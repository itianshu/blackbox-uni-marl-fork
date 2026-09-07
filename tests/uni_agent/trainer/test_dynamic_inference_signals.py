"""Tests for complete vLLM /metrics parsing, aggregation, and collection."""

import time

from uni_agent.trainer.dynamic_inference.metrics import (
    VLLMMetricsScraper,
    parse_vllm_metrics,
)
from uni_agent.trainer.dynamic_inference.signals import PolicySample, ServingSignalSource, SignalStore

_NAMES = ["kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio"]


class TestParseVLLMMetrics:
    def test_parses_all_numeric_vllm_series(self):
        text = (
            "# HELP vllm:gpu_cache_usage_perc Gauge\n"
            "# TYPE vllm:gpu_cache_usage_perc gauge\n"
            "vllm:gpu_cache_usage_perc 0.8312\n"
            'vllm:num_requests_running{model_name="x"} 3\n'
            "vllm:num_requests_waiting 7\n"
            "process_cpu_seconds_total 99\n"
        )
        snapshot = parse_vllm_metrics(text)
        assert snapshot.aggregate(_NAMES, "max") == 0.8312
        assert snapshot.aggregate("num_requests_running") == 3
        assert snapshot.aggregate("num_requests_waiting") == 7
        assert snapshot.aggregate("process_cpu_seconds_total") is None
        assert snapshot.samples[1].labels == {"model_name": "x"}

    def test_version_dependent_name_order(self):
        text = "vllm:kv_cache_usage_ratio{model=\"x\"} 0.42 1690000000000\n"
        snapshot = parse_vllm_metrics(text)
        assert snapshot.aggregate(_NAMES, "max") == 0.42

    def test_first_candidate_wins(self):
        text = (
            "vllm:gpu_cache_usage_perc 0.9\n"
            "vllm:kv_cache_usage_ratio 0.1\n"
        )
        snapshot = parse_vllm_metrics(text)
        assert snapshot.aggregate(_NAMES, "max") == 0.9
        # and the order in the config decides which one is "first"
        assert snapshot.aggregate(list(reversed(_NAMES)), "max") == 0.1

    def test_scientific_notation(self):
        snapshot = parse_vllm_metrics("vllm:gpu_cache_usage_perc 1.5e-1\n")
        assert snapshot.aggregate(_NAMES) == 0.15

    def test_malformed_and_non_finite_values_skipped(self):
        text = (
            "vllm:gpu_cache_usage_perc not-a-number\n"
            "vllm:num_requests_running NaN\n"
            "vllm:kv_cache_usage_ratio 0.7\n"
        )
        snapshot = parse_vllm_metrics(text)
        assert snapshot.aggregate(_NAMES) == 0.7
        assert snapshot.aggregate("num_requests_running") is None

    def test_prefix_must_not_match_loosely(self):
        text = "vllm:gpu_cache_usage_perc_total 0.99\n"
        assert parse_vllm_metrics(text).aggregate(_NAMES) is None

    def test_histogram_mean_and_p90(self):
        text = (
            'vllm:request_queue_time_seconds_bucket{le="0.1"} 4\n'
            'vllm:request_queue_time_seconds_bucket{le="0.5"} 9\n'
            'vllm:request_queue_time_seconds_bucket{le="+Inf"} 10\n'
            "vllm:request_queue_time_seconds_sum 2.5\n"
            "vllm:request_queue_time_seconds_count 10\n"
        )
        snapshot = parse_vllm_metrics(text)
        assert snapshot.histogram_mean("request_queue_time_seconds") == 0.25
        assert snapshot.histogram_quantile("request_queue_time_seconds", 0.9) == 0.5


class TestVLLMMetricsScraper:
    def test_scrape_returns_complete_snapshot(self, monkeypatch):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return (
                    b"vllm:kv_cache_usage_perc 0.6\n"
                    b"vllm:num_requests_waiting 4\n"
                )

        monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: _Response())
        scraper = VLLMMetricsScraper(_NAMES, timeout_s=0.1)
        snapshot = scraper.scrape("replica:8000")
        assert snapshot.aggregate("num_requests_waiting") == 4
        assert scraper.kv_cache_usage(snapshot) == 0.6

    def test_scrape_failure_returns_none(self, monkeypatch):
        scraper = VLLMMetricsScraper(_NAMES, timeout_s=0.1)

        def _boom(url, timeout):
            raise OSError("unreachable")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        assert scraper.scrape("127.0.0.1:1") is None


class _FakeLBHandle:
    """`get_status.remote()` -> a fake ObjectRef that `_FakeRay.get` resolves."""

    def __init__(self, status):
        self._status = status
        self.get_status = _FakeRemoteMethod(status)


class _FakeRemoteMethod:
    def __init__(self, status):
        self._status = status

    def remote(self):
        return ("resolved", self._status)


class _FakeRay:
    @staticmethod
    def get(ref):
        kind, value = ref
        return value


class TestServingSignalSource:
    def _handles(self, addresses):
        class _Replica:
            def __init__(self, addr):
                self.server_address = addr

        class _Handle:
            def __init__(self, addrs):
                self.replicas = [_Replica(a) for a in addrs]
                self.lb_handle = _FakeLBHandle({"servers": {}, "total_inflight": 0})

        return {name: _Handle(addrs) for name, addrs in addresses.items()}

    def test_scrape_throttled_by_interval(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ray", _FakeRay)

        scraped = []

        class _FakeScraper:
            def scrape(self, addr):
                scraped.append(addr)
                return parse_vllm_metrics(
                    "vllm:gpu_cache_usage_perc 0.5\n"
                    "vllm:num_requests_waiting 2\n"
                )

            def kv_cache_usage(self, snapshot):
                return snapshot.aggregate(_NAMES, "max")

        handles = self._handles({"a": ["h1:1"], "b": ["h2:2"]})
        source = ServingSignalSource(handles, scraper=_FakeScraper(), scrape_interval_s=100.0)

        samples = source.sample_all()
        assert sorted(scraped) == ["h1:1", "h2:2"]
        assert samples["a"].kv_cache_usage == {"h1:1": 0.5}
        assert samples["a"].vllm_metrics["h1:1"].aggregate(
            "num_requests_waiting") == 2
        assert samples["a"].metrics_fresh is True

        # next tick is inside the interval: no new scrapes
        scraped.clear()
        cached = source.sample_all()
        assert scraped == []
        assert cached["a"].metrics_fresh is False
        assert samples["a"].kv_cache_usage  # previous samples untouched

    def test_store_reports_only_fresh_scrape_attempts(self):
        store = SignalStore(["a"])

        class Source:
            fresh = True

            def sample_all(self):
                return {"a": PolicySample(
                    t=time.monotonic(),
                    kv_cache_usage={"a:0": 0.5} if self.fresh else {},
                    metrics_fresh=self.fresh,
                )}

        source = Source()
        assert store.refresh_now(source) is True
        assert store.latest_has_kv("a") is True
        source.fresh = False
        assert store.refresh_now(source) is False
        assert store.latest_has_kv("a") is False

    def test_lb_servers_included_in_scrape_targets(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ray", _FakeRay)

        scraped = []

        class _FakeScraper:
            def scrape(self, addr):
                scraped.append(addr)
                return None  # failures are skipped silently

            def kv_cache_usage(self, snapshot):
                raise AssertionError("failed scrapes have no snapshot")

        handles = self._handles({"a": ["h1:1"]})
        # the LB reports a guest server the home replicas don't know about
        handles["a"].lb_handle = _FakeLBHandle({"servers": {"guest:9": 3}, "total_inflight": 3})
        source = ServingSignalSource(handles, scraper=_FakeScraper(), scrape_interval_s=0.0)

        sample = source.sample_all()["a"]
        assert sorted(scraped) == ["guest:9", "h1:1"]
        assert sample.kv_cache_usage == {}  # scrape returned None
        assert sample.vllm_metrics == {}
        assert sample.inflight_per_replica == {"guest:9": 3}


class TestSignalStoreVLLMMetrics:
    def test_curated_observability_includes_queue_latency_and_counters(self):
        previous = parse_vllm_metrics(
            "vllm:num_requests_running 1\n"
            "vllm:num_requests_waiting 3\n"
            "vllm:prompt_tokens_total 80\n"
            "vllm:generation_tokens_total 20\n"
        )
        metrics = parse_vllm_metrics(
            "vllm:num_requests_running 3\n"
            "vllm:num_requests_waiting 7\n"
            "vllm:prompt_tokens_total 100\n"
            "vllm:generation_tokens_total 40\n"
            "vllm:prefix_cache_queries_total 20\n"
            "vllm:prefix_cache_hits_total 15\n"
            'vllm:request_queue_time_seconds_bucket{le="0.1"} 4\n'
            'vllm:request_queue_time_seconds_bucket{le="0.5"} 9\n'
            'vllm:request_queue_time_seconds_bucket{le="+Inf"} 10\n'
            "vllm:request_queue_time_seconds_sum 2.5\n"
            "vllm:request_queue_time_seconds_count 10\n"
            'vllm:time_to_first_token_seconds_bucket{le="1"} 8\n'
            'vllm:time_to_first_token_seconds_bucket{le="2"} 10\n'
            'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10\n'
            "vllm:time_to_first_token_seconds_sum 7\n"
            "vllm:time_to_first_token_seconds_count 10\n"
        )
        store = SignalStore(["a"])
        store.record("a", PolicySample(
            t=10.0,
            vllm_metrics={"replica-0": previous},
            metrics_fresh=True,
        ))
        store.record("a", PolicySample(
            t=12.0,
            vllm_metrics={"replica-0": metrics},
            metrics_fresh=True,
        ))

        observed = store.vllm_observability("a")
        assert observed["requests_running"] == 3
        assert observed["requests_waiting"] == 7
        assert observed["requests_waiting_window_p90"] == 7
        assert observed["request_queue_time_seconds_mean"] == 0.25
        assert observed["request_queue_time_seconds_p90"] == 0.5
        assert observed["time_to_first_token_seconds_mean"] == 0.7
        assert observed["time_to_first_token_seconds_p90"] == 1.5
        assert observed["prompt_tokens_total"] == 100
        assert observed["generation_tokens_total"] == 40
        assert observed["prompt_tokens_per_second"] == 10
        assert observed["generation_tokens_per_second"] == 10
        assert observed["prefix_cache_hit_rate"] == 0.75

    def test_raw_snapshot_retains_unrecognised_vllm_metrics(self):
        metrics = parse_vllm_metrics("vllm:future_metric_from_new_version 12\n")
        store = SignalStore(["a"])
        store.record("a", PolicySample(
            t=time.monotonic(),
            vllm_metrics={"replica-0": metrics},
            metrics_fresh=True,
        ))
        assert store.latest_vllm_snapshot("a").aggregate(
            "future_metric_from_new_version") == 12
