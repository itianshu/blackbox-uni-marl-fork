"""vLLM Prometheus parsing, scraping, and compact observability summaries."""

from __future__ import annotations

import logging
import math
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

MetricNames = str | list[str] | tuple[str, ...]


@dataclass(frozen=True)
class PrometheusSample:
    """One numeric vLLM Prometheus series."""

    name: str
    labels: dict[str, str]
    value: float


@dataclass
class VLLMMetricsSnapshot:
    """All numeric ``vllm:*`` series returned by one or more replicas."""

    samples: list[PrometheusSample] = field(default_factory=list)

    @classmethod
    def merge(cls, snapshots: list[VLLMMetricsSnapshot]) -> VLLMMetricsSnapshot:
        return cls([sample for snapshot in snapshots for sample in snapshot.samples])

    def values(self, metric_names: MetricNames) -> list[float]:
        names = (metric_names,) if isinstance(metric_names, str) else metric_names
        for name in names:
            values = [sample.value for sample in self.samples if sample.name == name]
            if values:
                return values
        return []

    def aggregate(self, metric_names: MetricNames, how: str = "sum") -> float | None:
        values = self.values(metric_names)
        if not values:
            return None
        reducers = {"max": max, "min": min, "first": lambda items: items[0]}
        return reducers.get(how, sum)(values)

    def histogram_mean(self, metric_names: MetricNames) -> float | None:
        names = (metric_names,) if isinstance(metric_names, str) else metric_names
        for name in names:
            total = self.aggregate(f"{name}_sum")
            count = self.aggregate(f"{name}_count")
            if total is not None and count is not None and count > 0:
                return total / count
        return None

    def histogram_quantile(self, metric_names: MetricNames, q: float) -> float | None:
        """Estimate a quantile from cumulative Prometheus histogram buckets."""
        names = (metric_names,) if isinstance(metric_names, str) else metric_names
        for name in names:
            buckets = self._histogram_buckets(name)
            if not buckets or buckets[-1][1] <= 0:
                continue
            target = min(1.0, max(0.0, q)) * buckets[-1][1]
            lower, previous_count = 0.0, 0.0
            for upper, count in buckets:
                if count < target:
                    lower, previous_count = upper, count
                    continue
                if math.isinf(upper):
                    return lower
                bucket_count = count - previous_count
                if bucket_count <= 0:
                    return upper
                return lower + (target - previous_count) / bucket_count * (upper - lower)
        return None

    def _histogram_buckets(self, name: str) -> list[tuple[float, float]]:
        buckets: dict[float, float] = {}
        for sample in self.samples:
            if sample.name != f"{name}_bucket" or "le" not in sample.labels:
                continue
            try:
                upper = float(sample.labels["le"])
            except ValueError:
                continue
            buckets[upper] = buckets.get(upper, 0.0) + sample.value
        return sorted(buckets.items())


_PROMETHEUS_LINE = re.compile(
    r'^\s*(vllm:[A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)'
)
_PROMETHEUS_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_vllm_metrics(metrics_text: str) -> VLLMMetricsSnapshot:
    """Parse every finite numeric ``vllm:*`` series from Prometheus text."""
    samples = []
    for line in metrics_text.splitlines():
        match = _PROMETHEUS_LINE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {
            label.group(1): label.group(2)
            for label in _PROMETHEUS_LABEL.finditer(match.group(2) or "")
        }
        samples.append(PrometheusSample(
            name=match.group(1).removeprefix("vllm:"), labels=labels, value=value,
        ))
    return VLLMMetricsSnapshot(samples)


class VLLMMetricsScraper:
    """Fetch and parse all vLLM metrics from one replica HTTP server."""

    def __init__(self, kv_metric_names: list[str], timeout_s: float = 1.0):
        self.kv_metric_names = tuple(kv_metric_names)
        self.timeout_s = timeout_s

    def scrape(self, server_address: str) -> VLLMMetricsSnapshot | None:
        url = f"http://{server_address}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
                return parse_vllm_metrics(response.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            logger.warning("dynamic_inference.metrics: scrape %s failed: %s", url, exc)
            return None

    def kv_cache_usage(self, snapshot: VLLMMetricsSnapshot) -> float | None:
        return snapshot.aggregate(self.kv_metric_names, how="max")


_GAUGES = {
    "requests_running": (("num_requests_running",), "sum"),
    "requests_waiting": (("num_requests_waiting",), "sum"),
    "requests_waiting_by_reason": (("num_requests_waiting_by_reason",), "sum"),
    "requests_swapped": (("num_requests_swapped",), "sum"),
    "engine_sleep_state_max": (("engine_sleep_state",), "max"),
    "gpu_cache_usage_max": (
        ("kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio"), "max",
    ),
    "cpu_cache_usage_max": (("cpu_cache_usage_perc", "cpu_cache_usage_ratio"), "max"),
}
_COUNTERS = {
    "prompt_tokens_total": ("prompt_tokens_total", "prompt_tokens"),
    "prompt_tokens_cached_total": ("prompt_tokens_cached_total", "prompt_tokens_cached"),
    "generation_tokens_total": ("generation_tokens_total", "generation_tokens"),
    "iteration_tokens_total": ("iteration_tokens_total",),
    "requests_success_total": ("request_success_total",),
    "requests_failure_total": ("request_failure_total", "request_failures_total"),
    "preemptions_total": ("num_preemptions_total", "preemptions_total"),
    "corrupted_requests_total": ("corrupted_requests_total", "corrupted_requests"),
}
_HISTOGRAMS = {
    "request_queue_time_seconds": ("request_queue_time_seconds", "request_waiting_time_seconds"),
    "time_to_first_token_seconds": ("time_to_first_token_seconds",),
    "time_per_output_token_seconds": (
        "request_time_per_output_token_seconds", "time_per_output_token_seconds", "inter_token_latency_seconds",
    ),
    "e2e_request_latency_seconds": ("e2e_request_latency_seconds",),
    "request_prefill_time_seconds": ("request_prefill_time_seconds",),
    "request_decode_time_seconds": ("request_decode_time_seconds",),
    "request_inference_time_seconds": ("request_inference_time_seconds",),
    "request_prompt_tokens": ("request_prompt_tokens",),
    "request_generation_tokens": ("request_generation_tokens",),
    "request_max_num_generation_tokens": ("request_max_num_generation_tokens",),
    "request_params_n": ("request_params_n",),
    "request_params_max_tokens": ("request_params_max_tokens",),
    "request_prefill_kv_computed_tokens": ("request_prefill_kv_computed_tokens",),
    "kv_block_lifetime_seconds": ("kv_block_lifetime_seconds",),
    "kv_block_idle_before_evict_seconds": ("kv_block_idle_before_evict_seconds",),
    "kv_block_reuse_gap_seconds": ("kv_block_reuse_gap_seconds",),
}
_CACHES = {
    "prefix_cache": ("prefix_cache_queries", "prefix_cache_hits"),
    "external_prefix_cache": ("external_prefix_cache_queries", "external_prefix_cache_hits"),
    "mm_cache": ("mm_cache_queries", "mm_cache_hits"),
}


def _counter_names(name: str) -> tuple[str, str]:
    return f"{name}_total", name


def summarize_vllm_metrics(
    snapshot: VLLMMetricsSnapshot,
    gauge_window_p90: Callable[[MetricNames, str], float | None],
    counter_rate: Callable[[MetricNames], float | None],
) -> dict[str, float | int]:
    """Build the stable, compact view exported by the scheduler."""
    if not snapshot.samples:
        return {}
    result: dict[str, float | int] = {"series_count": len(snapshot.samples)}
    for key, (names, aggregation) in _GAUGES.items():
        value = snapshot.aggregate(names, aggregation)
        if value is not None:
            result[key] = value
            p90 = gauge_window_p90(names, aggregation)
            if p90 is not None:
                result[f"{key}_window_p90"] = p90
    for key, names in _COUNTERS.items():
        value = snapshot.aggregate(names)
        if value is not None:
            result[key] = value
            rate = counter_rate(names)
            if rate is not None:
                result[f"{key.removesuffix('_total')}_per_second"] = rate
    for key, names in _HISTOGRAMS.items():
        mean = snapshot.histogram_mean(names)
        p90 = snapshot.histogram_quantile(names, 0.9)
        if mean is not None:
            result[f"{key}_mean"] = mean
        if p90 is not None:
            result[f"{key}_p90"] = p90
    for key, (queries, hits) in _CACHES.items():
        query_count = snapshot.aggregate(_counter_names(queries))
        hit_count = snapshot.aggregate(_counter_names(hits))
        if query_count is not None:
            result[f"{key}_queries_total"] = query_count
        if hit_count is not None:
            result[f"{key}_hits_total"] = hit_count
        if query_count is not None and hit_count is not None and query_count > 0:
            result[f"{key}_hit_rate"] = hit_count / query_count
    return result
