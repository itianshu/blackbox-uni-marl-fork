"""Collect LB load and vLLM metrics into rolling per-policy windows."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Protocol

from .metrics import (
    MetricNames,
    VLLMMetricsScraper,
    VLLMMetricsSnapshot,
    summarize_vllm_metrics,
)

logger = logging.getLogger(__name__)


def _stat(values: list[float], how: str, default: float | None = 0.0) -> float | None:
    """Apply the window statistic used consistently by every signal."""
    if not values:
        return default
    if how == "instant":
        return values[-1]
    if how == "max":
        return max(values)
    if how == "mean":
        return sum(values) / len(values)
    index = min(len(values) - 1, max(0, round(0.9 * (len(values) - 1))))
    return sorted(values)[index]


@dataclass
class PolicySample:
    """One poll of one policy's serving state."""

    t: float
    inflight_per_replica: dict[str, int] = field(default_factory=dict)
    total_inflight: int = 0
    kv_cache_usage: dict[str, float] = field(default_factory=dict)
    vllm_metrics: dict[str, VLLMMetricsSnapshot] = field(default_factory=dict)
    metrics_fresh: bool = False


class SignalSource(Protocol):
    def sample_all(self) -> dict[str, PolicySample]: ...


class ServingSignalSource:
    """Sample LB in-flight state and periodically scrape each vLLM server."""

    def __init__(
        self,
        handles: dict[str, Any],
        scraper: VLLMMetricsScraper | None = None,
        scrape_interval_s: float = 1.0,
    ):
        self._handles = handles
        self._scraper = scraper
        self._scrape_interval_s = max(0.0, float(scrape_interval_s))
        self._last_scrape_t = float("-inf")

    def sample_all(self) -> dict[str, PolicySample]:
        now = time.monotonic()
        scrape = self._scraper is not None and now - self._last_scrape_t >= self._scrape_interval_s
        if scrape:
            self._last_scrape_t = now

        statuses = self._load_statuses()
        inflight = {
            policy: {str(server): int(count) for server, count in (status.get("servers") or {}).items()}
            for policy, status in statuses.items()
        }
        metrics = self._scrape_metrics(self._scrape_targets(inflight)) if scrape else {}

        samples = {}
        for policy, status in statuses.items():
            policy_metrics = metrics.get(policy, {})
            kv_cache = {
                server: value
                for server, snapshot in policy_metrics.items()
                if (value := self._scraper.kv_cache_usage(snapshot)) is not None
            }
            per_replica = inflight[policy]
            samples[policy] = PolicySample(
                t=now,
                inflight_per_replica=per_replica,
                total_inflight=int(status.get("total_inflight", sum(per_replica.values()))),
                kv_cache_usage=kv_cache,
                vllm_metrics=policy_metrics,
                metrics_fresh=scrape,
            )
        return samples

    def _load_statuses(self) -> dict[str, dict]:
        import ray

        references = {}
        for policy, handle in self._handles.items():
            try:
                references[policy] = handle.lb_handle.get_status.remote()
            except Exception as exc:
                logger.warning("dynamic_inference.signals: %s get_status failed: %s", policy, exc)
                references[policy] = None

        statuses = {}
        for policy, reference in references.items():
            try:
                statuses[policy] = ray.get(reference) if reference is not None else {}
            except Exception as exc:
                logger.warning("dynamic_inference.signals: %s get_status failed: %s", policy, exc)
                statuses[policy] = {}
        return statuses

    def _scrape_targets(self, inflight: dict[str, dict[str, int]]) -> list[tuple[str, str]]:
        targets = []
        for policy, handle in self._handles.items():
            addresses = {getattr(replica, "server_address", None) for replica in handle.replicas}
            addresses.update(inflight[policy])
            targets.extend((policy, address) for address in addresses if address is not None)
        return targets

    def _scrape_metrics(self, targets: list[tuple[str, str]]) -> dict[str, dict[str, VLLMMetricsSnapshot]]:
        metrics = {policy: {} for policy in self._handles}
        if not targets:
            return metrics
        with ThreadPoolExecutor(max_workers=min(32, len(targets))) as pool:
            futures = {
                pool.submit(self._scraper.scrape, address): (policy, address)
                for policy, address in targets
            }
            for future in as_completed(futures):
                policy, address = futures[future]
                try:
                    snapshot = future.result()
                except Exception as exc:
                    logger.warning("dynamic_inference.signals: scrape %s failed: %s", address, exc)
                    continue
                if snapshot is not None:
                    metrics[policy][address] = snapshot
        return metrics


class SignalStore:
    """Rolling-window statistics over per-policy serving samples."""

    def __init__(
        self,
        policies: list[str],
        window_s: float = 5.0,
        max_num_seqs: dict[str, int] | None = None,
    ):
        self.policies = list(policies)
        self.window_s = window_s
        self.max_num_seqs = dict(max_num_seqs or {policy: 1 for policy in policies})
        self.samples: dict[str, deque[PolicySample]] = {policy: deque() for policy in policies}

    def record(self, policy: str, sample: PolicySample) -> None:
        samples = self.samples[policy]
        samples.append(sample)
        cutoff = sample.t - self.window_s
        while samples and samples[0].t < cutoff:
            samples.popleft()

    def record_all(self, samples: dict[str, PolicySample]) -> None:
        for policy, sample in samples.items():
            if policy in self.samples:
                self.record(policy, sample)

    def refresh_now(self, source: SignalSource) -> bool:
        samples = source.sample_all()
        self.record_all(samples)
        return any(sample.metrics_fresh for sample in samples.values())

    def latest_has_kv(self, policy: str) -> bool:
        samples = self.samples[policy]
        return bool(samples and samples[-1].kv_cache_usage)

    def latest_vllm_snapshot(self, policy: str) -> VLLMMetricsSnapshot:
        for sample in reversed(self.samples[policy]):
            if sample.vllm_metrics:
                return VLLMMetricsSnapshot.merge(list(sample.vllm_metrics.values()))
        return VLLMMetricsSnapshot()

    def stat_vllm_metric(
        self,
        policy: str,
        metric_names: MetricNames,
        how: str = "p90",
        replica_reduce: str = "sum",
    ) -> float | None:
        values = []
        for sample in self._metric_samples(policy):
            snapshot = VLLMMetricsSnapshot.merge(list(sample.vllm_metrics.values()))
            value = snapshot.aggregate(metric_names, replica_reduce)
            if value is not None:
                values.append(value)
        return _stat(values, how, default=None)

    def vllm_counter_rate(self, policy: str, metric_names: MetricNames) -> float | None:
        samples = self._metric_samples(policy)
        if len(samples) < 2 or samples[-1].t <= samples[0].t:
            return None
        first, last = samples[0], samples[-1]
        increase = 0.0
        observed = False
        for server in set(first.vllm_metrics) & set(last.vllm_metrics):
            start = first.vllm_metrics[server].aggregate(metric_names)
            end = last.vllm_metrics[server].aggregate(metric_names)
            if start is None or end is None:
                continue
            observed = True
            increase += end - start if end >= start else end
        return increase / (last.t - first.t) if observed else None

    def vllm_observability(self, policy: str) -> dict[str, float | int]:
        return summarize_vllm_metrics(
            self.latest_vllm_snapshot(policy),
            lambda names, aggregation: self.stat_vllm_metric(
                policy, names, replica_reduce=aggregation,
            ),
            lambda names: self.vllm_counter_rate(policy, names),
        )

    def stat_inflight(self, policy: str, how: str = "p90") -> float:
        values = [float(sample.total_inflight) for sample in self.samples[policy]]
        return _stat(values, how) or 0.0

    def stat_per_replica_inflight(self, policy: str, how: str = "p90") -> dict[str, float]:
        samples = self.samples[policy]
        servers = {server for sample in samples for server in sample.inflight_per_replica}
        return {
            server: _stat([
                float(sample.inflight_per_replica[server])
                for sample in samples
                if server in sample.inflight_per_replica
            ], how) or 0.0
            for server in servers
        }

    def kv_util(self, policy: str, how: str = "p90") -> float:
        values = [
            max(sample.kv_cache_usage.values())
            for sample in self.samples[policy]
            if sample.kv_cache_usage
        ]
        return _stat(values, how) or 0.0

    def _metric_samples(self, policy: str) -> list[PolicySample]:
        return [sample for sample in self.samples[policy] if sample.vllm_metrics]
