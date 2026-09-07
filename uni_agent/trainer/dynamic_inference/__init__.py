"""Multi-policy dynamic inference scheduling (RFC: multi-agent dynamic
inference scheduling).

Cross-policy replica borrowing over a many-to-many directed graph, triggered
by fresh serving metrics: an underloaded policy's (``home``) standalone
replicas are put to level-2 sleep and pre-created donor-architecture ``guest``
replicas are woken on the same cards,
receiving one directed weight push and joining the bottleneck (``donor``)
policy's load balancer for one step.

Serving signals are scraped from each replica's vLLM ``/metrics`` endpoint.
KV cache utilisation remains the scheduling input; queue, latency, throughput,
request, and cache statistics are retained for observation and future policies.
Borrow quantity uses discrete load equalisation across all topology-valid units
(:mod:`uni_agent.trainer.dynamic_inference.quantity`).

Public entry point: :class:`DynamicInferenceController`
(:meth:`DynamicInferenceController.maybe_create`), wired into
``MultiAgentsPPOTrainer``. The verl runtime patches live in
:mod:`uni_agent.trainer.dynamic_inference.patch` (applied via the yaml's
``worker_process_setup_hook`` / controller setup; the verl submodule stays
pristine).
"""

from .controller import BoundaryGate, DynamicInferenceController, PollLoop
from .executor import BoundaryExecutor
from .guest_engine import GuestEngineManager
from .metrics import (
    PrometheusSample,
    VLLMMetricsScraper,
    VLLMMetricsSnapshot,
    parse_vllm_metrics,
)
from .quantity import EqualisationStrategy
from .scheduler import MultiPolicyInferenceScheduler
from .signals import PolicySample, ServingSignalSource, SignalStore
from .types import (
    BorrowPairSpec,
    BorrowingConfig,
    BoundaryPlan,
    BorrowPlan,
    GuestUnit,
    LendEvent,
    LendRecord,
    PolicyInferenceHandles,
    ResourceUsageConfig,
    SchedulingConfig,
    StepStats,
    expand_pairs,
    required_slots_by_home,
    parse_scheduling_config,
    validate_against_handles,
    validate_policy_prerequisites,
)

__all__ = [
    "BoundaryExecutor",
    "BoundaryGate",
    "BoundaryPlan",
    "BorrowPairSpec",
    "BorrowPlan",
    "BorrowingConfig",
    "DynamicInferenceController",
    "EqualisationStrategy",
    "GuestEngineManager",
    "GuestUnit",
    "LendEvent",
    "LendRecord",
    "MultiPolicyInferenceScheduler",
    "PolicyInferenceHandles",
    "PolicySample",
    "PrometheusSample",
    "PollLoop",
    "ResourceUsageConfig",
    "SchedulingConfig",
    "ServingSignalSource",
    "SignalStore",
    "StepStats",
    "VLLMMetricsScraper",
    "VLLMMetricsSnapshot",
    "expand_pairs",
    "required_slots_by_home",
    "parse_scheduling_config",
    "parse_vllm_metrics",
    "validate_against_handles",
    "validate_policy_prerequisites",
]
