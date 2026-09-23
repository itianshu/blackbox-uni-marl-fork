"""Multi-policy dynamic inference scheduling (RFC: multi-agent dynamic
inference scheduling).

Cross-policy replica borrowing over a many-to-many directed graph, triggered
by fresh serving metrics: an underloaded policy's (``home``) standalone
replicas are put to level-2 sleep and pre-created donor-architecture ``guest``
replicas are woken on the same cards,
cloning published weights from an active same-policy vLLM and joining the bottleneck (``donor``)
policy's load balancer and checkpoint manager until returned.

Serving signals are sampled together from each load balancer and replica vLLM
``/metrics`` endpoint. A policy enters overload on either high KV utilisation
or a deep, long-lived request queue. Each borrow event selects at most one safe
atomic replica unit (:mod:`uni_agent.trainer.dynamic_inference.quantity`).

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
from .quantity import EqualisationStrategy, SingleUnitStrategy
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
    "SingleUnitStrategy",
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
