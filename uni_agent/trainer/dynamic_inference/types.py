"""Types and configuration for multi-policy dynamic inference scheduling.

Implements the data model of the RFC ``multi_agent_dynamic_inference_scheduling``:
per-policy inference handles, general directed borrow-graph declarations, the
KV-cache-utilisation scheduling configuration, and the borrow/return records
the scheduler and executor exchange.

The module is intentionally dependency-free (no ray/verl imports at module
level) so the whole decision layer stays CPU-unit-testable.

Terminology (follows the RFC):
- ``home``  = the lending (underloaded) policy;
- ``donor`` = the borrowing (overloaded / bottleneck) policy;
- ``guest`` = a pre-created donor-architecture replica sleeping on a home
  replica's card set, woken only while the unit is lent out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_MODES = ("static", "resource")


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class BorrowPairSpec:
    """Card-folding declaration for one (home -> donor) direction.

    One borrow unit groups ``N`` home replicas and folds their complete card
    set into ``M`` donor-architecture guest replicas.  Runtime validation
    requires the card equation ``N * home_cards == M * donor_cards``; runtime
    placement groups home replicas by physical node and validates the folded
    guest layout.
    """

    home: str
    donor: str
    home_replicas_per_unit: int = 1
    guest_replicas_per_unit: int = 1


@dataclass
class ResourceUsageConfig:
    """KV-cache-utilisation thresholds for scheduling decisions."""

    kv_enter: float = 0.85                # KV utilisation above which a policy may be judged bottleneck
    kv_exit: float = 0.6                  # bottleneck state exits at or below this
    kv_post_lend_max: float = 0.7         # predicted post-lend lender cap; below kv_enter for hysteresis
    kv_metric_names: list[str] = field(default_factory=lambda: [
        "kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio",
    ])                                    # candidate /metrics gauge names (vLLM version dependent)
    ema_alpha: float = 0.3


@dataclass
class BorrowingConfig:
    """Borrow-unit topology and guest-replica lifecycle knobs."""

    pairs: list[BorrowPairSpec] = field(default_factory=list)  # empty = all ordered pairs, 1:1
    guest_replica_rank_offset: int = 10000  # keeps guest vLLM actor names unique


@dataclass
class SchedulingConfig:
    """Top-level dynamic-inference-scheduling configuration."""

    enable: bool = False
    mode: str = "resource"                 # static | resource
    # --- bottleneck detection (KV cache utilisation) ---
    bottleneck_confirm_polls: int = 10
    # --- rebalance anti-jitter ---
    rebalance_confirm_polls: int = 2
    rebalance_settle_polls: int = 3
    min_lend_polls: int = 2
    borrow_cooldown_s: float = 10.0        # no new borrow event during this wall-clock interval
    return_confirm_polls: int = 10         # donor KV <= kv_exit before a guarded normal return
    # --- early return (home exhausted while lent out) ---
    early_return_confirm_polls: int = 10
    # --- polling / execution ---
    poll_interval_s: float = 0.2
    metrics_scrape_interval_s: float = 1.0  # vLLM /metrics scrape period
    borrow_drain_timeout_s: float = 10.0
    sleep_patch_mode: str = "patched"      # patched | collective_rpc (DP=1 fallback)
    # --- sub-blocks ---
    resource_usage: ResourceUsageConfig = field(default_factory=ResourceUsageConfig)
    borrowing: BorrowingConfig = field(default_factory=BorrowingConfig)


# --------------------------------------------------------------------------- #
# Runtime records
# --------------------------------------------------------------------------- #


@dataclass
class PolicyInferenceHandles:
    """Everything the scheduler/executor needs for one policy.

    Built by the trainer after every policy runtime exists. Fields holding
    Ray/verl objects are typed ``Any`` to keep this module import-light.
    """

    actor_rollout_wg: Any                         # trainer.actor_rollout_wg (weight-push sender)
    standalone_checkpoint_manager: Any            # CheckpointEngineManager (home replicas only)
    lb_handle: Any                                # global_load_balancer Ray actor handle
    rollout_config: Any                           # actor_rollout_ref.rollout (dataclass or DictConfig)
    model_config: Any
    replicas: list[Any] = field(default_factory=list)   # home RolloutReplica objects
    cards_per_replica: int = 0                    # nnodes * n_gpus_per_node
    nnodes: int = 1
    max_num_seqs: int = 64                        # vLLM running-batch cap (early-return signal)


@dataclass
class GuestUnit:
    """Pre-created sleeping guests on one (possibly grouped) home card set."""

    home_policy: str
    donor: str
    home_replicas: list[Any] = field(default_factory=list)
    guests: list[Any] = field(default_factory=list)
    in_use: bool = False


@dataclass
class BorrowPlan:
    """A borrow the scheduler wants executed at the boundary."""

    home_policy: str
    donor: str
    home_replicas: list[Any] = field(default_factory=list)


@dataclass
class LendRecord:
    """A physically active lend (one borrow unit)."""

    lend_id: int
    home_policy: str
    donor: str
    home_replicas: list[Any]
    guests: list[Any]
    since_step: int
    # early-return bookkeeping
    home_kv_hot_polls: int = 0
    # resumable return: physical steps already completed (executor retries from
    # here when a return failed mid-sequence)
    return_stage: int = 0


@dataclass
class BoundaryPlan:
    """Incremental allocation changes for one scheduling decision."""

    returns: list[LendRecord] = field(default_factory=list)
    renewals: list[LendRecord] = field(default_factory=list)
    borrows: list[BorrowPlan] = field(default_factory=list)


@dataclass
class StepStats:
    """Boundary input used by the scheduler."""

    step: int


@dataclass
class LendEvent:
    """A scheduling event for the metrics/log stream."""

    kind: str                                    # borrow | renew | return | early_return | bottleneck_change | disable
    step: int
    t: float
    home: str = ""
    donor: str = ""
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #


def _to_plain_dict(cfg: Any) -> dict:
    """Accept DictConfig / dict / None and return a plain dict."""
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except (ImportError, AttributeError):
        # AttributeError: some test stubs install a minimal omegaconf without
        # is_config — fall through to the plain-dict path.
        pass
    return dict(cfg)


def _build(spec_cls, data: dict, path: str):
    """Instantiate ``spec_cls`` and reject removed or misspelled keys."""
    valid = {f for f in spec_cls.__dataclass_fields__}  # noqa: C416 - readability
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"unknown {path} configuration keys: {unknown}")
    return spec_cls(**data)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_scheduling_config(cfg: Any) -> SchedulingConfig | None:
    """Parse the ``dynamic_inference_scheduling`` config block.

    Returns None when the block is absent or disabled. Raises ValueError on
    statically invalid combinations (mode whitelist, pair folding, ...).
    """
    data = _to_plain_dict(cfg)
    if not data or not data.get("enable", False):
        return None

    # Compatibility for configs written before scheduling moved from step
    # boundaries to complete metrics polls.
    for old, new in {
        "bottleneck_confirm_steps": "bottleneck_confirm_polls",
        "rebalance_confirm_steps": "rebalance_confirm_polls",
        "min_lend_steps": "min_lend_polls",
    }.items():
        if old not in data:
            continue
        if new in data:
            raise ValueError(f"configure only one of '{old}' and '{new}'")
        data[new] = data.pop(old)

    # Kept as an accepted no-op for old configs. Early return now uses the
    # same KV signal as borrowing instead of LB in-flight counts.
    data.pop("early_return_waiting_ratio", None)

    borrowing = _build(
        BorrowingConfig,
        _to_plain_dict(data.get("borrowing")),
        "dynamic_inference_scheduling.borrowing",
    )
    pairs = []
    for pair in borrowing.pairs:
        if isinstance(pair, BorrowPairSpec):
            pairs.append(pair)
        else:
            pairs.append(_build(BorrowPairSpec, _to_plain_dict(pair), "borrowing.pairs[]"))
    borrowing.pairs = pairs

    # sub-blocks are built explicitly below — exclude them from the splat or
    # a raw dict carrying e.g. 'borrowing' would pass it twice
    top_level = dict(data)
    top_level.pop("resource_usage", None)
    top_level.pop("borrowing", None)
    config = _build(
        SchedulingConfig,
        {
            **top_level,
            "resource_usage": _build(
                ResourceUsageConfig,
                _to_plain_dict(data.get("resource_usage")),
                "dynamic_inference_scheduling.resource_usage",
            ),
            "borrowing": borrowing,
        },
        "dynamic_inference_scheduling",
    )

    # --- static validation ---
    if config.mode not in _MODES:
        raise ValueError(
            f"unknown dynamic_inference_scheduling.mode '{config.mode}' "
            f"(supported modes: {list(_MODES)}; the coverage/fused schemes were removed)"
        )
    if config.sleep_patch_mode not in ("patched", "collective_rpc"):
        raise ValueError(f"unknown sleep_patch_mode '{config.sleep_patch_mode}'")
    ru = config.resource_usage
    if (
        not _is_finite_number(ru.kv_exit)
        or not _is_finite_number(ru.kv_enter)
        or not 0.0 <= ru.kv_exit < ru.kv_enter <= 1.0
    ):
        raise ValueError("resource_usage must satisfy 0 <= kv_exit < kv_enter <= 1")
    if (
        not _is_finite_number(ru.kv_post_lend_max)
        or not 0.0 < ru.kv_post_lend_max <= 1.0
    ):
        raise ValueError("resource_usage.kv_post_lend_max must be in (0, 1]")
    if not ru.kv_exit < ru.kv_post_lend_max < ru.kv_enter:
        raise ValueError(
            "resource_usage thresholds must satisfy "
            "kv_exit < kv_post_lend_max < kv_enter"
        )
    if not _is_finite_number(ru.ema_alpha) or not 0.0 < ru.ema_alpha <= 1.0:
        raise ValueError("resource_usage.ema_alpha must be in (0, 1]")
    if (
        not isinstance(ru.kv_metric_names, list)
        or not ru.kv_metric_names
        or any(not isinstance(name, str) or not name.strip()
               for name in ru.kv_metric_names)
    ):
        raise ValueError("resource_usage.kv_metric_names must be a non-empty list of names")
    for key in (
        "bottleneck_confirm_polls",
        "rebalance_confirm_polls",
        "rebalance_settle_polls",
        "min_lend_polls",
        "return_confirm_polls",
        "early_return_confirm_polls",
    ):
        value = getattr(config, key)
        if not _is_int(value) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("poll_interval_s", "metrics_scrape_interval_s", "borrow_drain_timeout_s"):
        value = getattr(config, key)
        if not _is_finite_number(value) or value <= 0:
            raise ValueError(f"{key} must be positive")
    if (
        not _is_finite_number(config.borrow_cooldown_s)
        or config.borrow_cooldown_s < 0
    ):
        raise ValueError("borrow_cooldown_s must be non-negative")
    if (
        not _is_int(config.borrowing.guest_replica_rank_offset)
        or config.borrowing.guest_replica_rank_offset < 0
    ):
        raise ValueError("borrowing.guest_replica_rank_offset must be a non-negative integer")
    seen = set()
    for pair in config.borrowing.pairs:
        if pair.home == pair.donor:
            raise ValueError(f"borrow pair {pair.home}->{pair.donor} is a self-pair")
        key = (pair.home, pair.donor)
        if key in seen:
            raise ValueError(f"duplicate borrow pair {key}")
        seen.add(key)
        if (
            not _is_int(pair.home_replicas_per_unit)
            or not _is_int(pair.guest_replicas_per_unit)
            or pair.home_replicas_per_unit < 1
            or pair.guest_replicas_per_unit < 1
        ):
            raise ValueError(
                f"borrow pair {pair.home}->{pair.donor}: home_replicas_per_unit "
                "and guest_replicas_per_unit must both be positive integers"
            )
    return config


def expand_pairs(config: SchedulingConfig, policy_names: list[str]) -> list[BorrowPairSpec]:
    """Resolve the effective pair list: declared pairs, or all ordered pairs 1:1.

    Empty ``borrowing.pairs`` expands to the complete directed graph for any
    number of policies.  This is suitable for homogeneous 1:1 topologies;
    heterogeneous policy layouts must declare their valid N:M edges explicitly.
    """
    if config.borrowing.pairs:
        known = set(policy_names)
        for pair in config.borrowing.pairs:
            for side in (pair.home, pair.donor):
                if side not in known:
                    raise ValueError(f"borrow pair references unknown policy '{side}'")
        return list(config.borrowing.pairs)
    return [
        BorrowPairSpec(home=home, donor=donor)
        for home in policy_names
        for donor in policy_names
        if home != donor
    ]


def required_slots_by_home(
    pairs: list[BorrowPairSpec],
    policy_names: list[str] | None = None,
) -> dict[str, int]:
    """Return the placement-group slot count required by each home policy.

    Every physical card needs one slot for its home worker and one additional
    fractional-GPU slot for each configured outgoing donor direction.  Guests
    remain asleep until selected, while the scheduler prevents two directions
    from activating on the same home replica at once.
    """
    outgoing = {name: 0 for name in (policy_names or [])}
    for pair in pairs:
        outgoing[pair.home] = outgoing.get(pair.home, 0) + 1
    return {home: 1 + count for home, count in outgoing.items()}


def validate_policy_prerequisites(
    config: SchedulingConfig,
    policy_rollout_configs: dict[str, Any],
) -> None:
    """Static per-policy checks against each policy's composed verl rollout config.

    Must run after ``parse_scheduling_config``; raises ValueError listing every
    violation (so the user sees all problems at once).
    """

    def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
        node = cfg
        for part in dotted.split("."):
            if node is None:
                return default
            if isinstance(node, dict):
                node = node.get(part)
            else:
                node = getattr(node, part, None)
        return default if node is None else node

    problems: list[str] = []
    for name, rollout_cfg in policy_rollout_configs.items():
        if str(_get(rollout_cfg, "name", "vllm")) != "vllm":
            problems.append(f"policy '{name}': rollout.name must be 'vllm' (got '{_get(rollout_cfg, 'name')}')")
        backend = _get(rollout_cfg, "checkpoint_engine.backend", "naive")
        if backend == "naive":
            problems.append(f"policy '{name}': checkpoint_engine.backend must be nccl/nixl/mooncake (separate-async forbids naive)")
        if _get(rollout_cfg, "enable_sleep_mode", True) is not True:
            problems.append(f"policy '{name}': rollout.enable_sleep_mode must be true (guest sleep/wake relies on it)")
        if _get(rollout_cfg, "free_cache_engine", True) is not True:
            problems.append(f"policy '{name}': rollout.free_cache_engine must be true")
        nnodes = int(_get(rollout_cfg, "nnodes", 1) or 1)
        gpus = int(_get(rollout_cfg, "n_gpus_per_node", 0) or 0)
        if nnodes < 1 or gpus < 1:
            problems.append(f"policy '{name}': rollout.nnodes/n_gpus_per_node must be positive (separate-async standalone replicas)")
        if str(_get(rollout_cfg, "full_determinism", False)).lower() in ("true", "1"):
            # not a hard error: guests still get traffic, just not preferentially
            import logging

            logging.getLogger(__name__).warning(
                "policy '%s': rollout.full_determinism is on — least-loaded routing is "
                "bypassed, borrowed guests receive uniform (not preferential) traffic",
                name,
            )
    if problems:
        raise ValueError("dynamic_inference_scheduling prerequisites failed:\n  - " + "\n  - ".join(problems))


def validate_against_handles(
    config: SchedulingConfig,
    handles: dict[str, PolicyInferenceHandles],
    pairs: list[BorrowPairSpec],
) -> None:
    """Reality checks that need live replica objects (run right before precreate).

    - every home policy can lend one complete unit and keep >= 1 replica;
    - every pair closes its N:M card equation and has a compatible node shape;
    - every home placement group has enough fractional worker slots for its
      home worker plus one sleeping guest-worker set per outgoing edge.
    """
    problems: list[str] = []
    required_slots = required_slots_by_home(pairs, list(handles))
    for policy, slots in required_slots.items():
        handle = handles.get(policy)
        if handle is None:
            continue
        for replica in handle.replicas:
            pool = getattr(replica, "resource_pool", None)
            actual = getattr(pool, "max_colocate_count", None)
            if actual is not None and actual < slots:
                problems.append(
                    f"policy '{policy}' placement group has {actual} worker slots; "
                    f"needs at least {slots} (1 home + {slots - 1} outgoing guests)"
                )
                break
    for pair in pairs:
        home_h = handles.get(pair.home)
        donor_h = handles.get(pair.donor)
        if home_h is None or donor_h is None:
            continue
        if len(home_h.replicas) < pair.home_replicas_per_unit + 1:
            problems.append(
                f"borrow pair {pair.home}->{pair.donor}: home policy needs at least "
                f"N+1={pair.home_replicas_per_unit + 1} replicas to lend one complete "
                f"unit and keep one (has {len(home_h.replicas)})"
            )
        home_cards = pair.home_replicas_per_unit * home_h.cards_per_replica
        guest_cards = pair.guest_replicas_per_unit * donor_h.cards_per_replica
        if home_cards != guest_cards:
            problems.append(
                f"borrow pair {pair.home}->{pair.donor}: card equation does not close: "
                f"N({pair.home_replicas_per_unit}) x home_cards({home_h.cards_per_replica}) "
                f"= {home_cards}, but M({pair.guest_replicas_per_unit}) x "
                f"donor_cards({donor_h.cards_per_replica}) = {guest_cards}"
            )
        if (home_h.cards_per_replica % max(1, home_h.nnodes) != 0
                or donor_h.cards_per_replica % max(1, donor_h.nnodes) != 0):
            problems.append(
                f"borrow pair {pair.home}->{pair.donor}: invalid per-node layout "
                f"({pair.home}: {home_h.cards_per_replica}/{home_h.nnodes}, "
                f"{pair.donor}: {donor_h.cards_per_replica}/{donor_h.nnodes}); "
                "replica cards must divide evenly across replica nodes"
            )
    if problems:
        raise ValueError("dynamic_inference_scheduling replica validation failed:\n  - " + "\n  - ".join(problems))
