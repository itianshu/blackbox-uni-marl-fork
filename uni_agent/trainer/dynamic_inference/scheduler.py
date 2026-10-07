"""Decision layer: composite serving-load detection and stable allocation.

Each decision selects one safe ordinary return or one new borrow. Pair
confirmation, post-change settling, minimum lend duration and cooldown
prevent repeated swaps:

- **Signal**: p90 over a window of per-policy replica-mean KV gauges, followed
  by an EMA. Sustained queue depth plus queue duration is an alternative entry
  signal when KV is low.
- **Quantity**: each event adds at most one topology-valid atomic replica unit.

The scheduler is a plain in-process class (no Ray actor): fresh metrics drive
decisions on the poll thread, while the controller's gate serializes physical
changes with step-boundary weight updates. The executor remains in the trainer
process because ``CheckpointEngineManager`` holds a non-serializable
``actor_wg``.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from typing import Any

from .types import (
    BorrowPairSpec,
    BorrowPlan,
    BoundaryPlan,
    LendEvent,
    LendRecord,
    PolicyInferenceHandles,
    SchedulingConfig,
    StepStats,
    expand_pairs,
)

logger = logging.getLogger(__name__)

_STAT_HOW = "p90"


class MultiPolicyInferenceScheduler:
    """When to borrow / return — never how (that is the executor's job).

    Dependencies are duck-typed so the whole layer is CPU-unit-testable:
    ``handles`` (PolicyInferenceHandles per policy), ``signals`` (SignalStore),
    ``guest_units`` (anything with ``unit_for(home, donor, home_replica)``).
    """

    def __init__(
        self,
        config: SchedulingConfig,
        handles: dict[str, PolicyInferenceHandles],
        signals: Any,
        guest_units: Any,
    ):
        self.config = config
        self.ru = config.resource_usage
        self.handles = handles
        self.signals = signals
        self.guest_units = guest_units
        self.policies = list(handles)
        self.pairs = expand_pairs(config, self.policies)
        self._pair_by_policies: dict[tuple[str, str], BorrowPairSpec] = {
            (pair.home, pair.donor): pair for pair in self.pairs
        }

        # ---- consecutive safe borrow-pair confirmation ----
        self._pending_borrow_pair: tuple[str, str] | None = None
        self._pending_borrow_polls = 0
        self._decision_count = 0
        self._lend_start_poll: dict[int, int] = {}
        self._settle_polls_remaining = 0
        self._borrow_cooldown_until = 0.0
        self._return_low_polls: dict[str, int] = {p: 0 for p in self.policies}

        # ---- borrow quantity ----
        from .quantity import SingleUnitStrategy

        self.quantity = SingleUnitStrategy()

        # ---- signal smoothing ----
        self.ema_kv: dict[str, float] = {p: 0.0 for p in self.policies}
        self.waiting_per_replica: dict[str, float] = {p: 0.0 for p in self.policies}
        self.queue_time_s: dict[str, float] = {p: 0.0 for p in self.policies}
        self._ema_initialized: set[str] = set()

        # ---- lend registry (single writer: this class, under the boundary gate) ----
        self.active_lends: list[LendRecord] = []
        self._lend_seq = itertools.count(1)

        # ---- failure blocking (executor writes after failed borrows/returns) ----
        self.pair_block_until: dict[tuple[str, str], int] = {}   # (home, donor) -> step

        # ---- observation ----
        self.events: list[LendEvent] = []
        self._counters = {
            "swap_episodes": 0, "early_returns": 0,
            "returns": 0, "disabled_by_failures": 0,
        }
        self.disabled = False   # repeated execution failures -> behave like static
        from .observability import SchedulingTrace
        self.trace = SchedulingTrace(config)

    # ================================================================ helpers
    def initial_cards(self, policy: str) -> int:
        handle = self.handles[policy]
        return len(handle.replicas) * handle.cards_per_replica

    def capacity_cards(self, policy: str) -> int:
        """Current serving capacity in cards (home awake + active guests)."""
        cards = self.initial_cards(policy)
        for lend in self.active_lends:
            if lend.home_policy == policy:
                cards -= len(lend.home_replicas) * self.handles[policy].cards_per_replica
            if lend.donor == policy:
                cards += len(lend.guests) * self.handles[policy].cards_per_replica
        return cards

    def pair_for(self, home: str, donor: str) -> BorrowPairSpec | None:
        return self._pair_by_policies.get((home, donor))

    def _lent_replica_ids(self, policy: str | None = None) -> set[int]:
        return {
            id(replica)
            for lend in self.active_lends
            if policy is None or lend.home_policy == policy
            for replica in lend.home_replicas
        }

    # ================================================================ polling
    def poll(self) -> bool:
        """Refresh signals under the poll gate; return whether all policies are ready."""
        if self.disabled:
            return False
        metrics_ready = True
        for policy in self.policies:
            # Update every policy even if an earlier sample was invalid.
            if not self._update_policy_signals(policy):
                metrics_ready = False
        if not metrics_ready:
            return False

        if self._settle_polls_remaining > 0:
            self._settle_polls_remaining -= 1
        self.trace.emit(self, "sample", getattr(self, "_trace_step", 0))
        if self._settle_polls_remaining > 0:
            return True
        self._update_early_return_confirmations()
        self._update_normal_return_confirmations()
        return True

    def _update_policy_signals(self, policy: str) -> bool:
        """Update one policy's load signals when its latest sample contains valid KV data."""
        if not self.signals.latest_has_kv(policy):
            return False
        kv = float(self.signals.kv_util(policy, _STAT_HOW))
        if not math.isfinite(kv) or not 0.0 <= kv <= 1.0:
            logger.warning(
                "dynamic_inference: ignoring invalid KV utilisation for %s: %r",
                policy, kv,
            )
            return False

        if policy not in self._ema_initialized:
            self.ema_kv[policy] = kv
            self._ema_initialized.add(policy)
        else:
            a = self.ru.ema_alpha
            self.ema_kv[policy] = a * kv + (1 - a) * self.ema_kv[policy]
        self.waiting_per_replica[policy] = float(
            self.signals.waiting_per_replica(policy, "mean")
        )
        histogram_wait = self.signals.histogram_window_quantile(
            policy,
            ("request_queue_time_seconds", "request_waiting_time_seconds"),
            0.9,
        )
        streak_wait = self.signals.waiting_streak_s(
            policy, self.ru.waiting_enter_per_replica,
        )
        self.queue_time_s[policy] = max(float(histogram_wait or 0.0), float(streak_wait))
        return True

    def _update_early_return_confirmations(self) -> None:
        """Count overload samples, confirming at most one lend per home per poll."""
        confirmed_homes = set()
        for lend in self.active_lends:
            if lend.home_policy in confirmed_homes:
                continue
            if self._is_overloaded(lend.home_policy):
                lend.home_kv_hot_polls += 1
                if lend.home_kv_hot_polls >= self.config.early_return_confirm_polls:
                    confirmed_homes.add(lend.home_policy)
            else:
                lend.home_kv_hot_polls = 0

    def decide_early_returns(self) -> list[LendRecord]:
        """Select at most one confirmed safety return after the settle window."""
        if self.disabled or self._settle_polls_remaining > 0:
            return []
        for lend in self.active_lends:
            if lend.home_kv_hot_polls >= self.config.early_return_confirm_polls:
                return [lend]
        return []

    def _update_normal_return_confirmations(self) -> None:
        """Count consecutive fully-relieved samples for policies using guests."""
        active_donors = {lend.donor for lend in self.active_lends}
        for policy in self.policies:
            if policy in active_donors and self._is_relieved(policy):
                self._return_low_polls[policy] += 1
            else:
                self._return_low_polls[policy] = 0

    # ================================================================ decide
    def decide(self, stats: StepStats) -> BoundaryPlan:
        """Select one ordinary return, otherwise one confirmed new borrow."""
        step = stats.step
        self._trace_step = step
        self._decision_count += 1

        if self.disabled:
            self._confirm_borrow(None)
            return BoundaryPlan(returns=self.active_lends[:1])
        if self._settle_polls_remaining > 0:
            return BoundaryPlan()

        lend = self._select_normal_return()
        if lend is not None:
            self._confirm_borrow(None)
            return BoundaryPlan(returns=[lend])

        if time.monotonic() < self._borrow_cooldown_until:
            self._confirm_borrow(None)
            return BoundaryPlan()

        candidate = self._pick_candidate()
        borrow = self.quantity.select_borrow(self, candidate, step)
        if self._confirm_borrow(borrow):
            return BoundaryPlan(borrows=[borrow])
        return BoundaryPlan()

    def _select_normal_return(self) -> LendRecord | None:
        """Return one mature, confirmed and individually safe lend.

        Prefer smaller units, then older lends. Evaluate against current
        capacity only; all other lends remain active until a later decision.
        """
        ordered = sorted(
            self.active_lends,
            key=lambda lend: (
                self._lend_guest_cards(lend), lend.since_step, lend.lend_id,
            ),
        )
        for lend in ordered:
            donor = lend.donor
            if (
                not self._is_relieved(donor)
                or self._return_low_polls[donor] < self.config.return_confirm_polls
            ):
                continue
            age = self._decision_count - self._lend_start_poll.get(id(lend), 0)
            if age < self.config.min_lend_polls:
                continue
            capacity = self.capacity_cards(donor)
            capacity_after = capacity - self._lend_guest_cards(lend)
            demand = max(0.0, self.ema_kv[donor]) * capacity
            if capacity_after > 0 and demand / capacity_after <= self.ru.kv_post_lend_max:
                return lend
        return None

    def _lend_guest_cards(self, lend: LendRecord) -> int:
        return len(lend.guests) * self.handles[lend.donor].cards_per_replica

    def _confirm_borrow(self, borrow: BorrowPlan | None) -> bool:
        """Confirm the directed pair, not a particular replica or full allocation."""
        pair = (borrow.home_policy, borrow.donor) if borrow is not None else None
        if pair is None:
            self._pending_borrow_pair = None
            self._pending_borrow_polls = 0
            return False
        if pair != self._pending_borrow_pair:
            self._pending_borrow_pair = pair
            self._pending_borrow_polls = 1
        else:
            self._pending_borrow_polls += 1
        return self._pending_borrow_polls >= self.config.rebalance_confirm_polls

    # ------------------------------------------------- layer 1: candidates
    def _pick_candidate(self) -> str | None:
        """Highest composite-load policy satisfying an entry condition."""
        cands = [p for p in self.policies if self._is_overloaded(p)]
        if not cands:
            return None
        return max(cands, key=self._load_score)

    def _queue_overloaded(self, policy: str) -> bool:
        return self.ru.queue_signal_enabled and (
            self.waiting_per_replica[policy] >= self.ru.waiting_enter_per_replica
            and self.queue_time_s[policy] >= self.ru.queue_time_enter_s
        )

    def _is_overloaded(self, policy: str) -> bool:
        return self.ema_kv[policy] >= self.ru.kv_enter or self._queue_overloaded(policy)

    def _is_relieved(self, policy: str) -> bool:
        if self.ema_kv[policy] > self.ru.kv_exit:
            return False
        return not self.ru.queue_signal_enabled or (
            self.waiting_per_replica[policy] <= self.ru.waiting_exit_per_replica
            and self.queue_time_s[policy] <= self.ru.queue_time_exit_s
        )

    def _load_score(self, policy: str) -> float:
        """Comparable pressure score used only to rank eligible policies."""
        kv = self.ema_kv[policy] / self.ru.kv_enter
        queue = 0.0
        if self.ru.queue_signal_enabled:
            queue = min(
                self.waiting_per_replica[policy] / self.ru.waiting_enter_per_replica,
                self.queue_time_s[policy] / self.ru.queue_time_enter_s,
            )
        return max(kv, queue)

    def _available_home_units(
        self,
        policy: str,
        donor: str,
        *,
        include_returning: bool = False,
    ) -> list[list[Any]]:
        """Topology-valid units ordered by migration cost.

        ``include_returning`` includes units in ``active_lends`` when planning
        a boundary: those units are returned before the new borrow plan runs.
        """
        pair = self.pair_for(policy, donor)
        if pair is None:
            return []
        lent = self._lent_replica_ids()
        groups = None
        candidate_units = getattr(self.guest_units, "candidate_units", None)
        if callable(candidate_units):
            groups = candidate_units(policy, donor)
        if groups is None:
            replicas = self._available_home_replicas(
                policy,
                include_returning=include_returning,
            )
            n = pair.home_replicas_per_unit
            groups = [replicas[i:i + n] for i in range(0, len(replicas) - n + 1, n)]
        groups = [
            list(group)
            for group in groups
            if group and (include_returning or not any(id(r) in lent for r in group))
        ]
        return sorted(groups, key=lambda g: self._unit_inflight_key(policy, g))

    def _available_home_replicas(
        self,
        policy: str,
        *,
        include_returning: bool = False,
    ) -> list[Any]:
        """Home replicas available for planning, cheapest to migrate.

        Ordered by per-replica in-flight ascending — in-flight proxies
        migration cost (abort + cold re-prefill of the unit's requests).
        ``include_returning`` also includes replicas whose current lend is
        scheduled to return before the new plan executes.
        """
        handle = self.handles[policy]
        lent = self._lent_replica_ids()
        avail = [
            r for r in handle.replicas
            if include_returning or id(r) not in lent
        ]
        per_replica = self.signals.stat_per_replica_inflight(policy, _STAT_HOW)

        def inflight_of(r) -> float:
            return per_replica.get(getattr(r, "server_address", None), 0.0)

        return sorted(avail, key=inflight_of)

    def _unit_inflight_key(self, lender: str, unit: list[Any]) -> float:
        per_replica = self.signals.stat_per_replica_inflight(lender, _STAT_HOW)
        return sum(per_replica.get(getattr(r, "server_address", None), 0.0) for r in unit)

    # ------------------------------------------------- lend registry hooks
    # (called by the executor/controller after physical success — the
    # scheduler stays the single writer of active_lends under the gate)
    def note_borrow_started(self, lend: LendRecord) -> None:
        self.trace.started[lend.lend_id] = time.monotonic()
        self.active_lends.append(lend)
        self._lend_start_poll[id(lend)] = self._decision_count
        self._borrow_cooldown_until = max(
            self._borrow_cooldown_until,
            time.monotonic() + self.config.borrow_cooldown_s,
        )
        self._start_settle()
        self._counters["swap_episodes"] += 1
        self._log_event("borrow", lend.since_step, lend.home_policy, lend.donor,
                        {"lend_id": lend.lend_id})

    def note_lend_returned(
        self, lend: LendRecord, step: int, early: bool = False,
    ) -> None:
        if lend in self.active_lends:
            self.active_lends.remove(lend)
        self._lend_start_poll.pop(id(lend), None)
        self._start_settle()
        if not early:
            self._counters["returns"] += 1
        else:
            self._counters["early_returns"] += 1
        self._log_event("early_return" if early else "return", step,
                        lend.home_policy, lend.donor,
                        {"lend_id": lend.lend_id,
                         "held_s": time.monotonic() - self.trace.started.pop(lend.lend_id)
                         if lend.lend_id in self.trace.started else None})

    def _start_settle(self) -> None:
        """Discard pre-change confirmations and observe the new allocation."""
        self._settle_polls_remaining = self.config.rebalance_settle_polls
        self._pending_borrow_pair = None
        self._pending_borrow_polls = 0
        self._return_low_polls = dict.fromkeys(self.policies, 0)
        for lend in self.active_lends:
            lend.home_kv_hot_polls = 0

    def next_lend_id(self) -> int:
        return next(self._lend_seq)

    def note_disabled_by_failures(self) -> None:
        if self.disabled:
            return
        self._counters["disabled_by_failures"] += 1
        self.disabled = True

    # ------------------------------------------------- observation
    def _log_event(self, kind, step, home, donor, detail):
        self.trace.emit(self, kind, step, lender=home, borrower=donor, **detail)
        self.events.append(LendEvent(kind=kind, step=step, t=round(time.time(), 3),
                                     home=home, donor=donor, detail=detail))
        if len(self.events) > 5000:
            del self.events[:2500]

    def metrics_snapshot(self) -> dict:
        snap = dict(self._counters)
        snap["active_lends"] = len(self.active_lends)
        snap["pending_borrow_pair"] = self._pending_borrow_pair
        snap["borrow_confirmations"] = self._pending_borrow_polls
        snap["disabled"] = self.disabled
        snap["kv_util"] = {p: round(self.ema_kv[p], 3) for p in self.policies}
        snap["waiting_per_replica"] = {
            p: round(self.waiting_per_replica[p], 3) for p in self.policies
        }
        snap["queue_time_s"] = {
            p: round(self.queue_time_s[p], 3) for p in self.policies
        }
        # Keep the aggregate mapping for programmatic inspection and expose
        # scalar values too: console/W&B loggers commonly discard nested maps.
        for policy in self.policies:
            snap[f"kv_util/{policy}"] = round(self.ema_kv[policy], 6)
            snap[f"waiting_per_replica/{policy}"] = round(
                self.waiting_per_replica[policy], 6,
            )
            snap[f"queue_time_s/{policy}"] = round(self.queue_time_s[policy], 6)
        snap["borrow_cooldown_remaining_s"] = round(
            max(0.0, self._borrow_cooldown_until - time.monotonic()), 3,
        )
        snap["return_low_polls"] = dict(self._return_low_polls)
        snap["vllm_metrics"] = {
            policy: self.signals.vllm_observability(policy)
            for policy in self.policies
        } if self.signals is not None else {}
        return snap
