"""Decision layer: composite serving-load detection and stable allocation.

Quantity output is treated as a desired allocation. Target confirmation,
post-change settling, minimum lend duration and in-place renewal prevent
discrete replica rounding from causing repeated swaps:

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
from collections import Counter
from typing import Any, Callable

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
        on_early_return: Callable[[LendRecord], None] | None = None,
    ):
        self.config = config
        self.ru = config.resource_usage
        self.handles = handles
        self.signals = signals
        self.guest_units = guest_units
        self._on_early_return = on_early_return
        self.policies = list(handles)
        self.pairs = expand_pairs(config, self.policies)
        self._pair_by_policies: dict[tuple[str, str], BorrowPairSpec] = {
            (pair.home, pair.donor): pair for pair in self.pairs
        }

        # ---- bottleneck state machine ----
        self.bottleneck: str | None = None
        self.confirm: dict[str, int] = {}
        self._pending_target: tuple | None = None
        self._pending_target_polls = 0
        self._decision_count = 0
        self._lend_start_poll: dict[int, int] = {}
        self._settle_polls_remaining = 0
        self._borrow_cooldown_until = 0.0
        self._return_low_polls: dict[str, int] = {p: 0 for p in self.policies}
        self._normal_return_ready_ids: set[int] = set()

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
            "swap_episodes": 0, "renewals": 0, "early_returns": 0,
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
    def poll(self, *, metrics_fresh: bool = True) -> bool:
        """Continuous signal sampling (poll thread; cheap, read-only).

        Refreshes the KV EMA and checks home exhaustion for early returns.
        Early-return *execution* goes through the controller's gated callback;
        if a boundary is active the controller defers it instead.

        Returns whether every policy supplied a valid fresh KV sample. Only a
        complete sample should drive an immediate rebalance decision.
        """
        if self.disabled:
            return False
        complete_scrape = False
        if metrics_fresh:
            a = self.ru.ema_alpha
            latest_has_kv = getattr(self.signals, "latest_has_kv", None)
            complete_scrape = True
            for p in self.policies:
                if callable(latest_has_kv) and not latest_has_kv(p):
                    complete_scrape = False
                    continue
                kv = float(self.signals.kv_util(p, _STAT_HOW))
                if not math.isfinite(kv) or not 0.0 <= kv <= 1.0:
                    complete_scrape = False
                    logger.warning(
                        "dynamic_inference: ignoring invalid KV utilisation for %s: %r",
                        p, kv,
                    )
                    continue
                if p not in self._ema_initialized:
                    self.ema_kv[p] = kv
                    self._ema_initialized.add(p)
                else:
                    self.ema_kv[p] = a * kv + (1 - a) * self.ema_kv[p]
                waiting_fn = getattr(self.signals, "waiting_per_replica", None)
                self.waiting_per_replica[p] = (
                    float(waiting_fn(p, "mean")) if callable(waiting_fn) else 0.0
                )
                histogram_fn = getattr(self.signals, "histogram_window_quantile", None)
                histogram_wait = (
                    histogram_fn(
                        p,
                        ("request_queue_time_seconds", "request_waiting_time_seconds"),
                        0.9,
                    )
                    if callable(histogram_fn) else None
                )
                streak_fn = getattr(self.signals, "waiting_streak_s", None)
                streak_wait = (
                    float(streak_fn(p, self.ru.waiting_enter_per_replica))
                    if callable(streak_fn) else 0.0
                )
                self.queue_time_s[p] = max(float(histogram_wait or 0.0), streak_wait)
            if complete_scrape and self._settle_polls_remaining > 0:
                self._settle_polls_remaining -= 1
        if complete_scrape:
            self.trace.emit(self, "sample", getattr(self, "_trace_step", 0))
            self._check_early_return()
            self._update_normal_return_confirmations()
        return complete_scrape

    def _check_early_return(self) -> None:
        """Return one lend when its home is persistently KV-saturated."""
        returned_homes = set()
        for lend in list(self.active_lends):
            if lend.home_policy in returned_homes:
                continue
            if self._is_overloaded(lend.home_policy):
                lend.home_kv_hot_polls += 1
                if lend.home_kv_hot_polls >= self.config.early_return_confirm_polls:
                    self._do_early_return(lend)
                    returned_homes.add(lend.home_policy)
            else:
                lend.home_kv_hot_polls = 0

    def _do_early_return(self, lend: LendRecord) -> None:
        self.trace.emit(self, "early_return_requested", getattr(self, "_trace_step", 0),
                        lend_id=lend.lend_id, lender=lend.home_policy, borrower=lend.donor,
                        reason="lender_composite_load_above_enter")
        logger.info(
            "dynamic_inference: early return %s->%s (home exhausted while lent out)",
            lend.home_policy, lend.donor,
        )
        self._counters["early_returns"] += 1
        if self._on_early_return is not None:
            self._on_early_return(lend)

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
        plan = BoundaryPlan()
        step = stats.step
        self._trace_step = step
        self._decision_count += 1

        if self.disabled:
            plan.returns = list(self.active_lends)
            return plan

        # A failed return may have already removed routing or put engines to
        # sleep. Resume it before making any allocation decision; unaffected
        # lends stay active in their owning policy's regular weight update.
        recovering = [lend for lend in self.active_lends if lend.return_stage > 0]
        if recovering:
            plan.returns = recovering
            plan.renewals = [lend for lend in self.active_lends if lend.return_stage == 0]
            return plan

        # ---------- layer 1: bottleneck detection ----------
        bn = self._bottleneck_fsm(self._pick_candidate(), step)

        # Ignore transient post-swap samples until enough fresh scrapes arrive.
        # Safety-driven early returns are evaluated independently in poll().
        if self._settle_polls_remaining > 0:
            plan.renewals = list(self.active_lends)
            return plan

        desired = []
        if bn is not None:
            desired = self.quantity.plan(self, bn, step)
        desired = self._guard_normal_returns(desired)
        desired = self._limit_new_borrows(desired)

        signature = self._target_signature(desired)
        if signature != self._pending_target:
            self._pending_target = signature
            self._pending_target_polls = 1
        else:
            self._pending_target_polls += 1

        current = {self._lend_key(lend): lend for lend in self.active_lends}
        if self._pending_target_polls < self.config.rebalance_confirm_polls:
            ready = [
                lend for lend in current.values()
                if id(lend) in self._normal_return_ready_ids
                and self._decision_count - self._lend_start_poll.get(id(lend), 0)
                >= self.config.min_lend_polls
            ]
            plan.returns = ready
            plan.renewals = [lend for lend in current.values() if lend not in ready]
            return plan

        desired_by_key = {self._plan_key(item): item for item in desired}
        retained_keys = set(desired_by_key) & set(current)

        # A new target cannot revoke a young lend. Load-driven early return is
        # a safety path and deliberately bypasses this minimum duration. Hold
        # the whole allocation while any undesired lend is young; mixing that
        # forced incumbent with additions from the new target could over-assign
        # capacity relative to the quantity calculation.
        if any(
            key not in retained_keys
            and self._decision_count - self._lend_start_poll.get(id(lend), 0)
            < self.config.min_lend_polls
            for key, lend in current.items()
        ):
            plan.renewals = list(current.values())
            return plan

        for key, lend in current.items():
            if key in retained_keys:
                plan.renewals.append(lend)
                retained_keys.add(key)
            else:
                plan.returns.append(lend)

        retained_replicas = {
            id(replica)
            for lend in plan.renewals
            for replica in lend.home_replicas
        }
        for key, borrow in desired_by_key.items():
            if key not in retained_keys and not any(
                id(replica) in retained_replicas for replica in borrow.home_replicas
            ):
                plan.borrows.append(borrow)
        return plan

    def _guard_normal_returns(self, desired: list[BorrowPlan]) -> list[BorrowPlan]:
        """Allow only confirmed, post-return-safe ordinary returns.

        Missing incumbents are proposed returns. For each donor, remove as many
        of those units as possible while its demand divided by remaining cards
        stays at or below ``kv_post_lend_max``. If any proposed return is held,
        suppress additions for this decision because the quantity plan assumed
        those cards had already moved back home.
        """
        desired_keys = {self._plan_key(plan) for plan in desired}
        self._normal_return_ready_ids.clear()
        proposed = [
            lend for lend in self.active_lends
            if self._lend_key(lend) not in desired_keys
        ]
        if not proposed:
            return desired

        allowed_ids: set[int] = set()
        blocked = False
        by_donor: dict[str, list[LendRecord]] = {}
        for lend in proposed:
            by_donor.setdefault(lend.donor, []).append(lend)

        for donor, lends in by_donor.items():
            if self._return_low_polls[donor] < self.config.return_confirm_polls:
                blocked = True
                continue
            capacity = self.capacity_cards(donor)
            demand = max(0.0, self.ema_kv[donor]) * capacity
            ordered = sorted(
                lends,
                key=lambda lend: (
                    self._lend_guest_cards(lend), lend.since_step, lend.lend_id,
                ),
            )
            for lend in ordered:
                capacity_after = capacity - self._lend_guest_cards(lend)
                if (
                    capacity_after <= 0
                    or demand / capacity_after > self.ru.kv_post_lend_max
                ):
                    blocked = True
                    continue
                allowed_ids.add(id(lend))
                capacity = capacity_after
        self._normal_return_ready_ids = allowed_ids

        active_keys = {self._lend_key(lend) for lend in self.active_lends}
        result = list(desired)
        for lend in proposed:
            if id(lend) not in allowed_ids:
                result.append(self._borrow_plan_for_lend(lend))
        if blocked:
            result = [plan for plan in result if self._plan_key(plan) in active_keys]
        return result

    def _lend_guest_cards(self, lend: LendRecord) -> int:
        return len(lend.guests) * self.handles[lend.donor].cards_per_replica

    @staticmethod
    def _borrow_plan_for_lend(lend: LendRecord) -> BorrowPlan:
        return BorrowPlan(
            home_policy=lend.home_policy,
            donor=lend.donor,
            home_replicas=list(lend.home_replicas),
        )

    def _limit_new_borrows(self, desired: list[BorrowPlan]) -> list[BorrowPlan]:
        """Retain incumbents but admit at most one new atomic unit.

        A successful physical borrow starts a global wall-clock cooldown. While
        it is active, no new units are admitted; desired removals still flow
        through so normal and safety returns are never delayed by the cooldown.
        """
        active_keys = {self._lend_key(lend) for lend in self.active_lends}
        additions = [plan for plan in desired if self._plan_key(plan) not in active_keys]
        incumbents = [
            plan for plan in desired if self._plan_key(plan) in active_keys
        ]
        if additions and time.monotonic() >= self._borrow_cooldown_until:
            incumbents.append(additions[0])
        return incumbents

    @staticmethod
    def _unit_key(home: str, donor: str, replicas: list[Any]) -> tuple:
        return home, donor, tuple(sorted(id(replica) for replica in replicas))

    def _plan_key(self, plan) -> tuple:
        return self._unit_key(plan.home_policy, plan.donor, plan.home_replicas)

    def _lend_key(self, lend: LendRecord) -> tuple:
        return self._unit_key(lend.home_policy, lend.donor, lend.home_replicas)

    @staticmethod
    def _target_signature(plans) -> tuple:
        """Confirm quantities, not transient replica choices within a pair."""
        counts = Counter((plan.home_policy, plan.donor) for plan in plans)
        return tuple(sorted((home, donor, count) for (home, donor), count in counts.items()))

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

    def _bottleneck_fsm(self, target: str | None, step: int) -> str | None:
        """Enter on ``bottleneck_confirm_polls`` consecutive confirmations;
        exit once the current bottleneck reaches ``kv_exit`` or below."""
        cur = self.bottleneck
        if target == cur:
            self.confirm.clear()
            return cur
        if target is None:
            if cur is not None and self._is_relieved(cur):
                self._log_bn_change(None, step)
                self.bottleneck = None
            self.confirm.clear()
            return self.bottleneck
        # candidate differs from the current bottleneck
        for k in list(self.confirm):
            if k != target:
                del self.confirm[k]
        self.confirm[target] = self.confirm.get(target, 0) + 1
        if self.confirm[target] >= self.config.bottleneck_confirm_polls:
            self._log_bn_change(target, step)
            self.bottleneck = target
            self.confirm.clear()
        return self.bottleneck

    def _log_bn_change(self, new_bn, step):
        if new_bn != self.bottleneck:
            self._log_event("bottleneck_change", step, "", new_bn or "",
                            {"from": self.bottleneck, "to": new_bn})

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
        self._return_low_polls[lend.donor] = 0
        self._settle_polls_remaining = self.config.rebalance_settle_polls
        self._counters["swap_episodes"] += 1
        self._log_event("borrow", lend.since_step, lend.home_policy, lend.donor,
                        {"lend_id": lend.lend_id})

    def note_lend_returned(
        self, lend: LendRecord, step: int, early: bool = False,
    ) -> None:
        if lend in self.active_lends:
            self.active_lends.remove(lend)
        self._lend_start_poll.pop(id(lend), None)
        self._return_low_polls[lend.donor] = 0
        self._settle_polls_remaining = self.config.rebalance_settle_polls
        if not early:
            self._counters["returns"] += 1
        self._log_event("early_return" if early else "return", step,
                        lend.home_policy, lend.donor,
                        {"lend_id": lend.lend_id,
                         "held_s": time.monotonic() - self.trace.started.pop(lend.lend_id)
                         if lend.lend_id in self.trace.started else None})

    def note_lend_renewed(self, lend: LendRecord, step: int) -> None:
        self._counters["renewals"] += 1
        self._log_event("renew", step, lend.home_policy, lend.donor,
                        {"lend_id": lend.lend_id})

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
        snap["bottleneck"] = self.bottleneck or ""
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
