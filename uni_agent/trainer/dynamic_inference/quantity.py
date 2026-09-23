"""Select at most one safe atomic replica unit for a borrow event.

The scheduler already decides whether a policy is overloaded and provides
hysteresis, confirmation, settling and cooldown. Quantity planning therefore
does not solve a multi-unit equalisation problem. It retains necessary
incumbents and selects the safest single available unit, or selects nothing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .types import BorrowPlan


class SingleUnitStrategy:
    """Retain needed lends and add no more than one safe atomic unit."""

    def plan(self, sched: Any, bottleneck: str, step: int) -> list[BorrowPlan]:
        retained = [
            BorrowPlan(
                home_policy=lend.home_policy,
                donor=lend.donor,
                home_replicas=list(lend.home_replicas),
            )
            for lend in sched.active_lends
            if not sched._is_relieved(lend.donor)
            and not sched._is_overloaded(lend.home_policy)
        ]

        # Hysteresis may retain the bottleneck after its entry condition has
        # cleared. Keep existing capacity in that band, but do not add more.
        if not sched._is_overloaded(bottleneck):
            return retained

        active_per_pair = Counter(
            (lend.home_policy, lend.donor) for lend in sched.active_lends
        )
        choices = []
        for order, pair in enumerate(sched.pairs):
            if pair.donor != bottleneck:
                continue
            edge = (pair.home, pair.donor)
            if step < sched.pair_block_until.get(edge, -1):
                continue
            if pair.max_units is not None and active_per_pair[edge] >= pair.max_units:
                continue
            if sched._is_overloaded(pair.home):
                continue
            for unit in sched._available_home_units(
                pair.home, pair.donor, include_returning=False,
            ):
                if not self._safe_after_lend(sched, pair.home, unit):
                    continue
                choices.append((
                    sched._load_score(pair.home),
                    sched._unit_inflight_key(pair.home, unit),
                    order,
                    BorrowPlan(
                        home_policy=pair.home,
                        donor=pair.donor,
                        home_replicas=list(unit),
                    ),
                ))

        if not choices:
            return retained
        return retained + [min(choices, key=lambda item: item[:3])[3]]

    @staticmethod
    def _safe_after_lend(sched: Any, lender: str, unit: list[Any]) -> bool:
        # Keep a home seed, not just borrowed capacity: an incoming guest can
        # be protectively recalled and must not be a policy's only weight source.
        lent = sched._lent_replica_ids()
        removed = {id(r) for r in unit}
        if not any(id(r) not in lent and id(r) not in removed
                   for r in sched.handles[lender].replicas):
            return False
        removed_cards = len(unit) * sched.handles[lender].cards_per_replica
        current_capacity = sched.capacity_cards(lender)
        capacity_after = current_capacity - removed_cards
        minimum = sched.handles[lender].cards_per_replica
        if capacity_after < minimum:
            return False

        # Freeze observed KV work while evaluating the smaller home capacity.
        demand = max(0.0, sched.ema_kv[lender]) * current_capacity
        return demand / capacity_after <= sched.ru.kv_post_lend_max


# Import compatibility for callers that used the previous strategy name.
EqualisationStrategy = SingleUnitStrategy
