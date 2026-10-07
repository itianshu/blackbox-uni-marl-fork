"""Select at most one safe atomic replica unit for a borrow event.

The scheduler already decides whether a policy is overloaded and provides
pair confirmation, settling and cooldown. Quantity planning therefore
does not solve a multi-unit equalisation problem. It selects one safe
available unit using current capacity, without assuming any pending returns.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .types import BorrowPlan


class SingleUnitStrategy:
    """Select one new safe atomic unit; existing lends are managed separately."""

    def select_borrow(self, sched: Any, bottleneck: str | None, step: int) -> BorrowPlan | None:
        if bottleneck is None or not sched._is_overloaded(bottleneck):
            return None

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
            return None
        return min(choices, key=lambda item: item[:3])[3]

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
