"""Borrow-quantity calculation: how many atomic units to borrow.

The built-in ``EqualisationStrategy`` estimates each policy's outstanding work
as ``utilisation * current serving cards``. Starting from the all-home capacity
baseline, it repeatedly ranks topology-valid directed-edge borrow units by the
resulting utilisation distribution. All currently hot donors participate, so
one home may lend disjoint units over multiple outgoing edges and one donor may
receive multiple incoming edges. Every unit that passes the lender safety
guards is selected; predicted global improvement is not an admission gate.

The strategy follows the scheduler's safety rails: a lender keeps >= 1 home
replica, pairs blocked by execution failures are skipped, and migration cost
breaks ties between otherwise equivalent candidate units.
"""

from __future__ import annotations

from typing import Any

from .types import BorrowPlan


class EqualisationStrategy:
    """Choose the discrete set of units that best equalises utilisation.

    Demand is held constant while capacity moves.  At each iteration every
    remaining atomic N:M unit is simulated, and the unit producing the lowest
    descending utilisation vector is selected first. Lexicographic comparison
    prioritises the peak, then the second-highest load, and so on. Selection
    stops only when candidates are exhausted or lender safety guards reject
    every remaining unit.
    """

    def plan(self, sched: Any, bottleneck: str, step: int) -> list[BorrowPlan]:
        # EMA utilisation describes the capacity serving *now*.  Convert it to
        # fixed demand before modelling the complete target allocation from the
        # all-home baseline.
        demand = {
            p: max(0.0, sched.ema_kv[p]) * sched.capacity_cards(p)
            for p in sched.policies
        }
        capacity = {p: sched.initial_cards(p) for p in sched.policies}

        # One confirmed scheduling episode may serve every currently hot donor,
        # not just the hottest policy used to drive the hysteresis FSM. This
        # lets disjoint units of one home policy use different outgoing edges.
        eligible_donors = {
            policy for policy in sched.policies
            if sched.ema_kv[policy] >= sched.ru.kv_enter
        }
        eligible_donors.add(bottleneck)

        # Enumerate post-return candidate units. Active units are eligible
        # because the boundary returns them before executing this borrow plan.
        candidates = []  # (stable_order, lender, donor, pair, unit_replicas)
        for pair in sched.pairs:
            lender, donor = pair.home, pair.donor
            if donor not in eligible_donors:
                continue
            if step < sched.pair_block_until.get((lender, donor), -1):
                continue
            max_units = max(
                0,
                (len(sched.handles[lender].replicas) - 1) // pair.home_replicas_per_unit,
            )
            units = sched._available_home_units(lender, donor, include_returning=True)
            for unit in units[:max_units]:
                candidates.append((len(candidates), lender, donor, pair, unit))
        if not candidates:
            return []

        relevant = set(sched.policies)

        def score(candidate_capacity: dict[str, int]) -> tuple[float, ...]:
            return tuple(sorted(
                (demand[p] / candidate_capacity[p] for p in relevant),
                reverse=True,
            ))

        borrows: list[BorrowPlan] = []
        active_keys = {
            sched._unit_key(lend.home_policy, lend.donor, lend.home_replicas)
            for lend in sched.active_lends
        }
        used: set[int] = set()
        remaining_candidates = list(candidates)
        while remaining_candidates:
            best = None
            for stable_order, lender, donor, pair, unit in remaining_candidates:
                if any(id(replica) in used for replica in unit):
                    continue
                home_cards = len(unit) * sched.handles[lender].cards_per_replica
                guest_cards = (
                    pair.guest_replicas_per_unit
                    * sched.handles[donor].cards_per_replica
                )
                home_capacity_after = capacity[lender] - home_cards
                if home_capacity_after < sched.handles[lender].cards_per_replica:
                    continue  # retain at least one complete home replica
                home_util_after = demand[lender] / home_capacity_after
                if home_util_after > sched.ru.kv_post_lend_max:
                    continue
                simulated = dict(capacity)
                simulated[lender] = home_capacity_after
                simulated[donor] += guest_cards
                candidate_score = score(simulated)
                rank = (
                    candidate_score,
                    0 if sched._unit_key(lender, donor, unit) in active_keys else 1,
                    sched._unit_inflight_key(lender, unit),
                    stable_order,
                )
                if best is None or rank < best[0]:
                    best = (rank, lender, donor, unit, simulated)
            if best is None:
                break
            _, lender, donor, unit, capacity = best
            borrows.append(BorrowPlan(
                home_policy=lender,
                donor=donor,
                home_replicas=list(unit),
            ))
            used.update(id(replica) for replica in unit)
            remaining_candidates = [
                candidate for candidate in remaining_candidates
                if not any(id(replica) in used for replica in candidate[4])
            ]
        return borrows
