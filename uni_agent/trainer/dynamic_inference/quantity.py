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

from dataclasses import dataclass
from typing import Any

from .types import BorrowPairSpec, BorrowPlan


_Rank = tuple[tuple[float, ...], int, float, int]


@dataclass
class _Candidate:
    """One topology-valid atomic unit that could be lent."""

    order: int
    lender: str
    donor: str
    pair: BorrowPairSpec
    home_replicas: list[Any]


@dataclass
class _Choice:
    """A safe candidate together with its simulated allocation."""

    candidate: _Candidate
    capacity_after: dict[str, int]
    rank: _Rank


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
        demand = self._estimate_demand(sched)
        capacity = {p: sched.initial_cards(p) for p in sched.policies}
        candidates = self._collect_candidates(sched, bottleneck, step)
        active_keys = {
            sched._unit_key(lend.home_policy, lend.donor, lend.home_replicas)
            for lend in sched.active_lends
        }

        borrows: list[BorrowPlan] = []
        while candidates:
            choice = self._choose_next(
                sched,
                candidates,
                demand,
                capacity,
                active_keys,
            )
            if choice is None:
                break

            selected = choice.candidate
            capacity = choice.capacity_after
            borrows.append(BorrowPlan(
                home_policy=selected.lender,
                donor=selected.donor,
                home_replicas=list(selected.home_replicas),
            ))
            candidates = self._without_overlaps(candidates, selected)

        return borrows

    @staticmethod
    def _estimate_demand(sched: Any) -> dict[str, float]:
        """Freeze current work before rebuilding from the all-home baseline."""
        return {
            policy: max(0.0, sched.ema_kv[policy]) * sched.capacity_cards(policy)
            for policy in sched.policies
        }

    @staticmethod
    def _collect_candidates(
        sched: Any,
        bottleneck: str,
        step: int,
    ) -> list[_Candidate]:
        """Return topology-valid units for all currently hot donors."""
        eligible_donors = {
            policy for policy in sched.policies
            if sched.ema_kv[policy] >= sched.ru.kv_enter
        }
        # The hysteresis FSM may retain a bottleneck below the enter threshold.
        eligible_donors.add(bottleneck)

        candidates: list[_Candidate] = []
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
            # Active units are eligible because the boundary returns them before
            # executing the newly selected target allocation.
            units = sched._available_home_units(lender, donor, include_returning=True)
            for unit in units[:max_units]:
                candidates.append(_Candidate(
                    order=len(candidates),
                    lender=lender,
                    donor=donor,
                    pair=pair,
                    home_replicas=list(unit),
                ))
        return candidates

    def _choose_next(
        self,
        sched: Any,
        candidates: list[_Candidate],
        demand: dict[str, float],
        capacity: dict[str, int],
        active_keys: set[tuple],
    ) -> _Choice | None:
        """Choose the safe unit with the best post-borrow allocation."""
        choices: list[_Choice] = []
        for candidate in candidates:
            choice = self._evaluate(
                sched,
                candidate,
                demand,
                capacity,
                active_keys,
            )
            if choice is not None:
                choices.append(choice)
        return min(choices, key=lambda choice: choice.rank, default=None)

    def _evaluate(
        self,
        sched: Any,
        candidate: _Candidate,
        demand: dict[str, float],
        capacity: dict[str, int],
        active_keys: set[tuple],
    ) -> _Choice | None:
        """Simulate one candidate, rejecting it if the lender becomes unsafe."""
        lender = candidate.lender
        donor = candidate.donor
        home_cards = (
            len(candidate.home_replicas)
            * sched.handles[lender].cards_per_replica
        )
        home_capacity_after = capacity[lender] - home_cards

        minimum_home_capacity = sched.handles[lender].cards_per_replica
        if home_capacity_after < minimum_home_capacity:
            return None
        if demand[lender] / home_capacity_after > sched.ru.kv_post_lend_max:
            return None

        capacity_after = dict(capacity)
        capacity_after[lender] = home_capacity_after
        capacity_after[donor] += (
            candidate.pair.guest_replicas_per_unit
            * sched.handles[donor].cards_per_replica
        )

        unit_key = sched._unit_key(lender, donor, candidate.home_replicas)
        utilisation_score = self._utilisation_score(demand, capacity_after)
        active_penalty = 0 if unit_key in active_keys else 1
        migration_cost = sched._unit_inflight_key(
            lender,
            candidate.home_replicas,
        )
        rank: _Rank = (
            utilisation_score,
            active_penalty,
            migration_cost,
            candidate.order,
        )
        return _Choice(candidate, capacity_after, rank)

    @staticmethod
    def _utilisation_score(
        demand: dict[str, float],
        capacity: dict[str, int],
    ) -> tuple[float, ...]:
        """Rank allocations by peak utilisation, then the next highest."""
        return tuple(sorted(
            (demand[policy] / capacity[policy] for policy in demand),
            reverse=True,
        ))

    @staticmethod
    def _without_overlaps(
        candidates: list[_Candidate],
        selected: _Candidate,
    ) -> list[_Candidate]:
        """Remove units sharing a physical home replica with the selection."""
        selected_ids = {id(replica) for replica in selected.home_replicas}
        return [
            candidate
            for candidate in candidates
            if selected_ids.isdisjoint(
                id(replica) for replica in candidate.home_replicas
            )
        ]
