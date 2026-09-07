"""Resolve physical replica placement and build node-compatible borrow units."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from .types import BorrowPairSpec, PolicyInferenceHandles

NodeLayout = tuple[str, ...]


async def replica_node_ids(replica: Any) -> NodeLayout | None:
    """Return one physical node id per worker, preserving resource-pool order."""
    known = getattr(replica, "node_ids", None)
    if known is not None:
        return tuple(map(str, known))

    workers = list(getattr(replica, "workers", []) or [])
    if not workers:
        return None

    import ray

    return tuple(await asyncio.gather(*[
        worker.__ray_call__.remote(
            lambda self: str(ray.get_runtime_context().get_node_id())
        )
        for worker in workers
    ]))


def fits_guest_layout(
    replicas: list[Any],
    layouts: dict[int, NodeLayout],
    guest_cards: int,
    guest_nodes: int,
) -> bool:
    """Check that every guest node-sized block stays on one physical node."""
    flat = tuple(node for replica in replicas for node in layouts[id(replica)])
    if guest_cards < 1 or guest_nodes < 1 or guest_cards % guest_nodes or len(flat) % guest_cards:
        return False

    cards_per_node = guest_cards // guest_nodes
    for guest_start in range(0, len(flat), guest_cards):
        guest = flat[guest_start:guest_start + guest_cards]
        nodes = [
            guest[start]
            for start in range(0, guest_cards, cards_per_node)
            if len(set(guest[start:start + cards_per_node])) == 1
        ]
        if len(nodes) != guest_nodes or len(set(nodes)) != guest_nodes:
            return False
    return True


def topology_groups(
    pair: BorrowPairSpec,
    replicas: list[Any],
    layouts: dict[int, NodeLayout],
    donor: PolicyInferenceHandles,
) -> list[list[Any]]:
    """Build disjoint compatible units, trying same-layout replicas first."""
    remaining = sorted(
        replicas,
        key=lambda replica: (layouts[id(replica)], getattr(replica, "replica_rank", 0)),
    )
    groups = []
    size = pair.home_replicas_per_unit
    while len(remaining) >= size:
        match = next((
            indices
            for indices in itertools.combinations(range(len(remaining)), size)
            if fits_guest_layout(
                [remaining[index] for index in indices], layouts,
                donor.cards_per_replica, donor.nnodes,
            )
        ), None)
        if match is None:
            break
        used = set(match)
        groups.append([remaining[index] for index in match])
        remaining = [replica for index, replica in enumerate(remaining) if index not in used]
    return groups


async def build_candidate_groups(
    handles: dict[str, PolicyInferenceHandles],
    pairs: list[BorrowPairSpec],
) -> dict[tuple[str, str], list[list[Any]]]:
    """Resolve topology once and freeze candidate units for every borrow edge."""
    replicas = {
        id(replica): replica
        for pair in pairs
        for replica in handles[pair.home].replicas
    }
    resolved = dict(zip(
        replicas,
        await asyncio.gather(*[replica_node_ids(replica) for replica in replicas.values()]),
    ))

    result = {}
    for pair in pairs:
        home = handles[pair.home]
        home_replicas = list(home.replicas)
        layouts = [resolved[id(replica)] for replica in home_replicas]
        if all(layout is None for layout in layouts):
            size = pair.home_replicas_per_unit
            groups = [
                home_replicas[index:index + size]
                for index in range(0, len(home_replicas) - size + 1, size)
            ]
        elif any(layout is None for layout in layouts):
            raise RuntimeError(
                f"borrow pair {pair.home}->{pair.donor}: cannot determine "
                "the physical node of every home replica"
            )
        else:
            node_layouts = dict(zip(map(id, home_replicas), layouts))
            if any(len(node_layouts[id(replica)]) != home.cards_per_replica for replica in home_replicas):
                raise RuntimeError(
                    f"borrow pair {pair.home}->{pair.donor}: worker/node metadata "
                    f"does not match {pair.home}'s {home.cards_per_replica}-card replica layout"
                )
            groups = topology_groups(pair, home_replicas, node_layouts, handles[pair.donor])
            if not groups:
                donor = handles[pair.donor]
                raise ValueError(
                    f"borrow pair {pair.home}->{pair.donor}: no node-local group of "
                    f"{pair.home_replicas_per_unit} home replicas can form a "
                    f"{donor.cards_per_replica}-card, {donor.nnodes}-node guest replica"
                )
        result[(pair.home, pair.donor)] = groups
    return result
