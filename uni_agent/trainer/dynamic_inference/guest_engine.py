"""Guest-replica pre-creation and lifecycle.

A *guest* is a donor-architecture vLLM replica pre-created on a borrow unit's
placement groups, holding dummy weights in permanent level-2 sleep. A unit may
aggregate N smaller home replicas into M larger guests, or split one larger
home replica into multiple smaller guests. It belongs to no
``CheckpointEngineManager`` and no load balancer
(invariant I1) until a borrow wakes it, receives a directed weight push, and
joins the donor's LB. In-place renewals refresh its weights at each boundary;
on return it sleeps again without any weight write-back.

Placement (the mechanism verl itself uses for colocated engines): the home
replica's PG is created with ``max_colocate_count = 1 + outgoing_edges``.
Its bundles reserve one CPU slot per worker and one physical GPU; the home and
each direction's sleeping guest worker reserve ``1/max_colocate_count`` GPU.
Thus every configured outgoing direction has a Ray scheduling slot while only
one engine on a physical card is awake at a time.
The existing pools are merged and/or split as views; no new placement group
is created at borrow time.

The pre-creation transition (training has not started, no in-flight requests):

    all participating homes sleep once
      -> initialize guests concurrently where their physical home replicas do
         not overlap; each initialized guest sleeps before releasing its cards
      -> all homes wake once

i.e. home weights are freed for the complete setup window. Replica-scoped
locks prevent two guests from initializing on the same physical cards, while
disjoint groups can initialize in parallel. The initial ``home.sleep()`` also
doubles as the **worker-patch probe**: a real level-2 sleep takes seconds;
verl's unpatched STANDALONE branch returns in milliseconds, which would
silently break every later borrow, so setup fails before the first boundary.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager

from .topology import build_candidate_groups
from .types import (
    BorrowPairSpec,
    GuestUnit,
    PolicyInferenceHandles,
    SchedulingConfig,
    validate_against_handles,
)

logger = logging.getLogger(__name__)

# Compatibility fallback for replica test doubles / older backends without
# server actor handles. Real vLLM actors are checked via engine.is_sleeping().
_SLEEP_PROBE_MIN_S = 0.5
_GUEST_LOAD_FORMAT = "dummy"


def _resource_pool_master_node_id(resource_pool) -> str:
    """Return the node used as rendezvous master by ``RayWorkerGroup``.

    Keep this selection in lockstep with verl's two WorkerGroup init paths:
    a sub-pool starts at its first selected bundle, while a normal pool uses
    the first placement group after sorting by node IP.
    """
    import ray
    from verl.single_controller.ray.base import (
        SubRayResourcePool,
        sort_placement_group_by_node_ip,
    )
    from verl.utils.device import get_device_name

    pgs = resource_pool.get_placement_groups(
        strategy="PACK", device_name=get_device_name(),
    )
    if isinstance(resource_pool, SubRayResourcePool):
        local_world_size = resource_pool.store[0]
        pg_index, bundle_index = divmod(
            resource_pool.start_bundle_index, local_world_size,
        )
        pg = pgs[pg_index]
    else:
        pg = sort_placement_group_by_node_ip(pgs)[0]
        bundle_index = 0

    specs = ray._private.state.state.placement_group_table(pg.id)
    return str(specs["bundles_to_node_id"][bundle_index])


def _set_field(cfg, name: str, value) -> None:
    """Set a field on a dataclass or DictConfig rollout config."""
    try:
        setattr(cfg, name, value)
    except Exception:
        from omegaconf import OmegaConf

        OmegaConf.update(cfg, name, value, force_add=True)


def _set_nested_field(cfg, path: str, value) -> bool:
    """Set a nested config field, returning False when an intermediate is absent.

    Guest rollout configs are deep copies of the donor config.  We must not use
    ``setattr(cfg, "a.b", ...)`` here: for dataclasses that creates an invalid
    attribute and for DictConfig it can create a literal dotted key.  Walking
    the existing nodes keeps this compatible with both config representations.
    """
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        try:
            node = getattr(node, part)
        except (AttributeError, KeyError, TypeError):
            try:
                node = node[part]
            except (KeyError, TypeError, IndexError):
                return False
        if node is None:
            return False
    _set_field(node, parts[-1], value)
    return True


class GuestEngineManager:
    """Owns every pre-created guest unit and its (home, donor, replica) index."""

    def __init__(
        self,
        config: SchedulingConfig,
        handles: dict[str, PolicyInferenceHandles],
        pairs: list[BorrowPairSpec],
    ):
        self.config = config
        self.handles = handles
        self.pairs = list(pairs)
        self.units: list[GuestUnit] = []
        # canonical group key -> GuestUnit
        self._index: dict[tuple[str, str, tuple[int, ...]], GuestUnit] = {}
        self._groups: dict[tuple[str, str], list[list]] = {}
        self._guest_rank = itertools.count(config.borrowing.guest_replica_rank_offset)
        # A port only needs to be unique on the rendezvous master node.  Keep
        # independent counters so different nodes can reuse the same slot.
        self._guest_port_slots_by_node: dict[str, int] = {}

    # ------------------------------------------------------------- lookup
    @staticmethod
    def _unit_key(home: str, donor: str, home_replicas: list) -> tuple:
        return home, donor, tuple(sorted(id(r) for r in home_replicas))

    def unit_for(self, home: str, donor: str, home_replicas: list) -> GuestUnit | None:
        """Look up a complete topology-valid home group."""
        return self._index.get(self._unit_key(home, donor, list(home_replicas)))

    def candidate_units(self, home: str, donor: str) -> list[list]:
        """Return the fixed, topology-valid home groups for one direction."""
        return [list(group) for group in self._groups.get((home, donor), [])]

    # ------------------------------------------------------------- creation
    async def precreate_all(self) -> int:
        """Validate reality constraints, then pre-create guests for every node-local unit."""
        validate_against_handles(self.config, self.handles, self.pairs)
        self._groups = await build_candidate_groups(self.handles, self.pairs)
        units = await self._build_all_units()
        try:
            async with self._homes_sleeping(units) as replica_locks:
                await self._initialize_all_units(units, replica_locks)
        except BaseException:
            self._kill_units(units)
            raise

        self._register_units(units)

        created = len(units)
        logger.info(
            "dynamic_inference.guest_engine: pre-created %d guest units across %d pairs",
            created, len(self.pairs),
        )
        return created

    async def _build_all_units(self) -> list[GuestUnit]:
        units = []
        for pair in self.pairs:
            for replicas in self._groups[(pair.home, pair.donor)]:
                units.append(await self._build_unit(pair, replicas))
        return units

    @asynccontextmanager
    async def _homes_sleeping(self, units: list[GuestUnit]):
        home_replicas = self._unique_home_replicas(units)
        replica_locks = {id(replica): asyncio.Lock() for replica in home_replicas}
        try:
            sleep_times = await self._sleep_all_homes(home_replicas)
            await self._verify_homes_sleeping(units, sleep_times)
            yield replica_locks
        finally:
            await self._wake_all_homes(home_replicas)

    async def _sleep_all_homes(self, replicas: list) -> dict[int, float]:
        try:
            elapsed = await self._gather_or_raise([
                self._timed_sleep(replica) for replica in replicas
            ])
        except Exception as exc:
            raise RuntimeError(
                "failed to sleep homes before guest precreation"
            ) from exc
        return dict(zip(map(id, replicas), elapsed))

    async def _verify_homes_sleeping(
        self,
        units: list[GuestUnit],
        sleep_times: dict[int, float],
    ) -> None:
        for unit in units:
            await self._probe_sleep(
                unit.home_policy,
                unit.home_replicas,
                min(sleep_times[id(replica)] for replica in unit.home_replicas),
            )

    async def _wake_all_homes(self, replicas: list) -> None:
        try:
            await self._gather_or_raise([
                replica.wake_up() for replica in replicas
            ])
        except Exception as exc:
            raise RuntimeError(
                "failed to restore homes after guest precreation"
            ) from exc

    async def _initialize_all_units(
        self,
        units: list[GuestUnit],
        replica_locks: dict[int, asyncio.Lock],
    ) -> None:
        await self._gather_or_raise([
            self._initialize_unit(unit, replica_locks) for unit in units
        ])

    @staticmethod
    async def _gather_or_raise(awaitables: list) -> list:
        results = list(await asyncio.gather(*awaitables, return_exceptions=True))
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]
        return results

    def _kill_units(self, units: list[GuestUnit]) -> None:
        self._kill_guest_actors([
            guest for unit in units for guest in unit.guests
        ])

    def _register_units(self, units: list[GuestUnit]) -> None:
        for unit in units:
            self.units.append(unit)
            self._index[self._unit_key(
                unit.home_policy, unit.donor, unit.home_replicas)] = unit

    def _guest_pools(self, home_replicas: list, guest_count: int, guest_cards: int) -> list:
        """Create resource-pool views that map M guests onto N home pools."""
        pools = [r.resource_pool for r in home_replicas]
        if len(pools) == 1:
            merged = pools[0]
        else:
            from verl.single_controller.ray.base import merge_resource_pool

            merged = pools[0]
            for pool in pools[1:]:
                merged = merge_resource_pool(merged, pool)
        if guest_count == 1:
            return [merged]
        from verl.single_controller.ray.base import split_resource_pool

        return split_resource_pool(merged, [guest_cards] * guest_count)

    def _allocate_guest_master_port_range(self, node_id: str) -> list[int] | None:
        """Allocate a node-local, end-exclusive rendezvous subrange.

        The pool is optional because only the cluster operator knows which
        ports are excluded from Ray workers, service ports, ephemeral ports,
        and other jobs.  When no pool is configured we deliberately leave
        selection to verl instead of embedding a machine-specific default.
        Slots are unique on one master node but reusable on different nodes.
        """
        port_range = self.config.borrowing.guest_master_port_range
        if port_range is None:
            return None
        stride = self.config.borrowing.guest_master_port_stride
        slot = self._guest_port_slots_by_node.get(node_id, 0)
        self._guest_port_slots_by_node[node_id] = slot + 1
        start = port_range[0] + slot * stride
        end = start + stride
        if end > port_range[1]:
            capacity = (port_range[1] - port_range[0]) // stride
            raise RuntimeError(
                "guest master-port pool exhausted on node "
                f"{node_id}: "
                f"range={port_range}, stride={stride}, capacity={capacity}, "
                f"requested_slot={slot}"
            )
        return [start, end]

    async def _make_guest(self, home: str, donor: str, home_replicas: list, pool, guest_idx: int):
        """Build one un-launched guest replica bound to a home-pool view."""
        donor_h = self.handles[donor]
        guest_cfg = copy.deepcopy(donor_h.rollout_config)
        _set_field(guest_cfg, "load_format", _GUEST_LOAD_FORMAT)
        # A guest has its own checkpoint-engine workers, but its config is
        # copied from the donor.  Reusing the donor's persistent NCCL group
        # makes the process-group registry bind a guest rank to the donor
        # topology; the next weight sync then fails with rank-mismatch
        # assertions (e.g. ``rank 1 != self.rank 2``).  Give every guest a
        # stable, unique group while retaining the donor's backend settings.
        guest_rank = next(self._guest_rank)
        _set_nested_field(
            guest_cfg,
            "checkpoint_engine.engine_kwargs.nccl.group_name",
            f"dynamic_guest_{home}_{donor}_{guest_rank}",
        )

        from verl.workers.rollout.replica import get_rollout_replica_class

        guest_cls = get_rollout_replica_class(str(getattr(guest_cfg, "name", "vllm")))
        guest = guest_cls(
            replica_rank=guest_rank,
            config=guest_cfg,
            model_config=donor_h.model_config,
            gpus_per_node=int(getattr(guest_cfg, "n_gpus_per_node", 8) or 8),
            name_suffix=(f"guest_{home}_u"
                         f"{'_'.join(str(r.replica_rank) for r in home_replicas)}_g{guest_idx}"),
        )
        # patch #2: init on the home replica's own PG — no new placement group
        guest._guest_external_pool = pool
        if self.config.borrowing.guest_master_port_range is not None:
            master_node_id = _resource_pool_master_node_id(pool)
            guest._guest_master_node_id = master_node_id
            guest._guest_master_port_range = self._allocate_guest_master_port_range(
                master_node_id,
            )
        return guest

    async def _build_unit(self, pair: BorrowPairSpec, home_replicas: list) -> GuestUnit:
        """Build one uninitialized guest unit without changing home state."""
        home, donor = pair.home, pair.donor
        home_replicas = list(home_replicas)
        if self.unit_for(home, donor, home_replicas) is not None:
            raise RuntimeError(f"guest unit for {home}->{donor} already exists")

        donor_h = self.handles[donor]
        pools = self._guest_pools(
            home_replicas, pair.guest_replicas_per_unit, donor_h.cards_per_replica)
        guests = [
            await self._make_guest(home, donor, home_replicas, pool, idx)
            for idx, pool in enumerate(pools)
        ]

        return GuestUnit(
            home_policy=home, donor=donor,
            home_replicas=home_replicas, guests=guests,
        )

    @staticmethod
    def _unique_home_replicas(units: list[GuestUnit]) -> list:
        replicas = {}
        for unit in units:
            for replica in unit.home_replicas:
                replicas.setdefault(id(replica), replica)
        return list(replicas.values())

    @staticmethod
    async def _timed_sleep(replica) -> float:
        t0 = time.perf_counter()
        await replica.sleep()
        return time.perf_counter() - t0

    async def _initialize_unit(
        self,
        unit: GuestUnit,
        replica_locks: dict[int, asyncio.Lock],
    ) -> None:
        """Initialize and sleep one unit while exclusively holding its home cards."""
        initialized_guests = []
        locks = [
            replica_locks[replica_id]
            for replica_id in sorted({id(replica) for replica in unit.home_replicas})
        ]
        async with AsyncExitStack() as stack:
            for lock in locks:
                await stack.enter_async_context(lock)
            try:
                for guest in unit.guests:
                    await guest.init_standalone()
                    initialized_guests.append(guest)
                    await guest.sleep()
            except BaseException:
                if initialized_guests:
                    await asyncio.gather(*[
                        guest.sleep() for guest in initialized_guests
                    ], return_exceptions=True)
                self._kill_guest_actors(unit.guests)
                raise

    async def _probe_sleep(self, home: str, replicas: list, dt: float) -> None:
        """Fail fast unless every home vLLM engine reports that it is sleeping."""
        if self.config.sleep_patch_mode != "patched":
            return

        actor_methods = []
        has_server_handles = False
        try:
            for replica in replicas:
                servers = getattr(replica, "servers", None)
                if servers is None:
                    continue
                has_server_handles = True
                for server in servers:
                    actor_methods.append(server.dynamic_inference_is_sleeping.remote())
            if actor_methods:
                states = await asyncio.gather(*actor_methods)
                if all(states):
                    return
                raise RuntimeError(
                    f"dynamic_inference: home policy '{home}' has a vLLM engine "
                    f"that is still awake after sleep(): states={states}"
                )
            if has_server_handles:
                raise RuntimeError(
                    f"dynamic_inference: home policy '{home}' exposes no vLLM server actors"
                )
        except Exception as exc:
            raise RuntimeError(
                f"dynamic_inference: failed to verify sleep state for home policy "
                f"'{home}'; worker sleep patch may not have reached vLLM actors: {exc}"
            ) from exc

        # Unit-test/legacy fallback when a replica object has no server handles.
        if dt < _SLEEP_PROBE_MIN_S:
            raise RuntimeError(
                f"dynamic_inference: home policy '{home}' sleep() returned in {dt:.3f}s — "
                "verl's unpatched STANDALONE branch is a no-op, so the worker-side "
                "sleep/wake patch (worker_process_setup_hook -> "
                "uni_agent.trainer.dynamic_inference.patch.apply_worker_patch) did not "
                "reach the vLLMHttpServer processes. Fix the hook, or set "
                "sleep_patch_mode: collective_rpc for the DP=1 fallback."
            )

    # ------------------------------------------------------------- borrowing
    async def guests_for(self, home: str, donor: str, home_replicas: list) -> list:
        """Resolve the pre-created guest replicas for a borrow unit.

        Marks the unit in-use. Raises RuntimeError if this unit, or another
        donor direction sharing any of its home replicas, is already lent out.
        """
        unit = self.unit_for(home, donor, home_replicas)
        if unit is None:
            ranks = [getattr(r, "replica_rank", "?") for r in home_replicas]
            raise RuntimeError(
                f"no pre-created guest unit for {home}->{donor} on replicas "
                f"{ranks} — precreate_all() not run?"
            )
        if unit.in_use:
            raise RuntimeError(f"guest unit {home}->{donor} is already lent out")
        requested = {id(replica) for replica in home_replicas}
        conflict = next(
            (
                other for other in self.units
                if other.in_use
                and requested.intersection(id(replica) for replica in other.home_replicas)
            ),
            None,
        )
        if conflict is not None:
            raise RuntimeError(
                f"home replicas are already lent on {conflict.home_policy}->{conflict.donor}; "
                f"cannot also activate {home}->{donor}"
            )
        unit.in_use = True
        return list(unit.guests)

    def release(self, home: str, donor: str, home_replicas: list) -> None:
        """Mark the units of a returned lend available again."""
        unit = self.unit_for(home, donor, home_replicas)
        if unit is not None:
            unit.in_use = False

    # ------------------------------------------------------------- teardown
    @staticmethod
    def _kill_guest_actors(guests: list) -> int:
        actors = [
            actor
            for guest in guests
            for actor in (
                list(getattr(guest, "servers", []) or [])
                + list(getattr(guest, "workers", []) or [])
            )
        ]
        if not actors:
            return 0
        import ray

        killed = 0
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
                killed += 1
            except Exception as exc:
                logger.warning("dynamic_inference.guest_engine: kill failed: %s", exc)
        return killed

    def kill_all(self) -> None:
        """Best-effort teardown of every guest actor (no orphaned vLLM servers)."""
        killed = self._kill_guest_actors([
            guest for unit in self.units for guest in unit.guests
        ])
        self.units.clear()
        self._index.clear()
        if killed:
            logger.info("dynamic_inference.guest_engine: killed %d guest actors", killed)
