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

The creation transition (training has not started, no in-flight requests):

    home.sleep()
      -> guest.init_standalone()
      -> guest.sleep()
      -> home.wake_up()

i.e. the home engine's weights are freed while the guest engine initialises
(dummy load), then the guest frees its weights and the home reloads. The
first ``home.sleep()`` doubles as the **worker-patch probe**: a real level-2
sleep takes seconds; verl's unpatched STANDALONE branch returns in
milliseconds, which would silently break every later borrow — so we fail the
run at setup time instead of at the first boundary.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import time

from .topology import build_candidate_groups
from .types import (
    BorrowPairSpec,
    GuestUnit,
    PolicyInferenceHandles,
    SchedulingConfig,
    validate_against_handles,
)

logger = logging.getLogger(__name__)

# A real level-2 vLLM sleep of any RL-sized model takes seconds; the unpatched
# STANDALONE no-op returns in ~ms. Below this the worker patch did not land.
_SLEEP_PROBE_MIN_S = 0.5
_GUEST_LOAD_FORMAT = "dummy"


def _set_field(cfg, name: str, value) -> None:
    """Set a field on a dataclass or DictConfig rollout config."""
    try:
        setattr(cfg, name, value)
    except Exception:
        from omegaconf import OmegaConf

        OmegaConf.update(cfg, name, value, force_add=True)


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
        created = 0
        for pair in self.pairs:
            home_h = self.handles[pair.home]
            for replicas in self._groups[(pair.home, pair.donor)]:
                await self._create_unit(pair.home, pair.donor, replicas)
                created += 1
        logger.info(
            "dynamic_inference.guest_engine: pre-created %d guest units across %d pairs",
            created, len(self.pairs),
        )
        return created

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

    async def _make_guest(self, home: str, donor: str, home_replicas: list, pool, guest_idx: int):
        """Build one un-launched guest replica bound to a home-pool view."""
        donor_h = self.handles[donor]
        guest_cfg = copy.deepcopy(donor_h.rollout_config)
        _set_field(guest_cfg, "load_format", _GUEST_LOAD_FORMAT)

        from verl.workers.rollout.replica import get_rollout_replica_class

        guest_cls = get_rollout_replica_class(str(getattr(guest_cfg, "name", "vllm")))
        guest = guest_cls(
            replica_rank=next(self._guest_rank),
            config=guest_cfg,
            model_config=donor_h.model_config,
            gpus_per_node=int(getattr(guest_cfg, "n_gpus_per_node", 8) or 8),
            name_suffix=(f"guest_{home}_u"
                         f"{'_'.join(str(r.replica_rank) for r in home_replicas)}_g{guest_idx}"),
        )
        # patch #2: init on the home replica's own PG — no new placement group
        guest._guest_external_pool = pool
        return guest

    async def _create_unit(self, home: str, donor: str, home_replicas: list) -> GuestUnit:
        """Run the pre-create transition and register the unit."""
        home_replicas = list(home_replicas)
        if self.unit_for(home, donor, home_replicas) is not None:
            raise RuntimeError(f"guest unit for {home}->{donor} already exists")

        pair = next(p for p in self.pairs if p.home == home and p.donor == donor)
        donor_h = self.handles[donor]
        pools = self._guest_pools(
            home_replicas, pair.guest_replicas_per_unit, donor_h.cards_per_replica)
        guests = [
            await self._make_guest(home, donor, home_replicas, pool, idx)
            for idx, pool in enumerate(pools)
        ]

        # Init-time transition: home must free its weights for the guest engine
        # to fit on the same cards. Always put successfully initialized guests
        # back to sleep and restore home service when setup fails midway.
        initialized_guests = []
        home_slept = False
        try:
            t0 = time.perf_counter()
            home_slept = True
            await asyncio.gather(*[replica.sleep() for replica in home_replicas])
            dt = time.perf_counter() - t0
            self._probe_sleep(home, dt)

            for guest in guests:
                await guest.init_standalone()
                initialized_guests.append(guest)

            t0 = time.perf_counter()
            await asyncio.gather(*[guest.sleep() for guest in initialized_guests])
            logger.info(
                "dynamic_inference.guest_engine: guest sleep in %.2fs (%s->%s)",
                time.perf_counter() - t0, home, donor,
            )
        except Exception:
            if initialized_guests:
                await asyncio.gather(
                    *[guest.sleep() for guest in initialized_guests],
                    return_exceptions=True,
                )
            self._kill_guest_actors(guests)
            raise
        finally:
            if home_slept:
                t0 = time.perf_counter()
                results = await asyncio.gather(
                    *[replica.wake_up() for replica in home_replicas],
                    return_exceptions=True,
                )
                failures = [result for result in results if isinstance(result, Exception)]
                if failures:
                    raise RuntimeError(
                        f"failed to restore home policy '{home}' after guest precreation: "
                        f"{failures[0]}"
                    ) from failures[0]
                logger.info(
                    "dynamic_inference.guest_engine: home wake in %.2fs (%s)",
                    time.perf_counter() - t0, home,
                )

        unit = GuestUnit(
            home_policy=home, donor=donor,
            home_replicas=home_replicas, guests=guests,
        )
        self.units.append(unit)
        self._index[self._unit_key(home, donor, home_replicas)] = unit
        return unit

    def _probe_sleep(self, home: str, dt: float) -> None:
        """Fail fast when the STANDALONE sleep patch did not reach the workers."""
        if self.config.sleep_patch_mode != "patched":
            return
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
