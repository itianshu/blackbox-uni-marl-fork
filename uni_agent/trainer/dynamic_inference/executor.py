"""Physical borrow/return sequences (the executor of the RFC).

Sequences mirror verl's own deactivation/activation order
(``dynamic_resource_controller.py``: remove-from-LB -> abort -> sleep;
add-to-LB last), so every swap satisfies the safety invariants:

- I1 unique ownership — guests only ever appear in a throwaway
  ``CheckpointEngineManager`` built for one directed push; they are never
  stored in any persistent manager;
- I2 serialization — the controller only calls these under its boundary gate,
  which serialises them with every policy's ``update_weights``;
- I3 one awake engine per card — borrow sleeps every home replica before
  waking any guest; return is the exact reverse pairing;
- I5 weights match LB membership — the directed push precedes ``add_servers``
  (borrow), follows ``wake_up`` (early return), and refreshes in-place renewals
  after every donor ``update_weights``;
- I6 remove-from-LB first — both sequences' literal step order.

Every stage is individually time-boxed and failure-handled: a borrow rolls
back to the pre-borrow state, a return records its progress on the
``LendRecord`` (``return_stage``) and resumes from there at the next complete
metrics decision or step boundary. A same-unit donor change keeps home asleep
and skips the otherwise redundant home wake and weight refresh. Repeated
failures on a pair block that pair; repeated failures overall disable the
scheduler (degrade to static).
"""

from __future__ import annotations

import asyncio
import logging
import time

from .types import BorrowPlan, LendRecord, PolicyInferenceHandles, SchedulingConfig

logger = logging.getLogger(__name__)

# how many consecutive pair failures before the whole feature degrades to static
_PAIR_FAILURE_LIMIT = 2


class BorrowFailedError(RuntimeError):
    """A borrow rolled back (home state restored); the pair gets blocked."""


class ReturnFailedError(RuntimeError):
    """A return is half-done; it stays active for the next gated retry."""


class BoundaryExecutor:
    def __init__(
        self,
        config: SchedulingConfig,
        handles: dict[str, PolicyInferenceHandles],
        guest_engine,
        scheduler=None,
    ):
        self.config = config
        self.handles = handles
        self.guest_engine = guest_engine
        self.scheduler = scheduler
        self._pair_failures: dict[tuple[str, str], int] = {}

    # ================================================================ public
    def borrow(
        self,
        plan: BorrowPlan,
        step: int,
        *,
        home_already_sleeping: bool = False,
    ) -> LendRecord | None:
        """Execute one borrow unit; returns the LendRecord (or None on rollback)."""
        try:
            return asyncio.run(self._borrow(plan, step, home_already_sleeping))
        except BorrowFailedError as exc:
            self._note_pair_failure(plan.home_policy, plan.donor, step, exc)
            return None

    def return_(
        self,
        lend: LendRecord,
        step: int,
        early: bool = False,
        *,
        reactivate_home: bool = True,
    ) -> bool:
        """Execute one return; True when complete, False for a later gated retry."""
        try:
            asyncio.run(self._return(lend, step, early, reactivate_home))
        except ReturnFailedError as exc:
            self._note_pair_failure(lend.home_policy, lend.donor, step, exc)
            return False
        self._pair_failures.pop((lend.home_policy, lend.donor), None)
        if self.scheduler is not None:
            self.scheduler.note_lend_returned(lend, step=step, early=early)
        return True

    def renew(self, lend: LendRecord, step: int) -> bool:
        """Refresh a still-routed guest with the donor's latest weights."""
        return self.renew_many([lend], step)

    def renew_many(self, lends: list[LendRecord], step: int) -> bool:
        """Refresh all retained lends for one donor in one process group."""
        if not lends:
            return True
        donors = {lend.donor for lend in lends}
        if len(donors) != 1:
            raise ValueError("renew_many requires all lends to have the same donor")
        donor = next(iter(donors))
        guests = [guest for lend in lends for guest in lend.guests]
        try:
            asyncio.run(self._directed_push(
                donor,
                guests,
                step,
                timeout=self.config.borrow_drain_timeout_s * 3,
            ))
        except Exception as exc:
            # A batch is one physical push attempt. Count it once per affected
            # direction, even when that direction currently owns several lend
            # units; otherwise one outage can immediately trip the consecutive
            # failure limit merely because the allocation is larger.
            failed_pairs = {
                (lend.home_policy, lend.donor)
                for lend in lends
            }
            for home, failed_donor in failed_pairs:
                self._note_pair_failure(home, failed_donor, step, exc)
            logger.warning(
                "dynamic_inference: renewal batch for donor %s failed at step %d: %s",
                donor, step, exc,
            )
            return False
        for lend in lends:
            self._pair_failures.pop((lend.home_policy, lend.donor), None)
            if self.scheduler is not None:
                self.scheduler.note_lend_renewed(lend, step)
        return True

    # ================================================================ borrow
    async def _borrow(
        self,
        plan: BorrowPlan,
        step: int,
        home_already_sleeping: bool = False,
    ) -> LendRecord:
        home_h = self.handles[plan.home_policy]
        donor_h = self.handles[plan.donor]
        unit = list(plan.home_replicas)
        timeout = self.config.borrow_drain_timeout_s
        guests: list = []
        stage = 4 if home_already_sleeping else 0
        try:
            if not home_already_sleeping:
                # 1. home LB: remove routing FIRST so retries land on remaining
                #    home replicas (I6)
                home_servers = {r.server_address: r.server_handle for r in unit}
                # Mark before the remote call: it may mutate the LB and then raise.
                # Re-adding the same address mapping during rollback is idempotent.
                stage = 1
                await self._ray(home_h.lb_handle.remove_servers.remote(list(home_servers)))
                # 2. abort in-flight (clients transparently retry elsewhere)
                await asyncio.wait_for(
                    asyncio.gather(*[r.abort_all_requests() for r in unit]), timeout)
                stage = 2
                # 3. home scm: stop covering these replicas in the policy's own
                #    update_weights (I1)
                home_h.standalone_checkpoint_manager.remove_replicas(unit)
                stage = 3
                # 4. home engines sleep (level-2 after the worker patch)
                t0 = time.perf_counter()
                # A gather can fail after only part of the unit slept. Rollback must
                # wake every home replica in that case.
                stage = 4
                await asyncio.wait_for(asyncio.gather(*[r.sleep() for r in unit]), timeout)
                logger.info("dynamic_inference: %s sleep in %.2fs", plan.home_policy, time.perf_counter() - t0)
            # 5. wake the pre-created guests (I3: home is fully asleep first)
            guests = await self.guest_engine.guests_for(plan.home_policy, plan.donor, unit)
            t0 = time.perf_counter()
            # guests_for marks the unit in-use. Even a partial wake failure must
            # sleep all guests and release that reservation.
            stage = 5
            await asyncio.wait_for(asyncio.gather(*[g.wake_up() for g in guests]), timeout)
            logger.info("dynamic_inference: %s guest wake in %.2fs", plan.donor, time.perf_counter() - t0)
            # 6. one directed weight push: donor weights -> guests (guests never
            #    enter the donor's persistent scm)
            t0 = time.perf_counter()
            await self._directed_push(plan.donor, guests, step, timeout=timeout * 3)
            logger.info("dynamic_inference: %s directed push in %.2fs", plan.donor, time.perf_counter() - t0)
            stage = 6
            # 7. donor LB: route to the guests (least-loaded sends them traffic
            #    naturally; I5 — weights are already correct)
            await self._ray(donor_h.lb_handle.add_servers.remote(
                {g.server_address: g.server_handle for g in guests}))
        except Exception as exc:
            await self._rollback_borrow(
                plan, unit, guests, stage, timeout, step,
                refresh_home=home_already_sleeping,
            )
            raise BorrowFailedError(
                f"borrow {plan.home_policy}->{plan.donor} failed at stage {stage}: {exc}") from exc

        # ---- success bookkeeping ----
        self._pair_failures.pop((plan.home_policy, plan.donor), None)
        lend = LendRecord(
            lend_id=self.scheduler.next_lend_id() if self.scheduler else 0,
            home_policy=plan.home_policy, donor=plan.donor,
            home_replicas=unit, guests=guests, since_step=step,
        )
        if self.scheduler is not None:
            self.scheduler.note_borrow_started(lend)
        logger.info("dynamic_inference: borrow %s -> %s at step %d",
                    plan.home_policy, plan.donor, step)
        return lend

    async def _rollback_borrow(
        self,
        plan,
        unit,
        guests,
        stage,
        timeout,
        step,
        *,
        refresh_home: bool = False,
    ) -> None:
        """Best-effort restore of the pre-borrow state; never raises."""
        home_h = self.handles[plan.home_policy]
        donor_h = self.handles[plan.donor]
        home_servers = {r.server_address: r.server_handle for r in unit}
        errors = []

        if stage >= 5:
            # add_servers may have failed after a partial LB mutation. Removing
            # all guest addresses is idempotent and prevents stale routes.
            try:
                await self._ray(donor_h.lb_handle.remove_servers.remote(
                    [g.server_address for g in guests]))
            except Exception as exc:
                errors.append(("donor LB remove", exc))
            try:
                await asyncio.wait_for(asyncio.gather(*[g.sleep() for g in guests]), timeout)
            except Exception as exc:
                errors.append(("guest sleep", exc))
            finally:
                self.guest_engine.release(plan.home_policy, plan.donor, unit)
        if stage >= 4:
            try:
                await asyncio.wait_for(asyncio.gather(*[r.wake_up() for r in unit]), timeout)
            except Exception as exc:
                errors.append(("home wake", exc))
            if refresh_home:
                try:
                    await self._directed_push(
                        plan.home_policy, unit, step, timeout=timeout * 3)
                except Exception as exc:
                    errors.append(("home weight refresh", exc))
        if stage >= 3:
            try:
                home_h.standalone_checkpoint_manager.add_replicas(unit)
            except Exception as exc:
                errors.append(("home SCM add", exc))
        if stage >= 1:
            try:
                await self._ray(home_h.lb_handle.add_servers.remote(home_servers))
            except Exception as exc:
                errors.append(("home LB add", exc))

        if errors:
            # rollback itself failed — the cluster state is inconsistent; block
            # the pair permanently and let the controller degrade to static
            logger.error(
                "dynamic_inference: ROLLBACK FAILED for %s->%s (stage %d): %s — "
                "disabling dynamic inference",
                plan.home_policy, plan.donor, stage,
                "; ".join(f"{label}: {exc}" for label, exc in errors),
            )
            if self.scheduler is not None:
                self.scheduler.note_disabled_by_failures()

    # ================================================================ return
    async def _return(
        self,
        lend: LendRecord,
        step: int,
        early: bool,
        reactivate_home: bool = True,
    ) -> None:
        home_h = self.handles[lend.home_policy]
        donor_h = self.handles[lend.donor]
        unit = list(lend.home_replicas)
        guests = list(lend.guests)
        timeout = self.config.borrow_drain_timeout_s
        home_servers = {r.server_address: r.server_handle for r in unit}
        try:
            if lend.return_stage < 1:
                # 1. donor LB: stop routing to the guests (I6)
                await self._ray(donor_h.lb_handle.remove_servers.remote(
                    [g.server_address for g in guests]))
                lend.return_stage = 1
            if lend.return_stage < 2:
                await asyncio.wait_for(
                    asyncio.gather(*[g.abort_all_requests() for g in guests]), timeout)
                lend.return_stage = 2
            if lend.return_stage < 3:
                # 3. guests sleep — NO weight write-back (the home replicas'
                #    weights are authoritative; the donor's trainer state moves on)
                t0 = time.perf_counter()
                await asyncio.wait_for(asyncio.gather(*[g.sleep() for g in guests]), timeout)
                logger.info("dynamic_inference: %s guest sleep in %.2fs",
                            lend.donor, time.perf_counter() - t0)
                lend.return_stage = 3
            if not reactivate_home:
                # A new guest will immediately reuse the same cards. Keep home
                # asleep and detached; the next borrow skips home teardown.
                lend.return_stage = 7
            if lend.return_stage < 4:
                # 4. home engines wake (I3: guests are asleep first)
                t0 = time.perf_counter()
                await asyncio.wait_for(asyncio.gather(*[r.wake_up() for r in unit]), timeout)
                logger.info("dynamic_inference: %s wake in %.2fs",
                            lend.home_policy, time.perf_counter() - t0)
                lend.return_stage = 4
            if lend.return_stage < 5:
                # 5. Refresh home before it re-enters the LB. Boundary returns
                #    happen before the regular update_weights, so omitting this
                #    push would expose stale home weights until that hook ends.
                t0 = time.perf_counter()
                await self._directed_push(lend.home_policy, unit, step, timeout=timeout * 3)
                logger.info("dynamic_inference: %s return push in %.2fs",
                            lend.home_policy, time.perf_counter() - t0)
                lend.return_stage = 5
            if lend.return_stage < 6:
                # 6. Re-attach to the persistent home weight manager.
                home_h.standalone_checkpoint_manager.add_replicas(unit)
                lend.return_stage = 6
            if lend.return_stage < 7:
                # 7. Route only after current home weights are present. Keeping
                #    SCM and LB as separate resumable stages avoids duplicate
                #    SCM entries when add_servers fails and is retried.
                await self._ray(home_h.lb_handle.add_servers.remote(home_servers))
                lend.return_stage = 7
        except Exception as exc:
            raise ReturnFailedError(
                f"return {lend.home_policy}->{lend.donor} failed at stage "
                f"{lend.return_stage}: {exc}") from exc
        self.guest_engine.release(lend.home_policy, lend.donor, unit)
        logger.info(
            "dynamic_inference: return %s -> %s at step %d%s",
            lend.home_policy, lend.donor, step, " (early)" if early else "",
        )

    # ================================================================ push
    async def _directed_push(self, policy: str, replicas, step: int, timeout: float) -> None:
        """One-shot weight push from ``policy``'s actor group to ``replicas``.

        Builds a throwaway ``CheckpointEngineManager`` around the policy's
        actor_wg and just the target replicas — the manager rebuilds its
        temporary worker group/process group on every ``update_weights``
        call, so nothing needs cleaning up afterwards (I1).
        """
        from verl.checkpoint_engine.base import CheckpointEngineManager

        handle = self.handles[policy]
        manager = CheckpointEngineManager(
            config=handle.standalone_checkpoint_manager.config,
            actor_wg=handle.actor_rollout_wg,
            replicas=list(replicas),
        )
        await asyncio.wait_for(manager.update_weights(step), timeout)

    # ================================================================ helpers
    async def _ray(self, obj_ref) -> None:
        """Await a Ray remote call from async context.

        ``ObjectRef`` implements ``__await__``; a plain value (fake handles in
        unit tests, or a future-like with ``result()``) is also accepted.
        """
        if hasattr(obj_ref, "__await__"):
            await obj_ref
        elif hasattr(obj_ref, "result"):
            obj_ref.result()
        # else: already a resolved value (test fakes) — nothing to wait on

    def _note_pair_failure(self, home: str, donor: str, step: int, exc: Exception) -> None:
        key = (home, donor)
        self._pair_failures[key] = self._pair_failures.get(key, 0) + 1
        count = self._pair_failures[key]
        if self.scheduler is not None:
            # block the direction for a while after a physical failure
            self.scheduler.pair_block_until[key] = step + 10
            if count >= _PAIR_FAILURE_LIMIT:
                logger.error(
                    "dynamic_inference: %d consecutive failures on pair %s->%s — "
                    "degrading to static scheduling",
                    count, home, donor,
                )
                self.scheduler.note_disabled_by_failures()
        logger.warning("dynamic_inference: pair %s->%s failure #%d at step %d: %s",
                       home, donor, count, step, exc)
