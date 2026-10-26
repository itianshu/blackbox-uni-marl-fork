"""Gated ownership transactions with fail-stop handling.

Successful guests belong to the donor checkpoint manager. A failed transaction
is never silently rolled back: remote work may still be running after timeout.
The executor retains its error and refuses all further mutations/synchronization.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from .replica_clone import abort_replica
from .types import BorrowPlan, LendRecord, PolicyInferenceHandles, SchedulingConfig

logger = logging.getLogger(__name__)

class BorrowFailedError(RuntimeError):
    """A borrow failed; remote state is uncertain and further work is forbidden."""


class ReturnFailedError(RuntimeError):
    """A return failed; its last completed stage is retained for diagnosis."""


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
        self.failure: BaseException | None = None
        self.transaction: str | None = None

    # ================================================================ public
    def borrow(
        self,
        plan: BorrowPlan,
        step: int,
        *,
        home_already_sleeping: bool = False,
    ) -> LendRecord:
        """Commit one borrow or raise and poison the executor."""
        self.assert_stable()
        self.transaction = "BORROWING"
        started = time.monotonic()
        transaction_id = uuid4().hex
        self._trace("borrow_start", step, transaction_id=transaction_id,
                    lender=plan.home_policy, borrower=plan.donor)
        try:
            result = asyncio.run(self._borrow(plan, step, home_already_sleeping))
            if self.scheduler is not None:
                self.scheduler.note_borrow_started(result)
            execution_s = time.monotonic() - started
            self._trace(
                "borrow_complete", step,
                transaction_id=transaction_id,
                lender=plan.home_policy,
                borrower=plan.donor,
                lend_id=result.lend_id,
                execution_s=execution_s,
                trigger_to_routing_ready_s=execution_s,
                stage_timings_s=result.borrow_timings,
                home_servers=[r.server_address for r in result.home_replicas],
                guest_servers=[g.server_address for g in result.guests],
            )
            self.transaction = None
            return result
        except BaseException as exc:
            self._trace("borrow_failed", step, transaction_id=transaction_id,
                        error=str(exc), execution_s=time.monotonic() - started)
            self.failure = exc
            raise

    def return_(
        self,
        lend: LendRecord,
        step: int,
        early: bool = False,
        *,
        reactivate_home: bool = True,
    ) -> bool:
        """Commit one return or raise and poison the executor."""
        self.assert_stable()
        self.transaction = "RETURNING"
        started = time.monotonic()
        transaction_id = uuid4().hex
        self._trace("return_start", step, transaction_id=transaction_id,
                    lend_id=lend.lend_id, lender=lend.home_policy,
                    borrower=lend.donor, early=early)
        try:
            stage_timings = asyncio.run(self._return(lend, step, early, reactivate_home))
        except BaseException as exc:
            self._trace("return_failed", step, transaction_id=transaction_id,
                        lend_id=lend.lend_id, error=str(exc),
                        execution_s=time.monotonic() - started)
            self.failure = exc
            raise
        self.transaction = None
        if self.scheduler is not None:
            self.scheduler.note_lend_returned(lend, step=step, early=early)
        execution_s = time.monotonic() - started
        self._trace(
            "return_complete", step,
            transaction_id=transaction_id,
            lend_id=lend.lend_id,
            lender=lend.home_policy,
            borrower=lend.donor,
            early=early,
            execution_s=execution_s,
            trigger_to_routing_ready_s=execution_s,
            stage_timings_s=stage_timings,
            home_servers=[r.server_address for r in lend.home_replicas],
            guest_servers=[g.server_address for g in lend.guests],
            reactivate_home=reactivate_home,
        )
        return True

    def _trace(self, kind, step, **detail):
        trace = getattr(self.scheduler, "trace", None)
        if trace is not None:
            trace.emit(self.scheduler, kind, step, **detail)

    def assert_stable(self) -> None:
        if self.failure is not None:
            raise RuntimeError("Topology transaction failed; restart required") from self.failure
        if self.transaction is not None:
            raise RuntimeError(f"Unfinished topology transaction: {self.transaction}")

    def validate_membership(self) -> None:
        self.assert_stable()
        expected = {p: list(h.replicas) for p, h in self.handles.items()}
        for lend in self.scheduler.active_lends:
            if lend.return_stage:
                raise RuntimeError(f"Unfinished return {lend.lend_id}: stage {lend.return_stage}")
            expected[lend.home_policy] = [r for r in expected[lend.home_policy]
                                           if r not in lend.home_replicas]
            expected[lend.donor].extend(lend.guests)
        seen = set()
        for policy, handle in self.handles.items():
            actual = handle.standalone_checkpoint_manager.replicas
            ids = [id(r) for r in actual]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise RuntimeError(f"Duplicate checkpoint ownership: {policy}")
            if set(ids) != {id(r) for r in expected[policy]}:
                raise RuntimeError(f"Checkpoint ownership mismatch: {policy}")
            seen.update(ids)

    @staticmethod
    def _add_replicas(manager, replicas):
        manager.add_replicas([r for r in replicas if r not in manager.replicas])

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
        timings: dict[str, float] = {}
        stage = 4 if home_already_sleeping else 0
        try:
            if not home_already_sleeping:
                if not any(r not in unit and r in home_h.replicas
                           for r in home_h.standalone_checkpoint_manager.replicas):
                    raise RuntimeError("Cannot lend the last published seed replica")
                # 1. home LB: remove routing FIRST so retries land on remaining
                #    home replicas (I6)
                home_servers = {r.server_address: r.server_handle for r in unit}
                # Mark before the remote call: it may mutate the LB and then raise.
                # A failure here is terminal because remote completion is uncertain.
                stage = 1
                t0 = time.monotonic()
                await self._ray(home_h.lb_handle.remove_servers.remote(list(home_servers)))
                timings["remove_home_routing"] = time.monotonic() - t0
                # 2. abort in-flight (clients transparently retry elsewhere)
                t0 = time.monotonic()
                await asyncio.wait_for(
                    asyncio.gather(*[abort_replica(r) for r in unit]), timeout)
                timings["abort_home_requests"] = time.monotonic() - t0
                stage = 2
                # 3. home scm: stop covering these replicas in the policy's own
                #    update_weights (I1)
                home_h.standalone_checkpoint_manager.remove_replicas(unit)
                stage = 3
                # 4. home engines sleep (level-2 after the worker patch)
                t0 = time.perf_counter()
                # A partial sleep failure leaves uncertain remote state and
                # therefore stops the transaction without waking another engine.
                stage = 4
                await asyncio.wait_for(asyncio.gather(*[r.sleep() for r in unit]), timeout)
                timings["home_sleep"] = time.perf_counter() - t0
                logger.info("dynamic_inference: %s sleep in %.2fs", plan.home_policy, timings["home_sleep"])
            # 5. wake the pre-created guests (I3: home is fully asleep first)
            t0 = time.monotonic()
            guests = await self.guest_engine.guests_for(plan.home_policy, plan.donor, unit)
            timings["reserve_guest"] = time.monotonic() - t0
            t0 = time.perf_counter()
            # guests_for reserves the unit. Retain that reservation on failure;
            # a timed-out wake can still be running remotely.
            stage = 5
            await asyncio.wait_for(asyncio.gather(*[g.wake_up() for g in guests]), timeout)
            timings["guest_wake"] = time.perf_counter() - t0
            logger.info("dynamic_inference: %s guest wake in %.2fs", plan.donor, timings["guest_wake"])
            # 6. Clone the target policy published vLLM version, without actor access.
            t0 = time.perf_counter()
            await self._clone_weights(
                plan.donor, guests, step,
                timeout=self.config.weight_sync_timeout_s,
            )
            timings["weight_sync"] = time.perf_counter() - t0
            logger.info("dynamic_inference: %s vLLM clone in %.2fs", plan.donor, timings["weight_sync"])
            stage = 6
            t0 = time.monotonic()
            self._add_replicas(donor_h.standalone_checkpoint_manager, guests)
            timings["publish_guest_to_checkpoint_manager"] = time.monotonic() - t0
            # 7. donor LB: route to the guests (least-loaded sends them traffic
            #    naturally; I5 — weights are already correct)
            t0 = time.monotonic()
            await self._ray(donor_h.lb_handle.add_servers.remote(
                {g.server_address: g.server_handle for g in guests}))
            timings["add_guest_routing"] = time.monotonic() - t0
        except Exception as exc:
            raise BorrowFailedError(
                f"borrow {plan.home_policy}->{plan.donor} failed at stage {stage}: {type(exc).__name__}: {exc!r}") from exc

        # ---- success bookkeeping ----
        lend = LendRecord(
            lend_id=self.scheduler.next_lend_id() if self.scheduler else 0,
            home_policy=plan.home_policy, donor=plan.donor,
            home_replicas=unit, guests=guests, since_step=step,
            borrow_timings=timings,
        )
        logger.info("dynamic_inference: borrow %s -> %s at step %d",
                    plan.home_policy, plan.donor, step)
        return lend

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
        timings: dict[str, float] = {}
        try:
            if lend.return_stage < 1:
                # 1. donor LB: stop routing to the guests (I6)
                t0 = time.monotonic()
                await self._ray(donor_h.lb_handle.remove_servers.remote(
                    [g.server_address for g in guests]))
                timings["remove_guest_routing"] = time.monotonic() - t0
                lend.return_stage = 1
            if lend.return_stage < 2:
                t0 = time.monotonic()
                await asyncio.wait_for(
                    asyncio.gather(*[abort_replica(g) for g in guests]), timeout)
                timings["abort_guest_requests"] = time.monotonic() - t0
                lend.return_stage = 2
            if lend.return_stage < 3:
                donor_h.standalone_checkpoint_manager.remove_replicas(guests)
                # 3. guests sleep — NO weight write-back (the home replicas'
                #    weights are authoritative; the donor's trainer state moves on)
                t0 = time.perf_counter()
                await asyncio.wait_for(asyncio.gather(*[g.sleep() for g in guests]), timeout)
                timings["guest_sleep"] = time.perf_counter() - t0
                logger.info("dynamic_inference: %s guest sleep in %.2fs",
                            lend.donor, timings["guest_sleep"])
                lend.return_stage = 3
            if not reactivate_home:
                # A new guest will immediately reuse the same cards. Keep home
                # asleep and detached; the next borrow skips home teardown.
                lend.return_stage = 7
            if lend.return_stage < 4:
                # 4. home engines wake (I3: guests are asleep first)
                t0 = time.perf_counter()
                await asyncio.wait_for(asyncio.gather(*[r.wake_up() for r in unit]), timeout)
                timings["home_wake"] = time.perf_counter() - t0
                logger.info("dynamic_inference: %s wake in %.2fs",
                            lend.home_policy, timings["home_wake"])
                lend.return_stage = 4
            if lend.return_stage < 5:
                # 5. Refresh home before it re-enters the LB. Boundary returns
                #    happen before the regular update_weights, so omitting this
                #    push would expose stale home weights until that hook ends.
                t0 = time.perf_counter()
                await self._clone_weights(
                    lend.home_policy, unit, step,
                    timeout=self.config.weight_sync_timeout_s,
                )
                timings["weight_sync"] = time.perf_counter() - t0
                logger.info("dynamic_inference: %s return vLLM clone in %.2fs",
                            lend.home_policy, timings["weight_sync"])
                lend.return_stage = 5
            if lend.return_stage < 6:
                # 6. Re-attach to the persistent home weight manager.
                t0 = time.monotonic()
                self._add_replicas(home_h.standalone_checkpoint_manager, unit)
                timings["publish_home_to_checkpoint_manager"] = time.monotonic() - t0
                lend.return_stage = 6
            if lend.return_stage < 7:
                # 7. Route only after current home weights are present. Keeping
                #    SCM and LB stages separate identifies which side completed
                #    if the remote routing change fails.
                t0 = time.monotonic()
                await self._ray(home_h.lb_handle.add_servers.remote(home_servers))
                timings["add_home_routing"] = time.monotonic() - t0
                lend.return_stage = 7
        except Exception as exc:
            raise ReturnFailedError(
                f"return {lend.home_policy}->{lend.donor} failed at stage "
                f"{lend.return_stage}: {type(exc).__name__}: {exc!r}") from exc
        self.guest_engine.release(lend.home_policy, lend.donor, unit)
        logger.info(
            "dynamic_inference: return %s -> %s at step %d%s",
            lend.home_policy, lend.donor, step, " (early)" if early else "",
        )
        return timings

    # ================================================================ push
    async def _clone_weights(self, policy: str, replicas, step: int, timeout: float) -> None:
        """Initialize from a published vLLM; caller holds topology/publish gate."""
        from .replica_clone import clone_replica
        manager = self.handles[policy].standalone_checkpoint_manager
        version = getattr(manager, "_dynamic_published_version", None)
        sources = [r for r in manager.replicas if r not in replicas
                   and getattr(r, "_dynamic_weight_version", None) == version]
        if version is None or not sources:
            raise RuntimeError(f"No verified published vLLM seed for {policy}")
        for target in replicas:
            target_host = target.server_address.rsplit(':', 1)[0]
            source = min(sources, key=lambda r: (
                r.server_address.rsplit(':', 1)[0] != target_host, r.server_address))
            result = await asyncio.wait_for(clone_replica(
                source, target, policy=policy, version=version, timeout=timeout,
                bucket_bytes=self.config.clone_bucket_megabytes << 20), timeout)
            self._trace("replica_clone", step, **result)

    # ================================================================ helpers
    async def _ray(self, obj_ref) -> None:
        """Await a Ray remote call from async context.

        ``ObjectRef`` implements ``__await__``; a plain value (fake handles in
        unit tests, or a future-like with ``result()``) is also accepted.
        """
        if hasattr(obj_ref, "__await__"):
            await asyncio.wait_for(obj_ref, self.config.borrow_drain_timeout_s)
        elif hasattr(obj_ref, "result"):
            obj_ref.result()
        # else: already a resolved value (test fakes) — nothing to wait on
