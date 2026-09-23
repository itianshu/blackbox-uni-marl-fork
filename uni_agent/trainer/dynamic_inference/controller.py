"""Dynamic inference scheduling controller — the trainer-facing facade.

Owns the four runtime pieces and enforces the serialization invariant (I2)
between everything physical:

- ``BoundaryGate`` — mutual exclusion between metrics-driven rebalancing,
  early returns and step-boundary weight updates (not actor compute);
- ``PollLoop`` — daemon thread sampling serving signals and immediately
  applying a confirmed useful allocation after a complete fresh KV scrape;
- the step boundary (see ``run_boundary``): finish requested returns, validate
  ownership and execute one combined weight update per policy.

Topology ownership stays in the trainer process alongside the persistent
checkpoint managers. Borrow/return transfers only read published vLLM weights. The gate keeps poll-thread decisions
serialized with boundaries owned by the outer trainer.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

from . import patch
from .executor import BoundaryExecutor
from .guest_engine import GuestEngineManager
from .metrics import VLLMMetricsScraper
from .scheduler import MultiPolicyInferenceScheduler
from .signals import ServingSignalSource, SignalStore
from .types import (
    BoundaryPlan,
    PolicyInferenceHandles,
    SchedulingConfig,
    StepStats,
    expand_pairs,
    parse_scheduling_config,
    required_slots_by_home,
    validate_policy_prerequisites,
)

logger = logging.getLogger(__name__)

# rolling-statistics window for the KV signal, in scrape periods
_SIGNAL_WINDOW_SAMPLES = 20


class BoundaryGate:
    """Topology transactions vs step-end publication; never actor training."""

    def __init__(self):
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._in_boundary = False
        self._generation = 0
        self.failure: BaseException | None = None
        self._pending_early_returns: list[Any] = []

    @contextmanager
    def boundary(self):
        with self._lock:
            self.raise_if_failed()
            was_in_boundary = self._in_boundary
            self._in_boundary = True
            try:
                yield
            except BaseException as exc:
                self.failure = exc
                raise
            finally:
                self._in_boundary = was_in_boundary
                self._generation += 1

    def raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError("Dynamic topology gate failed; refusing further work") from self.failure

    def snapshot_generation(self) -> int:
        # Reading an integer does not need to block metric sampling on publish.
        return self._generation

    def run_exclusive(self, fn, *, expected_generation: int | None = None) -> bool:
        """Run ``fn`` unless a boundary is active; defer then, return False."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self.raise_if_failed()
            if self._in_boundary:
                return False
            if (expected_generation is not None
                    and expected_generation != self._generation):
                return False
            fn()
            return True
        except BaseException as exc:
            self.failure = exc
            raise
        finally:
            self._lock.release()

    def defer_early_return(self, lend) -> None:
        with self._pending_lock:
            if lend not in self._pending_early_returns:
                self._pending_early_returns.append(lend)

    def take_pending(self) -> list:
        with self._pending_lock:
            pending, self._pending_early_returns = self._pending_early_returns, []
            return pending


class PollLoop(threading.Thread):
    """Background sampler: LB status + vLLM metrics -> SignalStore; scheduler.poll()."""

    def __init__(
        self,
        source,
        store,
        scheduler,
        gate: BoundaryGate,
        interval_s: float,
        on_metrics_ready=None,
    ):
        super().__init__(daemon=True, name="dynamic-inference-poll")
        self._source = source
        self._store = store
        self._scheduler = scheduler
        self._gate = gate
        self._interval = max(0.05, float(interval_s))
        self._on_metrics_ready = on_metrics_ready
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self._interval):
            try:
                generation = self._gate.snapshot_generation()
                samples = self._source.sample_all()
                self._gate.run_exclusive(
                    lambda: self._commit(samples),
                    expected_generation=generation,
                )
            except Exception:
                logger.warning("dynamic_inference: poll iteration failed", exc_info=True)
                if self._gate.failure is not None:
                    self.stop()

    def _commit(self, samples):
        self._store.record_all(samples)
        metrics_fresh = any(sample.metrics_fresh for sample in samples.values())
        metrics_ready = self._scheduler.poll(metrics_fresh=metrics_fresh)
        if metrics_ready and self._on_metrics_ready is not None:
            self._on_metrics_ready()

    def stop(self):
        self._stop_event.set()


class DynamicInferenceController:
    """Facade created by the outer trainer when the config block is enabled."""

    def __init__(self, config: SchedulingConfig, trainer: Any):
        self.config = config
        self.trainer = trainer
        self.gate = BoundaryGate()
        self.scheduler: MultiPolicyInferenceScheduler | None = None
        self.executor: BoundaryExecutor | None = None
        self.guest_engine: GuestEngineManager | None = None
        self.store: SignalStore | None = None
        self.poll_loop: PollLoop | None = None
        self.last_metrics: dict[str, Any] = {}
        self._last_step = 0

    # ================================================================ setup
    @classmethod
    def maybe_create(cls, trainer) -> "DynamicInferenceController | None":
        """Build the controller iff the config block exists and is enabled."""
        config = parse_scheduling_config(
            trainer.config.get("dynamic_inference_scheduling"))
        if config is None:
            return None
        return cls(config, trainer)

    def prepare_runtime_layout(self) -> None:
        """Reserve home and sleeping-guest slots before policy runtime setup.

        Standalone placement groups are created by each policy trainer before
        :meth:`setup` can pre-create guests.  Derive the required slot count
        from the borrowing graph and put it on each rollout config while those
        placement groups are still configurable.
        """
        policy_names = list(self.trainer.policy_configs)
        pairs = expand_pairs(self.config, policy_names)
        slots_by_home = required_slots_by_home(pairs, policy_names)
        patch.apply_patch()

        for policy_name, policy_config in self.trainer.policy_configs.items():
            rollout = policy_config.actor_rollout_ref.rollout
            custom = rollout.get("custom")
            if custom is None:
                from omegaconf import OmegaConf, open_dict

                with open_dict(rollout):
                    rollout.custom = OmegaConf.create({})
                custom = rollout.custom
            if isinstance(custom, dict):
                custom["dynamic_inference_max_colocate_count"] = slots_by_home[policy_name]
            else:
                from omegaconf import open_dict

                with open_dict(custom):
                    custom["dynamic_inference_max_colocate_count"] = slots_by_home[policy_name]

    def setup(self) -> None:
        trainer = self.trainer
        if trainer.trainer_mode != "separate_async":
            raise ValueError(
                "dynamic_inference_scheduling requires trainer.v1.trainer_mode="
                f"separate_async (got '{trainer.trainer_mode}'): borrowing "
                "operates on standalone rollout replicas"
            )

        # 1. static per-policy prerequisites (vllm backend, sleep mode, ...)
        validate_policy_prerequisites(
            self.config,
            {name: cfg.actor_rollout_ref.rollout for name, cfg in trainer.policy_configs.items()},
        )

        # 2. runtime handles per policy
        handles = self._build_handles()

        # 3. ensure the driver-side guest external-pool patch remains active;
        #    prepare_runtime_layout() installed it before home placement groups.
        #    The worker-side sleep/wake patch lands via the yaml's
        #    worker_process_setup_hook — precreate's sleep probe verifies it.

        self.scheduler = MultiPolicyInferenceScheduler(
            config=self.config, handles=handles, signals=None, guest_units=None,
            on_early_return=self._on_early_return,
        )
        self.guest_engine = GuestEngineManager(
            config=self.config, handles=handles, pairs=self.scheduler.pairs,
        )
        self.executor = BoundaryExecutor(
            config=self.config, handles=handles,
            guest_engine=self.guest_engine,
            scheduler=self.scheduler,
        )

        # 4. signals + decision wiring
        self.store = SignalStore(
            policies=list(handles),
            window_s=_SIGNAL_WINDOW_SAMPLES * self.config.metrics_scrape_interval_s,
            max_num_seqs={n: h.max_num_seqs for n, h in handles.items()},
        )
        self.scheduler.signals = self.store
        self.scheduler.guest_units = self.guest_engine

        # 5. pre-create guests (also the fail-fast sleep-probe), then poll
        import asyncio

        try:
            asyncio.run(self.guest_engine.precreate_all())
        except Exception:
            # Guest actors are cleaned up and home engines are restored by
            # precreate_all's failure path; keep this as best-effort teardown.
            self.guest_engine.kill_all()
            patch.restore()
            raise
        source = ServingSignalSource(
            handles,
            scraper=VLLMMetricsScraper(self.config.resource_usage.kv_metric_names),
            scrape_interval_s=0.0,
        )
        self.poll_loop = PollLoop(
            source, self.store, self.scheduler, self.gate,
            self.config.metrics_scrape_interval_s,
            on_metrics_ready=self._rebalance_now,
        )
        self.poll_loop.start()

        logger.info(
            "dynamic_inference: controller ready (policies=%s, pairs=%d)",
            list(handles), len(self.scheduler.pairs),
        )

    def _build_handles(self) -> dict[str, PolicyInferenceHandles]:
        trainer = self.trainer
        handles: dict[str, PolicyInferenceHandles] = {}
        for name, policy_trainer in trainer.policy_trainers.items():
            policy_config = trainer.policy_configs[name]
            rollout_cfg = policy_config.actor_rollout_ref.rollout
            server_manager = getattr(policy_trainer, "standalone_server_manager", None)
            if server_manager is None:
                raise ValueError(
                    f"policy '{name}' has no standalone_server_manager "
                    "(separate_async requires standalone rollout replicas)"
                )
            replicas = list(getattr(server_manager, "rollout_replicas", []) or [])
            if not replicas:
                raise ValueError(f"policy '{name}' has no standalone rollout replicas")
            cards = int(getattr(replicas[0], "world_size", 0) or 0)
            nnodes = int(getattr(replicas[0], "nnodes", 1) or 1)
            actor_wg = getattr(policy_trainer, "actor_rollout_wg", None)
            checkpoint_manager = getattr(
                policy_trainer, "standalone_checkpoint_manager", None)
            lb_handle = getattr(server_manager, "global_load_balancer", None)
            missing = [
                label for label, value in (
                    ("actor_rollout_wg", actor_wg),
                    ("standalone_checkpoint_manager", checkpoint_manager),
                    ("global_load_balancer", lb_handle),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"policy '{name}' is missing runtime handles: {missing}")
            if cards < 1 or nnodes < 1:
                raise ValueError(
                    f"policy '{name}' has invalid replica topology: "
                    f"world_size={cards}, nnodes={nnodes}")
            inconsistent = [
                replica for replica in replicas
                if int(getattr(replica, "world_size", 0) or 0) != cards
                or int(getattr(replica, "nnodes", 1) or 1) != nnodes
            ]
            if inconsistent:
                raise ValueError(
                    f"policy '{name}' has inconsistent standalone replica topologies")
            checkpoint_manager._dynamic_sync_timeout_s = self.config.weight_sync_timeout_s
            handles[name] = PolicyInferenceHandles(
                actor_rollout_wg=actor_wg,
                standalone_checkpoint_manager=checkpoint_manager,
                lb_handle=lb_handle,
                rollout_config=rollout_cfg,
                model_config=policy_config.actor_rollout_ref.model,
                replicas=replicas,
                cards_per_replica=cards,
                nnodes=nnodes,
                max_num_seqs=int(getattr(rollout_cfg, "max_num_seqs", 64) or 64),
            )
        return handles

    # ================================================================ boundary
    def run_boundary(self, trainer) -> None:
        """Require stable ownership, then synchronize each policy exactly once."""
        stats = self._build_stats(trainer)
        self._last_step = stats.step
        with self.gate.boundary():
            plan = self._boundary_maintenance_plan()
            deferred_early_ids = self._fold_deferred_early_returns(plan)
            self.executor.assert_stable()
            self._run_returns(plan.returns, stats.step, deferred_early_ids)
            self.executor.validate_membership()
            self._run_policy_hooks(trainer)
            self.executor.validate_membership()

        self.last_metrics = self.metrics_snapshot()

    def _rebalance_now(self) -> None:
        """Apply a metrics-driven allocation immediately on the poll thread."""
        step = int(self.trainer.global_steps)
        self._last_step = step
        plan = self.scheduler.decide(StepStats(step=step))
        if (plan.borrows or plan.returns) and getattr(self.scheduler, "trace", None) is not None:
            self.scheduler.trace.emit(self.scheduler, "decision", step,
                borrows=[{"lender": p.home_policy, "borrower": p.donor,
                          "home_servers": [getattr(r, "server_address", None) for r in p.home_replicas]}
                         for p in plan.borrows],
                returns=[p.lend_id for p in plan.returns])

        borrows_by_unit = {
            self._allocation_unit_key(plan.home_policy, plan.home_replicas): plan
            for plan in plan.borrows
        }
        handoffs = []
        regular_returns = []
        for lend in plan.returns:
            key = self._allocation_unit_key(lend.home_policy, lend.home_replicas)
            # Direct handoff assumes home is still fully asleep and detached.
            # A resumable return may already have woken or re-attached home, so
            # finish it normally and let the next borrow tear home down again.
            borrow = (
                borrows_by_unit.pop(key, None)
                if lend.return_stage == 0
                else None
            )
            if borrow is None:
                regular_returns.append(lend)
            else:
                handoffs.append((lend, borrow))

        if not self._run_returns(regular_returns, step, set()):
            self._log_skipped_borrows(len(plan.borrows), "a return failed")
            return

        for lend, borrow_plan in handoffs:
            if not self.executor.return_(
                lend, step, reactivate_home=False,
            ):
                self._log_skipped_borrows(len(plan.borrows), "a handoff return failed")
                return
            if self.executor.borrow(
                borrow_plan, step, home_already_sleeping=True,
            ) is None:
                self._log_skipped_borrows(
                    len(borrows_by_unit), "a handoff borrow failed")
                return

        for borrow_plan in borrows_by_unit.values():
            self.executor.borrow(borrow_plan, step)

    @staticmethod
    def _allocation_unit_key(home_policy: str, replicas: list[Any]) -> tuple:
        return home_policy, tuple(sorted(id(replica) for replica in replicas))

    @staticmethod
    def _log_skipped_borrows(count: int, reason: str) -> None:
        if count:
            logger.warning(
                "dynamic_inference: skipping %d immediate borrows because %s",
                count, reason,
            )

    def _boundary_maintenance_plan(self) -> BoundaryPlan:
        """Build the non-scheduling work that must remain at step boundaries."""
        active = list(self.scheduler.active_lends)
        if getattr(self.scheduler, "disabled", False):
            return BoundaryPlan(returns=active)
        return BoundaryPlan(returns=[lend for lend in active if lend.return_stage > 0])

    def _fold_deferred_early_returns(self, plan) -> set[int]:
        """Move still-active deferred returns out of renewals and into returns."""
        pending = [
            lend for lend in self.gate.take_pending()
            if lend in self.scheduler.active_lends
        ]
        for lend in pending:
            if lend not in plan.returns:
                plan.returns.append(lend)
            if lend in plan.renewals:
                plan.renewals.remove(lend)
        return {id(lend) for lend in pending}

    def _run_returns(self, lends, step: int, early_ids: set[int]) -> bool:
        """Return every requested lend; any failure prevents boundary progression."""
        succeeded = True
        for lend in lends:
            if not self.executor.return_(lend, step, early=id(lend) in early_ids):
                raise RuntimeError("Return executor reported failure")
        return succeeded

    @staticmethod
    def _run_policy_hooks(trainer) -> None:
        """Update every policy and wait for all hooks before surfacing errors."""
        futures = []
        policy_timings = {}
        for name, policy_trainer in trainer.policy_trainers.items():
            policy_timing = {}
            policy_trainer.timing_raw = policy_timing
            policy_timings[name] = policy_timing
            futures.append((name, trainer._policy_pool.submit(policy_trainer.on_step_end)))
        errors = []
        for name, future in futures:
            try:
                future.result()
            except Exception as exc:
                # A still-running update could race with cleanup or an early return.
                errors.append(exc)
            finally:
                for key, value in policy_timings[name].items():
                    trainer.timing_raw[f"{name}/{key}"] = value
        if errors:
            raise errors[0]

    def _build_stats(self, trainer) -> StepStats:
        return StepStats(step=trainer.global_steps)

    # ================================================================ early return
    def _on_early_return(self, lend) -> None:
        """Scheduler callback (poll thread): return the lend immediately."""
        def _run():
            self.executor.return_(lend, self._last_step, early=True)

        if not self.gate.run_exclusive(_run):
            # a boundary is running — the next decide() folds this lend in
            self.gate.defer_early_return(lend)

    # ================================================================ shutdown
    def shutdown(self) -> None:
        """Best-effort teardown; each step isolated (a failure must not block
        the rest of the trainer's cleanup)."""
        if self.poll_loop is not None:
            try:
                self.poll_loop.stop()
                self.poll_loop.join(timeout=max(1.0, self.config.poll_interval_s * 2))
            except Exception:
                pass
        try:
            with self.gate.boundary():
                for lend in list(self.scheduler.active_lends):
                    try:
                        self.executor.return_(lend, self._last_step, early=False)
                    except Exception:
                        logger.warning(
                            "dynamic_inference: final return of %s->%s failed",
                            lend.home_policy, lend.donor, exc_info=True)
        except Exception:
            logger.warning("dynamic_inference: final boundary failed", exc_info=True)
        try:
            self.guest_engine.kill_all()
        except Exception:
            logger.warning("dynamic_inference: guest kill_all failed", exc_info=True)
        try:
            patch.restore()
        except Exception:
            logger.warning("dynamic_inference: patch restore failed", exc_info=True)

    # ================================================================ metrics
    def metrics_snapshot(self) -> dict:
        if self.scheduler is None:
            return {}
        return {f"dynamic_inference/{k}": v for k, v in self.scheduler.metrics_snapshot().items()}
