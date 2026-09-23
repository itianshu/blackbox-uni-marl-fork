"""Opt-in deterministic GPU smoke; never used by ordinary experiment configs.

Drive real ownership transactions, retain a guest through the first training
step, and return it before the second. This verifies mechanics, not KV policy
quality. All training, LB, checkpoint and engine operations remain real.
"""
import inspect
import json
import logging

_APPLIED = False


def _emit(event, **detail):
    print("TRANSACTION_SMOKE " + json.dumps({"event": event, **detail}), flush=True)


def apply_patch():
    global _APPLIED
    if _APPLIED:
        return
    from examples.multi_agent_blackbox import verl_patch
    verl_patch.apply_patch()
    from uni_agent.trainer.dynamic_inference.controller import DynamicInferenceController
    from uni_agent.trainer.dynamic_inference.types import BorrowPlan
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    from verl.utils.ray_utils import auto_await

    # Hydra/Ray logging configuration can disable already-imported loggers.
    for name in ("patch", "executor", "controller", "replica_clone"):
        logger = logging.getLogger("uni_agent.trainer.dynamic_inference." + name)
        logger.disabled = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        logger.propagate = False

    original_setup = DynamicInferenceController.setup
    original_step = MultiAgentsPPOTrainer.train_step
    original_boundary = DynamicInferenceController.run_boundary

    def setup(self):
        original_setup(self)
        # No requests are being generated yet. Disable load-driven decisions
        # so they cannot undo the prescribed test sequence.
        self.poll_loop.stop()
        self.poll_loop.join(timeout=10)
        assert not self.poll_loop.is_alive(), "Smoke poll loop did not stop"
        self._smoke_steps = 0
        self._smoke_lend = None
        self._smoke_sync_calls = {p: [] for p in self.executor.handles}
        for policy, handle in self.executor.handles.items():
            manager = handle.standalone_checkpoint_manager
            update = manager.update_weights
            assert inspect.unwrap(update).__name__ == "_patched_checkpoint_update_weights", (
                "Smoke must exercise the bounded dynamic checkpoint update")

            async def counted_update(*args, _policy=policy, _manager=manager,
                                     _update=update, **kwargs):
                self._smoke_sync_calls[_policy].append(
                    [id(r) for r in _manager.replicas])
                return await _update(*args, **kwargs)

            manager.update_weights = auto_await(counted_update)
        _emit("ready", timeout_s=self.config.borrow_drain_timeout_s)

    def train_step(self):
        controller = self._dynamic_inference
        assert controller is not None, "Smoke requires dynamic inference enabled"
        controller._smoke_steps += 1
        sequence = controller._smoke_steps
        assert sequence <= 2, "Deterministic smoke requires exactly two steps"
        with controller.gate.boundary():
            if sequence == 1:
                unit = [controller.executor.handles["policy_2"].replicas[0]]
                controller._smoke_lend = controller.executor.borrow(
                    BorrowPlan("policy_2", "policy_1", unit), self.global_steps)
                _emit("borrowed", step=self.global_steps,
                      lend_id=controller._smoke_lend.lend_id)
            else:
                controller.executor.return_(controller._smoke_lend, self.global_steps)
                _emit("returned", step=self.global_steps)
            controller.executor.validate_membership()
        result = original_step(self)
        if sequence == 2:
            assert not controller.scheduler.active_lends
            _emit("passed", steps=2, borrows=1, returns=1,
                  policy_updates_per_step=1)
        return result

    def run_boundary(self, trainer):
        before = {p: len(calls) for p, calls in self._smoke_sync_calls.items()}
        original_boundary(self, trainer)
        for policy, calls in self._smoke_sync_calls.items():
            assert len(calls) - before[policy] == 1, (policy, "not one policy sync")
        guest_ids = {id(r) for r in self._smoke_lend.guests}
        p1_targets = set(self._smoke_sync_calls["policy_1"][-1])
        if self._smoke_steps == 1:
            assert guest_ids <= p1_targets, "Borrowed guest missed regular policy sync"
        else:
            assert guest_ids.isdisjoint(p1_targets), "Returned guest still synchronized"
        _emit("boundary_verified", step=trainer.global_steps,
              targets={p: len(calls[-1]) for p, calls in self._smoke_sync_calls.items()})

    DynamicInferenceController.setup = setup
    DynamicInferenceController.run_boundary = run_boundary
    MultiAgentsPPOTrainer.train_step = train_step
    _APPLIED = True
