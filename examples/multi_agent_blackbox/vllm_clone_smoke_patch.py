"""Opt-in GPU smoke: real return/reborrow overlaps actor update submission."""
import asyncio
import threading
import time

_APPLIED = False


def apply_patch():
    global _APPLIED
    if _APPLIED:
        return
    from examples.multi_agent_blackbox import transaction_smoke_patch as base
    base.apply_patch()
    original_emit = base._emit
    def emit(event, **detail):
        if event == 'passed':
            detail.update(borrows=2, returns=2, clone_transfers=4)
        original_emit(event, **detail)
    base._emit = emit
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    from uni_agent.trainer.dynamic_inference.types import BorrowPlan
    from uni_agent.trainer.dynamic_inference.guest_engine import GuestEngineManager
    build_units = GuestEngineManager._build_all_units
    async def smoke_units(self):
        # Only the explicitly exercised unit is needed; every operation on it
        # remains a real engine operation, including initialization and sleep.
        first = self.handles['policy_2'].replicas[0]
        self._groups = {key: [group for group in groups if first in group][:1]
                        for key, groups in self._groups.items()}
        return await build_units(self)
    GuestEngineManager._build_all_units = smoke_units
    from uni_agent.trainer.dynamic_inference import replica_clone
    original_clone = replica_clone.clone_replica
    async def verified_clone(source, destination, **kwargs):
        result = await original_clone(source, destination, **kwargs)
        params = {'temperature': 0.0, 'max_tokens': 8, 'logprobs': True}
        outputs = await asyncio.wait_for(asyncio.gather(*[
            r.server_handle.generate.remote(
                [1, 2, 3, 4], dict(params), result['transaction_id'] + suffix)
            for r, suffix in ((source, '-source'), (destination, '-target'))]), 60)
        a, b = outputs
        assert a.token_ids and a.token_ids == b.token_ids, 'Clone token output differs'
        assert a.log_probs is not None and b.log_probs is not None, 'verl generate did not return requested log probabilities'
        delta = max(abs(x-y) for x, y in zip(a.log_probs, b.log_probs))
        assert delta < 0.02, f'Clone log probability differs: {delta}'
        base._emit('clone_output_verified', transaction_id=result['transaction_id'],
                   token_ids=a.token_ids, max_logprob_delta=delta)
        return result
    replica_clone.clone_replica = verified_clone
    original = MultiAgentsPPOTrainer.update_policy_trainers

    def update(self, *args, **kwargs):
        controller = self._dynamic_inference
        if controller._smoke_steps != 1:
            return original(self, *args, **kwargs)
        entered = threading.Event()
        errors = []
        times = {}
        def topology():
            try:
                with controller.gate.boundary():
                    times['topology_start'] = time.monotonic()
                    entered.set()
                    lend = controller._smoke_lend
                    controller.executor.return_(lend, self.global_steps)
                    controller._smoke_lend = controller.executor.borrow(
                        BorrowPlan(lend.home_policy, lend.donor, lend.home_replicas), self.global_steps)
                    controller.executor.validate_membership()
                    times['topology_end'] = time.monotonic()
            except BaseException as exc:
                errors.append(exc)
                entered.set()
        thread = threading.Thread(target=topology, name='clone-smoke-topology', daemon=True)
        thread.start()
        assert entered.wait(30), 'Topology could not start before actor update'
        times['actor_start'] = time.monotonic()
        try:
            result = original(self, *args, **kwargs)
        finally:
            times['actor_end'] = time.monotonic()
            thread.join(timeout=2 * controller.config.weight_sync_timeout_s + 60)
        assert not thread.is_alive(), 'Topology did not finish; remote state unknown'
        if errors:
            raise errors[0]
        overlap = min(times['actor_end'], times['topology_end']) - max(times['actor_start'], times['topology_start'])
        assert overlap > 0, 'No overlap between topology and actor update'
        base._emit('actor_topology_overlap_verified', overlap_s=overlap, **times)
        return result

    MultiAgentsPPOTrainer.update_policy_trainers = update
    _APPLIED = True
