"""Preflight the opt-in smoke driver before expensive GPU initialization."""
import asyncio
import functools
import importlib
import logging
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace as NS

import pytest

# Import the real package before installing test-only module substitutes.
from uni_agent.trainer.dynamic_inference.types import BorrowPlan  # noqa: F401


def auto_await(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        coro = fn(*args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro
    return wrapped


@pytest.mark.parametrize('fault', [None, 'duplicate_sync', 'missing_guest'])
def test_smoke_driver_uses_executor_handles_and_verifies_boundaries(monkeypatch, capsys, fault):
    for name in ('patch', 'executor', 'controller'):
        logger = logging.getLogger('uni_agent.trainer.dynamic_inference.' + name)
        monkeypatch.setattr(logger, 'handlers', list(logger.handlers))
        for attr in ('disabled', 'level', 'propagate'):
            monkeypatch.setattr(logger, attr, getattr(logger, attr))

    async def _patched_checkpoint_update_weights():
        return {}

    class Controller:
        def setup(self):
            handles = {}
            for name, count in [('policy_1', 2), ('policy_2', 6), ('policy_3', 2)]:
                replicas = [object() for _ in range(count)]
                handles[name] = NS(replicas=replicas, standalone_checkpoint_manager=NS(
                    replicas=list(replicas), update_weights=auto_await(_patched_checkpoint_update_weights)))
            self.scheduler = NS(active_lends=[])
            self.config = NS(borrow_drain_timeout_s=60)
            self.poll_loop = NS(stop=lambda: None, join=lambda **kw: None, is_alive=lambda: False)
            self.gate = NS(boundary=nullcontext)

            def borrow(plan, step):
                home = handles[plan.home_policy].standalone_checkpoint_manager
                donor = handles[plan.donor].standalone_checkpoint_manager
                lend = NS(lend_id=1, guests=[object()], home_replicas=plan.home_replicas)
                home.replicas.remove(plan.home_replicas[0])
                if fault != 'missing_guest':
                    donor.replicas.extend(lend.guests)
                self.scheduler.active_lends.append(lend)
                return lend

            def return_(lend, step):
                handles['policy_1'].standalone_checkpoint_manager.replicas = list(handles['policy_1'].replicas)
                handles['policy_2'].standalone_checkpoint_manager.replicas = list(handles['policy_2'].replicas)
                self.scheduler.active_lends.remove(lend)

            # Deliberately no controller.handles: this mirrors production.
            self.executor = NS(handles=handles, borrow=borrow, return_=return_, validate_membership=lambda: None)

        def run_boundary(self, trainer):
            for handle in self.executor.handles.values():
                handle.standalone_checkpoint_manager.update_weights()
            if fault == 'duplicate_sync':
                self.executor.handles['policy_1'].standalone_checkpoint_manager.update_weights()

    class Trainer:
        def train_step(self):
            self._dynamic_inference.run_boundary(self)
            return {'step': self.global_steps}

    import examples.multi_agent_blackbox as package
    fake_patch = ModuleType('examples.multi_agent_blackbox.verl_patch')
    fake_patch.apply_patch = lambda: None
    monkeypatch.setitem(sys.modules, fake_patch.__name__, fake_patch)
    monkeypatch.setattr(package, 'verl_patch', fake_patch, raising=False)
    for name, attrs in {
        'uni_agent.trainer.dynamic_inference.controller': {'DynamicInferenceController': Controller},
        'uni_agent.trainer.multi_agents_ppo_trainer': {'MultiAgentsPPOTrainer': Trainer},
        'verl.utils.ray_utils': {'auto_await': auto_await},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    smoke = importlib.import_module('examples.multi_agent_blackbox.transaction_smoke_patch')
    monkeypatch.setattr(smoke, '_APPLIED', False)
    smoke.apply_patch()
    controller = Controller()
    controller.setup()
    trainer = Trainer()
    trainer._dynamic_inference = controller
    trainer.global_steps = 1
    if fault:
        with pytest.raises(AssertionError):
            trainer.train_step()
        assert '"event": "passed"' not in capsys.readouterr().out
        return
    trainer.train_step()
    trainer.global_steps = 2
    trainer.train_step()
    out = capsys.readouterr().out
    for event in ('borrowed', 'returned', 'passed'):
        assert f'"event": "{event}"' in out
    assert out.count('"event": "boundary_verified"') == 2
    assert [len(controller._smoke_sync_calls[p]) for p in controller.executor.handles] == [2, 2, 2]
