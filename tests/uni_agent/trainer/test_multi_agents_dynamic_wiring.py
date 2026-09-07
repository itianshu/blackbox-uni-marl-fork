"""Wiring tests: the controller's four-stage boundary and the trainer hooks.

Part 1 drives the real ``DynamicInferenceController`` against a fake outer
trainer with recording scheduler/executor fakes, asserting the stage ordering
invariants (I2) and the early-return deferral.
Part 2 drives the real ``MultiAgentsPPOTrainer`` (built via ``__new__`` with
injected fakes, bypassing hydra config resolution) to verify the four wiring
points: init, on_step_end dispatch, metrics merge, cleanup ordering.
"""

import importlib.util
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


def _ensure_attrs(name, **attrs):
    """Make ``name`` importable with the given attributes.

    Never shadows a real package; when another test file already installed a
    (possibly minimal) stub, only the missing attributes are added. Returns
    (freshly_installed_names, augmented_(module, attr)_pairs) for teardown.
    """
    mod = sys.modules.get(name)
    if mod is not None:
        augmented = []
        for key, value in attrs.items():
            if not hasattr(mod, key):
                setattr(mod, key, value)
                augmented.append((mod, key))
        return [], augmented
    try:
        if importlib.util.find_spec(name) is not None:
            return [], []          # the real package is importable; leave it be
    except ValueError:
        return [], []              # synthetic module without __spec__; unusable probe
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return [name], []


class _FakeOmegaConf:
    @staticmethod
    def is_config(cfg):
        return False

    @staticmethod
    def to_container(cfg, resolve=True):
        return cfg


def _install_import_stubs():
    """Stub modules the trainer imports at module level (no ray/numpy needed)."""
    installed, augmented = [], []
    for chunk in (
        _ensure_attrs("ray",
                      actor=SimpleNamespace(ActorHandle=object),
                      remote=lambda cls: cls,
                      util=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")),
        _ensure_attrs("transfer_queue"),
        _ensure_attrs("numpy", ndarray=object),
        _ensure_attrs("omegaconf",
                      DictConfig=dict,
                      OmegaConf=_FakeOmegaConf,
                      open_dict=lambda cfg: None),
    ):
        installed.extend(chunk[0])
        augmented.extend(chunk[1])
    return installed, augmented


@pytest.fixture(autouse=True)
def _dependency_stubs():
    """Install import stubs only for the duration of each test in this module —
    installing them at collection time would leak fake modules into every other
    test file in the session."""
    installed, augmented = _install_import_stubs()
    yield
    for name in installed:
        sys.modules.pop(name, None)
    for mod, key in augmented:
        try:
            delattr(mod, key)
        except AttributeError:
            pass


from uni_agent.trainer.dynamic_inference.controller import (  # noqa: E402
    BoundaryGate,
    DynamicInferenceController,
    PollLoop,
)
from uni_agent.trainer.dynamic_inference.types import (  # noqa: E402
    BoundaryPlan,
    BorrowPlan,
    LendRecord,
    parse_scheduling_config,
)


# --------------------------------------------------------------------- fakes
class FakeOuterTrainer:
    """The surface of MultiAgentsPPOTrainer that run_boundary touches."""

    def __init__(self, log, policies=("a", "b")):
        self.log = log
        self.policy_trainers = {name: SimpleNamespace() for name in policies}
        self._policy_pool = ThreadPoolExecutor(max_workers=len(policies))
        self.global_steps = 11
        self.replay_buffer = SimpleNamespace(get_sampleable_count=lambda: 7)

    def _run_trainer_hook(self, policy_name, hook_name):
        assert hook_name == "on_step_end"
        self.log.append(("hook", policy_name))

    def close(self):
        self._policy_pool.shutdown(wait=True)


class FakeScheduler:
    def __init__(self, log, plan):
        self.log = log
        self.plan = plan
        self.decide_calls = []
        self.active_lends = []

    def decide(self, stats):
        self.decide_calls.append(stats)
        self.log.append(("decide", stats.step))
        return self.plan

    def metrics_snapshot(self):
        return {"swap_episodes": 3, "bottleneck": "a", "disabled": False}


class FakeExecutor:
    def __init__(self, log):
        self.log = log

    def return_(self, lend, step, early=False, **kwargs):
        self.log.append(("return", lend.lend_id, step, early))
        return True

    def borrow(self, borrow_plan, step, **kwargs):
        self.log.append(("borrow", borrow_plan.home_policy, borrow_plan.donor, step))
        return object()

    def renew(self, lend, step):
        self.log.append(("renew", lend.lend_id, step))
        return True

    def renew_many(self, lends, step):
        self.log.append(("renew_batch", tuple(lend.lend_id for lend in lends), step))
        return True


def _lend(home="b", donor="a", lend_id=1):
    return LendRecord(lend_id=lend_id, home_policy=home, donor=donor,
                      home_replicas=[object()], guests=[object()], since_step=10)


def _controller(trainer, plan):
    controller = DynamicInferenceController.__new__(DynamicInferenceController)
    controller.config = parse_scheduling_config({"enable": True, "mode": "resource"})
    controller.trainer = trainer
    controller.gate = BoundaryGate()
    controller.scheduler = FakeScheduler(trainer.log, plan)
    controller.executor = FakeExecutor(trainer.log)
    controller.guest_engine = None
    controller.store = None
    controller.poll_loop = None
    controller.last_metrics = {}
    controller._last_step = 0
    return controller


# ------------------------------------------------------- boundary orchestration
class TestRunBoundary:
    def test_incomplete_return_precedes_policy_hooks(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend(lend_id=2, home="a", donor="b")
        lend.return_stage = 1
        controller = _controller(trainer, BoundaryPlan())
        controller.scheduler.active_lends.append(lend)

        controller.run_boundary(trainer)
        trainer.close()

        # A partially completed return is retried before every policy update.
        first_hook = min(i for i, e in enumerate(log) if e[0] == "hook")
        assert log.index(("return", 2, 11, False)) < first_hook
        assert not any(event[0] == "borrow" for event in log)
        assert {e for e in log if e[0] == "hook"} == {("hook", "a"), ("hook", "b")}

    def test_renewal_runs_after_hooks_without_returning(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        controller = _controller(trainer, BoundaryPlan())
        controller.scheduler.active_lends.append(lend)
        controller.run_boundary(trainer)
        trainer.close()

        renew_idx = log.index(("renew_batch", (lend.lend_id,), 11))
        assert renew_idx > max(i for i, event in enumerate(log) if event[0] == "hook")
        assert not any(event[0] == "return" for event in log)

    def test_metrics_prefixed_and_stored(self):
        log = []
        trainer = FakeOuterTrainer(log)
        controller = _controller(trainer, BoundaryPlan())
        controller.run_boundary(trainer)
        trainer.close()
        assert controller.last_metrics == {
            "dynamic_inference/swap_episodes": 3,
            "dynamic_inference/bottleneck": "a",
            "dynamic_inference/disabled": False,
        }

    def test_boundary_waits_for_all_policy_hooks_when_one_fails(self):
        log = []
        trainer = FakeOuterTrainer(log)

        def hook(policy_name, hook_name):
            log.append(("hook", policy_name))
            if policy_name == "a":
                raise RuntimeError("update failed")

        trainer._run_trainer_hook = hook
        controller = _controller(trainer, BoundaryPlan())
        with pytest.raises(RuntimeError, match="update failed"):
            controller.run_boundary(trainer)
        trainer.close()
        assert {event for event in log if event[0] == "hook"} == {
            ("hook", "a"), ("hook", "b"),
        }

    def test_failed_recovery_return_does_not_block_policy_hooks(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        lend.return_stage = 1
        controller = _controller(trainer, BoundaryPlan())
        controller.scheduler.active_lends.append(lend)

        def fail_return(lend, step, early=False):
            log.append(("return_failed", lend.lend_id, step, early))
            return False

        controller.executor.return_ = fail_return
        controller.run_boundary(trainer)
        trainer.close()

        assert ("return_failed", 1, 11, False) in log
        assert not any(event[0] == "borrow" for event in log)
        assert {event for event in log if event[0] == "hook"} == {
            ("hook", "a"), ("hook", "b"),
        }

    def test_failed_renewal_is_safely_returned(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        controller = _controller(trainer, BoundaryPlan())
        controller.scheduler.active_lends.append(lend)

        def fail_renew(active_lends, step):
            log.append(("renew_failed", tuple(lend.lend_id for lend in active_lends), step))
            return False

        controller.executor.renew_many = fail_renew
        controller.run_boundary(trainer)
        trainer.close()

        assert ("renew_failed", (lend.lend_id,), 11) in log
        assert ("return", lend.lend_id, 11, True) in log
        assert not any(event[0] == "borrow" for event in log)


class TestImmediateRebalance:
    def test_fresh_metrics_can_borrow_without_waiting_for_step_end(self):
        log = []
        trainer = FakeOuterTrainer(log)
        borrow = BorrowPlan(home_policy="b", donor="a", home_replicas=[object()])
        controller = _controller(trainer, BoundaryPlan(borrows=[borrow]))

        controller._rebalance_now()
        trainer.close()

        assert ("decide", 11) in log
        assert ("borrow", "b", "a", 11) in log
        assert not any(event[0] == "hook" for event in log)

    def test_return_failure_blocks_immediate_borrow(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        borrow = BorrowPlan(home_policy="b", donor="a", home_replicas=[object()])
        controller = _controller(
            trainer, BoundaryPlan(returns=[lend], borrows=[borrow]))

        controller.executor.return_ = lambda *args, **kwargs: False
        controller._rebalance_now()
        trainer.close()

        assert not any(event[0] == "borrow" for event in log)

    def test_same_unit_handoff_keeps_home_inactive(self):
        log = []
        trainer = FakeOuterTrainer(log)
        replica = object()
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[replica], guests=[object()], since_step=10,
        )
        borrow = BorrowPlan(
            home_policy="b", donor="c", home_replicas=[replica])
        controller = _controller(
            trainer, BoundaryPlan(returns=[lend], borrows=[borrow]))
        calls = []

        def return_(item, step, early=False, **kwargs):
            calls.append(("return", kwargs))
            return True

        def borrow_(item, step, **kwargs):
            calls.append(("borrow", kwargs))
            return object()

        controller.executor.return_ = return_
        controller.executor.borrow = borrow_
        controller._rebalance_now()
        trainer.close()

        assert calls == [
            ("return", {"reactivate_home": False}),
            ("borrow", {"home_already_sleeping": True}),
        ]

    def test_partial_return_is_not_treated_as_direct_handoff(self):
        log = []
        trainer = FakeOuterTrainer(log)
        replica = object()
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[replica], guests=[object()], since_step=10,
            return_stage=4,
        )
        borrow = BorrowPlan(
            home_policy="b", donor="c", home_replicas=[replica])
        controller = _controller(
            trainer, BoundaryPlan(returns=[lend], borrows=[borrow]))
        calls = []

        def return_(item, step, early=False, **kwargs):
            calls.append(("return", kwargs))
            return True

        def borrow_(item, step, **kwargs):
            calls.append(("borrow", kwargs))
            return object()

        controller.executor.return_ = return_
        controller.executor.borrow = borrow_
        controller._rebalance_now()
        trainer.close()

        assert calls == [("return", {}), ("borrow", {})]


class TestEarlyReturnDeferral:
    def test_early_return_deferred_during_boundary(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()  # donor a
        controller = _controller(trainer, BoundaryPlan())
        controller.scheduler.active_lends.append(lend)

        # an early return fires while a boundary is active: deferred, not run
        with controller.gate.boundary():
            controller._on_early_return(lend)
        assert not any(e[0] == "return" for e in log)
        assert controller.gate.take_pending() == [lend]

        # hand it back to the deferral list (the next boundary consumes it)
        controller.gate.defer_early_return(lend)
        controller.run_boundary(trainer)
        trainer.close()

        # the deferred early return was physically returned at the next boundary
        assert ("return", 1, 11, True) in log

    def test_stale_deferred_return_is_ignored(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        controller = _controller(trainer, BoundaryPlan())
        controller.gate.defer_early_return(lend)

        # The lend was already returned by the boundary that raced with the
        # deferral, so it is no longer present in the scheduler registry.
        controller.run_boundary(trainer)
        trainer.close()

        assert not any(event[0] == "return" for event in log)

    def test_early_return_runs_immediately_outside_boundary(self):
        log = []
        trainer = FakeOuterTrainer(log)
        lend = _lend()
        controller = _controller(trainer, BoundaryPlan())

        controller._on_early_return(lend)
        assert ("return", 1, 0, True) in log  # early=True, executed inline


class TestBoundaryGate:
    def test_exclusive_sections_serialize(self):
        gate = BoundaryGate()
        order = []

        with gate.boundary():
            ran = gate.run_exclusive(lambda: order.append("early"))  # deferred
            assert ran is False
        assert gate.run_exclusive(lambda: order.append("early")) is True
        assert order == ["early"]

    def test_cross_thread_call_is_deferred_without_waiting_for_boundary(self):
        gate = BoundaryGate()
        result = []

        with gate.boundary():
            thread = threading.Thread(
                target=lambda: result.append(gate.run_exclusive(lambda: None)))
            thread.start()
            thread.join(timeout=1.0)
            assert not thread.is_alive()
            assert result == [False]

    def test_poll_loop_stops_and_joins_cleanly(self):
        seen = threading.Event()

        class Source:
            def sample_all(self):
                return {"a": SimpleNamespace(metrics_fresh=True)}

        class Store:
            def record_all(self, samples):
                assert set(samples) == {"a"}

        class Scheduler:
            def poll(self, *, metrics_fresh):
                assert metrics_fresh is True
                seen.set()
                return True

        rebalanced = threading.Event()
        loop = PollLoop(
            Source(), Store(), Scheduler(), BoundaryGate(), 0.05,
            on_metrics_ready=rebalanced.set,
        )
        loop.start()
        assert seen.wait(timeout=1.0)
        assert rebalanced.wait(timeout=1.0)
        loop.stop()
        loop.join(timeout=1.0)
        assert not loop.is_alive()

    def test_stale_poll_result_is_rejected_after_boundary(self):
        gate = BoundaryGate()
        generation = gate.snapshot_generation()
        with gate.boundary():
            pass

        committed = []
        assert gate.run_exclusive(
            lambda: committed.append(True),
            expected_generation=generation,
        ) is False
        assert committed == []


# ------------------------------------------------------------- trainer wiring
class TestTrainerWiring:
    def _real_trainer(self, config):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        trainer.config = config
        trainer.trainer_mode = "separate_async"
        trainer.parameter_sync_step = 1
        trainer.train_batch_size = 4
        trainer.save_freq = -1
        trainer.total_training_steps = 1
        trainer.policy_configs = {}
        trainer.policy_trainers = {}
        trainer.replay_buffer = None
        trainer.timing_raw = {}
        trainer.agent_loop_manager = None
        trainer.global_steps = 3
        trainer._policy_pool = ThreadPoolExecutor(max_workers=2)
        trainer._dynamic_inference = None
        return trainer

    def _config(self, enabled):
        return SimpleNamespace(
            trainer=SimpleNamespace(save_freq=-1, total_training_steps=1,
                                    v1=SimpleNamespace(trainer_mode="separate_async")),
            get=lambda key, default=None: (
                {"enable": True, "mode": "resource"} if enabled else default
            ),
        )

    def test_disabled_config_leaves_controller_none(self):
        trainer = self._real_trainer(self._config(enabled=False))
        trainer._init_dynamic_inference()
        assert trainer._dynamic_inference is None

    def test_prepare_injects_graph_derived_home_slots(self, monkeypatch):
        from contextlib import nullcontext
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.dynamic_inference import patch as dynamic_patch

        applied = []
        monkeypatch.setattr(dynamic_patch, "apply_patch", lambda: applied.append(True))
        monkeypatch.setattr(trainer_module, "open_dict", nullcontext)
        config = {
            "dynamic_inference_scheduling": {
                "enable": True,
                "mode": "resource",
                "borrowing": {"pairs": []},
            },
        }
        trainer = self._real_trainer(config)
        trainer.policy_configs = {
            name: SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout={"custom": {}}),
            )
            for name in ("a", "b", "c")
        }

        trainer._prepare_dynamic_inference_slots()

        assert applied == [True]
        assert {
            name: cfg.actor_rollout_ref.rollout["custom"]["dynamic_inference_max_colocate_count"]
            for name, cfg in trainer.policy_configs.items()
        } == {"a": 3, "b": 3, "c": 3}

    def test_enabled_config_builds_and_setups_controller(self, monkeypatch):
        events = []

        class FakeController:
            def __init__(self, config, trainer):
                events.append(("create", config.mode))

            @classmethod
            def maybe_create(cls, trainer):
                config = parse_scheduling_config(
                    trainer.config.get("dynamic_inference_scheduling"))
                if config is None:
                    return None
                return cls(config, trainer)

            def setup(self):
                events.append(("setup",))

            def run_boundary(self, trainer):
                events.append(("run_boundary", trainer.global_steps))

            def shutdown(self):
                events.append(("shutdown",))

        import uni_agent.trainer.dynamic_inference.controller as controller_mod

        monkeypatch.setattr(controller_mod, "DynamicInferenceController", FakeController)

        trainer = self._real_trainer(self._config(enabled=True))
        trainer._init_dynamic_inference()
        assert trainer._dynamic_inference is not None
        assert ("create", "resource") in events
        assert ("setup",) in events

        # on_step_end delegates to the controller instead of the plain fan-out
        hooked = []
        trainer.policy_trainers = {
            "a": SimpleNamespace(on_step_end=lambda: hooked.append("a")),
            "b": SimpleNamespace(on_step_end=lambda: hooked.append("b")),
        }
        trainer.on_step_end()
        assert hooked == []  # hooks run inside the controller, not directly
        assert ("run_boundary", 3) in events

        # cleanup shuts the controller down before the per-policy teardown
        trainer.policy_trainers = {
            "a": SimpleNamespace(cleanup=lambda: events.append("policy_cleanup")),
        }
        trainer.cleanup()
        assert events.index(("shutdown",)) < events.index("policy_cleanup")
        assert trainer._dynamic_inference is None

    def test_feature_off_runs_original_on_step_end_path(self):
        trainer = self._real_trainer(self._config(enabled=False))
        hooked = []
        trainer.policy_trainers = {
            "a": SimpleNamespace(on_step_end=lambda: hooked.append("a")),
            "b": SimpleNamespace(on_step_end=lambda: hooked.append("b")),
        }
        trainer.on_step_end()
        assert sorted(hooked) == ["a", "b"]
        trainer._policy_pool.shutdown(wait=True)

    def test_shutdown_isolated_from_cleanup_failures(self):
        trainer = self._real_trainer(self._config(enabled=True))

        class ExplodingController:
            def shutdown(self):
                raise RuntimeError("boom")

        trainer._dynamic_inference = ExplodingController()
        trainer.policy_trainers = {}
        trainer.cleanup()  # must not raise
        assert trainer._dynamic_inference is None
