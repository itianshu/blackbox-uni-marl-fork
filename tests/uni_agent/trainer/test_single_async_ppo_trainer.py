import importlib.util
import sys
import types
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_single_async_module(monkeypatch):
    class FakePPOTrainer:
        pass

    class FakePPOTrainerSeparateAsync(FakePPOTrainer):
        _enable_hybrid_replicas = True

    class FakeHybridEngineMode(Enum):
        TRAINER = "trainer"
        ROLLOUT = "rollout"

    checkpoint_module = types.ModuleType("verl.checkpoint_engine")
    checkpoint_module.CheckpointEngineManager = type("CheckpointEngineManager", (), {})
    trainer_base_module = types.ModuleType("verl.trainer.ppo.v1.trainer_base")
    trainer_base_module.PPOTrainer = FakePPOTrainer
    separate_async_module = types.ModuleType("verl.trainer.ppo.v1.trainer_separate_async")
    separate_async_module.HybridEngineMode = FakeHybridEngineMode
    separate_async_module.PPOTrainerSeparateAsync = FakePPOTrainerSeparateAsync
    config_module = types.ModuleType("verl.utils.config")
    config_module.omega_conf_to_dataclass = lambda value: value
    llm_server_module = types.ModuleType("verl.workers.rollout.llm_server")
    llm_server_module.LLMServerManager = type("LLMServerManager", (), {})

    monkeypatch.setitem(sys.modules, "verl.checkpoint_engine", checkpoint_module)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.trainer_base", trainer_base_module)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.trainer_separate_async", separate_async_module)
    monkeypatch.setitem(sys.modules, "verl.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.llm_server", llm_server_module)

    module_path = (
        Path(__file__).resolve().parents[3]
        / "uni_agent"
        / "trainer"
        / "single_async_ppo_trainer.py"
    )
    module_name = "single_async_ppo_trainer_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakePPOTrainer


def test_init_training_runtime_stops_before_separate_async_standalone_setup(monkeypatch):
    trainer_module, ppo_trainer_cls = _load_single_async_module(monkeypatch)

    events = []
    trainer = object.__new__(trainer_module.SingleAsyncPPOTrainer)

    def base_setup(self):
        events.append("base_setup")
        self._init_resource_pool_mgr()

    monkeypatch.setattr(
        ppo_trainer_cls,
        "_setup",
        base_setup,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_module.PPOTrainerSeparateAsync,
        "_init_resource_pool_mgr",
        lambda self: events.append("separate_async_resource_pool"),
        raising=False,
    )

    trainer.init_training_runtime()

    assert events == ["base_setup", "separate_async_resource_pool"]


@pytest.mark.parametrize("hybrid_enabled", [True, False])
def test_init_standalone_rollout_runtime_builds_native_v1_components(monkeypatch, hybrid_enabled):
    trainer_module, _ = _load_single_async_module(monkeypatch)

    events = []
    standalone_manager = SimpleNamespace(get_replicas=lambda: ["standalone_replica"])

    class FakeLLMServerManager:
        @classmethod
        def create(cls, *, config, start_rank):
            events.append(("create_standalone", config, start_rank))
            return standalone_manager

    class FakeCheckpointEngineManager:
        def __init__(self, *, config, actor_wg, replicas):
            events.append(("create_checkpoint_manager", config, actor_wg, replicas))

    monkeypatch.setattr(trainer_module, "LLMServerManager", FakeLLMServerManager)
    monkeypatch.setattr(trainer_module, "CheckpointEngineManager", FakeCheckpointEngineManager)
    monkeypatch.setattr(trainer_module, "omega_conf_to_dataclass", lambda config: f"resolved:{config}")

    trainer = object.__new__(trainer_module.SingleAsyncPPOTrainer)
    trainer._enable_hybrid_replicas = hybrid_enabled
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(checkpoint_engine="checkpoint_config")
        )
    )
    trainer.actor_rollout_wg = "actor_wg"
    trainer.llm_server_manager = SimpleNamespace(rollout_replicas=["hybrid_0", "hybrid_1"])
    trainer.add_replicas_to_balancer = lambda: events.append("add_replicas_to_balancer")

    trainer.init_standalone_rollout_runtime()

    assert trainer.standalone_server_manager is standalone_manager
    assert isinstance(trainer.standalone_checkpoint_manager, FakeCheckpointEngineManager)
    expected_mode = (
        trainer_module.HybridEngineMode.ROLLOUT if hybrid_enabled else trainer_module.HybridEngineMode.TRAINER
    )
    assert trainer.current_mode is expected_mode
    assert events == [
        ("create_standalone", trainer.config, 2),
        (
            "create_checkpoint_manager",
            "resolved:checkpoint_config",
            "actor_wg",
            ["standalone_replica"],
        ),
    ] + (["add_replicas_to_balancer"] if hybrid_enabled else [])


def test_single_async_trainer_has_no_monolithic_init_runtime(monkeypatch):
    trainer_module, _ = _load_single_async_module(monkeypatch)

    assert not hasattr(trainer_module.SingleAsyncPPOTrainer, "init_runtime")


def test_outer_sample_end_switches_hybrid_without_standalone_sample_timers(monkeypatch):
    trainer_module, _ = _load_single_async_module(monkeypatch)
    trainer = object.__new__(trainer_module.SingleAsyncPPOTrainer)
    trainer.current_mode = trainer_module.HybridEngineMode.ROLLOUT
    events = []

    def standalone_sample_end(self):
        # New verl expects timers from its own step loop, which this adapter
        # deliberately does not run.
        self._step_sample_wait_seconds += 1

    def switch_to_trainer(self):
        events.extend(["remove_hybrid_routing", "abort_hybrid", "sleep_hybrid"])
        self.current_mode = trainer_module.HybridEngineMode.TRAINER

    monkeypatch.setattr(
        trainer_module.PPOTrainerSeparateAsync, "on_sample_end", standalone_sample_end, raising=False
    )
    monkeypatch.setattr(
        trainer_module.PPOTrainerSeparateAsync, "switch_to_trainer", switch_to_trainer, raising=False
    )

    trainer.on_sample_end()
    trainer.on_sample_end()

    assert trainer.current_mode is trainer_module.HybridEngineMode.TRAINER
    assert events == ["remove_hybrid_routing", "abort_hybrid", "sleep_hybrid"]


def test_get_llm_client_uses_abort_safe_adapter(monkeypatch):
    trainer_module, _ = _load_single_async_module(monkeypatch)

    adapter_module = types.ModuleType("uni_agent.trainer.resumable_llm_client")
    adapter_cls = type("ResumableFullyAsyncLLMServerClient", (), {})
    adapter_module.ResumableFullyAsyncLLMServerClient = adapter_cls
    monkeypatch.setitem(sys.modules, "uni_agent.trainer.resumable_llm_client", adapter_module)

    captured = {}
    trainer = object.__new__(trainer_module.SingleAsyncPPOTrainer)
    trainer.standalone_server_manager = SimpleNamespace(
        get_client=lambda *, client_cls: captured.setdefault("client_cls", client_cls) or "client"
    )

    trainer.get_llm_client()

    assert captured["client_cls"] is adapter_cls


def test_outer_step_end_publishes_without_standalone_fit_state(monkeypatch):
    trainer_module, _ = _load_single_async_module(monkeypatch)
    trainer = object.__new__(trainer_module.SingleAsyncPPOTrainer)
    trainer.global_steps = 2
    trainer.timing_raw = {}
    published = []

    @contextmanager
    def marked_timer(name, timing_raw, **kwargs):
        yield
        timing_raw[name] = 1.25

    debug = types.ModuleType("verl.utils.debug")
    debug.marked_timer = marked_timer
    monkeypatch.setitem(sys.modules, "verl.utils.debug", debug)

    def update_weights(step):
        published.append(step)
        return {"checkpoint_sync/seconds": 1.0}

    trainer.standalone_checkpoint_manager = SimpleNamespace(update_weights=update_weights)
    # No prev_step_profile, per-step timers, replay buffer, or adaptive-switch
    # state exists when MultiAgentsPPOTrainer owns the loop.
    trainer.on_step_end()

    assert published == [2]
    assert trainer.timing_raw == {"update_weights": 1.25}
    assert trainer._pending_sync_metrics == {"checkpoint_sync/seconds": 1.0}
