import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace


def _load_single_async_module(monkeypatch):
    class FakePPOTrainer:
        pass

    class FakePPOTrainerSeparateAsync(FakePPOTrainer):
        pass

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


def test_init_standalone_rollout_runtime_builds_native_v1_components(monkeypatch):
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
    assert trainer.current_mode is trainer_module.HybridEngineMode.ROLLOUT
    assert events == [
        ("create_standalone", trainer.config, 2),
        (
            "create_checkpoint_manager",
            "resolved:checkpoint_config",
            "actor_wg",
            ["standalone_replica"],
        ),
        "add_replicas_to_balancer",
    ]


def test_single_async_trainer_has_no_monolithic_init_runtime(monkeypatch):
    trainer_module, _ = _load_single_async_module(monkeypatch)

    assert not hasattr(trainer_module.SingleAsyncPPOTrainer, "init_runtime")
