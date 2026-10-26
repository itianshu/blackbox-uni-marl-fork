from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest


PATCH_PATH = Path("examples/multi_agent_blackbox/verl_patch.py")


@pytest.mark.parametrize("mode", ["colocated", "standalone"])
def test_replica_initializers_forward_resource_pool(monkeypatch, mode):
    loaded = _load_patch_module(monkeypatch)
    replica = types.SimpleNamespace(config=None, name_suffix="")
    pool = object()
    calls = []

    def original(self, resource_pool):
        calls.append((self, resource_pool))
        return "ready"

    monkeypatch.setattr(loaded.module, "_ORIG_REPLICA_INIT_" + mode.upper(), original)
    wrapper = getattr(loaded.module, "_patched_replica_init_" + mode)
    assert wrapper(replica, resource_pool=pool) == "ready"
    assert wrapper(replica, pool) == "ready"
    assert calls == [(replica, pool), (replica, pool)]


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _install_package(monkeypatch, name: str):
    package = types.ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)
    return package


def _install_module(monkeypatch, name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_patch_module(monkeypatch, *, install_vllm_targets: bool = True):
    ray = _install_package(monkeypatch, "ray")
    ray.nodes = lambda: []
    ray_util = _install_package(monkeypatch, "ray.util")
    ray.util = ray_util

    for name in (
        "verl",
        "verl.single_controller",
        "verl.single_controller.ray",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.trainer.ppo.v1",
        "verl.experimental",
        "verl.experimental.reward_loop",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.vllm_rollout",
    ):
        _install_package(monkeypatch, name)

    class FakeRayResourcePool:
        def __init__(self, *args, **kwargs):
            self.init_args = (args, kwargs)

    class FakeResourcePoolManager:
        def create_resource_pool(self):
            self.resource_pool_dict = {}

    class FakePPOTrainer:
        def _init_resource_pool_mgr(self):
            return None

    class FakeRewardLoopManager:
        def _init_reward_loop_workers(self):
            return None

    class FakeRolloutReplica:
        async def init_colocated(self, resource_pool):
            self.resource_pool = resource_pool

        async def init_standalone(self, resource_pool=None):
            self.resource_pool = resource_pool

    class FakeVLLMReplica(FakeRolloutReplica):
        def _get_server_name_prefix(self):
            return "original_creator_"

        async def launch_servers(self):
            return None

    class FakeServerAdapter:
        def _get_server_name_prefix(self):
            return "original_lookup_"

        async def update_weights(self, weights, global_steps=None, **kwargs):
            return None

    class FakeVLLMHttpServer:
        def __init__(self, *args, **kwargs):
            pass

    _install_module(monkeypatch, "verl.workers.rollout.replica", RolloutReplica=FakeRolloutReplica)

    _install_module(
        monkeypatch,
        "verl.single_controller.ray.base",
        RayResourcePool=FakeRayResourcePool,
        ResourcePoolManager=FakeResourcePoolManager,
    )
    _install_module(
        monkeypatch,
        "verl.trainer.ppo.v1.trainer_base",
        PPOTrainer=FakePPOTrainer,
    )
    _install_module(
        monkeypatch,
        "verl.experimental.reward_loop.reward_loop",
        RewardLoopManager=FakeRewardLoopManager,
    )
    if install_vllm_targets:
        _install_module(
            monkeypatch,
            "verl.workers.rollout.vllm_rollout.vllm_async_server",
            vLLMReplica=FakeVLLMReplica,
            vLLMHttpServer=FakeVLLMHttpServer,
        )
        _install_module(
            monkeypatch,
            "verl.workers.rollout.vllm_rollout.vllm_rollout",
            ServerAdapter=FakeServerAdapter,
        )

    module_name = f"_multi_agent_blackbox_verl_patch_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(
        module=module,
        RayResourcePool=FakeRayResourcePool,
        ResourcePoolManager=FakeResourcePoolManager,
        PPOTrainer=FakePPOTrainer,
        RewardLoopManager=FakeRewardLoopManager,
        vLLMReplica=FakeVLLMReplica,
        ServerAdapter=FakeServerAdapter,
    )


def test_apply_worker_patch_uses_policy_qualified_and_fallback_prefixes(monkeypatch):
    loaded = _load_patch_module(monkeypatch)
    loaded.module.apply_worker_patch()

    policy_adapter = loaded.ServerAdapter()
    policy_adapter.config = AttrDict(custom={"policy_name": "policy_1"})
    fallback_adapter = loaded.ServerAdapter()
    fallback_adapter.config = AttrDict(custom={})

    assert policy_adapter._get_server_name_prefix() == "policy_1_vllm_"
    assert fallback_adapter._get_server_name_prefix() == "vllm_"


def test_worker_patch_is_idempotent_and_restorable(monkeypatch):
    loaded = _load_patch_module(monkeypatch)
    original = loaded.ServerAdapter._get_server_name_prefix

    loaded.module.apply_worker_patch()
    installed = loaded.ServerAdapter._get_server_name_prefix
    loaded.module.apply_worker_patch()

    assert loaded.ServerAdapter._get_server_name_prefix is installed
    assert loaded.module._WORKER_PATCHED is True

    loaded.module.restore()

    assert loaded.ServerAdapter._get_server_name_prefix is original
    assert loaded.module._WORKER_PATCHED is False


def test_worker_patch_defers_import_until_worker_loads_target(monkeypatch):
    loaded = _load_patch_module(monkeypatch, install_vllm_targets=False)
    import threading

    started = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.target = kwargs["target"]

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    loaded.module.apply_worker_patch()

    assert len(started) == 1
    assert loaded.module._WORKER_PATCHED is True
    assert loaded.module._LOOKUP_PATCHED is False
    assert loaded.module._ORIG_SERVER_ADAPTER_NAME_PREFIX is None


def test_apply_patch_composes_worker_patch(monkeypatch):
    loaded = _load_patch_module(monkeypatch)
    original_lookup = loaded.ServerAdapter._get_server_name_prefix

    loaded.module.apply_patch()

    assert loaded.module._PATCHED is True
    assert loaded.module._WORKER_PATCHED is True
    assert loaded.ServerAdapter._get_server_name_prefix is not original_lookup

    loaded.module.restore()
    assert loaded.ServerAdapter._get_server_name_prefix is original_lookup


def test_creator_and_lookup_prefixes_match_for_each_policy(monkeypatch):
    loaded = _load_patch_module(monkeypatch)
    loaded.module.apply_patch()

    for policy_name in ("policy_1", "policy_2"):
        config = AttrDict(custom={"policy_name": policy_name})
        replica = loaded.vLLMReplica()
        replica.config = config
        adapter = loaded.ServerAdapter()
        adapter.config = config

        assert replica._get_server_name_prefix() == adapter._get_server_name_prefix()
        assert replica._get_server_name_prefix() == f"{policy_name}_vllm_"

    loaded.module.restore()


def test_reward_worker_names_start_with_policy(monkeypatch):
    loaded = _load_patch_module(monkeypatch)
    ray = sys.modules["ray"]
    ray.nodes = lambda: [
        {"NodeID": "node-1", "Alive": True, "Resources": {"CPU": 1}},
    ]
    ray.util.scheduling_strategies = types.SimpleNamespace(
        NodeAffinitySchedulingStrategy=lambda **kwargs: kwargs,
    )

    class FakeRewardWorker:
        names = []

        @classmethod
        def options(cls, **kwargs):
            cls.names.append(kwargs["name"])
            return cls

        @classmethod
        def remote(cls, *args):
            return types.SimpleNamespace(args=args)

    loaded.module.apply_patch()
    manager = loaded.RewardLoopManager()
    manager.config = AttrDict(
        policy_name="policy_1",
        reward=AttrDict(num_workers=2),
    )
    manager.reward_loop_workers_class = FakeRewardWorker
    manager.reward_router_address = "reward-router"

    manager._init_reward_loop_workers()

    assert FakeRewardWorker.names == [
        "policy_1_reward_loop_worker_0",
        "policy_1_reward_loop_worker_1",
    ]
    loaded.module.restore()
