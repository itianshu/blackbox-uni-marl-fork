"""Tests for the verl runtime patches (sleep/wake, guest external pool).

verl modules are stubbed in sys.modules — the patch module resolves every
verl import lazily (function bodies / apply time), so stubs fully control
what the patches see.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from uni_agent.trainer.dynamic_inference import patch


# --------------------------------------------------------------------- stubs
def _install_verl_stubs(monkeypatch):
    """Stub every verl module the driver-side patches import."""
    # verl.workers.rollout.replica with a RolloutReplica whose original
    # init_standalone records the call
    calls = []

    class FakeRolloutReplica:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name_suffix = ""
            self.replica_rank = kwargs.get("replica_rank", 0)
            self.workers = []
            self.resource_pool = None
            self.is_reward_model = False
            self.is_teacher_model = False
            self.gpus_per_replica_node = 8
            self.nnodes = 1

        def get_ray_class_with_init_args(self):
            return "ray_cls"

        async def launch_servers(self):
            calls.append(("launch_servers", self.replica_rank))

        async def init_standalone(self):
            calls.append(("orig_init_standalone", self))
            self.resource_pool = "fresh-pool"

    replica_mod = types.ModuleType("verl.workers.rollout.replica")
    replica_mod.RolloutReplica = FakeRolloutReplica

    class RolloutMode:
        STANDALONE = SimpleNamespace(value="standalone")

    replica_mod.RolloutMode = RolloutMode
    replica_mod.get_rollout_replica_class = lambda name: FakeRolloutReplica

    # verl.single_controller.ray with recording RayWorkerGroup / ResourcePoolManager
    class FakeRayWorkerGroup:
        def __init__(self, *, resource_pool=None, ray_cls_with_init=None,
                     bin_pack=False, name_prefix="", use_gpu=True, device_name=None):
            calls.append(("worker_group", resource_pool, name_prefix))
            self.workers = [f"worker:{name_prefix}"]

    class FakeResourcePoolManager:
        def __init__(self, *, resource_pool_spec, mapping, max_colocate_count):
            calls.append(("pool_manager", max_colocate_count))
            self.resource_pool_dict = {
                name: SimpleNamespace(
                    name=f"pool:{name}",
                    max_colocate_count=max_colocate_count,
                )
                for name in resource_pool_spec
            }

        def create_resource_pool(self):
            pass

    ray_mod = types.ModuleType("verl.single_controller.ray")
    ray_mod.RayWorkerGroup = FakeRayWorkerGroup
    ray_mod.ResourcePoolManager = FakeResourcePoolManager

    device_mod = types.ModuleType("verl.utils.device")
    device_mod.get_device_name = lambda: "cuda"

    for name in ("verl", "verl.utils", "verl.workers", "verl.workers.rollout",
                 "verl.single_controller"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.replica", replica_mod)
    monkeypatch.setitem(sys.modules, "verl.single_controller.ray", ray_mod)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device_mod)
    return replica_mod, calls


def _make_replica(replica_mod, **attrs):
    replica = replica_mod.RolloutReplica(replica_rank=1)
    for key, value in attrs.items():
        setattr(replica, key, value)
    return replica


@pytest.fixture(autouse=True)
def _reset_patch_module(monkeypatch):
    """Reset patch-module globals before and after every test."""
    patch.restore()
    monkeypatch.setattr(patch, "_SERVER_PATCHED", False)
    monkeypatch.setattr(patch, "_WORKER_PATCHED", False)
    monkeypatch.setattr(patch, "_ORIG_SERVER_SLEEP", None)
    monkeypatch.setattr(patch, "_ORIG_SERVER_WAKE_UP", None)
    yield
    patch.restore()


# ------------------------------------------------------- patch #2: guest pool
class TestInitStandalonePatch:
    def test_apply_and_restore_roundtrip(self, monkeypatch):
        replica_mod, calls = _install_verl_stubs(monkeypatch)
        original = replica_mod.RolloutReplica.init_standalone
        patch.apply_patch()
        assert replica_mod.RolloutReplica.init_standalone is not original
        # idempotent
        patch.apply_patch()
        patch.restore()
        assert replica_mod.RolloutReplica.init_standalone is original

    def test_default_path_delegates_to_original(self, monkeypatch):
        replica_mod, calls = _install_verl_stubs(monkeypatch)
        patch.apply_patch()
        replica = _make_replica(replica_mod)
        asyncio.run(replica.init_standalone())
        assert calls == [("orig_init_standalone", replica)]
        assert replica.resource_pool == "fresh-pool"

    def test_guest_path_reuses_external_pool(self, monkeypatch):
        replica_mod, calls = _install_verl_stubs(monkeypatch)
        patch.apply_patch()
        replica = _make_replica(replica_mod, _guest_external_pool="home-pool")
        asyncio.run(replica.init_standalone())

        assert ("worker_group", "home-pool", "rollout_guest_1") in calls
        # no new placement group was created
        assert not any(c[0] == "pool_manager" for c in calls)
        assert replica.resource_pool == "home-pool"
        assert replica.workers == ["worker:rollout_guest_1"]

    def test_guest_path_injects_policy_label(self, monkeypatch):
        replica_mod, calls = _install_verl_stubs(monkeypatch)
        patch.apply_patch()
        replica = _make_replica(
            replica_mod,
            _guest_external_pool="home-pool",
            config=SimpleNamespace(custom={"policy_name": "policy_b"}),
            name_suffix="_guest_b_0",
        )
        asyncio.run(replica.init_standalone())
        # the donor policy label is injected exactly like verl_patch does
        assert replica.name_suffix == "_policy_b_guest_b_0"
        assert calls[0][2] == "rollout_guest_1_policy_b_guest_b_0"

    def test_home_path_uses_graph_derived_slot_count(self, monkeypatch):
        replica_mod, calls = _install_verl_stubs(monkeypatch)
        patch.apply_patch()
        replica = _make_replica(
            replica_mod,
            config=SimpleNamespace(custom={
                "policy_name": "policy_a",
                "dynamic_inference_max_colocate_count": 4,
            }),
        )
        asyncio.run(replica.init_standalone())

        assert ("pool_manager", 4) in calls
        assert any(
            call[0] == "worker_group"
            and call[2] == "rollout_standalone_1_policy_a"
            for call in calls
        )
        assert replica.resource_pool.max_colocate_count == 4

# -------------------------------------------------- patch #1: sleep / wake
def _server_module():
    calls = []

    class FakeEngine:
        async def sleep(self, level=None):
            calls.append(("engine.sleep", level))

        async def wake_up(self, tags=None):
            calls.append(("engine.wake_up", tags))

        async def reset_prefix_cache(self, reset_connector=False):
            calls.append(("reset_prefix_cache", reset_connector))

        async def reset_encoder_cache(self):
            calls.append(("reset_encoder_cache",))

    class FakeServer:
        def __init__(self, rollout_mode="standalone", node_rank=0, free_cache_engine=True):
            self.rollout_mode = rollout_mode
            self.node_rank = node_rank
            self.config = SimpleNamespace(free_cache_engine=free_cache_engine)
            self.engine = FakeEngine()

        def _resolve_sleep_level(self):
            return 2

        def _get_wake_up_tags(self):
            return ["kv_cache", "weights"]

        async def sleep(self):
            calls.append(("orig_sleep",))

        async def wake_up(self, tags=None):
            calls.append(("orig_wake_up", tags))

    module = types.ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    module.vLLMHttpServer = FakeServer
    return module, calls


class TestServerSleepWakePatch:
    def test_standalone_sleep_hits_engine(self):
        module, calls = _server_module()
        patch._patch_server_sleep_wake_in_module(module)
        server = module.vLLMHttpServer()
        asyncio.run(server.sleep())
        # real level-2 sleep + encoder cache reset (mirrors verl's HYBRID branch)
        assert calls == [("engine.sleep", 2), ("reset_encoder_cache",)]

    def test_standalone_wake_hits_engine(self):
        module, calls = _server_module()
        patch._patch_server_sleep_wake_in_module(module)
        server = module.vLLMHttpServer()
        asyncio.run(server.wake_up())
        assert calls == [
            ("engine.wake_up", ["kv_cache", "weights"]),
            ("reset_prefix_cache", True),
        ]

    def test_hybrid_delegates_to_original(self):
        module, calls = _server_module()
        patch._patch_server_sleep_wake_in_module(module)
        server = module.vLLMHttpServer(rollout_mode=SimpleNamespace(value="hybrid"))
        asyncio.run(server.sleep())
        asyncio.run(server.wake_up())
        assert calls == [("orig_sleep",), ("orig_wake_up", None)]

    def test_node_rank_nonzero_still_sleeps_for_standalone(self):
        # vLLM's engine lives on rank 0; verl's original guard keeps node_rank 0
        module, calls = _server_module()
        patch._patch_server_sleep_wake_in_module(module)
        server = module.vLLMHttpServer(node_rank=1)
        asyncio.run(server.sleep())
        assert calls == [("orig_sleep",)]


class TestWorkerWatcher:
    def test_watcher_patches_module_once_loaded(self, monkeypatch):
        module, calls = _server_module()
        monkeypatch.setenv("UNI_AGENT_WORKER_HOOK_CHAIN", "")
        # target already importable -> the watcher patches within its poll tick
        monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.vllm_async_server", module)
        patch.apply_worker_patch()
        deadline = time.monotonic() + 5.0
        while not patch._SERVER_PATCHED and time.monotonic() < deadline:
            time.sleep(0.05)
        assert patch._SERVER_PATCHED
        server = module.vLLMHttpServer()
        asyncio.run(server.sleep())
        assert calls == [("engine.sleep", 2), ("reset_encoder_cache",)]

    def test_hook_chain_is_invoked(self, monkeypatch):
        chain_calls = []
        hook_mod = types.ModuleType("fake_worker_hooks")
        hook_mod.apply_worker_patch = lambda: chain_calls.append("chained")
        monkeypatch.setitem(sys.modules, "fake_worker_hooks", hook_mod)
        monkeypatch.setenv("UNI_AGENT_WORKER_HOOK_CHAIN", "fake_worker_hooks.apply_worker_patch")
        patch.apply_worker_patch()
        assert chain_calls == ["chained"]
