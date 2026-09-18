"""Runtime patches for verl that enable dynamic inference scheduling.

No verl file is modified. Three patches, following the conventions of
``examples/multi_agent_blackbox/verl_patch.py`` (module-level ``_ORIG_*``
globals, idempotent ``apply_patch``/``restore``, worker-side ``sys.modules``
watcher installed by Ray's ``worker_process_setup_hook``):

1. **STANDALONE sleep/wake (worker side)** — verl's ``vLLMHttpServer.sleep``/
   ``wake_up`` are no-ops on the STANDALONE branch (``"skip sleep in
   standalone mode"``), because nothing in vanilla separate-async ever sleeps a
   standalone replica. Guest lending needs real level-2 sleep/wake on those
   replicas, so the patch makes the branch call the engine directly, mirroring
   verl's own HYBRID/COLOCATED branches.

2. **``RolloutReplica.init_standalone`` pool layout (driver side)** — home
   replicas use a graph-derived ``max_colocate_count`` (one home slot plus one
   slot per outgoing donor direction), and a guest
   replica must NOT create its own placement group (a second ``rollout_pool_``
   PG cannot be scheduled onto cards already reserved by the home replica's
   PG). When ``replica._guest_external_pool`` is set, the patched init reuses
   a merged/split view of the home unit's existing PGs. Home and all guest
   workers reserve the same ``1 / max_colocate_count`` GPU fraction. Sleeping
   alternatives may therefore coexist in Ray while only one engine per card is
   awake — the same fractional-reservation mechanism verl uses for colocated
   engines.

3. **Topology-aware NCCL checkpoint groups (worker side)** — verl normally
   keeps checkpoint collectives alive between updates. Dynamic borrowing
   changes the rollout worker set, so surviving workers can receive a new rank
   or world size. The patch destroys and recreates a group only when that
   topology actually changes; unchanged updates retain verl's fast path.

Patch #1 runs in every Ray worker process via
``worker_process_setup_hook: uni_agent.trainer.dynamic_inference.patch.apply_worker_patch``
(that FQN chains the example patch's worker hook, see ``_WORKER_HOOK_CHAIN``).
Patch #2 runs in the driver via ``apply_patch()``, chained ON TOP of the
example patch: by the time the controller applies this module, the example
patch's wrappers are already installed, and delegation preserves them.
"""

from __future__ import annotations

import logging
import os
import itertools
import time

logger = logging.getLogger(__name__)

# Worker-side hooks to chain before our own (the yaml allows a single
# worker_process_setup_hook FQN, so ours must install the example's too).
# Override with env var UNI_AGENT_WORKER_HOOK_CHAIN="fqn1,fqn2" (empty = none).
_WORKER_HOOK_CHAIN = ["examples.multi_agent_blackbox.verl_patch.apply_worker_patch"]

_ORIG_INIT_STANDALONE = None      # RolloutReplica.init_standalone at apply time
_ORIG_SERVER_SLEEP = None         # vLLMHttpServer.sleep at patch time
_ORIG_SERVER_WAKE_UP = None       # vLLMHttpServer.wake_up at patch time
_ORIG_SERVER_IS_SLEEPING = None   # optional probe method at patch time
_PATCHED_SERVER_CLASS = None
_ORIG_NCCL_INIT_PROCESS_GROUP = None
_PATCHED_NCCL_CLASS = None
_ORIG_CHECKPOINT_BUILD_PROCESS_GROUP = None
_CHECKPOINT_MANAGER_PATCHED = False
_CHECKPOINT_GROUP_EPOCH = itertools.count()
_PATCHED = False                  # driver-side external-pool patch (#2)
_WORKER_PATCHED = False           # worker-side watcher armed (#1)
_WORKER_PATCH_GENERATION = 0      # invalidates watcher threads on restore
_SERVER_PATCHED = False           # vLLMHttpServer.sleep/wake_up actually wrapped
_NCCL_PATCHED = False             # NCCL checkpoint topology wrapper installed


# --------------------------------------------------------------------------- #
# Patch #1: real level-2 sleep/wake for STANDALONE replicas (worker side)
# --------------------------------------------------------------------------- #


async def _patched_server_sleep(self):
    """vLLMHttpServer.sleep with a real STANDALONE branch.

    Prefer verl's own ``_sleep_hybrid`` helper because it owns the
    MTP/LoRA/NPU level-selection logic.  Some verl revisions instead expose a
    ``_resolve_sleep_level`` helper, so retain that as a compatibility path.
    """
    if (
        self.rollout_mode is not None
        and getattr(self.rollout_mode, "value", self.rollout_mode) == "standalone"
        and self.node_rank == 0
        and getattr(self.config, "free_cache_engine", True)
    ):
        sleep_hybrid = getattr(self, "_sleep_hybrid", None)
        if callable(sleep_hybrid):
            await sleep_hybrid()
            return

        resolve_level = getattr(self, "_resolve_sleep_level", None)
        sleep_level = resolve_level() if callable(resolve_level) else 2
        await self.engine.sleep(level=sleep_level)
        await self.engine.reset_encoder_cache()
        return
    return await _ORIG_SERVER_SLEEP(self)


async def _patched_server_wake_up(self, tags=None):
    """vLLMHttpServer.wake_up with a real STANDALONE branch.

    Same tags as verl's HYBRID/COLOCATED branches (``["kv_cache", "weights"]``)
    plus the prefix-cache connector reset (a stale external KV store was built
    against the pre-sleep weights).
    """
    if (
        self.rollout_mode is not None
        and getattr(self.rollout_mode, "value", self.rollout_mode) == "standalone"
        and self.node_rank == 0
    ):
        await self.engine.wake_up(tags=tags or self._get_wake_up_tags())
        await self.engine.reset_prefix_cache(reset_connector=True)
        return
    return await _ORIG_SERVER_WAKE_UP(self, tags)


async def _dynamic_inference_is_sleeping(self):
    """Return vLLM's engine sleep state for the setup-time patch probe."""
    checker = getattr(self.engine, "is_sleeping", None)
    if not callable(checker):
        raise RuntimeError("vLLM engine does not expose is_sleeping()")
    return bool(await checker())


def _patch_server_sleep_wake_in_module(module) -> None:
    """Wrap vLLMHttpServer.sleep/wake_up in this process (idempotent)."""
    global _ORIG_SERVER_SLEEP, _ORIG_SERVER_WAKE_UP, _ORIG_SERVER_IS_SLEEPING
    global _PATCHED_SERVER_CLASS, _SERVER_PATCHED
    if _SERVER_PATCHED:
        return
    server_cls = getattr(module, "vLLMHttpServer", None)
    if server_cls is None:
        return
    _ORIG_SERVER_SLEEP = server_cls.sleep
    _ORIG_SERVER_WAKE_UP = server_cls.wake_up
    _ORIG_SERVER_IS_SLEEPING = getattr(server_cls, "dynamic_inference_is_sleeping", None)
    _PATCHED_SERVER_CLASS = server_cls
    server_cls.sleep = _patched_server_sleep
    server_cls.wake_up = _patched_server_wake_up
    server_cls.dynamic_inference_is_sleeping = _dynamic_inference_is_sleeping
    _SERVER_PATCHED = True
    logger.info(
        "dynamic_inference.patch: standalone level-2 sleep/wake enabled on vLLMHttpServer"
    )


def _dynamic_inference_set_group_name(self, group_name):
    """Switch to a driver-coordinated fresh group without touching old ranks."""
    self.group_name = str(group_name)
    self.rank = None
    self.world_size = None


def _patched_nccl_init_process_group(self, rank, world_size, master_metadata):
    """Let a fresh ZeroMQ SUB subscription reach the metadata publisher."""
    result = _ORIG_NCCL_INIT_PROCESS_GROUP(self, rank, world_size, master_metadata)
    if rank >= 0:
        # NCCLCheckpointEngine publishes bucket metadata immediately after
        # init. ZeroMQ PUB/SUB silently drops frames until a new SUB
        # subscription has propagated; dynamic topologies recreate that
        # connection on every sync and otherwise lose the first frame.
        time.sleep(float(os.environ.get("UNI_AGENT_NCCL_ZMQ_SETTLE_S", "1.0")))
    return result


def _patched_checkpoint_build_process_group(self, rollout):
    """Give every elastic NCCL sync a fresh group shared by all participants."""
    if getattr(self, "backend", None) != "nccl":
        return _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP(self, rollout)

    import ray

    try:
        backend_cfg = self.config.engine_kwargs.get("nccl", {})
        base_name = backend_cfg.get("group_name", "default")
    except Exception:
        base_name = "default"
    group_name = f"{base_name}__dynamic_epoch_{next(_CHECKPOINT_GROUP_EPOCH)}"
    actor_wg = self.actor_wg
    refs = actor_wg.execute_checkpoint_engine(
        ["dynamic_inference_set_group_name"] * actor_wg.world_size,
        group_name=[group_name] * actor_wg.world_size,
    )
    refs += rollout.execute_checkpoint_engine(
        ["dynamic_inference_set_group_name"] * rollout.world_size,
        group_name=[group_name] * rollout.world_size,
    )
    ray.get(refs)
    logger.info(
        "dynamic_inference.patch: checkpoint sync uses fresh group %s (%d rollout workers)",
        group_name, rollout.world_size,
    )
    return _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP(self, rollout)


def _patch_nccl_checkpoint_in_module(module) -> None:
    """Install topology-aware NCCL checkpoint group reuse (idempotent)."""
    global _ORIG_NCCL_INIT_PROCESS_GROUP, _PATCHED_NCCL_CLASS, _NCCL_PATCHED
    if _NCCL_PATCHED:
        return
    engine_cls = getattr(module, "NCCLCheckpointEngine", None)
    if engine_cls is None:
        return
    _ORIG_NCCL_INIT_PROCESS_GROUP = engine_cls.init_process_group
    _PATCHED_NCCL_CLASS = engine_cls
    engine_cls.dynamic_inference_set_group_name = _dynamic_inference_set_group_name
    engine_cls.init_process_group = _patched_nccl_init_process_group
    _NCCL_PATCHED = True
    logger.info("dynamic_inference.patch: topology-aware NCCL checkpoint groups enabled")


def apply_worker_patch() -> None:
    """Install the worker-side watcher (idempotent).

    Runs as Ray's ``worker_process_setup_hook`` in every worker process. It
    must NOT import verl directly (the hook runs before Ray configures the
    per-worker GPU environment; importing vLLM/verl here would initialise CUDA
    with the full device list and break FSDP) — a daemon thread polls
    ``sys.modules`` and patches ``vLLMHttpServer`` the moment worker code
    finishes importing it.

    The hook configured in the yaml points here, so this function first
    invokes the chained hooks (the example patch's worker patch) to keep their
    behaviour alive.
    """
    global _WORKER_PATCHED, _WORKER_PATCH_GENERATION
    if _WORKER_PATCHED:
        return

    chain = os.environ.get("UNI_AGENT_WORKER_HOOK_CHAIN")
    hook_fqns = [f.strip() for f in chain.split(",") if f.strip()] if chain is not None else _WORKER_HOOK_CHAIN
    for fqn in hook_fqns:
        try:
            import importlib

            module_name, _, attr = fqn.rpartition(".")
            getattr(importlib.import_module(module_name), attr)()
        except Exception as exc:
            logger.warning("dynamic_inference.patch: chained worker hook %s failed: %s", fqn, exc)

    import sys
    import threading
    import time

    _WORKER_PATCH_GENERATION += 1
    generation = _WORKER_PATCH_GENERATION

    targets = {
        "verl.workers.rollout.vllm_rollout.vllm_async_server": (
            "vLLMHttpServer", _patch_server_sleep_wake_in_module,
        ),
        "verl.checkpoint_engine.nccl_checkpoint_engine": (
            "NCCLCheckpointEngine", _patch_nccl_checkpoint_in_module,
        ),
    }

    def _wait_and_patch(target, class_name, installer):
        while target not in sys.modules:
            if generation != _WORKER_PATCH_GENERATION:
                return
            time.sleep(0.05)
        # "in sys.modules" fires while exec_module is still running (before the
        # class statement completes). Wait for the class attribute so we patch
        # the fully-defined class, not a half-loaded module.
        module = sys.modules[target]
        while not hasattr(module, class_name):
            if generation != _WORKER_PATCH_GENERATION:
                return
            time.sleep(0.05)
        try:
            installer(module)
        except Exception:
            logger.exception("dynamic_inference.patch: worker patch failed for %s", target)

    for target, (class_name, installer) in targets.items():
        threading.Thread(
            target=_wait_and_patch,
            args=(target, class_name, installer),
            daemon=True,
            name=f"di-{class_name}-patch",
        ).start()
    _WORKER_PATCHED = True


# --------------------------------------------------------------------------- #
# Patch #2: init_standalone external pool (driver side)
# --------------------------------------------------------------------------- #


def _policy_label_from_rollout_config(config) -> str | None:
    """Read ``config.custom.policy_name`` (same source of truth as verl_patch)."""
    try:
        custom = getattr(config, "custom", None) or {}
        getter = getattr(custom, "get", None)
        if callable(getter):
            return getter("policy_name")
    except Exception:
        pass
    return None


def _dynamic_max_colocate_count(config) -> int | None:
    """Read the graph-derived home/guest slot count from rollout.custom."""
    try:
        custom = getattr(config, "custom", None) or {}
        getter = getattr(custom, "get", None)
        value = getter("dynamic_inference_max_colocate_count") if callable(getter) else None
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


async def _patched_init_standalone(self):
    """RolloutReplica.init_standalone with a guest external-pool path.

    - ``_guest_external_pool`` set  -> guest path: reuse a view over the home
      unit's existing PGs (no new placement group), build the CheckpointEngineWorker
      group on it, launch servers. Guest actor names stay policy-unique: the
      donor label is injected into ``name_suffix`` exactly like the example
      patch does, and the guest construction adds its own
      ``guest_{home}_{rank}`` fragment.
    - graph-derived slot count set -> home path: create the normal standalone
      pool with that ``max_colocate_count`` instead of verl's hard-coded 2;
    - otherwise -> delegate to whatever ``init_standalone`` was when
      ``apply_patch()`` ran (the example patch's policy-suffixed wrapper if the
      example patch is active, else verl's original).
    """
    pool = getattr(self, "_guest_external_pool", None)
    if pool is not None:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.rollout.replica import RolloutMode
        from verl.utils.device import get_device_name

        self.rollout_mode = RolloutMode.STANDALONE
        self.resource_pool = pool
        # policy-unique actor names, mirroring verl_patch._policy_suffixed_replica
        label = _policy_label_from_rollout_config(getattr(self, "config", None))
        if label and not self.name_suffix.startswith(f"_{label}"):
            self.name_suffix = f"_{label}" + self.name_suffix
        worker_group = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_guest_{self.replica_rank}{self.name_suffix}",
            master_port_range=getattr(self, "_guest_master_port_range", None),
            use_gpu=True,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        # ``name_suffix`` is required on the CheckpointEngineWorker actor name
        # (multiple sleeping guests may share a replica rank namespace), but
        # verl's ServerAdapter lookup never appends that suffix to the vLLM
        # server name.  The replica rank itself is globally unique for guests,
        # so launch the server under the exact name the adapter will resolve.
        saved_suffix = self.name_suffix
        try:
            self.name_suffix = ""
            await self.launch_servers()
        finally:
            self.name_suffix = saved_suffix
        return

    max_colocate_count = _dynamic_max_colocate_count(getattr(self, "config", None))
    if max_colocate_count is not None:
        if max_colocate_count < 1:
            raise ValueError(
                "dynamic_inference_max_colocate_count must be positive, got "
                f"{max_colocate_count}"
            )
        from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
        from verl.workers.rollout.replica import RolloutMode
        from verl.utils.device import get_device_name

        self.rollout_mode = RolloutMode.STANDALONE
        label = _policy_label_from_rollout_config(getattr(self, "config", None))
        if label and not self.name_suffix.startswith(f"_{label}"):
            self.name_suffix = f"_{label}" + self.name_suffix

        if self.is_reward_model:
            resource_pool_name = f"rollout_pool_reward_{self.replica_rank}{self.name_suffix}"
            worker_prefix = f"rollout_reward_standalone_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            resource_pool_name = f"rollout_pool_teacher_{self.replica_rank}{self.name_suffix}"
            worker_prefix = f"rollout_teacher_standalone_{self.replica_rank}{self.name_suffix}"
        else:
            resource_pool_name = f"rollout_pool_{self.replica_rank}{self.name_suffix}"
            worker_prefix = f"rollout_standalone_{self.replica_rank}{self.name_suffix}"

        # Construct the same pool as verl's init_standalone, but with enough
        # fractional worker reservations for every configured outgoing edge.
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={
                resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
            },
            mapping=None,
            max_colocate_count=max_colocate_count,
        )
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=worker_prefix,
            use_gpu=True,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()
        return

    return await _ORIG_INIT_STANDALONE(self)


# --------------------------------------------------------------------------- #
# apply / restore
# --------------------------------------------------------------------------- #


def apply_patch() -> None:
    """Install the driver-side patch and arm the worker watcher (idempotent).

    Must run after the example patch's ``apply_patch()`` when that patch is
    active, so delegation in ``_patched_init_standalone`` chains onto the
    policy-suffixed wrapper.
    """
    global _PATCHED, _ORIG_INIT_STANDALONE
    global _CHECKPOINT_MANAGER_PATCHED, _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP
    if _PATCHED:
        return

    from verl.workers.rollout.replica import RolloutReplica

    _ORIG_INIT_STANDALONE = RolloutReplica.init_standalone
    RolloutReplica.init_standalone = _patched_init_standalone

    from verl.checkpoint_engine.base import CheckpointEngineManager

    _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP = CheckpointEngineManager.build_process_group
    CheckpointEngineManager.build_process_group = _patched_checkpoint_build_process_group
    _CHECKPOINT_MANAGER_PATCHED = True

    # Arm the sleep/wake watcher here as well (harmless in the driver; useful
    # if the driver process itself ever runs server code, and it matches the
    # example patch's apply/apply_worker pairing).
    apply_worker_patch()
    _PATCHED = True
    logger.info("dynamic_inference.patch: guest external-pool init enabled")


def restore() -> None:
    """Restore verl's original methods (idempotent)."""
    global _PATCHED, _ORIG_INIT_STANDALONE, _SERVER_PATCHED, _ORIG_SERVER_SLEEP, _ORIG_SERVER_WAKE_UP
    global _ORIG_SERVER_IS_SLEEPING, _PATCHED_SERVER_CLASS
    global _NCCL_PATCHED, _ORIG_NCCL_INIT_PROCESS_GROUP, _PATCHED_NCCL_CLASS
    global _CHECKPOINT_MANAGER_PATCHED, _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP
    global _WORKER_PATCHED, _WORKER_PATCH_GENERATION
    _WORKER_PATCH_GENERATION += 1
    _WORKER_PATCHED = False
    if _PATCHED:
        from verl.workers.rollout.replica import RolloutReplica

        RolloutReplica.init_standalone = _ORIG_INIT_STANDALONE
        _ORIG_INIT_STANDALONE = None
        _PATCHED = False
    if _CHECKPOINT_MANAGER_PATCHED:
        try:
            from verl.checkpoint_engine.base import CheckpointEngineManager

            CheckpointEngineManager.build_process_group = _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP
        except Exception:
            pass
        _ORIG_CHECKPOINT_BUILD_PROCESS_GROUP = None
        _CHECKPOINT_MANAGER_PATCHED = False
    if _SERVER_PATCHED:
        # Only meaningful in the process that patched it (worker processes
        # never call restore; the driver's in-process wrap is unwound here).
        try:
            server_cls = _PATCHED_SERVER_CLASS
            server_cls.sleep = _ORIG_SERVER_SLEEP
            server_cls.wake_up = _ORIG_SERVER_WAKE_UP
            if _ORIG_SERVER_IS_SLEEPING is None:
                delattr(server_cls, "dynamic_inference_is_sleeping")
            else:
                server_cls.dynamic_inference_is_sleeping = _ORIG_SERVER_IS_SLEEPING
        except Exception:
            pass
        _ORIG_SERVER_SLEEP = None
        _ORIG_SERVER_WAKE_UP = None
        _ORIG_SERVER_IS_SLEEPING = None
        _PATCHED_SERVER_CLASS = None
        _SERVER_PATCHED = False
    if _NCCL_PATCHED:
        try:
            _PATCHED_NCCL_CLASS.init_process_group = _ORIG_NCCL_INIT_PROCESS_GROUP
            delattr(_PATCHED_NCCL_CLASS, "dynamic_inference_set_group_name")
        except Exception:
            pass
        _ORIG_NCCL_INIT_PROCESS_GROUP = None
        _PATCHED_NCCL_CLASS = None
        _NCCL_PATCHED = False
        _SERVER_PATCHED = False
