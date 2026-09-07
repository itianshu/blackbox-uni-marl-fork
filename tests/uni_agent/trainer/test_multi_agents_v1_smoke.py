"""Smoke tests against the verl v1 PPOTrainer contract.

The test loads the real v1 ``PPOTrainer`` base/registry source from the sibling
``verl`` checkout when available, but replaces Ray/GPU-facing dependencies with
small local stubs. This keeps the check fast while still exercising the
registry-based initialization path used by ``MultiAgentsPPOTrainer``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack


class ConfigNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class SmokeKVBatchMeta:
    def __init__(self, *, partition_id="train", keys=None, tags=None, fields=None, extra_info=None):
        self.partition_id = partition_id
        self.keys = list(keys or [])
        self.tags = [dict(tag) for tag in (tags or [])]
        self.fields = fields
        self.extra_info = dict(extra_info or {})

    def __len__(self):
        return len(self.keys)


class SmokeReplayBuffer:
    def __init__(self, policy_name):
        self.policy_name = policy_name


class SmokeDataloader:
    def __init__(self, batches):
        self.batches = list(batches)

    def __iter__(self):
        return iter([dict(batch) for batch in self.batches])


class SmokeLLMClient:
    def __init__(self, policy_name):
        self.policy_name = policy_name
        self.calls = []

    async def generate(self, request_id, **kwargs):
        self.calls.append({"request_id": request_id, **kwargs})
        return f"{self.policy_name}:generated"


class RecordingAgentLoopManager:
    create_calls = []

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(kwargs)
        return SimpleNamespace(kind="agent_loop_manager", kwargs=kwargs)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _v1_trainer_base_path() -> Path:
    path = _workspace_root() / "verl" / "verl" / "trainer" / "ppo" / "v1" / "trainer_base.py"
    if not path.exists():
        pytest.skip("sibling verl checkout with v1 PPOTrainer source is not available")
    return path


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _install_parent_packages(monkeypatch, *names: str) -> None:
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)


def _install_verl_v1_import_stubs(monkeypatch) -> None:
    _install_parent_packages(
        monkeypatch,
        "verl",
        "verl.experimental",
        "verl.single_controller",
        "verl.trainer",
        "verl.trainer.ppo",
        "verl.trainer.ppo.v1",
        "verl.utils",
        "verl.utils.checkpoint",
        "verl.utils.dataset",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.utils",
        "torchdata",
    )

    ray_mod = _module(
        "ray",
        actor=SimpleNamespace(ActorHandle=object),
        get=lambda value: value,
        remote=lambda cls: cls,
        util=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"),
    )
    monkeypatch.setitem(sys.modules, "ray", ray_mod)

    transfer_queue_mod = _module(
        "transfer_queue",
        KVBatchMeta=SmokeKVBatchMeta,
        init=lambda config=None: None,
        close=lambda: None,
        kv_batch_put=lambda **kwargs: SmokeKVBatchMeta(
            keys=kwargs.get("keys"), tags=kwargs.get("tags"), fields=kwargs.get("fields")
        ),
        kv_batch_get=lambda **kwargs: TensorDict({}, batch_size=0),
        kv_clear=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "transfer_queue", transfer_queue_mod)

    class OmegaConf:
        @staticmethod
        def select(config, key, default=None):
            value = config
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, default)
                else:
                    value = getattr(value, part, default)
                if value is default:
                    return default
            return value

        @staticmethod
        def to_container(config, resolve=False):
            return config

    @contextlib.contextmanager
    def open_dict(config):
        yield config

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        _module("omegaconf", DictConfig=ConfigNode, OmegaConf=OmegaConf, open_dict=open_dict),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchdata.stateful_dataloader",
        _module("torchdata.stateful_dataloader", StatefulDataLoader=SmokeDataloader),
    )

    class RolloutMoELoadBalanceMetricsAccumulator:
        def __init__(self, model_config=None):
            self.model_config = model_config

    metric_utils = _module(
        "verl.trainer.ppo.metric_utils",
        RolloutMoELoadBalanceMetricsAccumulator=RolloutMoELoadBalanceMetricsAccumulator,
        compute_data_metrics=lambda *args, **kwargs: {},
        compute_moe_lb_metrics=lambda *args, **kwargs: {},
        compute_throughout_metrics=lambda *args, **kwargs: {},
        compute_timing_metrics=lambda *args, **kwargs: {},
        compute_variance_proxy_metrics=lambda *args, **kwargs: {},
        get_metric_data_with_optional_routed_experts=lambda *args, **kwargs: TensorDict({}, batch_size=0),
        process_validation_metrics=lambda *args, **kwargs: {},
    )
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.metric_utils", metric_utils)

    class SkipManager:
        @staticmethod
        def annotate_tq(role=None, phase=None):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def init(config):
            return None

        @staticmethod
        def set_step(step):
            return None

    def get_tensordict(batch_dict):
        source = {}
        batch_size = None
        for key, value in batch_dict.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list):
                source[key] = NonTensorStack.from_list([NonTensorData(item) for item in value])
                batch_size = len(value) if batch_size is None else batch_size
            else:
                source[key] = value
        return TensorDict(source=source, batch_size=[] if batch_size is None else [batch_size])

    def assign_non_tensor_data(tensor_dict, key, value):
        tensor_dict[key] = NonTensorData(value)

    def td_get(tensor_dict, key):
        if key not in tensor_dict:
            return None
        value = tensor_dict.get(key)
        if isinstance(value, NonTensorData):
            return value.data
        if isinstance(value, NonTensorStack):
            return [item.data if isinstance(item, NonTensorData) else item for item in value]
        return value

    stubs = {
        "verl.checkpoint_engine": _module("verl.checkpoint_engine", CheckpointEngineManager=object),
        "verl.experimental.agent_loop": _module("verl.experimental.agent_loop", AgentLoopManager=object),
        "verl.experimental.reward_loop": _module("verl.experimental.reward_loop", RewardLoopManager=object),
        "verl.experimental.teacher_loop": _module("verl.experimental.teacher_loop", MultiTeacherModelManager=object),
        "verl.protocol": _module("verl.protocol", DataProto=object, DataProtoFuture=object),
        "verl.single_controller.ray": _module(
            "verl.single_controller.ray",
            RayClassWithInitArgs=object,
            RayWorkerGroup=object,
            ResourcePoolManager=object,
            create_colocated_worker_cls=lambda class_dict: class_dict,
        ),
        "verl.trainer.distillation": _module(
            "verl.trainer.distillation",
            is_distillation_enabled=lambda config=None: False,
        ),
        "verl.trainer.ppo.core_algos": _module(
            "verl.trainer.ppo.core_algos",
            agg_loss=lambda *args, **kwargs: SimpleNamespace(detach=lambda: SimpleNamespace(item=lambda: 0.0)),
            get_kl_controller=lambda *args, **kwargs: None,
        ),
        "verl.trainer.ppo.padding_utils": _module(
            "verl.trainer.ppo.padding_utils",
            upsample_batch_to_divisible_size=lambda batch, *args, **kwargs: batch,
        ),
        "verl.trainer.ppo.ray_trainer": _module(
            "verl.trainer.ppo.ray_trainer",
            apply_kl_penalty=lambda data, *args, **kwargs: (data, {}),
            compute_spec_decode_metrics=lambda *args, **kwargs: {},
        ),
        "verl.trainer.ppo.rollout_corr_helper": _module(
            "verl.trainer.ppo.rollout_corr_helper",
            compute_rollout_correction_and_add_to_batch=lambda data, *args, **kwargs: (data, {}),
        ),
        "verl.trainer.ppo.utils": _module(
            "verl.trainer.ppo.utils",
            Role=SimpleNamespace(
                ActorRolloutRef="ActorRolloutRef",
                ActorRollout="ActorRollout",
                Critic="Critic",
                RewardModel="RewardModel",
                TeacherModel="TeacherModel",
            ),
            create_rl_dataset=lambda *args, **kwargs: None,
            create_rl_sampler=lambda *args, **kwargs: None,
            need_critic=lambda config: False,
            need_reference_policy=lambda config: False,
            need_teacher_policy=lambda config: False,
        ),
        "verl.trainer.ppo.v1.replay_buffer": _module(
            "verl.trainer.ppo.v1.replay_buffer",
            DAPO_FILTERED_REWARD_COUNTS_KEY="_dapo_filtered_reward_counts",
            ReplayBuffer=SmokeReplayBuffer,
            ReplayBufferAsync=SmokeReplayBuffer,
        ),
        "verl.trainer.ppo.v1.utils": _module(
            "verl.trainer.ppo.v1.utils",
            MetricsAggregator=object,
            compute_advantage_for_multi_trajectories=lambda data, *args, **kwargs: data,
        ),
        "verl.utils.tensordict_utils": _module(
            "verl.utils.tensordict_utils",
            get_tensordict=get_tensordict,
            assign_non_tensor_data=assign_non_tensor_data,
            get=td_get,
        ),
        "verl.utils.checkpoint.checkpoint_manager": _module(
            "verl.utils.checkpoint.checkpoint_manager",
            find_latest_ckpt_path=lambda *args, **kwargs: None,
        ),
        "verl.utils.config": _module("verl.utils.config", omega_conf_to_dataclass=lambda config: config),
        "verl.utils.dataset.rl_dataset": _module("verl.utils.dataset.rl_dataset", collate_fn=lambda batch: batch),
        "verl.utils.debug": _module("verl.utils.debug", marked_timer=lambda *args, **kwargs: contextlib.nullcontext()),
        "verl.utils.debug.metrics": _module("verl.utils.debug.metrics", calculate_debug_metrics=lambda *args, **kwargs: {}),
        "verl.utils.fs": _module("verl.utils.fs", copy_to_local=lambda path, *args, **kwargs: path),
        "verl.utils.import_utils": _module("verl.utils.import_utils", load_extern_type=lambda path, name: None),
        "verl.utils.metric": _module("verl.utils.metric", reduce_metrics=lambda metrics: metrics),
        "verl.utils.py_functional": _module("verl.utils.py_functional", rename_dict=lambda data, prefix: data),
        "verl.utils.seqlen_balancing": _module(
            "verl.utils.seqlen_balancing",
            calculate_workload=lambda values: values,
            get_seqlen_balanced_partitions=lambda values, k_partitions, equal_size=True: [list(range(len(values)))],
            log_seqlen_unbalance=lambda *args, **kwargs: {},
        ),
        "verl.utils.skip": _module("verl.utils.skip", SkipManager=SkipManager),
        "verl.utils.tracking": _module(
            "verl.utils.tracking",
            DapoFilteredRewardTableLogger=object,
            Tracking=object,
            ValidationGenerationsLogger=object,
        ),
        "verl.workers.config": _module(
            "verl.workers.config",
            CriticConfig=object,
            DistillationConfig=object,
            HFModelConfig=object,
        ),
        "verl.workers.engine_workers": _module(
            "verl.workers.engine_workers",
            ActorRolloutRefWorker=object,
            TrainingWorker=object,
            TrainingWorkerConfig=object,
        ),
        "verl.workers.rollout.llm_server": _module(
            "verl.workers.rollout.llm_server",
            LLMServerClient=object,
            LLMServerManager=object,
        ),
        "verl.workers.utils.losses": _module("verl.workers.utils.losses", value_loss=lambda *args, **kwargs: None),
        "verl.workers.utils.padding": _module(
            "verl.workers.utils.padding",
            response_from_nested=lambda data, mask: data,
            response_to_nested=lambda data, mask: data,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    verl_utils = sys.modules["verl.utils"]
    verl_utils.hf_processor = lambda *args, **kwargs: None
    verl_utils.hf_tokenizer = lambda *args, **kwargs: None
    verl_utils.tensordict_utils = sys.modules["verl.utils.tensordict_utils"]


def _load_v1_trainer_base(monkeypatch):
    _install_verl_v1_import_stubs(monkeypatch)
    path = _v1_trainer_base_path()
    module_name = "verl.trainer.ppo.v1.trainer_base"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    v1_package = sys.modules["verl.trainer.ppo.v1"]
    v1_package.PPOTrainer = module.PPOTrainer
    v1_package.register_trainer = module.register_trainer
    v1_package.get_trainer_cls = module.get_trainer_cls
    return module


def _v1_policy_config(policy_name: str) -> ConfigNode:
    return ConfigNode(
        policy_name=policy_name,
        algorithm=ConfigNode(
            use_kl_in_reward=False,
            kl_ctrl=ConfigNode(),
        ),
        trainer=ConfigNode(
            v1=ConfigNode(
                trainer_mode="sync",
                sync=ConfigNode(parameter_sync_step=1),
            )
        ),
        actor_rollout_ref=ConfigNode(
            model=ConfigNode(),
            rollout=ConfigNode(temperature=0.7),
        ),
    )


def test_multi_agents_trainer_initializes_v1_policy_runtimes(monkeypatch):
    trainer_base = _load_v1_trainer_base(monkeypatch)
    RecordingAgentLoopManager.create_calls.clear()

    class SmokeV1PPOTrainer(trainer_base.PPOTrainer):
        instances = []

        def __init__(self, config):
            self.setup_calls = 0
            self.on_init_end_calls = 0
            self.fit_calls = 0
            super().__init__(config)
            self.policy_name = config.policy_name
            self.global_steps = 0
            self.__class__.instances.append(self)

        def _build_replay_buffer(self):
            return SmokeReplayBuffer(self.config.policy_name)

        def _setup(self):
            self.setup_calls += 1
            self.tokenizer = f"tokenizer:{self.policy_name}"
            self.processor = f"processor:{self.policy_name}"
            self.llm_client = SmokeLLMClient(self.policy_name)
            self.llm_server_manager = SimpleNamespace(get_client=lambda: self.llm_client)
            self.reward_loop_manager = SimpleNamespace(
                reward_loop_worker_handles=[f"reward:{self.policy_name}"]
            )
            self.train_dataset = f"train_dataset:{self.policy_name}"
            self.val_dataset = f"val_dataset:{self.policy_name}"
            self.train_dataloader = SmokeDataloader(
                [
                    {
                        "raw_prompt": [[{"role": "user", "content": f"prompt:{self.policy_name}"}]],
                        "data_source": [f"source:{self.policy_name}"],
                    }
                ]
            )
            self.val_dataloader = SmokeDataloader([])

        def init_runtime(self):
            self._setup()

        def on_init_end(self):
            self.on_init_end_calls += 1

        def on_step_end(self):
            return None

        def on_sample_end(self):
            return None

        def fit(self, *args, **kwargs):
            self.fit_calls += 1
            raise AssertionError("MultiAgentsPPOTrainer must not call per-policy PPOTrainer.fit()")

    single_trainer_module = _module(
        "uni_agent.trainer.single_ppo_trainer",
        SinglePPOTrainer=SmokeV1PPOTrainer,
    )
    monkeypatch.setitem(sys.modules, "uni_agent.trainer.single_ppo_trainer", single_trainer_module)

    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    class TestableMultiAgentsPPOTrainer(MultiAgentsPPOTrainer):
        def _compose_policy_ppo_config(self, *, policy_name, config_name, policy_entry):
            from omegaconf import OmegaConf

            return OmegaConf.create(_v1_policy_config(policy_name))

        def _build_dataloader(self):
            source = self.policy_trainers["policy_1"]
            self.train_dataset = source.train_dataset
            self.val_dataset = source.val_dataset
            self.train_dataloader = source.train_dataloader
            self.val_dataloader = source.val_dataloader

        def _build_replay_buffer(self):
            self.replay_buffer = SmokeReplayBuffer("outer")

        def _load_checkpoint(self):
            self.global_steps = 0

    config = ConfigNode(
        policies=ConfigNode(
            policy_1=ConfigNode(ppo_trainer_config_name="ppo_trainer"),
            policy_2=ConfigNode(ppo_trainer_config_name="ppo_trainer"),
        ),
        actor_rollout_ref=ConfigNode(
            rollout=ConfigNode(
                temperature=0.6,
                custom=ConfigNode(
                    agent_framework=ConfigNode(
                        role_policy_mapping=ConfigNode(
                            agent_1="policy_1",
                            agent_2="policy_1",
                            agent_3="policy_2",
                        )
                    )
                ),
            )
        ),
        trainer=ConfigNode(
            total_training_steps=0,
            v1=ConfigNode(
                trainer_mode="sync",
                sync=ConfigNode(parameter_sync_step=1),
            ),
        ),
        data=ConfigNode(train_batch_size=1),
    )

    trainer = TestableMultiAgentsPPOTrainer(config=config)

    assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
    assert all(isinstance(policy_trainer, trainer_base.PPOTrainer) for policy_trainer in trainer.policy_trainers.values())

    trainer.init()
    assert [(item.policy_name, item.setup_calls, item.on_init_end_calls) for item in SmokeV1PPOTrainer.instances] == [
        ("policy_1", 1, 1),
        ("policy_2", 1, 1),
    ]
    assert [item.fit_calls for item in SmokeV1PPOTrainer.instances] == [0, 0]
    assert trainer.train_dataloader is trainer.policy_trainers["policy_1"].train_dataloader

    agent_loop_manager = RecordingAgentLoopManager.create(
        config=trainer.config,
        llm_client=trainer.get_multi_policy_llm_client(),
        reward_loop_worker_handles=trainer.get_reward_handles(),
        gateway_actor_kwargs=trainer.get_gateway_actor_kwargs(),
    )
    assert agent_loop_manager.kind == "agent_loop_manager"
    create_kwargs = RecordingAgentLoopManager.create_calls[0]
    assert create_kwargs["reward_loop_worker_handles"] == ["reward:policy_1", "reward:policy_2"]
    assert create_kwargs["gateway_actor_kwargs"] == {
        "tokenizer": "tokenizer:policy_1",
        "processor": "processor:policy_1",
        "policy_tokenizers": {
            "policy_1": "tokenizer:policy_1",
            "policy_2": "tokenizer:policy_2",
        },
        "policy_processors": {
            "policy_1": "processor:policy_1",
            "policy_2": "processor:policy_2",
        },
    }
    assert set(create_kwargs["llm_client"].policy_clients) == {"policy_1", "policy_2"}


def _ensure_flashinfer_workspace_writable():
    """Real verl imports flashinfer, which needs a writable JIT workspace dir.

    Only consulted before the first ``import verl`` in this process; if the
    configured ``FLASHINFER_WORKSPACE_BASE`` is missing/read-only, point it at
    a fresh temporary directory.
    """
    import os
    import tempfile

    base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if base and os.path.isdir(base) and os.access(base, os.W_OK):
        return
    os.environ["FLASHINFER_WORKSPACE_BASE"] = tempfile.mkdtemp(prefix="flashinfer_ws_")


def test_multi_agents_trainer_initializes_v1_policy_runtimes_real_verl(monkeypatch):
    """End-to-end smoke against the REAL verl v1 PPOTrainer.

    Unlike the stub-based variant above, this test imports the actual verl
    package (editable install pointing at the local checkout) and composes the
    real verl ``ppo_trainer`` Hydra config, then registers a smoke trainer via
    verl's registry and drives it through ``MultiAgentsPPOTrainer``.
    """
    _ensure_flashinfer_workspace_writable()

    try:
        from verl.trainer.ppo.v1 import trainer_base
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"real verl is not importable in this environment: {exc}")

    class SmokeV1PPOTrainer(trainer_base.PPOTrainer):
        instances = []

        def __init__(self, config):
            self.setup_calls = 0
            self.on_init_end_calls = 0
            self.fit_calls = 0
            super().__init__(config)
            self.policy_name = config.policy_name
            self.global_steps = 0
            self.__class__.instances.append(self)

        def _build_replay_buffer(self):
            return SmokeReplayBuffer(self.config.policy_name)

        def _setup(self):
            self.setup_calls += 1
            self.tokenizer = f"tokenizer:{self.policy_name}"
            self.processor = f"processor:{self.policy_name}"
            self.llm_client = SmokeLLMClient(self.policy_name)
            self.llm_server_manager = SimpleNamespace(get_client=lambda: self.llm_client)
            self.reward_loop_manager = SimpleNamespace(
                reward_loop_worker_handles=[f"reward:{self.policy_name}"]
            )
            self.train_dataset = f"train_dataset:{self.policy_name}"
            self.val_dataset = f"val_dataset:{self.policy_name}"
            self.train_dataloader = SmokeDataloader(
                [
                    {
                        "raw_prompt": [[{"role": "user", "content": f"prompt:{self.policy_name}"}]],
                        "data_source": [f"source:{self.policy_name}"],
                    }
                ]
            )
            self.val_dataloader = SmokeDataloader([])

        def init_runtime(self):
            self._setup()

        def on_init_end(self):
            self.on_init_end_calls += 1

        def on_step_end(self):
            return None

        def on_sample_end(self):
            return None

        def fit(self, *args, **kwargs):
            self.fit_calls += 1
            raise AssertionError("MultiAgentsPPOTrainer must not call per-policy PPOTrainer.fit()")

    SmokeV1PPOTrainer.instances.clear()

    # Compose the REAL verl ppo_trainer Hydra config, then snapshot it per policy.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    verl_config_dir = (_workspace_root() / "verl" / "verl" / "trainer" / "config").as_posix()
    with initialize_config_dir(config_dir=verl_config_dir, version_base=None):
        base_cfg = compose(
            config_name="ppo_trainer",
            overrides=["trainer.v1.trainer_mode=sync"],
        )
    base_policy = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    def _policy_config(policy_name):
        cfg = OmegaConf.create(OmegaConf.to_container(base_policy, resolve=True))
        cfg.policy_name = policy_name
        return cfg

    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    import uni_agent.trainer.single_ppo_trainer as single_ppo_module

    monkeypatch.setattr(single_ppo_module, "SinglePPOTrainer", SmokeV1PPOTrainer)

    class TestableMultiAgentsPPOTrainer(MultiAgentsPPOTrainer):
        def _compose_policy_ppo_config(self, *, policy_name, config_name, policy_entry):
            return _policy_config(policy_name)

        def _build_dataloader(self):
            source = self.policy_trainers["policy_1"]
            self.train_dataset = source.train_dataset
            self.val_dataset = source.val_dataset
            self.train_dataloader = source.train_dataloader
            self.val_dataloader = source.val_dataloader

        def _build_replay_buffer(self):
            self.replay_buffer = SmokeReplayBuffer("outer")

        def _load_checkpoint(self):
            self.global_steps = 0

    config = OmegaConf.create(
        {
            "policies": {
                "policy_1": {"ppo_trainer_config_name": "ppo_trainer"},
                "policy_2": {"ppo_trainer_config_name": "ppo_trainer"},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 0.6,
                    "custom": {
                        "agent_framework": {
                            "role_policy_mapping": {
                                "agent_1": "policy_1",
                                "agent_2": "policy_1",
                                "agent_3": "policy_2",
                            }
                        }
                    },
                }
            },
            "trainer": {
                "total_training_steps": 0,
                "save_freq": -1,
                "resume_mode": "disable",
                "v1": {
                    "trainer_mode": "sync",
                    "sync": {"parameter_sync_step": 1},
                },
            },
            "data": {"train_batch_size": 1},
        }
    )

    trainer = TestableMultiAgentsPPOTrainer(config=config)

    assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
    assert all(
        isinstance(policy_trainer, trainer_base.PPOTrainer)
        for policy_trainer in trainer.policy_trainers.values()
    )

    trainer.init()
    assert [
        (item.policy_name, item.setup_calls, item.on_init_end_calls)
        for item in SmokeV1PPOTrainer.instances
    ] == [
        ("policy_1", 1, 1),
        ("policy_2", 1, 1),
    ]
    assert [item.fit_calls for item in SmokeV1PPOTrainer.instances] == [0, 0]
