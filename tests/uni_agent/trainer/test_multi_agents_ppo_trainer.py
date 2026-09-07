import asyncio
import inspect
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack


def _install_dependency_stubs():
    if "ray" not in sys.modules:
        ray_stub = types.ModuleType("ray")
        ray_stub.actor = SimpleNamespace(ActorHandle=object)
        ray_stub.remote = lambda cls: cls
        ray_stub.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        sys.modules["ray"] = ray_stub
    else:
        ray_stub = sys.modules["ray"]
        if not hasattr(ray_stub, "actor"):
            ray_stub.actor = SimpleNamespace(ActorHandle=object)
        if not hasattr(ray_stub, "remote"):
            ray_stub.remote = lambda cls: cls
        if not hasattr(ray_stub, "util"):
            ray_stub.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")

    if "verl.workers.rollout.llm_server" not in sys.modules:
        llm_server_mod = types.ModuleType("verl.workers.rollout.llm_server")
        llm_server_mod.LLMServerClient = object

        for name in [
            "verl",
            "verl.utils",
            "verl.workers",
            "verl.workers.rollout",
        ]:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["verl.workers.rollout.llm_server"] = llm_server_mod

    for name in [
        "verl",
        "verl.utils",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    tensordict_utils_mod = sys.modules.get("verl.utils.tensordict_utils")
    if tensordict_utils_mod is None:
        tensordict_utils_mod = types.ModuleType("verl.utils.tensordict_utils")
        sys.modules["verl.utils.tensordict_utils"] = tensordict_utils_mod

    def get_tensordict(batch_dict):
        source = {}
        batch_size = None
        for key, value in batch_dict.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list):
                source[key] = NonTensorStack.from_list([NonTensorData(item) for item in value])
                value_batch_size = len(value)
                batch_size = value_batch_size if batch_size is None else batch_size
            else:
                source[key] = value
        return TensorDict(source=source, batch_size=[] if batch_size is None else [batch_size])

    def assign_non_tensor_data(tensor_dict, key, value):
        tensor_dict[key] = NonTensorData(value)

    def get(tensor_dict, key):
        if key not in tensor_dict:
            return None
        value = tensor_dict.get(key)
        if isinstance(value, NonTensorStack):
            return value.tolist()
        if isinstance(value, NonTensorData):
            return value.data
        return value

    if not hasattr(tensordict_utils_mod, "get_tensordict"):
        tensordict_utils_mod.get_tensordict = get_tensordict
    if not hasattr(tensordict_utils_mod, "assign_non_tensor_data"):
        tensordict_utils_mod.assign_non_tensor_data = assign_non_tensor_data
    if not hasattr(tensordict_utils_mod, "get"):
        tensordict_utils_mod.get = get
    sys.modules["verl.utils"].tensordict_utils = tensordict_utils_mod

    if "transfer_queue" not in sys.modules:
        tq_stub = types.ModuleType("transfer_queue")
        sys.modules["transfer_queue"] = tq_stub
    tq_stub = sys.modules["transfer_queue"]
    for name in ("init", "close", "kv_batch_put", "kv_put", "kv_clear"):
        if not hasattr(tq_stub, name):
            setattr(tq_stub, name, lambda *args, **kwargs: None)

    if "verl.utils.debug" not in sys.modules:
        for name in [
            "verl",
            "verl.utils",
        ]:
            sys.modules.setdefault(name, types.ModuleType(name))

        debug_mod = types.ModuleType("verl.utils.debug")

        @contextmanager
        def marked_timer(*args, **kwargs):
            yield

        debug_mod.marked_timer = marked_timer
        sys.modules["verl.utils.debug"] = debug_mod
        sys.modules["verl.utils"].debug = debug_mod

    tracking_mod = sys.modules.setdefault("verl.utils.tracking", types.ModuleType("verl.utils.tracking"))
    if not hasattr(tracking_mod, "Tracking"):
        class Tracking:
            def __init__(self, *args, **kwargs):
                self.logged = []

            def log(self, data, step):
                self.logged.append((data, step))

            def finish(self, exit_code=0):
                return None

        tracking_mod.Tracking = Tracking
    sys.modules["verl.utils"].tracking = tracking_mod


class RecordingLLMClient:
    def __init__(self, policy_name):
        self.policy_name = policy_name
        self.calls = []

    async def generate(self, request_id, **kwargs):
        self.calls.append({"request_id": request_id, **kwargs})
        return f"{self.policy_name}:generated"


class SimpleKVBatchMeta:
    def __init__(self, *, partition_id="train", keys=None, tags=None, fields=None, extra_info=None):
        self.partition_id = partition_id
        self.keys = list(keys or [])
        self.tags = [dict(tag) for tag in (tags or [])]
        self.fields = fields
        self.extra_info = dict(extra_info or {})

    def __len__(self):
        return len(self.keys)


class FakeReplayBuffer:
    def __init__(self, policy_name):
        self.policy_name = policy_name
        self.sample_calls = []
        self.next_sample = {policy_name: "sample:0"}

    def sample(self, *, global_steps, partition_id, batch_size):
        self.sample_calls.append(
            {
                "global_steps": global_steps,
                "partition_id": partition_id,
                "batch_size": batch_size,
            }
        )
        if callable(self.next_sample):
            return self.next_sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)
        return self.next_sample


class FakeDataloader:
    def __init__(self, batches):
        self.batches = list(batches)
        self.iter_calls = 0
        self.loaded_states = []
        self.saved_states = []

    def __iter__(self):
        self.iter_calls += 1
        return iter([dict(batch) for batch in self.batches])

    def __len__(self):
        return len(self.batches)

    def state_dict(self):
        state = {"iter_calls": self.iter_calls, "batch_count": len(self.batches)}
        self.saved_states.append(state)
        return state

    def load_state_dict(self, state):
        self.loaded_states.append(state)


class FakeWorkerGroup:
    def __init__(self, name):
        self.name = name
        self.load_calls = []
        self.save_calls = []

    def load_checkpoint(self, **kwargs):
        self.load_calls.append(kwargs)

    def save_checkpoint(self, *args, **kwargs):
        self.save_calls.append({"args": args, "kwargs": kwargs})


class FakeCheckpointManager:
    def __init__(self, policy_name):
        self.policy_name = policy_name
        self.sleep_calls = 0
        self.update_weight_steps = []

    def sleep_replicas(self):
        self.sleep_calls += 1

    def update_weights(self, global_steps):
        self.update_weight_steps.append(global_steps)


class FakeV1PPOTrainer:
    instances = []

    def __init__(self, config):
        self.config = config
        self.policy_name = config.policy_name
        self.replay_buffer = FakeReplayBuffer(self.policy_name)
        self.llm_client = RecordingLLMClient(self.policy_name)
        self.reward_handles = [f"reward:{self.policy_name}"]
        self.tokenizer = f"tokenizer:{self.policy_name}"
        self.processor = f"processor:{self.policy_name}"
        self.init_calls = 0
        self.runtime_init_calls = 0
        self.on_init_end_calls = 0
        self.fit_calls = 0
        self.global_steps = 0
        self.use_reference_policy = getattr(config, "use_reference_policy", False)
        self.use_critic = getattr(config, "use_critic", False)
        self.stage_calls = []
        self.checkpoint_manager = FakeCheckpointManager(self.policy_name)
        self.actor_rollout_wg = FakeWorkerGroup(f"actor:{self.policy_name}")
        self.critic_wg = FakeWorkerGroup(f"critic:{self.policy_name}")
        FakeV1PPOTrainer.instances.append(self)

    def init(self):
        self.init_calls += 1

    def init_runtime(self):
        self.runtime_init_calls += 1

    def on_init_end(self):
        self.on_init_end_calls += 1

    def get_llm_client(self):
        return self.llm_client

    def get_reward_handles(self):
        return self.reward_handles

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("per-policy PPOTrainer.fit() must not be called")

    def _record_stage(self, stage, batch, metrics=None):
        self.stage_calls.append((stage, list(batch.keys)))
        if metrics is not None:
            metrics[f"{stage}/count"] = len(batch)
        return batch

    def _balance_batch(self, batch, metrics=None, logging_prefix=None):
        self.stage_calls.append(("balance", list(batch.keys), logging_prefix))
        if metrics is not None:
            metrics["balance/count"] = len(batch)
        return batch

    def _compute_old_log_prob(self, batch, metrics=None):
        return self._record_stage("old_log_prob", batch, metrics)

    def _compute_ref_log_prob(self, batch, metrics=None):
        return self._record_stage("ref_log_prob", batch, metrics)

    def _compute_values(self, batch, metrics=None):
        return self._record_stage("values", batch, metrics)

    def _compute_advantage(self, batch, metrics=None):
        return self._record_stage("advantage", batch, metrics)

    def _update_critic(self, batch, metrics=None):
        return self._record_stage("update_critic", batch, metrics)

    def _update_actor(self, batch, metrics=None):
        return self._record_stage("update_actor", batch, metrics)


class FakeSharedAdvantagePPOTrainer(FakeV1PPOTrainer):
    shared_records = {}

    def _record_stage(self, stage, batch, metrics=None):
        super()._record_stage(stage, batch, metrics)
        return batch

    def _compute_advantage(self, batch, metrics=None):
        self.stage_calls.append(("advantage", list(batch.keys)))
        rollout_scores = {}
        for key in batch.keys:
            record = self.shared_records[key]
            rollout_scores.setdefault(record["rollout_id"], record["reward"])

        rewards = list(rollout_scores.values())
        mean_reward = sum(rewards) / len(rewards)
        for key in batch.keys:
            record = self.shared_records[key]
            record["advantage"] = record["reward"] - mean_reward
        if metrics is not None:
            metrics["advantage/rollout_count"] = len(rollout_scores)
        return batch

    def _update_actor(self, batch, metrics=None):
        self.stage_calls.append(("update_actor", list(batch.keys)))
        self.updated_records = {
            key: {
                "policy_name": self.shared_records[key]["policy_name"],
                "role": self.shared_records[key]["role"],
                "rollout_id": self.shared_records[key]["rollout_id"],
                "advantage": self.shared_records[key]["advantage"],
            }
            for key in batch.keys
        }
        if metrics is not None:
            metrics["update_actor/count"] = len(batch)
        return batch


class FakeV1PPOTrainerWithDataloader(FakeV1PPOTrainer):
    def init_runtime(self):
        super().init_runtime()
        self.train_dataset = f"train_dataset:{self.policy_name}"
        self.val_dataset = f"val_dataset:{self.policy_name}"
        self.train_dataloader = FakeDataloader(
            [
                {
                    "raw_prompt": [[{"role": "user", "content": f"prompt:{self.policy_name}"}]],
                    "reward_model": [{"ground_truth": f"answer:{self.policy_name}"}],
                    "tools_kwargs": [{"env": {"image": f"image:{self.policy_name}"}}],
                    "data_source": [f"source:{self.policy_name}"],
                }
            ]
        )
        self.val_dataloader = FakeDataloader([])


class FakeV1PPOTrainerWithLegacyInitWorkers(FakeV1PPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.init_workers_calls = 0

    def init_workers(self):
        self.init_workers_calls += 1


class FakeAgentFrameworkRolloutAdapter:
    create_calls = []

    def __init__(self):
        self.generated_prompts = []

    @classmethod
    def create(cls, **kwargs):
        cls.create_calls.append(kwargs)
        instance = cls()
        instance.kind = "agent_loop_manager"
        instance.kwargs = kwargs
        return instance

    def generate_sequences(self, prompts):
        self.generated_prompts.append(prompts)


class FakeTransferQueue:
    def __init__(self):
        self.init_calls = []
        self.close_calls = 0
        self.batch_puts = []

    def init(self, config=None):
        self.init_calls.append(config)

    def close(self):
        self.close_calls += 1

    def kv_batch_put(self, *, keys, partition_id, tags):
        self.batch_puts.append(
            {
                "keys": list(keys),
                "partition_id": partition_id,
                "tags": [dict(tag) for tag in tags],
            }
        )

class RecordingMarkedTimer:
    def __init__(self):
        self.calls = []

    @contextmanager
    def __call__(self, name, timing_raw, *args, **kwargs):
        self.calls.append(
            {
                "name": name,
                "timing_raw": timing_raw,
                "args": args,
                "kwargs": dict(kwargs),
            }
        )
        yield


class RecordingMultiAgentsPPOTrainerMixin:
    def _step_once(self, metrics, timing_raw, sample_batch_size):
        self.step_events.append(("sample", self.global_steps))
        multi_agent_batch = {"step": self.global_steps}
        self.step_events.append(("build", multi_agent_batch["step"]))
        per_policy_batches = {"policy_1": f"batch:{multi_agent_batch['step']}"}
        self.step_events.append(("update", self.global_steps, dict(per_policy_batches)))
        metrics["policy_1/loss"] = float(self.global_steps)
        return SimpleKVBatchMeta(keys=[f"key:{self.global_steps}"], tags=[{"policy_name": "policy_1"}])


def _policy_config(policy_name, **kwargs):
    kwargs.setdefault(
        "trainer",
        _NS(
            v1=_NS(
                trainer_mode="sync",
                sync=_NS(parameter_sync_step=1),
                separate_async=_NS(parameter_sync_step=1),
            )
        ),
    )
    return _NS(policy_name=policy_name, **kwargs)


class _NS(SimpleNamespace):
    """SimpleNamespace with OmegaConf-like .get() for minimal test configs."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _ns_deep(value):
    if isinstance(value, SimpleNamespace):
        return _NS(**{k: _ns_deep(v) for k, v in vars(value).items()})
    if isinstance(value, dict):
        return {k: _ns_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_ns_deep(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _config_with_policies(config=None, policy_configs=None):
    if isinstance(config, DictConfig):
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    else:
        config = _ns_deep(config) if config is not None else _NS()
    # MultiAgentsPPOTrainer.__init__ reads these outer fields directly (mirrors
    # the guaranteed keys in the real multi_agent_blackbox.yaml).
    if getattr(config, "data", None) is None:
        data_config = OmegaConf.create({"train_batch_size": 4}) if isinstance(config, DictConfig) else _NS(train_batch_size=4)
        if isinstance(config, DictConfig):
            with open_dict(config):
                config.data = data_config
        else:
            config.data = data_config
    trainer = getattr(config, "trainer", None)
    if trainer is None:
        trainer_defaults = _NS(
            save_freq=-1,
            total_training_steps=2,
            resume_mode="disable",
            default_hdfs_dir=None,
            del_local_ckpt_after_load=False,
            project_name="test",
            experiment_name="test",
            logger=["console"],
            v1=_NS(
                trainer_mode="sync",
                sync=_NS(parameter_sync_step=1),
                separate_async=_NS(parameter_sync_step=1),
            ),
        )
        if isinstance(config, DictConfig):
            with open_dict(config):
                config.trainer = OmegaConf.create(_plain(trainer_defaults))
        else:
            config.trainer = trainer_defaults
    else:
        if not hasattr(trainer, "save_freq"):
            trainer.save_freq = -1
        if not hasattr(trainer, "total_training_steps"):
            trainer.total_training_steps = 2
        if not hasattr(trainer, "resume_mode"):
            trainer.resume_mode = "disable"
        if not hasattr(trainer, "default_hdfs_dir"):
            trainer.default_hdfs_dir = None
        if not hasattr(trainer, "del_local_ckpt_after_load"):
            trainer.del_local_ckpt_after_load = False
        if not hasattr(trainer, "project_name"):
            trainer.project_name = "test"
        if not hasattr(trainer, "experiment_name"):
            trainer.experiment_name = "test"
        if not hasattr(trainer, "logger"):
            trainer.logger = ["console"]
        if not hasattr(trainer, "v1"):
            trainer.v1 = _NS(
                trainer_mode="sync",
                sync=_NS(parameter_sync_step=1),
                separate_async=_NS(parameter_sync_step=1),
            )
    if policy_configs is not None:
        config.policies = {
            policy_name: _NS(
                ppo_trainer_config_name="test_policy_config",
                ppo_trainer_overrides={},
            )
            for policy_name, policy_config in policy_configs.items()
        }
    if isinstance(config, DictConfig):
        return config
    return OmegaConf.create(_plain(config))


def _install_policy_trainer_stubs(policy_trainer_cls):
    for name in [
        "verl",
        "verl.trainer",
        "verl.trainer.ppo",
    ]:
        module = sys.modules.setdefault(name, types.ModuleType(name))
        if not hasattr(module, "__path__"):
            module.__path__ = []

    v1_mod = sys.modules.setdefault("verl.trainer.ppo.v1", types.ModuleType("verl.trainer.ppo.v1"))
    sys.modules["verl.trainer"].ppo = sys.modules["verl.trainer.ppo"]
    sys.modules["verl.trainer.ppo"].v1 = v1_mod

    sync_mod = types.ModuleType("uni_agent.trainer.single_ppo_trainer")
    sync_mod.SinglePPOTrainer = policy_trainer_cls
    sys.modules["uni_agent.trainer.single_ppo_trainer"] = sync_mod

    async_mod = types.ModuleType("uni_agent.trainer.single_async_ppo_trainer")
    async_mod.SingleAsyncPPOTrainer = policy_trainer_cls
    sys.modules["uni_agent.trainer.single_async_ppo_trainer"] = async_mod


def _test_multi_agents_trainer_cls(
    base_cls,
    *,
    policy_trainer_cls=FakeV1PPOTrainer,
    agent_loop_manager_cls=FakeAgentFrameworkRolloutAdapter,
    policy_configs=None,
):
    def should_stub_component(name):
        component = base_cls.__dict__.get(name)
        return component is None or component.__module__ == "uni_agent.trainer.multi_agents_ppo_trainer"

    class TestableMultiAgentsPPOTrainer(base_cls):
        test_agent_loop_manager_cls = agent_loop_manager_cls

        def _compose_policy_ppo_config(self, *, policy_name, config_name, policy_entry):
            if policy_configs is not None:
                policy_config = OmegaConf.create(_plain(policy_configs[policy_name]))
                with open_dict(policy_config):
                    policy_config.policy_name = policy_name
                return policy_config
            return super()._compose_policy_ppo_config(
                policy_name=policy_name,
                config_name=config_name,
                policy_entry=policy_entry,
            )

        if policy_configs is not None and should_stub_component("_build_dataloader"):
            def _build_dataloader(self):
                source = next(
                    (trainer for trainer in self.policy_trainers.values()
                     if getattr(trainer, "train_dataloader", None) is not None),
                    None,
                )
                if source is not None:
                    self.train_dataset = getattr(source, "train_dataset", None)
                    self.val_dataset = getattr(source, "val_dataset", None)
                    self.train_dataloader = source.train_dataloader
                    self.val_dataloader = source.val_dataloader
                    self.train_dataloader_it = None
                    for trainer in self.policy_trainers.values():
                        if trainer is source:
                            continue
                        trainer.train_dataset = None
                        trainer.val_dataset = None
                        trainer.train_dataloader = None
                        trainer.val_dataloader = None
                        trainer.train_dataloader_it = None

        if policy_configs is not None and should_stub_component("_build_replay_buffer"):
            def _build_replay_buffer(self):
                self.replay_buffer = FakeReplayBuffer("outer")
                for index, trainer in enumerate(self.policy_trainers.values()):
                    trainer.replay_buffer = self.replay_buffer if index == 0 else None

    return TestableMultiAgentsPPOTrainer


def _make_trainer(
    base_cls,
    *,
    config=None,
    policy_configs=None,
    policy_trainer_cls=FakeV1PPOTrainer,
    agent_loop_manager_cls=FakeAgentFrameworkRolloutAdapter,
):
    _install_policy_trainer_stubs(policy_trainer_cls)
    trainer_cls = _test_multi_agents_trainer_cls(
        base_cls,
        policy_trainer_cls=policy_trainer_cls,
        agent_loop_manager_cls=agent_loop_manager_cls,
        policy_configs=policy_configs,
    )
    return trainer_cls(config=_config_with_policies(config, policy_configs))


def _init_agent_loop_and_fit(trainer):
    trainer.init()
    agent_loop_manager = _build_test_agent_loop_manager(trainer)
    trainer.fit(agent_loop_manager)
    return agent_loop_manager


def _build_test_agent_loop_manager(trainer):
    agent_loop_manager_cls = getattr(
        trainer,
        "test_agent_loop_manager_cls",
        FakeAgentFrameworkRolloutAdapter,
    )
    return agent_loop_manager_cls.create(
        config=trainer.config,
        llm_client=trainer.get_multi_policy_llm_client(),
        reward_loop_worker_handles=trainer.get_reward_handles(),
        gateway_actor_kwargs=trainer.get_gateway_actor_kwargs(),
    )


def _td_get(batch, key):
    value = batch.get(key)
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, NonTensorStack):
        return [item.data if isinstance(item, NonTensorData) else item for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def setup_function():
    _install_dependency_stubs()
    FakeV1PPOTrainer.instances.clear()
    FakeAgentFrameworkRolloutAdapter.create_calls.clear()


class TestMultiAgentsPPOTrainer:
    def setup_method(self):
        _install_dependency_stubs()
        FakeV1PPOTrainer.instances.clear()
        FakeAgentFrameworkRolloutAdapter.create_calls.clear()

    def test_constructor_matches_v1_config_only_signature(self):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        parameters = inspect.signature(MultiAgentsPPOTrainer).parameters

        assert list(parameters) == ["config"]
        assert parameters["config"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["config"].annotation == "DictConfig"
        assert not hasattr(trainer_module, "_maybe_await_sync")
        assert not hasattr(trainer_module, "_import_transfer_queue")
        assert hasattr(trainer_module, "tq")
        assert not hasattr(MultiAgentsPPOTrainer, "_resolve_policy_trainer_cls")
        assert not hasattr(MultiAgentsPPOTrainer, "_write_prompt_tags_to_transfer_queue")
        assert not hasattr(MultiAgentsPPOTrainer, "_sync_policy_global_steps")
        assert not hasattr(MultiAgentsPPOTrainer, "build_gateway_actor_kwargs")
        assert not hasattr(MultiAgentsPPOTrainer, "collect_reward_loop_worker_handles")
        assert not hasattr(MultiAgentsPPOTrainer, "prepare_policy_batches_for_advantage")
        assert hasattr(MultiAgentsPPOTrainer, "_sync_policy_runtime_context")

    def test_creates_one_v1_trainer_per_policy_config(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )

        assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
        assert trainer.policy_trainers["policy_1"].config.policy_name == "policy_1"
        assert trainer.policy_trainers["policy_2"].config.policy_name == "policy_2"
        assert not hasattr(trainer, "initialize_policy_trainers")
        assert not hasattr(trainer, "init_policy_trainers")
        assert not hasattr(trainer, "create_policy_trainers")
        assert hasattr(trainer, "_create_policy_trainers")

    def test_rejects_unsupported_v1_trainer_modes(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        with pytest.raises(ValueError, match="Unsupported trainer.v1.trainer_mode: 'colocate_async'"):
            _make_trainer(
                MultiAgentsPPOTrainer,
                policy_configs={
                    "policy_1": _policy_config(
                        "policy_1",
                        trainer=_NS(v1=_NS(trainer_mode="colocate_async")),
                    )
                },
            )

    def test_can_resolve_policy_configs_from_config_policies(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )

        assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]

    def test_uses_policy_mapping_key_as_policy_name(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "first": _policy_config("ignored_name_1"),
                "second": _policy_config("ignored_name_2"),
            },
        )

        assert list(trainer.policy_trainers) == ["first", "second"]
        assert trainer.policy_configs["first"].policy_name == "first"
        assert trainer.policy_configs["second"].policy_name == "second"

    def test_preserves_per_policy_resource_config_from_policies(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        policy_1_config = SimpleNamespace(
            policy_name="policy_1",
            trainer=SimpleNamespace(
                n_gpus_per_node=2,
                default_local_dir="checkpoints/policy_1",
                v1=_NS(trainer_mode="sync", sync=_NS(parameter_sync_step=1)),
            ),
            actor_rollout_ref=SimpleNamespace(
                model=SimpleNamespace(path="/models/policy_1"),
                rollout=SimpleNamespace(tensor_model_parallel_size=2, gpu_memory_utilization=0.6),
            ),
        )
        policy_2_config = SimpleNamespace(
            policy_name="policy_2",
            trainer=SimpleNamespace(
                n_gpus_per_node=4,
                default_local_dir="checkpoints/policy_2",
                v1=_NS(trainer_mode="sync", sync=_NS(parameter_sync_step=1)),
            ),
            actor_rollout_ref=SimpleNamespace(
                model=SimpleNamespace(path="/models/policy_2"),
                rollout=SimpleNamespace(tensor_model_parallel_size=4, gpu_memory_utilization=0.8),
            ),
        )
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={"policy_1": policy_1_config, "policy_2": policy_2_config},
        )

        assert trainer.policy_trainers["policy_1"].config.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
        assert trainer.policy_trainers["policy_2"].config.actor_rollout_ref.rollout.tensor_model_parallel_size == 4

    def test_can_resolve_policy_configs_from_root_level_ppo_trainer_compose_spec(self, monkeypatch):
        import hydra
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        compose_calls = []

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            compose_calls.append(("source", config_module, version_base))
            yield

        def fake_compose(*, config_name):
            compose_calls.append(("compose", config_name))
            return OmegaConf.create(
                {
                    "trainer": {"v1": {"trainer_mode": "sync"}},
                    "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}},
                }
            )

        monkeypatch.setattr(hydra, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(hydra, "compose", fake_compose)
        config = SimpleNamespace(
            ppo_trainer_config_source="verl.trainer.config",
            policies={
                "policy_1": _NS(
                    ppo_trainer_config_name="ppo_trainer",
                    ppo_trainer_overrides={
                        "actor_rollout_ref": {"model": {"path": "/models/policy_1"}},
                        "data": {"train_files": ["train.parquet"]},
                    },
                ),
                "policy_2": _NS(
                    ppo_trainer_config_name="ppo_trainer",
                    ppo_trainer_overrides={
                        "actor_rollout_ref": {"model": {"path": "/models/policy_2"}},
                    },
                ),
            },
        )

        trainer = _make_trainer(MultiAgentsPPOTrainer, config=config)

        assert compose_calls == [
            ("source", "verl.trainer.config", None),
            ("compose", "ppo_trainer"),
            ("source", "verl.trainer.config", None),
            ("compose", "ppo_trainer"),
        ]
        assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
        assert trainer.policy_configs["policy_1"].policy_name == "policy_1"
        assert trainer.policy_configs["policy_2"].policy_name == "policy_2"
        assert trainer.policy_configs["policy_1"].actor_rollout_ref.actor.optim.lr == 1e-6
        assert trainer.policy_configs["policy_1"].actor_rollout_ref.model.path == "/models/policy_1"
        assert trainer.policy_configs["policy_2"].actor_rollout_ref.model.path == "/models/policy_2"
        assert trainer.policy_configs["policy_1"].data.train_files == ["train.parquet"]

    def test_disables_per_policy_resume_for_outer_checkpoint_ownership(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    resume_mode="auto",
                    default_local_dir="checkpoints/multi_agent_blackbox",
                )
            ),
            policy_configs={
                "policy_1": _policy_config(
                    "policy_1",
                    trainer=_NS(
                        resume_mode="auto",
                        v1=_NS(trainer_mode="sync", sync=_NS(parameter_sync_step=1)),
                    ),
                ),
                "policy_2": _policy_config(
                    "policy_2",
                    trainer=_NS(
                        resume_mode="resume_path",
                        resume_from_path="checkpoints/policy_2/global_step_8",
                        v1=_NS(trainer_mode="sync", sync=_NS(parameter_sync_step=1)),
                    ),
                ),
            },
        )

        assert trainer.policy_trainers["policy_1"].config.trainer.resume_mode == "disable"
        assert trainer.policy_trainers["policy_2"].config.trainer.resume_mode == "disable"

    def test_agent_framework_config_keeps_role_policy_mapping_with_policies(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        role_policy_mapping = {
            "agent_1": "policy_1",
            "agent_2": "policy_1",
            "agent_3": "policy_2",
        }
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    custom=SimpleNamespace(
                        agent_framework=SimpleNamespace(
                            role_policy_mapping=role_policy_mapping,
                        )
                    )
                )
            ),
        )
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )

        _build_test_agent_loop_manager(trainer)

        create_kwargs = FakeAgentFrameworkRolloutAdapter.create_calls[0]
        assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
        assert create_kwargs["config"] is trainer.config
        assert (
            create_kwargs["config"]
            .actor_rollout_ref
            .rollout
            .custom
            .agent_framework
            .role_policy_mapping
            == role_policy_mapping
        )

    def test_init_orders_policy_runtime_setup_before_checkpoint_and_rollout_sync(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingInitTrainer(MultiAgentsPPOTrainer):
            def _build_dataloader(self):
                self.init_events.append("build_dataloader")

            def _build_replay_buffer(self):
                self.init_events.append("build_replay_buffer")

            def _load_checkpoint(self):
                self.init_events.append("load_checkpoint")

        trainer = _make_trainer(
            RecordingInitTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        trainer.init_events = []
        for policy_name, policy_trainer in trainer.policy_trainers.items():
            policy_trainer.init_runtime = lambda name=policy_name: trainer.init_events.append(f"{name}:init_runtime")
            policy_trainer.on_init_end = lambda name=policy_name: trainer.init_events.append(f"{name}:on_init_end")

        trainer.init()

        assert trainer.init_events == [
            "policy_1:init_runtime",
            "policy_2:init_runtime",
            "build_dataloader",
            "build_replay_buffer",
            "load_checkpoint",
            "policy_1:on_init_end",
            "policy_2:on_init_end",
        ]
        assert [policy_trainer.fit_calls for policy_trainer in trainer.policy_trainers.values()] == [0, 0]

    def test_init_runtime_does_not_use_legacy_init_workers(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingInitTrainer(MultiAgentsPPOTrainer):
            def _build_dataloader(self):
                pass

            def _build_replay_buffer(self):
                pass

            def _load_checkpoint(self):
                pass

        trainer = _make_trainer(
            RecordingInitTrainer,
            policy_configs={"policy_1": _policy_config("policy_1")},
            policy_trainer_cls=FakeV1PPOTrainerWithLegacyInitWorkers,
        )

        trainer.init()

        policy_trainer = trainer.policy_trainers["policy_1"]
        assert policy_trainer.runtime_init_calls == 1
        assert policy_trainer.init_calls == 0
        assert policy_trainer.on_init_end_calls == 1
        assert policy_trainer.init_workers_calls == 0

    def test_train_step_exposes_v1_add_batch_to_generate_boundary(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        assert hasattr(MultiAgentsPPOTrainer, "_add_batch_to_generate")
        assert not hasattr(MultiAgentsPPOTrainer, "submit_multi_agent_prompts")
        assert not hasattr(MultiAgentsPPOTrainer, "_get_agent_loop_manager_for_step")

    def test_train_step_uses_v1_add_batch_to_generate_boundary(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            def _add_batch_to_generate(self):
                self.step_events.append(("add_batch_to_generate", self.global_steps))

        trainer = _make_trainer(
            RecordingTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )
        trainer.step_events = []

        trainer.train_step()

        assert trainer.step_events == [
            ("add_batch_to_generate", 0),
            ("sample", 0),
            ("build", 0),
            ("update", 0, {"policy_1": "batch:0"}),
        ]

    def test_builds_one_shared_agent_loop_manager_with_policy_routing_client(self):
        _install_dependency_stubs()

        from uni_agent.trainer.gateway.runtime import PolicyRoutingLLMClient
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        assert hasattr(MultiAgentsPPOTrainer, "get_multi_policy_llm_client")
        assert not hasattr(MultiAgentsPPOTrainer, "build_policy_routing_client")
        assert not hasattr(MultiAgentsPPOTrainer, "get_llm_client")

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )

        trainer.init()
        first_replay_buffer = trainer.replay_buffer
        agent_loop_manager = _build_test_agent_loop_manager(trainer)

        assert agent_loop_manager.kind == "agent_loop_manager"
        assert len(FakeAgentFrameworkRolloutAdapter.create_calls) == 1
        create_kwargs = FakeAgentFrameworkRolloutAdapter.create_calls[0]
        assert isinstance(create_kwargs["llm_client"], PolicyRoutingLLMClient)
        assert trainer.replay_buffer is first_replay_buffer
        assert trainer.policy_trainers["policy_1"].replay_buffer is first_replay_buffer
        assert trainer.policy_trainers["policy_2"].replay_buffer is None
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

        routed = asyncio.run(
            create_kwargs["llm_client"].generate(
                "request-1",
                prompt_ids=[1, 2],
                sampling_params={"temperature": 0.1},
                policy_name="policy_2",
            )
        )

        assert routed == "policy_2:generated"
        assert trainer.policy_trainers["policy_1"].llm_client.calls == []
        assert trainer.policy_trainers["policy_2"].llm_client.calls[0]["request_id"] == "request-1"

    def test_gateway_actor_kwargs_include_per_policy_tool_parser_names(self):
        _install_dependency_stubs()

        from omegaconf import OmegaConf

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_configs = {
            "policy_1": OmegaConf.create(
                {"actor_rollout_ref": {"rollout": {"multi_turn": {"format": "qwen3_coder"}}}}
            ),
            "policy_2": OmegaConf.create(
                {"actor_rollout_ref": {"rollout": {"multi_turn": {"format": "hermes"}}}}
            ),
            "policy_3": OmegaConf.create({}),
        }
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(tokenizer="tokenizer:policy_1", processor=None),
            "policy_2": SimpleNamespace(tokenizer="tokenizer:policy_2", processor=None),
            "policy_3": SimpleNamespace(tokenizer="tokenizer:policy_3", processor=None),
        }

        gateway_actor_kwargs = trainer.get_gateway_actor_kwargs()

        assert gateway_actor_kwargs["policy_tool_parser_names"] == {
            "policy_1": "qwen3_coder",
            "policy_2": "hermes",
        }

    def test_fit_runs_outer_training_loop_without_calling_policy_fit(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=2),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert trainer.agent_loop_manager is not None
        assert [policy_trainer.runtime_init_calls for policy_trainer in trainer.policy_trainers.values()] == [1, 1]
        assert [policy_trainer.fit_calls for policy_trainer in trainer.policy_trainers.values()] == [0, 0]
        assert trainer.global_steps == 3
        assert [policy_trainer.global_steps for policy_trainer in trainer.policy_trainers.values()] == [3, 3]
        assert trainer.step_events == [
            ("sample", 1),
            ("build", 1),
            ("update", 1, {"policy_1": "batch:1"}),
            ("sample", 2),
            ("build", 2),
            ("update", 2, {"policy_1": "batch:2"}),
        ]

    def test_outer_checkpoint_loads_and_saves_multi_policy_state(self, tmp_path):
        _install_dependency_stubs()

        import torch

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        checkpoint_dir = checkpoint_root / "global_step_3"
        checkpoint_dir.mkdir(parents=True)
        for policy_name, use_critic in (("policy_1", True), ("policy_2", False)):
            (checkpoint_dir / "policies" / policy_name / "actor").mkdir(parents=True)
            if use_critic:
                (checkpoint_dir / "policies" / policy_name / "Critic").mkdir()
        torch.save({"loaded": "outer-dataloader"}, checkpoint_dir / "data.pt")

        config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir),
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
                del_local_ckpt_after_load=False,
            ),
        )
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1", use_critic=True),
                "policy_2": _policy_config("policy_2", use_critic=False),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()

        assert trainer.global_steps == 3
        assert trainer.train_dataloader.loaded_states == [{"loaded": "outer-dataloader"}]
        policy_1 = trainer.policy_trainers["policy_1"]
        policy_2 = trainer.policy_trainers["policy_2"]
        assert policy_1.actor_rollout_wg.load_calls == [
            {
                "local_path": str(checkpoint_dir / "policies" / "policy_1" / "actor"),
                "del_local_after_load": False,
            }
        ]
        assert policy_1.critic_wg.load_calls == [
            {
                "local_path": str(checkpoint_dir / "policies" / "policy_1" / "Critic"),
                "del_local_after_load": False,
            }
        ]
        assert policy_2.actor_rollout_wg.load_calls == [
            {
                "local_path": str(checkpoint_dir / "policies" / "policy_2" / "actor"),
                "del_local_after_load": False,
            }
        ]
        assert policy_2.critic_wg.load_calls == []

        trainer.global_steps = 4
        trainer._save_checkpoint()

        save_dir = checkpoint_root / "global_step_4"
        assert (save_dir / "data.pt").exists()
        assert (checkpoint_root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8") == "4"
        assert policy_1.actor_rollout_wg.save_calls[0]["args"][0] == str(save_dir / "policies" / "policy_1" / "actor")
        assert policy_1.critic_wg.save_calls[0]["args"][0] == str(save_dir / "policies" / "policy_1" / "Critic")
        assert policy_2.actor_rollout_wg.save_calls[0]["args"][0] == str(save_dir / "policies" / "policy_2" / "actor")
        assert policy_2.critic_wg.save_calls == []

    def test_auto_resume_uses_checkpoint_tracker_instead_of_largest_directory(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        expected = checkpoint_root / "global_step_3"
        incomplete = checkpoint_root / "global_step_4"
        expected.mkdir(parents=True)
        incomplete.mkdir()
        (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="auto",
                default_local_dir=str(checkpoint_root),
            )
        )

        assert trainer._resolve_checkpoint_dir() == str(expected)

    def test_resume_path_accepts_trailing_separator(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_dir = tmp_path / "global_step_3"
        checkpoint_dir.mkdir()
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir) + "/",
                del_local_ckpt_after_load=False,
            )
        )
        trainer.policy_trainers = {}
        trainer.train_dataloader = None
        trainer.timing_raw = {}
        trainer.trainer_mode = "sync"

        trainer._load_checkpoint()

        assert trainer.global_steps == 3

    def test_checkpoint_load_rejects_missing_policy_actor_directory(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_dir = tmp_path / "global_step_3"
        checkpoint_dir.mkdir()
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir),
                del_local_ckpt_after_load=False,
            )
        )
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                actor_rollout_wg=FakeWorkerGroup("actor:policy_1"),
                use_critic=False,
            )
        }
        trainer.train_dataloader = None
        trainer.timing_raw = {}

        with pytest.raises(FileNotFoundError, match="policy_1.*actor"):
            trainer._load_checkpoint()

    def test_async_checkpoint_restores_saves_and_reissues_transfer_queue_state(self, tmp_path, monkeypatch):
        _install_dependency_stubs()

        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        checkpoint_dir = checkpoint_root / "global_step_3"
        (checkpoint_dir / "transfer_queue").mkdir(parents=True)
        calls = {"load": [], "save": [], "put": [], "clear": []}
        from verl.utils import tensordict_utils as tu

        restored_batch = tu.get_tensordict(
            {"uid": ["uid-a", "uid-b"], "raw_prompt": ["a", "b"]}
        )
        queue_items = {
            "train": {
                "uid-a": {"is_prompt": True, "status": "pending"},
                "uid-b": {"is_prompt": True, "status": "running"},
                "uid-c": {"is_prompt": True, "status": "finished"},
                "uid-a_0_0": {"policy_name": "policy_1"},
                "uid-b_0_0": {"policy_name": "policy_2"},
                "uid-c_0_0": {"policy_name": "policy_1"},
            }
        }
        monkeypatch.setattr(trainer_module.tq, "__version__", "0.1.9", raising=False)
        monkeypatch.setattr(
            trainer_module.tq,
            "load_checkpoint",
            lambda path: calls["load"].append(path),
            raising=False,
        )
        monkeypatch.setattr(
            trainer_module.tq,
            "save_checkpoint",
            lambda path, metadata: calls["save"].append((path, metadata)),
            raising=False,
        )
        monkeypatch.setattr(trainer_module.tq, "kv_list", lambda partition_id: queue_items, raising=False)
        monkeypatch.setattr(
            trainer_module.tq,
            "kv_batch_get",
            lambda **kwargs: restored_batch,
            raising=False,
        )
        monkeypatch.setattr(
            trainer_module.tq,
            "kv_clear",
            lambda **kwargs: calls["clear"].append(kwargs),
            raising=False,
        )
        monkeypatch.setattr(
            trainer_module.tq,
            "kv_batch_put",
            lambda **kwargs: calls["put"].append(kwargs),
        )

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir),
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
                del_local_ckpt_after_load=False,
            )
        )
        trainer.trainer_mode = "separate_async"
        trainer.policy_trainers = {}
        trainer.train_dataloader = None
        trainer.timing_raw = {}
        trainer.agent_loop_manager = FakeAgentFrameworkRolloutAdapter()

        trainer._load_checkpoint()
        trainer.global_steps = 4
        reissued = trainer._reissue_inflight_prompts()
        trainer._save_checkpoint()

        assert calls["load"] == [str(checkpoint_dir / "transfer_queue")]
        assert calls["clear"] == [
            {"keys": ["uid-a_0_0", "uid-b_0_0"], "partition_id": "train"}
        ]
        assert calls["put"] == [
            {
                "keys": ["uid-a", "uid-b"],
                "partition_id": "train",
                "tags": [
                    {"is_prompt": True, "status": "pending", "global_steps": 4},
                    {"is_prompt": True, "status": "pending", "global_steps": 4},
                ],
            }
        ]
        assert reissued == 2
        assert restored_batch["global_steps"] == 4
        assert trainer.agent_loop_manager.generated_prompts == [restored_batch]
        assert calls["save"] == [
            (
                str(checkpoint_root / "global_step_4" / "transfer_queue"),
                {"global_steps": 4},
            )
        ]

    def test_multi_policy_async_checkpoint_save_does_not_publish_outer_tracker(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
            )
        )
        trainer.policy_configs = {
            "policy_1": SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    actor=SimpleNamespace(checkpoint=SimpleNamespace(async_save=True))
                )
            )
        }
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                actor_rollout_wg=FakeWorkerGroup("actor:policy_1"),
                use_critic=False,
            )
        }
        trainer.global_steps = 4
        trainer.trainer_mode = "sync"
        trainer.train_dataloader = None

        assert trainer._has_async_checkpoint_save() is True
        trainer._save_checkpoint()

        assert not (checkpoint_root / "latest_checkpointed_iteration.txt").exists()

    def test_multi_policy_checkpoint_save_forwards_native_retention_limits(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
                max_actor_ckpt_to_keep=2,
                max_critic_ckpt_to_keep=3,
            )
        )
        trainer.policy_configs = {}
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                actor_rollout_wg=FakeWorkerGroup("actor:policy_1"),
                critic_wg=FakeWorkerGroup("critic:policy_1"),
                use_critic=True,
            )
        }
        trainer.global_steps = 4
        trainer.trainer_mode = "sync"
        trainer.train_dataloader = None

        trainer._save_checkpoint()

        actor_call = trainer.policy_trainers["policy_1"].actor_rollout_wg.save_calls[0]
        critic_call = trainer.policy_trainers["policy_1"].critic_wg.save_calls[0]
        assert actor_call["kwargs"]["max_ckpt_to_keep"] == 2
        assert critic_call["kwargs"]["max_ckpt_to_keep"] == 3

    def test_checkpoint_save_fails_fast_when_policy_actor_worker_is_unavailable(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
            )
        )
        trainer.policy_configs = {}
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                actor_rollout_wg=None,
                use_critic=False,
            )
        }
        trainer.global_steps = 1
        trainer.trainer_mode = "sync"
        trainer.train_dataloader = None

        with pytest.raises(AttributeError):
            trainer._save_checkpoint()

        assert not (checkpoint_root / "latest_checkpointed_iteration.txt").exists()

    def test_checkpoint_load_fails_fast_when_policy_actor_worker_is_unavailable(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_dir = tmp_path / "global_step_1"
        (checkpoint_dir / "policies" / "policy_1" / "actor").mkdir(parents=True)
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir),
                del_local_ckpt_after_load=False,
            )
        )
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                actor_rollout_wg=None,
                use_critic=False,
            )
        }
        trainer.train_dataloader = None
        trainer.timing_raw = {}
        trainer.trainer_mode = "sync"

        with pytest.raises(AttributeError):
            trainer._load_checkpoint()

    def test_build_per_policy_batches_groups_trajectories_by_policy_name(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        batch = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0", "uid_0_1", "uid_1_0"],
            tags=[
                {"policy_name": "policy_1", "role": "agent_1"},
                {"policy_name": "policy_2", "role": "agent_2"},
                {"policy_name": "policy_1", "role": "agent_3"},
            ],
            extra_info={"temperature": 0.7},
        )

        grouped = trainer.build_per_policy_batches(batch)

        assert list(grouped) == ["policy_1", "policy_2"]
        assert grouped["policy_1"].keys == ["uid_0_0", "uid_1_0"]
        assert grouped["policy_1"].tags == [
            {"policy_name": "policy_1", "role": "agent_1"},
            {"policy_name": "policy_1", "role": "agent_3"},
        ]
        assert grouped["policy_1"].partition_id == "train"
        assert grouped["policy_1"].extra_info == {"temperature": 0.7}
        assert grouped["policy_2"].keys == ["uid_0_1"]

    def test_step_once_prepares_advantage_once_then_updates_each_policy(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1", use_reference_policy=True, use_critic=True),
                "policy_2": _policy_config("policy_2", use_reference_policy=False, use_critic=False),
            },
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0", "uid_0_1", "uid_1_0"],
            tags=[
                {"policy_name": "policy_1", "role": "agent_1"},
                {"policy_name": "policy_2", "role": "agent_2"},
                {"policy_name": "policy_1", "role": "agent_3"},
            ],
        )
        trainer.replay_buffer = SimpleNamespace(
            sample=lambda **kwargs: (
                sample,
                {"training/off_policy/dropped_samples": 0},
            )
        )
        metrics = {}

        result = trainer._step_once(metrics=metrics, timing_raw={}, sample_batch_size=1)

        assert result.keys == ["uid_0_0", "uid_1_0", "uid_0_1"]
        assert metrics["training/off_policy/dropped_samples"] == 0
        assert metrics["policy_1/old_log_prob/count"] == 2
        assert metrics["policy_1/ref_log_prob/count"] == 2
        assert metrics["policy_1/values/count"] == 2
        assert metrics["policy_1/update_critic/count"] == 2
        assert metrics["policy_1/update_actor/count"] == 2
        assert metrics["policy_2/old_log_prob/count"] == 1
        assert "policy_2/ref_log_prob/count" not in metrics
        assert "policy_2/values/count" not in metrics
        assert "policy_2/update_critic/count" not in metrics
        assert metrics["policy_2/update_actor/count"] == 1

        policy_1 = trainer.policy_trainers["policy_1"]
        policy_2 = trainer.policy_trainers["policy_2"]
        assert policy_1.stage_calls == [
            ("balance", ["uid_0_0", "uid_1_0"], "global_seqlen"),
            ("old_log_prob", ["uid_0_0", "uid_1_0"]),
            ("ref_log_prob", ["uid_0_0", "uid_1_0"]),
            ("values", ["uid_0_0", "uid_1_0"]),
            ("advantage", ["uid_0_0", "uid_1_0", "uid_0_1"]),
            ("update_critic", ["uid_0_0", "uid_1_0"]),
            ("update_actor", ["uid_0_0", "uid_1_0"]),
        ]
        assert policy_2.stage_calls == [
            ("balance", ["uid_0_1"], "global_seqlen"),
            ("old_log_prob", ["uid_0_1"]),
            ("update_actor", ["uid_0_1"]),
        ]

    def test_step_once_reuses_initial_policy_batches_after_advantage(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class CountingTrainer(MultiAgentsPPOTrainer):
            def __init__(self, **kwargs):
                self.build_per_policy_batches_calls = 0
                super().__init__(**kwargs)

            def build_per_policy_batches(self, multi_agent_batch):
                self.build_per_policy_batches_calls += 1
                return super().build_per_policy_batches(multi_agent_batch)

        trainer = _make_trainer(
            CountingTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0", "uid_0_1"],
            tags=[
                {"policy_name": "policy_1", "role": "agent_1"},
                {"policy_name": "policy_2", "role": "agent_2"},
            ],
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: sample)

        trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=1)

        assert trainer.build_per_policy_batches_calls == 2

    def test_step_once_computes_advantage_from_policy_batches(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(MultiAgentsPPOTrainer):
            def __init__(self, **kwargs):
                self.advantage_policy_batch_keys = None
                super().__init__(**kwargs)

            def compute_multi_agent_advantage_from_policy_batches(self, per_policy_batches, metrics):
                self.advantage_policy_batch_keys = {
                    policy_name: list(batch.keys)
                    for policy_name, batch in per_policy_batches.items()
                }
                return super().compute_multi_agent_advantage_from_policy_batches(
                    per_policy_batches,
                    metrics,
                )

        trainer = _make_trainer(
            RecordingTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0", "uid_0_1"],
            tags=[
                {"policy_name": "policy_1", "role": "agent_1"},
                {"policy_name": "policy_2", "role": "agent_2"},
            ],
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: sample)

        trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=1)

        assert trainer.advantage_policy_batch_keys == {
            "policy_1": ["uid_0_0"],
            "policy_2": ["uid_0_1"],
        }

    def test_step_once_prepares_policy_batches_for_ppo_update(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(MultiAgentsPPOTrainer):
            def __init__(self, **kwargs):
                self.prepare_for_ppo_update_calls = 0
                super().__init__(**kwargs)

            def prepare_policy_batches_for_ppo_update(self, per_policy_batches, metrics):
                self.prepare_for_ppo_update_calls += 1
                return super().prepare_policy_batches_for_ppo_update(per_policy_batches, metrics)

        trainer = _make_trainer(
            RecordingTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0", "uid_0_1"],
            tags=[
                {"policy_name": "policy_1", "role": "agent_1"},
                {"policy_name": "policy_2", "role": "agent_2"},
            ],
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: sample)

        trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=1)

        assert trainer.prepare_for_ppo_update_calls == 1

    def test_prepare_policy_batches_prefixes_balance_metrics_once(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class FakeBalanceMetricsPPOTrainer(FakeV1PPOTrainer):
            def _balance_batch(self, batch, metrics=None, logging_prefix=None):
                self.stage_calls.append(("balance", list(batch.keys), logging_prefix))
                if metrics is not None:
                    metrics[f"{logging_prefix}/mean"] = 1.0
                return batch

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
            policy_trainer_cls=FakeBalanceMetricsPPOTrainer,
        )
        batch = SimpleKVBatchMeta(
            partition_id="train",
            keys=["uid_0_0"],
            tags=[{"policy_name": "policy_1", "role": "agent_1"}],
        )
        metrics = {}

        trainer.prepare_policy_batches_for_ppo_update({"policy_1": batch}, metrics)

        assert metrics["policy_1/global_seqlen/mean"] == 1.0
        assert "policy_1/policy_1/global_seqlen/mean" not in metrics

    def test_step_once_assigns_rollout_advantage_before_per_policy_updates(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        records = {
            "uid-1_0_0": {
                "uid": "uid-1",
                "rollout_id": "rollout-0",
                "sample_idx": 0,
                "policy_name": "policy_1",
                "role": "agent_1",
                "reward": 2.0,
            },
            "uid-1_0_1": {
                "uid": "uid-1",
                "rollout_id": "rollout-0",
                "sample_idx": 0,
                "policy_name": "policy_2",
                "role": "agent_2",
                "reward": 2.0,
            },
            "uid-1_1_0": {
                "uid": "uid-1",
                "rollout_id": "rollout-1",
                "sample_idx": 1,
                "policy_name": "policy_1",
                "role": "agent_1",
                "reward": 4.0,
            },
            "uid-1_1_1": {
                "uid": "uid-1",
                "rollout_id": "rollout-1",
                "sample_idx": 1,
                "policy_name": "policy_2",
                "role": "agent_2",
                "reward": 4.0,
            },
        }
        FakeSharedAdvantagePPOTrainer.shared_records = records
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6, n=2)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
            policy_trainer_cls=FakeSharedAdvantagePPOTrainer,
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=list(records),
            tags=[
                {
                    "uid": record["uid"],
                    "rollout_id": record["rollout_id"],
                    "sample_idx": record["sample_idx"],
                    "policy_name": record["policy_name"],
                    "role": record["role"],
                }
                for record in records.values()
            ],
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: sample)
        metrics = {}

        result = trainer._step_once(metrics=metrics, timing_raw={}, sample_batch_size=1)

        assert result.keys == [
            "uid-1_0_0",
            "uid-1_1_0",
            "uid-1_0_1",
            "uid-1_1_1",
        ]
        assert metrics["advantage/rollout_count"] == 2
        assert records["uid-1_0_0"]["advantage"] == -1.0
        assert records["uid-1_0_1"]["advantage"] == -1.0
        assert records["uid-1_1_0"]["advantage"] == 1.0
        assert records["uid-1_1_1"]["advantage"] == 1.0
        assert trainer.policy_trainers["policy_1"].updated_records == {
            "uid-1_0_0": {
                "policy_name": "policy_1",
                "role": "agent_1",
                "rollout_id": "rollout-0",
                "advantage": -1.0,
            },
            "uid-1_1_0": {
                "policy_name": "policy_1",
                "role": "agent_1",
                "rollout_id": "rollout-1",
                "advantage": 1.0,
            },
        }
        assert trainer.policy_trainers["policy_2"].updated_records == {
            "uid-1_0_1": {
                "policy_name": "policy_2",
                "role": "agent_2",
                "rollout_id": "rollout-0",
                "advantage": -1.0,
            },
            "uid-1_1_1": {
                "policy_name": "policy_2",
                "role": "agent_2",
                "rollout_id": "rollout-1",
                "advantage": 1.0,
            },
        }

    def test_step_once_supports_many_roles_mapping_to_fewer_policy_trainers(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        role_policy_mapping = {
            "agent_1": "policy_1",
            "agent_2": "policy_1",
            "agent_3": "policy_2",
            "agent_4": "policy_3",
        }
        rollout_rewards = [1.0, 2.0, 3.0, 4.0]
        records = {}
        for sample_idx, reward in enumerate(rollout_rewards):
            rollout_id = f"rollout-{sample_idx}"
            for record_idx, (role, policy_name) in enumerate(role_policy_mapping.items()):
                key = f"uid-1_{sample_idx}_{record_idx}"
                records[key] = {
                    "uid": "uid-1",
                    "rollout_id": rollout_id,
                    "sample_idx": sample_idx,
                    "record_idx": record_idx,
                    "policy_name": policy_name,
                    "role": role,
                    "reward": reward,
                }

        FakeSharedAdvantagePPOTrainer.shared_records = records
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(critic_warmup=0),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=0.6, n=4)),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
                "policy_3": _policy_config("policy_3"),
            },
            policy_trainer_cls=FakeSharedAdvantagePPOTrainer,
        )
        sample = SimpleKVBatchMeta(
            partition_id="train",
            keys=list(records),
            tags=[
                {
                    "uid": record["uid"],
                    "rollout_id": record["rollout_id"],
                    "sample_idx": record["sample_idx"],
                    "record_idx": record["record_idx"],
                    "policy_name": record["policy_name"],
                    "role": record["role"],
                }
                for record in records.values()
            ],
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: sample)

        trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=1)

        expected_advantages = {
            "rollout-0": -1.5,
            "rollout-1": -0.5,
            "rollout-2": 0.5,
            "rollout-3": 1.5,
        }
        for key, record in records.items():
            assert record["advantage"] == expected_advantages[record["rollout_id"]], key

        assert list(trainer.policy_trainers["policy_1"].updated_records) == [
            "uid-1_0_0",
            "uid-1_0_1",
            "uid-1_1_0",
            "uid-1_1_1",
            "uid-1_2_0",
            "uid-1_2_1",
            "uid-1_3_0",
            "uid-1_3_1",
        ]
        assert {
            record["role"] for record in trainer.policy_trainers["policy_1"].updated_records.values()
        } == {"agent_1", "agent_2"}
        assert list(trainer.policy_trainers["policy_2"].updated_records) == [
            "uid-1_0_2",
            "uid-1_1_2",
            "uid-1_2_2",
            "uid-1_3_2",
        ]
        assert {
            record["role"] for record in trainer.policy_trainers["policy_2"].updated_records.values()
        } == {"agent_3"}
        assert list(trainer.policy_trainers["policy_3"].updated_records) == [
            "uid-1_0_3",
            "uid-1_1_3",
            "uid-1_2_3",
            "uid-1_3_3",
        ]
        assert {
            record["role"] for record in trainer.policy_trainers["policy_3"].updated_records.values()
        } == {"agent_4"}

    def test_fit_uses_task_runner_owned_transfer_queue_lifecycle(self, monkeypatch):
        _install_dependency_stubs()

        from uni_agent.trainer import multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        fake_tq = FakeTransferQueue()
        monkeypatch.setattr(trainer_module, "tq", fake_tq)
        transfer_queue_config = SimpleNamespace(enable=False)
        config = SimpleNamespace(
            transfer_queue=transfer_queue_config,
            trainer=SimpleNamespace(total_training_steps=0),
        )
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )

        _init_agent_loop_and_fit(trainer)

        assert not hasattr(MultiAgentsPPOTrainer, "init_transfer_queue")
        assert not hasattr(MultiAgentsPPOTrainer, "close_transfer_queue")
        assert fake_tq.init_calls == []
        assert fake_tq.close_calls == 0
        assert transfer_queue_config.enable is False

    def test_does_not_fallback_to_prompt_loader_without_promoted_v1_dataloader(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        data_path = tmp_path / "train.jsonl"
        data_path.write_text('{"prompt": "build feature"}\n', encoding="utf-8")
        config = SimpleNamespace(
            training=SimpleNamespace(
                train_data_path=str(data_path),
                train_batch_size=1,
                prompt_loader=SimpleNamespace(
                    source_type="jsonl",
                    prompt_keys=["prompt"],
                    expected_keys=[],
                    train_repeat=False,
                    train_shuffle=False,
                ),
            ),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )
        trainer.step_events = []
        trainer.agent_loop_manager = _build_test_agent_loop_manager(trainer)

        trainer.train_step()

        assert len(trainer.agent_loop_manager.generated_prompts) == 0
        assert trainer.step_events == [
            ("sample", 0),
            ("build", 0),
            ("update", 0, {"policy_1": "batch:0"}),
        ]

    def test_fit_uses_injected_agent_loop_manager(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=2),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )
        trainer.step_events = []

        agent_loop_manager = _init_agent_loop_and_fit(trainer)

        assert trainer.agent_loop_manager is agent_loop_manager
        assert len(agent_loop_manager.generated_prompts) == 0

    def test_sync_runtime_hooks_drive_per_policy_checkpoint_managers(self, monkeypatch):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        marked_timer = RecordingMarkedTimer()
        debug_module = types.ModuleType("verl.utils.debug")
        debug_module.marked_timer = marked_timer
        monkeypatch.setitem(sys.modules, "verl.utils.debug", debug_module)
        sys.modules["verl.utils"].debug = debug_module

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        trainer.global_steps = 7
        trainer.timing_raw = {}

        trainer.on_sample_end()
        trainer.on_step_end()

        assert trainer.policy_trainers["policy_1"].checkpoint_manager.sleep_calls == 1
        assert trainer.policy_trainers["policy_2"].checkpoint_manager.sleep_calls == 1
        assert trainer.policy_trainers["policy_1"].checkpoint_manager.update_weight_steps == [7]
        assert trainer.policy_trainers["policy_2"].checkpoint_manager.update_weight_steps == [7]
        assert [call["name"] for call in marked_timer.calls] == [
            "policy_1/sleep_replicas",
            "policy_2/sleep_replicas",
            "policy_1/update_weights",
            "policy_2/update_weights",
        ]
        assert all(isinstance(call["timing_raw"], dict) for call in marked_timer.calls)
        assert all(call["kwargs"]["color"] == "red" for call in marked_timer.calls)

    def test_fit_loads_outer_checkpoint_before_training_loop(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            def _load_checkpoint(self):
                self.step_events.append(("load_checkpoint", self.global_steps))
                self.global_steps = 1

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=2),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert trainer.step_events == [
            ("load_checkpoint", 0),
            ("sample", 2),
            ("build", 2),
            ("update", 2, {"policy_1": "batch:2"}),
        ]
        assert trainer.global_steps == 3

    def test_fit_saves_outer_checkpoint_by_save_frequency(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            def _save_checkpoint(self):
                self.step_events.append(("save_checkpoint", self.global_steps))

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=2, save_freq=1),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert ("save_checkpoint", 1) in trainer.step_events
        assert ("save_checkpoint", 2) in trainer.step_events

    def test_shared_dataloader_uses_first_initialized_policy(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )

        trainer.init()

        first_policy_trainer = trainer.policy_trainers["policy_1"]
        assert trainer.train_dataset == "train_dataset:policy_1"
        assert trainer.val_dataset == "val_dataset:policy_1"
        assert trainer.train_dataloader is first_policy_trainer.train_dataloader
        assert trainer.val_dataloader is first_policy_trainer.val_dataloader

        second_policy_trainer = trainer.policy_trainers["policy_2"]
        assert second_policy_trainer.train_dataset is None
        assert second_policy_trainer.val_dataset is None
        assert second_policy_trainer.train_dataloader is None
        assert second_policy_trainer.val_dataloader is None
        assert second_policy_trainer.train_dataloader_it is None

    def test_train_step_uses_promoted_v1_dataloader_batch(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=1),
            data=SimpleNamespace(train_batch_size=1),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert len(trainer.agent_loop_manager.generated_prompts) == 1
        generated = trainer.agent_loop_manager.generated_prompts[0]
        uid = _td_get(generated, "uid")[0]
        assert isinstance(uid, str)
        assert uid
        assert _td_get(generated, "raw_prompt") == [[{"role": "user", "content": "prompt:policy_1"}]]
        assert _td_get(generated, "reward_model") == [{"ground_truth": "answer:policy_1"}]
        assert _td_get(generated, "tools_kwargs") == [{"env": {"image": "image:policy_1"}}]
        assert _td_get(generated, "data_source") == ["source:policy_1"]
        assert _td_get(generated, "global_steps") == 1

    def test_submit_batch_to_rollout_marks_prompt_pending_before_dispatch(self, monkeypatch):
        _install_dependency_stubs()

        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        fake_tq = FakeTransferQueue()
        monkeypatch.setattr(trainer_module, "tq", fake_tq)

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()
        trainer.agent_loop_manager = _build_test_agent_loop_manager(trainer)

        batch = trainer._next_train_batch()
        trainer._submit_batch_to_rollout(batch)

        assert len(fake_tq.batch_puts) == 1
        batch_put = fake_tq.batch_puts[0]
        assert batch_put["partition_id"] == "train"
        assert len(batch_put["keys"]) == 1
        assert batch_put["tags"] == [
            {
                "is_prompt": True,
                "status": "pending",
                "global_steps": 0,
            }
        ]
        assert len(trainer.agent_loop_manager.generated_prompts) == 1

    def test_rollout_submission_uses_sync_transfer_queue_api(self, monkeypatch):
        _install_dependency_stubs()

        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        no_sync_batch_put_tq = SimpleNamespace()
        monkeypatch.setattr(trainer_module, "tq", no_sync_batch_put_tq)

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()

        try:
            trainer._submit_batch_to_rollout(trainer._next_train_batch())
        except AttributeError:
            pass
        else:
            raise AssertionError("multi-agent trainer should call sync transfer_queue.kv_batch_put directly")
