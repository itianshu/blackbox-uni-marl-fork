import ast
import asyncio
import inspect
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Lock
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
        module = sys.modules.setdefault(name, types.ModuleType(name))
        if not hasattr(module, "__path__"):
            module.__path__ = []

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

    tq_stub = types.ModuleType("transfer_queue")
    for name in ("init", "close", "kv_batch_put", "kv_put", "kv_clear"):
        setattr(tq_stub, name, lambda *args, **kwargs: None)
    tq_stub.KVBatchMeta = SimpleKVBatchMeta
    sys.modules["transfer_queue"] = tq_stub

    trainer_module = sys.modules.get("uni_agent.trainer.multi_agents_ppo_trainer")
    if trainer_module is not None:
        trainer_module.tq = tq_stub

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
    skip_mod = sys.modules.setdefault("verl.utils.skip", types.ModuleType("verl.utils.skip"))
    if not hasattr(skip_mod, "SkipManager"):
        skip_mod.SkipManager = type("SkipManager", (), {"init": staticmethod(lambda config: None)})
    sys.modules["verl.utils"].skip = skip_mod

    trainer_mod = sys.modules.setdefault("verl.trainer", types.ModuleType("verl.trainer"))
    ppo_mod = sys.modules.setdefault("verl.trainer.ppo", types.ModuleType("verl.trainer.ppo"))
    v1_mod = sys.modules.setdefault("verl.trainer.ppo.v1", types.ModuleType("verl.trainer.ppo.v1"))
    trainer_mod.ppo = ppo_mod
    ppo_mod.v1 = v1_mod

    metric_mod = sys.modules.setdefault(
        "verl.trainer.ppo.metric_utils", types.ModuleType("verl.trainer.ppo.metric_utils")
    )
    metric_mod.compute_data_metrics = getattr(metric_mod, "compute_data_metrics", lambda *args, **kwargs: {})
    metric_mod.compute_timing_metrics = getattr(metric_mod, "compute_timing_metrics", lambda *args, **kwargs: {})
    metric_mod.compute_throughout_metrics = getattr(
        metric_mod, "compute_throughout_metrics", lambda *args, **kwargs: {}
    )
    metric_mod.process_validation_metrics = getattr(
        metric_mod, "process_validation_metrics", lambda *args, **kwargs: {}
    )
    utils_mod = sys.modules.setdefault("verl.trainer.ppo.utils", types.ModuleType("verl.trainer.ppo.utils"))
    utils_mod.create_rl_dataset = getattr(utils_mod, "create_rl_dataset", lambda *args, **kwargs: [])
    utils_mod.create_rl_sampler = getattr(utils_mod, "create_rl_sampler", lambda *args, **kwargs: None)
    replay_mod = sys.modules.setdefault(
        "verl.trainer.ppo.v1.replay_buffer", types.ModuleType("verl.trainer.ppo.v1.replay_buffer")
    )
    replay_mod.ReplayBuffer = getattr(replay_mod, "ReplayBuffer", object)
    replay_mod.ReplayBufferAsync = getattr(replay_mod, "ReplayBufferAsync", object)
    v1_utils_mod = sys.modules.setdefault(
        "verl.trainer.ppo.v1.utils", types.ModuleType("verl.trainer.ppo.v1.utils")
    )
    v1_utils_mod.MetricsAggregator = getattr(v1_utils_mod, "MetricsAggregator", object)
    dataset_mod = sys.modules.setdefault(
        "verl.utils.dataset.rl_dataset", types.ModuleType("verl.utils.dataset.rl_dataset")
    )
    dataset_mod.collate_fn = getattr(dataset_mod, "collate_fn", lambda batch: batch)

    protocol_mod = sys.modules.setdefault("verl.protocol", types.ModuleType("verl.protocol"))
    protocol_mod.DataProto = getattr(protocol_mod, "DataProto", object)


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
        self.training_runtime_init_calls = 0
        self.standalone_runtime_init_calls = 0
        self.on_init_end_calls = 0
        self.on_validate_begin_calls = 0
        self.on_validate_end_calls = 0
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

    def init_training_runtime(self):
        self.training_runtime_init_calls += 1

    def init_standalone_rollout_runtime(self):
        self.standalone_runtime_init_calls += 1

    def on_init_end(self):
        self.on_init_end_calls += 1

    def on_validate_begin(self):
        self.on_validate_begin_calls += 1

    def on_validate_end(self):
        self.on_validate_end_calls += 1

    def get_llm_client(self):
        return self.llm_client

    def get_reward_handles(self):
        return self.reward_handles

    def _consume_sync_metrics(self):
        metrics = getattr(self, "_pending_sync_metrics", None) or {}
        self._pending_sync_metrics = {}
        return metrics

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

    def _get_n_gpus_for_throughput(self):
        return 1

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
    def init_training_runtime(self):
        super().init_training_runtime()
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


def _policy_config_with_rollout(policy_name, *, total_gpus, tensor_parallel_size):
    return _policy_config(
        policy_name,
        actor_rollout_ref=_NS(
            rollout=_NS(
                nnodes=1,
                n_gpus_per_node=total_gpus,
                tensor_model_parallel_size=tensor_parallel_size,
                data_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        ),
    )


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
    if not hasattr(config, "ppo_trainer_config_name"):
        if isinstance(config, DictConfig):
            with open_dict(config):
                config.ppo_trainer_config_name = "test_policy_config"
        else:
            config.ppo_trainer_config_name = "test_policy_config"
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
            val_before_train=False,
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
        if not hasattr(trainer, "val_before_train"):
            trainer.val_before_train = False
        if not hasattr(trainer, "v1"):
            trainer.v1 = _NS(
                trainer_mode="sync",
                sync=_NS(parameter_sync_step=1),
                separate_async=_NS(parameter_sync_step=1),
            )
    if policy_configs is not None:
        config.policies = {
            policy_name: {"ppo_trainer_overrides": {}}
            for policy_name, policy_config in policy_configs.items()
        }
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(_plain(config))
    policy_names = list(getattr(config, "policies", {}) or {})
    if OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping",
        default=None,
    ) is None:
        with open_dict(config):
            config.actor_rollout_ref = config.get("actor_rollout_ref", {})
            config.actor_rollout_ref.rollout = config.actor_rollout_ref.get("rollout", {})
            config.actor_rollout_ref.rollout.custom = config.actor_rollout_ref.rollout.get("custom", {})
            config.actor_rollout_ref.rollout.custom.agent_framework = (
                config.actor_rollout_ref.rollout.custom.get("agent_framework", {})
            )
            config.actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping = {
                f"agent_{index}": policy_name
                for index, policy_name in enumerate(policy_names, start=1)
            }
    return config


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


def _install_metrics_aggregator_stub():
    from collections import defaultdict

    utils_mod = types.ModuleType("verl.trainer.ppo.v1.utils")

    class MetricsAggregator:
        def __init__(self):
            self.values = defaultdict(list)
            self.weights = defaultdict(list)

        def add_step_metrics(self, metrics, sample_count=0):
            for key, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                self.values[key].append(float(value))
                self.weights[key].append(sample_count)

        def get_aggregated_metrics(self):
            result = {}
            for key, values in self.values.items():
                if key in {
                    "training/off_policy/evicted_samples",
                    "training/filter_groups/evicted_samples",
                    "training/filter_groups/discarded_surplus_samples",
                    "training/rollout_failure/evicted_samples",
                }:
                    result[key] = sum(values)
                elif key.endswith("/max"):
                    result[key] = max(values)
                elif key.endswith("/min"):
                    result[key] = min(values)
                else:
                    weights = self.weights[key]
                    result[key] = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
            if {"global_seqlen/min", "global_seqlen/max"}.issubset(result):
                result["global_seqlen/minmax_diff"] = (
                    result["global_seqlen/max"] - result["global_seqlen/min"]
                )
            return result

    utils_mod.MetricsAggregator = MetricsAggregator
    sys.modules["verl.trainer.ppo.v1.utils"] = utils_mod
    trainer_module = sys.modules.get("uni_agent.trainer.multi_agents_ppo_trainer")
    if trainer_module is not None:
        trainer_module.MetricsAggregator = MetricsAggregator


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

        if policy_configs is not None and should_stub_component("_validate_stage_resources"):
            def _validate_stage_resources(self, stage):
                return None

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
        assert not hasattr(MultiAgentsPPOTrainer, "_call_v1_stage")
        assert hasattr(MultiAgentsPPOTrainer, "_sync_policy_runtime_context")

    def test_core_training_dataflow_has_concrete_type_annotations(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        expected_returns = {
            "_resolve_policy_configs": "dict[str, DictConfig]",
            "_create_policy_trainers": "dict[str, PolicyTrainer]",
            "step": "KVBatchMeta",
            "_step_once": "KVBatchMeta",
            "sample_multi_agent_batch": "KVBatchMeta",
            "build_per_policy_batches": "dict[str, KVBatchMeta]",
            "prepare_policy_batches_for_ppo_update": "dict[str, KVBatchMeta]",
            "compute_multi_agent_advantage_from_policy_batches": "KVBatchMeta",
            "compute_multi_agent_advantage": "KVBatchMeta",
            "update_policy_trainers": "dict[str, KVBatchMeta]",
            "_build_replay_buffer": "ReplayBuffer | ReplayBufferAsync",
            "_fetch_one_gen_batch": "TensorDict",
            "_next_train_batch": "TensorDict",
            "_make_batch_like": "KVBatchMeta",
            "_merge_policy_batches": "KVBatchMeta",
            "_collect_placement_groups": "list[PlacementGroup]",
        }
        for method_name, expected_return in expected_returns.items():
            signature = inspect.signature(getattr(MultiAgentsPPOTrainer, method_name))
            assert signature.return_annotation == expected_return

        expected_parameters = {
            ("build_per_policy_batches", "multi_agent_batch"): "KVBatchMeta",
            ("_add_advantage_metrics", "batch"): "KVBatchMeta",
            ("_add_data_metrics", "batch"): "KVBatchMeta",
            ("_submit_batch_to_rollout", "batch"): "TensorDict",
            ("_validation_final_record_keys", "batch"): "KVBatchMeta",
            ("_remove_placement_groups", "pgs"): "Iterable[PlacementGroup]",
        }
        for (method_name, parameter_name), expected_annotation in expected_parameters.items():
            signature = inspect.signature(getattr(MultiAgentsPPOTrainer, method_name))
            assert signature.parameters[parameter_name].annotation == expected_annotation

    def test_fit_uses_the_concrete_agent_framework_rollout_adapter_type(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        signature = inspect.signature(MultiAgentsPPOTrainer.fit)

        assert "agent_loop_manager" not in MultiAgentsPPOTrainer.__annotations__
        assert signature.parameters["agent_loop_manager"].annotation == "AgentFrameworkRolloutAdapter"
        assert signature.return_annotation == "None"

        source_path = Path(__file__).parents[3] / "uni_agent" / "trainer" / "framework" / "entry.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        adapter_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentFrameworkRolloutAdapter"
        )
        generate_sequences = next(
            node
            for node in adapter_class.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_sequences"
        )
        prompts = next(argument for argument in generate_sequences.args.args if argument.arg == "prompts")
        assert ast.unparse(prompts.annotation) == "TensorDict"
        assert ast.unparse(generate_sequences.returns) == "None"

    def test_joint_advantage_explicitly_selects_one_designated_policy_trainer(self):
        source_path = Path(__file__).parents[3] / "uni_agent" / "trainer" / "multi_agents_ppo_trainer.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MultiAgentsPPOTrainer"
        )
        method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_multi_agent_advantage"
        )

        assert not any(isinstance(node, ast.For) for node in ast.walk(method))
        assert any(
            isinstance(node, ast.Name) and node.id == "advantage_trainer"
            for node in ast.walk(method)
        )

    def test_stable_dependencies_are_imported_at_module_level(self):
        source_path = Path(__file__).parents[3] / "uni_agent" / "trainer" / "multi_agents_ppo_trainer.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        allowed_inline_modules = {
            "uni_agent.trainer.single_async_ppo_trainer",
            "uni_agent.trainer.single_ppo_trainer",
        }
        unexpected_inline_imports = [
            f"{module}:{node.lineno}"
            for scope in ast.walk(tree)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(scope)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for module in ([node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names])
            if module not in allowed_inline_modules
        ]

        assert unexpected_inline_imports == []

    def test_data_metric_dependencies_are_imported_at_module_level(self):
        source_path = Path(__file__).parents[3] / "uni_agent" / "trainer" / "multi_agents_ppo_trainer.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        assert {"DataProto", "compute_data_metrics"} <= imported_names

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
                config=SimpleNamespace(
                    data=SimpleNamespace(train_batch_size=4),
                    trainer=SimpleNamespace(
                        v1=SimpleNamespace(
                            trainer_mode="colocate_async",
                            colocate_async=SimpleNamespace(parameter_sync_step=1),
                        )
                    ),
                ),
                policy_configs={
                    "policy_1": _policy_config(
                        "policy_1",
                        trainer=_NS(v1=_NS(trainer_mode="sync")),
                    )
                },
            )

    def test_allows_separate_async_parameter_sync_step_greater_than_one(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        async_v1 = _NS(trainer_mode="separate_async", separate_async=_NS(parameter_sync_step=2))
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                data=SimpleNamespace(train_batch_size=4),
                trainer=SimpleNamespace(v1=async_v1),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1", trainer=_NS(v1=async_v1)),
                "policy_2": _policy_config("policy_2", trainer=_NS(v1=async_v1)),
            },
        )

        assert trainer.parameter_sync_step == 2
        assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]

    def test_outer_runtime_mode_selects_policy_trainer_class(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class SyncTrainer(FakeV1PPOTrainer):
            pass

        class AsyncTrainer(FakeV1PPOTrainer):
            pass

        config = SimpleNamespace(
            data=SimpleNamespace(train_batch_size=4),
            trainer=SimpleNamespace(
                v1=SimpleNamespace(
                    trainer_mode="separate_async",
                    separate_async=SimpleNamespace(parameter_sync_step=2),
                )
            ),
            policies={
                "policy_1": _NS(
                    ppo_trainer_config_name="ppo_trainer",
                    ppo_trainer_overrides={},
                )
            },
        )

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            yield

        def fake_compose(*, config_name):
            return OmegaConf.create({"trainer": {"v1": {}}, "data": {}})

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(trainer_module, "compose", fake_compose)

        _install_policy_trainer_stubs(SyncTrainer)
        import uni_agent.trainer.single_async_ppo_trainer as async_module

        async_module.SingleAsyncPPOTrainer = AsyncTrainer
        trainer_cls = _test_multi_agents_trainer_cls(MultiAgentsPPOTrainer, policy_configs=None)
        trainer = trainer_cls(config=_config_with_policies(config))

        assert isinstance(trainer.policy_trainers["policy_1"], AsyncTrainer)

    def test_outer_runtime_values_are_injected_into_policy_config(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = SimpleNamespace(
            data=SimpleNamespace(train_batch_size=4),
            trainer=SimpleNamespace(
                v1=SimpleNamespace(
                    trainer_mode="separate_async",
                    separate_async=SimpleNamespace(parameter_sync_step=2),
                )
            ),
            policies={
                "policy_1": _NS(
                    ppo_trainer_config_name="ppo_trainer",
                    ppo_trainer_overrides={},
                )
            },
        )

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            yield

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(
            trainer_module,
            "compose",
            lambda *, config_name: OmegaConf.create({"trainer": {"v1": {}}, "data": {}}),
        )

        trainer = _make_trainer(MultiAgentsPPOTrainer, config=config)
        policy_config = trainer.policy_configs["policy_1"]

        assert policy_config.trainer.v1.trainer_mode == "separate_async"
        assert policy_config.trainer.v1.separate_async.parameter_sync_step == 2

    def test_outer_algorithm_and_reward_override_policy_values(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = OmegaConf.create(
            {
                "data": {"train_batch_size": 4},
                "trainer": {
                    "v1": {
                        "trainer_mode": "sync",
                        "sync": {
                            "parameter_sync_step": 1,
                            "num_warmup_batches": 3,
                        },
                    }
                },
                "algorithm": {
                    "adv_estimator": "grpo",
                    "gamma": 1.0,
                    "filter_groups": None,
                },
                "reward": {
                    "num_workers": 8,
                    "custom_reward_function": {
                        "path": "pkg://outer.reward",
                        "name": "compute_score",
                    },
                },
                "policies": {
                    "policy_1": {
                        "ppo_trainer_config_name": "ppo_trainer",
                        "ppo_trainer_overrides": {
                            "algorithm": {"adv_estimator": "wrong", "gamma": 0.1},
                            "reward": {
                                "num_workers": 1,
                                "custom_reward_function": {
                                    "path": "pkg://policy.reward",
                                    "name": "wrong",
                                },
                            },
                        },
                    }
                },
            }
        )

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            yield

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(
            trainer_module,
            "compose",
            lambda *, config_name: OmegaConf.create(
                {
                    "trainer": {"v1": {}},
                    "data": {},
                    "algorithm": {"adv_estimator": "default"},
                    "reward": {"num_workers": 2},
                }
            ),
        )

        trainer = _make_trainer(MultiAgentsPPOTrainer, config=config)
        policy_config = trainer.policy_configs["policy_1"]

        assert policy_config.algorithm.adv_estimator == "grpo"
        assert policy_config.algorithm.gamma == 1.0
        assert policy_config.algorithm.filter_groups is None
        assert policy_config.reward.num_workers == 8
        assert policy_config.reward.custom_reward_function.path == "pkg://outer.reward"
        assert policy_config.reward.custom_reward_function.name == "compute_score"
        assert policy_config.trainer.v1.trainer_mode == "sync"
        assert policy_config.trainer.v1.sync.parameter_sync_step == 1
        assert policy_config.trainer.v1.sync.num_warmup_batches == 3

    def test_outer_runtime_values_override_legacy_policy_values(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = SimpleNamespace(
            data=SimpleNamespace(train_batch_size=4),
            trainer=SimpleNamespace(
                v1=SimpleNamespace(
                    trainer_mode="separate_async",
                    separate_async=SimpleNamespace(parameter_sync_step=2),
                )
            ),
            policies={
                "policy_1": _NS(
                    ppo_trainer_config_name="ppo_trainer",
                    ppo_trainer_overrides={
                        "trainer": {
                            "v1": {
                                "trainer_mode": "sync",
                                "separate_async": {"parameter_sync_step": 99},
                            }
                        }
                    },
                )
            },
        )

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            yield

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(
            trainer_module,
            "compose",
            lambda *, config_name: OmegaConf.create(
                {
                    "trainer": {
                        "v1": {
                            "trainer_mode": "sync",
                            "separate_async": {"parameter_sync_step": 1},
                        }
                    },
                    "data": {},
                }
            ),
        )

        trainer = _make_trainer(MultiAgentsPPOTrainer, config=config)
        policy_config = trainer.policy_configs["policy_1"]

        assert policy_config.trainer.v1.trainer_mode == "separate_async"
        assert policy_config.trainer.v1.separate_async.parameter_sync_step == 2

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
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
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
                    "data": {"train_files": ["base.parquet"]},
                    "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}},
                }
            )

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(trainer_module, "compose", fake_compose)
        config = SimpleNamespace(
            ppo_trainer_config_source="verl.trainer.config",
            ppo_trainer_config_name="ppo_trainer",
            data=_NS(train_batch_size=4, train_files=["outer.parquet"]),
            policies={
                "policy_1": _NS(
                    ppo_trainer_overrides={
                        "actor_rollout_ref": {"model": {"path": "/models/policy_1"}},
                        "data": {"train_files": ["policy.parquet"]},
                    },
                ),
                "policy_2": _NS(
                    ppo_trainer_config_source="custom.policy.config",
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
        assert trainer.policy_configs["policy_1"].data.train_files == ["policy.parquet"]

    def test_root_ppo_trainer_config_name_is_shared_by_all_policies(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        compose_calls = []

        @contextmanager
        def fake_initialize_config_module(*, config_module, version_base):
            yield

        def fake_compose(*, config_name):
            compose_calls.append(config_name)
            return OmegaConf.create({"trainer": {"v1": {}}, "data": {}})

        monkeypatch.setattr(trainer_module, "initialize_config_module", fake_initialize_config_module)
        monkeypatch.setattr(trainer_module, "compose", fake_compose)
        config = SimpleNamespace(
            ppo_trainer_config_source="verl.trainer.config",
            ppo_trainer_config_name="ppo_trainer",
            policies={
                "default_policy": _NS(ppo_trainer_overrides={}),
                "override_policy": _NS(
                    ppo_trainer_config_name="ppo_megatron_trainer",
                    ppo_trainer_overrides={},
                ),
            },
        )

        _make_trainer(MultiAgentsPPOTrainer, config=config)

        assert compose_calls == ["ppo_trainer", "ppo_trainer"]

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

    def test_rejects_role_mapping_to_unknown_policy_before_creating_policy_trainers(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "custom": {
                            "agent_framework": {
                                "role_policy_mapping": {
                                    "agent_1": "policy_1",
                                    "agent_2": "policy_3",
                                }
                            }
                        }
                    }
                }
            }
        )

        with pytest.raises(
            ValueError,
            match=r"unknown policies.*policy_3.*Configured policies.*policy_1.*policy_2",
        ):
            _make_trainer(
                MultiAgentsPPOTrainer,
                config=config,
                policy_configs={
                    "policy_1": _policy_config("policy_1"),
                    "policy_2": _policy_config("policy_2"),
                },
            )

        assert FakeV1PPOTrainer.instances == []

    def test_allows_multiple_roles_to_share_one_configured_policy(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "custom": {
                            "agent_framework": {
                                "role_policy_mapping": {
                                    "agent_1": "policy_1",
                                    "agent_2": "policy_1",
                                    "agent_3": "policy_2",
                                }
                            }
                        }
                    }
                }
            }
        )

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )

        assert trainer.role_policy_mapping == {
            "agent_1": "policy_1",
            "agent_2": "policy_1",
            "agent_3": "policy_2",
        }

    def test_warns_when_configured_policy_is_not_mapped(self, caplog):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "custom": {
                            "agent_framework": {
                                "role_policy_mapping": {"agent_1": "policy_1"}
                            }
                        }
                    }
                }
            }
        )

        with caplog.at_level("WARNING"):
            _make_trainer(
                MultiAgentsPPOTrainer,
                config=config,
                policy_configs={
                    "policy_1": _policy_config("policy_1"),
                    "policy_2": _policy_config("policy_2"),
                },
            )

        assert "Configured policies are not referenced by role_policy_mapping" in caplog.text
        assert "policy_2" in caplog.text

    def test_init_reserves_all_training_runtimes_before_starting_standalone_rollouts(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingInitTrainer(MultiAgentsPPOTrainer):
            def _validate_stage_resources(self, stage):
                self.init_events.append(f"validate_resources:{stage}")

            def _build_dataloader(self):
                self.init_events.append("build_dataloader")

            def _build_replay_buffer(self):
                self.init_events.append("build_replay_buffer")

            def _load_checkpoint(self):
                self.init_events.append("load_checkpoint")

        trainer = _make_trainer(
            RecordingInitTrainer,
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    v1=SimpleNamespace(
                        trainer_mode="separate_async",
                        separate_async=SimpleNamespace(parameter_sync_step=1),
                    )
                )
            ),
            policy_configs={
                "policy_1": _policy_config_with_rollout(
                    "policy_1", total_gpus=2, tensor_parallel_size=2
                ),
                "policy_2": _policy_config_with_rollout(
                    "policy_2", total_gpus=2, tensor_parallel_size=2
                ),
            },
        )
        trainer.init_events = []
        event_lock = Lock()
        stage_barriers = {
            "training": Barrier(2, timeout=5),
            "rollout": Barrier(2, timeout=5),
            "on_init_end": Barrier(2, timeout=5),
        }

        def run_policy_stage(policy_name, stage):
            with event_lock:
                trainer.init_events.append(f"{stage}:enter:{policy_name}")
            stage_barriers[stage].wait()
            with event_lock:
                trainer.init_events.append(f"{stage}:exit:{policy_name}")

        for policy_name, policy_trainer in trainer.policy_trainers.items():
            policy_trainer.init_training_runtime = (
                lambda name=policy_name: run_policy_stage(name, "training")
            )
            policy_trainer.init_standalone_rollout_runtime = (
                lambda name=policy_name: run_policy_stage(name, "rollout")
            )
            policy_trainer.on_init_end = lambda name=policy_name: run_policy_stage(name, "on_init_end")

        trainer.init()

        def event_index(event):
            return trainer.init_events.index(event)

        for stage in ("training", "rollout", "on_init_end"):
            enter_indices = [
                event_index(f"{stage}:enter:{policy_name}") for policy_name in trainer.policy_trainers
            ]
            exit_indices = [
                event_index(f"{stage}:exit:{policy_name}") for policy_name in trainer.policy_trainers
            ]
            assert max(enter_indices) < min(exit_indices)

        assert event_index("validate_resources:training") < min(
            event_index(f"training:enter:{name}") for name in trainer.policy_trainers
        )
        assert max(event_index(f"training:exit:{name}") for name in trainer.policy_trainers) < event_index(
            "build_dataloader"
        )
        assert event_index("load_checkpoint") < min(
            event_index(f"rollout:enter:{name}") for name in trainer.policy_trainers
        )
        assert event_index("load_checkpoint") < event_index("validate_resources:rollout")
        assert event_index("validate_resources:rollout") < min(
            event_index(f"rollout:enter:{name}") for name in trainer.policy_trainers
        )
        assert max(event_index(f"rollout:exit:{name}") for name in trainer.policy_trainers) < min(
            event_index(f"on_init_end:enter:{name}") for name in trainer.policy_trainers
        )
        assert [policy_trainer.fit_calls for policy_trainer in trainer.policy_trainers.values()] == [0, 0]

    def test_rollout_init_waves_group_policies_by_replica_node_footprint(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = self._resource_preflight_trainer(
            {
                "policy_1": _plain(
                    _policy_config_with_rollout(
                        "policy_1", total_gpus=8, tensor_parallel_size=4
                    )
                ),
                "policy_2": _plain(
                    _policy_config_with_rollout(
                        "policy_2", total_gpus=4, tensor_parallel_size=2
                    )
                ),
                "policy_3": _plain(
                    _policy_config_with_rollout(
                        "policy_3", total_gpus=4, tensor_parallel_size=2
                    )
                ),
            }
        )
        trainer.policy_trainers = {
            policy_name: SimpleNamespace()
            for policy_name in trainer.policy_configs
        }

        assert trainer._get_rollout_init_waves() == [
            (4, ["policy_1"]),
            (2, ["policy_2", "policy_3"]),
        ]

    def test_init_completes_larger_rollout_wave_before_starting_smaller_policies(self):
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
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    v1=SimpleNamespace(
                        trainer_mode="separate_async",
                        separate_async=SimpleNamespace(parameter_sync_step=1),
                    )
                )
            ),
            policy_configs={
                "policy_1": _policy_config_with_rollout(
                    "policy_1", total_gpus=8, tensor_parallel_size=4
                ),
                "policy_2": _policy_config_with_rollout(
                    "policy_2", total_gpus=4, tensor_parallel_size=2
                ),
                "policy_3": _policy_config_with_rollout(
                    "policy_3", total_gpus=4, tensor_parallel_size=2
                ),
            },
        )
        events = []
        event_lock = Lock()
        large_started = Event()
        release_large = Event()
        small_started = Event()
        small_barrier = Barrier(2, timeout=5)

        def record(event):
            with event_lock:
                events.append(event)

        def init_large_rollout():
            record("rollout:enter:policy_1")
            large_started.set()
            assert release_large.wait(timeout=5)
            record("rollout:exit:policy_1")

        def init_small_rollout(policy_name):
            record(f"rollout:enter:{policy_name}")
            small_started.set()
            small_barrier.wait()
            record(f"rollout:exit:{policy_name}")

        for policy_trainer in trainer.policy_trainers.values():
            policy_trainer.init_training_runtime = lambda: None
            policy_trainer.on_init_end = lambda: None
        trainer.policy_trainers["policy_1"].init_standalone_rollout_runtime = init_large_rollout
        trainer.policy_trainers["policy_2"].init_standalone_rollout_runtime = (
            lambda: init_small_rollout("policy_2")
        )
        trainer.policy_trainers["policy_3"].init_standalone_rollout_runtime = (
            lambda: init_small_rollout("policy_3")
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            init_future = executor.submit(trainer.init)
            assert large_started.wait(timeout=5)
            try:
                assert not small_started.wait(timeout=0.2)
            finally:
                release_large.set()
                init_future.result(timeout=5)

        assert events.index("rollout:exit:policy_1") < events.index("rollout:enter:policy_2")
        assert events.index("rollout:exit:policy_1") < events.index("rollout:enter:policy_3")
        assert max(
            events.index("rollout:enter:policy_2"),
            events.index("rollout:enter:policy_3"),
        ) < min(
            events.index("rollout:exit:policy_2"),
            events.index("rollout:exit:policy_3"),
        )

    def test_rollout_init_does_not_start_smaller_wave_after_larger_wave_failure(self):
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
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    v1=SimpleNamespace(
                        trainer_mode="separate_async",
                        separate_async=SimpleNamespace(parameter_sync_step=1),
                    )
                )
            ),
            policy_configs={
                "policy_1": _policy_config_with_rollout(
                    "policy_1", total_gpus=8, tensor_parallel_size=4
                ),
                "policy_2": _policy_config_with_rollout(
                    "policy_2", total_gpus=4, tensor_parallel_size=2
                ),
                "policy_3": _policy_config_with_rollout(
                    "policy_3", total_gpus=4, tensor_parallel_size=2
                ),
            },
        )
        smaller_wave_calls = []

        for policy_trainer in trainer.policy_trainers.values():
            policy_trainer.init_training_runtime = lambda: None
            policy_trainer.on_init_end = lambda: None
        trainer.policy_trainers["policy_1"].init_standalone_rollout_runtime = (
            lambda: (_ for _ in ()).throw(RuntimeError("policy_1 rollout init failed"))
        )
        trainer.policy_trainers["policy_2"].init_standalone_rollout_runtime = (
            lambda: smaller_wave_calls.append("policy_2")
        )
        trainer.policy_trainers["policy_3"].init_standalone_rollout_runtime = (
            lambda: smaller_wave_calls.append("policy_3")
        )

        with pytest.raises(RuntimeError, match="policy_1 rollout init failed"):
            trainer.init()

        assert smaller_wave_calls == []

    def test_init_training_runtime_does_not_use_legacy_init_workers(self):
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
        assert policy_trainer.training_runtime_init_calls == 1
        assert policy_trainer.init_calls == 0
        assert policy_trainer.on_init_end_calls == 1
        assert policy_trainer.init_workers_calls == 0

    def test_init_waits_for_the_current_policy_stage_before_propagating_failure(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingInitTrainer(MultiAgentsPPOTrainer):
            def _build_dataloader(self):
                raise AssertionError("outer initialization must not start after a policy failure")

        trainer = _make_trainer(
            RecordingInitTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
        )
        slow_policy_started = Event()
        release_slow_policy = Event()
        slow_policy_finished = Event()

        def fail_policy_init():
            raise RuntimeError("policy_1 init failed")

        def slow_policy_init():
            slow_policy_started.set()
            release_slow_policy.wait(timeout=5)
            slow_policy_finished.set()

        trainer.policy_trainers["policy_1"].init_training_runtime = fail_policy_init
        trainer.policy_trainers["policy_2"].init_training_runtime = slow_policy_init

        with ThreadPoolExecutor(max_workers=1) as caller_pool:
            init_future = caller_pool.submit(trainer.init)
            assert slow_policy_started.wait(timeout=5)
            assert not init_future.done()
            release_slow_policy.set()
            with pytest.raises(RuntimeError, match="policy_1 init failed"):
                init_future.result(timeout=5)

        assert slow_policy_finished.is_set()

    def test_consume_sync_metrics_delegates_to_each_policy_trainer(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        calls = []
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                _consume_sync_metrics=lambda: calls.append("policy_1") or {"bytes": 1}
            ),
            "policy_2": SimpleNamespace(
                _consume_sync_metrics=lambda: calls.append("policy_2") or {}
            ),
        }

        assert trainer._consume_sync_metrics() == {"policy_1/sync/bytes": 1}
        assert calls == ["policy_1", "policy_2"]

    @staticmethod
    def _resource_preflight_trainer(policy_configs):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_configs = {
            policy_name: OmegaConf.create(policy_config)
            for policy_name, policy_config in policy_configs.items()
        }
        return trainer

    @staticmethod
    def _set_available_ray_resources(monkeypatch, trainer_module, resources):
        monkeypatch.setattr(
            trainer_module.ray,
            "_private",
            SimpleNamespace(
                state=SimpleNamespace(
                    available_resources_per_node=lambda: resources,
                )
            ),
            raising=False,
        )

    def test_training_resource_preflight_rejects_aggregate_gpu_shortage(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module

        trainer = self._resource_preflight_trainer(
            {
                "policy_1": {"trainer": {"nnodes": 1, "n_gpus_per_node": 8}},
                "policy_2": {"trainer": {"nnodes": 1, "n_gpus_per_node": 8}},
            }
        )
        self._set_available_ray_resources(
            monkeypatch,
            trainer_module,
            {
                "node_1": {"GPU": 8},
                "node_2": {"GPU": 4},
            },
        )

        with pytest.raises(ValueError, match="training.*required 16.*available 12"):
            trainer._validate_stage_resources("training")

    def test_training_resource_preflight_rejects_unplaceable_strict_pack_topology(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module

        trainer = self._resource_preflight_trainer(
            {
                "policy_1": {"trainer": {"nnodes": 1, "n_gpus_per_node": 8}},
                "policy_2": {"trainer": {"nnodes": 1, "n_gpus_per_node": 8}},
            }
        )
        self._set_available_ray_resources(
            monkeypatch,
            trainer_module,
            {
                "node_1": {"GPU": 4},
                "node_2": {"GPU": 4},
                "node_3": {"GPU": 4},
                "node_4": {"GPU": 4},
            },
        )

        with pytest.raises(ValueError, match="training.*STRICT_PACK"):
            trainer._validate_stage_resources("training")

    def test_training_resource_preflight_includes_reward_and_teacher_pools(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module

        trainer = self._resource_preflight_trainer(
            {
                "policy_1": {
                    "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
                    "reward": {
                        "reward_model": {
                            "enable_resource_pool": True,
                            "nnodes": 1,
                            "n_gpus_per_node": 2,
                        }
                    },
                    "distillation": {
                        "enabled": True,
                        "nnodes": 1,
                        "n_gpus_per_node": 2,
                    },
                }
            }
        )
        self._set_available_ray_resources(
            monkeypatch,
            trainer_module,
            {"node_1": {"GPU": 4}},
        )

        with pytest.raises(ValueError, match="required 6.*reward_pool.*teacher_pool"):
            trainer._validate_stage_resources("training")

    def test_rollout_resource_preflight_uses_replica_tp_placement_shape(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module

        trainer = self._resource_preflight_trainer(
            {
                "policy_1": {
                    "actor_rollout_ref": {
                        "rollout": {
                            "nnodes": 1,
                            "n_gpus_per_node": 8,
                            "tensor_model_parallel_size": 2,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                        }
                    }
                }
            }
        )
        self._set_available_ray_resources(
            monkeypatch,
            trainer_module,
            {
                "node_1": {"GPU": 4},
                "node_2": {"GPU": 4},
            },
        )

        trainer._validate_stage_resources("rollout")

    def test_rollout_resource_preflight_rejects_multi_policy_aggregate_shortage(self, monkeypatch):
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module

        rollout_config = {
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 8,
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                }
            }
        }
        trainer = self._resource_preflight_trainer(
            {
                "policy_1": rollout_config,
                "policy_2": rollout_config,
            }
        )
        self._set_available_ray_resources(
            monkeypatch,
            trainer_module,
            {
                "node_1": {"GPU": 8},
                "node_2": {"GPU": 4},
            },
        )

        with pytest.raises(ValueError, match="rollout.*required 16.*available 12"):
            trainer._validate_stage_resources("rollout")

    def test_collect_placement_groups_includes_standalone_rollout_replicas(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class FakeResourcePool:
            def __init__(self, *placement_groups):
                self.pgs = list(placement_groups)

            def get_placement_groups(self):
                raise AssertionError("cleanup must not lazily create placement groups")

        trainer_pg = SimpleNamespace(id="trainer-pg")
        standalone_pg_0 = SimpleNamespace(id="standalone-pg-0")
        standalone_pg_1 = SimpleNamespace(id="standalone-pg-1")
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_trainers = {
            "policy_1": SimpleNamespace(
                resource_pool_manager=SimpleNamespace(
                    resource_pool_dict={"global": FakeResourcePool(trainer_pg)}
                ),
                standalone_server_manager=SimpleNamespace(
                    rollout_replicas=[
                        SimpleNamespace(resource_pool=FakeResourcePool(standalone_pg_0)),
                        SimpleNamespace(resource_pool=FakeResourcePool(standalone_pg_1)),
                    ]
                ),
            )
        }

        assert trainer._collect_placement_groups() == [
            trainer_pg,
            standalone_pg_0,
            standalone_pg_1,
        ]

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

    def test_step_keeps_caller_timing_raw_as_trainer_timing_context(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class TimingRecordingTrainer(MultiAgentsPPOTrainer):
            def _add_batch_to_generate(self):
                return None

            def _step_once(self, metrics, timing_raw, sample_batch_size):
                del metrics, sample_batch_size
                assert timing_raw is self.timing_raw
                timing_raw["custom_stage"] = 1.0
                return SimpleKVBatchMeta(
                    keys=["uid_0"],
                    tags=[{"policy_name": "policy_1"}],
                )

        trainer = _make_trainer(
            TimingRecordingTrainer,
            policy_configs={"policy_1": _policy_config("policy_1")},
        )
        timing_raw = {}

        trainer.step(metrics={}, timing_raw=timing_raw)

        assert timing_raw == {"custom_stage": 1.0}
        assert trainer.timing_raw is timing_raw

    def test_throughput_gpu_count_requires_policy_trainer_interface(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_trainers = {"policy_1": object()}

        with pytest.raises(AttributeError, match="_get_n_gpus_for_throughput"):
            trainer._get_n_gpus_for_throughput()

    def test_decoupled_step_tracks_local_trigger_step_per_policy(self):
        _install_dependency_stubs()
        _install_metrics_aggregator_stub()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class LocalTriggerRecordingTrainer(FakeV1PPOTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.local_trigger_steps = []

            def on_sample_end(self):
                return None

            def _compute_old_log_prob(self, batch, metrics=None):
                self.local_trigger_steps.append(self.local_trigger_step)
                return super()._compute_old_log_prob(batch, metrics)

        async_v1 = _NS(trainer_mode="separate_async", separate_async=_NS(parameter_sync_step=2))
        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            config=SimpleNamespace(
                data=SimpleNamespace(train_batch_size=4),
                trainer=SimpleNamespace(critic_warmup=0, v1=async_v1),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1", trainer=_NS(v1=async_v1)),
                "policy_2": _policy_config("policy_2", trainer=_NS(v1=async_v1)),
            },
            policy_trainer_cls=LocalTriggerRecordingTrainer,
        )
        batches = iter(
            [
                SimpleKVBatchMeta(
                    keys=["p1_0", "p2_0"],
                    tags=[{"policy_name": "policy_1"}, {"policy_name": "policy_2"}],
                ),
                SimpleKVBatchMeta(
                    keys=["p1_1"],
                    tags=[{"policy_name": "policy_1"}],
                ),
            ]
        )
        trainer.replay_buffer = SimpleNamespace(sample=lambda **_: next(batches))
        trainer._add_batch_to_generate = lambda: None

        assert not hasattr(trainer, "_policy_local_trigger_steps")
        assert not hasattr(trainer, "_last_step_policy_names")

        trainer.step(metrics={}, timing_raw={})

        assert trainer.policy_trainers["policy_1"].local_trigger_steps == [0, 1]
        assert trainer.policy_trainers["policy_2"].local_trigger_steps == [0]
        assert not hasattr(trainer, "_policy_local_trigger_steps")
        assert not hasattr(trainer, "_last_step_policy_names")

    def test_decoupled_step_aggregates_metrics_globally_and_per_policy(self):
        _install_metrics_aggregator_stub()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class MetricRecordingTrainer(MultiAgentsPPOTrainer):
            def __init__(self, **kwargs):
                self.iteration = 0
                super().__init__(**kwargs)

            def _add_batch_to_generate(self):
                return None

            def _step_once(self, metrics, timing_raw, sample_batch_size):
                del timing_raw, sample_batch_size
                self.iteration += 1
                if self.iteration == 1:
                    metrics.update(
                        {
                            "training/off_policy/evicted_samples": 2,
                            "training/off_policy/trajectory_staleness/mean": 1.0,
                            "policy_1/actor/loss/mean": 2.0,
                            "policy_1/global_seqlen/min": 10.0,
                            "policy_1/global_seqlen/max": 20.0,
                            "policy_1/global_seqlen/minmax_diff": 10.0,
                            "policy_2/actor/loss/mean": 5.0,
                        }
                    )
                    return SimpleKVBatchMeta(
                        keys=["p1_0", "p1_1", "p2_0"],
                        tags=[
                            {"policy_name": "policy_1"},
                            {"policy_name": "policy_1"},
                            {"policy_name": "policy_2"},
                        ],
                    )
                metrics.update(
                    {
                        "training/off_policy/evicted_samples": 3,
                        "training/off_policy/trajectory_staleness/mean": 4.0,
                        "policy_1/actor/loss/mean": 6.0,
                        "policy_1/global_seqlen/min": 0.0,
                        "policy_1/global_seqlen/max": 15.0,
                        "policy_1/global_seqlen/minmax_diff": 15.0,
                    }
                )
                return SimpleKVBatchMeta(
                    keys=["p1_2"],
                    tags=[{"policy_name": "policy_1"}],
                )

        async_v1 = _NS(trainer_mode="separate_async", separate_async=_NS(parameter_sync_step=2))
        trainer = _make_trainer(
            MetricRecordingTrainer,
            config=SimpleNamespace(
                data=SimpleNamespace(train_batch_size=4),
                trainer=SimpleNamespace(v1=async_v1),
            ),
            policy_configs={
                "policy_1": _policy_config("policy_1", trainer=_NS(v1=async_v1)),
                "policy_2": _policy_config("policy_2", trainer=_NS(v1=async_v1)),
            },
        )
        metrics = {}

        batch = trainer.step(metrics=metrics, timing_raw={})

        assert batch.keys == ["p1_0", "p1_1", "p2_0", "p1_2"]
        assert metrics["training/off_policy/evicted_samples"] == 5
        assert metrics["training/off_policy/trajectory_staleness/mean"] == pytest.approx(1.75)
        assert metrics["policy_1/actor/loss/mean"] == pytest.approx(10 / 3)
        assert metrics["policy_2/actor/loss/mean"] == pytest.approx(5.0)
        assert metrics["policy_1/global_seqlen/min"] == 0
        assert metrics["policy_1/global_seqlen/max"] == 20
        assert metrics["policy_1/global_seqlen/minmax_diff"] == 20

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
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert trainer.agent_loop_manager is not None
        assert [
            policy_trainer.training_runtime_init_calls
            for policy_trainer in trainer.policy_trainers.values()
        ] == [1, 1]
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
        trainer.train_dataloader = FakeDataloader([])
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
        trainer.train_dataloader = FakeDataloader([])
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
        trainer.config = _NS(
            trainer=_NS(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_dir),
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
                del_local_ckpt_after_load=False,
            )
        )
        trainer.trainer_mode = "separate_async"
        trainer.policy_configs = {}
        trainer.policy_trainers = {}
        trainer.train_dataloader = FakeDataloader([])
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

    def test_validation_selects_one_final_record_per_mas_rollout(self):
        _install_dependency_stubs()
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        batch = SimpleKVBatchMeta(
            partition_id="val",
            keys=["uid_a_0_0", "uid_a_0_1", "uid_a_1_0"],
            tags=[
                {"uid": "uid_a", "sample_idx": 0, "record_idx": 0},
                {"uid": "uid_a", "sample_idx": 0, "record_idx": 1},
                {"uid": "uid_a", "sample_idx": 1, "record_idx": 0},
            ],
        )

        assert MultiAgentsPPOTrainer._validation_final_record_keys(batch) == [
            "uid_a_0_1",
            "uid_a_1_0",
        ]

    def test_fit_runs_outer_initial_validation_and_supports_val_only(self):
        _install_dependency_stubs()
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(total_training_steps=2, val_before_train=True, val_only=True),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()
        trainer.step_events = []
        validation_hook_states = []

        def validate():
            validation_hook_states.append(
                {
                    policy_name: (
                        policy_trainer.on_validate_begin_calls,
                        policy_trainer.on_validate_end_calls,
                    )
                    for policy_name, policy_trainer in trainer.policy_trainers.items()
                }
            )
            return {"val-core/test/reward/mean@1": 1.0}

        trainer._validate = validate

        trainer.fit(FakeAgentFrameworkRolloutAdapter.create())

        assert trainer.global_steps == 0
        assert trainer.step_events == []
        assert validation_hook_states == [
            {
                "policy_1": (1, 0),
                "policy_2": (1, 0),
            }
        ]
        assert all(
            policy_trainer.on_validate_end_calls == 1
            for policy_trainer in trainer.policy_trainers.values()
        )

    def test_outer_validation_uses_val_partition_and_final_mas_records(self, monkeypatch):
        _install_dependency_stubs()
        import torch
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={"policy_1": _policy_config("policy_1")},
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()
        trainer.global_steps = 5
        trainer.val_dataloader = FakeDataloader(
            [{"raw_prompt": [[{"role": "user", "content": "question"}]]}]
        )
        trainer.agent_loop_manager = FakeAgentFrameworkRolloutAdapter()
        val_batch = SimpleKVBatchMeta(
            partition_id="val",
            keys=["uid_0_0", "uid_0_1", "uid_1_0"],
            tags=[
                {"uid": "uid", "sample_idx": 0, "record_idx": 0},
                {"uid": "uid", "sample_idx": 0, "record_idx": 1},
                {"uid": "uid", "sample_idx": 1, "record_idx": 0},
            ],
        )
        trainer.replay_buffer.next_sample = (val_batch, {})
        calls = {"put": [], "get": [], "clear": []}
        monkeypatch.setattr(
            trainer_module.tq, "kv_batch_put", lambda **kwargs: calls["put"].append(kwargs)
        )

        def get_fields(**kwargs):
            calls["get"].append(kwargs)
            return {
                "uid": ["uid", "uid"],
                "rm_scores": torch.nested.as_nested_tensor(
                    [torch.tensor([0.0, 2.0]), torch.tensor([3.0])],
                    layout=torch.jagged,
                ),
                "num_turns": [2, 4],
                "data_source": ["math", "math"],
                "extra_fields": [{}, {}],
            }

        monkeypatch.setattr(trainer_module.tq, "kv_batch_get", get_fields, raising=False)
        monkeypatch.setattr(
            trainer_module.tq, "kv_clear", lambda **kwargs: calls["clear"].append(kwargs)
        )

        metrics = trainer._validate()

        assert calls["put"][0]["partition_id"] == "val"
        assert calls["get"][0]["keys"] == ["uid_0_1", "uid_1_0"]
        assert calls["clear"] == [{"keys": val_batch.keys, "partition_id": "val"}]
        assert trainer.replay_buffer.sample_calls == [
            {"global_steps": 5, "partition_id": "val", "batch_size": 1}
        ]
        generated = trainer.agent_loop_manager.generated_prompts[0]
        assert _td_get(generated, "validate") is True
        assert _td_get(generated, "global_steps") == 5
        assert metrics["validation/reward/mean"] == pytest.approx(2.5)
        assert metrics["val-aux/num_turns/mean"] == pytest.approx(3.0)
        assert trainer.policy_trainers["policy_1"].checkpoint_manager.update_weight_steps == []
        assert trainer.policy_trainers["policy_1"].checkpoint_manager.sleep_calls == 0

    def test_fit_validates_at_test_frequency_and_last_step(self):
        _install_dependency_stubs()
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            pass

        config = SimpleNamespace(
            transfer_queue=SimpleNamespace(enable=False),
            trainer=SimpleNamespace(
                total_training_steps=3,
                val_before_train=False,
                test_freq=2,
            ),
        )
        trainer = _make_trainer(
            RecordingTrainer,
            config=config,
            policy_configs={"policy_1": _policy_config("policy_1")},
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.init()
        trainer.step_events = []
        validation_steps = []

        def validate():
            validation_steps.append(trainer.global_steps)
            return {"validation/reward/mean": float(trainer.global_steps)}

        trainer._validate = validate
        trainer.fit(FakeAgentFrameworkRolloutAdapter.create())

        assert validation_steps == [2, 3]
        assert trainer.policy_trainers["policy_1"].on_validate_begin_calls == 2
        assert trainer.policy_trainers["policy_1"].on_validate_end_calls == 2

    def test_multi_policy_async_checkpoint_save_does_not_publish_outer_tracker(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = _NS(
            trainer=_NS(
                default_local_dir=str(checkpoint_root),
                default_hdfs_dir=None,
            )
        )
        trainer.policy_configs = {
            "policy_1": OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "actor": {"checkpoint": {"async_save": True}},
                    }
                }
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
        trainer.train_dataloader = FakeDataloader([])

        assert trainer._has_async_checkpoint_save() is True
        trainer._save_checkpoint()

        assert not (checkpoint_root / "latest_checkpointed_iteration.txt").exists()

    def test_multi_policy_checkpoint_save_forwards_native_retention_limits(self, tmp_path):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        checkpoint_root = tmp_path / "multi_agent_ckpts"
        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.config = _NS(
            trainer=_NS(
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
        trainer.train_dataloader = FakeDataloader([])

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
        trainer.config = _NS(
            trainer=_NS(
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
        trainer.train_dataloader = FakeDataloader([])

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
        trainer.train_dataloader = FakeDataloader([])
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
            fields=["responses", "response_mask"],
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
        assert grouped["policy_1"].fields == ["responses", "response_mask"]
        assert grouped["policy_1"].extra_info == {"temperature": 0.7}
        assert grouped["policy_2"].keys == ["uid_0_1"]
        assert grouped["policy_2"].fields == ["responses", "response_mask"]
        assert grouped["policy_2"].extra_info == {"temperature": 0.7}

        grouped["policy_1"].extra_info["temperature"] = 0.3
        assert grouped["policy_2"].extra_info == {"temperature": 0.7}

    def test_make_batch_like_rejects_non_kv_batch_constructor(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class NonKVBatch:
            def __init__(self, keys, tags):
                self.keys = keys
                self.tags = tags

        batch = object.__new__(NonKVBatch)
        batch.partition_id = "train"
        batch.keys = ["uid_0_0"]
        batch.tags = [{"policy_name": "policy_1"}]
        batch.fields = ["responses"]
        batch.extra_info = {"temperature": 0.7}

        with pytest.raises(TypeError):
            MultiAgentsPPOTrainer._make_batch_like(
                batch,
                keys=batch.keys,
                tags=batch.tags,
            )

    def test_make_batch_like_requires_partition_id(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class MissingPartitionBatch:
            def __init__(self, *, partition_id, keys, tags, fields, extra_info):
                self.partition_id = partition_id
                self.keys = keys
                self.tags = tags
                self.fields = fields
                self.extra_info = extra_info

        batch = object.__new__(MissingPartitionBatch)
        batch.keys = ["uid_0_0"]
        batch.tags = [{"policy_name": "policy_1"}]
        batch.fields = ["responses"]
        batch.extra_info = {"temperature": 0.7}

        with pytest.raises(AttributeError, match="partition_id"):
            MultiAgentsPPOTrainer._make_batch_like(
                batch,
                keys=batch.keys,
                tags=batch.tags,
            )

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

        with pytest.raises(AttributeError, match="train_dataloader"):
            trainer.train_step()

        assert len(trainer.agent_loop_manager.generated_prompts) == 0
        assert trainer.step_events == []

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
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.step_events = []

        agent_loop_manager = _init_agent_loop_and_fit(trainer)

        assert trainer.agent_loop_manager is agent_loop_manager
        assert len(agent_loop_manager.generated_prompts) == 2

    def test_sync_runtime_hooks_drive_per_policy_checkpoint_managers(self, monkeypatch):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        marked_timer = RecordingMarkedTimer()
        debug_module = types.ModuleType("verl.utils.debug")
        debug_module.marked_timer = marked_timer
        monkeypatch.setitem(sys.modules, "verl.utils.debug", debug_module)
        sys.modules["verl.utils"].debug = debug_module
        import uni_agent.trainer.multi_agents_ppo_trainer as trainer_module
        monkeypatch.setattr(trainer_module, "marked_timer", marked_timer)

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

    def test_sync_sample_end_exposes_missing_sleep_replicas_interface(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={"policy_1": _policy_config("policy_1")},
        )
        trainer.policy_trainers["policy_1"].checkpoint_manager = None

        with pytest.raises(AttributeError):
            trainer.on_sample_end()

    def test_get_reward_handles_exposes_missing_policy_interface(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.policy_trainers = {"policy_1": SimpleNamespace()}

        with pytest.raises(AttributeError):
            trainer.get_reward_handles()

    def test_separate_async_step_end_exposes_missing_lifecycle_interface(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.trainer_mode = "separate_async"
        trainer.global_steps = 1
        trainer.timing_raw = {}
        trainer.policy_trainers = {"policy_1": SimpleNamespace()}
        trainer._policy_pool = ThreadPoolExecutor(max_workers=1)

        try:
            with pytest.raises(AttributeError):
                trainer.on_step_end()
        finally:
            trainer._policy_pool.shutdown(wait=True)

    def test_update_weights_exposes_missing_checkpoint_interface(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.global_steps = 1
        trainer.policy_trainers = {"policy_1": SimpleNamespace(checkpoint_manager=None)}

        with pytest.raises(AttributeError):
            trainer._update_weights_one_policy("policy_1")

    def test_separate_async_hooks_isolate_and_prefix_policy_timing(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class SeparateAsyncHookTrainer(FakeV1PPOTrainer):
            def on_step_end(self):
                # Mirrors verl's hook: the native hook records an unprefixed
                # stage in the trainer-owned timing context.
                self.timing_raw["update_weights"] = self.policy_name

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={
                "policy_1": _policy_config("policy_1"),
                "policy_2": _policy_config("policy_2"),
            },
            policy_trainer_cls=SeparateAsyncHookTrainer,
        )
        trainer.trainer_mode = "separate_async"
        trainer.global_steps = 7
        trainer.timing_raw = {"step": 1.0}
        trainer._sync_policy_runtime_context()

        trainer.on_step_end()

        assert trainer.timing_raw == {
            "step": 1.0,
            "policy_1/update_weights": "policy_1",
            "policy_2/update_weights": "policy_2",
        }
        policy_timings = [
            trainer.policy_trainers[name].timing_raw
            for name in ("policy_1", "policy_2")
        ]
        assert policy_timings[0] is not policy_timings[1]
        assert policy_timings == [
            {"update_weights": "policy_1"},
            {"update_weights": "policy_2"},
        ]

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
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
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

    def test_fit_saves_outer_checkpoint_before_policy_step_end(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        class RecordingTrainer(RecordingMultiAgentsPPOTrainerMixin, MultiAgentsPPOTrainer):
            def _save_checkpoint(self):
                self.step_events.append(("save_checkpoint", self.global_steps))

            def on_step_end(self):
                self.step_events.append(("on_step_end", self.global_steps))
                super().on_step_end()

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
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )
        trainer.step_events = []

        _init_agent_loop_and_fit(trainer)

        assert trainer.step_events == [
            ("sample", 1),
            ("build", 1),
            ("update", 1, {"policy_1": "batch:1"}),
            ("save_checkpoint", 1),
            ("on_step_end", 1),
            ("sample", 2),
            ("build", 2),
            ("update", 2, {"policy_1": "batch:2"}),
            ("save_checkpoint", 2),
            ("on_step_end", 2),
        ]

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

    def test_dataloaders_are_created_by_init_not_constructor(self):
        _install_dependency_stubs()

        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = _make_trainer(
            MultiAgentsPPOTrainer,
            policy_configs={"policy_1": _policy_config("policy_1")},
            policy_trainer_cls=FakeV1PPOTrainerWithDataloader,
        )

        assert not hasattr(trainer, "train_dataloader")
        assert not hasattr(trainer, "val_dataloader")

        trainer.init()

        assert trainer.train_dataloader is trainer.policy_trainers["policy_1"].train_dataloader
        assert trainer.val_dataloader is trainer.policy_trainers["policy_1"].val_dataloader

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

    def test_submit_batch_to_rollout_is_train_only_like_native_v1(self):
        _install_dependency_stubs()

        import inspect
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        signature = inspect.signature(MultiAgentsPPOTrainer._submit_batch_to_rollout)

        assert "partition_id" not in signature.parameters

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

    def test_add_data_metrics_reports_trajectory_staleness_by_policy(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.global_steps = 12
        trainer.policy_trainers = {"policy_1": object(), "policy_2": object()}
        batch = SimpleKVBatchMeta(
            keys=["k1", "k2", "k3", "k4"],
            tags=[
                {"policy_name": "policy_1", "min_global_steps": 8, "max_global_steps": 10},
                {"policy_name": "policy_1", "min_global_steps": 11, "max_global_steps": 11},
                {"policy_name": "policy_2", "min_global_steps": 9, "max_global_steps": 9},
                {
                    "policy_name": "policy_2",
                    "is_padding": True,
                    "min_global_steps": 1,
                    "max_global_steps": 1,
                },
            ],
        )
        metrics = {}

        trainer._add_data_metrics(batch, metrics)

        assert metrics["training/off_policy/trajectory_spans/mean"] == pytest.approx(5 / 3)
        assert metrics["training/off_policy/trajectory_spans/max"] == 3
        assert metrics["training/off_policy/trajectory_spans/min"] == 1
        assert metrics["training/off_policy/trajectory_staleness/mean"] == pytest.approx(1)
        assert metrics["training/off_policy/trajectory_staleness/max"] == 2
        assert metrics["training/off_policy/trajectory_staleness/min"] == 0
        assert metrics["training/off_policy/trajectory_staleness_worst/mean"] == pytest.approx(5 / 3)
        assert metrics["training/off_policy/trajectory_staleness_worst/max"] == 3
        assert metrics["training/off_policy/trajectory_staleness_worst/min"] == 0

        assert metrics["policy_1/off_policy/trajectory_spans/mean"] == pytest.approx(2)
        assert metrics["policy_1/off_policy/trajectory_staleness/mean"] == pytest.approx(0.5)
        assert metrics["policy_1/off_policy/trajectory_staleness_worst/mean"] == pytest.approx(1.5)
        assert metrics["policy_2/off_policy/trajectory_spans/mean"] == pytest.approx(1)
        assert metrics["policy_2/off_policy/trajectory_staleness/mean"] == pytest.approx(2)
        assert metrics["policy_2/off_policy/trajectory_staleness_worst/mean"] == pytest.approx(2)

    def test_add_data_metrics_skips_staleness_when_version_metadata_is_absent(self):
        from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

        trainer = object.__new__(MultiAgentsPPOTrainer)
        trainer.global_steps = 12
        trainer.policy_trainers = {"policy_1": object()}
        batch = SimpleKVBatchMeta(
            keys=["k1"],
            tags=[{"policy_name": "policy_1"}],
        )
        metrics = {}

        trainer._add_data_metrics(batch, metrics)

        assert not any("off_policy/trajectory_" in key for key in metrics)
