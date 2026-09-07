from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import InvalidVersion, Version


logger = logging.getLogger(__name__)


def _tq_supports_checkpoint() -> bool:
    """Whether the installed TransferQueue can snapshot and restore state."""
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


class MultiAgentsPPOTrainer:
    """PPO orchestration layer for multi-agent blackbox training.

    The class owns per-policy v1 PPO trainer runtimes and builds one shared
    AgentFrameworkRolloutAdapter for multi-agent rollout collection.
    """
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        # Capture guaranteed outer config values once (verl-native style): the
        # trainer loop and helper methods read these instance attributes instead
        # of re-walking the config tree on every access.
        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        if self.parameter_sync_step <= 0:
            raise ValueError(f"parameter_sync_step must be positive, got {self.parameter_sync_step}")
        self.train_batch_size = self.config.data.train_batch_size
        self.save_freq = self.config.trainer.save_freq
        self.total_training_steps = self.config.trainer.total_training_steps
        if self.total_training_steps < 0:
            raise ValueError(f"total_training_steps must be non-negative, got {self.total_training_steps}")

        self.policy_configs = self._resolve_policy_configs()
        self.replay_buffer = None

        self.policy_trainers: dict[str, Any] = {}
        self.agent_loop_manager = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataloader_it = None
        self.global_steps = 0
        self.timing_raw: dict[str, Any] = {}

        self._create_policy_trainers()
        # Dynamic inference scheduling (cross-policy replica borrowing); None
        # unless config.dynamic_inference_scheduling.enable is true. Built in
        # init() once every policy runtime exists.
        self._dynamic_inference = None
        # Per-policy phase methods (compute_log_prob / update_actor /
        # update_weights) run on disjoint GPU sets, so they are submitted to
        # this pool to overlap instead of idling one policy's GPUs. verl's
        # phase methods are synchronous (they block on ray.get), so threads are
        # the natural parallelism unit; TQ creates a fresh zmq socket per call
        # and ray.get is thread-safe, making this safe.
        self._policy_pool = ThreadPoolExecutor(max_workers=max(1, len(self.policy_trainers)))

    def _resolve_policy_configs(self) -> dict[str, Any]:
        policies = self.config.get("policies")
        resolved = {}
        for policy_key, policy_entry in (policies.items() if policies is not None else []):
            # The policy name is the dict key itself; the yaml has no per-policy
            # "name" override field.
            policy_name = policy_key
            config_name = policy_entry.get("ppo_trainer_config_name")
            if not config_name:
                raise ValueError(
                    f"policy '{policy_name}' requires config.policies['{policy_name}'].ppo_trainer_config_name "
                    "(e.g. 'ppo_trainer' or 'ppo_megatron_trainer')"
                )
            resolved[policy_name] = self._compose_policy_ppo_config(
                policy_name=policy_name,
                config_name=config_name,
                policy_entry=policy_entry,
            )

        if not resolved:
            raise ValueError("MultiAgentsPPOTrainer requires config.policies[*].ppo_trainer_config_name")
        return resolved

    def _compose_policy_ppo_config(self, *, policy_name: str, config_name: str, policy_entry: Any):
        from hydra import compose, initialize_config_module

        # Policy trainer configs come from a hydra config module (e.g. verl's
        # "verl.trainer.config"); per-policy source overrides the outer default.
        source = (
            policy_entry.get("ppo_trainer_config_source")
            or self.config.get("ppo_trainer_config_source")
            or "verl.trainer.config"
        )
        with initialize_config_module(config_module=source, version_base=None):
            policy_config = compose(config_name=config_name)

        overrides = policy_entry.get("ppo_trainer_overrides")
        if overrides is not None:
            set_struct = getattr(OmegaConf, "set_struct", None)
            if callable(set_struct):
                set_struct(policy_config, False)
            if isinstance(overrides, DictConfig):
                overrides = OmegaConf.to_container(overrides, resolve=True)
            policy_config = OmegaConf.merge(policy_config, OmegaConf.create(overrides))

        with open_dict(policy_config):
            policy_config.policy_name = policy_name
        return policy_config

    def _create_policy_trainers(self) -> dict[str, Any]:
        for policy_name, policy_config in self.policy_configs.items():
            # The outer trainer owns the checkpoint lifecycle (load/save under
            # checkpoints/.../policies/<policy>), so a per-policy trainer must
            # never self-resume; verl's default resume_mode is "auto", which
            # would conflict with the outer ownership.
            trainer_config = policy_config.get("trainer")
            if trainer_config is not None:
                with open_dict(trainer_config):
                    trainer_config.resume_mode = "disable"

            trainer_mode = policy_config.trainer.v1.trainer_mode
            if trainer_mode == "sync":
                from uni_agent.trainer.single_ppo_trainer import SinglePPOTrainer
                trainer_cls = SinglePPOTrainer
            elif trainer_mode == "separate_async":
                parameter_sync_step = policy_config.trainer.v1.separate_async.parameter_sync_step
                if parameter_sync_step != 1:
                    raise ValueError(
                        "MultiAgentsPPOTrainer separate_async mode only supports parameter_sync_step=1 "
                        "for now (Decoupled PPO is a deferred phase); got "
                        f"parameter_sync_step={parameter_sync_step}."
                    )
                from uni_agent.trainer.single_async_ppo_trainer import SingleAsyncPPOTrainer
                trainer_cls = SingleAsyncPPOTrainer
            else:
                raise ValueError(f"Unsupported trainer.v1.trainer_mode: {trainer_mode!r}")
            self.policy_trainers[policy_name] = trainer_cls(config=policy_config)
        return self.policy_trainers

    def init(self) -> None:
        """Initialize all components of the multi-agent trainer.

        1. Per-policy v1 runtime setup for each policy; ``on_init_end()`` is
           intentionally deferred.
        2. Shared outer dataloader: single prompt stream for all policies.
        3. Shared outer replay buffer over the TransferQueue.
        4. Outer checkpoint phase; the outer trainer owns the shared checkpoint
           lifecycle and establishes actor/critic state and ``global_steps``.
        5. Invoke each policy's ``on_init_end()`` after the outer
           initialization phase to synchronize current actor weights to that
           policy's rollout replicas.
        """
        self._prepare_dynamic_inference_slots()
        for policy_trainer in self.policy_trainers.values():
            policy_trainer.init_runtime()
        self._build_dataloader()
        self._build_replay_buffer()
        self._load_checkpoint()
        self._init_dynamic_inference()

    def _prepare_dynamic_inference_slots(self) -> None:
        """Reserve one home slot plus one sleeping-guest slot per outgoing edge.

        Standalone placement groups are created inside each policy trainer's
        ``init()``, before the dynamic controller can pre-create guests.  Put
        the graph-derived slot count on every rollout config and install the
        local ``init_standalone`` wrapper before those placement groups exist.
        """
        from uni_agent.trainer.dynamic_inference import patch
        from uni_agent.trainer.dynamic_inference.types import (
            expand_pairs,
            required_slots_by_home,
            parse_scheduling_config,
        )

        scheduling = parse_scheduling_config(
            self.config.get("dynamic_inference_scheduling"))
        if scheduling is None or scheduling.mode == "static":
            return

        policy_names = list(self.policy_configs)
        pairs = expand_pairs(scheduling, policy_names)
        slots_by_home = required_slots_by_home(pairs, policy_names)
        patch.apply_patch()

        for policy_name, policy_config in self.policy_configs.items():
            rollout = policy_config.actor_rollout_ref.rollout
            custom = rollout.get("custom")
            if custom is None:
                with open_dict(rollout):
                    rollout.custom = OmegaConf.create({})
                custom = rollout.custom
            with open_dict(custom):
                custom["dynamic_inference_max_colocate_count"] = slots_by_home[policy_name]

    def _init_dynamic_inference(self) -> None:
        """Build the dynamic inference controller when the config block enables it.

        All policy runtimes (standalone replicas holding initial weights) must
        already exist: setup pre-creates the guest replicas and probes the
        worker-side sleep patch before training starts.
        """
        from uni_agent.trainer.dynamic_inference.controller import DynamicInferenceController

        controller = DynamicInferenceController.maybe_create(self)
        if controller is None:
            return
        controller.setup()
        self._dynamic_inference = controller

    def _shutdown_dynamic_inference(self) -> None:
        """Return every active lend, stop the poll loop, kill guest replicas."""
        if self._dynamic_inference is None:
            return
        try:
            self._dynamic_inference.shutdown()
        except Exception as exc:
            logger.warning("dynamic inference shutdown failed: %s", exc)
        finally:
            self._dynamic_inference = None

        for policy_trainer in self.policy_trainers.values():
            policy_trainer.on_init_end()

    def get_multi_policy_llm_client(self) -> list[Any] | None:
        from uni_agent.trainer.gateway.runtime import PolicyRoutingLLMClient

        policy_clients = {}
        for policy_name, trainer in self.policy_trainers.items():
            get_llm_client = getattr(trainer, "get_llm_client", None)
            if not callable(get_llm_client):
                raise AttributeError(f"PPO trainer for policy '{policy_name}' has no get_llm_client()")
            policy_clients[policy_name] = get_llm_client()
        return PolicyRoutingLLMClient(policy_clients)

    def get_reward_handles(self) -> list[Any] | None:
        """Aggregate reward-worker handles from all policy trainers.

        All policies currently share the same rule-based reward function, so any
        worker can score any trajectory; aggregating lets the framework load-
        balance across all workers instead of only the first policy's. If
        policies ever diverge in reward functions, this must become per-policy
        routing.
        """
        handles: list[Any] = []
        for trainer in self.policy_trainers.values():
            get_reward_handles = getattr(trainer, "get_reward_handles", None)
            if not callable(get_reward_handles):
                continue
            trainer_handles = get_reward_handles()
            if trainer_handles:
                handles.extend(list(trainer_handles))
        return handles or None

    def get_gateway_actor_kwargs(self) -> dict[str, Any]:
        policy_tokenizers = {}
        policy_processors = {}
        policy_tool_parser_names = {}
        for policy_name, trainer in self.policy_trainers.items():
            tokenizer = getattr(trainer, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(f"PPO trainer for policy '{policy_name}' has no tokenizer")
            policy_tokenizers[policy_name] = tokenizer
            processor = getattr(trainer, "processor", None)
            policy_processors[policy_name] = processor
            tool_parser_name = OmegaConf.select(
                self.policy_configs[policy_name],
                "actor_rollout_ref.rollout.multi_turn.format",
                default=None,
            )
            if tool_parser_name:
                policy_tool_parser_names[policy_name] = str(tool_parser_name)

        first_policy_name = next(iter(policy_tokenizers))
        gateway_actor_kwargs: dict[str, Any] = {
            "tokenizer": policy_tokenizers[first_policy_name],
            "policy_tokenizers": policy_tokenizers,
            "policy_processors": policy_processors,
        }
        if policy_tool_parser_names:
            gateway_actor_kwargs["policy_tool_parser_names"] = policy_tool_parser_names
        default_processor = policy_processors[first_policy_name]
        if default_processor is not None:
            gateway_actor_kwargs["processor"] = default_processor
        return gateway_actor_kwargs

    def fit(self, agent_loop_manager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager
        from verl.utils.tracking import Tracking
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        # verl semantics: global_steps is 1-based and is advanced before
        # on_train_begin, so warmup batches and update_weights/rollout version
        # tags all use the current step number.
        self.global_steps += 1
        self._sync_policy_runtime_context()
        self._reissue_inflight_prompts()
        self.on_train_begin()
        succeeded = False
        try:
            while self.global_steps <= self.total_training_steps:
                step_metrics = self.train_step()
                if self._should_save_checkpoint():
                    self._save_checkpoint()
                # Mirror verl v1 trainer.fit(): record per-step metrics
                # (loss/adv/grad_norm, prefixed per policy) to the configured
                # backend. Sorting keeps the two policies' metrics grouped.
                self.logger.log(data=dict(sorted(step_metrics.items())), step=self.global_steps)
                self.global_steps += 1
                self._sync_policy_runtime_context()
            succeeded = True
        finally:
            self.on_train_end()
            tracking = getattr(self, "logger", None)
            if tracking is not None:
                tracking.finish(exit_code=0 if succeeded else 1)

    def train_step(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        self.timing_raw = {}
        self._sync_policy_runtime_context()
        self.on_step_begin()
        batch = self.step(metrics=metrics, timing_raw=self.timing_raw)
        self.on_step_end()
        # separate_async: each policy's on_step_end stores the standalone
        # checkpoint manager's weight-sync metrics in _pending_sync_metrics;
        # verl's native fit() merges them via _consume_sync_metrics().
        metrics.update(self._consume_sync_metrics())
        if self._dynamic_inference is not None:
            metrics.update(self._dynamic_inference.last_metrics or {})
        # Mirror verl v1 trainer.fit(): evict the sampled trajectory records
        # from TransferQueue at the end of each step. ReplayBuffer.sample()
        # only clears the prompt uids, so without this the per-trajectory
        # {uid}_{sample_idx}_{record_idx} records accumulate in SimpleStorage
        # and eventually exceed total_storage_size on long runs.
        if batch is not None:
            tq.kv_clear(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
            )
        metrics["training/global_step"] = self.global_steps
        return dict(metrics)

    def _consume_sync_metrics(self) -> dict[str, Any]:
        """Consume each policy trainer's pending weight-sync metrics.

        separate_async's per-policy ``on_step_end`` stores the standalone
        checkpoint manager's sync metrics (engine-side per-sync stats) in
        ``trainer._pending_sync_metrics``. The outer trainer merges them,
        prefixed per policy, so the console/wandb output stays namespaced.
        """
        metrics: dict[str, Any] = {}
        for policy_name, trainer in self.policy_trainers.items():
            pending = getattr(trainer, "_pending_sync_metrics", None)
            if not pending:
                continue
            for key, value in pending.items():
                metrics[f"{policy_name}/sync/{key}"] = value
            try:
                trainer._pending_sync_metrics = {}
            except Exception:
                pass
        return metrics

    def step(self, metrics: dict[str, Any] | None = None, timing_raw: dict[str, Any] | None = None):
        metrics = metrics if metrics is not None else {}
        timing_raw = timing_raw if timing_raw is not None else {}
        train_batch_size = self.train_batch_size
        parameter_sync_step = self.parameter_sync_step
        if train_batch_size % parameter_sync_step != 0:
            raise ValueError(
                f"train_batch_size ({train_batch_size}) must be divisible by "
                f"parameter_sync_step ({parameter_sync_step})"
            )
        sample_batch_size = train_batch_size // parameter_sync_step

        self._add_batch_to_generate()
        step_batches = [
            self._step_once(metrics, timing_raw, sample_batch_size)
            for _ in range(parameter_sync_step)
        ]
        if len(step_batches) == 1:
            return step_batches[0]

        return self._make_batch_like(
            step_batches[0],
            keys=[key for batch in step_batches for key in batch.keys],
            tags=[tag for batch in step_batches for tag in batch.tags],
        )

    def _step_once(self, metrics: dict[str, Any], timing_raw: dict[str, Any], sample_batch_size: int):
        del timing_raw
        multi_agent_batch = self.sample_multi_agent_batch(sample_batch_size=sample_batch_size, metrics=metrics)
        per_policy_batches = self.build_per_policy_batches(multi_agent_batch)
        per_policy_batches = self.prepare_policy_batches_for_ppo_update(per_policy_batches, metrics)
        multi_agent_batch = self.compute_multi_agent_advantage_from_policy_batches(
            per_policy_batches,
            metrics,
        )
        self._add_data_metrics(multi_agent_batch, metrics)
        # Chain the advantage-computed batch into the update, mirroring verl's
        # standard flow (`batch = _compute_advantage(batch); _update_actor(batch)`).
        # Without this, per-policy batches never carry the advantages that
        # _compute_advantage wrote back, and worker-side ppo_loss fails with
        # KeyError('advantages').
        per_policy_batches = self.build_per_policy_batches(multi_agent_batch)
        self.update_policy_trainers(per_policy_batches, metrics=metrics)
        return multi_agent_batch

    def sample_multi_agent_batch(self, sample_batch_size: int | None = None, metrics: dict[str, Any] | None = None):
        self.on_sample_begin()
        result = self.replay_buffer.sample(
            global_steps=self.global_steps,
            partition_id="train",
            batch_size=sample_batch_size if sample_batch_size is not None else self.train_batch_size,
        )
        self.on_sample_end()

        off_policy_metrics = {}
        if isinstance(result, tuple):
            batch, off_policy_metrics = result
        else:
            batch = result
        if metrics is not None and off_policy_metrics:
            metrics.update(off_policy_metrics)

        # NOTE: sampling temperature is intentionally NOT attached here. The
        # outer config has no actor_rollout_ref.rollout.temperature (it is
        # per-policy), and verl's per-policy _compute_old_log_prob writes the
        return batch

    def build_per_policy_batches(self, multi_agent_batch):
        if not hasattr(multi_agent_batch, "keys") or not hasattr(multi_agent_batch, "tags"):
            raise TypeError("multi_agent_batch must be a KVBatchMeta-like object with keys and tags")
        if len(multi_agent_batch.keys) != len(multi_agent_batch.tags):
            raise ValueError("multi_agent_batch keys and tags must have the same length")

        grouped: dict[str, dict[str, list[Any]]] = {}
        for key, tag in zip(multi_agent_batch.keys, multi_agent_batch.tags, strict=True):
            policy_name = tag.get("policy_name")
            if not policy_name:
                raise ValueError(f"Multi-agent trajectory tag for key {key!r} is missing policy_name")
            if policy_name not in self.policy_trainers:
                raise ValueError(f"Unknown policy_name in multi-agent trajectory tag: {policy_name}")
            group = grouped.setdefault(str(policy_name), {"keys": [], "tags": []})
            group["keys"].append(key)
            group["tags"].append(tag)

        return {
            policy_name: self._make_batch_like(
                multi_agent_batch,
                keys=grouped[policy_name]["keys"],
                tags=grouped[policy_name]["tags"],
            )
            for policy_name in self.policy_trainers
            if policy_name in grouped
        }

    def prepare_policy_batches_for_ppo_update(self, per_policy_batches, metrics: dict[str, Any]):
        """Run per-policy balance/log-prob stages concurrently on disjoint GPUs."""
        futures = [
            self._policy_pool.submit(self._prepare_policy_batch_for_update, policy_name, batch)
            for policy_name, batch in per_policy_batches.items()
        ]
        prepared = {}
        timing_raw: dict[str, float] = {}
        for future in futures:
            policy_name, batch, policy_metrics, per_policy_timing = future.result()
            self._prefix_metrics(metrics, policy_name, policy_metrics)
            timing_raw.update(per_policy_timing)
            prepared[policy_name] = batch
        self.timing_raw.update(timing_raw)
        return prepared

    def _prepare_policy_batch_for_update(
        self, policy_name: str, batch
    ) -> tuple[str, Any, dict[str, Any], dict[str, float]]:
        """Run one policy's pre-update stages (balance + old/ref log-probs + values)."""
        from verl.utils.debug import marked_timer

        trainer = self.policy_trainers[policy_name]
        policy_metrics: dict[str, Any] = {}
        timing_raw: dict[str, float] = {}

        balance_batch = getattr(trainer, "_balance_batch", None)
        if callable(balance_batch):
            batch = self._call_v1_stage(
                balance_batch,
                batch,
                policy_metrics,
                logging_prefix="global_seqlen",
            )

        compute_old_log_prob = getattr(trainer, "_compute_old_log_prob", None)
        if callable(compute_old_log_prob):
            with marked_timer(f"{policy_name}/old_log_prob", timing_raw, color="blue"):
                batch = self._call_v1_stage(compute_old_log_prob, batch, policy_metrics)

        if getattr(trainer, "use_reference_policy", False):
            compute_ref_log_prob = getattr(trainer, "_compute_ref_log_prob", None)
            if callable(compute_ref_log_prob):
                with marked_timer(f"{policy_name}/ref_log_prob", timing_raw, color="olive"):
                    batch = self._call_v1_stage(compute_ref_log_prob, batch, policy_metrics)

        if getattr(trainer, "use_critic", False):
            compute_values = getattr(trainer, "_compute_values", None)
            if callable(compute_values):
                with marked_timer(f"{policy_name}/values", timing_raw, color="cyan"):
                    batch = self._call_v1_stage(compute_values, batch, policy_metrics)

        return policy_name, batch, policy_metrics, timing_raw

    def compute_multi_agent_advantage_from_policy_batches(self, per_policy_batches, metrics: dict[str, Any]):
        # Advantage needs the merged rollout/group view. The per-policy batches
        # keep the same TQ keys, so updates can reuse them after advantage is
        # written back to the shared trajectory records.
        multi_agent_batch = self._merge_policy_batches(per_policy_batches)
        return self.compute_multi_agent_advantage(multi_agent_batch, metrics)
    def compute_multi_agent_advantage(self, multi_agent_batch, metrics: dict[str, Any]):
        for trainer in self.policy_trainers.values():
            compute_advantage = getattr(trainer, "_compute_advantage", None)
            if callable(compute_advantage):
                batch = self._call_v1_stage(compute_advantage, multi_agent_batch, metrics)
                self._add_advantage_metrics(multi_agent_batch, metrics)
                return batch
        raise AttributeError("At least one policy trainer must provide _compute_advantage()")

    def _add_advantage_metrics(self, batch, metrics: dict[str, Any]) -> None:
        """Log per-policy advantage distribution stats from the TransferQueue.

        verl's per-policy ``_compute_advantage`` writes token-level
        ``advantages``/``returns`` back to the shared trajectory records but
        does not emit any advantage metrics itself. This reads them back (one
        extra TQ fetch per step) and aggregates masked distribution stats per
        policy, so GRPO group-normalized advantages become observable.
        """
        if batch is None or not getattr(batch, "keys", None) or not getattr(batch, "tags", None):
            return
        try:
            import torch

            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
                select_fields=["advantages", "response_mask"],
            )
            padded = data.to_padded_tensor()
            advantages = padded["advantages"].detach().cpu().float()
            response_mask = padded["response_mask"].to(bool).cpu()
        except Exception as exc:
            logger.warning("failed to compute advantage metrics: %s", exc)
            return

        per_policy: dict[str, list[torch.Tensor]] = {policy: [] for policy in self.policy_trainers}
        for i, tag in enumerate(batch.tags):
            policy_name = tag.get("policy_name")
            if policy_name not in per_policy:
                continue
            values = advantages[i][response_mask[i]]
            if values.numel() == 0:
                continue
            per_policy[policy_name].append(values)

        for policy_name, tensors in per_policy.items():
            if not tensors:
                continue
            values = torch.cat(tensors)
            stats = {
                "advantages_mean": values.mean().item(),
                # Population std (correction=0): a single value or all-equal
                # advantages yields 0.0 instead of NaN (torch's default
                # sample std with correction=1 returns NaN for n=1).
                "advantages_std": values.std(correction=0).item(),
                "advantages_min": values.min().item(),
                "advantages_max": values.max().item(),
                "advantages_abs_mean": values.abs().mean().item(),
                "advantages_positive_ratio": (values > 0).float().mean().item(),
            }
            for key, value in stats.items():
                metrics[f"{policy_name}/actor/{key}"] = value

    def _add_data_metrics(self, batch, metrics: dict[str, Any]) -> None:
        """Mirror verl v1 ``_compute_metrics``' data-metrics portion, per policy.

        Fetches score/reward/advantage/return/length/num-turns fields from the
        TransferQueue and aggregates them through verl's
        ``compute_data_metrics`` (token-level masked, identical semantics to
        native verl), split per policy. Keys are prefixed ``policy_X/``, e.g.
        ``policy_1/critic/score/mean`` / ``policy_1/response_length/mean``.
        """
        if batch is None or not getattr(batch, "keys", None) or not getattr(batch, "tags", None):
            return
        try:
            import torch
            from verl import DataProto
            from verl.trainer.ppo.metric_utils import compute_data_metrics

            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
                select_fields=[
                    "prompts",
                    "responses",
                    "response_mask",
                    "advantages",
                    "returns",
                    "rm_scores",
                    "num_turns",
                ],
            )
            # num_turns is a per-row scalar field; capture it before padding.
            num_turns = np.array(data.pop("num_turns").tolist())
            prompt_length = data["prompts"].offsets().diff()
            response_length = data["responses"].offsets().diff()

            data = data.to_padded_tensor()
            if "token_level_scores" not in data:
                data["token_level_scores"] = data["rm_scores"]
            if "token_level_rewards" not in data:
                data["token_level_rewards"] = data["rm_scores"]
            data["prompt_length"] = prompt_length.float()
            data["response_length"] = response_length.float()
            dp = DataProto(batch=data)
        except Exception as exc:
            logger.warning("failed to compute data metrics: %s", exc)
            return

        for policy_name in self.policy_trainers:
            mask = np.array(
                [
                    not tag.get("is_padding", False) and tag.get("policy_name") == policy_name
                    for tag in batch.tags
                ],
                dtype=bool,
            )
            if not mask.any():
                continue
            try:
                sub = dp.select_idxs(mask)
                for key, value in compute_data_metrics(sub, use_critic=False).items():
                    metrics[f"{policy_name}/{key}"] = value
                policy_num_turns = num_turns[mask]
                if policy_num_turns.size > 0:
                    metrics[f"{policy_name}/num_turns/mean"] = policy_num_turns.mean().item()
                    metrics[f"{policy_name}/num_turns/max"] = policy_num_turns.max().item()
                    metrics[f"{policy_name}/num_turns/min"] = policy_num_turns.min().item()
            except Exception as exc:
                logger.warning("failed to compute data metrics for %s: %s", policy_name, exc)

    def update_policy_trainers(self, per_policy_batches, metrics: dict[str, Any] | None = None):
        """Run per-policy critic/actor updates concurrently on disjoint GPUs."""
        metrics = metrics if metrics is not None else {}
        critic_warmup = self.config.trainer.get("critic_warmup", 0)
        futures = [
            self._policy_pool.submit(self._update_one_policy, policy_name, batch, critic_warmup)
            for policy_name, batch in per_policy_batches.items()
        ]
        updated = {}
        timing_raw: dict[str, float] = {}
        for future in futures:
            policy_name, batch, policy_metrics, per_policy_timing = future.result()
            self._prefix_metrics(metrics, policy_name, policy_metrics)
            timing_raw.update(per_policy_timing)
            updated[policy_name] = batch
        self.timing_raw.update(timing_raw)
        return updated
    def _update_one_policy(
        self, policy_name: str, batch, critic_warmup: int
    ) -> tuple[str, Any, dict[str, Any], dict[str, float]]:
        """Run one policy's critic/actor update."""
        from verl.utils.debug import marked_timer

        trainer = self.policy_trainers[policy_name]
        policy_metrics: dict[str, Any] = {}
        timing_raw: dict[str, float] = {}

        if getattr(trainer, "use_critic", False):
            update_critic = getattr(trainer, "_update_critic", None)
            if callable(update_critic):
                with marked_timer(f"{policy_name}/update_critic", timing_raw, color="pink"):
                    batch = self._call_v1_stage(update_critic, batch, policy_metrics)

        if critic_warmup <= self.global_steps:
            update_actor = getattr(trainer, "_update_actor", None)
            if callable(update_actor):
                with marked_timer(f"{policy_name}/update_actor", timing_raw, color="red"):
                    batch = self._call_v1_stage(update_actor, batch, policy_metrics)

        return policy_name, batch, policy_metrics, timing_raw

    def _build_dataloader(self) -> None:
        """Build the single shared prompt stream used by multi-agent sampling.

        Mirrors verl PPOTrainer._init_dataloader but runs once at the outer
        level: SinglePPOTrainer skips per-policy dataloader creation, so the
        full dataset is loaded only once instead of once per policy.
        """
        source_trainer = None
        for trainer in self.policy_trainers.values():
            if getattr(trainer, "tokenizer", None) is not None:
                source_trainer = trainer
                break
        if source_trainer is None:
            raise RuntimeError(
                "MultiAgentsPPOTrainer requires at least one policy trainer with a tokenizer "
                "to build the shared dataloader"
            )
        tokenizer = source_trainer.tokenizer
        processor = getattr(source_trainer, "processor", None)

        from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.utils.dataset.rl_dataset import collate_fn

        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        # Mirror verl: exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups")
        dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        sync_refill_failed_groups = bool(self.config.sampler.sync_refill_failed_groups)
        trainer_mode = self.trainer_mode
        requires_exact_refill = dapo_enabled or sync_refill_failed_groups or trainer_mode != "sync"
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size")
            if user_gen_batch_size not in (None, 1):
                logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config.data):
                self.config.data.gen_batch_size = 1

        gen_batch_size = self.config.data.get("gen_batch_size") or self.train_batch_size
        dataloader_num_workers = self.config.data.dataloader_num_workers
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=dataloader_num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=dataloader_num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _build_replay_buffer(self):
        """Build the outer shared ReplayBuffer used by multi-agent sampling.

        Mirrors verl PPOTrainer._build_replay_buffer but sources every knob
        from the outer config (the outer sampler/algorithm sections are the
        single source of truth; per-policy sampler configs are not used).
        refill_fn is bound directly to the outer trainer, so DAPO group
        filtering / sync_refill_failed_groups refills never route through a
        per-policy trainer that lacks an agent_loop_manager.
        """
        from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

        trainer_mode = self.trainer_mode
        buffer_cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync

        max_off_policy_threshold = self.config.sampler.max_off_policy_threshold
        max_off_policy_strategy = self.config.sampler.max_off_policy_strategy
        sampler_kwargs = self.config.sampler.sampler_kwargs
        sync_refill_failed_groups = bool(self.config.sampler.sync_refill_failed_groups)
        filter_groups_metric = self._resolve_filter_groups_metric()
        train_batch_size = self.train_batch_size
        gen_batch_size = (
            1
            if filter_groups_metric is not None or sync_refill_failed_groups or trainer_mode != "sync"
            else self.config.data.get("gen_batch_size") or train_batch_size
        )
        max_inflight_gen_batches = 1
        if filter_groups_metric is not None:
            filter_groups = self.config.algorithm.get("filter_groups")
            max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)

        self.replay_buffer = buffer_cls(
            trainer_mode=trainer_mode,
            trainer_config=OmegaConf.create({}),
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
            filter_groups_metric=filter_groups_metric,
            sync_refill_failed_groups=sync_refill_failed_groups,
            train_batch_size=train_batch_size,
            gen_batch_size=int(gen_batch_size),
            max_inflight_gen_batches=int(max_inflight_gen_batches),
        )
        return self.replay_buffer

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling.

        Mirrors verl PPOTrainer._resolve_filter_groups_metric, reading the
        outer config.algorithm.filter_groups (inherited by every policy).
        """
        filter_groups = self.config.algorithm.get("filter_groups")
        filter_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric")
        if not filter_metric:
            raise ValueError("algorithm.filter_groups.metric must be set when group filtering is enabled")

        reward_model = self.config.reward.get("reward_model")
        streaming_reward_path = reward_model is None or (
            not reward_model.get("enable", False) or reward_model.get("enable_resource_pool", False)
        )
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def _load_checkpoint(self) -> None:
        checkpoint_dir = self._resolve_checkpoint_dir()
        if checkpoint_dir is None:
            self.global_steps = 0
            return

        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        step_text = checkpoint_name.removeprefix("global_step_")
        if checkpoint_name == step_text or not step_text.isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir}")
        self.global_steps = int(step_text)

        policy_paths = {}
        for policy_name, trainer in self.policy_trainers.items():
            policy_checkpoint_dir = os.path.join(checkpoint_dir, "policies", policy_name)
            actor_path = os.path.join(policy_checkpoint_dir, "actor")
            if not os.path.isdir(actor_path):
                raise FileNotFoundError(
                    f"Checkpoint for policy {policy_name!r} is missing actor directory: {actor_path}"
                )

            critic_path = None
            if trainer.use_critic:
                critic_path = os.path.join(policy_checkpoint_dir, "Critic")
                if not os.path.isdir(critic_path):
                    raise FileNotFoundError(
                        f"Checkpoint for policy {policy_name!r} is missing critic directory: {critic_path}"
                    )
            policy_paths[policy_name] = (actor_path, critic_path)

        del_local_after_load = bool(self.config.trainer.del_local_ckpt_after_load)
        for policy_name, trainer in self.policy_trainers.items():
            actor_path, critic_path = policy_paths[policy_name]
            trainer.actor_rollout_wg.load_checkpoint(
                local_path=actor_path,
                del_local_after_load=del_local_after_load,
            )

            if trainer.use_critic:
                trainer.critic_wg.load_checkpoint(
                    local_path=critic_path,
                    del_local_after_load=del_local_after_load,
                )

        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        if self.train_dataloader is not None and os.path.exists(dataloader_path):
            import torch

            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
        elif self.train_dataloader is not None:
            logger.warning("No dataloader state found at %s; starting its state from scratch", dataloader_path)

        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq_checkpoint_path = os.path.join(checkpoint_dir, "transfer_queue")
            if os.path.exists(tq_checkpoint_path):
                logger.info("Loading TransferQueue state from %s", tq_checkpoint_path)
                tq.load_checkpoint(tq_checkpoint_path)
        self._sync_policy_runtime_context()

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Re-dispatch restored pending/running prompt groups after resume."""
        if self.trainer_mode == "sync" or not _tq_supports_checkpoint():
            return 0

        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0
        if self.agent_loop_manager is None:
            raise RuntimeError("agent_loop_manager must be initialized before reissuing prompts")

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        trajectory_prefixes = tuple(f"{uid}_" for uid in inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.startswith(trajectory_prefixes)
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        from verl.utils import tensordict_utils as tu

        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [
            {"is_prompt": True, "status": "pending", "global_steps": self.global_steps}
            for _ in inflight_uids
        ]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        logger.info(
            "Re-issued %d in-flight prompts at step %d and cleared %d partial trajectories",
            len(inflight_uids),
            self.global_steps,
            len(old_trajectory_keys),
        )
        return len(inflight_uids)

    def _resolve_checkpoint_dir(self) -> str | None:
        resume_mode = self.config.trainer.resume_mode
        if resume_mode == "disable":
            return None
        if resume_mode == "resume_path":
            checkpoint_dir = self.config.trainer.resume_from_path
            if not checkpoint_dir:
                raise ValueError("trainer.resume_from_path is required when trainer.resume_mode='resume_path'")
            checkpoint_dir = os.path.abspath(os.path.normpath(str(checkpoint_dir)))
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return checkpoint_dir
        if resume_mode == "auto":
            checkpoint_root = self.config.trainer.default_local_dir
            if not checkpoint_root:
                return None
            checkpoint_root = os.path.abspath(str(checkpoint_root))
            if not os.path.isdir(checkpoint_root):
                return None
            tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
            if not os.path.isfile(tracker_path):
                logger.info("No checkpoint tracker found at %s; training from scratch", tracker_path)
                return None
            with open(tracker_path, encoding="utf-8") as file:
                step_text = file.read().strip()
            if not step_text.isdigit():
                raise ValueError(f"Invalid checkpoint step in {tracker_path}: {step_text!r}")
            checkpoint_dir = os.path.join(checkpoint_root, f"global_step_{step_text}")
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint tracker points to a missing directory: {checkpoint_dir}"
                )
            return checkpoint_dir
        raise ValueError(f"Unknown trainer.resume_mode: {resume_mode}")

    def _save_checkpoint(self) -> None:
        checkpoint_root = self.config.trainer.default_local_dir
        if not checkpoint_root:
            raise ValueError("trainer.default_local_dir is required to save a multi-agent checkpoint")
        checkpoint_dir = os.path.join(str(checkpoint_root), f"global_step_{self.global_steps}")
        policies_dir = os.path.join(checkpoint_dir, "policies")
        os.makedirs(policies_dir, exist_ok=True)

        default_hdfs_dir = self.config.trainer.default_hdfs_dir
        remove_previous = self._trainer_option("remove_previous_ckpt_in_save", False)
        if remove_previous:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated; use max_actor_ckpt_to_keep=1 "
                "and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            1 if remove_previous else self._trainer_option("max_actor_ckpt_to_keep", None)
        )
        max_critic_ckpt_to_keep = (
            1 if remove_previous else self._trainer_option("max_critic_ckpt_to_keep", None)
        )
        for policy_name, trainer in self.policy_trainers.items():
            policy_checkpoint_dir = os.path.join(policies_dir, policy_name)
            os.makedirs(policy_checkpoint_dir, exist_ok=True)

            actor_remote_path = (
                None
                if default_hdfs_dir is None
                else posixpath.join(
                    str(default_hdfs_dir),
                    f"global_step_{self.global_steps}",
                    "policies",
                    policy_name,
                    "actor",
                )
            )
            trainer.actor_rollout_wg.save_checkpoint(
                os.path.join(policy_checkpoint_dir, "actor"),
                actor_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

            if trainer.use_critic:
                critic_remote_path = (
                    None
                    if default_hdfs_dir is None
                    else posixpath.join(
                        str(default_hdfs_dir),
                        f"global_step_{self.global_steps}",
                        "policies",
                        policy_name,
                        "Critic",
                    )
                )
                trainer.critic_wg.save_checkpoint(
                    os.path.join(policy_checkpoint_dir, "Critic"),
                    critic_remote_path,
                    self.global_steps,
                    max_ckpt_to_keep=max_critic_ckpt_to_keep,
                )

        if self.train_dataloader is not None:
            import torch

            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self.train_dataloader.state_dict(), os.path.join(checkpoint_dir, "data.pt"))

        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq.save_checkpoint(
                os.path.join(checkpoint_dir, "transfer_queue"),
                metadata={"global_steps": self.global_steps},
            )

        if self._has_async_checkpoint_save():
            logger.warning(
                "Skipping multi-policy latest checkpoint tracker at step %d because at least one "
                "policy checkpoint uses async_save; publish the tracker only after all policy saves complete.",
                self.global_steps,
            )
            return

        os.makedirs(str(checkpoint_root), exist_ok=True)
        latest_path = os.path.join(str(checkpoint_root), "latest_checkpointed_iteration.txt")
        with open(latest_path, "w", encoding="utf-8") as file:
            file.write(str(self.global_steps))

    def _trainer_option(self, name: str, default: Any = None) -> Any:
        trainer_config = self.config.trainer
        getter = getattr(trainer_config, "get", None)
        if callable(getter):
            return getter(name, default)
        return getattr(trainer_config, name, default)

    def _has_async_checkpoint_save(self) -> bool:
        """Return whether any policy model checkpoint is written asynchronously."""
        checkpoint_paths = (
            "actor_rollout_ref.actor.checkpoint.async_save",
            "critic.checkpoint.async_save",
        )
        for policy_config in getattr(self, "policy_configs", {}).values():
            for path in checkpoint_paths:
                value = policy_config
                for component in path.split("."):
                    if isinstance(value, Mapping):
                        value = value.get(component)
                    else:
                        value = getattr(value, component, None)
                    if value is None:
                        break
                if bool(value):
                    return True
        return False

    def _should_save_checkpoint(self) -> bool:
        save_freq = self.save_freq
        if save_freq <= 0:
            return False
        return self.global_steps >= self.total_training_steps or self.global_steps % save_freq == 0

    def _fetch_one_gen_batch(self):
        if self.train_dataloader is None:
            return None
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        from verl.utils import tensordict_utils as tu

        batch_dict["uid"] = np.array([str(uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None):
        """Fetch and coalesce the requested number of prompts.

        Mirrors verl's v1 trainer semantics: ``num_prompts`` must be a positive
        multiple of ``data.gen_batch_size`` (defaults to ``data.train_batch_size``),
        and is submitted in whole gen-batch dataloader fetches.
        """
        if self.train_dataloader is None:
            return None
        train_batch_size = self.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        # Read the dataloader's actual fetch granularity instead of the outer
        # config default: verl forces per-policy data.gen_batch_size=1 when
        # DAPO group filtering / sync_refill_failed_groups is enabled, so the
        # outer fetch loop must use the same granularity as the shared
        # dataloader (1 sample per fetch) rather than train_batch_size.
        gen_batch_size = int(getattr(self.train_dataloader, "batch_size", None) or train_batch_size)
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
                f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
            )

        from verl.utils import tensordict_utils as tu

        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)]
        if any(chunk is None for chunk in chunks):
            return None
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the agent loop manager."""
        batch = self._next_train_batch(num_prompts)
        if batch is None:
            return 0
        return self._submit_batch_to_rollout(batch)

    def _add_batch_to_generate(self) -> None:
        batch = self._next_train_batch()
        if batch is None:
            return
        if len(batch) == 0:
            return
        self._submit_batch_to_rollout(batch)

    def _submit_batch_to_rollout(self, batch) -> int:
        """Register prompts in TransferQueue and dispatch them for generation.

        Returns the number of submitted prompts (refill_fn contract).
        """
        if batch is None:
            return 0

        from verl.utils import tensordict_utils as tu

        uid_values = tu.get(batch, "uid")
        if uid_values is None:
            raise ValueError("MultiAgentsPPOTrainer requires batch['uid'] before rollout submission")
        uid_values = uid_values.tolist() if hasattr(uid_values, "tolist") else list(uid_values)
        if not uid_values:
            return 0

        tags = [
            {
                "is_prompt": True,
                "status": "pending",
                "global_steps": self.global_steps,
            }
            for _ in uid_values
        ]
        put_kwargs = {
            "keys": [str(uid) for uid in uid_values],
            "partition_id": "train",
            "tags": tags,
        }
        # Mirror verl: async trainers persist prompt fields in TQ so in-flight
        # prompts can be re-issued after a checkpoint resume.
        trainer_mode = self.trainer_mode
        if trainer_mode != "sync":
            from tensordict.tensorclass import NonTensorData

            fields = batch.select(
                *[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]
            )
            put_kwargs["fields"] = fields
        tq.kv_batch_put(**put_kwargs)
        if self.agent_loop_manager is None:
            raise RuntimeError("agent_loop_manager must be passed to fit() before rollout submission")
        self.agent_loop_manager.generate_sequences(batch)
        return len(uid_values)

    def _sync_policy_runtime_context(self) -> None:
        for trainer in self.policy_trainers.values():
            trainer.global_steps = self.global_steps
            trainer.timing_raw = self.timing_raw

    def on_train_begin(self) -> None:
        if self.config.get("skip") is not None:
            from verl.utils.skip import SkipManager

            SkipManager.init(self.config)
        if self.trainer_mode == "sync":
            return
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()

    def on_train_end(self) -> None:
        # Shut down dataloader workers while the interpreter is still alive.
        # torch's DataLoader registers worker pids with a SIGCHLD handler; if the
        # iterator is never garbage-collected (StatefulDataLoader keeps its own
        # ``_iterator`` reference), ``_shutdown_workers`` never runs and workers
        # remain registered. When the interpreter exits, their teardown SIGKILL
        # surfaces as ``RuntimeError: DataLoader worker ... killed by signal:
        # Killed`` from the atexit path, making a successful run exit non-zero.
        self._close_dataloader()

    def _close_dataloader(self) -> None:
        """Terminate dataloader workers cleanly and drop iterator references.

        Safe to call multiple times (idempotent): after the first call the
        attributes are None and torch's ``_shutdown_workers`` is itself guarded
        by an internal ``_shutdown`` flag.
        """
        for attr in ("train_dataloader_it", "val_dataloader_it"):
            it = getattr(self, attr, None)
            if it is None:
                continue
            try:
                shutdown = getattr(it, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                logger.warning("failed to shut down %s", attr, exc_info=True)
            finally:
                setattr(self, attr, None)

        for attr in ("train_dataloader", "val_dataloader"):
            dl = getattr(self, attr, None)
            if dl is None:
                continue
            try:
                # StatefulDataLoader keeps ``self._iterator`` alive, so the
                # iterator's ``__del__`` (which calls ``_shutdown_workers``)
                # would otherwise never run. Shut it down and drop the ref.
                internal_it = getattr(dl, "_iterator", None)
                if internal_it is not None:
                    shutdown = getattr(internal_it, "_shutdown_workers", None)
                    if callable(shutdown):
                        shutdown()
                    dl._iterator = None
            except Exception:
                logger.warning("failed to shut down %s", attr, exc_info=True)
            finally:
                setattr(self, attr, None)

    def on_step_begin(self) -> None:
        return None

    def on_step_end(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            futures = [
                self._policy_pool.submit(self._update_weights_one_policy, policy_name)
                for policy_name in self.policy_trainers
            ]
            timing_raw: dict[str, Any] = {}
            for future in futures:
                _, per_policy_timing = future.result()
                timing_raw.update(per_policy_timing)
            self.timing_raw.update(timing_raw)
            return
        # separate_async: delegate to verl's per-policy on_step_end (standalone
        # checkpoint manager; update_weights contains abort/resume internally).
        # Dynamic inference borrows on fresh metrics; the step boundary only
        # retries returns, updates home weights, and refreshes active guests.
        if self._dynamic_inference is not None:
            self._dynamic_inference.run_boundary(self)
            return
        futures = [
            self._policy_pool.submit(self._run_trainer_hook, policy_name, "on_step_end")
            for policy_name in self.policy_trainers
        ]
        for future in futures:
            future.result()

    def _run_trainer_hook(self, policy_name: str, hook_name: str) -> None:
        """Run one policy trainer's verl lifecycle hook (e.g. on_step_end/on_sample_end)."""
        trainer = self.policy_trainers[policy_name]
        hook = getattr(trainer, hook_name, None)
        if callable(hook):
            hook()

    def _update_weights_one_policy(self, policy_name: str) -> tuple[None, dict[str, Any]]:
        """Wake one policy's vLLM replicas with the current weights."""
        from verl.utils.debug import marked_timer

        timing_raw: dict[str, Any] = {}
        trainer = self.policy_trainers[policy_name]
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if checkpoint_manager is None:
            return None, timing_raw
        update_weights = getattr(checkpoint_manager, "update_weights", None)
        if not callable(update_weights):
            return None, timing_raw
        with marked_timer(f"{policy_name}/update_weights", timing_raw, color="red"):
            update_weights(self.global_steps)
        return None, timing_raw

    def on_sample_begin(self) -> None:
        return None

    def on_sample_end(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            from verl.utils.debug import marked_timer

            for policy_name, trainer in self.policy_trainers.items():
                checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
                if checkpoint_manager is None:
                    continue
                sleep_replicas = getattr(checkpoint_manager, "sleep_replicas", None)
                if not callable(sleep_replicas):
                    continue
                with marked_timer(f"{policy_name}/sleep_replicas", self.timing_raw, color="red"):
                    sleep_replicas()
            return
        # separate_async: delegate to verl's per-policy on_sample_end
        # (hybrid replicas switch to trainer mode; standalone replicas keep running).
        futures = [
            self._policy_pool.submit(self._run_trainer_hook, policy_name, "on_sample_end")
            for policy_name in self.policy_trainers
        ]
        for future in futures:
            future.result()

    @staticmethod
    def _call_v1_stage(method, batch, metrics: dict[str, Any], **kwargs):
        result = method(batch, metrics=metrics, **kwargs)
        return batch if result is None else result

    @staticmethod
    def _prefix_metrics(metrics: dict[str, Any], policy_name: str, policy_metrics: dict[str, Any]) -> None:
        for key, value in policy_metrics.items():
            metrics[f"{policy_name}/{key}"] = value

    @staticmethod
    def _make_batch_like(template, *, keys: list[str], tags: list[dict[str, Any]]):
        kwargs = {
            "partition_id": getattr(template, "partition_id", "train"),
            "keys": list(keys),
            "tags": [dict(tag) for tag in tags],
        }
        if hasattr(template, "fields"):
            kwargs["fields"] = getattr(template, "fields")
        if hasattr(template, "extra_info"):
            kwargs["extra_info"] = dict(getattr(template, "extra_info") or {})

        try:
            return template.__class__(**kwargs)
        except TypeError:
            try:
                from transfer_queue import KVBatchMeta
            except ImportError:
                from verl.utils.transferqueue_utils import KVBatchMeta

            return KVBatchMeta(**kwargs)

    def _merge_policy_batches(self, per_policy_batches: Mapping[str, Any]):
        merged_keys: list[str] = []
        merged_tags: list[dict[str, Any]] = []
        template = None
        for policy_name in self.policy_trainers:
            batch = per_policy_batches.get(policy_name)
            if batch is None:
                continue
            if template is None:
                template = batch
            merged_keys.extend(list(batch.keys))
            merged_tags.extend([dict(tag) for tag in batch.tags])
        if template is None:
            raise ValueError("Cannot merge an empty per-policy batch mapping")
        return self._make_batch_like(template, keys=merged_keys, tags=merged_tags)

    def cleanup(self) -> None:
        """Release Ray resources held by this trainer and its agent framework.

        Framework shutdown is a prerequisite because active rollouts still
        access policy and Gateway resources. Later cleanup steps are
        best-effort and isolated from one another:

        1. stop framework background rollouts and remote MAS tasks;
        2. per-policy v1 trainer cleanup (no-op with current verl, kept for
           forward compatibility);
        3. remove per-policy placement groups (frees vLLM/worker GPU actors);
        4. shut down gateway actors owned by the agent framework runtime.
        """
        framework = (
            getattr(self.agent_loop_manager, "framework", None)
            if self.agent_loop_manager is not None
            else None
        )
        shutdown_framework = getattr(framework, "shutdown", None)
        if callable(shutdown_framework):
            try:
                shutdown_framework()
            except Exception as exc:
                logger.warning("multi-agent framework shutdown failed: %s", exc)
                # Active rollout tasks may still access policies and Gateway.
                # Preserve those resources and surface the failure so callers
                # do not treat a partial cleanup as successful.
                raise

        # Return every active borrow, stop the poll loop and kill the guest
        # replicas before the per-policy teardown removes their placement
        # groups (guests ride the home replicas' pools but own Ray actors).
        self._shutdown_dynamic_inference()

        for trainer in self.policy_trainers.values():
            cleanup = getattr(trainer, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("cleanup failed for a policy trainer: %s", exc)

        try:
            self._remove_placement_groups(self._collect_placement_groups())
        except Exception as exc:
            logger.warning("placement group cleanup failed: %s", exc)

        try:
            self._shutdown_gateway_actors()
        except Exception as exc:
            logger.warning("gateway shutdown failed: %s", exc)

        try:
            self._policy_pool.shutdown(wait=True)
        except Exception as exc:
            logger.warning("policy thread pool shutdown failed: %s", exc)

    def _collect_placement_groups(self) -> list:
        """Collect all placement groups owned by per-policy trainers."""
        pgs = []
        for policy_trainer in self.policy_trainers.values():
            rp_mgr = getattr(policy_trainer, "resource_pool_manager", None)
            if rp_mgr is None:
                continue
            try:
                pools = rp_mgr.resource_pool_dict.values()
            except Exception:
                pools = []
            for pool in pools:
                try:
                    pgs.extend(pool.get_placement_groups())
                except Exception:
                    continue
        return pgs

    def _remove_placement_groups(self, pgs) -> None:
        """Best-effort removal of placement groups, deduplicated by PG id."""
        seen = set()
        for pg in pgs:
            pg_id = str(pg.id)
            if pg_id in seen:
                continue
            seen.add(pg_id)
            try:
                ray.util.remove_placement_group(pg)
                logger.info("removed placement group %s", pg_id)
            except Exception as exc:
                logger.warning("failed to remove placement group %s: %s", pg_id, exc)

    def _shutdown_gateway_actors(self) -> None:
        """Shut down gateway Ray actors owned by the agent framework runtime."""
        framework = getattr(self.agent_loop_manager, "framework", None) if self.agent_loop_manager is not None else None
        session_runtime = getattr(framework, "session_runtime", None)
        actors = getattr(session_runtime, "owned_gateway_actors", None) or []
        for gateway in actors:
            try:
                ray.get(gateway.shutdown.remote())
                logger.info("shut down gateway actor")
            except Exception as exc:
                logger.warning("failed to shutdown gateway actor: %s", exc)
        if session_runtime is not None:
            session_runtime.owned_gateway_actors = []
            session_runtime.gateway_manager = None

__all__ = ["MultiAgentsPPOTrainer"]
