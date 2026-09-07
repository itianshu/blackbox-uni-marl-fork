from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import partial
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf
import torch
import yaml
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.transferqueue_utils import tq
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask

from .multi_modal_postprocess import compute_multi_modal_inputs, compute_position_ids
from .types import SessionHandle, SessionRuntime, Trajectory

logger = logging.getLogger(__name__)


class AgentFramework(ABC):
    """Abstract base for framework implementations.

    Phase A: entry.py owns session runtime construction and passes it in.
    Subclasses receive shared entry resources plus the raw config for
    subclass-specific field parsing.

    Phase B: trainer inlines entry; this from_config contract remains.
    """

    @classmethod
    @abstractmethod
    async def from_config(
        cls,
        *,
        config,
        session_runtime,
        processor=None,
        policy_processors=None,
        replay_buffer,
        reward_loop_worker_handles=None,
    ) -> "AgentFramework":
        ...

    @abstractmethod
    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run agent sessions and write finalized trajectories to TransferQueue."""
        ...


def _short_failure_reason(error: BaseException) -> str:
    message = str(error)
    if not message:
        message = error.__class__.__name__
    return message[:512]


_TQ_NESTED_SEQUENCE_FIELDS = {
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
    "teacher_logprobs",
    "teacher_ids",
}


def _list_of_tq_fields_to_tensordict(fields: list[dict[str, object]]) -> TensorDict:
    td = tu.list_of_dict_to_tensordict(fields)
    for key in _TQ_NESTED_SEQUENCE_FIELDS:
        if key not in fields[0]:
            continue
        values = [field[key] for field in fields]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        ragged_idx = 2 if key == "position_ids" and values[0].dim() == 2 else None
        td[key] = tu.nested_tensor_from_tensor_list(values, ragged_idx=ragged_idx)
    return td


def _trajectory_to_reward_dataproto(trajectory, sample_fields):
    """Build a single-sample DataProto for RewardLoopWorker.compute_score.

    Field shape matches AgentLoopWorker._compute_score
    (verl/experimental/agent_loop/agent_loop.py:753-772). Only fields actually
    consumed by NaiveRewardManager.run_single / RewardLoopWorker dispatch are
    populated; tool_extra_fields / num_turns are passed via non_tensor_batch
    for parity.
    """
    import numpy as np
    from verl.protocol import DataProto

    prompt_ids = torch.tensor(trajectory.prompt_ids, dtype=torch.long).unsqueeze(0)
    response_ids = torch.tensor(trajectory.response_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )

    non_tensor_batch: dict[str, object] = {}
    for key in ("raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
        if key in sample_fields:
            non_tensor_batch[key] = np.array([sample_fields[key]], dtype=object)
    non_tensor_batch["__num_turns__"] = np.array([trajectory.num_turns])

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _load_yaml_file(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class OpenAICompatibleAgentFramework(AgentFramework):
    """Reference AgentFramework implementation for OpenAI-compatible agent loops.

    Each sample in the batch is run as an independent session: the agent
    communicates with the Gateway via standard ``/v1/chat/completions``
    requests, and the Gateway collects token-level trajectories.  After
    finalization, ``_score_trajectories`` dispatches the session's final
    trajectory to a RewardLoopWorker and broadcasts the score back to all
    trajectories in the session (matching
    ``AgentLoopWorkerTQ._agent_loop_postprocess``); the framework then writes
    them to the TransferQueue schema consumed by sync training.
    """

    def __init__(
        self,
        session_runtime: SessionRuntime,
        agent_runner,
        *,
        reward_loop_worker_handles=None,
        processor=None,
        replay_buffer=None,
        rollout_config=None,
        completion_timeout: float | None = 30.0,
        wait_for_completion_after_agent_run: bool = False,
        max_concurrent_sessions: int = 0,
    ):
        self.session_runtime = session_runtime
        self.agent_runner = agent_runner
        self.reward_loop_worker_handles = list(reward_loop_worker_handles) if reward_loop_worker_handles else None
        self._processor = processor
        # These are optional so direct framework construction and adapter-based
        # construction can share the same implementation.
        self._replay_buffer = replay_buffer
        self._rollout_config = rollout_config
        self.completion_timeout = completion_timeout
        self.wait_for_completion_after_agent_run = wait_for_completion_after_agent_run
        self._max_concurrent_sessions = max_concurrent_sessions
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    async def from_config(
        cls,
        *,
        config,
        session_runtime,
        processor=None,
        policy_processors=None,
        replay_buffer,
        reward_loop_worker_handles=None,
    ) -> "OpenAICompatibleAgentFramework":
        del policy_processors
        # Current recipes keep framework settings under rollout.custom.agent_framework.
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        agent_runner_fqn = af_cfg.get("agent_runner_fqn")
        if not agent_runner_fqn:
            raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.agent_runner_fqn is required")

        agent_runner = load_class_from_fqn(str(agent_runner_fqn), description="agent runner")
        runner_kwargs = dict(
            OmegaConf.to_container(OmegaConf.create(af_cfg.get("agent_runner_kwargs", {})), resolve=True) or {}
        )
        tool_config_path = af_cfg.get("tool_config_path")
        if tool_config_path:
            tool_config = initialize_tools_from_config(tool_config_path)
            if not tool_config:
                raise ValueError(f"tool config did not initialize any tools: {tool_config_path}")
            runner_kwargs["tool_config"] = tool_config
        if runner_kwargs:
            agent_runner = partial(agent_runner, **runner_kwargs)

        completion_timeout = af_cfg.get("completion_timeout_seconds")
        return cls(
            session_runtime=session_runtime,
            agent_runner=agent_runner,
            reward_loop_worker_handles=reward_loop_worker_handles,
            processor=processor,
            replay_buffer=replay_buffer,
            rollout_config=config.actor_rollout_ref.rollout,
            completion_timeout=completion_timeout,
            wait_for_completion_after_agent_run=completion_timeout is not None,
            max_concurrent_sessions=int(af_cfg.get("max_concurrent_sessions", 0)),
        )

    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run rollout-manager generation and write outputs into TransferQueue."""
        if self._replay_buffer is None and self._rollout_config is None:
            raise RuntimeError("OpenAICompatibleAgentFramework requires replay_buffer or rollout_config for generate_sequences")
        if self._rollout_config is None:
            raise RuntimeError("OpenAICompatibleAgentFramework requires rollout_config for generate_sequences")

        global_steps = tu.get(prompts, "global_steps")
        if global_steps is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['global_steps']")

        partition_id = "val" if "validate" in prompts.keys() else "train"
        if partition_id == "val":
            val_kwargs = self._rollout_config.get("val_kwargs", {})
            num_sessions = int(val_kwargs.get("n"))
        else:
            num_sessions = int(self._rollout_config.get("n"))

        uids = tu.get(prompts, "uid")
        if uids is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['uid'] for replay_buffer")
        uid_values = uids.tolist() if hasattr(uids, "tolist") else list(uids)
        if self._replay_buffer is not None:
            self._replay_buffer.add(
                partition_id,
                {str(uid): {"global_steps": global_steps, "status": "running"} for uid in uid_values},
            )

        stats = await self._run_batch_to_tq(
            prompts,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        logger.info(
            "generate_sequences summary: num_input_prompts=%s num_success_sessions=%s "
            "num_failed_sessions=%s num_success_outputs=%s num_failed_uids=%s failure_reasons=%s",
            stats["num_input_prompts"],
            stats["num_success_sessions"],
            stats["num_failed_sessions"],
            stats["num_success_outputs"],
            stats["num_failed_uids"],
            stats["failure_reasons"][:3],
        )
        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )
        return None

    async def _run_batch_to_tq(
        self,
        prompts: TensorDict,
        *,
        global_steps: int,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict:
        """Run all prompts in a batch and aggregate prompt/session stats."""
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")

        raw_prompts = tu.get(prompts, "raw_prompt")
        if raw_prompts is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['raw_prompt']")

        # Batch layer: each sample/prompt owns its own group of rollout.n sessions.
        # Prompt tasks are isolated so one prompt failure does not drop the whole batch.
        tasks = [
            self._run_prompt_sessions_to_tq(
                prompts=prompts,
                raw_prompt=raw_prompts[sample_index],
                sample_index=sample_index,
                global_steps=global_steps,
                partition_id=partition_id,
                num_sessions=num_sessions,
            )
            for sample_index in range(len(prompts))
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        failure_reasons: list[str] = []
        stats = {
            "num_input_prompts": len(prompts),
            "num_success_sessions": 0,
            "num_failed_sessions": 0,
            "num_success_outputs": 0,
            "num_failed_uids": 0,
            "failure_reasons": failure_reasons,
        }
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                stats["num_failed_sessions"] += num_sessions
                stats["num_failed_uids"] += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            stats["num_success_sessions"] += outcome["num_success_sessions"]
            stats["num_failed_sessions"] += outcome["num_failed_sessions"]
            stats["num_success_outputs"] += outcome["num_success_outputs"]
            stats["num_failed_uids"] += outcome["num_failed_uids"]
            failure_reasons.extend(outcome["failure_reasons"])
        return stats

    async def _run_prompt_sessions_to_tq(
        self,
        *,
        prompts: TensorDict,
        raw_prompt,
        sample_index: int,
        global_steps: int,
        partition_id: str,
        num_sessions: int,
    ) -> dict:
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        uid = sample_fields.get("uid")
        if uid is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['uid'] for TransferQueue output")
        uid = str(uid)

        # Prompt layer: rollout.n sessions race independently for the same uid.
        # Successful sessions are written to TQ; failed sessions only affect this uid's stats.
        tasks = [
            self._run_session_with_concurrency_limit(
                prompts=prompts,
                raw_prompt=raw_prompt,
                sample_index=sample_index,
                session_id=f"session-{sample_index}-{session_index}-{uuid4().hex}",
                runner_kwargs={
                    key: sample_fields[key]
                    for key in ("tools_kwargs", "agent_name")
                    if key in sample_fields
                },
            )
            for session_index in range(num_sessions)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        success_sessions = 0
        failed_sessions = 0
        success_outputs = 0
        failure_reasons: list[str] = []
        for session_index, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed_sessions += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue

            trajectories, session_sample_fields = outcome
            if not trajectories:
                failed_sessions += 1
                failure_reasons.append(f"empty trajectories for uid={uid} session_index={session_index}")
                continue

            success_sessions += 1
            await self._write_session_trajectories_to_tq(
                uid=uid,
                session_index=session_index,
                trajectories=trajectories,
                sample_fields=session_sample_fields,
                global_steps=global_steps,
                partition_id=partition_id,
            )
            success_outputs += len(trajectories)

        if success_sessions > 0:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
            failed_uids = 0
        else:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
            failed_uids = 1

        return {
            "num_success_sessions": success_sessions,
            "num_failed_sessions": failed_sessions,
            "num_success_outputs": success_outputs,
            "num_failed_uids": failed_uids,
            "failure_reasons": failure_reasons,
        }

    async def _run_session_with_concurrency_limit(
        self,
        *,
        prompts: TensorDict,
        raw_prompt,
        sample_index: int,
        session_id: str | None = None,
        runner_kwargs: dict[str, object] | None = None,
    ) -> tuple[list[Trajectory], dict[str, object]]:
        if self._max_concurrent_sessions <= 0:
            return await self._run_session(
                prompts=prompts,
                raw_prompt=raw_prompt,
                sample_index=sample_index,
                session_id=session_id,
                runner_kwargs=runner_kwargs,
            )
        # Lazy-init Semaphore on first use and rebind if the running loop
        # changed: asyncio.Semaphore binds to the loop at construction, but
        # Ray actors may run sessions on a different loop than __init__.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrent_sessions)
            self._semaphore_loop = loop
        async with self._semaphore:
            return await self._run_session(
                prompts=prompts,
                raw_prompt=raw_prompt,
                sample_index=sample_index,
                session_id=session_id,
                runner_kwargs=runner_kwargs,
            )

    async def _run_session(
        self,
        *,
        prompts: TensorDict,
        raw_prompt,
        sample_index: int,
        session_id: str | None = None,
        runner_kwargs: dict[str, object] | None = None,
    ) -> tuple[list[Trajectory], dict[str, object]]:
        """Run one gateway session lifecycle and return finalized trajectories."""
        session_id = session_id or f"session-{sample_index}-0-{uuid4().hex}"
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        session = self._prepare_session_handle(await self.session_runtime.create_session(session_id))
        try:
            await self.agent_runner(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                session_runtime=self.session_runtime,
                **(runner_kwargs or {}),
            )
            if self.wait_for_completion_after_agent_run:
                await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
            session_trajectories = await self.session_runtime.finalize_session(session_id)
        except Exception:
            await self.session_runtime.abort_session(session_id)
            raise

        # Score the session's trajectories immediately after finalization,
        # consistent with VERL's per-sample reward path.
        if not self.reward_loop_worker_handles or not session_trajectories:
            return session_trajectories, sample_fields

        annotations = await self._score_trajectories(session_trajectories, sample_fields)
        scored_trajectories = []
        for traj, (score, extra) in zip(session_trajectories, annotations, strict=True):
            scored_trajectories.append(
                replace(
                    traj,
                    reward_score=score,
                    extra_fields={**traj.extra_fields, "reward_extra_info": extra},
                )
            )
        return scored_trajectories, sample_fields

    def _prepare_session_handle(self, session: SessionHandle) -> SessionHandle:
        """Adapt the gateway session handle for the agent's client protocol.

        The gateway hands out base_url ending in ``/v1`` (the OpenAI SDK
        convention).  Subclasses targeting other API protocols can override
        this to reshape the handle before it reaches the agent runner.
        """
        return session

    async def _score_trajectories(
        self,
        session_trajectories: list[Trajectory],
        sample_fields: dict[str, object],
    ) -> list[tuple[float, dict[str, object]]]:
        """Score the session's final trajectory and broadcast (score, extra_info) to all.

        Mirrors AgentLoopWorkerTQ._agent_loop_postprocess
        (verl/trainer/main_ppo_sync.py:353-396): only the final trajectory (the
        session's last interaction segment) is dispatched to RewardLoopWorker;
        its score + reward_extra_info are then broadcast to every trajectory in
        the session. Subclasses can override this method to implement custom
        session-to-trajectory scoring policies.
        """
        assert self.reward_loop_worker_handles is not None
        assert session_trajectories, "expected non-empty session_trajectories"

        final_trajectory = session_trajectories[-1]
        data = _trajectory_to_reward_dataproto(final_trajectory, sample_fields)
        worker = random.choice(self.reward_loop_worker_handles)
        result = await worker.compute_score.remote(data)

        if "reward_score" not in result:
            raise ValueError(
                f"RewardLoopWorker result missing 'reward_score' key for uid={sample_fields.get('uid')}"
            )
        score = float(result["reward_score"])
        extra = dict(result.get("reward_extra_info") or {})
        return [(score, extra)] * len(session_trajectories)

    def _extract_sample_fields(self, *, prompts: TensorDict, sample_index: int) -> dict[str, object]:
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value if value.ndim == 0 else value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data
        return sample_fields

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_index: int,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int,
        partition_id: str,
    ) -> None:
        keys = []
        fields = []
        tags = []
        for index, trajectory in enumerate(trajectories):
            field, tag = self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=sample_fields,
                session_index=session_index,
                global_steps=global_steps,
                uid=uid,
            )
            keys.append(f"{uid}_{session_index}_{index}")
            fields.append(field)
            tags.append(tag)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=_list_of_tq_fields_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
        session_index: int,
        global_steps: int,
        uid: str = "",
    ) -> tuple[dict[str, object], dict[str, object]]:
        prompts = torch.tensor(trajectory.prompt_ids, dtype=torch.long)
        responses = torch.tensor(trajectory.response_ids, dtype=torch.long)
        response_mask = torch.tensor(trajectory.response_mask, dtype=torch.long)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        multi_modal_inputs = compute_multi_modal_inputs(
            self._processor,
            input_ids.unsqueeze(0),
            trajectory.multi_modal_data,
        )
        if self._processor is None:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        else:
            position_ids = compute_position_ids(
                self._processor,
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)

        field: dict[str, object] = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "loss_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }
        if trajectory.response_logprobs is not None:
            field["rollout_log_probs"] = torch.tensor(trajectory.response_logprobs, dtype=torch.float32)
        else:
            field["rollout_log_probs"] = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.routed_experts is not None:
            field["routed_experts"] = (
                torch.from_numpy(trajectory.routed_experts.copy())
                if hasattr(trajectory.routed_experts, "copy") and not isinstance(trajectory.routed_experts, torch.Tensor)
                else trajectory.routed_experts
            )
        rm_scores = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.reward_score is not None and responses.numel() > 0:
            rm_scores[-1] = float(trajectory.reward_score)
        field["rm_scores"] = rm_scores

        field.update(trajectory.extra_fields)
        field.pop("multi_modal_data", None)
        for key in ("uid", "raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
            if key in sample_fields:
                field[key] = sample_fields[key]
        field["session_id"] = session_index
        field["global_steps"] = global_steps
        field["num_turns"] = torch.tensor(int(trajectory.num_turns), dtype=torch.long)

        prompt_len = prompts.size(0)
        response_len = responses.size(0)
        tag = {
            "global_steps": global_steps,
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
            "uid": uid,
        }
        return field, tag


class AnthropicCompatibleAgentFramework(OpenAICompatibleAgentFramework):
    """AgentFramework implementation for Anthropic-API agent loops.

    Orchestration (session lifecycle, scoring, TransferQueue output) is
    identical to ``OpenAICompatibleAgentFramework``; the agent instead talks
    to the Gateway via the Anthropic Messages API
    (``/sessions/{session_id}/v1/messages``), which the gateway converts to
    its internal OpenAI message format for chat templating and token-level
    trajectory collection.

    The session handle's ``base_url`` is reshaped for the Anthropic SDK: the
    OpenAI convention embeds the ``/v1`` suffix in base_url, while
    ``anthropic.Anthropic(base_url=...)`` appends ``/v1/messages`` itself.
    Agent runners can therefore pass ``session.base_url`` directly to
    ``anthropic.Anthropic`` / ``AsyncAnthropic`` (any ``api_key`` works; the
    gateway does not authenticate).
    """

    def _prepare_session_handle(self, session: SessionHandle) -> SessionHandle:
        if session.base_url and session.base_url.endswith("/v1"):
            return replace(session, base_url=session.base_url[: -len("/v1")])
        return session


class MultiAgentFramework(AgentFramework):
    """AgentFramework implementation for external multi-agent blackbox rollouts.

    This class owns the multi-agent rollout lifecycle. Trainer-level multi-policy
    batch construction and PPO update orchestration are intentionally kept out of
    this framework implementation.
    """

    def __init__(
        self,
        session_runtime: SessionRuntime,
        multi_agent_runner,
        *,
        role_policy_mapping: dict[str, str],
        reward_loop_worker_handles=None,
        processor=None,
        policy_processors: dict[str, object] | None = None,
        rollout_config=None,
        completion_timeout: float | None = 30.0,
        wait_for_completion_after_runner: bool = False,
        max_concurrent_rollouts: int = 0,
    ):
        self.session_runtime = session_runtime
        self.multi_agent_runner = multi_agent_runner
        self.role_policy_mapping = dict(role_policy_mapping)
        self.reward_loop_worker_handles = list(reward_loop_worker_handles) if reward_loop_worker_handles else None
        self._processor = processor
        self._policy_processors = dict(policy_processors or {})
        self._rollout_config = rollout_config
        self.completion_timeout = completion_timeout
        self.wait_for_completion_after_runner = wait_for_completion_after_runner
        self._max_concurrent_rollouts = max_concurrent_rollouts
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_thread: "threading.Thread | None" = None
        self._bg_tasks: set = set()
        self._bg_drain_future: concurrent.futures.Future | None = None

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @classmethod
    async def from_config(
        cls,
        *,
        config,
        session_runtime,
        processor=None,
        policy_processors=None,
        replay_buffer=None,
        reward_loop_worker_handles=None,
    ) -> "MultiAgentFramework":
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        runner_fqn = af_cfg.get("multi_agent_runner_fqn") or af_cfg.get("agent_runner_fqn")
        if not runner_fqn:
            raise ValueError(
                "actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_fqn is required"
            )

        role_policy_mapping = dict(
            OmegaConf.to_container(OmegaConf.create(af_cfg.get("role_policy_mapping", {})), resolve=True) or {}
        )
        if not role_policy_mapping:
            raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping is required")

        multi_agent_runner = load_class_from_fqn(str(runner_fqn), description="multi-agent runner")
        runner_kwargs = dict(
            OmegaConf.to_container(OmegaConf.create(af_cfg.get("multi_agent_runner_kwargs", {})), resolve=True) or {}
        )
        mas_config_path = runner_kwargs.get("mas_config_path")
        if mas_config_path and "mas_config" not in runner_kwargs:
            runner_kwargs["mas_config"] = _load_yaml_file(str(mas_config_path))
        if runner_kwargs:
            multi_agent_runner = partial(multi_agent_runner, **runner_kwargs)

        completion_timeout = af_cfg.get("completion_timeout_seconds")
        return cls(
            session_runtime=session_runtime,
            multi_agent_runner=multi_agent_runner,
            role_policy_mapping=role_policy_mapping,
            reward_loop_worker_handles=reward_loop_worker_handles,
            processor=processor,
            policy_processors=policy_processors,
            rollout_config=config.actor_rollout_ref.rollout,
            completion_timeout=completion_timeout,
            wait_for_completion_after_runner=completion_timeout is not None,
            max_concurrent_rollouts=int(af_cfg.get("max_concurrent_rollouts", 0)),
        )

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        """Return a long-lived event loop owned by this framework instance.

        fire-and-forget rollout tasks run here; a fresh auto_await loop would
        close after generate_sequences returns and kill the tasks.
        """
        if self._bg_loop is None:
            self._bg_loop = asyncio.new_event_loop()
            self._bg_thread = threading.Thread(
                target=self._bg_loop.run_forever,
                daemon=True,
                name="ma-framework-bg",
            )
            self._bg_thread.start()
        return self._bg_loop

    async def _run_batch_to_tq_guarded(self, prompts, *, global_steps, partition_id, num_rollouts) -> None:
        try:
            stats = await self._run_batch_to_tq(
                prompts,
                global_steps=global_steps,
                partition_id=partition_id,
                num_rollouts=num_rollouts,
            )
            logger.info(
                "multi-agent generate_sequences summary: num_input_prompts=%s num_success_rollouts=%s "
                "num_failed_rollouts=%s num_success_outputs=%s num_failed_uids=%s failure_reasons=%s",
                stats["num_input_prompts"],
                stats["num_success_rollouts"],
                stats["num_failed_rollouts"],
                stats["num_success_outputs"],
                stats["num_failed_uids"],
                stats["failure_reasons"][:3],
            )
            if stats["num_success_outputs"] == 0:
                raise RuntimeError(
                    f"All multi-agent rollouts failed at global_steps={global_steps}. "
                    f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
                )
        except Exception as exc:
            try:
                uids = tu.get(prompts, "uid")
                if uids is not None:
                    for uid in uids.tolist() if hasattr(uids, "tolist") else list(uids):
                        await tq.async_kv_put(
                            key=str(uid),
                            partition_id=partition_id,
                            tag={"status": "failure"},
                        )
            except Exception:
                pass
            logger.exception("background multi-agent rollout batch failed: %s", exc)

    async def generate_sequences(self, prompts) -> None:
        """Run multi-agent rollouts and write finalized role trajectories to TransferQueue."""
        if self._rollout_config is None:
            raise RuntimeError("MultiAgentFramework requires rollout_config for generate_sequences")

        global_steps = tu.get(prompts, "global_steps")
        if global_steps is None:
            raise ValueError("MultiAgentFramework requires prompts['global_steps']")
        global_steps = self._scalar(global_steps)

        partition_id = "val" if "validate" in prompts.keys() else "train"
        if partition_id == "val":
            val_kwargs = self._rollout_config.get("val_kwargs", {})
            num_rollouts = int(val_kwargs.get("n"))
        else:
            num_rollouts = int(self._rollout_config.get("n"))

        if len(prompts) == 0:
            raise ValueError("generate_sequences requires a non-empty batch")
        if num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
        if tu.get(prompts, "raw_prompt") is None:
            raise ValueError("MultiAgentFramework requires prompts['raw_prompt']")
        if tu.get(prompts, "uid") is None:
            raise ValueError("MultiAgentFramework requires prompts['uid']")

        loop = self._ensure_background_loop()
        task = asyncio.run_coroutine_threadsafe(
            self._run_batch_to_tq_guarded(
                prompts,
                global_steps=global_steps,
                partition_id=partition_id,
                num_rollouts=num_rollouts,
            ),
            loop,
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return None

    async def _run_batch_to_tq(
        self,
        prompts,
        *,
        global_steps: int,
        partition_id: str,
        num_rollouts: int,
    ) -> dict:
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")

        raw_prompts = tu.get(prompts, "raw_prompt")
        if raw_prompts is None:
            raise ValueError("MultiAgentFramework requires prompts['raw_prompt']")

        uid_to_status = {}
        sample_fields_by_index = []
        for sample_index in range(len(prompts)):
            sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
            uid = sample_fields.get("uid")
            if uid is None:
                raise ValueError("MultiAgentFramework requires prompts['uid'] for TransferQueue output")
            uid_to_status[str(uid)] = {"global_steps": global_steps, "status": "running"}
            sample_fields_by_index.append(sample_fields)

        for uid, tag in uid_to_status.items():
            await tq.async_kv_put(
                key=uid,
                partition_id=partition_id,
                tag=dict(tag),
            )

        tasks = [
            self._run_prompt_rollouts_to_tq(
                raw_prompt=raw_prompts[sample_index],
                sample_fields=sample_fields_by_index[sample_index],
                sample_index=sample_index,
                global_steps=global_steps,
                partition_id=partition_id,
                num_rollouts=num_rollouts,
            )
            for sample_index in range(len(prompts))
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        failure_reasons: list[str] = []
        stats = {
            "num_input_prompts": len(prompts),
            "num_success_rollouts": 0,
            "num_failed_rollouts": 0,
            "num_success_outputs": 0,
            "num_failed_uids": 0,
            "failure_reasons": failure_reasons,
        }
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                stats["num_failed_rollouts"] += num_rollouts
                stats["num_failed_uids"] += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            stats["num_success_rollouts"] += outcome["num_success_rollouts"]
            stats["num_failed_rollouts"] += outcome["num_failed_rollouts"]
            stats["num_success_outputs"] += outcome["num_success_outputs"]
            stats["num_failed_uids"] += outcome["num_failed_uids"]
            failure_reasons.extend(outcome["failure_reasons"])
        return stats

    async def _run_prompt_rollouts_to_tq(
        self,
        *,
        raw_prompt,
        sample_fields: dict[str, object],
        sample_index: int,
        global_steps: int,
        partition_id: str,
        num_rollouts: int,
    ) -> dict:
        uid = str(sample_fields["uid"])

        tasks = [
            self._run_rollout_with_concurrency_limit(
                raw_prompt=raw_prompt,
                rollout_id=f"multi-agent-rollout-{sample_index}-{sample_idx}-{uuid4().hex}",
                sample_index=sample_index,
                sample_fields=sample_fields,
                sample_idx=sample_idx,
                runner_kwargs={
                    key: sample_fields[key]
                    for key in ("tools_kwargs", "agent_name")
                    if key in sample_fields
                },
            )
            for sample_idx in range(num_rollouts)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        success_rollouts = 0
        failed_rollouts = 0
        success_outputs = 0
        failure_reasons: list[str] = []
        for sample_idx, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed_rollouts += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            rollout_result, trajectories = outcome
            if not trajectories:
                failed_rollouts += 1
                failure_reasons.append(f"empty trajectories for uid={uid} sample_idx={sample_idx}")
                continue

            success_rollouts += 1
            await self._write_multi_agent_rollout_to_tq(
                uid=uid,
                sample_idx=sample_idx,
                rollout_id=rollout_result.rollout_id,
                trajectories=trajectories,
                sample_fields=sample_fields,
                global_steps=global_steps,
                partition_id=partition_id,
            )
            success_outputs += len(trajectories)

        if success_rollouts > 0:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
            failed_uids = 0
        else:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
            failed_uids = 1

        return {
            "num_success_rollouts": success_rollouts,
            "num_failed_rollouts": failed_rollouts,
            "num_success_outputs": success_outputs,
            "num_failed_uids": failed_uids,
            "failure_reasons": failure_reasons,
        }

    async def _run_rollout_with_concurrency_limit(
        self,
        *,
        raw_prompt,
        rollout_id: str,
        sample_index: int,
        sample_fields: dict[str, object],
        sample_idx: int,
        runner_kwargs: dict[str, object] | None = None,
    ) -> tuple[object, list[Trajectory]]:
        async def run_and_annotate() -> tuple[object, list[Trajectory]]:
            rollout_result = await self.run_rollout(
                raw_prompt=raw_prompt,
                rollout_id=rollout_id,
                sample_index=sample_index,
                runner_kwargs=runner_kwargs,
            )
            if not rollout_result.trajectories:
                return rollout_result, []
            trajectories = await self._annotate_rollout_trajectories(
                rollout_result=rollout_result,
                sample_fields=sample_fields,
                sample_idx=sample_idx,
            )
            return rollout_result, trajectories

        if self._max_concurrent_rollouts <= 0:
            return await run_and_annotate()
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrent_rollouts)
            self._semaphore_loop = loop
        async with self._semaphore:
            return await run_and_annotate()

    async def run_rollout(
        self,
        *,
        raw_prompt,
        rollout_id: str | None = None,
        sample_index: int = 0,
        runner_kwargs: dict[str, object] | None = None,
    ):
        """Run one external MAS rollout and return finalized Gateway trajectories."""
        rollout_id = rollout_id or f"multi-agent-rollout-{sample_index}-{uuid4().hex}"
        rollout = None
        create_task = asyncio.create_task(
            self.session_runtime.create_multi_agent_rollout(
                rollout_id,
                role_policy_mapping=self.role_policy_mapping,
            )
        )
        try:
            # Shield the Gateway RPC so cancellation cannot leave a remotely
            # created rollout without first recovering its manager mapping.
            rollout = await asyncio.shield(create_task)
            runner_result = await self._execute_multi_agent_runner(
                raw_prompt=raw_prompt,
                rollout=rollout,
                rollout_id=rollout_id,
                sample_index=sample_index,
                runner_kwargs=runner_kwargs,
            )
            reward_info = None
            if isinstance(runner_result, dict):
                reward_info = runner_result.get("reward_info")
            if reward_info is not None:
                await self.session_runtime.complete_multi_agent_rollout(rollout_id, reward_info=reward_info)
            if self.wait_for_completion_after_runner:
                await self.session_runtime.wait_for_multi_agent_rollout_completion(
                    rollout_id,
                    timeout=self.completion_timeout,
                )
            return await self.session_runtime.finalize_multi_agent_rollout(rollout_id)
        except asyncio.CancelledError:
            if rollout is None and not create_task.cancelled():
                try:
                    rollout = await create_task
                except Exception:
                    logger.warning(
                        "Gateway rollout creation failed while cancellation was being handled: %s",
                        rollout_id,
                        exc_info=True,
                    )
            if rollout is not None:
                await self.session_runtime.abort_multi_agent_rollout(rollout_id)
            raise
        except Exception:
            if rollout is not None:
                await self.session_runtime.abort_multi_agent_rollout(rollout_id)
            raise

    async def _execute_multi_agent_runner(
        self,
        *,
        raw_prompt,
        rollout,
        rollout_id: str,
        sample_index: int,
        runner_kwargs: dict[str, object] | None,
    ):
        """Execute the configured runner; recipes may override the process boundary."""
        del rollout_id
        return await self.multi_agent_runner(
            raw_prompt=raw_prompt,
            rollout=rollout,
            sample_index=sample_index,
            session_runtime=self.session_runtime,
            role_policy_mapping=self.role_policy_mapping,
            **(runner_kwargs or {}),
        )
        
    async def _annotate_rollout_trajectories(
        self,
        *,
        rollout_result,
        sample_fields: dict[str, object],
        sample_idx: int,
    ) -> list[Trajectory]:
        reward_score = self._reward_score_from_info(rollout_result.reward_info)
        reward_extra_info = dict(rollout_result.reward_info.get("reward_extra_info") or {})

        if reward_score is None and self.reward_loop_worker_handles and rollout_result.trajectories:
            final_trajectory = rollout_result.trajectories[-1]
            reward_sample_fields = sample_fields
            if rollout_result.reward_info:
                extra_info = dict(sample_fields.get("extra_info") or {})
                reward_sample_fields = {
                    **sample_fields,
                    "extra_info": {**extra_info, **rollout_result.reward_info},
                }
            (reward_score, reward_extra_info) = (
                await OpenAICompatibleAgentFramework._score_trajectories(
                    self,
                    [final_trajectory],
                    reward_sample_fields,
                )
            )[0]

        annotated = []
        for record_idx, trajectory in enumerate(rollout_result.trajectories):
            extra_fields = dict(trajectory.extra_fields)
            role_session_id = extra_fields.pop("session_id", None) or extra_fields.get("role_session_id")
            role = extra_fields.get("role")
            if role_session_id is not None:
                extra_fields["role_session_id"] = role_session_id
            extra_fields.setdefault("rollout_id", rollout_result.rollout_id)
            extra_fields["sample_idx"] = sample_idx
            extra_fields["record_idx"] = record_idx
            if reward_extra_info:
                extra_fields["reward_extra_info"] = reward_extra_info
            annotated.append(
                replace(
                    trajectory,
                    reward_score=reward_score if reward_score is not None else trajectory.reward_score,
                    reward_info=dict(rollout_result.reward_info),
                    extra_fields=extra_fields,
                )
            )
        return annotated

    @staticmethod
    def _reward_score_from_info(reward_info: dict[str, object]) -> float | None:
        for key in ("reward_score", "score", "reward"):
            value = reward_info.get(key)
            if value is not None:
                return float(value)
        return None

    def _extract_sample_fields(self, *, prompts, sample_index: int) -> dict[str, object]:
        return OpenAICompatibleAgentFramework._extract_sample_fields(
            self,
            prompts=prompts,
            sample_index=sample_index,
        )

    def _processor_for_policy(self, policy_name: str | None):
        if policy_name is None:
            return self._processor
        return self._policy_processors.get(policy_name, self._processor)

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
        sample_idx: int,
        record_idx: int,
        rollout_id: str,
        uid: str,
        global_steps: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        processor = self._processor_for_policy(trajectory.extra_fields.get("policy_name"))
        prompts = torch.tensor(trajectory.prompt_ids, dtype=torch.long)
        responses = torch.tensor(trajectory.response_ids, dtype=torch.long)
        response_mask = torch.tensor(trajectory.response_mask, dtype=torch.long)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        multi_modal_inputs = compute_multi_modal_inputs(
            processor,
            input_ids.unsqueeze(0),
            trajectory.multi_modal_data,
        )
        if processor is None:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        else:
            position_ids = compute_position_ids(
                processor,
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)

        field: dict[str, object] = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "loss_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }
        if trajectory.response_logprobs is not None:
            field["rollout_log_probs"] = torch.tensor(trajectory.response_logprobs, dtype=torch.float32)
        else:
            field["rollout_log_probs"] = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.routed_experts is not None:
            field["routed_experts"] = (
                torch.from_numpy(trajectory.routed_experts.copy())
                if hasattr(trajectory.routed_experts, "copy") and not isinstance(trajectory.routed_experts, torch.Tensor)
                else trajectory.routed_experts
            )
        rm_scores = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.reward_score is not None and responses.numel() > 0:
            rm_scores[-1] = float(trajectory.reward_score)
        field["rm_scores"] = rm_scores

        field.update(trajectory.extra_fields)
        field.pop("multi_modal_data", None)
        for key in ("uid", "raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
            if key in sample_fields:
                field[key] = sample_fields[key]
        field["global_steps"] = global_steps
        field["num_turns"] = torch.tensor(int(trajectory.num_turns), dtype=torch.long)
        field["rollout_id"] = rollout_id
        field["sample_idx"] = sample_idx
        field["record_idx"] = record_idx

        prompt_len = prompts.size(0)
        response_len = responses.size(0)
        tag = {
            "global_steps": global_steps,
            "min_global_steps": trajectory.extra_fields.get("min_global_steps"),
            "max_global_steps": trajectory.extra_fields.get("max_global_steps"),
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
            "uid": uid,
        }
        tag.update(
            {
                "rollout_id": rollout_id,
                "sample_idx": sample_idx,
                "record_idx": record_idx,
                "role": field.get("role"),
                "policy_name": field.get("policy_name"),
            }
        )
        return field, tag

    async def _write_multi_agent_rollout_to_tq(
        self,
        *,
        uid: str,
        sample_idx: int,
        rollout_id: str,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int,
        partition_id: str,
    ) -> None:
        keys = []
        fields = []
        tags = []
        for record_idx, trajectory in enumerate(trajectories):
            field, tag = self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=sample_fields,
                sample_idx=sample_idx,
                record_idx=record_idx,
                rollout_id=rollout_id,
                uid=uid,
                global_steps=global_steps,
            )
            keys.append(f"{uid}_{sample_idx}_{record_idx}")
            fields.append(field)
            tags.append(tag)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=_list_of_tq_fields_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

    def shutdown(self, timeout: float = 30.0) -> None:
        """Cancel background work, let coroutine cleanup finish, and close its loop."""
        loop = self._bg_loop
        if loop is None:
            return
        deadline = time.monotonic() + max(0.0, timeout)

        if threading.current_thread() is self._bg_thread:
            raise RuntimeError("MultiAgentFramework.shutdown() cannot run on its background thread")

        for background_future in list(self._bg_tasks):
            background_future.cancel()

        async def drain_cancelled_tasks() -> None:
            current_task = asyncio.current_task()
            pending_tasks = [
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        if loop.is_running():
            drain_future = getattr(self, "_bg_drain_future", None)
            if drain_future is None or drain_future.done():
                drain_future = asyncio.run_coroutine_threadsafe(drain_cancelled_tasks(), loop)
                self._bg_drain_future = drain_future
            try:
                drain_future.result(timeout=max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError(
                    "timed out draining multi-agent framework background tasks"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "failed to drain multi-agent framework background tasks"
                ) from exc
            loop.call_soon_threadsafe(loop.stop)

        if self._bg_thread is not None:
            self._bg_thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._bg_thread.is_alive():
                raise TimeoutError("timed out stopping multi-agent framework background thread")

        if not loop.is_closed():
            loop.close()
        self._bg_tasks.clear()
        self._bg_drain_future = None
        self._bg_loop = None
        self._bg_thread = None
