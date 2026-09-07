"""Example-specific multi-agent framework with isolated Ray runner tasks."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from functools import partial
from typing import Any

import ray

from uni_agent.trainer.framework.framework import MultiAgentFramework

from examples.multi_agent_blackbox.remote_runner import remote_multi_agent_run

logger = logging.getLogger(__name__)

_RAY_TASK_OPTION_KEYS = frozenset({
    "num_cpus",
    "num_gpus",
    "resources",
    "scheduling_strategy",
})


class RemoteMultiAgentFramework(MultiAgentFramework):
    """Execute each external MAS rollout in an independent Ray worker process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._remote_tasks: dict[str, Any] = {}
        self._remote_task_cancel_grace: dict[str, float] = {}

    def _resolve_runner(self) -> tuple[str, dict[str, Any]]:
        runner = self.multi_agent_runner
        runner_kwargs: dict[str, Any] = {}
        while isinstance(runner, partial):
            runner_kwargs = {**dict(runner.keywords or {}), **runner_kwargs}
            runner = runner.func
        return f"{runner.__module__}.{runner.__qualname__}", runner_kwargs

    @staticmethod
    def _extract_ray_task_config(
        runner_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        resolved_kwargs = copy.deepcopy(runner_kwargs)
        execution = resolved_kwargs.get("execution")
        if not isinstance(execution, dict):
            return resolved_kwargs, {}, 2.0

        ray_config = execution.pop("ray", None)
        if not isinstance(ray_config, dict):
            return resolved_kwargs, {}, 2.0
        unknown = set(ray_config) - _RAY_TASK_OPTION_KEYS - {"cancel_grace_seconds"}
        if unknown:
            raise ValueError(f"Unsupported external MAS Ray task options: {sorted(unknown)}")
        task_options = {
            key: value
            for key, value in ray_config.items()
            if key in _RAY_TASK_OPTION_KEYS and value is not None
        }
        cancel_grace = float(ray_config.get("cancel_grace_seconds", 2.0))
        if cancel_grace < 0:
            raise ValueError("execution.ray.cancel_grace_seconds must be non-negative")
        return resolved_kwargs, task_options, cancel_grace

    @staticmethod
    async def _cancel_remote_task(remote_ref, *, grace_seconds: float) -> None:
        ray.cancel(remote_ref, force=False)
        if grace_seconds <= 0:
            ray.cancel(remote_ref, force=True)
            return
        try:
            _, pending = await asyncio.to_thread(
                ray.wait,
                [remote_ref],
                num_returns=1,
                timeout=grace_seconds,
            )
        except Exception:
            pending = [remote_ref]
        if pending:
            ray.cancel(remote_ref, force=True)

    async def _execute_multi_agent_runner(
        self,
        *,
        raw_prompt,
        rollout,
        rollout_id: str,
        sample_index: int,
        runner_kwargs: dict[str, object] | None,
    ):
        runner_fqn, configured_kwargs = self._resolve_runner()
        resolved_kwargs = {**configured_kwargs, **(runner_kwargs or {})}
        resolved_kwargs, ray_task_options, cancel_grace = self._extract_ray_task_config(resolved_kwargs)
        remote_task = (
            remote_multi_agent_run.options(**ray_task_options)
            if ray_task_options
            else remote_multi_agent_run
        )
        remote_ref = remote_task.remote(
            runner_fqn=runner_fqn,
            raw_prompt=raw_prompt,
            rollout=rollout,
            sample_index=sample_index,
            role_policy_mapping=self.role_policy_mapping,
            runner_kwargs=resolved_kwargs,
        )
        self._remote_tasks[rollout_id] = remote_ref
        if not hasattr(self, "_remote_task_cancel_grace"):
            self._remote_task_cancel_grace = {}
        self._remote_task_cancel_grace[rollout_id] = cancel_grace
        try:
            return await asyncio.wrap_future(remote_ref.future())
        except asyncio.CancelledError:
            try:
                await self._cancel_remote_task(remote_ref, grace_seconds=cancel_grace)
            except Exception:
                logger.warning("failed to cancel remote MAS rollout %s", rollout_id, exc_info=True)
            raise
        finally:
            self._remote_tasks.pop(rollout_id, None)
            getattr(self, "_remote_task_cancel_grace", {}).pop(rollout_id, None)

    def shutdown(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        remote_refs = list(self._remote_tasks.values())
        cancellation_errors = []
        for remote_ref in remote_refs:
            try:
                ray.cancel(remote_ref, force=False)
            except Exception as exc:
                cancellation_errors.append(exc)
                logger.warning("failed to cancel remote MAS rollout during shutdown", exc_info=True)

        if cancellation_errors:
            raise RuntimeError("failed to cancel one or more remote MAS rollouts") from cancellation_errors[0]

        if remote_refs:
            grace = max(getattr(self, "_remote_task_cancel_grace", {}).values(), default=2.0)
            cooperative_timeout = min(grace, max(0.0, deadline - time.monotonic()))
            _, pending_refs = ray.wait(
                remote_refs,
                num_returns=len(remote_refs),
                timeout=cooperative_timeout,
            )
            for remote_ref in pending_refs:
                ray.cancel(remote_ref, force=True)
            if pending_refs:
                remaining_timeout = max(0.0, deadline - time.monotonic())
                _, pending_refs = ray.wait(
                    pending_refs,
                    num_returns=len(pending_refs),
                    timeout=remaining_timeout,
                )
            if pending_refs:
                raise TimeoutError(
                    f"timed out waiting for {len(pending_refs)} remote MAS rollout task(s) to stop"
                )

        super().shutdown(timeout=max(0.0, deadline - time.monotonic()))
        self._remote_tasks.clear()
        getattr(self, "_remote_task_cancel_grace", {}).clear()
