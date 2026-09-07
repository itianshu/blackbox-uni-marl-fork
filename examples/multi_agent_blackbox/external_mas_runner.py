"""Generic command-driven external MAS runner.

Each invocation receives a rollout-scoped Gateway URL, injects it into a copy
of the MAS configuration, and launches the configured command in the Ray
worker that owns this rollout task.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from examples.multi_agent_blackbox.multi_agent_runner import extract_user_task
from examples.multi_agent_blackbox.process_backend import LocalProcessResult, run_local_process


def _safe_filename_component(value: Any, *, default: str = "rollout") -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or default


def _set_path(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = [key for key in dotted_path.split(".") if key]
    if not keys:
        raise ValueError("injection paths must not be empty")
    current = root
    for key in keys[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot inject {dotted_path!r}: {key!r} is not a mapping")
        current = child
    current[keys[-1]] = value


def inject_gateway_config(
    template: Mapping[str, Any],
    *,
    gateway_url: str,
    roles: Sequence[str],
    injection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copied MAS config with Gateway URLs and role model names."""
    resolved = copy.deepcopy(dict(template))
    injection = dict(injection or {})
    _set_path(resolved, str(injection.get("global_base_url_path", "llm.base_url")), gateway_url)

    agents_path = str(injection.get("agents_path", "agents"))
    agents: dict[str, Any] = {}
    _set_path(resolved, agents_path, agents)
    original_agents = template
    for component in agents_path.split("."):
        if not isinstance(original_agents, Mapping):
            original_agents = {}
            break
        original_agents = original_agents.get(component, {})
    if isinstance(original_agents, Mapping):
        agents.update(copy.deepcopy(dict(original_agents)))

    inject_agent_base_url = bool(injection.get("inject_agent_base_url", True))
    agent_base_url_path = str(injection.get("agent_base_url_path", "llm.base_url"))
    agent_model_path = str(injection.get("agent_model_path", "model"))
    for role in roles:
        role_config = agents.setdefault(str(role), {})
        if not isinstance(role_config, dict):
            raise ValueError(f"MAS config for role {role!r} must be a mapping")
        if inject_agent_base_url:
            _set_path(role_config, agent_base_url_path, gateway_url)
        _set_path(role_config, agent_model_path, str(role))
    return resolved


def _format_values(values: Mapping[str, Any], context: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(value).format_map(context) for key, value in values.items()}


def _build_command(command_config: Mapping[str, Any], context: Mapping[str, str]) -> tuple[str | list[str], bool]:
    shell = bool(command_config.get("shell", False))
    argv = command_config.get("argv")
    template = command_config.get("template")
    if argv is not None:
        if shell or not isinstance(argv, Sequence) or isinstance(argv, str | bytes):
            raise ValueError("command.argv must be a sequence and cannot be combined with shell=true")
        return [str(part).format_map(context) for part in argv], False
    if not isinstance(template, str) or not template:
        raise ValueError("command.argv or command.template is required")
    if shell:
        quoted_context = dict(context)
        if os.name == "posix":
            quoted_context = {key: shlex.quote(value) for key, value in context.items()}
        return template.format_map(quoted_context), True
    parts = shlex.split(template, posix=os.name != "nt")
    return [part.format_map(context) for part in parts], False


def _parse_result(process_result: LocalProcessResult, result_config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(result_config.get("mode", "stdout_json"))
    if mode == "none":
        return {}
    if mode == "stdout_text":
        text = process_result.stdout.strip()
        return {"final_result": text, "reward_info": {"final_result": text}}
    if mode != "stdout_json":
        raise ValueError(f"Unsupported external MAS result mode: {mode!r}")

    for line in reversed(process_result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("External MAS stdout did not contain a JSON object")


async def external_mas_runner(
    *,
    raw_prompt: Any,
    rollout: Any,
    sample_index: int,
    session_runtime: Any,
    role_policy_mapping: Mapping[str, str],
    execution: Mapping[str, Any],
    command: Mapping[str, Any],
    config: Mapping[str, Any],
    timeout: Mapping[str, Any] | float | int | None = None,
    result: Mapping[str, Any] | None = None,
    logs: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Launch one external MAS command against the rollout Gateway."""
    del session_runtime, sample_index
    gateway_url = getattr(rollout, "base_url", None)
    if not gateway_url:
        raise ValueError("external_mas_runner requires rollout.base_url")

    backend = str(execution.get("backend", "local_process"))
    if backend != "local_process":
        raise NotImplementedError(f"External MAS execution backend {backend!r} is not implemented")

    template_path = config.get("template_path")
    if not template_path:
        raise ValueError("config.template_path is required")
    with Path(str(template_path)).expanduser().open(encoding="utf-8") as file:
        template = yaml.safe_load(file) or {}
    if not isinstance(template, Mapping):
        raise ValueError("External MAS config template must contain a mapping")

    injected = inject_gateway_config(
        template,
        gateway_url=str(gateway_url),
        roles=list(role_policy_mapping),
        injection=config.get("injection"),
    )
    temp_dir = config.get("temp_dir")
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="uni_mas_",
        dir=str(temp_dir) if temp_dir else None,
        delete=False,
        encoding="utf-8",
    )
    config_path = Path(temporary.name)
    try:
        yaml.safe_dump(injected, temporary, sort_keys=False, allow_unicode=True)
        temporary.close()

        prompt = extract_user_task(raw_prompt)
        context = {
            "config_path": str(config_path),
            "prompt": prompt,
            "gateway_url": str(gateway_url),
            "rollout_id": str(getattr(rollout, "rollout_id", "")),
        }
        process_command, shell = _build_command(command, context)
        process_env = _format_values(command.get("env", {}), context)
        timeout_seconds = timeout.get("seconds") if isinstance(timeout, Mapping) else timeout

        process_result = await run_local_process(
            process_command,
            work_dir=str(command["work_dir"]) if command.get("work_dir") else None,
            env=process_env,
            timeout_seconds=float(timeout_seconds) if timeout_seconds is not None else None,
            shell=shell,
        )
        log_dir = (logs or {}).get("directory")
        if log_dir:
            output_dir = Path(str(log_dir)).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            rollout_name = _safe_filename_component(getattr(rollout, "rollout_id", "rollout"))
            (output_dir / f"{rollout_name}.stdout.log").write_text(
                process_result.stdout,
                encoding="utf-8",
                newline="",
            )
            (output_dir / f"{rollout_name}.stderr.log").write_text(
                process_result.stderr,
                encoding="utf-8",
                newline="",
            )
        return _parse_result(process_result, result or {})
    finally:
        temporary.close()
        config_path.unlink(missing_ok=True)
