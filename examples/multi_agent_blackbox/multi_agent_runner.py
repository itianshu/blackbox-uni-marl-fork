"""Minimal external MAS runner for MultiAgentFramework.

This example is intentionally small: it shows the contract between an external
multi-agent system and Uni-Agent's gateway-backed multi-agent rollout runtime.
Production integrations can use the same runner contract with a richer MAS
implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def extract_user_task(raw_prompt: Any) -> str:
    """Extract a user task from a raw prompt string or chat-message list."""
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, Mapping):
        return _content_to_text(raw_prompt.get("content", raw_prompt))
    if isinstance(raw_prompt, list):
        for message in reversed(raw_prompt):
            if isinstance(message, Mapping) and message.get("role") == "user":
                return _content_to_text(message.get("content", ""))
        return _content_to_text(raw_prompt)
    return str(raw_prompt)


def _agent_config(mas_config: Mapping[str, Any] | None, role: str) -> dict[str, Any]:
    agents = (mas_config or {}).get("agents", {})
    if isinstance(agents, Mapping):
        value = agents.get(role, {})
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def build_role_messages(
    *,
    raw_prompt: Any,
    role_policy_mapping: Mapping[str, str],
    mas_config: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Build the first private message list for each configured MAS role."""
    task = extract_user_task(raw_prompt)
    messages_by_role: dict[str, list[dict[str, str]]] = {}
    for role in role_policy_mapping:
        agent_cfg = _agent_config(mas_config, role)
        system_prompt = agent_cfg.get(
            "system_prompt",
            f"You are {role}. Complete your assigned part of the task.",
        )
        messages_by_role[role] = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": task},
        ]
    return messages_by_role


async def _chat_completion(
    *,
    base_url: str,
    role: str,
    messages: list[dict[str, str]],
    agent_cfg: Mapping[str, Any],
    max_tokens: int,
    request_timeout_seconds: float,
) -> str:
    import httpx

    payload: dict[str, Any] = {
        "model": role,
        "messages": messages,
        "max_tokens": int(agent_cfg.get("max_tokens", max_tokens)),
    }
    # These vLLM/OpenAI request fields are useful for deterministic load tests:
    # min_tokens + ignore_eos keep a request alive long enough for the dynamic
    # scheduler to observe sustained KV-cache pressure.  They remain opt-in so
    # normal example behaviour is unchanged.
    if "min_tokens" in agent_cfg:
        payload["min_tokens"] = int(agent_cfg["min_tokens"])
    if "ignore_eos" in agent_cfg:
        payload["ignore_eos"] = bool(agent_cfg["ignore_eos"])
    tools = agent_cfg.get("tools")
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=request_timeout_seconds) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return str(data["choices"][0]["message"].get("content", ""))


async def multi_agent_runner(
    *,
    raw_prompt: Any,
    rollout,
    sample_index: int,
    session_runtime,
    role_policy_mapping: Mapping[str, str],
    mas_config: Mapping[str, Any] | None = None,
    tools_kwargs: Mapping[str, Any] | None = None,
    max_tokens: int = 1024,
    request_timeout_seconds: float = 600.0,
    **_,
) -> dict[str, Any]:
    """Run one multi-agent rollout against a role-aware Uni-Agent gateway.

    The gateway exposes one rollout-level ``base_url``. Each request sets the
    OpenAI ``model`` field to the MAS role name; the gateway maps that role to
    the configured trainable policy. Temperature is intentionally NOT sent:
    sampling falls back to each policy's vLLM generation config, which is the
    same value the trainer uses to recompute log-probs, keeping the two sides
    consistent (same mechanism as swe_agent_blackbox training).
    """
    del session_runtime, sample_index, tools_kwargs

    if not getattr(rollout, "base_url", None):
        raise ValueError("multi_agent_runner requires rollout.base_url")

    missing_roles = set(role_policy_mapping) - set(getattr(rollout, "sessions", {}))
    if missing_roles:
        raise ValueError(f"rollout is missing role sessions: {sorted(missing_roles)}")

    role_messages = build_role_messages(
        raw_prompt=raw_prompt,
        role_policy_mapping=role_policy_mapping,
        mas_config=mas_config,
    )

    agent_outputs: dict[str, str] = {}
    for role in role_policy_mapping:
        messages = list(role_messages[role])
        if agent_outputs:
            prior_outputs = "\n\n".join(f"{name}: {output}" for name, output in agent_outputs.items())
            messages.append({"role": "user", "content": f"Previous agent outputs:\n{prior_outputs}"})

        agent_outputs[role] = await _chat_completion(
            base_url=rollout.base_url,
            role=role,
            messages=messages,
            agent_cfg=_agent_config(mas_config, role),
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )

    final_role = next(reversed(role_policy_mapping))
    final_result = agent_outputs.get(final_role, "")
    return {
        "final_result": final_result,
        "agent_outputs": agent_outputs,
        "reward_info": {
            "final_result": final_result,
            "agent_outputs": agent_outputs,
        },
    }
