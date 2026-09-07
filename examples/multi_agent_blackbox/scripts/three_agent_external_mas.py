"""Command-line three-agent MAS used by the external runner example.

The script deliberately has no dependency on trainer or Gateway internals. It
reads the rollout-local YAML produced by ``external_mas_runner``, sends one
OpenAI-compatible request per configured role, and prints one JSON result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import yaml


def _agent_config(config: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    agents = config.get("agents", {})
    if not isinstance(agents, Mapping):
        return {}
    value = agents.get(role, {})
    return value if isinstance(value, Mapping) else {}


def _base_url(config: Mapping[str, Any], agent_config: Mapping[str, Any]) -> str:
    llm_config = agent_config.get("llm", {})
    if isinstance(llm_config, Mapping) and llm_config.get("base_url"):
        return str(llm_config["base_url"])
    global_llm = config.get("llm", {})
    if isinstance(global_llm, Mapping) and global_llm.get("base_url"):
        return str(global_llm["base_url"])
    raise ValueError("MAS config requires llm.base_url or agents.<role>.llm.base_url")


async def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    agent_config: Mapping[str, Any],
    default_max_tokens: int,
    request_timeout_seconds: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(agent_config.get("max_tokens", default_max_tokens)),
    }
    tools = agent_config.get("tools")
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=request_timeout_seconds) as client:
        response = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Gateway response does not contain choices")
    message = choices[0].get("message", {})
    if not isinstance(message, Mapping):
        raise ValueError("Gateway response choice does not contain a message")
    return str(message.get("content", ""))


async def run_three_agent_mas(
    *,
    config: Mapping[str, Any],
    prompt: str,
) -> dict[str, Any]:
    agents = config.get("agents", {})
    if not isinstance(agents, Mapping) or not agents:
        raise ValueError("MAS config requires a non-empty agents mapping")

    settings = config.get("mas", {})
    if not isinstance(settings, Mapping):
        settings = {}
    default_max_tokens = int(settings.get("max_tokens", 1024))
    request_timeout_seconds = float(settings.get("request_timeout_seconds", 600.0))
    outputs: dict[str, str] = {}

    for role, raw_agent_config in agents.items():
        role = str(role)
        agent_config = raw_agent_config if isinstance(raw_agent_config, Mapping) else {}
        system_prompt = str(
            agent_config.get("system_prompt", f"You are {role}. Complete your assigned part of the task.")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if outputs:
            prior_outputs = "\n\n".join(f"{name}: {output}" for name, output in outputs.items())
            messages.append({"role": "user", "content": f"Previous agent outputs:\n{prior_outputs}"})

        model = str(agent_config.get("model", role))
        outputs[role] = await _chat_completion(
            base_url=_base_url(config, agent_config),
            model=model,
            messages=messages,
            agent_config=agent_config,
            default_max_tokens=default_max_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )

    final_result = outputs[next(reversed(outputs))]
    return {
        "final_result": final_result,
        "agent_outputs": outputs,
        "reward_info": {"final_result": final_result, "agent_outputs": outputs},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the rollout-local MAS YAML")
    parser.add_argument("--prompt", required=True, help="User task passed to the MAS")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, Mapping):
        raise ValueError("MAS config must contain a mapping")
    result = asyncio.run(run_three_agent_mas(config=config, prompt=args.prompt))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
