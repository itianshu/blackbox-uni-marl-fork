"""Random-routing multi-agent workload harness.

The harness models a request entering through one agent and moving between
agents for a random number of turns.  Only the immediately preceding answer is
handed to the next agent, which keeps prompts bounded even for ten-turn runs.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

from examples.multi_agent_blackbox.multi_agent_runner import (
    _agent_config,
    _chat_completion,
    extract_user_task,
)


def _integer(config: Mapping[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value <= 0:
        raise ValueError(f"harness.{name} must be positive, got {value}")
    return value


def _stable_rng(*, seed: int, sample_index: int, task: str) -> random.Random:
    material = f"{seed}\0{sample_index}\0{task}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(derived_seed)


def build_random_route(
    *,
    roles: Sequence[str],
    entry_agent: str,
    min_rounds: int,
    max_rounds: int,
    rng: random.Random,
    route_weights: Mapping[str, Any] | None = None,
) -> list[str]:
    """Route a bounded conversation whose first turn is handled by the entry agent."""
    if not roles:
        raise ValueError("random-routing harness requires at least one role")
    if entry_agent not in roles:
        raise ValueError(f"entry_agent {entry_agent!r} is not in configured roles {list(roles)!r}")
    if min_rounds <= 0 or max_rounds < min_rounds:
        raise ValueError(
            f"invalid round range: min_rounds={min_rounds}, max_rounds={max_rounds}"
        )
    if max_rounds > 10:
        raise ValueError(f"max_rounds cannot exceed the harness safety limit of 10, got {max_rounds}")

    weights = [float((route_weights or {}).get(role, 1.0)) for role in roles]
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("route_weights must be non-negative with at least one positive value")

    rounds = rng.randint(min_rounds, max_rounds)
    # The entry agent is a real processing hop, not just metadata attached to a
    # request that starts at an arbitrary worker.  Subsequent hops are random.
    return [entry_agent, *rng.choices(list(roles), weights=weights, k=rounds - 1)]


def _sample_token_budget(
    *,
    rng: random.Random,
    harness_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
) -> tuple[int, int | None]:
    minimum = int(agent_config.get("output_tokens_min", harness_config.get("output_tokens_min", 4096)))
    maximum = int(agent_config.get("output_tokens_max", harness_config.get("output_tokens_max", 8192)))
    step = _integer(harness_config, "output_tokens_step", 512)
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"invalid output token range: {minimum}..{maximum}")

    choices = list(range(minimum, maximum + 1, step))
    if choices[-1] != maximum:
        choices.append(maximum)
    max_tokens = rng.choice(choices)

    minimum_ratio = float(harness_config.get("minimum_completion_ratio", 0.0))
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("harness.minimum_completion_ratio must be between 0 and 1")
    min_tokens = max(1, int(max_tokens * minimum_ratio)) if minimum_ratio else None
    return max_tokens, min_tokens


def _messages_for_turn(
    *,
    task: str,
    role: str,
    entry_agent: str,
    agent_config: Mapping[str, Any],
    turn_index: int,
    total_turns: int,
    previous_role: str | None,
    previous_output: str | None,
) -> list[dict[str, str]]:
    system_prompt = agent_config.get(
        "system_prompt",
        f"You are {role}. Analyze the task thoroughly and preserve the final answer.",
    )
    messages = [{"role": "system", "content": str(system_prompt)}]
    if previous_output is None:
        content = (
            f"This request entered through {entry_agent}. "
            "Inspect the request, begin the work, and prepare it for a possible downstream agent. "
            f"You are turn {turn_index + 1} of {total_turns}.\n\nTask:\n{task}"
        )
    else:
        content = (
            f"Continue the task from {previous_role}. You are turn {turn_index + 1} "
            f"of {total_turns}. Check the prior work, extend or correct it, and retain "
            "a clear final answer for a possible downstream agent.\n\n"
            f"Original task:\n{task}\n\nPrevious agent output:\n{previous_output}"
        )
    messages.append({"role": "user", "content": content})
    return messages


async def random_routing_agent_runner(
    *,
    raw_prompt: Any,
    rollout,
    sample_index: int,
    session_runtime,
    role_policy_mapping: Mapping[str, str],
    mas_config: Mapping[str, Any] | None = None,
    tools_kwargs: Mapping[str, Any] | None = None,
    max_tokens: int = 8192,
    request_timeout_seconds: float = 7200.0,
    **_,
) -> dict[str, Any]:
    """Execute a bounded, reproducible random route through the MAS roles."""
    del session_runtime, tools_kwargs

    if not getattr(rollout, "base_url", None):
        raise ValueError("random_routing_agent_runner requires rollout.base_url")
    roles = list(role_policy_mapping)
    missing_roles = set(roles) - set(getattr(rollout, "sessions", {}))
    if missing_roles:
        raise ValueError(f"rollout is missing role sessions: {sorted(missing_roles)}")

    config = dict((mas_config or {}).get("harness", {}))
    task = extract_user_task(raw_prompt)
    rng = _stable_rng(
        seed=int(config.get("random_seed", 42)),
        sample_index=sample_index,
        task=task,
    )
    entry_agent = str(config.get("entry_agent", "agent_1"))
    route = build_random_route(
        roles=roles,
        entry_agent=entry_agent,
        min_rounds=_integer(config, "min_rounds", 1),
        max_rounds=_integer(config, "max_rounds", 10),
        rng=rng,
        route_weights=config.get("route_weights"),
    )

    previous_role: str | None = None
    previous_output: str | None = None
    latest_agent_outputs: dict[str, str] = {}
    routing_trace: list[dict[str, Any]] = []
    for turn_index, role in enumerate(route):
        agent_config = _agent_config(mas_config, role)
        sampled_max_tokens, sampled_min_tokens = _sample_token_budget(
            rng=rng,
            harness_config=config,
            agent_config=agent_config,
        )
        # The sampled per-turn budget takes precedence over legacy fixed values.
        request_config = dict(agent_config)
        request_config["max_tokens"] = min(sampled_max_tokens, int(max_tokens))
        request_config.pop("ignore_eos", None)
        if sampled_min_tokens is not None:
            request_config["min_tokens"] = min(sampled_min_tokens, request_config["max_tokens"])
        else:
            request_config.pop("min_tokens", None)

        output = await _chat_completion(
            base_url=rollout.base_url,
            role=role,
            messages=_messages_for_turn(
                task=task,
                role=role,
                entry_agent=entry_agent,
                agent_config=agent_config,
                turn_index=turn_index,
                total_turns=len(route),
                previous_role=previous_role,
                previous_output=previous_output,
            ),
            agent_cfg=request_config,
            max_tokens=request_config["max_tokens"],
            request_timeout_seconds=request_timeout_seconds,
        )
        latest_agent_outputs[role] = output
        routing_trace.append(
            {
                "turn": turn_index + 1,
                "agent": role,
                "policy": role_policy_mapping[role],
                "max_tokens": request_config["max_tokens"],
                "min_tokens": request_config.get("min_tokens"),
                "output_chars": len(output),
            }
        )
        previous_role, previous_output = role, output

    final_result = previous_output or ""
    reward_info = {
        "final_result": final_result,
        "agent_outputs": latest_agent_outputs,
        "round_count": len(route),
        "entry_agent": entry_agent,
        "route": route,
        "routing_trace": routing_trace,
    }
    return {
        "final_result": final_result,
        "agent_outputs": latest_agent_outputs,
        "route": route,
        "routing_trace": routing_trace,
        "reward_info": reward_info,
    }
