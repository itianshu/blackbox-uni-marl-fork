"""Factory entry + trainer-facing adapters for the agent framework stack.

`build_agent_framework` owns gateway-universal wiring so framework subclasses
only handle their own agent runner, reward dispatch, and framework-specific
config fields.

`AgentFrameworkRolloutAdapter` satisfies the sync trainer's
`agent_loop_manager_class` extension-point contract; recipes wire it in via
yaml without authoring per-recipe glue:

    actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter

`FullyAsyncAgentFrameworkRolloutAdapter` satisfies the fully-async rollouter's
`agent_loop_manager_class` contract; it writes directly to TransferQueue
instead of relying on a local ReplayBuffer.
"""

from __future__ import annotations

from omegaconf import OmegaConf
from tensordict import TensorDict

from uni_agent.trainer.framework.framework import AgentFramework
from uni_agent.trainer.gateway.runtime import GatewayServingRuntime
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.ray_utils import auto_await
from verl.workers.config.model import HFModelConfig

_DEFAULT_FRAMEWORK_CLASS = "uni_agent.trainer.framework.framework.OpenAICompatibleAgentFramework"


async def build_agent_framework(
    *,
    config,
    llm_client,
    replay_buffer,
    reward_loop_worker_handles=None,
    gateway_actor_kwargs=None,
) -> AgentFramework:
    """Build GatewayServingRuntime, then delegate subclass-specific wiring."""
    # Current recipes keep framework settings under rollout.custom.agent_framework.
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}

    model_config = None
    framework_processor = None
    if gateway_actor_kwargs is None:
        # Match AgentLoopWorker pattern: self-load tokenizer/processor via HFModelConfig.
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        framework_processor = model_config.processor
        gateway_actor_kwargs = {
            "tokenizer": model_config.tokenizer,
            "processor": model_config.processor,
        }
    else:
        gateway_actor_kwargs = dict(gateway_actor_kwargs)
        framework_processor = gateway_actor_kwargs.get("processor")
    tool_parser_name = OmegaConf.select(config, "actor_rollout_ref.rollout.multi_turn.format", default=None)
    if tool_parser_name is not None and "tool_parser_name" not in gateway_actor_kwargs:
        gateway_actor_kwargs["tool_parser_name"] = tool_parser_name

    session_runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=int(af_cfg["gateway_count"]),
        gateway_actor_kwargs=gateway_actor_kwargs,
    )

    framework_cls = load_class_from_fqn(str(af_cfg.get("framework_class_fqn", _DEFAULT_FRAMEWORK_CLASS)))
    return await framework_cls.from_config(
        config=config,
        session_runtime=session_runtime,
        processor=framework_processor,
        policy_processors=gateway_actor_kwargs.get("policy_processors"),
        replay_buffer=replay_buffer,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


class AgentFrameworkRolloutAdapter:
    """Trainer-facing adapter satisfying the `agent_loop_manager_class` contract.

    Holds zero recipe-specific logic; every agent-framework recipe wires the
    same class in yaml. Phase B will let `main_ppo_sync.py` call
    `build_agent_framework` directly and this adapter can retire.
    """

    def __init__(self) -> None:
        self.framework = None

    @classmethod
    @auto_await
    async def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        replay_buffer=None,
        gateway_actor_kwargs=None,
        **_,
    ) -> "AgentFrameworkRolloutAdapter":
        del teacher_client

        framework = await build_agent_framework(
            config=config,
            llm_client=llm_client,
            replay_buffer=replay_buffer,
            reward_loop_worker_handles=reward_loop_worker_handles,
            gateway_actor_kwargs=gateway_actor_kwargs,
        )

        instance = cls()
        instance.framework = framework
        return instance

    @auto_await
    async def generate_sequences(self, prompts: TensorDict) -> None:
        if self.framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        return await self.framework.generate_sequences(prompts)


class FullyAsyncAgentFrameworkRolloutAdapter:
    """Async-rollouter adapter for agent frameworks (Gateway + subprocess runner).

    Satisfies the fully-async rollouter's ``agent_loop_manager_class`` contract.
    The ``llm_client`` injected by the rollouter is a
    :class:`FullyAsyncLLMServerClient`, so the Gateway automatically gains
    partial rollout resume capability — weight-sync interruptions are
    transparent to the subprocess agent_runner.
    """

    def __init__(self) -> None:
        self.framework = None

    @classmethod
    @auto_await
    async def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        gateway_actor_kwargs=None,
        **_,
    ) -> "FullyAsyncAgentFrameworkRolloutAdapter":
        del teacher_client
        framework = await build_agent_framework(
            config=config,
            llm_client=llm_client,
            replay_buffer=None,
            reward_loop_worker_handles=reward_loop_worker_handles,
            gateway_actor_kwargs=gateway_actor_kwargs,
        )
        instance = cls()
        instance.framework = framework
        return instance

    async def generate_sequences_single(self, prompts):
        """TQ async path entry — writes to TransferQueue, returns None."""
        return await self.framework.generate_sequences(prompts)

    async def generate_sequences(self, prompts):
        """Compatibility entry for validation path."""
        return await self.framework.generate_sequences(prompts)
