import asyncio
import inspect
import json
import sys
import types
from concurrent.futures import Future
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        text = ""
        if tools:
            text += f"[tools:{json.dumps(tools, sort_keys=True)}]"
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, (list, dict)):
                content = json.dumps(content, sort_keys=True)
            text += f"<{message['role']}>{content}</>"
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) for i in ids).decode("utf-8", errors="ignore")


class PrefixTokenizer(FakeTokenizer):
    def __init__(self, prefix):
        self.prefix = prefix

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, tools=None, return_dict=False, **kwargs
    ):
        encoded = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        if not tokenize:
            return f"{self.prefix}:{encoded}"
        return list(self.prefix.encode("utf-8")) + encoded

class RecordingBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        self.requests.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "kwargs": dict(kwargs),
            }
        )
        output = self.outputs.pop(0)
        text, stop_reason = output[:2]
        extra_fields = output[2] if len(output) > 2 else {}
        return SimpleNamespace(
            token_ids=list(text.encode("utf-8")),
            log_probs=None,
            stop_reason=stop_reason,
            extra_fields=dict(extra_fields),
        )


def _install_dependency_stubs():
    if "omegaconf" not in sys.modules:
        omegaconf_mod = types.ModuleType("omegaconf")

        class OmegaConf:
            @staticmethod
            def select(config, key, default=None):
                return default

            @staticmethod
            def create(value):
                return value

            @staticmethod
            def to_container(value, resolve=False):
                return value

        omegaconf_mod.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf_mod
    if "ray" not in sys.modules:
        ray_stub = types.ModuleType("ray")
        ray_stub.remote = lambda cls: cls
        ray_stub.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        sys.modules["ray"] = ray_stub
    if "verl.experimental.agent_loop.tool_parser" not in sys.modules:
        tool_parser_mod = types.ModuleType("verl.experimental.agent_loop.tool_parser")

        class ToolParser:
            @staticmethod
            def get_tool_parser(name, tokenizer):
                return None

        tool_parser_mod.ToolParser = ToolParser
        chat_template_mod = types.ModuleType("verl.utils.chat_template")
        chat_template_mod.apply_chat_template = (
            lambda tokenizer, messages, tools=None, add_generation_prompt=True, **kwargs: tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        )
        chat_template_mod.initialize_system_prompt = lambda tokenizer, **kwargs: None
        tokenizer_mod = types.ModuleType("verl.utils.tokenizer")
        tokenizer_mod.__path__ = []
        tokenizer_mod.normalize_token_ids = lambda ids: list(ids)
        tool_registry_mod = types.ModuleType("verl.tools.tool_registry")
        tool_registry_mod.initialize_tools_from_config = lambda path: []
        import_utils_mod = types.ModuleType("verl.utils.import_utils")
        import_utils_mod.load_class_from_fqn = lambda fqn, description=None: None
        transferqueue_utils_mod = types.ModuleType("verl.utils.transferqueue_utils")
        transferqueue_utils_mod.tq = SimpleNamespace()
        tensordict_utils_mod = types.ModuleType("verl.utils.tensordict_utils")
        tensordict_utils_mod.get = lambda tensor_dict, key: tensor_dict[key]
        model_mod = types.ModuleType("verl.utils.model")
        model_mod.compute_position_id_with_mask = lambda attention_mask: attention_mask
        llm_server_mod = types.ModuleType("verl.workers.rollout.llm_server")
        llm_server_mod.LLMServerClient = object
        rollout_utils_mod = types.ModuleType("verl.workers.rollout.utils")
        rollout_utils_mod.run_uvicorn = lambda *args, **kwargs: None

        for name in [
            "verl",
            "verl.experimental",
            "verl.experimental.agent_loop",
            "verl.tools",
            "verl.utils",
            "verl.workers",
            "verl.workers.rollout",
        ]:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["verl.experimental.agent_loop.tool_parser"] = tool_parser_mod
        sys.modules["verl.utils.chat_template"] = chat_template_mod
        sys.modules["verl.utils.tokenizer"] = tokenizer_mod
        sys.modules["verl.utils.tokenizer.chat_template"] = chat_template_mod
        sys.modules["verl.tools.tool_registry"] = tool_registry_mod
        sys.modules["verl.utils.import_utils"] = import_utils_mod
        sys.modules["verl.utils.transferqueue_utils"] = transferqueue_utils_mod
        sys.modules["verl.utils.tensordict_utils"] = tensordict_utils_mod
        sys.modules["verl.utils.model"] = model_mod
        sys.modules["verl.workers.rollout.llm_server"] = llm_server_mod
        sys.modules["verl.workers.rollout.utils"] = rollout_utils_mod


def _make_gateway(backend):
    _install_dependency_stubs()

    from uni_agent.trainer.gateway.gateway import _GatewayActor

    actor = _GatewayActor(FakeTokenizer(), backend)
    actor._server_base_url = "http://testserver"
    return actor


class RecordingLLMClient:
    def __init__(self, label):
        self.label = label
        self.calls = []

    async def generate(self, request_id, **kwargs):
        self.calls.append({"request_id": request_id, **kwargs})
        return self.label


class ImmediateObjectRef:
    def __init__(self, value=None):
        self._future = Future()
        self._future.set_result(value)

    def future(self):
        return self._future


class RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return ImmediateObjectRef(self._func(*args, **kwargs))


class FakeGatewayForManager:
    def __init__(self):
        self.created_rollouts = []
        self.finalized_rollouts = []
        self.aborted_rollouts = []
        self.create_multi_agent_rollout = RemoteMethod(self._create_multi_agent_rollout)
        self.finalize_multi_agent_rollout = RemoteMethod(self._finalize_multi_agent_rollout)
        self.abort_multi_agent_rollout = RemoteMethod(self._abort_multi_agent_rollout)

    def _create_multi_agent_rollout(self, rollout_id, **kwargs):
        from uni_agent.trainer.gateway.types import MultiAgentRolloutHandle, RoleSessionInfo

        role_policy_mapping = dict(kwargs["role_policy_mapping"])
        self.created_rollouts.append(rollout_id)
        return MultiAgentRolloutHandle(
            rollout_id=rollout_id,
            sessions={
                role: RoleSessionInfo(role=role, session_id=f"{rollout_id}:{role}", policy_name=policy_name)
                for role, policy_name in role_policy_mapping.items()
            },
            role_policy_mapping=role_policy_mapping,
        )

    def _finalize_multi_agent_rollout(self, rollout_id):
        from uni_agent.trainer.gateway.types import MultiAgentRolloutResult

        self.finalized_rollouts.append(rollout_id)
        return MultiAgentRolloutResult(rollout_id=rollout_id, trajectories=[])

    def _abort_multi_agent_rollout(self, rollout_id):
        self.aborted_rollouts.append(rollout_id)


class TestGatewayManager:
    def test_multi_agent_rollout_counts_as_one_scheduling_unit(self):
        asyncio.run(self._run_multi_agent_rollout_counts_as_one_scheduling_unit())

    async def _run_multi_agent_rollout_counts_as_one_scheduling_unit(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.manager import GatewayManager

        first_gateway = FakeGatewayForManager()
        second_gateway = FakeGatewayForManager()
        manager = GatewayManager([first_gateway, second_gateway])

        await manager.create_multi_agent_rollout(
            "rollout-1",
            role_policy_mapping={
                "agent_1": "policy_1",
                "agent_2": "policy_1",
                "agent_3": "policy_2",
            },
        )

        assert manager.active_sessions_per_gateway == [1, 0]

        await manager.create_multi_agent_rollout(
            "rollout-2",
            role_policy_mapping={"agent_1": "policy_1"},
        )
        assert manager.active_sessions_per_gateway == [1, 1]

        await manager.finalize_multi_agent_rollout("rollout-1")
        assert manager.active_sessions_per_gateway == [0, 1]

        await manager.abort_multi_agent_rollout("rollout-2")
        assert manager.active_sessions_per_gateway == [0, 0]


class TestGatewayActorApiLayout:
    def test_multi_agent_apis_follow_existing_single_agent_apis(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _GatewayActor

        source_lines, _ = inspect.getsourcelines(_GatewayActor)
        observed_order = []
        expected_order = [
            "start",
            "shutdown",
            "create_session",
            "complete_session",
            "wait_for_completion",
            "finalize_session",
            "abort_session",
            "create_multi_agent_rollout",
            "finalize_multi_agent_rollout",
            "complete_multi_agent_rollout",
            "abort_multi_agent_rollout",
            "wait_for_multi_agent_rollout_completion",
        ]

        for line in source_lines:
            stripped = line.strip()
            if not stripped.startswith("async def "):
                continue
            method_name = stripped.removeprefix("async def ").split("(", 1)[0]
            if method_name in expected_order:
                observed_order.append(method_name)

        assert observed_order == expected_order


class TestPolicyRoutingLLMClient:
    def test_routes_generation_by_policy_name(self):
        asyncio.run(self._run_route())

    async def _run_route(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.runtime import PolicyRoutingLLMClient

        policy_1 = RecordingLLMClient("policy_1_result")
        policy_2 = RecordingLLMClient("policy_2_result")
        router = PolicyRoutingLLMClient({"policy_1": policy_1, "policy_2": policy_2})

        result = await router.generate(
            "request-1",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 4},
            policy_name="policy_2",
            role="agent_3",
            rollout_id="rollout-1",
            agent_role="legacy-agent-role",
            priority=7,
        )

        assert result == "policy_2_result"
        assert policy_1.calls == []
        assert policy_2.calls == [
            {
                "request_id": "request-1",
                "prompt_ids": [1, 2],
                "sampling_params": {"max_tokens": 4},
                "image_data": None,
                "video_data": None,
                "agent_role": "legacy-agent-role",
                "priority": 7,
            }
        ]

    def test_gateway_runtime_delegates_policy_routing_client(self):
        asyncio.run(self._run_gateway_runtime_route())

    async def _run_gateway_runtime_route(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.runtime import GatewayServingRuntime, PolicyRoutingLLMClient

        policy_1 = RecordingLLMClient("policy_1_result")
        policy_2 = RecordingLLMClient("policy_2_result")
        router = PolicyRoutingLLMClient({"policy_1": policy_1, "policy_2": policy_2})
        runtime = GatewayServingRuntime(llm_client=router, gateway_count=0)

        result = await runtime.generate(
            "request-2",
            prompt_ids=[7, 8],
            sampling_params={"temperature": 0.2},
            policy_name="policy_1",
            role="agent_1",
            rollout_id="rollout-2",
        )

        assert result == "policy_1_result"
        assert policy_1.calls == [
            {
                "request_id": "request-2",
                "prompt_ids": [7, 8],
                "sampling_params": {"temperature": 0.2},
                "image_data": None,
                "video_data": None,
            }
        ]
        assert policy_2.calls == []


class TestMultiAgentFrameworkExports:
    def test_framework_module_and_compat_module_export_same_multi_agent_class(self):
        _install_dependency_stubs()

        from uni_agent.trainer.framework import MultiAgentFramework as package_class
        from uni_agent.trainer.framework.framework import (
            MultiAgentFramework as framework_module_class,
        )
        from uni_agent.trainer.framework.multi_agent import (
            MultiAgentFramework as compat_module_class,
        )

        assert framework_module_class is compat_module_class
        assert package_class is framework_module_class


class RecordingTQ:
    def __init__(self):
        self.kv_puts = []
        self.batch_puts = []

    async def async_kv_put(self, *, key, partition_id, tag):
        self.kv_puts.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.batch_puts.append(
            {
                "keys": list(keys),
                "fields": list(fields),
                "tags": [dict(tag) for tag in tags],
                "partition_id": partition_id,
            }
        )


class FakeMultiAgentSessionRuntime:
    def __init__(self):
        self.created = []
        self.completed = []
        self.waited = []
        self.finalized = []
        self.aborted = []
        self.finalize_results = None

    async def create_multi_agent_rollout(self, rollout_id, **kwargs):
        from uni_agent.trainer.gateway.types import MultiAgentRolloutHandle, RoleSessionInfo

        self.created.append({"rollout_id": rollout_id, **kwargs})
        return MultiAgentRolloutHandle(
            rollout_id=rollout_id,
            base_url=f"http://testserver/rollouts/{rollout_id}/v1",
            sessions={
                role: RoleSessionInfo(role=role, session_id=f"{rollout_id}:{role}", policy_name=policy_name)
                for role, policy_name in kwargs["role_policy_mapping"].items()
            },
            role_policy_mapping=dict(kwargs["role_policy_mapping"]),
        )

    async def complete_multi_agent_rollout(self, rollout_id, reward_info=None):
        self.completed.append({"rollout_id": rollout_id, "reward_info": dict(reward_info or {})})

    async def wait_for_multi_agent_rollout_completion(self, rollout_id, timeout=None):
        self.waited.append({"rollout_id": rollout_id, "timeout": timeout})

    async def finalize_multi_agent_rollout(self, rollout_id):
        from uni_agent.trainer.framework.types import Trajectory
        from uni_agent.trainer.gateway.types import MultiAgentRolloutResult, RoleSessionInfo

        self.finalized.append(rollout_id)
        if self.finalize_results is not None:
            if callable(self.finalize_results):
                return self.finalize_results(rollout_id)
            return self.finalize_results[rollout_id]
        return MultiAgentRolloutResult(
            rollout_id=rollout_id,
            trajectories=[
                Trajectory(
                    prompt_ids=[1, 2],
                    response_ids=[3],
                    response_mask=[1],
                    reward_info={"reward_score": 0.75},
                    num_turns=3,
                    extra_fields={
                        "rollout_id": rollout_id,
                        "session_id": f"{rollout_id}:agent_1",
                        "role": "agent_1",
                        "policy_name": "policy_1",
                    },
                ),
                Trajectory(
                    prompt_ids=[4],
                    response_ids=[5, 6],
                    response_mask=[1, 1],
                    reward_info={"reward_score": 0.75},
                    num_turns=2,
                    extra_fields={
                        "rollout_id": rollout_id,
                        "session_id": f"{rollout_id}:agent_2",
                        "role": "agent_2",
                        "policy_name": "policy_1",
                    },
                ),
            ],
            sessions={
                "agent_1": RoleSessionInfo("agent_1", f"{rollout_id}:agent_1", "policy_1"),
                "agent_2": RoleSessionInfo("agent_2", f"{rollout_id}:agent_2", "policy_1"),
            },
            reward_info={"reward_score": 0.75, "reward_extra_info": {"judge": "ok"}},
        )

    async def abort_multi_agent_rollout(self, rollout_id):
        self.aborted.append(rollout_id)


class TestMultiAgentFrameworkTQ:
    def test_from_config_loads_mas_config_for_multi_agent_runner(self):
        asyncio.run(self._run_from_config_loads_mas_config_for_multi_agent_runner())

    async def _run_from_config_loads_mas_config_for_multi_agent_runner(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework

        runner_calls = []

        async def runner(**kwargs):
            runner_calls.append(kwargs)
            return {}

        class FakeOmegaConf:
            @staticmethod
            def select(config, key, default=None):
                if key == "actor_rollout_ref.rollout.custom.agent_framework":
                    return {
                        "multi_agent_runner_fqn": "tests.fake_runner",
                        "multi_agent_runner_kwargs": {
                            "runner_option": "enabled",
                            "mas_config_path": "mas.yaml",
                        },
                        "role_policy_mapping": {"agent_1": "policy_1"},
                    }
                return default

            @staticmethod
            def create(value):
                return value

            @staticmethod
            def to_container(value, resolve=False):
                return value

        original_omega_conf = framework_module.OmegaConf
        original_load_class_from_fqn = framework_module.load_class_from_fqn
        original_load_yaml_file = framework_module._load_yaml_file
        framework_module.OmegaConf = FakeOmegaConf
        framework_module.load_class_from_fqn = lambda fqn, description=None: runner
        framework_module._load_yaml_file = lambda path: {
            "agents": {
                "agent_1": {
                    "model": "agent_1",
                    "tools": [{"name": "spawn_agent"}],
                },
                "agent_2": {
                    "model": "agent_2",
                    "tools": [{"name": "execute_bash"}],
                },
            }
        }
        try:
            config = SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    rollout={"n": 1, "val_kwargs": {"n": 1}},
                )
            )
            runtime = FakeMultiAgentSessionRuntime()
            framework = await MultiAgentFramework.from_config(
                config=config,
                session_runtime=runtime,
                processor=None,
                replay_buffer=None,
                reward_loop_worker_handles=None,
            )
            await framework.run_rollout(raw_prompt=[{"role": "user", "content": "solve"}], rollout_id="rollout-1")
        finally:
            framework_module.OmegaConf = original_omega_conf
            framework_module.load_class_from_fqn = original_load_class_from_fqn
            framework_module._load_yaml_file = original_load_yaml_file

        assert runtime.created[0]["role_policy_mapping"] == {"agent_1": "policy_1"}
        assert runner_calls[0]["runner_option"] == "enabled"
        assert runner_calls[0]["mas_config_path"] == "mas.yaml"
        assert runner_calls[0]["mas_config"] == {
            "agents": {
                "agent_1": {
                    "model": "agent_1",
                    "tools": [{"name": "spawn_agent"}],
                },
                "agent_2": {
                    "model": "agent_2",
                    "tools": [{"name": "execute_bash"}],
                },
            }
        }

    def test_run_rollout_waits_for_completion_when_configured(self):
        asyncio.run(self._run_rollout_waits_for_completion())

    async def _run_rollout_waits_for_completion(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework.framework import MultiAgentFramework

        async def runner(**kwargs):
            return {}

        runtime = FakeMultiAgentSessionRuntime()
        framework = MultiAgentFramework(
            session_runtime=runtime,
            multi_agent_runner=runner,
            role_policy_mapping={"agent_1": "policy_1"},
            rollout_config={"n": 1, "val_kwargs": {"n": 1}},
            completion_timeout=12.5,
            wait_for_completion_after_runner=True,
        )

        result = await framework.run_rollout(raw_prompt=[{"role": "user", "content": "solve"}], rollout_id="rollout-1")

        assert result.rollout_id == "rollout-1"
        assert runtime.completed == []
        assert runtime.waited == [{"rollout_id": "rollout-1", "timeout": 12.5}]
        assert runtime.finalized == ["rollout-1"]

    def test_trajectory_to_tq_field_and_tag_adds_multi_agent_metadata(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework.framework import MultiAgentFramework
        from uni_agent.trainer.framework.types import Trajectory

        framework = MultiAgentFramework(
            session_runtime=FakeMultiAgentSessionRuntime(),
            multi_agent_runner=lambda **_: None,
            role_policy_mapping={"agent_1": "policy_1"},
            rollout_config={"n": 1, "val_kwargs": {"n": 1}},
        )
        trajectory = Trajectory(
            prompt_ids=[1, 2],
            response_ids=[3],
            response_mask=[1],
            reward_score=0.5,
            num_turns=2,
            extra_fields={
                "role": "agent_1",
                "policy_name": "policy_1",
                "role_session_id": "rollout-1:agent_1",
                "global_steps": 10,
                "min_global_steps": 8,
                "max_global_steps": 10,
            },
        )

        field, tag = framework._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields={"uid": "prompt-uid"},
            sample_idx=2,
            record_idx=3,
            rollout_id="rollout-1",
            uid="prompt-uid",
            global_steps=9,
        )

        assert field["uid"] == "prompt-uid"
        assert "session_id" not in field
        assert "prompt_group_id" not in field
        assert field["rollout_id"] == "rollout-1"
        assert field["sample_idx"] == 2
        assert field["record_idx"] == 3
        assert field["role"] == "agent_1"
        assert "agent_role" not in field
        assert field["policy_name"] == "policy_1"
        assert field["role_session_id"] == "rollout-1:agent_1"
        assert field["rm_scores"][-1].item() == 0.5

        assert tag["uid"] == "prompt-uid"
        assert "prompt_group_id" not in tag
        assert tag["rollout_id"] == "rollout-1"
        assert tag["sample_idx"] == 2
        assert tag["record_idx"] == 3
        assert tag["role"] == "agent_1"
        assert tag["policy_name"] == "policy_1"
        assert tag["global_steps"] == 9
        assert tag["min_global_steps"] == 8
        assert tag["max_global_steps"] == 10

    def test_multi_agent_reward_worker_receives_rollout_result_extra_info(self):
        asyncio.run(self._run_multi_agent_reward_worker_receives_rollout_result_extra_info())

    async def _run_multi_agent_reward_worker_receives_rollout_result_extra_info(self):
        _install_dependency_stubs()
        protocol_mod = types.ModuleType("verl.protocol")

        class DataProto:
            def __init__(self, batch=None, non_tensor_batch=None):
                self.batch = batch
                self.non_tensor_batch = non_tensor_batch or {}

        protocol_mod.DataProto = DataProto
        sys.modules["verl.protocol"] = protocol_mod
        from uni_agent.trainer.framework.framework import MultiAgentFramework
        from uni_agent.trainer.framework.types import Trajectory
        from uni_agent.trainer.gateway.types import MultiAgentRolloutResult, RoleSessionInfo

        captured = {}

        class RecordingComputeScore:
            async def remote(self, data):
                captured["extra_info"] = dict(data.non_tensor_batch["extra_info"][0])
                return {
                    "reward_score": 0.8,
                    "reward_extra_info": {"judge": "reward_worker"},
                }

        framework = MultiAgentFramework(
            session_runtime=FakeMultiAgentSessionRuntime(),
            multi_agent_runner=lambda **_: None,
            role_policy_mapping={"agent_1": "policy_1"},
            rollout_config={"n": 1, "val_kwargs": {"n": 1}},
            reward_loop_worker_handles=[SimpleNamespace(compute_score=RecordingComputeScore())],
        )
        rollout_result = MultiAgentRolloutResult(
            rollout_id="rollout-1",
            trajectories=[
                Trajectory(
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[1],
                    num_turns=1,
                    extra_fields={
                        "session_id": "rollout-1:agent_1",
                        "role": "agent_1",
                        "policy_name": "policy_1",
                    },
                )
            ],
            sessions={
                "agent_1": RoleSessionInfo(
                    role="agent_1",
                    session_id="rollout-1:agent_1",
                    policy_name="policy_1",
                )
            },
            reward_info={
                "final_result": "final answer",
                "agent_outputs": {"agent_1": "draft"},
            },
        )

        annotated = await framework._annotate_rollout_trajectories(
            rollout_result=rollout_result,
            sample_fields={
                "uid": "prompt-uid",
                "raw_prompt": [{"role": "user", "content": "solve"}],
                "data_source": "multi_agent",
                "reward_model": {"ground_truth": {"answer": "final answer"}},
                "extra_info": {"existing": "value"},
            },
            sample_idx=0,
        )

        assert captured["extra_info"]["existing"] == "value"
        assert captured["extra_info"]["final_result"] == "final answer"
        assert captured["extra_info"]["agent_outputs"] == {"agent_1": "draft"}
        assert annotated[0].reward_score == 0.8
        assert annotated[0].extra_fields["reward_extra_info"] == {"judge": "reward_worker"}

    def test_trajectory_to_tq_field_and_tag_uses_policy_processor(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework
        from uni_agent.trainer.framework.types import Trajectory

        class MarkerProcessor:
            def __init__(self, name):
                self.name = name

        calls = []

        def fake_compute_multi_modal_inputs(processor, input_ids, multi_modal_data):
            calls.append(("multi_modal_inputs", processor.name, dict(multi_modal_data or {})))
            return {"processor_name": processor.name}

        def fake_compute_position_ids(processor, input_ids, attention_mask, multi_modal_inputs):
            calls.append(("position_ids", processor.name, dict(multi_modal_inputs)))
            return torch.full_like(attention_mask, fill_value=7)

        original_compute_multi_modal_inputs = framework_module.compute_multi_modal_inputs
        original_compute_position_ids = framework_module.compute_position_ids
        framework_module.compute_multi_modal_inputs = fake_compute_multi_modal_inputs
        framework_module.compute_position_ids = fake_compute_position_ids
        try:
            framework = MultiAgentFramework(
                session_runtime=FakeMultiAgentSessionRuntime(),
                multi_agent_runner=lambda **_: None,
                role_policy_mapping={"agent_1": "policy_1", "agent_2": "policy_2"},
                rollout_config={"n": 1, "val_kwargs": {"n": 1}},
                processor=MarkerProcessor("fallback"),
                policy_processors={
                    "policy_1": MarkerProcessor("policy_1"),
                    "policy_2": MarkerProcessor("policy_2"),
                },
            )
            trajectory = Trajectory(
                prompt_ids=[1, 2],
                response_ids=[3],
                response_mask=[1],
                multi_modal_data={"images": ["image"]},
                extra_fields={
                    "role": "agent_2",
                    "policy_name": "policy_2",
                },
            )

            field, _ = framework._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields={"uid": "prompt-uid"},
                sample_idx=0,
                record_idx=0,
                rollout_id="rollout-1",
                uid="prompt-uid",
                global_steps=9,
            )
        finally:
            framework_module.compute_multi_modal_inputs = original_compute_multi_modal_inputs
            framework_module.compute_position_ids = original_compute_position_ids

        assert calls[0] == ("multi_modal_inputs", "policy_2", {"images": ["image"]})
        assert calls[1] == ("position_ids", "policy_2", {"processor_name": "policy_2"})
        assert field["multi_modal_inputs"] == {"processor_name": "policy_2"}

    def test_generate_sequences_writes_rollout_records_with_v1_compatible_keys(self):
        asyncio.run(self._run_generate_sequences())

    async def _run_generate_sequences(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework

        recorded_tq = RecordingTQ()
        original_tq = framework_module.tq
        original_converter = framework_module._list_of_tq_fields_to_tensordict
        framework_module.tq = recorded_tq
        framework_module._list_of_tq_fields_to_tensordict = lambda fields: fields

        async def runner(**kwargs):
            return {"reward_info": {"reward_score": 0.75, "reward_extra_info": {"judge": "ok"}}}

        prompts = TensorDict(
            {
                "uid": NonTensorStack(NonTensorData("prompt-uid")),
                "raw_prompt": NonTensorStack(NonTensorData([{"role": "user", "content": "solve"}])),
                "global_steps": torch.tensor([9]),
            },
            batch_size=1,
        )
        rollout_config = {"n": 1, "val_kwargs": {"n": 1}}
        runtime = FakeMultiAgentSessionRuntime()
        framework = MultiAgentFramework(
            session_runtime=runtime,
            multi_agent_runner=runner,
            role_policy_mapping={"agent_1": "policy_1", "agent_2": "policy_1"},
            rollout_config=rollout_config,
        )

        try:
            await framework.generate_sequences(prompts)
        finally:
            framework_module.tq = original_tq
            framework_module._list_of_tq_fields_to_tensordict = original_converter

        assert runtime.created[0]["role_policy_mapping"] == {"agent_1": "policy_1", "agent_2": "policy_1"}
        assert runtime.completed[0]["reward_info"]["reward_score"] == 0.75
        assert runtime.finalized == [runtime.created[0]["rollout_id"]]
        assert runtime.aborted == []

        assert recorded_tq.batch_puts[0]["partition_id"] == "train"
        assert recorded_tq.batch_puts[0]["keys"] == ["prompt-uid_0_0", "prompt-uid_0_1"]
        first_field = recorded_tq.batch_puts[0]["fields"][0]
        second_field = recorded_tq.batch_puts[0]["fields"][1]
        assert first_field["uid"] == "prompt-uid"
        assert "session_id" not in first_field
        assert "prompt_group_id" not in first_field
        assert first_field["sample_idx"] == 0
        assert first_field["record_idx"] == 0
        assert first_field["rollout_id"] == runtime.created[0]["rollout_id"]
        assert first_field["role"] == "agent_1"
        assert "agent_role" not in first_field
        assert first_field["role_session_id"] == f"{runtime.created[0]['rollout_id']}:agent_1"
        assert first_field["policy_name"] == "policy_1"
        assert first_field["rm_scores"][-1].item() == 0.75
        assert first_field["reward_extra_info"] == {"judge": "ok"}
        assert second_field["record_idx"] == 1
        assert second_field["role"] == "agent_2"

        assert recorded_tq.batch_puts[0]["tags"][0]["uid"] == "prompt-uid"
        assert "prompt_group_id" not in recorded_tq.batch_puts[0]["tags"][0]
        assert recorded_tq.batch_puts[0]["tags"][0]["sample_idx"] == 0
        assert recorded_tq.batch_puts[0]["tags"][0]["record_idx"] == 0
        assert recorded_tq.batch_puts[0]["tags"][0]["role"] == "agent_1"
        assert recorded_tq.batch_puts[0]["tags"][0]["policy_name"] == "policy_1"
        assert recorded_tq.kv_puts[-1] == {
            "key": "prompt-uid",
            "partition_id": "train",
            "tag": {"status": "finished"},
        }

    def test_generate_sequences_writes_policy_routable_multi_rollout_records_to_tq(self):
        asyncio.run(self._run_generate_sequences_writes_policy_routable_multi_rollout_records_to_tq())

    async def _run_generate_sequences_writes_policy_routable_multi_rollout_records_to_tq(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework
        from uni_agent.trainer.framework.types import Trajectory
        from uni_agent.trainer.gateway.types import MultiAgentRolloutResult, RoleSessionInfo

        recorded_tq = RecordingTQ()
        original_tq = framework_module.tq
        original_converter = framework_module._list_of_tq_fields_to_tensordict
        framework_module.tq = recorded_tq
        framework_module._list_of_tq_fields_to_tensordict = lambda fields: fields

        role_policy_mapping = {
            "agent_1": "policy_1",
            "agent_2": "policy_1",
            "agent_3": "policy_2",
        }

        def make_result(rollout_id, reward_score):
            return MultiAgentRolloutResult(
                rollout_id=rollout_id,
                trajectories=[
                    Trajectory(
                        prompt_ids=[1],
                        response_ids=[10 + record_idx],
                        response_mask=[1],
                        num_turns=record_idx + 1,
                        extra_fields={
                            "session_id": f"{rollout_id}:{role}",
                            "role": role,
                            "policy_name": policy_name,
                        },
                    )
                    for record_idx, (role, policy_name) in enumerate(role_policy_mapping.items())
                ],
                sessions={
                    role: RoleSessionInfo(role=role, session_id=f"{rollout_id}:{role}", policy_name=policy_name)
                    for role, policy_name in role_policy_mapping.items()
                },
                reward_info={"reward_score": reward_score},
            )

        async def runner(**kwargs):
            return {"reward_info": {"reward_score": 0.5}}

        prompts = TensorDict(
            {
                "uid": NonTensorStack(NonTensorData("prompt-uid")),
                "raw_prompt": NonTensorStack(NonTensorData([{"role": "user", "content": "solve"}])),
                "global_steps": torch.tensor([9]),
            },
            batch_size=1,
        )
        runtime = FakeMultiAgentSessionRuntime()
        runtime.finalize_results = lambda rollout_id: make_result(rollout_id, reward_score=0.5)
        framework = MultiAgentFramework(
            session_runtime=runtime,
            multi_agent_runner=runner,
            role_policy_mapping=role_policy_mapping,
            rollout_config={"n": 2, "val_kwargs": {"n": 1}},
        )

        try:
            await framework.generate_sequences(prompts)
        finally:
            framework_module.tq = original_tq
            framework_module._list_of_tq_fields_to_tensordict = original_converter

        first_rollout_id = runtime.created[0]["rollout_id"]
        second_rollout_id = runtime.created[1]["rollout_id"]

        assert len(runtime.created) == 2
        assert runtime.created[0]["role_policy_mapping"] == role_policy_mapping
        assert recorded_tq.batch_puts[0]["keys"] == [
            "prompt-uid_0_0",
            "prompt-uid_0_1",
            "prompt-uid_0_2",
        ]
        assert recorded_tq.batch_puts[1]["keys"] == [
            "prompt-uid_1_0",
            "prompt-uid_1_1",
            "prompt-uid_1_2",
        ]

        all_fields = recorded_tq.batch_puts[0]["fields"] + recorded_tq.batch_puts[1]["fields"]
        all_tags = recorded_tq.batch_puts[0]["tags"] + recorded_tq.batch_puts[1]["tags"]
        expected = [
            (0, 0, "agent_1", "policy_1", first_rollout_id),
            (0, 1, "agent_2", "policy_1", first_rollout_id),
            (0, 2, "agent_3", "policy_2", first_rollout_id),
            (1, 0, "agent_1", "policy_1", second_rollout_id),
            (1, 1, "agent_2", "policy_1", second_rollout_id),
            (1, 2, "agent_3", "policy_2", second_rollout_id),
        ]
        for field, tag, (sample_idx, record_idx, role, policy_name, rollout_id) in zip(
            all_fields,
            all_tags,
            expected,
            strict=True,
        ):
            assert field["uid"] == "prompt-uid"
            assert "session_id" not in field
            assert field["rollout_id"] == rollout_id
            assert field["sample_idx"] == sample_idx
            assert field["record_idx"] == record_idx
            assert field["role"] == role
            assert "agent_role" not in field
            assert field["role_session_id"] == f"{rollout_id}:{role}"
            assert field["policy_name"] == policy_name
            assert field["rm_scores"][-1].item() == 0.5
            assert tag["uid"] == "prompt-uid"
            assert tag["rollout_id"] == rollout_id
            assert tag["sample_idx"] == sample_idx
            assert tag["record_idx"] == record_idx
            assert tag["role"] == role
            assert tag["policy_name"] == policy_name

    def test_generate_sequences_marks_prompt_running_and_does_not_touch_replay_buffer(self):
        asyncio.run(self._run_generate_sequences_marks_prompt_running_and_does_not_touch_replay_buffer())

    async def _run_generate_sequences_marks_prompt_running_and_does_not_touch_replay_buffer(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework

        recorded_tq = RecordingTQ()
        original_tq = framework_module.tq
        original_converter = framework_module._list_of_tq_fields_to_tensordict
        framework_module.tq = recorded_tq
        framework_module._list_of_tq_fields_to_tensordict = lambda fields: fields

        async def runner(**kwargs):
            return {"reward_info": {"reward_score": 0.75}}

        prompts = TensorDict(
            {
                "uid": NonTensorStack(NonTensorData("prompt-uid")),
                "raw_prompt": NonTensorStack(NonTensorData([{"role": "user", "content": "solve"}])),
                "global_steps": torch.tensor([11]),
            },
            batch_size=1,
        )
        framework = MultiAgentFramework(
            session_runtime=FakeMultiAgentSessionRuntime(),
            multi_agent_runner=runner,
            role_policy_mapping={"agent_1": "policy_1"},
            rollout_config={"n": 1, "val_kwargs": {"n": 1}},
        )

        try:
            await framework.generate_sequences(prompts)
        finally:
            framework_module.tq = original_tq
            framework_module._list_of_tq_fields_to_tensordict = original_converter

        assert recorded_tq.kv_puts[0] == {
            "key": "prompt-uid",
            "partition_id": "train",
            "tag": {"global_steps": 11, "status": "running"},
        }
        assert recorded_tq.kv_puts[-1] == {
            "key": "prompt-uid",
            "partition_id": "train",
            "tag": {"status": "finished"},
        }

    def test_generate_sequences_limits_concurrent_multi_agent_rollouts(self):
        asyncio.run(self._run_generate_sequences_with_concurrency_limit())

    async def _run_generate_sequences_with_concurrency_limit(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework

        recorded_tq = RecordingTQ()
        original_tq = framework_module.tq
        original_converter = framework_module._list_of_tq_fields_to_tensordict
        framework_module.tq = recorded_tq
        framework_module._list_of_tq_fields_to_tensordict = lambda fields: fields

        running = 0
        max_running = 0

        async def runner(**kwargs):
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.01)
            running -= 1
            return {"reward_info": {"reward_score": 1.0}}

        prompts = TensorDict(
            {
                "uid": NonTensorStack(NonTensorData("prompt-uid")),
                "raw_prompt": NonTensorStack(NonTensorData([{"role": "user", "content": "solve"}])),
                "global_steps": torch.tensor([9]),
            },
            batch_size=1,
        )
        runtime = FakeMultiAgentSessionRuntime()
        framework = MultiAgentFramework(
            session_runtime=runtime,
            multi_agent_runner=runner,
            role_policy_mapping={"agent_1": "policy_1"},
            rollout_config={"n": 3, "val_kwargs": {"n": 1}},
            max_concurrent_rollouts=1,
        )

        try:
            await framework.generate_sequences(prompts)
        finally:
            framework_module.tq = original_tq
            framework_module._list_of_tq_fields_to_tensordict = original_converter

        assert max_running == 1
        assert len(runtime.created) == 3
        assert recorded_tq.batch_puts[0]["keys"] == ["prompt-uid_0_0", "prompt-uid_0_1"]
        assert recorded_tq.batch_puts[1]["keys"] == ["prompt-uid_1_0", "prompt-uid_1_1"]
        assert recorded_tq.batch_puts[2]["keys"] == ["prompt-uid_2_0", "prompt-uid_2_1"]

    def test_multi_agent_rollout_annotation_starts_before_sibling_rollout_finishes(self):
        asyncio.run(self._run_multi_agent_rollout_annotation_starts_before_sibling_rollout_finishes())

    async def _run_multi_agent_rollout_annotation_starts_before_sibling_rollout_finishes(self):
        _install_dependency_stubs()
        from uni_agent.trainer.framework import framework as framework_module
        from uni_agent.trainer.framework.framework import MultiAgentFramework
        from uni_agent.trainer.framework.types import Trajectory

        first_annotation_started = asyncio.Event()
        second_rollout_timed_out_waiting_for_annotation = False
        written_sample_indices = []

        class TimingFramework(MultiAgentFramework):
            async def run_rollout(self, *, rollout_id=None, **kwargs):
                nonlocal second_rollout_timed_out_waiting_for_annotation
                sample_idx = int(rollout_id.split("-")[4])
                if sample_idx == 1:
                    try:
                        await asyncio.wait_for(first_annotation_started.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        second_rollout_timed_out_waiting_for_annotation = True
                return SimpleNamespace(
                    rollout_id=rollout_id,
                    trajectories=[
                        Trajectory(
                            prompt_ids=[1],
                            response_ids=[2],
                            response_mask=[1],
                            extra_fields={"role": "agent_1", "policy_name": "policy_1"},
                        )
                    ],
                    reward_info={"reward_score": 1.0},
                )

            async def _annotate_rollout_trajectories(self, **kwargs):
                first_annotation_started.set()
                return kwargs["rollout_result"].trajectories

            async def _write_multi_agent_rollout_to_tq(self, *, sample_idx, **kwargs):
                written_sample_indices.append(sample_idx)

        original_tq = framework_module.tq
        framework_module.tq = RecordingTQ()
        try:
            framework = TimingFramework(
                session_runtime=object(),
                multi_agent_runner=lambda **_: None,
                role_policy_mapping={"agent_1": "policy_1"},
                rollout_config={"n": 2, "val_kwargs": {"n": 1}},
                max_concurrent_rollouts=2,
            )

            result = await framework._run_prompt_rollouts_to_tq(
                raw_prompt=[{"role": "user", "content": "solve"}],
                sample_fields={"uid": "prompt-uid"},
                sample_index=0,
                global_steps=1,
                partition_id="train",
                num_rollouts=2,
            )
        finally:
            framework_module.tq = original_tq

        assert second_rollout_timed_out_waiting_for_annotation is False
        assert written_sample_indices == [0, 1]
        assert result["num_success_rollouts"] == 2


class TestMultiAgentGateway:
    def test_single_agent_chat_turn_keeps_existing_private_interface(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _GatewayActor

        signature = inspect.signature(_GatewayActor._chat_completions_turn)

        assert list(signature.parameters) == ["self", "session_id", "payload"]


class TestBuildSamplingParams:
    def test_requests_logprobs_by_default(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _build_sampling_params

        params = _build_sampling_params(
            payload={"temperature": 0.7},
            base_sampling_params={},
            allowed_request_sampling_param_keys=frozenset({"temperature", "logprobs"}),
        )

        assert params["logprobs"] is True
        assert params["temperature"] == 0.7

    def test_client_can_opt_out_of_logprobs(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _build_sampling_params

        params = _build_sampling_params(
            payload={"logprobs": False},
            base_sampling_params={},
            allowed_request_sampling_param_keys=frozenset({"temperature", "logprobs"}),
        )

        assert params["logprobs"] is False

    def test_default_allowed_request_keys_include_logprobs(self):
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS

        assert "logprobs" in _DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS

    def test_rollout_routes_roles_to_policy_backends_and_finalizes_metadata(self):
        asyncio.run(self._run_rollout())

    async def _run_rollout(self):
        import httpx

        backend = RecordingBackend(
            [
                ("from agent 1", "completed", {"global_steps": 10, "min_global_steps": 9, "max_global_steps": 10}),
                ("from agent 3", "completed", {"global_steps": 8, "min_global_steps": 8, "max_global_steps": 8}),
            ]
        )
        actor = _make_gateway(backend)
        handle = await actor.create_multi_agent_rollout(
            "rollout-1",
            role_policy_mapping={
                "agent_1": "policy_1",
                "agent_2": "policy_1",
                "agent_3": "policy_2",
            },
        )

        assert handle.rollout_id == "rollout-1"
        assert handle.base_url == "http://testserver/rollouts/rollout-1/v1"
        assert handle.sessions["agent_1"].policy_name == "policy_1"
        assert handle.sessions["agent_3"].policy_name == "policy_2"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=actor._app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/rollouts/rollout-1/v1/chat/completions",
                json={
                    "model": "agent_1",
                    "messages": [{"role": "user", "content": "work on subtask"}],
                    "max_tokens": 16,
                },
            )
            second = await client.post(
                "/rollouts/rollout-1/v1/chat/completions",
                json={
                    "model": "agent_3",
                    "messages": [{"role": "user", "content": "review subtask"}],
                    "max_tokens": 16,
                },
            )

        assert first.status_code == 200
        assert first.json()["choices"][0]["message"]["content"] == "from agent 1"
        assert second.status_code == 200
        assert second.json()["choices"][0]["message"]["content"] == "from agent 3"

        assert backend.requests[0]["request_id"] == "rollout-1:agent_1"
        assert backend.requests[0]["kwargs"]["rollout_id"] == "rollout-1"
        assert backend.requests[0]["kwargs"]["role"] == "agent_1"
        assert backend.requests[0]["kwargs"]["policy_name"] == "policy_1"
        assert backend.requests[1]["request_id"] == "rollout-1:agent_3"
        assert backend.requests[1]["kwargs"]["policy_name"] == "policy_2"

        await actor.complete_multi_agent_rollout("rollout-1", reward_info={"reward_score": 1.0})
        result = await actor.finalize_multi_agent_rollout("rollout-1")

        assert result.rollout_id == "rollout-1"
        assert set(result.sessions) == {"agent_1", "agent_2", "agent_3"}
        assert len(result.trajectories) == 2
        by_role = {trajectory.extra_fields["role"]: trajectory for trajectory in result.trajectories}
        assert by_role["agent_1"].extra_fields["policy_name"] == "policy_1"
        assert by_role["agent_1"].extra_fields["rollout_id"] == "rollout-1"
        assert by_role["agent_1"].extra_fields["min_global_steps"] == 9
        assert by_role["agent_1"].extra_fields["max_global_steps"] == 10
        assert by_role["agent_1"].reward_info["reward_score"] == 1.0
        assert by_role["agent_3"].extra_fields["policy_name"] == "policy_2"
        assert by_role["agent_3"].extra_fields["min_global_steps"] == 8
        assert by_role["agent_3"].extra_fields["max_global_steps"] == 8

    def test_rollout_uses_policy_specific_tokenizer(self):
        asyncio.run(self._run_rollout_uses_policy_specific_tokenizer())

    def test_rollout_accumulates_version_range_across_multi_turn_generation(self):
        asyncio.run(self._run_rollout_accumulates_version_range_across_multi_turn_generation())

    async def _run_rollout_accumulates_version_range_across_multi_turn_generation(self):
        import httpx

        backend = RecordingBackend(
            [
                ("first", "completed", {"global_steps": 8, "min_global_steps": 7, "max_global_steps": 8}),
                ("second", "completed", {"global_steps": 10, "min_global_steps": 9, "max_global_steps": 10}),
            ]
        )
        actor = _make_gateway(backend)
        actor._system_prompt = []
        await actor.create_session("session-1")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=actor._app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/sessions/session-1/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "first question"}],
                    "max_tokens": 16,
                },
            )
            second = await client.post(
                "/sessions/session-1/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "follow-up"},
                    ],
                    "max_tokens": 16,
                },
            )

        assert first.status_code == 200
        assert second.status_code == 200
        trajectories = await actor.finalize_session("session-1")
        assert len(trajectories) == 1
        assert trajectories[0].extra_fields["min_global_steps"] == 7
        assert trajectories[0].extra_fields["max_global_steps"] == 10

    async def _run_rollout_uses_policy_specific_tokenizer(self):
        import httpx

        backend = RecordingBackend([("from policy 1", "completed"), ("from policy 2", "completed")])
        _install_dependency_stubs()
        from uni_agent.trainer.gateway.gateway import _GatewayActor

        actor = _GatewayActor(
            FakeTokenizer(),
            backend,
            policy_tokenizers={
                "policy_1": PrefixTokenizer("P1"),
                "policy_2": PrefixTokenizer("P2"),
            },
        )
        actor._server_base_url = "http://testserver"
        await actor.create_multi_agent_rollout(
            "rollout-1",
            role_policy_mapping={
                "agent_1": "policy_1",
                "agent_2": "policy_2",
            },
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=actor._app),
            base_url="http://testserver",
        ) as client:
            await client.post(
                "/rollouts/rollout-1/v1/chat/completions",
                json={
                    "model": "agent_1",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
            await client.post(
                "/rollouts/rollout-1/v1/chat/completions",
                json={
                    "model": "agent_2",
                    "messages": [{"role": "user", "content": "second"}],
                },
            )

        assert backend.requests[0]["prompt_ids"][:2] == list(b"P1")
        assert backend.requests[1]["prompt_ids"][:2] == list(b"P2")

    def test_rollout_uses_policy_specific_tool_parser_names(self, monkeypatch):
        _install_dependency_stubs()
        import uni_agent.trainer.gateway.gateway as gateway_module

        class RecordingToolParser:
            calls = []

            @staticmethod
            def get_tool_parser(name, tokenizer):
                parser = SimpleNamespace(name=name, tokenizer=tokenizer)
                RecordingToolParser.calls.append(parser)
                return parser

        monkeypatch.setattr(gateway_module, "ToolParser", RecordingToolParser)

        actor = gateway_module._GatewayActor(
            FakeTokenizer(),
            RecordingBackend([]),
            policy_tokenizers={
                "policy_1": PrefixTokenizer("P1"),
                "policy_2": PrefixTokenizer("P2"),
            },
            policy_tool_parser_names={
                "policy_1": "qwen3_coder",
                "policy_2": "hermes",
            },
        )

        assert actor._tool_parser_for_policy("policy_1").name == "qwen3_coder"
        assert actor._tool_parser_for_policy("policy_2").name == "hermes"
        assert actor._tool_parser_for_policy("unknown") is None
        assert [(call.name, call.tokenizer.prefix) for call in RecordingToolParser.calls] == [
            ("qwen3_coder", "P1"),
            ("hermes", "P2"),
        ]

    def test_rollout_does_not_register_anthropic_messages_endpoint(self):
        asyncio.run(self._run_rollout_does_not_register_anthropic_messages_endpoint())

    async def _run_rollout_does_not_register_anthropic_messages_endpoint(self):
        import httpx

        backend = RecordingBackend([("should not be used", "completed")])
        actor = _make_gateway(backend)
        await actor.create_multi_agent_rollout(
            "rollout-1",
            role_policy_mapping={"agent_1": "policy_1"},
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=actor._app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/rollouts/rollout-1/v1/messages",
                json={
                    "model": "agent_1",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "work on subtask"}],
                },
            )

        assert response.status_code == 404
        assert backend.requests == []

    def test_wait_for_multi_agent_rollout_completion_unblocks_on_complete(self):
        asyncio.run(self._run_wait_for_multi_agent_rollout_completion())

    async def _run_wait_for_multi_agent_rollout_completion(self):
        backend = RecordingBackend([])
        actor = _make_gateway(backend)
        await actor.create_multi_agent_rollout(
            "rollout-wait",
            role_policy_mapping={"agent_1": "policy_1"},
        )

        waiter = asyncio.create_task(
            actor.wait_for_multi_agent_rollout_completion("rollout-wait", timeout=1.0)
        )
        await asyncio.sleep(0)

        assert not waiter.done()

        await actor.complete_multi_agent_rollout("rollout-wait", reward_info={"reward_score": 1.0})
        await waiter
