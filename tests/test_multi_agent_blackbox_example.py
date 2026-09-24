"""Lightweight checks for the multi-agent blackbox training example."""

from __future__ import annotations

import importlib
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml


EXAMPLE_DIR = Path("examples/multi_agent_blackbox")


def test_multi_agent_runner_importable():
    module = importlib.import_module("examples.multi_agent_blackbox.multi_agent_runner")

    assert module.multi_agent_runner is not None
    assert module.build_role_messages is not None


def test_worker_setup_hook_dotted_path_importable():
    module = importlib.import_module("examples.multi_agent_blackbox.verl_patch")

    assert callable(module.apply_worker_patch)


def test_multi_agent_runner_builds_role_messages_from_abstract_agents():
    from examples.multi_agent_blackbox.multi_agent_runner import build_role_messages

    messages = build_role_messages(
        raw_prompt=[{"role": "user", "content": "solve the task"}],
        role_policy_mapping={
            "agent_1": "policy_1",
            "agent_2": "policy_1",
            "agent_3": "policy_2",
        },
        mas_config={
            "agents": {
                "agent_1": {"system_prompt": "You plan and summarize."},
                "agent_2": {"system_prompt": "You inspect."},
                "agent_3": {"system_prompt": "You verify."},
            }
        },
    )

    assert list(messages) == ["agent_1", "agent_2", "agent_3"]
    assert messages["agent_1"][0] == {"role": "system", "content": "You plan and summarize."}
    assert messages["agent_1"][1]["content"] == "solve the task"
    assert messages["agent_2"][0]["content"] == "You inspect."


def test_multi_agent_runner_returns_rollout_result_for_reward_worker(monkeypatch):
    from examples.multi_agent_blackbox import multi_agent_runner as runner_module

    async def fake_chat_completion(*, role, **_):
        return {
            "agent_1": "plan",
            "agent_2": "work",
            "agent_3": "final answer",
        }[role]

    monkeypatch.setattr(runner_module, "_chat_completion", fake_chat_completion)

    result = asyncio.run(
        runner_module.multi_agent_runner(
            raw_prompt=[{"role": "user", "content": "solve"}],
            rollout=SimpleNamespace(
                base_url="http://gateway/rollouts/rollout-1/v1",
                sessions={
                    "agent_1": object(),
                    "agent_2": object(),
                    "agent_3": object(),
                },
            ),
            sample_index=0,
            session_runtime=None,
            role_policy_mapping={
                "agent_1": "policy_1",
                "agent_2": "policy_1",
                "agent_3": "policy_2",
            },
        )
    )

    assert "reward_score" not in result["reward_info"]
    assert result["reward_info"]["final_result"] == "final answer"
    assert result["reward_info"]["agent_outputs"] == {
        "agent_1": "plan",
        "agent_2": "work",
        "agent_3": "final answer",
    }


def test_random_routing_harness_uses_entry_agent_and_bounded_handoffs(monkeypatch):
    from examples.multi_agent_blackbox import random_routing_harness as harness

    calls = []

    async def fake_chat_completion(*, role, messages, agent_cfg, **_):
        calls.append({"role": role, "messages": messages, "agent_cfg": dict(agent_cfg)})
        return f"output from {role}"

    monkeypatch.setattr(harness, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        harness.random_routing_agent_runner(
            raw_prompt=[{"role": "user", "content": "solve"}],
            rollout=SimpleNamespace(
                base_url="http://gateway/rollouts/random/v1",
                sessions={"agent_1": object(), "agent_2": object(), "agent_3": object()},
            ),
            sample_index=7,
            session_runtime=None,
            role_policy_mapping={
                "agent_1": "policy_1",
                "agent_2": "policy_2",
                "agent_3": "policy_3",
            },
            mas_config={
                "harness": {
                    "entry_agent": "agent_1",
                    "min_rounds": 3,
                    "max_rounds": 3,
                    "random_seed": 42,
                    "route_weights": {"agent_1": 0, "agent_2": 1, "agent_3": 0},
                    "output_tokens_min": 4096,
                    "output_tokens_max": 4096,
                    "minimum_completion_ratio": 0.75,
                },
                "agents": {},
            },
            max_tokens=8192,
        )
    )

    assert result["route"] == ["agent_1", "agent_2", "agent_2"]
    assert [call["role"] for call in calls] == result["route"]
    assert all(call["agent_cfg"]["max_tokens"] == 4096 for call in calls)
    assert all(call["agent_cfg"]["min_tokens"] == 3072 for call in calls)
    assert all("ignore_eos" not in call["agent_cfg"] for call in calls)
    assert "entered through agent_1" in calls[0]["messages"][-1]["content"]
    assert "Previous agent output:\noutput from agent_1" in calls[1]["messages"][-1]["content"]
    assert result["reward_info"]["round_count"] == 3
    assert result["reward_info"]["entry_agent"] == "agent_1"
    assert result["reward_info"]["final_result"] == "output from agent_2"


def test_random_routing_harness_can_fix_both_endpoints(monkeypatch):
    from examples.multi_agent_blackbox import random_routing_harness as harness

    calls = []

    async def fake_chat_completion(*, role, **_):
        calls.append(role)
        return f"output from {role}"

    monkeypatch.setattr(harness, "_chat_completion", fake_chat_completion)
    result = asyncio.run(
        harness.random_routing_agent_runner(
            raw_prompt="solve",
            rollout=SimpleNamespace(
                base_url="http://gateway/rollouts/random/v1",
                sessions={"agent_1": object(), "agent_2": object()},
            ),
            sample_index=3,
            session_runtime=None,
            role_policy_mapping={"agent_1": "policy_1", "agent_2": "policy_2"},
            mas_config={
                "harness": {
                    "entry_agent": "agent_1",
                    "final_agent": "agent_1",
                    "min_rounds": 7,
                    "max_rounds": 7,
                    "random_seed": 42,
                    "route_weights": {"agent_1": 1, "agent_2": 1},
                    "output_tokens_min": 128,
                    "output_tokens_max": 128,
                    "minimum_completion_ratio": 0.9,
                },
                "agents": {},
            },
            max_tokens=128,
        )
    )

    assert len(result["route"]) == 7
    assert result["route"][0] == result["route"][-1] == "agent_1"
    assert calls == result["route"]
    assert result["reward_info"]["final_agent"] == "agent_1"
    assert result["final_result"] == "output from agent_1"


def test_random_routing_recipe_selects_new_harness_and_random_token_range():
    mas_cfg = yaml.safe_load(
        (EXAMPLE_DIR / "config" / "mas_config_random_routing.yaml").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(
        (EXAMPLE_DIR / "config" / "multi_agent_blackbox_borrow_random_routing.yaml").read_text(
            encoding="utf-8"
        )
    )

    harness_cfg = mas_cfg["harness"]
    assert harness_cfg["entry_agent"] == "agent_1"
    assert (harness_cfg["min_rounds"], harness_cfg["max_rounds"]) == (7, 10)
    assert harness_cfg["output_tokens_min"] >= 8192
    assert harness_cfg["output_tokens_max"] > harness_cfg["output_tokens_min"]
    assert all("ignore_eos" not in agent for agent in mas_cfg["agents"].values())
    framework_cfg = cfg["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]
    assert framework_cfg["multi_agent_runner_fqn"].endswith(
        "random_routing_harness.random_routing_agent_runner"
    )
    assert framework_cfg["multi_agent_runner_kwargs"]["max_tokens"] == 10240
    script = (EXAMPLE_DIR / "scripts" / "run_e2e_borrow_verify.sh").read_text(encoding="utf-8")
    assert 'BORROW_LOAD_PROFILE:-standard}" == "random_routing"' in script
    assert "multi_agent_blackbox_borrow_random_routing" in script
    assert "mas_config_random_routing.yaml" in script


def test_chat_completion_forwards_deterministic_length_options(monkeypatch):
    from examples.multi_agent_blackbox import multi_agent_runner as runner_module

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "done"}}]}

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, json):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    result = asyncio.run(
        runner_module._chat_completion(
            base_url="http://gateway/v1",
            role="agent_1",
            messages=[{"role": "user", "content": "load"}],
            agent_cfg={"max_tokens": 64, "min_tokens": 63, "ignore_eos": True},
            max_tokens=32,
            request_timeout_seconds=12.0,
        )
    )

    assert result == "done"
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["min_tokens"] == 63
    assert captured["payload"]["ignore_eos"] is True


def test_chat_completion_retries_interrupted_dynamic_inference_request(monkeypatch):
    from examples.multi_agent_blackbox import multi_agent_runner as runner_module

    attempts = []

    class FakeTransportError(Exception):
        pass

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "recovered"}}]}

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, json):
            attempts.append((url, json))
            if len(attempts) == 1:
                raise FakeTransportError("replica was reconfigured")
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(
            AsyncClient=FakeClient,
            TransportError=FakeTransportError,
            TimeoutException=TimeoutError,
        ),
    )

    result = asyncio.run(
        runner_module._chat_completion(
            base_url="http://gateway/v1",
            role="agent_2",
            messages=[{"role": "user", "content": "load"}],
            agent_cfg={"request_max_retries": 2, "request_retry_backoff_seconds": 0},
            max_tokens=32,
            request_timeout_seconds=12.0,
        )
    )

    assert result == "recovered"
    assert len(attempts) == 2


def test_borrow_verify_recipe_forces_asymmetric_load_and_lends_to_both_busy_policies():
    mas_cfg = yaml.safe_load(
        (EXAMPLE_DIR / "config" / "mas_config_borrow_verify.yaml").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(
        (EXAMPLE_DIR / "config" / "multi_agent_blackbox_borrow_verify.yaml").read_text(
            encoding="utf-8"
        )
    )

    for agent in mas_cfg["agents"].values():
        assert "min_tokens" not in agent  # async resume can reduce the remaining budget
        assert agent["ignore_eos"] is True
    assert mas_cfg["agents"]["agent_2"]["max_tokens"] < mas_cfg["agents"]["agent_1"]["max_tokens"]
    assert mas_cfg["agents"]["agent_2"]["max_tokens"] < mas_cfg["agents"]["agent_3"]["max_tokens"]
    dynamic = cfg["dynamic_inference_scheduling"]
    assert dynamic["resource_usage"]["kv_enter"] > dynamic["resource_usage"]["kv_post_lend_max"]
    assert dynamic["borrowing"]["guest_master_port_range"] == [20010, 20522]
    assert dynamic["borrowing"]["guest_master_port_stride"] == 4
    assert dynamic["resource_usage"]["queue_signal_enabled"] is False
    assert dynamic["borrowing"]["pairs"] == [
        {"home": "policy_1", "donor": "policy_2"},
        {"home": "policy_1", "donor": "policy_3"},
        {"home": "policy_2", "donor": "policy_1"},
        {"home": "policy_2", "donor": "policy_3"},
        {"home": "policy_3", "donor": "policy_1"},
        {"home": "policy_3", "donor": "policy_2"},
    ]
    assert cfg["policies"]["policy_2"]["ppo_trainer_overrides"]["trainer"]["n_gpus_per_node"] == 4
    assert cfg["policies"]["policy_2"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["n_gpus_per_node"] == 12
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["disable_log_stats"] is False
    assert cfg["policies"]["policy_2"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["disable_log_stats"] is False

    script = (EXAMPLE_DIR / "scripts" / "run_e2e_borrow_verify.sh").read_text(encoding="utf-8")
    assert 'TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-multi_agent_blackbox_borrow_verify}"' in script
    assert 'POLICY_2_N_GPUS_PER_NODE="${POLICY_2_N_GPUS_PER_NODE:-4}"' in script
    assert 'POLICY_2_ROLLOUT_N_GPUS_PER_NODE="${POLICY_2_ROLLOUT_N_GPUS_PER_NODE:-12}"' in script
    assert 'DYNAMIC_INFERENCE_SCHEDULING="${DYNAMIC_INFERENCE_SCHEDULING:-true}"' in script


def test_multi_agent_reward_function_scores_injected_final_result():
    from examples.multi_agent_blackbox.reward import compute_score

    result = compute_score(
        data_source="multi_agent",
        solution_str="ignored",
        ground_truth={"answer": "final answer"},
        extra_info={
            "final_result": "The final answer is here.",
            "agent_outputs": {"agent_1": "plan", "agent_2": "work"},
        },
    )

    assert result["score"] == 1.0
    assert result["reward_extra_info"]["reward_source"] == "expected_substring"
    assert result["reward_extra_info"]["num_agent_outputs"] == 2


def test_multi_agent_blackbox_yaml_exposes_framework_and_policy_mapping():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))
    af_cfg = cfg["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]

    assert list(cfg)[-1] == "policies"
    assert af_cfg["framework_class_fqn"] == (
        "examples.multi_agent_blackbox.framework.RemoteMultiAgentFramework"
    )
    assert af_cfg["multi_agent_runner_fqn"] == "examples.multi_agent_blackbox.multi_agent_runner.multi_agent_runner"
    assert af_cfg["role_policy_mapping"] == {
        "agent_1": "policy_1",
        "agent_2": "policy_2",
        "agent_3": "policy_3",
    }
    assert cfg["actor_rollout_ref"]["rollout"]["n"] == 4
    assert "reward_function_fqn" not in af_cfg["multi_agent_runner_kwargs"]
    assert cfg["reward"]["custom_reward_function"] == {
        "path": "pkg://examples/multi_agent_blackbox.reward",
        "name": "compute_score",
    }
    assert "models" not in cfg
    assert set(cfg["policies"]) == {"policy_1", "policy_2", "policy_3"}
    assert "name" not in cfg["policies"]["policy_1"]
    assert "name" not in cfg["policies"]["policy_2"]
    assert "name" not in cfg["policies"]["policy_3"]
    assert cfg["ppo_trainer_config_name"] == "ppo_trainer"
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_1"]
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_2"]
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_3"]


def test_external_multi_agent_recipe_exposes_command_runner_contract():
    cfg = yaml.safe_load(
        (EXAMPLE_DIR / "config" / "multi_agent_blackbox_external.yaml").read_text(encoding="utf-8")
    )
    runner_cfg = cfg["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]
    kwargs = runner_cfg["multi_agent_runner_kwargs"]

    assert runner_cfg["multi_agent_runner_fqn"].endswith(
        "external_mas_runner.external_mas_runner"
    )
    assert kwargs["execution"]["backend"] == "local_process"
    assert kwargs["execution"]["ray"]["num_cpus"] == 1
    assert kwargs["execution"]["ray"]["scheduling_strategy"] == "SPREAD"
    assert kwargs["command"]["argv"][2] == (
        "examples.multi_agent_blackbox.scripts.three_agent_external_mas"
    )
    assert kwargs["config"]["injection"]["agent_model_path"] == "model"


def test_external_mas_training_script_selects_external_recipe_without_callable_config():
    script_path = EXAMPLE_DIR / "scripts" / "run_external_mas_train.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "--config-name=multi_agent_blackbox_external" in content
    assert "python -m examples.multi_agent_blackbox.scripts.three_agent_external_mas" not in content
    assert "multi_agent_runner_kwargs.mas_config_path" not in content
    assert "MAS_CONFIG_PATH" not in content
    assert "POLICY_1_MODEL_PATH" in content
    assert "POLICY_2_MODEL_PATH" in content
    assert "data.train_files" in content
    assert "data.val_files" in content
    assert "for var in POLICY_1_MODEL_PATH POLICY_2_MODEL_PATH POLICY_3_MODEL_PATH; do" in content
    assert "--config-name=multi_agent_blackbox_external" in content
    assert "--config-path=\"${REPO_ROOT}/examples/multi_agent_blackbox/config\"" in content
    assert 'data.train_files="[\'${TRAIN_DATA}\']"' in content
    assert 'data.val_files="[\'${VAL_DATA}\']"' in content
    assert "actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_kwargs.config.template_path=${MAS_TEMPLATE_PATH}" in content


def test_multi_agent_launchers_configure_shared_outer_ppo_mini_batch_size():
    for script_name in ("run_e2e_train.sh", "run_external_mas_train.sh"):
        content = (EXAMPLE_DIR / "scripts" / script_name).read_text(encoding="utf-8")

        assert 'PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"' in content
        assert "PPO_MINI_BATCH_SIZE" in content
        assert "trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}" in content
        assert "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" in content
        assert "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.actor.ppo_mini_batch_size" not in content
        assert "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.actor.ppo_mini_batch_size" not in content


def test_multi_agent_blackbox_yaml_uses_public_ppo_trainer_base_per_policy():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert "hydra" not in cfg
    assert "defaults" not in cfg
    assert cfg["ppo_trainer_config_source"] == "verl.trainer.config"
    assert "policy_ppo_trainer_base" not in cfg

    assert cfg["ppo_trainer_config_name"] == "ppo_trainer"
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_1"]
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_2"]
    assert "ppo_trainer_config_name" not in cfg["policies"]["policy_3"]
    for policy_name in ("policy_1", "policy_2", "policy_3"):
        assert "use_v1" not in cfg["policies"][policy_name]["ppo_trainer_overrides"].get("trainer", {})
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["data"]["train_batch_size"] == "${data.train_batch_size}"
    assert (
        cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["actor_rollout_ref"]["actor"]
        ["ppo_mini_batch_size"]
        == "${actor_rollout_ref.actor.ppo_mini_batch_size}"
    )
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6

    assert cfg["policies"]["policy_2"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6
    assert cfg["policies"]["policy_3"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6


def test_multi_agent_blackbox_yaml_projects_shared_ppo_mini_batch_size_to_policies():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert cfg["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 32
    for policy_name in ("policy_1", "policy_2", "policy_3"):
        assert (
            cfg["policies"][policy_name]["ppo_trainer_overrides"]["actor_rollout_ref"]["actor"]
            ["ppo_mini_batch_size"]
            == "${actor_rollout_ref.actor.ppo_mini_batch_size}"
        )


def test_multi_agent_blackbox_yaml_keeps_shared_runtime_topology_at_outer_level():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert cfg["trainer"]["v1"]["trainer_mode"] == "separate_async"
    assert cfg["trainer"]["v1"]["separate_async"]["parameter_sync_step"] == 1
    for policy_name in ("policy_1", "policy_2", "policy_3"):
        policy_trainer = cfg["policies"][policy_name]["ppo_trainer_overrides"].get("trainer", {})
        assert "v1" not in policy_trainer


def test_multi_agent_blackbox_yaml_keeps_top_p_per_policy():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert "top_p" not in cfg["actor_rollout_ref"]["rollout"]
    for policy_name in ("policy_1", "policy_2", "policy_3"):
        assert (
            cfg["policies"][policy_name]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["top_p"]
            == 1.0
        )


def test_multi_agent_blackbox_yaml_keeps_shared_data_and_algorithm_at_root_level():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    for policy_name in ("policy_1", "policy_2", "policy_3"):
        ppo_config = cfg["policies"][policy_name]["ppo_trainer_overrides"]

        assert cfg["data"]["train_files"] == "???"
        assert cfg["data"]["val_files"] == "???"
        assert ppo_config["data"]["train_batch_size"] == "${data.train_batch_size}"
        # algorithm/reward are injected from the outer config during policy
        # composition and are intentionally absent from YAML overrides.
        assert "algorithm" not in ppo_config
        assert "reward" not in ppo_config

        assert ppo_config["actor_rollout_ref"]["rollout"]["n"] == "${actor_rollout_ref.rollout.n}"
        # Temperature is owned per policy (no top-level reference): each policy
        # holds its own value, which drives both sampling and recomputation.
        assert ppo_config["actor_rollout_ref"]["rollout"]["temperature"] == 1.0
        assert ppo_config["actor_rollout_ref"]["rollout"]["top_p"] == 1.0


def test_multi_agent_blackbox_yaml_has_v1_compatible_transfer_queue_and_ray_defaults():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert cfg["transfer_queue"] == {
        "enable": True,
        "metrics": {
            "enabled": False,
            "port": 0,
        },
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 100000,
                "num_data_storage_units": 8,
            },
            "MooncakeStore": {
                "auto_init": False,
                "metadata_server": "localhost:50123",
                "master_server_address": "localhost:50124",
                "local_hostname": "localhost",
                "protocol": "tcp",
                "global_segment_size": 4294967296,
                "local_buffer_size": 1073741824,
                "device_name": "",
            },
        },
    }
    assert cfg["ray_kwargs"] == {
        "ray_init": {
            "num_cpus": None,
            "runtime_env": {
                "worker_process_setup_hook": (
                    "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
                ),
            },
        },
        "timeline_json_file": None,
    }


def test_multi_agent_blackbox_yaml_documents_per_policy_resource_isolation():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    policy_1 = cfg["policies"]["policy_1"]["ppo_trainer_overrides"]
    policy_2 = cfg["policies"]["policy_2"]["ppo_trainer_overrides"]
    policy_3 = cfg["policies"]["policy_3"]["ppo_trainer_overrides"]

    assert policy_1["trainer"]["nnodes"] == 1
    assert policy_2["trainer"]["nnodes"] == 1
    assert policy_3["trainer"]["nnodes"] == 1
    assert policy_1["trainer"]["n_gpus_per_node"] == 8
    assert policy_2["trainer"]["n_gpus_per_node"] == 8
    assert policy_3["trainer"]["n_gpus_per_node"] == 8
    assert "n_training_gpus_per_node" not in policy_1["trainer"]
    assert "n_training_gpus_per_node" not in policy_2["trainer"]
    assert "n_training_gpus_per_node" not in policy_3["trainer"]
    assert policy_1["trainer"]["default_local_dir"].endswith("/policy_1")
    assert policy_2["trainer"]["default_local_dir"].endswith("/policy_2")
    assert policy_3["trainer"]["default_local_dir"].endswith("/policy_3")
    assert policy_1["actor_rollout_ref"]["model"]["path"] == "???"
    assert policy_2["actor_rollout_ref"]["model"]["path"] == "???"
    assert policy_3["actor_rollout_ref"]["model"]["path"] == "???"
    assert policy_1["actor_rollout_ref"]["actor"]["fsdp_config"]["fsdp_size"] == 8
    assert policy_2["actor_rollout_ref"]["actor"]["fsdp_config"]["fsdp_size"] == 8
    assert policy_3["actor_rollout_ref"]["actor"]["fsdp_config"]["fsdp_size"] == 8
    assert policy_1["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 2
    assert policy_2["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 2
    assert policy_3["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 2
    assert policy_1["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] == "qwen3_coder"
    assert policy_2["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] == "hermes"
    assert policy_3["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] == "hermes"
    assert "enable" not in policy_1["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert "enable" not in policy_2["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert "enable" not in policy_3["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert "served_model_name" not in policy_1["actor_rollout_ref"]["rollout"]
    assert "served_model_name" not in policy_2["actor_rollout_ref"]["rollout"]
    assert "served_model_name" not in policy_3["actor_rollout_ref"]["rollout"]
    assert "prometheus" not in policy_1["actor_rollout_ref"]["rollout"]
    assert "prometheus" not in policy_2["actor_rollout_ref"]["rollout"]
    assert "prometheus" not in policy_3["actor_rollout_ref"]["rollout"]


def test_multi_agent_blackbox_hydra_config_initializes_multi_policy_trainer(monkeypatch):
    from hydra import compose, initialize_config_dir

    sys.modules.pop("uni_agent.trainer.multi_agents_ppo_trainer", None)
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    class ConfigSmokePolicyTrainer:
        instances = []

        def __init__(self, config):
            self.config = config
            self.__class__.instances.append(self)

    async_trainer_module = types.ModuleType("uni_agent.trainer.single_async_ppo_trainer")
    async_trainer_module.SingleAsyncPPOTrainer = ConfigSmokePolicyTrainer
    monkeypatch.setitem(sys.modules, "uni_agent.trainer.single_async_ppo_trainer", async_trainer_module)

    config_dir = str((EXAMPLE_DIR / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="multi_agent_blackbox",
            overrides=[
                "data.train_files=[train.parquet]",
                "data.val_files=[val.parquet]",
                "trainer.total_training_steps=1",
                "actor_rollout_ref.rollout.n=8",
                "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.model.path=/models/policy_1",
                "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.model.path=/models/policy_2",
                "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.model.path=/models/policy_3",
                "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.top_p=0.95",
                "policies.policy_1.ppo_trainer_overrides.trainer.nnodes=2",
                "policies.policy_2.ppo_trainer_overrides.trainer.nnodes=3",
                "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.temperature=0.5",
                "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.temperature=0.8",
                ],
            )

    trainer = MultiAgentsPPOTrainer(config=cfg)

    assert list(trainer.policy_trainers) == ["policy_1", "policy_2", "policy_3"]
    assert len(ConfigSmokePolicyTrainer.instances) == 3
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.actor.optim.lr == 1e-6
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.model.path == "/models/policy_1"
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.model.path == "/models/policy_2"
    assert trainer.policy_configs["policy_3"].actor_rollout_ref.model.path == "/models/policy_3"
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.rollout.n == 8
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.n == 8
    assert trainer.policy_configs["policy_3"].actor_rollout_ref.rollout.n == 8
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.rollout.temperature == 0.5
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.temperature == 0.8
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.rollout.top_p == 0.95
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.top_p == 1.0
    assert trainer.policy_configs["policy_1"].trainer.nnodes == 2
    assert trainer.policy_configs["policy_2"].trainer.nnodes == 3
    assert cfg.actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping == {
        "agent_1": "policy_1",
        "agent_2": "policy_2",
        "agent_3": "policy_3",
    }


def test_e2e_training_script_uses_multi_agents_entrypoint_and_config():
    script = EXAMPLE_DIR / "scripts" / "run_e2e_train.sh"
    content = script.read_text(encoding="utf-8")

    assert "-m uni_agent.trainer.main_multi_agents_ppo" in content
    assert '--config-name="${TRAIN_CONFIG_NAME}"' in content
    assert "--config-path=\"${REPO_ROOT}/examples/multi_agent_blackbox/config\"" in content
    assert "actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_kwargs.mas_config_path" in content
    assert "actor_rollout_ref.rollout.n=${ROLLOUT_N}" in content
    assert "POLICY_1_NNODES" in content
    assert "POLICY_2_NNODES" in content
    assert "POLICY_3_NNODES" in content
    assert "POLICY_1_N_GPUS_PER_NODE" in content
    assert "POLICY_2_N_GPUS_PER_NODE" in content
    assert "POLICY_3_N_GPUS_PER_NODE" in content
    assert "POLICY_1_TENSOR_PARALLEL_SIZE" in content
    assert "POLICY_2_TENSOR_PARALLEL_SIZE" in content
    assert "POLICY_3_TENSOR_PARALLEL_SIZE" in content
    assert "policies.policy_1.ppo_trainer_overrides.trainer.nnodes=${POLICY_1_NNODES}" in content
    assert "policies.policy_2.ppo_trainer_overrides.trainer.nnodes=${POLICY_2_NNODES}" in content
    assert "policies.policy_1.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_1_N_GPUS_PER_NODE}" in content
    assert "policies.policy_2.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_2_N_GPUS_PER_NODE}" in content
    assert (
        "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_1_TENSOR_PARALLEL_SIZE}"
        in content
    )
    assert (
        "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_2_TENSOR_PARALLEL_SIZE}"
        in content
    )


def test_readme_points_to_training_entrypoint():
    readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")

    assert "TaskRunner `apply_patch()`" in readme
    assert "Ray worker `apply_worker_patch()`" in readme
    assert "worker_process_setup_hook" in readme
    assert "shared checkout" in readme
    assert "gateway_count` (currently 8" in readme
    assert "fresh training Driver" in readme
    assert "policy_1_reward_loop_worker_0" in readme
    assert "policy_1_vllm_" in readme
    assert "vllm_policy_1_" not in readme
    assert "pytest -q tests/test_multi_agent_blackbox_example.py" in readme
    assert "bash examples/multi_agent_blackbox/scripts/run_e2e_train.sh" in readme
    assert "bash examples/multi_agent_blackbox/scripts/run_external_mas_train.sh" in readme
    assert "python -m uni_agent.trainer.main_multi_agents_ppo" in readme
    assert "POLICY_1_MODEL_PATH" in readme
    assert "POLICY_2_MODEL_PATH" in readme
