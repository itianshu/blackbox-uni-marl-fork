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
        "agent_2": "policy_1",
        "agent_3": "policy_2",
    }
    assert cfg["actor_rollout_ref"]["rollout"]["n"] == 4
    assert "reward_function_fqn" not in af_cfg["multi_agent_runner_kwargs"]
    assert cfg["reward"]["custom_reward_function"] == {
        "path": "pkg://examples/multi_agent_blackbox.reward",
        "name": "compute_score",
    }
    assert "models" not in cfg
    assert set(cfg["policies"]) == {"policy_1", "policy_2"}
    assert "name" not in cfg["policies"]["policy_1"]
    assert "name" not in cfg["policies"]["policy_2"]
    assert cfg["policies"]["policy_1"]["ppo_trainer_config_name"] == "ppo_trainer"
    assert cfg["policies"]["policy_2"]["ppo_trainer_config_name"] == "ppo_trainer"


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
    assert "for var in POLICY_1_MODEL_PATH POLICY_2_MODEL_PATH; do" in content
    assert "--config-name=multi_agent_blackbox_external" in content
    assert "--config-path=\"${REPO_ROOT}/examples/multi_agent_blackbox/config\"" in content
    assert 'data.train_files="[\'${TRAIN_DATA}\']"' in content
    assert 'data.val_files="[\'${VAL_DATA}\']"' in content
    assert "actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_kwargs.config.template_path=${MAS_TEMPLATE_PATH}" in content


def test_multi_agent_blackbox_yaml_uses_public_ppo_trainer_base_per_policy():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    assert "hydra" not in cfg
    assert "defaults" not in cfg
    assert cfg["ppo_trainer_config_source"] == {
        "kind": "config_module",
        "value": "verl.trainer.config",
    }
    assert "policy_ppo_trainer_base" not in cfg

    assert cfg["policies"]["policy_1"]["ppo_trainer_config_name"] == "ppo_trainer"
    assert cfg["policies"]["policy_2"]["ppo_trainer_config_name"] == "ppo_trainer"
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["data"]["train_batch_size"] == "${data.train_batch_size}"
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 4
    assert cfg["policies"]["policy_1"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6

    assert cfg["policies"]["policy_2"]["ppo_trainer_overrides"]["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.6


def test_multi_agent_blackbox_yaml_propagates_top_level_data_and_algorithm_to_policies():
    cfg = yaml.safe_load((EXAMPLE_DIR / "config" / "multi_agent_blackbox.yaml").read_text(encoding="utf-8"))

    for policy_name in ("policy_1", "policy_2"):
        ppo_config = cfg["policies"][policy_name]["ppo_trainer_overrides"]

        assert ppo_config["data"]["train_files"] == "${data.train_files}"
        assert ppo_config["data"]["val_files"] == "${data.val_files}"
        assert ppo_config["data"]["train_batch_size"] == "${data.train_batch_size}"
        assert ppo_config["data"]["val_batch_size"] == "${data.val_batch_size}"
        assert ppo_config["data"]["return_raw_chat"] == "${data.return_raw_chat}"

        assert ppo_config["algorithm"]["adv_estimator"] == "${algorithm.adv_estimator}"
        assert ppo_config["algorithm"]["gamma"] == "${algorithm.gamma}"
        assert ppo_config["algorithm"]["lam"] == "${algorithm.lam}"

        assert ppo_config["reward"]["custom_reward_function"] == {
            "path": "${reward.custom_reward_function.path}",
            "name": "${reward.custom_reward_function.name}",
        }

        assert ppo_config["actor_rollout_ref"]["rollout"]["n"] == "${actor_rollout_ref.rollout.n}"
        # Temperature is owned per policy (no top-level reference): each policy
        # holds its own value, which drives both sampling and recomputation.
        assert ppo_config["actor_rollout_ref"]["rollout"]["temperature"] == 1.0
        assert ppo_config["actor_rollout_ref"]["rollout"]["top_p"] == "${actor_rollout_ref.rollout.top_p}"


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

    assert policy_1["trainer"]["nnodes"] == 2
    assert policy_2["trainer"]["nnodes"] == 2
    assert policy_1["trainer"]["n_gpus_per_node"] == 8
    assert policy_2["trainer"]["n_gpus_per_node"] == 8
    assert "n_training_gpus_per_node" not in policy_1["trainer"]
    assert "n_training_gpus_per_node" not in policy_2["trainer"]
    assert policy_1["trainer"]["default_local_dir"].endswith("/policy_1")
    assert policy_2["trainer"]["default_local_dir"].endswith("/policy_2")
    assert policy_1["actor_rollout_ref"]["model"]["path"] == "${oc.env:POLICY_1_MODEL_PATH,???}"
    assert policy_2["actor_rollout_ref"]["model"]["path"] == "${oc.env:POLICY_2_MODEL_PATH,???}"
    assert policy_1["actor_rollout_ref"]["actor"]["fsdp_config"]["fsdp_size"] == 8
    assert policy_2["actor_rollout_ref"]["actor"]["fsdp_config"]["fsdp_size"] == 8
    assert policy_1["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 4
    assert policy_2["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 8
    assert policy_1["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] == "qwen3_coder"
    assert policy_2["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] == "hermes"
    assert "enable" not in policy_1["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert "enable" not in policy_2["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert "served_model_name" not in policy_1["actor_rollout_ref"]["rollout"]
    assert "served_model_name" not in policy_2["actor_rollout_ref"]["rollout"]
    assert "prometheus" not in policy_1["actor_rollout_ref"]["rollout"]
    assert "prometheus" not in policy_2["actor_rollout_ref"]["rollout"]


def test_multi_agent_blackbox_hydra_config_initializes_multi_policy_trainer(monkeypatch):
    from hydra import compose, initialize_config_dir

    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(kv_batch_put=lambda **kwargs: None))
    sys.modules.pop("uni_agent.trainer.multi_agents_ppo_trainer", None)
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    class ConfigSmokePolicyTrainer:
        instances = []

        def __init__(self, config):
            self.config = config
            self.__class__.instances.append(self)

    monkeypatch.setenv("POLICY_1_MODEL_PATH", "/models/policy_1")
    monkeypatch.setenv("POLICY_2_MODEL_PATH", "/models/policy_2")
    for name in ["verl", "verl.trainer", "verl.trainer.ppo"]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    v1_module = types.ModuleType("verl.trainer.ppo.v1")
    v1_module.get_trainer_cls = lambda trainer_mode: ConfigSmokePolicyTrainer
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1", v1_module)
    sys.modules["verl.trainer"].ppo = sys.modules["verl.trainer.ppo"]
    sys.modules["verl.trainer.ppo"].v1 = v1_module

    config_dir = str((EXAMPLE_DIR / "config").resolve())
    verl_config_dir = (Path(__file__).resolve().parents[2] / "verl" / "verl" / "trainer" / "config").as_posix()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="multi_agent_blackbox",
            overrides=[
                "ppo_trainer_config_source.kind=config_dir",
                f"ppo_trainer_config_source.value={verl_config_dir}",
                "data.train_files=[train.parquet]",
                "data.val_files=[val.parquet]",
                "trainer.total_training_steps=1",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.rollout.top_p=0.95",
                "policies.policy_1.ppo_trainer_overrides.trainer.nnodes=2",
                "policies.policy_2.ppo_trainer_overrides.trainer.nnodes=3",
                "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.temperature=0.5",
                "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.temperature=0.8",
                ],
            )

    trainer = MultiAgentsPPOTrainer(config=cfg)

    assert list(trainer.policy_trainers) == ["policy_1", "policy_2"]
    assert len(ConfigSmokePolicyTrainer.instances) == 2
    assert trainer.policy_configs["policy_1"].data.train_files == ["train.parquet"]
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.actor.optim.lr == 1e-6
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.model.path == "/models/policy_1"
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.model.path == "/models/policy_2"
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.rollout.n == 8
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.n == 8
    assert trainer.policy_configs["policy_1"].actor_rollout_ref.rollout.temperature == 0.5
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.temperature == 0.8
    assert trainer.policy_configs["policy_2"].actor_rollout_ref.rollout.top_p == 0.95
    assert trainer.policy_configs["policy_1"].trainer.nnodes == 2
    assert trainer.policy_configs["policy_2"].trainer.nnodes == 3
    assert cfg.actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping == {
        "agent_1": "policy_1",
        "agent_2": "policy_1",
        "agent_3": "policy_2",
    }


def test_run_train_script_uses_multi_agents_entrypoint_and_config():
    script = EXAMPLE_DIR / "scripts" / "run_train.sh"
    content = script.read_text(encoding="utf-8")

    assert "python3 -m uni_agent.trainer.main_multi_agents_ppo" in content
    assert "--config-name=multi_agent_blackbox" in content
    assert "--config-path=\"$(pwd)/examples/multi_agent_blackbox/config\"" in content
    assert "actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_kwargs.mas_config_path" in content
    assert "actor_rollout_ref.rollout.n=${ROLLOUT_N}" in content
    assert "POLICY_1_GPUS" in content
    assert "POLICY_2_GPUS" in content
    assert "POLICY_1_NNODES" in content
    assert "POLICY_2_NNODES" in content
    assert "POLICY_1_TP" in content
    assert "POLICY_2_TP" in content
    assert 'POLICY_1_NNODES="${POLICY_1_NNODES:-2}"' in content
    assert 'POLICY_2_NNODES="${POLICY_2_NNODES:-2}"' in content
    assert 'POLICY_1_GPUS="${POLICY_1_GPUS:-8}"' in content
    assert 'POLICY_2_GPUS="${POLICY_2_GPUS:-8}"' in content
    assert 'POLICY_1_TP="${POLICY_1_TP:-4}"' in content
    assert 'POLICY_2_TP="${POLICY_2_TP:-8}"' in content
    assert "policies.policy_1.ppo_trainer_overrides.trainer.nnodes=${POLICY_1_NNODES}" in content
    assert "policies.policy_2.ppo_trainer_overrides.trainer.nnodes=${POLICY_2_NNODES}" in content
    assert "policies.policy_1.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_1_GPUS}" in content
    assert "policies.policy_2.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_2_GPUS}" in content
    assert (
        "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_1_TP}"
        in content
    )
    assert (
        "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_2_TP}"
        in content
    )


def test_verify_vllm_servers_checks_worker_patch_before_trainer_init():
    script = EXAMPLE_DIR / "scripts" / "verify_vllm_servers.py"
    content = script.read_text(encoding="utf-8")
    main_body = content.split("def main() -> None:", maxsplit=1)[1]

    assert "examples.multi_agent_blackbox.verl_patch.apply_worker_patch" in content
    assert "@ray.remote" in content
    assert "probe_worker_lookup_prefix.remote()" in content
    assert 'probe["prefix"] != "policy_1_vllm_"' in content
    assert main_body.index("_verify_worker_lookup_patch()") < main_body.index(
        "trainer = MultiAgentsPPOTrainer(config=config)"
    )


def test_readme_points_to_training_entrypoint():
    readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")

    assert "TaskRunner `apply_patch()`" in readme
    assert "Ray worker `apply_worker_patch()`" in readme
    assert "worker_process_setup_hook" in readme
    assert "shared checkout" in readme
    assert "gateway_count: 4" in readme
    assert "fresh training Driver" in readme
    assert "policy_1_reward_loop_worker_0" in readme
    assert "policy_1_vllm_" in readme
    assert "vllm_policy_1_" not in readme
    assert "bash examples/multi_agent_blackbox/scripts/run_verify_vllm_servers.sh" in readme
    assert "bash examples/multi_agent_blackbox/scripts/run_train.sh" in readme
    assert "python -m uni_agent.trainer.main_multi_agents_ppo" in readme
    assert "POLICY_1_MODEL_PATH" in readme
    assert "POLICY_2_MODEL_PATH" in readme
