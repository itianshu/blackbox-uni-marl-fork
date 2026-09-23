from __future__ import annotations

import importlib
import sys
import types
import pytest
from contextlib import nullcontext
from pathlib import Path


class ConfigNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeRay:
    def __init__(self):
        self.initialized = False
        self.init_calls = []
        self.get_calls = []
        self.timeline_calls = []

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.initialized = True

    def get(self, object_ref):
        self.get_calls.append(object_ref)
        return object_ref

    def timeline(self, filename=None):
        self.timeline_calls.append(filename)

    def remote(self, cls):
        return cls


class FakeOmegaConf:
    resolved = []

    @staticmethod
    def merge(*configs):
        merged = ConfigNode()
        for config in configs:
            if not config:
                continue
            for key, value in dict(config).items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = FakeOmegaConf.merge(merged[key], value)
                else:
                    merged[key] = value
        return merged

    @staticmethod
    def create(config):
        if isinstance(config, ConfigNode):
            return config
        if isinstance(config, dict):
            return ConfigNode(config)
        return config

    @staticmethod
    def to_container(config, resolve=False):
        if isinstance(config, dict):
            return {key: FakeOmegaConf.to_container(value, resolve=resolve) for key, value in config.items()}
        return config

    @staticmethod
    def select(config, key, default=None):
        node = config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @classmethod
    def resolve(cls, config):
        cls.resolved.append(config)


def _install_entry_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "uni_agent.trainer.multi_agents_ppo_trainer",
                        types.SimpleNamespace(MultiAgentsPPOTrainer=object))
    fake_ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        types.SimpleNamespace(
            init=lambda config: None,
            close=lambda: None,
            kv_batch_put=lambda **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hydra",
        types.SimpleNamespace(main=lambda **_: (lambda func: func)),
    )
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        types.SimpleNamespace(DictConfig=ConfigNode, OmegaConf=FakeOmegaConf, open_dict=lambda config: nullcontext(config)),
    )
    for name in ["verl", "verl.trainer", "verl.utils"]:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "verl.trainer.constants_ppo",
        types.SimpleNamespace(
            get_ppo_ray_runtime_env=lambda config: ConfigNode(env_vars=ConfigNode(BASE_ENV="1"))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.device",
        types.SimpleNamespace(auto_set_device=lambda config: setattr(config.trainer, "device", "cuda")),
    )
    sys.modules.pop("uni_agent.trainer.main_multi_agents_ppo", None)
    module = importlib.import_module("uni_agent.trainer.main_multi_agents_ppo")
    return module, fake_ray


def test_run_multi_agents_ppo_initializes_ray_and_runs_remote_task(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    module, fake_ray = _install_entry_stubs(monkeypatch)
    assert not hasattr(module, "_enable_transfer_queue")
    assert not hasattr(module, "_merge_ray_runtime_env")

    class FakeRemoteRun:
        calls = []

        @classmethod
        def remote(cls, config):
            cls.calls.append(config)
            return "run-ref"

    class FakeTaskRunner:
        remote_calls = 0
        run = FakeRemoteRun

        @classmethod
        def remote(cls):
            cls.remote_calls += 1
            return cls

    config = ConfigNode(
        transfer_queue=ConfigNode(enable=True),
        ray_kwargs=ConfigNode(
            ray_init=ConfigNode(
                num_cpus=8,
                runtime_env=ConfigNode(
                    env_vars=ConfigNode(EXISTING_ENV="1"),
                    worker_process_setup_hook=(
                        "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
                    ),
                ),
            ),
            timeline_json_file="timeline.json",
        ),
        trainer=ConfigNode(device="cuda"),
    )

    module.run_multi_agents_ppo(config, task_runner_class=FakeTaskRunner)

    assert config.transfer_queue.enable is True
    assert FakeTaskRunner.remote_calls == 1
    assert FakeRemoteRun.calls == [config]
    assert fake_ray.get_calls == ["run-ref"]
    assert fake_ray.timeline_calls == ["timeline.json"]
    assert fake_ray.init_calls == [
        {
            "num_cpus": 8,
            "runtime_env": {
                "env_vars": {
                    "BASE_ENV": "1",
                    "EXISTING_ENV": "1",
                    "TRANSFER_QUEUE_ENABLE": "1",
                    "PYTHONPATH": str(Path(module.__file__).resolve().parents[2]),
                },
                "worker_process_setup_hook": (
                    "examples.multi_agent_blackbox.verl_patch.apply_worker_patch"
                ),
            },
        }
    ]


def test_apply_example_patch_loads_and_applies_patch(monkeypatch):
    module, _ = _install_entry_stubs(monkeypatch)
    applied = []
    fake_patch_module = types.SimpleNamespace(apply_patch=lambda: applied.append("applied"))
    monkeypatch.setitem(sys.modules, "examples.multi_agent_blackbox.verl_patch", fake_patch_module)

    config = ConfigNode(example_patch_fqn="examples.multi_agent_blackbox.verl_patch")
    module._apply_example_patch(config)
    assert applied == ["applied"]

    module._apply_example_patch(ConfigNode())
    assert applied == ["applied"]


def test_task_runner_builds_agent_loop_manager_and_calls_fit(monkeypatch):
    module, _ = _install_entry_stubs(monkeypatch)
    assert not hasattr(module, "_enable_transfer_queue")
    FakeOmegaConf.resolved.clear()
    events = []

    fake_tq = types.SimpleNamespace(
        init=lambda config: events.append(("tq.init", config)),
        close=lambda: events.append(("tq.close", None)),
    )
    monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)

    class FakeAgentFrameworkRolloutAdapter:
        create_calls = []

        @classmethod
        def create(cls, **kwargs):
            cls.create_calls.append(kwargs)
            events.append(("agent_loop_manager.create", kwargs))
            return "agent-loop-manager"

    monkeypatch.setitem(
        sys.modules,
        "uni_agent.trainer.framework.entry",
        types.SimpleNamespace(AgentFrameworkRolloutAdapter=FakeAgentFrameworkRolloutAdapter),
    )

    class FakeTrainer:
        instances = []

        def __init__(self, *, config):
            self.config = config
            self.init_calls = 0
            self.fit_calls = 0
            self.fit_agent_loop_managers = []
            self.__class__.instances.append(self)
            events.append(("trainer.__init__", config))

        def init(self):
            self.init_calls += 1
            events.append(("trainer.init", None))

        def get_multi_policy_llm_client(self):
            return "multi-policy-llm-client"

        def get_replay_buffer(self):
            return "replay-buffer"

        def get_reward_handles(self):
            return ["reward-handle"]

        def get_gateway_actor_kwargs(self):
            return {"tokenizer": "tokenizer"}

        def cleanup(self):
            pass

        def fit(self, agent_loop_manager):
            self.fit_calls += 1
            self.fit_agent_loop_managers.append(agent_loop_manager)
            events.append(("trainer.fit", agent_loop_manager))

    monkeypatch.setattr(module, "MultiAgentsPPOTrainer", FakeTrainer)
    assert not hasattr(module.MultiAgentsTaskRunner, "get_multi_policy_llm_client")

    config = ConfigNode(
        transfer_queue=ConfigNode(enable=False),
        trainer=ConfigNode(device="cuda"),
    )

    runner = module.MultiAgentsTaskRunner()
    runner.run(config)

    assert config.transfer_queue.enable is True
    assert FakeOmegaConf.resolved == [config]
    assert len(FakeTrainer.instances) == 1
    assert FakeTrainer.instances[0].config is config
    assert FakeTrainer.instances[0].init_calls == 1
    assert FakeTrainer.instances[0].fit_calls == 1
    assert FakeTrainer.instances[0].fit_agent_loop_managers == ["agent-loop-manager"]
    assert runner.trainer is FakeTrainer.instances[0]
    assert runner.agent_loop_manager == "agent-loop-manager"
    assert events == [
        ("tq.init", config.transfer_queue),
        ("trainer.__init__", config),
        ("trainer.init", None),
        ("agent_loop_manager.create", FakeAgentFrameworkRolloutAdapter.create_calls[0]),
        ("trainer.fit", "agent-loop-manager"),
        ("tq.close", None),
    ]
    assert FakeAgentFrameworkRolloutAdapter.create_calls == [
        {
            "config": config,
            "llm_client": "multi-policy-llm-client",
            "reward_loop_worker_handles": ["reward-handle"],
            "gateway_actor_kwargs": {"tokenizer": "tokenizer"},
        }
    ]


@pytest.mark.parametrize("primary_fails,cleanup_fails,queue_fails,expected", [
    (True, True, True, "primary"),
    (True, False, True, "primary"),
    (False, True, True, "cleanup"),
    (False, False, True, "queue"),
])
def test_task_runner_preserves_first_error(monkeypatch, primary_fails, cleanup_fails, queue_fails, expected):
    module, _ = _install_entry_stubs(monkeypatch)
    def fail(message):
        raise RuntimeError(message)
    class Trainer:
        def __init__(self, **kwargs):
            pass
        def init(self):
            if primary_fails:
                fail("primary")
        def fit(self, manager):
            pass
        def cleanup(self):
            if cleanup_fails:
                fail("cleanup")
    monkeypatch.setattr(module, "MultiAgentsPPOTrainer", Trainer)
    monkeypatch.setattr(module.MultiAgentsTaskRunner, "init_agent_loop_manager", lambda self: None)
    monkeypatch.setitem(sys.modules, "transfer_queue", types.SimpleNamespace(
        init=lambda config: None, close=lambda: fail("queue") if queue_fails else None))
    with pytest.raises(RuntimeError, match=expected):
        module.MultiAgentsTaskRunner().run(ConfigNode(transfer_queue=ConfigNode(enable=False)))
