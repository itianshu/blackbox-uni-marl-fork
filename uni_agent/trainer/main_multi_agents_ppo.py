from __future__ import annotations

import logging
import importlib
import os
import sys
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _apply_example_patch(config) -> None:
    """Apply an example-owned runtime patch before trainer construction.

    verl is treated as a shared/public package, so example-specific fixes live
    in the example directory and are loaded here through the optional
    ``config.example_patch_fqn`` key (e.g.
    ``examples.multi_agent_blackbox.verl_patch``, which makes each policy's Ray
    placement-group name prefix unique).
    """
    patch_fqn = OmegaConf.select(config, "example_patch_fqn", default=None)
    if not patch_fqn:
        return
    module = importlib.import_module(str(patch_fqn))
    apply_patch = getattr(module, "apply_patch", None)
    if callable(apply_patch):
        apply_patch()
        logger.info("applied example patch from %s", patch_fqn)


def run_multi_agents_ppo(config, task_runner_class) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        # runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # Hydra composes configs as OmegaConf structs, which reject adding keys
        # (e.g. env_vars) that the YAML does not declare. Convert to a plain
        # mutable container before injecting TRANSFER_QUEUE_ENABLE / PYTHONPATH.
        runtime_env_kwargs = OmegaConf.to_container(
            ray_init_kwargs.get("runtime_env", {}), resolve=True
        ) or {}

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # Force Ray workers to import THIS checkout's example patch
        # (worker_process_setup_hook). Workers inherit a cluster/node
        # PYTHONPATH that may point at another uni-marl checkout whose
        # verl_patch lacks apply_worker_patch, which would break the hook.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        runtime_env_vars = runtime_env_kwargs.setdefault("env_vars", {})
        existing_pythonpath = runtime_env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        runtime_env_vars["PYTHONPATH"] = repo_root + (":" + existing_pythonpath if existing_pythonpath else "")

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info("ray init kwargs: %s", ray_init_kwargs)
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = (getattr(config, "ray_kwargs", {}) or {}).get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class MultiAgentsTaskRunner:
    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        """Initialize the agent loop manager to generate multi-agent rollouts."""
        if self.config is None or self.trainer is None:
            raise RuntimeError("trainer and config must be initialized before agent_loop_manager")

        from uni_agent.trainer.framework.entry import AgentFrameworkRolloutAdapter

        self.agent_loop_manager = AgentFrameworkRolloutAdapter.create(
            config=self.config,
            llm_client=self.trainer.get_multi_policy_llm_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
            gateway_actor_kwargs=self.trainer.get_gateway_actor_kwargs(),
        )
        return self.agent_loop_manager

    def run(self, config: DictConfig):
        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        try:
            _apply_example_patch(config)
            self.trainer = MultiAgentsPPOTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            # ``sys.exception`` is Python 3.11+, while this project supports
            # Python 3.10.  Capture the active exception without replacing it
            # if either cleanup path also fails.
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            try:
                # Release placement groups and gateway actors so a shared Ray
                # cluster can be reused immediately by the next job.
                if self.trainer is not None:
                    self.trainer.cleanup()
            except Exception as exc:
                cleanup_error = exc
                if primary_error is not None:
                    logger.exception("Trainer cleanup failed after the primary experiment error")

            try:
                tq.close()
            except Exception:
                if primary_error is None and cleanup_error is None:
                    raise
                logger.exception("Transfer queue cleanup failed")

            if primary_error is None and cleanup_error is not None:
                raise cleanup_error


@hydra.main(config_path="config", config_name="multi_agent_blackbox", version_base=None)
def main(config):
    auto_set_device(config)
    run_multi_agents_ppo(config, task_runner_class=MultiAgentsTaskRunner)


if __name__ == "__main__":
    main()
