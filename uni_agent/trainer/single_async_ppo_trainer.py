"""Minimal separate-async single-policy v1 PPO trainer for multi-agent orchestration.

Mirrors SinglePPOTrainer (sync): the outer MultiAgentsPPOTrainer owns the shared
dataloader and the shared ReplayBufferAsync. Inheriting PPOTrainerSeparateAsync
gives us:

- native trainer/FSDP setup plus standalone vLLM runtime components;
- get_llm_client() -> FullyAsyncLLMServerClient (partial-rollout resume);
- on_step_end(): standalone_checkpoint_manager.update_weights (abort/resume built-in);
- on_sample_end(): hybrid switch-to-trainer semantics.
"""

from __future__ import annotations

from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.llm_server import LLMServerManager


class SingleAsyncPPOTrainer(PPOTrainerSeparateAsync):
    """Separate-async v1 PPO trainer driven by the outer multi-agent trainer."""

    def init_training_runtime(self) -> None:
        """Initialize the primary trainer-owned runtime, excluding standalone rollout.

        Calling the v1 base implementation directly establishes the policy's
        trainer placement groups, distributed model workers, and hybrid
        rollout replicas without entering ``PPOTrainerSeparateAsync._setup()``,
        which would immediately allocate standalone rollout GPUs.
        """
        PPOTrainer._setup(self)

    def init_standalone_rollout_runtime(self) -> None:
        """Initialize the native v1 standalone rollout runtime."""
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        self.standalone_server_manager = LLMServerManager.create(
            config=self.config,
            start_rank=hybrid_num_replicas,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        if self._enable_hybrid_replicas:
            self.current_mode = HybridEngineMode.ROLLOUT
            self.add_replicas_to_balancer()
        else:
            self.current_mode = HybridEngineMode.TRAINER

    def _build_replay_buffer(self):
        """The outer trainer owns the shared async replay buffer."""
        return None

    def _init_dataloader(self):
        """The outer trainer owns the shared prompt stream."""
        from uni_agent.trainer.policy_lifecycle import initialize_policy_training_steps

        initialize_policy_training_steps(self)

    def get_llm_client(self):
        """Build the Uni-Agent client that fixes min_tokens on abort resume."""
        from uni_agent.trainer.resumable_llm_client import ResumableFullyAsyncLLMServerClient

        return self.standalone_server_manager.get_client(
            client_cls=ResumableFullyAsyncLLMServerClient
        )

    def on_sample_end(self) -> None:
        """Release hybrid rollout memory once the outer trainer has a batch.

        The outer trainer owns sampling and never runs the standalone trainer's
        per-step sampling timers. Its sample-end hook must perform the hybrid
        transition directly rather than enter verl's private timer lifecycle.
        """
        if self.current_mode == HybridEngineMode.ROLLOUT:
            self.switch_to_trainer()

    def on_step_end(self) -> None:
        """Publish weights without entering verl's standalone fit lifecycle.

        Multi-agent scheduling owns step boundaries and hybrid transitions.
        The native hook additionally reads profiling and adaptive-switch state
        initialized only by the standalone trainer's fit/step hooks.
        """
        from verl.utils.debug import marked_timer

        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = dict(
                self.standalone_checkpoint_manager.update_weights(self.global_steps) or {}
            )

    def fit(self, agent_loop_manager):
        raise RuntimeError(
            "SingleAsyncPPOTrainer cannot run its own fit(): it has no dataloader or "
            "replay buffer. Use MultiAgentsPPOTrainer for multi-agent orchestration."
        )

    def step(self, metrics, timing_raw):
        raise RuntimeError(
            "SingleAsyncPPOTrainer cannot run its own step(): it has no dataloader or "
            "replay buffer. Use MultiAgentsPPOTrainer for multi-agent orchestration."
        )
