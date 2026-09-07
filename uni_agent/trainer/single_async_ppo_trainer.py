"""Minimal separate-async single-policy v1 PPO trainer for multi-agent orchestration.

Mirrors SinglePPOTrainer (sync): the outer MultiAgentsPPOTrainer owns the shared
dataloader and the shared ReplayBufferAsync. Inheriting PPOTrainerSeparateAsync
gives us:

- _setup(): standalone vLLM replicas + standalone_checkpoint_manager;
- get_llm_client() -> FullyAsyncLLMServerClient (partial-rollout resume);
- on_step_end(): standalone_checkpoint_manager.update_weights (abort/resume built-in);
- on_sample_end(): hybrid switch-to-trainer semantics.
"""

from __future__ import annotations

from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync


class SingleAsyncPPOTrainer(PPOTrainerSeparateAsync):
    def init_runtime(self) -> None:
        """Initialize the per-policy runtime without syncing rollout weights.

        ``MultiAgentsPPOTrainer`` owns the shared checkpoint lifecycle.  It
        therefore lets the outer trainer establish actor/critic state first;
        the inherited ``on_init_end()`` hook can then synchronize the current
        actor weights to standalone rollout replicas.
        """
        self._setup()

    def _build_replay_buffer(self):
        """The outer trainer owns the shared async replay buffer."""
        return None

    def _init_dataloader(self):
        """The outer trainer owns the shared prompt stream."""
        return None

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
