#!/usr/bin/env bash
# Deterministic real-GPU borrow/sync/return/sync correctness run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export BORROW_LOAD_PROFILE=random_routing
export TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2
export PROMPT_LENGTH=4096 RESPONSE_LENGTH=1024 MAX_MODEL_LEN=6144
export MAS_CONFIG_PATH="${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_transaction_smoke.yaml"
exec bash "${SCRIPT_DIR}/run_e2e_borrow_verify.sh" \
  example_patch_fqn=examples.multi_agent_blackbox.transaction_smoke_patch \
  'dynamic_inference_scheduling.borrowing.pairs=[{home:policy_2,donor:policy_1}]' \
  actor_rollout_ref.rollout.custom.agent_framework.max_concurrent_rollouts=32 \
  dynamic_inference_scheduling.resource_usage.kv_enter=0.01 \
  dynamic_inference_scheduling.resource_usage.kv_exit=0.003 \
  dynamic_inference_scheduling.resource_usage.kv_post_lend_max=0.007 \
  dynamic_inference_scheduling.min_lend_polls=10 \
  dynamic_inference_scheduling.borrow_drain_timeout_s=60 \
  dynamic_inference_scheduling.bottleneck_confirm_polls=2 \
  dynamic_inference_scheduling.rebalance_confirm_polls=2 \
  trainer.test_freq=-1 "$@"
