#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# The container injects /opt/tiger/bi_verl ahead of site-packages, but that
# checkout is newer than the v1 trainer API used by this repository.  Pin the
# same verl tree as the established experiments.
VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-/mnt/bn/chenghao1026/zzh/zzh_0813/verl}"
export PYTHONPATH="${REPO_ROOT}:${VERL_SOURCE_ROOT}"
export PYTHON="${PYTHON:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3}"

# The default fixture has only eight rows. Generate enough distinct indices for
# warmup plus eight 128-prompt steps without depending on dataloader cycling.
if [[ -z "${TRAIN_DATA:-}" && -z "${MOCK_DATA_DIR:-}" ]]; then
  export MOCK_DATA_DIR="${SCRIPT_DIR}/mock_data_2policy_2b"
  "${PYTHON}" "${SCRIPT_DIR}/generate_mock_mas_data.py" \
    --repeat 256 --output-dir "${MOCK_DATA_DIR}"
fi

export TRAIN_CONFIG_NAME="multi_agent_blackbox_2policy_2b_comparison"
export MAS_CONFIG_PATH="${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_2agent_variable.yaml"

export POLICY_1_MODEL_PATH="${POLICY_1_MODEL_PATH:-/mnt/bn/chenghao1026/models/Qwen3-1.7B}"
export POLICY_2_MODEL_PATH="${POLICY_2_MODEL_PATH:-/mnt/bn/chenghao1026/models/Qwen3-1.7B}"
export POLICY_1_N_GPUS_PER_NODE=8
export POLICY_2_N_GPUS_PER_NODE=4
export POLICY_1_FSDP_SIZE=8
export POLICY_2_FSDP_SIZE=4
export POLICY_1_ROLLOUT_NNODES=3
export POLICY_2_ROLLOUT_NNODES=2
export POLICY_1_ROLLOUT_N_GPUS_PER_NODE=4
export POLICY_2_ROLLOUT_N_GPUS_PER_NODE=4
export POLICY_1_TENSOR_PARALLEL_SIZE=1
export POLICY_2_TENSOR_PARALLEL_SIZE=1

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-12288}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-10240}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Two-policy 2B comparison: train=8/4 FSDP8/FSDP4+SP2; rollout=12/8 TP1 (20 replicas total)"

exec bash "${SCRIPT_DIR}/run_e2e_train.sh" \
  '~policies.policy_3' \
  '~actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping.agent_3' \
  policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.data_parallel_size=1 \
  policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.data_parallel_size=1 \
  "$@"
