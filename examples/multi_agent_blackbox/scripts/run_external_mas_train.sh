#!/usr/bin/env bash
set -euo pipefail

# Command-driven external MAS training launcher.
# This intentionally mirrors run_e2e_train.sh; only the Hydra recipe and MAS
# template override differ.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RAY_ADDRESS="${RAY_ADDRESS:-10.122.121.62:20001}"
POLICY_1_MODEL_PATH="${POLICY_1_MODEL_PATH:-/mnt/bn/chenghao1026/models/Qwen2.5-0.5B-Instruct}"
POLICY_2_MODEL_PATH="${POLICY_2_MODEL_PATH:-/mnt/bn/chenghao1026/models/Qwen2.5-0.5B-Instruct}"

MOCK_DATA_DIR="${MOCK_DATA_DIR:-${REPO_ROOT}/examples/multi_agent_blackbox/scripts/mock_data}"
TRAIN_DATA="${TRAIN_DATA:-${MOCK_DATA_DIR}/mock_mas_train.parquet}"
VAL_DATA="${VAL_DATA:-${MOCK_DATA_DIR}/mock_mas_val.parquet}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((PROMPT_LENGTH + RESPONSE_LENGTH + 512))}"

POLICY_1_NNODES="${POLICY_1_NNODES:-1}"
POLICY_1_N_GPUS_PER_NODE="${POLICY_1_N_GPUS_PER_NODE:-2}"
POLICY_1_FSDP_SIZE="${POLICY_1_FSDP_SIZE:-${POLICY_1_N_GPUS_PER_NODE}}"
POLICY_1_ROLLOUT_NNODES="${POLICY_1_ROLLOUT_NNODES:-1}"
POLICY_1_ROLLOUT_N_GPUS_PER_NODE="${POLICY_1_ROLLOUT_N_GPUS_PER_NODE:-2}"
POLICY_1_TENSOR_PARALLEL_SIZE="${POLICY_1_TENSOR_PARALLEL_SIZE:-2}"

POLICY_2_NNODES="${POLICY_2_NNODES:-1}"
POLICY_2_N_GPUS_PER_NODE="${POLICY_2_N_GPUS_PER_NODE:-2}"
POLICY_2_FSDP_SIZE="${POLICY_2_FSDP_SIZE:-${POLICY_2_N_GPUS_PER_NODE}}"
POLICY_2_ROLLOUT_NNODES="${POLICY_2_ROLLOUT_NNODES:-1}"
POLICY_2_ROLLOUT_N_GPUS_PER_NODE="${POLICY_2_ROLLOUT_N_GPUS_PER_NODE:-2}"
POLICY_2_TENSOR_PARALLEL_SIZE="${POLICY_2_TENSOR_PARALLEL_SIZE:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/multi_agent_blackbox/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/external_mas_${TIMESTAMP}.log}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/multi_agent_blackbox/external_mas_${TIMESTAMP}}"
MAS_TEMPLATE_PATH="${MAS_TEMPLATE_PATH:-${REPO_ROOT}/examples/multi_agent_blackbox/config/external_mas_template.yaml}"

PYTHON="${PYTHON:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export PYTHON
export RAY_ADDRESS
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

for var in POLICY_1_MODEL_PATH POLICY_2_MODEL_PATH; do
    val="${!var}"
    if [[ -z "${val}" ]]; then
        echo "ERROR: ${var} is empty" >&2
        exit 1
    fi
    if [[ "${val}" == *'~'* ]]; then
        echo "ERROR: ${var} contains an unexpanded '~': '${val}'" >&2
        exit 1
    fi
    if [[ ! -d "${val}" ]]; then
        echo "ERROR: ${var} directory does not exist: '${val}'" >&2
        exit 1
    fi
done

echo "=== Multi-Agent Blackbox External MAS (v1 separate_async) ==="
echo "Ray address:      ${RAY_ADDRESS}"
echo "Nodes:            p1 train=${POLICY_1_NNODES}/rollout=${POLICY_1_ROLLOUT_NNODES}, p2 train=${POLICY_2_NNODES}/rollout=${POLICY_2_ROLLOUT_NNODES}"
echo "Train GPUs/node:  p1=${POLICY_1_N_GPUS_PER_NODE}, p2=${POLICY_2_N_GPUS_PER_NODE} (FSDP: p1=${POLICY_1_FSDP_SIZE}, p2=${POLICY_2_FSDP_SIZE}, TP: p1=${POLICY_1_TENSOR_PARALLEL_SIZE}, p2=${POLICY_2_TENSOR_PARALLEL_SIZE})"
echo "Rollout GPUs/node: p1=${POLICY_1_ROLLOUT_N_GPUS_PER_NODE}, p2=${POLICY_2_ROLLOUT_N_GPUS_PER_NODE} (standalone)"
echo "Steps:            ${TOTAL_TRAINING_STEPS}, batch=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, warmup=${NUM_WARMUP_BATCHES}"
echo "Sequence:         prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}, max_model_len=${MAX_MODEL_LEN}"
echo "Policy 1 model:   ${POLICY_1_MODEL_PATH}"
echo "Policy 2 model:   ${POLICY_2_MODEL_PATH}"
echo "Train data:       ${TRAIN_DATA}"
echo "Val data:         ${VAL_DATA}"
echo "MAS template:     ${MAS_TEMPLATE_PATH}"
echo "Log file:         ${LOG_PATH}"
echo "============================================"

"${PYTHON}" -u -m uni_agent.trainer.main_multi_agents_ppo \
    --config-name=multi_agent_blackbox_external \
    --config-path="${REPO_ROOT}/examples/multi_agent_blackbox/config" \
    \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${TRAIN_BATCH_SIZE} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.custom.agent_framework.multi_agent_runner_kwargs.config.template_path=${MAS_TEMPLATE_PATH} \
    \
    policies.policy_1.ppo_trainer_overrides.trainer.nnodes=${POLICY_1_NNODES} \
    policies.policy_1.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_1_N_GPUS_PER_NODE} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.model.path=${POLICY_1_MODEL_PATH} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.actor.fsdp_config.fsdp_size=${POLICY_1_FSDP_SIZE} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.nnodes=${POLICY_1_ROLLOUT_NNODES} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.n_gpus_per_node=${POLICY_1_ROLLOUT_N_GPUS_PER_NODE} \
    policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_1_TENSOR_PARALLEL_SIZE} \
    \
    policies.policy_2.ppo_trainer_overrides.trainer.nnodes=${POLICY_2_NNODES} \
    policies.policy_2.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_2_N_GPUS_PER_NODE} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.model.path=${POLICY_2_MODEL_PATH} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.actor.fsdp_config.fsdp_size=${POLICY_2_FSDP_SIZE} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.nnodes=${POLICY_2_ROLLOUT_NNODES} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.n_gpus_per_node=${POLICY_2_ROLLOUT_N_GPUS_PER_NODE} \
    policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_2_TENSOR_PARALLEL_SIZE} \
    2>&1 | tee "${LOG_PATH}"
