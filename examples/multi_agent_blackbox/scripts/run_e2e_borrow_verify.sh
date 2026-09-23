#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Optional reproducible high-KV workload; keep the previous recipe available.
if [[ "${BORROW_LOAD_PROFILE:-standard}" == "high_kv" ]]; then
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export PROMPT_LENGTH="${PROMPT_LENGTH:-12288}"
    export RESPONSE_LENGTH="${RESPONSE_LENGTH:-10240}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
    export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-multi_agent_blackbox_borrow_high_kv}"
    export MAS_CONFIG_PATH="${MAS_CONFIG_PATH:-${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_borrow_high_kv.yaml}"
elif [[ "${BORROW_LOAD_PROFILE:-standard}" == "random_routing" ]]; then
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export PROMPT_LENGTH="${PROMPT_LENGTH:-12288}"
    export RESPONSE_LENGTH="${RESPONSE_LENGTH:-10240}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
    export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-multi_agent_blackbox_borrow_random_routing}"
    export MAS_CONFIG_PATH="${MAS_CONFIG_PATH:-${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_random_routing.yaml}"
fi

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-9216}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-13824}"
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    RAY_CLUSTER_FILE="${RAY_CLUSTER_FILE:-/tmp/ray_zzh_255/ray_current_cluster}"
    if [[ ! -s "${RAY_CLUSTER_FILE}" ]]; then
        echo "ERROR: start the cluster with ${SCRIPT_DIR}/start_ray_cluster.sh or set RAY_ADDRESS" >&2
        exit 1
    fi
    export RAY_ADDRESS="$(cat "${RAY_CLUSTER_FILE}")"
fi
# Use the original zzh environment and its installed verl.
export PYTHON="${PYTHON:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3}"
# Prevent the system PYTHONPATH from shadowing zzh_env's installed verl.
export PYTHONPATH="${REPO_ROOT}"
# Give the load test 1024 indexed rows instead of cycling the eight-row mock set.
if [[ -z "${TRAIN_DATA:-}" && -z "${MOCK_DATA_DIR:-}" ]]; then
    export MOCK_DATA_DIR="${SCRIPT_DIR}/mock_data_borrow"
    "${PYTHON}" "${SCRIPT_DIR}/generate_mock_mas_data.py" --repeat 128 --output-dir "${MOCK_DATA_DIR}"
fi
# 12 training GPUs + 20 standalone rollout GPUs; all replicas use TP=2.
export POLICY_1_N_GPUS_PER_NODE="${POLICY_1_N_GPUS_PER_NODE:-4}"
export POLICY_1_ROLLOUT_N_GPUS_PER_NODE="${POLICY_1_ROLLOUT_N_GPUS_PER_NODE:-4}"
export POLICY_1_TENSOR_PARALLEL_SIZE=2
export POLICY_2_N_GPUS_PER_NODE="${POLICY_2_N_GPUS_PER_NODE:-4}"
export POLICY_2_ROLLOUT_N_GPUS_PER_NODE="${POLICY_2_ROLLOUT_N_GPUS_PER_NODE:-12}"
export POLICY_2_TENSOR_PARALLEL_SIZE=2
export DYNAMIC_INFERENCE_SCHEDULING="${DYNAMIC_INFERENCE_SCHEDULING:-true}"
export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-multi_agent_blackbox_borrow_verify}"
export MAS_CONFIG_PATH="${MAS_CONFIG_PATH:-${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_borrow_verify.yaml}"

echo "Borrow verification: train GPUs=4/4/4; rollout GPUs=4/12/4; TP2 replicas=2/6/2; all policies can lend to each other"
exec bash "${SCRIPT_DIR}/run_e2e_train.sh" "$@"
