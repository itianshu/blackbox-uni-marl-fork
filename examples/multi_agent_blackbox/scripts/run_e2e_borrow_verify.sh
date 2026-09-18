#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export DYNAMIC_INFERENCE_SCHEDULING=true
export TRAIN_CONFIG_NAME=multi_agent_blackbox_borrow_verify
export MAS_CONFIG_PATH="${REPO_ROOT}/examples/multi_agent_blackbox/config/mas_config_borrow_verify.yaml"

exec bash "${SCRIPT_DIR}/run_e2e_train.sh"
