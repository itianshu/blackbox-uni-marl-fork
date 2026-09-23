#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep two TP2 home replicas per policy and one exercised guest unit.
export POLICY_2_ROLLOUT_N_GPUS_PER_NODE="${POLICY_2_ROLLOUT_N_GPUS_PER_NODE:-4}"
export MOCK_DATA_DIR="${MOCK_DATA_DIR:-${SCRIPT_DIR}/mock_data_clone_smoke}"
"${PYTHON:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3}" \
  "${SCRIPT_DIR}/generate_mock_mas_data.py" --repeat 4 --output-dir "${MOCK_DATA_DIR}"
exec bash "${SCRIPT_DIR}/run_transaction_smoke.sh" \
  example_patch_fqn=examples.multi_agent_blackbox.vllm_clone_smoke_patch \
  dynamic_inference_scheduling.borrow_drain_timeout_s=180 \
  dynamic_inference_scheduling.weight_sync_timeout_s=600 "$@"
