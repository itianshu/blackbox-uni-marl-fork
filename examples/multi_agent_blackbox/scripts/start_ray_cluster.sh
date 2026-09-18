#!/usr/bin/env bash
# Start an independent Ray 2.55.1 head cluster, isolated from the system
# bytedray (2.46.0.18) cluster that is already running on this host.
#
# The system cluster owns /opt/tiger/ray, ports 11052 (GCS) and 9248
# (dashboard). This script deliberately uses separate ports and a separate
# temp dir so the two clusters never interfere.
#
# GCS and dashboard ports must stay OUTSIDE the default worker ports range
# (10002-19999) that Ray 2.55.x reserves, hence 20001/20002 instead of
# 16379/18265.
#
# Verify afterwards:
#   "${RAY_BIN}" status --address="$(cat "${TEMP_DIR}/ray_current_cluster")"
# Do not run a bare `ray stop` on this host: it may stop the system bytedray
# cluster as well. Stop this independent cluster by terminating the processes
# whose command line/session path contains TEMP_DIR.

set -euo pipefail

RAY_BIN="${RAY_BIN:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/ray}"
# Outside Ray's default worker port range (10002-19999).
GCS_PORT="${GCS_PORT:-20001}"
# Outside Ray's default worker port range (10002-19999).
DASHBOARD_PORT="${DASHBOARD_PORT:-20002}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-20003}"
DASHBOARD_AGENT_GRPC_PORT="${DASHBOARD_AGENT_GRPC_PORT:-20004}"
DASHBOARD_AGENT_LISTEN_PORT="${DASHBOARD_AGENT_LISTEN_PORT:-20005}"
MIN_WORKER_PORT="${MIN_WORKER_PORT:-21000}"
MAX_WORKER_PORT="${MAX_WORKER_PORT:-29999}"
# End-exclusive range intentionally left out of Ray's worker pool. Training
# launchers may split it between policy and guest torch rendezvous groups.
TORCH_MASTER_PORT_START="${TORCH_MASTER_PORT_START:-18000}"
TORCH_MASTER_PORT_END="${TORCH_MASTER_PORT_END:-18128}"
NUM_CPUS="${NUM_CPUS:-99}"
NUM_GPUS="${NUM_GPUS:-8}"
BLOCK="${BLOCK:-false}"
TEMP_DIR="${TEMP_DIR:-/tmp/ray_zzh_255}"
# Leave empty to let Ray auto-detect the node IP (recommended for a single
# node); set explicitly only when you need to bind a specific interface.
NODE_IP="${NODE_IP:-}"

if (( TORCH_MASTER_PORT_START < 1024 \
      || TORCH_MASTER_PORT_START >= TORCH_MASTER_PORT_END \
      || TORCH_MASTER_PORT_END > 65536 )); then
    echo "ERROR: invalid end-exclusive torch master-port range [${TORCH_MASTER_PORT_START}, ${TORCH_MASTER_PORT_END})" >&2
    exit 1
fi
if (( MIN_WORKER_PORT < TORCH_MASTER_PORT_END \
      && MAX_WORKER_PORT >= TORCH_MASTER_PORT_START )); then
    echo "ERROR: Ray worker ports ${MIN_WORKER_PORT}-${MAX_WORKER_PORT} overlap torch master-port range [${TORCH_MASTER_PORT_START}, ${TORCH_MASTER_PORT_END})" >&2
    exit 1
fi
if [[ -r /proc/sys/net/ipv4/ip_local_port_range ]]; then
    read -r EPHEMERAL_PORT_START EPHEMERAL_PORT_END < /proc/sys/net/ipv4/ip_local_port_range
    if (( TORCH_MASTER_PORT_START <= EPHEMERAL_PORT_END \
          && TORCH_MASTER_PORT_END > EPHEMERAL_PORT_START )); then
        echo "ERROR: torch master-port range [${TORCH_MASTER_PORT_START}, ${TORCH_MASTER_PORT_END}) overlaps Linux ephemeral ports ${EPHEMERAL_PORT_START}-${EPHEMERAL_PORT_END}" >&2
        exit 1
    fi
fi

echo "=== Starting independent Ray head cluster ==="
echo "Ray binary:     ${RAY_BIN}"
echo "GCS port:       ${GCS_PORT}"
echo "Dashboard port: ${DASHBOARD_PORT}"
echo "Ray client port:${RAY_CLIENT_SERVER_PORT}"
echo "Worker ports:   ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"
echo "Torch ports:    [${TORCH_MASTER_PORT_START}, ${TORCH_MASTER_PORT_END}) reserved"
echo "Resources:      ${NUM_CPUS} CPUs, ${NUM_GPUS} GPUs"
echo "Block:          ${BLOCK}"
echo "Temp dir:       ${TEMP_DIR}"
echo "============================================="

start_args=(
    --head
    --port="${GCS_PORT}" \
    --dashboard-port="${DASHBOARD_PORT}" \
    --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
    --dashboard-agent-grpc-port="${DASHBOARD_AGENT_GRPC_PORT}" \
    --dashboard-agent-listen-port="${DASHBOARD_AGENT_LISTEN_PORT}" \
    --min-worker-port="${MIN_WORKER_PORT}" \
    --max-worker-port="${MAX_WORKER_PORT}" \
    --num-cpus="${NUM_CPUS}" \
    --num-gpus="${NUM_GPUS}" \
    --temp-dir="${TEMP_DIR}" \
    --disable-usage-stats
)
if [[ -n "${NODE_IP}" ]]; then
    start_args+=(--node-ip-address="${NODE_IP}")
fi
if [[ "${BLOCK}" == "true" ]]; then
    start_args+=(--block)
fi

"${RAY_BIN}" start "${start_args[@]}"

if [[ "${BLOCK}" != "true" ]]; then
    echo
    echo "Cluster address: $(cat "${TEMP_DIR}/ray_current_cluster")"
fi
