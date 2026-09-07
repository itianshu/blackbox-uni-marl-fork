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
# Stop this cluster (do NOT run a bare `ray stop` here: RAY_TMPDIR=/opt/tiger
# in this shell would stop the system bytedray cluster instead):
#   "${RAY_BIN}" stop --temp-dir="${TEMP_DIR}"

set -euo pipefail

RAY_BIN="${RAY_BIN:-/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/ray}"
# Outside Ray's default worker port range (10002-19999).
GCS_PORT="${GCS_PORT:-20001}"
# Outside Ray's default worker port range (10002-19999).
DASHBOARD_PORT="${DASHBOARD_PORT:-20002}"
TEMP_DIR="${TEMP_DIR:-/tmp/ray_zzh_255}"
# Leave empty to let Ray auto-detect the node IP (recommended for a single
# node); set explicitly only when you need to bind a specific interface.
NODE_IP="${NODE_IP:-}"

echo "=== Starting independent Ray head cluster ==="
echo "Ray binary:     ${RAY_BIN}"
echo "GCS port:       ${GCS_PORT}"
echo "Dashboard port: ${DASHBOARD_PORT}"
echo "Temp dir:       ${TEMP_DIR}"
echo "============================================="

start_args=(
    --head
    --port="${GCS_PORT}" \
    --dashboard-port="${DASHBOARD_PORT}" \
    --temp-dir="${TEMP_DIR}" \
    --disable-usage-stats
)
if [[ -n "${NODE_IP}" ]]; then
    start_args+=(--node-ip-address="${NODE_IP}")
fi

"${RAY_BIN}" start "${start_args[@]}"

echo
echo "Cluster address: $(cat "${TEMP_DIR}/ray_current_cluster")"
