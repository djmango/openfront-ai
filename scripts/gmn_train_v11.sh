#!/usr/bin/env bash
# Thin 1×GPU launcher for Give Me A Node / SF Compute H100 nodes.
# Wraps pod_train_v11.sh with a smaller env band so one H100 stays near
# the util SLO without the 4×A100 shard recipe.
#
#   NUM_GPUS=1 bash scripts/gmn_train_v11.sh
#   # or via MCP run_command (detached) once the node is running
set -uo pipefail

export RUN_NAME="${RUN_NAME:-ppo_v11}"
export NUM_GPUS="${NUM_GPUS:-1}"
# Start conservative; --auto-scale-envs climbs toward MAX_ENVS for util.
export NUM_ENVS="${NUM_ENVS:-20}"
export MAX_ENVS="${MAX_ENVS:-32}"
export MIN_ENVS="${MIN_ENVS:-8}"
export TARGET_GPU_UTIL="${TARGET_GPU_UTIL:-0.92}"
export NODE_FRACTION="${NODE_FRACTION:-0}"
export REPO_DIR="${REPO_DIR:-$HOME/openfront-ai}"
export GIT_REF="${GIT_REF:-master}"
# H100 single-GPU: no multi-GPU NCCL path needed.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/pod_train_v11.sh" "$@"
