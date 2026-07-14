#!/usr/bin/env bash
# ROCm/AMD (MI300X) counterpart to `pod_train_v8.sh` - restart-proof
# training for the Rust `oftrain` trainer on a RunPod ROCm pod. Deliberately
# a SEPARATE script rather than a branch inside `pod_train_v8.sh`: the
# CUDA path is proven on real training runs and this keeps it byte-for-byte
# untouched rather than risking a conditional regressing it. Same
# structure/safety checks as `pod_train_v8.sh` (deployed-commit assertion,
# crash-loop backoff, HF checkpoint sync) - see that script's comments for
# the rationale behind each; this file only calls out where the ROCm path
# actually differs.
#
#   RUN_NAME=ppo_v8_rocm NUM_GPUS=8 bash scripts/pod_train_v8_rocm.sh
#
# As a pod start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v8_rocm.sh | RUN_NAME=ppo_v8_rocm NUM_GPUS=8 bash"
#
# This script itself is the "gate": the existing pod_train_v8.sh (CUDA, the
# default/proven path) is completely untouched, and picking ROCm is just a
# matter of running this file instead of that one on an AMD pod - no shared
# ENGINE_BACKEND-style branching inside a single script that could regress
# the CUDA path.
#
# ############################################################################
# HONESTY NOTE (see rust/oftrain/ROCM.md for the full writeup): this script
# has NOT been run against a real AMD/ROCm pod. It mirrors pod_train_v8.sh's
# proven structure and fixes/versions confirmed via web research (the
# tch-rs GitHub issue for the link bug, and the actual torch wheel index at
# download.pytorch.org for available ROCm versions), but the venv
# bootstrap, the build.rs HIP link fix, and the sanity checks below are
# unverified end-to-end without real ROCm hardware. Treat first runs on an
# actual MI300X pod as the real test, not a rerun of something proven.
# ############################################################################
#
# Known deliberate difference from pod_train_v8.sh: torch-sys 0.24 (tch-rs
# 0.24, our pinned version) hard-requires PyTorch "2.11.0" exactly unless
# LIBTORCH_BYPASS_VERSION_CHECK=1 is set (see torch-sys's build.rs
# `version_check`). As of this writing, the ROCm wheel index
# (https://download.pytorch.org/whl/rocm7.0/torch/) tops out at
# "2.10.0+rocm7.0" - no 2.11.0 ROCm build exists yet - so this script pins
# the nearest available ROCm torch version and sets the bypass flag. If a
# 2.11.0+rocmN.N wheel has since shipped, prefer switching ROCM_TORCH_VERSION
# to that and dropping the bypass.
set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_v8_rocm}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_ENVS="${NUM_ENVS:-64}"
ROLLOUT_LEN="${ROLLOUT_LEN:-32}"
STAGE="${STAGE:-0}"
NODE_FRACTION="${NODE_FRACTION:-0}"
EXTRA_ARGS="${EXTRA_ARGS:---amp --pinned-h2d}"
REPO_DIR="${REPO_DIR:-/root/openfront-ai}"
CKPT_DIR="$REPO_DIR/rust/checkpoints/$RUN_NAME"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-600}"
HF_REPO_ID="${HF_REPO_ID:-djmango/openfront-rl}"
HF_RUN_PREFIX="${HF_RUN_PREFIX:-$RUN_NAME}"
# Nearest ROCm-published version to torch-sys 0.24's expected "2.11.0" -
# see the HONESTY NOTE above. Override if a matching/newer ROCm wheel ships.
ROCM_TORCH_VERSION="${ROCM_TORCH_VERSION:-2.10.0}"
# ROCm channel to install against (https://download.pytorch.org/whl/rocmX.Y).
ROCM_VERSION="${ROCM_VERSION:-7.0}"

# --- bootstrap: repo (identical to pod_train_v8.sh) ---
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SKIP_SYNC:-}" ]; then
  git fetch origin master || true
  git reset --hard origin/master || true
  git submodule update --init || true
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master 2>/dev/null)" ]; then
    echo "FATAL: HEAD $(git rev-parse --short HEAD) != origin/master; refusing to train stale code"
    exit 1
  fi
  echo "deployed commit: $(git rev-parse --short HEAD)"
fi

# --- rust toolchain (identical to pod_train_v8.sh) ---
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
fi
. "$HOME/.cargo/env" 2>/dev/null || true

# --- Node.js + openfront/bridge deps, same node-fraction gate as
# pod_train_v8.sh (see that script's comment) ---
if [ "$(python3 -c "print(1 if float('$NODE_FRACTION') > 0 else 0)")" = "1" ]; then
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    apt-get install -y nodejs >/dev/null
  fi
  [ -d openfront/node_modules ] || (cd openfront && npm install --silent)
  echo "node engine mix enabled (--node-fraction $NODE_FRACTION): node $(node --version), tsx ready"
fi

# --- ROCm-linked libtorch venv. Differs from pod_train_v8.sh's CUDA venv
# in: (1) the pip index URL (rocmX.Y instead of cuNNN), (2)
# ROCM_TORCH_VERSION rather than the cu128 TORCH_VERSION (see HONESTY NOTE),
# (3) LIBTORCH_BYPASS_VERSION_CHECK=1 to work around torch-sys's hardcoded
# "2.11.0" expectation, and (4) OFTRAIN_ROCM=1 so oftrain's build.rs emits
# the -ltorch_hip/-lc10_hip force-link fix (see build.rs's doc comment and
# https://github.com/LaurentMazare/tch-rs/issues/1015). ---
VENV="$REPO_DIR/rust/.libtorch-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "torch==$ROCM_TORCH_VERSION" --index-url "https://download.pytorch.org/whl/rocm$ROCM_VERSION"
fi
PY_TAG="$("$VENV/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
TORCH_LIB="$VENV/lib/$PY_TAG/site-packages/torch"
mkdir -p "$REPO_DIR/rust/.cargo"
cat > "$REPO_DIR/rust/.cargo/config.toml" <<EOF
[env]
LIBTORCH = "$TORCH_LIB"
LIBTORCH_BYPASS_VERSION_CHECK = "1"
OFTRAIN_ROCM = "1"
LD_LIBRARY_PATH = "$TORCH_LIB/lib:/opt/rocm/lib"
EOF

cd "$REPO_DIR/rust"
OFTRAIN_ROCM=1 cargo build --release -p oftrain --features native-engine
# ROCm equivalent of pod_train_v8.sh's libtorch_cuda.so link-check (see
# devlog 2026-07-09 and build.rs's doc comment for the underlying
# --as-needed drop bug) - confirm before spending any GPU-hours, not after
# wondering why util is stuck near 0.
if ! readelf -d target/release/oftrain | grep -q libtorch_hip.so; then
  echo "FATAL: libtorch_hip.so not linked into oftrain - ROCm is silently missing (see build.rs, ROCM.md)"
  exit 1
fi
if ! command -v rocm-smi >/dev/null 2>&1 && ! command -v amd-smi >/dev/null 2>&1; then
  echo "WARNING: neither rocm-smi nor amd-smi found on PATH - gpu_util.rs's GpuUtilSampler will report no GPU utilization (training still runs)"
fi

"$VENV/bin/pip" install --quiet huggingface_hub 2>/dev/null || true
PYTHON="$VENV/bin/python"

OFTRAIN_ROCM=1 cargo build --release -p ofhub
OFHF="$REPO_DIR/rust/target/release/ofhf"

mkdir -p "$CKPT_DIR"

# --- resume seed: automated launches require the safetensors/state pair. ---
if [ ! -f "$CKPT_DIR/latest.safetensors" ] || [ ! -f "$CKPT_DIR/latest.state.json" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN is required to restore checkpoints from Hugging Face" >&2
    exit 1
  fi
  "$OFHF" pull --checkpoint-dir "$CKPT_DIR" --repo-id "$HF_REPO_ID" \
    --run-prefix "$HF_RUN_PREFIX" || true
fi
if { [ -f "$CKPT_DIR/latest.safetensors" ] && [ ! -f "$CKPT_DIR/latest.state.json" ]; } \
  || { [ ! -f "$CKPT_DIR/latest.safetensors" ] && [ -f "$CKPT_DIR/latest.state.json" ]; }
then
  echo "FATAL: latest.safetensors and latest.state.json must exist as a pair" >&2
  exit 1
fi

# --- background safetensors/state/manifest HF sync. Fail loud without token. ---
if [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: HF_TOKEN is required for ofhf sync-loop (fail loud)" >&2
  exit 1
fi
"$OFHF" sync-loop \
  --checkpoint-dir "$CKPT_DIR" --repo-id "$HF_REPO_ID" \
  --run-prefix "$HF_RUN_PREFIX" --interval "$HF_SYNC_INTERVAL_SECONDS" \
  >>"/tmp/hf_sync_$RUN_NAME.log" 2>&1 &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

# --- crash-proof training loop (identical structure to pod_train_v8.sh,
# see that script's comment; PYTORCH_CUDA_ALLOC_CONF and --device cuda:0
# are unchanged from the CUDA path on purpose - ROCm's torch build reuses
# the same torch.cuda/at::Device(DeviceType::CUDA) API surface, see
# ROCM.md) ---
ulimit -n 65535 2>/dev/null || true
FAST_EXITS=0
while true; do
  RESUME=""
  if [ -f "$CKPT_DIR/latest.safetensors" ]; then
    RESUME="--resume $CKPT_DIR/latest.safetensors"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME (rocm) num_gpus=$NUM_GPUS envs/shard=$NUM_ENVS $RESUME ==="
  START_TS=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  LD_LIBRARY_PATH="$TORCH_LIB/lib:/opt/rocm/lib" \
    ./target/release/oftrain --engine native --node-fraction "$NODE_FRACTION" --num-envs "$NUM_ENVS" --num-gpus "$NUM_GPUS" \
    --rollout-len "$ROLLOUT_LEN" --stage "$STAGE" --device cuda:0 \
    --ckpt-dir "$CKPT_DIR" $EXTRA_ARGS $RESUME \
    >> "/tmp/train_$RUN_NAME.log" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ "$ELAPSED" -lt 120 ]; then
    FAST_EXITS=$((FAST_EXITS + 1))
  else
    FAST_EXITS=0
  fi
  BACKOFF=$(( FAST_EXITS >= 2 ? (FAST_EXITS >= 4 ? 600 : 60) : 10 ))
  echo "=== trainer exited ($RC) after ${ELAPSED}s; fast-exits=$FAST_EXITS, restarting in ${BACKOFF}s ===" \
    | tee -a "/tmp/train_$RUN_NAME.log"
  sleep "$BACKOFF"
done
