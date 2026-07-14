#!/usr/bin/env bash
# Restart-proof training for the Rust `oftrain` trainer (v8 port) on a
# RunPod pod. Rust-flavored equivalent of `pod_train.sh` (the Python-era
# ppo_v2-v7 wrapper): bootstraps the repo + libtorch/CUDA venv + build,
# resumes from a local or HF-synced checkpoint, crash-loops with backoff,
# and periodically pushes the latest checkpoint to `djmango/openfront-rl`
# so a fresh pod (or one after total disk loss) can pick training back up.
#
#   RUN_NAME=ppo_v8 NUM_GPUS=8 bash scripts/pod_train_v8.sh
#
# As a RunPod `dockerArgs` pod start command - deliberately does NOT run
# this script as the container's own foreground process (see docs/devlog.html's
# 2026-07-12 "don't tie the container's lifecycle to the training script"
# entry for why that caused repeated "container not found" SSH failures
# that terminating-and-relaunching pods did NOT fix, since it wasn't a
# host-specific problem): starts sshd explicitly, launches this script
# fully detached in the background, and keeps the container itself alive
# with `sleep infinity` regardless of what happens to the training process.
#   bash -c "service ssh start 2>/dev/null || /usr/sbin/sshd; nohup bash -c 'curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v8.sh -o /root/pod_train_v8.sh && RUN_NAME=ppo_v8 NUM_GPUS=8 bash /root/pod_train_v8.sh' > /root/bootstrap.log 2>&1 & disown; sleep infinity"
#
# To hedge native's known weaker parity at higher bot counts by running a
# fraction of envs on the real Node/TS engine (see oftrain's `--engine` doc
# comment), set NODE_FRACTION (0.0-1.0, default 0 = pure native, no extra
# bootstrap cost):
#   RUN_NAME=ppo_v8 NUM_GPUS=4 NODE_FRACTION=0.2 bash scripts/pod_train_v8.sh
#
# If a pod fails to actually train (crash-loops immediately, "CUDA unknown
# error" in the trainer's own log at /tmp/train_$RUN_NAME.log) despite
# `nvidia-smi` looking healthy: this has already happened on a RunPod
# community-cloud host once (see docs/devlog.html's "v8 launch" entry,
# 2026-07-12) and was NOT fixable in code - `cuInit()` itself failed for the
# compiled binary on that specific host while plain `python3 -c "import
# torch"` worked fine in the identical environment. Don't sink time
# bisecting it again - terminate that pod and relaunch (prefer secure cloud
# over community; a different physical host resolved it instantly last
# time, zero code changes needed).
#
# See docs/devlog.html's "ppo_v8 launch plan" section for the full runbook,
# config rationale, and sizing math this script implements.

set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_v8}"
NUM_GPUS="${NUM_GPUS:-1}"
V81_CURRICULUM="${V81_CURRICULUM:-0}"
# Envs per GPU/shard. Live A40 A/Bs found 48 faster than 64 once the
# persistent compact path was enabled (64 increased stage-2 tail latency).
if [ "$V81_CURRICULUM" = "1" ]; then
  NUM_ENVS="${NUM_ENVS:-24}"
else
  NUM_ENVS="${NUM_ENVS:-48}"
fi
STAGE_ENV_TARGETS="${STAGE_ENV_TARGETS:-}"
ROLLOUT_LEN="${ROLLOUT_LEN:-32}"
# Match rl/ppo.py's optimizer cadence: its --minibatch is a target sample
# count (128), while oftrain's --minibatches is a count. Derive the latter
# so scaling env workers does not silently make each critic update weaker.
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
MINIBATCHES=$((NUM_ENVS * ROLLOUT_LEN / MINIBATCH_SIZE))
[ "$MINIBATCHES" -ge 1 ] || MINIBATCHES=1
STAGE="${STAGE:-0}"
# Fraction (0.0-1.0) of env workers that run the real Node/TS engine
# instead of native, to hedge native's known parity gaps at higher bot
# counts (see oftrain's `--engine` doc comment) while still getting
# native's ~10x tick speed for the majority. 0 (default) = pure native,
# same as before this option existed - no extra bootstrap cost in that case.
NODE_FRACTION="${NODE_FRACTION:-0}"
# Validated Jul-13 one-GPU recipe: actor/learner CUDA state remains on
# persistent owner threads, rollout payloads cross threads as compact host
# data, and two env groups overlap stepping with actor inference. Keep
# fp16-rollout opt-in until it receives the same extended CUDA soak.
EXTRA_ARGS="${EXTRA_ARGS:---amp --foveate --compact-rollout --persistent-actors --pipeline-groups=true --coarse-ckpt ../weights/ae/ae_v31_d16c32.encoder.safetensors --ckpt ../weights/ae/ae_v31_d8c32.encoder.safetensors}"
V81_ARGS=""
if [ "$V81_CURRICULUM" = "1" ]; then
  V81_ARGS="--v81-curriculum"
fi
if [ -n "$STAGE_ENV_TARGETS" ]; then
  V81_ARGS="$V81_ARGS --stage-env-targets $STAGE_ENV_TARGETS"
fi
REPO_DIR="${REPO_DIR:-/root/openfront-ai}"
CKPT_DIR="$REPO_DIR/rust/checkpoints/$RUN_NAME"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-600}"
HF_REPO_ID="${HF_REPO_ID:-djmango/openfront-rl}"
HF_RUN_PREFIX="${HF_RUN_PREFIX:-$RUN_NAME}"
# The current RunPod A40 host advertises direct CUDA P2P, but its first NCCL
# collective wedges on that transport. Shared-memory transport reduced the
# same 48 MiB gradient in ~113 ms. Override only after a host-specific P2P
# smoke succeeds (for example on a validated NVLink box).
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_VERSION="2.11.0" # tch 0.24's C++ shim needs this exact version - see devlog
AE_DIR="${AE_DIR:-$REPO_DIR/weights/ae}"

# --- bootstrap: repo ---
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SKIP_SYNC:-}" ]; then
  # Same "deployed code MUST match origin/master" assertion as pod_train.sh
  # - a silently-failed pull once ran a whole day's training on stale code.
  git fetch origin master || true
  git reset --hard origin/master || true
  git submodule update --init || true
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master 2>/dev/null)" ]; then
    echo "FATAL: HEAD $(git rev-parse --short HEAD) != origin/master; refusing to train stale code"
    exit 1
  fi
  echo "deployed commit: $(git rev-parse --short HEAD)"
fi

# --- rust toolchain ---
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
fi
. "$HOME/.cargo/env" 2>/dev/null || true

# --- Node.js + openfront/bridge deps, ONLY if any envs will run the Node
# engine (--node-fraction > 0) - the pure-native path (the default) never
# needed this, and Bridge::spawn shells out to openfront/node_modules/.bin/
# tsx, so it must exist before oftrain is even launched, not just before
# the eventual `--engine node` call site fails. Same install lines as
# pod_train.sh (the Python-era script), which already proved this works. ---
if [ "$(python3 -c "print(1 if float('$NODE_FRACTION') > 0 else 0)")" = "1" ]; then
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    apt-get install -y nodejs >/dev/null
  fi
  [ -d openfront/node_modules ] || (cd openfront && npm install --silent)
  echo "node engine mix enabled (--node-fraction $NODE_FRACTION): node $(node --version), tsx ready"
fi

# --- CUDA-linked libtorch venv (see devlog: must be exactly $TORCH_VERSION,
# tch 0.24's C++ shim calls ATen ops that don't exist in older/newer torch
# headers and fails to *compile*, not just link, against a mismatched one) ---
VENV="$REPO_DIR/rust/.libtorch-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
fi
PY_TAG="$("$VENV/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
TORCH_LIB="$VENV/lib/$PY_TAG/site-packages/torch"
NVRTC_LIB="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_INCLUDE="/usr/local/cuda/include"
CUDA_LIB="/usr/local/cuda/lib64"
if [ ! -f "$CUDA_INCLUDE/cuda_runtime_api.h" ]; then
  CUDA_INCLUDE="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_runtime/include"
  CUDA_LIB="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_runtime/lib"
fi
NCCL_ROOT="$VENV/lib/$PY_TAG/site-packages/nvidia/nccl"
NCCL_INCLUDE="$NCCL_ROOT/include"
NCCL_LIB="$NCCL_ROOT/lib"
NCCL_LINK_LIB="$NCCL_LIB"
if [ ! -f "$NCCL_LIB/libnccl.so" ] && [ -f "$NCCL_LIB/libnccl.so.2" ]; then
  # NVIDIA runtime wheels may omit the unversioned linker name.
  NCCL_LINK_LIB="$REPO_DIR/rust/.nccl-link"
  mkdir -p "$NCCL_LINK_LIB"
  ln -sf "$NCCL_LIB/libnccl.so.2" "$NCCL_LINK_LIB/libnccl.so"
fi
LIBTORCH_CXX11_ABI="$("$VENV/bin/python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
mkdir -p "$REPO_DIR/rust/.cargo"
cat > "$REPO_DIR/rust/.cargo/config.toml" <<EOF
[env]
LIBTORCH = "$TORCH_LIB"
LIBTORCH_CXX11_ABI = "$LIBTORCH_CXX11_ABI"
LD_LIBRARY_PATH = "$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB"
EOF

cd "$REPO_DIR/rust"
BUILD_FEATURES="native-engine"
if [ "$NUM_GPUS" -gt 1 ] && [ -f "$CUDA_INCLUDE/cuda_runtime_api.h" ] \
  && [ -f "$CUDA_LIB/libcudart.so" ] \
  && [ -f "$NCCL_INCLUDE/nccl.h" ] && [ -f "$NCCL_LINK_LIB/libnccl.so" ]; then
  export CUDA_INCLUDE_DIR="$CUDA_INCLUDE"
  export CUDA_LIB_DIR="$CUDA_LIB"
  export NCCL_INCLUDE_DIR="$NCCL_INCLUDE"
  export NCCL_LIB_DIR="$NCCL_LINK_LIB"
  BUILD_FEATURES="$BUILD_FEATURES,nccl"
  echo "NCCL preflight: headers and runtime found; enabling device gradient all-reduce"
else
  echo "NCCL preflight: unavailable or single-GPU launch; building CPU gradient fallback"
fi
if ! cargo build --release -p oftrain --features "$BUILD_FEATURES"; then
  if [[ "$BUILD_FEATURES" == *nccl* ]]; then
    echo "WARNING: NCCL build preflight failed; rebuilding with the CPU gradient fallback" >&2
    BUILD_FEATURES="native-engine"
    cargo build --release -p oftrain --features "$BUILD_FEATURES" || exit 1
  else
    exit 1
  fi
fi
# CUDA-not-actually-linked footgun (see devlog 2026-07-09): confirm before
# spending any GPU-hours, not after wondering why util is stuck near 0.
if ! readelf -d target/release/oftrain | grep -q libtorch_cuda.so; then
  echo "FATAL: libtorch_cuda.so not linked into oftrain - CUDA is silently missing (see devlog)"
  exit 1
fi
if [[ "$BUILD_FEATURES" == *nccl* ]]; then
  if ! LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" ldd target/release/oftrain \
    | grep -q 'libnccl.so.*=>'; then
    echo "WARNING: NCCL runtime preflight failed; rebuilding with the CPU gradient fallback" >&2
    BUILD_FEATURES="native-engine"
    cargo build --release -p oftrain --features "$BUILD_FEATURES" || exit 1
  fi
fi
# Host-level CUDA-init footgun (see devlog 2026-07-09 "v8 launch" entry): a
# RunPod community-cloud host once had a working nvidia-smi/driver but a
# broken cuInit() for THIS compiled binary specifically (worked fine from
# plain python3 -c "import torch" in the identical env) - not fixable in
# code, only by relaunching on a different host. Warn-only, deliberately
# NOT fatal (a real GPU pod is the only place this can ever run, and this
# exact invocation - --num-envs 1 --updates 0, minimal but still spawning a
# real native env worker - has only been validated on CPU where the CUDA
# panic masks any other failure mode this specific shape could have; a bug
# *in this check itself* must never be able to kill a launch the real
# trainer, with its own crash-loop/backoff, would have survived).
if ! LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" OFTRAIN_EXPLICIT_CUINIT=1 \
  timeout 30 ./target/release/oftrain --engine native --node-fraction 0 --num-envs 1 --num-gpus 1 \
  --rollout-len 1 --updates 0 --device cuda:0 --ckpt-dir /tmp/oftrain_cuda_preflight \
  2>&1 | grep -q "explicit cuInit(0) -> 0"; then
  echo "WARNING: cuInit() preflight didn't report success (see 2026-07-09 devlog entry for the" \
       "known-bad-host failure mode this checks for) - proceeding anyway; the real trainer's own" \
       "crash-loop will retry/backoff if this host genuinely can't init CUDA." >&2
fi

"$VENV/bin/pip" install --quiet huggingface_hub safetensors numpy 2>/dev/null || true
PYTHON="$VENV/bin/python"

# Fine + coarse AE encoder safetensors for oftrain --ckpt / --coarse-ckpt.
mkdir -p "$AE_DIR"
if [ ! -f "$AE_DIR/ae_v31_d8c32.encoder.safetensors" ] || [ ! -f "$AE_DIR/ae_v31_d16c32.encoder.safetensors" ]; then
  echo "=== fetching/exporting AE encoders into $AE_DIR ==="
  AE_DIR="$AE_DIR" PYTHON="$PYTHON" bash "$REPO_DIR/scripts/fetch_ae_encoders.sh"
fi

# Build ofhub (HF sync) alongside oftrain.
cargo build --release -p ofhub
OFHF="$REPO_DIR/rust/target/release/ofhf"

mkdir -p "$CKPT_DIR"

# --- resume seed: current/future runs restore a complete safetensors pair
# only. Explicit `oftrain --resume old.ot` remains available for manual
# legacy migrations, but automated launches never select it. ---
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

# --- background HF sync: immutable snapshots of latest, best-eval,
# milestones, state sidecars, and the run manifest. Fail loud without token. ---
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

# --- crash-proof training loop (backoff, not an instant relaunch into the
# same wall - see pod_train.sh's FAST_EXITS precedent) ---
ulimit -n 65535 2>/dev/null || true
FAST_EXITS=0
while true; do
  RESUME=""
  if [ -f "$CKPT_DIR/latest.safetensors" ]; then
    RESUME="--resume $CKPT_DIR/latest.safetensors"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME num_gpus=$NUM_GPUS envs/shard=$NUM_ENVS $RESUME ==="
  START_TS=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
  NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  OFTRAIN_NCCL_TIMEOUT_SECONDS="${OFTRAIN_NCCL_TIMEOUT_SECONDS:-60}" \
  LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" \
    ./target/release/oftrain --engine native --node-fraction "$NODE_FRACTION" --num-envs "$NUM_ENVS" --num-gpus "$NUM_GPUS" \
    --rollout-len "$ROLLOUT_LEN" --minibatches "$MINIBATCHES" --stage "$STAGE" --device cuda:0 \
    --ckpt-dir "$CKPT_DIR" $V81_ARGS $EXTRA_ARGS $RESUME \
    >> "/tmp/train_$RUN_NAME.log" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ -f "$CKPT_DIR/restart_request.json" ]; then
    REQUESTED_ENVS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["requested_envs_per_shard"])' "$CKPT_DIR/restart_request.json")
    echo "=== intentional stage resize requested: $NUM_ENVS -> $REQUESTED_ENVS envs/shard; restarting now ===" \
      | tee -a "/tmp/train_$RUN_NAME.log"
    NUM_ENVS="$REQUESTED_ENVS"
    MINIBATCHES=$((NUM_ENVS * ROLLOUT_LEN / MINIBATCH_SIZE))
    [ "$MINIBATCHES" -ge 1 ] || MINIBATCHES=1
    FAST_EXITS=0
    continue
  fi
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
