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
# As a pod start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v8.sh | RUN_NAME=ppo_v8 NUM_GPUS=8 bash"
#
# See docs/devlog.html's "ppo_v8 launch plan" section for the full runbook,
# config rationale, and sizing math this script implements.

set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_v8}"
NUM_GPUS="${NUM_GPUS:-1}"
# Envs PER GPU/shard (see the v8 plan's sizing section: 64 is the
# proven-safe ceiling for the full, non-foveated policy before OOM).
NUM_ENVS="${NUM_ENVS:-64}"
ROLLOUT_LEN="${ROLLOUT_LEN:-32}"
STAGE="${STAGE:-0}"
# Frozen v8 launch config (see devlog): full policy (no --gc/--blocks
# override), AMP on, pinned H2D on, entropy floor at its default. Override
# via EXTRA_ARGS if deliberately deviating from the plan.
EXTRA_ARGS="${EXTRA_ARGS:---amp --pinned-h2d}"
REPO_DIR="${REPO_DIR:-/root/openfront-ai}"
CKPT_DIR="$REPO_DIR/rust/checkpoints/$RUN_NAME"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-600}"
TORCH_VERSION="2.11.0" # tch 0.24's C++ shim needs this exact version - see devlog

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
mkdir -p "$REPO_DIR/rust/.cargo"
cat > "$REPO_DIR/rust/.cargo/config.toml" <<EOF
[env]
LIBTORCH = "$TORCH_LIB"
LD_LIBRARY_PATH = "$TORCH_LIB/lib:$NVRTC_LIB"
EOF

cd "$REPO_DIR/rust"
cargo build --release -p oftrain --features native-engine
# CUDA-not-actually-linked footgun (see devlog 2026-07-09): confirm before
# spending any GPU-hours, not after wondering why util is stuck near 0.
if ! readelf -d target/release/oftrain | grep -q libtorch_cuda.so; then
  echo "FATAL: libtorch_cuda.so not linked into oftrain - CUDA is silently missing (see devlog)"
  exit 1
fi

"$VENV/bin/pip" install --quiet huggingface_hub 2>/dev/null || true
PYTHON="$VENV/bin/python"

mkdir -p "$CKPT_DIR"

# --- resume seed: if the local checkpoint is gone (fresh pod, disk wiped)
# but a synced copy exists on HF, pull it down before starting ---
if [ ! -f "$CKPT_DIR/latest.ot" ]; then
  "$PYTHON" - "$RUN_NAME" "$CKPT_DIR" <<'PYEOF' || true
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

run, dest = sys.argv[1], Path(sys.argv[2])
for f in ("latest.ot", "latest.state.json"):
    try:
        p = hf_hub_download("djmango/openfront-rl", f"{run}/{f}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f).write_bytes(Path(p).read_bytes())
        print(f"restored {f} from HF")
    except Exception as e:
        print(f"{f}: no HF copy ({e.__class__.__name__})")
PYEOF
fi

# --- background HF sync: push the latest checkpoint periodically so
# training survives total pod/disk loss, not just an in-pod crash ---
(
  while true; do
    sleep "$HF_SYNC_INTERVAL_SECONDS"
    if [ -f "$CKPT_DIR/latest.ot" ]; then
      "$PYTHON" - "$RUN_NAME" "$CKPT_DIR" <<'PYEOF' 2>&1 | sed 's/^/[hf-sync] /'
import sys
from pathlib import Path
from huggingface_hub import HfApi

run, ckpt_dir = sys.argv[1], Path(sys.argv[2])
api = HfApi()
try:
    api.create_repo("djmango/openfront-rl", exist_ok=True, repo_type="model")
    for f in ("latest.ot", "latest.state.json"):
        p = ckpt_dir / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{run}/{f}", repo_id="djmango/openfront-rl")
    print("synced latest checkpoint")
except Exception as e:
    print(f"sync failed: {e}")
PYEOF
    fi
  done
) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

# --- crash-proof training loop (backoff, not an instant relaunch into the
# same wall - see pod_train.sh's FAST_EXITS precedent) ---
ulimit -n 65535 2>/dev/null || true
FAST_EXITS=0
while true; do
  RESUME=""
  [ -f "$CKPT_DIR/latest.ot" ] && RESUME="--resume $CKPT_DIR/latest.ot"
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME num_gpus=$NUM_GPUS envs/shard=$NUM_ENVS $RESUME ==="
  START_TS=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB" \
    ./target/release/oftrain --engine native --num-envs "$NUM_ENVS" --num-gpus "$NUM_GPUS" \
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
