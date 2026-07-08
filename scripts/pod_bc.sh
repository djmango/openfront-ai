#!/usr/bin/env bash
# Self-bootstrapping, restart-proof behavior-cloning training for a RunPod pod.
#
# Mirrors scripts/pod_train.sh, but the dataset is the replayed human games:
# it pulls the map tars + bc-sidecar tar from the HF dataset repo
# (djmango/openfront-human-games) into /workspace/openfront-ai/data-human
# (~110GB extracted; size the pod volume >= 200GB).
#
#   RUN_NAME=bc_v1 bash scripts/pod_bc.sh              # feedforward (option 2)
#   RUN_NAME=bc_seq_v1 SEQ=8 bash scripts/pod_bc.sh    # temporal (option 3)
#
# As a pod start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_bc.sh | RUN_NAME=bc_v1 bash"

set -uo pipefail

RUN_NAME="${RUN_NAME:-bc_v1}"
SEQ="${SEQ:-0}"
BATCH="${BATCH:-96}"
# Dynamic area-based batch sizing (rl/bc.py --batch-cells): 0 derives the
# latent-cell budget from BATCH at the largest curriculum grid, so BATCH
# stays the reference size for the biggest maps and small-map buckets run
# up to 4x BATCH. Set BC_COMPILE=0 to opt out of per-bucket torch.compile.
BATCH_CELLS="${BATCH_CELLS:-0}"
export BC_COMPILE="${BC_COMPILE:-1}"
ACCUM="${ACCUM:-1}"
STEPS="${STEPS:-60000}"
WORKERS="${WORKERS:-16}"
# AE-latent cache budget (rl/bc.py --z-cache-gb). With the default
# disk-backed slabs (--z-cache-dir auto -> runs/bc/zcache) the budget is
# DISK, not RAM: file pages are kernel-reclaimable, so this can be sized
# to the full dataset (~440GB fp16) even on cgroup-capped pods (bc2:
# 116GiB cap; bc3: 250GB cap - anonymous slabs got the trainer
# OOM-killed there). Pure-RAM budgets (Z_CACHE_DIR="") are clamped to
# the cgroup limit in rl/bc.py.
Z_CACHE_GB="${Z_CACHE_GB:-450}"
Z_CACHE_DIR="${Z_CACHE_DIR:-auto}"
# Optional pre-v6 checkpoint to warm-start from (rl.bc --init-extend).
# Safe to leave set across relaunches: rl.bc ignores it once --resume is
# passed, and the loop below passes --resume as soon as bc.pt exists.
INIT_EXTEND="${INIT_EXTEND:-}"
REPO_DIR=/workspace/openfront-ai
# Keep the HF cache off the small container disk.
export HF_HOME=/workspace/hf-cache

if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  grep -qF "$PUBLIC_KEY" ~/.ssh/authorized_keys 2>/dev/null \
    || echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
fi
if ! pgrep -x sshd >/dev/null 2>&1; then
  mkdir -p /run/sshd
  ssh-keygen -A >/dev/null 2>&1 || true
  /usr/sbin/sshd || true
fi

# --- bootstrap ---
mkdir -p /workspace
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SKIP_SYNC:-}" ]; then
  # Deployed code MUST match origin/master (a silently failed pull once ran
  # a pod on stale code all day). Pods never carry local commits.
  git fetch origin master || true
  git reset --hard origin/master || true
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master 2>/dev/null)" ]; then
    echo "FATAL: HEAD $(git rev-parse --short HEAD) != origin/master; refusing to train stale code"
    exit 1
  fi
  echo "deployed commit: $(git rev-parse --short HEAD)"
fi
command -v tmux >/dev/null 2>&1 || apt-get install -y tmux >/dev/null
pip install -q tensorboard huggingface_hub orjson zstandard 2>/dev/null | tail -0 || true

# --- optional Rust hot paths (rl/native.py falls back to numpy if absent).
# Rebuild whenever the installed module is older than rust/ofrs/Cargo.toml
# (a stale ofrs would e.g. silently drop z_key -> 0% AE-latent-cache hits).
OFRS_WANT=$(sed -n 's/^version = "\(.*\)"/\1/p' rust/ofrs/Cargo.toml | head -1)
OFRS_HAVE=$(python -c "import ofrs; print(getattr(ofrs, '__version__', '0'))" 2>/dev/null || echo none)
if [ "$OFRS_HAVE" != "$OFRS_WANT" ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y -q >/dev/null 2>&1 || true
  fi
  . "$HOME/.cargo/env" 2>/dev/null || true
  if command -v cargo >/dev/null 2>&1; then
    pip install -q ./rust/ofrs && echo "ofrs native paths built ($OFRS_WANT)" \
      || echo "ofrs build failed; using numpy fallbacks"
  else
    echo "no rust toolchain; using numpy fallbacks"
  fi
fi

if [ ! -f runs/ae_v31_d8c32/ae_v3.pt ]; then
  mkdir -p runs/ae_v31_d8c32
  python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('djmango/openfront-tile-autoencoder', 'ae_v31_d8c32.pt')
shutil.copy(p, 'runs/ae_v31_d8c32/ae_v3.pt')
print('fetched AE checkpoint (v3.1 d8c32)')
"
fi

# --- human dataset ---
# Preferred: prebuilt cache-bc tars (prefeaturized frames + v2 sidecars,
# ~26GB) uploaded from a previous pod. Training only reads cache-bc/, so
# this skips the 110GB raw download, the engine replay, and prefeaturize.
# Set RAW_DATA=1 to force the old full pipeline (needed to REBUILD caches).
if [ ! -f data-human/.complete ] && [ -z "${RAW_DATA:-}" ]; then
  python - <<'EOF'
import os
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo = "djmango/openfront-human-games"
api = HfApi()
tars = [f for f in api.list_repo_files(repo, repo_type="dataset")
        if f.startswith("cache-bc/") and f.endswith(".tar")]
Path("data-human").mkdir(exist_ok=True)
for i, f in enumerate(sorted(tars)):
    print(f"[{i+1}/{len(tars)}] {f}", flush=True)
    p = hf_hub_download(repo, f, repo_type="dataset")
    with tarfile.open(p) as t:
        t.extractall("data-human")
    os.remove(os.path.realpath(p))
if tars:
    Path("data-human/.complete").touch()
EOF
fi

# Fallback / cache-rebuild path: raw map tars + bc sidecars (~110GB).
# Marker file makes the (long) download+extract idempotent across restarts.
if [ ! -f data-human/.complete ]; then
  python - <<'EOF'
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo = "djmango/openfront-human-games"
api = HfApi()
tars = [f for f in api.list_repo_files(repo, repo_type="dataset")
        if f.endswith(".tar") and (f.startswith("maps/") or f.startswith("bc/"))]
Path("data-human").mkdir(exist_ok=True)
import os
for i, f in enumerate(sorted(tars)):
    print(f"[{i+1}/{len(tars)}] {f}", flush=True)
    p = hf_hub_download(repo, f, repo_type="dataset")
    with tarfile.open(p) as t:
        t.extractall("data-human")
    os.remove(os.path.realpath(p))  # drop the cached blob; disk is tight
EOF
  n=$(ls -d data-human/*/*/ 2>/dev/null | wc -l)
  echo "extracted $n game dirs"
  [ "$n" -gt 0 ] && touch data-human/.complete
fi

# --- prefeaturize into cache-bc (idempotent; ~10 min parallel, one-time) ---
# This is what takes BC from ~34 ex/s (gzip+JSON per sample) to GPU-bound.
PYTHONPATH=. python scripts/prefeaturize_bc.py --data data-human \
  --workers "$WORKERS" 2>&1 | tail -5

# Resume seed from HF if local checkpoint is gone.
if [ ! -f "runs/bc/$RUN_NAME/bc.pt" ]; then
  mkdir -p "runs/bc/$RUN_NAME"
  python -c "
from huggingface_hub import hf_hub_download
import shutil
try:
    p = hf_hub_download('djmango/openfront-rl', '$RUN_NAME/bc.pt')
    shutil.copy(p, 'runs/bc/$RUN_NAME/bc.pt')
    print('restored checkpoint from HF')
except Exception as e:
    print(f'no HF checkpoint ({e.__class__.__name__}); starting fresh')
" || true
fi

# --- crash-proof training loop (with crash-loop backoff: an auto-restart
# that relaunches into the same wall every 10s is not auto-recovery) ---
FAST_EXITS=0
while true; do
  RESUME=""
  if [ -f "runs/bc/$RUN_NAME/bc.pt" ]; then
    RESUME="--resume runs/bc/$RUN_NAME/bc.pt"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME (seq=$SEQ accum=$ACCUM) $RESUME ==="
  START_TS=$(date +%s)
  # MALLOC_*: recurring batch buffers must come from a reusable heap, or
  # every batch pays mmap/munmap + zero-page faults that degrade as
  # physical memory fragments - the slow decay (bc_v4: 88 -> 12 ex/s over
  # ~5h; col 3.4s -> 14s per window even after 9c030f7's 256MB threshold).
  # The real fix is in code (persistent staged buffers for collate/stack
  # outputs, slab-allocated z-cache); the thresholds keep whatever mid-size
  # transients remain on the heap. ARENA_MAX stays at 2: 1 serialized the
  # 16 rayon sampler threads on one malloc lock (smp 0 -> 12s per window).
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=1073741824 \
    python -m rl.bc --data data-human --name "$RUN_NAME" --seq "$SEQ" \
    --batch "$BATCH" --batch-cells "$BATCH_CELLS" \
    --accum "$ACCUM" --steps "$STEPS" --workers "$WORKERS" \
    --z-cache-gb "$Z_CACHE_GB" --z-cache-dir "$Z_CACHE_DIR" \
    ${INIT_EXTEND:+--init-extend "$INIT_EXTEND"} $RESUME \
    2>&1 | tee -a "/tmp/bc_$RUN_NAME.log"
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ "$ELAPSED" -lt 120 ]; then
    FAST_EXITS=$((FAST_EXITS + 1))
  else
    FAST_EXITS=0
  fi
  BACKOFF=$(( FAST_EXITS >= 2 ? (FAST_EXITS >= 4 ? 600 : 60) : 10 ))
  echo "=== trainer exited ($?) after ${ELAPSED}s; fast-exits=$FAST_EXITS, restarting in ${BACKOFF}s ===" \
    | tee -a "/tmp/bc_$RUN_NAME.log"
  sleep "$BACKOFF"
done
