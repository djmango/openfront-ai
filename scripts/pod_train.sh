#!/usr/bin/env bash
# Self-bootstrapping, restart-proof RL training for a RunPod pod.
#
# Idempotent: safe to run as the pod's start command (survives pod
# restarts/migrations, when /workspace may or may not have survived) or
# manually in tmux. Training auto-resumes from the run's checkpoint, which
# carries optimizer state, curriculum stage, and step counters.
#
#   RUN_NAME=ppo_v2c ENVS=48 bash scripts/pod_train.sh
#   RUN_NAME=ppo_v61 ENVS=384 NPROC=4 bash scripts/pod_train.sh  # multi-GPU DDP
#
# As a pod start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train.sh | RUN_NAME=ppo_v2c bash"

set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_auto}"
# ENVS must be divisible by NPROC (each rank owns ENVS/NPROC envs);
# v6.1 fat-node target: ENVS=384 NPROC=4.
ENVS="${ENVS:-48}"
# GPUs: 1 -> plain python (single-process, identical to pre-v6.1);
# >1 -> torchrun DDP, one rank per GPU.
NPROC="${NPROC:-1}"
STAGE="${STAGE:-0}"
# OOM lever for late-curriculum maps: the 1/8 grid costs ~4x v3's conv
# activations at World/Asia sizes; drop to 64 if stage 6+ OOMs.
MINIBATCH="${MINIBATCH:-128}"
# INIT_BC=bc_v4 warm-starts a FRESH run from that BC run's checkpoint on HF
# (ignored once the run has its own policy.pt; --init defers to --resume).
INIT_BC="${INIT_BC:-}"
# INIT_EXTEND=/path/to/policy.pt warm-starts a FRESH run from a pre-v6
# checkpoint via --init-extend (head growth). Same resume-wins semantics:
# skipped once the run has its own policy.pt, and rl/ppo.py ignores
# --init-extend whenever --resume is passed, so crash relaunches can't
# re-extend and wipe progress.
INIT_EXTEND="${INIT_EXTEND:-}"
REPO_DIR=/workspace/openfront-ai

# When run as the pod start command this replaces the image's /start.sh,
# so keep SSH reachable ourselves.
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

# --- bootstrap (skips anything already present) ---
mkdir -p /workspace
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SKIP_SYNC:-}" ]; then
  # Deployed code MUST match origin/master: a silently failed pull once ran
  # a pod on stale code for a whole day. Pods never carry local commits, so
  # hard-sync and assert.
  git fetch origin master || true
  git reset --hard origin/master || true
  git submodule update --init || true
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master 2>/dev/null)" ]; then
    echo "FATAL: HEAD $(git rev-parse --short HEAD) != origin/master; refusing to train stale code"
    exit 1
  fi
  echo "deployed commit: $(git rev-parse --short HEAD)"
fi  # rsynced copies have no .git; run whatever is present

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y nodejs >/dev/null
fi
command -v tmux >/dev/null 2>&1 || apt-get install -y tmux >/dev/null
[ -d openfront/node_modules ] || (cd openfront && npm install --silent)

# System python (image torch matches the driver); add the small extras.
pip install -q tensorboard huggingface_hub 2>/dev/null | tail -0 || true

# --- optional Rust hot paths (rl/native.py falls back to numpy if absent) ---
if ! python -c "import ofrs" 2>/dev/null; then
  if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y -q >/dev/null 2>&1 || true
  fi
  . "$HOME/.cargo/env" 2>/dev/null || true
  if command -v cargo >/dev/null 2>&1; then
    pip install -q ./rust/ofrs && echo "ofrs native paths built" || echo "ofrs build failed; using numpy fallbacks"
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

# Resume seed: if the local checkpoint is gone (disk wiped) but a synced
# copy exists on HF, pull it down before starting.
if [ ! -f "runs/rl/$RUN_NAME/policy.pt" ]; then
  mkdir -p "runs/rl/$RUN_NAME"
  python -c "
from huggingface_hub import hf_hub_download
import shutil
try:
    p = hf_hub_download('djmango/openfront-rl', '$RUN_NAME/policy.pt')
    shutil.copy(p, 'runs/rl/$RUN_NAME/policy.pt')
    print('restored checkpoint from HF')
except Exception as e:
    print(f'no HF checkpoint ({e.__class__.__name__}); starting fresh')
" || true
fi

# BC warm start: fetch the BC checkpoint if this run hasn't produced its
# own policy.pt yet (after that, resume wins and --init is a no-op).
INIT=""
if [ -n "$INIT_BC" ] && [ ! -f "runs/rl/$RUN_NAME/policy.pt" ]; then
  if [ ! -f "runs/bc/$INIT_BC/bc_init.pt" ]; then
    mkdir -p "runs/bc/$INIT_BC"
    python -c "
from huggingface_hub import hf_hub_download
import shutil
# Prefer the best-holdout checkpoint over the last one.
for f in ('bc_best.pt', 'bc.pt'):
    try:
        p = hf_hub_download('djmango/openfront-rl', f'$INIT_BC/{f}')
        shutil.copy(p, 'runs/bc/$INIT_BC/bc_init.pt')
        print(f'fetched BC checkpoint $INIT_BC/{f}')
        break
    except Exception as e:
        print(f'{f}: {e.__class__.__name__}')
else:
    raise SystemExit('INIT_BC set but no BC checkpoint found on HF')
"
  fi
  INIT="--init runs/bc/$INIT_BC/bc_init.pt"
fi
if [ -n "$INIT_EXTEND" ]; then
  if [ ! -f "$INIT_EXTEND" ]; then
    echo "FATAL: INIT_EXTEND=$INIT_EXTEND does not exist"
    exit 1
  fi
  INIT="$INIT --init-extend $INIT_EXTEND"
fi

# --- tensorboard on the pod's exposed http port ---
if ! pgrep -f "tensorboard.*19123" >/dev/null; then
  nohup tensorboard --logdir runs/rl --port 19123 --host 0.0.0.0 \
    >/tmp/tensorboard.log 2>&1 &
fi

# --- crash-proof training loop (with crash-loop backoff: an auto-restart
# that relaunches into the same wall every 10s is not auto-recovery) ---
FAST_EXITS=0
while true; do
  RESUME=""
  if [ -f "runs/rl/$RUN_NAME/policy.pt" ]; then
    RESUME="--resume runs/rl/$RUN_NAME/policy.pt"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME $RESUME ==="
  START_TS=$(date +%s)
  # MALLOC_*: see pod_bc.sh - keeps large per-batch buffers on the reusable
  # heap instead of per-batch mmap/munmap (glibc caps its dynamic threshold
  # at 32MB; collated batches are bigger), preventing slow page-fault decay.
  if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --standalone --nproc_per_node $NPROC -m"
  else
    LAUNCH="python -m"
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=268435456 \
    $LAUNCH rl.ppo --envs "$ENVS" --updates 100000 --rollout 32 \
    --minibatch "$MINIBATCH" --name "$RUN_NAME" --stage "$STAGE" $RESUME $INIT \
    2>&1 | tee -a "/tmp/train_$RUN_NAME.log"
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ "$ELAPSED" -lt 120 ]; then
    FAST_EXITS=$((FAST_EXITS + 1))
  else
    FAST_EXITS=0
  fi
  BACKOFF=$(( FAST_EXITS >= 2 ? (FAST_EXITS >= 4 ? 600 : 60) : 10 ))
  echo "=== trainer exited ($?) after ${ELAPSED}s; fast-exits=$FAST_EXITS, restarting in ${BACKOFF}s ===" \
    | tee -a "/tmp/train_$RUN_NAME.log"
  sleep "$BACKOFF"
done
