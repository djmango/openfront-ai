#!/usr/bin/env bash
# Self-bootstrapping, restart-proof RL training for a RunPod pod.
#
# Idempotent: safe to run as the pod's start command (survives pod
# restarts/migrations, when /workspace may or may not have survived) or
# manually in tmux. Training auto-resumes from the run's checkpoint, which
# carries optimizer state, curriculum stage, and step counters.
#
#   RUN_NAME=ppo_v2c ENVS=48 bash scripts/pod_train.sh
#
# As a pod start command:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train.sh | RUN_NAME=ppo_v2c bash"

set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_auto}"
ENVS="${ENVS:-48}"
STAGE="${STAGE:-0}"
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
if [ -d .git ]; then
  git pull --ff-only || true
  git submodule update --init || true
fi  # rsynced copies have no .git; run whatever is present

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y nodejs >/dev/null
fi
command -v tmux >/dev/null 2>&1 || apt-get install -y tmux >/dev/null
[ -d openfront/node_modules ] || (cd openfront && npm install --silent)

# System python (image torch matches the driver); add the small extras.
pip install -q tensorboard huggingface_hub 2>/dev/null | tail -0 || true

if [ ! -f runs/ae_v3/ae_v3.pt ]; then
  mkdir -p runs/ae_v3
  python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('djmango/openfront-tile-autoencoder', 'ae_v3.pt')
shutil.copy(p, 'runs/ae_v3/ae_v3.pt')
print('fetched AE checkpoint')
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

# --- tensorboard on the pod's exposed http port ---
if ! pgrep -f "tensorboard.*19123" >/dev/null; then
  nohup tensorboard --logdir runs/rl --port 19123 --host 0.0.0.0 \
    >/tmp/tensorboard.log 2>&1 &
fi

# --- crash-proof training loop ---
while true; do
  RESUME=""
  if [ -f "runs/rl/$RUN_NAME/policy.pt" ]; then
    RESUME="--resume runs/rl/$RUN_NAME/policy.pt"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME $RESUME ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    python -m rl.ppo --envs "$ENVS" --updates 100000 --rollout 32 \
    --minibatch 128 --name "$RUN_NAME" --stage "$STAGE" $RESUME \
    2>&1 | tee -a "/tmp/train_$RUN_NAME.log"
  echo "=== trainer exited ($?); restarting in 10s ===" | tee -a "/tmp/train_$RUN_NAME.log"
  sleep 10
done
