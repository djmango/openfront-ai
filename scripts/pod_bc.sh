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
STEPS="${STEPS:-60000}"
WORKERS="${WORKERS:-16}"
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
[ -d .git ] && git pull --ff-only || true
command -v tmux >/dev/null 2>&1 || apt-get install -y tmux >/dev/null
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

# --- human dataset: map tars + bc sidecars from HF ---
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

# --- crash-proof training loop ---
while true; do
  RESUME=""
  if [ -f "runs/bc/$RUN_NAME/bc.pt" ]; then
    RESUME="--resume runs/bc/$RUN_NAME/bc.pt"
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME (seq=$SEQ) $RESUME ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
    python -m rl.bc --data data-human --name "$RUN_NAME" --seq "$SEQ" \
    --batch "$BATCH" --steps "$STEPS" --workers "$WORKERS" $RESUME \
    2>&1 | tee -a "/tmp/bc_$RUN_NAME.log"
  echo "=== trainer exited ($?); restarting in 10s ===" | tee -a "/tmp/bc_$RUN_NAME.log"
  sleep 10
done
