#!/usr/bin/env bash
# Train and publish the v7 coarse /16 AE used by the foveated PPO stream.
#
# Start command example:
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_ae16.sh | bash"

set -euo pipefail

REPO_DIR=/workspace/openfront-ai
OUT="${OUT:-runs/ae_v31_d16c32}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-32}"

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

mkdir -p /workspace
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin master
git reset --hard origin/master
git submodule update --init

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y nodejs >/dev/null
fi
command -v tmux >/dev/null 2>&1 || apt-get install -y tmux >/dev/null
pip install -q numpy zstandard "huggingface_hub[hf_transfer]" tensorboard

if ! pgrep -f "tensorboard.*19123" >/dev/null; then
  nohup tensorboard --logdir runs --port 19123 --host 0.0.0.0 \
    >/tmp/tensorboard.log 2>&1 &
fi

# Data staging is idempotent and skips already-extracted/cached games.
WORKERS="$WORKERS" bash scripts/runpod_bootstrap.sh

if [ ! -f runs/ae_v31_d8c32/ae_v3.pt ]; then
  mkdir -p runs/ae_v31_d8c32
  python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("djmango/openfront-tile-autoencoder", "ae_v31_d8c32.pt")
shutil.copy(p, "runs/ae_v31_d8c32/ae_v3.pt")
print("fetched ae_v31_d8c32")
PY
fi

if [ ! -f "$OUT/ae_v3.pt" ] || [ -n "${FORCE:-}" ]; then
  PYTHONPATH=. python -m ae.train_v3 \
    --data data,data-human \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --latent-down 16 \
    --latent-c 32 \
    --init runs/ae_v31_d8c32/ae_v3.pt \
    --out "$OUT" \
    2>&1 | tee -a /tmp/ae_v31_d16c32.log
fi

PYTHONPATH=. python scripts/eval_v3.py \
  --ckpt "$OUT/ae_v3.pt" \
  --data data-human \
  --samples "${EVAL_SAMPLES:-512}" \
  --out "$OUT/eval_human.json"

if [ -n "${HF_TOKEN:-}" ]; then
  python - <<'PY'
from huggingface_hub import HfApi
from pathlib import Path
api = HfApi()
api.create_repo("djmango/openfront-tile-autoencoder", exist_ok=True)
api.upload_file(
    path_or_fileobj="runs/ae_v31_d16c32/ae_v3.pt",
    path_in_repo="ae_v31_d16c32.pt",
    repo_id="djmango/openfront-tile-autoencoder",
)
eval_path = Path("runs/ae_v31_d16c32/eval_human.json")
if eval_path.exists():
    api.upload_file(
        path_or_fileobj=str(eval_path),
        path_in_repo="ae_v31_d16c32_eval_human.json",
        repo_id="djmango/openfront-tile-autoencoder",
    )
print("uploaded ae_v31_d16c32")
PY
else
  echo "HF_TOKEN unset; skipped AE upload"
fi

echo "ae16 complete: $OUT/ae_v3.pt"
