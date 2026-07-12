#!/usr/bin/env bash
# Download fine + coarse SpatialAE checkpoints from Hugging Face and export
# encoder-only safetensors for `oftrain --ckpt` / `--coarse-ckpt`.
#
#   bash scripts/fetch_ae_encoders.sh
#   AE_DIR=weights/ae bash scripts/fetch_ae_encoders.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
AE_DIR="${AE_DIR:-weights/ae}"
mkdir -p runs/ae_v31_d8c32 runs/ae_v31_d16c32 "$AE_DIR"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python3
  fi
fi

"$PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path
for name, dest in [
    ("ae_v31_d8c32.pt", "runs/ae_v31_d8c32/ae_v3.pt"),
    ("ae_v31_d16c32.pt", "runs/ae_v31_d16c32/ae_v3.pt"),
]:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if Path(dest).exists():
        print(f"keep {dest}")
        continue
    p = hf_hub_download("djmango/openfront-tile-autoencoder", name)
    shutil.copy(p, dest)
    print(f"fetched {name} -> {dest}")
PY

PYTHONPATH=. "$PYTHON" scripts/export_safetensors.py \
  --ae runs/ae_v31_d8c32/ae_v3.pt \
  --out "$AE_DIR/ae_v31_d8c32.encoder.safetensors"
PYTHONPATH=. "$PYTHON" scripts/export_safetensors.py \
  --ae runs/ae_v31_d16c32/ae_v3.pt \
  --expected-down 16 \
  --out "$AE_DIR/ae_v31_d16c32.encoder.safetensors"
echo "AE encoders ready under $AE_DIR"
