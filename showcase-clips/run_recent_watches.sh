#!/usr/bin/env bash
# Recent games at the live stage (u806 / s16): a few stochastic watches,
# train-matched tick budget. One watch each — no seed hunting.
set -euo pipefail
cd /workspace
export PATH="/workspace/rust/.libtorch-venv/bin:$PATH"
TORCH_ROOT="/workspace/rust/.libtorch-venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${TORCH_ROOT}/torch/lib:${TORCH_ROOT}/nvidia/cu13/lib:${TORCH_ROOT}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p showcase-clips /opt/cursor/artifacts

POLICY=rust/checkpoints/ppo_v11/latest.safetensors
STATE=rust/checkpoints/ppo_v11/latest.state.json
AE=weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors
COARSE=weights/ae/ae_v32_nostatic_d16c32.encoder.safetensors
OFTRAIN=./rust/target/release/oftrain
MAX_TICKS=21000
MAX_STEPS=$((MAX_TICKS / 10 + 64))

UPD=$(python3 -c "import json;print(json.load(open('$STATE'))['update'])")
STAGE=$(python3 -c "import json;print(json.load(open('$STATE'))['stage'])")
TAG_PREFIX="ppo_v11_u${UPD}"
# s16 lobby from curriculum
BOTS=26; N=3; DIFF=Easy

echo "CHECKPOINT update=$UPD stage=$STAGE lobby=${BOTS}b/${N}n $DIFF ticks=$MAX_TICKS"

run_watch() {
  local TAG=$1 MAP=$2 SEED=$3
  local OUT="showcase-clips/${TAG_PREFIX}_${TAG}.json"
  local LOG="showcase-clips/${TAG_PREFIX}_${TAG}.watch.log"
  echo "=== WATCH $TAG map=$MAP seed=$SEED $(date -u +%H:%M:%S) ===" | tee "$LOG"
  "$OFTRAIN" --watch --watch-stochastic=true --engine native \
    --policy "$POLICY" --ckpt "$AE" --coarse-ckpt "$COARSE" \
    --stage "$STAGE" --map "$MAP" --bots "$BOTS" --nations "$N" --difficulty "$DIFF" \
    --seed "$SEED" --device cuda:0 --amp=true --foveate=true \
    --persistent-actors=true --recurrent-policy=true \
    --max-episode-ticks "$MAX_TICKS" --max-steps "$MAX_STEPS" \
    --record "$OUT" --debug true >>"$LOG" 2>&1
  python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
d=json.loads(Path("showcase-clips/${TAG_PREFIX}_${TAG}.debug.json").read_text())
print(f"outcome={d.get('outcome')} end_tick={d.get('end_tick')}")
PY
}

run_watch "s${STAGE}_europe" Europe "recent_eu_a"
run_watch "s${STAGE}_world" World "recent_world_a"
run_watch "s${STAGE}_asia" Asia "recent_asia_a"
echo ALL_RECENT_WATCHES_DONE
