#!/usr/bin/env bash
# Human-facing watch episodes: stochastic + native + full train tick budget,
# distinct non-Onion maps. See .cursor/rules/showcase-clips.mdc.
set -euo pipefail
cd /workspace
export PATH="/workspace/rust/.libtorch-venv/bin:$PATH"
TORCH_ROOT="/workspace/rust/.libtorch-venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${TORCH_ROOT}/torch/lib:${TORCH_ROOT}/nvidia/cu13/lib:${TORCH_ROOT}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p showcase-clips /opt/cursor/artifacts

POLICY=rust/checkpoints/ppo_v11/latest.safetensors
AE=weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors
COARSE=weights/ae/ae_v32_nostatic_d16c32.encoder.safetensors
OFTRAIN=./rust/target/release/oftrain

# Shared with train (ofcore::DEFAULT_MAX_EPISODE_TICKS).
MAX_TICKS=21000

run_watch() {
  local STAGE=$1 N=$2 BOTS=$3 TAG=$4 MAP=$5 SEED=$6
  local OUT="showcase-clips/ppo_v11_u676_${TAG}.json"
  local LOG="showcase-clips/ppo_v11_u676_${TAG}.watch.log"
  echo "=== WATCH stage=$STAGE n=$N bots=$BOTS map=$MAP seed=$SEED ($TAG) $(date -u +%H:%M:%S) ===" | tee "$LOG"
  "$OFTRAIN" \
    --watch \
    --watch-stochastic=true \
    --engine native \
    --policy "$POLICY" \
    --ckpt "$AE" \
    --coarse-ckpt "$COARSE" \
    --stage "$STAGE" \
    --map "$MAP" \
    --bots "$BOTS" \
    --nations "$N" \
    --difficulty Easy \
    --seed "$SEED" \
    --device cuda:0 \
    --amp=true \
    --foveate=true \
    --persistent-actors=true \
    --recurrent-policy=true \
    --max-episode-ticks "$MAX_TICKS" \
    --record "$OUT" \
    --debug true \
    >>"$LOG" 2>&1
  echo "EXIT:$? wrote $OUT ($(date -u +%H:%M:%S))" | tee -a "$LOG"
  # surface outcome
  python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
dbg=Path("showcase-clips/ppo_v11_u676_${TAG}.debug.json")
if dbg.exists():
    d=json.loads(dbg.read_text())
    print(f"outcome={d.get('outcome')} end_tick={d.get('end_tick')}")
PY
  tail -20 "$LOG"
}

# Current train stage (~13): 26 bots / 2 nations. Future denser lobbies use the
# first Easy curriculum rows that actually have 4n / 8n (policy is still s13 —
# those are harder / OOD, but density-matched).
run_watch 13 2 26 n2_europe Europe watch_n2_a
run_watch 20 4 34 n4_pangaea Pangaea watch_n4_a
run_watch 26 8 66 n8_northamerica NorthAmerica watch_n8_a
echo ALL_WATCHES_DONE
ls -lh showcase-clips/ppo_v11_u676_n{2_europe,4_pangaea,8_northamerica}.{json,debug.json} 2>/dev/null || true
