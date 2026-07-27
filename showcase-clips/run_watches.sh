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

# Shared with train (ofcore::DEFAULT_MAX_EPISODE_TICKS / pod --max-episode-ticks).
MAX_TICKS=21000
MAX_STEPS=$((MAX_TICKS / 10 + 64))

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
    --max-steps "$MAX_STEPS" \
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
#
# Prefer full-horizon episodes (timeout/win at the 21000 train budget). Retries
# on death so we don't ship mid-game losses as "the demo".
run_until_full() {
  local STAGE=$1 N=$2 BOTS=$3 TAG_BASE=$4 MAP=$5 SEED_BASE=$6
  local ATTEMPT=1
  while (( ATTEMPT <= 6 )); do
    local TAG="${TAG_BASE}"
    local SEED="${SEED_BASE}${ATTEMPT}"
    # first attempt keeps the canonical tag for the map; retries get _rN
    if (( ATTEMPT > 1 )); then
      TAG="${TAG_BASE}_r${ATTEMPT}"
    fi
    run_watch "$STAGE" "$N" "$BOTS" "$TAG" "$MAP" "$SEED"
    local OUTCOME
    OUTCOME=$(python3 - <<PY
import json
from pathlib import Path
p=Path("showcase-clips/ppo_v11_u676_${TAG}.debug.json")
print(json.loads(p.read_text()).get("outcome","?") if p.exists() else "?")
PY
)
    if [[ "$OUTCOME" == "timeout" || "$OUTCOME" == "win" ]]; then
      echo "FULL_HORIZON tag=$TAG outcome=$OUTCOME"
      # promote retry to canonical tag if needed
      if (( ATTEMPT > 1 )); then
        for ext in json debug.json thinking.json watch.log; do
          src="showcase-clips/ppo_v11_u676_${TAG}.${ext}"
          dst="showcase-clips/ppo_v11_u676_${TAG_BASE}.${ext}"
          [[ -f "$src" ]] && cp -f "$src" "$dst"
        done
      fi
      return 0
    fi
    echo "RETRY tag=$TAG outcome=$OUTCOME (want timeout/win)"
    ATTEMPT=$((ATTEMPT + 1))
  done
  echo "WARN: no full-horizon episode for $TAG_BASE after retries" >&2
  return 1
}

run_until_full 13 2 26 n2_europe Europe watch_n2_
# World tends to reach the full 21000-tick horizon more often than Pangaea at
# this OOD 4n density; retries still cover deaths.
run_until_full 20 4 34 n4_world World watch_n4_
# Prefer maps whose GameMapType value == id (Europe) for 8n — NorthAmerica
# remaps for Zod and has shown client "You died" divergence on timeout records.
run_until_full 26 8 66 n8_europe Europe watch_n8_
echo ALL_WATCHES_DONE
ls -lh showcase-clips/ppo_v11_u676_n{2_europe,4_world,8_europe}.{json,debug.json} 2>/dev/null || true
