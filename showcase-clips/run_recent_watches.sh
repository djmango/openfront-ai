#!/usr/bin/env bash
# Recent games at the *live* curriculum lobby (bots/nations/difficulty from
# stage table — not hardcoded). Train-matched tick budget + decision_ticks.
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
# decision every 15 ticks + headroom (ofcore::watch_max_steps_for_ticks)
MAX_STEPS=$((MAX_TICKS / 15 + 64))

UPD=$(python3 -c "import json;print(json.load(open('$STATE'))['update'])")
STAGE=$(python3 -c "import json;print(json.load(open('$STATE'))['stage'])")
read -r BOTS N DIFF DT <<<"$(python3 - <<PY
import re
from pathlib import Path
text = Path("rust/ofcore/src/curriculum.rs").read_text()
pairs = [(int(a), int(b)) for a, b in re.findall(
    r"\((\d+),\s*(\d+)\)",
    re.search(r"pub const V10_BOT_NATION_DENSITY:.*?=\s*\[(.*?)\];", text, re.S).group(1),
)]
med = int(re.search(r"V10_MEDIUM_START:\s*usize\s*=\s*(\d+)", text).group(1))
hard = int(re.search(r"V10_HARD_START:\s*usize\s*=\s*(\d+)", text).group(1))
# decision_ticks: V10 is 15 for every stage
s = int("$STAGE")
bots, n = pairs[s]
diff = "Easy" if s < med else ("Medium" if s < hard else "Hard")
dt = 15
print(bots, n, diff, dt)
PY
)"
TAG_PREFIX="ppo_v11_u${UPD}"

echo "CHECKPOINT update=$UPD stage=$STAGE lobby=${BOTS}b/${N}n $DIFF decision_ticks=$DT ticks=$MAX_TICKS"

run_watch() {
  local TAG=$1 MAP=$2 SEED=$3
  local OUT="showcase-clips/${TAG_PREFIX}_${TAG}.json"
  local LOG="showcase-clips/${TAG_PREFIX}_${TAG}.watch.log"
  echo "=== WATCH $TAG map=$MAP seed=$SEED $(date -u +%H:%M:%S) ===" | tee "$LOG"
  "$OFTRAIN" --watch --watch-stochastic=true --engine node \
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
print(f"outcome={d.get('outcome')} end_tick={d.get('end_tick')} decision_ticks={d.get('decision_ticks')}")
PY
}

# Leave map unset for true train sampling? Keep a few maps for variety, but
# lobby always matches curriculum for STAGE.
run_watch "s${STAGE}_europe" Europe "diag_eu_a"
run_watch "s${STAGE}_world" World "diag_world_a"
run_watch "s${STAGE}_asia" Asia "diag_asia_a"
run_watch "s${STAGE}_pangaea" Pangaea "diag_pang_a"
echo ALL_RECENT_WATCHES_DONE
