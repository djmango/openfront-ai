#!/usr/bin/env bash
# Progress ladder watches against the latest ppo_v11 checkpoint.
# Tick budget MUST match training (`ofcore::DEFAULT_MAX_EPISODE_TICKS` /
# pod_train_v11 `--max-episode-ticks 21000`).
set -euo pipefail
cd /workspace
export PATH="/workspace/rust/.libtorch-venv/bin:$PATH"
TORCH_ROOT="/workspace/rust/.libtorch-venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${TORCH_ROOT}/torch/lib:${TORCH_ROOT}/nvidia/cu13/lib:${TORCH_ROOT}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p showcase-clips /workspace/artifacts

POLICY=rust/checkpoints/ppo_v11/latest.safetensors
STATE=rust/checkpoints/ppo_v11/latest.state.json
AE=weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors
COARSE=weights/ae/ae_v32_nostatic_d16c32.encoder.safetensors
OFTRAIN=./rust/target/release/oftrain

# Single source of truth with training.
MAX_TICKS=21000
# decision every 15 ticks + headroom (ofcore::watch_max_steps_for_ticks)
MAX_STEPS=$((MAX_TICKS / 15 + 64))

read_stage_lobby() {
  # prints: bots nations difficulty
  local STAGE=$1
  python3 - <<PY
import re
from pathlib import Path
text = Path("rust/ofcore/src/curriculum.rs").read_text()
pairs = [(int(a), int(b)) for a, b in re.findall(
    r"\((\d+),\s*(\d+)\)",
    re.search(r"pub const V10_BOT_NATION_DENSITY:.*?=\s*\[(.*?)\];", text, re.S).group(1),
)]
med = int(re.search(r"V10_MEDIUM_START:\s*usize\s*=\s*(\d+)", text).group(1))
hard = int(re.search(r"V10_HARD_START:\s*usize\s*=\s*(\d+)", text).group(1))
s = int("$STAGE")
bots, n = pairs[s]
diff = "Easy" if s < med else ("Medium" if s < hard else "Hard")
print(bots, n, diff)
PY
}

UPD=$(python3 -c "import json;print(json.load(open('$STATE'))['update'])")
CUR=$(python3 -c "import json;print(json.load(open('$STATE'))['stage'])")
TAG_PREFIX="ppo_v11_u${UPD}"

echo "CHECKPOINT update=$UPD stage=$CUR"
echo "TRAIN_MATCHED max-episode-ticks=$MAX_TICKS max-steps=$MAX_STEPS (must equal pod --max-episode-ticks 21000)"

run_watch() {
  local STAGE=$1 TAG=$2 MAP=$3 SEED=$4
  local BOTS N DIFF
  read -r BOTS N DIFF <<<"$(read_stage_lobby "$STAGE")"
  local OUT="showcase-clips/${TAG_PREFIX}_${TAG}.json"
  local LOG="showcase-clips/${TAG_PREFIX}_${TAG}.watch.log"
  echo "=== WATCH stage=$STAGE n=$N bots=$BOTS diff=$DIFF map=$MAP seed=$SEED ticks=$MAX_TICKS ($TAG) $(date -u +%H:%M:%S) ===" | tee "$LOG"
  "$OFTRAIN" \
    --watch \
    --watch-stochastic=true \
    --engine node \
    --policy "$POLICY" \
    --ckpt "$AE" \
    --coarse-ckpt "$COARSE" \
    --stage "$STAGE" \
    --map "$MAP" \
    --bots "$BOTS" \
    --nations "$N" \
    --difficulty "$DIFF" \
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
  python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
dbg=Path("showcase-clips/${TAG_PREFIX}_${TAG}.debug.json")
d=json.loads(dbg.read_text()) if dbg.exists() else {}
print(f"outcome={d.get('outcome')} end_tick={d.get('end_tick')} (train_budget={$MAX_TICKS})")
assert d.get("end_tick") is None or int(d["end_tick"]) <= $MAX_TICKS + 2, d
# timeout must be at the train budget, not an earlier silent cap
if d.get("outcome") == "timeout":
    assert int(d["end_tick"]) >= $MAX_TICKS, (d, "timeout before train budget - tick mismatch")
PY
  tail -15 "$LOG"
}

# current / next / +10 / first Medium (V10_MEDIUM_START=36)
run_watch "$CUR" "s${CUR}_current" Europe "prog_cur_a"
run_watch $((CUR + 1)) "s$((CUR + 1))_next" World "prog_next_a"
run_watch $((CUR + 10)) "s$((CUR + 10))_plus10" Asia "prog_p10_a"
run_watch 36 "s36_medium" Pangaea "prog_med_a"

echo ALL_PROGRESS_WATCHES_DONE
ls -lh showcase-clips/${TAG_PREFIX}_s*.{json,debug.json} 2>/dev/null || true
