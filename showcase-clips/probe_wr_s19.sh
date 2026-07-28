#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PATH="/workspace/rust/.libtorch-venv/bin:$PATH"
TORCH_ROOT="/workspace/rust/.libtorch-venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${TORCH_ROOT}/torch/lib:${TORCH_ROOT}/nvidia/cu13/lib:${TORCH_ROOT}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
OFTRAIN=./rust/target/release/oftrain
POLICY=rust/checkpoints/ppo_v11/latest.safetensors
AE=weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors
COARSE=weights/ae/ae_v32_nostatic_d16c32.encoder.safetensors
MAX_TICKS=21000
MAX_STEPS=2164
UPD=$(python3 -c "import json;print(json.load(open('rust/checkpoints/ppo_v11/latest.state.json'))['update'])")
STAGE=19
BOTS=30
N=4
DIFF=Easy
OUTDIR=showcase-clips/wr_probe_u${UPD}_s${STAGE}
mkdir -p "$OUTDIR"
MAPS=(Europe World Asia Pangaea Britannia Africa GreatLakes NorthAmerica)
echo "PROBE update=$UPD stage=$STAGE ${BOTS}b/${N}n train_WR_window=~0.425"
i=0
for MAP in "${MAPS[@]}"; do
  i=$((i+1))
  SEED="probe_${MAP}_${i}"
  TAG="${i}_${MAP}"
  LOG="$OUTDIR/${TAG}.watch.log"
  echo "=== $TAG $(date -u +%H:%M:%S) ===" | tee "$LOG"
  "$OFTRAIN" --watch --watch-stochastic=true --engine node \
    --policy "$POLICY" --ckpt "$AE" --coarse-ckpt "$COARSE" \
    --stage "$STAGE" --map "$MAP" --bots "$BOTS" --nations "$N" --difficulty "$DIFF" \
    --seed "$SEED" --device cuda:0 --amp=true --foveate=true \
    --persistent-actors=true --recurrent-policy=true \
    --max-episode-ticks "$MAX_TICKS" --max-steps "$MAX_STEPS" \
    --record "$OUTDIR/${TAG}.json" --debug true >>"$LOG" 2>&1
  python3 -c "import json;from pathlib import Path;d=json.loads(Path('${OUTDIR}/${TAG}.debug.json').read_text());print('outcome=%s end=%s dt=%s'%(d.get('outcome'),d.get('end_tick'),d.get('decision_ticks')))" | tee -a "$LOG"
done
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
outdir = sorted(Path("showcase-clips").glob("wr_probe_u*_s19"))[-1]
outs = []
for p in sorted(outdir.glob("*.debug.json")):
    d = json.load(open(p))
    outs.append(d.get("outcome"))
    print(p.name, d.get("outcome"), d.get("end_tick"), "dt", d.get("decision_ticks"))
c = Counter(outs)
n = len(outs)
wins = c.get("win", 0)
print("SUMMARY", dict(c), f"WR={wins}/{n}={wins/max(n,1):.3f} (train window ~0.425)")
(outdir / "summary.txt").write_text(f"{dict(c)} WR={wins}/{n}\n")
PY
echo PROBE_DONE
