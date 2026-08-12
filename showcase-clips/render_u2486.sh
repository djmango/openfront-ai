#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PATH=/tmp/bin-stubs:$PATH
export BROWSER=none SKIP_BROWSER_OPEN=true PYTHONUNBUFFERED=1
export OF_FORCE_GPU=1 OF_REFUSE_SOFTGL=1
export DISPLAY=${DISPLAY:-:99}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OFSHOWCASE=/workspace/rust/target/release/ofshowcase
mkdir -p /workspace/artifacts
# ensure Xvfb
if ! (ls /tmp/.X99-lock >/dev/null 2>&1); then
  Xvfb :99 -screen 0 1920x1080x24 >/tmp/of-xvfb.log 2>&1 &
  sleep 1
fi

PREFIX=ppo_v11_u2486
# Prefer win + long timeout + a death for honest sample; skip asia (short death)
for TAG in s23_pangaea s23_europe s23_world s36_medium_europe; do
  REC="showcase-clips/${PREFIX}_${TAG}.json"
  OUT="/workspace/artifacts/${PREFIX}_${TAG}.webm"
  LOG="showcase-clips/${PREFIX}_${TAG}.render.log"
  if [[ ! -f "$REC" ]]; then
    echo "MISSING $REC" | tee "$LOG"
    continue
  fi
  if [[ -f "$OUT" && -s "$OUT" ]]; then
    echo "SKIP existing $OUT" | tee "$LOG"
    continue
  fi
  echo "=== RENDER $TAG ===" | tee "$LOG"
  uv run --no-project --with playwright python scripts/render_client_replay.py \
    --record "$REC" --out "$OUT" \
    --speed max --timeout 2400 --trim-gameplay --width 1280 --height 720 --device-scale-factor 1 \
    2>&1 | tee -a "$LOG"
  ec=${PIPESTATUS[0]}
  echo "RENDER_EC_${TAG}:${ec}" | tee -a "$LOG"
  ls -lh "$OUT" 2>&1 | tee -a "$LOG" || true
done
echo ALL_U2486_RENDERS_DONE
