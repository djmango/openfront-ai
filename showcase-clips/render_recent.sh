#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PATH=/tmp/bin-stubs:$PATH
export BROWSER=none SKIP_BROWSER_OPEN=true PYTHONUNBUFFERED=1
export OF_CLIENT_COMMIT=f73501ae71a02c39e66e14f3e580ecaf95f76502
export OF_FORCE_GPU=1 OF_REFUSE_SOFTGL=1
export DISPLAY=${DISPLAY:-:99}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
pgrep -x Xvfb >/dev/null || Xvfb :99 -screen 0 1920x1080x24 >/tmp/of-xvfb.log 2>&1 &
sleep 1

PREFIX=ppo_v11_u806
for TAG in s16_europe s16_world s16_asia; do
  REC="showcase-clips/${PREFIX}_${TAG}.json"
  OUT="/opt/cursor/artifacts/${PREFIX}_${TAG}.webm"
  LOG="showcase-clips/${PREFIX}_${TAG}.render.log"
  echo "=== RENDER $TAG ===" | tee "$LOG"
  uv run --no-project --with playwright python scripts/render_client_replay.py \
    --record "$REC" --out "$OUT" \
    --speed max --timeout 2400 --trim-gameplay --width 1280 --height 720 --device-scale-factor 1 \
    2>&1 | tee -a "$LOG"
  echo "RENDER_EC_${TAG}:${PIPESTATUS[0]}" | tee -a "$LOG"
  ls -lh "$OUT" | tee -a "$LOG"
done
echo ALL_RECENT_RENDERS_DONE
