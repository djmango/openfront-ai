#!/usr/bin/env bash
# Run ON the homelab host (not your laptop).
# Rebuilds the showcase image from master, restarts with GPU, and kicks clip generation.
set -euo pipefail

SRC=/var/lib/openfront-eval/src
DATA=/var/lib/openfront-eval/data

echo "[1/5] sync openfront-ai master"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch origin master
  git -C "$SRC" reset --hard origin/master
  git -C "$SRC" submodule sync --recursive
  git -C "$SRC" submodule update --init --recursive
  OPENFRONT_PIN="$(tr -d '[:space:]' < "$SRC/openfront.commit")"
  git -C "$SRC/openfront" fetch origin "$OPENFRONT_PIN" 2>/dev/null || true
  git -C "$SRC/openfront" checkout --force "$OPENFRONT_PIN"
else
  git clone --recurse-submodules --branch master \
    https://github.com/djmango/openfront-ai.git "$SRC"
fi

echo "[2/5] build image (playwright + ffmpeg + CUDA torch)"
export DOCKER_BUILDKIT=1
docker build -t openfront-eval:local -f "$SRC/docker/Dockerfile" "$SRC"

echo "[3/5] restart container with GPU (CDI device, not --gpus)"
docker rm -f openfront-eval 2>/dev/null || true
docker run -d \
  --name openfront-eval \
  --restart unless-stopped \
  --device nvidia.com/gpu=all \
  -p 127.0.0.1:8086:8086 \
  -p "[::1]:8086:8086" \
  -v "$DATA:/data" \
  -e RUN_NAME="${RUN_NAME:-ppo_v81}" \
  -e STAGE="${STAGE:-4}" \
  -e SHOWCASE_WATCH_STAGE="${SHOWCASE_WATCH_STAGE:-4}" \
  -e SHOWCASE_MAPS="${SHOWCASE_MAPS:-Onion,Pangaea,Caucasus,BlackSea,BetweenTwoSeas,World,Asia}" \
  -e REFRESH_HOURS="${REFRESH_HOURS:-1}" \
  -e LIVE_SHOWCASE="${LIVE_SHOWCASE:-0}" \
  -e CLIP_MAX_SEC="${CLIP_MAX_SEC:-90}" \
  -e AE_CKPT="${AE_CKPT:-runs/ae_v31_d8c32/ae_v3.pt}" \
  -e PLAY_MAP="${PLAY_MAP:-Onion}" \
  -e PLAY_BOTS="${PLAY_BOTS:-10}" \
  -e PLAY_NATIONS="${PLAY_NATIONS:-1}" \
  -e PLAY_START_DELAY="${PLAY_START_DELAY:-15}" \
  -e ADMIN_BOT_API_KEY="${ADMIN_BOT_API_KEY:-WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION}" \
  --memory 8g --cpus 6 \
  openfront-eval:local

echo "[4/5] wait for health"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8086/status >/dev/null 2>&1; then
    echo "showcase up"
    break
  fi
  sleep 5
done

echo "[5/5] trigger clip backfill (if clips dir empty)"
docker exec openfront-eval mkdir -p /data/clips
docker exec openfront-eval python3 -c "
import json
from pathlib import Path
p = Path('/data/state.json')
s = json.loads(p.read_text()) if p.exists() else {}
s['hero_clips'] = []
p.write_text(json.dumps(s, indent=2) + '\n')
"
docker exec openfront-eval pkill -f eval_daemon.py || true
sleep 2
# entrypoint will not restart eval_daemon after pkill; start manually
docker exec -d openfront-eval /app/.venv/bin/python scripts/eval_daemon.py

echo "done. check: curl -s http://127.0.0.1:8086/status | python3 -m json.tool"
echo "clips: ls -la $DATA/clips/"
