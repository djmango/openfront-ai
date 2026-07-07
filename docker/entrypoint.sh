#!/usr/bin/env bash
set -euo pipefail

cd /app

DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR
export PYTHONPATH=/app
export PATH="/app/.venv/bin:/app/openfront/node_modules/.bin:${PATH}"

mkdir -p "${DATA_DIR}/records" "${DATA_DIR}/policy"

export SKIP_BROWSER_OPEN=true
export ADMIN_BOT_API_KEY="${ADMIN_BOT_API_KEY:-WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION}"

echo "[entrypoint] starting OpenFront dev stack (needed when a visitor clicks Play)"
(
  cd openfront
  exec npm run dev:host
) &

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:9000" >/dev/null 2>&1; then
    echo "[entrypoint] dev stack ready"
    break
  fi
  sleep 1
done

echo "[entrypoint] replay archive API on :8987"
python scripts/serve_replay.py \
  --records "${DATA_DIR}/records" \
  --state "${DATA_DIR}/state.json" \
  --bind 0.0.0.0 \
  --port 8987 &

echo "[entrypoint] replay showcase daemon (RUN_NAME=${RUN_NAME:-ppo_v4})"
python scripts/eval_daemon.py &

echo "[entrypoint] showcase hub on :8988 (watch + on-demand play)"
python scripts/showcase_hub.py &

echo "[entrypoint] Caddy on :8086"
exec caddy run --config /app/docker/Caddyfile --adapter caddyfile
