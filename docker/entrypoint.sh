#!/usr/bin/env bash
set -euo pipefail

cd /app

DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR
export PYTHONPATH=/app
export PATH="/app/.venv/bin:/app/openfront/node_modules/.bin:/app/rust/target/release:${PATH}"
export OFTRAIN_BIN="${OFTRAIN_BIN:-/app/rust/target/release/oftrain}"
# Showcase clip Chromium: SoftGL is reliable in this image; GPU WebGL still falls back.
export OF_FORCE_SWIFTSHADER="${OF_FORCE_SWIFTSHADER:-1}"
PY="/app/.venv/bin/python"

mkdir -p "${DATA_DIR}/records" "${DATA_DIR}/policy" "${DATA_DIR}/clips" "${DATA_DIR}/replay-spool"

export SKIP_BROWSER_OPEN=true
export ADMIN_BOT_API_KEY="${ADMIN_BOT_API_KEY:-WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION}"

echo "[entrypoint] starting OpenFront dev stack (needed when a visitor clicks Play)"
(
  cd openfront
  exec npm run dev:host
) &

ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:9000" >/dev/null 2>&1; then
    echo "[entrypoint] dev stack ready"
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" != "1" ]; then
  echo "[entrypoint] FATAL: OpenFront dev stack on :9000 never became ready" >&2
  exit 1
fi

OFSHOWCASE="${OFSHOWCASE:-/app/rust/target/release/ofshowcase}"

echo "[entrypoint] replay archive API on :8987"
"$OFSHOWCASE" archive \
  --records "${DATA_DIR}/records" \
  --clips "${DATA_DIR}/clips" \
  --state "${DATA_DIR}/state.json" \
  --bind 0.0.0.0 \
  --port 8987 &

echo "[entrypoint] replay showcase daemon (RUN_NAME=${RUN_NAME:-ppo_v81})"
"$OFSHOWCASE" daemon &

echo "[entrypoint] showcase hub on :8988 (watch + on-demand play; featured map=latest)"
"$OFSHOWCASE" hub --port 8988 &

echo "[entrypoint] Caddy on :8086"
exec caddy run --config /app/docker/Caddyfile --adapter caddyfile
