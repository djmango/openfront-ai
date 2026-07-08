#!/usr/bin/env bash
# Ensure the dedicated openfront submodule is checked out at PARITY_COMMIT.
# Never touches /Users/djmango/github/openfront-ai/openfront (webbot).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
ENGINE="$ROOT/openfront"

if [[ ! -d "$ENGINE/.git" && ! -f "$ENGINE/.git" ]]; then
  echo "[ensure_parity_openfront] initializing openfront submodule..."
  git -C "$ROOT" submodule update --init openfront
fi

cd "$ENGINE"
cur="$(git rev-parse --short=12 HEAD 2>/dev/null || true)"
want="$(git rev-parse --short=12 "${PARITY_COMMIT}^{commit}" 2>/dev/null || true)"
if [[ -z "$want" ]]; then
  echo "[ensure_parity_openfront] fetching ${PARITY_COMMIT}..."
  git remote get-url fork >/dev/null 2>&1 \
    || git remote add fork https://github.com/djmango/OpenFrontIO.git
  git fetch --depth 1 fork "$PARITY_COMMIT" 2>/dev/null \
    || git fetch --depth 1 origin "$PARITY_COMMIT" 2>/dev/null \
    || git fetch fork "$PARITY_COMMIT" \
    || git fetch origin "$PARITY_COMMIT"
  want="$(git rev-parse --short=12 "${PARITY_COMMIT}^{commit}")"
fi

if [[ "$cur" != "$want" ]]; then
  echo "[ensure_parity_openfront] checkout $want (was ${cur:-none})"
  git checkout -q "$want"
else
  echo "[ensure_parity_openfront] already at $want"
fi

if [[ ! -d node_modules/.bin ]]; then
  echo "[ensure_parity_openfront] npm install (parity openfront)..."
  npm install --ignore-scripts
fi
