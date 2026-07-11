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
  # GitHub's smart-HTTP fetch only accepts a *full* 40-char SHA for an
  # arbitrary (non-tip) commit want - a short prefix is treated as a ref
  # name lookup and fails with "couldn't find remote ref" even though the
  # object is perfectly reachable. On a cold submodule cache (no local
  # object yet to expand the prefix from), resolve it to the full SHA via
  # GitHub's REST API first so the fetch has something it'll actually serve.
  if ! git fetch --depth 1 fork "$PARITY_COMMIT" 2>/dev/null \
      && ! git fetch --depth 1 origin "$PARITY_COMMIT" 2>/dev/null; then
    full_sha="$(curl -sf "https://api.github.com/repos/djmango/OpenFrontIO/commits/${PARITY_COMMIT}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || true)"
    if [[ -n "$full_sha" ]]; then
      git fetch --depth 1 fork "$full_sha" 2>/dev/null \
        || git fetch --depth 1 origin "$full_sha"
    else
      git fetch fork "$PARITY_COMMIT" || git fetch origin "$PARITY_COMMIT"
    fi
  fi
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
