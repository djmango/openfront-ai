#!/usr/bin/env bash
# Sync rust-fast to homelab and run the 78-record parity gate.
# Homelab default shell is fish; remote work always runs under bash.
set -euo pipefail

HOST="${HOMELAB_HOST:-skg@homelab}"
REMOTE_DIR="${HOMELAB_DIR:-openfront-ai-rust-fast}"
PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
SYNC="${HOMELAB_SYNC:-1}"
PULL="${HOMELAB_PULL:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$SYNC" == "1" ]]; then
  echo "[run_homelab_parity] rsync $ROOT -> $HOST:~/$REMOTE_DIR"
  rsync -az --delete \
    --exclude target \
    --exclude '.git' \
    --exclude 'logs/*.log' \
    "$ROOT/" "$HOST:~/$REMOTE_DIR/"
elif [[ "$PULL" == "1" ]]; then
  echo "[run_homelab_parity] git pull on $HOST:~/$REMOTE_DIR"
  ssh "$HOST" bash -lc "cd ~/$REMOTE_DIR && git fetch origin && git pull --ff-only"
fi

REMOTE_CMD="cd ~/$REMOTE_DIR && OPENFRONT_REPO=~/openfront-ai PARITY_COMMIT=${PARITY_COMMIT} bash scripts/homelab_parity.sh"
echo "[run_homelab_parity] host=$HOST commit=$PARITY_COMMIT"
ssh "$HOST" bash -lc "$REMOTE_CMD"
