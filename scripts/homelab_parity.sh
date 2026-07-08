#!/usr/bin/env bash
# Run native 78-record parity gate on homelab (or any remote host).
set -euo pipefail

OPENFRONT_REPO="${OPENFRONT_REPO:-$HOME/openfront-ai}"
OPENFRONT_WORKTREE="${OPENFRONT_WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
LOG_DIR="${LOG_DIR:-$OPENFRONT_WORKTREE/logs}"
mkdir -p "$LOG_DIR"

export OPENFRONT_REPO
export PARITY_COMMIT

cd "$OPENFRONT_WORKTREE/rust"
echo "[homelab_parity] repo=$OPENFRONT_REPO commit=$PARITY_COMMIT"
echo "[homelab_parity] building release (nix gcc wrapper)..."
nix-shell -p gcc cargo rustc --run 'cargo build --release -p openfront-engine'

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/parity_${PARITY_COMMIT}_${STAMP}.log"
echo "[homelab_parity] running gate -> $LOG"
nix-shell -p gcc cargo rustc --run \
  'cargo test -p openfront-engine replay::tests::multi_record_parity_report --release -- --ignored --nocapture' \
  2>&1 | tee "$LOG"
echo "[homelab_parity] done"
