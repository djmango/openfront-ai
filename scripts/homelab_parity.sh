#!/usr/bin/env bash
# Run native 78-record parity gate on homelab (or any remote host).
set -euo pipefail

OPENFRONT_WORKTREE="${OPENFRONT_WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
# Prefer this worktree's own openfront submodule + records symlink; never the webbot checkout.
OPENFRONT_REPO="${OPENFRONT_REPO:-$OPENFRONT_WORKTREE}"
PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
LOG_DIR="${LOG_DIR:-$OPENFRONT_WORKTREE/logs}"
mkdir -p "$LOG_DIR"

export OPENFRONT_REPO
export PARITY_COMMIT

cd "$OPENFRONT_WORKTREE/rust"
echo "[homelab_parity] repo=$OPENFRONT_REPO commit=$PARITY_COMMIT"
if [[ ! -d "$OPENFRONT_REPO/openfront/src" ]]; then
  echo "[homelab_parity] missing $OPENFRONT_REPO/openfront (init submodule / rsync openfront)" >&2
  exit 1
fi
echo "[homelab_parity] building release (nix gcc wrapper)..."
nix-shell -p gcc cargo rustc --run 'cargo build --release -p openfront-engine'

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/parity_${PARITY_COMMIT}_${STAMP}.log"
echo "[homelab_parity] running gate -> $LOG"
nix-shell -p gcc cargo rustc --run \
  'cargo test -p openfront-engine replay::tests::multi_record_parity_report --release -- --ignored --nocapture' \
  2>&1 | tee "$LOG"
echo "[homelab_parity] done"
