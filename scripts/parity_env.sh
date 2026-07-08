#!/usr/bin/env bash
# Source this (or rely on script defaults) so parity never touches the webbot openfront checkout.
# Usage: source scripts/parity_env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENFRONT_REPO="${OPENFRONT_REPO:-$ROOT}"
export OPENFRONT_WORKTREE="${OPENFRONT_WORKTREE:-$ROOT}"
export OPENFRONT_ENGINE_DIR="${OPENFRONT_ENGINE_DIR:-$ROOT/openfront}"
export PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
echo "[parity_env] OPENFRONT_REPO=$OPENFRONT_REPO"
echo "[parity_env] OPENFRONT_ENGINE_DIR=$OPENFRONT_ENGINE_DIR"
echo "[parity_env] PARITY_COMMIT=$PARITY_COMMIT"
