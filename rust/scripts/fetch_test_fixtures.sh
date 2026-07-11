#!/usr/bin/env bash
# Fetches the exact archived-game fixtures rust/engine's #[cfg(test)] suite
# needs (records/0c4c7d7993c9/<gameID>.json.gz) directly from the public
# OpenFront API, by game ID. `records/` is gitignored (see .gitignore) -
# these fixtures are meant to be fetched locally by each dev/CI run, not
# committed - this script is the one-command way to do that instead of
# reverse-engineering the exact IDs out of `cargo test`'s failure list.
#
# `scripts/fetch_games.py` (repo root) fetches games by time window instead
# and is the right tool for building new datasets; this script exists only
# because these specific 14 IDs are hardcoded fixture references inside
# rust/engine/src/{replay,bootstrap}.rs and execution/spawn_util.rs.
#
# Usage (from openfront-ai/rust/):
#   bash scripts/fetch_test_fixtures.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/records/0c4c7d7993c9"
API="https://api.openfront.io/public/game"

# Every game ID referenced by a `records/0c4c7d7993c9/<ID>.json.gz` fixture
# path in rust/engine/src - keep in sync if a new test adds a new ID (`rg
# -o '[A-Za-z0-9]{8}(?=\.json\.gz)' rust/engine/src` from the repo root lists
# every ID currently referenced).
GAME_IDS=(
  1MFxEdwr 2dG9dxmX 3QNU4eJa 6k4SnLrH 7MVmc1cR GiQovEcP HxWdr5PK
  MdPDuVXZ fdh3gYAF fkVh9QtC jby2gMJF rN7wbZ1Y tCFq6nPn x7pvCXU3
)

mkdir -p "$OUT_DIR"
for gid in "${GAME_IDS[@]}"; do
  dest="$OUT_DIR/${gid}.json.gz"
  if [[ -f "$dest" ]]; then
    echo "[$gid] already present, skipping"
    continue
  fi
  echo "[$gid] fetching..."
  tmp="$(mktemp)"
  if ! curl -sf -m 30 "$API/$gid" -o "$tmp"; then
    echo "[$gid] FAILED to fetch" >&2
    rm -f "$tmp"
    continue
  fi
  gzip -c "$tmp" > "$dest"
  rm -f "$tmp"
  echo "[$gid] saved -> $dest"
  sleep 0.3
done

echo "done: $(ls "$OUT_DIR" | wc -l | tr -d ' ') fixture(s) in $OUT_DIR"
