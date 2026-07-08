#!/usr/bin/env bash
# Replay all downloaded game records, bucketed by engine commit.
#
# Replays are only deterministic on the engine commit the game ran on, so for
# each records/<commit>/ bucket this checks out the openfront submodule at
# that commit, replays the bucket (hash-verified), then restores the pin.
#
# Usage: datagen/replay_all.sh [records_dir] [out_dir] [parallelism]
# BC=1 passes --bc (dump behavior-cloning sidecars; incremental over
# already-replayed games). REBC=1 additionally passes --rebc to regenerate
# existing sidecars (e.g. formatVersion 2 -> 3 upgrade for v6 action labels).
set -euo pipefail
cd "$(dirname "$0")/.."

RECORDS=${1:-records}
OUT=${2:-data-human}
JOBS=${3:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
BCFLAG="${BC:+--bc}${REBC:+ --rebc}"
TSX=openfront/node_modules/.bin/tsx

PIN=$(git -C openfront rev-parse HEAD)
restore() { git -C openfront checkout -q "$PIN"; }
trap restore EXIT

for bucket in "$RECORDS"/*/; do
  commit=$(basename "$bucket")
  [ "$commit" = "unknown" ] && { echo "skipping unknown-commit bucket"; continue; }
  n=$(ls "$bucket" | wc -l)
  echo "=== bucket $commit: $n records ==="
  if ! git -C openfront checkout -q "$commit" 2>/dev/null; then
    git -C openfront fetch origin --quiet || true
    git -C openfront checkout -q "$commit" || { echo "cannot checkout $commit, skipping"; continue; }
  fi
  # Shard the bucket across parallel workers; replay.ts skips games that
  # already have meta.json (and, with --rebc, an up-to-date bc.json.gz), so
  # reruns are incremental. One game per worker: a single replay runs for
  # many minutes, so batching (-n 4) serialized long games behind each other
  # and left the box mostly idle on bucket tails.
  ls "$bucket"*.json.gz | xargs -P "$JOBS" -n 1 sh -c '
    tmp=$(mktemp -d)
    for f in "$@"; do ln -s "$(realpath "$f")" "$tmp/"; done
    nice -n 15 '"$TSX"' datagen/replay.ts --records "$tmp" --out '"$OUT"' '"$BCFLAG"' 2>&1 | grep -Ev "not found|QuickChat|cannot build|cannot send|Constructor|Failed to find"
    rm -rf "$tmp"
  ' sh
done
echo "all buckets done"
