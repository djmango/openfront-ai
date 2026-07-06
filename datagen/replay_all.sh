#!/usr/bin/env bash
# Replay all downloaded game records, bucketed by engine commit.
#
# Replays are only deterministic on the engine commit the game ran on, so for
# each records/<commit>/ bucket this checks out the openfront submodule at
# that commit, replays the bucket (hash-verified), then restores the pin.
#
# Usage: datagen/replay_all.sh [records_dir] [out_dir] [parallelism]
set -euo pipefail
cd "$(dirname "$0")/.."

RECORDS=${1:-records}
OUT=${2:-data-human}
JOBS=${3:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
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
  # already have meta.json, so reruns are incremental.
  ls "$bucket"*.json.gz | xargs -P "$JOBS" -n 4 sh -c '
    tmp=$(mktemp -d)
    for f in "$@"; do ln -s "$(realpath "$f")" "$tmp/"; done
    '"$TSX"' datagen/replay.ts --records "$tmp" --out '"$OUT"' 2>&1 | grep -Ev "not found|QuickChat|cannot build|Constructor"
    rm -rf "$tmp"
  ' sh
done
echo "all buckets done"
