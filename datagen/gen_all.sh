#!/usr/bin/env bash
# Parallel data generation across maps. Usage: gen_all.sh <games-per-map> <parallelism>
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-25}"
PAR="${2:-10}"
MAPS=(Onion Pangaea World Asia Africa Australia Britannia BlackSea Caucasus BetweenTwoSeas)

for map in "${MAPS[@]}"; do
  echo "$map"
done | xargs -P "$PAR" -I {} \
  openfront/node_modules/.bin/tsx datagen/generate.ts \
    --map {} --games "$GAMES" --seed 1 2>&1 | grep -E "done|Error|error" || true

echo "all maps complete"
