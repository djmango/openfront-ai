#!/usr/bin/env bash
# Third-opinion sweep: dims from jq (not from any Rust code), hash + land from
# the C program. Emits the same TSV as `ofcuda_map --all --size all` so the two
# outputs diff directly.
set -uo pipefail
MAPS=/opt/data/workspaces/skg/openfront-ai/openfront/resources/maps
BIN=/opt/data/workspaces/skg/ofcuda_map/third_opinion
for dir in "$MAPS"/*/; do
  name=$(basename "$dir")
  for pair in "normal:map:map.bin" "4x:map4x:map4x.bin" "16x:map16x:map16x.bin"; do
    size=${pair%%:*}; rest=${pair#*:}; key=${rest%%:*}; file=${rest#*:}
    w=$(jq -r ".$key.width" "$dir/manifest.json")
    h=$(jq -r ".$key.height" "$dir/manifest.json")
    ml=$(jq -r ".$key.num_land_tiles" "$dir/manifest.json")
    out=$("$BIN" "$dir$file" "$w" "$h") || { printf '%s\t%s\tERROR\t0\t0\t0\tERR:c-corpus\n' "$name" "$size"; continue; }
    dims=$(printf '%s' "$out" | cut -f1)
    land=$(printf '%s' "$out" | cut -f2)
    hash=$(printf '%s' "$out" | cut -f3)
    printf '%s\t%s\t%s\t%s\t%s\t%s\tOK\n' "$name" "$size" "$dims" "$land" "$ml" "$hash"
  done
done
