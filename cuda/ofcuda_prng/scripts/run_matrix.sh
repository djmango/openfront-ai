#!/usr/bin/env bash
# Per-N spawn / initial-state proof for the CUDA port (ofcuda_prng).
#
# Ground truth ALWAYS comes from the real engine: `spawnall --agents N --nations S`
# invokes the ofcuda_spawn oracle, which drives `openfront-engine` itself. The
# port is then checked against that reference, never against itself.
#
# Usage:
#   run_matrix.sh [--gpu] [--diag] N [N ...]        # nations = 0
#   NATIONS=1 run_matrix.sh N ...                   # nations variant
set -u
PRNG=/opt/data/workspaces/skg/ofcuda_prng
WRAP=/opt/data/workspaces/skg/ofcuda_env.sh
BIN=$PRNG/target/release/spawnall
FLAGS="--no-gpu"
if [ "${1:-}" = "--gpu" ]; then FLAGS=""; shift; fi
if [ "${1:-}" = "--diag" ]; then FLAGS="$FLAGS --diag"; shift; fi
SPEC="${NATIONS:-0}"

for N in "$@"; do
  out=$(bash "$WRAP" "$BIN" --agents "$N" --nations "$SPEC" $FLAGS 2>&1)
  agg=$(echo "$out" | grep -E "^(ids|spawned|tiles|owned-set|counts|gpu_vs)" | tr '\n' ' ')
  star=$(echo "$out" | grep -E "^(land_unowned|->)" | tr '\n' ' ')
  printf 'N=%-5s nations=%-9s %s %s\n' "$N" "$SPEC" "$agg" "$star"
done
