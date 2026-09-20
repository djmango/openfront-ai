#!/usr/bin/env bash
# Regenerate the oracle dumps with the CURRENT oracle binary (PEXEC/CADENCE/FRIEND)
# and capture the engine's own cluster-removal stream (OF_ENG_CLUSTER=1) next to
# each dump as <dump>.clusters.
#
# usage: gen_oracles.sh <cells-file> <oracle-out-dir> [nations-override]
#   cells-file lines: <map> <agents> [nations] [ticks]
set -euo pipefail
PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
export PATH
HERE=/opt/data/workspaces/skg/ofcuda_matrix
ORACLE="$HERE/oracle/target/release/oracle"
CELLS="$1"
OUT="$2"
mkdir -p "$OUT"
n=0
while read -r map agents nations ticks; do
  case "$map" in ''|'#'*) continue;; esac
  [ -n "${nations:-}" ] || nations=0
  [ -n "${ticks:-}" ] || ticks=60
  dump="$OUT/${map}_n${agents}_nat${nations}_t${ticks}.dump"
  echo "=== oracle $map N=$agents nat=$nations t=$ticks"
  t0=$SECONDS
  OF_ENG_CLUSTER=1 nice -n 10 "$ORACLE" replay --maps "$map" --ns "$agents" \
      --nations "$nations" --ticks "$ticks" --seed parity --out-dir "$OUT" \
      > /dev/null 2> "$dump.stderr"
  grep '^ENG_CLUSTER_REMOVE' "$dump.stderr" > "$dump.clusters" || true
  echo "    $(wc -l < "$dump.clusters") cluster removals, $((SECONDS-t0))s"
  n=$((n+1))
done < "$CELLS"
echo "regenerated $n dumps into $OUT"
