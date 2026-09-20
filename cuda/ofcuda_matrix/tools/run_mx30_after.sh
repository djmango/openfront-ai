#!/usr/bin/env bash
# Run the 30-cell 60-tick matrix against out/matrix/oracle and tally.
set -u
M=/opt/data/workspaces/skg/ofcuda_matrix
BIN=/opt/data/workspaces/skg/.target-ofcuda-botai/release/ofcuda_matrix
OUT=$M/out/mx30_after
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$M"
while read -r map N nat ticks; do
  [ -z "${map:-}" ] && continue
  d="$OUT/${map}_n${N}_nat${nat}_t${ticks}"
  bash /opt/data/workspaces/skg/ofcuda_env.sh "$BIN" --map "$map" --agents "$N" --nations "$nat" \
    --ticks "$ticks" --out "$d" --oracle-dir "$M/out/matrix/oracle" --refs-dir "$M/out/matrix/refs" \
    >"$d.log" 2>&1 || echo "RUNFAIL $map $N $nat $ticks"
  tail -1 "$d/cells.tsv" >> "$OUT/cells.tsv"
  echo "done $map $N $nat $ticks"
done < tools/cells_mx30_t60.txt