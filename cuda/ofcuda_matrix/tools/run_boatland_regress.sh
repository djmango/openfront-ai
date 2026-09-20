#!/usr/bin/env bash
# Regression suite for the boat-landing player-target change.
set -u
M=/opt/data/workspaces/skg/ofcuda_matrix
BIN=/opt/data/workspaces/skg/.target-ofcuda-boatland/release/ofcuda_matrix
cd "$M"
echo "START $(date -Is)"

# --- mx30: 30-cell 60-tick matrix ---
OUT=$M/out/blreg_mx30
rm -rf "$OUT"; mkdir -p "$OUT"
while read -r map N nat ticks; do
  [ -z "${map:-}" ] && continue
  d="$OUT/${map}_n${N}_nat${nat}_t${ticks}"
  bash /opt/data/workspaces/skg/ofcuda_env.sh "$BIN" --map "$map" --agents "$N" --nations "$nat" \
    --ticks "$ticks" --out "$d" --oracle-dir "$M/out/matrix/oracle" --refs-dir "$M/out/matrix/refs" \
    >"$d.log" 2>&1 || echo "RUNFAIL $map $N $nat $ticks"
done < tools/cells_mx30_t60.txt
echo "MX30 cells_txt_lines=$(grep -vc '^\s*$' tools/cells_mx30_t60.txt)"
echo "MX30 matched=$(grep -l 'hash matched 60/60' "$OUT"/*/*/result.txt 2>/dev/null | wc -l)"
echo "MX30 results:"
for r in "$OUT"/*/*/result.txt; do
  [ -f "$r" ] || continue
  printf '%s :: %s :: %s\n' "$(basename "$(dirname "$(dirname "$r")")")" "$(grep -m1 'hash matched' "$r")" "$(grep -m1 'first divergence' "$r")"
done

# --- t250 ---
d=$M/out/blreg_t250; rm -rf "$d"
env bash /opt/data/workspaces/skg/ofcuda_env.sh "$BIN" --map pangaea --agents 488 --nations 0 \
  --ticks 250 --out "$d" --oracle-dir "$M/out/matrix/oracle" --refs-dir "$M/out/matrix/refs" >"$d.log" 2>&1
echo "=== t250 ==="; grep -E "hash matched|first divergence|troops matched|SELFDRIVE totals" "$d/pangaea_n488_nat0_t250/result.txt" | head

# --- t500 cancel model ON (default) ---
d=$M/out/blreg_t500on; rm -rf "$d"
env bash /opt/data/workspaces/skg/ofcuda_env.sh "$BIN" --map pangaea --agents 488 --nations 0 \
  --ticks 500 --out "$d" --oracle-dir "$M/out/matrix/oracle" --refs-dir "$M/out/matrix/refs" >"$d.log" 2>&1
echo "=== t500 cancel ON ==="; grep -E "hash matched|first divergence|troops matched|SELFDRIVE totals" "$d/pangaea_n488_nat0_t500/result.txt" | head

echo "DONE $(date -Is)"