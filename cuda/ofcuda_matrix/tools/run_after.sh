#!/usr/bin/env bash
set -u
M=/opt/data/workspaces/skg/ofcuda_matrix
BIN=/opt/data/workspaces/skg/.target-ofcuda-botai/release/ofcuda_matrix
cd "$M"
run() { # name env-extra map N nat ticks oracle-dir
  local name="$1" extra="$2" map="$3" N="$4" nat="$5" ticks="$6" od="$7"
  local d="$M/out/$name"
  rm -rf "$d"
  env $extra bash /opt/data/workspaces/skg/ofcuda_env.sh "$BIN" --map "$map" --agents "$N" --nations "$nat" \
    --ticks "$ticks" --out "$d" --oracle-dir "$od" --refs-dir "$M/out/matrix/refs" >"$d.log" 2>&1
  echo "=== $name ==="
  tail -1 "$d/cells.tsv" | cut -f1,4,9,10,11,12,17,18 - | sed 's/\t/ /g'
  grep -m1 "first divergence" "$d/${map}_n${N}_nat${nat}_t${ticks}/result.txt" || echo "first divergence: none"
  grep -m1 "note:" "$d/${map}_n${N}_nat${nat}_t${ticks}/result.txt"
}
run reg_t250  "" pangaea 488 0 250 "$M/out/matrix/oracle"
run sdb_f20  "OFCUDA_MATRIX_FREEZE_ATTACKS=20" pangaea 488 0 500 "$M/out/matrix/oracle"
run sdb_f60  "OFCUDA_MATRIX_FREEZE_ATTACKS=60" pangaea 488 0 500 "$M/out/matrix/oracle"