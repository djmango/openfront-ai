#!/usr/bin/env bash
# boundary-413 fix verification: freeze-20, freeze-60, plain t437.
# NOTE: the result path is derived from the ticks actually passed - a hardcoded
# `t500` path here previously reported VOID for the t437 arm even though its
# result.txt existed (absent output is not evidence of absence).
set -u
cd /opt/data/workspaces/skg/ofcuda_matrix
ENV=/opt/data/workspaces/skg/ofcuda_env.sh
BIN=${BIN:-/opt/data/workspaces/skg/.target-ofcuda-b413/release/ofcuda_matrix}
ORC=out/matrix/oracle
REF=out/matrix/refs

run() {
  local name="$1" ticks="$2"; shift 2
  local dir="out/fix413_${name}"
  rm -rf "$dir"; mkdir -p "$dir"
  local t0=$SECONDS
  "$@" > "$dir/run.log" 2>&1
  local rc=$? wall=$((SECONDS - t0))
  local res="$dir/pangaea_n488_nat0_t${ticks}/result.txt"
  echo "=== $name rc=$rc wall=${wall}s"
  if [ ! -f "$res" ]; then
    echo "VOID: expected result file $res does not exist (arm exited in ${wall}s)"
    find "$dir" -name result.txt | head -5
  else
    echo "result: $res"
  fi
}

run b20_t434 434 env OFCUDA_MATRIX_FREEZE_ATTACKS=20 bash "$ENV" "$BIN" --map pangaea --agents 488 --nations 0 --ticks 434 --out out/fix413_b20_t434 --oracle-dir "$ORC" --refs-dir "$REF"
run b60_t434 434 env OFCUDA_MATRIX_FREEZE_ATTACKS=60 bash "$ENV" "$BIN" --map pangaea --agents 488 --nations 0 --ticks 434 --out out/fix413_b60_t434 --oracle-dir "$ORC" --refs-dir "$REF"
run p437 437 bash "$ENV" "$BIN" --map pangaea --agents 488 --nations 0 --ticks 437 --out out/fix413_p437 --oracle-dir "$ORC" --refs-dir "$REF"
echo ALLDONE
