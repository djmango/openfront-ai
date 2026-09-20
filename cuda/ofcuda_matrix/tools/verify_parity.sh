#!/usr/bin/env bash
# Canonical parity verification. Pin a commit, build from a pristine worktree,
# run the standard arm set with the correct environment, print a fixed report.
set -u
S=/opt/data/workspaces/skg
M=$S/ofcuda_matrix
O=$M/out/matrix/oracle; RF=$M/out/matrix/refs
REF=${1:?usage: verify_parity.sh <commit> [ticks_for_long_arms]}
TICKS=${2:-434}
OUT=${OUT:-/tmp/parity-vb}; rm -rf $OUT; mkdir -p $OUT
export PATH=/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export CARGO_TARGET_DIR=$S/.target-parity-verify
cd /tmp/lgr-wt || { echo NO_WORKTREE; exit 1; }
git checkout -q --detach "$REF" || { echo PIN_FAIL; exit 1; }
echo "pinned $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
W=/tmp/lgr-wt/cuda/ofcuda_matrix
cd $W || exit 1
if nice -n 10 bash $S/ofcuda_env.sh cargo oxide build -- --release --bin ofcuda_matrix > $OUT/build.log 2>&1; then echo "BUILD OK ptx_fma=$(grep -c fma.rn.f64 $W/ofcuda_matrix.ptx)"; else echo BUILD_FAIL; tail -5 $OUT/build.log; exit 1; fi
BIN=$CARGO_TARGET_DIR/release/ofcuda_matrix
run(){ tag=$1; ticks=$2; shift 2; s=$SECONDS; env "$@" "$BIN" --map pangaea --agents 488 --nations 0 --ticks $ticks --oracle-dir $O --refs-dir $RF --out $OUT/$tag > $OUT/$tag.log 2>&1; echo "[$tag] $((SECONDS-s))s rc=$?"; }
run freeze20 $TICKS OFCUDA_MATRIX_FREEZE_ATTACKS=20
run freeze60 $TICKS OFCUDA_MATRIX_FREEZE_ATTACKS=60
run freeze20_full 500 OFCUDA_MATRIX_FREEZE_ATTACKS=20
run plain437 437
run t250 250
for t in freeze20 freeze60 plain437 t250; do
  f=$(ls $OUT/$t/pangaea_n488_nat0_*/result.txt 2>/dev/null | head -1)
  echo "=== $t"
  if [ -n "$f" ]; then grep -aE "hash matched|troops matched|first divergence|SELFDRIVE totals" "$f" | head -4; else echo "  VOID: no result file"; fi
done
echo "=== freeze20_full (a capacity refusal is expected here, not a divergence)"
grep -aE "CELL FAILED|MAX_SLOTS|more than 1024" $OUT/freeze20_full.log | head -2
s=$SECONDS; $BIN --cells-file $W/tools/cells_mx30_t60.txt --oracle-dir $O --refs-dir $RF --out $OUT/mx > $OUT/mx.log 2>&1; echo "[mx30] $((SECONDS-s))s"; grep -aE "cells: [0-9]+ passed" $OUT/mx.log
echo PARITYVERIFYDONE
