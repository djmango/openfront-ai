#!/usr/bin/env bash
# Interleaved T-sweep at fixed N: contention cancels across T, so the
# minima per T can be fitted as fixed + marginal.
set -u
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
export LD_LIBRARY_PATH=/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
BIN=/tmp/ofcuda_tick_check/release/gpu_env
RUN="nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh $BIN"
OUT=/tmp/ge_tsweep.txt
: > $OUT
for R in 1 2 3; do
  for T in 1 2 4 8 16; do
    $RUN --ticks $T --envs 1024 --mode fast 2>/dev/null \
      | grep -E "^N=1024 +mode=fast-1launch" \
      | sed "s/^/run=$R /" >> $OUT
  done
done
echo TSWEEPDONE
