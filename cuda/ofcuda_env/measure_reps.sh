#!/usr/bin/env bash
# Interleaved repeats to bound GPU contention (live training shares the box).
set -u
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
export LD_LIBRARY_PATH=/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
cd /opt/data/workspaces/skg/ofcuda_env
BIN=/tmp/ofcuda_tick_check/release/gpu_env
RUN="nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh $BIN"
T=8
OUT=/tmp/ge_rep.txt
: > $OUT
for R in 1 2 3 4 5; do
  for N in 1 64 256 1024 4096; do
    $RUN --ticks $T --envs $N --mode fast 2>/dev/null \
      | grep -E "^N=[0-9]+ +mode=fast-1launch" \
      | sed "s/^/run=$R /" >> $OUT
  done
done
echo "=== contention snapshot ===" >> $OUT
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $OUT 2>&1
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv >> $OUT 2>&1
echo REPEATS_DONE
