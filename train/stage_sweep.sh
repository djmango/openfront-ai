#!/usr/bin/env bash
# Sweep a checkpoint over curriculum stages to find where the win rate is informative.
#
# Why: the win condition is `percentage_owned > 80% || time_limit_reached`, so at
# the time limit the TILE LEADER wins. Stage 0 is 2 Easy bots (V10_BOT_NATION_DENSITY
# (2,0)) - a uniform-random policy out-tiles them, so every policy reads ~1.000
# there and stage 0 cannot separate a trained policy from an all-zero control.
# This finds the stage where the numbers actually separate.
#
# Usage: stage_sweep.sh [episodes] [threads] [stage...]
# Output: one `### <ckpt> @ stage N` block per condition, each with its CUDA_EVAL
# line and one OFEP line per finished episode (tick= won= died=).
set -uo pipefail

PL=/opt/data/workspaces/skg/PufferLib5
RUN=/opt/data/workspaces/skg/openfront-train
OMP_LIB_DIR=/nix/store/mczwc9iws3bjamnx0fi5kxgn9gpvin2g-openmp-21.1.8/lib
UNITS="openfront-train.service openfront-sampler.service"
WATCHDOG_OFF="$RUN/WATCHDOG_OFF"

EPISODES="${1:-4}"
THREADS="${2:-4}"
shift 2 2>/dev/null || true
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(0 4 6)
fi
CKPTS=(/tmp/zeros_ckpt.bin latest)

touch "$WATCHDOG_OFF"
restore() {
    rm -f "$WATCHDOG_OFF"
    sudo -n systemctl start $UNITS 2>/dev/null || true
}
trap restore EXIT INT TERM
sudo -n systemctl stop $UNITS 2>/dev/null || true
sleep 2

cd "$PL"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:/opt/data/cuda-libs:$OMP_LIB_DIR:${LD_LIBRARY_PATH:-}"
# Pin the ladder: without it a stage promotes mid-eval and the win rate mixes stages.
export OPENFRONT_HOLD_STAGE=1
# A period far larger than any episode: keep the per-episode OFEP lines, drop the
# per-N-decision spam.
export OF_TRACE=200000

for st in "${STAGES[@]}"; do
    for ck in "${CKPTS[@]}"; do
        echo "### $(basename "$ck") @ stage $st  ($EPISODES episodes total, $THREADS envs)"
        timeout 240 ./puffer-eval eval "$ck" --headless --env.stage="$st" \
            --vec.total_agents="$THREADS" --vec.num_threads="$THREADS" \
            --base.eval_episodes="$EPISODES" 2>&1 | grep -aE "CUDA_EVAL|OFEP"
    done
done
