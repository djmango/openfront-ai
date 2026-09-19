#!/usr/bin/env bash
# Launch the OpenFront x PufferLib 5.0 trainer (headless, CUDA, this VM).
#
# Environment glue only: nvcc/NCCL/cudart come from nixpkgs + the nvidia-*-cu12
# pip wheels, so the loader paths must be set here rather than at link time.
#
# Gradient steps per rollout, not samples, is what makes this learn:
#   total_minibatches = replay_ratio * (horizon * agents) / minibatch_size
# With MB=4096 and horizon*agents=4096 that is exactly 1 update per rollout,
# which left the policy at uniform-over-legal (entropy ~= ln(legal actions),
# kl 0.000) for 5.5M steps. Keep MB small relative to the batch.
set -euo pipefail

PL=/opt/data/workspaces/skg/PufferLib5
OMP_LIB_DIR=/nix/store/mczwc9iws3bjamnx0fi5kxgn9gpvin2g-openmp-21.1.8/lib

cd "$PL"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:/opt/data/cuda-libs:$OMP_LIB_DIR:${LD_LIBRARY_PATH:-}"

AGENTS=${AGENTS:-64}
THREADS=${THREADS:-8}
MB=${MB:-512}
REPLAY=${REPLAY:-4}
STEPS=${STEPS:-20000000}
# Samples per rollout are HORIZON * AGENTS; raising HORIZON adds data per update
# without another env session (engine sessions, not buffers, dominate RAM here).
HORIZON=${HORIZON:-64}
# Cosine horizon in epochs for the LR/ent_coef anneal; 0 = derive from STEPS.
# Set it when the anneal should finish before the run does.
SCHED_EPOCHS=${SCHED_EPOCHS:-0}

extra=()

# LOAD=auto resolves the newest checkpoint across every run dir, so a reboot or
# crash resumes from the most recent weights instead of a path baked into a
# unit file (which silently goes stale and re-trains from old weights).
if [ "${LOAD:-}" = "auto" ]; then
    LOAD=$(ls -t "$PL"/checkpoints/openfront/*/*.bin 2>/dev/null | head -1 || true)
    if [ -n "$LOAD" ]; then
        echo "[train] LOAD=auto -> $LOAD" >&2
    else
        echo "[train] LOAD=auto found no checkpoint; starting from random init" >&2
    fi
fi

[ -n "${LR:-}" ] && extra+=("--train.learning_rate=${LR}")
[ -n "${LOAD:-}" ] && extra+=("--base.load_model_path=${LOAD}")
[ -n "${ENT:-}" ] && extra+=("--train.ent_coef=${ENT}")
[ "${HORIZON}" != "64" ] && extra+=("--train.horizon=${HORIZON}")
[ "${SCHED_EPOCHS}" != "0" ] && extra+=("--train.schedule_epochs=${SCHED_EPOCHS}")

exec ./puffer train --headless \
    --vec.total_agents="$AGENTS" \
    --vec.num_threads="$THREADS" \
    --train.minibatch_size="$MB" \
    --train.replay_ratio="$REPLAY" \
    --base.async=0 \
    --base.cudagraphs=-1 \
    --train.total_timesteps="$STEPS" \
    "${extra[@]}" \
    "$@"