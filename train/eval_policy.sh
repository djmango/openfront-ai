#!/usr/bin/env bash
# Evaluate a checkpoint's WIN RATE on a pinned curriculum stage.
#
# Why this exists: the trainer's own win rate is confounded. Envs advance to
# harder stages mid-run, so the rolling window mixes stages and the number is
# not comparable between experiments. This pins the ladder (OPENFRONT_HOLD_STAGE)
# and plays N episodes on one stage with a fixed seed, so two checkpoints can be
# compared directly.
#
# Each vec env gets `train.seed + i`, so several threads play genuinely different
# games rather than N copies of one.
#
# The trainer is paused for the duration: it is CPU-bound and saturates all 8
# cores, and a nice'd eval alongside it measured ~8x slower per episode. The
# watchdog timer is stood down too - otherwise it sees a "dead" trainer every 2
# minutes and starts one, which fights the eval for CPU and leaves a duplicate.
#
# Usage: eval_policy.sh [<ckpt.bin>|latest] [episodes] [threads]
#   default: latest, 200 episodes, 8 threads
#   STAGE=<n> pins the curriculum stage the eval STARTS on (default 0).
#   PAUSE_TRAIN=0 keeps the trainer and watchdog alone (slow, quick look only)
#
# Which STAGE to use matters more than it looks. Stage 0 has 2 Easy bots, and
# the engine's win condition is `percentage_owned > 80% || time_limit_reached`
# (execution/win_check.rs), so at the time limit the TILE LEADER wins. A
# uniform-random policy out-tiles 2 weak bots, so stage 0 reports ~1.000 wins
# for a trained policy AND for an all-zero control: it cannot measure skill.
# Stage 4 (14 bots) is where the tile race stops being free.
#
# Output: one CUDA_EVAL line. The number to read is
#   win_rate_eval = wins / (wins + losses)
# (env/win_rate in that line is the 40-episode rolling window and reads 0.000
# until the window fills, so ignore it for short evals.)
set -euo pipefail

PL=/opt/data/workspaces/skg/PufferLib5
RUN=/opt/data/workspaces/skg/openfront-train
OMP_LIB_DIR=/nix/store/mczwc9iws3bjamnx0fi5kxgn9gpvin2g-openmp-21.1.8/lib
UNITS="openfront-train.service openfront-sampler.service"
# Documented maintenance flag in watch_train.sh: the watchdog does nothing at
# all while this file exists.
WATCHDOG_OFF="$RUN/WATCHDOG_OFF"

# NixOS keeps the setuid sudo in /run/wrappers/bin; the `sudo` on the system
# profile PATH is the raw store binary and is NOT setuid, so a bare `sudo`
# fails with "must be owned by uid 0 and have the setuid bit set". Every call
# in here used to end in `|| true`, so that failure was invisible: on
# 2026-09-18 05:46 the freeze became a no-op (eval and trainer split the 8
# cores) and the thaw failed too, leaving the trainer SIGSTOPped until it was
# noticed by hand 13 minutes later. Resolve the wrapper explicitly and verify
# BOTH directions instead of assuming either one worked.
if [ -z "${SUDO:-}" ]; then
    if [ -x /run/wrappers/bin/sudo ]; then SUDO=/run/wrappers/bin/sudo; else SUDO=sudo; fi
fi
trainer_pid() { systemctl show -p MainPID --value openfront-train.service 2>/dev/null; }
proc_state() { local st; read -r _ _ st _ < "/proc/$1/stat" 2>/dev/null || true; echo "${st:-?}"; }

CKPT="${1:-latest}"
EPISODES="${2:-200}"
THREADS="${3:-8}"
STAGE="${STAGE:-0}"
PAUSE_TRAIN="${PAUSE_TRAIN:-1}"

restore() {
    if [ "$PAUSE_TRAIN" = "1" ]; then
        thaw
        kill "${SAFETY:-0}" 2>/dev/null || true
        rm -f "$WATCHDOG_OFF"
    fi
}
trap restore EXIT INT TERM

# Freeze, never restart. `systemctl stop` + `start` gives a NEW process, and the
# curriculum is not in the checkpoint: every env drops to stage 0 with an empty
# 40-episode window. So each measurement used to cost ~1.7M steps of ladder
# warm-up, and two win rates were never taken at the same ladder position.
# SIGSTOP keeps the process and its ladder intact and frees the CPU just the same.
freeze() {
    $SUDO -n systemctl kill -s SIGSTOP $UNITS 2>/dev/null || true
    local p st; p=$(trainer_pid); st=$(proc_state "$p")
    if [ "$st" != T ]; then
        echo "[eval] WARNING: freeze did NOT take (trainer pid=$p state=$st) - this eval will share the 8 cores with the trainer" >&2
    fi
}
thaw() {
    $SUDO -n systemctl kill -s SIGCONT $UNITS 2>/dev/null || true
    local p st; p=$(trainer_pid); st=$(proc_state "$p")
    if [ "$st" = T ]; then
        $SUDO -n kill -CONT "$p" 2>/dev/null || true
        sleep 2; st=$(proc_state "$p")
    fi
    if [ "$st" = T ]; then
        echo "[eval] WARNING: TRAINER pid=$p IS STILL STOPPED after thaw. Fix with: $SUDO systemctl kill -s SIGCONT $UNITS" >&2
    fi
}

if [ "$PAUSE_TRAIN" = "1" ]; then
    touch "$WATCHDOG_OFF"
    echo "[eval] freezing $UNITS (watchdog stood down; process + curriculum survive)" >&2
    freeze
    # Safety net: a hard-killed eval must not leave the trainer frozen forever.
    ( sleep "${EVAL_MAX_SECONDS:-2400}"; $SUDO -n systemctl kill -s SIGCONT $UNITS 2>/dev/null || true ) &
    SAFETY=$!
    sleep 2
fi

cd "$PL"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:/opt/data/cuda-libs:$OMP_LIB_DIR:${LD_LIBRARY_PATH:-}"
# Pin the ladder. Without this a long eval promotes at 40 episodes per env and
# the reported win rate silently mixes two stages.
export OPENFRONT_HOLD_STAGE=1

./puffer-eval eval "$CKPT" --headless \
    --env.stage="$STAGE" \
    --vec.total_agents="$THREADS" \
    --vec.num_threads="$THREADS" \
    --base.eval_episodes="$EPISODES" \
    2>&1 | grep -aE "CUDA_EVAL|error|Error|assert" || true
