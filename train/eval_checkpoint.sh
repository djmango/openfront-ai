#!/usr/bin/env bash
# Cron-friendly checkpoint eval: append a timestamped stage-0 win rate to the
# eval log so an experiment's before/after is recorded without a human.
#
# Pauses the trainer for the duration (eval_policy.sh stops it, stands the
# watchdog down with WATCHDOG_OFF, and restores both), so keep the episode count
# modest on a repeating schedule.
#
# Usage: eval_checkpoint.sh [episodes_per_env] [threads]
set -euo pipefail

DIR=/opt/data/workspaces/skg/openfront-train
OUT="$DIR/experiments/eval-log.txt"
EPISODES="${1:-25}"
THREADS="${2:-8}"

mkdir -p "$DIR/experiments"
EXP=$(ls -t "$DIR/experiments"/*.meta 2>/dev/null | head -1 | xargs -r basename | sed 's/\.meta$//')

{
    echo "--- $(date -Is)  exp=${EXP:-none}  episodes/env=${EPISODES}"
    bash "$DIR/eval_policy.sh" latest "$EPISODES" "$THREADS" 2>&1 \
        | grep -a CUDA_EVAL \
        || echo "  (eval produced no CUDA_EVAL line)"
} >> "$OUT"

tail -4 "$OUT"
