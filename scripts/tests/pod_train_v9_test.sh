#!/usr/bin/env bash
# Smoke checks for V9 sparse-win launch wiring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v8.sh"
WRAPPER="$ROOT/scripts/pod_train_v9.sh"

grep -q 'V9_MODE="${V9_MODE:-0}"' "$SCRIPT"
grep -q 'RUN_NAME="${RUN_NAME:-ppo_v9}"' "$SCRIPT"
grep -q -- '--v9-curriculum' "$SCRIPT"
grep -q -- '--v9-sparse-win' "$SCRIPT"
grep -q -- '--gamma 0.9997' "$SCRIPT"
grep -q -- '--ret-clip 2' "$SCRIPT"
grep -q 'parallel sparse-win experiment' "$SCRIPT"
grep -q 'V9_MODE=1' "$WRAPPER"

rg -n 'v9_curriculum|v9_sparse_win' \
  "$ROOT/rust/oftrain/src/main.rs" >/dev/null
rg -n 'V9_REWARD_PROFILE|v9_sparse_win|sparse_terminal_reward|CurriculumSchedule::V9' \
  "$ROOT/rust/ofcore/src/curriculum.rs" >/dev/null
rg -n 'v9_sparse_win|sparse_terminal_reward' \
  "$ROOT/rust/oftrain/src/vecenv.rs" >/dev/null
rg -n 'CurriculumSchedule::V9|V9_REWARD_PROFILE' \
  "$ROOT/rust/oftrain/src/train.rs" >/dev/null

echo "pod_train_v9_test: ok"
