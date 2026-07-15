#!/usr/bin/env bash
# Smoke checks for V10 anti-death-spiral launch wiring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v8.sh"
WRAP="$ROOT/scripts/pod_train_v10.sh"

grep -q 'V10_MODE="${V10_MODE:-0}"' "$SCRIPT"
grep -q 'RUN_NAME="${RUN_NAME:-ppo_v10}"' "$SCRIPT"
grep -q -- '--v10-curriculum' "$SCRIPT"
grep -q -- '--v86-death-penalty 3.0' "$SCRIPT"
grep -q -- '--v10-survival-coef 0.01' "$SCRIPT"
grep -q -- '--v10-diplo-panic 0.08' "$SCRIPT"
grep -q -- '--v10-combat-action 0.02' "$SCRIPT"
grep -q -- '--migrate-v86-to-v10' "$SCRIPT"
grep -q 'V10_SOURCE_PREFIX="${V10_SOURCE_PREFIX:-ppo_v86}"' "$SCRIPT"
grep -q 'v10-anti-spiral-v1' "$ROOT/rust/ofcore/src/curriculum.rs"
grep -q 'V10_MODE=1' "$WRAP"

rg -n 'v10_curriculum|migrate_v86_to_v10|v10_survival_coef|v10_diplo_panic|v10_combat_action' \
  "$ROOT/rust/oftrain/src/main.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
rg -n 'V10_REWARD_PROFILE|v10_reward_active|should_demote_v10|should_advance_v10|V10_BOT_NATION_DENSITY' \
  "$ROOT/rust/ofcore/src/curriculum.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
rg -n 'v10_survival_reward|v10_diplo_panic_penalty|v10_combat_action_bonus|uses_v83_closeout' \
  "$ROOT/rust/oftrain/src/vecenv.rs" "$ROOT/rust/ofcore/src/curriculum.rs" >/dev/null

echo "pod_train_v10_test: ok"
