#!/usr/bin/env bash
# Smoke checks for the default V10 trainer launch wiring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v10.sh"
WRAP="$ROOT/scripts/pod_train_v8.sh"

# No legacy mode switches.
! grep -q 'V10_MODE=' "$SCRIPT"
! grep -q 'V9_MODE=' "$SCRIPT"
! grep -q 'V86_MODE=' "$SCRIPT"
! grep -q 'V85_MODE=' "$SCRIPT"
! grep -q 'V84_MODE=' "$SCRIPT"
! grep -q 'V83_MODE=' "$SCRIPT"

grep -q 'RUN_NAME="${RUN_NAME:-ppo_v10}"' "$SCRIPT"
grep -q 'NUM_GPUS="${NUM_GPUS:-4}"' "$SCRIPT"
! grep -q -- '--v10-curriculum' "$SCRIPT"
grep -q -- '--v86-death-penalty 3.0' "$SCRIPT"
grep -q -- '--v10-survival-coef 0.01' "$SCRIPT"
grep -q -- '--v10-diplo-panic 0.08' "$SCRIPT"
grep -q -- '--v10-combat-action 0.02' "$SCRIPT"
grep -q -- '--v10-timeout-closeout 20.0' "$SCRIPT"
grep -q -- '--v85-extra-win-bonus 200.0' "$SCRIPT"
grep -q -- '--v84-fast-win-coef 40.0' "$SCRIPT"
grep -q -- '--v10-closeout-entry 25.0' "$SCRIPT"
grep -q -- '--max-episode-ticks 21000' "$SCRIPT"
grep -q -- '--migrate-v86-to-v10' "$SCRIPT"
grep -q 'v10-anti-spiral-v1' "$ROOT/rust/ofcore/src/curriculum.rs"

# Wrapper is a pure compatibility alias.
grep -q 'pod_train_v10.sh' "$WRAP"
! grep -q 'V10_MODE=1' "$WRAP"

rg -n 'migrate_v86_to_v10|v10_survival_coef|v10_diplo_panic|v10_combat_action|v10_timeout_closeout' \
  "$ROOT/rust/oftrain/src/main.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
rg -n 'V10_REWARD_PROFILE|v10_reward_active|should_demote_v10|should_advance_v10|V10_BOT_NATION_DENSITY|V10_EASY_RAMP_LEN|V10_CLOSEOUT_STAGE' \
  "$ROOT/rust/ofcore/src/curriculum.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
grep -q 'V10_EASY_RAMP_LEN: usize = 30' "$ROOT/rust/ofcore/src/curriculum.rs"
grep -q 'V10_STAGE_COUNT: usize = 100' "$ROOT/rust/ofcore/src/curriculum.rs"

echo "pod_train_v10_test: ok"
