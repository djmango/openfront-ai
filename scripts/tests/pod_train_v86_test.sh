#!/usr/bin/env bash
# Smoke checks for V8.6 attack-fair launch wiring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v8.sh"

grep -q 'V86_MODE="${V86_MODE:-0}"' "$SCRIPT"
grep -q 'RUN_NAME="${RUN_NAME:-ppo_v86}"' "$SCRIPT"
grep -q -- '--v81-dominance-threshold 0.30' "$SCRIPT"
grep -q -- '--v86-delta-loss 5.5' "$SCRIPT"
grep -q -- '--v86-attack-symmetric-loss=true' "$SCRIPT"
grep -q -- '--v86-skip-combat-churn=true' "$SCRIPT"
grep -q -- '--v86-death-penalty 10.0' "$SCRIPT"
grep -q -- '--v85-premature-retreat=-0.03' "$SCRIPT"
grep -q -- '--migrate-v85-to-v86' "$SCRIPT"
grep -q 'V86_SOURCE_PREFIX="${V86_SOURCE_PREFIX:-ppo_v85}"' "$SCRIPT"
grep -q 'v8.6-attack-fair-v1' "$SCRIPT"

rg -n 'migrate_v85_to_v86|v86_delta_loss|v86_attack_symmetric_loss|v86_skip_combat_churn|v86_death_penalty' \
  "$ROOT/rust/oftrain/src/main.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
rg -n 'V86_REWARD_PROFILE|v86_reward_active|reward_profile_id' \
  "$ROOT/rust/ofcore/src/curriculum.rs" >/dev/null
rg -n 'v86_skip_combat_churn|has_sourced_attack|death_penalty|or_insert' \
  "$ROOT/rust/oftrain/src/vecenv.rs" >/dev/null

echo "pod_train_v86_test: ok"
