#!/usr/bin/env bash
# Smoke checks for V8.5 launch wiring in scripts/pod_train_v8.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v8.sh"

grep -q 'V85_MODE="${V85_MODE:-0}"' "$SCRIPT"
grep -q 'RUN_NAME="${RUN_NAME:-ppo_v85}"' "$SCRIPT"
grep -q -- '--v85-tempo-share-threshold 0.30' "$SCRIPT"
grep -q -- '--v85-extra-win-bonus 30.0' "$SCRIPT"
grep -q -- '--v85-embargo-bad-stop=-0.15' "$SCRIPT"
grep -q -- '--v85-premature-retreat=-0.10' "$SCRIPT"
grep -q -- '--migrate-v84-to-v85' "$SCRIPT"
grep -q 'V85_SOURCE_PREFIX="${V85_SOURCE_PREFIX:-ppo_v84}"' "$SCRIPT"
grep -q -- '--v84-tempo-coef 0.015' "$SCRIPT"
grep -q -- '--v84-fast-win-coef 12.0' "$SCRIPT"

rg -n 'migrate_v84_to_v85|v85_extra_win_bonus|v85_embargo_bad_stop' \
  "$ROOT/rust/oftrain/src/main.rs" >/dev/null
rg -n 'V85_REWARD_PROFILE' "$ROOT/rust/ofcore/src/curriculum.rs" >/dev/null

echo "pod_train_v85_test: ok"
