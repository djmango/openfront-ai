#!/usr/bin/env bash
# Smoke checks for V8.4 launch wiring in scripts/pod_train_v8.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/pod_train_v8.sh"

grep -q 'V84_MODE="${V84_MODE:-0}"' "$SCRIPT"
grep -q 'RUN_NAME="${RUN_NAME:-ppo_v84}"' "$SCRIPT"
grep -q 'ROLLOUT_LEN="${ROLLOUT_LEN:-64}"' "$SCRIPT"
grep -q 'BPTT_CHUNK_LEN="${BPTT_CHUNK_LEN:-32}"' "$SCRIPT"
grep -q -- '--v84-boat-useful 0.15' "$SCRIPT"
grep -q -- '--v84-boat-destroyed=-0.20' "$SCRIPT"
grep -q -- '--v84-tempo-coef 0.005' "$SCRIPT"
grep -q -- '--v84-fast-win-coef 8.0' "$SCRIPT"
grep -q -- '--migrate-v83-to-v84' "$SCRIPT"
grep -q 'V84_SOURCE_PREFIX="${V84_SOURCE_PREFIX:-ppo_v83}"' "$SCRIPT"

# CLI surface
rg -n 'migrate_v83_to_v84|v84_boat_useful|v84_tempo_coef|v84_fast_win_coef' \
  "$ROOT/rust/oftrain/src/main.rs" >/dev/null

# Reward profile constant
rg -n 'V84_REWARD_PROFILE' "$ROOT/rust/ofcore/src/curriculum.rs" >/dev/null

echo "pod_train_v84_test: ok"
