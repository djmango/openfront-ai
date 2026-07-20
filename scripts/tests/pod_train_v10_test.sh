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
# Pure-native is the production default; non-zero mix needs an explicit opt-in.
grep -q 'NODE_FRACTION="${NODE_FRACTION:-0}"' "$SCRIPT"
grep -q 'ALLOW_NODE_MIX="${ALLOW_NODE_MIX:-0}"' "$SCRIPT"
grep -q 'ALLOW_NODE_MIX=1' "$SCRIPT"
grep -q -- '--allow-node-mix' "$SCRIPT"
# Do not advertise a concrete non-zero NODE_FRACTION in examples (operators
# copy-paste it). `set -e` ignores `! cmd` failures — assert explicitly.
if grep -nE 'NODE_FRACTION=0\.[1-9]' "$SCRIPT"; then
  echo "FAIL: non-zero NODE_FRACTION example still present in pod_train_v10.sh" >&2
  exit 1
fi
# Recommended dockerArgs must pin NODE_FRACTION=0 and curl v10 (not v8+V10_MODE).
grep -q 'NODE_FRACTION=0 NCCL_P2P_DISABLE=1' "$SCRIPT"
grep -q 'curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v10.sh' "$SCRIPT"
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
rg -n 'V10_REWARD_PROFILE|v10_reward_active|should_demote_v10|should_advance_v10|V10_BOT_NATION_DENSITY|V10_EASY_RAMP_LEN|V10_CLOSEOUT_STAGE|V10_MAP_WARMUP_LEN|V10_BROAD_STAGE' \
  "$ROOT/rust/ofcore/src/curriculum.rs" "$ROOT/rust/oftrain/src/train.rs" >/dev/null
grep -q 'V10_EASY_RAMP_LEN: usize = 30' "$ROOT/rust/ofcore/src/curriculum.rs"
grep -q 'V10_MAP_WARMUP_LEN: usize = 8' "$ROOT/rust/ofcore/src/curriculum.rs"
grep -q 'V10_STAGE_COUNT: usize = 100' "$ROOT/rust/ofcore/src/curriculum.rs"
# Early stages must mix maps (bridge → broad), not Onion-only.
grep -q 'push(&V10_BRIDGE_MAPS, "Easy", 15, V10_MAP_WARMUP_LEN)' \
  "$ROOT/rust/ofcore/src/curriculum.rs"
grep -q 'push(&V10_BROAD_MAPS, "Easy", 15, 38)' \
  "$ROOT/rust/ofcore/src/curriculum.rs"
! grep -q 'push(ONION,' "$ROOT/rust/ofcore/src/curriculum.rs"

echo "pod_train_v10_test: ok"
