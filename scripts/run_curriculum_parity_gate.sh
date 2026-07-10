#!/usr/bin/env bash
# Curriculum-representative counterpart to run_outcome_gate.sh: the 78-record
# archive at records/$PARITY_COMMIT/ is uniformly 400-bot/125-human
# mega-games (see docs/devlog.html, "Why the gate took 4 hours..."), which is
# nowhere near the bot counts the RL curriculum actually trains on
# (rust/ofcore/src/curriculum.rs stages(): 0/5/10/30/50/80/120/150). This
# generates a small TS self-play record set at those exact bot counts and
# runs the same TS-vs-native outcome comparison against it, then breaks the
# pass rate down BY BOT COUNT so it's clear where (if anywhere) divergence
# starts becoming a problem.
#
# Usage (from openfront-ai/):
#   scripts/run_curriculum_parity_gate.sh [--regenerate]
#
# Env overrides: CURRICULUM_RECORDS_DIR, CURRICULUM_GAMES_PER_BUCKET,
# CURRICULUM_TICKS, CURRICULUM_MAX_TIMER, CURRICULUM_PARITY_LABEL,
# CURRICULUM_JOBS, CURRICULUM_RECORD_TIMEOUT_SECONDS.
set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
source "$ROOT/scripts/parity_env.sh" >&2
bash "$ROOT/scripts/ensure_parity_openfront.sh" >&2

RECORDS_DIR="${CURRICULUM_RECORDS_DIR:-$ROOT/records/curriculum-parity-v1}"
GAMES_PER_BUCKET="${CURRICULUM_GAMES_PER_BUCKET:-5}"
TICKS="${CURRICULUM_TICKS:-4500}"
MAX_TIMER="${CURRICULUM_MAX_TIMER:-6}"
PARITY_LABEL="${CURRICULUM_PARITY_LABEL:-curriculum-parity-v1}"
JOBS="${CURRICULUM_JOBS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
RECORD_TIMEOUT_SECONDS="${CURRICULUM_RECORD_TIMEOUT_SECONDS:-300}"

CACHE_ROOT="${OUTCOME_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openfront-ai/outcomes}"
CACHE_FILE="$CACHE_ROOT/$PARITY_LABEL.json"
mkdir -p "$CACHE_ROOT"

if [[ "${1:-}" == "--regenerate" || ! -d "$RECORDS_DIR" ]]; then
  echo "[curriculum_parity] generating self-play records -> $RECORDS_DIR" >&2
  rm -rf "$RECORDS_DIR" "$RECORDS_DIR.manifest.json" "$CACHE_FILE" "$CACHE_FILE.records"
  pushd "$ROOT" >/dev/null
  "$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/datagen/gen_curriculum_parity.ts" \
    --out "$RECORDS_DIR" \
    --games-per-bucket "$GAMES_PER_BUCKET" \
    --ticks "$TICKS" \
    --max-timer "$MAX_TIMER" 1>&2
  popd >/dev/null
fi

NUM_RECORDS="$(find "$RECORDS_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.json.gz' \) | wc -l | tr -d ' ')"

pushd "$ROOT" >/dev/null
"$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/datagen/replay.ts" \
  --outcome-oracle \
  --records "$RECORDS_DIR" \
  --cache "$CACHE_FILE" \
  --parity-commit "$PARITY_LABEL" \
  --limit 0 \
  --jobs "$JOBS" \
  --record-timeout-seconds "$RECORD_TIMEOUT_SECONDS" 1>&2
popd >/dev/null

CARGO_ARGS=(
  run
  --quiet
  --manifest-path "$ROOT/rust/Cargo.toml"
  -p openfront-engine
  --release
  --bin outcome_gate
  --
  --repo "$ROOT"
  --records "$RECORDS_DIR"
  --oracle "$CACHE_FILE"
  --parity-commit "$PARITY_LABEL"
  --expected-records "$NUM_RECORDS"
  --required-passes 0
  --jobs "$JOBS"
  --record-timeout-seconds "$RECORD_TIMEOUT_SECONDS"
)
if [[ -n "${OUTCOME_TARGET_DIR:-}" ]]; then
  export CARGO_TARGET_DIR="$OUTCOME_TARGET_DIR"
fi
REPORT_FILE="$(mktemp -t curriculum_gate_report).json"
cargo "${CARGO_ARGS[@]}" > "$REPORT_FILE"
echo "[curriculum_parity] full report -> $REPORT_FILE" >&2

python3 "$ROOT/scripts/analyze_curriculum_parity.py" "$REPORT_FILE"
