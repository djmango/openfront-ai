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
# CURRICULUM_JOBS, CURRICULUM_RECORD_TIMEOUT_SECONDS, CURRICULUM_BUCKETS
# (comma-separated bot counts, e.g. "0,5" - passed through to
# gen_curriculum_parity.ts's --buckets, for fast targeted re-checks of one
# bucket instead of paying for all 8), CURRICULUM_PARITY_COMMIT,
# CURRICULUM_SKIP_NATIVE_CACHE=1 to force a cold native replay.
set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
source "$ROOT/scripts/parity_env.sh" >&2

# Unlike run_outcome_gate.sh/run_parity_gate.sh, this gate has NO dependency
# on the frozen archived-game PARITY_COMMIT (records/0c4c7d7993c9/ is real
# captured gameplay tied to that exact historical commit; the records here
# are freshly synthesized every run, with no archival tie to any specific
# commit). Reusing parity_env.sh's frozen PARITY_COMMIT default here was a
# real bug: it silently pinned the TS side of this gate to a commit that
# predates upstream TS's own `neighbors4`/`forEachNeighbor` order-unification
# fix (openfront commit 22d5aba5a, "standardize cardinal-neighbor iteration
# on neighbors() N,S,W,E order", #4495) - the SAME class of bug (and, for
# `AttackExecution.addNeighbors`, literally the same functions) that
# docs/bot-ai-parity-investigation/ already root-caused and fixed on the
# native side by matching native's `for_each_neighbor4` to N,S,W,E. Native
# already matches current upstream TS; the stale oracle commit did not, so
# every synthetic self-play game compared native against out-of-date TS
# neighbor-order behavior instead of current TS behavior, producing a
# systematic (and entirely spurious) native-vs-TS rate divergence - see
# docs/bot-ai-parity-rate/README.md for the full tick-level bisection.
#
# Override PARITY_COMMIT here (before ensure_parity_openfront.sh pins the
# checkout) to whatever openfront commit THIS superproject checkout actually
# has pinned via .gitmodules/the gitlink - i.e. "current master", not a
# frozen historical snapshot - so this gate always tracks whatever TS commit
# native is meant to mirror, even as that pin advances over time. Override
# with CURRICULUM_PARITY_COMMIT if a specific commit is needed instead.
export PARITY_COMMIT="${CURRICULUM_PARITY_COMMIT:-$(git -C "$ROOT" rev-parse HEAD:openfront 2>/dev/null || echo "$PARITY_COMMIT")}"

bash "$ROOT/scripts/ensure_parity_openfront.sh" >&2

RECORDS_DIR="${CURRICULUM_RECORDS_DIR:-$ROOT/records/curriculum-parity-v1}"
GAMES_PER_BUCKET="${CURRICULUM_GAMES_PER_BUCKET:-5}"
TICKS="${CURRICULUM_TICKS:-4500}"
MAX_TIMER="${CURRICULUM_MAX_TIMER:-6}"
PARITY_LABEL="${CURRICULUM_PARITY_LABEL:-curriculum-parity-v1}"
NPROC="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
JOBS="${CURRICULUM_JOBS:-$NPROC}"
RECORD_TIMEOUT_SECONDS="${CURRICULUM_RECORD_TIMEOUT_SECONDS:-300}"

if [[ "$JOBS" =~ ^[0-9]+$ ]] && (( JOBS < NPROC / 2 && JOBS < 8 )); then
  echo "[curriculum_parity] warning: CURRICULUM_JOBS=$JOBS on ${NPROC}-core host; unset it to use all cores" >&2
fi

CACHE_ROOT="${OUTCOME_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openfront-ai/outcomes}"
CACHE_FILE="$CACHE_ROOT/$PARITY_LABEL.json"
NATIVE_CACHE_FILE="$CACHE_ROOT/$PARITY_LABEL.native.json"
mkdir -p "$CACHE_ROOT"

if [[ "${1:-}" == "--regenerate" || ! -d "$RECORDS_DIR" ]]; then
  echo "[curriculum_parity] generating self-play records -> $RECORDS_DIR" >&2
  rm -rf "$RECORDS_DIR" "$RECORDS_DIR.manifest.json" "$CACHE_FILE" "$CACHE_FILE.records" "$NATIVE_CACHE_FILE"
  GEN_ARGS=(
    --out "$RECORDS_DIR"
    --games-per-bucket "$GAMES_PER_BUCKET"
    --ticks "$TICKS"
    --max-timer "$MAX_TIMER"
  )
  if [[ -n "${CURRICULUM_BUCKETS:-}" ]]; then
    GEN_ARGS+=(--buckets "$CURRICULUM_BUCKETS")
  fi
  pushd "$ROOT" >/dev/null
  "$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/datagen/gen_curriculum_parity.ts" "${GEN_ARGS[@]}" 1>&2
  popd >/dev/null
fi

# Follow symlinks (curriculum-parity-v4 is often a link into /workspace/records).
NUM_RECORDS="$(find -L "$RECORDS_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.json.gz' \) | wc -l | tr -d ' ')"
if [[ "$NUM_RECORDS" == "0" ]]; then
  echo "[curriculum_parity] warning: found 0 records under $RECORDS_DIR; skipping expected-count check" >&2
fi

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

if [[ -n "${OUTCOME_TARGET_DIR:-}" ]]; then
  export CARGO_TARGET_DIR="$OUTCOME_TARGET_DIR"
fi
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/rust/target}"
BIN="$TARGET_DIR/release/outcome_gate"

# Build once, then exec the binary. `cargo run` re-checks/rebuilds on every
# gate invocation and was dominating iterative loops even when sources were
# unchanged.
echo "[curriculum_parity] building outcome_gate (release) -> $BIN" >&2
cargo build --quiet --manifest-path "$ROOT/rust/Cargo.toml" -p openfront-engine --release --bin outcome_gate

if [[ "${CURRICULUM_SKIP_NATIVE_CACHE:-0}" == "1" ]]; then
  rm -f "$NATIVE_CACHE_FILE"
fi

REPORT_FILE="$(mktemp -t curriculum_gate_report.XXXXXX).json"
GATE_ARGS=(
  --repo "$ROOT"
  --records "$RECORDS_DIR"
  --oracle "$CACHE_FILE"
  --parity-commit "$PARITY_LABEL"
  --expected-records "$NUM_RECORDS"
  --required-passes 0
  --jobs "$JOBS"
  --record-timeout-seconds "$RECORD_TIMEOUT_SECONDS"
  --native-cache "$NATIVE_CACHE_FILE"
)
"$BIN" "${GATE_ARGS[@]}" > "$REPORT_FILE"
echo "[curriculum_parity] full report -> $REPORT_FILE" >&2

python3 "$ROOT/scripts/analyze_curriculum_parity.py" "$REPORT_FILE"
