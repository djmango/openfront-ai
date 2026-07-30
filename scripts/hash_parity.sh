#!/usr/bin/env bash
# One-command hash/bit-parity probe: native + TS dump in parallel as NDJSON,
# online compare with early-stop, optional unit expand around first diverge.
#
# This replaces the dumb coarse-then-fine full-replay loop for day-to-day
# tip hash work: stream NDJSON from both engines, early-stop on first diverge,
# then optionally expand with OF_DUMP_UNITS near the window.
#
# Usage (from repo root):
#   scripts/hash_parity.sh <record.json.gz> [--max-ticks N] [--every N] [--skip-before N]
#
# Env:
#   CARGO_TARGET_DIR   - build/run tick_dump from here (default: rust/target)
#   HASH_PARITY_JOBS   - unused placeholder for future multi-record fanout
#   OF_DUMP_*          - forwarded to dumpers (UNITS auto-enabled on expand pass)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="${1:?usage: hash_parity.sh <record.json.gz> [--max-ticks N] [--every N]}"
shift || true

MAX_TICKS=""
EVERY=1
SKIP_BEFORE=5
EXPAND_PAD=25
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-ticks) MAX_TICKS="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --skip-before) SKIP_BEFORE="$2"; shift 2 ;;
    --expand-pad) EXPAND_PAD="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

GAME_ID="$(basename "$RECORD" | sed -E 's/\.json(\.gz)?$//')"
TMP="/tmp/hash_parity.$GAME_ID"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/rust/target}"
TICK_DUMP="$TARGET_DIR/release/tick_dump"
TSX="$ROOT/openfront/node_modules/.bin/tsx"

echo "[hash_parity] $GAME_ID every=$EVERY max_ticks=${MAX_TICKS:-full}" >&2

if [[ ! -x "$TICK_DUMP" ]]; then
  echo "[hash_parity] building tick_dump into $TARGET_DIR" >&2
  cargo build --quiet --release --manifest-path "$ROOT/rust/Cargo.toml" \
    -p openfront-engine --bin tick_dump >&2
fi

# Clear prior streams so the comparator does not read stale lines.
rm -f "$TMP.native.ndjson" "$TMP.ts.ndjson" "$TMP.native.err" "$TMP.ts.err"
: >"$TMP.native.ndjson"
: >"$TMP.ts.ndjson"

NATIVE_ARGS=(--repo "$ROOT" --record "$RECORD" --every "$EVERY" --out "$TMP.native.ndjson" --ndjson)
TS_MAX_ARG=()
if [[ -n "$MAX_TICKS" ]]; then
  NATIVE_ARGS+=(--max-ticks "$MAX_TICKS")
  TS_MAX_ARG=("$MAX_TICKS")
fi

# Parent shells may export dump filters from a prior expand pass.
unset OF_DUMP_TICKS_FROM OF_DUMP_UNITS OF_DUMP_UNITS_FROM || true

echo "[hash_parity] launching native + TS dumps in parallel" >&2
"$TICK_DUMP" "${NATIVE_ARGS[@]}" >"$TMP.native.err" 2>&1 &
NATIVE_PID=$!
OF_DUMP_NDJSON=1 "$TSX" "$ROOT/scripts/dump_ts_tick_state.ts" \
  "$RECORD" "$EVERY" "$TMP.ts.ndjson" "${TS_MAX_ARG[@]}" \
  >"$TMP.ts.err" 2>&1 &
TS_PID=$!

cleanup() {
  kill "$NATIVE_PID" "$TS_PID" 2>/dev/null || true
  wait "$NATIVE_PID" 2>/dev/null || true
  wait "$TS_PID" 2>/dev/null || true
}
trap cleanup EXIT

set +e
COMPARE_OUT="$(
  uv run --no-project python "$ROOT/scripts/stream_compare_ticks.py" \
    "$TMP.native.ndjson" "$TMP.ts.ndjson" \
    --skip-before "$SKIP_BEFORE" \
    --idle-timeout 180 \
    --startup-timeout 900
)"
COMPARE_STATUS=$?
set -e
echo "$COMPARE_OUT"

DIVERGENT_TICK="$(echo "$COMPARE_OUT" | grep -oP 'DIVERGENCE_TICK=\K[0-9]+' | tail -1 || true)"
DIVERGENT_LAYER="$(echo "$COMPARE_OUT" | grep -oP 'DIVERGENCE_LAYER=\K\S+' | tail -1 || true)"

# Always stop dumpers once compare finishes (agree or diverge).
cleanup
trap - EXIT

if [[ $COMPARE_STATUS -eq 0 ]]; then
  echo "[hash_parity] PASS - engines agree through streamed range" >&2
  echo "HASH_PARITY_PASS=1"
  exit 0
fi

if [[ $COMPARE_STATUS -ne 1 || -z "$DIVERGENT_TICK" ]]; then
  echo "[hash_parity] compare failed (status=$COMPARE_STATUS); see $TMP.*.err" >&2
  tail -20 "$TMP.native.err" "$TMP.ts.err" >&2 || true
  exit "$COMPARE_STATUS"
fi

echo "[hash_parity] FAIL at tick $DIVERGENT_TICK layer=${DIVERGENT_LAYER:-unknown}" >&2
echo "HASH_PARITY_PASS=0"
echo "DIVERGENCE_TICK=$DIVERGENT_TICK"
echo "DIVERGENCE_LAYER=${DIVERGENT_LAYER:-unknown}"

# Auto expand: re-dump only near the window WITH units (still from tick 0,
# but retain from window start - avoids multi-GB JSON and answers "which unit").
# gameHash-only misses still expand: player fields may agree while unit/attack
# state under the game hash does not.
if [[ "${DIVERGENT_LAYER:-}" == "units" || "${DIVERGENT_LAYER:-}" == "hash" || "${DIVERGENT_LAYER:-}" == "gameHash" || "${HASH_PARITY_ALWAYS_EXPAND:-0}" == "1" ]]; then
  if [[ "$DIVERGENT_TICK" -gt "$EXPAND_PAD" ]]; then FROM=$((DIVERGENT_TICK - EXPAND_PAD)); else FROM=0; fi
  TO=$((DIVERGENT_TICK + EXPAND_PAD))
  if [[ -n "$MAX_TICKS" && "$TO" -gt "$MAX_TICKS" ]]; then TO=$MAX_TICKS; fi
  echo "[hash_parity] expand pass with OF_DUMP_UNITS ticks $FROM..$TO" >&2
  export OF_DUMP_UNITS=1
  export OF_DUMP_UNITS_FROM="$FROM"
  export OF_DUMP_TICKS_FROM="$FROM"
  "$TICK_DUMP" --repo "$ROOT" --record "$RECORD" --every 1 --max-ticks "$TO" \
    --out "$TMP.native.expand.json" >&2
  "$TSX" "$ROOT/scripts/dump_ts_tick_state.ts" \
    "$RECORD" 1 "$TMP.ts.expand.json" "$TO" >&2
  echo "[hash_parity] expand diff:" >&2
  uv run --no-project python "$ROOT/scripts/diff_tick_dumps.py" \
    "$TMP.native.expand.json" "$TMP.ts.expand.json" \
    --fields alive,tiles,troops,gold,hash,hashBits,unitsHash,numUnits \
    --skip-before "$FROM" || true
  echo "[hash_parity] expand dumps: $TMP.{native,ts}.expand.json" >&2
fi

echo "[hash_parity] streams kept at $TMP.{native,ts}.ndjson" >&2
exit 1
