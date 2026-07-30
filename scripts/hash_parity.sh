#!/usr/bin/env bash
# One-command hash/bit-parity probe: native + TS dump in parallel as NDJSON,
# online compare with early-stop, then mid-game-resume unit dump at diverge
# via dump daemons (ADVANCE to tick — no second full trailing replay).
#
# Usage (from repo root):
#   scripts/hash_parity.sh <record.json.gz> [--max-ticks N] [--every N] [--skip-before N]
#
# For true binary search over a long horizon, prefer scripts/hash_bisect.sh.
#
# Env:
#   CARGO_TARGET_DIR   - build/run tick_dump from here (default: rust/target)
#   HASH_PARITY_ALWAYS_EXPAND - expand even when layer is not units/hash/gameHash
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

rm -f "$TMP.native.ndjson" "$TMP.ts.ndjson" "$TMP.native.err" "$TMP.ts.err"
: >"$TMP.native.ndjson"
: >"$TMP.ts.ndjson"

NATIVE_ARGS=(--repo "$ROOT" --record "$RECORD" --every "$EVERY" --out "$TMP.native.ndjson" --ndjson)
TS_MAX_ARG=()
if [[ -n "$MAX_TICKS" ]]; then
  NATIVE_ARGS+=(--max-ticks "$MAX_TICKS")
  TS_MAX_ARG=("$MAX_TICKS")
fi

unset OF_DUMP_TICKS_FROM OF_DUMP_UNITS OF_DUMP_UNITS_FROM OF_DUMP_CONTROL || true

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

# Expand via dump daemons: warm once to the diverge tick (forward-only resume
# inside each daemon), DUMP with units. Avoids a second trailing full-game dump
# and avoids multi-GB every-tick JSON for the whole prefix.
if [[ "${DIVERGENT_LAYER:-}" == "units" || "${DIVERGENT_LAYER:-}" == "hash" || "${DIVERGENT_LAYER:-}" == "gameHash" || "${HASH_PARITY_ALWAYS_EXPAND:-0}" == "1" ]]; then
  echo "[hash_parity] daemon expand at tick $DIVERGENT_TICK (units)" >&2
  EXP_DIR="$TMP.expand"
  rm -rf "$EXP_DIR"
  mkdir -p "$EXP_DIR"
  mkfifo "$EXP_DIR/native.in" "$EXP_DIR/ts.in"
  "$TICK_DUMP" --daemon --repo "$ROOT" --record "$RECORD" \
    <"$EXP_DIR/native.in" >"$EXP_DIR/native.out" 2>"$EXP_DIR/native.err" &
  NP=$!
  "$TSX" "$ROOT/scripts/dump_ts_tick_state.ts" --daemon "$RECORD" \
    <"$EXP_DIR/ts.in" >"$EXP_DIR/ts.out" 2>"$EXP_DIR/ts.err" &
  TP=$!
  exec {EIN}>"$EXP_DIR/native.in"
  exec {EIT}>"$EXP_DIR/ts.in"
  wait_ok() {
    local f="$1" i
    for i in $(seq 1 12000); do
      grep -q '^OK' "$f" 2>/dev/null && return 0
      grep -q '^ERR' "$f" 2>/dev/null && { cat "$f" >&2; return 1; }
      sleep 0.05
    done
    return 1
  }
  wait_ok "$EXP_DIR/native.out"
  wait_ok "$EXP_DIR/ts.out"
  : >"$EXP_DIR/native.out"; : >"$EXP_DIR/ts.out"
  echo "ADVANCE $DIVERGENT_TICK" >&"$EIN"
  echo "ADVANCE $DIVERGENT_TICK" >&"$EIT"
  wait_ok "$EXP_DIR/native.out"
  wait_ok "$EXP_DIR/ts.out"
  : >"$EXP_DIR/native.out"; : >"$EXP_DIR/ts.out"
  echo "DUMP $TMP.native.expand.json units" >&"$EIN"
  echo "DUMP $TMP.ts.expand.json units" >&"$EIT"
  wait_ok "$EXP_DIR/native.out"
  wait_ok "$EXP_DIR/ts.out"
  echo QUIT >&"$EIN" 2>/dev/null || true
  echo QUIT >&"$EIT" 2>/dev/null || true
  exec {EIN}>&-; exec {EIT}>&-
  wait "$NP" 2>/dev/null || true
  wait "$TP" 2>/dev/null || true
  echo "[hash_parity] expand diff (single tick $DIVERGENT_TICK with units):" >&2
  uv run --no-project python "$ROOT/scripts/diff_tick_dumps.py" \
    "$TMP.native.expand.json" "$TMP.ts.expand.json" \
    --fields alive,tiles,troops,gold,hash,hashBits,unitsHash,numUnits \
    --skip-before 0 || true
  echo "[hash_parity] expand dumps: $TMP.{native,ts}.expand.json" >&2
fi

echo "[hash_parity] streams kept at $TMP.{native,ts}.ndjson" >&2
echo "[hash_parity] for logarithmic resume search: scripts/hash_bisect.sh $RECORD" >&2
exit 1
