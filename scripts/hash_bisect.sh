#!/usr/bin/env bash
# True binary search for first native-vs-TS hash diverge using dump daemons.
#
# Each engine stays warm in a daemon process: ADVANCE only moves forward
# (mid-game resume). On disagreement we RESET and re-ADVANCE to the last
# known-good tick. This replaces "replay from 0 at every probe."
#
# Assumes diverge is sticky (once hashes differ they stay different).
#
# Usage (repo root):
#   scripts/hash_bisect.sh <record.json.gz> [--max-ticks N] [--skip-before N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="${1:?usage: hash_bisect.sh <record.json.gz> [--max-ticks N]}"
shift || true

MAX_TICKS=""
SKIP_BEFORE=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-ticks) MAX_TICKS="$2"; shift 2 ;;
    --skip-before) SKIP_BEFORE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

GAME_ID="$(basename "$RECORD" | sed -E 's/\.json(\.gz)?$//')"
TMP="/tmp/hash_bisect.$GAME_ID"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/rust/target}"
TICK_DUMP="$TARGET_DIR/release/tick_dump"
TSX="$ROOT/openfront/node_modules/.bin/tsx"
FIELDS="${HASH_BISECT_FIELDS:-alive,tiles,hashBits,unitsHash,numUnits}"

# Always invoke cargo so source edits are picked up (see hash_parity.sh).
echo "[hash_bisect] ensuring tick_dump" >&2
cargo build --quiet --release --manifest-path "$ROOT/rust/Cargo.toml" \
  -p openfront-engine --bin tick_dump >&2

if [[ -z "$MAX_TICKS" ]]; then
  MAX_TICKS="$(uv run --no-project python - "$RECORD" <<'PY'
import gzip, json, sys
p = sys.argv[1]
raw = gzip.open(p).read() if p.endswith(".gz") else open(p, "rb").read()
r = json.loads(raw)
info = r.get("info") or {}
n = info.get("numTurns") or info.get("num_turns")
if not n:
    turns = r.get("turns") or []
    n = max((t.get("turnNumber") or t.get("turn_number") or 0) for t in turns) if turns else 0
print(int(n))
PY
)"
fi
if [[ "$MAX_TICKS" -le "$SKIP_BEFORE" ]]; then
  echo "[hash_bisect] max_ticks=$MAX_TICKS too small" >&2
  exit 2
fi

echo "[hash_bisect] $GAME_ID max_ticks=$MAX_TICKS skip_before=$SKIP_BEFORE" >&2
rm -rf "$TMP"
mkdir -p "$TMP"
NATIVE_IN_FIFO="$TMP/native.in"
TS_IN_FIFO="$TMP/ts.in"
mkfifo "$NATIVE_IN_FIFO" "$TS_IN_FIFO"

"$TICK_DUMP" --daemon --repo "$ROOT" --record "$RECORD" \
  <"$NATIVE_IN_FIFO" >"$TMP/native.stdout" 2>"$TMP/native.err" &
NATIVE_PID=$!
"$TSX" "$ROOT/scripts/dump_ts_tick_state.ts" --daemon "$RECORD" \
  <"$TS_IN_FIFO" >"$TMP/ts.stdout" 2>"$TMP/ts.err" &
TS_PID=$!

exec {NATIVE_IN}>"$NATIVE_IN_FIFO"
exec {TS_IN}>"$TS_IN_FIFO"

cleanup() {
  echo QUIT >&"$NATIVE_IN" 2>/dev/null || true
  echo QUIT >&"$TS_IN" 2>/dev/null || true
  exec {NATIVE_IN}>&- 2>/dev/null || true
  exec {TS_IN}>&- 2>/dev/null || true
  kill "$NATIVE_PID" "$TS_PID" 2>/dev/null || true
  wait "$NATIVE_PID" 2>/dev/null || true
  wait "$TS_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_ok() {
  local out="$1"
  local i
  for i in $(seq 1 12000); do
    if grep -q '^OK' "$out" 2>/dev/null; then
      return 0
    fi
    if grep -q '^ERR' "$out" 2>/dev/null; then
      echo "[hash_bisect] daemon error in $out:" >&2
      cat "$out" >&2
      return 1
    fi
    sleep 0.05
  done
  echo "[hash_bisect] timeout waiting for OK in $out" >&2
  tail -30 "$TMP/native.err" "$TMP/ts.err" >&2 || true
  return 1
}

send_both() {
  local cmd="$1"
  : >"$TMP/native.stdout"
  : >"$TMP/ts.stdout"
  echo "$cmd" >&"$NATIVE_IN"
  echo "$cmd" >&"$TS_IN"
  wait_ok "$TMP/native.stdout"
  wait_ok "$TMP/ts.stdout"
}

# Wait for boot OK lines.
wait_ok "$TMP/native.stdout"
wait_ok "$TMP/ts.stdout"
echo "[hash_bisect] daemons ready" >&2

agree_at() {
  local tick="$1"
  local snap_n="$TMP/n.$tick.json"
  local snap_t="$TMP/t.$tick.json"
  send_both "ADVANCE $tick"
  : >"$TMP/native.stdout"
  : >"$TMP/ts.stdout"
  echo "DUMP $snap_n" >&"$NATIVE_IN"
  echo "DUMP $snap_t" >&"$TS_IN"
  wait_ok "$TMP/native.stdout"
  wait_ok "$TMP/ts.stdout"
  set +e
  uv run --no-project python "$ROOT/scripts/diff_tick_dumps.py" \
    "$snap_n" "$snap_t" --fields "$FIELDS" --skip-before 0 >"$TMP/diff.$tick.out" 2>"$TMP/diff.$tick.err"
  local st=$?
  set -e
  return "$st"
}

LO=$SKIP_BEFORE
HI=$MAX_TICKS

echo "[hash_bisect] probe max tick $HI" >&2
if agree_at "$HI"; then
  echo "[hash_bisect] PASS — engines agree through tick $HI" >&2
  echo "HASH_PARITY_PASS=1"
  echo "DIVERGENCE_TICK="
  cleanup
  trap - EXIT
  exit 0
fi

echo "[hash_bisect] diverge by $HI — binary searching" >&2
send_both "RESET"
send_both "ADVANCE $LO"

PROBES=0
LAST_BAD=$HI
while [[ $((HI - LO)) -gt 1 ]]; do
  MID=$(( (LO + HI) / 2 ))
  PROBES=$((PROBES + 1))
  echo "[hash_bisect] probe #$PROBES lo=$LO mid=$MID hi=$HI" >&2
  if agree_at "$MID"; then
    LO=$MID
  else
    HI=$MID
    LAST_BAD=$MID
    send_both "RESET"
    send_both "ADVANCE $LO"
  fi
done

echo "[hash_bisect] FAIL first diverge in ($LO, $HI] — at $HI" >&2
echo "HASH_PARITY_PASS=0"
echo "DIVERGENCE_TICK=$HI"
echo "DIVERGENCE_LAYER=bisect"
echo "HASH_BISECT_PROBES=$PROBES"
echo "HASH_BISECT_LO=$LO"
cat "$TMP/diff.$LAST_BAD.out" 2>/dev/null || true

cleanup
trap - EXIT
exit 1
