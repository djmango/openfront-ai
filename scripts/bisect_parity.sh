#!/usr/bin/env bash
# One-command tick-level bisection for a native-vs-TS full-game divergence.
# Built to replace the ad-hoc "write a custom test, run it, eyeball the
# output, repeat" loop every prior bisection this session did by hand - that
# loop is what made bisections the slowest category of parity work. This
# automates the exact same idea (coarse dump -> diff -> narrow -> fine dump
# -> diff) as a single call.
#
# Two-pass strategy (NOT per-probe binary search - both engines only support
# replaying from tick 0, there's no mid-game resume, so "binary search" would
# just mean re-running from scratch at every probed tick, which is strictly
# worse than this):
#   1. Coarse pass: dump both engines every --coarse-every ticks, up to
#      --max-ticks. Diff to find the first diverging checkpoint window.
#   2. Fine pass: re-dump both engines every tick, but ONLY up to that
#      window (usually a tiny fraction of --max-ticks) - this is where the
#      time savings actually come from, since the TS side is far slower
#      per-tick than native and this avoids replaying the rest of the game.
#
# Usage (from openfront-ai/):
#   scripts/bisect_parity.sh <record.json.gz> [--max-ticks N] [--coarse-every N] [--fields f1,f2,...]
#
# Pick --max-ticks from context you already have (e.g. outcome_gate's
# reported terminal ticks for this record + a margin) rather than the full
# 20000-tick horizon when you can - every tick you don't need to ask for is
# time you don't spend waiting on the TS side.
#
# Output: prints the first diverging tick + exact player/field, and leaves
# /tmp/bisect_parity.<gameID>.{native,ts}.{coarse,fine}.json for follow-up
# inspection (e.g. cross-referencing against the TS source for that tick).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="${1:?usage: bisect_parity.sh <record.json.gz> [--max-ticks N] [--coarse-every N] [--fields f1,f2,...]}"
shift || true

MAX_TICKS=20000
COARSE_EVERY=200
FIELDS="alive,tiles,troops,gold,hash,numUnits"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-ticks) MAX_TICKS="$2"; shift 2 ;;
    --coarse-every) COARSE_EVERY="$2"; shift 2 ;;
    --fields) FIELDS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

GAME_ID="$(basename "$RECORD" | sed -E 's/\.json(\.gz)?$//')"
TMP="/tmp/bisect_parity.$GAME_ID"

echo "[bisect_parity] $GAME_ID: coarse pass (every=$COARSE_EVERY, max=$MAX_TICKS)" >&2

cargo run --quiet --release --manifest-path "$ROOT/rust/Cargo.toml" -p openfront-engine --bin tick_dump -- \
  --repo "$ROOT" --record "$RECORD" --every "$COARSE_EVERY" --max-ticks "$MAX_TICKS" \
  --out "$TMP.native.coarse.json" >&2

"$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/scripts/dump_ts_tick_state.ts" \
  "$RECORD" "$COARSE_EVERY" "$TMP.ts.coarse.json" "$MAX_TICKS" >&2

set +e
DIFF_OUT="$(python3 "$ROOT/scripts/diff_tick_dumps.py" "$TMP.native.coarse.json" "$TMP.ts.coarse.json" --fields "$FIELDS")"
DIFF_STATUS=$?
set -e
echo "$DIFF_OUT"

if [[ $DIFF_STATUS -eq 0 ]]; then
  echo "[bisect_parity] no divergence found up to tick $MAX_TICKS at $COARSE_EVERY-tick granularity - try a larger --max-ticks, or the two engines genuinely agree here" >&2
  exit 0
fi

DIVERGENT_TICK="$(echo "$DIFF_OUT" | grep -oP 'DIVERGENCE_TICK=\K[0-9]+' | tail -1)"
if [[ -z "$DIVERGENT_TICK" ]]; then
  echo "[bisect_parity] couldn't parse a divergence tick out of the diff output above - inspect it manually" >&2
  exit 1
fi

FINE_MAX=$((DIVERGENT_TICK + COARSE_EVERY))
if (( FINE_MAX > MAX_TICKS )); then FINE_MAX=$MAX_TICKS; fi

echo "[bisect_parity] coarse divergence near tick $DIVERGENT_TICK - fine pass up to tick $FINE_MAX (every=1)" >&2

# Only retain snapshots near the divergence window so large bot-count games
# do not blow past V8's JSON.stringify string limit.
FINE_FROM=$((DIVERGENT_TICK > COARSE_EVERY ? DIVERGENT_TICK - COARSE_EVERY : 0))
export OF_DUMP_TICKS_FROM="$FINE_FROM"

cargo run --quiet --release --manifest-path "$ROOT/rust/Cargo.toml" -p openfront-engine --bin tick_dump -- \
  --repo "$ROOT" --record "$RECORD" --every 1 --max-ticks "$FINE_MAX" \
  --out "$TMP.native.fine.json" >&2

"$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/scripts/dump_ts_tick_state.ts" \
  "$RECORD" 1 "$TMP.ts.fine.json" "$FINE_MAX" >&2

echo "[bisect_parity] exact tick-for-tick diff:" >&2
python3 "$ROOT/scripts/diff_tick_dumps.py" "$TMP.native.fine.json" "$TMP.ts.fine.json" --fields "$FIELDS"
echo "[bisect_parity] dumps kept at $TMP.{native,ts}.{coarse,fine}.json for follow-up" >&2
