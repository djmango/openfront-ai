#!/usr/bin/env bash
# Additive helper: dump per-player owned tiles + attacks from BOTH engines at
# every tick over a short window, for a single early-curriculum record, so the
# exact differing tile / attack can be located. New file; nothing existing is modified.
set -uo pipefail
ROOT="/opt/data/workspaces/skg/openfront-ai"
cd "$ROOT"
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
REC="${1:?record}"
MAX="${2:-313}"
TAG="${3:-own}"
export OF_DUMP_OWNED_TILES=1 OF_DUMP_ATTACKS=1 OF_DUMP_NDJSON=1
"$ROOT/rust/target/release/tick_dump" --repo "$ROOT" --record "$REC" --every 1 \
  --max-ticks "$MAX" --out "/tmp/$TAG.native.ndjson" --ndjson >"/tmp/$TAG.native.log" 2>&1 &
NP=$!
"$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/scripts/dump_ts_tick_state.ts" \
  "$REC" 1 "/tmp/$TAG.ts.ndjson" "$MAX" >"/tmp/$TAG.ts.log" 2>&1 &
TP=$!
wait $NP; NR=$?
wait $TP; TR=$?
echo "native rc=$NR ts rc=$TR"
wc -l "/tmp/$TAG.native.ndjson" "/tmp/$TAG.ts.ndjson"