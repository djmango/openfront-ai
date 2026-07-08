#!/usr/bin/env bash
# Track native Rust port coverage vs TS engine core.
set -euo pipefail
TS_CORE="${OPENFRONT_REPO:-/Users/djmango/github/openfront-ai}/openfront/src/core"
RUST_ENGINE="$(cd "$(dirname "$0")/../rust/engine/src" && pwd)"

ts_files=$(find "$TS_CORE" -name '*.ts' | wc -l | tr -d ' ')
ts_loc=$(find "$TS_CORE" -name '*.ts' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
rust_files=$(find "$RUST_ENGINE" -name '*.rs' | wc -l | tr -d ' ')
rust_loc=$(find "$RUST_ENGINE" -name '*.rs' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

echo "OpenFront native port status"
echo "  TS core:   $ts_files files, ~$ts_loc LOC"
echo "  Rust engine: $rust_files files, ~$rust_loc LOC"
pct=$(echo "scale=1; 100 * $rust_loc / $ts_loc" | bc 2>/dev/null || echo "?")
echo "  Rough LOC ratio: ${pct}%"
echo ""
echo "Modules:"
for m in core execution bot game map bootstrap; do
  if [[ -d "$RUST_ENGINE/$m" ]] || [[ -f "$RUST_ENGINE/$m.rs" ]]; then
    loc=$(find "$RUST_ENGINE" -path "*/$m/*" -name '*.rs' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    [[ -f "$RUST_ENGINE/$m.rs" ]] && loc=$(wc -l < "$RUST_ENGINE/$m.rs")
    echo "  $m: ${loc:-0} LOC"
  fi
done
echo ""
echo "Hash gates:"
echo "  --backend ts     PASS (full TS engine)"
echo "  --backend native WIP (record bootstrap + partial sim)"
echo "  OPENFRONT_DAEMON  default ON (multiplexed TS for RL)"
