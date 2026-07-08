#!/usr/bin/env bash
# Hash-verify human records with engine commit pin (TS parity oracle).
set -euo pipefail
WORKTREE="${OPENFRONT_WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
export OPENFRONT_REPO="${OPENFRONT_REPO:-/Users/djmango/github/openfront-ai}"
LIMIT="${1:-5}"
BIN="$WORKTREE/rust/target/release/openfront-replay"
if [[ ! -x "$BIN" ]]; then
  (cd "$WORKTREE/rust" && cargo build --release -p openfront-engine)
fi
mapfile -t FILES < <(find "$OPENFRONT_REPO/records" -name '*.json.gz' 2>/dev/null | head -n "$LIMIT")
OK=0 FAIL=0
for f in "${FILES[@]}"; do
  if OUT=$("$BIN" "$f" --repo "$WORKTREE" --backend ts 2>&1) && echo "$OUT" | grep -q '^ok=true'; then
    OK=$((OK + 1))
    HC=$(echo "$OUT" | sed -n 's/.*hashes_checked=\([0-9]*\).*/\1/p')
    echo "PASS $(basename "$f") hashes=$HC"
  else
    FAIL=$((FAIL + 1))
    echo "FAIL $(basename "$f")"
    echo "$OUT" | tail -3
  fi
done
echo "parity: $OK passed, $FAIL failed (engine pinned per record)"
[[ $FAIL -eq 0 ]]
