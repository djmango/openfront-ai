#!/usr/bin/env bash
# Pull public human GameRecords that match the *currently deployed* OpenFront
# tip (same gitCommit the live lobbies are hashing), so parity work isn't
# polluted by older pins.
#
# Probes a handful of the newest finished Public games, takes the modal
# gitCommit, then runs fetch_games.py with --git-commit.
#
# Usage:
#   scripts/fetch_latest_human_games.sh
#   scripts/fetch_latest_human_games.sh --max-games 80 --days 3
#   scripts/fetch_latest_human_games.sh --all-modes   # include Team lobbies
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_GAMES=60
DAYS=3
MIN_PLAYERS=8
OUT="${OUT:-$ROOT/records}"
MODE="Free For All"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-games) MAX_GAMES="$2"; shift 2 ;;
    --days) DAYS="$2"; shift 2 ;;
    --min-players) MIN_PLAYERS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --all-modes) MODE=""; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

LIVE_TIP="$(
  uv run --no-project python - <<'PY'
import json, sys, time, urllib.parse, urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

API = "https://api.openfront.io/public"
UA = {"User-Agent": "openfront-ai-research (fetch_latest_human_games)"}
end = datetime.now(timezone.utc)
start = end - timedelta(hours=12)
params = urllib.parse.urlencode(
    {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "type": "Public",
        "limit": 100,
        "offset": 0,
    }
)
req = urllib.request.Request(f"{API}/games?{params}", headers=UA)
with urllib.request.urlopen(req, timeout=60) as resp:
    listed = json.load(resp)
listed.sort(key=lambda g: g.get("end") or "", reverse=True)
counts: Counter[str] = Counter()
for g in listed[:20]:
    req = urllib.request.Request(f"{API}/game/{g['game']}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        rec = json.load(resp)
    commit = (rec.get("gitCommit") or "")[:12]
    if commit:
        counts[commit] += 1
    time.sleep(0.15)
if not counts:
    sys.exit("could not probe live gitCommit from recent games")
tip, n = counts.most_common(1)[0]
total = sum(counts.values())
print(
    f"[fetch_latest] probed {total} recent games; tip={tip} ({n}/{total})",
    file=sys.stderr,
)
for c, k in counts.most_common():
    print(f"  {c}: {k}", file=sys.stderr)
print(tip)
PY
)"
echo "[fetch_latest] using LIVE_TIP=$LIVE_TIP" >&2

END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_ISO="$(
  python3 -c "from datetime import datetime,timedelta,timezone;print((datetime.now(timezone.utc)-timedelta(days=${DAYS})).strftime('%Y-%m-%dT%H:%M:%SZ'))"
)"

ARGS=(
  --start "$START_ISO"
  --end "$END_ISO"
  --out "$OUT"
  --git-commit "$LIVE_TIP"
  --min-players "$MIN_PLAYERS"
  --max-games "$MAX_GAMES"
)
if [[ -n "$MODE" ]]; then
  ARGS+=(--mode "$MODE")
fi

echo "[fetch_latest] fetch_games.py ${ARGS[*]}" >&2
cd "$ROOT"
uv run --no-project python scripts/fetch_games.py "${ARGS[@]}"
N="$(ls "$OUT/$LIVE_TIP"/*.json.gz 2>/dev/null | wc -l | tr -d ' ')"
echo "[fetch_latest] tip bucket: $OUT/$LIVE_TIP ($N games)" >&2
echo "$LIVE_TIP"
