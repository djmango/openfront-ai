#!/usr/bin/env bash
# Play against the RL agent in a live local OpenFront lobby.
#
#   bash scripts/play_live.sh                 # dev server already up; prompt for lobby ID
#   bash scripts/play_live.sh --restart       # kill vite + game server, restart, then play
#   bash scripts/play_live.sh --game <ID>     # skip the lobby ID prompt
#
# Before clicking Start in the browser:
#   - set Start Delay to 60+ seconds (host modal, default is 3)
#   - lower Bots / Nations for a fair fight (default is 400)
#   - wait until "AgentRL" appears in the lobby, then Start Game
#
# The MODEL overlay panel (the agent's live decisions) appears automatically
# in the browser; disable with localStorage.setItem("rlDebugOverlay", "0").

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

HOST="${HOST:-localhost:9000}"
RUN_NAME="${RUN_NAME:-ppo_v2c}"
POLICY="${POLICY:-runs/rl/$RUN_NAME/policy.pt}"
AE="${AE:-runs/ae_v3/ae_v3.pt}"
GAME=""
RESTART=0

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) RESTART=1 ;;
    --game) GAME="${2:?--game requires a lobby ID}"; shift ;;
    --policy) POLICY="$2"; shift ;;
    --run-name) RUN_NAME="$2"; POLICY="runs/rl/$RUN_NAME/policy.pt"; shift ;;
    --host) HOST="$2"; shift ;;
    --ae) AE="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
  shift
done

stop_dev_server() {
  pkill -f "tsx src/server/Server.ts" 2>/dev/null || true
  pkill -f "openfront.*vite" 2>/dev/null || true
  pkill -f "node_modules/.bin/vite" 2>/dev/null || true
  sleep 1
}

wait_for_server() {
  echo "waiting for http://$HOST ..."
  for _ in $(seq 1 60); do
    if curl -sf "http://$HOST" >/dev/null 2>&1; then
      echo "dev server ready at http://$HOST"
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for dev server (see /tmp/openfront-dev.log)" >&2
  exit 1
}

ensure_ae() {
  if [[ -f "$AE" ]]; then
    return
  fi
  mkdir -p "$(dirname "$AE")"
  echo "fetching AE checkpoint -> $AE"
  uv run python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path
dest = Path("runs/ae_v3/ae_v3.pt")
dest.parent.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("djmango/openfront-tile-autoencoder", "ae_v3.pt")
shutil.copy(p, dest)
print(f"saved {dest}")
PY
}

ensure_policy() {
  if [[ -f "$POLICY" ]]; then
    return
  fi
  mkdir -p "$(dirname "$POLICY")"
  echo "fetching policy from Hugging Face (djmango/openfront-rl/$RUN_NAME/policy.pt) -> $POLICY"
  RUN_NAME="$RUN_NAME" POLICY="$POLICY" uv run python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

run = os.environ["RUN_NAME"]
dest = Path(os.environ["POLICY"])
dest.parent.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("djmango/openfront-rl", f"{run}/policy.pt")
shutil.copy(p, dest)
print(f"saved {dest}")
PY
}

if [[ "$RESTART" -eq 1 ]]; then
  echo "stopping existing dev server ..."
  stop_dev_server
  echo "starting dev server (log: /tmp/openfront-dev.log) ..."
  (cd openfront && npm run dev > /tmp/openfront-dev.log 2>&1 &)
fi

wait_for_server

ensure_ae
ensure_policy

if [[ -z "$GAME" ]]; then
  cat <<'EOF'

Browser steps (do these before pressing Enter):
  1. Create Lobby (private)
  2. Set Start Delay to 60+ seconds  (default is 3 — not enough time to launch the bot)
  3. Lower Bots / Nations            (default 400 is brutal)
  4. Copy the lobby ID (top right)

Public lobbies on the home screen auto-start in ~5s in dev — use a private lobby instead.

EOF
  read -r -p "lobby ID: " GAME
fi

if [[ -z "$GAME" ]]; then
  echo "no lobby ID provided" >&2
  exit 1
fi

echo "joining lobby $GAME as AgentRL (policy: $POLICY)"
echo "wait for AgentRL in the lobby, then click Start Game."
exec uv run python -m rl.play --policy "$POLICY" --ckpt "$AE" --game "$GAME" --host "$HOST"
