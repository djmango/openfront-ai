#!/usr/bin/env bash
# Join a local OpenFront lobby as AgentRL via the in-browser webbot (ONNX).
#
#   bash scripts/play_live.sh --game '<lobby URL or 8-char ID>'
#   bash scripts/play_live.sh --restart --game '<URL>'
#   bash scripts/play_live.sh --headed --game '<URL>'   # visible Chromium
#
# Lobby: private, fewer bots, Asia/World/BlackSea (not Levant). Spawn phase
# is patched to 15s; game auto-starts when AgentRL joins (uses Start Delay).
# MODEL overlay:
#   localStorage.setItem("rlDebugHost", "http://localhost:8988")

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

HOST="${HOST:-localhost:9000}"
RUN_NAME="${RUN_NAME:-ppo_v81}"
DEFAULT_POLICY="rust/checkpoints/$RUN_NAME/latest.safetensors"
POLICY="${POLICY:-$DEFAULT_POLICY}"
AE="${AE:-weights/ae/ae_v31_d8c32.encoder.safetensors}"
ONNX_DIR="${ONNX_DIR:-openfront/resources/webbot/models}"
HF_POLICY_REPO="${HF_POLICY_REPO:-djmango/openfront-rl}"
HF_AE_REPO="${HF_AE_REPO:-djmango/openfront-tile-autoencoder}"
DEBUG_PORT="${DEBUG_PORT:-8988}"
GAME=""
WORKER_PATH=""
RESTART=0
HEADED=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) RESTART=1 ;;
    --headed) HEADED=1 ;;
    --game) GAME="${2:?}"; shift ;;
    --run-name)
      RUN_NAME="$2"
      POLICY="rust/checkpoints/$RUN_NAME/latest.safetensors"
      shift
      ;;
    --policy) POLICY="$2"; shift ;;
    --ae) AE="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --debug-port) DEBUG_PORT="$2"; shift ;;
    --worker-path) WORKER_PATH="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
  shift
done

# Parse lobby URL/ID. Capture game ID before any second =~ (BASH_REMATCH
# is overwritten by each match - that bug once joined game "w1").
if [[ -n "$GAME" ]]; then
  raw="$GAME"
  if [[ "$raw" =~ /game/([A-Za-z0-9]{8}) ]]; then
    GAME="${BASH_REMATCH[1]}"
    if [[ -z "$WORKER_PATH" && "$raw" =~ /(w[0-9]+)/game/ ]]; then
      WORKER_PATH="${BASH_REMATCH[1]}"
    fi
  else
    GAME="${raw//[$'\t\r\n ']}"
  fi
fi

kill_dev_servers() {
  pkill -f "tsx src/server/Server.ts" 2>/dev/null || true
  pkill -f "node_modules/.bin/vite" 2>/dev/null || true
  pkill -f "concurrently.*start:client" 2>/dev/null || true
  # Vite without strictPort used to fall back to :9001 when :9000 was busy.
  for p in 9000 9001; do
    pids=$(lsof -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)
    [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
  done
  sleep 1
}

wait_for_server() {
  # Prefer the hostname as given (vite often binds ::1 only; 127.0.0.1 fails).
  echo "waiting for http://$HOST ..."
  for _ in $(seq 1 90); do
    curl -sf "http://$HOST" >/dev/null 2>&1 && { echo "ready"; return 0; }
    sleep 1
  done
  echo "timed out (see /tmp/openfront-dev.log)" >&2
  exit 1
}

start_dev_server() {
  echo "starting server (log: /tmp/openfront-dev.log) ..."
  # Don't auto-open a second browser tab; host lobby is already open.
  (cd openfront && SKIP_BROWSER_OPEN=true npm run dev > /tmp/openfront-dev.log 2>&1 &)
}

if [[ "$RESTART" -eq 1 ]]; then
  kill_dev_servers
  start_dev_server
elif ! curl -sf "http://$HOST" >/dev/null 2>&1; then
  # Stale vite on :9001 with nothing on :9000 looks "down" - clear both.
  kill_dev_servers
  start_dev_server
elif lsof -tiTCP:9001 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "duplicate vite on :9001 detected; restarting single server ..."
  kill_dev_servers
  start_dev_server
fi
wait_for_server

# Pin openfront to the fork commit that has webbot/. Upstream HEAD has no
# webbot/, and something (IDE submodule sync) keeps resetting us there.
WEBBOT_COMMIT="${WEBBOT_COMMIT:-7225742fc1c822ee6c4e74f0eafabaa156e8e7e2}"
want="$(git -C openfront rev-parse "$WEBBOT_COMMIT" 2>/dev/null || true)"
cur="$(git -C openfront rev-parse HEAD 2>/dev/null || true)"
need_restart=0
if [[ -z "$want" || "$cur" != "$want" || ! -f openfront/src/client/webbot/main.ts ]]; then
  echo "pinning openfront to webbot commit ${WEBBOT_COMMIT:0:9} (was ${cur:0:9}) ..."
  if ! git -C openfront cat-file -e "${WEBBOT_COMMIT}^{commit}" 2>/dev/null; then
    git -C openfront remote get-url fork >/dev/null 2>&1 \
      || git -C openfront remote add fork https://github.com/djmango/OpenFrontIO.git
    git -C openfront fetch fork --quiet
  fi
  mkdir -p /tmp/webbot-models-bak
  cp -f openfront/resources/webbot/models/*.onnx /tmp/webbot-models-bak/ 2>/dev/null || true
  git -C openfront checkout -f "$WEBBOT_COMMIT"
  mkdir -p openfront/resources/webbot/models
  cp -f /tmp/webbot-models-bak/*.onnx openfront/resources/webbot/models/ 2>/dev/null || true
  need_restart=1
fi
if [[ ! -f openfront/src/client/webbot/main.ts ]]; then
  echo "still no webbot/ after pin - aborting" >&2
  exit 1
fi
# Multiplayer spawn phase: 300 ticks (~30s) -> 150 ticks (15s) for webbot tests.
spawn_patched="$(python3 - <<'PY'
from pathlib import Path
p = Path("openfront/src/core/configuration/Config.ts")
t = p.read_text()
old = """  numSpawnPhaseTurns(): number {
    if (this._gameConfig.gameType === GameType.Singleplayer) {
      return 100;
    }
    if (this.isRandomSpawn()) {
      return 150;
    }
    return 300;
  }"""
new = """  numSpawnPhaseTurns(): number {
    if (this._gameConfig.gameType === GameType.Singleplayer) {
      return 100;
    }
    if (this.isRandomSpawn()) {
      return 150;
    }
    // Local webbot testing: 15s spawn window (150 ticks @ 10/s). Upstream is 300.
    return 150;
  }"""
if old in t:
    p.write_text(t.replace(old, new, 1))
    print("1")
elif "return 150;" in t and "Local webbot testing: 15s" in t:
    print("0")
elif "return 50;" in t and "numSpawnPhaseTurns" in t:
    # Upgrade older 5s patch.
    t2 = t.replace(
        """    // Local webbot testing: 5s spawn window (50 ticks @ 10/s). Upstream is 300.
    return 50;""",
        """    // Local webbot testing: 15s spawn window (150 ticks @ 10/s). Upstream is 300.
    return 150;""",
        1,
    )
    if t2 != t:
        p.write_text(t2)
        print("1")
    else:
        print("-1")
else:
    print("-1")
PY
)"
if [[ "$spawn_patched" == "1" ]]; then
  echo "patched spawn phase to 15s (150 ticks)"
  need_restart=1
elif [[ "$spawn_patched" == "-1" ]]; then
  echo "WARN: numSpawnPhaseTurns pattern not found" >&2
fi
if [[ "$need_restart" -eq 1 ]]; then
  kill_dev_servers
  start_dev_server
  wait_for_server
fi

# Ensure policy + ONNX (re-export if policy is newer than the onnx).
mkdir -p "$ONNX_DIR"
POLICY_STATE="${POLICY%.safetensors}.state.json"
if [[ ! -f "$POLICY" || ! -f "$POLICY_STATE" ]]; then
  mkdir -p "$(dirname "$POLICY")"
  echo "fetching policy checkpoint for $RUN_NAME from HF ..."
  RUN_NAME="$RUN_NAME" POLICY="$POLICY" HF_POLICY_REPO="$HF_POLICY_REPO" uv run python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

dest = Path(os.environ["POLICY"])
repo = os.environ["HF_POLICY_REPO"]
run = os.environ["RUN_NAME"]
weights = f"{run}/latest.safetensors"
state = f"{run}/latest.state.json"
shutil.copy2(hf_hub_download(repo, weights), dest)
state_dest = dest.with_name(f"{dest.stem}.state.json")
shutil.copy2(hf_hub_download(repo, state), state_dest)
print(f"saved {dest}")
PY
fi
if [[ ! -f "$AE" ]]; then
  mkdir -p "$(dirname "$AE")"
  AE="$AE" HF_AE_REPO="$HF_AE_REPO" uv run python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
dest = Path(os.environ["AE"])
repo = os.environ["HF_AE_REPO"]
# Prefer published encoder safetensors; fall back to legacy name.
for name in ("ae_v31_d8c32.encoder.safetensors", "ae_v31_d8c32.pt"):
    try:
        src = hf_hub_download(repo, name)
        break
    except Exception:
        src = None
if src is None:
    raise SystemExit(f"could not fetch AE encoder from {repo}")
if src.endswith(".pt"):
    raise SystemExit(
        f"only found legacy .pt at {src}; publish/download .encoder.safetensors"
    )
shutil.copy(src, dest)
meta = Path(src).with_suffix(".json")
if meta.exists():
    shutil.copy(meta, dest.with_suffix(".json"))
print(f"saved {dest}")
PY
fi
if [[ ! -f "$ONNX_DIR/policy.onnx" || "$POLICY" -nt "$ONNX_DIR/policy.onnx" ]]; then
  echo "exporting ONNX ($POLICY) -> $ONNX_DIR ..."
  PYTHONPATH="$REPO_DIR" uv run python "$REPO_DIR/scripts/export_onnx.py" \
    --ae "$AE" --policy "$POLICY" --out "$ONNX_DIR"
fi

if [[ -z "$GAME" ]]; then
  cat <<'EOF'

Create a private lobby (fewer bots, Asia/World/BlackSea). Spawn phase is 15s;
game auto-starts when AgentRL joins. Paste the URL/ID.

EOF
  read -r -p "lobby: " raw
  if [[ "$raw" =~ /game/([A-Za-z0-9]{8}) ]]; then
    GAME="${BASH_REMATCH[1]}"
    if [[ -z "$WORKER_PATH" && "$raw" =~ /(w[0-9]+)/game/ ]]; then
      WORKER_PATH="${BASH_REMATCH[1]}"
    fi
  else
    GAME="${raw//[$'\t\r\n ']}"
  fi
fi

if [[ ! "$GAME" =~ ^[A-Za-z0-9]{8}$ ]]; then
  echo "bad lobby ID: $GAME" >&2
  exit 1
fi

echo "joining $GAME as AgentRL (webbot / $RUN_NAME)"
echo "game auto-starts when AgentRL joins (Start Delay from lobby)."
echo "overlay: localStorage.setItem(\"rlDebugHost\", \"http://localhost:$DEBUG_PORT\")"

uv run playwright install chromium >/dev/null 2>&1 || true
args=(
  --host "$HOST"
  --game "$GAME"
  --debug-port "$DEBUG_PORT"
  --name AgentRL
)
[[ -n "$WORKER_PATH" ]] && args+=(--worker-path "$WORKER_PATH")
[[ "$HEADED" -eq 1 ]] && args+=(--headed)
exec uv run python scripts/webbot_launcher.py "${args[@]}"
