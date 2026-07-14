#!/usr/bin/env bash
# Open a local tunnel to TensorBoard on a RunPod pod.
#
# Usage:
#   scripts/tensorboard_tunnel.sh  # ppo-v8-4xA40-fresh / ppo_v82
#   LOCAL_PORT=19124 scripts/tensorboard_tunnel.sh
#   OPEN_BROWSER=0 scripts/tensorboard_tunnel.sh

set -euo pipefail

POD="${1:-ppo-v8-4xA40-fresh}"
LOCAL_PORT="${LOCAL_PORT:-19123}"
REMOTE_PORT="${REMOTE_PORT:-19123}"
RUN_NAME="${RUN_NAME:-ppo_v82}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/root/openfront-ai}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

command -v python3 >/dev/null || {
  echo "python3 is required" >&2
  exit 1
}
command -v ssh >/dev/null || {
  echo "ssh is required" >&2
  exit 1
}
SCRIPT_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent)' "$0")"
REPO_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent)' "$SCRIPT_DIR")"
LOCAL_BRIDGE="$SCRIPT_DIR/metrics_jsonl_to_tensorboard.py"
TEMP_KEY=""
cleanup() {
  if [ -n "$TEMP_KEY" ]; then
    rm -f "$TEMP_KEY"
  fi
}
trap cleanup EXIT

CONNECTION=""
if command -v runpodctl >/dev/null; then
  INFO="$(runpodctl ssh info "$POD" 2>/dev/null || true)"
  CONNECTION="$(python3 -c '
import json, sys
try:
    info = json.load(sys.stdin)
    print(info["ip"], info["port"], info["ssh_key"]["path"], sep="\t")
except Exception:
    pass
' <<<"$INFO")"
fi

if [ -n "$CONNECTION" ]; then
  IFS=$'\t' read -r IP PORT KEY <<<"$CONNECTION"
else
  echo "runpodctl is not authenticated; using encrypted repository credentials."
  SOPS_BIN="$(command -v sops || true)"
  if [ -z "$SOPS_BIN" ] && [ -x "$HOME/.local/bin/sops" ]; then
    SOPS_BIN="$HOME/.local/bin/sops"
  fi
  if [ -z "$SOPS_BIN" ]; then
    echo "sops is required to decrypt secrets/runpod*.enc.yaml" >&2
    exit 1
  fi
  command -v curl >/dev/null || {
    echo "curl is required for the RunPod API fallback" >&2
    exit 1
  }

  RUNPOD_API_KEY="$("$SOPS_BIN" --decrypt --output-type json \
    "$REPO_DIR/secrets/runpod.enc.yaml" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["runpod_api_key"])')"
  PODS="$(curl -fsS -H 'content-type: application/json' \
    --data '{"query":"query Pods { myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
    "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY")"
  unset RUNPOD_API_KEY
  CONNECTION="$(python3 -c '
import json, sys
target = sys.argv[1]
for pod in json.load(sys.stdin).get("data", {}).get("myself", {}).get("pods", []):
    if target not in (pod.get("id"), pod.get("name")):
        continue
    for port in (pod.get("runtime") or {}).get("ports", []):
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            print(port["ip"], port["publicPort"], sep="\t")
            raise SystemExit
' "$POD" <<<"$PODS")"
  if [ -z "$CONNECTION" ]; then
    echo "No running RunPod named or identified by '$POD' has public SSH." >&2
    exit 1
  fi
  IFS=$'\t' read -r IP PORT <<<"$CONNECTION"

  TEMP_KEY="$(mktemp "${TMPDIR:-/tmp}/openfront-runpod-key.XXXXXX")"
  KEY="$TEMP_KEY"
  umask 077
  "$SOPS_BIN" --decrypt --output-type json "$REPO_DIR/secrets/runpod_ssh.enc.yaml" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["runpod_ssh_private_key"])' \
    > "$KEY"
  chmod 600 "$KEY"
fi

if command -v lsof >/dev/null && lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null; then
  echo "localhost:$LOCAL_PORT is already in use. Try: LOCAL_PORT=19124 $0 $POD" >&2
  exit 1
fi

echo "Deploying the metrics bridge to $POD..."
ssh -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=15 \
  root@"$IP" -p "$PORT" \
  "repo='$REMOTE_REPO_DIR'; \
   if [ ! -d \"\$repo\" ] && [ -d /workspace/openfront-ai ]; then repo=/workspace/openfront-ai; fi; \
   mkdir -p \"\$repo/scripts\"; \
   umask 022; \
   tee \"\$repo/scripts/metrics_jsonl_to_tensorboard.py\" >/dev/null; \
   chmod 755 \"\$repo/scripts/metrics_jsonl_to_tensorboard.py\"" \
  < "$LOCAL_BRIDGE"

echo "Ensuring the $RUN_NAME bridge and TensorBoard are running..."
ssh -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=15 \
  root@"$IP" -p "$PORT" \
  "repo='$REMOTE_REPO_DIR'; \
   if [ ! -d \"\$repo\" ] && [ -d /workspace/openfront-ai ]; then repo=/workspace/openfront-ai; fi; \
   python=\"\$repo/rust/.libtorch-venv/bin/python\"; \
   metrics=\"\$repo/rust/checkpoints/$RUN_NAME/metrics.jsonl\"; \
   out=\"\$repo/runs/rl/$RUN_NAME\"; \
   mkdir -p \"\$out\"; \
   if ! \"\$python\" -c 'import tensorboard' >/dev/null 2>&1; then \
     \"\$python\" -m pip install --quiet tensorboard; \
   fi; \
   if ! pgrep -f \"metrics_jsonl_to_tensorboard.py --metrics \$metrics\" >/dev/null; then \
     nohup \"\$python\" \"\$repo/scripts/metrics_jsonl_to_tensorboard.py\" \
       --metrics \"\$metrics\" --out-dir \"\$out\" >/tmp/tb_bridge_$RUN_NAME.log 2>&1 & \
   fi; \
   if ! pgrep -f 'tensorboard.*--port ${REMOTE_PORT}' >/dev/null; then \
     nohup \"\$repo/rust/.libtorch-venv/bin/tensorboard\" --logdir \"\$repo/runs/rl\" \
       --port ${REMOTE_PORT} --host 127.0.0.1 >/tmp/tensorboard.log 2>&1 & \
   fi"

URL="http://127.0.0.1:${LOCAL_PORT}"
echo "TensorBoard ($RUN_NAME): $URL"
echo "Press Ctrl-C to close the tunnel."
if [ "$OPEN_BROWSER" != "0" ]; then
  (
    sleep 2
    if command -v open >/dev/null; then
      open "$URL"
    elif command -v xdg-open >/dev/null; then
      xdg-open "$URL"
    elif command -v cmd.exe >/dev/null; then
      cmd.exe /c start "$URL"
    fi
  ) >/dev/null 2>&1 &
fi
ssh -N \
  -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  root@"$IP" -p "$PORT"
