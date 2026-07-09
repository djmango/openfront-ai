#!/usr/bin/env bash
# Open a local tunnel to TensorBoard on a RunPod pod.
#
# Usage:
#   scripts/tensorboard_tunnel.sh                 # openfront-rl4, localhost:19123
#   scripts/tensorboard_tunnel.sh openfront-rl4   # explicit pod name
#   LOCAL_PORT=19124 scripts/tensorboard_tunnel.sh openfront-rl4

set -euo pipefail

POD="${1:-openfront-v7}"
LOCAL_PORT="${LOCAL_PORT:-19123}"
REMOTE_PORT="${REMOTE_PORT:-19123}"

command -v runpodctl >/dev/null || {
  echo "runpodctl is required" >&2
  exit 1
}
command -v python3 >/dev/null || {
  echo "python3 is required" >&2
  exit 1
}

INFO="$(runpodctl ssh info "$POD")"
IP="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["ip"])' <<<"$INFO")"
PORT="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["port"])' <<<"$INFO")"
KEY="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["ssh_key"]["path"])' <<<"$INFO")"

if command -v lsof >/dev/null && lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null; then
  echo "localhost:$LOCAL_PORT is already in use. Try: LOCAL_PORT=19124 $0 $POD" >&2
  exit 1
fi

echo "Ensuring TensorBoard is running on $POD:$REMOTE_PORT..."
ssh -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=15 \
  root@"$IP" -p "$PORT" \
  "cd /workspace/openfront-ai && if ! pgrep -f 'tensorboard.*${REMOTE_PORT}' >/dev/null; then nohup tensorboard --logdir runs/rl --port ${REMOTE_PORT} --host 127.0.0.1 >/tmp/tensorboard.log 2>&1 & fi"

echo "TensorBoard: http://127.0.0.1:${LOCAL_PORT}"
echo "Press Ctrl-C to close the tunnel."
exec ssh -N \
  -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  root@"$IP" -p "$PORT"
