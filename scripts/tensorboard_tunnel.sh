#!/usr/bin/env bash
# Open a local tunnel to TensorBoard on a RunPod pod.
#
# Usage:
#   scripts/tensorboard_tunnel.sh                  # ppo_v8, localhost:19123
#   RUN_NAME=ppo_v82 scripts/tensorboard_tunnel.sh openfront-rl4
#   LOCAL_PORT=19124 scripts/tensorboard_tunnel.sh openfront-rl4

set -euo pipefail

POD="${1:-openfront-v7}"
LOCAL_PORT="${LOCAL_PORT:-19123}"
REMOTE_PORT="${REMOTE_PORT:-19123}"
RUN_NAME="${RUN_NAME:-ppo_v8}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/root/openfront-ai}"

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

echo "Ensuring the $RUN_NAME metrics bridge and TensorBoard are running on $POD..."
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
   if ! pgrep -f \"metrics_jsonl_to_tensorboard.py --metrics \$metrics\" >/dev/null; then \
     nohup \"\$python\" \"\$repo/scripts/metrics_jsonl_to_tensorboard.py\" \
       --metrics \"\$metrics\" --out-dir \"\$out\" >/tmp/tb_bridge_$RUN_NAME.log 2>&1 & \
   fi; \
   if ! pgrep -f 'tensorboard.*--port ${REMOTE_PORT}' >/dev/null; then \
     nohup \"\$repo/rust/.libtorch-venv/bin/tensorboard\" --logdir \"\$repo/runs/rl\" \
       --port ${REMOTE_PORT} --host 127.0.0.1 >/tmp/tensorboard.log 2>&1 & \
   fi"

echo "TensorBoard ($RUN_NAME): http://127.0.0.1:${LOCAL_PORT}"
echo "Press Ctrl-C to close the tunnel."
exec ssh -N \
  -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  root@"$IP" -p "$PORT"
