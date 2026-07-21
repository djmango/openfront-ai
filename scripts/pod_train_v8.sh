#!/usr/bin/env bash
# Compatibility shim for existing RunPod curl commands. V10 is the real launcher.
# Prefer invoking pod_train_v10.sh directly on new pods.
#
# Legacy dockerArgs often look like:
#   curl …/pod_train_v8.sh && V10_MODE=1 bash /root/pod_train_v8.sh
# Those must keep working until the pod is recreated with the v10 dockerArgs
# documented in pod_train_v10.sh.
set -uo pipefail
# v10 refuses mode envs; drop the legacy flag before exec.
unset V10_MODE V9_MODE V86_MODE V85_MODE V84_MODE V83_MODE
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT/pod_train_v10.sh" ]; then
  exec bash "$ROOT/pod_train_v10.sh" "$@"
fi
REPO_DIR="${REPO_DIR:-/root/openfront-ai}"
if [ -f "$REPO_DIR/scripts/pod_train_v10.sh" ]; then
  exec bash "$REPO_DIR/scripts/pod_train_v10.sh" "$@"
fi
# Curl-only bootstrap: dockerArgs downloaded this shim into /root/ alone.
curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v10.sh \
  -o "$ROOT/pod_train_v10.sh"
exec bash "$ROOT/pod_train_v10.sh" "$@"
