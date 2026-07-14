#!/usr/bin/env bash
# Thin wrapper for the parallel V9 sparse-win curriculum experiment.
# Equivalent to: V9_MODE=1 bash scripts/pod_train_v8.sh
#
#   bash scripts/pod_train_v9.sh
#   NUM_GPUS=4 bash scripts/pod_train_v9.sh
#
# RunPod dockerArgs (detached, same pattern as pod_train_v8.sh):
#   bash -c "service ssh start 2>/dev/null || /usr/sbin/sshd; nohup bash -c 'curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v9.sh -o /root/pod_train_v9.sh && NUM_GPUS=4 bash /root/pod_train_v9.sh' > /root/bootstrap.log 2>&1 & disown; sleep infinity"

set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
V9_MODE=1 exec bash "$ROOT/pod_train_v8.sh" "$@"
