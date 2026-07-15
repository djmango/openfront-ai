#!/usr/bin/env bash
# Thin wrapper for the V10 anti-death-spiral curriculum.
# Equivalent to: V10_MODE=1 bash scripts/pod_train_v8.sh
#
#   bash scripts/pod_train_v10.sh
#   NUM_GPUS=4 bash scripts/pod_train_v10.sh
#
# RunPod dockerArgs (detached, same pattern as pod_train_v8.sh):
#   bash -c "service ssh start 2>/dev/null || /usr/sbin/sshd; nohup bash -c 'curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v10.sh -o /root/pod_train_v10.sh && NUM_GPUS=4 bash /root/pod_train_v10.sh' > /root/bootstrap.log 2>&1 & disown; sleep infinity"

set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
V10_MODE=1 exec bash "$ROOT/pod_train_v8.sh" "$@"
