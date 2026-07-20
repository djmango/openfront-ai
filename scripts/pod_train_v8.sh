#!/usr/bin/env bash
# Compatibility shim for existing RunPod curl commands. V10 is the real launcher.
# Prefer invoking pod_train_v10.sh directly on new pods.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/pod_train_v10.sh" "$@"
