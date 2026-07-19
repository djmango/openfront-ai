#!/usr/bin/env bash
# Alias for the default V10 trainer (pod_train_v8.sh).
# Kept so existing RunPod dockerArgs that curl pod_train_v10.sh keep working.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/pod_train_v8.sh" "$@"
