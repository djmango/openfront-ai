#!/usr/bin/env bash
# ROCm/AMD launcher was an unverified fork of the old CUDA pod_train path and
# never wired V10 curriculum/reward. Disabled rather than silently training
# Legacy on MI300X.
#
# See rust/oftrain/ROCM.md. When ROCm is ready, reintroduce a launcher that
# mirrors scripts/pod_train_v8.sh's V10 defaults.
set -euo pipefail
echo "FATAL: scripts/pod_train_v8_rocm.sh is disabled (unverified; not V10)." >&2
echo "Use the CUDA launcher: bash scripts/pod_train_v8.sh" >&2
echo "ROCm notes: rust/oftrain/ROCM.md" >&2
exit 1
