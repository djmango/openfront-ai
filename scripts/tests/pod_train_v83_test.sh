#!/usr/bin/env bash
# Obsolete: legacy Vx mode launches were removed. V10 is the only trainer path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
! grep -q 'V83_MODE\|V84_MODE\|V85_MODE\|V86_MODE\|V9_MODE\|V10_MODE' "$ROOT/scripts/pod_train_v8.sh"
echo "$(basename "$0"): ok (legacy modes removed)"
