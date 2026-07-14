#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo/scripts/pod_train_v8.sh"

bash -n "$script"

python3 - "$script" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = [
    'RUN_NAME="${RUN_NAME:-ppo_v83}"',
    'NUM_GPUS="${NUM_GPUS:-4}"',
    "--v83-curriculum",
    "--migrate-v82-to-v83",
    "--persistent-actors",
    "--work-conserving-actors",
    "--recurrent-policy",
    "--bptt-chunk-len 16",
    "--compact-rollout",
    "--fp16-rollout",
    'V83_SOURCE_PREFIX="${V83_SOURCE_PREFIX:-ppo_v82}"',
    'NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"',
    '"$OFHF" sync-loop',
]
for item in required:
    assert item in text, f"missing V8.3 launch requirement: {item}"

target_resume = 'RESUME="--resume $CKPT_DIR/latest.safetensors"'
seed_resume = (
    'RESUME="--resume $V83_SEED_DIR/latest.safetensors '
    '--migrate-v82-to-v83"'
)
assert text.index(target_resume) < text.index(seed_resume)
assert 'cp ' not in text, "deployment must not copy over the V8.2 source"
PY

echo "pod_train_v83 tests passed"
