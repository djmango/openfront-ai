#!/usr/bin/env bash
# Launch parity gate on homelab (remote default shell is fish  -  always invoke bash).
set -euo pipefail

HOST="${HOMELAB_HOST:-skg@homelab}"
BRANCH="${HOMELAB_BRANCH:-rust-ofrs-fast}"
PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"
PULL="${HOMELAB_PULL:-1}"

REMOTE_CMD="cd ~/openfront-ai-rust-fast"
if [[ "$PULL" == "1" ]]; then
  REMOTE_CMD+=" && git fetch origin && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}"
fi
REMOTE_CMD+=" && OPENFRONT_REPO=~/openfront-ai PARITY_COMMIT=${PARITY_COMMIT} bash scripts/homelab_parity.sh"

echo "[run_homelab_parity] host=$HOST branch=$BRANCH commit=$PARITY_COMMIT"
ssh "$HOST" bash -lc "$REMOTE_CMD"
