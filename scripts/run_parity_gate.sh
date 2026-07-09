#!/usr/bin/env bash
# Pin dedicated openfront submodule, then run native 78-record gate.
# Never touches the webbot openfront checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=parity_env.sh
source "$ROOT/scripts/parity_env.sh"
bash "$ROOT/scripts/ensure_parity_openfront.sh"
if [[ ! -e "$ROOT/records" ]]; then
  echo "[run_parity_gate] linking records from openfront-ai..."
  ln -s /Users/djmango/github/openfront-ai/records "$ROOT/records"
fi
cd "$ROOT/rust"
echo "[run_parity_gate] OPENFRONT_REPO=$OPENFRONT_REPO PARITY_COMMIT=$PARITY_COMMIT"
exec cargo test -p openfront-engine replay::tests::multi_record_parity_report --release -- --ignored --nocapture
