#!/usr/bin/env bash
# Cache TypeScript outcomes, then compare all PARITY_COMMIT records natively.
# Stdout is the stable machine-readable gate report; setup logs go to stderr.
set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
# shellcheck source=parity_env.sh
source "$ROOT/scripts/parity_env.sh" >&2
bash "$ROOT/scripts/ensure_parity_openfront.sh" >&2

RECORDS_DIR="$ROOT/records/$PARITY_COMMIT"
if [[ ! -d "$RECORDS_DIR" ]]; then
  echo "[run_outcome_gate] records directory not found: $RECORDS_DIR" >&2
  exit 2
fi

CACHE_ROOT="${OUTCOME_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openfront-ai/outcomes}"
CACHE_FILE="${OUTCOME_CACHE_FILE:-$CACHE_ROOT/$PARITY_COMMIT.json}"
OUTCOME_LIMIT="${OUTCOME_LIMIT:-0}"
OUTCOME_JOBS="${OUTCOME_JOBS:-4}"
OUTCOME_RECORD_TIMEOUT_SECONDS="${OUTCOME_RECORD_TIMEOUT_SECONDS:-1800}"
if ! [[ "$OUTCOME_LIMIT" =~ ^[0-9]+$ && "$OUTCOME_JOBS" =~ ^[1-9][0-9]*$ \
  && "$OUTCOME_RECORD_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run_outcome_gate] limit must be >= 0; jobs and timeout must be >= 1" >&2
  exit 2
fi
EXPECTED_RECORDS="${OUTCOME_EXPECTED_RECORDS:-$([[ "$OUTCOME_LIMIT" -gt 0 ]] && echo "$OUTCOME_LIMIT" || echo 78)}"
REQUIRED_PASSES="${OUTCOME_REQUIRED_PASSES:-$([[ "$OUTCOME_LIMIT" -gt 0 ]] && echo 0 || echo 55)}"
mkdir -p "$(dirname "$CACHE_FILE")"

# The TS-side replay (below) is the expensive part - real engine sim of
# every record, ~1hr+ cold. If nobody's cached oracle for this exact
# PARITY_COMMIT exists locally, try pulling one from the shared HF dataset
# before paying to regenerate it; `datagen/replay.ts --outcome-oracle`
# validates schemaVersion+recordSetHash itself and only fills in gaps, so a
# stale/partial/wrong-commit download is harmless - it just gets ignored or
# topped up.
if [[ ! -s "$CACHE_FILE" && "${OUTCOME_SKIP_HF_FETCH:-0}" != "1" ]] && command -v python3 >/dev/null 2>&1; then
  echo "[run_outcome_gate] no local oracle cache, trying HF (djmango/openfront-human-games:outcome-oracle/$PARITY_COMMIT.json)..." >&2
  python3 - "$PARITY_COMMIT" "$CACHE_FILE" >&2 <<'PYEOF' || echo "[run_outcome_gate] HF fetch failed/unavailable, will regenerate" >&2
import sys
from pathlib import Path
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit(1)
commit, dest = sys.argv[1], Path(sys.argv[2])
path = hf_hub_download(
    "djmango/openfront-human-games", f"outcome-oracle/{commit}.json", repo_type="dataset"
)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(Path(path).read_bytes())
print(f"[run_outcome_gate] fetched cached oracle from HF -> {dest}")
PYEOF
fi

pushd "$ROOT" >/dev/null
"$ROOT/openfront/node_modules/.bin/tsx" "$ROOT/datagen/replay.ts" \
  --outcome-oracle \
  --records "$RECORDS_DIR" \
  --cache "$CACHE_FILE" \
  --parity-commit "$PARITY_COMMIT" \
  --limit "$OUTCOME_LIMIT" \
  --jobs "$OUTCOME_JOBS" \
  --record-timeout-seconds "$OUTCOME_RECORD_TIMEOUT_SECONDS" 1>&2
popd >/dev/null

CARGO_ARGS=(
  run
  --quiet
  --manifest-path "$ROOT/rust/Cargo.toml"
  -p openfront-engine
  --release
  --bin outcome_gate
  --
  --repo "$ROOT"
  --records "$RECORDS_DIR"
  --oracle "$CACHE_FILE"
  --parity-commit "$PARITY_COMMIT"
  --expected-records "$EXPECTED_RECORDS"
  --required-passes "$REQUIRED_PASSES"
  --jobs "$OUTCOME_JOBS"
  --record-timeout-seconds "$OUTCOME_RECORD_TIMEOUT_SECONDS"
)
if [[ "$OUTCOME_LIMIT" -gt 0 ]]; then
  CARGO_ARGS+=(--limit "$OUTCOME_LIMIT")
fi
if [[ -n "${OUTCOME_TARGET_DIR:-}" ]]; then
  export CARGO_TARGET_DIR="$OUTCOME_TARGET_DIR"
fi
exec cargo "${CARGO_ARGS[@]}"
