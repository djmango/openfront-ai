#!/usr/bin/env bash
# Decrypts secrets/*.enc.yaml (SOPS+age, see .sops.yaml) into shell env vars
# for the current session. Requires SOPS_AGE_KEY (the age private key) to be
# set as an env var/secret - never commit that key anywhere in this repo.
#
# Usage:
#   source scripts/load_secrets.sh
# then use $RUNPOD_API_KEY etc. Add more keys to secrets/*.enc.yaml and this
# script's export list as new secrets are needed - don't create a second,
# differently-shaped secrets mechanism.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOPS_BIN="$(command -v sops || echo "$HOME/.local/bin/sops")"

if [ -z "${SOPS_AGE_KEY:-}" ] && [ -z "${SOPS_AGE_KEY_FILE:-}" ]; then
  echo "[load_secrets] SOPS_AGE_KEY (or SOPS_AGE_KEY_FILE) not set - can't decrypt secrets/*.enc.yaml. Skipping." >&2
  return 0 2>/dev/null || exit 0
fi

if [ ! -x "$SOPS_BIN" ]; then
  echo "[load_secrets] sops not found on PATH or at ~/.local/bin/sops - install it first." >&2
  return 1 2>/dev/null || exit 1
fi

for f in "$ROOT"/secrets/*.enc.yaml; do
  [ -f "$f" ] || continue
  # --output-type json avoids a pyyaml dependency - stdlib json is always
  # available, sops can re-emit any encrypted format as any output format.
  while IFS='=' read -r key val; do
    [ -z "$key" ] && continue
    export "$(echo "$key" | tr '[:lower:]' '[:upper:]')=$val"
  done < <("$SOPS_BIN" --decrypt --output-type json "$f" | python3 -c "
import sys, json
for k, v in json.load(sys.stdin).items():
    print(f'{k}={v}')
" 2>/dev/null)
done

echo "[load_secrets] loaded secrets from $ROOT/secrets/*.enc.yaml"
