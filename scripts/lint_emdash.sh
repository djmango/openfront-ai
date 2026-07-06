#!/usr/bin/env bash
# Fail if any em dashes (U+2014) exist in the repo.
#
# Excluded:
#   openfront/  - upstream submodule, not ours to clean
#   patches/    - diff context lines must quote upstream code verbatim
#
# Usage:
#   scripts/lint_emdash.sh            lint all tracked files
#   scripts/lint_emdash.sh --staged   lint staged changes only (pre-commit)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DASH=$(printf '\xe2\x80\x94')
EXCLUDE=(':!openfront' ':!patches')

if [[ "${1:-}" == "--staged" ]]; then
  matches=$(git diff --cached -U0 -- "${EXCLUDE[@]}" | grep -n "^+.*${DASH}" || true)
else
  matches=$(git grep -nI "${DASH}" -- "${EXCLUDE[@]}" || true)
fi

if [[ -n "$matches" ]]; then
  echo "em dashes found (use '-', ',' or ':' instead):"
  echo "$matches"
  exit 1
fi
