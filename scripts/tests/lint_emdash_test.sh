#!/usr/bin/env bash
# Smoke test for scripts/lint_emdash.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LINT="$ROOT/scripts/lint_emdash.sh"
TMP="$(mktemp -d)"
PLANT="$ROOT/docs/.emdash_lint_probe.html"
cleanup() { rm -f "$PLANT"; rm -rf "$TMP"; }
trap cleanup EXIT

EMDASH="$(printf '\xe2\x80\x94')"
ENT_NAME='&'md'ash;'

# Repo must currently be clean.
bash "$LINT"

# Positive: planted em dash is detected.
printf '<p>probe %s fail</p>\n' "$EMDASH" >"$PLANT"
if bash "$LINT" "$PLANT"; then
  echo "FAIL: expected lint to catch planted em dash" >&2
  exit 1
fi
rm -f "$PLANT"

# Entity form.
printf '<p>probe %s fail</p>\n' "$ENT_NAME" >"$PLANT"
if bash "$LINT" "$PLANT"; then
  echo "FAIL: expected lint to catch HTML entity em dash" >&2
  exit 1
fi
rm -f "$PLANT"

# Devlog itself must stay clean.
bash "$LINT" docs/devlog.html

echo "lint_emdash_test: ok"
