#!/usr/bin/env bash
# Fail if any em dashes exist in tracked sources.
# Catches U+2014 and common HTML/numeric entities for the same glyph.
# Prefer ASCII '-' / ',' / ':' / parentheses in prose and comments.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# UTF-8 for U+2014 (bash $'\uXXXX' is not portable across versions).
DASH="$(printf '\xe2\x80\x94')"
# Build entity needles without writing the searchable literals into this file.
_ent_name=mdash
_ent_dec=8212
_ent_hex=2014
ENT_NAME="&${_ent_name};"
ENT_DEC="&#${_ent_dec};"
ENT_HEX="&#x${_ent_hex};"
ENT_HEX_UP="&#X${_ent_hex};"
ENTITY_RE="${ENT_NAME}|${ENT_DEC}|${ENT_HEX}|${ENT_HEX_UP}"

EXCLUDE=(
  ':(exclude)openfront/**'
  ':(exclude)patches/**'
  ':(exclude)scripts/lint_emdash.sh'
  ':(exclude)scripts/tests/lint_emdash_test.sh'
)

MODE="all"
PATHS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --staged) MODE="staged"; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/lint_emdash.sh [--staged] [path ...]

Fail if em dashes (U+2014) or HTML entities for them appear in tracked files.
Excludes openfront/, patches/, and this linter's own sources.
Explicit file paths are scanned even if untracked (for tests / probes).
EOF
      exit 0
      ;;
    *) PATHS+=("$1"); shift ;;
  esac
done

if [[ ${#PATHS[@]} -eq 0 ]]; then
  PATHS=(.)
fi

scan_files() {
  local f
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    grep -nF "${DASH}" -- "$f" | sed "s|^|${f}:|" || true
    grep -nIE "${ENTITY_RE}" -- "$f" | sed "s|^|${f}:|" || true
  done
}

hits=""
if [[ "$MODE" == "staged" ]]; then
  hits=$(git diff --cached -U0 -- "${PATHS[@]}" \
    | grep -nE "^\+.*(${DASH}|${ENTITY_RE})" || true)
else
  # If every path is an existing file, scan those files directly (tracked or not).
  all_files=1
  for p in "${PATHS[@]}"; do
    if [[ ! -f "$p" ]]; then
      all_files=0
      break
    fi
  done
  if [[ "$all_files" -eq 1 ]]; then
    hits=$(scan_files "${PATHS[@]}" | sed '/^$/d' || true)
  else
    unicode=$(git grep -nIF "${DASH}" -- "${EXCLUDE[@]}" "${PATHS[@]}" || true)
    entities=$(git grep -nIE "${ENTITY_RE}" -- "${EXCLUDE[@]}" "${PATHS[@]}" || true)
    hits=$(printf '%s\n%s\n' "$unicode" "$entities" | sed '/^$/d' || true)
  fi
fi

if [[ -n "${hits}" ]]; then
  echo "em dashes found (use '-', ',', ':' or parentheses instead):"
  echo "${hits}"
  exit 1
fi

echo "emdash lint ok"
