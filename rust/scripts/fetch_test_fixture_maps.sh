#!/usr/bin/env bash
# Fetches frozen per-commit map assets (`map.bin`/`manifest.json`/etc.) that
# `rust/engine/src/core/terrain.rs`'s `map_dir_for_commit` prefers over the
# live `openfront/` submodule checkout when replaying archived records.
#
# Map binaries drift over time just like any other upstream game content
# (coastline/balance tweaks) - they are ordinary tracked assets, not pinned
# to `PARITY_COMMIT` the way TS source/behavior is. An archived record's
# hash checkpoints were computed against whatever terrain existed at its own
# `gitCommit`, so replaying it against a *later* map silently diverges from
# tick 0 of any tile-geometry-sensitive decision (spawn tile legality,
# territory shape, etc.) - not a native-vs-TS logic bug, just a stale
# terrain snapshot. `records/frozen-maps/` is gitignored (via `records/`)
# like the record fixtures themselves - fetch locally, don't commit.
#
# Usage (from openfront-ai/rust/):
#   bash scripts/fetch_test_fixture_maps.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT="0c4c7d7993c91bd058af2790c5b9f7b48fa8e90b"
COMMIT_SHORT="${COMMIT:0:12}"
OUT_ROOT="$ROOT/records/frozen-maps/$COMMIT_SHORT"

# Every map key a `records/0c4c7d7993c9/<ID>.json.gz` fixture is known to
# need a period-correct snapshot for (add more here if another archived
# record's map is found to have drifted too; see
# docs/bot-ai-parity-nation-relations/README.md for how this was
# diagnosed). `twolakes` (jby2gMJF) is confirmed drifted; `balkans`
# (fdh3gYAF/GiQovEcP/fkVh9QtC) and `world` (tCFq6nPn) are fetched
# defensively even though not individually confirmed - harmless if
# identical to the live submodule (map_dir_for_commit just prefers an
# identical frozen copy), and cheap to include in the same tarball fetch.
MAP_KEYS=(twolakes balkans world)

TARBALL_URL="https://codeload.github.com/djmango/OpenFrontIO/tar.gz/$COMMIT"

need_fetch=0
for key in "${MAP_KEYS[@]}"; do
  if [[ ! -f "$OUT_ROOT/$key/manifest.json" ]]; then
    need_fetch=1
  fi
done
if [[ "$need_fetch" -eq 0 ]]; then
  echo "[fetch_test_fixture_maps] all frozen maps already present in $OUT_ROOT"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "[fetch_test_fixture_maps] downloading $COMMIT_SHORT source tarball..."
curl -sf -m 120 "$TARBALL_URL" -o "$TMP/repo.tar.gz"
tar -xzf "$TMP/repo.tar.gz" -C "$TMP"
SRC_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'OpenFrontIO-*')"

mkdir -p "$OUT_ROOT"
for key in "${MAP_KEYS[@]}"; do
  src="$SRC_DIR/resources/maps/$key"
  dest="$OUT_ROOT/$key"
  if [[ ! -d "$src" ]]; then
    echo "[fetch_test_fixture_maps] WARNING: $key not found at $COMMIT_SHORT, skipping" >&2
    continue
  fi
  mkdir -p "$dest"
  cp "$src"/* "$dest"/
  echo "[fetch_test_fixture_maps] $key -> $dest"
done

echo "done: $(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') frozen map(s) in $OUT_ROOT"
