#!/usr/bin/env bash
# Per-cell device plane dump -> MP4 + stills. Reuses the COMMITTED render script
# (/opt/data/workspaces/skg/openfront-ai/scripts/render_device_planes.py) with the
# map's REAL tile dimensions, so the aspect ratio is never squared.
#
#   tools/render_cells.sh [cells-file] [out-with-dumps] [report-dir]
set -euo pipefail
REPO_ROOT=/opt/data/workspaces/skg            # absolute: never CWD-dependent
MX=$REPO_ROOT/ofcuda_matrix
SRC=$REPO_ROOT/openfront-ai/scripts
RENDER=$SRC/render_device_planes.py
MAPS=$REPO_ROOT/openfront-ai/openfront/resources/maps
CELLS=${1:-$MX/tools/cells_renders.txt}
OUT=${2:?usage: render_cells.sh [cells-file] <out-with-dumps> <report-dir>}
REP=${3:?usage: render_cells.sh [cells-file] <out-with-dumps> <report-dir>}

# --- (1) validate every input up front, fail once with the full list ---------
missing=()
[ -e "$RENDER" ] || missing+=("$RENDER")
[ -e "$CELLS" ]  || missing+=("$CELLS")
manifests=()
while read -r map agents nations ticks; do
  [ -z "${map:-}" ] && continue
  case "$map" in \#*) continue ;; esac
  nations=${nations:-0}
  ticks=${ticks:-60}
  man="$MAPS/$map/manifest.json"
  [ -e "$man" ] || missing+=("$man")
  manifests+=("$man")
  d="$OUT/${map}_n${agents}_nat${nations}_t${ticks}"
  for f in planes.bin terrain.bin; do
    [ -e "$d/$f" ] || missing+=("$d/$f")
  done
done < "$CELLS"
if [ ${#missing[@]} -gt 0 ]; then
  echo "FATAL: ${#missing[@]} missing input(s):" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi
mkdir -p "$REP/stills"

# --- (2) real dims per map (manifest width/height; land count is NOT used,
#         4 manifests undercount their own bytes) ------------------------------
# --- (2) real dims per map. The manifest nests the dims one level down:
#         m["map"]["width"|"height"] (also map4x/map16x). There is NO top-level
#         width/height. `num_land_tiles` is NEVER used: it undercounts the real
#         byte land count on china/tierradelfuego/unitedstates/losangeles.
dims() {
  python3 - "$MAPS/$1/manifest.json" <<'PY'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
try:
    print(int(m["map"]["width"]), int(m["map"]["height"]))
except (KeyError, TypeError):
    sys.exit(
        f"FATAL: {p}: no map.width/map.height. top-level keys = "
        f"{sorted(m)}; 'map' keys = {sorted(m.get('map', {})) if isinstance(m.get('map'), dict) else type(m.get('map'))}"
    )
PY
}

# --- (3) render, asserting W*H against the terrain size before each cell ------
: > "$REP/render_dims.tsv"
printf 'cell\tmap\tW\tH\taspect\tframes\tout_px\tterrain_ok\n' >> "$REP/render_dims.tsv"
while read -r map agents nations ticks; do
  [ -z "${map:-}" ] && continue
  case "$map" in \#*) continue ;; esac
  nations=${nations:-0}
  ticks=${ticks:-60}
  cell="${map}_n${agents}_nat${nations}_t${ticks}"
  dir="$OUT/$cell"
  read -r W H <<<"$(dims "$map")"
  tsz=$(stat -c%s "$dir/terrain.bin")
  if [ "$tsz" -ne $((W * H)) ]; then
    echo "FATAL: $cell terrain.bin is $tsz bytes, manifest says ${W}x${H}=$((W*H))" >&2
    exit 1
  fi
  echo "=== $cell ${W}x${H} (aspect $(python3 -c "print(f'{$W/$H:.4f}')"))"
  python3 "$RENDER" "$dir/planes.bin" "$dir/terrain.bin" \
      "$REP/${cell}.mp4" "$REP/stills/${cell}" "$W" "$H"
  nf=$(python3 -c "import os;print(os.path.getsize('$dir/planes.bin')//($W*$H*2))")
  px=$(python3 -c "w,h=$W,$H; s=min(800/w,800/h); print(f'{int(w*s)}x{int(h*s)}')")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tyes\n' "$cell" "$map" "$W" "$H" \
      "$(python3 -c "print(f'{$W/$H:.4f}')")" "$nf" "$px" >> "$REP/render_dims.tsv"
done < "$CELLS"
echo "movies in $REP, stills in $REP/stills, dims in $REP/render_dims.tsv"
