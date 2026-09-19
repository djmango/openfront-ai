#!/usr/bin/env bash
# Per-cell device plane dump -> MP4 + stills. Reuses the COMMITTED render script
# (/opt/data/workspaces/skg/openfront-ai/scripts/render_device_planes.py) with the
# map's REAL tile dimensions, so the aspect ratio is never squared.
#
#   tools/render_cells.sh <cells-file> <out-with-dumps> <report-dir>
#   tools/render_cells.sh <out-with-dumps> <report-dir>      (cells file defaults)
#
# Health rules (learned the hard way: a run that produced ZERO movies used to
# print its banner and exit 0):
#   * the cells argument must be a REGULAR FILE - a directory (the classic
#     swapped-argument mistake) is a hard failure, not an empty cell list;
#   * a cell list that parses to zero cells is a failure;
#   * the parsed cell count is printed, so a zero is visible in one line;
#   * after the loop, #MP4 and #stills must both equal the cell count, and
#     render_dims.tsv must have data rows - otherwise exit non-zero.
set -euo pipefail
REPO_ROOT=/opt/data/workspaces/skg            # absolute: never CWD-dependent
MX=$REPO_ROOT/ofcuda_matrix
SRC=$REPO_ROOT/openfront-ai/scripts
RENDER=$SRC/render_device_planes.py
MAPS=$REPO_ROOT/openfront-ai/openfront/resources/maps
DEFAULT_CELLS=$MX/tools/cells_renders.txt

case $# in
  2) CELLS=$DEFAULT_CELLS; OUT=$1; REP=$2 ;;
  3) CELLS=$1; OUT=$2; REP=$3 ;;
  *) echo "usage: render_cells.sh <cells-file> <out-with-dumps> <report-dir>" >&2; exit 2 ;;
esac

# --- (0) the cells argument must be a regular file ---------------------------
if [ ! -f "$CELLS" ]; then
  echo "FATAL: cells file '$CELLS' is not a regular file" >&2
  if [ -d "$CELLS" ]; then
    echo "       it is a DIRECTORY - the arguments are swapped. Correct form:" >&2
    echo "       render_cells.sh <cells-file> <out-with-dumps> <report-dir>" >&2
    echo "       e.g. render_cells.sh $DEFAULT_CELLS <out-with-dumps> <report-dir>" >&2
  fi
  exit 2
fi
[ -d "$OUT" ] || { echo "FATAL: dump dir '$OUT' does not exist" >&2; exit 2; }

N_CELLS=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "$CELLS" || true)
if [ "$N_CELLS" -eq 0 ]; then
  echo "FATAL: '$CELLS' parses to 0 cells - nothing would be rendered" >&2
  exit 2
fi
echo "rendering $N_CELLS cells from $CELLS -> movies+stills in $REP"

# --- (1) validate every input up front, fail once with the full list ---------
missing=()
[ -e "$RENDER" ] || missing+=("$RENDER")
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
n_mp4=0
n_png=0
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
  [ -s "$REP/${cell}.mp4" ] && n_mp4=$((n_mp4 + 1))
  np=$(ls -1 "$REP/stills/${cell}"_*.png 2>/dev/null | wc -l)
  if [ "$np" -eq 0 ]; then
    echo "FATAL: $cell produced no stills in $REP/stills" >&2
    exit 1
  fi
  n_png=$((n_png + np))
done < "$CELLS"

# --- (4) assert the artifacts exist and cover every cell ---------------------
n_dims=$(grep -vc '^cell' "$REP/render_dims.tsv" || true)
bad=0
if [ "$n_mp4" -ne "$N_CELLS" ]; then
  echo "FATAL: $n_mp4 movie(s) for $N_CELLS cell(s)" >&2; bad=1
fi
if [ "$n_png" -lt "$N_CELLS" ]; then
  echo "FATAL: $n_png still(s) for $N_CELLS cell(s) (expected >=1 each)" >&2; bad=1
fi
if [ "$n_dims" -eq 0 ]; then
  echo "FATAL: $REP/render_dims.tsv has 0 data rows" >&2; bad=1
fi
if [ "$bad" -ne 0 ]; then exit 1; fi
echo "OK: $n_mp4 movies, $n_png stills, $n_dims dims rows for $N_CELLS cells"
echo "movies in $REP, stills in $REP/stills, dims in $REP/render_dims.tsv"
