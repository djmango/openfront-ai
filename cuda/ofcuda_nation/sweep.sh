#!/usr/bin/env bash
# Nation spawn parity sweep: for every (map, N, nations) cell, build the ENGINE
# reference (ofcuda_spawn) + the INDEPENDENT engine oracle (ofcuda_nation), run
# the CUDA port's spawn layer (spawnall) against the reference, and cross-check
# the port's sampler against the oracle's engine-print evidence.
#
# Nothing here calls the port to validate itself: every "engine" number is
# produced by an engine-linking binary.
set -u
S=/opt/data/workspaces/skg
OUT=/opt/data/workspaces/skg/ofcuda_nation/out
REF=$OUT/refs
ORC=$OUT/oracle
CELL=$OUT/port
mkdir -p "$REF" "$ORC" "$CELL"
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"

SPAWN_BIN=$S/ofcuda_spawn/target/release/ofcuda_spawn
NATION_BIN=$S/ofcuda_nation/target/release/ofcuda_nation
SPAWNALL=$S/ofcuda_prng/target/release/spawnall
ENVSH=$S/ofcuda_env.sh
TABLE=$OUT/nation_parity.tsv
: > "$TABLE"

run() {  # map N nations
  local map=$1 n=$2 nat=$3
  local base="${map}_a${n}_nat${nat}"
  local ref="$REF/$base.spawn.txt" orc="$ORC/${map}_n${n}_nat${nat}.oracle.txt" port="$CELL/$base.dump.txt" log="$CELL/$base.spawnall.log"
  local st=OK notes=""

  if [ ! -s "$ref" ]; then
    "$SPAWN_BIN" --map "$map" --agents "$n" --nations "$nat" --seed parity --out "$ref" >/dev/null 2>&1 \
      || { notes="ref generation FAILED"; st=FAIL; printf "%s\t%s\t%s\t%s\t%s\n" "$map" "$n" "$nat" "$st" "$notes" >> "$TABLE"; return; }
  fi
  if [ ! -s "$orc" ]; then
    "$NATION_BIN" --map "$map" --agents "$n" --nations "$nat" --seed parity --out "$orc" >/dev/null 2>&1 \
      || { notes="oracle FAILED"; st=FAIL; printf "%s\t%s\t%s\t%s\t%s\n" "$map" "$n" "$nat" "$st" "$notes" >> "$TABLE"; return; }
  fi

  bash "$ENVSH" "$SPAWNALL" "$ref" --dump "$port" > "$log" 2>&1
  local rc=$?
  local all natline
  all=$(grep -o "ALL_BIT_EXACT [a-z]*" "$log" | tail -1 | awk '{print $2}')
  natline=$(grep -E "^nation[0-9]+ sid=" "$log" | head -3 | tr '\n' ';')
  local eng_natur nat_natur eng_tiles port_tiles
  eng_natur=$(grep -c '^NATION ' "$orc")
  nat_natur=$(grep -oE "engine nations [0-9]+  port nations [0-9]+" "$log" | head -1)
  eng_tiles=$(grep -oE "^NATIONOWNED [0-9]+ n=[0-9]+" "$orc" | sed 's/.*n=//' | tr '\n' ',')
  port_tiles=$(grep -oE "^nationowned [0-9]+ .*" "$port" | awk '{print NF-2}' | tr '\n' ',')
  local plane_nonzero derived id_eq
  plane_nonzero=$(grep -oE "^PLANE_AT_NATION_TICK_NONZERO [0-9]+" "$orc" | awk '{print $2}')
  derived=$(grep -oE "derived_eq_engine=[01]" "$orc" | cut -d= -f2 | tr '\n' ',')
  id_eq=$(grep -oE "id_eq_engine=[01]" "$orc" | cut -d= -f2 | tr '\n' ',')

  # per-nation sampler evidence: engine tries (oracle) vs port tries (dump)
  local eng_tries port_tries
  eng_tries=$(grep -oE "^NATIONTRIES [0-9]+ [0-9]+" "$orc" | awk '{print $3}' | tr '\n' ',')
  port_tries=$(grep -oE "^nationtries [0-9]+ [0-9]+" "$port" | awk '{print $3}' | tr '\n' ',')

  if [ "$rc" != 0 ] || [ "$all" != "true" ]; then st=FAIL; fi
  # cross-checks that must hold regardless of the port's own tallies
  if [ "$plane_nonzero" != "0" ]; then st=FAIL; notes="$notes PLANE_NOT_EMPTY"; fi
  case "$derived" in *0*) st=FAIL; notes="$notes ORACLE_DERIVED_TILE!=ENGINE";; esac
  case "$id_eq" in *0*) st=FAIL; notes="$notes ORACLE_ID!=ENGINE";; esac
  if [ "$eng_tries" != "$port_tries" ]; then st=FAIL; notes="$notes TRIES engine=$eng_tries port=$port_tries"; fi

  printf "%s\t%s\t%s\t%s\tnations=%s\ttiles(engine)=%s tiles(port)=%s\ttries engine=%s port=%s\tderived_eq=%s id_eq=%s\t%s\n" \
    "$map" "$n" "$nat" "$st" "$nat_natur" "$eng_tiles" "$port_tiles" "$eng_tries" "$port_tries" "$derived" "$id_eq" "$notes" >> "$TABLE"
  printf "%s nations=%s n=%s -> %s %s\n" "$map" "$nat" "$n" "$st" "$natline"
}

for map in pangaea africa europe world thebox onion; do
  for n in 7 18 64; do
    for nat in 1 2; do
      run "$map" "$n" "$nat"
    done
  done
done
# The map that declares 0 nations (`nations: []`): the engine still creates the
# requested nations (`create_random_nations` fabricates them, `coordinates:None`)
# and places them with the generic `find_spawn`. nations=0 is the control.
for n in 7 18 64; do
  for nat in 0 1 2; do
    run baikalnukewars "$n" "$nat"
  done
done
echo "=== table: $TABLE ==="
cat "$TABLE"
