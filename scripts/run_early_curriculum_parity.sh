#!/usr/bin/env bash
# Early-curriculum hash-parity sweep: every-tick native-vs-TS over a set of
# zero-nation tribe-bot records. New file (additive); does not modify existing scripts.
set -uo pipefail
ROOT="/opt/data/workspaces/skg/openfront-ai"
cd "$ROOT"
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"

OUT_DIR="${1:-records/early-curriculum-parity}"
MAX_TICKS="${MAX_TICKS:-4500}"
EVERY="${EVERY:-1}"
JOBS="${JOBS:-4}"
RES="/tmp/early_curr_parity_results.tsv"
: >"$RES"

run_one() {
  local rec="$1"
  local id out st tick layer
  id="$(basename "$rec" | sed -E 's/\.json(gz)?$//')"
  out="/tmp/early_curr_parity.$id.out"
  bash "$ROOT/scripts/hash_parity.sh" "$rec" --every "$EVERY" --max-ticks "$MAX_TICKS" \
    >"$out" 2>"/tmp/early_curr_parity.$id.err"
  st=$?
  tick="$(grep -oP 'DIVERGENCE_TICK=\K[0-9]+' "$out" | tail -1 || true)"
  layer="$(grep -oP 'DIVERGENCE_LAYER=\K\S+' "$out" | tail -1 || true)"
  local pass=false
  [[ $st -eq 0 ]] && pass=true
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$pass" "${tick:-}" "${layer:-}" "$st" >>"$RES"
}
export -f run_one
export ROOT MAX_TICKS EVERY RES

find "$OUT_DIR" -maxdepth 1 -type f -name '*.json.gz' | sort \
  | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "=== RESULTS (every=$EVERY max_ticks=$MAX_TICKS jobs=$JOBS) ==="
sort "$RES"