#!/usr/bin/env bash
# Multi-record hash/bit-parity gate for tip (or any) record set.
# For each record, runs `scripts/hash_parity.sh` and reports first diverge.
#
# Unlike outcome_gate (winner tolerance), this fails a record on the first
# native-vs-TS tick/field mismatch. Use for tip full-parity work.
#
# Usage:
#   PARITY_COMMIT=dd1277e245b5 scripts/run_hash_parity_gate.sh
#   HASH_PARITY_LIMIT=5 HASH_PARITY_MAX_TICKS=3000 scripts/run_hash_parity_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=parity_env.sh
source "$ROOT/scripts/parity_env.sh" >&2
bash "$ROOT/scripts/ensure_parity_openfront.sh" >&2

RECORDS_DIR="${HASH_PARITY_RECORDS:-$ROOT/records/$PARITY_COMMIT}"
LIMIT="${HASH_PARITY_LIMIT:-0}"
MAX_TICKS="${HASH_PARITY_MAX_TICKS:-}"
EVERY="${HASH_PARITY_EVERY:-1}"
JOBS="${HASH_PARITY_JOBS:-1}"
SKIP_BEFORE="${HASH_PARITY_SKIP_BEFORE:-5}"
OUT_JSON="${HASH_PARITY_OUT:-/tmp/hash_parity_gate.$PARITY_COMMIT.json}"
# Set HASH_PARITY_USE_BISECT=1 to use daemon binary-search instead of streaming.

mapfile -t RECORDS < <(find "$RECORDS_DIR" -maxdepth 1 -type f \( -name '*.json.gz' -o -name '*.json' \) | sort)
if [[ "$LIMIT" -gt 0 ]]; then
  RECORDS=("${RECORDS[@]:0:$LIMIT}")
fi
if [[ ${#RECORDS[@]} -eq 0 ]]; then
  echo "[hash_parity_gate] no records in $RECORDS_DIR" >&2
  exit 2
fi

echo "[hash_parity_gate] ${#RECORDS[@]} record(s) jobs=$JOBS every=$EVERY max_ticks=${MAX_TICKS:-full} bisect=${HASH_PARITY_USE_BISECT:-0}" >&2

# Build tick_dump once.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/rust/target}"
cargo build --quiet --release --manifest-path "$ROOT/rust/Cargo.toml" \
  -p openfront-engine --bin tick_dump >&2

run_one() {
  local record="$1"
  local id
  id="$(basename "$record" | sed -E 's/\.json(\.gz)?$//')"
  local args=("$record")
  [[ -n "$MAX_TICKS" ]] && args+=(--max-ticks "$MAX_TICKS")
  args+=(--skip-before "$SKIP_BEFORE")
  local out="/tmp/hash_parity_gate.$id.out"
  set +e
  if [[ "${HASH_PARITY_USE_BISECT:-0}" == "1" ]]; then
    "$ROOT/scripts/hash_bisect.sh" "${args[@]}" >"$out" 2>"/tmp/hash_parity_gate.$id.err"
  else
    args+=(--every "$EVERY")
    "$ROOT/scripts/hash_parity.sh" "${args[@]}" >"$out" 2>"/tmp/hash_parity_gate.$id.err"
  fi
  local st=$?
  set -e
  local tick layer pass
  tick="$(grep -oP 'DIVERGENCE_TICK=\K[0-9]+' "$out" | tail -1 || true)"
  layer="$(grep -oP 'DIVERGENCE_LAYER=\K\S+' "$out" | tail -1 || true)"
  if [[ $st -eq 0 ]]; then pass=true; else pass=false; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$pass" "${tick:-}" "${layer:-}" "$st"
}

export -f run_one
export ROOT SKIP_BEFORE MAX_TICKS EVERY
export HASH_PARITY_USE_BISECT="${HASH_PARITY_USE_BISECT:-0}"

RESULTS_FILE="$(mktemp)"
if [[ "$JOBS" -le 1 ]]; then
  for r in "${RECORDS[@]}"; do run_one "$r" | tee -a "$RESULTS_FILE"; done
else
  printf '%s\n' "${RECORDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} | tee "$RESULTS_FILE"
fi

uv run --no-project python - "$RESULTS_FILE" "$OUT_JSON" "$PARITY_COMMIT" <<'PY'
import json, sys
path, out_path, commit = sys.argv[1], sys.argv[2], sys.argv[3]
records = []
passes = 0
for line in open(path):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        continue
    gid, pass_s, tick, layer, st = parts[:5]
    ok = pass_s == "true"
    passes += int(ok)
    records.append({
        "gameId": gid,
        "pass": ok,
        "divergenceTick": int(tick) if tick else None,
        "divergenceLayer": layer or None,
        "exitCode": int(st),
    })
report = {
    "schemaVersion": 1,
    "parityCommit": commit,
    "summary": {
        "pass": passes,
        "total": len(records),
        "gatePass": passes == len(records) and len(records) > 0,
    },
    "records": records,
}
open(out_path, "w").write(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["summary"]))
sys.exit(0 if report["summary"]["gatePass"] else 1)
PY
