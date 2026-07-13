#!/usr/bin/env bash
# Fail-stop, isolated V8 control vs V8.1 trainer orchestration for 4-GPU RunPod hosts.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  SOURCE_CHECKPOINT=/path/to/latest.safetensors \
    bash scripts/runpod_v81_ab.sh [--dry-run]

Required:
  SOURCE_CHECKPOINT       Frozen starting .safetensors checkpoint.

Important overrides:
  RUN_ROOT                New, non-existent experiment directory.
  TRAINER_BIN             Integrated oftrain binary (default: rust/target/release/oftrain).
  CONTROL_GPUS / V81_GPUS Physical GPU pairs (defaults: 0,1 and 2,3).
  NUM_ENVS, ROLLOUT_LEN, MINIBATCH_SIZE, STAGE, UPDATES
  REPORT_EVERY, CKPT_EVERY, EVAL_EVERY, EVAL_EPISODES
  COMMON_EXTRA_ARGS       Whitespace-separated arguments applied to both arms.
  CONTROL_EXTRA_ARGS      Control-only arguments (default: none).
  V81_EXTRA_ARGS          V8.1-only arguments (default: persistent candidate flags).

Arm-specific arguments cannot override stage, env count, rollout, minibatches,
updates, checkpoint paths, devices, seeds, eval cadence, or checkpoint cadence.
The script never restarts a failed process. Any non-zero exit, including a CUDA
failure, stops the peer and exits non-zero.
EOF
}

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { usage >&2; exit 2; }

die() {
  echo "FATAL: $*" >&2
  exit 1
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_positive() {
  is_uint "$2" && [ "$2" -gt 0 ] || die "$1 must be a positive integer (got '$2')"
}

validate_gpu_pair() {
  local label="$1" value="$2"
  [[ "$value" =~ ^[0-9]+,[0-9]+$ ]] || die "$label must contain exactly two numeric GPU IDs"
  [ "${value%,*}" != "${value#*,}" ] || die "$label contains a duplicate GPU ID"
}

contains_protected_arg() {
  local token name
  for token in "$@"; do
    for name in \
      stage num-envs num-gpus rollout-len minibatches updates ckpt-dir resume init \
      device ckpt-every log-every eval-every eval-episodes eval-device async-eval \
      engine node-fraction auto-scale-envs min-envs max-envs autoscale-check-every
    do
      if [ "$token" = "--$name" ] || [[ "$token" == "--$name="* ]]; then
        return 0
      fi
    done
  done
  return 1
}

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"
[ -n "$SOURCE_CHECKPOINT" ] || die "SOURCE_CHECKPOINT is required"
SOURCE_CHECKPOINT="$(realpath -e "$SOURCE_CHECKPOINT")" || die "source checkpoint does not exist"
case "$SOURCE_CHECKPOINT" in
  *.safetensors) CKPT_EXT=".safetensors"; SOURCE_STATE="${SOURCE_CHECKPOINT%.safetensors}.state.json" ;;
  *) die "SOURCE_CHECKPOINT must end in .safetensors" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRAINER_BIN="${TRAINER_BIN:-$REPO_DIR/rust/target/release/oftrain}"
REPORTER="${REPORTER:-$SCRIPT_DIR/v81_ab_report.py}"
RUN_ID="${RUN_ID:-v81-ab-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/workspace/oftrain-ab/$RUN_ID}"
CONTROL_GPUS="${CONTROL_GPUS:-0,1}"
V81_GPUS="${V81_GPUS:-2,3}"
NUM_ENVS="${NUM_ENVS:-24}"
ROLLOUT_LEN="${ROLLOUT_LEN:-32}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
UPDATES="${UPDATES:-1000000}"
REPORT_EVERY="${REPORT_EVERY:-25}"
CKPT_EVERY="${CKPT_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
HEALTH_INTERVAL_SECONDS="${HEALTH_INTERVAL_SECONDS:-15}"
NODE_FRACTION="${NODE_FRACTION:-0}"
FINE_CKPT="${FINE_CKPT:-$REPO_DIR/weights/ae/ae_v31_d8c32.encoder.safetensors}"
COARSE_CKPT="${COARSE_CKPT:-$REPO_DIR/weights/ae/ae_v31_d16c32.encoder.safetensors}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:---persistent-actors --work-conserving-actors --compact-rollout --pipeline-groups=true}"
CONTROL_EXTRA_ARGS="${CONTROL_EXTRA_ARGS:-}"
V81_EXTRA_ARGS="${V81_EXTRA_ARGS:---v81-dom-coef 0.25 --v81-dominant-loss=true --v81-curriculum}"

validate_gpu_pair CONTROL_GPUS "$CONTROL_GPUS"
validate_gpu_pair V81_GPUS "$V81_GPUS"
IFS=, read -r control_gpu_a control_gpu_b <<<"$CONTROL_GPUS"
IFS=, read -r v81_gpu_a v81_gpu_b <<<"$V81_GPUS"
for gpu in "$control_gpu_a" "$control_gpu_b"; do
  [ "$gpu" != "$v81_gpu_a" ] && [ "$gpu" != "$v81_gpu_b" ] \
    || die "CONTROL_GPUS and V81_GPUS must be disjoint"
done

for pair in \
  "NUM_ENVS:$NUM_ENVS" "ROLLOUT_LEN:$ROLLOUT_LEN" \
  "MINIBATCH_SIZE:$MINIBATCH_SIZE" "UPDATES:$UPDATES" \
  "REPORT_EVERY:$REPORT_EVERY" "CKPT_EVERY:$CKPT_EVERY" \
  "EVAL_EPISODES:$EVAL_EPISODES" "HEALTH_INTERVAL_SECONDS:$HEALTH_INTERVAL_SECONDS"
do
  require_positive "${pair%%:*}" "${pair#*:}"
done
is_uint "$EVAL_EVERY" || die "EVAL_EVERY must be a non-negative integer"

SOURCE_UPDATE=0
SOURCE_STAGE=0
if [ -f "$SOURCE_STATE" ]; then
  read -r SOURCE_UPDATE SOURCE_STAGE < <(
    python3 - "$SOURCE_STATE" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(state["update"]), int(state["stage"]))
PY
  ) || die "source state sidecar is not valid JSON"
fi
STAGE="${STAGE:-$SOURCE_STAGE}"
is_uint "$STAGE" || die "STAGE must be a non-negative integer"
if [ -f "$SOURCE_STATE" ] && [ "$STAGE" -ne "$SOURCE_STAGE" ]; then
  die "STAGE=$STAGE conflicts with frozen sidecar stage=$SOURCE_STAGE"
fi
[ "$UPDATES" -gt "$SOURCE_UPDATE" ] \
  || die "UPDATES=$UPDATES must exceed source update $SOURCE_UPDATE"

MINIBATCHES=$((NUM_ENVS * ROLLOUT_LEN / MINIBATCH_SIZE))
[ "$MINIBATCHES" -ge 1 ] || MINIBATCHES=1
[ $((NUM_ENVS * ROLLOUT_LEN % MINIBATCH_SIZE)) -eq 0 ] \
  || die "NUM_ENVS*ROLLOUT_LEN must be divisible by MINIBATCH_SIZE"

read -r -a common_extra <<<"$COMMON_EXTRA_ARGS"
read -r -a control_extra <<<"$CONTROL_EXTRA_ARGS"
read -r -a v81_extra <<<"$V81_EXTRA_ARGS"
contains_protected_arg "${common_extra[@]}" && die "COMMON_EXTRA_ARGS overrides an A/B-controlled argument"
contains_protected_arg "${control_extra[@]}" && die "CONTROL_EXTRA_ARGS overrides an A/B-controlled argument"
contains_protected_arg "${v81_extra[@]}" && die "V81_EXTRA_ARGS overrides an A/B-controlled argument"

common_args=(
  --engine native
  --node-fraction "$NODE_FRACTION"
  --num-envs "$NUM_ENVS"
  --num-gpus 2
  --rollout-len "$ROLLOUT_LEN"
  --minibatches "$MINIBATCHES"
  --stage "$STAGE"
  --updates "$UPDATES"
  --device cuda:0
  --ckpt-every "$CKPT_EVERY"
  --log-every 1
  --eval-every "$EVAL_EVERY"
  --eval-episodes "$EVAL_EPISODES"
  --amp
  --fp16-rollout
  --foveate
  --ckpt "$FINE_CKPT"
  --coarse-ckpt "$COARSE_CKPT"
  "${common_extra[@]}"
)

RUN_ROOT="$(realpath -m "$RUN_ROOT")"
case "$SOURCE_CHECKPOINT" in
  "$RUN_ROOT"/*) die "RUN_ROOT cannot contain the source checkpoint" ;;
esac
[ ! -e "$RUN_ROOT" ] || die "RUN_ROOT already exists; refusing to overwrite it: $RUN_ROOT"

source_hash="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
control_dir="$RUN_ROOT/control"
v81_dir="$RUN_ROOT/v81"
control_seed="$control_dir/checkpoints/source_seed$CKPT_EXT"
v81_seed="$v81_dir/checkpoints/source_seed$CKPT_EXT"
control_log="$control_dir/trainer.log"
v81_log="$v81_dir/trainer.log"

control_cmd=("$TRAINER_BIN" "${common_args[@]}" --ckpt-dir "$control_dir/checkpoints" --resume "$control_seed" "${control_extra[@]}")
v81_cmd=("$TRAINER_BIN" "${common_args[@]}" --ckpt-dir "$v81_dir/checkpoints" --resume "$v81_seed" "${v81_extra[@]}")

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'source_sha256=%s\nsource_update=%s\nstage=%s\nminibatches=%s\n' \
    "$source_hash" "$SOURCE_UPDATE" "$STAGE" "$MINIBATCHES"
  printf 'control env CUDA_VISIBLE_DEVICES=%q TMPDIR=%q ' "$CONTROL_GPUS" "$control_dir/tmp"
  printf '%q ' "${control_cmd[@]}"
  printf '\n'
  printf 'v81 env CUDA_VISIBLE_DEVICES=%q TMPDIR=%q ' "$V81_GPUS" "$v81_dir/tmp"
  printf '%q ' "${v81_cmd[@]}"
  printf '\n'
  exit 0
fi

[ -x "$TRAINER_BIN" ] || die "trainer binary is not executable: $TRAINER_BIN"
[ -x "$REPORTER" ] || die "reporter is not executable: $REPORTER"
[ -f "$FINE_CKPT" ] || die "fine AE checkpoint not found: $FINE_CKPT"
[ -f "$COARSE_CKPT" ] || die "coarse AE checkpoint not found: $COARSE_CKPT"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required outside dry-run mode"
available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
for gpu in "$control_gpu_a" "$control_gpu_b" "$v81_gpu_a" "$v81_gpu_b"; do
  printf '%s\n' "$available_gpus" | awk -v wanted="$gpu" '$1 == wanted { found=1 } END { exit !found }' \
    || die "GPU $gpu is not available"
done

mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
mkdir -p "$control_dir/checkpoints" "$control_dir/tmp" "$v81_dir/checkpoints" "$v81_dir/tmp" "$RUN_ROOT/source"
frozen_source="$RUN_ROOT/source/checkpoint$CKPT_EXT"
cp --reflink=auto --preserve=mode,timestamps "$SOURCE_CHECKPOINT" "$frozen_source"
[ "$(sha256sum "$frozen_source" | awk '{print $1}')" = "$source_hash" ] \
  || die "frozen checkpoint hash mismatch"
chmod a-w "$frozen_source"
cp --reflink=auto "$frozen_source" "$control_seed"
cp --reflink=auto "$frozen_source" "$v81_seed"
chmod a-w "$control_seed" "$v81_seed"
if [ -f "$SOURCE_STATE" ]; then
  frozen_state="$RUN_ROOT/source/checkpoint.state.json"
  cp --reflink=auto "$SOURCE_STATE" "$frozen_state"
  chmod a-w "$frozen_state"
  cp --reflink=auto "$frozen_state" "${control_seed%$CKPT_EXT}.state.json"
  cp --reflink=auto "$frozen_state" "${v81_seed%$CKPT_EXT}.state.json"
  chmod a-w "${control_seed%$CKPT_EXT}.state.json" "${v81_seed%$CKPT_EXT}.state.json"
fi

python3 - "$RUN_ROOT/manifest.json" "$SOURCE_CHECKPOINT" "$source_hash" "$SOURCE_UPDATE" \
  "$STAGE" "$NUM_ENVS" "$ROLLOUT_LEN" "$MINIBATCHES" "$CONTROL_GPUS" "$V81_GPUS" \
  "$CKPT_EVERY" "$REPORT_EVERY" "${control_cmd[*]}" "${v81_cmd[*]}" <<'PY'
import json, sys, time
(path, source, digest, source_update, stage, envs, rollout, minibatches,
 control_gpus, v81_gpus, ckpt_every, report_every, control_cmd, v81_cmd) = sys.argv[1:]
manifest = {
    "schema": 1,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source": {"path": source, "sha256": digest, "update": int(source_update)},
    "parity": {
        "stage": int(stage), "num_envs_per_gpu": int(envs),
        "rollout_len": int(rollout), "minibatches": int(minibatches),
        "seed_scheme": "oftrain fixed tch seed 0; env workers w{index}-ep{episode}",
    },
    "control": {"gpus": control_gpus, "command": control_cmd},
    "v81": {"gpus": v81_gpus, "command": v81_cmd},
    "ckpt_every": int(ckpt_every),
    "report_every": int(report_every),
    "restart_policy": "never",
}
open(path, "w", encoding="utf-8").write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

events="$RUN_ROOT/events.jsonl"
event() {
  local kind="$1" arm="${2:-}" detail="${3:-}"
  python3 - "$events" "$kind" "$arm" "$detail" <<'PY'
import json, sys, time
path, kind, arm, detail = sys.argv[1:]
row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind}
if arm:
    row["arm"] = arm
if detail:
    row["detail"] = detail
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

control_pid=""
v81_pid=""
reporter_pid=""
stop_children() {
  trap - INT TERM EXIT
  for pid in "$control_pid" "$v81_pid" "$reporter_pid"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
}
trap 'event interrupted "" "signal received"; stop_children; exit 130' INT TERM
trap stop_children EXIT

(
  cd "$REPO_DIR"
  exec env CUDA_VISIBLE_DEVICES="$CONTROL_GPUS" TMPDIR="$control_dir/tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUST_BACKTRACE=1 \
    "${control_cmd[@]}"
) >"$control_log" 2>&1 &
control_pid=$!
(
  cd "$REPO_DIR"
  exec env CUDA_VISIBLE_DEVICES="$V81_GPUS" TMPDIR="$v81_dir/tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUST_BACKTRACE=1 \
    "${v81_cmd[@]}"
) >"$v81_log" 2>&1 &
v81_pid=$!
"$REPORTER" \
  --control-metrics "$control_dir/checkpoints/metrics.jsonl" \
  --control-log "$control_log" \
  --v81-metrics "$v81_dir/checkpoints/metrics.jsonl" \
  --v81-log "$v81_log" \
  --start-update "$SOURCE_UPDATE" \
  --every "$REPORT_EVERY" \
  --jsonl "$RUN_ROOT/comparisons.jsonl" \
  --latest-markdown "$RUN_ROOT/latest-comparison.md" \
  >"$RUN_ROOT/reporter.log" 2>&1 &
reporter_pid=$!
event launched "" "control_pid=$control_pid v81_pid=$v81_pid reporter_pid=$reporter_pid"

failure=0
while [ -n "$control_pid" ] || [ -n "$v81_pid" ]; do
  if ! kill -0 "$reporter_pid" 2>/dev/null; then
    set +e; wait "$reporter_pid"; reporter_rc=$?; set -e
    event process_exit reporter "rc=$reporter_rc"
    failure=1
    break
  fi
  for arm in control v81; do
    if [ "$arm" = control ]; then pid="$control_pid"; log="$control_log"; else pid="$v81_pid"; log="$v81_log"; fi
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      set +e; wait "$pid"; rc=$?; set -e
      event process_exit "$arm" "rc=$rc"
      if [ "$arm" = control ]; then control_pid=""; else v81_pid=""; fi
      if [ "$rc" -ne 0 ]; then
        if python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
raise SystemExit(0 if re.search(r"cuda|cuinit|cudnn|device-side assert|c10_cuda", text, re.I) else 1)
PY
        then
          event cuda_failure "$arm" "fail-stop; no restart"
        fi
        failure=1
        break 2
      fi
    fi
  done
  event health "" "control=${control_pid:-exited} v81=${v81_pid:-exited} reporter=$reporter_pid"
  sleep "$HEALTH_INTERVAL_SECONDS"
done

if [ "$failure" -ne 0 ]; then
  event fail_stop "" "stopping remaining processes"
  stop_children
  set +e
  [ -n "$control_pid" ] && wait "$control_pid"
  [ -n "$v81_pid" ] && wait "$v81_pid"
  set -e
else
  # Let the reporter consume final flushed lines before stopping it.
  sleep 3
  kill "$reporter_pid" 2>/dev/null || true
  set +e; wait "$reporter_pid"; set -e
fi
reporter_pid=""

current_source_hash="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
if [ "$current_source_hash" != "$source_hash" ]; then
  event source_checkpoint_changed "" "expected=$source_hash actual=$current_source_hash"
  die "source checkpoint changed during the run"
fi
event completed "" "status=$failure source_sha256=$source_hash"
trap - EXIT
exit "$failure"
