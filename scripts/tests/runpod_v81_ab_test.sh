#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
orchestrator="$repo/scripts/runpod_v81_ab.sh"
reporter="$repo/scripts/v81_ab_report.py"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

bash -n "$orchestrator"
python3 - "$reporter" <<'PY'
import ast, sys
ast.parse(open(sys.argv[1], encoding="utf-8").read())
PY

printf 'frozen checkpoint bytes\n' >"$tmp/source.safetensors"
cat >"$tmp/source.state.json" <<'JSON'
{"update": 100, "stage": 3, "ent_scale": 1.0, "lr_now": 0.0001,
 "total_env_steps": 42, "recent_wins": []}
JSON
before="$(sha256sum "$tmp/source.safetensors")"

output="$(
  SOURCE_CHECKPOINT="$tmp/source.safetensors" \
  RUN_ROOT="$tmp/new-run" \
  REPO_DIR="$repo" \
  TRAINER_BIN=/bin/true \
  UPDATES=110 \
  REPORT_EVERY=5 \
  bash "$orchestrator" --dry-run
)"
after="$(sha256sum "$tmp/source.safetensors")"
[ "$before" = "$after" ]
[ ! -e "$tmp/new-run" ]
[[ "$output" == *"source_update=100"* ]]
[[ "$output" == *"stage=3"* ]]
[[ "$output" == *"CUDA_VISIBLE_DEVICES=0\\,1"* ]]
[[ "$output" == *"CUDA_VISIBLE_DEVICES=2\\,3"* ]]
[[ "$output" == *"--persistent-actors"* ]]
[[ "$output" == *"--ckpt-dir $tmp/new-run/control/checkpoints"* ]]
[[ "$output" == *"--ckpt-dir $tmp/new-run/v81/checkpoints"* ]]

printf 'legacy\n' >"$tmp/legacy.ot"
if SOURCE_CHECKPOINT="$tmp/legacy.ot" RUN_ROOT="$tmp/nope" UPDATES=110 \
  bash "$orchestrator" --dry-run >/dev/null 2>&1
then
  echo "legacy .ot unexpectedly passed new v8.1 launch validation" >&2
  exit 1
fi

if SOURCE_CHECKPOINT="$tmp/source.safetensors" RUN_ROOT="$tmp/nope" \
  CONTROL_GPUS=0,1 V81_GPUS=1,2 UPDATES=110 \
  bash "$orchestrator" --dry-run >/dev/null 2>&1
then
  echo "overlapping GPU pairs unexpectedly passed validation" >&2
  exit 1
fi

if SOURCE_CHECKPOINT="$tmp/source.safetensors" RUN_ROOT="$tmp/nope" \
  CONTROL_EXTRA_ARGS="--stage 4" UPDATES=110 \
  bash "$orchestrator" --dry-run >/dev/null 2>&1
then
  echo "protected arm argument unexpectedly passed validation" >&2
  exit 1
fi

python3 - "$reporter" "$tmp" <<'PY'
import importlib.util
import json
import pathlib
import sys

spec = importlib.util.spec_from_file_location("v81_ab_report", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tmp = pathlib.Path(sys.argv[2])
metrics = tmp / "metrics.jsonl"
log = tmp / "trainer.log"
metrics.write_text(json.dumps({
    "update": 104, "stage": 3, "loss/vf": 0.25, "loss/ent": 2.5,
    "win_rate": 0.6, "eval/win": 0.5, "eval/score": 0.75,
}) + "\n", encoding="utf-8")
log.write_text(
    "[update   104] steps/s=   81.5 decisions_total=1 eps_done=1 "
    "recent_reward=  12.25 pg=+0.1 v=0.25 ent=2.5\n",
    encoding="utf-8",
)
rows = module.read_jsonl(metrics)
stdout = module.read_stdout(log)
snapshot = module.arm_snapshot(rows, stdout, 104)
assert snapshot["rolling_win_rate"] == 0.6
assert snapshot["recent_reward"] == 12.25
assert snapshot["throughput_steps_s"] == 81.5
assert snapshot["eval"]["score"] == 0.75
assert module.deltas(snapshot, {**snapshot, "recent_reward": 14.0})["recent_reward"] == 1.75
PY

mkdir "$tmp/bin"
cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
printf '0\n1\n2\n3\n'
SH
cat >"$tmp/bin/fake-oftrain" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
ckpt_dir=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--ckpt-dir" ]; then
    ckpt_dir="$2"
    shift 2
  else
    shift
  fi
done
[ -n "$ckpt_dir" ]
for update in 100 101 102 103 104; do
  printf '{"update":%s,"stage":3,"loss/vf":0.25,"loss/ent":2.5,"win_rate":0.6,"eval/win":null,"eval/score":null}\n' \
    "$update" >>"$ckpt_dir/metrics.jsonl"
  printf '[update %5s] steps/s=%7.1f decisions_total=1 eps_done=1 recent_reward=%8.3f pg=+0.1 v=0.25 ent=2.5\n' \
    "$update" 81.5 12.25
done
printf 'arm output\n' >"$ckpt_dir/latest.safetensors"
SH
chmod +x "$tmp/bin/nvidia-smi" "$tmp/bin/fake-oftrain"
touch "$tmp/fine.safetensors" "$tmp/coarse.safetensors"

PATH="$tmp/bin:$PATH" \
SOURCE_CHECKPOINT="$tmp/source.safetensors" \
RUN_ROOT="$tmp/live-run" \
REPO_DIR="$repo" \
TRAINER_BIN="$tmp/bin/fake-oftrain" \
FINE_CKPT="$tmp/fine.safetensors" \
COARSE_CKPT="$tmp/coarse.safetensors" \
UPDATES=105 \
REPORT_EVERY=5 \
CKPT_EVERY=5 \
HEALTH_INTERVAL_SECONDS=1 \
bash "$orchestrator"

[ "$(sha256sum "$tmp/source.safetensors")" = "$before" ]
[ -f "$tmp/live-run/control/checkpoints/latest.safetensors" ]
[ -f "$tmp/live-run/v81/checkpoints/latest.safetensors" ]
[ -f "$tmp/live-run/comparisons.jsonl" ]
python3 - "$tmp/live-run/comparisons.jsonl" "$tmp/live-run/events.jsonl" <<'PY'
import json, sys
reports = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
events = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
assert reports[-1]["report_update"] == 104
assert reports[-1]["control"]["recent_reward"] == 12.25
assert reports[-1]["v81"]["throughput_steps_s"] == 81.5
assert reports[-1]["delta"]["vf_loss"] == 0.0
assert sum(row["event"] == "process_exit" for row in events) == 2
assert events[-1]["event"] == "completed"
PY

echo "runpod_v81_ab tests passed"
