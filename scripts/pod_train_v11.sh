#!/usr/bin/env bash
# Restart-proof V11 training for the Rust `oftrain` trainer on a RunPod pod.
# Schema: neighbor pack + ego-split attacks + unit pointer + LSTM + bigger trunk.
# Requires AE v3.2 no-static encoders (ae_v32_nostatic_d{8,16}c32).
# Target recipe: 2×A100 80GB SXM (override NUM_GPUS/MAX_ENVS as needed).
# Bootstraps the repo + libtorch/CUDA venv + build, resumes from a local or
# HF-synced checkpoint, crash-loops with backoff, and periodically pushes the
# latest checkpoint to `djmango/openfront-rl`.
#
#   bash scripts/pod_train_v11.sh
#   NUM_GPUS=4 bash scripts/pod_train_v11.sh
#
# Knob sources of truth (do not duplicate elsewhere):
#   1. THIS SCRIPT — launch recipe (NUM_ENVS/MAX_ENVS, EXTRA_ARGS, TRAIN_ARGS,
#      NCCL, HF sync). Optional overrides: /root/ppo_v11.env (see
#      scripts/ppo_v11.env.example).
#   2. rust/ofcore/src/curriculum.rs — stages, bots/nations, win gates,
#      V10_ENV_TARGETS floors (V11 keeps the same curriculum).
#   3. rust/oftrain clap defaults in main.rs — mirror this recipe (rollout 48,
#      BPTT 24, epochs 8, balance-train-collect, amp/fp16/compact, persistent
#      + work-conserving + recurrent, autoscale target 0.92 / max-envs 32) so a
#      bare `oftrain` matches production; pods still pass them explicitly.
#
# Production is pure-native (NODE_FRACTION=0). A non-zero mix is a slow parity
# hedge and requires an explicit opt-in: ALLOW_NODE_MIX=1 NODE_FRACTION=<frac> ...
#
# RunPod dockerArgs (detached; keep the container alive with sleep infinity).
# Pin NODE_FRACTION=0 so leftover shell/template env cannot reintroduce a mix.
# The script self-flocks; never start a second copy alongside a live trainer:
#   bash -c "service ssh start 2>/dev/null || /usr/sbin/sshd; nohup bash -c 'set -a; [ -f /root/ppo_v11.env ] && . /root/ppo_v11.env; set +a; curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/pod_train_v11.sh -o /root/pod_train_v11.sh && NUM_GPUS=2 NODE_FRACTION=0 MAX_ENVS=24 NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 bash /root/pod_train_v11.sh' > /root/bootstrap.log 2>&1 & disown; sleep infinity"
#
# If a pod fails to actually train (crash-loops immediately, "CUDA unknown
# error" in /tmp/train_$RUN_NAME.log) despite nvidia-smi looking healthy:
# terminate and relaunch on another host (known cuInit host footgun).

set -uo pipefail

RUN_NAME="${RUN_NAME:-ppo_v11}"
# Single-instance supervisor: docker CMD + manual starts must not dual-launch
# oftrain (that OOMs every GPU and takes the run down).
BOOTSTRAP_LOCK="${BOOTSTRAP_LOCK:-/tmp/${RUN_NAME}.bootstrap.lock}"
exec 9>"$BOOTSTRAP_LOCK"
if ! flock -n 9; then
  echo "FATAL: another $RUN_NAME bootstrap already holds $BOOTSTRAP_LOCK" >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-2}"
# Envs per GPU/shard. Start near the util-healthy band; autoscale can still
# grow to MAX_ENVS. (Clap default for bare oftrain remains 4 for local smokes.)
NUM_ENVS="${NUM_ENVS:-24}"
STAGE_ENV_TARGETS="${STAGE_ENV_TARGETS:-}"
# Persistent owners cannot live-spawn env workers; autoscale grows via the same
# restart_request.json path as stage env targets.
AUTO_SCALE_ENVS="${AUTO_SCALE_ENVS:-1}"
# Cap for 80GB A100. Was 24 (~74% VRAM); 32 uses the MEM_BLOCK=0.90 headroom
# so util-driven growth can continue when collect-bound at mid mem.
MAX_ENVS="${MAX_ENVS:-32}"
MIN_ENVS="${MIN_ENVS:-10}"
# Aim autoscale / ops target at the util SLO (balance handles train fill).
TARGET_GPU_UTIL="${TARGET_GPU_UTIL:-0.92}"
AUTOSCALE_CHECK_EVERY="${AUTOSCALE_CHECK_EVERY:-5}"
AUTOSCALE_STEP="${AUTOSCALE_STEP:-2}"
# Shorter rollout lowers the collect barrier; pair with high epochs + balance
# so train wall tracks collect even when episode length drifts.
ROLLOUT_LEN="${ROLLOUT_LEN:-48}"
BPTT_CHUNK_LEN="${BPTT_CHUNK_LEN:-24}"
# Floor epochs for ≥90% util when collect stretches (observed 180→350s).
# --balance-train-collect can still grow up to MAX_EPOCHS.
EPOCHS="${EPOCHS:-8}"
# Match rl/ppo.py's optimizer cadence: its --minibatch is a target sample
# count (128), while oftrain's --minibatches is a count.
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
# Always launch inside the autoscale band. Curriculum floors (e.g. 24) are
# aspirational; MAX_ENVS is the VRAM ceiling on this pod class.
if [ "$NUM_ENVS" -gt "$MAX_ENVS" ]; then NUM_ENVS=$MAX_ENVS; fi
if [ "$NUM_ENVS" -lt "$MIN_ENVS" ]; then NUM_ENVS=$MIN_ENVS; fi
MINIBATCHES=$((NUM_ENVS * ROLLOUT_LEN / MINIBATCH_SIZE))
[ "$MINIBATCHES" -ge 1 ] || MINIBATCHES=1
STAGE="${STAGE:-0}"
# Fraction (0.0-1.0) of env workers that run the real Node/TS engine
# instead of native. Default 0 (pure native). Non-zero requires ALLOW_NODE_MIX=1
# — otherwise every "quick hedge" relaunch silently tanks collect throughput.
NODE_FRACTION="${NODE_FRACTION:-0}"
ALLOW_NODE_MIX="${ALLOW_NODE_MIX:-0}"
CKPT_KEEP_LAST="${CKPT_KEEP_LAST:-48}"
DISK_WARN_PCT="${DISK_WARN_PCT:-85}"
DISK_CRIT_PCT="${DISK_CRIT_PCT:-92}"
# Inactive run dirs under rust/checkpoints/ that may be deleted when disk is
# critical (never deletes $RUN_NAME).
STALE_CKPT_RUNS="${STALE_CKPT_RUNS:-ppo_v8,ppo_v8_fast_native,ppo_v81,ppo_v82,ppo_v83,ppo_v84,ppo_v85,ppo_v86,ppo_v9,ppo_v10}"

# Validated recipe: persistent CUDA owners + compact rollout + recurrent BPTT.
# Work-conserving batching knobs (inference scheduling only):
# - wait-ms=50 / same-shape-prefer: was 2ms → ~70% singleton AE batches
# - target-batch=2: stage-25 has ~16 unique map shapes vs 14 envs/shard, so
#   target=8 never fills and always burns the full wait before dispatch
# - padding-waste=0.50: after wait, allow slightly more mixed-shape compact pad
# --balance-train-collect adapts epochs toward train_s≈collect_hwm so mean
# util stays ≥90% when collect wall drifts (see balance.rs).
MAX_EPOCHS="${MAX_EPOCHS:-12}"
BALANCE_TARGET_RATIO="${BALANCE_TARGET_RATIO:-0.95}"
EXTRA_ARGS="${EXTRA_ARGS:---amp --foveate --compact-rollout --fp16-rollout --pinned-h2d --persistent-actors --work-conserving-actors --pipeline-groups=true --actor-target-batch 2 --actor-max-wait-ms 15 --actor-max-padding-waste 0.50 --recurrent-policy --bptt-chunk-len $BPTT_CHUNK_LEN --epochs $EPOCHS --balance-train-collect --max-epochs $MAX_EPOCHS --balance-target-ratio $BALANCE_TARGET_RATIO --ckpt-every 5 --ckpt-keep-last $CKPT_KEEP_LAST --eval-every 0 --log-every 1 --coarse-ckpt ../weights/ae/ae_v32_nostatic_d16c32.encoder.safetensors --ckpt ../weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors}"

# V11 anti-death-spiral on the closeout ladder. Dense reward with softer death,
# survival / diplo-panic / combat priors, and radical win bonus so finishing
# dominates shaping. All reward stage mins = 0.
TRAIN_ARGS="--max-episode-ticks 21000 --v83-close-coef 4.0 --v83-churn-coef 0.06 --v81-dom-coef 0.25 --v81-dominant-loss=true --v81-dominance-threshold 0.30 --v81-delta-loss-dominant 5.0 --v81-min-stage 0 --v81-churn-coef 0.05 --v81-churn-window 16 --v81-churn-min-stage 0 --v84-boat-useful 0.15 --v84-boat-destroyed=-0.20 --v84-boat-cancelled=-0.03 --v84-boat-own-shore=-0.05 --v84-boat-min-stage 0 --v84-tempo-coef 0.015 --v84-tempo-min-stage 0 --v84-fast-win-coef 40.0 --v85-tempo-share-threshold 0.30 --v85-extra-win-bonus 200.0 --v85-embargo-bad-stop=-0.15 --v85-embargo-good-stop 0.02 --v85-embargo-min-stage 0 --v85-premature-retreat=-0.03 --v85-thrash-reengage=-0.03 --v85-combat-min-stage 0 --v86-delta-loss 5.5 --v86-attack-symmetric-loss --v86-skip-combat-churn --v86-death-penalty 3.0 --v10-survival-coef 0.01 --v10-diplo-panic 0.08 --v10-diplo-panic-share 0.35 --v10-diplo-panic-tick-frac 0.55 --v10-combat-action 0.02 --v10-timeout-closeout 20.0 --v10-closeout-entry 25.0"
if [ -n "$STAGE_ENV_TARGETS" ]; then
  TRAIN_ARGS="$TRAIN_ARGS --stage-env-targets $STAGE_ENV_TARGETS"
fi
if [ "$AUTO_SCALE_ENVS" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --auto-scale-envs --min-envs $MIN_ENVS --max-envs $MAX_ENVS --target-gpu-util $TARGET_GPU_UTIL --autoscale-check-every $AUTOSCALE_CHECK_EVERY --autoscale-step $AUTOSCALE_STEP"
fi

REPO_DIR="${REPO_DIR:-/root/openfront-ai}"
CKPT_DIR="$REPO_DIR/rust/checkpoints/$RUN_NAME"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-600}"
HF_REPO_ID="${HF_REPO_ID:-djmango/openfront-rl}"
HF_RUN_PREFIX="${HF_RUN_PREFIX:-$RUN_NAME}"
# The current RunPod A40 host advertises direct CUDA P2P, but its first NCCL
# collective wedges on that transport. Shared-memory transport reduced the
# same 48 MiB gradient in ~113 ms. Override only after a host-specific P2P
# smoke succeeds (for example on a validated NVLink box).
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_VERSION="2.11.0" # tch 0.24's C++ shim needs this exact version - see devlog
AE_DIR="${AE_DIR:-$REPO_DIR/weights/ae}"

# --- refuse recurring footguns early (before long bootstrap) ---
# Legacy RunPod dockerArgs still set V11_MODE=1 while curling pod_train_v8.sh.
# Ignore that flag so a reboot with a frozen CMD does not refuse to train;
# the v8 shim also unsets it. Other mode envs remain hard errors.
if [ -n "${V10_MODE:-}" ] || [ -n "${V11_MODE:-}" ]; then
  echo "WARNING: ignoring legacy mode env V10_MODE/V11_MODE (pod_train_v11.sh is already V11)."
  unset V10_MODE V11_MODE
fi
if [ -n "${V9_MODE:-}" ] || [ -n "${V86_MODE:-}" ]; then
  echo "FATAL: legacy mode env (V9_MODE/V86_MODE) is no longer supported."
  echo "       Use scripts/pod_train_v11.sh directly."
  exit 1
fi
if [ "$(python3 -c "print(1 if float('$NODE_FRACTION') > 0 else 0)")" = "1" ] \
  && [ "$ALLOW_NODE_MIX" != "1" ]; then
  echo "FATAL: NODE_FRACTION=$NODE_FRACTION without ALLOW_NODE_MIX=1."
  echo "       Production training is pure-native (NODE_FRACTION=0)."
  echo "       Node mix is a slow parity hedge — re-run with ALLOW_NODE_MIX=1 if you"
  echo "       truly need it, or unset NODE_FRACTION."
  exit 1
fi

# --- bootstrap: repo ---
# GIT_REF selects the training code. Default master; V11 pre-merge pods should
# set GIT_REF=sully/v11-plan-lstm-bc-210b (or the merged commit once on master).
GIT_REF="${GIT_REF:-master}"
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SKIP_SYNC:-}" ]; then
  # Deployed code MUST match origin/$GIT_REF — a silently-failed pull once ran
  # a whole day's training on stale code.
  git fetch origin "$GIT_REF" || true
  git checkout -B "$GIT_REF" "origin/$GIT_REF" 2>/dev/null \
    || git reset --hard "origin/$GIT_REF" || true
  git submodule update --init || true
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$GIT_REF" 2>/dev/null)" ]; then
    echo "FATAL: HEAD $(git rev-parse --short HEAD) != origin/$GIT_REF; refusing to train stale code"
    exit 1
  fi
  echo "deployed commit: $(git rev-parse --short HEAD) (GIT_REF=$GIT_REF)"
fi

# --- rust toolchain ---
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
fi
. "$HOME/.cargo/env" 2>/dev/null || true

# --- Node.js + openfront/bridge deps, ONLY if any envs will run the Node
# engine (--node-fraction > 0) - the pure-native path (the default) never
# needed this, and Bridge::spawn shells out to openfront/node_modules/.bin/
# tsx, so it must exist before oftrain is even launched, not just before
# the eventual `--engine node` call site fails. Same install lines as
# pod_train.sh (the Python-era script), which already proved this works. ---
if [ "$(python3 -c "print(1 if float('$NODE_FRACTION') > 0 else 0)")" = "1" ]; then
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    apt-get install -y nodejs >/dev/null
  fi
  [ -d openfront/node_modules ] || (cd openfront && npm install --silent)
  echo "node engine mix enabled (--node-fraction $NODE_FRACTION ALLOW_NODE_MIX=1): node $(node --version), tsx ready"
fi

# --- CUDA-linked libtorch venv (see devlog: must be exactly $TORCH_VERSION,
# tch 0.24's C++ shim calls ATen ops that don't exist in older/newer torch
# headers and fails to *compile*, not just link, against a mismatched one) ---
VENV="$REPO_DIR/rust/.libtorch-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
fi
PY_TAG="$("$VENV/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
TORCH_LIB="$VENV/lib/$PY_TAG/site-packages/torch"
NVRTC_LIB="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_INCLUDE="/usr/local/cuda/include"
CUDA_LIB="/usr/local/cuda/lib64"
if [ ! -f "$CUDA_INCLUDE/cuda_runtime_api.h" ]; then
  CUDA_INCLUDE="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_runtime/include"
  CUDA_LIB="$VENV/lib/$PY_TAG/site-packages/nvidia/cuda_runtime/lib"
fi
NCCL_ROOT="$VENV/lib/$PY_TAG/site-packages/nvidia/nccl"
NCCL_INCLUDE="$NCCL_ROOT/include"
NCCL_LIB="$NCCL_ROOT/lib"
NCCL_LINK_LIB="$NCCL_LIB"
if [ ! -f "$NCCL_LIB/libnccl.so" ] && [ -f "$NCCL_LIB/libnccl.so.2" ]; then
  # NVIDIA runtime wheels may omit the unversioned linker name.
  NCCL_LINK_LIB="$REPO_DIR/rust/.nccl-link"
  mkdir -p "$NCCL_LINK_LIB"
  ln -sf "$NCCL_LIB/libnccl.so.2" "$NCCL_LINK_LIB/libnccl.so"
fi
LIBTORCH_CXX11_ABI="$("$VENV/bin/python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
mkdir -p "$REPO_DIR/rust/.cargo"
cat > "$REPO_DIR/rust/.cargo/config.toml" <<EOF
[env]
LIBTORCH = "$TORCH_LIB"
LIBTORCH_CXX11_ABI = "$LIBTORCH_CXX11_ABI"
LD_LIBRARY_PATH = "$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB"
EOF

cd "$REPO_DIR/rust"
BUILD_FEATURES="native-engine"
if [ "$NUM_GPUS" -gt 1 ] && [ -f "$CUDA_INCLUDE/cuda_runtime_api.h" ] \
  && [ -f "$CUDA_LIB/libcudart.so" ] \
  && [ -f "$NCCL_INCLUDE/nccl.h" ] && [ -f "$NCCL_LINK_LIB/libnccl.so" ]; then
  export CUDA_INCLUDE_DIR="$CUDA_INCLUDE"
  export CUDA_LIB_DIR="$CUDA_LIB"
  export NCCL_INCLUDE_DIR="$NCCL_INCLUDE"
  export NCCL_LIB_DIR="$NCCL_LINK_LIB"
  BUILD_FEATURES="$BUILD_FEATURES,nccl"
  echo "NCCL preflight: headers and runtime found; enabling device gradient all-reduce"
else
  echo "NCCL preflight: unavailable or single-GPU launch; building CPU gradient fallback"
fi
if ! cargo build --release -p oftrain --features "$BUILD_FEATURES"; then
  if [[ "$BUILD_FEATURES" == *nccl* ]]; then
    echo "WARNING: NCCL build preflight failed; rebuilding with the CPU gradient fallback" >&2
    BUILD_FEATURES="native-engine"
    cargo build --release -p oftrain --features "$BUILD_FEATURES" || exit 1
  else
    exit 1
  fi
fi
# CUDA-not-actually-linked footgun (see devlog 2026-07-09): confirm before
# spending any GPU-hours, not after wondering why util is stuck near 0.
if ! readelf -d target/release/oftrain | grep -q libtorch_cuda.so; then
  echo "FATAL: libtorch_cuda.so not linked into oftrain - CUDA is silently missing (see devlog)"
  exit 1
fi
if [[ "$BUILD_FEATURES" == *nccl* ]]; then
  if ! LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" ldd target/release/oftrain \
    | grep -q 'libnccl.so.*=>'; then
    echo "WARNING: NCCL runtime preflight failed; rebuilding with the CPU gradient fallback" >&2
    BUILD_FEATURES="native-engine"
    cargo build --release -p oftrain --features "$BUILD_FEATURES" || exit 1
  fi
fi
# Host-level CUDA-init footgun (see devlog 2026-07-09 "v8 launch" entry): a
# RunPod community-cloud host once had a working nvidia-smi/driver but a
# broken cuInit() for THIS compiled binary specifically (worked fine from
# plain python3 -c "import torch" in the identical env) - not fixable in
# code, only by relaunching on a different host. Warn-only, deliberately
# NOT fatal (a real GPU pod is the only place this can ever run, and this
# exact invocation - --num-envs 1 --updates 0, minimal but still spawning a
# real native env worker - has only been validated on CPU where the CUDA
# panic masks any other failure mode this specific shape could have; a bug
# *in this check itself* must never be able to kill a launch the real
# trainer, with its own crash-loop/backoff, would have survived).
if ! LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" OFTRAIN_EXPLICIT_CUINIT=1 \
  timeout 30 ./target/release/oftrain --engine native --node-fraction 0 --num-envs 1 --num-gpus 1 \
  --rollout-len 1 --updates 0 --device cuda:0 --ckpt-dir /tmp/oftrain_cuda_preflight \
  2>&1 | grep -q "explicit cuInit(0) -> 0"; then
  echo "WARNING: cuInit() preflight didn't report success (see 2026-07-09 devlog entry for the" \
       "known-bad-host failure mode this checks for) - proceeding anyway; the real trainer's own" \
       "crash-loop will retry/backoff if this host genuinely can't init CUDA." >&2
fi

"$VENV/bin/pip" install --quiet huggingface_hub safetensors numpy 2>/dev/null || true
PYTHON="$VENV/bin/python"

# Build ofhub before AE fetch: fetch_ae_encoders.sh uses `ofhf pull-ae`.
cargo build --release -p ofhub
OFHF="$REPO_DIR/rust/target/release/ofhf"

# Fine + coarse AE encoder safetensors for oftrain --ckpt / --coarse-ckpt.
mkdir -p "$AE_DIR"
if [ ! -f "$AE_DIR/ae_v32_nostatic_d8c32.encoder.safetensors" ] || [ ! -f "$AE_DIR/ae_v32_nostatic_d16c32.encoder.safetensors" ]; then
  echo "=== fetching/exporting AE encoders into $AE_DIR ==="
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN is required to fetch AE encoders from Hugging Face" >&2
    exit 1
  fi
  AE_DIR="$AE_DIR" PYTHON="$PYTHON" OFHF="$OFHF" bash "$REPO_DIR/scripts/fetch_ae_encoders.sh"
fi
if [ ! -f "$AE_DIR/ae_v32_nostatic_d8c32.encoder.safetensors" ] || [ ! -f "$AE_DIR/ae_v32_nostatic_d16c32.encoder.safetensors" ]; then
  echo "FATAL: AE encoder safetensors missing under $AE_DIR after fetch" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"

# Cap local numbered checkpoints + reclaim stale inactive runs when the
# overlay is near full. HF already retains the full policy_update* backlog.
disk_used_pct() {
  df -P "$1" 2>/dev/null | awk 'NR==2 { gsub(/%/, "", $5); print $5; exit }'
}

prune_local_numbered_ckpts() {
  local dir="$1" keep="$2"
  [ -d "$dir" ] || return 0
  [ "$keep" -gt 0 ] 2>/dev/null || return 0
  local -a files=()
  while IFS= read -r f; do
    [ -n "$f" ] && files+=("$f")
  done < <(ls -1 "$dir"/policy_update*.safetensors 2>/dev/null | sort -V)
  local n=${#files[@]}
  if [ "$n" -le "$keep" ]; then
    return 0
  fi
  local drop=$((n - keep))
  local i
  for ((i = 0; i < drop; i++)); do
    local stem="${files[$i]%.safetensors}"
    rm -f "${files[$i]}" "${stem}.state.json"
    echo "[disk] pruned local $(basename "${files[$i]}")"
  done
}

reclaim_stale_ckpt_runs() {
  local root="$REPO_DIR/rust/checkpoints"
  [ -d "$root" ] || return 0
  local IFS=','
  local run
  for run in $STALE_CKPT_RUNS; do
    run="$(echo "$run" | tr -d '[:space:]')"
    [ -n "$run" ] || continue
    [ "$run" = "$RUN_NAME" ] && continue
    if [ -d "$root/$run" ]; then
      local sz
      sz=$(du -sh "$root/$run" 2>/dev/null | awk '{print $1}')
      echo "[disk] removing stale checkpoint dir $root/$run ($sz)"
      rm -rf "$root/$run"
    fi
  done
}

ensure_disk_headroom() {
  local pct
  pct="$(disk_used_pct "$CKPT_DIR")"
  [ -n "$pct" ] || return 0
  prune_local_numbered_ckpts "$CKPT_DIR" "$CKPT_KEEP_LAST"
  if [ "$pct" -ge "$DISK_WARN_PCT" ]; then
    echo "[disk] WARNING: ${pct}% used on $(df -Ph "$CKPT_DIR" | awk 'NR==2{print $1,$5,$4" free"}')"
  fi
  if [ "$pct" -ge "$DISK_CRIT_PCT" ]; then
    echo "[disk] CRITICAL: ${pct}% used — reclaiming stale inactive checkpoint runs"
    reclaim_stale_ckpt_runs
    prune_local_numbered_ckpts "$CKPT_DIR" "$CKPT_KEEP_LAST"
    pct="$(disk_used_pct "$CKPT_DIR")"
    echo "[disk] after reclaim: ${pct}% used"
  fi
  if [ -n "$pct" ] && [ "$pct" -ge 97 ]; then
    echo "FATAL: disk ${pct}% full after reclaim; refusing to launch trainer" >&2
    exit 1
  fi
}

ensure_disk_headroom

# --- resume seed: current/future runs restore a complete safetensors pair
# only. Explicit `oftrain --resume old.ot` remains available for manual
# legacy migrations, but automated launches never select it. ---
if [ ! -f "$CKPT_DIR/latest.safetensors" ] || [ ! -f "$CKPT_DIR/latest.state.json" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN is required to restore checkpoints from Hugging Face" >&2
    exit 1
  fi
  "$OFHF" pull --checkpoint-dir "$CKPT_DIR" --repo-id "$HF_REPO_ID" \
    --run-prefix "$HF_RUN_PREFIX" || true
fi
if { [ -f "$CKPT_DIR/latest.safetensors" ] && [ ! -f "$CKPT_DIR/latest.state.json" ]; } \
  || { [ ! -f "$CKPT_DIR/latest.safetensors" ] && [ -f "$CKPT_DIR/latest.state.json" ]; }
then
  echo "FATAL: latest.safetensors and latest.state.json must exist as a pair" >&2
  exit 1
fi
if [ -f "$CKPT_DIR/latest.safetensors" ] && [ ! -f "$CKPT_DIR/manifest.json" ]; then
  echo "FATAL: V11 resume requires manifest.json beside the checkpoint pair" >&2
  exit 1
fi

# --- background HF sync: immutable snapshots of latest, best-eval,
# milestones, state sidecars, and the run manifest. Fail loud without token. ---
if [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: HF_TOKEN is required for ofhf sync-loop (fail loud)" >&2
  exit 1
fi
"$OFHF" sync-loop \
  --checkpoint-dir "$CKPT_DIR" --repo-id "$HF_REPO_ID" \
  --run-prefix "$HF_RUN_PREFIX" --interval "$HF_SYNC_INTERVAL_SECONDS" \
  >>"/tmp/hf_sync_$RUN_NAME.log" 2>&1 &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

# --- crash-proof training loop (backoff, not an instant relaunch into the
# same wall - see pod_train.sh's FAST_EXITS precedent) ---
ulimit -n 65535 2>/dev/null || true
FAST_EXITS=0
while true; do
  RESUME=""
  if [ -f "$CKPT_DIR/latest.safetensors" ]; then
    # V11 is a fresh schema (units + LSTM + C_GRID=99). Do not auto-migrate
    # V10/V8.x weights — resume only same-run V11 checkpoints.
    RESUME="--resume $CKPT_DIR/latest.safetensors"
  fi
  ensure_disk_headroom
  NODE_MIX_ARGS=()
  if [ "$(python3 -c "print(1 if float('$NODE_FRACTION') > 0 else 0)")" = "1" ]; then
    NODE_MIX_ARGS=(--allow-node-mix)
  fi
  echo "=== $(date -u +%FT%TZ) launching $RUN_NAME num_gpus=$NUM_GPUS envs/shard=$NUM_ENVS node_fraction=$NODE_FRACTION $RESUME ==="
  START_TS=$(date +%s)
  # Sampled apply/prepare timings on stderr (collect bottleneck diagnostics).
  OF_COLLECT_PROFILE="${OF_COLLECT_PROFILE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
  NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  OFTRAIN_NCCL_TIMEOUT_SECONDS="${OFTRAIN_NCCL_TIMEOUT_SECONDS:-60}" \
  LD_LIBRARY_PATH="$TORCH_LIB/lib:$NVRTC_LIB:$CUDA_LIB:$NCCL_LIB:$NCCL_LINK_LIB" \
    ./target/release/oftrain --engine native --node-fraction "$NODE_FRACTION" "${NODE_MIX_ARGS[@]}" \
    --num-envs "$NUM_ENVS" --num-gpus "$NUM_GPUS" \
    --rollout-len "$ROLLOUT_LEN" --minibatches "$MINIBATCHES" --stage "$STAGE" --device cuda:0 \
    --ckpt-dir "$CKPT_DIR" $TRAIN_ARGS $EXTRA_ARGS $RESUME \
    >> "/tmp/train_$RUN_NAME.log" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START_TS ))
  if [ -f "$CKPT_DIR/restart_request.json" ]; then
    REQUESTED_ENVS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["requested_envs_per_shard"])' "$CKPT_DIR/restart_request.json")
    RESIZE_REASON=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason","resize"))' "$CKPT_DIR/restart_request.json")
    # Consume the request here. oftrain also deletes it after a successful
    # spawn, but if startup fails the file would otherwise pin the supervisor
    # in a tight 26→26 relaunch loop.
    mv -f "$CKPT_DIR/restart_request.json" "$CKPT_DIR/restart_request.json.last"
    # Defense in depth: oftrain should already clamp to --max-envs, but never
    # relaunch above MAX_ENVS (curriculum floors of 24 used to OOM A40s).
    CLAMPED_ENVS=$REQUESTED_ENVS
    if [ "$CLAMPED_ENVS" -gt "$MAX_ENVS" ]; then CLAMPED_ENVS=$MAX_ENVS; fi
    if [ "$CLAMPED_ENVS" -lt "$MIN_ENVS" ]; then CLAMPED_ENVS=$MIN_ENVS; fi
    if [ "$CLAMPED_ENVS" -eq "$NUM_ENVS" ]; then
      echo "=== resize request ($RESIZE_REASON) $REQUESTED_ENVS capped to $CLAMPED_ENVS (= current); relaunching same size ===" \
        | tee -a "/tmp/train_$RUN_NAME.log"
    else
      echo "=== intentional resize ($RESIZE_REASON): $NUM_ENVS -> $CLAMPED_ENVS envs/shard (requested $REQUESTED_ENVS); restarting now ===" \
        | tee -a "/tmp/train_$RUN_NAME.log"
      NUM_ENVS="$CLAMPED_ENVS"
      MINIBATCHES=$((NUM_ENVS * ROLLOUT_LEN / MINIBATCH_SIZE))
      [ "$MINIBATCHES" -ge 1 ] || MINIBATCHES=1
    fi
    FAST_EXITS=0
    continue
  fi
  if [ "$ELAPSED" -lt 120 ]; then
    FAST_EXITS=$((FAST_EXITS + 1))
  else
    FAST_EXITS=0
  fi
  # Keep downtime short: 10s → 30s → 60s max (never sit for 10 minutes).
  BACKOFF=$(( FAST_EXITS >= 2 ? (FAST_EXITS >= 4 ? 60 : 30) : 10 ))
  echo "=== trainer exited ($RC) after ${ELAPSED}s; fast-exits=$FAST_EXITS, restarting in ${BACKOFF}s ===" \
    | tee -a "/tmp/train_$RUN_NAME.log"
  sleep "$BACKOFF"
done
