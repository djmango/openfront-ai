#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for autoencoder training: repo + deps + both
# datasets (bot "data/" + human "data-human/") fully prefeaturized.
#
# Idempotent: every stage skips work that is already done, so it is safe to
# re-run after an interruption. Run it inside the pod:
#
#   bash -c "curl -fsSL https://raw.githubusercontent.com/djmango/openfront-ai/master/scripts/runpod_bootstrap.sh | bash"
#
# or, if the repo is already at /workspace/openfront-ai:
#
#   bash /workspace/openfront-ai/scripts/runpod_bootstrap.sh
#
# The HF tars contain raw snapshots (states/*.gz) without the cache/ subdirs,
# so after download+extract this runs `ofae prefeaturize` (CPU-parallel). On a
# 48-vCPU community pod the whole thing is bounded by download + prefeaturize.

set -euo pipefail

REPO_DIR=/workspace/openfront-ai
# nproc on community pods reports host cores, not the rented slice; cap it.
WORKERS="${WORKERS:-$(( $(nproc) < 32 ? $(nproc) : 32 ))}"

# --- repo ---
mkdir -p /workspace
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/djmango/openfront-ai "$REPO_DIR"
fi
cd "$REPO_DIR"

# --- rust toolchain + ofae (needs libtorch via setup_libtorch / torch venv) ---
if [ ! -x rust/target/release/ofae ]; then
  bash scripts/setup_libtorch.sh || true
  (cd rust && cargo build --release -p ofae)
fi
OFAE="${OFAE:-$REPO_DIR/rust/target/release/ofae}"

# --- download + extract datasets from HF ---
# maps/<map>.tar extracts to <map>/<game-id>/{terrain.bin,meta.json,states/}
# HF cache must live on the big /workspace volume; the default ~/.cache is on
# the small container disk and fills up.
pip install -q "huggingface_hub[hf_transfer]"
export HF_HOME=/workspace/hf-cache
stage_dataset() { # $1 = hf dataset repo, $2 = local root (data | data-human)
  local repo="$1" root="$2"
  mkdir -p "$root"
  python - "$repo" "$root" <<'EOF'
import shutil, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo, root = sys.argv[1], Path(sys.argv[2])
info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
tars = {s.rfilename: s.size for s in info.siblings
        if s.rfilename.startswith("maps/") and s.rfilename.endswith(".tar")}
bad = []
for t in sorted(tars):
    map_name = Path(t).stem
    if (root / map_name / ".extracted").exists():
        print(f"skip {repo}/{t}", flush=True)
        continue
    if not tars[t]:
        print(f"WARN {repo}/{t} is empty on HF; skipping", flush=True)
        bad.append(t)
        continue
    print(f"fetch {repo}/{t}", flush=True)
    p = hf_hub_download(repo, t, repo_type="dataset")
    # Some tars can be truncated on HF (upload raced the tar writer).
    # Skip them and drop partial extractions; a later re-run of this
    # script fetches them once fixed (idempotent via .extracted markers).
    r = subprocess.run(["tar", "-xf", p, "-C", str(root)])
    if r.returncode != 0:
        print(f"WARN {repo}/{t} corrupt (tar exit {r.returncode}); skipping", flush=True)
        shutil.rmtree(root / map_name, ignore_errors=True)
        bad.append(t)
    else:
        (root / map_name / ".extracted").touch()
    Path(p).unlink()  # drop the cached tar; raw extraction is enough
print(f"{repo} -> {root}: done" + (f" (SKIPPED: {bad})" if bad else ""), flush=True)
EOF
}

stage_dataset djmango/openfront-snapshots data
stage_dataset djmango/openfront-human-games data-human
rm -rf /workspace/hf-cache  # reclaim the transient download cache

# --- featurize (skips games whose cache/index.json already exists) ---
"$OFAE" prefeaturize --data data --workers "$WORKERS"
"$OFAE" prefeaturize --data data-human --workers "$WORKERS"

echo "bootstrap complete - train with:"
echo "  $OFAE train --data data,data-human --steps 40000 --batch-size 64 --latent-down 8 --out runs/ae_v32_nostatic_d8c32"
