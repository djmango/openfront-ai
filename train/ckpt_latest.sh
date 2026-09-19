#!/usr/bin/env bash
# Print the current OpenFront training run's checkpoint dir, or its newest
# checkpoint file.
#
# Layout is <checkpoint_dir>/<env>/<run_id>/, one .bin per checkpoint named by
# global_step, and run_id is a launch timestamp - so every launch that dies
# early leaves a fresh EMPTY run dir behind (15 accumulated before the first
# cleanup). Never select with `ls -dt */ | head -1`: an empty dir can win, and
# the result is a resume from the wrong run.
#
#   bash ckpt_latest.sh           # newest checkpoint file (resume point)
#   bash ckpt_latest.sh --dir     # the run dir holding it
#   bash ckpt_latest.sh --list    # every run dir, newest checkpoint first
#   RUN_ID=<id> bash ckpt_latest.sh   # pin a specific run
#
# Env: ROOT (default the PufferLib5 checkpoint root)
set -euo pipefail

ROOT=${ROOT:-/opt/data/workspaces/skg/PufferLib5/checkpoints/openfront}
mode=${1:-}

newest_in() { # newest_in <dir> -> newest *.bin path, or empty
    find "$1" -maxdepth 1 -name '*.bin' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-
}

if [[ -n "${RUN_ID:-}" ]]; then
    dir=$ROOT/$RUN_ID
    [[ -d $dir ]] || { echo "no such run dir: $dir" >&2; exit 1; }
elif [[ $mode != --list && $mode != --prune ]]; then
    # The run in progress is the one holding the globally newest checkpoint.
    newest=$(find "$ROOT" -mindepth 2 -maxdepth 2 -name '*.bin' \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [[ -n ${newest:-} ]] || { echo "no checkpoints under $ROOT" >&2; exit 1; }
    dir=$(dirname "$newest")
fi

case "$mode" in
    --dir)  echo "$dir" ;;
    --list)
        for d in "$ROOT"/*/; do
            f=$(newest_in "$d")
            if [[ -n $f ]]; then
                printf '%-18s %s  (%s)\n' "$(basename "$d")" "$(basename "$f")" \
                    "$(find "$d" -maxdepth 1 -name '*.bin' | wc -l)x .bin"
            else
                printf '%-18s EMPTY\n' "$(basename "$d")"
            fi
        done
        ;;
    --prune)
        # rmdir only, so this can never delete a checkpoint: it removes the
        # empty run dirs every aborted launch leaves behind. Refuses while a
        # trainer is live, because the run dir of a just-started run is empty
        # until its first checkpoint lands.
        if pgrep -f 'puffer train' >/dev/null 2>&1; then
            echo "refusing to prune: a trainer is running (its run dir may still be empty)" >&2
            exit 1
        fi
        n=0
        for d in "$ROOT"/*/; do
            if [[ -z $(newest_in "$d") ]] && rmdir "$d" 2>/dev/null; then
                echo "removed empty run dir $(basename "$d")"
                n=$((n + 1))
            fi
        done
        echo "pruned $n empty run dir(s)"
        ;;
    *)
        f=$(newest_in "$dir")
        [[ -n $f ]] || { echo "run dir $dir holds no checkpoints" >&2; exit 1; }
        echo "$f"
        ;;
esac