#!/usr/bin/env bash
# Launch a training experiment: override the unit's environment, restart from the
# newest checkpoint, and record where in the log this experiment begins.
#
# Usage: run_experiment.sh <name> [VAR=VAL ...]
#   e.g. run_experiment.sh e1_bigbatch HORIZON=256 SCHED_EPOCHS=400
#
# The NixOS unit pins the baseline config; a runtime drop-in overrides only the
# variables named here, so everything unmentioned keeps its pinned value.
# /etc is read-only on NixOS and the drop-in lives in /run, so a reboot returns
# the unit to the pinned baseline (and LOAD=auto resumes the newest checkpoint).
#
# Read the result with exp_report.sh <name>; it reads only this experiment's
# slice of the log, so experiments cannot be confused with each other.
set -euo pipefail

NAME="${1:?usage: run_experiment.sh <name> [VAR=VAL ...]}"; shift
UNIT=openfront-train.service
DIR=/opt/data/workspaces/skg/openfront-train/experiments
DROPIN=/run/systemd/system/${UNIT}.d
LOG=/opt/data/workspaces/skg/openfront-train/current.log

mkdir -p "$DIR"
[ -f "$LOG" ] || : > "$LOG"
OFFSET=$(stat -c %s "$LOG")

# Clean slate for the experiment override only, so experiments never inherit
# each other. limits.conf (StartLimitBurst) lives in the same dir and must
# survive: without it a few restarts lock the unit out with start-limit-hit.
sudo -n rm -f "$DROPIN/override.conf"
if [ "$#" -gt 0 ]; then
    sudo -n mkdir -p "$DROPIN"
    printf '[Service]\nEnvironment=%s\n' "$*" \
        | sudo -n tee "$DROPIN/override.conf" >/dev/null
fi
sudo -n systemctl daemon-reload

{
    echo "name=$NAME"
    echo "started=$(date -Is)"
    echo "log=$LOG"
    echo "offset=$OFFSET"
    echo "overrides=${*:-none}"
} > "$DIR/$NAME.meta"

sudo -n systemctl restart "$UNIT" openfront-sampler.service
echo "[exp] $NAME launched; overrides: ${*:-none}"
sleep 20
grep -a "pufferl:" "$LOG" | tail -1 || true
systemctl is-active "$UNIT"
