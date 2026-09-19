#!/usr/bin/env bash
# Report an experiment from its OWN log slice: the resume line plus the win rate,
# entropy, KL and curriculum state over the sampler rows written since it began.
#
# Usage: exp_report.sh <name> [n_rows]
#   n_rows limits the average to the last N sampler rows (default: all of them).
#
# The window is slice-by-offset for the trainer log (exact) and by wall-clock
# start time for the sampler rows (the sampler log has no offset record). Rows
# carry only HH:MM:SS, so a window that crosses midnight would read short - run
# it again or pass n_rows in that case.
set -euo pipefail

DIR=/opt/data/workspaces/skg/openfront-train/experiments
NAME="${1:?usage: exp_report.sh <name> [n_rows]}"
N="${2:-0}"
META="$DIR/$NAME.meta"
[ -f "$META" ] || { echo "no such experiment: $NAME" >&2; exit 1; }

LOG=$(grep -m1 '^log=' "$META" | cut -d= -f2-)
OFF=$(grep -m1 '^offset=' "$META" | cut -d= -f2-)
STARTED=$(grep -m1 '^started=' "$META" | cut -d= -f2-)
START_HMS=$(printf '%s' "$STARTED" | cut -dT -f2 | cut -d- -f1 | cut -d+ -f1)
SAMPLES="${LOG%/*}/samples-current.log"
SLICE="$DIR/$NAME.log"

tail -c +$((OFF + 1)) "$LOG" > "$SLICE" 2>/dev/null || : > "$SLICE"

echo "=== experiment $NAME ==="
printf 'started   %s\n' "$STARTED"
grep -m1 '^overrides=' "$META" | sed 's/^/          /'
echo
echo "--- resume line ---"
grep -a "pufferl:" "$SLICE" | tail -1 | sed 's/^/  /' || echo "  (none yet)"
echo
echo "--- sampler rows since start ---"
awk -v t="$START_HMS" -v n="$N" '
    /^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] / {
        if (substr($1, 1, 8) < t) { next }
        if (v["steps"] != "") { }   # keep v fresh
        rows[++k] = $0
    }
    END {
        if (k == 0) { print "  (no sampler rows yet; the sampler adds one every ~5 min)"; exit }
        lo = (n > 0 && k > n) ? k - n + 1 : 1
        for (i = lo; i <= k; i++) {
            split(rows[i], a, " ")
            for (j = 2; j <= length(a); j++) {
                split(a[j], kv, "=")
                v[kv[1]] = kv[2]
            }
            m++
            sw += v["wins"]; se += v["entropy"]; skl += v["kl"]
            ss += v["score"]; sl += v["losses"]
        }
        # Compute the rate in its own statement: a bare `>` inside printf is
        # parsed by awk as output redirection, not as a comparison.
        wr = (sw + sl) > 0 ? sw / (sw + sl) : 0
        printf "  rows %s .. %s (%d rows)\n", substr(rows[lo],1,8), substr(rows[k],1,8), m
        printf "  mean wins=%.3f losses=%.3f  (win rate %.3f)\n", sw/m, sl/m, wr
        printf "  mean entropy=%.3f  mean kl=%.4f  mean score=%.1f\n", se/m, skl/m, ss/m
        printf "  last: steps=%s epoch=%s stage=%s advances=%s stage_wins=%s win_rate=%s\n", \
            v["steps"], v["epoch"], v["stage"], v["stage_advances"], v["stage_wins"], v["win_rate"]
    }
' "$SAMPLES"
