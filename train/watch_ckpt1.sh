#!/usr/bin/env bash
# Wait for run2's first checkpoint, then summarise how far the policy has learned.
#
# The trainer writes checkpoints as <run-dir>/%016ld.bin named by global_step, so
# watch for ANY .bin rather than a hardcoded name (the step the interval fires on
# is an implementation detail we do not want to get wrong).
#
# Exits when it has written the report, which fires the notification.
set -u

# The trainer creates a fresh timestamped checkpoint dir per run, and aborted
# launches leave empty ones behind, so ask the shared helper for the run that
# actually holds checkpoints (never `ls -dt */ | head -1`, which an empty dir
# can win). Pin with RUN_ID=<id> if you want a specific run.
CKPT=$(bash /opt/data/workspaces/skg/openfront-train/ckpt_latest.sh --dir)
CKPT=${CKPT%/}
RUN=/opt/data/workspaces/skg/openfront-train/run4.log
TREND=/opt/data/workspaces/skg/openfront-train/samples4.log
REPORT=/opt/data/workspaces/skg/openfront-train/checkpoint1-report.txt

# last <n> values of a key=value field, for averaging
series() { grep -o "$1=[0-9.eE+-]*" "$TREND" 2>/dev/null | cut -d= -f2 | grep -E '^[0-9.eE+-]+$'; }
lastval() { tail -1 "$TREND" 2>/dev/null | grep -o "$1=[0-9.eE+-]*" | tail -1 | cut -d= -f2; }
avg() { awk '{s+=$1; n++} END { if (n) printf "%.4f", s/n; else printf "n/a" }'; }
maxof() { awk 'BEGIN{m=""} {if ($1+0>m+0 || m=="") m=$1} END{print (m==""?"n/a":m)}'; }
dash() { tr -d '\r' < "$RUN" 2>/dev/null | grep -a "$1" | tail -1; }

dead=0
for i in $(seq 1 720); do            # up to 6 hours
    if ls "$CKPT"/*.bin >/dev/null 2>&1; then break; fi
    if ! pgrep -f "puffer train" >/dev/null 2>&1; then dead=1; break; fi
    sleep 30
done

{
    echo "=== CHECKPOINT ==="
    if ls "$CKPT"/*.bin >/dev/null 2>&1; then
        # let the write settle: wait for the size to stop changing
        f=$(ls "$CKPT"/*.bin | head -1)
        prev=-1
        for j in $(seq 1 60); do
            cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
            [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ] && break
            prev=$cur; sleep 10
        done
        ls -la "$CKPT"/*.bin
    elif [ "$dead" = "1" ]; then
        echo "TRAINER DIED before writing a checkpoint (see run2.log tail below)"
    else
        echo "TIMEOUT: no checkpoint after 6h"
    fi

    echo
    echo "=== RUN STATE ==="
    for k in Steps SPS Epoch Uptime "To go"; do dash "$k"; done
    grep -ao "VRAM: *[0-9.]*G *RAM: *[0-9.]*G" "$RUN" | tail -1

    echo
    echo "=== LEARNING TREND (samples4.log) ==="
    echo "samples: $(wc -l < "$TREND" 2>/dev/null)"
    echo "--- first ---"; head -2 "$TREND" 2>/dev/null | tail -1
    echo "--- mid   ---"; sed -n "$(( $(wc -l < "$TREND" 2>/dev/null) / 2 ))p" "$TREND" 2>/dev/null
    echo "--- last  ---"; tail -1 "$TREND" 2>/dev/null

    echo
    echo "=== HOW FAR HAS IT LEARNED ==="
    printf 'win rate      over ALL samples : %s\n' "$(series wins | avg)"
    printf 'win rate      over last 60     : %s\n' "$(series wins | tail -60 | avg)"
    printf 'score         over ALL samples : %s\n' "$(series score | avg)"
    printf 'score         over last 60     : %s\n' "$(series score | tail -60 | avg)"
    printf 'score         max seen         : %s\n' "$(series score | maxof)"
    printf 'mean_reward   over last 60     : %s\n' "$(series mean_reward | tail -60 | avg)"
    printf 'entropy       last             : %s\n' "$(lastval entropy)"
    printf 'policy loss   last             : %s\n' "$(lastval policy)"
    printf 'stage         max seen         : %s\n' "$(series stage | maxof)"
    printf 'stage_advances max seen        : %s\n' "$(series advances | maxof)"
    printf 'noop_degraded  last            : %s\n' "$(lastval noop_degraded)"
    printf 'episodes      last             : %s\n' "$(lastval episodes)"

    echo
    echo "=== CURRICULUM / STAGE EVENTS IN LOG ==="
    grep -aiE "stage[_ ]?(advance|win|up|clear)|curriculum" "$RUN" 2>/dev/null | grep -av "^│" | tail -10
    echo "(no lines above = no stage event logged yet)"

    echo
    echo "=== ANOMALIES ==="
    grep -aiE "error|assert|abort|terminate|nan|inf" "$RUN" 2>/dev/null | grep -av "^│" | tail -5
    echo "(no lines above = clean)"
} > "$REPORT" 2>&1

echo "wrote $REPORT"