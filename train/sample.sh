#!/usr/bin/env bash
# Sample the OpenFront training dashboard into a trend file (one line per
# minute) so progress can be read without scrolling the full dashboard log.
set -u
LOG=${LOG:-/opt/data/workspaces/skg/openfront-train/run1.log}
OUT=${OUT:-/opt/data/workspaces/skg/openfront-train/samples.log}

field() { # field <label> <pattern>
    printf '%s' "$1" | grep -aoE "$2" | tail -1
}

while true; do
    d=$(tail -c 5000 "$LOG" 2>/dev/null | tr -d '\r')
    steps=$(field "$d" 'Steps *[0-9.]+[KMB]?')
    sps=$(field "$d" 'SPS *[0-9.]+[KMB]?')
    epoch=$(field "$d" 'Epoch *[0-9]+')
    pol=$(field "$d" 'policy *[0-9.-]+')
    val=$(field "$d" 'value *[0-9.-]+')
    score=$(field "$d" 'score *[0-9.-]+')
    epe=$(field "$d" 'episodes *[0-9.]+')
    wins=$(field "$d" 'wins *[0-9.]+')
    adv=$(field "$d" 'stage_advances *[0-9.]+')
    printf '%s steps=%s sps=%s epoch=%s policy=%s value=%s score=%s episodes=%s wins=%s advances=%s load=%s\n' \
        "$(date +%H:%M:%S)" "${steps#* }" "${sps#* }" "${epoch#* }" "${pol#* }" "${val#* }" \
        "${score#* }" "${epe#* }" "${wins#* }" "${adv#* }" "$(cut -d' ' -f1 /proc/loadavg)" >>"$OUT"
    sleep 60
done