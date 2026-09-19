#!/usr/bin/env bash
# Trend sampler for the OpenFront x PufferLib 5.0 trainer.
#
# Why this exists: run1.log is a multi-block dashboard redraw, and the env/*
# block only refreshes when an episode completes. The previous sampler used
# `tail -c 5000` and a naive label strip, so score/wins/episodes/advances came
# out blank even while the run was healthy.
#
# Two rules make it reliable:
#   1. window large enough to hold several whole dashboard blocks (60000);
#   2. take the LAST occurrence of each label and strip the label with awk
#      ($NF), so values with internal spaces cannot corrupt the field;
#   3. anchor every label with a LEADING SPACE, otherwise `wins` matches
#      `stage_wins` and `kl` matches `old_kl` (the char before those labels
#      is '_', not a space), which silently reports the wrong number.
set -u
LOG=${LOG:-/opt/data/workspaces/skg/openfront-train/run2.log}
OUT=${OUT:-/opt/data/workspaces/skg/openfront-train/samples2.log}
INTERVAL=${INTERVAL:-60}

field() { # field <dashboard text> <regex>
    printf '%s' "$1" | grep -aoE "$2" | tail -1 | awk '{print $NF}'
}

while true; do
    d=$(tail -c 60000 "$LOG" 2>/dev/null | tr -d '\r')
    printf '%s steps=%s sps=%s epoch=%s policy=%s value=%s entropy=%s kl=%s clipfrac=%s score=%s wins=%s losses=%s episodes=%s stage=%s advances=%s stage_wins=%s win_rate=%s death_rate=%s ticks=%s mean_reward=%s noop_degraded=%s episode_length=%s load=%s\n' \
        "$(date +%H:%M:%S)" \
        "$(field "$d" ' Steps *[0-9.]+[KMB]?')" \
        "$(field "$d" ' SPS *[0-9.]+[KMB]?')" \
        "$(field "$d" ' Epoch *[0-9]+')" \
        "$(field "$d" ' policy *[-0-9.]+')" \
        "$(field "$d" ' value *[-0-9.]+')" \
        "$(field "$d" ' entropy *[-0-9.]+')" \
        "$(field "$d" ' kl *[-0-9.]+')" \
        "$(field "$d" ' clipfrac *[-0-9.]+')" \
        "$(field "$d" ' score *[-0-9.]+')" \
        "$(field "$d" ' wins *[-0-9.]+')" \
        "$(field "$d" ' losses *[-0-9.]+')" \
        "$(field "$d" ' episodes *[-0-9.]+')" \
        "$(field "$d" ' stage *[-0-9.]+')" \
        "$(field "$d" ' stage_advances *[-0-9.]+')" \
        "$(field "$d" ' stage_wins *[-0-9.]+')" \
        "$(field "$d" ' win_rate *[-0-9.]+')" \
        "$(field "$d" ' death_rate *[-0-9.]+')" \
        "$(field "$d" ' ticks *[-0-9.]+')" \
        "$(field "$d" ' mean_reward *[-0-9.]+')" \
        "$(field "$d" ' noop_degraded *[-0-9.]+')" \
        "$(field "$d" ' episode_length *[-0-9.]+')" \
        "$(cut -d' ' -f1 /proc/loadavg)" >>"$OUT"
    sleep "$INTERVAL"
done