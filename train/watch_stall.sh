#!/usr/bin/env bash
# Watch the trainer for a stall and capture WHY before killing it.
#
# Background: the first two post-fix runs froze mid-run with the log stopped,
# no CPU use, no kernel event, no coredump and no crash message. A freeze that
# leaves no trace is expensive to diagnose, so this watchdog converts the next
# one into evidence: per-thread state and kernel stack, plus the helper
# processes the engine depends on.
#
# A stall is "the log has not changed for STALL seconds while the process is
# still alive". The trainer redraws its dashboard far more often than that.
set -u

LOG=${LOG:-/opt/data/workspaces/skg/openfront-train/run3.log}
REPORT=${REPORT:-/opt/data/workspaces/skg/openfront-train/stall-report.txt}
STALL=${STALL:-240}
PATTERN="[p]uffer train"

capture() {
    local pid=$1 stalled=$2
    echo "=== STALL DETECTED $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "no log output for ${stalled}s; pid=$pid"
    echo
    echo "--- process ---"
    ps -o pid,stat,etime,time,nlwp,rss,cmd -p "$pid" 2>&1
    echo
    echo "--- /proc/$pid/status ---"
    grep -E "^(State|Threads|VmRSS|VmSize|voluntary_ctxt_switches)" "/proc/$pid/status" 2>&1
    echo
    echo "--- per-thread kernel stack (root) ---"
    # wchan names the kernel function a thread is blocked in, which is the
    # decisive clue: futex_wait = a lock or condvar, pipe_read = blocked pipe IO
    # (the multiplexed daemon path talks over pipes), do_exit = already dead.
    for t in /proc/"$pid"/task/*; do
        [ -d "$t" ] || continue
        tid=$(basename "$t")
        st=$(awk '{print $3}' "$t/stat" 2>/dev/null)
        wc=$(cat "$t/wchan" 2>/dev/null)
        kst=$(sudo -n cat "$t/stack" 2>/dev/null | head -2 | tr '\n' '|' | sed 's/|$//')
        printf 'tid=%-7s state=%-2s wchan=%-22s stack=%s\n' "$tid" "${st:-?}" "${wc:-?}" "${kst:-?}"
    done
    echo
    echo "--- engine helper processes (tsx / bridge / node / daemon) ---"
    ps -eo pid,ppid,stat,etime,pcpu,rss,cmd 2>/dev/null \
        | grep -E "tsx|bridge|engine_daemon|[n]ode " | grep -v grep
    echo "(none = the engine helper the trainer talks to is GONE, which would"
    echo " block a thread on a pipe read forever)"
    echo
    echo "--- how the engine path was configured ---"
    echo "OPENFRONT_DAEMON=${OPENFRONT_DAEMON:-<unset: daemon path, default ON>}"
    echo
    echo "--- last 25 log lines ---"
    tail -25 "$LOG" 2>&1
}

last=""
stalled=0
while true; do
    if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
        printf '%s trainer no longer running, watchdog exiting\n' "$(date '+%H:%M:%S')" >>"$REPORT"
        exit 0
    fi
    cur=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" = "$last" ]; then
        stalled=$(( stalled + 15 ))
    else
        stalled=0
        last=$cur
    fi
    if [ "$stalled" -ge "$STALL" ]; then
        pid=$(pgrep -f "$PATTERN" | head -1)
        capture "$pid" "$stalled" >"$REPORT" 2>&1
        kill -TERM "$pid" 2>/dev/null
        echo "--- watchdog SIGTERMed pid=$pid to stop it burning wall clock ---" >>"$REPORT"
        exit 0
    fi
    sleep 15
done