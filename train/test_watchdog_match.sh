#!/usr/bin/env bash
# Rigorous decoy test for the watchdog's process matching.
# The decoy keeps the string "puffer train" in its command line WITHOUT being a
# trainer: argv[0] is a different program whose text happens to mention it.
set -u
W=/opt/data/workspaces/skg/openfront-train/watch_train.sh
R=/opt/data/workspaces/skg/openfront-train/stall-report.txt
REAL=$(systemctl show -p MainPID --value openfront-train.service)

bash -c 'exec -a "tail -f puffer train.log" /var/lib/hermes/venv/bin/python -c "import time; time.sleep(60)"' >/dev/null 2>&1 &
decoy=$!
sleep 2

echo "real trainer: $REAL | decoy: $decoy"
echo "decoy cmdline: [$(tr '\0' ' ' <"/proc/$decoy/cmdline")]"
echo "OLD matcher pgrep -f '[p]uffer train' matches: [$(pgrep -f '[p]uffer train' | tr '\n' ' ')]  <- includes the decoy = the bug"
echo

before=$(wc -l <"$R")
DRY=1 UNIT=openfront-scratch.service bash "$W"    # precision: what does it consider a trainer?
bash "$W"                                          # the real 2-minute run
sleep 1

kill -0 "$decoy" 2>/dev/null && echo "decoy after real run: ALIVE (good)" || echo "decoy after real run: KILLED (bad)"
kill -0 "$REAL"  2>/dev/null && echo "trainer after real run: ALIVE (good)" || echo "trainer after real run: KILLED (bad)"
echo "new report lines (should name only the real trainer, never the decoy):"
tail -n +$((before + 1)) "$R"
kill "$decoy" 2>/dev/null
wait "$decoy" 2>/dev/null
echo "cleanup done"
