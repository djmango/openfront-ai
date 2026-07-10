#!/usr/bin/env python3
"""Summarize an outcome_gate JSON report BY BOT COUNT for the curriculum
self-play record set (see datagen/gen_curriculum_parity.ts and
scripts/run_curriculum_parity_gate.sh).

Record filenames are "curr-b<BOTS>-s<SEED>-<map>.json(.gz)"; bot count is
parsed straight from the filename rather than joined against the manifest,
so this also works if the manifest.json isn't handy.

Usage:
    python3 scripts/analyze_curriculum_parity.py /tmp/curriculum_gate_report.json
"""
import json
import re
import sys
from collections import defaultdict

RECORD_RE = re.compile(r"curr-b(\d+)-")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <outcome_gate_report.json>", file=sys.stderr)
        sys.exit(2)
    report = json.load(open(sys.argv[1]))

    by_bots: dict[int, list[dict]] = defaultdict(list)
    for rec in report["records"]:
        m = RECORD_RE.match(rec["record"])
        if not m:
            continue
        by_bots[int(m.group(1))].append(rec)

    print(f"parity_commit={report.get('parityCommit')} "
          f"oracle_record_set_hash={report.get('oracleRecordSetHash')}")
    print(f"overall: {report['summary']['pass']}/{report['summary']['total']} pass\n")

    print(f"{'bots':>5} {'pass':>6} {'total':>6} {'rate':>6}  "
          f"{'avg_tiles_ratio':>15}  {'avg_land_share_delta(wrong_winner)':>36}")
    rows = []
    for bots in sorted(by_bots):
        recs = by_bots[bots]
        passed = sum(1 for r in recs if r["category"] == "pass")
        total = len(recs)
        ratios = []
        deltas = []
        for r in recs:
            exp, act = r.get("expected"), r.get("actual")
            if exp and act:
                et = sum(p["tiles"] for p in exp["finalRanking"])
                at = sum(p["tiles"] for p in act["finalRanking"])
                if et > 0:
                    ratios.append(at / et)
            if (
                r["category"] == "wrong_winner"
                and exp
                and act
                and exp.get("winnerLandShare") is not None
                and act.get("winnerLandShare") is not None
            ):
                deltas.append(abs(exp["winnerLandShare"] - act["winnerLandShare"]))
        avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        avg_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        rows.append((bots, passed, total, avg_ratio, avg_delta))
        print(
            f"{bots:>5} {passed:>6} {total:>6} {passed / total * 100:>5.0f}%  "
            f"{avg_ratio:>15.4f}  {avg_delta:>36.4f}"
        )

    print("\n=== non-pass records (winner/tick/land-share detail) ===")
    for bots in sorted(by_bots):
        for r in by_bots[bots]:
            if r["category"] == "pass":
                continue
            exp, act = r.get("expected") or {}, r.get("actual") or {}
            print(
                f"bots={bots:<4} {r['record']:<32} category={r['category']:<16} "
                f"expected_winner={exp.get('winner')!r:<32} actual_winner={act.get('winner')!r:<32} "
                f"expected_tick={exp.get('terminalTick')} actual_tick={act.get('terminalTick')} "
                f"expected_share={exp.get('winnerLandShare')} actual_share={act.get('winnerLandShare')}"
            )


if __name__ == "__main__":
    main()
