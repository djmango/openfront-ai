#!/usr/bin/env python3
"""Align oftrain JSONL/stdout metrics into periodic V8 vs V8.1 reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


UPDATE_RE = re.compile(
    r"\[update\s+(?P<update>\d+)\].*?"
    r"steps/s=\s*(?P<throughput>[-+.\deE]+).*?"
    r"recent_reward=\s*(?P<reward>[-+.\deE]+)"
)


def read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    rows[int(row["update"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # The trainer may be appending the final line while we read.
                    continue
    except FileNotFoundError:
        pass
    return rows


def read_stdout(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = UPDATE_RE.search(line)
                if match:
                    rows[int(match["update"])] = {
                        "reward": float(match["reward"]),
                        "throughput": float(match["throughput"]),
                    }
    except FileNotFoundError:
        pass
    return rows


def latest_eval(rows: dict[int, dict[str, Any]], update: int) -> dict[str, Any] | None:
    for key in sorted((key for key in rows if key <= update), reverse=True):
        row = rows[key]
        if row.get("eval/win") is not None or row.get("eval/score") is not None:
            return {
                "update": key,
                "win": row.get("eval/win"),
                "score": row.get("eval/score"),
            }
    return None


def arm_snapshot(
    metrics: dict[int, dict[str, Any]],
    stdout: dict[int, dict[str, float]],
    update: int,
) -> dict[str, Any]:
    row = metrics[update]
    text = stdout.get(update, {})
    return {
        "update": update,
        "rolling_win_rate": row.get("win_rate"),
        "recent_reward": text.get("reward"),
        "vf_loss": row.get("loss/vf"),
        "entropy": row.get("loss/ent"),
        "throughput_steps_s": text.get("throughput"),
        "stage": row.get("stage"),
        "eval": latest_eval(metrics, update),
    }


def deltas(control: dict[str, Any], v81: dict[str, Any]) -> dict[str, float | None]:
    fields = (
        "rolling_win_rate",
        "recent_reward",
        "vf_loss",
        "entropy",
        "throughput_steps_s",
        "stage",
    )
    result: dict[str, float | None] = {}
    for field in fields:
        left, right = control.get(field), v81.get(field)
        result[field] = None if left is None or right is None else float(right) - float(left)
    return result


def markdown(report: dict[str, Any]) -> str:
    def value(item: Any) -> str:
        return "n/a" if item is None else f"{item:.4g}" if isinstance(item, float) else str(item)

    control, v81 = report["control"], report["v81"]
    lines = [
        f"# V8.1 A/B report — update {report['report_update']}",
        "",
        "| metric | control | v8.1 | delta (v8.1-control) |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "rolling_win_rate",
        "recent_reward",
        "vf_loss",
        "entropy",
        "throughput_steps_s",
        "stage",
    ):
        lines.append(
            f"| {key} | {value(control[key])} | {value(v81[key])} | "
            f"{value(report['delta'][key])} |"
        )
    lines.extend(
        [
            "",
            f"- control eval: `{json.dumps(control['eval'], sort_keys=True)}`",
            f"- v8.1 eval: `{json.dumps(v81['eval'], sort_keys=True)}`",
            f"- generated: {report['generated_at']}",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--v81-metrics", type=Path, required=True)
    parser.add_argument("--v81-log", type=Path, required=True)
    parser.add_argument("--start-update", type=int, required=True)
    parser.add_argument("--every", type=int, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--latest-markdown", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be positive")
    target = args.start_update + args.every - 1
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    while True:
        control_metrics = read_jsonl(args.control_metrics)
        v81_metrics = read_jsonl(args.v81_metrics)
        control_stdout = read_stdout(args.control_log)
        v81_stdout = read_stdout(args.v81_log)
        while (
            target in control_metrics
            and target in v81_metrics
            and target in control_stdout
            and target in v81_stdout
        ):
            control = arm_snapshot(
                control_metrics, control_stdout, target
            )
            v81 = arm_snapshot(v81_metrics, v81_stdout, target)
            report = {
                "schema": 1,
                "report_update": target,
                "updates_since_start": target - args.start_update + 1,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "control": control,
                "v81": v81,
                "delta": deltas(control, v81),
            }
            with args.jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, sort_keys=True) + "\n")
            atomic_write(args.latest_markdown, markdown(report))
            print(json.dumps({"event": "comparison_report", **report}), flush=True)
            target += args.every
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
