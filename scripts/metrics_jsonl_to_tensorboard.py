#!/usr/bin/env python3
"""Continuously relay oftrain's metrics.jsonl into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Iterator


TAG_MAP = {
    "loss/pg": "loss/policy",
    "loss/vf": "loss/value",
    "loss/ent": "loss/entropy",
    "loss/entq": "loss/entropy_q",
    "lr": "loss/lr",
    "win_rate": "curriculum/rolling_win",
    "stage": "curriculum/stage",
}
EPISODE_TAG_MAP = {
    "reward": "episode/reward",
    "stage": "curriculum/episode_stage",
    "rehearsal": "curriculum/rehearsal",
}
METADATA_KEYS = {"event", "update", "env_steps", "map"}


def tag_for(key: str, event: str | None) -> str:
    if event == "episode":
        return EPISODE_TAG_MAP.get(key, key)
    return TAG_MAP.get(key, key)


def numeric_items(value: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Flatten numeric JSON values while ignoring labels and nulls."""
    if isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        yield prefix, float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from numeric_items(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            yield from numeric_items(child, child_prefix)


def write_row(writer: Any, row: dict[str, Any], episode_step: int) -> int:
    event = row.get("event")
    if event == "episode":
        step = episode_step
        episode_step += 1
    else:
        step = int(row.get("env_steps", row.get("update", 0)))

    for key, value in row.items():
        if key in METADATA_KEYS:
            continue
        for flattened_key, number in numeric_items(value, key):
            writer.add_scalar(tag_for(flattened_key, event), number, step)
    return episode_step


def relay(
    metrics: Path,
    writer: Any,
    poll_seconds: float,
    from_start: bool,
    stop: Any,
) -> None:
    offset = 0
    episode_step = 0
    initialized = False
    pending = b""

    while not stop():
        try:
            size = metrics.stat().st_size
        except FileNotFoundError:
            time.sleep(poll_seconds)
            continue

        if not initialized:
            offset = 0 if from_start else size
            initialized = True
        elif size < offset:
            # The trainer recreated/truncated the metrics file.
            offset = 0
            pending = b""

        with metrics.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()

        if chunk:
            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    print(f"warning: skipping malformed metrics row: {error}", flush=True)
                    continue
                if isinstance(row, dict):
                    episode_step = write_row(writer, row, episode_step)
            writer.flush()
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Only process rows appended after startup (default replays history).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    writer = SummaryWriter(str(args.out_dir), flush_secs=max(1, int(args.poll_seconds)))
    try:
        relay(
            args.metrics,
            writer,
            args.poll_seconds,
            from_start=not args.tail,
            stop=lambda: stopping,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
