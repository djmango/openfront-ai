#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from thinking_compact import compact_debug, compact_debug_json, load_thinking_for_record  # noqa: E402


def test_compact_is_small_and_keeps_endpoints(tmp_path: Path | None = None) -> None:
    actions = [f"a{i}" for i in range(21)]
    actions[0] = "noop"
    actions[1] = "attack"
    log = []
    for i in range(120):
        action = "noop" if i % 7 != 0 else "attack"
        probs = [0.01] * 21
        probs[0 if action == "noop" else 1] = 0.8
        log.append(
            {
                "tick": (i + 1) * 10,
                "action": action,
                "desc": f"{action} x",
                "value": 1.23 if action == "attack" else 0.1,
                "probs": probs,
            }
        )
    debug = {"actions": actions, "log": log, "outcome": "win", "end_tick": 1200}
    blob = compact_debug(debug, stride=15)
    text = compact_debug_json(debug, stride=15)
    assert blob["n"] == 120
    assert blob["o"] == "win"
    assert len(text) < 8_000
    assert len(blob["s"]) < 60
    # First and last ticks retained.
    ticks = [row[0] for row in blob["s"]]
    assert ticks[0] == 10
    assert ticks[-1] == 1200


def test_load_from_debug_sidecar(tmp_path: Path) -> None:
    record = tmp_path / "game.json"
    record.write_text("{}")
    debug = {
        "actions": ["noop", "attack"],
        "log": [
            {
                "tick": 10,
                "action": "attack",
                "desc": "attack",
                "value": 0.5,
                "probs": [0.2, 0.8],
            }
        ],
        "outcome": "death",
        "end_tick": 10,
    }
    (tmp_path / "game.debug.json").write_text(json.dumps(debug))
    text = load_thinking_for_record(record)
    obj = json.loads(text)
    assert obj["o"] == "death"
    assert obj["s"]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_compact_is_small_and_keeps_endpoints()
    with TemporaryDirectory() as d:
        test_load_from_debug_sidecar(Path(d))
    print("thinking_compact_test: ok")
