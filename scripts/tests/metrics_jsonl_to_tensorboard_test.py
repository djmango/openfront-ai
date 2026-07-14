#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "metrics_jsonl_to_tensorboard.py"
SPEC = importlib.util.spec_from_file_location("metrics_jsonl_to_tensorboard", SCRIPT)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))


class MetricsBridgeTest(unittest.TestCase):
    def test_update_uses_env_steps_and_python_compatible_tags(self) -> None:
        writer = FakeWriter()
        bridge.write_row(
            writer,
            {
                "update": 7,
                "env_steps": 1234,
                "stage": 2,
                "loss/pg": 0.25,
                "win_rate": None,
                "gpu": {"util_pct": 97.0},
            },
            0,
        )

        self.assertIn(("loss/policy", 0.25, 1234), writer.scalars)
        self.assertIn(("curriculum/stage", 2.0, 1234), writer.scalars)
        self.assertIn(("gpu/util_pct", 97.0, 1234), writer.scalars)
        self.assertFalse(any(tag == "win_rate" for tag, _, _ in writer.scalars))

    def test_episode_rows_get_monotonic_steps_and_reward_tags(self) -> None:
        writer = FakeWriter()
        next_step = bridge.write_row(
            writer,
            {
                "event": "episode",
                "update": 9,
                "map": "World",
                "reward": 1.5,
                "rehearsal": True,
                "reward/action_churn": -0.2,
            },
            4,
        )

        self.assertEqual(next_step, 5)
        self.assertIn(("episode/reward", 1.5, 4), writer.scalars)
        self.assertIn(("curriculum/rehearsal", 1.0, 4), writer.scalars)
        self.assertIn(("reward/action_churn", -0.2, 4), writer.scalars)

    def test_non_finite_and_text_values_are_ignored(self) -> None:
        writer = FakeWriter()
        bridge.write_row(
            writer,
            {"update": 1, "bad": float("nan"), "label": "x", "good": 3},
            0,
        )
        self.assertEqual(writer.scalars, [("good", 3.0, 1)])


if __name__ == "__main__":
    unittest.main()
