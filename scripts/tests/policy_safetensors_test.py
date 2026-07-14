#!/usr/bin/env python3
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import torch

from scripts.policy_safetensors import (
    _python_key,
    load_oftrain_safetensors,
    map_oftrain_state,
)


class PolicySafetensorsMappingTest(unittest.TestCase):
    def test_maps_towers_and_concatenates_qkv_strictly(self) -> None:
        source = {
            "tf.0.q.weight": torch.full((2, 2), 1.0),
            "tf.0.k.weight": torch.full((2, 2), 2.0),
            "tf.0.v.weight": torch.full((2, 2), 3.0),
            "head_action.weight": torch.zeros(3, 4),
        }
        expected = {
            "player_tf.layers.0.self_attn.in_proj_weight": torch.empty(6, 2),
            "head_action.weight": torch.empty(3, 4),
        }
        mapped = map_oftrain_state(source, expected)
        self.assertEqual(mapped.keys(), expected.keys())
        self.assertEqual(
            mapped["player_tf.layers.0.self_attn.in_proj_weight"][:, 0].tolist(),
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(
            _python_key("grid_fine.block.3.conv2.bias"),
            "grid_fine_net.5.conv2.bias",
        )

    def test_rejects_unmapped_or_shape_mismatched_tensors(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmapped"):
            map_oftrain_state({"unknown.weight": torch.zeros(1)}, {})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            map_oftrain_state(
                {"head_action.weight": torch.zeros(2, 2)},
                {"head_action.weight": torch.empty(3, 2)},
            )

    def test_transient_python_mapping_rejects_recurrent_schema_v2(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "latest.safetensors"
            checkpoint.write_bytes(b"not-read")
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "oftrain-safetensors",
                        "manifest_schema_version": 1,
                        "architecture": {
                            "schema_version": 2,
                            "recurrent": {
                                "context_schema": "action-outcome-v1",
                            },
                        },
                    }
                )
            )
            with mock.patch("safetensors.torch.load_file"):
                with self.assertRaisesRegex(ValueError, "recurrent.*v2"):
                    load_oftrain_safetensors(torch.nn.Linear(1, 1), checkpoint)


if __name__ == "__main__":
    unittest.main()
