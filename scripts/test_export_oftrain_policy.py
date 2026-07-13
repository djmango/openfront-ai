import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from ae.model_v3 import MAX_SLOTS
from rl.obs import (
    BUILD_TYPES,
    C_GRID,
    C_GRID_FINE,
    N_ACTIONS,
    N_LOCAL,
    N_SCALARS,
    NUKE_TYPES,
    P_FEAT,
)
from rl.policy import Policy
from scripts.export_oftrain_policy import (
    ARCHITECTURE,
    MAPPING,
    ExportError,
    convert_tensors,
    export_policy,
)


def python_state(seed: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        key: torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.01
        for key, value in Policy().state_dict().items()
    }


def rust_state_from_python(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = {}
    for entry in MAPPING:
        value = state[entry.destination]
        if entry.transform == "identity":
            source[entry.sources[0]] = value.clone()
        elif entry.transform == "concat_qkv_dim0":
            chunks = value.chunk(3, dim=0)
            source.update({key: chunk.clone() for key, chunk in zip(entry.sources, chunks)})
        else:
            raise AssertionError(entry.transform)
    return source


def fixed_observation() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    batch, fine_h, fine_w = 2, 6, 8
    coarse_h, coarse_w = 3, 4
    return {
        "grid_coarse": torch.randn(batch, C_GRID, coarse_h, coarse_w, generator=generator),
        "grid_coarse_valid": torch.ones(batch, coarse_h, coarse_w),
        "grid_fine": torch.randn(batch, C_GRID_FINE, fine_h, fine_w, generator=generator),
        "grid_fine_valid": torch.ones(batch, fine_h, fine_w),
        "fine_coverage": torch.ones(batch, fine_h, fine_w),
        "fine_origin": torch.zeros(batch, 2, dtype=torch.long),
        "legal_tile_fine": torch.ones(batch, fine_h, fine_w),
        "legal_tile_coarse": torch.ones(batch, coarse_h, coarse_w),
        "coarse_has_land": torch.ones(batch, coarse_h, coarse_w),
        "coarse_has_water": torch.ones(batch, coarse_h, coarse_w),
        "players": torch.randn(batch, MAX_SLOTS, P_FEAT, generator=generator),
        "pmask": torch.ones(batch, MAX_SLOTS),
        "local": torch.randn(batch, N_LOCAL, 64, 64, generator=generator),
        "scalars": torch.randn(batch, N_SCALARS, generator=generator),
        "legal_actions": torch.ones(batch, N_ACTIONS),
        "legal_build": torch.ones(batch, len(BUILD_TYPES)),
        "legal_nuke": torch.ones(batch, len(NUKE_TYPES)),
    }


class ExportOftrainPolicyTests(unittest.TestCase):
    def setUp(self):
        self.python = python_state()
        self.rust = rust_state_from_python(self.python)

    def test_mapping_is_complete_and_qkv_order_is_explicit(self):
        converted = convert_tensors(self.rust)
        self.assertEqual(set(converted), set(self.python))
        key = "player_tf.layers.0.self_attn.in_proj_weight"
        self.assertTrue(torch.equal(converted[key], self.python[key]))
        q, k, v = converted[key].chunk(3, dim=0)
        self.assertTrue(torch.equal(q, self.rust["tf.0.q.weight"]))
        self.assertTrue(torch.equal(k, self.rust["tf.0.k.weight"]))
        self.assertTrue(torch.equal(v, self.rust["tf.0.v.weight"]))

    def test_refuses_missing_extra_and_shape_mismatches(self):
        missing = dict(self.rust)
        missing.pop("head_value.bias")
        with self.assertRaisesRegex(ExportError, "source key mismatch.*head_value.bias"):
            convert_tensors(missing)

        extra = dict(self.rust)
        extra["not_a_policy_tensor"] = torch.zeros(1)
        with self.assertRaisesRegex(ExportError, "source key mismatch.*not_a_policy_tensor"):
            convert_tensors(extra)

        wrong = dict(self.rust)
        wrong["head_action.weight"] = wrong["head_action.weight"][:-1]
        with self.assertRaisesRegex(ExportError, "shape mismatch for head_action.weight"):
            convert_tensors(wrong)

        wrong_dtype = dict(self.rust)
        wrong_dtype["head_action.weight"] = wrong_dtype["head_action.weight"].half()
        with self.assertRaisesRegex(ExportError, "dtype mismatch for head_action.weight"):
            convert_tensors(wrong_dtype)

        balanced_qkv = dict(self.rust)
        balanced_qkv["tf.0.q.weight"] = balanced_qkv["tf.0.q.weight"][:-1]
        balanced_qkv["tf.0.k.weight"] = torch.cat(
            [balanced_qkv["tf.0.k.weight"], balanced_qkv["tf.0.k.weight"][:1]]
        )
        with self.assertRaisesRegex(ExportError, "shape mismatch.*tf.0.q.weight"):
            convert_tensors(balanced_qkv)

    def test_exported_checkpoint_loads_strictly_and_carries_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "latest.safetensors"
            state = directory / "latest.state.json"
            output = directory / "policy.pt"
            save_file(self.rust, source)
            state.write_text(json.dumps({"update": 123, "stage": 6, "other": "preserved nowhere"}))

            export_policy(source, state, output)
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["update"], 123)
            self.assertEqual(checkpoint["stage"], 6)
            self.assertEqual(checkpoint["architecture"], ARCHITECTURE)
            self.assertEqual(
                checkpoint["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )
            loaded = Policy()
            loaded.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def test_fixed_observation_all_heads_match_after_round_trip(self):
        reference = Policy().eval()
        reference.load_state_dict(self.python, strict=True)
        exported = Policy().eval()
        exported.load_state_dict(convert_tensors(self.rust), strict=True)
        observation = fixed_observation()
        with torch.no_grad():
            expected = reference(observation)
            actual = exported(observation)
        self.assertEqual(set(expected), set(actual))
        for head in expected:
            torch.testing.assert_close(actual[head], expected[head], rtol=1e-6, atol=1e-7)

    def test_state_json_requires_non_negative_integer_update_and_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "weights.safetensors"
            save_file(self.rust, source)
            state = directory / "state.json"
            output = directory / "policy.pt"
            for invalid in (
                {"update": True, "stage": 0},
                {"update": 0, "stage": -1},
                {"update": "4", "stage": 0},
            ):
                state.write_text(json.dumps(invalid))
                with self.assertRaises(ExportError):
                    export_policy(source, state, output)


if __name__ == "__main__":
    unittest.main()
