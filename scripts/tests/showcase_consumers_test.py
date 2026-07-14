#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from rl import showcase_util
from rl.watch import load_policy_checkpoint
from scripts.export_onnx import load_export_policy


class ShowcasePolicyTest(unittest.TestCase):
    def test_hf_paths_and_revision_select_current_and_legacy_formats(self) -> None:
        self.assertEqual(
            showcase_util.hf_policy_paths("ppo_v81"),
            ("ppo_v81/latest.safetensors", "ppo_v81/latest.state.json"),
        )
        self.assertEqual(
            showcase_util.hf_policy_paths("ppo_v5"),
            ("ppo_v5/policy.pt", None),
        )
        api = MagicMock()
        api.get_paths_info.return_value = [SimpleNamespace(blob_id="weights-revision")]
        with patch("huggingface_hub.HfApi", return_value=api):
            self.assertEqual(showcase_util.hf_policy_revision("ppo_v81"), "weights-revision")
        api.get_paths_info.assert_called_once_with(
            showcase_util.HF_POLICY_REPO,
            ["ppo_v81/latest.safetensors"],
        )

    def test_downloads_and_caches_weights_with_state_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            remote_weights = temp / "remote.safetensors"
            remote_state = temp / "remote.state.json"
            remote_weights.write_bytes(b"safe weights")
            remote_state.write_text('{"update": 81, "stage": 4}')
            policy_dir = temp / "cache"
            revision_path = temp / "revision.txt"

            def download(_repo: str, path: str) -> str:
                return str(remote_state if path.endswith(".json") else remote_weights)

            with (
                patch.object(showcase_util, "POLICY_DIR", policy_dir),
                patch.object(showcase_util, "REVISION_PATH", revision_path),
                patch.object(showcase_util, "hf_policy_revision", return_value="rev-81"),
                patch("huggingface_hub.hf_hub_download", side_effect=download) as hub_download,
            ):
                policy = showcase_util.ensure_policy("ppo_v81")
                self.assertEqual(policy, policy_dir / "ppo_v81/latest.safetensors")
                self.assertEqual(policy.read_bytes(), b"safe weights")
                self.assertEqual(
                    policy.with_name("latest.state.json").read_text(),
                    '{"update": 81, "stage": 4}',
                )
                self.assertEqual(revision_path.read_text(), "rev-81")
                self.assertEqual(
                    showcase_util.policy_meta(policy),
                    {
                        "policy_update": 81,
                        "policy_stage": 4,
                        "policy_sha256": hashlib.sha256(b"safe weights").hexdigest()[:16],
                    },
                )
                self.assertEqual(
                    hub_download.call_args_list,
                    [
                        call(showcase_util.HF_POLICY_REPO, "ppo_v81/latest.safetensors"),
                        call(showcase_util.HF_POLICY_REPO, "ppo_v81/latest.state.json"),
                    ],
                )
                showcase_util.ensure_policy("ppo_v81")
                self.assertEqual(hub_download.call_count, 2)

    def test_state_metadata_is_required_for_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            policy = Path(raw) / "latest.safetensors"
            policy.write_bytes(b"weights")
            with self.assertRaisesRegex(FileNotFoundError, "state metadata"):
                showcase_util.policy_meta(policy)


class LoaderDispatchTest(unittest.TestCase):
    def test_watch_dispatches_safetensors_to_strict_loader(self) -> None:
        policy = MagicMock()
        state = {"update": 81, "stage": 4}
        with patch(
            "scripts.policy_safetensors.load_oftrain_safetensors",
            return_value={"state": state},
        ) as loader:
            self.assertIs(
                load_policy_checkpoint(policy, "/cache/ppo_v81/latest.safetensors", "cpu"),
                state,
            )
        loader.assert_called_once_with(policy, Path("/cache/ppo_v81/latest.safetensors"))

    def test_policy_pt_is_only_allowed_for_named_legacy_runs(self) -> None:
        policy = MagicMock()
        state = {"model_state_dict": {"weight": "value"}, "update": 7}
        with patch("torch.load", return_value=state) as torch_load:
            self.assertIs(
                load_policy_checkpoint(policy, "/cache/ppo_v7/policy.pt", "cpu"),
                state,
            )
        torch_load.assert_called_once()
        policy.load_state_dict.assert_called_once_with(
            state["model_state_dict"], strict=True
        )
        with self.assertRaisesRegex(ValueError, "only.*legacy"):
            load_policy_checkpoint(policy, "/cache/ppo_v81/policy.pt", "cpu")

    def test_onnx_loader_is_strict_and_never_converts_to_policy_pt(self) -> None:
        policy = MagicMock()
        state = {"update": 81}
        with (
            patch("scripts.export_onnx.Policy", return_value=policy),
            patch(
                "scripts.policy_safetensors.load_oftrain_safetensors",
                return_value={"state": state},
            ) as loader,
        ):
            loaded_policy, loaded_state = load_export_policy(
                "/cache/ppo_v81/latest.safetensors"
            )
        self.assertIs(loaded_policy, policy)
        self.assertIs(loaded_state, state)
        loader.assert_called_once_with(
            policy, Path("/cache/ppo_v81/latest.safetensors")
        )
        with patch("scripts.export_onnx.Policy", return_value=policy):
            with self.assertRaisesRegex(ValueError, "only.*legacy"):
                load_export_policy("/cache/ppo_v81/policy.pt")


if __name__ == "__main__":
    unittest.main()
