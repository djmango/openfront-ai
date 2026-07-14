#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hf_checkpoint_sync.py"
SPEC = importlib.util.spec_from_file_location("hf_checkpoint_sync", SCRIPT)
assert SPEC and SPEC.loader
sync_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_module
SPEC.loader.exec_module(sync_module)


class FakeApi:
    def __init__(self, remote: dict[str, bytes] | None = None) -> None:
        self.remote = remote or {}
        self.uploads: list[dict[str, object]] = []

    def upload_file(self, **kwargs: object) -> None:
        source = Path(str(kwargs["path_or_fileobj"]))
        self.uploads.append({**kwargs, "body": source.read_bytes()})

    def hf_hub_download(self, _repo: str, name: str, **_kwargs: object) -> str:
        if name not in self.remote:
            raise FileNotFoundError(name)
        path = Path(self.temp) / Path(name).name
        path.write_bytes(self.remote[name])
        return str(path)


class CheckpointSyncTest(unittest.TestCase):
    def test_file_selection_is_safetensors_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            selected = {
                "latest.safetensors",
                "latest.state.json",
                "best_eval.safetensors",
                "best_eval.state.json",
                "manifest.json",
                "policy_update100.safetensors",
                "policy_update100.state.json",
            }
            ignored = {
                "latest.ot",
                "policy.pt",
                "policy_update100.ot",
                "latest.tmp.safetensors",
                "metrics.jsonl",
            }
            for name in selected | ignored:
                (checkpoints / name).write_bytes(name.encode())
            self.assertEqual(
                {path.name for path in sync_module.discover(checkpoints)}, selected
            )

    def test_sync_uploads_manifest_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "latest.safetensors").write_bytes(b"weights")
            (checkpoints / "latest.state.json").write_text('{"update": 7}')
            (checkpoints / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "oftrain-safetensors",
                        "manifest_schema_version": 1,
                        "update": 7,
                        "stage": 2,
                    }
                )
            )
            api = FakeApi()
            sync = sync_module.CheckpointSync(
                checkpoints, "owner/repo", "ppo_v81", api
            )
            self.assertEqual(sync.sync_once(), 3)
            self.assertEqual(sync.sync_once(), 0)
            self.assertEqual(
                {call["path_in_repo"] for call in api.uploads},
                {
                    "ppo_v81/latest.safetensors",
                    "ppo_v81/latest.state.json",
                    "ppo_v81/manifest.json",
                },
            )

    def test_restore_requires_safetensors_and_state_and_ignores_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            api = FakeApi(
                {
                    "ppo_v81/latest.safetensors": b"weights",
                    "ppo_v81/latest.state.json": b'{"update": 8}',
                    "ppo_v81/manifest.json": (
                        b'{"format":"oftrain-safetensors",'
                        b'"manifest_schema_version":1,'
                        b'"architecture":{"schema_version":1}}'
                    ),
                    "ppo_v81/latest.ot": b"legacy",
                    "ppo_v81/policy.pt": b"legacy-python",
                }
            )
            api.temp = directory
            destination = Path(directory) / "restore"
            self.assertTrue(
                sync_module.restore_latest(
                    api, "owner/repo", "ppo_v81", destination
                )
            )
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {"latest.safetensors", "latest.state.json", "manifest.json"},
            )

    def test_restore_does_not_install_unpaired_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            api = FakeApi({"ppo_v81/latest.safetensors": b"weights"})
            api.temp = directory
            destination = Path(directory) / "restore"
            self.assertFalse(
                sync_module.restore_latest(
                    api, "owner/repo", "ppo_v81", destination
                )
            )
            self.assertFalse((destination / "latest.safetensors").exists())

    def test_recurrent_manifest_validation_and_mismatch_failures(self) -> None:
        valid = {
            "format": "oftrain-safetensors",
            "manifest_schema_version": 1,
            "architecture": {
                "schema_version": 2,
                "recurrent": {
                    "cell": "gru",
                    "hidden_size": 256,
                    "context_schema": "action-outcome-v1",
                    "context_features": 14,
                    "context_embedding": 128,
                    "bptt_length": 16,
                    "rollout_length": 32,
                    "residual_initialization": "zero-output-projection",
                    "hidden_reset_policy": "episode_done",
                },
            },
        }
        self.assertEqual(
            sync_module.validate_manifest_bytes(json.dumps(valid).encode()),
            valid,
        )
        for mutate in (
            lambda value: value["architecture"].update(schema_version=3),
            lambda value: value["architecture"]["recurrent"].update(hidden_size=0),
            lambda value: value["architecture"]["recurrent"].update(
                context_schema="unknown"
            ),
            lambda value: value["architecture"]["recurrent"].update(
                hidden_reset_policy="never"
            ),
        ):
            candidate = json.loads(json.dumps(valid))
            mutate(candidate)
            with self.assertRaises(ValueError):
                sync_module.validate_manifest_bytes(json.dumps(candidate).encode())


if __name__ == "__main__":
    unittest.main()
