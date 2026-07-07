"""Shared helpers for homelab showcase + live lobby daemons."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__import__("os").environ.get("DATA_DIR", "/data"))
POLICY_DIR = DATA_DIR / "policy"
REVISION_PATH = DATA_DIR / "policy_revision.txt"


def hf_policy_revision(run_name: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().get_paths_info("djmango/openfront-rl", [f"{run_name}/policy.pt"])[0]
    return str(getattr(info, "blob_id", "") or getattr(info, "last_modified", ""))


def ensure_ae(ae_path: Path) -> Path:
    dest = REPO / ae_path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    name = dest.name
    src = hf_hub_download("djmango/openfront-tile-autoencoder", name)
    shutil.copy2(src, dest)
    return dest


def ensure_policy(run_name: str) -> Path:
    dest = POLICY_DIR / run_name / "policy.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and REVISION_PATH.exists():
        if REVISION_PATH.read_text().strip() == hf_policy_revision(run_name):
            return dest
    from huggingface_hub import hf_hub_download

    src = hf_hub_download("djmango/openfront-rl", f"{run_name}/policy.pt")
    shutil.copy2(src, dest)
    REVISION_PATH.write_text(hf_policy_revision(run_name))
    return dest


def policy_meta(policy: Path) -> dict:
    import torch

    pt = torch.load(policy, map_location="cpu", weights_only=False)
    return {
        "policy_update": pt.get("update"),
        "policy_stage": pt.get("stage"),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()[:16],
    }


def write_json(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
