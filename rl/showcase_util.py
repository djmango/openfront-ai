"""Shared helpers for homelab showcase + live lobby daemons."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
POLICY_DIR = DATA_DIR / "policy"
CLIPS_DIR = DATA_DIR / "clips"
REVISION_PATH = DATA_DIR / "policy_revision.txt"


def showcase_seeds() -> list[str]:
    raw = os.environ.get("SHOWCASE_SEEDS", "showcase0,showcase1,showcase2")
    return [s.strip() for s in raw.split(",") if s.strip()]


def showcase_maps() -> list[str]:
    """Maps to pre-render for /watch; rotates one per hour on the hub."""
    raw = os.environ.get("SHOWCASE_MAPS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    from rl.curriculum import ALL_MAPS

    return list(ALL_MAPS)


def map_seed(map_name: str) -> str:
    return map_name.lower().replace(" ", "_")


def featured_showcase_entry(state: dict, now: float | None = None) -> dict | None:
    """Pick the replay entry for the current hour (UTC)."""
    entries = state.get("maps")
    if entries:
        t = time.time() if now is None else now
        return entries[int(t // 3600) % len(entries)]
    if state.get("game_id"):
        return state
    return None


def featured_game_id(state: dict) -> str | None:
    entry = featured_showcase_entry(state)
    if not entry:
        return None
    gid = entry.get("game_id")
    return str(gid) if gid else None


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

    hf_name = os.environ.get("AE_HF_NAME")
    if not hf_name:
        hf_name = "ae_v31_d8c32.pt" if "ae_v31_d8c32" in str(ae_path) else dest.name
    src = hf_hub_download("djmango/openfront-tile-autoencoder", hf_name)
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


def hero_clip_urls(state: dict) -> list[str]:
    """Public URLs for pre-rendered landing-page client clips."""
    urls: list[str] = []
    for entry in state.get("hero_clips") or []:
        if isinstance(entry, str):
            urls.append(entry if entry.startswith("/") else f"/archive/clips/{entry}")
        elif isinstance(entry, dict) and entry.get("url"):
            urls.append(str(entry["url"]))
    if urls:
        return urls
    if CLIPS_DIR.is_dir():
        for path in sorted(CLIPS_DIR.glob("*.webm")):
            urls.append(f"/archive/clips/{path.name}")
    return urls
