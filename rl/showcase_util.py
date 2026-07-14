"""Shared helpers for homelab showcase + live lobby daemons."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
POLICY_DIR = DATA_DIR / "policy"
CLIPS_DIR = DATA_DIR / "clips"
REVISION_PATH = DATA_DIR / "policy_revision.txt"
HF_POLICY_REPO = "djmango/openfront-rl"
LEGACY_POLICY_RUNS = frozenset({"ppo_v5", "ppo_v7"})


def showcase_seeds() -> list[str]:
    raw = os.environ.get("SHOWCASE_SEEDS", "showcase0,showcase1,showcase2")
    return [s.strip() for s in raw.split(",") if s.strip()]


def showcase_maps() -> list[str]:
    """Maps to pre-render for /watch; hub picks a random featured entry."""
    raw = os.environ.get("SHOWCASE_MAPS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    from rl.curriculum import ALL_MAPS

    return list(ALL_MAPS)


def map_seed(map_name: str) -> str:
    return map_name.lower().replace(" ", "_")


def featured_showcase_entry(state: dict, now: float | None = None) -> dict | None:
    """Pick a random replay entry from ``state["maps"]``.

    ``now`` is accepted for call-site compatibility but ignored. Falls back to
    a legacy top-level ``game_id`` entry when ``maps`` is empty.
    """
    del now  # formerly used for hourly UTC rotation
    entries = state.get("maps")
    if entries:
        return random.choice(entries)
    if state.get("game_id"):
        return state
    return None


def featured_game_id(state: dict) -> str | None:
    entry = featured_showcase_entry(state)
    if not entry:
        return None
    gid = entry.get("game_id")
    return str(gid) if gid else None


def hf_policy_paths(run_name: str) -> tuple[str, str | None]:
    """Return the weights and state paths published for a policy run."""
    if run_name in LEGACY_POLICY_RUNS:
        return f"{run_name}/policy.pt", None
    return f"{run_name}/latest.safetensors", f"{run_name}/latest.state.json"


def hf_policy_revision(run_name: str) -> str:
    from huggingface_hub import HfApi

    weights_path, _ = hf_policy_paths(run_name)
    info = HfApi().get_paths_info(HF_POLICY_REPO, [weights_path])[0]
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
    weights_path, state_path = hf_policy_paths(run_name)
    dest = POLICY_DIR / weights_path
    state_dest = POLICY_DIR / state_path if state_path else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    cache_complete = dest.exists() and (state_dest is None or state_dest.exists())
    if cache_complete and REVISION_PATH.exists():
        if REVISION_PATH.read_text().strip() == hf_policy_revision(run_name):
            return dest
    from huggingface_hub import hf_hub_download

    revision = hf_policy_revision(run_name)
    src = hf_hub_download(HF_POLICY_REPO, weights_path)
    shutil.copy2(src, dest)
    if state_path and state_dest:
        state_src = hf_hub_download(HF_POLICY_REPO, state_path)
        shutil.copy2(state_src, state_dest)
    REVISION_PATH.write_text(revision)
    return dest


def policy_meta(policy: Path) -> dict:
    if policy.suffix == ".safetensors":
        state_path = policy.with_name(f"{policy.stem}.state.json")
        if not state_path.is_file():
            raise FileNotFoundError(f"missing safetensors state metadata: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    elif policy.suffix == ".pt" and policy.parent.name in LEGACY_POLICY_RUNS:
        import torch

        state = torch.load(policy, map_location="cpu", weights_only=False)
    else:
        raise ValueError(
            "policy must be a current .safetensors checkpoint or an explicitly "
            "legacy ppo_v5/ppo_v7 policy.pt"
        )
    return {
        "policy_update": state.get("update"),
        "policy_stage": state.get("stage"),
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
