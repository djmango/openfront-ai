"""Background worker for the homelab eval showcase.

Pulls the latest RL policy checkpoint from Hugging Face, runs greedy episodes
via rl.watch (with --record for the MODEL overlay sidecar), renders trimmed
client WebM clips, and writes /data/state.json for the hub + archive API.

Usage (normally started by docker/entrypoint.sh):
  uv run python scripts/eval_daemon.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rl.showcase_util import (
    CLIPS_DIR,
    ensure_ae,
    ensure_policy,
    hero_clip_urls,
    hf_policy_revision,
    policy_meta,
    showcase_seeds,
    utc_now,
    write_json,
)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RECORDS_DIR = DATA_DIR / "records"
STATE_PATH = DATA_DIR / "state.json"
AE_PATH = Path(os.environ.get("AE_CKPT", "runs/ae_v31_d8c32/ae_v3.pt"))
RUN_NAME = os.environ.get("RUN_NAME", "ppo_v4")
STAGE = int(os.environ.get("STAGE", "4"))
MAP = os.environ.get("MAP") or None
SHOWCASE_WATCH_STAGE = int(os.environ.get("SHOWCASE_WATCH_STAGE", "0"))
REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", "6"))
CLIP_MAX_SEC = int(os.environ.get("CLIP_MAX_SEC", "90"))
CLIP_WIDTH = int(os.environ.get("CLIP_WIDTH", "1920"))
CLIP_HEIGHT = int(os.environ.get("CLIP_HEIGHT", "1080"))
CLIP_CRF = int(os.environ.get("CLIP_CRF", "18"))


def log(msg: str) -> None:
    print(f"[eval_daemon] {msg}", flush=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def policy_changed(run_name: str) -> bool:
    from rl.showcase_util import REVISION_PATH

    if not REVISION_PATH.exists():
        return True
    try:
        return REVISION_PATH.read_text().strip() != hf_policy_revision(run_name)
    except Exception as exc:
        log(f"revision check failed ({exc}); regenerating")
        return True


def needs_showcase(state: dict, run_name: str) -> bool:
    if not state.get("game_id"):
        return True
    if policy_changed(run_name):
        return True
    urls = hero_clip_urls(state)
    if not urls:
        return True
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        if not (CLIPS_DIR / name).is_file():
            return True
    return False


def run_watch(policy: Path, ae: Path, seed: str, record: Path, stage: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "rl.watch",
        "--policy",
        str(policy),
        "--ckpt",
        str(ae),
        "--stage",
        str(stage),
        "--seed",
        seed,
        "--record",
        str(record),
    ]
    if MAP:
        cmd.extend(["--map", MAP])
    subprocess.run(cmd, cwd=REPO, check=True)


def render_client_clip(record: Path, out: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/render_client_replay.py",
        "--record",
        str(record),
        "--out",
        str(out),
        "--reuse-services",
        "--trim-gameplay",
        "--max-duration",
        str(CLIP_MAX_SEC),
        "--width",
        str(CLIP_WIDTH),
        "--height",
        str(CLIP_HEIGHT),
        "--crf",
        str(CLIP_CRF),
    ]
    subprocess.run(cmd, cwd=REPO, check=True)


def generate_clip(policy: Path, ae: Path, seed: str) -> dict:
    base = f"{RUN_NAME}_s{SHOWCASE_WATCH_STAGE}_{seed}"
    record = RECORDS_DIR / f"{base}.json"
    clip = CLIPS_DIR / f"{seed}.webm"
    if not record.exists():
        log(f"clip {seed}: rl.watch stage {SHOWCASE_WATCH_STAGE} -> {record.name}")
        run_watch(policy, ae, seed, record, SHOWCASE_WATCH_STAGE)
    else:
        log(f"clip {seed}: reusing {record.name}")
    if not clip.exists():
        log(f"clip {seed}: render client video -> {clip.name}")
        render_client_clip(record, clip)
    else:
        log(f"clip {seed}: reusing {clip.name}")
    meta = json.loads(record.read_text())
    return {
        "seed": seed,
        "game_id": meta["info"]["gameID"],
        "map": meta["info"].get("map") or MAP,
        "clip": str(clip),
        "url": f"/archive/clips/{clip.name}",
    }


def generate_showcase(policy: Path, ae: Path) -> dict:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    clip_infos: list[dict] = []
    for seed in showcase_seeds():
        try:
            clip_infos.append(generate_clip(policy, ae, seed))
        except Exception as exc:
            log(f"clip {seed} failed: {exc}")

    if not clip_infos:
        raise RuntimeError("no showcase clips generated")

    primary = clip_infos[0]
    state = {
        "game_id": primary["game_id"],
        "run_name": RUN_NAME,
        "stage": STAGE,
        "map": primary.get("map"),
        "record": str(RECORDS_DIR / f"{RUN_NAME}_s{STAGE}_{primary['seed']}.json"),
        "hero_clips": [c["url"] for c in clip_infos],
        "clips": clip_infos,
        "generated_at": utc_now(),
        **policy_meta(policy),
    }
    write_json(STATE_PATH, state)
    from rl.showcase_util import REVISION_PATH

    REVISION_PATH.write_text(hf_policy_revision(RUN_NAME))
    log(
        f"showcase ready: {len(clip_infos)} clip(s), "
        f"game_id={state['game_id']} update={state.get('policy_update')}"
    )
    return state


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ae = ensure_ae(AE_PATH)

    while True:
        try:
            if needs_showcase(load_state(), RUN_NAME):
                policy = ensure_policy(RUN_NAME)
                generate_showcase(policy, ae)
            else:
                log(f"policy unchanged ({RUN_NAME}); next check in {REFRESH_HOURS}h")
        except Exception as exc:
            log(f"showcase generation failed: {exc}")
            write_json(
                STATE_PATH,
                {**load_state(), "error": str(exc), "failed_at": utc_now()},
            )

        time.sleep(max(REFRESH_HOURS, 0.25) * 3600)


if __name__ == "__main__":
    main()
