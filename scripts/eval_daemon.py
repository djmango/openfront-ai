"""Background worker for the homelab eval showcase.

Pulls the latest RL policy checkpoint from Hugging Face, runs one greedy
episode via rl.watch (with --record for the MODEL overlay sidecar), and
writes /data/state.json so serve_replay.py can redirect visitors to the
current replay.

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
    ensure_ae,
    ensure_policy,
    hf_policy_revision,
    policy_meta,
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
SEED = os.environ.get("SEED", "showcase0")
REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", "6"))


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


def generate_showcase(policy: Path, ae: Path) -> dict:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{RUN_NAME}_s{STAGE}_{SEED}"
    record = RECORDS_DIR / f"{base}.json"
    log(f"running rl.watch -> {record.name}")

    cmd = [
        sys.executable,
        "-m",
        "rl.watch",
        "--policy",
        str(policy),
        "--ckpt",
        str(ae),
        "--stage",
        str(STAGE),
        "--seed",
        SEED,
        "--record",
        str(record),
        "--out",
        str(RECORDS_DIR / f"{base}.webm"),
        "--no-debug",
    ]
    if MAP:
        cmd.extend(["--map", MAP])

    subprocess.run(cmd, cwd=REPO, check=True)

    meta = json.loads(record.read_text())
    state = {
        "game_id": meta["info"]["gameID"],
        "run_name": RUN_NAME,
        "stage": STAGE,
        "map": meta["info"].get("map") or MAP,
        "seed": SEED,
        "record": str(record),
        "generated_at": utc_now(),
        **policy_meta(policy),
    }
    write_json(STATE_PATH, state)
    from rl.showcase_util import REVISION_PATH

    REVISION_PATH.write_text(hf_policy_revision(RUN_NAME))
    log(f"showcase ready: game_id={state['game_id']} update={state.get('policy_update')}")
    return state


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ae = ensure_ae(AE_PATH)

    while True:
        try:
            if policy_changed(RUN_NAME) or not load_state().get("game_id"):
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
