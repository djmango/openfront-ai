"""Render a GameRecord as a video of the REAL OpenFront client - actual
game graphics: terrain art, units, factories, nukes, leaderboard, the lot.

Drives a headless Chromium (Playwright) against the local dev client,
replaying the record served by the archive-API shim. Client hooks (patch in
patches/client-replay-tooling.patch, applied in a detached worktree at the
record's engine commit so the openfront submodule stays clean):
  - replayViewAs: the viewer adopts the agent's identity - self-player
    styling, gold spawn ring, crown when first, "You Won!" modal
  - replayFitMap: camera starts centered on the whole map
  - RlDebugOverlay: in-client model panel (chosen action, value,
    action-probability bars, recent-actions log) synced to the sim tick;
    it fetches <apiHost>/debug/<gameID>, which the shim serves from the
    rl.watch debug sidecar (<record>.debug.json)

The same overlay works in a normal browser (manual serve_replay workflow)
and in live games against rl.play --debug-port (localStorage rlDebugHost).

Usage:
  uv run python scripts/render_client_replay.py \
      --record replays/v3_stage0.json --out replays/v3_client.webm

Requires: `uv run playwright install chromium` (one-time). Starts
serve_replay.py itself; starts the vite client too unless one is already
on --client-port.
"""

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url: str, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1.0)
    raise SystemExit(f"timed out waiting for {url}")


PATCH = REPO / "patches/client-replay-tooling.patch"
OPENFRONT = REPO / "openfront"


def record_engine_commit(record: Path) -> str:
    """Engine commit the record ran on; fall back to the submodule pin."""
    data = json.loads(record.read_text())
    commit = data.get("gitCommit") or data.get("info", {}).get("gitCommit")
    if commit and commit != "DEV":
        return commit
    return subprocess.check_output(
        ["git", "-C", str(OPENFRONT), "rev-parse", "HEAD"], text=True,
    ).strip()


@contextlib.contextmanager
def client_worktree(commit: str):
    """Detached openfront worktree at `commit` with replay-tooling patch applied.

    The main submodule working tree is never touched; node_modules is
    symlinked from the pin checkout when present."""
    wt = Path(tempfile.mkdtemp(prefix="openfront-client-"))
    try:
        subprocess.run(
            ["git", "-C", str(OPENFRONT), "worktree", "add", "--detach", str(wt), commit],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "apply", str(PATCH)], cwd=wt, check=True,
        )
        pin_nm = OPENFRONT / "node_modules"
        if pin_nm.is_dir() and not (wt / "node_modules").exists():
            os.symlink(pin_nm, wt / "node_modules")
        print(f"client worktree {commit[:12]} at {wt}")
        yield wt
    finally:
        subprocess.run(
            ["git", "-C", str(OPENFRONT), "worktree", "remove", "--force", str(wt)],
            capture_output=True,
        )
        shutil.rmtree(wt, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="GameRecord JSON from rl.watch --record")
    ap.add_argument("--out", default=None, help="output .webm (default: <record>.client.webm)")
    ap.add_argument("--speed", default="max", choices=["0.5", "1", "2", "max"])
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=1200, help="max seconds to wait for game end")
    ap.add_argument("--api-port", type=int, default=8987)
    ap.add_argument("--client-port", type=int, default=9000)
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                    help="model debug panel from <record>.debug.json (--no-overlay to disable)")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    record = Path(args.record).resolve()
    game_id = json.loads(record.read_text())["info"]["gameID"]
    out = Path(args.out or record.with_suffix(".client.webm"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"gameID {game_id} -> {out}")

    sidecar = record.with_suffix(".debug.json")
    if args.overlay and sidecar.exists():
        print(f"model overlay from {sidecar.name}")
    elif args.overlay:
        print(f"no {sidecar.name} - rendering without the model overlay "
              "(re-run rl.watch --record to get one)")

    commit = record_engine_commit(record)
    procs: list[subprocess.Popen] = []
    with client_worktree(commit) as client_dir:
        try:
            procs.append(subprocess.Popen(
                [sys.executable, "scripts/serve_replay.py",
                 "--records", str(record.parent), "--port", str(args.api_port)],
                cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
            if not port_open(args.client_port):
                print("starting vite client (first boot takes ~15s)...")
                # IPv4 explicitly: vite defaults to ::1 which headless chromium
                # won't reach when resolving localhost to 127.0.0.1.
                procs.append(subprocess.Popen(
                    ["npm", "run", "start:client", "--", "--host", "127.0.0.1"],
                    cwd=client_dir,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            wait_http(f"http://localhost:{args.client_port}", 90)

            speed_label = {"0.5": "×0.5", "1": "×1", "2": "×2"}.get(args.speed)

            with sync_playwright() as pw, tempfile.TemporaryDirectory() as td:
                # Client rejects software WebGL; force hardware ANGLE (Metal on
                # macOS falls back gracefully elsewhere).
                browser = pw.chromium.launch(
                    headless=not args.headed,
                    args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"],
                )
                ctx = browser.new_context(
                    viewport={"width": args.width, "height": args.height},
                    record_video_dir=td,
                    record_video_size={"width": args.width, "height": args.height},
                )
                # Must land before the app boots: archive API base, agent
                # identity, map-centered camera, no auto-focus on the player.
                overlay_flag = "1" if args.overlay else "0"
                ctx.add_init_script(
                    f'localStorage.setItem("apiHost", "http://localhost:{args.api_port}");'
                    'localStorage.setItem("replayViewAs", "1");'
                    'localStorage.setItem("replayFitMap", "1");'
                    f'localStorage.setItem("rlDebugOverlay", "{overlay_flag}");'
                    'localStorage.setItem("settings.goToPlayer", "false");'
                    'localStorage.setItem("username", "AGENT");'
                )
                page = ctx.new_page()
                page.goto(f"http://localhost:{args.client_port}/game/{game_id}")

                # Sidebar replay toggle appearing = game booted and is ticking.
                toggle = page.locator('img[alt="replay"]')
                toggle.wait_for(state="visible", timeout=120_000)
                toggle.click()
                panel = page.locator("replay-panel button").first
                panel.wait_for(state="visible", timeout=10_000)
                if speed_label:
                    page.locator("replay-panel button", has_text=speed_label).click()
                else:
                    page.locator("replay-panel button").last.click()  # max speed
                print("replay running...")

                # Done when the win modal shows (or the turn feed runs dry).
                t0 = time.time()
                while time.time() - t0 < args.timeout:
                    if page.locator("win-modal div.fixed").count() > 0:
                        print(f"game over after {time.time() - t0:.0f}s")
                        page.wait_for_timeout(4000)  # linger on the win screen
                        break
                    page.wait_for_timeout(2000)
                else:
                    print("timeout reached; saving what we have")

                page.close()
                video = page.video.path() if page.video else None
                ctx.close()
                browser.close()
                if not video or not Path(video).exists():
                    raise SystemExit("no video captured")
                shutil.move(video, out)

            print(f"wrote {out}")
        finally:
            for p in procs:
                p.terminate()


if __name__ == "__main__":
    main()
