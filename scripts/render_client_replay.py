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

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
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


def chromium_args() -> list[str]:
    base = ["--no-sandbox", "--disable-dev-shm-usage"]
    if platform.system() == "Darwin":
        return base + ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]
    return base + ["--use-gl=angle", "--enable-gpu", "--ignore-gpu-blocklist"]


def trim_video(
    src: Path,
    dst: Path,
    start_sec: float,
    max_duration: float | None,
    *,
    crf: int = 18,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if src != dst:
            shutil.move(src, dst)
        return

    cmd = [ffmpeg, "-y"]
    if start_sec > 0.05:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(src)])
    if max_duration and max_duration > 0:
        cmd.extend(["-t", f"{max_duration:.3f}"])
    cmd.extend([
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
        "-row-mt", "1", "-an", str(dst),
    ])
    subprocess.run(cmd, check=True, capture_output=True)
    if src != dst and src.exists():
        src.unlink()


@contextlib.contextmanager
def client_worktree(commit: str):
    """Detached openfront worktree at `commit` with replay-tooling patch applied."""
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


def render_record(
    record: Path,
    out: Path,
    *,
    speed: str = "max",
    width: int = 1920,
    height: int = 1080,
    device_scale_factor: float = 2,
    crf: int = 18,
    timeout: int = 1200,
    api_port: int = 8987,
    client_port: int = 9000,
    headed: bool = False,
    overlay: bool = True,
    reuse_services: bool = False,
    trim_gameplay: bool = False,
    max_duration: int | None = None,
) -> None:
    from playwright.sync_api import sync_playwright

    game_id = json.loads(record.read_text())["info"]["gameID"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"gameID {game_id} -> {out}")

    sidecar = record.with_suffix(".debug.json")
    if overlay and sidecar.exists():
        print(f"model overlay from {sidecar.name}")
    elif overlay:
        print(f"no {sidecar.name} - rendering without the model overlay")

    procs: list[subprocess.Popen] = []
    client_ctx = (
        contextlib.nullcontext(OPENFRONT)
        if reuse_services
        else client_worktree(record_engine_commit(record))
    )

    with client_ctx as client_dir:
        try:
            if not reuse_services or not port_open(api_port):
                procs.append(subprocess.Popen(
                    [sys.executable, "scripts/serve_replay.py",
                     "--records", str(record.parent), "--port", str(api_port)],
                    cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            if not reuse_services or not port_open(client_port):
                print("starting vite client (first boot takes ~15s)...")
                procs.append(subprocess.Popen(
                    ["npm", "run", "start:client", "--", "--host", "127.0.0.1"],
                    cwd=client_dir,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ))
            wait_http(f"http://localhost:{client_port}", 90)

            speed_label = {"0.5": "×0.5", "1": "×1", "2": "×2"}.get(speed)

            with sync_playwright() as pw, tempfile.TemporaryDirectory() as td:
                browser = pw.chromium.launch(
                    headless=not headed,
                    args=chromium_args(),
                )
                ctx = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=device_scale_factor,
                    record_video_dir=td,
                    record_video_size={"width": width, "height": height},
                )
                overlay_flag = "1" if overlay else "0"
                ctx.add_init_script(
                    f'localStorage.setItem("apiHost", "http://127.0.0.1:{api_port}");'
                    'localStorage.setItem("replayViewAs", "1");'
                    'localStorage.setItem("replayFitMap", "1");'
                    f'localStorage.setItem("rlDebugOverlay", "{overlay_flag}");'
                    'localStorage.setItem("settings.goToPlayer", "false");'
                    'localStorage.setItem("username", "AGENT");'
                )
                page = ctx.new_page()
                page_open_t0 = time.time()
                page.goto(f"http://localhost:{client_port}/game/{game_id}")

                toggle = page.locator('img[alt="replay"]')
                toggle.wait_for(state="visible", timeout=120_000)
                toggle.click()
                panel = page.locator("replay-panel button").first
                panel.wait_for(state="visible", timeout=10_000)
                if speed_label:
                    page.locator("replay-panel button", has_text=speed_label).click()
                else:
                    page.locator("replay-panel button").last.click()
                page.wait_for_timeout(800)
                gameplay_t0 = time.time()
                print("replay running...")

                t0 = time.time()
                gameplay_duration: float | None = None
                while time.time() - t0 < timeout:
                    if page.locator("win-modal div.fixed").count() > 0:
                        # Stop before the "You died" modal; don't linger on end cards.
                        gameplay_duration = max(0.0, time.time() - gameplay_t0 - 1.0)
                        print(
                            f"game over after {time.time() - t0:.0f}s "
                            f"(trimming to {gameplay_duration:.1f}s gameplay)"
                        )
                        break
                    page.wait_for_timeout(500)
                else:
                    print("timeout reached; saving what we have")

                page.close()
                video = page.video.path() if page.video else None
                ctx.close()
                browser.close()
                if not video or not Path(video).exists():
                    raise SystemExit("no video captured")

                raw = Path(video)
                trim_start = (gameplay_t0 - page_open_t0) if trim_gameplay else 0.0
                if trim_gameplay:
                    print(f"trimming {trim_start:.1f}s of load-in")
                trim_end = gameplay_duration if gameplay_duration is not None else max_duration
                trim_video(raw, out, trim_start, trim_end, crf=crf)

            print(f"wrote {out}")
        finally:
            for p in procs:
                p.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="GameRecord JSON from rl.watch --record")
    ap.add_argument("--out", default=None, help="output .webm (default: <record>.client.webm)")
    ap.add_argument("--speed", default="max", choices=["0.5", "1", "2", "max"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--device-scale-factor", type=float, default=2,
                    help="emulate retina DPR so WebGL renders at 2x backing resolution")
    ap.add_argument("--crf", type=int, default=18,
                    help="VP9 quality for ffmpeg trim pass (lower = sharper, default 18)")
    ap.add_argument("--timeout", type=int, default=1200, help="max seconds to wait for game end")
    ap.add_argument("--api-port", type=int, default=8987)
    ap.add_argument("--client-port", type=int, default=9000)
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                    help="model debug panel from <record>.debug.json (--no-overlay to disable)")
    ap.add_argument("--reuse-services", action="store_true",
                    help="use existing serve_replay + vite on api/client ports")
    ap.add_argument("--trim-gameplay", action="store_true",
                    help="ffmpeg-trim load-in before replay starts")
    ap.add_argument("--max-duration", type=int, default=0,
                    help="cap output length in seconds (0 = full replay)")
    args = ap.parse_args()

    record = Path(args.record).resolve()
    out = Path(args.out or record.with_suffix(".client.webm"))
    max_duration = args.max_duration if args.max_duration > 0 else None
    render_record(
        record,
        out,
        speed=args.speed,
        width=args.width,
        height=args.height,
        device_scale_factor=args.device_scale_factor,
        crf=args.crf,
        timeout=args.timeout,
        api_port=args.api_port,
        client_port=args.client_port,
        headed=args.headed,
        overlay=args.overlay,
        reuse_services=args.reuse_services,
        trim_gameplay=args.trim_gameplay,
        max_duration=max_duration,
    )


if __name__ == "__main__":
    main()
