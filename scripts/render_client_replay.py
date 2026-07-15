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
    action-probability bars, recent-actions log) synced to the sim tick,
    plus a red ✕ on the map when the decision includes a tile target;
    it fetches <apiHost>/debug/<gameID>, which the shim serves from the
    rl.watch debug sidecar (<record>.debug.json)

The same overlay works in a normal browser (manual `ofshowcase archive`
workflow) and in live games against webbot / rl.play --debug-port
(localStorage rlDebugHost).

Usage:
  uv run python scripts/render_client_replay.py \
      --record replays/v3_stage0.json --out replays/v3_client.webm

Requires: `uv run playwright install chromium` (one-time) and a built
`ofshowcase` (`cargo build --release -p ofhub`). Starts `ofshowcase
archive` itself; starts the vite client too unless one is already on
--client-port.
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


def ofshowcase_bin() -> Path:
    env = os.environ.get("OFSHOWCASE")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for candidate in (
        REPO / "rust/target/release/ofshowcase",
        REPO / "rust/target/debug/ofshowcase",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    which = shutil.which("ofshowcase")
    if which:
        return Path(which)
    raise FileNotFoundError(
        "ofshowcase not found; build with: cargo build --release -p ofhub "
        "(or set OFSHOWCASE=/path/to/ofshowcase)"
    )


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


def linux_has_gpu() -> bool:
    """True when Chromium can likely use a real GPU (NVIDIA / DRM) instead of SoftGL."""
    if os.environ.get("OF_FORCE_SWIFTSHADER", "").strip() in ("1", "true", "yes"):
        return False
    if os.environ.get("OF_FORCE_GPU", "").strip() in ("1", "true", "yes"):
        return True
    # NVIDIA CDI / classic device nodes commonly present in GPU containers.
    for path in ("/dev/nvidia0", "/dev/dri/renderD128", "/dev/dri/card0"):
        if Path(path).exists():
            return True
    return False


def chromium_args(*, soft_gl: bool | None = None) -> list[str]:
    """Chromium flags for headless OpenFront WebGL.

    Prefer a real GPU on Linux (homelab/showcase with NVIDIA). SoftGL/SwiftShader
    is known to crawl at ~1fps and produces near-static hero clips - keep it as
    fallback only. Always allow SoftGL in the OpenFront client (rlAllowSoftwareGL)
    because Chromium often silently falls back to SwiftShader even when we ask
    for GL/ANGLE.
    """
    base = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-gpu-blocklist", "--enable-gpu"]
    if platform.system() == "Darwin":
        return base + ["--use-angle=metal"]
    use_soft = soft_gl if soft_gl is not None else not linux_has_gpu()
    if use_soft:
        return base + [
            "--use-gl=angle",
            "--use-angle=swiftshader-webgl",
            "--enable-unsafe-swiftshader",
        ]
    # NVIDIA headless: ANGLE+EGL tends to stick to the discrete GPU better than
    # --use-angle=gl (which often silently falls back to SwiftShader).
    return base + [
        "--use-gl=angle",
        "--use-angle=gl-egl",
        "--enable-unsafe-swiftshader",  # last-resort if EGL fails at runtime
    ]


def soft_gl_defaults(*, width: int, height: int, device_scale_factor: float) -> tuple[int, int, float]:
    """Cheaper SoftGL capture so wall-clock clips still show gameplay motion."""
    return (
        min(width, 1280),
        min(height, 720),
        min(device_scale_factor, 1.0),
    )


def trim_video(
    src: Path,
    dst: Path,
    start_sec: float,
    max_duration: float | None,
    *,
    crf: int = 18,
    speedup: float = 1.0,
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
    vf_parts: list[str] = []
    if speedup > 1.01:
        # SoftGL records wall-clock at ~1fps; pack more sim progress into the clip.
        vf_parts.append(f"setpts=PTS/{speedup:.4f}")
    # Playwright records at 25fps. Normalize published clips to 30fps so
    # browser playback is smooth and consistent after timestamp compression.
    vf_parts.append("fps=30")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
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


def load_episode_meta(record: Path) -> dict:
    """Outcome + end tick from oftrain --watch debug sidecar (fallback: record info)."""
    sidecar = record.with_suffix(".debug.json")
    meta = {"outcome": "death", "end_tick": None}
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        meta["outcome"] = data.get("outcome", "death")
        meta["end_tick"] = data.get("end_tick")
    if meta["end_tick"] is None:
        info = json.loads(record.read_text()).get("info", {})
        meta["end_tick"] = info.get("num_turns")
    return meta


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
    death_trim_ticks: int = 55,
    win_hold_sec: float = 2.5,
) -> None:
    from playwright.sync_api import sync_playwright

    game_id = json.loads(record.read_text())["info"]["gameID"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"gameID {game_id} -> {out}")

    sidecar = record.with_suffix(".debug.json")
    episode = load_episode_meta(record)
    print(f"episode outcome: {episode['outcome']} (end tick {episode['end_tick']})")
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

    use_soft_gl = platform.system() != "Darwin" and not linux_has_gpu()
    chrome_args = chromium_args(soft_gl=use_soft_gl)
    render_width, render_height, render_dpr = width, height, device_scale_factor
    if use_soft_gl:
        render_width, render_height, render_dpr = soft_gl_defaults(
            width=width, height=height, device_scale_factor=device_scale_factor
        )
        print(
            f"SoftGL fallback: {render_width}x{render_height} dpr={render_dpr} "
            "(mount /dev/nvidia* or set OF_FORCE_GPU=1 for real WebGL)"
        )
    else:
        print(f"Chromium WebGL: GPU ({' '.join(chrome_args[-3:])})")

    with client_ctx as client_dir:
        try:
            if not reuse_services or not port_open(api_port):
                procs.append(subprocess.Popen(
                    [
                        str(ofshowcase_bin()),
                        "archive",
                        "--records",
                        str(record.parent),
                        "--port",
                        str(api_port),
                        "--bind",
                        "127.0.0.1",
                    ],
                    cwd=REPO,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
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
                    args=chrome_args,
                )
                ctx = browser.new_context(
                    viewport={"width": render_width, "height": render_height},
                    device_scale_factor=render_dpr,
                    record_video_dir=td,
                    record_video_size={"width": render_width, "height": render_height},
                )
                overlay_flag = "1" if overlay else "0"
                # Always allow SoftGL: Chromium often falls back to SwiftShader
                # even when we request GL/EGL, and OpenFront otherwise refuses
                # to boot (no replay button → 120s timeout).
                ctx.add_init_script(
                    f'localStorage.setItem("apiHost", "http://127.0.0.1:{api_port}");'
                    'localStorage.setItem("replayViewAs", "1");'
                    'localStorage.setItem("replayFitMap", "1");'
                    f'localStorage.setItem("rlDebugOverlay", "{overlay_flag}");'
                    'localStorage.setItem("rlAllowSoftwareGL", "1");'
                    # Rendering every deterministic update is the SoftGL
                    # bottleneck. The patched client consumes every update but
                    # draws one in ten while recording.
                    'localStorage.setItem("replayRenderEvery", "10");'
                    'localStorage.setItem("settings.goToPlayer", "false");'
                    'localStorage.setItem("username", "AGENT");'
                )
                page = ctx.new_page()
                page_open_t0 = time.time()
                page.goto(
                    f"http://localhost:{client_port}/game/{game_id}",
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )

                toggle = page.locator('img[alt="replay"]')
                toggle.wait_for(state="visible", timeout=120_000)
                # DOM click: Playwright's real click hangs under SwiftShader when
                # overlays/hit-testing stall ("performing click action" forever).
                toggle.evaluate("el => el.click()")
                panel = page.locator("replay-panel button").first
                panel.wait_for(state="visible", timeout=30_000)
                if speed_label:
                    page.locator("replay-panel button", has_text=speed_label).evaluate(
                        "el => el.click()"
                    )
                else:
                    page.locator("replay-panel button").last.evaluate(
                        "el => el.click()"
                    )
                page.wait_for_timeout(800)
                gameplay_t0 = time.time()
                start_tick = page.evaluate(
                    "() => (typeof window.__replayTick === 'number' "
                    "? window.__replayTick : 0)"
                )
                last_tick = start_tick
                target_tick_rate = float(
                    os.environ.get("CLIP_GAME_TICKS_PER_SEC", "20")
                )
                target_ticks = (
                    int(target_tick_rate * max_duration)
                    if max_duration is not None
                    else None
                )
                print("replay running...")

                end_tick = episode.get("end_tick")
                outcome = episode.get("outcome", "death")
                stop_tick = (
                    int(end_tick) - death_trim_ticks
                    if end_tick and outcome == "death"
                    else None
                )
                t0 = time.time()
                gameplay_duration: float | None = None
                win_hold_t0: float | None = None
                # SoftGL advances ~1 tick/sec; give more wall-clock so Max-speed
                # still covers meaningful gameplay before --max-duration.
                soft_budget = (
                    float(max_duration) * 8.0
                    if use_soft_gl and max_duration is not None
                    else None
                )
                while time.time() - t0 < timeout:
                    # Wall-clock cap (SoftGL gets a longer budget; tick hooks lag).
                    wall_cap = soft_budget if soft_budget is not None else (
                        float(max_duration) if max_duration is not None else None
                    )
                    if wall_cap is not None and time.time() - gameplay_t0 >= wall_cap:
                        gameplay_duration = time.time() - gameplay_t0
                        print(
                            f"max-duration budget {wall_cap:.0f}s reached "
                            f"(tick={page.evaluate('() => window.__replayTick ?? null')})"
                        )
                        break
                    tick = page.evaluate(
                        "() => (typeof window.__replayTick === 'number' "
                        "? window.__replayTick : null)"
                    )
                    if tick is not None:
                        last_tick = tick
                    if (
                        target_ticks is not None
                        and tick is not None
                        and tick - start_tick >= target_ticks
                    ):
                        gameplay_duration = time.time() - gameplay_t0
                        print(
                            f"clip target reached at tick {tick} "
                            f"({target_ticks} gameplay ticks)"
                        )
                        break
                    if (
                        stop_tick is not None
                        and tick is not None
                        and tick >= stop_tick
                    ):
                        gameplay_duration = max(0.0, time.time() - gameplay_t0 - 0.2)
                        print(
                            f"stopping before death at tick {tick} "
                            f"(trim to {gameplay_duration:.1f}s gameplay)"
                        )
                        break

                    modal = page.locator("win-modal div.fixed")
                    if modal.count() > 0:
                        if outcome == "win":
                            if win_hold_t0 is None:
                                win_hold_t0 = time.time()
                                print("win modal - holding for celebration")
                            elif time.time() - win_hold_t0 >= win_hold_sec:
                                gameplay_duration = time.time() - gameplay_t0
                                break
                        else:
                            gameplay_duration = max(
                                0.0, time.time() - gameplay_t0 - 1.5
                            )
                            print(
                                f"death modal detected late "
                                f"(trim to {gameplay_duration:.1f}s gameplay)"
                            )
                            break

                    if (
                        outcome == "win"
                        and end_tick is not None
                        and tick is not None
                        and tick >= int(end_tick)
                        and win_hold_t0 is None
                    ):
                        # Win tick reached; wait for modal then celebrate.
                        for _ in range(30):
                            if modal.count() > 0:
                                win_hold_t0 = time.time()
                                break
                            page.wait_for_timeout(100)
                        if win_hold_t0 is None:
                            gameplay_duration = time.time() - gameplay_t0
                            break

                    page.wait_for_timeout(100)
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
                recorded = gameplay_duration if gameplay_duration is not None else (
                    time.time() - gameplay_t0
                )
                ticks_rendered = max(0, int(last_tick or 0) - int(start_tick or 0))
                tick_rate = ticks_rendered / recorded if recorded > 0 else 0.0
                print(
                    f"captured {ticks_rendered} ticks in {recorded:.1f}s "
                    f"({tick_rate:.1f} ticks/sec before video speedup)"
                )
                speedup = 1.0
                if use_soft_gl and max_duration is not None and ticks_rendered > 0:
                    # Wall-clock compression alone preserved the host's slow
                    # simulation rate. Derive output duration from actual game
                    # ticks so every published clip is at least the requested
                    # gameplay speed, even when SoftGL never reaches its target.
                    trim_end = min(
                        float(max_duration),
                        ticks_rendered / target_tick_rate,
                    )
                    speedup = recorded / trim_end
                    print(
                        f"SoftGL speedup {speedup:.1f}x "
                        f"({recorded:.0f}s wall, {ticks_rendered} ticks "
                        f"-> {trim_end:.1f}s clip at {target_tick_rate:g} ticks/sec)"
                    )
                elif max_duration is not None:
                    trim_end = min(recorded, float(max_duration))
                else:
                    trim_end = gameplay_duration
                trim_video(
                    raw,
                    out,
                    trim_start,
                    trim_end,
                    crf=crf,
                    speedup=speedup,
                )

            print(f"wrote {out}")
        finally:
            for p in procs:
                p.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="GameRecord JSON from oftrain --watch")
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
                    help="use existing ofshowcase archive + vite on api/client ports")
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
