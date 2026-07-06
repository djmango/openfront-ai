"""Render a GameRecord as a video of the REAL OpenFront client — actual
game graphics: terrain art, units, factories, nukes, leaderboard, the lot.

Drives a headless Chromium (Playwright) against the local dev client,
replaying the record served by the archive-API shim. Client hooks (patch in
patches/client-replay-tooling.patch, pre-applied to the submodule):
  - replayViewAs: the viewer adopts the agent's identity — self-player
    styling, gold spawn ring, crown when first, "You Won!" modal
  - replayFitMap: camera starts centered on the whole map
  - window.__replayTick: current sim tick, drives the model debug overlay

If rl.watch wrote a debug sidecar (<record>.debug.json) next to the record,
the video gets a live model panel: chosen action, value estimate,
action-probability bars, and a recent-actions log, synced to the sim tick.

Usage:
  uv run python scripts/render_client_replay.py \
      --record replays/v3_stage0.json --out replays/v3_client.webm

Requires: `uv run playwright install chromium` (one-time). Starts
serve_replay.py itself; starts the vite client too unless one is already
on --client-port.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# In-page model panel fed by the watch.py sidecar; follows the replay via
# window.__replayTick (exposed by the submodule patch).
OVERLAY_JS = """
(payload) => {
  const data = JSON.parse(payload);
  const log = data.log, actions = data.actions;
  const el = document.createElement("div");
  el.style.cssText =
    "position:fixed;left:8px;top:160px;width:250px;z-index:10005;" +
    "background:rgba(12,14,22,.88);color:#d2d6e0;font:11px/1.4 ui-monospace,monospace;" +
    "padding:8px 10px;border-radius:8px;pointer-events:none";
  document.body.appendChild(el);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  let idx = 0;
  const recent = [];
  setInterval(() => {
    const t = window.__replayTick;
    if (t === undefined || !log.length) return;
    while (idx < log.length && log[idx].tick <= t) {
      const e = log[idx++];
      if (e.action !== "noop") {
        recent.push("t" + e.tick + " " + e.desc);
        if (recent.length > 8) recent.shift();
      }
    }
    const e = idx > 0 ? log[idx - 1] : null;
    let html = '<b style="color:#fff">MODEL</b> <span style="color:#787e8c">tick ' + t + "</span><br>";
    if (!e) { el.innerHTML = html + "spawn phase"; return; }
    html += "tiles " + e.tiles.toLocaleString() + "  troops " + e.troops.toLocaleString();
    if (e.value !== undefined) html += "  v " + e.value.toFixed(2);
    html += '<br><span style="color:#5adc5a">' + esc(e.desc) + "</span>";
    if (e.probs) {
      for (let i = 0; i < actions.length; i++) {
        const hi = actions[i] === e.action;
        const col = hi ? "#5adc5a" : e.probs[i] > 0.01 ? "#d2d6e0" : "#787e8c";
        html +=
          '<div style="display:flex;align-items:center;margin:1px 0">' +
          '<span style="width:90px;color:' + col + '">' + esc(actions[i].slice(0, 13)) + "</span>" +
          '<span style="flex:1;height:7px;background:#282c3a;border-radius:2px">' +
          '<span style="display:block;height:7px;border-radius:2px;width:' +
          Math.round(e.probs[i] * 100) + "%;background:" + (hi ? "#5adc5a" : "#466ebe") +
          '"></span></span></div>';
      }
    }
    if (recent.length) {
      html += '<div style="color:#787e8c;margin-top:4px">recent:<br>' +
        recent.map(esc).join("<br>") + "</div>";
    }
    el.innerHTML = html;
  }, 100);
}
"""


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


def ensure_patch() -> None:
    """The client hooks live uncommitted in the submodule working tree;
    re-apply the patch if a submodule update/reset wiped them."""
    markers = [
        ("openfront/src/client/LocalServer.ts", "replayViewAs"),
        ("openfront/src/client/TransformHandler.ts", "replayFitMap"),
        ("openfront/src/client/ClientGameRunner.ts", "__replayTick"),
    ]
    present = [m in (REPO / f).read_text() for f, m in markers]
    if all(present):
        return
    if any(present):
        raise SystemExit(
            "openfront submodule has a partial replay-tooling patch; "
            "run: git -C openfront checkout -- src/client && "
            "git -C openfront apply ../patches/client-replay-tooling.patch"
        )
    subprocess.run(
        ["git", "apply", str(REPO / "patches/client-replay-tooling.patch")],
        cwd=REPO / "openfront", check=True,
    )
    print("re-applied patches/client-replay-tooling.patch")


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
    overlay_payload = None
    if args.overlay and sidecar.exists():
        overlay_payload = sidecar.read_text()
        print(f"model overlay from {sidecar.name}")
    elif args.overlay:
        print(f"no {sidecar.name} — rendering without the model overlay "
              "(re-run rl.watch --record to get one)")

    ensure_patch()

    procs: list[subprocess.Popen] = []
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
                cwd=REPO / "openfront",
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
            ctx.add_init_script(
                f'localStorage.setItem("apiHost", "http://localhost:{args.api_port}");'
                'localStorage.setItem("replayViewAs", "1");'
                'localStorage.setItem("replayFitMap", "1");'
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
            if overlay_payload is not None:
                page.evaluate(OVERLAY_JS, overlay_payload)
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
