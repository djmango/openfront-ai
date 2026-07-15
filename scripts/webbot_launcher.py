"""Launch a headless Chromium tab that plays a live OpenFront lobby as the
in-browser ONNX agent (src/client/webbot/ in the openfront submodule), in
place of the old server-side GPU/CPU process (rl.play).

The agent's observation featurization + AE/policy inference all run inside
the page via onnxruntime-web (WASM, single-threaded, off the main thread in
a Web Worker) - this process just drives the browser tab and stays alive for
the lifetime of the game so `ofshowcase hub` can track/terminate it exactly
like the old rl.play subprocess (poll/wait/terminate on this PID).

Exits as soon as src/client/webbot/main.ts sets window.__webbotDone (game
won/lost) so the hub's active-lobby poll frees up and the next /play visitor
gets a fresh lobby instead of piling into a finished game.

If --debug-port is given, serves the same {actions, log} JSON shape the old
rl.play --debug-port server did at /debug/<gameID>, so the client's MODEL
overlay (patches/client-replay-tooling.patch's RlDebugOverlay, proxied by
`ofshowcase hub`'s /play/debug/<id>) keeps working - just sourced from this
page's window.__webbotDebug instead of a separate Python inference process.

Usage:
  python scripts/webbot_launcher.py --host 127.0.0.1:9000 \
      --game <gameID> --worker-path w0 --debug-port 8989
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_GAME_SECONDS = 30 * 60  # safety net if the page never signals game-over


def chromium_args() -> list[str]:
    import os
    from pathlib import Path

    base = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--enable-gpu",
        "--ignore-gpu-blocklist",
    ]
    force_soft = os.environ.get("OF_FORCE_SWIFTSHADER", "").strip() in ("1", "true", "yes")
    force_gpu = os.environ.get("OF_FORCE_GPU", "").strip() in ("1", "true", "yes")
    has_gpu = force_gpu or any(
        Path(p).exists() for p in ("/dev/nvidia0", "/dev/dri/renderD128", "/dev/dri/card0")
    )
    if force_soft or not has_gpu:
        return base + [
            "--use-gl=angle",
            "--use-angle=swiftshader-webgl",
            "--enable-unsafe-swiftshader",
        ]
    return base + ["--use-gl=angle", "--use-angle=gl"]


# Local-dev Chromium console is flooded with ad/CORS/cosmetics noise that
# has nothing to do with the bot. Keep only webbot status + real failures.
_NOISE = (
    "CORS policy",
    "Access to fetch",
    "Access to XMLHttpRequest",
    "cloudflareinsights",
    "optable.co",
    "playwire",
    "PubAdsService",
    "Audigent",
    "googletag",
    "prebid",
    "Turnstile",
    "id5id",
    "Lit is in dev mode",
    "web-share",
    "cosmetics",
    "getNews",
    "profane_words",
    "ERR_CONNECTION_REFUSED",
    "ERR_CERT_DATE_INVALID",
    "ERR_CONNECTION_CLOSED",
    "Failed to load resource",
    "[vite]",
    "Refresh failed",
    "No JWT found",
    "Refreshing jwt",
    "homepage ads",
    "gutter ads",
    "Ramp.js",
    "Confiant",
    "CrazyGames",
    "destroyUnits",
    "Closing host lobby",
    "Error checking lobby",
    "Not running on CrazyGames",
    "shouldRefresh is false",
    # Nation/bot spawn failures on crowded maps (not AgentRL).
    "SpawnExecution: cannot spawn",
    "cannot spawn ",
)


def _console_interesting(text: str) -> bool:
    if any(n in text for n in _NOISE):
        return False
    # webbot/main.ts logs as `[webbot] ...`; pageerror handler is always kept.
    t = text.lower()
    return (
        text.startswith("[webbot]")
        or "[webbot]" in text
        or "pageerror" in t
        or "game ended" in t
        or "loading models" in t
        or "invalid string" in t
        or "[webbot] spawn" in t
        or "websocket closed" in t
        or "connection refused" in t
        or "host left" in t
        or "lobby full" in t
    )


class _DebugState:
    def __init__(self) -> None:
        self.body = b'{"actions":[],"log":[]}'
        self.lock = threading.Lock()

    def set(self, body: bytes) -> None:
        with self.lock:
            self.body = body

    def get(self) -> bytes:
        with self.lock:
            return self.body


def make_debug_server(game_id: str, state: _DebugState, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") != f"/debug/{game_id}":
                self.send_response(404)
                self.end_headers()
                return
            body = state.get()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *a: object) -> None:
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:9000")
    ap.add_argument("--game", required=True)
    ap.add_argument("--worker-path", default="")
    ap.add_argument("--name", default="AgentRL")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--debug-port", type=int, default=0)
    ap.add_argument("--headed", action="store_true",
                    help="open a visible Chromium window (default: headless)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every browser console line (default: filter ad/CORS noise)")
    args = ap.parse_args()

    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    if not __import__("re").fullmatch(r"[A-Za-z0-9]{8}", args.game):
        raise SystemExit(
            f"bad --game {args.game!r}: need an 8-char lobby ID "
            "(did a worker path like 'w1' get passed by mistake?)"
        )

    # Hit "/" with ?webbot= - NOT /wN/game/<id>. Transport resolves the
    # worker from gameID itself (ClientEnv.workerPath). --worker-path is
    # kept for logging only.
    _ = args.worker_path  # unused; Transport derives worker from gameID
    url = f"http://{args.host}/?webbot={args.game}&name={args.name}"
    if args.greedy:
        url += "&greedy=1"

    stop = False

    def _stop(*_a: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    debug_state = _DebugState()
    debug_srv: ThreadingHTTPServer | None = None
    if args.debug_port:
        debug_srv = make_debug_server(args.game, debug_state, args.debug_port)
        threading.Thread(target=debug_srv.serve_forever, daemon=True).start()
        print(f"[webbot] debug feed :{args.debug_port}/debug/{args.game}", flush=True)

    def on_console(m) -> None:
        text = m.text
        if args.verbose or _console_interesting(text):
            print(f"[webbot:{args.game}] {text}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, args=chromium_args())
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.add_init_script('localStorage.setItem("rlAllowSoftwareGL", "1");')
        # Block third-party ad/telemetry so local join isn't fighting CORS.
        page.route(
            "**/*",
            lambda route: (
                route.abort()
                if any(
                    h in route.request.url
                    for h in (
                        "optable.co",
                        "cloudflareinsights.com",
                        "googlesyndication",
                        "googletagservices",
                        "doubleclick.net",
                        "amazon-adsystem",
                        "playwire",
                        "audigent",
                        "id5-sync",
                        "prebid",
                    )
                )
                else route.continue_()
            ),
        )
        page.on("console", on_console)
        page.on("pageerror", lambda e: print(f"[webbot:{args.game}] pageerror: {e}", flush=True))
        page.on("crash", lambda: print(f"[webbot:{args.game}] page crashed", flush=True))

        print(f"[webbot] joining {url}", flush=True)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        t0 = time.time()
        last_status = ""
        try:
            while not stop:
                if page.is_closed():
                    print(f"[webbot:{args.game}] page closed, exiting", flush=True)
                    break
                try:
                    done = page.evaluate("window.__webbotDone ?? null")
                    if done is not None:
                        print(f"[webbot:{args.game}] game ended: {json.dumps(done)}", flush=True)
                        break
                    # Console events are flaky under Vite/HMR; poll the overlay.
                    status = page.evaluate(
                        "document.getElementById('webbot-overlay')?.querySelector('div')?.textContent ?? ''"
                    )
                    if status and status != last_status:
                        print(f"[webbot:{args.game}] {status}", flush=True)
                        last_status = status
                    if debug_srv is not None:
                        payload = page.evaluate("JSON.stringify(window.__webbotDebug ?? {actions:[],log:[]})")
                        debug_state.set(payload.encode())
                except PlaywrightError:
                    pass  # transient - page mid-navigation, try again next tick
                if time.time() - t0 > MAX_GAME_SECONDS:
                    print(f"[webbot:{args.game}] max runtime hit, exiting", flush=True)
                    break
                time.sleep(1.0)
        finally:
            if debug_srv is not None:
                debug_srv.shutdown()
            with contextlib.suppress(Exception):
                browser.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
