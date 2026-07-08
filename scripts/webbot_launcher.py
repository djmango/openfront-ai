"""Launch a headless Chromium tab that plays a live OpenFront lobby as the
in-browser ONNX agent (src/client/webbot/ in the openfront submodule), in
place of the old server-side GPU/CPU process (rl.play).

The agent's observation featurization + AE/policy inference all run inside
the page via onnxruntime-web (WASM, single-threaded, off the main thread in
a Web Worker) - this process just drives the browser tab and stays alive for
the lifetime of the game so showcase_hub.py can track/terminate it exactly
like it did the old rl.play subprocess (poll/wait/terminate on this PID).

Usage:
  python scripts/webbot_launcher.py --host 127.0.0.1:9000 \
      --game <gameID> --worker-path w0
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time


def chromium_args() -> list[str]:
    return ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:9000")
    ap.add_argument("--game", required=True)
    ap.add_argument("--worker-path", default="")
    ap.add_argument("--name", default="AgentRL")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    prefix = f"/{args.worker_path}" if args.worker_path else ""
    url = f"http://{args.host}{prefix}/game/{args.game}?webbot={args.game}&name={args.name}"
    if args.greedy:
        url += "&greedy=1"

    stop = False

    def _stop(*_a: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=chromium_args())
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda m: print(f"[webbot:{args.game}] {m.text}", flush=True))
        page.on("pageerror", lambda e: print(f"[webbot:{args.game}] pageerror: {e}", flush=True))
        page.on("crash", lambda: print(f"[webbot:{args.game}] page crashed", flush=True))

        print(f"[webbot_launcher] navigating {url}", flush=True)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        try:
            while not stop:
                if page.is_closed():
                    print(f"[webbot:{args.game}] page closed, exiting", flush=True)
                    break
                time.sleep(1.0)
        finally:
            with contextlib.suppress(Exception):
                browser.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
