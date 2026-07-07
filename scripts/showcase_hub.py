"""Homelab showcase hub: view-only replay by default, on-demand 1v1 play.

  GET /         -> landing (watch link + play button)
  GET /watch    -> redirect to latest checkpoint replay
  GET /replay   -> alias for /watch
  GET /play     -> create 1v1+bots lobby, launch agent, join as human
  GET /play/debug/<id> -> MODEL overlay feed for the active live game
  GET /status   -> JSON hub status
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rl.showcase_util import ensure_ae, ensure_policy, preview_webm_for_state, utc_now, write_json

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
REPLAY_STATE = DATA_DIR / "state.json"
HUB_STATE = DATA_DIR / "hub_state.json"
AE_PATH = Path(os.environ.get("AE_CKPT", "runs/ae_v31_d8c32/ae_v3.pt"))
RUN_NAME = os.environ.get("RUN_NAME", "ppo_v4")
CLIENT_HOST = os.environ.get("CLIENT_HOST", "127.0.0.1:9000")
ADMIN_KEY = os.environ.get(
    "ADMIN_BOT_API_KEY",
    "WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION",
)
ADMIN_HEADER = "x-admin-bot-key"
PLAY_MAP = os.environ.get("PLAY_MAP", "Onion")
PLAY_BOTS = int(os.environ.get("PLAY_BOTS", "10"))
PLAY_NATIONS = int(os.environ.get("PLAY_NATIONS", "1"))
PLAY_START_DELAY = int(os.environ.get("PLAY_START_DELAY", "90"))
DEBUG_PORT = int(os.environ.get("PLAY_DEBUG_PORT", "8989"))

_active_proc: subprocess.Popen | None = None
_active_game: str | None = None
_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[showcase_hub] {msg}", flush=True)


def load_replay_state() -> dict:
    if REPLAY_STATE.exists():
        try:
            return json.loads(REPLAY_STATE.read_text())
        except Exception:
            pass
    return {}


def load_hub_state() -> dict:
    if HUB_STATE.exists():
        try:
            return json.loads(HUB_STATE.read_text())
        except Exception:
            pass
    return {}


def http_json(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", ADMIN_HEADER: ADMIN_KEY},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {url} -> {exc.code}: {exc.read().decode()}") from exc


def play_config() -> dict:
    return {
        "gameMap": PLAY_MAP,
        "gameType": "Private",
        "bots": PLAY_BOTS,
        "difficulty": "Easy",
        "nations": PLAY_NATIONS,
        "startDelay": PLAY_START_DELAY,
    }


def create_play_lobby() -> dict:
    base = f"http://{CLIENT_HOST}"
    info = http_json("POST", f"{base}/api/adminbot/create_game", play_config())
    log(f"play lobby {info['gameID']} ({PLAY_MAP}, {PLAY_NATIONS} nations, {PLAY_BOTS} bots)")
    return info


def launch_agent(game_id: str, policy: Path, ae: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "rl.play",
        "--policy",
        str(policy),
        "--ckpt",
        str(ae),
        "--game",
        game_id,
        "--host",
        CLIENT_HOST,
        "--debug-port",
        str(DEBUG_PORT),
        "--debug-bind",
        "0.0.0.0",
    ]
    return subprocess.Popen(cmd, cwd=REPO)


def proxy_debug(game_id: str) -> bytes | None:
    url = f"http://127.0.0.1:{DEBUG_PORT}/debug/{game_id}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.read()
    except Exception:
        return None


LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenFront RL Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; }
    @keyframes drift {
      0% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(-2%, 1%) scale(1.03); }
      100% { transform: translate(0, 0) scale(1); }
    }
    @keyframes glow {
      0%, 100% { box-shadow: 0 0 24px rgba(90,220,90,.25), 0 0 60px rgba(70,110,190,.15); }
      50% { box-shadow: 0 0 36px rgba(90,220,90,.45), 0 0 80px rgba(70,110,190,.28); }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(18px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scan {
      0% { transform: translateY(-100%); }
      100% { transform: translateY(100%); }
    }
    @keyframes pulse {
      0%, 100% { opacity: .35; }
      50% { opacity: .7; }
    }
    body {
      min-height: 100vh; color: #d2d6e0;
      font: 16px/1.5 system-ui, sans-serif;
      background: #07080f;
      overflow-x: hidden;
    }
    .bg {
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background:
        radial-gradient(ellipse 80% 60% at 20% 20%, rgba(70,110,190,.18), transparent 55%),
        radial-gradient(ellipse 70% 50% at 80% 70%, rgba(42,90,42,.2), transparent 50%),
        #07080f;
    }
    .grid {
      position: fixed; inset: -50%;
      background-image:
        linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
      background-size: 48px 48px;
      animation: drift 18s ease-in-out infinite;
      mask-image: radial-gradient(ellipse 70% 60% at 50% 45%, #000 20%, transparent 75%);
    }
    .wrap {
      position: relative; z-index: 1;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 1fr;
      gap: 2rem;
      align-items: center;
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.25rem 2.5rem;
    }
    @media (min-width: 900px) {
      .wrap { grid-template-columns: 1fr 1.15fr; padding: 3rem 2rem; }
    }
    .copy { animation: rise .7s ease both; }
    .eyebrow {
      display: inline-block;
      font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
      color: #5adc5a; margin-bottom: .75rem;
      border: 1px solid rgba(90,220,90,.35);
      padding: .35rem .6rem; border-radius: 999px;
      background: rgba(42,90,42,.15);
    }
    h1 {
      font-size: clamp(1.8rem, 4vw, 2.6rem);
      line-height: 1.1; color: #fff; margin-bottom: .75rem;
      letter-spacing: -.02em;
    }
    .lead { color: #8b92a3; margin-bottom: 1.5rem; max-width: 34rem; }
    .btns { display: flex; flex-wrap: wrap; gap: .75rem; }
    a.btn {
      display: inline-block; padding: .85rem 1.25rem; border-radius: 10px;
      text-decoration: none; font-weight: 600; transition: transform .15s ease, filter .15s ease;
    }
    a.btn:hover { transform: translateY(-2px); filter: brightness(1.08); }
    .watch { background: linear-gradient(135deg, #466ebe, #5a84d6); color: #fff; }
    .play {
      background: rgba(42,90,42,.55); color: #5adc5a;
      border: 1px solid rgba(90,220,90,.55);
    }
    .meta { margin-top: 1.25rem; font-size: .8rem; color: #505868; }
    .stage {
      animation: rise .9s .15s ease both;
      position: relative;
    }
    .frame {
      position: relative; border-radius: 16px; overflow: hidden;
      border: 1px solid rgba(255,255,255,.08);
      background: #0c0e16;
      animation: glow 4s ease-in-out infinite;
      aspect-ratio: 16 / 10;
    }
    .frame::after {
      content: ""; position: absolute; inset: 0; pointer-events: none;
      background: linear-gradient(180deg, transparent 70%, rgba(7,8,15,.55));
    }
    .scanline {
      position: absolute; inset: 0; pointer-events: none; opacity: .08;
      background: linear-gradient(180deg, transparent, rgba(90,220,90,.9), transparent);
      height: 30%; animation: scan 5s linear infinite;
    }
    video {
      width: 100%; height: 100%; object-fit: cover; display: block;
      background: #0c0e16;
    }
    .badge {
      position: absolute; top: 12px; left: 12px; z-index: 2;
      font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
      color: #9be89b; background: rgba(0,0,0,.55);
      border: 1px solid rgba(90,220,90,.35);
      padding: .3rem .5rem; border-radius: 6px;
      animation: pulse 2.4s ease-in-out infinite;
    }
    .placeholder {
      width: 100%; height: 100%; display: grid; place-items: center;
      color: #5f6675; font-size: .9rem;
      background: linear-gradient(135deg, #10131f, #0c1018);
    }
    .placeholder span {
      width: 12px; height: 12px; border-radius: 50%;
      background: #5adc5a; display: inline-block; margin-right: .5rem;
      animation: pulse 1.2s ease-in-out infinite;
    }
  </style>
</head>
<body>
  <div class="bg"></div>
  <div class="grid"></div>
  <div class="wrap">
    <section class="copy">
      <div class="eyebrow">Live RL showcase</div>
      <h1>OpenFront Agent</h1>
      <p class="lead">Watch the latest checkpoint play in real time with the model
         overlay, or jump into a 1v1 on a bot-filled map.</p>
      <div class="btns">
        <a class="btn watch" href="/watch">Watch full replay</a>
        <a class="btn play" href="/play">Play vs Agent</a>
      </div>
      <p class="meta">policy: %%RUN_NAME%%</p>
    </section>
    <section class="stage">
      <div class="frame">
        <div class="badge">Agent playing</div>
        <div class="scanline"></div>
        %%PREVIEW%%
      </div>
    </section>
  </div>
</body>
</html>
"""


def preview_markup(replay: dict) -> str:
    if preview_webm_for_state(replay):
        return (
            '<video autoplay muted loop playsinline preload="auto" '
            'src="/archive/preview.webm"></video>'
        )
    return (
        '<div class="placeholder">'
        '<div><span></span>Generating showcase preview…</div>'
        "</div>"
    )


def render_landing(replay: dict) -> str:
    return (
        LANDING_HTML.replace("%%RUN_NAME%%", str(replay.get("run_name", RUN_NAME)))
        .replace("%%PREVIEW%%", preview_markup(replay))
    )


class HubHandler(BaseHTTPRequestHandler):
    policy_path: Path | None = None
    ae_path: Path | None = None

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path == "/":
            replay = load_replay_state()
            html = render_landing(replay)
            self._send(200, html.encode(), "text/html; charset=utf-8")
            return

        if path in ("/watch", "/replay"):
            gid = load_replay_state().get("game_id")
            if not gid:
                self._send(503, b'{"status":"warming","message":"replay generating"}')
                return
            self.send_response(302)
            self.send_header("Location", f"/game/{gid}")
            self.end_headers()
            return

        if path == "/status":
            payload = {
                "replay": load_replay_state(),
                "hub": load_hub_state(),
                "play_config": play_config(),
            }
            self._send(200, json.dumps(payload).encode())
            return

        if path.startswith("/play/debug/"):
            game_id = path.split("/play/debug/", 1)[1].strip("/")
            body = proxy_debug(game_id)
            if body is None:
                self._send(404, b'{"error":"no live debug feed"}')
            else:
                self._send(200, body)
            return

        if path == "/play":
            global _active_proc, _active_game
            with _lock:
                if _active_game and _active_proc and _active_proc.poll() is None:
                    log(f"reusing active lobby {_active_game}")
                    self.send_response(302)
                    self.send_header("Location", f"/game/{_active_game}")
                    self.end_headers()
                    return

                if self.policy_path is None or self.ae_path is None:
                    self._send(503, b'{"error":"policy not ready"}')
                    return

                try:
                    info = create_play_lobby()
                except Exception as exc:
                    log(f"lobby create failed: {exc}")
                    self._send(500, json.dumps({"error": str(exc)}).encode())
                    return

                game_id = info["gameID"]
                _active_game = game_id
                _active_proc = launch_agent(game_id, self.policy_path, self.ae_path)

                write_json(
                    HUB_STATE,
                    {
                        "game_id": game_id,
                        "status": "lobby",
                        "config": play_config(),
                        "run_name": RUN_NAME,
                        "started_at": utc_now(),
                    },
                )
                log(f"agent joining {game_id}, you have {PLAY_START_DELAY}s after Start")

            self.send_response(302)
            self.send_header("Location", f"/game/{game_id}")
            self.end_headers()
            return

        self._send(404, b'{"error":"unknown route"}')

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    port = int(os.environ.get("HUB_PORT", "8988"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log("loading policy + encoder")
    ae = ensure_ae(AE_PATH)
    policy = ensure_policy(RUN_NAME)
    HubHandler.policy_path = policy
    HubHandler.ae_path = ae

    srv = ThreadingHTTPServer(("0.0.0.0", port), HubHandler)
    log(f"hub on :{port} (watch=/watch, play=/play)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
