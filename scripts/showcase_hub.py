"""Homelab showcase hub: hourly replay spectate, on-demand 1v1 play.

  GET /         -> landing (featured map clip; watch link)
  GET /watch    -> archived replay for the current hour's map
  GET /replay   -> alias for /watch
  GET /play     -> create 1v1+bots lobby, launch agent, join as human
  GET /play/debug/<id> -> MODEL overlay feed for the active play lobby
  GET /status   -> JSON hub status
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rl.showcase_util import (
    ensure_ae,
    ensure_policy,
    featured_showcase_entry,
    utc_now,
    write_json,
)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
REPLAY_STATE = DATA_DIR / "state.json"
HUB_STATE = DATA_DIR / "hub_state.json"
AE_PATH = Path(os.environ.get("AE_CKPT", "runs/ae_v31_d8c32/ae_v3.pt"))
RUN_NAME = os.environ.get("RUN_NAME", "ppo_v6")
CLIENT_HOST = os.environ.get("CLIENT_HOST", "127.0.0.1:9000")
ADMIN_KEY = os.environ.get(
    "ADMIN_BOT_API_KEY",
    "WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION",
)
ADMIN_HEADER = "x-admin-bot-key"
PLAY_MAP = os.environ.get("PLAY_MAP", "Onion")
PLAY_BOTS = int(os.environ.get("PLAY_BOTS", "10"))
PLAY_NATIONS = int(os.environ.get("PLAY_NATIONS", "1"))
PLAY_START_DELAY = int(os.environ.get("PLAY_START_DELAY", "30"))
DEBUG_PORT = int(os.environ.get("PLAY_DEBUG_PORT", "8989"))
LIVE_DEBUG_PORT = int(os.environ.get("LIVE_DEBUG_PORT", "8990"))
LIVE_SHOWCASE = os.environ.get("LIVE_SHOWCASE", "0") != "0"
WORKER_BASE_PORT = int(os.environ.get("WORKER_BASE_PORT", "3001"))

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
    log(
        f"play lobby {info['gameID']} ({PLAY_MAP}, {PLAY_NATIONS} nations, "
        f"{PLAY_BOTS} bots, worker {info.get('workerIndex', '?')})"
    )
    return info


def worker_path_for(game_id: str) -> str:
    hub = load_hub_state()
    if hub.get("game_id") == game_id and hub.get("worker_path"):
        return str(hub["worker_path"])
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    h = 0
    for ch in game_id:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
    if h & 0x80000000:
        h = -((~h + 1) & 0xFFFFFFFF)
    return f"w{h % num_workers}"


def play_redirect(game_id: str, worker_path: str | None = None) -> str:
    wp = worker_path or worker_path_for(game_id)
    return f"/{wp}/game/{game_id}"


def wait_for_webbot_join(
    game_id: str, worker_path: str, *, timeout_s: float = 40.0
) -> bool:
    """Poll the public lobby-info route until the webbot's client shows up.

    launch_webbot_agent() only *starts* a subprocess - cold Chromium boot +
    page load + ONNX init can take several seconds. If we arm the game-start
    countdown immediately (the old behavior), it can fire before the webbot's
    "join" message ever reaches the server, so gameStartInfo.players ends up
    empty and the bot spends the whole match retrying a spawn that can never
    land (server logs "player with clientID ... not found" forever). Block
    here instead so the countdown only starts once the bot is actually in
    the lobby.
    """
    url = f"http://{CLIENT_HOST}/{worker_path}/api/game/{game_id}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                info = json.loads(resp.read().decode())
            if len(info.get("clients", [])) > 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_play_lobby(game_id: str, worker_index: int) -> dict:
    """Arm the lobby countdown via the admin bot on the owning game worker."""
    base = f"http://127.0.0.1:{WORKER_BASE_PORT + worker_index}"
    info = http_json(
        "POST",
        f"{base}/api/adminbot/game/{game_id}/intent",
        {"type": "toggle_game_start_timer"},
    )
    log(f"countdown armed for {game_id} (delay {PLAY_START_DELAY}s)")
    return info


def launch_agent(
    game_id: str,
    policy: Path,
    ae: Path,
    *,
    debug_port: int = DEBUG_PORT,
) -> subprocess.Popen:
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
        str(debug_port),
        "--debug-bind",
        "0.0.0.0",
    ]
    return subprocess.Popen(cmd, cwd=REPO)


def launch_webbot_agent(
    game_id: str,
    worker_path: str,
    *,
    debug_port: int = DEBUG_PORT,
) -> subprocess.Popen:
    """Headless-browser agent: inference runs client-side (onnxruntime-web),
    no server GPU/CPU model process needed. See scripts/webbot_launcher.py.
    """
    cmd = [
        sys.executable,
        "scripts/webbot_launcher.py",
        "--host",
        CLIENT_HOST,
        "--game",
        game_id,
        "--worker-path",
        worker_path,
        "--debug-port",
        str(debug_port),
    ]
    return subprocess.Popen(cmd, cwd=REPO)


def proxy_debug(game_id: str) -> bytes | None:
    hub = load_hub_state()
    ports: list[int] = []
    if hub.get("live_game_id") == game_id:
        ports.append(LIVE_DEBUG_PORT)
    ports.append(DEBUG_PORT)
    if LIVE_DEBUG_PORT not in ports:
        ports.append(LIVE_DEBUG_PORT)
    for port in ports:
        url = f"http://127.0.0.1:{port}/debug/{game_id}"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.read()
        except Exception:
            continue
    return None


def live_game_active(hub: dict) -> bool:
    gid = hub.get("live_game_id")
    if not gid:
        return False
    if hub.get("live_status") not in ("lobby", "countdown", "playing"):
        return False
    pid = hub.get("live_pid")
    if pid is not None:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return False
    return True


def watch_target() -> tuple[str, str, dict | None]:
    """Return (redirect path, mode, featured entry). Watch is replay-only."""
    replay = load_replay_state()
    featured = featured_showcase_entry(replay)
    gid = featured.get("game_id") if featured else replay.get("game_id")
    if gid:
        return f"/game/{gid}", "replay", featured
    return "", "none", None


LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenFront RL Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; }
    body {
      line-height: 1.45;
      font-size: 18px;
      padding: 3rem 1.25rem 4rem;
      color: #000;
      background: #fff;
      text-align: center;
    }
    .page {
      max-width: 920px;
      margin: 0 auto;
    }
    h1 {
      font-size: clamp(2.4rem, 7vw, 4rem);
      font-weight: 800;
      letter-spacing: -.03em;
      line-height: 1.05;
      margin-bottom: 1rem;
    }
    .lead {
      font-size: clamp(1.1rem, 2.5vw, 1.35rem);
      font-weight: 500;
      max-width: 36rem;
      margin: 0 auto 2rem;
      color: #111;
    }
    .lead a { font-weight: 600; }
    .preview {
      margin: 0 auto 2rem;
      max-width: 860px;
      border: 2px solid #000;
      background: #000;
    }
    .preview video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      border: 0;
      background: #000;
    }
    .preview iframe {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 0;
      background: #000;
    }
    .placeholder {
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      color: #888;
      font-size: 1rem;
      background: #111;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 1rem 1.5rem;
      margin-bottom: 1.25rem;
    }
    .actions a {
      font-size: 1.15rem;
      font-weight: 700;
      text-decoration: underline;
      text-underline-offset: 4px;
    }
    .meta {
      font-size: .95rem;
      color: #666;
      margin-bottom: 1.5rem;
    }
    .links {
      font-size: 1rem;
      color: #666;
    }
    .links a { font-weight: 600; }
    .sep { margin: 0 .5rem; }
  </style>
</head>
<body>
  <main class="page">
    <h1>OpenFront Agent</h1>
    <p class="lead">A reinforcement learning agent that plays
      <a href="https://openfront.io">OpenFront.io</a>, trained on the real
      game engine with live model overlay. Play it 1v1.</p>

    <figure class="preview">
      %%PREVIEW%%
    </figure>

    <div class="actions">
      <a href="/watch">%%WATCH_LABEL%%</a>
      <a href="/play">Play vs Agent</a>
    </div>

    <p class="meta">policy: %%RUN_NAME%%</p>

    <p class="links">
      <a href="https://skg.gg" target="_blank" rel="noopener">skg.gg</a>
      <span class="sep">·</span>
      <a href="https://skg.gg/pages/openfront-devlog/" target="_blank" rel="noopener">Devlog</a>
      <span class="sep">·</span>
      <a href="https://github.com/djmango/openfront-ai" target="_blank" rel="noopener">GitHub</a>
    </p>
  </main>
</body>
</html>
"""


def preview_markup(replay: dict) -> str:
    featured = featured_showcase_entry(replay)
    clip_url = featured.get("url") if featured else None
    if not clip_url:
        for entry in replay.get("hero_clips") or []:
            clip_url = entry if isinstance(entry, str) else entry.get("url")
            if clip_url:
                break
    if clip_url:
        map_label = featured.get("map", "") if featured else ""
        title = f"Replay preview ({map_label})" if map_label else "Replay preview"
        return (
            f'<video autoplay muted loop playsinline preload="auto" '
            f'src="{clip_url}" title="{title}"></video>'
        )
    return '<div class="placeholder">Preview loading...</div>'


def render_landing(replay: dict) -> str:
    _, mode, featured = watch_target()
    map_name = featured.get("map") if featured else replay.get("map")
    watch_label = "Watch replay" if mode == "replay" else "Watch"
    if map_name:
        watch_label = f"Watch {map_name}"
    return (
        LANDING_HTML.replace("%%RUN_NAME%%", str(replay.get("run_name", RUN_NAME)))
        .replace("%%PREVIEW%%", preview_markup(replay))
        .replace("%%WATCH_LABEL%%", watch_label)
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
            target, mode, featured = watch_target()
            if not target:
                self._send(503, b'{"status":"warming","message":"no replay yet"}')
                return
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("X-Showcase-Watch", mode)
            if featured and featured.get("map"):
                self.send_header("X-Showcase-Map", str(featured["map"]))
            self.end_headers()
            return

        if path == "/status":
            target, mode, featured = watch_target()
            replay = load_replay_state()
            next_rotate = None
            maps = replay.get("maps") or []
            if maps:
                hour = int(time.time() // 3600)
                next_rotate = (hour + 1) * 3600
            payload = {
                "watch": {
                    "url": target or None,
                    "mode": mode,
                    "map": featured.get("map") if featured else replay.get("map"),
                    "game_id": featured.get("game_id") if featured else replay.get("game_id"),
                    "next_rotate_at": next_rotate,
                },
                "replay": replay,
                "hub": load_hub_state(),
                "play_config": play_config(),
                "live_showcase": LIVE_SHOWCASE,
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
            redirect = None
            with _lock:
                if _active_game and _active_proc and _active_proc.poll() is None:
                    log(f"reusing active lobby {_active_game}")
                    redirect = play_redirect(_active_game)
                else:
                    try:
                        info = create_play_lobby()
                    except Exception as exc:
                        log(f"lobby create failed: {exc}")
                        self._send(500, json.dumps({"error": str(exc)}).encode())
                        return

                    game_id = info["gameID"]
                    worker_index = int(info["workerIndex"])
                    worker_path = info.get("workerPath") or f"w{worker_index}"
                    _active_game = game_id
                    _active_proc = launch_webbot_agent(game_id, worker_path)

                    hub_payload = {
                        "game_id": game_id,
                        "status": "lobby",
                        "config": play_config(),
                        "run_name": RUN_NAME,
                        "started_at": utc_now(),
                        "worker_index": worker_index,
                        "worker_path": worker_path,
                    }
                    try:
                        if not wait_for_webbot_join(game_id, worker_path):
                            log(
                                f"webbot didn't join {game_id} within timeout; "
                                "arming countdown anyway"
                            )
                        started = start_play_lobby(game_id, worker_index)
                        hub_payload["status"] = "countdown"
                        hub_payload["starts_at"] = started.get("startsAt")
                    except Exception as exc:
                        log(f"lobby start failed: {exc}")
                        hub_payload["start_error"] = str(exc)

                    write_json(HUB_STATE, hub_payload)
                    redirect = play_redirect(game_id, worker_path)
                    log(f"agent joining {game_id}; redirect -> {redirect}")

            if redirect:
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
            return

        self._send(404, b'{"error":"unknown route"}')

    def log_message(self, fmt: str, *args) -> None:
        pass


def live_showcase_loop(policy: Path, ae: Path) -> None:
    """Run continuous bot lobbies so /watch can spectate the current game."""
    backoff = 15
    while True:
        try:
            with _lock:
                if _active_proc and _active_proc.poll() is None:
                    time.sleep(5)
                    continue

            info = create_play_lobby()
            game_id = info["gameID"]
            worker_index = int(info["workerIndex"])
            worker_path = info.get("workerPath") or f"w{worker_index}"
            proc = launch_agent(
                game_id, policy, ae, debug_port=LIVE_DEBUG_PORT,
            )

            hub_payload: dict = {
                "live_game_id": game_id,
                "live_worker_path": worker_path,
                "live_status": "lobby",
                "live_pid": proc.pid,
                "run_name": RUN_NAME,
                "started_at": utc_now(),
                "worker_index": worker_index,
                "worker_path": worker_path,
            }
            write_json(HUB_STATE, hub_payload)

            try:
                started = start_play_lobby(game_id, worker_index)
                hub_payload["live_status"] = "countdown"
                hub_payload["starts_at"] = started.get("startsAt")
                write_json(HUB_STATE, hub_payload)
            except Exception as exc:
                log(f"live lobby start failed: {exc}")
                proc.terminate()
                raise

            hub_payload["live_status"] = "playing"
            write_json(HUB_STATE, hub_payload)
            log(f"live showcase playing {game_id} -> {play_redirect(game_id, worker_path)}")

            rc = proc.wait()
            hub_payload["live_status"] = "ended"
            hub_payload["ended_at"] = utc_now()
            hub_payload["exit_code"] = rc
            write_json(HUB_STATE, hub_payload)
            log(f"live game {game_id} ended (rc={rc}); starting next")
            backoff = 15
        except Exception as exc:
            log(f"live showcase error: {exc}")
            write_json(
                HUB_STATE,
                {
                    **load_hub_state(),
                    "live_status": "error",
                    "live_error": str(exc),
                    "failed_at": utc_now(),
                },
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)


def main() -> None:
    port = int(os.environ.get("HUB_PORT", "8988"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log("loading policy + encoder")
    ae = ensure_ae(AE_PATH)
    policy = ensure_policy(RUN_NAME)
    HubHandler.policy_path = policy
    HubHandler.ae_path = ae

    if LIVE_SHOWCASE:
        threading.Thread(
            target=live_showcase_loop,
            args=(policy, ae),
            daemon=True,
            name="live-showcase",
        ).start()
        log("live showcase loop enabled")

    srv = ThreadingHTTPServer(("0.0.0.0", port), HubHandler)
    log(f"hub on :{port} (watch=/watch, play=/play)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
