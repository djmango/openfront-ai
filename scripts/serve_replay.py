"""Serve agent GameRecords to the real OpenFront client replay viewer.

The dev client fetches archived games from `${apiHost}/game/<gameID>`.
This server mimics that endpoint for records saved by rl/watch.py.

Workflow (automated: scripts/render_client_replay.py does all of this and
records a webm):
  1. uv run python -m rl.watch --policy ... --record records-rl/game.json
  2. uv run python scripts/serve_replay.py --records records-rl --port 8987
  3. (cd openfront && npm run start:client -- --host 127.0.0.1)
  4. In the browser devtools console on the client page:
       localStorage.setItem("apiHost", "http://localhost:8987")
       localStorage.setItem("replayViewAs", "1")  // adopt agent perspective
  5. Open http://localhost:9000/game/<gameID> (ID printed by step 1).
     The client fetches the record here and replays it in the full game UI.
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


from rl.showcase_util import featured_game_id


def load_state(state_path: Path | None) -> dict:
    if state_path is None or not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def build_index(records_dir: Path) -> dict[str, Path]:
    idx = {}
    for f in records_dir.rglob("*.json"):
        try:
            gid = json.loads(f.read_text())["info"]["gameID"]
            idx[gid] = f
        except Exception:
            continue
    return idx


class Handler(BaseHTTPRequestHandler):
    index: dict[str, Path] = {}
    records_dir: Path = Path("records-rl")
    clips_dir: Path | None = None
    state_path: Path | None = None

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._send(204, b"")

    def _refresh_index(self) -> None:
        self.index = build_index(self.records_dir)

    def do_GET(self) -> None:
        if self.path == "/replay":
            state = load_state(self.state_path)
            gid = featured_game_id(state) or state.get("game_id")
            if not gid:
                self._send(
                    503,
                    b'{"status":"warming","message":"showcase replay is generating"}',
                )
                return
            self.send_response(302)
            self.send_header("Location", f"/game/{gid}")
            self.end_headers()
            return
        if self.path == "/status":
            self._refresh_index()
            payload = {
                "records": len(self.index),
                **load_state(self.state_path),
            }
            self._send(200, json.dumps(payload).encode())
            return

        m = re.fullmatch(r"/clips/([A-Za-z0-9_.-]+\.webm)", self.path)
        if m and self.clips_dir is not None:
            clip = self.clips_dir / m.group(1)
            if clip.is_file():
                self._send_file(clip, "video/webm")
            else:
                self._send(404, b'{"error":"clip not found"}')
            return

        self._refresh_index()
        # Archived-game fetch: /game/<id>
        m = re.fullmatch(r"/game/([A-Za-z0-9]{8})", self.path)
        if m:
            f = self.index.get(m.group(1))
            if f is None:
                self._send(404, b'{"error":"not found"}')
            else:
                self._send(200, f.read_bytes())
            return
        # Model debug sidecar (rl.watch --record): /debug/<id>
        m = re.fullmatch(r"/debug/([A-Za-z0-9]{8})", self.path)
        if m:
            f = self.index.get(m.group(1))
            side = f.with_suffix(".debug.json") if f else None
            if side is None or not side.exists():
                self._send(404, b'{"error":"no debug sidecar"}')
            else:
                self._send(200, side.read_bytes())
            return
        # Live-lobby existence probe: force the archive path.
        m = re.fullmatch(r"/api/game/([A-Za-z0-9]{8})/exists", self.path)
        if m:
            self._send(200, b'{"exists":false}')
            return
        self._send(404, b'{"error":"unknown route"}')

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records-rl")
    ap.add_argument("--port", type=int, default=8987)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--state", default=None, help="state.json from eval_daemon (for / and /status)")
    ap.add_argument("--clips", default=None, help="pre-rendered landing clips directory")
    args = ap.parse_args()

    records_dir = Path(args.records)
    Handler.records_dir = records_dir
    Handler.state_path = Path(args.state) if args.state else None
    Handler.clips_dir = Path(args.clips) if args.clips else None
    Handler.index = build_index(records_dir)
    print(f"serving {len(Handler.index)} record(s) on http://{args.bind}:{args.port}")
    for gid, f in Handler.index.items():
        print(f"  {gid}  {f}")
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
