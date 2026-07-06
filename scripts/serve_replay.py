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

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._send(204, b"")

    def do_GET(self) -> None:
        # Archived-game fetch: /game/<id>
        m = re.fullmatch(r"/game/([A-Za-z0-9]{8})", self.path)
        if m:
            f = self.index.get(m.group(1))
            if f is None:
                self._send(404, b'{"error":"not found"}')
            else:
                self._send(200, f.read_bytes())
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
    args = ap.parse_args()

    Handler.index = build_index(Path(args.records))
    print(f"serving {len(Handler.index)} record(s) on http://localhost:{args.port}")
    for gid, f in Handler.index.items():
        print(f"  {gid}  {f}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
