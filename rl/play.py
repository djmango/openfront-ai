"""Play against the trained agent in a real OpenFront lobby.

Workflow:
  1. (cd openfront && npm run dev)   # client :9000 + game server
  2. In the browser: Create Lobby (private), copy the lobby ID.
  3. uv run python -m rl.play --policy /tmp/policy.pt --game <LOBBY_ID>
     -> "AgentRL" appears in the lobby; click Start.
  4. Fight it.

The Node side (bridge/play.ts) speaks the real client websocket protocol
and mirrors the sim locally; this side picks actions with the policy.
"""

import argparse
import base64
import gzip
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from rl.env import FALLOUT_BIT, OWNER_MASK, REPO_ROOT, TSX
from rl.obs import ACTIONS, ObsBuilder, encode_grids, load_ae
from rl.policy import Policy
from rl.ppo import OBS_KEYS
from rl.ppo_translate import IntentTranslator, my_tiles
from rl.watch import describe


class _EnvShim:
    """Just enough of OpenFrontEnv's surface for IntentTranslator."""

    def __init__(self, width: int, height: int, terrain: np.ndarray):
        self.width = width
        self.height = height
        self.terrain = terrain


def parse_game_id(s: str) -> str:
    """Accept a bare 8-char lobby ID or a full lobby URL
    (http://host/w1/game/<ID>?lobby&s=...)."""
    m = re.search(r"/game/([A-Za-z0-9]{8})", s)
    if m:
        return m.group(1)
    s = s.strip()
    if re.fullmatch(r"[A-Za-z0-9]{8}", s):
        return s
    raise SystemExit(f"could not parse a lobby ID from {s!r}")


def start_debug_server(game_id: str, port: int, log: list, lock: threading.Lock):
    """Serve the model's decisions at /debug/<gameID> for the in-client
    overlay (set localStorage rlDebugHost = "http://localhost:<port>")."""

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == f"/debug/{game_id}":
                with lock:
                    body = json.dumps(
                        {"actions": list(ACTIONS), "log": list(log), "live": True}
                    ).encode()
                code = 200
            else:
                body, code = b'{"error":"not found"}', 404
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"debug overlay feed: http://localhost:{port}/debug/{game_id}")
    print('  in the browser console: localStorage.setItem("rlDebugHost", '
          f'"http://localhost:{port}")')


def decode_tiles(obs: dict, width: int, height: int) -> dict:
    raw = gzip.decompress(base64.b64decode(obs["tiles"]))
    state = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    obs["owners"] = state & OWNER_MASK
    obs["fallout"] = (state >> FALLOUT_BIT) & 1
    del obs["tiles"]
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--game", required=True, help="lobby ID from the browser")
    ap.add_argument("--host", default="localhost:9000")
    ap.add_argument("--debug-port", type=int, default=8988,
                    help="serve decisions for the in-client overlay (0 = off)")
    args = ap.parse_args()
    args.game = parse_game_id(args.game)

    device = "cpu"
    ae = load_ae(args.ckpt, device)
    policy = Policy()
    state = torch.load(args.policy, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    print(f"policy loaded (update {state.get('update', '?')}); joining {args.game}")

    proc = subprocess.Popen(
        [str(TSX), str(REPO_ROOT / "bridge" / "play.ts"),
         "--game", args.game, "--host", args.host],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # bridge logs pass through to the terminal
        cwd=REPO_ROOT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout

    env: _EnvShim | None = None
    builder = ObsBuilder()
    translator: IntentTranslator | None = None
    rng = np.random.default_rng()
    steps = 0

    debug_log: list[dict] = []
    debug_lock = threading.Lock()
    if args.debug_port:
        start_debug_server(args.game, args.debug_port, debug_log, debug_lock)

    for line in proc.stdout:
        msg = json.loads(line)
        event = msg.get("event")

        if event == "lobby":
            continue
        if event == "start":
            from rl.curriculum import GH_MAX, GW_MAX
            from rl.obs import REGION

            gh, gw = msg["height"] // REGION, msg["width"] // REGION
            if gh > GH_MAX or gw > GW_MAX:
                print(
                    f"note: map grid {gh}x{gw} exceeds the training max "
                    f"{GH_MAX}x{GW_MAX}; the policy will run but is out of "
                    "distribution here"
                )
            terr = gzip.decompress(base64.b64decode(msg["terrain"]))
            terrain = np.frombuffer(terr, dtype=np.uint8).reshape(
                msg["height"], msg["width"]
            )
            env = _EnvShim(msg["width"], msg["height"], terrain)
            builder.start_game(terrain)
            translator = IntentTranslator(env, builder)  # type: ignore[arg-type]
            print(f"game started: {msg['width']}x{msg['height']}, spawning")
            continue
        if event == "end":
            print(f"game over: winner={msg.get('winner')}, alive={msg.get('alive')}")
            break
        if event != "obs" or env is None or translator is None:
            continue

        obs = decode_tiles(msg, env.width, env.height)
        if obs["spawnPhase"]:
            # The policy places its own spawn (spawn action + tile head);
            # random fallback keeps live games moving if the snap fails.
            raw = builder.prepare(obs)
            o = encode_grids(ae, [raw], device)[0]
            ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
            with torch.no_grad():
                choices, _, _ = policy.act(ot)
            intents = translator.translate(choices[0], obs)
            if not intents:
                land = (env.terrain >> 7) & 1
                ys, xs = np.nonzero(land)
                i = rng.integers(len(ys))
                intents = [
                    {"type": "spawn", "tile": int(ys[i]) * env.width + int(xs[i])}
                ]
        elif not obs["alive"]:
            intents = []
        else:
            raw = builder.prepare(obs)
            o = encode_grids(ae, [raw], device)[0]
            ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
            with torch.no_grad():
                choices, _, _ = policy.act(ot, debug=args.debug_port > 0)
            choice = choices[0]
            if args.debug_port:
                me = next(
                    (p for p in obs["entities"]["players"] if p["id"] == obs["me"]),
                    None,
                )
                entry = {
                    "tick": obs["tick"],
                    "desc": describe(choice, obs),
                    "action": ACTIONS[choice["action"]],
                    "tiles": me["tiles"] if me else 0,
                    "troops": int(me["troops"]) if me else 0,
                }
                dbg = choice.get("debug")
                if dbg is not None:
                    entry["value"] = round(float(dbg["value"]), 3)
                    entry["probs"] = [round(float(p), 4) for p in dbg["action_probs"]]
                with debug_lock:
                    debug_log.append(entry)
            intents = translator.translate(choice, obs)
            steps += 1
            if intents:
                print(f"tick {obs['tick']} (tiles {my_tiles(obs)}): "
                      + "; ".join(json.dumps(i) for i in intents))
            elif steps % 50 == 0:
                print(f"tick {obs['tick']}: idle, my tiles {my_tiles(obs)}")

        proc.stdin.write(json.dumps({"intents": intents}) + "\n")
        proc.stdin.flush()

    proc.terminate()


if __name__ == "__main__":
    main()
