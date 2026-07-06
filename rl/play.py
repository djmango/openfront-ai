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
import subprocess

import numpy as np
import torch

from rl.env import FALLOUT_BIT, OWNER_MASK, REPO_ROOT, TSX
from rl.obs import ObsBuilder, encode_grids, load_ae
from rl.policy import Policy
from rl.ppo import OBS_KEYS
from rl.ppo_translate import IntentTranslator, my_tiles


class _EnvShim:
    """Just enough of OpenFrontEnv's surface for IntentTranslator."""

    def __init__(self, width: int, height: int, terrain: np.ndarray):
        self.width = width
        self.height = height
        self.terrain = terrain


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
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--game", required=True, help="lobby ID from the browser")
    ap.add_argument("--host", default="localhost:9000")
    args = ap.parse_args()

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
    spawn_tile: int | None = None
    steps = 0

    for line in proc.stdout:
        msg = json.loads(line)
        event = msg.get("event")

        if event == "lobby":
            continue
        if event == "start":
            terr = gzip.decompress(base64.b64decode(msg["terrain"]))
            terrain = np.frombuffer(terr, dtype=np.uint8).reshape(
                msg["height"], msg["width"]
            )
            env = _EnvShim(msg["width"], msg["height"], terrain)
            builder.start_game(terrain)
            translator = IntentTranslator(env, builder)  # type: ignore[arg-type]
            land = (terrain >> 7) & 1
            ys, xs = np.nonzero(land)
            i = rng.integers(len(ys))
            spawn_tile = int(ys[i]) * msg["width"] + int(xs[i])
            print(f"game started: {msg['width']}x{msg['height']}, spawning")
            continue
        if event == "end":
            print(f"game over: winner={msg.get('winner')}, alive={msg.get('alive')}")
            break
        if event != "obs" or env is None or translator is None:
            continue

        obs = decode_tiles(msg, env.width, env.height)
        if obs["spawnPhase"]:
            intents = [{"type": "spawn", "tile": spawn_tile}] if spawn_tile else []
        elif not obs["alive"]:
            intents = []
        else:
            raw = builder.prepare(obs)
            o = encode_grids(ae, [raw], device)[0]
            ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
            with torch.no_grad():
                choices, _, _ = policy.act(ot)
            intents = translator.translate(choices[0], obs)
            steps += 1
            if steps % 50 == 0:
                print(f"tick {obs['tick']}: my tiles {my_tiles(obs)}")

        proc.stdin.write(json.dumps({"intents": intents}) + "\n")
        proc.stdin.flush()

    proc.terminate()


if __name__ == "__main__":
    main()
