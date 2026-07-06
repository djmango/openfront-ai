"""Random-agent smoke test: spawn, expand, print stats every 50 ticks.

Usage:
  uv run python -m rl.smoke --map Onion --steps 100
"""

import argparse

import numpy as np

from rl.env import OpenFrontEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Onion")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    env = OpenFrontEnv()
    obs = env.reset(args.map, seed="smoke1")
    print(f"reset: {env.width}x{env.height}, tick {obs['tick']}, me={obs['me']}")

    rng = np.random.default_rng(0)

    # Spawn on a random land tile during spawn phase.
    land = (env.terrain >> 7) & 1
    ys, xs = np.nonzero(land)
    while obs["spawnPhase"]:
        i = rng.integers(len(ys))
        tile = int(ys[i]) * env.width + int(xs[i])
        obs = env.step([{"type": "spawn", "tile": tile}], ticks=10)
    me = obs["me"]
    print(f"spawn phase over at tick {obs['tick']}, me={me}, alive={obs['alive']}")

    for step in range(args.steps):
        intents = []
        legal = obs["legal"]["actions"]
        # Random policy: attack a random bordering enemy or expand into
        # neutral land with 30% of troops.
        if legal and obs["alive"]:
            targets = legal.get("attackable", [])
            troops = int(legal.get("troops", 0) * 0.3)
            if troops > 50:
                target = (
                    int(rng.choice(targets))
                    if targets and rng.random() < 0.5
                    else None
                )
                intents.append(
                    {"type": "attack", "targetID": target, "troops": troops}
                )
        obs = env.step(intents, ticks=10)
        if not obs["alive"]:
            print(f"died at tick {obs['tick']}")
            break
        if obs["winner"] is not None:
            print(f"game over at tick {obs['tick']}: winner={obs['winner']}")
            break
        if step % 10 == 0:
            mine = [p for p in obs["entities"]["players"] if p["id"] == me]
            tiles = mine[0]["tiles"] if mine else 0
            n_alive = sum(p["alive"] for p in obs["entities"]["players"])
            print(
                f"tick {obs['tick']:5d}  my-tiles {tiles:6d}  "
                f"troops {legal.get('troops', 0):7d}  alive-players {n_alive}"
            )

    env.close()
    print("smoke test complete")


if __name__ == "__main__":
    main()
