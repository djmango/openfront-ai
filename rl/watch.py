"""Run one greedy episode and save an engine GameRecord for client replay.

The real OpenFront client replays the record via `ofshowcase archive`.
For real-graphics video, render it with scripts/render_client_replay.py.

Usage:
  uv run python -m rl.watch --policy rust/checkpoints/ppo_v81/latest.safetensors --stage 3 \\
      --record records-rl/game.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl.curriculum import STAGES
from rl.env import OpenFrontEnv
from rl.obs import ACTIONS, ObsBuilder, encode_grids, load_ae
from rl.policy import Policy
from rl.ppo import OBS_KEYS
from rl.ppo_translate import IntentTranslator, my_tiles, spawn_randomly
from rl.showcase_util import LEGACY_POLICY_RUNS


def load_policy_checkpoint(
    policy: Policy, checkpoint: str | Path, device: str
) -> dict:
    """Strictly dispatch current safetensors and frozen legacy checkpoints."""
    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        from scripts.policy_safetensors import load_oftrain_safetensors

        metadata = load_oftrain_safetensors(policy, path)
        state = metadata["state"]
        if not isinstance(state, dict):
            raise ValueError("invalid safetensors state metadata")
        return state
    if path.suffix == ".pt" and path.parent.name in LEGACY_POLICY_RUNS:
        state = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model_state_dict"], strict=True)
        return state
    raise ValueError(
        "policy must be .safetensors; policy.pt is supported only for "
        "explicitly legacy ppo_v5/ppo_v7 runs"
    )


def describe(choice: dict, obs: dict) -> str:
    """One-line human description of a policy choice."""
    from rl.curriculum import GW_MAX
    from rl.obs import BUILD_TYPES, NUKE_TYPES, REGION

    name = ACTIONS[choice["action"]]
    parts = [name]
    if "player_slot" in choice:
        parts.append(f"-> P{choice['player_slot']}")
    if "tile_region" in choice:
        gy, gx = divmod(choice["tile_region"], GW_MAX)
        parts.append(f"@({gx},{gy})")
    if "build_type" in choice:
        parts.append(BUILD_TYPES[choice["build_type"]])
    if "nuke_type" in choice:
        unit, up = NUKE_TYPES[choice["nuke_type"]]
        parts.append(unit if up is None else f"{unit} {'up' if up else 'down'}")
    if "quantity_frac" in choice:
        frac = choice["quantity_frac"]
        troops = int(obs["legal"]["actions"].get("troops", 0) * frac)
        parts.append(f"{frac * 100:.0f}% ({troops:,})")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--map", default=None, help="override map (default: first in stage pool)")
    ap.add_argument(
        "--nations",
        default=None,
        help='override nations count or "default"/"disabled"',
    )
    ap.add_argument("--bots", type=int, default=None, help="override bot count")
    ap.add_argument("--difficulty", default=None, help="override difficulty")
    ap.add_argument("--record", default=None,
                    help="engine GameRecord JSON; default records-rl/<run>_s<stage>_<seed>.json")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", default="watch0")
    ap.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=True,
        help="write .debug.json sidecar for the client MODEL overlay (--no-debug to disable)",
    )
    args = ap.parse_args()

    run = Path(args.policy).parent.name or "policy"
    base = f"{run}_s{args.stage}_{args.seed}"
    if args.record is None:
        args.record = f"records-rl/{base}.json"
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)

    st = STAGES[args.stage]
    map_name = args.map or st.maps[0]
    nations = st.nations if args.nations is None else args.nations
    if isinstance(nations, str) and nations not in ("default", "disabled"):
        nations = int(nations)
    bots = st.bots if args.bots is None else args.bots
    difficulty = st.difficulty if args.difficulty is None else args.difficulty
    print(
        f"stage {args.stage}: {map_name}, nations={nations}, "
        f"{bots} bots, {difficulty}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = load_ae(args.ckpt, device)
    policy = Policy().to(device)
    state = load_policy_checkpoint(policy, args.policy, device)
    policy.eval()
    print(f"device: {device}")
    print(f"policy from update {state.get('update', '?')}, stage {state.get('stage', '?')}")

    env = OpenFrontEnv()
    builder = ObsBuilder()
    obs = env.reset(
        map_name, seed=args.seed, bots=bots, difficulty=difficulty,
        nations=nations,
    )
    builder.start_game(env.terrain)
    rng = np.random.default_rng(0)
    translator = IntentTranslator(env, builder)
    for _ in range(8):
        if not obs["spawnPhase"]:
            break
        raw = builder.prepare(obs)
        o = encode_grids(ae, [raw], device)[0]
        ot = {k: torch.from_numpy(o[k])[None].to(device) for k in OBS_KEYS}
        choices, _, _ = policy.act(ot)
        obs = env.step(translator.translate(choices[0], obs), ticks=10)
    if obs["spawnPhase"]:
        obs = spawn_randomly(env, rng)
    builder._slot_lut(obs["entities"]["players"])

    debug_log: list[dict] = []
    episode_outcome = "death"
    end_tick = 0

    for step in range(args.max_steps):
        raw = builder.prepare(obs)
        o = encode_grids(ae, [raw], device)[0]
        ot = {k: torch.from_numpy(o[k])[None].to(device) for k in OBS_KEYS}
        choices, _, _ = policy.act(ot, debug=args.debug)
        choice = choices[0]
        choice["_desc"] = describe(choice, obs)
        if args.record:
            me = next(
                (p for p in obs["entities"]["players"] if p["id"] == obs["me"]),
                None,
            )
            entry = {
                "tick": obs["tick"],
                "desc": choice["_desc"],
                "action": ACTIONS[choice["action"]],
                "tiles": me["tiles"] if me else 0,
                "troops": int(me["troops"]) if me else 0,
            }
            dbg = choice.get("debug")
            if dbg is not None:
                entry["value"] = round(float(dbg["value"]), 3)
                entry["probs"] = [round(float(p), 4) for p in dbg["action_probs"]]
            debug_log.append(entry)
        obs = env.step(translator.translate(choice, obs), ticks=10)

        if step % 100 == 0:
            print(
                f"step {step}, tick {obs['tick']}, my tiles {my_tiles(obs)}, alive {obs['alive']}",
                flush=True,
            )
        if not obs["alive"] or obs["winner"] is not None:
            end_tick = int(obs["tick"])
            w = obs.get("winner")
            won = (
                w is not None
                and isinstance(w, list)
                and len(w) > 1
                and w[1] == "AGENTRL1"
            )
            episode_outcome = "win" if won else "death"
            print(
                f"episode over at tick {end_tick}: alive={obs['alive']}, "
                f"winner={w}, outcome={episode_outcome}",
                flush=True,
            )
            break

    if args.record:
        info = env.save_record(str(Path(args.record).resolve()))
        print(f"game record: {info['saved']} (gameID {info['gameID']}, {info['turns']} turns)")
        if args.debug:
            sidecar = Path(args.record).with_suffix(".debug.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "actions": ACTIONS,
                        "log": debug_log,
                        "outcome": episode_outcome,
                        "end_tick": end_tick,
                    }
                )
            )
            print(f"debug sidecar: {sidecar} ({len(debug_log)} decisions)")
    env.close()


if __name__ == "__main__":
    main()
