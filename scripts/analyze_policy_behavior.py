"""Compare policy behavior across curriculum stages.

Runs the real engine with the current policy and prints compact episode
summaries: action mix, final placement, resource trajectory, and structure
usage. Intended as a diagnostic, not a training dependency.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from rl.curriculum import STAGES, placement, sample_episode, strengths
from rl.env import OpenFrontEnv
from rl.obs import ACTIONS, BUILD_TYPES, ObsBuilder, collate, encode_grids, load_ae
from rl.policy import NEEDS_QUANTITY, Policy
from rl.ppo_translate import IntentTranslator, spawn_randomly

STRUCT_TYPES = {"City", "Port", "Defense Post", "Missile Silo", "SAM Launcher", "Factory"}
OBS_KEYS = [
    "grid", "grid_valid", "legal_tile", "local", "players", "pmask", "scalars",
    "legal_actions", "legal_ptarget", "legal_build", "legal_nuke",
]


def me_player(obs: dict) -> dict | None:
    me = obs.get("me", -1)
    for p in obs["entities"]["players"]:
        if p["id"] == me:
            return p
    return None


def own_structs(obs: dict) -> collections.Counter[str]:
    me = obs.get("me", -1)
    c: collections.Counter[str] = collections.Counter()
    for u in obs["entities"].get("units", []):
        if u["owner"] == me and u["type"] in STRUCT_TYPES and not u["constructing"]:
            c[u["type"]] += 1
    return c


def snapshot(obs: dict, land_total: int) -> dict:
    p = me_player(obs)
    s = strengths(obs["entities"], land_total).get(obs.get("me", -1), 0.0)
    if p is None:
        return {"tick": obs["tick"], "alive": False, "strength": s}
    return {
        "tick": obs["tick"],
        "alive": bool(p["alive"]),
        "tiles": int(p["tiles"]),
        "troops": int(p["troops"]),
        "gold": float(p["gold"]),
        "strength": float(s),
        "structures": dict(own_structs(obs)),
    }


@torch.no_grad()
def run_one(policy, ae, device: str, stage: int, seed: str, greedy: bool) -> dict:
    rng = np.random.default_rng(abs(hash(seed)) % (2**32))
    st_rng = np.random.default_rng(abs(hash(("stage", seed))) % (2**32))
    map_name, bots, difficulty, nations, rehearsal = sample_episode(stage, st_rng)

    env = OpenFrontEnv()
    try:
        obs = env.reset(map_name, seed=seed, bots=bots, difficulty=difficulty, nations=nations)
        builder = ObsBuilder()
        builder.start_game(env.terrain)
        translator = IntentTranslator(env, builder)
        land_total = max(1, int(((env.terrain >> 7) & 1).sum()))

        actions: collections.Counter[str] = collections.Counter()
        builds: collections.Counter[str] = collections.Counter()
        quantities: collections.Counter[str] = collections.Counter()
        wasted_total = 0
        spawn_steps = 0
        snaps: list[dict] = []
        max_decisions = 15000 // STAGES[stage].decision_ticks + 64

        for step in range(max_decisions):
            if step in (0, 10, 25, 50, 100, 200, 400, 700, 1000):
                snaps.append(snapshot(obs, land_total))

            raw = builder.prepare(obs)
            enc = encode_grids(ae, [raw], device)[0]
            ot = {k: torch.from_numpy(v).to(device) for k, v in collate([enc], OBS_KEYS).items()}
            choice = policy.act(ot, greedy=greedy)[0][0]
            action = ACTIONS[choice["action"]]
            actions[action] += 1
            if action == "build":
                builds[BUILD_TYPES[choice.get("build_type", 0)]] += 1
            if action in NEEDS_QUANTITY and "quantity_frac" in choice:
                # Decile-bucket the scalar Beta fraction for the summary.
                decile = min(9, int(choice["quantity_frac"] * 10))
                quantities[f"{action}@{decile * 10}-{decile * 10 + 10}%"] += 1
            intents = translator.translate(choice, obs)
            obs = env.step(intents, ticks=STAGES[stage].decision_ticks)
            wasted = int(obs.get("wasted", 0))
            if not intents and action not in ("noop", "spawn"):
                wasted += 1
            wasted_total += wasted

            if obs["spawnPhase"]:
                spawn_steps += 1
                if spawn_steps >= 8:
                    obs = spawn_randomly(env, rng)
                continue

            if not obs["alive"] or obs["winner"] is not None or obs["tick"] >= 15000:
                break

        p = me_player(obs)
        won = isinstance(obs["winner"], list) and len(obs["winner"]) > 1 and obs["winner"][1] == "AGENTRL1"
        place, n = placement(obs["entities"], obs["me"], obs["alive"], land_total)
        snaps.append(snapshot(obs, land_total))
        return {
            "stage": stage,
            "seed": seed,
            "map": map_name,
            "bots": bots,
            "nations": nations,
            "rehearsal": rehearsal,
            "won": won,
            "winner": obs["winner"],
            "alive": bool(obs["alive"]),
            "place": place,
            "n_players": n,
            "final_tick": int(obs["tick"]),
            "final_tiles": int(p["tiles"]) if p else 0,
            "final_troops": int(p["troops"]) if p else 0,
            "final_gold": float(p["gold"]) if p else 0.0,
            "final_structures": dict(own_structs(obs)),
            "actions": dict(actions),
            "build_choices": dict(builds),
            "quantities": dict(quantities),
            "wasted": wasted_total,
            "snapshots": snaps,
        }
    finally:
        env.close()


def summarize(stage: int, rows: list[dict]) -> None:
    wins = np.mean([r["won"] for r in rows])
    places = [r["place"] for r in rows]
    ticks = [r["final_tick"] for r in rows]
    tiles = [r["final_tiles"] for r in rows]
    wasted = [r["wasted"] for r in rows]
    acts: collections.Counter[str] = collections.Counter()
    builds: collections.Counter[str] = collections.Counter()
    quants: collections.Counter[str] = collections.Counter()
    maps: collections.Counter[str] = collections.Counter()
    for r in rows:
        acts.update(r["actions"])
        builds.update(r["build_choices"])
        quants.update(r.get("quantities", {}))
        maps[r["map"]] += 1
    total = sum(acts.values()) or 1
    top_actions = ", ".join(f"{k}:{v / total:.1%}" for k, v in acts.most_common(8))
    q_total = sum(quants.values()) or 1
    quant_mix = ", ".join(
        f"{k}:{v / q_total:.1%}" for k, v in quants.most_common(12)
    )
    print(f"\nSTAGE {stage}  n={len(rows)}  maps={dict(maps)}")
    print(
        f"  win={wins:.2f}  place_mean={np.mean(places):.2f}  "
        f"tick_mean={np.mean(ticks):.0f}  tiles_mean={np.mean(tiles):.0f}  "
        f"wasted_mean={np.mean(wasted):.1f}"
    )
    print(f"  actions: {top_actions}")
    print(f"  build choices: {dict(builds)}")
    print(f"  quantity mix: {quant_mix or 'none'}")
    for r in rows:
        print(
            f"  ep {r['seed']} {r['map']} won={r['won']} place={r['place']}/{r['n_players']} "
            f"tick={r['final_tick']} tiles={r['final_tiles']} wasted={r['wasted']} "
            f"winner={r['winner']} structures={r['final_structures']}"
        )
        for s in r["snapshots"][-4:]:
            print(f"    snap {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="runs/rl/ppo_v5/policy.pt")
    ap.add_argument("--ae", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--stages", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = load_ae(args.ae, device)
    policy = Policy().to(device)
    ck = torch.load(args.policy, map_location=device, weights_only=False)
    policy.load_state_dict(ck["model_state_dict"])
    policy.eval()
    print(f"device={device} policy_update={ck.get('update')} ckpt_stage={ck.get('stage')} greedy={args.greedy}")

    all_rows = []
    for stage in args.stages:
        rows = [
            run_one(policy, ae, device, stage, f"behavior-s{stage}-{i}", args.greedy)
            for i in range(args.episodes)
        ]
        all_rows.extend(rows)
        summarize(stage, rows)

    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
