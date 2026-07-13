"""Inference-only fixed-seed evaluator for a checkout's Python policy schema.

This file is copied into a historical source archive by
paired_policy_benchmark.py. Imports therefore resolve against that checkout,
while its bridge/openfront paths are symlinked to the current tree.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch


def schema_from_state(state: dict) -> dict:
    sd = state["model_state_dict"]
    grid_key = "grid_coarse_net.0.weight" if "grid_coarse_net.0.weight" in sd else "grid_net.0.weight"
    quantity = int(sd["head_quantity.weight"].shape[0])
    return {
        "grid_channels": int(sd[grid_key].shape[1]),
        "player_features": int(sd["player_in.weight"].shape[1]),
        "scalars": 11 if int(sd["trunk.0.weight"].shape[1]) == 715 else 8,
        "local_planes": int(sd["local_net.0.weight"].shape[1]),
        "actions": int(sd["head_action.weight"].shape[0]),
        "build_types": int(sd["head_build.weight"].shape[0]),
        "nuke_types": int(sd["head_nuke.weight"].shape[0]),
        "quantity": "beta" if quantity == 2 else f"categorical-{quantity}",
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--coarse-ae")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--max-ticks", type=int, default=15000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from rl.curriculum import STAGES, sample_episode
    from rl.obs import (
        BUILD_TYPES,
        C_GRID,
        N_ACTIONS,
        N_LOCAL,
        N_SCALARS,
        NUKE_TYPES,
        P_FEAT,
        collate,
        encode_grids,
        load_ae,
    )
    from rl.policy import Policy
    from rl.vec import VecEnv

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    observed = schema_from_state(state)
    source_schema = {
        "grid_channels": C_GRID,
        "player_features": P_FEAT,
        "scalars": N_SCALARS,
        "local_planes": N_LOCAL,
        "actions": N_ACTIONS,
        "build_types": len(BUILD_TYPES),
        "nuke_types": len(NUKE_TYPES),
        "quantity": observed["quantity"],
    }
    if observed != source_schema:
        raise SystemExit(
            "checkpoint/source schema mismatch (refusing partial load): "
            f"checkpoint={observed}, source={source_schema}"
        )
    if observed["grid_channels"] == 43:
        obs_keys = [
            "grid", "grid_valid", "legal_tile", "local", "players", "pmask",
            "scalars", "legal_actions", "legal_ptarget", "legal_build",
            "legal_nuke",
        ]
    else:
        obs_keys = [
            "grid_fine", "grid_fine_valid", "fine_coverage", "fine_origin",
            "legal_tile_fine", "grid_coarse", "grid_coarse_valid",
            "legal_tile_coarse", "coarse_has_land", "coarse_has_water",
            "local", "players", "pmask", "scalars", "legal_actions",
            "legal_ptarget", "legal_build", "legal_nuke",
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy().to(device)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    ae = load_ae(args.ae, device)
    coarse_ae = load_ae(args.coarse_ae, device) if args.coarse_ae else None
    encode_has_coarse = "coarse_ae" in inspect.signature(encode_grids).parameters

    vec = VecEnv(args.episodes, args.stage, args.max_ticks, 10)
    results: dict[int, dict] = {}
    try:
        cap = args.max_ticks // STAGES[args.stage].decision_ticks + 64
        for _ in range(cap):
            pending = [i for i in range(args.episodes) if i not in results]
            if not pending:
                break
            kwargs = {"coarse_ae": coarse_ae} if encode_has_coarse else {}
            encoded = encode_grids(ae, vec.obs_group(pending), device, **kwargs)
            batch = {
                key: torch.from_numpy(value).to(device)
                for key, value in collate(encoded, obs_keys).items()
            }
            choices, _, _ = policy.act(batch, greedy=True)
            vec.send_group(pending, choices)
            for index, (_, _, info) in zip(pending, vec.recv_group(pending)):
                if info is not None:
                    results[index] = info
    finally:
        vec.close()

    episodes = []
    for index, info in sorted(results.items()):
        # Reproduce the worker's first scenario so config is explicit in the
        # artifact and can be checked against every other policy.
        rng = np.random.default_rng(1000 + index)
        map_name, bots, difficulty, nations, rehearsal = sample_episode(args.stage, rng)
        if map_name != info["map"]:
            raise RuntimeError(f"scenario reconstruction mismatch at worker {index}")
        episodes.append(
            {
                "index": index,
                "seed": f"w{index}-ep0",
                "map": info["map"],
                "bots": bots,
                "difficulty": difficulty,
                "nations": nations,
                "rehearsal": rehearsal,
                "decision_ticks": STAGES[args.stage].decision_ticks,
                "won": bool(info["won"]),
                "place": int(info["place"]),
                "n_players": int(info["n_players"]),
                "score": float(info["score"]),
                "final_tick": int(info["final_tick"]),
                "final_tiles": float(info["final_tiles"]),
            }
        )
    report = {
        "format": 1,
        "mode": "indirect-scripted-bot",
        "runner": "python",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "engine": "node-ts",
        "stage": args.stage,
        "max_ticks": args.max_ticks,
        "schema": observed,
        "episodes": episodes,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
