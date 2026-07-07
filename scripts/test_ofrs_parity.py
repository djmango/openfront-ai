"""Parity check: ofrs.Sampler (Rust) vs rl.bc_data.BCSampler (Python).

Randomness makes draw-for-draw comparison impossible, so we compare the
deterministic core on every (game, step): actors only, first label per
actor (ofrs debug_step_samples vs a Python replica of _step_samples with
the same determinism). Every array in the raw dict must match exactly.

  PYTHONPATH=. python scripts/test_ofrs_parity.py --data data-human
"""

import argparse
from pathlib import Path

import numpy as np
import ofrs

from rl.bc_data import NOOP_CHOICE, BCSampler, label_to_choice

ARRAY_KEYS = [
    "owners", "fallout_packed", "terrain_static", "static", "transient",
    "clut", "players", "pmask", "scalars", "legal_actions", "legal_ptarget",
    "legal_build", "legal_nuke", "legal_tile",
]


def python_step_samples(sampler: BCSampler, game, step) -> list[dict]:
    """_step_samples with the randomness pinned: actors only, first label."""
    tick = step["tick"]
    entities = game.entities(tick)
    builder = sampler._builder(game)
    out = []
    for cid, ls in step["labels"].items():
        if not ls or cid not in step["legal"]:
            continue
        leg = step["legal"][cid]
        raw = builder.prepare(sampler._obs(game, tick, entities, leg, spawn=False))
        choice = label_to_choice(ls[0], game, entities, leg["me"])
        if choice is None:
            continue
        raw["legal_actions"][choice["action"]] = 1.0
        if choice["player_slot"] >= 0:
            raw["legal_ptarget"][choice["action"], choice["player_slot"]] = 1.0
        if choice["build_type"] >= 0:
            raw["legal_build"][choice["build_type"]] = 1.0
        if choice["nuke_type"] >= 0:
            raw["legal_nuke"][choice["nuke_type"]] = 1.0
        p = game.placements.get(cid, {"placement": 0.5})
        raw["choice"] = choice
        raw["cond"] = min(7, int(float(p["placement"]) * 8))
        out.append(raw)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--max-steps", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    py = BCSampler([Path(d) for d in args.data], holdout_every=1000, holdout=False)
    rs = ofrs.Sampler([str(d) for d in args.data], holdout_every=1000,
                      holdout=False, noop_frac=0.15, spawn_frac=0.03, seed=0)
    assert rs.n_games() == len(py.games), (rs.n_games(), len(py.games))

    checked = 0
    for gi, game in enumerate(py.games):
        n = game.n_steps() if not args.max_steps else min(game.n_steps(), args.max_steps)
        for si in range(n):
            step = game.step(si)
            want = python_step_samples(py, game, step)
            got = rs.debug_step_samples(gi, si)
            assert len(got) == len(want), (
                f"game {gi} step {si}: {len(got)} rust vs {len(want)} python samples"
            )
            for j, (r, w) in enumerate(zip(got, want)):
                for k in ARRAY_KEYS:
                    a, b = np.asarray(r[k]), np.asarray(w[k])
                    assert a.dtype == b.dtype, f"g{gi} s{si} #{j} {k}: dtype {a.dtype} vs {b.dtype}"
                    assert a.shape == b.shape, f"g{gi} s{si} #{j} {k}: shape {a.shape} vs {b.shape}"
                    if not np.array_equal(a, b):
                        bad = np.argwhere(a != b)[:5]
                        raise AssertionError(
                            f"g{gi} s{si} #{j} {k}: {len(np.argwhere(a != b))} mismatches, "
                            f"first at {bad.tolist()}: rust={a[tuple(bad[0])]} py={b[tuple(bad[0])]}"
                        )
                assert r["me_slot"] == w["me_slot"], f"g{gi} s{si} #{j} me_slot"
                assert dict(r["choice"]) == w["choice"], (
                    f"g{gi} s{si} #{j} choice: {dict(r['choice'])} vs {w['choice']}"
                )
                assert r["cond"] == w["cond"], f"g{gi} s{si} #{j} cond"
            checked += len(want)
        print(f"game {gi} ok ({n} steps)", flush=True)

    # Spawn + window paths: structural smoke (they share featurize/label code
    # with the exactly-checked path; randomness precludes exact comparison).
    b = rs.sample_batch(32)
    assert len(b) == 32 and all("choice" in s for s in b)
    w = rs.sample_window(4)
    assert len(w) in (0, 4)
    if w:
        assert all(s["choice"]["action"] == NOOP_CHOICE["action"] for s in w[:-1])
    print(f"parity ok: {checked} samples compared exactly")


if __name__ == "__main__":
    main()
