#!/usr/bin/env python3
"""Confirm ofenv (Rust TS bridge) matches Node bridge/env.ts on the same trajectory."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("OPENFRONT_REPO", Path(__file__).resolve().parent.parent))
TSX = REPO / "openfront/node_modules/.bin/tsx"
OWNER_MASK = 0x0FFF


def node_env():
    proc = subprocess.Popen(
        [str(TSX), str(REPO / "bridge/env.ts")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=REPO,
    )

    def rpc(msg: dict) -> dict:
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        out = json.loads(line)
        if "tilesBin" in out:
            out["tiles_raw"] = proc.stdout.read(int(out.pop("tilesBin")))
        return out

    return proc, rpc


def owners(raw: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(raw, dtype="<u2").reshape(h, w) & OWNER_MASK


def main() -> None:
    os.environ["OPENFRONT_REPO"] = str(REPO)
    os.environ["OPENFRONT_ENV"] = "native"
    from rl.env import OpenFrontEnv

    seed, bots, map_name = "parity-check", 12, "Onion"
    proc, rpc = node_env()
    r = OpenFrontEnv()
    try:
        n = rpc(
            {
                "op": "reset",
                "map": map_name,
                "seed": seed,
                "bots": bots,
                "difficulty": "Medium",
            }
        )
        rust = r.reset(map_name=map_name, seed=seed, bots=bots)
        h, w = n["height"], n["width"]

        for key in ("tick", "width", "height", "spawnPhase", "me", "alive"):
            assert n[key] == rust[key], f"reset mismatch {key}: {n[key]} != {rust[key]}"

        n2 = rpc({"op": "step", "intents": [], "ticks": 10})
        rust2 = r.step([], ticks=10)
        for key in ("tick", "spawnPhase", "alive"):
            assert n2[key] == rust2[key], f"step mismatch {key}: {n2[key]} != {rust2[key]}"

        no = owners(n2["tiles_raw"], h, w)
        assert np.array_equal(no, rust2["owners"]), "tile owner mismatch after step"
        print("env parity ok: reset + 10 ticks identical")
    finally:
        proc.stdin.write(b'{"op":"close"}\n')
        proc.stdin.flush()
        proc.terminate()
        r.close()


if __name__ == "__main__":
    main()
