"""Python wrapper over the Node environment bridge (bridge/env.ts).

Speaks JSONL over the subprocess's stdio. One env instance = one persistent
Node process; reset() starts a fresh game in the same process.
"""

import base64
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TSX = REPO_ROOT / "openfront" / "node_modules" / ".bin" / "tsx"

OWNER_MASK = 0x0FFF
FALLOUT_BIT = 13
DEFENSE_BONUS_BIT = 14


class OpenFrontEnv:
    def __init__(self):
        # Binary stdio: JSON header lines, with obs tile state following as
        # a raw frame of "tilesBin" bytes (no gzip/base64 - the codec was
        # measurable CPU on both sides at 48 envs).
        self.proc = subprocess.Popen(
            [str(TSX), str(REPO_ROOT / "bridge" / "env.ts")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT,
        )
        self.width = 0
        self.height = 0
        self.terrain: np.ndarray | None = None
        self.me = -1

    def _read_exact(self, n: int) -> bytes:
        assert self.proc.stdout
        chunks = []
        while n > 0:
            b = self.proc.stdout.read(n)
            if not b:
                raise RuntimeError("bridge died mid-frame")
            chunks.append(b)
            n -= len(b)
        return b"".join(chunks)

    def _rpc(self, msg: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("bridge died")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(f"bridge error: {out['error']}")
        if "tilesBin" in out:
            out["tiles_raw"] = self._read_exact(int(out.pop("tilesBin")))
        return out

    def reset(
        self,
        map_name: str = "Onion",
        seed: str = "0",
        bots: int = 100,
        difficulty: str = "Medium",
        nations: int | str = "default",
    ) -> dict:
        obs = self._rpc(
            {"op": "reset", "map": map_name, "seed": seed, "bots": bots,
             "difficulty": difficulty, "nations": nations}
        )
        self.width, self.height = obs["width"], obs["height"]
        terr = gzip.decompress(base64.b64decode(obs["terrain"]))
        self.terrain = np.frombuffer(terr, dtype=np.uint8).reshape(
            self.height, self.width
        )
        return self._decode(obs)

    def step(self, intents: list[dict], ticks: int = 10) -> dict:
        return self._decode(self._rpc({"op": "step", "intents": intents, "ticks": ticks}))

    def _decode(self, obs: dict) -> dict:
        state = np.frombuffer(obs.pop("tiles_raw"), dtype="<u2").reshape(
            self.height, self.width
        )
        obs["owners"] = state & OWNER_MASK
        obs["fallout"] = (state >> FALLOUT_BIT) & 1
        obs["defense_bonus"] = (state >> DEFENSE_BONUS_BIT) & 1
        self.me = obs["me"]
        return obs

    def save_record(self, path: str) -> dict:
        """Dump the episode so far as a GameRecord the real client can replay."""
        return self._rpc({"op": "save_record", "path": path})

    def close(self) -> None:
        try:
            self._rpc({"op": "close"})
        except Exception:
            pass
        self.proc.terminate()
