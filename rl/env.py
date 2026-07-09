"""Python wrapper over the RL environment.

Default: Node bridge (bridge/env.ts) via JSONL subprocess.
Optional: native Rust engine (ofenv) when built - set OPENFRONT_ENV=native.
"""

import base64
import gzip
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TSX = REPO_ROOT / "openfront" / "node_modules" / ".bin" / "tsx"

OWNER_MASK = 0x0FFF
FALLOUT_BIT = 13
DEFENSE_BONUS_BIT = 14

USE_NATIVE = os.environ.get("OPENFRONT_ENV", "").lower() == "native"

try:
    import ofenv as _ofenv
except ImportError:
    _ofenv = None

HAVE_NATIVE_ENV = _ofenv is not None


class _NativeBridge:
    """ofenv.NativeEnv with the same decode shape as the Node bridge."""

    def __init__(self):
        if _ofenv is None:
            raise RuntimeError("ofenv not built; pip install ./rust/ofenv")
        self._env = _ofenv.NativeEnv()
        self.width = 0
        self.height = 0
        self.terrain: np.ndarray | None = None
        self.me = -1

    def reset(self, map_name="Onion", seed="0", bots=100, **_kw) -> dict:
        obs = self._env.reset(map_name, seed, bots)
        self.width, self.height = self._env.width, self._env.height
        terr = gzip.decompress(base64.b64decode(self._env.terrain))
        self.terrain = np.frombuffer(terr, dtype=np.uint8).reshape(
            self.height, self.width
        )
        return self._decode(dict(obs))

    def step(self, intents: list[dict], ticks: int = 10) -> dict:
        return self._decode(self._env.step(intents, ticks))

    def _decode(self, obs: dict) -> dict:
        obs = dict(obs)
        state = np.frombuffer(obs.pop("tiles_raw"), dtype="<u2").reshape(
            self.height, self.width
        )
        obs["owners"] = state & OWNER_MASK
        obs["fallout"] = (state >> FALLOUT_BIT) & 1
        self.me = obs["me"]
        return obs

    def save_record(self, path: str) -> dict:
        raise NotImplementedError("native env save_record not implemented yet")

    def close(self) -> None:
        pass


class OpenFrontEnv:
    def __init__(self):
        if USE_NATIVE and HAVE_NATIVE_ENV:
            self._impl = _NativeBridge()
            return
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
        self._impl = None

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
        if self._impl is not None:
            return self._impl.reset(
                map_name=map_name, seed=seed, bots=bots, difficulty=difficulty, nations=nations
            )
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
        if self._impl is not None:
            return self._impl.step(intents, ticks)
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
        if self._impl is not None:
            return self._impl.save_record(path)
        return self._rpc({"op": "save_record", "path": path})

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            return
        try:
            self._rpc({"op": "close"})
        except Exception:
            pass
        self.proc.terminate()
