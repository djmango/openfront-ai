"""Vectorized envs: one OS process per env (bridge + featurization), so
JSON decode, gzip, and numpy work all escape the GIL. The main process
only does batched AE encode + policy forward on the GPU.

Pipe payloads stay small: static terrain ships once per episode and is
cached main-side; per-step traffic is the uint8 owner grid plus small
arrays (~300KB/env).
"""

import multiprocessing as mp
import sys

import numpy as np

from rl.env import OpenFrontEnv
from rl.obs import ObsBuilder
from rl.ppo_translate import IntentTranslator, my_tiles, spawn_randomly


class EnvWorker:
    def __init__(self, idx: int, map_name: str, max_episode_ticks: int, decision_ticks: int):
        self.idx = idx
        self.map_name = map_name
        self.max_episode_ticks = max_episode_ticks
        self.decision_ticks = decision_ticks
        self.env = OpenFrontEnv()
        self.builder = ObsBuilder()  # prepare() only; encode is centralized
        self.rng = np.random.default_rng(1000 + idx)
        self.episode = 0
        self.ep_reward = 0.0
        self.ep_len = 0
        self.obs: dict = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        self.obs = self.env.reset(self.map_name, seed=f"w{self.idx}-ep{self.episode}")
        self.builder.start_game(self.env.terrain)
        self.obs = spawn_randomly(self.env, self.rng)
        self.translator = IntentTranslator(self.env, self.builder)
        self.land_total = max(1, int(((self.env.terrain >> 7) & 1).sum()))
        self.prev_tiles = my_tiles(self.obs)
        self.ep_reward = 0.0
        self.ep_len = 0
        self.episode += 1

    def prepare(self) -> dict:
        return self.builder.prepare(self.obs)

    def apply(self, choice: dict) -> tuple[float, bool, dict | None]:
        """Translate + step. Auto-resets on done; returns (reward, done, ep_info)."""
        intents = self.translator.translate(choice, self.obs)
        self.obs = self.env.step(intents, ticks=self.decision_ticks)

        tiles = my_tiles(self.obs)
        reward = (tiles - self.prev_tiles) / self.land_total * 10.0
        self.prev_tiles = tiles
        done = False
        if not self.obs["alive"]:
            reward -= 5.0
            done = True
        elif self.obs["winner"] is not None:
            w = self.obs["winner"]
            won = isinstance(w, list) and len(w) > 1 and w[1] == "Agent"
            reward += 10.0 if won else -2.0
            done = True
        elif self.obs["tick"] >= self.max_episode_ticks:
            done = True

        self.ep_reward += reward
        self.ep_len += 1
        info = None
        if done:
            info = {
                "reward": self.ep_reward,
                "length": self.ep_len,
                "final_tiles": tiles,
                "final_tick": self.obs["tick"],
            }
            self.reset_episode()
        return reward, done, info


def _child(idx: int, map_name: str, max_ticks: int, decision_ticks: int, conn) -> None:
    try:
        worker = EnvWorker(idx, map_name, max_ticks, decision_ticks)
        sent_episode = -1

        def pack(raw: dict, result) -> dict:
            nonlocal sent_episode
            msg = {"raw": raw, "result": result, "episode": worker.episode}
            if worker.episode != sent_episode:
                sent_episode = worker.episode
            else:
                msg["raw"] = {k: v for k, v in raw.items() if k != "terrain"}
            return msg

        conn.send(pack(worker.prepare(), None))
        while True:
            choice = conn.recv()
            if choice is None:
                break
            result = worker.apply(choice)
            conn.send(pack(worker.prepare(), result))
        worker.env.close()
    except Exception as e:
        conn.send({"error": repr(e)})
        raise
    finally:
        conn.close()


class VecEnv:
    def __init__(self, n: int, map_name: str, max_episode_ticks: int, decision_ticks: int):
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        self.pipes = []
        self.procs = []
        self.terrains: list[np.ndarray | None] = [None] * n
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_child,
                args=(i, map_name, max_episode_ticks, decision_ticks, child),
                daemon=True,
            )
            p.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(p)
        self._pending: list[dict] = [self._recv(i) for i in range(n)]

    def _recv(self, i: int) -> dict:
        msg = self.pipes[i].recv()
        if "error" in msg:
            raise RuntimeError(f"env worker {i}: {msg['error']}")
        raw = msg["raw"]
        if "terrain" in raw:
            self.terrains[i] = raw["terrain"]
        else:
            raw["terrain"] = self.terrains[i]
        return msg

    def obs(self) -> list[dict]:
        """Current raw observations (state t)."""
        return [m["raw"] for m in self._pending]

    def step(self, choices: list[dict]) -> list[tuple[float, bool, dict | None]]:
        """Send actions, collect (reward, done, info); next obs via obs()."""
        for pipe, c in zip(self.pipes, choices):
            pipe.send(c)
        self._pending = [self._recv(i) for i in range(len(self.pipes))]
        return [m["result"] for m in self._pending]

    def close(self) -> None:
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
