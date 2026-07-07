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

from rl.curriculum import (
    STAGES,
    W_DEATH,
    W_DELTA,
    W_STR,
    placement,
    placement_score,
    sample_episode,
    strengths,
    terminal_reward,
    timeweight,
)
from rl.env import OpenFrontEnv
from rl.obs import ObsBuilder
from rl.ppo_translate import IntentTranslator, my_tiles, spawn_randomly


class EnvWorker:
    def __init__(self, idx: int, stage_val, max_episode_ticks: int, decision_ticks: int):
        self.idx = idx
        self.stage_val = stage_val  # mp.Value: current curriculum stage
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
        self.stage = int(self.stage_val.value)
        self.dt = STAGES[self.stage].decision_ticks
        self.map_name, bots, difficulty, nations, self.rehearsal = sample_episode(
            self.stage, self.rng
        )
        # The episode starts inside the spawn phase: the policy's first
        # decision is WHERE to spawn (spawn action + tile-region head).
        self.obs = self.env.reset(
            self.map_name,
            seed=f"w{self.idx}-ep{self.episode}",
            bots=bots,
            difficulty=difficulty,
            nations=nations,
        )
        self.builder.start_game(self.env.terrain)
        self.translator = IntentTranslator(self.env, self.builder)
        self.land_total = max(1, int(((self.env.terrain >> 7) & 1).sum()))
        self.prev_strength = 0.0
        self.spawn_steps = 0
        self.ep_reward = 0.0
        self.ep_len = 0
        self.episode += 1

    def prepare(self) -> dict:
        return self.builder.prepare(self.obs)

    def apply(self, choice: dict) -> tuple[float, bool, dict | None]:
        """Translate + step. Auto-resets on done; returns (reward, done, ep_info)."""
        intents = self.translator.translate(choice, self.obs)
        self.obs = self.env.step(intents, ticks=self.dt)

        if self.obs["spawnPhase"]:
            # Masked spawn picks should always land; if the phase somehow
            # stalls (snap failure), fall back to a random spawn.
            self.spawn_steps += 1
            if self.spawn_steps >= 8:
                self.obs = spawn_randomly(self.env, self.rng)
            else:
                self.ep_len += 1
                return 0.0, False, None

        tiles = my_tiles(self.obs)
        mine = strengths(self.obs["entities"], self.land_total).get(self.obs["me"], 0.0)
        tw = timeweight(self.obs["tick"])
        reward = W_STR * mine * tw + W_DELTA * (mine - self.prev_strength)
        self.prev_strength = mine

        done = False
        won = False
        if not self.obs["alive"]:
            reward -= W_DEATH
            done = True
        elif self.obs["winner"] is not None:
            # Engine emits ["player", clientID] for human wins (clientID,
            # not username) and ["nation", name] for nation wins.
            w = self.obs["winner"]
            won = isinstance(w, list) and len(w) > 1 and w[1] == "AGENTRL1"
            done = True
        elif self.obs["tick"] >= self.max_episode_ticks:
            done = True

        info = None
        if done:
            place, n = placement(
                self.obs["entities"], self.obs["me"], self.obs["alive"], self.land_total
            )
            reward += terminal_reward(place, won)
            self.ep_reward += reward
            self.ep_len += 1
            info = {
                "reward": self.ep_reward,
                "length": self.ep_len,
                "final_tiles": tiles,
                "final_tick": self.obs["tick"],
                "place": place,
                "n_players": n,
                "score": placement_score(place, n),
                "won": won,
                "stage": self.stage,
                "map": self.map_name,
                "rehearsal": self.rehearsal,
            }
            self.reset_episode()
        else:
            self.ep_reward += reward
            self.ep_len += 1
        return reward, done, info


def _child(idx: int, stage_val, max_ticks: int, decision_ticks: int, conn) -> None:
    try:
        worker = EnvWorker(idx, stage_val, max_ticks, decision_ticks)
        sent_episode = -1

        def pack(raw: dict, result) -> dict:
            # terrain_static is constant per episode: ship it once, the
            # parent re-attaches it (dynamic fallout ships every step as
            # packed bits - it must NOT ride inside the cached tensor).
            nonlocal sent_episode
            msg = {"raw": raw, "result": result, "episode": worker.episode}
            if worker.episode != sent_episode:
                sent_episode = worker.episode
            else:
                msg["raw"] = {k: v for k, v in raw.items() if k != "terrain_static"}
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
    def __init__(self, n: int, start_stage: int, max_episode_ticks: int, decision_ticks: int):
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        self.stage_val = ctx.Value("i", start_stage)
        self.pipes = []
        self.procs = []
        self.terrains: list[np.ndarray | None] = [None] * n
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_child,
                args=(i, self.stage_val, max_episode_ticks, decision_ticks, child),
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
        if "terrain_static" in raw:
            self.terrains[i] = raw["terrain_static"]
        else:
            raw["terrain_static"] = self.terrains[i]
        return msg

    def obs(self) -> list[dict]:
        """Current raw observations (state t)."""
        return [m["raw"] for m in self._pending]

    def step(self, choices: list[dict]) -> list[tuple[float, bool, dict | None]]:
        """Send actions, collect (reward, done, info); next obs via obs()."""
        n = len(self.pipes)
        self.send_group(range(n), choices)
        return self.recv_group(range(n))

    # Group variants let the trainer pipeline: while one half of the envs
    # steps in their Node processes, the GPU acts for the other half.
    def obs_group(self, idxs) -> list[dict]:
        return [self._pending[i]["raw"] for i in idxs]

    def send_group(self, idxs, choices) -> None:
        for i, c in zip(idxs, choices):
            self.pipes[i].send(c)

    def recv_group(self, idxs) -> list[tuple[float, bool, dict | None]]:
        out = []
        for i in idxs:
            self._pending[i] = self._recv(i)
            out.append(self._pending[i]["result"])
        return out

    def set_stage(self, stage: int) -> None:
        """Workers pick this up at their next episode reset."""
        with self.stage_val.get_lock():
            self.stage_val.value = stage

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
