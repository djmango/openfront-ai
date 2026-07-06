"""Vectorized envs: N bridge processes stepped in parallel threads.

The Node engine does the heavy lifting in its own process per env, so
threads are enough on the Python side; the frozen-AE encode and policy
forward run batched across envs on the GPU.
"""

from concurrent.futures import ThreadPoolExecutor

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


class VecEnv:
    def __init__(self, n: int, map_name: str, max_episode_ticks: int, decision_ticks: int):
        self.pool = ThreadPoolExecutor(max_workers=n)
        self.workers = list(
            self.pool.map(
                lambda i: EnvWorker(i, map_name, max_episode_ticks, decision_ticks),
                range(n),
            )
        )

    def prepare(self) -> list[dict]:
        return list(self.pool.map(lambda w: w.prepare(), self.workers))

    def apply(self, choices: list[dict]) -> list[tuple[float, bool, dict | None]]:
        return list(
            self.pool.map(lambda wc: wc[0].apply(wc[1]), zip(self.workers, choices))
        )

    def close(self) -> None:
        for w in self.workers:
            w.env.close()
        self.pool.shutdown()
