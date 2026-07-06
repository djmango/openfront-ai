"""Intent translation + episode helpers shared by ppo.py and vec.py."""

import numpy as np

from rl.env import OpenFrontEnv
from rl.obs import ACTIONS, BUILD_TYPES, NUKE_TYPES, ObsBuilder
from rl.policy import QUANTITY_FRACS


class IntentTranslator:
    """choice dict -> engine intent JSON. Region pointers snap to a land
    tile inside the 16x16 region (engine validates the rest)."""

    def __init__(self, env: OpenFrontEnv, builder: ObsBuilder):
        self.env = env
        self.builder = builder
        land = (env.terrain >> 7) & 1
        self.land = land[: builder.h16, : builder.w16]
        self.gh = builder.h16 // 16
        self.gw = builder.w16 // 16
        self.rng = np.random.default_rng(0)

    def region_tile(self, region: int) -> int | None:
        # Region indices address the padded GW_MAX-wide policy grid.
        from rl.curriculum import GW_MAX

        gy, gx = divmod(region, GW_MAX)
        if gy >= self.gh or gx >= self.gw:
            return None  # padded region; masked, but stay safe
        block = self.land[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16]
        ys, xs = np.nonzero(block)
        if len(ys) == 0:
            return None
        i = self.rng.integers(len(ys))
        y, x = gy * 16 + int(ys[i]), gx * 16 + int(xs[i])
        return y * self.env.width + x

    def slot_to_pid(self, slot: int, obs: dict) -> str | None:
        """Slot -> persistent player ID (intents reference players by pid)."""
        lut = self.builder.lut
        small_ids = np.nonzero(lut == slot)[0]
        if not len(small_ids):
            return None
        small = int(small_ids[0])
        for p in obs["entities"]["players"]:
            if p["id"] == small:
                return p["pid"]
        return None

    def translate(self, choice: dict, obs: dict) -> list[dict]:
        name = ACTIONS[choice["action"]]
        legal = obs["legal"]["actions"]
        troops = legal.get("troops", 0)
        frac = QUANTITY_FRACS[choice.get("quantity", 2)]

        if name == "noop":
            return []
        if name == "expand":
            return [{"type": "attack", "targetID": None, "troops": int(troops * frac)}]
        if name == "attack":
            pid = self.slot_to_pid(choice["player_slot"], obs)
            if pid is None:
                return []
            return [{"type": "attack", "targetID": pid, "troops": int(troops * frac)}]
        if name == "boat":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "boat", "dst": tile, "troops": int(troops * frac)}]
        if name == "build":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "build_unit", "unit": BUILD_TYPES[choice["build_type"]], "tile": tile}]
        if name == "launch_nuke":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "build_unit", "unit": NUKE_TYPES[choice["nuke_type"]], "tile": tile}]
        if name == "retreat":
            attacks = legal.get("attacks", [])
            return [{"type": "cancel_attack", "attackID": attacks[-1]}] if attacks else []

        pid = self.slot_to_pid(choice.get("player_slot", -1), obs)
        if pid is None:
            return []
        if name == "alliance_request":
            return [{"type": "allianceRequest", "recipient": pid}]
        if name == "alliance_reject":
            return [{"type": "allianceReject", "requestor": pid}]
        if name == "break_alliance":
            return [{"type": "breakAlliance", "recipient": pid}]
        if name == "donate_gold":
            gold = int(float(legal.get("gold", 0)) * frac)
            return [{"type": "donate_gold", "recipient": pid, "gold": gold}]
        if name == "donate_troops":
            return [{"type": "donate_troops", "recipient": pid, "troops": int(troops * frac)}]
        if name == "embargo":
            return [{"type": "embargo", "targetID": pid, "action": "start"}]
        return []


def my_tiles(obs: dict) -> int:
    for p in obs["entities"]["players"]:
        if p["id"] == obs["me"]:
            return p["tiles"]
    return 0


def spawn_randomly(env: OpenFrontEnv, rng: np.random.Generator) -> dict:
    land = (env.terrain >> 7) & 1
    ys, xs = np.nonzero(land)
    obs = None
    while True:
        i = rng.integers(len(ys))
        tile = int(ys[i]) * env.width + int(xs[i])
        obs = env.step([{"type": "spawn", "tile": tile}], ticks=10)
        if not obs["spawnPhase"]:
            return obs


