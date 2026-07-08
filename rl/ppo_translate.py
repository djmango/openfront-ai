"""Intent translation + episode helpers shared by ppo.py and vec.py."""

import numpy as np

from rl.obs import (
    ACTIONS,
    BUILD_TYPES,
    IMPASSABLE_MAGNITUDE,
    MAGNITUDE_MASK,
    NUKE_TYPES,
    REGION,
    ObsBuilder,
)


class IntentTranslator:
    """choice dict -> engine intent JSON. Region pointers snap to a tile
    inside the REGION x REGION block that passes the same cheap validity
    checks the engine runs at execution (ownership, passability, shore for
    ports) so region picks rarely become silently-discarded intents; the
    engine still validates the rest, and the env penalizes what slips
    through (see bridge/env.ts countWasted)."""

    SHORELINE_BIT = 6  # terrain byte layout, see GameMapImpl

    def __init__(self, env, builder: ObsBuilder):
        self.env = env
        self.builder = builder
        land = (env.terrain >> 7) & 1
        self.land = land[: builder.hr, : builder.wr]
        self.mag = (env.terrain & MAGNITUDE_MASK)[: builder.hr, : builder.wr]
        self.shore = ((env.terrain >> self.SHORELINE_BIT) & 1)[
            : builder.hr, : builder.wr
        ]
        self.passable = (self.land == 1) & (self.mag < IMPASSABLE_MAGNITUDE)
        self.gh = builder.hr // REGION
        self.gw = builder.wr // REGION
        self.rng = np.random.default_rng(0)

    def region_tile(self, region: int, valid: np.ndarray | None = None) -> int | None:
        # Region indices address the padded policy grid, whose width is
        # GW_MAX or the map's own grid width if larger (see encode_grids).
        from rl.curriculum import GW_MAX

        gy, gx = divmod(region, max(GW_MAX, self.gw))
        if gy >= self.gh or gx >= self.gw:
            return None  # padded region; masked, but stay safe
        sl = (
            slice(gy * REGION, (gy + 1) * REGION),
            slice(gx * REGION, (gx + 1) * REGION),
        )
        block = self.land[sl] if valid is None else valid[sl]
        ys, xs = np.nonzero(block)
        if len(ys) == 0:
            return None
        i = self.rng.integers(len(ys))
        y, x = gy * REGION + int(ys[i]), gx * REGION + int(xs[i])
        return y * self.env.width + x

    def spawn_tile(self, region: int, obs: dict) -> int | None:
        """Valid spawn tile inside the region: land, unowned, passable -
        mirrors SpawnExecution's validity so a masked pick always lands."""
        owners = obs["owners"][: self.builder.hr, : self.builder.wr]
        valid = self.passable & (owners == 0)
        return self.region_tile(region, valid)

    def _owners(self, obs: dict) -> np.ndarray:
        return obs["owners"][: self.builder.hr, : self.builder.wr]

    def boat_tile(self, region: int, obs: dict) -> int | None:
        """Boat destination inside the region: passable land not owned by
        self or an ally - mirrors TransportShipExecution's target checks
        (a boat to own/friendly territory is silently discarded)."""
        me = obs["me"]
        valid = self.passable & (self._owners(obs) != me)
        for a, b, _exp in obs["entities"]["alliances"]:
            ally = b if a == me else a if b == me else None
            if ally is not None:
                valid &= self._owners(obs) != ally
        return self.region_tile(region, valid)

    def build_tile(self, region: int, obs: dict, unit: str) -> int | None:
        """Structure site inside the region: own territory (engine's
        validStructureSpawnTiles requires ownership); ports additionally
        need own shore within the engine's port-spawn search radius.
        Warships target water (the engine spawns them at the closest own
        port sharing the target's water component)."""
        if unit == "Warship":
            return self.region_tile(region, self.land == 0)
        valid = self.passable & (self._owners(obs) == obs["me"])
        if unit == "Port":
            valid &= self.shore == 1
        return self.region_tile(region, valid)

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

    def region_center(self, region: int) -> tuple[int, int]:
        """(y, x) tile coordinates of the region's center."""
        from rl.curriculum import GW_MAX

        gy, gx = divmod(region, max(GW_MAX, self.gw))
        return gy * REGION + REGION // 2, gx * REGION + REGION // 2

    def nearest_own_unit(
        self, region: int, obs: dict, ids: list[int], types: set[str] | None = None
    ) -> dict | None:
        """Own unit closest to the region center, restricted to the given
        engine unit ids (from the bridge legality lists) and optionally to
        a type set. The tile head picks WHERE, this snaps to WHAT."""
        idset = set(ids)
        cands = [
            u
            for u in obs["entities"]["units"]
            if u["owner"] == obs["me"]
            and u.get("uid") in idset
            and (types is None or u["type"] in types)
        ]
        if not cands:
            return None
        cy, cx = self.region_center(region)
        return min(cands, key=lambda u: (u["y"] - cy) ** 2 + (u["x"] - cx) ** 2)

    def translate(self, choice: dict, obs: dict) -> list[dict]:
        name = ACTIONS[choice["action"]]
        legal = obs["legal"]["actions"]
        troops = legal.get("troops", 0)
        frac = min(1.0, max(0.01, float(choice.get("quantity_frac", 0.25))))

        if name == "noop":
            return []
        if name == "spawn":
            tile = self.spawn_tile(choice["tile_region"], obs)
            if tile is None:
                return []
            return [{"type": "spawn", "tile": tile}]
        if name == "expand":
            return [{"type": "attack", "targetID": None, "troops": int(troops * frac)}]
        if name == "attack":
            pid = self.slot_to_pid(choice["player_slot"], obs)
            if pid is None:
                return []
            return [{"type": "attack", "targetID": pid, "troops": int(troops * frac)}]
        if name == "boat":
            tile = self.boat_tile(choice["tile_region"], obs)
            if tile is None:
                return []
            return [{"type": "boat", "dst": tile, "troops": int(troops * frac)}]
        if name == "build":
            unit = BUILD_TYPES[choice["build_type"]]
            tile = self.build_tile(choice["tile_region"], obs, unit)
            if tile is None:
                return []
            return [{"type": "build_unit", "unit": unit, "tile": tile}]
        if name == "launch_nuke":
            tile = self.region_tile(choice["tile_region"], self.passable)
            if tile is None:
                return []
            unit, up = NUKE_TYPES[choice["nuke_type"]]
            intent = {"type": "build_unit", "unit": unit, "tile": tile}
            if up is not None:  # MIRV has no arc
                intent["rocketDirectionUp"] = up
            return [intent]
        if name == "retreat":
            # Targeted: cancel the newest non-retreating attack on the
            # chosen player (slot 0 = terra-nullius expand); fall back to
            # the newest attack overall.
            attacks = legal.get("attacks", [])
            if not attacks:
                return []
            slot = choice.get("player_slot", -1)
            lut = self.builder.lut
            aid = attacks[-1]
            matches = [
                a
                for a in obs["entities"]["attacks"]
                if a["from"] == obs["me"]
                and not a.get("retreating")
                and (int(lut[a["to"]]) if a["to"] else 0) == slot
            ]
            if matches:
                aid = matches[-1]["aid"]
            return [{"type": "cancel_attack", "attackID": aid}]
        if name == "upgrade_structure":
            u = self.nearest_own_unit(
                choice["tile_region"], obs, legal.get("upgradable", [])
            )
            if u is None:
                return []
            return [{"type": "upgrade_structure", "unit": u["type"], "unitId": u["uid"]}]
        if name == "move_warship":
            ids = legal.get("warships", [])
            tile = self.region_tile(choice["tile_region"], self.land == 0)
            if not ids or tile is None:
                return []
            return [{"type": "move_warship", "unitIds": ids, "tile": tile}]
        if name == "cancel_boat":
            u = self.nearest_own_unit(
                choice["tile_region"], obs, legal.get("boats", [])
            )
            if u is None:
                return []
            return [{"type": "cancel_boat", "unitID": u["uid"]}]
        if name == "delete_unit":
            u = self.nearest_own_unit(
                choice["tile_region"], obs, legal.get("deletable", [])
            )
            if u is None:
                return []
            return [{"type": "delete_unit", "unitId": u["uid"]}]

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
        if name == "embargo_stop":
            return [{"type": "embargo", "targetID": pid, "action": "stop"}]
        if name == "target_player":
            return [{"type": "targetPlayer", "target": pid}]
        if name == "alliance_extension":
            return [{"type": "allianceExtension", "recipient": pid}]
        return []


def my_tiles(obs: dict) -> int:
    for p in obs["entities"]["players"]:
        if p["id"] == obs["me"]:
            return p["tiles"]
    return 0


def spawn_randomly(env, rng: np.random.Generator) -> dict:
    """Fallback spawner (random land tile). v4 spawns via the policy's
    spawn action; this remains as a safety valve if the spawn phase stalls."""
    land = (env.terrain >> 7) & 1
    ys, xs = np.nonzero(land)
    obs = None
    while True:
        i = rng.integers(len(ys))
        tile = int(ys[i]) * env.width + int(xs[i])
        obs = env.step([{"type": "spawn", "tile": tile}], ticks=10)
        if not obs["spawnPhase"]:
            return obs


