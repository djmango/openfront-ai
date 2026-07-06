"""Observation builder: frozen spatial-AE latent + exact bypass features.

Per DESIGN.md: the AE compresses the map; everything small reaches the
policy raw. Output tensors per step:

  grid:     (C_GRID, H/16, W/16) — frozen z_grid (64ch) + ego ownership
            planes (3) + transient unit planes (8, exact) = 75ch
  players:  (MAX_SLOTS, P_FEAT) per-player bypass features (exact)
  pmask:    (MAX_SLOTS,) slot exists
  scalars:  (N_SCALARS,) own/global state
  legal_*:  action masks (exact, from the bridge)
"""

from pathlib import Path

import numpy as np
import torch

from ae.model_v3 import MAX_SLOTS, NUM_STATIC, SpatialAE
from ae.units import STATIC_INDICES, UNIT_CLASS_INDEX

IS_LAND_BIT = 7
MAGNITUDE_MASK = 0x1F
MAGNITUDE_NORM = 31.0

# Transient planes (exact bypass, rendered at 1/16): warship, transport,
# transport-destination, trade ship, nuke, nuke-impact-point, samlock-nuke,
# construction.
N_TRANSIENT = 8
C_GRID = 64 + 3 + N_TRANSIENT

P_FEAT = 12
N_SCALARS = 8

ACTIONS = [
    "noop",
    "attack",  # player target + quantity
    "expand",  # attack with null target (neutral land)
    "boat",  # tile target + quantity
    "build",  # build-type + tile target
    "launch_nuke",  # nuke-type + tile target
    "alliance_request",  # player target
    "alliance_reject",  # player target
    "break_alliance",  # player target
    "donate_gold",  # player target
    "donate_troops",  # player target
    "embargo",  # player target
    "retreat",  # cancel newest attack
]
N_ACTIONS = len(ACTIONS)

BUILD_TYPES = ["City", "Port", "Defense Post", "Missile Silo", "SAM Launcher", "Factory"]
NUKE_TYPES = ["Atom Bomb", "Hydrogen Bomb", "MIRV"]


def log_norm(x: float) -> float:
    return float(np.log10(1.0 + max(0.0, x)) / 8.0)


class ObsBuilder:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.ae = SpatialAE(latent_c=ckpt["args"]["latent_c"]).to(device)
        self.ae.load_state_dict(ckpt["model_state_dict"])
        self.ae.eval()
        self.device = device
        self.lut: np.ndarray | None = None
        self._terr_t: torch.Tensor | None = None

    def start_game(self, terrain: np.ndarray) -> None:
        """Call at reset; caches terrain tensors and clears the slot LUT."""
        self.lut = None
        h, w = terrain.shape
        h16, w16 = h - h % 16, w - w % 16
        self.h16, self.w16 = h16, w16
        t = terrain[:h16, :w16]
        self._terr_np = t
        self._land_mag = np.stack(
            [
                ((t >> IS_LAND_BIT) & 1).astype(np.float32),
                ((t & MAGNITUDE_MASK) / MAGNITUDE_NORM).astype(np.float32),
            ]
        )

    def _slot_lut(self, players: list[dict]) -> np.ndarray:
        if self.lut is None:
            ids = sorted(p["id"] for p in players)
            lut = np.zeros(4096, dtype=np.int64)
            for slot, sid in enumerate(ids, start=1):
                lut[sid] = min(slot, MAX_SLOTS - 1)
            self.lut = lut
        return self.lut

    def build(self, obs: dict) -> dict:
        ents = obs["entities"]
        lut = self._slot_lut(ents["players"])
        me_slot = int(lut[obs["me"]]) if obs["me"] >= 0 else 0
        h16, w16 = self.h16, self.w16
        gh, gw = h16 // 16, w16 // 16

        owners = lut[obs["owners"][:h16, :w16]]
        fallout = obs["fallout"][:h16, :w16].astype(np.float32)
        terrain = np.concatenate([self._land_mag, fallout[None]])

        static = np.zeros((NUM_STATIC, gh, gw), dtype=np.float32)
        transient = np.zeros((N_TRANSIENT, gh, gw), dtype=np.float32)
        static_pos = {i: k for k, i in enumerate(STATIC_INDICES)}
        for u in ents["units"]:
            ci = UNIT_CLASS_INDEX.get(u["type"])
            if ci is None:
                continue
            gy, gx = u["y"] // 16, u["x"] // 16
            if not (0 <= gy < gh and 0 <= gx < gw):
                continue
            if ci in static_pos and not u["constructing"]:
                static[static_pos[ci], gy, gx] = 1.0
            t = u["type"]
            if t == "Warship":
                transient[0, gy, gx] = 1.0
            elif t == "Transport":
                transient[1, gy, gx] = 1.0
                if u["tx"] is not None:
                    transient[2, u["ty"] // 16, u["tx"] // 16] = 1.0
            elif t == "Trade Ship":
                transient[3, gy, gx] = 1.0
            elif t in ("Atom Bomb", "Hydrogen Bomb", "MIRV"):
                transient[4, gy, gx] = 1.0
                if u["tx"] is not None:
                    transient[5, u["ty"] // 16, u["tx"] // 16] = 1.0
                if u["samLock"]:
                    transient[6, gy, gx] = 1.0
            if u["constructing"]:
                transient[7, gy, gx] = 1.0

        with torch.no_grad():
            z = self.ae.encode(
                torch.from_numpy(owners[None]).to(self.device),
                torch.from_numpy(terrain[None]).to(self.device),
                torch.from_numpy(static[None]).to(self.device),
            )[0].cpu().numpy()

        # Ego ownership planes at 1/16 (fraction of region owned).
        allies = self._ally_slots(ents, me_slot)
        own = (owners == me_slot).astype(np.float32)
        ally = np.isin(owners, list(allies)).astype(np.float32) if allies else np.zeros_like(own)
        enemy = ((owners > 0) & (owners != me_slot) & ~np.isin(owners, list(allies))).astype(np.float32)
        ego = np.stack(
            [
                own.reshape(gh, 16, gw, 16).mean(axis=(1, 3)),
                ally.reshape(gh, 16, gw, 16).mean(axis=(1, 3)),
                enemy.reshape(gh, 16, gw, 16).mean(axis=(1, 3)),
            ]
        )

        grid = np.concatenate([z, ego, transient]).astype(np.float32)
        players, pmask = self._player_feats(ents, lut, me_slot)
        scalars = self._scalars(obs, ents, me_slot)
        masks = self._masks(obs, lut)
        return {
            "grid": grid,
            "players": players,
            "pmask": pmask,
            "scalars": scalars,
            "me_slot": me_slot,
            **masks,
        }

    def _ally_slots(self, ents: dict, me_slot: int) -> set[int]:
        assert self.lut is not None
        out = set()
        for a, b, _exp in ents["alliances"]:
            sa, sb = int(self.lut[a]), int(self.lut[b])
            if sa == me_slot:
                out.add(sb)
            elif sb == me_slot:
                out.add(sa)
        return out

    def _player_feats(
        self, ents: dict, lut: np.ndarray, me_slot: int
    ) -> tuple[np.ndarray, np.ndarray]:
        feats = np.zeros((MAX_SLOTS, P_FEAT), dtype=np.float32)
        mask = np.zeros(MAX_SLOTS, dtype=np.float32)
        allies = self._ally_slots(ents, me_slot)
        atk_between: dict[int, float] = {}
        for a in ents["attacks"]:
            sa, sb = int(lut[a["from"]]), int(lut[a["to"]]) if a["to"] else 0
            if sa == me_slot:
                atk_between[sb] = atk_between.get(sb, 0.0) + a["troops"]
            elif sb == me_slot:
                atk_between[sa] = atk_between.get(sa, 0.0) - a["troops"]
        for p in ents["players"]:
            slot = int(lut[p["id"]])
            if slot <= 0:
                continue
            mask[slot] = 1.0
            feats[slot] = [
                1.0 if p["alive"] else 0.0,
                log_norm(p["troops"]),
                log_norm(float(p["gold"])),
                log_norm(p["tiles"]),
                1.0 if p["traitor"] else 0.0,
                1.0 if slot in allies else 0.0,
                1.0 if me_slot in [int(lut[e]) for e in p["embargoes"]] else 0.0,
                1.0 if slot == me_slot else 0.0,
                log_norm(abs(atk_between.get(slot, 0.0))),
                1.0 if atk_between.get(slot, 0.0) > 0 else 0.0,
                len(p["reqsIn"]) / 4.0,
                len(p["reqsOut"]) / 4.0,
            ]
        return feats, mask

    def _scalars(self, obs: dict, ents: dict, me_slot: int) -> np.ndarray:
        legal = obs["legal"]["actions"]
        n_alive = sum(p["alive"] for p in ents["players"])
        return np.array(
            [
                obs["tick"] / 15000.0,
                1.0 if obs["spawnPhase"] else 0.0,
                1.0 if obs["alive"] else 0.0,
                log_norm(legal.get("troops", 0)),
                log_norm(float(legal.get("gold", 0))),
                n_alive / 128.0,
                len(legal.get("attacks", [])) / 8.0,
                me_slot / MAX_SLOTS,
            ],
            dtype=np.float32,
        )

    def _masks(self, obs: dict, lut: np.ndarray) -> dict:
        legal = obs["legal"]["actions"]
        act = np.zeros(N_ACTIONS, dtype=np.float32)
        ptarget = np.zeros((N_ACTIONS, MAX_SLOTS), dtype=np.float32)

        def fill(action: str, ids: list[int]) -> None:
            if ids:
                act[ACTIONS.index(action)] = 1.0
                for i in ids:
                    ptarget[ACTIONS.index(action), int(lut[i])] = 1.0

        act[ACTIONS.index("noop")] = 1.0
        if obs["alive"] and legal:
            fill("attack", legal.get("attackable", []))
            fill("alliance_request", legal.get("allianceRequestable", []))
            fill("alliance_reject", legal.get("allianceRejectable", []))
            fill("break_alliance", legal.get("breakable", []))
            fill("donate_gold", legal.get("donatableGold", []))
            fill("donate_troops", legal.get("donatableTroops", []))
            fill("embargo", legal.get("embargoable", []))
            act[ACTIONS.index("expand")] = 1.0
            act[ACTIONS.index("boat")] = 1.0 if legal.get("troops", 0) > 100 else 0.0
            build_ok = [t for t in BUILD_TYPES if t in legal.get("buildableTypes", [])]
            act[ACTIONS.index("build")] = 1.0 if build_ok else 0.0
            nukes_ok = [t for t in NUKE_TYPES if t in legal.get("buildableTypes", [])]
            act[ACTIONS.index("launch_nuke")] = (
                1.0 if nukes_ok and legal.get("hasSilo") else 0.0
            )
            act[ACTIONS.index("retreat")] = 1.0 if legal.get("attacks") else 0.0
        build_mask = np.array(
            [1.0 if t in obs["legal"]["actions"].get("buildableTypes", []) else 0.0 for t in BUILD_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(BUILD_TYPES), dtype=np.float32)
        nuke_mask = np.array(
            [1.0 if t in obs["legal"]["actions"].get("buildableTypes", []) else 0.0 for t in NUKE_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(NUKE_TYPES), dtype=np.float32)
        return {
            "legal_actions": act,
            "legal_ptarget": ptarget,
            "legal_build": build_mask,
            "legal_nuke": nuke_mask,
        }
