"""Observation builder: frozen spatial-AE latent + exact bypass features.

Per DESIGN.md: the AE compresses the map; everything small reaches the
policy raw. v4 layout (d8c32 encoder, everything at 1/8 resolution):

  grid:     (C_GRID, H/8, W/8) - frozen z_grid (32ch) + ego ownership
            planes (3) + defense-bonus fraction (1, v7, exact bypass) +
            transient unit planes (N_TRANSIENT, exact)
  local:    (N_LOCAL, LOCAL, LOCAL) raw owner-map crop centered on the
            agent's territory - exact borders where the agent acts most
            (own/ally/enemy/land/defense-bonus, v7)
  players:  (MAX_SLOTS, P_FEAT) per-player bypass features (exact)
  pmask:    (MAX_SLOTS,) slot exists
  scalars:  (N_SCALARS,) own/global state
  legal_*:  action masks (exact, from the bridge), including legal_tile
            (H/8, W/8) - all-ones normally, valid spawn regions during
            the spawn phase

Split of labor (v4): prepare() is cheap numpy over SMALL state (entity
planes, player features, masks). All full-resolution work - slot classing,
ego pooling, fallout unpacking, the AE encode, the local crop - happens
batched on the GPU in encode_grids(). Static terrain (land+magnitude)
ships once per episode; dynamic fallout ships every step as packed bits
(the old design baked fallout into the cached terrain tensor, so the AE
saw frozen fallout after an episode's first step).
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ae.model_v3 import MAX_SLOTS, NUM_STATIC, SpatialAE
from ae.units import STATIC_INDICES, UNIT_CLASS_INDEX

IS_LAND_BIT = 7
MAGNITUDE_MASK = 0x1F
MAGNITUDE_NORM = 31.0
IMPASSABLE_MAGNITUDE = 31  # land tiles at this magnitude cannot be owned

# v4 spatial resolution: one latent cell / tile-pointer region per 8x8 tiles.
REGION = 8
LATENT_C = 32  # d8c32; asserted against the checkpoint in load_ae()

# Transient planes (exact bypass, rendered at 1/REGION). v7: every *_BASE
# below is an own/ally/enemy triplet (index BASE + {0,1,2}); a unit's class
# comes straight from `clut[owner_slot]` (1 own, 2 ally, 3 enemy) - same
# ego-relative classing the AE-bypass ownership planes already use. Values
# are intensities (log_norm troops/health fraction/level fraction) where a
# natural magnitude exists, else flat presence. The two TR_ATTACK_* channels
# are shared across owners - the player-feature stream carries who/whom.
TR_WARSHIP = 0  # position; value = health_fraction
TR_TRANSPORT = 3  # position; value = log_norm(troops carried)
TR_TRANSPORT_DEST = 6  # binary
TR_TRADE = 9  # position; binary
TR_TRADE_DEST = 12  # binary (previously missing)
TR_NUKE = 15  # Atom/Hydrogen/MIRV position; binary
TR_NUKE_IMPACT = 18  # binary
TR_NUKE_SAMLOCK = 21  # nuke position, targeted by a SAM; binary
TR_CONSTRUCTION = 24  # any structure under construction; binary
TR_SAM_MISSILE = 27  # position; binary
TR_SAM_MISSILE_IMPACT = 30  # binary
TR_MIRV_WARHEAD = 33  # position; binary (independently-targetable sub-warheads)
TR_MIRV_WARHEAD_IMPACT = 36  # binary
TR_TRAIN = 39  # position; value = log_norm(cargo troops)
TR_SILO_COOLDOWN = 42  # Missile Silo, currently in cooldown; binary
TR_SAM_COOLDOWN = 45  # SAM Launcher, currently in cooldown; binary
TR_STATION = 48  # City/Port/Factory with an active connected TrainStation
TR_ATTACK_SRC = 51  # all active attacks' source tile; value = log_norm(troops)
TR_ATTACK_RETREAT = 52  # same, retreating attacks only (overlays TR_ATTACK_SRC)
N_TRANSIENT = 53
# +1 region-averaged defense-bonus fraction channel, appended right after
# the ego ownership planes (same bypass treatment - never touches the
# frozen AE's input, which stays land+magnitude+fallout only).
N_DEFENSE_BONUS = 1
C_GRID = LATENT_C + 3 + N_DEFENSE_BONUS + N_TRANSIENT

# Raw local owner-map crop (exact-borders bypass): own/ally/enemy/land/
# defense-bonus (v7) planes over a LOCAL x LOCAL tile window centered on
# own territory.
LOCAL = 64
N_LOCAL = 5

# v7: +9 columns (global attack pressure out/in, retreating fraction,
# alliance expiry countdown, target-marked flag, troop/gold income,
# doomsday in-clock flag + ticks) - see ObsBuilder._player_feats.
P_FEAT = 21
# v7: +3 scalars (own troop/gold income, doomsday-clock-enabled flag) -
# see ObsBuilder._scalars.
N_SCALARS = 11

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
    "retreat",  # player target: cancel attack on that player (slot 0 = expand)
    "spawn",  # tile target; only legal (and forced) during the spawn phase
    # v6 additions (appended so v5 action indices stay stable)
    "upgrade_structure",  # tile target -> nearest own upgradable structure
    "move_warship",  # tile target: move all own warships to that water region
    "cancel_boat",  # tile target -> nearest own in-flight transport
    "delete_unit",  # tile target -> nearest own deletable structure
    "embargo_stop",  # player target: lift an active embargo
    "target_player",  # player target: mark for allies/bots
    "alliance_extension",  # player target: renew an expiring alliance
]
N_ACTIONS = len(ACTIONS)

BUILD_TYPES = [
    "City", "Port", "Defense Post", "Missile Silo", "SAM Launcher", "Factory",
    "Warship",
]
# 5-way nuke head: (engine unit, rocketDirectionUp). The arc flag flips the
# parabolic flight path (SAM evasion); MIRV ignores it entirely
# (MirvExecution never receives the flag), so it gets a single row.
NUKE_TYPES = [
    ("Atom Bomb", True),
    ("Atom Bomb", False),
    ("Hydrogen Bomb", True),
    ("Hydrogen Bomb", False),
    ("MIRV", None),
]
NUKE_UNITS = ["Atom Bomb", "Hydrogen Bomb", "MIRV"]


def log_norm(x: float) -> float:
    return float(np.log10(1.0 + max(0.0, x)) / 8.0)


def load_ae(ckpt_path: str | Path, device: str = "cpu") -> SpatialAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    ae = SpatialAE(
        latent_c=a["latent_c"],
        terrain_cond=a.get("terrain_cond", False),
        upsample_decoder=a.get("upsample_decoder", False),
        latent_down=a.get("latent_down", 16),
    ).to(device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    if ae.latent_down != REGION or ae.latent_c != LATENT_C:
        raise ValueError(
            f"encoder {ckpt_path} is {ae.latent_c}ch @ 1/{ae.latent_down}; "
            f"v4 expects {LATENT_C}ch @ 1/{REGION} (ae_v31_d8c32)"
        )
    return ae


class ObsBuilder:
    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        device: str = "cpu",
        ae: SpatialAE | None = None,
    ):
        # ae=None with a ckpt loads a private copy; pass ae to share weights,
        # or leave both unset when encoding happens centrally (see prepare()).
        self.ae = ae if ae is not None else (load_ae(ckpt_path, device) if ckpt_path else None)
        self.device = device
        self.lut: np.ndarray | None = None

    def start_game(self, terrain: np.ndarray) -> None:
        """Call at reset; caches terrain tensors and clears the slot LUT."""
        self.lut = None
        h, w = terrain.shape
        hr, wr = h - h % REGION, w - w % REGION
        self.hr, self.wr = hr, wr
        t = terrain[:hr, :wr]
        self._terr_np = t
        self._land = ((t >> IS_LAND_BIT) & 1).astype(np.uint8)
        self._mag = (t & MAGNITUDE_MASK).astype(np.uint8)
        self._terrain_static = np.stack(
            [
                self._land.astype(np.float32),
                (self._mag / MAGNITUDE_NORM).astype(np.float32),
            ]
        )

    def _make_lut(self, players: list[dict]) -> np.ndarray:
        ids = sorted(p["id"] for p in players)
        lut = np.zeros(4096, dtype=np.int64)
        for slot, sid in enumerate(ids, start=1):
            lut[sid] = min(slot, MAX_SLOTS - 1)
        return lut

    def _slot_lut(self, players: list[dict], spawn_phase: bool = False) -> np.ndarray:
        # During the spawn phase the roster is still filling in (tribe bots
        # spawn over several ticks), so the LUT is rebuilt fresh each step
        # and only frozen on the first post-spawn observation - slot
        # identities stay stable for the whole playable episode.
        if spawn_phase:
            return self._make_lut(players) if self.lut is None else self.lut
        if self.lut is None:
            self.lut = self._make_lut(players)
        return self.lut

    def prepare(self, obs: dict) -> dict:
        """Numpy-only featurization of SMALL state (thread-safe). All
        full-resolution work happens in encode_grids() on the GPU."""
        ents = obs["entities"]
        spawn_phase = bool(obs.get("spawnPhase"))
        lut = self._slot_lut(ents["players"], spawn_phase)
        me_slot = int(lut[obs["me"]]) if obs["me"] >= 0 else 0
        hr, wr = self.hr, self.wr
        gh, gw = hr // REGION, wr // REGION

        # Fast paths for the BC cache (pre-slotted owners, pre-packed
        # fallout); the live env ships raw smallID owners + a 0/1 grid.
        if "owners_slots" in obs:
            owners = obs["owners_slots"]
        else:
            lut_u8 = lut.astype(np.uint8)
            owners = lut_u8[obs["owners"][:hr, :wr]]
        if "fallout_packed" in obs:
            fallout_packed = obs["fallout_packed"]
        else:
            fallout_packed = np.packbits(
                obs["fallout"][:hr, :wr].astype(np.uint8), axis=1
            )
        if "defense_bonus_packed" in obs:
            defense_bonus_packed = obs["defense_bonus_packed"]
        else:
            defense_bonus_packed = np.packbits(
                obs["defense_bonus"][:hr, :wr].astype(np.uint8), axis=1
            )

        # Slot -> ego class LUT (0 neutral/unowned, 1 own, 2 ally, 3 enemy);
        # the GPU expands it over the full-res owner grid. Computed before
        # the unit loop below so per-unit ownership (own/ally/enemy) reuses
        # this exact classing instead of a second ally-membership check.
        allies, ally_expiry = self._ally_slots(ents, me_slot, lut)
        clut = np.full(MAX_SLOTS, 3, dtype=np.uint8)
        clut[0] = 0
        for s in allies:
            clut[s] = 2
        if me_slot > 0:
            clut[me_slot] = 1

        static = np.zeros((NUM_STATIC, gh, gw), dtype=np.float32)
        transient = np.zeros((N_TRANSIENT, gh, gw), dtype=np.float32)
        static_pos = {i: k for k, i in enumerate(STATIC_INDICES)}
        for u in ents["units"]:
            ci = UNIT_CLASS_INDEX.get(u["type"])
            if ci is None:
                continue
            gy, gx = u["y"] // REGION, u["x"] // REGION
            if not (0 <= gy < gh and 0 <= gx < gw):
                continue
            # Units are always player-owned; clut[0] (neutral) is a
            # can't-happen fallback rendered as enemy so a bookkeeping edge
            # case never gets silently dropped.
            cls = int(clut[int(lut[u["owner"]])])
            oc = cls - 1 if cls else 2
            if ci in static_pos and not u["constructing"]:
                # Intensity instead of flat presence: how upgraded is this
                # structure (soft-normalized - the engine has no hard level
                # cap). max(): a region can hold >1 structure of a class.
                level_frac = min(float(u.get("level") or 1) / 10.0, 1.0)
                si = static_pos[ci]
                static[si, gy, gx] = max(static[si, gy, gx], level_frac)
            # Targets can sit in the cropped edge strip (maps are trimmed to
            # multiples of REGION), so bounds-check them like unit positions.
            ty, tx = (
                (u["ty"] // REGION, u["tx"] // REGION) if u["tx"] is not None else (-1, -1)
            )
            target_ok = 0 <= ty < gh and 0 <= tx < gw
            t = u["type"]
            if t == "Warship":
                # .get(): old BC caches predate health/maxHealth.
                health, max_health = u.get("health"), u.get("maxHealth")
                hf = health / max_health if health is not None and max_health else 1.0
                ch = TR_WARSHIP + oc
                transient[ch, gy, gx] = max(transient[ch, gy, gx], hf)
            elif t == "Transport":
                ch = TR_TRANSPORT + oc
                transient[ch, gy, gx] = max(transient[ch, gy, gx], log_norm(u["troops"]))
                if target_ok:
                    transient[TR_TRANSPORT_DEST + oc, ty, tx] = 1.0
            elif t == "Trade Ship":
                transient[TR_TRADE + oc, gy, gx] = 1.0
                if target_ok:
                    transient[TR_TRADE_DEST + oc, ty, tx] = 1.0
            elif t in ("Atom Bomb", "Hydrogen Bomb", "MIRV"):
                transient[TR_NUKE + oc, gy, gx] = 1.0
                if target_ok:
                    transient[TR_NUKE_IMPACT + oc, ty, tx] = 1.0
                if u["samLock"]:
                    transient[TR_NUKE_SAMLOCK + oc, gy, gx] = 1.0
            elif t == "SAMMissile":
                transient[TR_SAM_MISSILE + oc, gy, gx] = 1.0
                if target_ok:
                    transient[TR_SAM_MISSILE_IMPACT + oc, ty, tx] = 1.0
            elif t == "MIRV Warhead":
                transient[TR_MIRV_WARHEAD + oc, gy, gx] = 1.0
                if target_ok:
                    transient[TR_MIRV_WARHEAD_IMPACT + oc, ty, tx] = 1.0
            elif t == "Train":
                ch = TR_TRAIN + oc
                transient[ch, gy, gx] = max(transient[ch, gy, gx], log_norm(u["troops"]))
            if u["constructing"]:
                transient[TR_CONSTRUCTION + oc, gy, gx] = 1.0
            if u.get("cooldown"):  # .get(): old BC caches predate cooldown
                if t == "Missile Silo":
                    transient[TR_SILO_COOLDOWN + oc, gy, gx] = 1.0
                elif t == "SAM Launcher":
                    transient[TR_SAM_COOLDOWN + oc, gy, gx] = 1.0
            if u.get("station"):  # .get(): old BC caches predate station
                transient[TR_STATION + oc, gy, gx] = 1.0

        # Attack fronts (v7): source-tile intensity across ALL active
        # attacks, not just ego's - the single biggest pre-v7 blind spot
        # (third-party land wars were invisible). Shared across owners; the
        # player-feature stream carries who's attacking whom. Broad-front
        # expand attacks have no single source tile (srcX/srcY null) and
        # are naturally skipped - there's no meaningful point to plot.
        for a in ents["attacks"]:
            # .get(): old BC caches predate srcX/srcY.
            src_x, src_y = a.get("srcX"), a.get("srcY")
            if src_x is None or src_y is None:
                continue
            gy, gx = src_y // REGION, src_x // REGION
            if not (0 <= gy < gh and 0 <= gx < gw):
                continue
            val = log_norm(a["troops"])
            transient[TR_ATTACK_SRC, gy, gx] = max(transient[TR_ATTACK_SRC, gy, gx], val)
            if a["retreating"]:
                transient[TR_ATTACK_RETREAT, gy, gx] = max(
                    transient[TR_ATTACK_RETREAT, gy, gx], val
                )

        players, pmask = self._player_feats(
            ents, lut, me_slot, allies, ally_expiry, obs["tick"]
        )
        scalars = self._scalars(obs, ents, me_slot)
        masks = self._masks(obs, lut, spawn_phase)
        masks["legal_tile"] = self._legal_tile(owners, spawn_phase, gh, gw)
        return {
            "owners": owners,  # uint8 slots; full-res work happens on GPU
            "fallout_packed": fallout_packed,
            "defense_bonus_packed": defense_bonus_packed,
            "terrain_static": self._terrain_static,  # (2, H, W); static per game
            "static": static,
            "transient": transient,
            "clut": clut,
            "players": players,
            "pmask": pmask,
            "scalars": scalars,
            "me_slot": me_slot,
            **masks,
        }

    def _legal_tile(
        self, owners: np.ndarray, spawn_phase: bool, gh: int, gw: int
    ) -> np.ndarray:
        """Tile-pointer region mask. Normally all-ones (region choice is
        engine-snapped); during the spawn phase, regions containing at least
        one valid spawn tile (land, unowned, passable) - mirrors
        SpawnExecution.getSpawn(center)."""
        if not spawn_phase:
            return np.ones((gh, gw), dtype=np.float32)
        valid = (self._land == 1) & (self._mag < IMPASSABLE_MAGNITUDE) & (owners == 0)
        return (
            valid.reshape(gh, REGION, gw, REGION)
            .any(axis=(1, 3))
            .astype(np.float32)
        )

    def _ally_slots(
        self, ents: dict, me_slot: int, lut: np.ndarray
    ) -> tuple[set[int], dict[int, int]]:
        out = set()
        expiry: dict[int, int] = {}  # ally slot -> that alliance's expiresAt tick
        for a, b, exp in ents["alliances"]:
            sa, sb = int(lut[a]), int(lut[b])
            if sa == me_slot:
                out.add(sb)
                expiry[sb] = exp
            elif sb == me_slot:
                out.add(sa)
                expiry[sa] = exp
        return out, expiry

    def _player_feats(
        self,
        ents: dict,
        lut: np.ndarray,
        me_slot: int,
        allies: set[int],
        ally_expiry: dict[int, int],
        tick: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        feats = np.zeros((MAX_SLOTS, P_FEAT), dtype=np.float32)
        mask = np.zeros(MAX_SLOTS, dtype=np.float32)
        atk_between: dict[int, float] = {}
        # v7: global attack pressure (vs anyone, not just ego) + retreating
        # fraction, per attacking player - closes the third-party-attack
        # gap the spatial TR_ATTACK_* channels also address (this carries
        # how much per player; the grid carries where).
        out_troops: dict[int, float] = {}
        in_troops: dict[int, float] = {}
        atk_total: dict[int, int] = {}
        atk_retreating: dict[int, int] = {}
        for a in ents["attacks"]:
            sa, sb = int(lut[a["from"]]), int(lut[a["to"]]) if a["to"] else 0
            if sa == me_slot:
                atk_between[sb] = atk_between.get(sb, 0.0) + a["troops"]
            elif sb == me_slot:
                atk_between[sa] = atk_between.get(sa, 0.0) - a["troops"]
            out_troops[sa] = out_troops.get(sa, 0.0) + a["troops"]
            if sb:
                in_troops[sb] = in_troops.get(sb, 0.0) + a["troops"]
            atk_total[sa] = atk_total.get(sa, 0) + 1
            if a["retreating"]:
                atk_retreating[sa] = atk_retreating.get(sa, 0) + 1
        # Target marks (v7): who ego or an ally has painted as a priority
        # target - mirrors the existing embargo-by-ego column pattern.
        marking_slots = allies | {me_slot}
        marked: set[int] = set()
        for p in ents["players"]:
            if int(lut[p["id"]]) in marking_slots:
                marked.update(int(lut[t]) for t in p.get("targets", []))
        for p in ents["players"]:
            slot = int(lut[p["id"]])
            if slot <= 0:
                continue
            mask[slot] = 1.0
            n_atk = atk_total.get(slot, 0)
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
                # v7 additions below
                log_norm(out_troops.get(slot, 0.0)),
                log_norm(in_troops.get(slot, 0.0)),
                atk_retreating.get(slot, 0) / n_atk if n_atk else 0.0,
                max(0.0, min(1.0, (ally_expiry[slot] - tick) / 3000.0))
                if slot in ally_expiry
                else 0.0,
                1.0 if slot in marked else 0.0,
                log_norm(p.get("troopIncome", 0.0)),
                log_norm(float(p.get("goldIncome", 0.0))),
                1.0 if p.get("doomsday") else 0.0,
                min(1.0, p.get("doomsdayTicks", 0) / 3000.0),
            ]
        return feats, mask

    def _scalars(self, obs: dict, ents: dict, me_slot: int) -> np.ndarray:
        legal = obs["legal"]["actions"]
        n_alive = sum(p["alive"] for p in ents["players"])
        me_p = next((p for p in ents["players"] if p["id"] == obs["me"]), None)
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
                # v7: own income rate (matches DESIGN.md's aspirational
                # global "income rate" scalar) + doomsday-clock-enabled flag.
                log_norm(me_p.get("troopIncome", 0.0)) if me_p else 0.0,
                log_norm(float(me_p.get("goldIncome", 0.0))) if me_p else 0.0,
                1.0 if ents.get("doomsdayEnabled") else 0.0,
            ],
            dtype=np.float32,
        )

    def _masks(self, obs: dict, lut: np.ndarray, spawn_phase: bool) -> dict:
        legal = obs["legal"]["actions"]
        act = np.zeros(N_ACTIONS, dtype=np.float32)
        ptarget = np.zeros((N_ACTIONS, MAX_SLOTS), dtype=np.float32)

        def fill(action: str, ids: list[int]) -> None:
            if ids:
                act[ACTIONS.index(action)] = 1.0
                for i in ids:
                    ptarget[ACTIONS.index(action), int(lut[i])] = 1.0

        if spawn_phase:
            # The spawn decision is forced: nothing else (not even noop) is
            # legal, so the policy must place itself before playing.
            act[ACTIONS.index("spawn")] = 1.0
        else:
            act[ACTIONS.index("noop")] = 1.0
        if obs["alive"] and legal and not spawn_phase:
            fill("attack", legal.get("attackable", []))
            fill("alliance_request", legal.get("allianceRequestable", []))
            fill("alliance_reject", legal.get("allianceRejectable", []))
            fill("break_alliance", legal.get("breakable", []))
            fill("donate_gold", legal.get("donatableGold", []))
            fill("donate_troops", legal.get("donatableTroops", []))
            fill("embargo", legal.get("embargoable", []))
            # v6 player-target actions. Old BC caches predate
            # stopEmbargoable/extendable/deletable: missing key -> illegal.
            fill("embargo_stop", legal.get("stopEmbargoable", []))
            fill("target_player", legal.get("targetable", []))
            fill("alliance_extension", legal.get("extendable", []))
            # canExpand/canBoat are exact engine-state checks from the
            # bridge (neutral border, boat cap + own shore). Default True
            # for old BC caches whose legality blobs predate the keys.
            act[ACTIONS.index("expand")] = 1.0 if legal.get("canExpand", True) else 0.0
            act[ACTIONS.index("boat")] = (
                1.0
                if legal.get("canBoat", True) and legal.get("troops", 0) > 100
                else 0.0
            )
            build_ok = [t for t in BUILD_TYPES if t in legal.get("buildableTypes", [])]
            act[ACTIONS.index("build")] = 1.0 if build_ok else 0.0
            nukes_ok = [t for t in NUKE_UNITS if t in legal.get("buildableTypes", [])]
            act[ACTIONS.index("launch_nuke")] = (
                1.0 if nukes_ok and legal.get("hasSilo") else 0.0
            )
            # Targeted retreat: the player head picks WHICH attack to cancel
            # (slot of its target; slot 0 covers terra-nullius expands).
            if legal.get("attacks"):
                act[ACTIONS.index("retreat")] = 1.0
                r = ACTIONS.index("retreat")
                for a in obs["entities"]["attacks"]:
                    if a["from"] == obs["me"]:
                        ptarget[r, int(lut[a["to"]]) if a["to"] else 0] = 1.0
            act[ACTIONS.index("upgrade_structure")] = (
                1.0 if legal.get("upgradable") else 0.0
            )
            act[ACTIONS.index("move_warship")] = 1.0 if legal.get("warships") else 0.0
            act[ACTIONS.index("cancel_boat")] = 1.0 if legal.get("boats") else 0.0
            act[ACTIONS.index("delete_unit")] = 1.0 if legal.get("deletable") else 0.0
        build_mask = np.array(
            [1.0 if t in legal.get("buildableTypes", []) else 0.0 for t in BUILD_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(BUILD_TYPES), dtype=np.float32)
        nuke_mask = np.array(
            [1.0 if u in legal.get("buildableTypes", []) else 0.0 for u, _ in NUKE_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(NUKE_TYPES), dtype=np.float32)
        return {
            "legal_actions": act,
            "legal_ptarget": ptarget,
            "legal_build": build_mask,
            "legal_nuke": nuke_mask,
        }


class ZCache:
    """Cache over frozen-AE latents, keyed (game, tick).

    The AE input (owners + terrain/fallout + static unit planes) is
    per-(game, tick) and the AE is frozen, so z is identical for every
    player sample at the same snapshot - BC re-encoded it per sample, per
    epoch (~90% of wall). Latents are stored as fp16 numpy views into
    1GiB slab arenas under a byte budget (--z-cache-gb); once the budget
    is reached the cache stops inserting (with uniform random sampling
    that equals LRU's hit rate, without any churn). PPO must never use
    this: its episodes are unique, and encode_grids only consults the
    cache for raws that carry a "z_key" (only the BC samplers attach one).

    Slabs, not per-entry allocs: ~100k x 2.4MB mallocs interleaved with
    the trainer's big transient batch buffers fragmented the glibc heap
    within an hour (compact_stall climbing, collate 3.4s -> 25s per
    window - the same decay 9c030f7 fought). Each slab sits above the
    pinned MALLOC_MMAP_THRESHOLD, so it gets its own mmap and the heap
    the batch buffers recycle stays compact.

    disk_dir: slabs become file-backed memmaps plus an append-only
    index.jsonl, giving (a) warm restarts - a relaunch reopens the store
    and starts at the previous run's hit rate instead of re-encoding for
    hours, and (b) survival on cgroup-capped pods: the full human dataset
    is ~440GB of fp16 latents but bc3's container is capped at 250GB -
    anonymous slabs get the trainer OOM-killed before the cache is even
    two-thirds full, while file-backed pages are kernel-reclaimable, so
    the hot set stays in RAM and the cold tail pages in from disk.
    Writes go through the page cache (no sync I/O on the put path).

    Also keeps a per-game land plane on the GPU (uint8): cache hits skip
    the terrain_static upload (AE-only input), but _local_crops still
    needs land - re-uploading 9.6MB fp32 per sample would eat most of the
    win, while all games together are ~0.7GB as uint8."""

    SLAB = 1 << 30

    def __init__(self, max_bytes: int, disk_dir: str | Path | None = None):
        self.max_bytes = max_bytes
        self._d: dict[tuple, np.ndarray] = {}
        self._land: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._slabs: list[np.ndarray] = []
        self._off = 0
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._dir = Path(disk_dir) if disk_dir else None
        self._index_f = None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._load_disk()
            # Line-buffered: an OOM-killed run loses at most the last line
            # (the loader skips a torn tail and overwrites orphaned bytes).
            self._index_f = (self._dir / "index.jsonl").open("a", buffering=1)

    def _load_disk(self) -> None:
        idx = self._dir / "index.jsonl"
        paths = sorted(self._dir.glob("slab*.bin"))
        if not idx.exists() or not paths:
            return
        self._slabs = [np.memmap(p, dtype=np.uint8, mode="r+") for p in paths]
        ends = [0] * len(self._slabs)
        with idx.open() as f:
            for line in f:
                try:
                    k, s, off, shape, dt = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    break  # torn tail line from a killed writer
                if s >= len(self._slabs):
                    break
                nb = int(np.prod(shape)) * np.dtype(dt).itemsize
                self._d[(k[0], int(k[1]))] = (
                    self._slabs[s][off : off + nb].view(np.dtype(dt)).reshape(shape)
                )
                ends[s] = max(ends[s], off + ((nb + 15) & ~15))
                self.bytes += nb
        self._off = ends[-1]

    def _new_slab(self) -> bool:
        if (len(self._slabs) + 1) * self.SLAB > self.max_bytes:
            return False  # budget reached: cache saturates, never churns
        if self._dir is not None:
            s = np.memmap(
                self._dir / f"slab{len(self._slabs):04d}.bin",
                dtype=np.uint8, mode="w+", shape=(self.SLAB,),
            )
        else:
            s = np.empty(self.SLAB, dtype=np.uint8)
        self._slabs.append(s)
        self._off = 0
        return True

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            z = self._d.get(key)
            if z is None:
                self.misses += 1
                return None
            self.hits += 1
            return z

    def put(self, key: tuple, z: np.ndarray) -> None:
        with self._lock:
            if key in self._d:
                return
            nb = (z.nbytes + 15) & ~15  # keep slab offsets fp16-aligned
            if not self._slabs or self._off + nb > self.SLAB:
                if not self._new_slab():
                    return
            view = (
                self._slabs[-1][self._off : self._off + z.nbytes]
                .view(z.dtype)
                .reshape(z.shape)
            )
            view[...] = z
            if self._index_f is not None:
                self._index_f.write(json.dumps(
                    [list(key), len(self._slabs) - 1, self._off,
                     list(z.shape), z.dtype.str]
                ) + "\n")
            self._off += nb
            self._d[key] = view
            self.bytes += z.nbytes

    def __len__(self) -> int:
        return len(self._d)

    def land(self, raw: dict, device: str) -> torch.Tensor:
        """(H, W) uint8 land plane on GPU, cached per game."""
        game = raw["z_key"][0]
        t = self._land.get(game)
        if t is None:
            t = torch.from_numpy(
                raw["terrain_static"][0].astype(np.uint8)
            ).to(device)
            self._land[game] = t
        return t


_BIT_SHIFTS: dict[str, torch.Tensor] = {}
_warned_oversize = False


def _unpack_bits(packed: torch.Tensor, w: int) -> torch.Tensor:
    """(B, H, W/8) uint8 -> (B, H, W) float32 on-device."""
    key = str(packed.device)
    if key not in _BIT_SHIFTS:
        _BIT_SHIFTS[key] = torch.tensor(
            [7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8, device=packed.device
        )
    bits = (packed.unsqueeze(-1) >> _BIT_SHIFTS[key]) & 1
    return bits.reshape(packed.shape[0], packed.shape[1], w).float()


def _local_crops(
    classmap: torch.Tensor, land: torch.Tensor, defense_bonus: torch.Tensor
) -> torch.Tensor:
    """(B, N_LOCAL, LOCAL, LOCAL) crops centered on own-territory centroid
    (map center when the agent owns nothing, e.g. the spawn phase)."""
    B, H, W = classmap.shape
    own = (classmap == 1).float()
    if H < LOCAL or W < LOCAL:
        ph, pw = max(0, LOCAL - H), max(0, LOCAL - W)
        classmap = F.pad(classmap, (0, pw, 0, ph))
        land = F.pad(land, (0, pw, 0, ph))
        defense_bonus = F.pad(defense_bonus, (0, pw, 0, ph))
        own = F.pad(own, (0, pw, 0, ph))
        B, H, W = classmap.shape
    counts = own.sum(dim=(1, 2))
    ys = torch.arange(H, device=classmap.device, dtype=torch.float32)
    xs = torch.arange(W, device=classmap.device, dtype=torch.float32)
    cy = (own.sum(2) * ys).sum(1) / counts.clamp(min=1)
    cx = (own.sum(1) * xs).sum(1) / counts.clamp(min=1)
    cy = torch.where(counts > 0, cy, torch.full_like(cy, H / 2))
    cx = torch.where(counts > 0, cx, torch.full_like(cx, W / 2))
    y0 = (cy - LOCAL / 2).round().long().clamp(0, H - LOCAL)
    x0 = (cx - LOCAL / 2).round().long().clamp(0, W - LOCAL)

    # Batched gather (one kernel) instead of a per-sample Python loop: the
    # loop profiled at ~0.9ms/sample - the single biggest cost of the
    # cache-hit encode path once the AE was out of the picture.
    ar = torch.arange(LOCAL, device=classmap.device)
    yy = (y0[:, None] + ar)[:, :, None]  # (B, LOCAL, 1)
    xx = (x0[:, None] + ar)[:, None, :]  # (B, 1, LOCAL)
    bi = torch.arange(B, device=classmap.device)[:, None, None]
    cm = classmap[bi, yy, xx]
    ld = land[bi, yy, xx]
    db = defense_bonus[bi, yy, xx]
    return torch.stack(
        [
            (cm == 1).float(),
            (cm == 2).float(),
            (cm == 3).float(),
            ld.float(),
            db.float(),
        ],
        dim=1,
    )


# Reused per-thread staging buffers for the big full-res stacks below.
# Fresh np.stack allocations here ran ~0.5GB/batch (terrain_static alone is
# 19MB/sample), all above glibc's mmap threshold: every batch paid
# mmap+munmap page faults, and on hosts with fragmented physical memory
# (bc2: compact_stall in the millions) that fault path degraded 3-4x over
# hours - the "bc decay". A persistent buffer faults once and stays warm.
# Keyed by name + per-sample shape, grown to the largest batch seen;
# thread-local because PPO encodes from two acting threads concurrently.
_staging = threading.local()


def _staged_stack(
    name: str, arrays: list[np.ndarray], parallel: bool = False
) -> np.ndarray:
    bufs = getattr(_staging, "bufs", None)
    if bufs is None:
        bufs = _staging.bufs = {}
    n = len(arrays)
    key = (name, arrays[0].shape, arrays[0].dtype.str)
    buf = bufs.get(key)
    if buf is None or buf.shape[0] < n:
        buf = np.empty((n, *arrays[0].shape), arrays[0].dtype)
        bufs[key] = buf
    out = buf[:n]
    if parallel and n > 1:
        # Sources may be cold disk-backed memmaps (ZCache disk_dir): the
        # copies page-fault, so overlap the I/O instead of faulting serially.
        list(_copy_pool().map(
            lambda t: np.copyto(out[t[0]], t[1]), enumerate(arrays)
        ))
    else:
        np.stack(arrays, out=out)
    return out


_COPY_POOL: ThreadPoolExecutor | None = None


def _copy_pool() -> ThreadPoolExecutor:
    global _COPY_POOL
    if _COPY_POOL is None:
        _COPY_POOL = ThreadPoolExecutor(max_workers=8)
    return _COPY_POOL


@torch.no_grad()
def encode_grids(
    ae: SpatialAE,
    raws: list[dict],
    device: str,
    fp16: bool = False,
    z_cache: ZCache | None = None,
    torch_out: bool = False,
) -> list[dict]:
    """Batched full-resolution featurization on the GPU: frozen-AE encode,
    ego ownership pooling, fallout unpacking, and the local owner crop.
    Envs may be on different maps: work per shape group. Grids come back at
    NATIVE latent size (no padding); use collate() to pad a mixed batch to
    its own max before stacking. Tile regions use the global GW_MAX
    coordinate convention regardless of padding (see Policy.act), so a grid
    may never exceed GH_MAX x GW_MAX.

    fp16=True halves the device->host transfer and downstream collate /
    rollout-buffer / host->device cost (PPO's main-loop bottleneck). Values
    are safe in half precision: AE latents, {0..1} pooled fractions, and
    small integer unit counts. Consumers must cast back to float32 (or run
    under autocast) after upload.

    z_cache (BC only): raws carrying a "z_key" reuse cached AE latents and
    skip the encode plus the terrain/fallout/static uploads (AE-only
    inputs); per-player work (ego, local crop) still runs per sample.
    Duplicate keys within one batch (a BC step draw emits every actor at
    the same snapshot) encode once. PPO passes neither key nor cache, so
    its path is untouched.

    torch_out=True (BC trainer): "grid" and "local" come back as device
    tensors instead of numpy - the numpy path downloads ~300MB of grids
    per 96-batch only for collate to stack and re-upload them, which was
    the single biggest cost of the cache-hit feed path. Callers must use
    a collate that understands tensor grids (rl.bc.collate does); PPO
    keeps numpy (its rollout buffer lives on the CPU)."""
    from rl.curriculum import GH_MAX, GW_MAX

    n = len(raws)
    # Resolve cached latents: z_hit holds fp16 CPU latents; rep_of maps a
    # duplicate in-batch miss to the index that will actually encode.
    z_hit: dict[int, np.ndarray] = {}
    rep_of: dict[int, int] = {}
    if z_cache is not None:
        seen: dict[tuple, int] = {}
        for i, r in enumerate(raws):
            k = r.get("z_key")
            if k is None:
                continue
            z = z_cache.get(k)
            if z is not None:
                z_hit[i] = z
            elif k in seen:
                rep_of[i] = seen[k]
            else:
                seen[k] = i
    encode_idx = [i for i in range(n) if i not in z_hit and i not in rep_of]
    needed_reps = set(rep_of.values())

    groups: dict[tuple, list[int]] = {}
    for i in encode_idx:
        groups.setdefault(raws[i]["owners"].shape, []).append(i)

    # Cap the full-res pixel footprint of one encode chunk: enc_stem's
    # activations run ~32ch x B x H x W, so an unlucky batch of same-map
    # large-map samples tried 7-14GiB single allocations and OOM'd (bc2
    # crash-looped deterministically at step 27k - the resumed sampler
    # redraws the same batch). Chunking bounds the peak; results identical.
    MAX_ENC_PIX = 16_000_000
    chunks: list[list[int]] = []
    for (H, W), idxs in groups.items():
        per = max(1, MAX_ENC_PIX // (H * W))
        chunks.extend(idxs[k : k + per] for k in range(0, len(idxs), per))

    grid_by_idx: dict[int, np.ndarray] = {}
    local_by_idx: dict[int, np.ndarray] = {}
    z_gpu: dict[int, torch.Tensor] = {}
    for idxs in chunks:
        owners = torch.from_numpy(
            _staged_stack("owners", [raws[i]["owners"] for i in idxs])
        ).to(device)
        terr = torch.from_numpy(
            _staged_stack("terr", [raws[i]["terrain_static"] for i in idxs])
        ).to(device)
        packed = torch.from_numpy(
            _staged_stack("packed", [raws[i]["fallout_packed"] for i in idxs])
        ).to(device)
        db_packed = torch.from_numpy(
            _staged_stack(
                "db_packed", [raws[i]["defense_bonus_packed"] for i in idxs]
            )
        ).to(device)
        static = torch.from_numpy(
            _staged_stack("static", [raws[i]["static"] for i in idxs])
        ).to(device)
        clut = torch.from_numpy(
            np.stack([raws[i]["clut"] for i in idxs])
        ).long().to(device)

        B, H, W = owners.shape
        fallout = _unpack_bits(packed, W)
        terrain = torch.cat([terr, fallout.unsqueeze(1)], dim=1)
        # Defense bonus never reaches the (frozen) AE - exact bypass only,
        # same treatment as ego ownership below.
        defense_bonus = _unpack_bits(db_packed, W)
        # The frozen AE runs in bf16 on cuda: profiling showed the encode
        # phase dominating BC's feed thread (enc 60s vs smp 0s per window)
        # and it shares the GPU with the trainer's forward/backward.
        if owners.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = ae.encode(owners.long(), terrain, static)
        else:
            z = ae.encode(owners.long(), terrain, static)

        if z_cache is not None:
            z16 = z.half().cpu().numpy()
            for j, i in enumerate(idxs):
                k = raws[i].get("z_key")
                if k is not None:
                    z_cache.put(k, z16[j])  # put copies into its slab
                if i in needed_reps:
                    z_gpu[i] = z[j].float()

        classmap = torch.gather(
            clut, 1, owners.long().reshape(B, -1)
        ).reshape(B, H, W)
        ego = torch.stack(
            [(classmap == c).float() for c in (1, 2, 3)], dim=1
        )
        ego = F.avg_pool2d(ego, REGION)
        db_pooled = F.avg_pool2d(defense_bonus.unsqueeze(1), REGION)

        grid = torch.cat([z.float(), ego, db_pooled], dim=1)
        local = _local_crops(classmap, terr[:, 0], defense_bonus)
        if torch_out:
            # Append transient planes here (GPU) so "grid" is complete and
            # never round-trips through the host.
            tr = torch.from_numpy(
                _staged_stack("transient", [raws[i]["transient"] for i in idxs])
            ).to(device)
            grid = torch.cat([grid, tr], dim=1)
        if fp16:
            grid, local = grid.half(), local.half()
        if not torch_out:
            grid, local = grid.cpu().numpy(), local.cpu().numpy()
        for j, i in enumerate(idxs):
            grid_by_idx[i] = grid[j]
            local_by_idx[i] = local[j]

    # Cache-hit / in-batch-duplicate path: no AE, no terrain/fallout/static
    # upload - just owners+clut (per-player featurization) and the latent.
    cached_idx = [i for i in range(n) if i in z_hit or i in rep_of]
    groups_c: dict[tuple, list[int]] = {}
    for i in cached_idx:
        groups_c.setdefault(raws[i]["owners"].shape, []).append(i)
    chunks_c: list[list[int]] = []
    for (H, W), idxs in groups_c.items():
        per = max(1, MAX_ENC_PIX // (H * W))
        chunks_c.extend(idxs[k : k + per] for k in range(0, len(idxs), per))

    for idxs in chunks_c:
        owners = torch.from_numpy(
            _staged_stack("owners", [raws[i]["owners"] for i in idxs])
        ).to(device)
        clut = torch.from_numpy(
            np.stack([raws[i]["clut"] for i in idxs])
        ).long().to(device)
        land = torch.stack([z_cache.land(raws[i], device) for i in idxs]).float()
        db_packed = torch.from_numpy(
            _staged_stack(
                "db_packed", [raws[i]["defense_bonus_packed"] for i in idxs]
            )
        ).to(device)

        B, H, W = owners.shape
        # Defense bonus changes every tick, unlike the cached AE latent -
        # always decoded fresh (cheap: unpack + avg_pool, no AE involved).
        defense_bonus = _unpack_bits(db_packed, W)
        zs = torch.empty(
            B, LATENT_C, H // REGION, W // REGION, device=device
        )
        # One host->device upload per unique key, not per sample: a step
        # draw emits ~10 players per snapshot, and with a disk-backed
        # cache each redundant copy would also be a cold page-fault read.
        hit_groups: dict[tuple, list[int]] = {}
        for j, i in enumerate(idxs):
            if i in z_hit:
                hit_groups.setdefault(raws[i]["z_key"], []).append(j)
        if hit_groups:
            keys = list(hit_groups)
            zu = torch.from_numpy(
                _staged_stack(
                    "zhit", [z_hit[idxs[hit_groups[k][0]]] for k in keys],
                    parallel=z_cache._dir is not None,
                )
            ).to(device).float()
            for u, k in enumerate(keys):
                for j in hit_groups[k]:
                    zs[j] = zu[u]
        for j, i in enumerate(idxs):
            if i in rep_of:
                zs[j] = z_gpu[rep_of[i]]

        classmap = torch.gather(
            clut, 1, owners.long().reshape(B, -1)
        ).reshape(B, H, W)
        ego = torch.stack(
            [(classmap == c).float() for c in (1, 2, 3)], dim=1
        )
        ego = F.avg_pool2d(ego, REGION)
        db_pooled = F.avg_pool2d(defense_bonus.unsqueeze(1), REGION)

        grid = torch.cat([zs, ego, db_pooled], dim=1)
        local = _local_crops(classmap, land, defense_bonus)
        if torch_out:
            tr = torch.from_numpy(
                _staged_stack("transient", [raws[i]["transient"] for i in idxs])
            ).to(device)
            grid = torch.cat([grid, tr], dim=1)
        if fp16:
            grid, local = grid.half(), local.half()
        if not torch_out:
            grid, local = grid.cpu().numpy(), local.cpu().numpy()
        for j, i in enumerate(idxs):
            grid_by_idx[i] = grid[j]
            local_by_idx[i] = local[j]

    consumed = (
        "owners", "terrain_static", "fallout_packed", "defense_bonus_packed",
        "static", "clut", "transient", "z_key",
    )
    dtype = np.float16 if fp16 else np.float32
    out = []
    global _warned_oversize
    for i, r in enumerate(raws):
        o = {k: v for k, v in r.items() if k not in consumed}
        if torch_out:
            grid = grid_by_idx[i]  # transient already appended on device
        else:
            grid = np.concatenate([grid_by_idx[i], r["transient"].astype(dtype)])
        gh, gw = grid.shape[1], grid.shape[2]
        if (gh > GH_MAX or gw > GW_MAX) and not _warned_oversize:
            # Training can't get here (curriculum and the BC stride picker
            # both respect the budget); live play on a huge map can, and the
            # net runs fine there - just out of distribution.
            _warned_oversize = True
            print(
                f"warning: grid {gh}x{gw} exceeds curriculum max "
                f"{GH_MAX}x{GW_MAX}; policy is out of distribution",
                flush=True,
            )
        o["grid"] = grid
        o["grid_valid"] = np.ones((gh, gw), dtype=np.float32)
        o["local"] = local_by_idx[i]
        out.append(o)
    return out


def _staged_out(name: str, shape: tuple, dtype) -> np.ndarray:
    """Persistent per-thread flat buffer reshaped to `shape` (contiguous).
    Same rationale as _staged_stack: collate outputs run 60-200MB per
    batch, and fresh mallocs that size mmap/munmap every batch (glibc
    non-main arenas cap heap segments at 64MB regardless of
    MALLOC_MMAP_THRESHOLD_), which degrades badly on fragmented hosts -
    BC's col phase decayed 3.4s -> 14s per 50-step window on bc2."""
    bufs = getattr(_staging, "outs", None)
    if bufs is None:
        bufs = _staging.outs = {}
    numel = int(np.prod(shape))
    key = (name, np.dtype(dtype).str)
    buf = bufs.get(key)
    if buf is None or buf.size < numel:
        buf = np.empty(numel, dtype=dtype)
        bufs[key] = buf
    return buf[:numel].reshape(shape)


def collate(
    obs_list: list[dict], keys: list[str], staged: bool = False
) -> dict[str, np.ndarray]:
    """Stack per-env obs dicts into batch arrays, zero-padding 'grid',
    'grid_valid', and 'legal_tile' to the largest grid in THIS batch (not
    the curriculum-wide max - that wasted ~9x conv compute on small maps).

    staged=True returns views into persistent per-thread buffers, valid
    only until the caller's next collate on the same thread. Safe when
    the arrays are uploaded to the GPU immediately (BC does; PPO keeps
    rollout grids on the CPU, so it must NOT pass staged)."""
    from rl.native import collate_grids, collate_masks, stack

    out = {}
    gh = max(o["grid"].shape[1] for o in obs_list)
    gw = max(o["grid"].shape[2] for o in obs_list)
    B = len(obs_list)
    for k in keys:
        if k == "grid":
            # dtype-preserving: ppo stores rollout grids as fp16 to halve
            # collate/PCIe cost; everyone else passes fp32 through unchanged
            if staged:
                g0 = obs_list[0]["grid"]
                b = _staged_out("col_grid", (B, g0.shape[0], gh, gw), g0.dtype)
                b.fill(0)
                for i, o in enumerate(obs_list):
                    g = o["grid"]
                    b[i, :, : g.shape[1], : g.shape[2]] = g
            else:
                b = collate_grids([o["grid"] for o in obs_list], gh, gw)
        elif k in ("grid_valid", "legal_tile"):
            if staged:
                b = _staged_out(f"col_{k}", (B, gh, gw), np.float32)
                b.fill(0)
                for i, o in enumerate(obs_list):
                    m = o[k]
                    b[i, : m.shape[0], : m.shape[1]] = m
            else:
                b = collate_masks([o[k] for o in obs_list], gh, gw)
        else:
            if staged:
                a0 = np.asarray(obs_list[0][k])
                b = _staged_out(f"col_{k}", (B, *a0.shape), a0.dtype)
                for i, o in enumerate(obs_list):
                    b[i] = o[k]
            else:
                b = stack([o[k] for o in obs_list])
        out[k] = b
    return out
