"""Observation builder: frozen spatial-AE latent + exact bypass features.

Per DESIGN.md: the AE compresses the map; everything small reaches the
policy raw. v4 layout (d8c32 encoder, everything at 1/8 resolution):

  grid:     (C_GRID, H/8, W/8) - frozen z_grid (32ch) + ego ownership
            planes (3) + transient unit planes (8, exact) = 43ch
  local:    (N_LOCAL, LOCAL, LOCAL) raw owner-map crop centered on the
            agent's territory - exact borders where the agent acts most
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

# Transient planes (exact bypass, rendered at 1/REGION): warship, transport,
# transport-destination, trade ship, nuke, nuke-impact-point, samlock-nuke,
# construction.
N_TRANSIENT = 8
C_GRID = LATENT_C + 3 + N_TRANSIENT

# Raw local owner-map crop (exact-borders bypass): own/ally/enemy/land
# planes over a LOCAL x LOCAL tile window centered on own territory.
LOCAL = 64
N_LOCAL = 4

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
    "spawn",  # tile target; only legal (and forced) during the spawn phase
]
N_ACTIONS = len(ACTIONS)

BUILD_TYPES = ["City", "Port", "Defense Post", "Missile Silo", "SAM Launcher", "Factory"]
NUKE_TYPES = ["Atom Bomb", "Hydrogen Bomb", "MIRV"]


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
            if ci in static_pos and not u["constructing"]:
                static[static_pos[ci], gy, gx] = 1.0
            # Targets can sit in the cropped edge strip (maps are trimmed to
            # multiples of REGION), so bounds-check them like unit positions.
            ty, tx = (
                (u["ty"] // REGION, u["tx"] // REGION) if u["tx"] is not None else (-1, -1)
            )
            target_ok = 0 <= ty < gh and 0 <= tx < gw
            t = u["type"]
            if t == "Warship":
                transient[0, gy, gx] = 1.0
            elif t == "Transport":
                transient[1, gy, gx] = 1.0
                if target_ok:
                    transient[2, ty, tx] = 1.0
            elif t == "Trade Ship":
                transient[3, gy, gx] = 1.0
            elif t in ("Atom Bomb", "Hydrogen Bomb", "MIRV"):
                transient[4, gy, gx] = 1.0
                if target_ok:
                    transient[5, ty, tx] = 1.0
                if u["samLock"]:
                    transient[6, gy, gx] = 1.0
            if u["constructing"]:
                transient[7, gy, gx] = 1.0

        # Slot -> ego class LUT (0 neutral/unowned, 1 own, 2 ally, 3 enemy);
        # the GPU expands it over the full-res owner grid.
        allies = self._ally_slots(ents, me_slot, lut)
        clut = np.full(MAX_SLOTS, 3, dtype=np.uint8)
        clut[0] = 0
        for s in allies:
            clut[s] = 2
        if me_slot > 0:
            clut[me_slot] = 1

        players, pmask = self._player_feats(ents, lut, me_slot, allies)
        scalars = self._scalars(obs, ents, me_slot)
        masks = self._masks(obs, lut, spawn_phase)
        masks["legal_tile"] = self._legal_tile(owners, spawn_phase, gh, gw)
        return {
            "owners": owners,  # uint8 slots; full-res work happens on GPU
            "fallout_packed": fallout_packed,
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

    def _ally_slots(self, ents: dict, me_slot: int, lut: np.ndarray) -> set[int]:
        out = set()
        for a, b, _exp in ents["alliances"]:
            sa, sb = int(lut[a]), int(lut[b])
            if sa == me_slot:
                out.add(sb)
            elif sb == me_slot:
                out.add(sa)
        return out

    def _player_feats(
        self, ents: dict, lut: np.ndarray, me_slot: int, allies: set[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        feats = np.zeros((MAX_SLOTS, P_FEAT), dtype=np.float32)
        mask = np.zeros(MAX_SLOTS, dtype=np.float32)
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
            [1.0 if t in legal.get("buildableTypes", []) else 0.0 for t in BUILD_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(BUILD_TYPES), dtype=np.float32)
        nuke_mask = np.array(
            [1.0 if t in legal.get("buildableTypes", []) else 0.0 for t in NUKE_TYPES],
            dtype=np.float32,
        ) if obs["alive"] and legal else np.zeros(len(NUKE_TYPES), dtype=np.float32)
        return {
            "legal_actions": act,
            "legal_ptarget": ptarget,
            "legal_build": build_mask,
            "legal_nuke": nuke_mask,
        }

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
    classmap: torch.Tensor, land: torch.Tensor
) -> torch.Tensor:
    """(B, N_LOCAL, LOCAL, LOCAL) crops centered on own-territory centroid
    (map center when the agent owns nothing, e.g. the spawn phase)."""
    B, H, W = classmap.shape
    own = (classmap == 1).float()
    if H < LOCAL or W < LOCAL:
        ph, pw = max(0, LOCAL - H), max(0, LOCAL - W)
        classmap = F.pad(classmap, (0, pw, 0, ph))
        land = F.pad(land, (0, pw, 0, ph))
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

    out = torch.empty(B, N_LOCAL, LOCAL, LOCAL, device=classmap.device)
    for b in range(B):
        cm = classmap[b, y0[b] : y0[b] + LOCAL, x0[b] : x0[b] + LOCAL]
        out[b, 0] = cm == 1
        out[b, 1] = cm == 2
        out[b, 2] = cm == 3
        out[b, 3] = land[b, y0[b] : y0[b] + LOCAL, x0[b] : x0[b] + LOCAL]
    return out


@torch.no_grad()
def encode_grids(ae: SpatialAE, raws: list[dict], device: str) -> list[dict]:
    """Batched full-resolution featurization on the GPU: frozen-AE encode,
    ego ownership pooling, fallout unpacking, and the local owner crop.
    Envs may be on different maps: work per shape group. Grids come back at
    NATIVE latent size (no padding); use collate() to pad a mixed batch to
    its own max before stacking. Tile regions use the global GW_MAX
    coordinate convention regardless of padding (see Policy.act), so a grid
    may never exceed GH_MAX x GW_MAX."""
    from rl.curriculum import GH_MAX, GW_MAX

    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(raws):
        groups.setdefault(r["owners"].shape, []).append(i)

    grid_by_idx: dict[int, np.ndarray] = {}
    local_by_idx: dict[int, np.ndarray] = {}
    for idxs in groups.values():
        owners = torch.from_numpy(
            np.stack([raws[i]["owners"] for i in idxs])
        ).to(device)
        terr = torch.from_numpy(
            np.stack([raws[i]["terrain_static"] for i in idxs])
        ).to(device)
        packed = torch.from_numpy(
            np.stack([raws[i]["fallout_packed"] for i in idxs])
        ).to(device)
        static = torch.from_numpy(
            np.stack([raws[i]["static"] for i in idxs])
        ).to(device)
        clut = torch.from_numpy(
            np.stack([raws[i]["clut"] for i in idxs])
        ).long().to(device)

        B, H, W = owners.shape
        fallout = _unpack_bits(packed, W)
        terrain = torch.cat([terr, fallout.unsqueeze(1)], dim=1)
        z = ae.encode(owners.long(), terrain, static)

        classmap = torch.gather(
            clut, 1, owners.long().reshape(B, -1)
        ).reshape(B, H, W)
        ego = torch.stack(
            [(classmap == c).float() for c in (1, 2, 3)], dim=1
        )
        ego = F.avg_pool2d(ego, REGION)

        grid = torch.cat([z.float(), ego], dim=1).cpu().numpy()
        local = _local_crops(classmap, terr[:, 0]).cpu().numpy()
        for j, i in enumerate(idxs):
            grid_by_idx[i] = grid[j]
            local_by_idx[i] = local[j]

    consumed = ("owners", "terrain_static", "fallout_packed", "static", "clut", "transient")
    out = []
    global _warned_oversize
    for i, r in enumerate(raws):
        o = {k: v for k, v in r.items() if k not in consumed}
        grid = np.concatenate([grid_by_idx[i], r["transient"]]).astype(np.float32)
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
        o["local"] = local_by_idx[i].astype(np.float32)
        out.append(o)
    return out


def collate(obs_list: list[dict], keys: list[str]) -> dict[str, np.ndarray]:
    """Stack per-env obs dicts into batch arrays, zero-padding 'grid',
    'grid_valid', and 'legal_tile' to the largest grid in THIS batch (not
    the curriculum-wide max - that wasted ~9x conv compute on small maps)."""
    out = {}
    gh = max(o["grid"].shape[1] for o in obs_list)
    gw = max(o["grid"].shape[2] for o in obs_list)
    for k in keys:
        if k == "grid":
            b = np.zeros((len(obs_list), obs_list[0]["grid"].shape[0], gh, gw), dtype=np.float32)
            for i, o in enumerate(obs_list):
                g = o["grid"]
                b[i, :, : g.shape[1], : g.shape[2]] = g
        elif k in ("grid_valid", "legal_tile"):
            b = np.zeros((len(obs_list), gh, gw), dtype=np.float32)
            for i, o in enumerate(obs_list):
                v = o[k]
                b[i, : v.shape[0], : v.shape[1]] = v
        else:
            b = np.stack([o[k] for o in obs_list])
        out[k] = b
    return out
