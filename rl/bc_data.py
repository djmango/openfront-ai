"""Behavior-cloning dataset over replayed human games.

Assembles (observation, action, placement) samples from data-human game dirs
that carry bc.json.gz sidecars (datagen/replay.ts --bc):

  states/t<tick>.bin.gz   owner/fallout grid (shared across players at a tick)
  states/t<tick>.json.gz  entities
  bc.json.gz              per-snapshot legality per living human + their
                          intents in the following window (normalized to the
                          policy action space) + final placements

One *step* is (game, snapshot, clientID). Steps where the player acted yield
action labels; steps where they didn't are no-op supervision, downsampled via
--noop-frac at the sampler level (humans idle ~90% of decision steps; naive
training collapses to "always noop").

Obs assembly reuses rl.obs.ObsBuilder.prepare() — identical featurization to
the live PPO env — so BC weights drop into PPO as initialization.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.obs import ACTIONS, BUILD_TYPES, NUKE_TYPES, ObsBuilder
from rl.policy import NEEDS_PLAYER, NEEDS_QUANTITY, NEEDS_TILE, QUANTITY_FRACS

OWNER_MASK = 0x0FFF
FALLOUT_BIT = 13

N_PLACEMENT_BUCKETS = 8


def placement_bucket(p: float) -> int:
    """0 = bottom of the lobby, N-1 = winner-tier."""
    return min(N_PLACEMENT_BUCKETS - 1, int(p * N_PLACEMENT_BUCKETS))


def quantity_bucket(amt: float | None, avail: float) -> int:
    """Nearest QUANTITY_FRACS bucket for an absolute amount; the client
    default (amt=None) plays like the 25% slider."""
    if amt is None or avail <= 0:
        return 2
    frac = min(1.0, max(0.0, float(amt) / float(avail)))
    return int(np.argmin([abs(frac - f) for f in QUANTITY_FRACS]))


@dataclass
class GameHandle:
    path: Path
    meta: dict
    bc: dict
    terrain: np.ndarray  # strided by ds
    lut: np.ndarray  # smallID -> slot (built from the earliest snapshot)
    gh: int  # grid dims after striding
    gw: int
    ds: int  # tile-space stride so the grid fits (GH_MAX, GW_MAX)

    @property
    def steps(self) -> list[dict]:
        return self.bc["steps"]


def _slot_lut_from_players(players: list[dict]) -> np.ndarray:
    from ae.model_v3 import MAX_SLOTS

    ids = sorted(p["id"] for p in players)
    lut = np.zeros(4096, dtype=np.int64)
    for slot, sid in enumerate(ids, start=1):
        lut[sid] = min(slot, MAX_SLOTS - 1)
    return lut


def load_game(path: Path, gh_max: int, gw_max: int) -> GameHandle | None:
    bc_path = path / "bc.json.gz"
    if not bc_path.exists() or not (path / "meta.json").exists():
        return None  # sidecar without snapshots (partial sync)
    meta = json.loads((path / "meta.json").read_text())
    h, w = meta["height"], meta["width"]

    # Human games run on Normal-size maps (up to ~4100 tiles wide), while the
    # policy grid is budgeted for Compact/small training maps. Stride tile
    # space by 2/4 until it fits — the engine's own Compact mode is exactly a
    # 2x-downscaled Normal map, so strided games stay in-distribution.
    ds = 1
    while True:
        hs, ws = len(range(0, h, ds)), len(range(0, w, ds))
        gh, gw = (hs - hs % 16) // 16, (ws - ws % 16) // 16
        if gh <= gh_max and gw <= gw_max:
            break
        ds *= 2
        if ds > 8:
            return None

    bc = json.loads(gzip.decompress(bc_path.read_bytes()))
    if not bc["steps"]:
        return None
    terrain = np.frombuffer((path / "terrain.bin").read_bytes(), dtype=np.uint8)
    terrain = terrain.reshape(h, w)[::ds, ::ds]

    first = bc["steps"][0]["tick"]
    ents0 = _load_entities(path, first)
    if ents0 is None:
        return None
    return GameHandle(
        path=path,
        meta=meta,
        bc=bc,
        terrain=terrain,
        lut=_slot_lut_from_players(ents0["players"]),
        gh=gh,
        gw=gw,
        ds=ds,
    )


def scale_entities(entities: dict, ds: int) -> dict:
    """Unit coordinates live in tile space; stride them with the grid."""
    if ds == 1:
        return entities
    units = [
        {
            **u,
            "x": u["x"] // ds,
            "y": u["y"] // ds,
            "tx": u["tx"] // ds if u.get("tx") is not None else None,
            "ty": u["ty"] // ds if u.get("ty") is not None else None,
        }
        for u in entities["units"]
    ]
    return {**entities, "units": units}


def _load_entities(path: Path, tick: int) -> dict | None:
    f = path / "states" / f"t{tick:06d}.json.gz"
    if not f.exists():
        return None
    return json.loads(gzip.decompress(f.read_bytes()))


def _load_grid(
    path: Path, tick: int, h: int, w: int, ds: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    raw = gzip.decompress((path / "states" / f"t{tick:06d}.bin.gz").read_bytes())
    state = np.frombuffer(raw, dtype="<u2").reshape(h, w)[::ds, ::ds]
    return (state & OWNER_MASK).astype(np.int64), ((state >> FALLOUT_BIT) & 1)


def label_to_choice(
    label: dict, game: GameHandle, entities: dict, client_smallid: int
) -> dict | None:
    """Map a normalized BC label to policy head targets. Unused heads are -1
    (matches Policy.evaluate choice tensors)."""
    name = label["a"]
    if name not in ACTIONS:
        return None
    choice = {
        "action": ACTIONS.index(name),
        "player_slot": -1,
        "tile_region": -1,
        "build_type": -1,
        "nuke_type": -1,
        "quantity": -1,
    }
    if name in NEEDS_PLAYER:
        slot = int(game.lut[label["t"]])
        if slot <= 0:
            return None
        choice["player_slot"] = slot
    if name in NEEDS_TILE:
        gx, gy = (label["x"] // game.ds) // 16, (label["y"] // game.ds) // 16
        if not (0 <= gy < game.gh and 0 <= gx < game.gw):
            return None
        # Flattened over the *padded* (GH_MAX, GW_MAX) grid: padding sits at
        # bottom/right so in-bounds indices are unchanged.
        from rl.curriculum import GW_MAX

        choice["tile_region"] = gy * GW_MAX + gx
    if name == "build":
        choice["build_type"] = BUILD_TYPES.index(label["unit"])
    if name == "launch_nuke":
        choice["nuke_type"] = NUKE_TYPES.index(label["unit"])
    if name in NEEDS_QUANTITY:
        me = next(
            (p for p in entities["players"] if p["id"] == client_smallid), None
        )
        avail = float(me["gold"]) if name == "donate_gold" else float(me["troops"]) if me else 0.0
        choice["quantity"] = quantity_bucket(label.get("amt"), avail)
    return choice


NOOP_CHOICE = {
    "action": ACTIONS.index("noop"),
    "player_slot": -1,
    "tile_region": -1,
    "build_type": -1,
    "nuke_type": -1,
    "quantity": -1,
}


class BCSampler:
    """Random (game, step) sampler that emits every acting player at the
    chosen snapshot (amortizing the grid/entity decode) plus a controlled
    ration of no-op players.

    Emits "raw" prepare() dicts; the trainer batches AE encoding on GPU via
    rl.obs.encode_grids, exactly like the PPO rollout path.
    """

    def __init__(
        self,
        roots: list[Path],
        holdout_every: int = 10,
        holdout: bool = False,
        noop_frac: float = 0.15,
        seed: int = 0,
    ):
        from rl.curriculum import GH_MAX, GW_MAX

        self.rng = random.Random(seed)
        self.noop_frac = noop_frac
        self.games: list[Path] = []
        for root in roots:
            for meta in sorted(root.glob("*/*/bc.json.gz")):
                # Stable split (builtin hash() is salted per process).
                digest = hashlib.md5(meta.parent.name.encode()).digest()
                if (digest[0] % holdout_every == 0) == holdout:
                    self.games.append(meta.parent)
        if not self.games:
            raise FileNotFoundError(f"no bc.json.gz under {roots} (holdout={holdout})")
        self.gh_max, self.gw_max = GH_MAX, GW_MAX
        self._handles: dict[Path, GameHandle | None] = {}
        self._builders: dict[Path, ObsBuilder] = {}

    def _handle(self, path: Path) -> GameHandle | None:
        if path not in self._handles:
            # Cap the cache: handles hold bc docs (~tens of MB as objects).
            if len(self._handles) > 16:
                self._handles.clear()
                self._builders.clear()
            self._handles[path] = load_game(path, self.gh_max, self.gw_max)
        return self._handles[path]

    def _builder(self, game: GameHandle) -> ObsBuilder:
        if game.path not in self._builders:
            b = ObsBuilder()  # no AE: encode happens centrally
            b.start_game(game.terrain)
            b.lut = game.lut
            self._builders[game.path] = b
        return self._builders[game.path]

    def sample_step(self) -> list[dict]:
        """All samples for one random (game, snapshot): raw obs + choice +
        placement bucket per emitted player."""
        for _ in range(64):
            game = self._handle(self.rng.choice(self.games))
            if game is None:
                continue
            step = self.rng.choice(game.steps)
            out = self._step_samples(game, step)
            if out:
                return out
        raise RuntimeError("could not draw a non-empty BC step in 64 tries")

    def _step_samples(self, game: GameHandle, step: dict) -> list[dict]:
        tick = step["tick"]
        h, w = game.meta["height"], game.meta["width"]
        entities = _load_entities(game.path, tick)
        if entities is None:
            return []
        entities = scale_entities(entities, game.ds)
        owners, fallout = _load_grid(game.path, tick, h, w, game.ds)
        builder = self._builder(game)
        placements = game.bc["placements"]

        actors = [c for c, ls in step["labels"].items() if ls and c in step["legal"]]
        noops = [c for c in step["legal"] if c not in step["labels"]]
        take = actors[:]
        n_noop = max(1, int(len(take) * self.noop_frac)) if noops else 0
        take += self.rng.sample(noops, min(n_noop, len(noops)))

        out = []
        for cid in take:
            leg = step["legal"][cid]
            obs = {
                "tick": tick,
                "spawnPhase": False,
                "me": leg["me"],
                "alive": True,
                "owners": owners,
                "fallout": fallout,
                "entities": entities,
                "legal": leg["legal"],
            }
            raw = builder.prepare(obs)
            if cid in step["labels"]:
                label = self.rng.choice(step["labels"][cid])
                choice = label_to_choice(label, game, entities, leg["me"])
                if choice is None:
                    continue
                # The human did it, so it was legal: force the labeled option
                # into the masks (our legality approximation can be narrower).
                raw["legal_actions"][choice["action"]] = 1.0
                if choice["player_slot"] >= 0:
                    raw["legal_ptarget"][choice["action"], choice["player_slot"]] = 1.0
                if choice["build_type"] >= 0:
                    raw["legal_build"][choice["build_type"]] = 1.0
                if choice["nuke_type"] >= 0:
                    raw["legal_nuke"][choice["nuke_type"]] = 1.0
            else:
                choice = dict(NOOP_CHOICE)
            p = placements.get(cid, {"placement": 0.5})
            raw["choice"] = choice
            raw["cond"] = placement_bucket(float(p["placement"]))
            out.append(raw)
        return out

    def sample_batch(self, n: int) -> list[dict]:
        out: list[dict] = []
        while len(out) < n:
            out.extend(self.sample_step())
        return out[:n]

    def sample_window(self, k: int) -> list[dict]:
        """One player's last-k consecutive decision steps (for --seq BC).

        Returns exactly k raw samples, step-major, with the supervised
        choice on the last one (earlier steps carry noop placeholders that
        the trainer discards). Empty list when the draw fails; caller
        retries."""
        game = self._handle(self.rng.choice(self.games))
        if game is None or len(game.steps) < 1:
            return []
        steps = game.steps

        # Prefer windows ending on an acted step (same act/noop balance
        # logic as sample_step, applied to window ends).
        want_actor = self.rng.random() > self.noop_frac
        ends = list(range(len(steps)))
        self.rng.shuffle(ends)
        for end in ends[:16]:
            step = steps[end]
            cands = [
                c for c, ls in step["labels"].items() if ls and c in step["legal"]
            ] if want_actor else [
                c for c in step["legal"] if c not in step["labels"]
            ]
            if not cands:
                continue
            cid = self.rng.choice(cands)
            idxs = list(range(max(0, end - k + 1), end + 1))
            idxs = [idxs[0]] * (k - len(idxs)) + idxs  # left-pad short histories
            if any(cid not in steps[i]["legal"] for i in idxs):
                continue  # player wasn't alive across the whole window
            out = []
            for j, i in enumerate(idxs):
                sample = self._one_sample(
                    game, steps[i], cid, labeled=(j == k - 1)
                )
                if sample is None:
                    break
                out.append(sample)
            if len(out) == k:
                return out
        return []

    def _one_sample(
        self, game: GameHandle, step: dict, cid: str, labeled: bool
    ) -> dict | None:
        tick = step["tick"]
        h, w = game.meta["height"], game.meta["width"]
        entities = _load_entities(game.path, tick)
        if entities is None:
            return None
        entities = scale_entities(entities, game.ds)
        owners, fallout = _load_grid(game.path, tick, h, w, game.ds)
        leg = step["legal"][cid]
        obs = {
            "tick": tick,
            "spawnPhase": False,
            "me": leg["me"],
            "alive": True,
            "owners": owners,
            "fallout": fallout,
            "entities": entities,
            "legal": leg["legal"],
        }
        raw = self._builder(game).prepare(obs)
        choice = dict(NOOP_CHOICE)
        if labeled and step["labels"].get(cid):
            label = self.rng.choice(step["labels"][cid])
            c = label_to_choice(label, game, entities, leg["me"])
            if c is None:
                return None
            choice = c
            raw["legal_actions"][choice["action"]] = 1.0
            if choice["player_slot"] >= 0:
                raw["legal_ptarget"][choice["action"], choice["player_slot"]] = 1.0
            if choice["build_type"] >= 0:
                raw["legal_build"][choice["build_type"]] = 1.0
            if choice["nuke_type"] >= 0:
                raw["legal_nuke"][choice["nuke_type"]] = 1.0
        p = game.bc["placements"].get(cid, {"placement": 0.5})
        raw["choice"] = choice
        raw["cond"] = placement_bucket(float(p["placement"]))
        return raw
