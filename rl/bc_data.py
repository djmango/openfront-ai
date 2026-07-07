"""Behavior-cloning dataset over prefeaturized human-game caches.

Reads the cache-bc/ layout written by scripts/prefeaturize_bc.py (zstd-1
frames + per-step orjson blobs, everything pre-strided to the policy grid
budget). Per-sample CPU is ~1ms: one frame decode, one entity decode, and
cheap small-state featurization via rl.obs.ObsBuilder.prepare() - the
full-resolution work (AE encode, ego pooling, local crop) happens batched
on the GPU in rl.obs.encode_grids, exactly like the PPO rollout path, so
BC weights drop into PPO as initialization.

One *step* is (game, step_idx, clientID). Steps where the player acted
yield action labels; steps where they didn't are no-op supervision,
downsampled via --noop-frac at the sampler level (humans idle ~90% of
decision steps; naive training collapses to "always noop"). Sidecars with
spawn supervision (formatVersion 2) additionally yield spawn-placement
samples, drawn at --spawn-frac.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path

import numpy as np
import zstandard as zstd

_TLS = threading.local()


def _dctx() -> zstd.ZstdDecompressor:
    # One decompressor per thread: the trainer prefetches from a thread
    # pool and zstd contexts are not thread-safe (nor fork/pickle-able).
    if not hasattr(_TLS, "d"):
        _TLS.d = zstd.ZstdDecompressor()
    return _TLS.d

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

from rl.obs import ACTIONS, BUILD_TYPES, NUKE_TYPES, REGION, ObsBuilder
from rl.policy import NEEDS_PLAYER, NEEDS_QUANTITY, NEEDS_TILE, QUANTITY_FRACS

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


class CachedGame:
    """View over one game's cache-bc/ (mmapped; cheap enough to keep every
    game resident - no LRU or sticky sampling needed anymore)."""

    def __init__(self, path: Path):
        cache = path / "cache-bc"
        idx = json.loads((cache / "index.json").read_text())
        self.path = path
        self.ds = idx["ds"]
        self.hr, self.wr = idx["hr"], idx["wr"]
        self.gh, self.gw = self.hr // REGION, self.wr // REGION
        self.placements = idx["placements"]
        self.spawn_steps = idx["spawn_steps"]
        self.n_spawn = idx["n_spawn"]
        self._tick_row = {t: i for i, t in enumerate(idx["ticks"])}
        self._frame_off = np.asarray(idx["frame_offsets"], dtype=np.int64)
        self._ent_off = np.asarray(idx["ent_offsets"], dtype=np.int64)
        self._step_off = np.asarray(idx["step_offsets"], dtype=np.int64)
        self._frames = np.memmap(cache / "frames.zst", dtype=np.uint8, mode="r")
        self._ents = np.memmap(cache / "ents.zst", dtype=np.uint8, mode="r")
        self._steps = np.memmap(cache / "steps.zst", dtype=np.uint8, mode="r")
        self.terrain = np.load(cache / "terrain.npy")
        self.lut = np.load(cache / "lut.npy").astype(np.int64)

    def _d(self, buf: np.ndarray, off: np.ndarray, i: int, max_out: int) -> bytes:
        return _dctx().decompress(
            buf[off[i] : off[i + 1]].tobytes(), max_output_size=max_out
        )

    def n_steps(self) -> int:
        return len(self._step_off) - 1

    def step(self, i: int) -> dict:
        return _loads(self._d(self._steps, self._step_off, i, 1 << 26))

    def frame(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        """(owner slots uint8 (hr, wr), packed fallout (hr, wr/8))."""
        i = self._tick_row[tick]
        raw = self._d(self._frames, self._frame_off, i, self.hr * self.wr * 2)
        hw = self.hr * self.wr
        slots = np.frombuffer(raw[:hw], dtype=np.uint8).reshape(self.hr, self.wr)
        packed = np.frombuffer(raw[hw:], dtype=np.uint8).reshape(self.hr, -1)
        return slots, packed

    def entities(self, tick: int) -> dict:
        i = self._tick_row[tick]
        return _loads(self._d(self._ents, self._ent_off, i, 1 << 27))


def label_to_choice(
    label: dict, game: CachedGame, entities: dict, client_smallid: int
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
        gx = (label["x"] // game.ds) // REGION
        gy = (label["y"] // game.ds) // REGION
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
    chosen snapshot (amortizing the frame/entity decode) plus a controlled
    ration of no-op players and spawn-placement samples.

    Emits "raw" prepare() dicts; the trainer batches AE encoding on GPU via
    rl.obs.encode_grids, exactly like the PPO rollout path.
    """

    def __init__(
        self,
        roots: list[Path],
        holdout_every: int = 10,
        holdout: bool = False,
        noop_frac: float = 0.15,
        spawn_frac: float = 0.03,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.noop_frac = noop_frac
        self.spawn_frac = spawn_frac
        self.games: list[CachedGame] = []
        for root in roots:
            for idx in sorted(root.glob("*/*/cache-bc/index.json")):
                game_dir = idx.parent.parent
                # Stable split (builtin hash() is salted per process).
                digest = hashlib.md5(game_dir.name.encode()).digest()
                if (digest[0] % holdout_every == 0) == holdout:
                    self.games.append(CachedGame(game_dir))
        if not self.games:
            raise FileNotFoundError(
                f"no cache-bc under {roots} (holdout={holdout}); "
                "run scripts/prefeaturize_bc.py"
            )
        self.spawn_games = [g for g in self.games if g.n_spawn > 0]
        self._builders: dict[Path, ObsBuilder] = {}

    def _builder(self, game: CachedGame) -> ObsBuilder:
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
            if self.spawn_games and self.rng.random() < self.spawn_frac:
                out = self._spawn_samples(self.rng.choice(self.spawn_games))
            else:
                game = self.rng.choice(self.games)
                out = self._step_samples(game, game.step(self.rng.randrange(game.n_steps())))
            if out:
                return out
        raise RuntimeError("could not draw a non-empty BC step in 64 tries")

    def _obs(self, game: CachedGame, tick: int, entities: dict, leg: dict,
             spawn: bool) -> dict:
        owners, fallout_packed = game.frame(tick)
        return {
            "tick": tick,
            "spawnPhase": spawn,
            "me": leg["me"],
            "alive": not spawn,
            "owners_slots": owners,
            "fallout_packed": fallout_packed,
            "entities": entities,
            "legal": leg.get("legal", {"actions": {}}),
        }

    def _spawn_samples(self, game: CachedGame) -> list[dict]:
        """One spawn-placement sample: the map state at the moment a human
        picked their spawn, supervised with their actual pick."""
        from rl.curriculum import GW_MAX

        step = self.rng.choice(game.spawn_steps)
        cands = list(step["labels"].items())
        if not cands:
            return []
        cid, label = self.rng.choice(cands)
        gx = (label["x"] // game.ds) // REGION
        gy = (label["y"] // game.ds) // REGION
        if not (0 <= gy < game.gh and 0 <= gx < game.gw):
            return []
        entities = game.entities(step["tick"])
        raw = self._builder(game).prepare(
            self._obs(game, step["tick"], entities,
                      {"me": label.get("me", -1)}, spawn=True)
        )
        choice = dict(NOOP_CHOICE)
        choice["action"] = ACTIONS.index("spawn")
        choice["tile_region"] = gy * GW_MAX + gx
        # The human did spawn there, so the region was legal at that tick;
        # force it into the mask (our reconstruction can be narrower).
        raw["legal_tile"][gy, gx] = 1.0
        p = game.placements.get(cid, {"placement": 0.5})
        raw["choice"] = choice
        raw["cond"] = placement_bucket(float(p["placement"]))
        return [raw]

    def _step_samples(self, game: CachedGame, step: dict) -> list[dict]:
        tick = step["tick"]
        entities = game.entities(tick)
        builder = self._builder(game)
        placements = game.placements

        actors = [c for c, ls in step["labels"].items() if ls and c in step["legal"]]
        noops = [c for c in step["legal"] if c not in step["labels"]]
        take = actors[:]
        n_noop = max(1, int(len(take) * self.noop_frac)) if noops else 0
        take += self.rng.sample(noops, min(n_noop, len(noops)))

        out = []
        for cid in take:
            leg = step["legal"][cid]
            raw = builder.prepare(self._obs(game, tick, entities, leg, spawn=False))
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
        game = self.rng.choice(self.games)
        if game.n_steps() < 1:
            return []

        # Prefer windows ending on an acted step (same act/noop balance
        # logic as sample_step, applied to window ends).
        want_actor = self.rng.random() > self.noop_frac
        ends = list(range(game.n_steps()))
        self.rng.shuffle(ends)
        for end in ends[:16]:
            step = game.step(end)
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
            window = [game.step(i) if i != end else step for i in idxs]
            if any(cid not in s["legal"] for s in window):
                continue  # player wasn't alive across the whole window
            out = []
            for j, s in enumerate(window):
                sample = self._one_sample(game, s, cid, labeled=(j == k - 1))
                if sample is None:
                    break
                out.append(sample)
            if len(out) == k:
                return out
        return []

    def _one_sample(
        self, game: CachedGame, step: dict, cid: str, labeled: bool
    ) -> dict | None:
        tick = step["tick"]
        entities = game.entities(tick)
        leg = step["legal"][cid]
        raw = self._builder(game).prepare(
            self._obs(game, tick, entities, leg, spawn=False)
        )
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
        p = game.placements.get(cid, {"placement": 0.5})
        raw["choice"] = choice
        raw["cond"] = placement_bucket(float(p["placement"]))
        return raw
