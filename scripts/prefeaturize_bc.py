"""One-time featurization of BC human-game data into fast-decoding caches.

The bc.json.gz + gzip-states pipeline costs ~15-20ms of CPU per sample
(gzip grid decompress + entities JSON parse + full-res featurization),
which capped BC training at ~34 ex/s. This pass converts each game with a
bc.json.gz sidecar into cache-bc/:

  frames.zst    per-step zstd-1 frames: strided owner-slot uint8 (hr*wr)
                ++ packbits fallout (hr*wr/8). ~0.3ms to decode.
  ents.zst      per-step zstd-1 orjson entity blobs, coordinates already
                strided. ~0.1ms to decode.
  steps.zst     per-step zstd-1 orjson blobs of the bc step (legality per
                living human + labels). Replaces the monolithic 30MB
                bc.json.gz parse (~250ms) with ~50us per access.
  index.json    dims, stride, byte offsets, placements, spawn steps.
  terrain.npy   strided owner terrain (uint8), trimmed to the region grid.
  lut.npy       smallID -> slot LUT frozen from the first post-spawn step.

Striding happens at build time (Normal-size human maps are 2/4x the
policy's grid budget), so training never touches full-resolution data.
Spawn supervision (sidecar formatVersion 2, datagen/replay.ts) rides along
as extra frames keyed by the spawn ticks.

Usage:
  PYTHONPATH=. python scripts/prefeaturize_bc.py --data data-human [--workers 8]
"""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

try:
    import orjson

    _loads = orjson.loads
    _dumps = orjson.dumps
except ImportError:  # pragma: no cover
    _loads = json.loads
    _dumps = lambda o: json.dumps(o).encode()  # noqa: E731

from ae.model_v3 import MAX_SLOTS
from rl.curriculum import GH_MAX, GW_MAX
from rl.obs import REGION

OWNER_MASK = 0x0FFF
FALLOUT_BIT = 13

CACHE_FORMAT = 1


def pick_stride(h: int, w: int) -> int | None:
    """Smallest power-of-2 stride so the featurized grid fits the policy
    budget. The engine's Compact mode is exactly a 2x-downscaled Normal
    map, so strided games stay in-distribution."""
    ds = 1
    while ds <= 8:
        hs, ws = len(range(0, h, ds)), len(range(0, w, ds))
        gh, gw = (hs - hs % REGION) // REGION, (ws - ws % REGION) // REGION
        if gh <= GH_MAX and gw <= GW_MAX:
            return ds
        ds *= 2
    return None


def scale_entities(entities: dict, ds: int) -> dict:
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


def build_cache(game_dir: Path) -> str:
    cache = game_dir / "cache-bc"
    bc_path = game_dir / "bc.json.gz"
    if not bc_path.exists() or not (game_dir / "meta.json").exists():
        return f"no-sidecar {game_dir.name}"
    idx_path = cache / "index.json"
    # Rebuild when the sidecar was regenerated (e.g. formatVersion upgrade).
    if idx_path.exists() and idx_path.stat().st_mtime >= bc_path.stat().st_mtime:
        return f"skip {game_dir.name}"

    meta = json.loads((game_dir / "meta.json").read_text())
    h, w = meta["height"], meta["width"]
    ds = pick_stride(h, w)
    if ds is None:
        return f"too-big {game_dir.name}"

    bc = _loads(gzip.decompress(bc_path.read_bytes()))
    steps = [s for s in bc["steps"] if _has_state(game_dir, s["tick"])]
    if not steps:
        return f"empty {game_dir.name}"
    spawn_steps = [
        s for s in bc.get("spawn_steps", []) if _has_state(game_dir, s["tick"])
    ]

    hs, ws = len(range(0, h, ds)), len(range(0, w, ds))
    hr, wr = hs - hs % REGION, ws - ws % REGION

    terrain = np.frombuffer((game_dir / "terrain.bin").read_bytes(), dtype=np.uint8)
    terrain = np.ascontiguousarray(terrain.reshape(h, w)[::ds, ::ds][:hr, :wr])

    # Slot LUT frozen from the first post-spawn snapshot: by then every
    # nation/tribe/human exists, so slots are stable across the episode.
    ents0 = _load_entities(game_dir, steps[0]["tick"])
    ids = sorted(p["id"] for p in ents0["players"])
    lut = np.zeros(4096, dtype=np.uint8)
    for slot, sid in enumerate(ids, start=1):
        lut[sid] = min(slot, MAX_SLOTS - 1)

    comp = zstd.ZstdCompressor(level=1)
    all_ticks = [s["tick"] for s in spawn_steps] + [s["tick"] for s in steps]
    frame_off = np.zeros(len(all_ticks) + 1, dtype=np.int64)
    ent_off = np.zeros(len(all_ticks) + 1, dtype=np.int64)
    with open(cache_tmp(cache, "frames.zst"), "wb") as ff, open(
        cache_tmp(cache, "ents.zst"), "wb"
    ) as ef:
        for i, tick in enumerate(all_ticks):
            raw = gzip.decompress(
                (game_dir / "states" / f"t{tick:06d}.bin.gz").read_bytes()
            )
            state = np.frombuffer(raw, dtype="<u2").reshape(h, w)[::ds, ::ds][:hr, :wr]
            slots = lut[state & OWNER_MASK]
            fallout = np.packbits(
                ((state >> FALLOUT_BIT) & 1).astype(np.uint8), axis=1
            )
            blob = comp.compress(slots.tobytes() + fallout.tobytes())
            ff.write(blob)
            frame_off[i + 1] = frame_off[i] + len(blob)

            ents = scale_entities(_load_entities(game_dir, tick), ds)
            eblob = comp.compress(_dumps(ents))
            ef.write(eblob)
            ent_off[i + 1] = ent_off[i] + len(eblob)

    step_off = np.zeros(len(steps) + 1, dtype=np.int64)
    with open(cache_tmp(cache, "steps.zst"), "wb") as sf:
        for i, s in enumerate(steps):
            blob = comp.compress(_dumps(s))
            sf.write(blob)
            step_off[i + 1] = step_off[i] + len(blob)

    # open() handles: np.save would append ".npy" to the ".tmp" names.
    with open(cache_tmp(cache, "terrain.npy"), "wb") as f:
        np.save(f, terrain)
    with open(cache_tmp(cache, "lut.npy"), "wb") as f:
        np.save(f, lut)
    (cache / "index.json.tmp").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "ds": ds,
                "hr": hr,
                "wr": wr,
                "ticks": all_ticks,
                "n_spawn": len(spawn_steps),
                "frame_offsets": frame_off.tolist(),
                "ent_offsets": ent_off.tolist(),
                "step_offsets": step_off.tolist(),
                "placements": bc["placements"],
                "spawn_steps": spawn_steps,
            }
        )
    )
    for name in ("frames.zst", "ents.zst", "steps.zst", "terrain.npy", "lut.npy", "index.json"):
        (cache / f"{name}.tmp").rename(cache / name)
    return f"done {game_dir.name}: {len(steps)} steps, {len(spawn_steps)} spawns, ds={ds}"


def cache_tmp(cache: Path, name: str) -> Path:
    cache.mkdir(exist_ok=True)
    return cache / f"{name}.tmp"


def _has_state(game_dir: Path, tick: int) -> bool:
    return (game_dir / "states" / f"t{tick:06d}.bin.gz").exists()


def _load_entities(game_dir: Path, tick: int) -> dict:
    return _loads(
        gzip.decompress((game_dir / "states" / f"t{tick:06d}.json.gz").read_bytes())
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    game_dirs = sorted(
        {p.parent for root in args.data for p in Path(root).glob("*/*/bc.json.gz")}
    )
    print(f"{len(game_dirs)} games with sidecars")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(build_cache, game_dirs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
