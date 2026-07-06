"""One-time featurization of snapshot data into fast-decoding caches.

The gzip+JSON pipeline costs ~10ms of CPU per sample, which caps training
at ~500 ex/s even with worker processes. This pass converts each game to:

  cache/frames.zst   per-snapshot zstd-1 frames, concatenated. Each frame is
                     owner-slot uint8 (h*w) ++ packbits fallout (h*ceil(w/8)).
                     zstd-1 decodes at GB/s: ~0.5ms per frame vs ~10ms before
                     (raw memmaps would be ideal but 250 games of uint8 grids
                     is ~600GB; zstd-1 is ~50x smaller).
  cache/units.npy    int32 (m, 7): snap_idx, class, x, y, tx, ty, owner_slot
                     (tx/ty = -1 when the unit has no target tile)
  cache/index.json   shapes + frame byte offsets + per-snapshot unit offsets

Usage:
  PYTHONPATH=. python scripts/prefeaturize.py --data data [--workers 8]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

from ae.dataset import FALLOUT_BIT, OWNER_MASK, load_game
from ae.model import MAX_SLOTS
from ae.train import slot_lut
from ae.units import UNIT_CLASS_INDEX


def build_cache(game_dir: Path) -> str:
    rec = load_game(game_dir)
    cache = game_dir / "cache"
    if (cache / "index.json").exists():
        return f"skip {game_dir.name}"
    cache.mkdir(exist_ok=True)

    lut8 = slot_lut(rec).astype(np.uint8)  # slots < MAX_SLOTS=128 fit in u8
    lut = slot_lut(rec)
    n, h, w = rec.num_snapshots, rec.height, rec.width

    comp = zstd.ZstdCompressor(level=1)
    frame_offsets = np.zeros(n + 1, dtype=np.int64)
    unit_rows: list[np.ndarray] = []
    unit_offsets = np.zeros(n + 1, dtype=np.int64)

    with open(cache / "frames.zst", "wb") as fh:
        for i in range(n):
            state = rec.state(i)
            slots = lut8[state & OWNER_MASK]
            fallout = np.packbits((state >> FALLOUT_BIT) & 1, axis=1)
            frame = slots.tobytes() + fallout.tobytes()
            blob = comp.compress(frame)
            fh.write(blob)
            frame_offsets[i + 1] = frame_offsets[i] + len(blob)

            ents = rec.entities(i)
            rows = []
            for u in ents.get("units", []):
                ci = UNIT_CLASS_INDEX.get(u["type"])
                if ci is None:
                    continue
                rows.append(
                    (
                        i,
                        ci,
                        u["x"],
                        u["y"],
                        u.get("tx") if u.get("tx") is not None else -1,
                        u.get("ty") if u.get("ty") is not None else -1,
                        int(lut[u["owner"]]),
                    )
                )
            arr = np.array(rows, dtype=np.int32).reshape(-1, 7)
            unit_rows.append(arr)
            unit_offsets[i + 1] = unit_offsets[i] + len(arr)

    units = (
        np.concatenate(unit_rows) if unit_rows else np.zeros((0, 7), dtype=np.int32)
    )
    np.save(cache / "units.npy", units)
    (cache / "index.json").write_text(
        json.dumps(
            {
                "n": n,
                "h": h,
                "w": w,
                "max_slots": MAX_SLOTS,
                "frame_offsets": frame_offsets.tolist(),
                "unit_offsets": unit_offsets.tolist(),
            }
        )
    )
    return f"done {game_dir.name}: {n} snaps, {len(units)} unit rows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    game_dirs = sorted(p.parent for p in Path(args.data).rglob("meta.json"))
    print(f"{len(game_dirs)} games")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(build_cache, game_dirs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
