"""Train the spatial-only AE (v3) from prefeaturized caches.

Requires scripts/prefeaturize.py to have been run on the data root first.
Per-sample cost is an mmap slice + a few small numpy ops (~microseconds),
so the pipeline no longer caps GPU throughput.

Usage:
  python -m ae.train_v3 --data data --steps 10000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd

from ae.dataset import IS_LAND_BIT, MAGNITUDE_MASK
from ae.model_v3 import (
    NUM_STATIC,
    STATIC_CLASS_WEIGHTS,
    SpatialAE,
)
from ae.train import border_weight
from ae.units import STATIC_INDICES

MAGNITUDE_NORM = 31.0


class CachedGame:
    """View over one game's prefeaturized cache (zstd-1 frames, mmapped)."""

    def __init__(self, game_dir: Path):
        cache = game_dir / "cache"
        idx = json.loads((cache / "index.json").read_text())
        self.n, self.h, self.w = idx["n"], idx["h"], idx["w"]
        self.frames = np.memmap(cache / "frames.zst", dtype=np.uint8, mode="r")
        self.frame_offsets = np.asarray(idx["frame_offsets"], dtype=np.int64)
        self.units = np.load(cache / "units.npy", mmap_mode="r")
        self.unit_offsets = np.asarray(idx["unit_offsets"], dtype=np.int64)
        self._dctx: zstd.ZstdDecompressor | None = None
        # Terrain kept as raw bytes; float conversion happens on crops only
        # (full-map float32 planes were ~16MB/game and OOM-killed workers).
        self.terr = np.fromfile(game_dir / "terrain.bin", dtype=np.uint8).reshape(
            self.h, self.w
        )
        # Static-structure units only, pre-sorted by snapshot via offsets.
        units_cls = np.asarray(self.units[:, 1])
        self.static_mask = np.isin(units_cls, STATIC_INDICES)
        self.static_class = np.searchsorted(
            STATIC_INDICES, units_cls
        )  # valid where static_mask

    def frame(self, si: int) -> tuple[np.ndarray, np.ndarray]:
        """(owner slots uint8 (h, w), packed fallout (h, w/8)) for snapshot si."""
        if self._dctx is None:  # per-process; zstd contexts don't fork/pickle
            self._dctx = zstd.ZstdDecompressor()
        lo, hi = self.frame_offsets[si], self.frame_offsets[si + 1]
        raw = self._dctx.decompress(
            self.frames[lo:hi].tobytes(), max_output_size=self.h * self.w * 2
        )
        hw = self.h * self.w
        slots = np.frombuffer(raw[:hw], dtype=np.uint8).reshape(self.h, self.w)
        packed = np.frombuffer(raw[hw:], dtype=np.uint8).reshape(self.h, -1)
        return slots, packed

    def sample(self, rng: np.random.Generator, crop: int):
        si = int(rng.integers(self.n))
        y0 = int(rng.integers(max(1, (self.h - crop) // 16 + 1))) * 16
        x0 = int(rng.integers(max(1, (self.w - crop) // 16 + 1))) * 16

        slots_full, fall_full = self.frame(si)
        slots = slots_full[y0 : y0 + crop, x0 : x0 + crop].astype(np.int64)
        fall_packed = fall_full[y0 : y0 + crop, x0 // 8 : (x0 + crop) // 8]
        fallout = np.unpackbits(fall_packed, axis=1).astype(np.float32)
        terr = self.terr[y0 : y0 + crop, x0 : x0 + crop]
        terrain = np.stack(
            [
                ((terr >> IS_LAND_BIT) & 1).astype(np.float32),
                ((terr & MAGNITUDE_MASK) / MAGNITUDE_NORM).astype(np.float32),
                fallout,
            ]
        )

        g = crop // 16
        planes = np.zeros((NUM_STATIC, g, g), dtype=np.float32)
        lo, hi = self.unit_offsets[si], self.unit_offsets[si + 1]
        rows = self.units[lo:hi]
        m = self.static_mask[lo:hi]
        if m.any():
            rows = rows[m]
            cls = self.static_class[lo:hi][m]
            gx = (rows[:, 2] - x0) // 16
            gy = (rows[:, 3] - y0) // 16
            ok = (gx >= 0) & (gx < g) & (gy >= 0) & (gy < g)
            np.add.at(planes, (cls[ok], gy[ok], gx[ok]), 1.0)
        return slots, terrain, np.minimum(planes, 1.0)


class CachedDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_root: str, crop: int, seed: int = 0):
        self.data_root = data_root
        self.crop = crop
        self.seed = seed
        self._games: list[CachedGame] | None = None

    def __iter__(self):
        if self._games is None:
            dirs = sorted(
                p.parent.parent for p in Path(self.data_root).rglob("cache/index.json")
            )
            if not dirs:
                raise SystemExit(
                    f"no caches under {self.data_root}; run scripts/prefeaturize.py"
                )
            self._games = [CachedGame(d) for d in dirs]
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed * 100_003 + wid)
        games = self._games
        while True:
            yield games[rng.integers(len(games))].sample(rng, self.crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--latent-c", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--border-weight", type=float, default=4.0)
    ap.add_argument("--w-units", type=float, default=1.0)
    ap.add_argument("--unit-pos-weight", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/ae_v3")
    args = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    dataset = CachedDataset(args.data, args.crop)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=6 if args.workers else None,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )
    batches = iter(loader)

    model = SpatialAE(latent_c=args.latent_c).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_pos_w = (
        torch.tensor(STATIC_CLASS_WEIGHTS, device=device).view(1, -1, 1, 1)
        * args.unit_pos_weight
    )

    t0 = time.time()
    for step in range(1, args.steps + 1):
        owners, terrain, planes = (
            t.to(device, non_blocking=True) for t in next(batches)
        )

        tile_logits, unit_logits, _ = model(owners, terrain, planes)

        per_tile = F.cross_entropy(tile_logits, owners, reduction="none")
        weights = border_weight(owners, args.border_weight)
        loss_tiles = (per_tile * weights).sum() / weights.sum()

        loss_units = F.binary_cross_entropy_with_logits(
            unit_logits, planes, pos_weight=unit_pos_w
        )
        loss = loss_tiles + args.w_units * loss_units

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (tile_logits.argmax(1) == owners).float().mean().item()
                occ = planes > 0.5
                n_occ = occ.sum().item()
                unit_rec = (
                    ((unit_logits > 0) & occ).sum().item() / n_occ
                    if n_occ
                    else float("nan")
                )
            rate = step * args.batch_size / (time.time() - t0)
            print(
                f"step {step:5d}  loss {loss.item():.4f}  "
                f"tiles {loss_tiles.item():.4f}  units {loss_units.item():.4f}  "
                f"acc {acc:.4f}  unit-rec {unit_rec:.2f}  {rate:.1f} ex/s",
                flush=True,
            )

        if step % 500 == 0 or step == args.steps:
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                out_dir / "ae_v3.pt",
            )

    print(f"saved {out_dir / 'ae_v3.pt'}")


if __name__ == "__main__":
    main()
