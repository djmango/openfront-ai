"""Train the tile autoencoder on headless-game snapshots.

Usage:
  uv run python ae/train.py --data data --steps 2000 --crop 256
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ae.dataset import GameRecord, iter_games, load_game
from ae.model import MAX_SLOTS, TileAutoencoder

MAGNITUDE_NORM = 31.0


_slot_lut_cache: dict[Path, np.ndarray] = {}


def slot_lut(record: GameRecord) -> np.ndarray:
    """LUT from raw owner smallID -> static per-game slot (1..MAX_SLOTS-1).

    Slots are assigned by ascending smallID once per game, so a player keeps
    the same slot for every snapshot of that game regardless of territory.
    """
    cached = _slot_lut_cache.get(record.game_dir)
    if cached is not None:
        return cached
    small_ids = sorted(p["id"] for p in record.players(0))
    lut = np.zeros(4096, dtype=np.int64)
    for slot, sid in enumerate(small_ids, start=1):
        lut[sid] = min(slot, MAX_SLOTS - 1)
    _slot_lut_cache[record.game_dir] = lut
    return lut


def owner_slots(record: GameRecord, snapshot_idx: int) -> np.ndarray:
    return slot_lut(record)[record.owners(snapshot_idx)]


def terrain_channels(record: GameRecord, snapshot_idx: int) -> np.ndarray:
    land = record.land().astype(np.float32)
    mag = (record.magnitude() / MAGNITUDE_NORM).astype(np.float32)
    fallout = record.fallout(snapshot_idx).astype(np.float32)
    return np.stack([land, mag, fallout])


def encode_crop(
    record: GameRecord, snapshot_idx: int, y0: int, x0: int, crop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (owner-slot, terrain-channels) arrays for one crop.

    Decompresses the full snapshot (unavoidable with gzip) but does the
    LUT and float conversions on the crop only.
    """
    from ae.dataset import FALLOUT_BIT, IS_LAND_BIT, MAGNITUDE_MASK, OWNER_MASK

    state = record.state(snapshot_idx)[y0 : y0 + crop, x0 : x0 + crop]
    terr = record.terrain[y0 : y0 + crop, x0 : x0 + crop]
    slots = slot_lut(record)[state & OWNER_MASK]
    land = ((terr >> IS_LAND_BIT) & 1).astype(np.float32)
    mag = ((terr & MAGNITUDE_MASK) / MAGNITUDE_NORM).astype(np.float32)
    fallout = ((state >> FALLOUT_BIT) & 1).astype(np.float32)
    return slots, np.stack([land, mag, fallout])


def border_weight(owners: torch.Tensor, weight: float) -> torch.Tensor:
    """Per-tile loss weights: `1 + weight` on ownership borders, 1 elsewhere."""
    w = torch.ones_like(owners, dtype=torch.float32)
    diff = torch.zeros_like(owners, dtype=torch.bool)
    diff[:, 1:, :] |= owners[:, 1:, :] != owners[:, :-1, :]
    diff[:, :-1, :] |= owners[:, 1:, :] != owners[:, :-1, :]
    diff[:, :, 1:] |= owners[:, :, 1:] != owners[:, :, :-1]
    diff[:, :, :-1] |= owners[:, :, 1:] != owners[:, :, :-1]
    return w + weight * diff.float()


class SnapshotSampler:
    """Samples random (game, snapshot, crop) examples.

    gzip decompression dominates sampling cost and releases the GIL, so
    examples are built in a thread pool, and the next batch is prefetched
    while the GPU works on the current one.
    """

    def __init__(
        self, data_root: str, crop: int, seed: int = 0, workers: int = 8
    ):
        self.records = list(iter_games(data_root))
        if not self.records:
            raise SystemExit(f"no games found under {data_root}")
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self._pending = None
        total = sum(len(r.meta["snapshots"]) for r in self.records)
        print(f"dataset: {len(self.records)} games, {total} snapshots")

    def _sample_one(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        rec = self.records[rng.integers(len(self.records))]
        si = int(rng.integers(len(rec.meta["snapshots"])))
        y0 = int(rng.integers(max(1, rec.height - self.crop + 1)))
        x0 = int(rng.integers(max(1, rec.width - self.crop + 1)))
        return encode_crop(rec, si, y0, x0, self.crop)

    def _submit(self, batch_size: int):
        seeds = self.rng.integers(0, 2**63, size=batch_size)
        return [self.pool.submit(self._sample_one, int(s)) for s in seeds]

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pending is None:
            self._pending = self._submit(batch_size)
        futures = self._pending
        self._pending = self._submit(batch_size)  # prefetch next batch

        owners_out = np.empty((batch_size, self.crop, self.crop), dtype=np.int64)
        terrain_out = np.empty(
            (batch_size, 3, self.crop, self.crop), dtype=np.float32
        )
        for b, fut in enumerate(futures):
            owners_out[b], terrain_out[b] = fut.result()
        return torch.from_numpy(owners_out), torch.from_numpy(terrain_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--latent-c", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--border-weight", type=float, default=4.0)
    parser.add_argument("--out", default="runs/ae")
    args = parser.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    sampler = SnapshotSampler(args.data, args.crop)
    model = TileAutoencoder(latent_c=args.latent_c).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, latent_c={args.latent_c}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        owners, terrain = sampler.sample_batch(args.batch_size)
        owners = owners.to(device)
        terrain = terrain.to(device)

        logits, _ = model(owners, terrain)
        per_tile = F.cross_entropy(logits, owners, reduction="none")
        weights = border_weight(owners, args.border_weight)
        loss = (per_tile * weights).sum() / weights.sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == owners).float().mean().item()
            rate = step * args.batch_size / (time.time() - t0)
            print(
                f"step {step:5d}  loss {loss.item():.4f}  "
                f"tile-acc {acc:.4f}  {rate:.1f} ex/s",
                flush=True,
            )

        if step % 500 == 0 or step == args.steps:
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                out_dir / "ae.pt",
            )

    print(f"saved {out_dir / 'ae.pt'}")


def render_reconstruction(
    checkpoint: str, game_dir: str, snapshot_idx: int, out_path: str
) -> None:
    """Reconstruct a full map snapshot and save original|reconstruction PNG."""
    from PIL import Image

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TileAutoencoder(latent_c=ckpt["args"]["latent_c"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rec = load_game(game_dir)
    slots = owner_slots(rec, snapshot_idx)
    terr = terrain_channels(rec, snapshot_idx)
    h, w = slots.shape
    h16, w16 = h - h % 16, w - w % 16
    owners = torch.from_numpy(slots[None, :h16, :w16])
    terrain = torch.from_numpy(terr[None, :, :h16, :w16])
    with torch.no_grad():
        logits, z = model(owners, terrain)
    pred = logits.argmax(dim=1)[0].numpy()

    rng = np.random.default_rng(0)
    palette = rng.integers(50, 255, size=(MAX_SLOTS, 3), dtype=np.uint8)
    palette[0] = (40, 40, 40)

    both = np.concatenate([slots[:h16, :w16], pred], axis=1)
    img = palette[both]
    Image.fromarray(img).save(out_path)
    ratio = (h16 * w16 * 16) / z[0].numel()
    print(f"wrote {out_path} (compression {ratio:.0f}x vs uint16 grid)")


if __name__ == "__main__":
    main()
