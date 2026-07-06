"""Train the unified state autoencoder (tiles + units + players).

Usage:
  uv run python -m ae.train_v2 --data data --steps 20000
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ae.dataset import GameRecord, iter_games
from ae.model_v2 import (
    MAX_SLOTS,
    NUM_DIPLO,
    NUM_UNIT_CLASSES,
    PLAYER_FEAT_DIM,
    UNIT_CLASS_WEIGHTS,
    UNIT_CLASSES,
    UnifiedStateAE,
)
from ae.train import border_weight, slot_lut

UNIT_CLASS_INDEX = {name: i for i, name in enumerate(UNIT_CLASSES)}
MAGNITUDE_NORM = 31.0


def log_norm(x: float) -> float:
    """log10(1+x) squashed to roughly [0, 1] for troops/gold magnitudes."""
    return float(np.log10(1.0 + max(0.0, x)) / 8.0)


class Featurizer:
    """Builds model inputs for (game, snapshot, crop) samples."""

    def __init__(self, rec: GameRecord):
        self.rec = rec
        self.lut = slot_lut(rec)
        self.land_tiles = max(1, int(rec.land().sum()))

    def spatial(self, si: int, y0: int, x0: int, crop: int):
        from ae.dataset import FALLOUT_BIT, IS_LAND_BIT, MAGNITUDE_MASK, OWNER_MASK

        state = self.rec.state(si)[y0 : y0 + crop, x0 : x0 + crop]
        terr = self.rec.terrain[y0 : y0 + crop, x0 : x0 + crop]
        slots = self.lut[state & OWNER_MASK]
        land = ((terr >> IS_LAND_BIT) & 1).astype(np.float32)
        mag = ((terr & MAGNITUDE_MASK) / MAGNITUDE_NORM).astype(np.float32)
        fallout = ((state >> FALLOUT_BIT) & 1).astype(np.float32)
        return slots, np.stack([land, mag, fallout])

    def unit_planes(
        self, entities: dict, y0: int, x0: int, crop: int
    ) -> np.ndarray:
        g = crop // 16
        planes = np.zeros((NUM_UNIT_CLASSES, g, g), dtype=np.float32)
        for u in entities["units"]:
            ci = UNIT_CLASS_INDEX.get(u["type"])
            if ci is None:
                continue
            gy, gx = (u["y"] - y0) // 16, (u["x"] - x0) // 16
            if 0 <= gy < g and 0 <= gx < g:
                planes[ci, gy, gx] += 1.0
        return np.log1p(planes)

    def player_feats(
        self, entities: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = np.zeros((MAX_SLOTS, PLAYER_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(MAX_SLOTS, dtype=np.float32)
        # Pairwise diplomacy targets: [allied, embargoes] per (i, j).
        self._diplo = np.zeros((NUM_DIPLO, MAX_SLOTS, MAX_SLOTS), dtype=np.float32)

        n_allies: dict[int, int] = {}
        for a, b, _exp in entities["alliances"]:
            n_allies[a] = n_allies.get(a, 0) + 1
            n_allies[b] = n_allies.get(b, 0) + 1
            sa, sb = int(self.lut[a]), int(self.lut[b])
            self._diplo[0, sa, sb] = 1.0
            self._diplo[0, sb, sa] = 1.0
        atk_out: dict[int, float] = {}
        atk_in: dict[int, float] = {}
        for atk in entities["attacks"]:
            atk_out[atk["from"]] = atk_out.get(atk["from"], 0.0) + atk["troops"]
            atk_in[atk["to"]] = atk_in.get(atk["to"], 0.0) + atk["troops"]

        for p in entities["players"]:
            slot = int(self.lut[p["id"]])
            if slot <= 0:
                continue
            mask[slot] = 1.0
            for e in p.get("embargoes", []):
                self._diplo[1, slot, int(self.lut[e])] = 1.0
            feats[slot] = [
                1.0 if p["alive"] else 0.0,
                log_norm(p["troops"]),
                log_norm(float(p["gold"])),
                p["tiles"] / self.land_tiles,
                1.0 if p.get("traitor") else 0.0,
                1.0 if p.get("disconnected") else 0.0,
                n_allies.get(p["id"], 0) / 8.0,
                len(p.get("embargoes", [])) / 8.0,
                (len(p.get("reqsIn", [])) + len(p.get("reqsOut", []))) / 4.0,
                log_norm(atk_out.get(p["id"], 0.0)),
                log_norm(atk_in.get(p["id"], 0.0)),
            ]
        return feats, mask, self._diplo


def sample_example(featurizers: list[Featurizer], rng, crop: int):
    """One random (game, snapshot, crop) example as numpy arrays."""
    fz = featurizers[rng.integers(len(featurizers))]
    rec = fz.rec
    si = int(rng.integers(rec.num_snapshots))
    # Crop origin snapped to 16 so unit planes align with the latent grid.
    y0 = int(rng.integers(max(1, (rec.height - crop) // 16 + 1))) * 16
    x0 = int(rng.integers(max(1, (rec.width - crop) // 16 + 1))) * 16
    slots, terr = fz.spatial(si, y0, x0, crop)
    entities = rec.entities(si)
    planes = fz.unit_planes(entities, y0, x0, crop)
    pfeats, pmask, diplo = fz.player_feats(entities)
    return slots, terr, planes, pfeats, pmask, diplo


class SampleDataset(torch.utils.data.IterableDataset):
    """Infinite stream of random samples; each DataLoader worker process
    loads its own featurizer list and RNG, sidestepping the GIL entirely
    (gzip decompress + numpy featurization are the bottleneck, not the GPU).
    """

    def __init__(self, data_root: str, crop: int, seed: int = 0):
        self.data_root = data_root
        self.crop = crop
        self.seed = seed
        self._featurizers: list[Featurizer] | None = None

    def __iter__(self):
        if self._featurizers is None:
            self._featurizers = [Featurizer(r) for r in iter_games(self.data_root)]
            if not self._featurizers:
                raise SystemExit(f"no games found under {self.data_root}")
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed * 100_003 + wid)
        while True:
            yield sample_example(self._featurizers, rng, self.crop)


class SamplerV2:
    """Thread-pool sampler kept for eval scripts (no worker processes)."""

    def __init__(self, data_root: str, crop: int, seed: int = 0, workers: int = 8):
        self.featurizers = [Featurizer(r) for r in iter_games(data_root)]
        if not self.featurizers:
            raise SystemExit(f"no games found under {data_root}")
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self._pending = None
        total = sum(f.rec.num_snapshots for f in self.featurizers)
        print(f"dataset: {len(self.featurizers)} games, {total} snapshots")

    def _sample_one(self, seed: int):
        rng = np.random.default_rng(seed)
        return sample_example(self.featurizers, rng, self.crop)

    def _submit(self, n: int):
        seeds = self.rng.integers(0, 2**63, size=n)
        return [self.pool.submit(self._sample_one, int(s)) for s in seeds]

    def sample_batch(self, n: int):
        if self._pending is None:
            self._pending = self._submit(n)
        futures = self._pending
        self._pending = self._submit(n)

        c, g = self.crop, self.crop // 16
        owners = np.empty((n, c, c), dtype=np.int64)
        terrain = np.empty((n, 3, c, c), dtype=np.float32)
        planes = np.empty((n, NUM_UNIT_CLASSES, g, g), dtype=np.float32)
        pfeats = np.empty((n, MAX_SLOTS, PLAYER_FEAT_DIM), dtype=np.float32)
        pmask = np.empty((n, MAX_SLOTS), dtype=np.float32)
        diplo = np.empty((n, NUM_DIPLO, MAX_SLOTS, MAX_SLOTS), dtype=np.float32)
        for b, fut in enumerate(futures):
            owners[b], terrain[b], planes[b], pfeats[b], pmask[b], diplo[b] = (
                fut.result()
            )
        return tuple(
            torch.from_numpy(a)
            for a in (owners, terrain, planes, pfeats, pmask, diplo)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    # Curve from the 20k run plateaued by ~8-10k steps on every head.
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--latent-c", type=int, default=64)
    ap.add_argument("--latent-d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--border-weight", type=float, default=4.0)
    ap.add_argument("--w-units", type=float, default=1.0)
    ap.add_argument("--w-players", type=float, default=5.0)
    ap.add_argument("--w-diplo", type=float, default=1.0)
    # 50.0 gave recall 0.96 but precision 0.13 (predicted alliances everywhere).
    ap.add_argument("--diplo-pos-weight", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    # Base pos-weight for unit occupancy BCE, multiplied by per-class rarity
    # weights. MSE on counts collapsed to all-zeros (recall 0.0 on every class).
    ap.add_argument("--unit-pos-weight", type=float, default=20.0)
    ap.add_argument("--out", default="runs/ae_v2")
    args = ap.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    dataset = SampleDataset(args.data, args.crop)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
        pin_memory=device == "cuda",
    )
    batches = iter(loader)
    model = UnifiedStateAE(latent_c=args.latent_c, latent_d=args.latent_d).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_pos_w = (
        torch.tensor(UNIT_CLASS_WEIGHTS, device=device).view(1, -1, 1, 1)
        * args.unit_pos_weight
    )
    diplo_pos_w = torch.tensor(args.diplo_pos_weight, device=device)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        owners, terrain, planes, pfeats, pmask, diplo = (
            t.to(device, non_blocking=True) for t in next(batches)
        )

        (tile_logits, unit_pred, player_pred, diplo_logits), _ = model(
            owners, terrain, planes, pfeats, pmask, diplo
        )

        per_tile = F.cross_entropy(tile_logits, owners, reduction="none")
        weights = border_weight(owners, args.border_weight)
        loss_tiles = (per_tile * weights).sum() / weights.sum()

        # Unit occupancy as detection: BCE with rarity-scaled positive weight
        # so the head can't win by predicting "no units anywhere".
        unit_occ = (planes > 0).float()
        loss_units = F.binary_cross_entropy_with_logits(
            unit_pred, unit_occ, pos_weight=unit_pos_w
        )

        per_player = F.mse_loss(player_pred, pfeats, reduction="none").mean(-1)
        loss_players = (per_player * pmask).sum() / pmask.sum().clamp(min=1)

        # Pairwise diplomacy BCE over pairs of existing slots, positives
        # heavily upweighted (alliances are ~0.1% of pairs).
        pair_mask = (pmask.unsqueeze(2) * pmask.unsqueeze(1)).unsqueeze(1)
        per_pair = F.binary_cross_entropy_with_logits(
            diplo_logits, diplo, pos_weight=diplo_pos_w, reduction="none"
        )
        loss_diplo = (per_pair * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        loss = (
            loss_tiles
            + args.w_units * loss_units
            + args.w_players * loss_players
            + args.w_diplo * loss_diplo
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (tile_logits.argmax(1) == owners).float().mean().item()
                # Alliance recall/precision on this batch.
                ally_true = diplo[:, 0] > 0.5
                ally_pred = (diplo_logits[:, 0] > 0) & (pair_mask[:, 0] > 0)
                n_true = ally_true.sum().item()
                n_pred = ally_pred.sum().item()
                tp = (ally_pred & ally_true).sum().item()
                ally_rec = tp / n_true if n_true else float("nan")
                ally_prec = tp / n_pred if n_pred else float("nan")
                # Unit-cell recall: of cells truly containing units, fraction hit.
                occ_true = unit_occ > 0.5
                n_occ = occ_true.sum().item()
                unit_rec = (
                    ((unit_pred > 0) & occ_true).sum().item() / n_occ
                    if n_occ
                    else float("nan")
                )
            rate = step * args.batch_size / (time.time() - t0)
            print(
                f"step {step:5d}  loss {loss.item():.4f}  "
                f"tiles {loss_tiles.item():.4f}  units {loss_units.item():.4f}  "
                f"players {loss_players.item():.4f}  diplo {loss_diplo.item():.4f}  "
                f"acc {acc:.4f}  ally-rec {ally_rec:.2f}  ally-prec {ally_prec:.2f}  "
                f"unit-rec {unit_rec:.2f}  {rate:.1f} ex/s",
                flush=True,
            )

        if step % 500 == 0 or step == args.steps:
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                out_dir / "ae_v2.pt",
            )

    print(f"saved {out_dir / 'ae_v2.pt'}")


if __name__ == "__main__":
    main()
