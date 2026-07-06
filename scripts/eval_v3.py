"""Quick fidelity check for the spatial-only AE (v3).

Reports overall + border tile accuracy and per-class static-structure
detection precision/recall through the latent.

Usage:
  PYTHONPATH=. python scripts/eval_v3.py --ckpt runs/ae_v3/ae_v3.pt --data data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ae.model_v3 import SpatialAE
from ae.train_v3 import CachedDataset
from ae.units import STATIC_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default="eval_out/eval_v3.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = SpatialAE(latent_c=ckpt["args"]["latent_c"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = CachedDataset(args.data, args.crop, seed=1234)
    it = iter(ds)

    tile_ok = tile_n = border_ok = border_n = 0
    tp = np.zeros(len(STATIC_CLASSES))
    fp = np.zeros(len(STATIC_CLASSES))
    fn = np.zeros(len(STATIC_CLASSES))

    B = 32
    for _ in range(args.samples // B):
        batch = [next(it) for _ in range(B)]
        owners = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        terrain = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        planes = torch.from_numpy(np.stack([b[2] for b in batch])).to(device)
        with torch.no_grad():
            tile_logits, unit_logits, _ = model(owners, terrain, planes)
            pred = tile_logits.argmax(1)

        ok = pred == owners
        tile_ok += ok.sum().item()
        tile_n += ok.numel()
        # Border tiles: any 4-neighbour differs in true ownership.
        diff = torch.zeros_like(owners, dtype=torch.bool)
        diff[:, 1:, :] |= owners[:, 1:, :] != owners[:, :-1, :]
        diff[:, :-1, :] |= owners[:, 1:, :] != owners[:, :-1, :]
        diff[:, :, 1:] |= owners[:, :, 1:] != owners[:, :, :-1]
        diff[:, :, :-1] |= owners[:, :, 1:] != owners[:, :, :-1]
        border_ok += (ok & diff).sum().item()
        border_n += diff.sum().item()

        up = (unit_logits > 0).cpu().numpy()
        ut = (planes > 0.5).cpu().numpy()
        tp += (up & ut).sum(axis=(0, 2, 3))
        fp += (up & ~ut).sum(axis=(0, 2, 3))
        fn += (~up & ut).sum(axis=(0, 2, 3))

    report = {
        "tile_acc": round(tile_ok / tile_n, 4),
        "border_acc": round(border_ok / max(1, border_n), 4),
    }
    for i, name in enumerate(STATIC_CLASSES):
        report[f"{name}_precision"] = round(tp[i] / max(1.0, tp[i] + fp[i]), 4)
        report[f"{name}_recall"] = round(tp[i] / max(1.0, tp[i] + fn[i]), 4)
        report[f"{name}_true_cells"] = int(tp[i] + fn[i])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
