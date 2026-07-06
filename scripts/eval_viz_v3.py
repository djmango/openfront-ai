"""Visuals for the spatial-only AE (v3): loss curve, reconstructions, PCA.

Produces under --out:
  - loss_curve_v3.png   loss + tile-acc over training (parsed from log)
  - recon_v3_<map>.png  original | reconstruction, full map
  - latent_pca_v3_<map>.png

Usage:
  PYTHONPATH=. python scripts/eval_viz_v3.py --ckpt runs/ae_v3/ae_v3.pt --log train_v3.log --data data
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ae.model_v3 import MAX_SLOTS, NUM_STATIC, SpatialAE
from ae.train_v3 import CachedGame
from ae.units import STATIC_INDICES

RENDER_MAPS = ["onion", "world", "africa"]
SNAPSHOT_FRAC = 0.6

IS_LAND_BIT = 7
MAGNITUDE_MASK = 0x1F
MAGNITUDE_NORM = 31.0


def plot_curves(log_path: str, out: Path) -> None:
    steps, losses, accs = [], [], []
    pat = re.compile(r"step\s+(\d+)\s+loss ([\d.]+)\s+.*acc ([\d.]+)")
    for line in Path(log_path).read_text().splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(steps, losses, color="tab:red", lw=1)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("total loss (border-weighted CE + unit BCE)", color="tab:red")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(steps, accs, color="tab:blue", lw=1)
    ax2.set_ylabel("tile accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.02)
    ax1.set_title(
        "Spatial AE v3 (10 maps, 256x256 crops, batch 64, RTX 5090, ~700 ex/s)"
    )
    fig.tight_layout()
    fig.savefig(out / "loss_curve_v3.png", dpi=120)
    plt.close(fig)
    print(f"loss curve: {len(steps)} points")


def render_map(model: SpatialAE, game_dir: Path, out: Path, map_name: str) -> None:
    game = CachedGame(game_dir)
    si = int(game.n * SNAPSHOT_FRAC)
    slots_full, _ = game.frame(si)
    h16, w16 = game.h - game.h % 16, game.w - game.w % 16
    slots = slots_full[:h16, :w16].astype(np.int64)

    terr = game.terr[:h16, :w16]
    fallout = np.zeros_like(slots, dtype=np.float32)
    terrain = np.stack(
        [
            ((terr >> IS_LAND_BIT) & 1).astype(np.float32),
            ((terr & MAGNITUDE_MASK) / MAGNITUDE_NORM).astype(np.float32),
            fallout,
        ]
    )

    g_h, g_w = h16 // 16, w16 // 16
    planes = np.zeros((NUM_STATIC, g_h, g_w), dtype=np.float32)
    lo, hi = game.unit_offsets[si], game.unit_offsets[si + 1]
    rows = np.asarray(game.units[lo:hi])
    m = game.static_mask[lo:hi]
    if m.any():
        rows = rows[m]
        cls = game.static_class[lo:hi][m]
        gx, gy = rows[:, 2] // 16, rows[:, 3] // 16
        ok = (gx < g_w) & (gy < g_h)
        np.add.at(planes, (cls[ok], gy[ok], gx[ok]), 1.0)
    planes = np.minimum(planes, 1.0)

    with torch.no_grad():
        tile_logits, _, z = model(
            torch.from_numpy(slots[None]),
            torch.from_numpy(terrain[None]),
            torch.from_numpy(planes[None]),
        )
    pred = tile_logits.argmax(dim=1)[0].numpy()

    rng = np.random.default_rng(0)
    pal = rng.integers(50, 255, size=(MAX_SLOTS, 3), dtype=np.uint8)
    pal[0] = (38, 38, 48)
    gap = np.full((h16, 8, 3), 255, dtype=np.uint8)
    plt.imsave(
        out / f"recon_v3_{map_name}.png",
        np.concatenate([pal[slots], gap, pal[pred]], axis=1),
    )

    zg = z[0].numpy().reshape(z.shape[1], -1).T
    zg = zg - zg.mean(axis=0)
    _, _, vt = np.linalg.svd(zg, full_matrices=False)
    pcs = (zg @ vt[:3].T).reshape(g_h, g_w, 3)
    plo, phi = np.percentile(pcs, [2, 98], axis=(0, 1))
    pcs = np.clip((pcs - plo) / (phi - plo + 1e-9), 0, 1)
    plt.imsave(out / f"latent_pca_v3_{map_name}.png", pcs)

    acc = float((pred == slots).mean())
    border = np.zeros_like(slots, dtype=bool)
    border[1:, :] |= slots[1:, :] != slots[:-1, :]
    border[:-1, :] |= slots[1:, :] != slots[:-1, :]
    border[:, 1:] |= slots[:, 1:] != slots[:, :-1]
    border[:, :-1] |= slots[:, 1:] != slots[:, :-1]
    bacc = float((pred[border] == slots[border]).mean())
    print(f"{map_name}: acc={acc:.4f} border_acc={bacc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--log", default="train_v3.log")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(args.log, out)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SpatialAE(latent_c=ckpt["args"]["latent_c"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for map_name in RENDER_MAPS:
        dirs = sorted((Path(args.data) / map_name).glob("*/cache/index.json"))
        if not dirs:
            print(f"skip {map_name}")
            continue
        render_map(model, dirs[0].parent.parent, out, map_name)


if __name__ == "__main__":
    main()
