"""Evaluation visuals + metrics for a trained checkpoint.

Produces, under --out:
  - loss_curve.png        loss + tile-acc over training steps (parsed from log)
  - recon_<map>.png       original | reconstruction side-by-side, full map
  - latent_pca_<map>.png  top-3 PCA components of the latent grid as RGB
  - metrics.json          overall + border-tile accuracy per map

Usage:
  python scripts/eval_viz.py --ckpt runs/ae_v1/ae.pt --log train_v1.log --data data --out eval_out
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ae.dataset import load_game
from ae.model import MAX_SLOTS, TileAutoencoder
from ae.train import owner_slots, terrain_channels

RENDER_MAPS = ["onion", "world", "africa"]
SNAPSHOT_FRAC = 0.6  # mid-late game: plenty of players, complex borders


def plot_curves(log_path: str, out: Path) -> None:
    steps, losses, accs = [], [], []
    pat = re.compile(r"step\s+(\d+)\s+loss ([\d.]+)\s+tile-acc ([\d.]+)")
    for line in Path(log_path).read_text().splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(steps, losses, color="tab:red", lw=1)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("border-weighted CE loss", color="tab:red")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(steps, accs, color="tab:blue", lw=1)
    ax2.set_ylabel("tile accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.02)
    ax1.set_title("Tile autoencoder training (10 maps, 256x256 crops, RTX 3070)")
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"loss curve: {len(steps)} points")


def palette() -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(50, 255, size=(MAX_SLOTS, 3), dtype=np.uint8)
    pal[0] = (38, 38, 48)
    return pal


def eval_map(
    model: TileAutoencoder, game_dir: Path, out: Path, map_name: str
) -> dict:
    rec = load_game(game_dir)
    si = int(rec.num_snapshots * SNAPSHOT_FRAC)
    slots = owner_slots(rec, si)
    terr = terrain_channels(rec, si)
    h, w = slots.shape
    h16, w16 = h - h % 16, w - w % 16
    slots = slots[:h16, :w16]
    owners_t = torch.from_numpy(slots[None])
    terrain_t = torch.from_numpy(terr[None, :, :h16, :w16])

    with torch.no_grad():
        logits, z = model(owners_t, terrain_t)
    pred = logits.argmax(dim=1)[0].numpy()

    # Metrics: overall and border-tile accuracy
    border = np.zeros_like(slots, dtype=bool)
    border[1:, :] |= slots[1:, :] != slots[:-1, :]
    border[:-1, :] |= slots[1:, :] != slots[:-1, :]
    border[:, 1:] |= slots[:, 1:] != slots[:, :-1]
    border[:, :-1] |= slots[:, 1:] != slots[:, :-1]
    acc = float((pred == slots).mean())
    border_acc = float((pred[border] == slots[border]).mean())

    pal = palette()
    gap = np.full((h16, 8, 3), 255, dtype=np.uint8)
    img = np.concatenate([pal[slots], gap, pal[pred]], axis=1)
    plt.imsave(out / f"recon_{map_name}.png", img)

    # Latent PCA -> RGB
    zg = z[0].numpy().reshape(z.shape[1], -1).T  # (H/16*W/16, C)
    zg = zg - zg.mean(axis=0)
    _, _, vt = np.linalg.svd(zg, full_matrices=False)
    pcs = (zg @ vt[:3].T).reshape(h16 // 16, w16 // 16, 3)
    lo, hi = np.percentile(pcs, [2, 98], axis=(0, 1))
    pcs = np.clip((pcs - lo) / (hi - lo + 1e-9), 0, 1)
    plt.imsave(out / f"latent_pca_{map_name}.png", pcs)

    n_players = sum(p["alive"] for p in rec.meta["snapshots"][si]["players"])
    print(f"{map_name}: acc={acc:.4f} border_acc={border_acc:.4f} players={n_players}")
    return {
        "map": map_name,
        "tick": rec.meta["snapshots"][si]["tick"],
        "alive_players": n_players,
        "tile_acc": acc,
        "border_tile_acc": border_acc,
        "latent_shape": list(z[0].shape),
        "grid_shape": [h16, w16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ae_v1/ae.pt")
    ap.add_argument("--log", default="train_v1.log")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plot_curves(args.log, out)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TileAutoencoder(latent_c=ckpt["args"]["latent_c"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics = []
    for map_name in RENDER_MAPS:
        game_dirs = sorted((Path(args.data) / map_name).glob("*/meta.json"))
        if not game_dirs:
            print(f"skip {map_name}: no games")
            continue
        metrics.append(eval_map(model, game_dirs[0].parent, out, map_name))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
