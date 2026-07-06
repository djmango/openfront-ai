"""Semantic-fidelity evaluation of the unified state AE.

Pixel accuracy is not the point; this scores whether *strategically load-
bearing* facts survive the latent bottleneck:

  - alliance / embargo pair precision+recall
  - nuke-in-flight detection recall (any predicted mass in a cell that
    truly contains an airborne nuke)
  - silo / SAM / warship / transport cell detection recall
  - troop-strength ordering: for random alive-player pairs, does the
    reconstruction preserve who is stronger?
  - alive-flag accuracy

Usage:
  python scripts/eval_semantic.py --ckpt runs/ae_v2/ae_v2.pt --data data --samples 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ae.model_v2 import UNIT_CLASSES, UnifiedStateAE
from ae.train_v2 import Featurizer, SamplerV2

DETECT_CLASSES = {
    "nuke": ["Atom Bomb", "Hydrogen Bomb", "MIRV"],
    "silo": ["Missile Silo"],
    "sam": ["SAM Launcher"],
    "warship": ["Warship"],
    "transport": ["Transport"],
    "city": ["City"],
}


def prf(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    tp = float((pred & true).sum())
    p = tp / max(1.0, float(pred.sum()))
    r = tp / max(1.0, float(true.sum()))
    return p, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ae_v2/ae_v2.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default="eval_out/semantic.json")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = UnifiedStateAE(
        latent_c=ckpt["args"]["latent_c"], latent_d=ckpt["args"]["latent_d"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sampler = SamplerV2(args.data, args.crop, seed=1234)

    cls_idx = {name: i for i, name in enumerate(UNIT_CLASSES)}
    det_pred = {k: [] for k in DETECT_CLASSES}
    det_true = {k: [] for k in DETECT_CLASSES}
    diplo_preds, diplo_trues = [], []
    diplo_logit_vals, diplo_masks = [], []
    troop_pairs_ok = 0
    troop_pairs_n = 0
    alive_ok = 0
    alive_n = 0

    rng = np.random.default_rng(7)
    done = 0
    while done < args.samples:
        owners, terrain, planes, pfeats, pmask, diplo = sampler.sample_batch(8)
        with torch.no_grad():
            (tile_logits, unit_pred, player_pred, diplo_logits), _ = model(
                owners, terrain, planes, pfeats, pmask, diplo
            )

        for k, names in DETECT_CLASSES.items():
            idxs = [cls_idx[n] for n in names]
            t = planes[:, idxs].sum(1).numpy() > 0  # >= 1 unit in cell
            # unit_pred is per-class occupancy logits; detect if any class fires.
            p = unit_pred[:, idxs].max(dim=1).values.numpy() > 0
            det_true[k].append(t)
            det_pred[k].append(p)

        pair_mask = (
            (pmask.unsqueeze(2) * pmask.unsqueeze(1)).unsqueeze(1).numpy() > 0
        )
        diplo_preds.append((diplo_logits.numpy() > 0) & pair_mask)
        diplo_trues.append((diplo.numpy() > 0.5) & pair_mask)
        diplo_logit_vals.append(diplo_logits.numpy())
        diplo_masks.append(pair_mask)

        # Troop ordering + alive flags on reconstructed player stats.
        pf, pp, pm = pfeats.numpy(), player_pred.numpy(), pmask.numpy()
        for b in range(pf.shape[0]):
            slots = np.where(pm[b] > 0)[0]
            alive_true = pf[b, slots, 0] > 0.5
            alive_pred = pp[b, slots, 0] > 0.5
            alive_ok += int((alive_true == alive_pred).sum())
            alive_n += len(slots)
            live = slots[alive_true]
            if len(live) >= 2:
                for _ in range(10):
                    i, j = rng.choice(live, 2, replace=False)
                    ti, tj = pf[b, i, 1], pf[b, j, 1]
                    if abs(ti - tj) < 1e-3:
                        continue
                    troop_pairs_n += 1
                    if (ti > tj) == (pp[b, i, 1] > pp[b, j, 1]):
                        troop_pairs_ok += 1
        done += 8

    report: dict = {"samples": done}
    for k in DETECT_CLASSES:
        t = np.concatenate(det_true[k]).ravel()
        p = np.concatenate(det_pred[k]).ravel()
        prec, rec = prf(p, t)
        report[f"{k}_precision"] = round(prec, 4)
        report[f"{k}_recall"] = round(rec, 4)
        report[f"{k}_true_cells"] = int(t.sum())
    for r, name in enumerate(["alliance", "embargo"]):
        t = np.concatenate([x[:, r].ravel() for x in diplo_trues])
        p = np.concatenate([x[:, r].ravel() for x in diplo_preds])
        prec, rec = prf(p, t)
        report[f"{name}_precision"] = round(prec, 4)
        report[f"{name}_recall"] = round(rec, 4)
        report[f"{name}_true_pairs"] = int(t.sum())
        # The BCE pos-weight shifts the natural operating point away from
        # logit 0; sweep thresholds to find the best-F1 point, which tells us
        # whether the latent actually separates the classes.
        logits = np.concatenate([x[:, r].ravel() for x in diplo_logit_vals])
        mask = np.concatenate([x[:, 0].ravel() for x in diplo_masks])
        lv, tv = logits[mask], t[mask]
        best = (0.0, 0.0, 0.0, 0.0)  # f1, thresh, prec, rec
        for thresh in np.arange(-2.0, 8.01, 0.25):
            pv = lv > thresh
            pr, rc = prf(pv, tv)
            f1 = 2 * pr * rc / max(1e-9, pr + rc)
            if f1 > best[0]:
                best = (f1, float(thresh), pr, rc)
        report[f"{name}_best_f1"] = round(best[0], 4)
        report[f"{name}_best_thresh"] = best[1]
        report[f"{name}_best_precision"] = round(best[2], 4)
        report[f"{name}_best_recall"] = round(best[3], 4)
    report["troop_ordering_acc"] = round(troop_pairs_ok / max(1, troop_pairs_n), 4)
    report["alive_flag_acc"] = round(alive_ok / max(1, alive_n), 4)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
