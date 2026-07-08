"""Fine-grained timing of the warm (100%-hit) encode path: where do the
~2.4ms/sample go once the AE encode is gone? Mirrors the cache-hit chunk
loop in rl.obs.encode_grids with a cuda-sync timer around each section.

  PYTHONPATH=. python scripts/prof_zcache_warm.py --data data-human --batch 96 --batches 10
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.bc_data import BCSampler
from rl.obs import LATENT_C, REGION, ZCache, _local_crops, _staged_stack, encode_grids, load_ae
from rl.bc import OBS_KEYS
from rl.obs import collate as obs_collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--ae", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--batches", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = load_ae(args.ae, device)
    sampler = BCSampler([Path(d) for d in args.data])
    batches = [sampler.sample_batch(args.batch) for _ in range(args.batches)]
    n = sum(len(b) for b in batches)

    cache = ZCache(int(50e9))
    for b in batches:  # fill
        encode_grids(ae, [dict(r) for r in b], device, z_cache=cache)

    t = defaultdict(float)

    def tick(name, t0):
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        t[name] += t1 - t0
        return t1

    MAX_ENC_PIX = 16_000_000
    if device == "cuda":
        torch.cuda.synchronize()
    tall = time.time()
    for b in batches:
        raws = b
        groups = {}
        for i, r in enumerate(raws):
            groups.setdefault(r["owners"].shape, []).append(i)
        chunks = []
        for (H, W), idxs in groups.items():
            per = max(1, MAX_ENC_PIX // (H * W))
            chunks.extend(idxs[k : k + per] for k in range(0, len(idxs), per))
        t0 = time.time()
        for idxs in chunks:
            zl = [cache.get(raws[i]["z_key"]) for i in idxs]
            t0 = tick("cache_get", t0)
            owners = torch.from_numpy(
                _staged_stack("owners", [raws[i]["owners"] for i in idxs])
            ).to(device)
            t0 = tick("owners_up", t0)
            clut = torch.from_numpy(
                np.stack([raws[i]["clut"] for i in idxs])
            ).long().to(device)
            land = torch.stack(
                [cache.land(raws[i], device) for i in idxs]
            ).float()
            t0 = tick("clut_land", t0)
            B, H, W = owners.shape
            zs = torch.from_numpy(
                _staged_stack("zhit", zl)
            ).to(device).float()
            t0 = tick("z_up", t0)
            classmap = torch.gather(
                clut, 1, owners.long().reshape(B, -1)
            ).reshape(B, H, W)
            ego = torch.stack([(classmap == c).float() for c in (1, 2, 3)], dim=1)
            ego = F.avg_pool2d(ego, REGION)
            grid = torch.cat([zs, ego], dim=1)
            t0 = tick("ego_gpu", t0)
            local = _local_crops(classmap, land)
            t0 = tick("local", t0)
            grid, local = grid.cpu().numpy(), local.cpu().numpy()
            t0 = tick("download", t0)
        # per-sample assembly (transient concat) + collate, as in training
        enc = encode_grids(ae, [dict(r) for r in raws], device, z_cache=cache)
        t0 = tick("full_encode_grids", t0)
        obs_collate(enc, OBS_KEYS)
        t0 = tick("collate", t0)
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.time() - tall
    print(f"n={n}  total {total:.2f}s  ({n/total:.0f} ex/s incl. double work)")
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v:6.2f}s  {1e3 * v / n:.3f} ms/sample")


if __name__ == "__main__":
    main()
