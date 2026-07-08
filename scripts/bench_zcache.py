"""Measure the AE-latent cache's steady-state ceiling without waiting for
the training-time hit rate to climb: encode the same raw batches cold
(misses, fills the cache) then warm (100% hits) and report ex/s for each.

  PYTHONPATH=. python scripts/bench_zcache.py --data data-human --batch 96 --batches 20
"""

import argparse
import time
from pathlib import Path

import torch

from rl.bc_data import BCSampler
from rl.obs import ZCache, encode_grids, load_ae
from rl.bc import OBS_KEYS
from rl.obs import collate as obs_collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--ae", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--collate", action="store_true",
                    help="also time obs collate on the warm pass")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = load_ae(args.ae, device)
    sampler = BCSampler([Path(d) for d in args.data])
    print(f"device: {device}  games: {len(sampler.games)}  "
          f"native: {sampler._native is not None}")

    t = time.time()
    batches = [sampler.sample_batch(args.batch) for _ in range(args.batches)]
    n = sum(len(b) for b in batches)
    print(f"sampled {n} raws in {time.time() - t:.1f}s")

    def run(tag: str, z_cache: ZCache | None) -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for b in batches:
            enc = encode_grids(ae, [dict(r) for r in b], device, z_cache=z_cache)
            if args.collate:
                obs_collate(enc, OBS_KEYS)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        zc = ""
        if z_cache is not None:
            zc = (f"  hits {z_cache.hits} misses {z_cache.misses} "
                  f"cache {z_cache.bytes / 1e9:.1f}GB")
        print(f"{tag}: {dt:.2f}s  {n / dt:.1f} ex/s{zc}", flush=True)

    run("no-cache  ", None)
    cache = ZCache(int(50e9))
    run("cold(fill)", cache)
    cache.hits = cache.misses = 0
    run("warm(hits)", cache)
    cache.hits = cache.misses = 0
    run("warm(hits)", cache)


if __name__ == "__main__":
    main()
