"""Build per-game static-structure sidecars (cache/static.npz) in parallel.

Extracts static-structure unit rows (cities, ports, silos, ...) from each
game's units.npy so training workers never have to scan the full unit table.
Idempotent; run after prefeaturize.py.

Usage:
  PYTHONPATH=. python scripts/build_static_cache.py --data data,data-human --workers 12
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ae.train_v3 import CachedGame


def build(game_dir: Path) -> str:
    if (game_dir / "cache" / "static.npz").exists():
        return f"skip {game_dir.name}"
    CachedGame(game_dir)  # constructor builds + persists the sidecar
    return f"done {game_dir.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data,data-human")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    dirs = sorted(
        p.parent.parent
        for root in args.data.split(",")
        for p in Path(root).rglob("cache/index.json")
    )
    print(f"{len(dirs)} games")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, msg in enumerate(ex.map(build, dirs)):
            if (i + 1) % 50 == 0 or i == len(dirs) - 1:
                print(f"[{i + 1}/{len(dirs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
