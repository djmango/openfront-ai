"""Loader for headless-game snapshot data produced by datagen/generate.ts.

Supports two on-disk layouts:
  - new: states/t<tick>.bin.gz  (one gzipped uint16-le grid per snapshot)
  - old: states.bin             (all snapshots concatenated, memory-mapped)
"""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OWNER_MASK = 0x0FFF
FALLOUT_BIT = 13
DEFENSE_BONUS_BIT = 14

# Terrain byte bits (see GameMapImpl)
IS_LAND_BIT = 7
SHORELINE_BIT = 6
OCEAN_BIT = 5
MAGNITUDE_MASK = 0x1F


@dataclass
class GameRecord:
    game_dir: Path
    meta: dict
    terrain: np.ndarray  # (h, w) uint8
    state_files: list[Path] = field(default_factory=list)  # new format
    states_mm: np.ndarray | None = None  # old format, memory-mapped

    @property
    def width(self) -> int:
        return self.meta["width"]

    @property
    def height(self) -> int:
        return self.meta["height"]

    @property
    def num_snapshots(self) -> int:
        return len(self.meta["snapshots"])

    def state(self, i: int) -> np.ndarray:
        """Raw packed uint16 tile state for snapshot i, shape (h, w)."""
        if self.states_mm is not None:
            return self.states_mm[i]
        raw = gzip.decompress(self.state_files[i].read_bytes())
        return np.frombuffer(raw, dtype="<u2").reshape(self.height, self.width)

    def owners(self, i: int) -> np.ndarray:
        """Owner smallID per tile (0 = unowned) for snapshot i."""
        return self.state(i) & OWNER_MASK

    def fallout(self, i: int) -> np.ndarray:
        return (self.state(i) >> FALLOUT_BIT) & 1

    def land(self) -> np.ndarray:
        return (self.terrain >> IS_LAND_BIT) & 1

    def magnitude(self) -> np.ndarray:
        return self.terrain & MAGNITUDE_MASK


def load_game(game_dir: str | Path) -> GameRecord:
    game_dir = Path(game_dir)
    meta = json.loads((game_dir / "meta.json").read_text())
    h, w = meta["height"], meta["width"]
    terrain = np.fromfile(game_dir / "terrain.bin", dtype=np.uint8).reshape(h, w)
    n = len(meta["snapshots"])

    states_dir = game_dir / "states"
    if states_dir.is_dir():
        files = sorted(states_dir.glob("t*.bin.gz"))
        if len(files) != n:
            raise ValueError(
                f"{game_dir}: {len(files)} state files but {n} snapshots in meta"
            )
        return GameRecord(game_dir, meta, terrain, state_files=files)

    states = np.memmap(game_dir / "states.bin", dtype="<u2", mode="r", shape=(n, h, w))
    return GameRecord(game_dir, meta, terrain, states_mm=states)


def iter_games(data_root: str | Path):
    for meta_path in sorted(Path(data_root).rglob("meta.json")):
        yield load_game(meta_path.parent)


if __name__ == "__main__":
    import sys

    from PIL import Image

    game = load_game(sys.argv[1])
    out_dir = game.game_dir / "preview"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    # Stable distinct colors per owner id; 0 (unowned) rendered as terrain.
    palette = rng.integers(50, 255, size=(4096, 3), dtype=np.uint8)

    land = game.land().astype(bool)
    for i in np.linspace(0, game.num_snapshots - 1, 6, dtype=int):
        owners = game.owners(i)
        img = np.zeros((game.height, game.width, 3), dtype=np.uint8)
        img[~land] = (30, 60, 110)
        img[land] = (170, 160, 130)
        owned = owners > 0
        img[owned] = palette[owners[owned]]
        tick = game.meta["snapshots"][i]["tick"]
        Image.fromarray(img).save(out_dir / f"tick_{tick:06d}.png")
        print(f"wrote preview for tick {tick}")
