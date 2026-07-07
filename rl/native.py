"""Optional Rust hot paths (rust/ofrs, PyO3). Pure-Python numpy fallbacks
keep every entry point working when the extension isn't built; callers
import from here and never check availability themselves.

Build locally / on pods:  pip install ./rust/ofrs   (needs a Rust toolchain)
"""

from __future__ import annotations

import numpy as np

try:
    import ofrs as _ofrs
except ImportError:  # pragma: no cover - extension not built
    _ofrs = None

HAVE_NATIVE = _ofrs is not None


def decode_frame(blob: bytes, hr: int, wr: int, dctx) -> tuple[np.ndarray, np.ndarray]:
    """zstd frame blob -> (owner slots (hr, wr) u8, packed fallout (hr, wr/8) u8).

    `dctx` is the caller's thread-local zstd decompressor, used only by the
    fallback path.
    """
    if _ofrs is not None:
        return _ofrs.decode_frame(blob, hr, wr)
    hw = hr * wr
    raw = dctx.decompress(blob, max_output_size=hw * 2)
    slots = np.frombuffer(raw[:hw], dtype=np.uint8).reshape(hr, wr)
    packed = np.frombuffer(raw[hw:], dtype=np.uint8).reshape(hr, -1)
    return slots, packed


def collate_grids(grids: list[np.ndarray], gh: int, gw: int) -> np.ndarray:
    """Pad+stack (C, h, w) arrays (all same C and dtype) to (B, C, gh, gw)."""
    if _ofrs is not None and grids[0].dtype in (np.float32, np.float16):
        fn = (
            _ofrs.collate_grids_f32
            if grids[0].dtype == np.float32
            else _ofrs.collate_grids_f16
        )
        return fn([np.ascontiguousarray(g) for g in grids], gh, gw)
    b = np.zeros((len(grids), grids[0].shape[0], gh, gw), dtype=grids[0].dtype)
    for i, g in enumerate(grids):
        b[i, :, : g.shape[1], : g.shape[2]] = g
    return b


def collate_masks(masks: list[np.ndarray], gh: int, gw: int) -> np.ndarray:
    """Pad+stack (h, w) float32 arrays to (B, gh, gw)."""
    if _ofrs is not None:
        return _ofrs.collate_masks(
            [np.ascontiguousarray(m, dtype=np.float32) for m in masks], gh, gw
        )
    b = np.zeros((len(masks), gh, gw), dtype=np.float32)
    for i, v in enumerate(masks):
        b[i, : v.shape[0], : v.shape[1]] = v
    return b
