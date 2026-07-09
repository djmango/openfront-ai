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


def decode_frame(
    blob: bytes, hr: int, wr: int, dctx
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """zstd frame blob -> (owner slots (hr, wr) u8, packed fallout (hr, wr/8)
    u8, packed defense bonus (hr, wr/8) u8) (v7: CACHE_FORMAT=2, three
    planes - regenerate stale caches with scripts/prefeaturize_bc.py).

    `dctx` is the caller's thread-local zstd decompressor, used only by the
    fallback path.
    """
    if _ofrs is not None:
        return _ofrs.decode_frame(blob, hr, wr)
    hw = hr * wr
    plane = hr * (wr // 8)
    raw = dctx.decompress(blob, max_output_size=hw + 2 * plane)
    slots = np.frombuffer(raw[:hw], dtype=np.uint8).reshape(hr, wr)
    fallout = np.frombuffer(raw[hw : hw + plane], dtype=np.uint8).reshape(hr, -1)
    defense_bonus = np.frombuffer(raw[hw + plane :], dtype=np.uint8).reshape(hr, -1)
    return slots, fallout, defense_bonus


def _c_contig(a: np.ndarray) -> np.ndarray:
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)


def collate_grids(grids: list[np.ndarray], gh: int, gw: int) -> np.ndarray:
    """Pad+stack (C, h, w) arrays (all same C and dtype) to (B, C, gh, gw)."""
    if _ofrs is not None and grids[0].dtype in (np.float32, np.float16):
        fn = (
            _ofrs.collate_grids_f32
            if grids[0].dtype == np.float32
            else _ofrs.collate_grids_f16
        )
        return fn([_c_contig(g) for g in grids], gh, gw)
    b = np.zeros((len(grids), grids[0].shape[0], gh, gw), dtype=grids[0].dtype)
    for i, g in enumerate(grids):
        b[i, :, : g.shape[1], : g.shape[2]] = g
    return b


def stack(arrays: list[np.ndarray]) -> np.ndarray:
    """np.stack for equal-shape f32/f16 arrays (parallel copy, GIL-free);
    anything else falls back to numpy."""
    a0 = arrays[0]
    if _ofrs is not None and isinstance(a0, np.ndarray) and a0.dtype in (
        np.float32,
        np.float16,
    ):
        fn = _ofrs.stack_f32 if a0.dtype == np.float32 else _ofrs.stack_f16
        flat = fn([np.ascontiguousarray(a) for a in arrays])
        return flat.reshape(len(arrays), *a0.shape)
    return np.stack(arrays)


def pack_arrays(msg: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize an env-worker message for unpack_arrays: u32 header length
    ++ header json ++ raw array buffers. Cheaper than pickle on both sides
    (used by rl/vec.py; works without the extension too)."""
    import json

    specs, bufs = [], []
    for k, v in arrays.items():
        v = np.ascontiguousarray(v)
        specs.append([k, v.dtype.str, list(v.shape)])
        bufs.append(v.tobytes())
    header = json.dumps({"arrays": specs, **msg}).encode()
    return b"".join([len(header).to_bytes(4, "little"), header, *bufs])


def unpack_arrays(payload: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Inverse of pack_arrays: (rest-of-header dict, {key: array})."""
    import json

    if _ofrs is not None:
        rest, arrays = _ofrs.unpack_arrays(payload)
        return json.loads(rest), arrays
    hlen = int.from_bytes(payload[:4], "little")
    header = json.loads(payload[4 : 4 + hlen])
    off = 4 + hlen
    arrays = {}
    for k, dt, shape in header.pop("arrays"):
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        arrays[k] = np.frombuffer(payload[off : off + n], dtype=dt).reshape(shape)
        off += n
    return header, arrays


def collate_masks(masks: list[np.ndarray], gh: int, gw: int) -> np.ndarray:
    """Pad+stack (h, w) float32 arrays to (B, gh, gw)."""
    if _ofrs is not None:
        return _ofrs.collate_masks([_c_contig(m.astype(np.float32, copy=False)) for m in masks], gh, gw)
    b = np.zeros((len(masks), gh, gw), dtype=np.float32)
    for i, v in enumerate(masks):
        b[i, : v.shape[0], : v.shape[1]] = v
    return b
