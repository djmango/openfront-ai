#!/usr/bin/env python3
"""Benchmark ofrs vs numpy on paths that actually matter. Run after maturin develop."""

import argparse
import time

import numpy as np
import zstandard as zstd

from rl.native import HAVE_NATIVE, collate_grids, decode_frame


def bench(name: str, fn, n: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"  {name}: {ms:.2f} ms")
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-n", type=int, default=500)
    ap.add_argument("--collate-n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    print(f"ofrs native: {HAVE_NATIVE}")

    # wr must be divisible by 8 (packbits fallout plane)
    hr, wr = 128, 248
    owners = np.random.randint(0, 30, (hr, wr), dtype=np.uint8)
    fall = np.random.randint(0, 2, (hr, wr), dtype=np.uint8)
    packed = np.packbits(fall, axis=1)
    blob = zstd.ZstdCompressor(level=1).compress(owners.tobytes() + packed.tobytes())
    dctx = zstd.ZstdDecompressor()

    def py_dec():
        hw = hr * wr
        raw = dctx.decompress(blob, max_output_size=hw * 2)
        np.frombuffer(raw[:hw], np.uint8).reshape(hr, wr)
        np.frombuffer(raw[hw:], np.uint8).reshape(hr, -1)

    print("decode_frame (2nd+ rust call reuses thread-local scratch):")
    bench("python", py_dec, args.decode_n)
    if HAVE_NATIVE:
        bench("rust", lambda: decode_frame(blob, hr, wr, dctx), args.decode_n)

    gh, gw = 128, 248
    gs = [
        np.random.rand(43, 100 + (i % 20), 180 + (i % 30)).astype(np.float16)
        for i in range(args.batch)
    ]

    def py_col():
        b = np.zeros((len(gs), 43, gh, gw), np.float16)
        for i, g in enumerate(gs):
            b[i, :, : g.shape[1], : g.shape[2]] = g
        return b

    print(f"collate_grids_f16 batch={args.batch} pad->{gh}x{gw}:")
    bench("python", py_col, args.collate_n)
    if HAVE_NATIVE:
        import ofrs

        gs_c = [np.ascontiguousarray(g) for g in gs]
        bench("rust direct", lambda: ofrs.collate_grids_f16(gs_c, gh, gw), args.collate_n)
        bench("native wrapper", lambda: collate_grids(gs, gh, gw), args.collate_n)


if __name__ == "__main__":
    main()
