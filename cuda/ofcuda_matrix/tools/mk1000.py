#!/usr/bin/env python3
"""Nearest-neighbour resample of a cell's plane/terrain dump to 1000x1000.

The committed render scripts (openfront-ai/scripts/render_device_planes.py) are
hardcoded to W = H = 1000. A cell on thebox (2048x2048) or giantworldmap
(4108x1948) cannot be fed to them directly, and rewriting those scripts is not
allowed, so the dump is decimated here instead - one sample per output pixel, no
interpolation, so owner ids survive exactly (no blended owners).

Usage:
  mk1000.py <planes.bin> <terrain.bin> <W> <H> <outdir>

Writes <outdir>/planes.bin (u16, 1000*1000 per frame, frame count preserved) and
<outdir>/terrain.bin (u8, 1000*1000).
"""
import array
import os
import sys

OUT = 1000


def main() -> int:
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    planes_path, terrain_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    outdir = sys.argv[5]
    os.makedirs(outdir, exist_ok=True)

    # The output->source index map is built once and reused for every frame.
    xmap = [x * w // OUT for x in range(OUT)]
    ymap = [y * h // OUT for y in range(OUT)]
    idx = [ymap[y] * w + xmap[x] for y in range(OUT) for x in range(OUT)]

    frame_bytes = w * h * 2
    total = os.path.getsize(planes_path)
    frames = total // frame_bytes
    with open(planes_path, "rb") as f, open(os.path.join(outdir, "planes.bin"), "wb") as o:
        for fi in range(frames):
            buf = f.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            src = array.array("H")
            src.frombytes(buf)
            if sys.byteorder != "little":
                src.byteswap()
            out = array.array("H", bytes(OUT * OUT * 2))
            for k in range(OUT * OUT):
                out[k] = src[idx[k]]
            if sys.byteorder != "little":
                out.byteswap()
            o.write(out.tobytes())
    print(f"planes: {frames} frames {w}x{h} -> {OUT}x{OUT}", flush=True)

    tb = open(terrain_path, "rb").read()
    if len(tb) < w * h:
        print(f"error: terrain {terrain_path} is {len(tb)} bytes, need {w*h}")
        return 1
    outb = bytearray(OUT * OUT)
    for k in range(OUT * OUT):
        outb[k] = tb[idx[k]]
    with open(os.path.join(outdir, "terrain.bin"), "wb") as o:
        o.write(outb)
    print(f"terrain: {w}x{h} -> {OUT}x{OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
