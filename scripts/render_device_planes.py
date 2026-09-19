#!/usr/bin/env python3
"""Render a planes.bin-style file to an MP4 + PNG stills (stdlib + ffmpeg/magick).

Usage:
  render_planes.py <planes.bin> <terrain.bin> <out.mp4> <still_prefix> [W H]

W and H are the map's real tile dimensions (from its manifest.json). They are
optional only for backwards compatibility, where they default to 1000x1000:
pass them for any non-square map, otherwise the plane is squashed into a square
and the picture lies about the geometry.

Water = the 12 most frequent terrain values (measured: 63 is open ocean, then the
shallow-water gradient). Owners get a coral-red / sky-blue palette. Stills keep
the plane's true aspect ratio (fitted into an 800x800 box, never squared) and the
video is letterboxed onto a square canvas the same way.
"""
import array, collections, os, subprocess, sys

if len(sys.argv) >= 5:
    planes_path, terrain_path, out_mp4, still_prefix = sys.argv[1:5]
    W = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    H = int(sys.argv[6]) if len(sys.argv) > 6 else W
else:
    # Backwards-compatible default (original single-argument-free invocation).
    planes_path = "/tmp/ofhash_record_late/planes.bin"
    terrain_path = "/tmp/ofhash_record_late/terrain.bin"
    out_mp4 = "/tmp/cuda_game.mp4"
    still_prefix = "/tmp/cuda_still"
    W = H = 1000
_psz = os.path.getsize(planes_path)
if _psz % (W * H * 2) != 0:
    sys.exit(
        f"FATAL: {planes_path} is {_psz} bytes, not a multiple of "
        f"W*H*2 = {W * H * 2} for {W}x{H} - wrong W/H for this plane"
    )
FRAMES = _psz // (W * H * 2)
print(f"planes {planes_path} {W}x{H} frames={FRAMES}", flush=True)

terrain = open(terrain_path, "rb").read()
hist = collections.Counter(terrain)
WATER = {v for v, _ in hist.most_common(12)}
print("terrain distinct =", len(hist), "min =", min(hist), "max =", max(hist))
print("top terrain values:", hist.most_common(12))
print("water set:", sorted(WATER))
lo, hi = min(hist), max(hist)

WATER_RGB = (26, 40, 64)


def land_rgb(t):
    f = (t - lo) / (hi - lo) if hi != lo else 0.0
    g = int(60 + 120 * f)
    return (int(g * 0.92), g, int(g * 0.84))


LAND_TBL = [land_rgb(t) for t in range(256)]
BASE = [(232, 93, 76), (86, 157, 232), (240, 196, 76), (144, 214, 116), (206, 122, 214)]


def frame_rgb(owners):
    rgb = bytearray(W * H * 3)
    j = 0
    for i in range(W * H):
        o = owners[i]
        if o:
            c = BASE[(o - 1) % len(BASE)]
        elif terrain[i] in WATER:
            c = WATER_RGB
        else:
            c = LAND_TBL[terrain[i]]
        rgb[j] = c[0]
        rgb[j + 1] = c[1]
        rgb[j + 2] = c[2]
        j += 3
    return bytes(rgb)


planes = open(planes_path, "rb")
cmd = ["ffmpeg", "-y", "-loglevel", "error",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "12", "-i", "-",
       "-vf", f"scale=720:720:force_original_aspect_ratio=decrease:flags=neighbor,"
              f"pad=720:720:(ow-iw)/2:(oh-ih)/2:color=0x101010",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4]
enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for k in range(FRAMES):
    a = array.array("H")
    a.frombytes(planes.read(W * H * 2))
    if sys.byteorder == "big":
        a.byteswap()
    frame = frame_rgb(a)
    enc.stdin.write(frame)
    if k % 25 == 0:
        print("frame", k, "of", FRAMES, flush=True)
    if k in (0, FRAMES // 4, FRAMES // 2, FRAMES - 1):
        raw = f"/tmp/_rp{k}.rgb"
        open(raw, "wb").write(frame)
        png = f"{still_prefix}_{k:03d}.png"
        subprocess.run(["magick", "-size", f"{W}x{H}", "-depth", "8", f"rgb:{raw}",
                        "-resize", "800x800", png], check=True)
        os.remove(raw)
        print("wrote", png, flush=True)

enc.stdin.close()
enc.wait()
print("wrote", out_mp4)
