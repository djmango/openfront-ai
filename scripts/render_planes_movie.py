#!/usr/bin/env python3
"""Render every per-tick plane to an MP4 (stdlib + ffmpeg, no numpy)."""
import array, collections, os, subprocess, sys

D = "/tmp/ofhash_record_late"
W = H = 1000
FRAMES = os.path.getsize(f"{D}/planes.bin") // (W * H * 2)

terrain = open(f"{D}/terrain.bin", "rb").read()
hist = collections.Counter(terrain)
top = hist.most_common(8)
print("terrain distinct =", len(hist), "min =", min(hist), "max =", max(hist))
print("top terrain values:", top)
WATER = top[0][0]
print("treating terrain value", WATER, "as water (most frequent)")
lo, hi = min(hist), max(hist)

# OpenFront-ish palette: water dark blue, land grey-green shading
WATER_RGB = (26, 40, 64)

def land_rgb(t):
    if hi == lo:
        return (96, 100, 96)
    f = (t - lo) / (hi - lo)
    g = int(60 + 120 * f)
    return (int(g * 0.92), g, int(g * 0.84))

def owner_rgb(oid):
    if oid == 0:
        return None
    base = [(232, 93, 76), (86, 157, 232), (240, 196, 76), (144, 214, 116), (206, 122, 214)]
    return base[(oid - 1) % len(base)]

LAND_LUT = bytes()
# build a 256-entry lookup for land shading
land_tbl = [land_rgb(t) for t in range(256)]

def frame_rgb(owners):
    rgb = bytearray(W * H * 3)
    j = 0
    for i in range(W * H):
        o = owners[i]
        if o:
            c = owner_rgb(o)
        elif terrain[i] == WATER:
            c = WATER_RGB
        else:
            c = land_tbl[terrain[i]]
        rgb[j] = c[0]; rgb[j + 1] = c[1]; rgb[j + 2] = c[2]
        j += 3
    return bytes(rgb)

planes = open(f"{D}/planes.bin", "rb")
cmd = ["ffmpeg", "-y", "-loglevel", "error",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "12", "-i", "-",
       "-vf", "scale=720:720:flags=neighbor",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "/tmp/cuda_game.mp4"]
enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for k in range(FRAMES):
    a = array.array("H")
    a.frombytes(planes.read(W * H * 2))
    if sys.byteorder == "big":
        a.byteswap()
    enc.stdin.write(frame_rgb(a))
    if k % 25 == 0:
        print("frame", k, "of", FRAMES, flush=True)
    if k in (0, FRAMES // 4, FRAMES // 2, FRAMES - 1):
        raw = f"/tmp/_f{k}.rgb"
        open(raw, "wb").write(frame_rgb(a))
        subprocess.run(["magick", "-size", f"{W}x{H}", "-depth", "8", f"rgb:{raw}",
                        "-resize", "800x800", f"/tmp/cuda_still_{k:03d}.png"], check=True)
        os.remove(raw)

enc.stdin.close()
enc.wait()
print("wrote /tmp/cuda_game.mp4")
print("stills:", sorted(f for f in os.listdir("/tmp") if f.startswith("cuda_still_")))
