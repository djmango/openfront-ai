#!/usr/bin/env python3
"""Side-by-side device-vs-reference render with the per-frame differing-tile
count stamped on every frame (stdlib + ffmpeg/magick, no numpy/PIL).

Usage:
  render_sbs.py <device_planes.bin> <ref_planes.bin> <terrain.bin> <out.mp4> <still_prefix>

Left pane  = CUDA DEVICE (device_planes.bin, read back from d_bytes)
Right pane = REFERENCE   (planes.bin, the oracle dump)
Differing tiles are outlined in magenta in both panes (nothing is outlined when
the count is 0). The count itself is printed per frame to stdout and stamped
into the label bar of every frame.
"""
import array, collections, os, subprocess, sys

dev_path, ref_path, terrain_path, out_mp4, still_prefix = sys.argv[1:6]
W = H = 1000
PANES_H = 1000
BAR_H = 64
FRAME_W = 2 * W
FRAME_H = PANES_H + BAR_H

n_dev = os.path.getsize(dev_path) // (W * H * 2)
n_ref = os.path.getsize(ref_path) // (W * H * 2)
FRAMES = min(n_dev, n_ref)
print(f"device planes={n_dev} ref planes={n_ref} frames={FRAMES}", flush=True)

terrain = open(terrain_path, "rb").read()
hist = collections.Counter(terrain)
WATER = {v for v, _ in hist.most_common(12)}
lo, hi = min(hist), max(hist)
WATER_RGB = (26, 40, 64)
LAND_TBL = [((lambda g: (int(g * 0.92), g, int(g * 0.84)))(int(60 + 120 * ((t - lo) / (hi - lo) if hi != lo else 0.0)))) for t in range(256)]
BASE = [(232, 93, 76), (86, 157, 232), (240, 196, 76), (144, 214, 116), (206, 122, 214)]
DIFF_RGB = (255, 0, 255)


def pane(owners, diff_idx):
    """RGB bytes for one 1000x1000 pane; diff tiles outlined magenta."""
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
    for i in diff_idx:
        # 3x3 outline of magenta (only hits neighbours inside the pane)
        x, y = i % W, i // W
        for yy in range(max(0, y - 1), min(H, y + 2)):
            for xx in range(max(0, x - 1), min(W, x + 2)):
                if xx in (x - 1, x + 1) or yy in (y - 1, y + 1):
                    p = (yy * W + xx) * 3
                    rgb[p], rgb[p + 1], rgb[p + 2] = DIFF_RGB
    return bytes(rgb)


_label_cache = {}


def label_bar(text):
    """Raw RGB of a FRAME_W x BAR_H bar with `text` centred."""
    if text not in _label_cache:
        raw = "/tmp/_sbs_label.rgb"
        subprocess.run(["magick", "-size", f"{FRAME_W}x{BAR_H}", "xc:black",
                        "-fill", "white", "-pointsize", "34", "-gravity", "center",
                        "-annotate", "+0+0", text, "-depth", "8", f"rgb:{raw}"], check=True)
        data = open(raw, "rb").read()
        assert len(data) == FRAME_W * BAR_H * 3, (len(data), FRAME_W * BAR_H * 3)
        _label_cache[text] = data
        os.remove(raw)
    return _label_cache[text]


dev = open(dev_path, "rb")
ref = open(ref_path, "rb")
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FRAME_W}x{FRAME_H}", "-r", "12", "-i", "-",
     "-vf", "scale=1400:-2:flags=neighbor",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4],
    stdin=subprocess.PIPE)

diff_counts = []
for k in range(FRAMES):
    ad, ar = array.array("H"), array.array("H")
    ad.frombytes(dev.read(W * H * 2))
    ar.frombytes(ref.read(W * H * 2))
    if sys.byteorder == "big":
        ad.byteswap()
        ar.byteswap()
    diff = [i for i in range(W * H) if ad[i] != ar[i]]
    n = len(diff)
    diff_counts.append(n)
    bar = label_bar(
        f"tick {k}:  device vs reference differing tiles = {n}   "
        f"[left: CUDA DEVICE  |  right: REFERENCE]"
    )
    enc.stdin.write(bar + pane(ad, diff) + pane(ar, diff))
    if k % 25 == 0 or n:
        print(f"frame {k}: differing tiles = {n}", flush=True)
    if k in (0, FRAMES // 4, FRAMES // 2, FRAMES - 1):
        raw = f"/tmp/_sbs{k}.rgb"
        open(raw, "wb").write(bar + pane(ad, diff) + pane(ar, diff))
        png = f"{still_prefix}_{k:03d}.png"
        subprocess.run(["magick", "-size", f"{FRAME_W}x{FRAME_H}", "-depth", "8", f"rgb:{raw}",
                        "-resize", "1600x", png], check=True)
        os.remove(raw)
        print("wrote", png, flush=True)
    # diff-map still: magenta where tiles differ, black where identical
    if k in (0, FRAMES // 2):
        dm = bytearray(W * H * 3)
        for i in diff:
            p = i * 3
            dm[p], dm[p + 1], dm[p + 2] = DIFF_RGB
        raw = f"/tmp/_dm{k}.rgb"
        open(raw, "wb").write(bytes(dm))
        png = f"/tmp/cuda_diffmap_{k:03d}.png"
        subprocess.run(["magick", "-size", f"{W}x{H}", "-depth", "8", f"rgb:{raw}",
                        "-resize", "600x600", png], check=True)
        os.remove(raw)
        print("wrote", png, flush=True)

enc.stdin.close()
enc.wait()
print("wrote", out_mp4)
print("frames=", FRAMES, "total_differing_tiles=", sum(diff_counts),
      "max_per_frame=", max(diff_counts), "min_per_frame=", min(diff_counts))
print("per_frame_diff_counts:", diff_counts)
