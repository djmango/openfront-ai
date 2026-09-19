#!/usr/bin/env python3
"""Contact sheet: one frame per dumped cell, map + N burned into each label.

Reads the stills written by the COMMITTED render script
(openfront-ai/scripts/render_device_planes.py writes <prefix>_<frame>.png) and
montages them into one PNG grid with `magick montage -label`.

Usage:
  contact_sheet.py <manifest> <out.png> [tile]
    manifest lines: <label>|<still.png>
"""
import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    man, out = sys.argv[1], sys.argv[2]
    tile = sys.argv[3] if len(sys.argv) > 3 else "4x"
    rows = []
    missing = []
    for line in open(man):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label, _, png = line.partition("|")
        if not os.path.exists(png):
            missing.append(png)
            continue
        rows.append((label, png))
    if missing:
        print(f"WARNING: {len(missing)} still(s) missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    if not rows:
        print("no stills to tile")
        return 1
    cmd = [
        "magick",
        "montage",
        # Font settings MUST come before the per-image -label arguments: magick
        # applies them as they are parsed, so a -pointsize placed after the
        # images labels nothing and the sheet comes out caption-less.
        "-font",
        "DejaVu-Sans",
        "-pointsize",
        "22",
        "-background",
        "#1b1b1b",
        "-fill",
        "white",
    ]
    for label, png in rows:
        cmd += ["-label", label, png]
    cmd += [
        "-tile",
        tile,
        "-geometry",
        "380x380+8+8",
        out,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
