#!/usr/bin/env python3
"""Convert an ofcuda_matrix v2 TEXT oracle dump into the JSONL record shape
that `ofcuda_tick::parse_dump` + `ofcuda_env/src/bin/extras.rs` read.

This is a READER for the env's verify path, not a second oracle parser: the
v2 text dump and its `.clusters` sidecar stay the oracle, and the env is
driven from the JSONL this produces so it can be compared against them.

    python3 v2_to_jsonl.py <in.dump> <out.ndjson>

Field meanings taken from ofcuda_matrix/src/lib.rs:
  OWNED   <b> <sid> <n> <tiles...>                 owned tile order
  BORDER  <b> <sid> <n> <tiles...>                 border set
  PLAYER  <b> <sid> <type> <id> <tiles> <troops> <gold> <nvec> <prefix>
  ATTACK  <b> <owner> <target> <troops_bits> <src> <heap_len> <border_len> <id>
  CADENCE <b> <sid> <lcc> <ltc> <disc>             (NOT part of the JSONL;
                                                   the engine's own counters)
"""
import json
import struct
import sys
from collections import defaultdict


def f64_from_bits(tok):
    try:
        return struct.unpack("<d", struct.pack("<Q", int(tok, 16)))[0]
    except Exception:
        return 0.0


def main():
    src, dst = sys.argv[1], sys.argv[2]
    owned = defaultdict(dict)   # b -> sid -> tiles
    border = defaultdict(dict)
    players = defaultdict(dict)  # b -> sid -> dict
    attacks = defaultdict(list)  # b -> [ ... ]

    with open(src, "r") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split()
            if not f:
                continue
            tag = f[0]
            try:
                if tag == "OWNED":
                    owned[int(f[1])][int(f[2])] = [int(x) for x in f[4:4 + int(f[3])]]
                elif tag == "BORDER":
                    border[int(f[1])][int(f[2])] = [int(x) for x in f[4:4 + int(f[3])]]
                elif tag == "PLAYER":
                    players[int(f[1])][int(f[2])] = {
                        "id": f[4],
                        "smallId": int(f[2]),
                        "name": f[4],
                        "playerType": {"H": "Human", "B": "Bot", "N": "Nation"}.get(f[3], "Bot"),
                        "tiles": int(f[5]),
                        "troops": int(float(f[6])),
                        "gold": int(float(f[7])),
                    }
                elif tag == "ATTACK":
                    attacks[int(f[1])].append({
                        "ownerSmallId": int(f[2]),
                        "targetSmallId": int(f[3]),
                        "troops": int(f64_from_bits(f[4])),
                        "attackLive": True,
                    })
            except (IndexError, ValueError):
                continue

    ticks = sorted(set(owned) | set(players) | set(attacks))
    with open(dst, "w") as out:
        out.write(json.dumps({"type": "header", "engine": "ofcuda_matrix-v2", "every": 1}) + "\n")
        for b in ticks:
            ps = []
            for sid, base in sorted(players.get(b, {}).items()):
                p = dict(base)
                p["ownedTiles"] = owned.get(b, {}).get(sid, [])
                p["ownedOrder"] = p["ownedTiles"]
                p["borderOrder"] = border.get(b, {}).get(sid, [])
                p["alive"] = bool(p["tiles"])
                ps.append(p)
            out.write(json.dumps({
                "tick": b,
                "players": ps,
                "attacks": attacks.get(b, []),
            }) + "\n")
    print(f"{src} -> {dst}: {len(ticks)} tick records")


if __name__ == "__main__":
    main()
