#!/usr/bin/env python3
"""Regenerate (or verify) the two ground-truth case files under cases/ from a
FIXED-engine tick_dump NDJSON dump.

The dumps are NOT kept: they are big and the tip can be re-run in seconds. This
script exists so "the case file came from the fixed engine" is checkable instead
of asserted. Usage:

  # 1. dump with the fixed native tip (see README "the dumps")
  cd /opt/data/workspaces/skg/openfront-ai
  export PATH=/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH
  OF_DUMP_OWNED_TILES=1 OF_DUMP_BORDER_ORDER=1 OF_DUMP_OWNED_ORDER=1 \
    OF_DUMP_TICKS_FROM=300 ./rust/target/release/tick_dump --repo . \
    --record records/early-curriculum-parity/curr-b002-s1-pangaea.json.gz \
    --every 1 --max-ticks 332 --out /tmp/g.b002.ndjson --ndjson
  # (same for curr-b007-s3-pangaea.json.gz, OF_DUMP_TICKS_FROM=580 --max-ticks 583)

  # 2. verify the preserved case files against it
  python3 gen_case_files.py verify b002 /tmp/g.b002.ndjson
  python3 gen_case_files.py verify b007 /tmp/g.b007.ndjson

  # 3. or rewrite them from the dump (--write), then re-run the crate:
  python3 gen_case_files.py write b002 /tmp/g.b002.ndjson

Both modes re-derive the per-tick state-plane hash with the fixed contract
(FNV-1a 64, offset 0xcbf29ce484222325, prime 0x100000001b3, each u16 as two
little-endian bytes) and exit non-zero on any disagreement, so a stale or
pre-fix dump cannot silently become ground truth.
"""

import json
import sys

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF
W = H = 1000

# Which (game, first claimed player) each case file describes. b002's window is
# the composed-tick case; b007's single tick is the contested-border case.
SPECS = {
    "b002": {
        "case": "cases/b002-t300-331.ticksteps.json",
        "record": "curr-b002-s1-pangaea",
        "tick0": 300,
        "tick1": 331,
    },
    "b007": {
        "case": "cases/b007-t581-sid2.contested.json",
        "record": "curr-b007-s3-pangaea",
        "tick0": 581,
        "tick1": 582,
        "owner_sid": 2,
    },
}


def fnv1a_u16_le(words):
    h = FNV_OFFSET
    for w in words:
        b = w.to_bytes(2, "little")
        h = ((h ^ b[0]) * FNV_PRIME) & MASK
        h = ((h ^ b[1]) * FNV_PRIME) & MASK
    return h


def load_dump(path):
    out = {}
    for line in open(path):
        rec = json.loads(line)
        if rec.get("type") == "header":
            continue
        out[rec["tick"]] = rec
    return out


def plane(dump, tick):
    p = [0] * (W * H)
    for pl in dump[tick]["players"]:
        for t in pl["ownedTiles"]:
            p[t] = pl["smallId"]
    return p


def players(dump, tick):
    return {p["smallId"]: p for p in dump[tick]["players"]}


def verify_b002(dump, spec, write):
    tick0, tick1 = spec["tick0"], spec["tick1"]
    cur = json.load(open(spec["case"]))
    steps = []
    for tick in range(tick0, tick1 + 1):
        blocks = []
        if tick > tick0:
            before, after = players(dump, tick - 1), players(dump, tick)
            for sid in sorted(after):
                if sid not in before:
                    continue
                new = set(after[sid]["ownedTiles"]) - set(before[sid]["ownedTiles"])
                if not new:
                    continue
                order = after[sid]["ownedOrder"]
                prev_len = len(before[sid]["ownedOrder"])
                claimed = order[prev_len:]
                assert set(claimed) == new, (tick, sid, len(claimed), len(new))
                mine = sorted(before[sid]["ownedTiles"])
                any_ = sorted({t for p in dump[tick - 1]["players"] for t in p["ownedTiles"]})
                blocks.append(
                    {
                        "tick": tick,
                        "ownerSid": sid,
                        "border": before[sid]["borderOrder"],
                        "ownedAny": any_,
                        "ownedMine": mine,
                        "budget": len(claimed),
                        "expectedClaimSet": claimed,
                        "expectedHash": hex(fnv1a_u16_le(plane(dump, tick))),
                    }
                )
        steps.append(
            {
                "tick": tick,
                "expectedHash": hex(fnv1a_u16_le(plane(dump, tick))),
                "blocks": blocks,
            }
        )
    new_file = {
        "game": spec["record"],
        "tick0": tick0,
        "players0": [
            {"smallId": sid, "ownedTiles": players(dump, tick0)[sid]["ownedTiles"]}
            for sid in sorted(players(dump, tick0))
        ],
        "steps": steps,
    }
    return cur, new_file


def verify_b007(dump, spec, write):
    tick0, tick1 = spec["tick0"], spec["tick1"]
    sid = spec["owner_sid"]
    cur = json.load(open(spec["case"]))
    before, after = players(dump, tick0), players(dump, tick1)
    claimed = after[sid]["ownedOrder"][len(before[sid]["ownedOrder"]) :]
    new_file = {
        "game": spec["record"],
        "tick": tick0,
        "claim_tick": tick1,
        "owner_sid": sid,
        "players": [
            {
                "smallId": s,
                "name": p["name"],
                "ownedTiles": p["ownedTiles"],
                "borderOrder": p["borderOrder"],
                "ownedOrder": p["ownedOrder"],
            }
            for s, p in sorted(players(dump, tick0).items())
        ],
        "expected_claims": claimed,
    }
    return cur, new_file


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("verify", "write"):
        print(__doc__)
        return 2
    mode, key, dump_path = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = SPECS[key]
    dump = load_dump(dump_path)
    have = sorted(dump)
    need = [spec["tick0"], spec["tick1"]]
    missing = [t for t in need if t not in dump]
    if missing:
        print(f"dump {dump_path} is missing ticks {missing} (have {have[:3]}..{have[-3:]})")
        return 1
    ok = True
    for t in range(spec["tick0"], spec["tick1"] + 1):
        if t not in dump:
            print(f"dump is missing tick {t}")
            ok = False
    fn = verify_b002 if key == "b002" else verify_b007
    cur, new_file = fn(dump, spec, mode == "write")
    if mode == "write":
        with open(spec["case"], "w") as f:
            json.dump(new_file, f, indent=1)
        print(f"wrote {spec['case']} from {dump_path}")
        return 0
    same = json.dumps(cur, sort_keys=True) == json.dumps(new_file, sort_keys=True)
    if same:
        print(f"{spec['case']} MATCHES {dump_path} (fixed-engine re-derivation agrees)")
        return 0 if ok else 1
    # Report the first field that differs, so the failure is actionable.
    for k in sorted(set(cur) | set(new_file)):
        if json.dumps(cur.get(k), sort_keys=True) != json.dumps(new_file.get(k), sort_keys=True):
            print(f"{spec['case']} DIFFERS from {dump_path} at key '{k}'")
            break
    return 1


if __name__ == "__main__":
    sys.exit(main())
