#!/usr/bin/env python3
"""Online NDJSON tick compare for native vs TS dump streams.

Reads two growing `.ndjson` files (header line + one TickSnapshot per line),
compares checkpoints as they appear, and exits at the first hard divergence.
Used by `scripts/hash_parity.sh` so both engines can run in parallel and be
killed early instead of replaying the rest of a long FFA.

Exit: 0 = agree through EOF, 1 = divergence, 2 = usage/IO error.
Prints machine-readable `DIVERGENCE_TICK=N` and `DIVERGENCE_LAYER=...`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_FIELDS = [
    "alive",
    "tiles",
    "troops",
    "gold",
    "hashBits",
    "hash",
    "unitsHash",
    "numUnits",
]
SOFT_FIELDS = {"troops", "gold"}
# Prefer these labels when classifying the diverge layer.
LAYER_PRIORITY = [
    ("presence", "presence"),
    ("alive", "alive"),
    ("tiles", "tiles"),
    ("unitsHash", "units"),
    ("numUnits", "units"),
    ("hashBits", "hash"),
    ("hash", "hash"),
    ("troops", "troops"),
    ("gold", "gold"),
]


def normalize(field: str, value: Any) -> Any:
    if field == "gold" and value is not None:
        return int(value)
    if field == "unitsHash" and value is not None:
        return int(value)
    if field in ("hashBits", "gameHashBits") and value is not None:
        return str(value)
    return value


def players_by_id(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Join on stable player id — identity can collide for bots (nation:Name)."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in snap.get("players", []):
        key = str(p.get("id") if p.get("id") is not None else p.get("identity"))
        out[key] = p
    return out


def players_by_identity(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Kept for expand dumps that still label by identity in messages.
    return {p["identity"]: p for p in snap.get("players", [])}


def diff_players(
    native: Dict[str, Dict[str, Any]],
    ts: Dict[str, Dict[str, Any]],
    fields: List[str],
) -> List[Tuple[str, str, Any, Any]]:
    diffs: List[Tuple[str, str, Any, Any]] = []
    for ident in sorted(set(native) | set(ts)):
        n = native.get(ident)
        t = ts.get(ident)
        if n is None or t is None:
            diffs.append((ident, "presence", n is not None, t is not None))
            continue
        for f in fields:
            nv = normalize(f, n.get(f))
            tv = normalize(f, t.get(f))
            if nv != tv:
                diffs.append((ident, f, nv, tv))
    return diffs


def classify_layer(diffs: List[Tuple[str, str, Any, Any]]) -> str:
    fields = {d[1] for d in diffs}
    for field, layer in LAYER_PRIORITY:
        if field in fields:
            return layer
    return "unknown"


def hard_diffs(diffs: List[Tuple[str, str, Any, Any]]) -> List[Tuple[str, str, Any, Any]]:
    hard = [d for d in diffs if d[1] not in SOFT_FIELDS]
    return hard if hard else diffs


def wait_line(fh, path: str, deadline: float) -> Optional[str]:
    """Read one line from an open file, waiting for writers to append."""
    while True:
        pos = fh.tell()
        line = fh.readline()
        if line:
            return line
        if time.time() > deadline:
            return None
        # Writer may still be catching up; also detect writer death via mtime stall
        # is handled by caller timeout. Sleep briefly.
        fh.seek(pos)
        time.sleep(0.05)


def open_stream(path: str, timeout: float):
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise FileNotFoundError(path)
        time.sleep(0.05)
    return open(path, "r", encoding="utf-8")


def iter_snaps(fh, path: str, idle_timeout: float) -> Iterable[Dict[str, Any]]:
    # Skip/consume header
    while True:
        line = wait_line(fh, path, time.time() + idle_timeout)
        if line is None:
            return
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "header":
            break
        # No header — treat as first snap
        yield obj
        break
    while True:
        line = wait_line(fh, path, time.time() + idle_timeout)
        if line is None:
            return
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def unit_deep_diffs(
    n_units: Optional[List[Dict[str, Any]]],
    t_units: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if n_units is None and t_units is None:
        return []
    n_by_id = {u["id"]: u for u in (n_units or [])}
    t_by_id = {u["id"]: u for u in (t_units or [])}
    out = []
    for uid in sorted(set(n_by_id) | set(t_by_id)):
        nu, tu = n_by_id.get(uid), t_by_id.get(uid)
        if nu is None or tu is None:
            out.append(f"  unit#{uid} presence native={nu is not None} ts={tu is not None}")
            continue
        for f in (
            "unitType",
            "tile",
            "hash",
            "level",
            "underConstruction",
            "health",
            "veterancy",
            "veterancyProgress",
            "targetTile",
            "patrolTile",
            "retreatPort",
            "retreating",
            "docked",
        ):
            if nu.get(f) != tu.get(f):
                out.append(f"  unit#{uid} {f} native={nu.get(f)} ts={tu.get(f)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("native_ndjson")
    ap.add_argument("ts_ndjson")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    ap.add_argument("--skip-before", type=int, default=5)
    ap.add_argument(
        "--idle-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the next line before treating a stream as finished",
    )
    ap.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for ndjson files to appear",
    )
    ap.add_argument("--compare-game-hash", action="store_true", default=True)
    args = ap.parse_args()
    fields = [f for f in args.fields.split(",") if f]

    try:
        n_fh = open_stream(args.native_ndjson, args.startup_timeout)
        t_fh = open_stream(args.ts_ndjson, args.startup_timeout)
    except Exception as e:
        print(f"[stream_compare_ticks] open error: {e}", file=sys.stderr)
        return 2

    n_iter = iter_snaps(n_fh, args.native_ndjson, args.idle_timeout)
    t_iter = iter_snaps(t_fh, args.ts_ndjson, args.idle_timeout)

    compared = 0
    while True:
        try:
            n_snap = next(n_iter)
        except StopIteration:
            n_snap = None
        try:
            t_snap = next(t_iter)
        except StopIteration:
            t_snap = None

        if n_snap is None and t_snap is None:
            print(
                f"[stream_compare_ticks] no divergence across {compared} checkpoints",
                file=sys.stderr,
            )
            return 0
        if n_snap is None or t_snap is None:
            print(
                f"[stream_compare_ticks] stream ended early "
                f"(native={'yes' if n_snap else 'no'}, ts={'yes' if t_snap else 'no'}) "
                f"after {compared} checkpoints",
                file=sys.stderr,
            )
            return 2

        n_tick = int(n_snap["tick"])
        t_tick = int(t_snap["tick"])
        if n_tick != t_tick:
            print(
                f"[stream_compare_ticks] tick alignment break native={n_tick} ts={t_tick}",
                file=sys.stderr,
            )
            print(f"DIVERGENCE_TICK={min(n_tick, t_tick)}")
            print("DIVERGENCE_LAYER=tick_align")
            return 1

        if n_tick <= args.skip_before:
            continue

        compared += 1
        if args.compare_game_hash and n_snap.get("gameHash") != t_snap.get("gameHash"):
            # Still do player diff so we report the real field(s).
            pass

        all_diffs = diff_players(
            players_by_id(n_snap), players_by_id(t_snap), fields
        )
        diffs = hard_diffs(all_diffs)
        n_ghb = n_snap.get("gameHashBits")
        t_ghb = t_snap.get("gameHashBits")
        if n_ghb is not None and t_ghb is not None:
            # Prefer IEEE bits — JSON numbers / i64 truncation disagree past 2^53
            # even when the underlying f64 hash matches.
            game_hash_mismatch = str(n_ghb) != str(t_ghb)
        else:
            game_hash_mismatch = (
                args.compare_game_hash
                and n_snap.get("gameHash") is not None
                and t_snap.get("gameHash") is not None
                and n_snap.get("gameHash") != t_snap.get("gameHash")
            )
        if not args.compare_game_hash:
            game_hash_mismatch = False
        if not diffs and not game_hash_mismatch:
            continue

        layer = classify_layer(diffs) if diffs else "gameHash"
        print(f"[stream_compare_ticks] FIRST DIVERGENCE at tick {n_tick} layer={layer}")
        if game_hash_mismatch or (
            n_snap.get("gameHash") != t_snap.get("gameHash")
            and (n_ghb is not None or t_ghb is not None)
        ):
            print(
                f"  gameHash native={n_snap.get('gameHash')} ts={t_snap.get('gameHash')}"
            )
            print(f"  gameHashBits native={n_ghb} ts={t_ghb}")
        n_by_id = players_by_id(n_snap)
        t_by_id = players_by_id(t_snap)
        report = diffs if diffs else all_diffs
        for pid, field, nv, tv in report[:40]:
            label = pid
            np = n_by_id.get(pid) or {}
            tp = t_by_id.get(pid) or {}
            ident = np.get("identity") or tp.get("identity")
            if ident:
                label = f"{ident}(id={pid})"
            print(f"  {label}: {field} native={nv} ts={tv}")
            if field in ("unitsHash", "numUnits", "hash", "hashBits"):
                for line in unit_deep_diffs(np.get("units"), tp.get("units")):
                    print(line)
        if len(report) > 40:
            print(f"  ... ({len(report) - 40} more field diffs)")
        if not report and game_hash_mismatch:
            print(
                "  (no per-player field diffs after id-join; "
                "gameHashBits diverge — check player iteration order in hash fold)"
            )
        print(f"DIVERGENCE_TICK={n_tick}")
        print(f"DIVERGENCE_LAYER={layer}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
