#!/usr/bin/env python3
"""Compare native vs TS tick-dump JSON (see rust/engine/src/bin/tick_dump.rs
and scripts/dump_ts_tick_state.ts) and report the first tick where any
entity's tile count diverges by more than a relative threshold.

Usage:
    python3 scripts/compare_tick_dumps.py <native.json> <ts.json> \
        [--rel-threshold 0.10] [--abs-threshold 20] [--out report.json]
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def index_by_tick(dump):
    return {t["tick"]: t for t in dump["ticks"]}


def entities_by_identity(tick_snapshot):
    return {p["identity"]: p for p in tick_snapshot["players"]}


def rel_diff(a, b):
    denom = max(abs(a), abs(b), 1)
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("ts")
    ap.add_argument("--rel-threshold", type=float, default=0.10)
    ap.add_argument("--abs-threshold", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    native = load(args.native)
    ts = load(args.ts)
    native_by_tick = index_by_tick(native)
    ts_by_tick = index_by_tick(ts)
    common_ticks = sorted(set(native_by_tick) & set(ts_by_tick))
    if not common_ticks:
        print("no common ticks between the two dumps", file=sys.stderr)
        sys.exit(2)

    rows = []
    first_divergence = None
    for tick in common_ticks:
        n_snap = native_by_tick[tick]
        t_snap = ts_by_tick[tick]
        n_ent = entities_by_identity(n_snap)
        t_ent = entities_by_identity(t_snap)
        all_ids = sorted(set(n_ent) | set(t_ent))
        tick_max_rel = 0.0
        worst = None
        for ident in all_ids:
            n_tiles = n_ent.get(ident, {}).get("tiles", 0)
            t_tiles = t_ent.get(ident, {}).get("tiles", 0)
            if n_tiles < args.abs_threshold and t_tiles < args.abs_threshold:
                continue
            d = rel_diff(n_tiles, t_tiles)
            if d > tick_max_rel:
                tick_max_rel = d
                worst = {
                    "identity": ident,
                    "nativeTiles": n_tiles,
                    "tsTiles": t_tiles,
                    "nativeAlive": n_ent.get(ident, {}).get("alive"),
                    "tsAlive": t_ent.get(ident, {}).get("alive"),
                    "relDiff": d,
                }
        row = {
            "tick": tick,
            "nativeTotalOwned": n_snap["totalOwnedTiles"],
            "tsTotalOwned": t_snap["totalOwnedTiles"],
            "nativeInSpawn": n_snap.get("inSpawnPhase"),
            "tsInSpawn": t_snap.get("inSpawnPhase"),
            "worstEntity": worst,
        }
        rows.append(row)
        if first_divergence is None and worst is not None and worst["relDiff"] > args.rel_threshold:
            first_divergence = row

    print(f"compared {len(common_ticks)} common ticks "
          f"({common_ticks[0]}..{common_ticks[-1]})")
    print(f"native final_tick={native['finalTick']} ts final_tick={ts['finalTick']}")
    if first_divergence is None:
        print(f"no divergence found above rel-threshold={args.rel_threshold} "
              f"(abs-threshold={args.abs_threshold})")
    else:
        print(f"FIRST DIVERGENCE at tick {first_divergence['tick']}:")
        print(json.dumps(first_divergence, indent=2))

    print("\n--- per-tick summary (tick, native_total, ts_total, worst entity rel diff) ---")
    for row in rows:
        worst = row["worstEntity"]
        wd = f"{worst['relDiff']:.3f} ({worst['identity']})" if worst else "-"
        marker = " <== DIVERGENCE" if first_divergence and row["tick"] == first_divergence["tick"] else ""
        print(f"tick={row['tick']:>6}  native_total={row['nativeTotalOwned']:>8}  "
              f"ts_total={row['tsTotalOwned']:>8}  worst_rel_diff={wd}{marker}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "firstDivergence": first_divergence}, f, indent=2)
        print(f"\nwrote full report to {args.out}")

    sys.exit(0 if first_divergence is None else 1)


if __name__ == "__main__":
    main()
