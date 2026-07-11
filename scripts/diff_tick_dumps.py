#!/usr/bin/env python3
"""Diff a native `tick_dump` JSON against a TS `dump_ts_tick_state.ts` JSON
and report the first checkpoint tick at which any player's state diverges.

Both dumpers share one schema (see rust/engine/src/bin/tick_dump.rs and
scripts/dump_ts_tick_state.ts): a list of per-`every`-tick snapshots, each
with a list of per-player {identity, id, name, tiles, troops, gold, alive,
hash, numUnits}. `identity` (clientID- or nation-name-keyed) is the join key
- stable across both engines, unlike array position.

This is the comparison half of bisect_parity.sh's coarse-then-fine loop; run
directly if you already have two dump files and just want the diff.

Usage:
  python3 scripts/diff_tick_dumps.py <native.json> <ts.json> [--fields tiles,troops,gold,hash,numUnits,alive]

Exit code 0 = no divergence found in the overlapping tick range, 1 = found
(and printed), 2 = usage/load error.

Note: both dump harnesses (tick_dump.rs / dump_ts_tick_state.ts) can show a
transient `alive` mismatch in the first few ticks even for engines that
agree perfectly later - nation/bot spawn registration order during game
setup isn't guaranteed to land on the exact same tick between the two
harnesses' own init paths (tick_dump.rs's `bootstrap::game_from_record` vs
dump_ts_tick_state.ts's `createGameRunner()`-mirroring init), independent of
whatever the *real* replay path (outcome_gate/replay.ts) does. --skip-before
defaults to a few ticks to filter this out; raise it if early-game noise
still shows up, but don't mistake it for a genuine bug without checking
whether outcome_gate's own replay (not this tool) agrees.
"""
import json
import sys
import argparse


DEFAULT_FIELDS = ["alive", "tiles", "troops", "gold", "hash", "numUnits"]
# `troops` is TS-side-rounded and gold can differ by rounding-mode noise on
# both sides even when nothing is actually wrong; treat these as "soft" -
# still reported, but don't count as the *first* divergence on their own
# unless nothing harder (identity/alive/hash) differs at the same tick.
SOFT_FIELDS = {"troops", "gold"}


def load(path):
    with open(path) as f:
        data = json.load(f)
    by_tick = {}
    for snap in data["ticks"]:
        by_tick[snap["tick"]] = {p["identity"]: p for p in snap["players"]}
    return data, by_tick


def camel(field):
    # native's tick_dump.rs is serde camelCase too, so no translation needed,
    # but keep this indirection in case a snake_case dump ever sneaks in.
    return field


def normalize(field, value):
    # `gold` is a JS BigInt on the TS side, serialized as a decimal string to
    # dodge Number precision loss (see dump_ts_tick_state.ts); native emits a
    # plain JSON integer. Compare numerically on both sides so this isn't a
    # spurious int-vs-str divergence on every single tick.
    if field == "gold" and value is not None:
        return int(value)
    return value


def diff_at_tick(native_players, ts_players, fields):
    diffs = []
    all_ids = set(native_players) | set(ts_players)
    for ident in sorted(all_ids):
        n = native_players.get(ident)
        t = ts_players.get(ident)
        if n is None or t is None:
            diffs.append((ident, "presence", n is not None, t is not None))
            continue
        for f in fields:
            nv = normalize(f, n.get(camel(f)))
            tv = normalize(f, t.get(camel(f)))
            if nv != tv:
                diffs.append((ident, f, nv, tv))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native_json")
    ap.add_argument("ts_json")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    ap.add_argument("--skip-before", type=int, default=5, help="ignore checkpoints at or before this tick (init-order noise, see module docstring)")
    args = ap.parse_args()
    fields = args.fields.split(",")

    try:
        native_data, native_by_tick = load(args.native_json)
        ts_data, ts_by_tick = load(args.ts_json)
    except Exception as e:
        print(f"[diff_tick_dumps] load error: {e}", file=sys.stderr)
        return 2

    common_ticks = sorted(t for t in set(native_by_tick) & set(ts_by_tick) if t > args.skip_before)
    if not common_ticks:
        print("[diff_tick_dumps] no overlapping checkpoint ticks between the two dumps", file=sys.stderr)
        return 2

    print(
        f"[diff_tick_dumps] comparing {len(common_ticks)} checkpoints "
        f"(native final={native_data['finalTick']}, ts final={ts_data['finalTick']})",
        file=sys.stderr,
    )

    first_hard_tick = None
    first_any_tick = None
    for tick in common_ticks:
        diffs = diff_at_tick(native_by_tick[tick], ts_by_tick[tick], fields)
        if not diffs:
            continue
        if first_any_tick is None:
            first_any_tick = (tick, diffs)
        hard = [d for d in diffs if d[1] == "presence" or d[1] not in SOFT_FIELDS]
        if hard and first_hard_tick is None:
            first_hard_tick = (tick, diffs)
            break

    report = first_hard_tick or first_any_tick
    if report is None:
        print(f"[diff_tick_dumps] no divergence in {fields} across {len(common_ticks)} checkpoints - engines agree here")
        return 0

    tick, diffs = report
    print(f"FIRST DIVERGENCE at checkpoint tick {tick}:")
    for ident, field, nv, tv in diffs:
        print(f"  {ident}: {field} native={nv!r} ts={tv!r}")
    # Machine-readable line for bisect_parity.sh to grep out the tick.
    print(f"DIVERGENCE_TICK={tick}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
