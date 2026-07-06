"""Download archived OpenFront games from the public API.

Lists finished games in a time window, filters to real multiplayer games,
downloads each full record (including the per-turn intent log), and stores
them gzipped, bucketed by the engine commit the game ran on:

    records/<gitCommit>/<gameID>.json.gz

The commit bucket matters: replays are only deterministic on the same engine
commit, so datagen/replay.ts should be run against the matching submodule
checkout (or accept hash-verification failures on drifted games).

Usage:
    uv run python scripts/fetch_games.py \
        --start 2026-07-03T00:00:00Z --end 2026-07-05T00:00:00Z \
        --min-players 8 --max-games 500

API docs: openfront/docs/API.md (max 2-day window, 1000 games per page).
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

API = "https://api.openfront.io/public"
UA = {"User-Agent": "openfront-ai-research (github.com/djmango/openfront-ai)"}


def get_json(url: str, retries: int = 4) -> object:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  retry {attempt + 1} after {wait}s: {e}")
            time.sleep(wait)
    raise AssertionError("unreachable")


def list_games(start: str, end: str) -> list[dict]:
    """Page through the listing endpoint for one <=2 day window."""
    games: list[dict] = []
    offset = 0
    while True:
        params = {
            "start": start,
            "end": end,
            "type": "Public",
            "limit": 1000,
            "offset": offset,
        }
        url = f"{API}/games?{urllib.parse.urlencode(params)}"
        page = get_json(url)
        assert isinstance(page, list)
        games.extend(page)
        if len(page) < 1000:
            return games
        offset += len(page)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="ISO timestamp")
    ap.add_argument("--end", required=True, help="ISO timestamp")
    ap.add_argument("--out", default="records")
    ap.add_argument("--min-players", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=500)
    ap.add_argument(
        "--mode", default=None, help='optional filter: "Free For All" or "Team"'
    )
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between fetches")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    have = {p.stem.removesuffix(".json") for p in out_root.glob("*/*.json.gz")}

    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    # The listing endpoint caps the window at 2 days; walk it in chunks.
    listed: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=2), end)
        chunk = list_games(iso(cursor), iso(chunk_end))
        listed.extend(chunk)
        print(f"listed {len(chunk)} games in {iso(cursor)}..{iso(chunk_end)}")
        cursor = chunk_end

    candidates = [
        g
        for g in listed
        if (g.get("numPlayers") or 0) >= args.min_players
        and (args.mode is None or g.get("mode") == args.mode)
    ]
    # Biggest lobbies first: most human decisions per download.
    candidates.sort(key=lambda g: -(g.get("numPlayers") or 0))
    print(
        f"{len(listed)} public games listed, {len(candidates)} pass filters "
        f"(min {args.min_players} players), downloading up to {args.max_games}"
    )

    done = failed = 0
    for g in candidates:
        if done >= args.max_games:
            break
        gid = g["game"]
        if gid in have:
            done += 1
            continue
        try:
            record = get_json(f"{API}/game/{gid}")
        except Exception as e:
            print(f"[{gid}] fetch failed: {e}")
            failed += 1
            continue
        if not isinstance(record, dict) or "turns" not in record:
            print(f"[{gid}] no turn data, skipping")
            failed += 1
            continue
        commit = record.get("gitCommit", "unknown")[:12]
        dest = out_root / commit / f"{gid}.json.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(json.dumps(record).encode(), 6))
        done += 1
        n_turns = len(record["turns"])
        n_hashes = sum(1 for t in record["turns"] if t.get("hash") is not None)
        print(
            f"[{gid}] saved: {g.get('numPlayers')} players, {n_turns} turns, "
            f"{n_hashes} hashes, commit {commit} ({done}/{args.max_games})"
        )
        time.sleep(args.sleep)

    print(f"done: {done} saved, {failed} failed")
    by_commit: dict[str, int] = {}
    for p in out_root.glob("*/*.json.gz"):
        by_commit[p.parent.name] = by_commit.get(p.parent.name, 0) + 1
    for c, n in sorted(by_commit.items(), key=lambda kv: -kv[1]):
        print(f"  commit {c}: {n} games")


if __name__ == "__main__":
    main()
