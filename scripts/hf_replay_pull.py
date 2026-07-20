#!/usr/bin/env python3
"""Pull latest parquet shard(s) from openfront-replays into local records/.

Materializes `record_json` columns as GameRecord JSON files so
`ofshowcase archive` can serve them for Watch / client replay.

Usage:
  uv run --with pyarrow python scripts/hf_replay_pull.py
  uv run --with pyarrow python scripts/hf_replay_pull.py --limit 50 --out /data/records
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

DATASET_REPO = os.environ.get("HF_REPLAYS_REPO", "djmango/openfront-replays")


def default_out() -> Path:
    return Path(os.environ.get("DATA_DIR", ".")) / "records" / "hf-replays"


def list_shards(api, repo: str) -> list[str]:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        raise SystemExit(f"list_repo_files failed: {e}") from e
    shards = sorted(
        f for f in files if f.startswith("data/train-") and f.endswith(".parquet")
    )
    return shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DATASET_REPO)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=100, help="max games to materialize")
    ap.add_argument("--shards", type=int, default=1, help="how many newest shards to read")
    args = ap.parse_args()

    out = args.out or default_out()
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    token = os.environ.get("HF_TOKEN", "").strip() or None
    api = HfApi(token=token)
    shards = list_shards(api, args.repo)
    if not shards:
        print(f"no parquet shards in {args.repo}")
        return

    chosen = shards[-args.shards :]
    written = 0
    latest_game_id = None
    latest_created = -1
    with tempfile.TemporaryDirectory(prefix="of-replay-pull-") as tmp:
        tmp_path = Path(tmp)
        for remote in reversed(chosen):
            local = hf_hub_download(
                repo_id=args.repo,
                filename=remote,
                repo_type="dataset",
                token=token,
                local_dir=tmp_path,
            )
            table = pq.read_table(local)
            # Prefer newest created_at within the shard.
            rows = table.to_pylist()
            rows.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
            for row in rows:
                if written >= args.limit:
                    break
                gid = str(row.get("game_id") or "")
                raw = row.get("record_json") or ""
                if not gid or not raw:
                    continue
                dest = out / f"{gid}.json"
                if isinstance(raw, bytes):
                    dest.write_bytes(raw)
                else:
                    # Normalize to compact JSON file.
                    try:
                        dest.write_text(json.dumps(json.loads(raw), separators=(",", ":")))
                    except json.JSONDecodeError:
                        dest.write_text(str(raw))
                thinking = row.get("thinking_json") or ""
                if thinking:
                    if isinstance(thinking, bytes):
                        thinking = thinking.decode("utf-8", errors="replace")
                    (out / f"{gid}.thinking.json").write_text(str(thinking))
                created = int(row.get("created_at") or 0)
                if created >= latest_created:
                    latest_created = created
                    latest_game_id = gid
                written += 1
            if written >= args.limit:
                break

    index = {
        "repo": args.repo,
        "count": written,
        "featured_game_id": latest_game_id,
        "out": str(out),
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {written} GameRecords under {out} (featured={latest_game_id})")


if __name__ == "__main__":
    main()
