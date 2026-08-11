#!/usr/bin/env python3
"""Prune djmango/openfront-rl toward a sparse ~20GB layout.

Keeps:
  - ppo_v11 latest/manifest + curriculum_* + thinned policy_update milestones
  - prior major ppo_* runs: latest.* + manifest.json only
  - .gitattributes / README.md

Deletes dense every-~5-update milestone histories and legacy bc_*/early ppo_*
blobs that are near-duplicates for Hub serving.

Usage:
  uv run --no-project --with huggingface_hub python scripts/hf_prune_openfront_rl.py --dry-run
  uv run --no-project --with huggingface_hub python scripts/hf_prune_openfront_rl.py --execute
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter

from huggingface_hub import CommitOperationDelete, HfApi

REPO_ID = "djmango/openfront-rl"
CURRENT_RUN = "ppo_v11"
MAJOR_PRIOR = {
    "ppo_v10",
    "ppo_v9",
    "ppo_v86",
    "ppo_v85",
    "ppo_v84",
    "ppo_v83",
    "ppo_v82",
    "ppo_v81",
}
ROOT_KEEP = {".gitattributes", "README.md"}
MILESTONE_RE = re.compile(r"^policy_update(\d+)\.(safetensors|state\.json)$")


def select_v11_updates(updates: list[int]) -> set[int]:
    if not updates:
        return set()
    max_u = max(updates)
    keep: set[int] = {updates[0], max_u}
    for u in updates:
        recent = u >= max_u - 200
        step = 25 if recent else 100
        if u % step == 0:
            keep.add(u)
    return keep


def plan(files: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[tuple[str, int]], set[int]]:
    v11_updates = sorted(
        int(m.group(1))
        for path, _ in files
        if path.startswith(f"{CURRENT_RUN}/")
        for m in [MILESTONE_RE.fullmatch(path.split("/", 1)[1])]
        if m
    )
    keep_updates = select_v11_updates(v11_updates)
    keep: list[tuple[str, int]] = []
    delete: list[tuple[str, int]] = []

    for path, size in files:
        if path in ROOT_KEEP:
            keep.append((path, size))
            continue

        prefix, _, name = path.partition("/")
        if not name:
            # unexpected root file
            if size < 1_000_000:
                keep.append((path, size))
            else:
                delete.append((path, size))
            continue

        if prefix == CURRENT_RUN:
            if name in {"latest.safetensors", "latest.state.json", "manifest.json"}:
                keep.append((path, size))
            elif name.startswith("curriculum_"):
                keep.append((path, size))
            else:
                m = MILESTONE_RE.fullmatch(name)
                if m and int(m.group(1)) in keep_updates:
                    keep.append((path, size))
                else:
                    delete.append((path, size))
            continue

        if prefix in MAJOR_PRIOR and name in {
            "latest.safetensors",
            "latest.state.json",
            "manifest.json",
        }:
            keep.append((path, size))
            continue

        delete.append((path, size))

    return keep, delete, keep_updates


def batched(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", REPO_ID))
    parser.add_argument("--execute", action="store_true", help="actually delete (default dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="print plan only (default)")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--upload-readme", type=str, default="", help="path to README.md to upload")
    args = parser.parse_args()
    dry = not args.execute

    token = os.environ.get("HF_TOKEN")
    if args.execute and not token:
        raise SystemExit("HF_TOKEN required for --execute")
    api = HfApi(token=token)
    info = api.repo_info(args.repo_id, repo_type="model", files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
    keep, delete, keep_updates = plan(files)
    keep_b = sum(s for _, s in keep)
    del_b = sum(s for _, s in delete)
    print(f"repo={args.repo_id} files={len(files)}")
    print(f"keep {len(keep)} files ({keep_b / 1e9:.2f} GB)")
    print(f"delete {len(delete)} files ({del_b / 1e9:.2f} GB)")
    print(f"ppo_v11 milestone updates kept ({len(keep_updates)}): {sorted(keep_updates)}")
    print("delete by prefix:", Counter(p.split("/")[0] for p, _ in delete).most_common())

    if dry:
        print("[dry-run] no deletions performed; pass --execute to apply")
        if args.upload_readme:
            print(f"[dry-run] would upload README from {args.upload_readme}")
        return 0

    paths = [p for p, _ in delete]
    for i, batch in enumerate(batched(paths, args.batch_size), 1):
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        msg = f"Prune dense checkpoints (batch {i}, {len(batch)} files)"
        print(f"[prune] commit {i}: deleting {len(batch)} files ...", flush=True)
        for attempt in range(6):
            try:
                api.create_commit(
                    repo_id=args.repo_id,
                    repo_type="model",
                    operations=ops,
                    commit_message=msg,
                )
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                wait = min(60, 2**attempt)
                print(f"[prune] retry after {exc!r} sleep={wait}s", flush=True)
                time.sleep(wait)
        time.sleep(1)

    if args.upload_readme:
        from pathlib import Path

        readme_path = Path(args.upload_readme).expanduser().resolve()
        if not readme_path.is_file():
            raise SystemExit(f"README not found: {readme_path}")
        print(f"[prune] uploading model card {readme_path}", flush=True)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card (README.md)",
        )

    info2 = api.repo_info(args.repo_id, repo_type="model", files_metadata=True)
    total = sum(s.size or 0 for s in (info2.siblings or []))
    print(f"[prune] done. files={len(info2.siblings or [])} total={total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
