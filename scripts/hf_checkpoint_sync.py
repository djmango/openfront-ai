#!/usr/bin/env python3
"""Safely restore and back up oftrain safetensors checkpoints on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable


RESTORE_FILES = ("latest.safetensors", "latest.state.json", "manifest.json")
EXACT_FILES = {
    *RESTORE_FILES,
    "best_eval.safetensors",
    "best_eval.state.json",
}
MILESTONE_RE = re.compile(r"^policy_update\d+\.(?:safetensors|state\.json)$")
CHUNK_SIZE = 8 * 1024 * 1024


class UnstableSourceError(RuntimeError):
    """A trainer output changed while it was being snapshotted."""


def validate_manifest_bytes(body: bytes) -> dict[str, Any]:
    """Validate checkpoint identity before a restore is made visible."""
    manifest = json.loads(body)
    if manifest.get("format") != "oftrain-safetensors":
        raise ValueError("unsupported checkpoint format")
    if manifest.get("manifest_schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest schema")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("missing checkpoint architecture")
    schema = architecture.get("schema_version")
    if schema not in {1, 2}:
        raise ValueError(f"unsupported checkpoint architecture schema: {schema!r}")
    if schema == 2:
        recurrent = architecture.get("recurrent")
        required = {
            "cell": "gru",
            "context_schema": "action-outcome-v1",
            "residual_initialization": "zero-output-projection",
            "hidden_reset_policy": "episode_done",
        }
        if not isinstance(recurrent, dict) or any(
            recurrent.get(key) != value for key, value in required.items()
        ):
            raise ValueError("invalid recurrent architecture schema")
        for positive in (
            "hidden_size",
            "context_features",
            "context_embedding",
            "bptt_length",
            "rollout_length",
        ):
            if not isinstance(recurrent.get(positive), int) or recurrent[positive] <= 0:
                raise ValueError(f"invalid recurrent {positive}")
    return manifest


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def snapshot(source: Path, directory: Path) -> tuple[Path, str, int, tuple[int, ...]]:
    before = signature(source)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{source.name}.{os.getpid()}.tmp"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, tmp.open("xb") as dst:
            while chunk := src.read(CHUNK_SIZE):
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if signature(source) != before:
            raise UnstableSourceError(f"{source} changed while being snapshotted")
        sha = digest.hexdigest()
        stable = directory / f"{sha}-{source.name}"
        if stable.exists():
            tmp.unlink()
        else:
            os.replace(tmp, stable)
            stable.chmod(0o444)
        return stable, sha, before[2], before
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def discover(checkpoint_dir: Path) -> list[Path]:
    """Select only current oftrain interchange files; never .ot or policy.pt."""
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file()
            and (path.name in EXACT_FILES or MILESTONE_RE.fullmatch(path.name))
        ),
        key=lambda path: path.name,
    )


def normalize_prefix(prefix: str) -> str:
    normalized = prefix.strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in PurePosixPath(normalized).parts):
        raise ValueError(f"invalid Hugging Face run prefix: {prefix!r}")
    return normalized


def is_transient(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return (
        status in {408, 409, 425, 429}
        or isinstance(status, int)
        and status >= 500
        or isinstance(error, (ConnectionError, TimeoutError))
    )


def retry(
    operation: Callable[[], Any],
    *,
    max_retries: int,
    sleep: Callable[[float], None],
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_retries or not is_transient(error):
                raise
            sleep(min(60.0, 2.0**attempt))
    raise AssertionError("unreachable")


class CheckpointSync:
    def __init__(
        self,
        checkpoint_dir: Path,
        repo_id: str,
        run_prefix: str,
        api: Any,
        *,
        dry_run: bool = False,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.repo_id = repo_id
        self.run_prefix = normalize_prefix(run_prefix)
        self.api = api
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.sleep = sleep
        self.state_dir = checkpoint_dir / ".hf-sync"
        self.snapshot_dir = self.state_dir / "snapshots"
        self.state_path = self.state_dir / "sync-manifest.json"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                self.state["repo_id"] != repo_id
                or self.state["run_prefix"] != self.run_prefix
            ):
                raise RuntimeError("sync state targets a different HF run")
        except FileNotFoundError:
            self.state = {
                "schema": 1,
                "repo_id": repo_id,
                "run_prefix": self.run_prefix,
                "files": {},
            }

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def sync_once(self) -> int:
        uploaded = 0
        for source in discover(self.checkpoint_dir):
            remote = f"{self.run_prefix}/{source.name}"
            stable: Path | None = None
            try:
                source_signature = list(signature(source))
                previous = self.state["files"].get(remote, {})
                if previous.get("source_signature") == source_signature:
                    continue
                stable, sha, size, source_signature_tuple = snapshot(
                    source, self.snapshot_dir
                )
                if previous.get("sha256") == sha:
                    previous["source_signature"] = list(source_signature_tuple)
                    self.save_state()
                    continue
                if not self.dry_run:
                    retry(
                        lambda: self.api.upload_file(
                            path_or_fileobj=str(stable),
                            path_in_repo=remote,
                            repo_id=self.repo_id,
                            repo_type="model",
                            commit_message=f"Sync {source.name} ({sha[:12]})",
                        ),
                        max_retries=self.max_retries,
                        sleep=self.sleep,
                    )
                    self.state["files"][remote] = {
                        "sha256": sha,
                        "bytes": size,
                        "source_signature": list(source_signature_tuple),
                        "uploaded_at": utc_now(),
                    }
                    self.save_state()
                uploaded += 1
                print(f"[hf-sync] {'would upload' if self.dry_run else 'uploaded'} {remote}")
            except UnstableSourceError as error:
                print(f"[hf-sync] skipped unstable source: {error}", flush=True)
            except Exception as error:
                print(f"[hf-sync] upload failed for {source.name}: {error}", flush=True)
            finally:
                if stable is not None:
                    stable.unlink(missing_ok=True)
        return uploaded


def restore_latest(api: Any, repo_id: str, run_prefix: str, destination: Path) -> bool:
    """Restore only a complete, manifest-validated safetensors checkpoint."""
    prefix = normalize_prefix(run_prefix)
    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    try:
        for name in RESTORE_FILES:
            staged[name] = Path(
                api.hf_hub_download(repo_id, f"{prefix}/{name}", repo_type="model")
            )
        validate_manifest_bytes(staged["manifest.json"].read_bytes())
    except Exception as error:
        print(f"[hf-sync] no complete compatible safetensors checkpoint to restore: {error}")
        return False
    for name in RESTORE_FILES:
        target = destination / name
        tmp = target.with_name(f".{name}.restore.tmp")
        tmp.write_bytes(staged[name].read_bytes())
        os.replace(tmp, target)
    print(f"[hf-sync] restored {prefix}/latest.safetensors and state")
    return True


def create_api(token: str | None = None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "djmango/openfront-rl"))
    parser.add_argument("--run-prefix", default=os.environ.get("HF_RUN_PREFIX", "ppo_v81"))
    parser.add_argument("--interval", type=float, default=float(os.environ.get("HF_SYNC_INTERVAL_SECONDS", "600")))
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore-latest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.max_retries < 0:
        raise SystemExit("interval must be positive and max-retries non-negative")
    api = create_api(os.environ.get("HF_TOKEN"))
    if args.restore_latest:
        return 0 if restore_latest(
            api, args.repo_id, args.run_prefix, args.checkpoint_dir
        ) else 1
    if not args.dry_run:
        api.whoami()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    sync = CheckpointSync(
        args.checkpoint_dir,
        args.repo_id,
        args.run_prefix,
        api,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )
    sync.save_state()
    while True:
        sync.sync_once()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
