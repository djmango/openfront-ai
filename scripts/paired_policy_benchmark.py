"""Paired fixed-seed Rust-vs-frozen-policy benchmark (indirect).

All policies face the same scripted bots in separate episodes. This is not a
head-to-head tournament: the engine exposes only one controllable human.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXPECTED_SCHEMAS = {
    "current_rust": {
        "grid_channels": 89, "player_features": 21, "scalars": 11,
        "local_planes": 5, "actions": 21, "build_types": 7,
        "nuke_types": 5, "quantity": "beta",
    },
    "ppo_v5": {
        "grid_channels": 43, "player_features": 12, "scalars": 8,
        "local_planes": 4, "actions": 14, "build_types": 6,
        "nuke_types": 3, "quantity": "categorical-5",
    },
    "ppo_v7": {
        "grid_channels": 89, "player_features": 21, "scalars": 11,
        "local_planes": 5, "actions": 21, "build_types": 7,
        "nuke_types": 5, "quantity": "beta",
    },
}
SOURCE_REFS = {"ppo_v5": "23197d4^", "ppo_v7": "HEAD"}


def current_engine_path(name: str) -> Path:
    candidates = []
    if os.environ.get("OPENFRONT_ENGINE_ROOT"):
        candidates.append(Path(os.environ["OPENFRONT_ENGINE_ROOT"]) / name)
    candidates.extend((REPO / name, REPO.parent / name))
    for candidate in candidates:
        marker = (
            candidate / "node_modules" / ".bin" / "tsx"
            if name == "openfront"
            else candidate / "env.ts"
        )
        if marker.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"current engine path {name!r} not found; set OPENFRONT_ENGINE_ROOT"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    if len(values) == 1:
        return [values[0], values[0]]
    a = np.asarray(values, dtype=np.float64)
    means = a[rng.integers(0, len(a), size=(20_000, len(a)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def wilson_ci(wins: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    p = wins / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def validate(reports: dict[str, dict]) -> list[tuple]:
    for label, report in reports.items():
        if report.get("mode") != "indirect-scripted-bot":
            raise ValueError(f"{label}: unexpected mode {report.get('mode')!r}")
        if report.get("schema") != EXPECTED_SCHEMAS[label]:
            raise ValueError(
                f"{label}: incompatible schema; expected {EXPECTED_SCHEMAS[label]}, "
                f"got {report.get('schema')}"
            )
    baseline = reports["current_rust"]
    keys = ["engine", "stage", "max_ticks"]
    for label, report in reports.items():
        for key in keys:
            if report.get(key) != baseline.get(key):
                raise ValueError(
                    f"{label}: {key}={report.get(key)!r} differs from "
                    f"current_rust={baseline.get(key)!r}"
                )
    scenario_keys = (
        "index", "seed", "map", "bots", "difficulty", "nations",
        "decision_ticks",
    )
    scenarios = [
        tuple(e[key] for key in scenario_keys) for e in baseline["episodes"]
    ]
    for label, report in reports.items():
        other = [
            tuple(e[key] for key in scenario_keys) for e in report["episodes"]
        ]
        if other != scenarios:
            raise ValueError(
                f"{label}: scenario list differs from current Rust; "
                "refusing an unpaired comparison"
            )
    return scenarios


def summarize(reports: dict[str, dict]) -> dict:
    scenarios = validate(reports)
    rng = np.random.default_rng(20260713)
    summary: dict[str, dict] = {}
    for label, report in reports.items():
        episodes = report["episodes"]
        wins = [float(e["won"]) for e in episodes]
        places = [float(e["place"]) for e in episodes]
        scores = [float(e["score"]) for e in episodes]
        summary[label] = {
            "episodes": len(episodes),
            "wins": int(sum(wins)),
            "win_rate": float(np.mean(wins)),
            "win_rate_ci95_wilson": wilson_ci(int(sum(wins)), len(wins)),
            "mean_place": float(np.mean(places)),
            "mean_place_ci95_bootstrap": bootstrap_ci(places, rng),
            "mean_score": float(np.mean(scores)),
            "mean_score_ci95_bootstrap": bootstrap_ci(scores, rng),
        }
    paired = {}
    base = {e["index"]: e for e in reports["current_rust"]["episodes"]}
    for label in ("ppo_v5", "ppo_v7"):
        old = {e["index"]: e for e in reports[label]["episodes"]}
        score_delta = [base[i]["score"] - old[i]["score"] for i, *_ in scenarios]
        win_delta = [float(base[i]["won"]) - float(old[i]["won"]) for i, *_ in scenarios]
        paired[f"current_rust-minus-{label}"] = {
            "mean_win_delta": float(np.mean(win_delta)),
            "mean_win_delta_ci95_bootstrap": bootstrap_ci(win_delta, rng),
            "mean_score_delta": float(np.mean(score_delta)),
            "mean_score_delta_ci95_bootstrap": bootstrap_ci(score_delta, rng),
        }
    return {
        "format": 1,
        "mode": "indirect-scripted-bot",
        "warning": "Not head-to-head: each policy separately faces identical scripted bots.",
        "scenario_count": len(scenarios),
        "results": summary,
        "paired_differences": paired,
    }


def fetch_old(label: str, destination: Path) -> Path:
    from huggingface_hub import hf_hub_download

    source = hf_hub_download("djmango/openfront-rl", f"{label}/policy.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def historical_tree(ref: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    tar_path = destination.parent / f"{destination.name}.tar"
    tar_path.write_bytes(archive)
    destination.mkdir()
    with tarfile.open(tar_path) as bundle:
        # The archive is produced locally by `git archive`, not supplied by
        # an external party. Avoid `filter=` for Python builds whose tarfile
        # module predates that argument despite satisfying project metadata.
        bundle.extractall(destination)
    tar_path.unlink()
    for name in ("openfront", "bridge"):
        target = destination / name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        source = current_engine_path(name)
        if name == "bridge":
            # Copy so TypeScript's relative ../openfront imports resolve via
            # the temporary tree instead of the symlink's physical parent.
            shutil.copytree(source, target)
        else:
            target.symlink_to(source, target_is_directory=True)


def run_python(
    label: str,
    checkpoint: Path,
    ae: Path,
    coarse_ae: Path | None,
    stage: int,
    episodes: int,
    max_ticks: int,
    output: Path,
    temp: Path,
) -> None:
    tree = temp / f"source-{label}"
    historical_tree(SOURCE_REFS[label], tree)
    worker = tree / "scripts" / "python_policy_benchmark.py"
    shutil.copy2(REPO / "scripts" / "python_policy_benchmark.py", worker)
    command = [
        sys.executable, str(worker), "--checkpoint", str(checkpoint),
        "--ae", str(ae), "--stage", str(stage), "--episodes", str(episodes),
        "--max-ticks", str(max_ticks), "--out", str(output),
    ]
    if coarse_ae and label == "ppo_v7":
        command += ["--coarse-ae", str(coarse_ae)]
    env = {**os.environ, "PYTHONPATH": str(tree), "OPENFRONT_ENV": ""}
    subprocess.run(command, cwd=tree, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-checkpoint", required=True)
    ap.add_argument("--rust-bin", default="rust/target/release/oftrain")
    ap.add_argument("--rust-ae", required=True, help="exported encoder safetensors")
    ap.add_argument("--rust-coarse-ae")
    ap.add_argument("--python-ae", required=True, help="PyTorch AE checkpoint")
    ap.add_argument("--python-coarse-ae")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--max-ticks", type=int, default=15000)
    ap.add_argument("--out", default="eval_out/paired-policy-benchmark.json")
    ap.add_argument("--checkpoint-dir", default="runs/rl/frozen-eval")
    args = ap.parse_args()

    rust_checkpoint = Path(args.rust_checkpoint).resolve()
    if rust_checkpoint.suffix != ".safetensors":
        raise SystemExit(
            "--rust-checkpoint must be a current .safetensors checkpoint; "
            "ppo_v5/ppo_v7 policy.pt files are frozen legacy fixtures only"
        )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    old = {
        label: fetch_old(label, checkpoint_dir / label / "policy.pt")
        for label in ("ppo_v5", "ppo_v7")
    }
    with tempfile.TemporaryDirectory(prefix="openfront-paired-") as td:
        temp = Path(td)
        raw = {
            "current_rust": temp / "current_rust.json",
            "ppo_v5": temp / "ppo_v5.json",
            "ppo_v7": temp / "ppo_v7.json",
        }
        rust_command = [
            str(Path(args.rust_bin).resolve()),
            "--benchmark-out", str(raw["current_rust"]),
            "--resume", str(rust_checkpoint),
            "--ckpt", str(Path(args.rust_ae).resolve()),
            "--stage", str(args.stage), "--eval-episodes", str(args.episodes),
            "--max-episode-ticks", str(args.max_ticks), "--engine", "node",
            "--device", "cuda" if __import__("torch").cuda.is_available() else "cpu",
        ]
        if args.rust_coarse_ae:
            rust_command += ["--coarse-ckpt", str(Path(args.rust_coarse_ae).resolve())]
        subprocess.run(rust_command, cwd=REPO, check=True)
        for label in ("ppo_v5", "ppo_v7"):
            run_python(
                label, old[label], Path(args.python_ae).resolve(),
                Path(args.python_coarse_ae).resolve() if args.python_coarse_ae else None,
                args.stage, args.episodes, args.max_ticks, raw[label], temp,
            )
        reports = {label: json.loads(path.read_text()) for label, path in raw.items()}
        for label, report in reports.items():
            report["checkpoint_sha256"] = sha256(
                rust_checkpoint if label == "current_rust" else old[label]
            )
        final = summarize(reports)
        final["inputs"] = {
            label: {
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "schema": report["schema"],
            }
            for label, report in reports.items()
        }
        out.write_text(json.dumps(final, indent=2) + "\n")
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
