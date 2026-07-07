"""Training-history comparison graphs for the devlog.

Pulls scalars out of whatever TB event files / jsonl logs survive from each
run generation and renders the three panels the project actually argues
about: curriculum progress, throughput engineering, and the BC data
pipeline. v3's TB events died with its pod (only devlog milestones and the
HF checkpoint survive), so it appears as annotated reference lines.

Usage:
  uv run python scripts/make_progress_graphs.py \
      --v4-events /tmp/tb/ppo_v4 --bc-v4-log /tmp/tb/bc_v4_log.jsonl
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def load_run(
    pattern: str, tags: list[str]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[float]]:
    """Merge scalars across a run's (restart-fragmented) event files.

    Returns (tag -> (wall_time_seconds, values) sorted by wall time,
    restart boundary wall times = first event of each file).
    """
    acc: dict[str, list[tuple[float, float]]] = {t: [] for t in tags}
    starts = []
    for f in sorted(glob.glob(pattern)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        have = ea.Tags()["scalars"]
        first = None
        for t in tags:
            if t in have:
                rows = [(s.wall_time, s.value) for s in ea.Scalars(t)]
                acc[t].extend(rows)
                if rows:
                    first = rows[0][0] if first is None else min(first, rows[0][0])
        if first is not None:
            starts.append(first)
    out = {}
    for t, rows in acc.items():
        if not rows:
            continue
        rows.sort()
        w, v = zip(*rows)
        out[t] = (np.array(w), np.array(v))
    return out, starts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-events", default="/tmp/tb/ppo_v4")
    ap.add_argument("--v2c-events", default="runs/rl/ppo_v2c_final")
    ap.add_argument("--bc-v4-log", default="/tmp/tb/bc_v4_log.jsonl")
    ap.add_argument("--bc-pilot-log", default="runs/bc_v0_pilot_log.jsonl")
    ap.add_argument("--out", default="docs/graphs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tags = [
        "curriculum/stage",
        "curriculum/rolling_win",
        "curriculum/rolling_score",
        "perf/game_ticks_per_s",
        "eval/win",
    ]
    v4, v4_starts = load_run(f"{args.v4_events}/events*", tags)
    v2c, _ = load_run(f"{args.v2c_events}/events*", tags)

    # ---- fig 1: curriculum progress across generations ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for run, color, label in [(v2c, "tab:gray", "ppo_v2c (warm start; final hours, retired)"),
                              (v4, "tab:blue", "ppo_v4 (from scratch, 1/8 res)")]:
        if "curriculum/stage" not in run:
            continue
        w, v = run["curriculum/stage"]
        h = (w - w[0]) / 3600
        ax1.step(h, v, where="post", color=color, label=label, lw=1.8)
        if "curriculum/rolling_win" in run:
            w2, v2 = run["curriculum/rolling_win"]
            ax2.plot((w2 - w[0]) / 3600, v2, color=color, lw=1.2, alpha=0.85)
    # v3: curves died with its pod; devlog milestones only.
    ax1.axhline(4, color="tab:green", ls="--", lw=1.2, alpha=0.7)
    ax1.text(0.1, 4.08, "ppo_v3 final: stage 4, roll-win 0.6-0.7 after ~1 day"
             " (TB curves lost with pod; ckpt on HF)",
             color="tab:green", fontsize=8.5)
    ax2.axhspan(0.6, 0.7, color="tab:green", alpha=0.12)
    ax1.set_ylabel("curriculum stage")
    ax1.set_title("Curriculum progress by run generation (hours since run start)")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.set_ylabel("rolling win rate (current stage)")
    ax2.set_xlabel("hours since run start")
    fig.tight_layout()
    fig.savefig(out / "curriculum_progress.png", dpi=140)

    # ---- fig 2: v4 throughput engineering timeline ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w, v = v4["perf/game_ticks_per_s"]
    t0 = w[0]
    # Plot each restart segment separately so smoothing doesn't bridge gaps.
    bounds = sorted(v4_starts) + [w[-1] + 1]
    for i in range(len(bounds) - 1):
        m = (w >= bounds[i]) & (w < bounds[i + 1])
        if not m.any():
            continue
        h, vv = (w[m] - t0) / 3600, v[m]
        ax.plot(h, vv, color="tab:blue", lw=1.0, alpha=0.35)
        k = 5
        if len(vv) > k:
            ax.plot(h[k - 1:], np.convolve(vv, np.ones(k) / k, mode="valid"),
                    color="tab:blue", lw=2.0)
    labels = {
        1: "OOM restart\n(minibatch 128->64)",
        3: "fp16 + pinned +\nprefetch (8a0da05)",
        4: "v4.1 async\noverlap",
    }
    for i, s in enumerate(sorted(v4_starts)):
        if i == 0:
            continue
        hh = (s - t0) / 3600
        ax.axvline(hh, color="tab:orange", ls="--", lw=1.0, alpha=0.7)
        if i in labels:
            ax.text(hh + 0.03, ax.get_ylim()[1] * 0.86, labels[i],
                    fontsize=8, color="tab:orange")
    ax.set_xlabel("hours (ppo_v4 wall clock)")
    ax.set_ylabel("game-ticks / s")
    ax.set_title("ppo_v4 throughput (per-update window; decays within a window as maps grow)")
    if len(v2c.get("perf/game_ticks_per_s", ((), ()))[0]):
        med = float(np.median(v2c["perf/game_ticks_per_s"][1]))
        ax.axhline(med, color="tab:gray", ls=":", lw=1.2)
        ax.text(0.05, med * 1.05, f"v2c-era median ({med:.0f})", color="tab:gray",
                fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=140)

    # ---- fig 3: BC pipeline, starved pilot vs cache-fed v4 ----
    def load_jsonl(p: str) -> dict[str, np.ndarray]:
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        rows = [r for r in rows if "loss" in r]
        # Restarts re-log overlapping step ranges; keep the last occurrence.
        by_step = {r["step"]: r for r in rows}
        rows = [by_step[s] for s in sorted(by_step)]
        keys = {k for r in rows for k in r
                if all(isinstance(r2.get(k, 0.0), (int, float)) for r2 in rows)}
        return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
                for k in keys}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for p, color, label in [(args.bc_pilot_log, "tab:gray", "bc_v0 pilot (34 ex/s, 58 games)"),
                            (args.bc_v4_log, "tab:red", "bc_v4 (cache-bc, 291 games)")]:
        if not Path(p).exists():
            continue
        d = load_jsonl(p)
        k = 9
        loss = d["loss"]
        sm = np.convolve(loss, np.ones(k) / k, mode="valid") if len(loss) > k else loss
        ax1.plot(d["step"], loss, color=color, alpha=0.25, lw=0.8)
        ax1.plot(d["step"][k - 1:] if len(loss) > k else d["step"], sm,
                 color=color, lw=1.8, label=label)
        if "action_no_noop" in d:
            acc = d["action_no_noop"]
            sma = np.convolve(acc, np.ones(k) / k, mode="valid") if len(acc) > k else acc
            ax2.plot(d["step"][k - 1:] if len(acc) > k else d["step"], sma,
                     color=color, lw=1.8)
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("BC loss")
    ax1.set_title("Behavior cloning loss")
    ax1.legend(fontsize=8.5)
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("acted-step action accuracy")
    ax2.set_title("Action accuracy (non-noop steps)")
    fig.tight_layout()
    fig.savefig(out / "bc_pipeline.png", dpi=140)

    print(f"wrote {out}/curriculum_progress.png, throughput.png, bc_pipeline.png")


if __name__ == "__main__":
    main()
