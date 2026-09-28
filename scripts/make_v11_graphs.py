#!/usr/bin/env python3
"""Progress figures for the final mainline run (ppo_v11).

Data source: docs/v11_milestones.csv, extracted from the published checkpoints on
Hugging Face (djmango/openfront-rl, ppo_v11/*.state.json). Each row is one saved
update with the curriculum stage and the trainer's own 40-episode rolling window
at that moment. Nothing here is smoothed or re-derived; the curves are the
values the trainer wrote into its checkpoints.

Run: python3 scripts/make_v11_graphs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "graphs"
CSV = ROOT / "docs" / "v11_milestones.csv"

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2b6cb0"
WARN = "#c05621"
GATE = "#e6f0f7"


def load():
    rows = []
    with CSV.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "update": int(r["update"]),
                    "stage": int(r["stage"]),
                    "steps": int(r["total_env_steps"]),
                    "wr": float(r["rolling40_win_rate"]) if r["rolling40_win_rate"] else None,
                    "conv": float(r["conversion"]) if r["conversion"] else None,
                    "death": float(r["death_rate"]) if r["death_rate"] else None,
                }
            )
    rows.sort(key=lambda r: r["update"])
    return rows


def style(ax, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)


def ladder(rows):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
    )

    # stage step curve
    xs, ys = [], []
    last = None
    for r in rows:
        if r["stage"] != last:
            xs.append(r["update"])
            ys.append(r["stage"])
            last = r["stage"]
    xs.append(rows[-1]["update"])
    ys.append(last)
    top.step(xs, ys, where="post", color=ACCENT, linewidth=1.8)
    top.fill_between(xs, ys, step="post", color=ACCENT, alpha=0.08)
    style(top, ylabel="curriculum stage")
    top.set_ylim(0, 30)

    top.annotate(
        "stage 24 at u1306",
        xy=(1306, 24),
        xytext=(1400, 28),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    top.annotate(
        "u1780 demote to 23,\nthen ~1200 updates at the same rung",
        xy=(2100, 23),
        xytext=(1150, 15.5),
        fontsize=9,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=0.9),
    )
    top.annotate(
        "final: u2561 / stage 26",
        xy=(2561, 26),
        xytext=(2050, 12),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    top.set_title(
        "ppo_v11: the ladder climb, and where it stopped",
        color=INK,
        fontsize=12,
        loc="left",
        pad=10,
    )

    # rolling win rate
    pts = [(r["update"], r["wr"]) for r in rows if r["wr"] is not None]
    bottom.plot([p[0] for p in pts], [p[1] for p in pts], color=INK, linewidth=1.4)
    bottom.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, color=INK, zorder=3)
    bottom.axhspan(0.6, 0.9, color=GATE, zorder=0)
    bottom.text(
        40,
        0.93,
        "ladder gate: 0.60 to 0.90 depending on the rung",
        fontsize=8.5,
        color=MUTED,
    )
    style(bottom, xlabel="policy update", ylabel="rolling win rate (last 40 episodes)")
    bottom.set_ylim(0, 1.05)
    bottom.set_xlim(0, 2620)

    fig.tight_layout()
    fig.savefig(OUT / "v11_ladder.png", dpi=150, facecolor="white")
    plt.close(fig)


def endgame(rows):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    conv = [(r["update"], r["conv"]) for r in rows if r["conv"] is not None]
    death = [(r["update"], r["death"]) for r in rows if r["death"] is not None]
    ax.plot([p[0] for p in conv], [p[1] for p in conv], color=ACCENT, linewidth=1.5, label="conversion")
    ax.plot([p[0] for p in death], [p[1] for p in death], color=WARN, linewidth=1.5, label="death rate")
    style(ax, xlabel="policy update", ylabel="share of the last 40 episodes")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 2620)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(
        "ppo_v11: the two numbers the ladder gate does not see",
        color=INK,
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax.annotate(
        "conversion stays near 0.8 while the win rate sinks:\nepisodes reach the endgame and then do not close",
        xy=(1600, 0.6),
        xytext=(90, 0.08),
        fontsize=9,
        color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8),
    )
    fig.tight_layout()
    fig.savefig(OUT / "v11_endgame.png", dpi=150, facecolor="white")
    plt.close(fig)


def stage0_reward():
    """Measured on the CUDA/OpenFront environment port, stage 0, 12,696 decisions.

    Source: openfront-train/RESULTS.md (reward_harness.py over the built engine).
    """
    labels = ["terminal", "closeout", "strength", "strength delta", "survival", "death", "tempo"]
    shares = [91.4, 8.0, 2.0, 1.2, 0.5, -2.7, -0.3]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    colors = [ACCENT if s >= 0 else WARN for s in shares]
    ax.bar(labels, shares, color=colors, width=0.62)
    for i, s in enumerate(shares):
        ax.text(i, s + (2 if s >= 0 else -6), f"{s:+.1f}%", ha="center", fontsize=9, color=INK)
    style(ax, ylabel="share of total return")
    ax.set_ylim(-12, 105)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(
        "What the reward actually was at stage 0 (12,696 measured decisions)",
        color=INK,
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax.text(
        2.15,
        62,
        "The terminal term is 91% of the return, and 0.19% of decisions are terminal.\n"
        "A uniform random policy won 6 of 6 stage-0 games; every deliberate policy lost.",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "stage0_reward_shares.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    ladder(rows)
    endgame(rows)
    stage0_reward()
    print("wrote", ", ".join(str(OUT / n) for n in
          ("v11_ladder.png", "v11_endgame.png", "stage0_reward_shares.png")))


if __name__ == "__main__":
    main()
