"""Reward landscape graphs for the devlog and tuning.

Renders the MDP the PPO agent actually sees: dense strength shaping,
asymmetric deltas, terminal placement, and a few archetypal episode
trajectories. Constants are imported from rl.curriculum so the plots stay
in sync with training code.

Usage:
  uv run python scripts/make_reward_graphs.py
  uv run python scripts/make_reward_graphs.py --out docs/graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl.curriculum import (
    K_BUILD,
    K_ECO,
    K_LAND,
    K_MIL,
    PLACE_POW,
    W_DEATH,
    W_DELTA_GAIN,
    W_DELTA_LOSS,
    W_PLACE,
    W_STR,
    W_WASTE,
    W_WIN,
    terminal_reward,
    timeweight,
)

# v4 constants (pre-v5) for before/after panels.
PLACE_POW_V4 = 0.7
W_DELTA_SYMM = 5.0


def _term_v4(place: int, won: bool) -> float:
    r = W_PLACE * place ** -PLACE_POW_V4
    if won:
        r += W_WIN
    return r


def dense_step(
    strength: float,
    prev: float,
    tick: int,
    *,
    wasted: int = 0,
    symmetric_delta: bool = False,
) -> float:
    tw = timeweight(tick)
    delta = strength - prev
    if symmetric_delta:
        dcoef = W_DELTA_SYMM
    else:
        dcoef = W_DELTA_GAIN if delta >= 0 else W_DELTA_LOSS
    return W_STR * strength * tw + dcoef * delta - W_WASTE * wasted


def simulate(
    ticks: np.ndarray,
    strength: np.ndarray,
    *,
    place: int,
    won: bool,
    dead: bool = False,
    wasted: int = 0,
    symmetric_delta: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (tick, cumulative_reward), per-step dense, terminal total."""
    cum = np.zeros(len(ticks))
    total = 0.0
    prev = float(strength[0])
    for i, (t, s) in enumerate(zip(ticks, strength)):
        r = dense_step(float(s), prev, int(t), wasted=wasted, symmetric_delta=symmetric_delta)
        total += r
        cum[i] = total
        prev = float(s)
    if dead:
        total -= W_DEATH
    term = _term_v4(place, won) if symmetric_delta else terminal_reward(place, won)
    total += term
    cum[-1] = total
    return cum, term, total


def fig_terminal(out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ranks = np.arange(1, 11)
    for n, ls in [(4, "-"), (8, "--"), (16, ":")]:
        ax1.plot(
            ranks,
            [W_PLACE * p ** -PLACE_POW_V4 for p in ranks],
            ls=ls,
            color="tab:orange",
            lw=1.6,
            label=f"v4 pow={PLACE_POW_V4}, n={n}" if n == 4 else None,
        )
        ax1.plot(
            ranks,
            [W_PLACE * p ** -PLACE_POW for p in ranks],
            ls=ls,
            color="tab:blue",
            lw=1.6,
            label=f"v5 pow={PLACE_POW}, n={n}" if n == 4 else None,
        )

    ax1.axhline(W_WIN, color="tab:green", ls=":", lw=1.2)
    ax1.text(1.05, W_WIN + 1.2, f"+ win bonus W_WIN={W_WIN}", color="tab:green", fontsize=8.5)
    ax1.set_xlabel("placement rank (1 = best)")
    ax1.set_ylabel("terminal reward (no win bonus)")
    ax1.set_title("Terminal placement curve (v4 vs v5)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xticks(ranks)

    # Highlight the safe-second gap that motivated v5.
    labels = ["1st", "2nd", "3rd", "5th", "10th"]
    places = [1, 2, 3, 5, 10]
    v4 = [_term_v4(p, False) for p in places]
    v5 = [terminal_reward(p, False) for p in places]
    v4w = [_term_v4(p, True) for p in places]
    v5w = [terminal_reward(p, True) for p in places]
    x = np.arange(len(labels))
    w = 0.2
    ax2.bar(x - 1.5 * w, v4, w, label="v4 terminal", color="tab:orange")
    ax2.bar(x - 0.5 * w, v5, w, label="v5 terminal", color="tab:blue")
    ax2.bar(x + 0.5 * w, v4w, w, label="v4 + win", color="tab:orange", alpha=0.45)
    ax2.bar(x + 1.5 * w, v5w, w, label="v5 + win", color="tab:blue", alpha=0.45)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("reward")
    ax2.set_title("Selected ranks: terminal only vs + win")
    ax2.legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    fig.savefig(out / "reward_terminal.png", dpi=140)
    plt.close(fig)


def fig_dense(out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Timeweight + dense strength signal at zero delta.
    ticks = np.linspace(0, 15000, 300)
    ax = axes[0, 0]
    ax.plot(ticks, [timeweight(int(t)) for t in ticks], color="tab:purple", lw=2)
    ax.set_xlabel("engine tick")
    ax.set_ylabel("timeweight")
    ax.set_title("timeweight(t): 0.5 early → 1.0 by tick 8000")
    ax2 = ax.twinx()
    for s, c in [(0.1, "tab:blue"), (0.25, "tab:green"), (0.4, "tab:red")]:
        ax2.plot(
            ticks,
            [W_STR * s * timeweight(int(t)) for t in ticks],
            color=c,
            lw=1.2,
            label=f"strength={s}",
        )
    ax2.set_ylabel("dense reward / step (delta=0)")
    ax2.legend(loc="center right", fontsize=8)

    # Delta asymmetry at fixed strength.
    ax = axes[0, 1]
    deltas = np.linspace(-0.08, 0.08, 161)
    ax.plot(deltas, W_DELTA_SYMM * deltas, color="tab:orange", lw=2, label="v4 symmetric")
    gain = np.where(deltas >= 0, W_DELTA_GAIN * deltas, W_DELTA_LOSS * deltas)
    ax.plot(deltas, gain, color="tab:blue", lw=2, label="v5 loss-averse")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("strength delta per decision")
    ax.set_ylabel("delta term reward")
    ax.set_title(f"Per-step delta (W_GAIN={W_DELTA_GAIN}, W_LOSS={W_DELTA_LOSS})")
    ax.legend(fontsize=8)

    # Heatmap: strength x delta at late game.
    ax = axes[1, 0]
    s = np.linspace(0.02, 0.45, 44)
    d = np.linspace(-0.06, 0.06, 49)
    ss, dd = np.meshgrid(s, d)
    tw = timeweight(8000)
    rr = W_STR * ss * tw + np.where(dd >= 0, W_DELTA_GAIN, W_DELTA_LOSS) * dd
    im = ax.imshow(
        rr,
        origin="lower",
        aspect="auto",
        extent=[s[0], s[-1], d[0], d[-1]],
        cmap="RdYlGn",
    )
    ax.set_xlabel("strength")
    ax.set_ylabel("delta")
    ax.set_title("Dense reward heatmap @ tick 8000 (v5)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Wasted-intent tax.
    ax = axes[1, 1]
    wasted = np.arange(0, 6)
    base = dense_step(0.25, 0.25, 8000)
    ax.bar(wasted, [base - W_WASTE * w for w in wasted], color="tab:gray")
    ax.set_xlabel("wasted intents this step")
    ax.set_ylabel("dense reward @ strength=0.25, delta=0")
    ax.set_title(f"W_WASTE={W_WASTE} per silently-discarded intent")
    for w in wasted:
        ax.text(w, base - W_WASTE * w + 0.002, f"{base - W_WASTE * w:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "reward_dense.png", dpi=140)
    plt.close(fig)


def _strength_curve(kind: str, ticks: np.ndarray) -> np.ndarray:
    t = ticks.astype(float)
    if kind == "winner":
        return np.clip(0.04 + 0.31 * (1 - np.exp(-t / 5000)), 0, 0.38)
    if kind == "safe_second":
        rise = 0.04 + 0.26 * (1 - np.exp(-t / 4000))
        return np.where(t < 7000, rise, 0.27 + 0.01 * np.sin(t / 800))
    if kind == "expand_collapse":
        peak = 0.05 + 0.58 * np.exp(-((t - 7500) ** 2) / (2 * 2500**2))
        decay = np.where(t > 8500, peak * np.exp(-(t - 8500) / 1800), peak)
        return np.clip(decay, 0.02, None)
    if kind == "waste_spam":
        return np.full_like(t, 0.22)
    raise ValueError(kind)


def fig_scenarios(out: Path) -> None:
    ticks = np.arange(0, 15001, 150)  # decision every 150 ticks ~ stage 0-2
    scenarios = [
        ("Winner (1st + win)", "winner", 1, True, False, 0),
        ("Safe 2nd (timeout)", "safe_second", 2, False, False, 0),
        ("Expand → collapse (3rd)", "expand_collapse", 3, False, True, 0),
        ("Waste spam (2nd)", "waste_spam", 2, False, False, 4),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for label, kind, place, won, dead, wasted in scenarios:
        s = _strength_curve(kind, ticks)
        cum_v4, _, tot_v4 = simulate(
            ticks, s, place=place, won=won, dead=dead, wasted=wasted, symmetric_delta=True
        )
        cum_v5, _, tot_v5 = simulate(
            ticks, s, place=place, won=won, dead=dead, wasted=wasted, symmetric_delta=False
        )
        ax1.plot(ticks / 1000, s, lw=1.8, label=label)
        ax2.plot(ticks / 1000, cum_v4, ls="--", lw=1.2, alpha=0.7)
        ax2.plot(ticks / 1000, cum_v5, lw=2.0, label=f"{label}: v4={tot_v4:.0f} v5={tot_v5:.0f}")

    ax1.set_ylabel("composite strength")
    ax1.set_title("Archetypal episode strength traces")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("engine tick (thousands)")
    ax2.set_ylabel("cumulative return")
    ax2.set_title("Cumulative return: dashed=v4 MDP, solid=v5 (labels show totals)")
    ax2.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(out / "reward_scenarios.png", dpi=140)
    plt.close(fig)


def fig_strength_blend(out: Path) -> None:
    """How land / military / economy / structure shares blend into strength."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # (land, mil, eco, build) shares per archetype.
    archetypes = {
        "Territory hog\n(60% land)": (0.60, 0.20, 0.20, 0.10),
        "Balanced\n(even shares)": (0.33, 0.33, 0.34, 0.33),
        "Eco island\n(10% land, rich)": (0.10, 0.15, 0.75, 0.50),
        "Military rush\n(25% land, stacked troops)": (0.25, 0.65, 0.10, 0.05),
    }
    names = list(archetypes)
    vals = [
        K_LAND * l + K_MIL * m + K_ECO * e + K_BUILD * b
        for l, m, e, b in archetypes.values()
    ]
    parts = np.array([
        [K_LAND * l, K_MIL * m, K_ECO * e, K_BUILD * b]
        for l, m, e, b in archetypes.values()
    ])
    x = np.arange(len(names))
    ax.bar(x, parts[:, 0], label=f"land (K={K_LAND})", color="tab:green")
    ax.bar(x, parts[:, 1], bottom=parts[:, 0], label=f"mil (K={K_MIL})", color="tab:red")
    ax.bar(
        x,
        parts[:, 2],
        bottom=parts[:, 0] + parts[:, 1],
        label=f"eco (K={K_ECO})",
        color="goldenrod",
    )
    ax.bar(
        x,
        parts[:, 3],
        bottom=parts[:, 0] + parts[:, 1] + parts[:, 2],
        label=f"structures (K={K_BUILD}, v5.1)",
        color="tab:blue",
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylabel("composite strength (normalized shares)")
    ax.set_title("Strength index by play style (shares of map / alive totals)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reward_strength.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/graphs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_terminal(out)
    fig_dense(out)
    fig_scenarios(out)
    fig_strength_blend(out)
    print(f"wrote reward graphs to {out}/")


if __name__ == "__main__":
    main()
