"""Curriculum stages and the strength-based reward.

Reward design (v2):
- Dense signal is a composite STRENGTH index, not raw land: territory
  alone mis-scores legit strategies (tiny-island economies that stack
  gold). Strength blends land share (of the whole map), military share,
  economic share, and (v5.1) structure-value share (of everything held by
  living players). Each decision
  step earns w_str * strength * timeweight, where timeweight ramps
  0.5 -> 1.0 over the first 8000 ticks: being strong late is worth double
  being strong early, and unlike pure deltas (v1) this does not net to
  zero when the agent dies.
- Small delta term retained (w_delta * d_strength) for immediate credit.
- Terminal placement: w_place * placement^-0.7 (power law: 1st ~ w_place,
  2nd ~ 0.62x, 10th ~ 0.2x, 100th ~ 0.04x), plus w_win extra for an
  outright engine win. Placement = 1 + players still alive if we died,
  else rank among the living by the same strength index.
- Flat small death penalty w_death.
- Wasted-intent penalty w_waste per intent the engine silently discarded
  (doomed boat, invalid build site, expand with no neutral border). These
  are otherwise reward-identical to noop but with occasional upside, so
  the policy farms them as free lottery tickets (v3/v4 replays: 40-80% of
  decisions were boats/builds that did nothing).

Curriculum (v2):
- Each stage is a POOL of maps plus a bot count and difficulty; the map is
  sampled per episode so the agent never overfits a single layout.
- Anti-forgetting rehearsal: REHEARSAL_P of episodes replay a map pool from
  an earlier (already-cleared) stage but at the CURRENT stage's bot count
  and difficulty - old maps come back harder, so mastery has to hold up.
  Rehearsal episodes still train the policy but do not count toward
  advancement stats.
- Advancement is win-gated: the agent must WIN (engine win, not just
  survive) - rolling win rate over the last WINDOW on-stage episodes must
  exceed the stage's win_at before moving on. v5.1: gates are per-stage,
  demanding dominance (0.9) where wins are cheap and tapering as they get
  genuinely hard.
"""

import math
from dataclasses import dataclass

import numpy as np

W_STR = 0.02
# v5: asymmetric strength delta (loss-aversion shaping). v4 games showed
# hyper-expansion followed by collapse; symmetric W_DELTA priced a lost
# tile the same as a gained one, so defense was never worth learning.
W_DELTA_GAIN = 5.0
W_DELTA_LOSS = 6.5
W_PLACE = 15.0
W_WIN = 30.0
W_DEATH = 1.0
W_WASTE = 0.01  # per silently-discarded intent; makes noop dominate them
# v5: 0.7 -> 1.5. ppo_v4 converged to a "safe second" attractor (score 0.82,
# win rate 0, entropy collapsed): at 0.7 the 1st-vs-2nd terminal gap was
# only 5.8 (15.0 vs 9.2) before the win bonus. At 1.5 it's 9.7 (15.0 vs
# 5.3), and 3rd drops to 2.9 - placement alone now pushes for the top.
PLACE_POW = 1.5

# Strength blend: land still matters most (the win condition is territorial)
# but military and economy make island/eco strategies score honestly.
# v5.1 adds structures: without them, buying a building read as a strength
# LOSS (gold share drops, the building counts nowhere), so the loss-averse
# delta punished construction. Now structure value is its own component and
# completing a building pays an immediate delta.
K_LAND = 0.40
K_MIL = 0.20
K_ECO = 0.25
K_BUILD = 0.15

# Structure values ~ in-game gold cost ceilings / 1M (Config.ts unitInfo);
# level multiplies (upgraded cities/silos are worth their reinvestment).
# Under-construction units count once finished.
STRUCT_VALUE = {
    "City": 1.0,
    "Port": 1.0,
    "Factory": 1.0,
    "Missile Silo": 1.0,
    "Defense Post": 0.25,
    "SAM Launcher": 3.0,
}

WINDOW = 40
REHEARSAL_P = 0.25  # fraction of episodes replaying earlier maps, harder


@dataclass(frozen=True)
class Stage:
    maps: tuple[str, ...]
    bots: int
    difficulty: str
    nations: int | str = "default"  # exact opponent count, or the map's default
    decision_ticks: int = 10  # engine ticks per policy decision
    # v5.1: per-stage advancement gate. A flat 0.5 let the BC-warm-started
    # v5 clear stages 0-3 at 21/40 wins and arrive at stage 4 without ever
    # mastering the close-out kill; it then saturated at 2nd place (score
    # 0.84, wins 0.17). Early stages now demand dominance where wins are
    # cheap; later stages taper back as winning gets genuinely hard.
    win_at: float = 0.5


# Map pool (all in the AE training set) with featurized grid sizes (H//8 x W//8):
#   Onion 64x64, Pangaea 125x125, Caucasus 125x156, BlackSea 137x187,
#   BetweenTwoSeas 132x223, World 125x250, Asia 150x250.
ALL_MAPS = (
    "Onion", "Pangaea", "Caucasus", "BlackSea", "BetweenTwoSeas", "World", "Asia",
)

# Early stages pin the opponent count (1v1, then 1v3) so wins are actually
# reachable and the win signal is dense; later stages return to the map's
# full nation roster plus tribe bots.
STAGES = [
    Stage(("Onion",), 0, "Easy", nations=1, decision_ticks=15, win_at=0.9),
    Stage(("Onion",), 0, "Easy", nations=3, decision_ticks=15, win_at=0.8),
    Stage(("Onion", "Pangaea"), 5, "Easy", nations=3, decision_ticks=15, win_at=0.75),
    Stage(("Pangaea", "Caucasus"), 10, "Easy", nations=6, decision_ticks=15, win_at=0.65),
    Stage(("Pangaea", "Caucasus", "BlackSea"), 30, "Easy", win_at=0.55),
    Stage(("BlackSea", "BetweenTwoSeas", "Caucasus"), 30, "Medium"),
    Stage(("World", "Asia", "BlackSea"), 50, "Medium"),
    Stage(("World", "Asia", "BetweenTwoSeas", "Caucasus"), 80, "Medium", win_at=0.45),
    Stage(ALL_MAPS, 80, "Hard", win_at=0.4),
    Stage(ALL_MAPS, 120, "Hard", win_at=0.35),
    Stage(ALL_MAPS, 150, "Impossible", win_at=0.3),  # terminal; gate moot
]

# Largest featurized grid across curriculum maps (Asia 2000x1200 -> 150x250
# at the v4 1/8 latent resolution).
GH_MAX = 150
GW_MAX = 250


def sample_episode(
    stage: int, rng: np.random.Generator
) -> tuple[str, int, str, int | str, bool]:
    """Pick (map, bots, difficulty, nations, is_rehearsal) for one episode.

    Rehearsal draws a map from a random earlier stage's pool but keeps the
    current stage's bots/difficulty/nations: old maps with harder opposition.
    """
    cur = STAGES[stage]
    if stage > 0 and rng.random() < REHEARSAL_P:
        past = STAGES[int(rng.integers(stage))]
        return str(rng.choice(past.maps)), cur.bots, cur.difficulty, cur.nations, True
    return str(rng.choice(cur.maps)), cur.bots, cur.difficulty, cur.nations, False


def timeweight(tick: int) -> float:
    return 0.5 + 0.5 * min(1.0, tick / 8000.0)


def strengths(entities: dict, land_total: int) -> dict[int, float]:
    """Composite strength per living player: blended land / military /
    economic / structural position. Land share is absolute (fraction of the
    map); troops, gold, and structure value are shares of what living
    players hold."""
    alive = [p for p in entities["players"] if p["alive"]]
    tot_troops = sum(p["troops"] for p in alive) + 1e-9
    tot_gold = sum(float(p["gold"]) for p in alive) + 1e-9
    sv: dict[int, float] = {}
    for u in entities.get("units", ()):
        v = STRUCT_VALUE.get(u["type"])
        if v is not None and not u["constructing"]:
            sv[u["owner"]] = sv.get(u["owner"], 0.0) + v * max(1, u.get("level", 1))
    tot_sv = sum(sv.get(p["id"], 0.0) for p in alive) + 1e-9
    return {
        p["id"]: (
            K_LAND * (p["tiles"] / land_total)
            + K_MIL * (p["troops"] / tot_troops)
            + K_ECO * (float(p["gold"]) / tot_gold)
            + K_BUILD * (sv.get(p["id"], 0.0) / tot_sv)
        )
        for p in alive
    }


def placement(entities: dict, me: int, agent_alive: bool, land_total: int) -> tuple[int, int]:
    """Returns (place, n_players). Dead: behind everyone still alive.
    Alive: ranked among the living by composite strength. The engine may
    drop a dead agent from the player list entirely, so count it back in
    and clamp so place never exceeds n."""
    ids = {p["id"] for p in entities["players"]}
    n = len(ids) + (0 if me in ids else 1)
    s = strengths(entities, land_total)
    if not agent_alive or me not in s:
        others_alive = sum(1 for pid in s if pid != me)
        return min(1 + others_alive, n), n
    mine = s[me]
    better = sum(1 for pid, v in s.items() if pid != me and v > mine)
    return 1 + better, n


def placement_score(place: int, n: int) -> float:
    return (n - place) / max(1, n - 1)


def terminal_reward(place: int, won: bool) -> float:
    r = W_PLACE * place ** -PLACE_POW
    if won:
        r += W_WIN
    return r
