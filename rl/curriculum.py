"""Curriculum stages and the strength-based reward.

Reward design (v2):
- Dense signal is a composite STRENGTH index, not raw land: territory
  alone mis-scores legit strategies (tiny-island economies that stack
  gold). Strength blends land share (of the whole map), military share,
  and economic share (of everything held by living players). Each decision
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

Stage advancement: rolling mean placement score over the last WINDOW
episodes; advance when it clears ADVANCE_AT. Score = (N - place) / (N - 1),
1.0 = first of N, 0 = last.
"""

import math
from dataclasses import dataclass

W_STR = 0.02
W_DELTA = 5.0
W_PLACE = 15.0
W_WIN = 15.0
W_DEATH = 1.0
PLACE_POW = 0.7

# Strength blend: land still matters most (the win condition is territorial)
# but military and economy make island/eco strategies score honestly.
K_LAND = 0.45
K_MIL = 0.25
K_ECO = 0.30

WINDOW = 40
ADVANCE_AT = 0.80


@dataclass(frozen=True)
class Stage:
    map_name: str
    bots: int
    difficulty: str


STAGES = [
    Stage("Onion", 10, "Easy"),
    Stage("Onion", 30, "Easy"),
    Stage("Onion", 60, "Medium"),
    Stage("Onion", 100, "Medium"),
    Stage("Pangaea", 60, "Medium"),
    Stage("Pangaea", 100, "Medium"),
    Stage("Pangaea", 100, "Hard"),
]

# Largest featurized grid across curriculum maps (Pangaea 1000x1000 -> 62).
GH_MAX = 62
GW_MAX = 62


def timeweight(tick: int) -> float:
    return 0.5 + 0.5 * min(1.0, tick / 8000.0)


def strengths(entities: dict, land_total: int) -> dict[int, float]:
    """Composite strength per living player: blended land / military /
    economic position. Land share is absolute (fraction of the map);
    troops and gold are shares of what living players hold."""
    alive = [p for p in entities["players"] if p["alive"]]
    tot_troops = sum(p["troops"] for p in alive) + 1e-9
    tot_gold = sum(float(p["gold"]) for p in alive) + 1e-9
    return {
        p["id"]: (
            K_LAND * (p["tiles"] / land_total)
            + K_MIL * (p["troops"] / tot_troops)
            + K_ECO * (float(p["gold"]) / tot_gold)
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
