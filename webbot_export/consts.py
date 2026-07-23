"""Observation / action-space constants shared with oftrain + webbot.

Kept in sync with rust/ofcore feat layout.
"""

MAX_SLOTS = 128
REGION = 8
LATENT_C = 32

# Exact static + ego + defense-bonus + transient (must match oftrain C_GRID).
N_STATIC = 6
N_TRANSIENT = 53
N_DEFENSE_BONUS = 1
C_GRID = LATENT_C + N_STATIC + 3 + N_DEFENSE_BONUS + N_TRANSIENT  # 95
C_GRID_FINE = C_GRID + 1

LOCAL = 64
N_LOCAL = 5
P_FEAT = 21
N_SCALARS = 11

ACTIONS = [
    "noop",
    "attack",
    "expand",
    "boat",
    "build",
    "launch_nuke",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "embargo",
    "retreat",
    "spawn",
    "upgrade_structure",
    "move_warship",
    "cancel_boat",
    "delete_unit",
    "embargo_stop",
    "target_player",
    "alliance_extension",
]
N_ACTIONS = len(ACTIONS)

BUILD_TYPES = [
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
    "Warship",
]
NUKE_TYPES = [
    ("Atom Bomb", True),
    ("Atom Bomb", False),
    ("Hydrogen Bomb", True),
    ("Hydrogen Bomb", False),
    ("MIRV", None),
]

__all__ = [
    "ACTIONS",
    "BUILD_TYPES",
    "C_GRID",
    "C_GRID_FINE",
    "LATENT_C",
    "LOCAL",
    "MAX_SLOTS",
    "N_ACTIONS",
    "N_LOCAL",
    "N_SCALARS",
    "NUKE_TYPES",
    "P_FEAT",
    "REGION",
]
