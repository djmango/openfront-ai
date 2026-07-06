"""Canonical unit-class taxonomy shared by cache, models, and evals.

STATIC_CLASSES are persistent structures the spatial AE compresses.
TRANSIENT_CLASSES (mobile/projectile) bypass the AE: they are few, exact,
and fully described by (owner, type, position, target) — the policy reads
them raw instead of through a lossy latent.
"""

UNIT_CLASSES = [
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
    "Warship",
    "Transport",
    "Trade Ship",
    "Atom Bomb",
    "Hydrogen Bomb",
    "MIRV",
]
UNIT_CLASS_INDEX = {name: i for i, name in enumerate(UNIT_CLASSES)}

STATIC_CLASSES = [
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
]
STATIC_INDICES = [UNIT_CLASS_INDEX[n] for n in STATIC_CLASSES]

TRANSIENT_CLASSES = [
    "Warship",
    "Transport",
    "Trade Ship",
    "Atom Bomb",
    "Hydrogen Bomb",
    "MIRV",
]
