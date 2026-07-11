//! Shared warship-veterancy math (TS `core/game/Veterancy.ts`).
//!
//! Integer-only, mirroring TS's own "no floats in src/core" discipline so the
//! engine and (if this project ever grows a renderer) a client would derive
//! identical effective max health.

/// Effective max health for a warship at a given veterancy level.
///
/// Each veterancy level adds `health_bonus_percent`% of base max health,
/// floored to an integer. Returns `base_max_health` unchanged at veterancy 0
/// (and therefore for any non-warship unit, which is always veterancy 0).
pub fn max_health_with_veterancy(base_max_health: i32, veterancy: i32, health_bonus_percent: i32) -> i32 {
    if veterancy <= 0 {
        return base_max_health;
    }
    base_max_health + (base_max_health * veterancy * health_bonus_percent) / 100
}
