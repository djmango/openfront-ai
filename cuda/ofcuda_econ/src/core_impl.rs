// Shared core of the per-tick ECONOMY / STATE-PLANE update (slice "econ").
//
// This file is the canonical source. It is included by the host reference
// (`src/lib.rs`, `mod core_impl`) and copied VERBATIM into cuda-oxide's device
// module (`src/main.rs`, `mod kernels` - see the BEGIN/END VERBATIM COPY
// markers there, and the `device_core_is_a_verbatim_copy` test in lib.rs that
// fails the moment the two copies drift apart). So the income arithmetic has
// exactly one implementation that can disagree with the engine - not one per
// side.
//
// 1:1 ports, with citations (TS is the authority; the Rust engine is the
// cross-check that already reached every-tick parity with TS):
//   * `PlayerExecution.tick` income block   - openfront/src/core/execution/PlayerExecution.ts:75-81
//                                             (Rust live path: rust/engine/src/execution/player.rs:82-88)
//   * `Config.maxTroops`                    - openfront/src/core/configuration/Config.ts:791-823
//                                             (rust/engine/src/core/config.rs:363-390)
//   * `Config.troopIncreaseRate`            - openfront/src/core/configuration/Config.ts:825-857
//                                             (rust/engine/src/core/config.rs:433-458)
//   * `Config.goldAdditionRate`             - openfront/src/core/configuration/Config.ts:859-868
//                                             (rust/engine/src/core/config.rs:477-481)
//   * `PlayerImpl.addTroops` / `removeTroops` - openfront/src/core/game/PlayerImpl.ts:1177-1189
//                                             (rust/engine/src/game.rs:1130-1155)
//   * `Util.toInt`                          - openfront/src/core/Util.ts (floor)
//                                             (rust/engine/src/util.rs:26-35)
//
// `tick_player_income` (rust/engine/src/game.rs:1269) is DEAD CODE: nothing
// calls it. The live per-tick income is `execution/player.rs::PlayerExecution::tick`
// (registered per player by the execution manager), which calls
// `Game::troop_increase_rate_raw_for` + `Config::gold_addition_rate` exactly as
// the TS `PlayerExecution.tick` does. `try_merge_land_attack` is likewise not on
// the income path. See README.md "Dead code" for the grep that establishes it.
//
// (This file is `#[no_std]`-clean on purpose: no allocation, no float methods
// beyond `powf`/`floor`/`min`.)

/// `PlayerType` discriminants, as they cross the host/device boundary.
pub const PT_HUMAN: u32 = 0;
pub const PT_BOT: u32 = 1;
pub const PT_NATION: u32 = 2;

/// `Difficulty` discriminants (`Config._gameConfig.difficulty`).
pub const DIFF_EASY: u32 = 0;
pub const DIFF_MEDIUM: u32 = 1;
pub const DIFF_HARD: u32 = 2;
pub const DIFF_IMPOSSIBLE: u32 = 3;

/// `Config.cityTroopIncrease()` (`Config.ts:392-394`; `config.rs:392-394`).
pub const CITY_TROOP_INCREASE: f64 = 250_000.0;

/// `Util.toInt` (`util.rs:26-35`): `floor`, saturating to the i32 range at the
/// infinities (Rust's `as i32` already saturates, so only the explicit inf
/// handling is needed for parity with the crate's Python probe).
#[inline]
pub fn to_int(num: f64) -> i32 {
    if num.is_infinite() {
        return if num.is_sign_positive() { i32::MAX } else { i32::MIN };
    }
    num.floor() as i32
}

/// `a * b + c`, with the multiply and the add kept as two separately rounded
/// IEEE operations. `black_box` is an identity to the language but opaque to the
/// optimiser, which is what stops `a*b+c` from being contracted into a single
/// `fma.rn.f64` on the device. Rust on x86-64 emits the two-step form anyway
/// (measured: this core reproduces the engine's own dump), so on the host this
/// only documents the intent.
#[inline]
pub fn survive_opt(p: f64) -> f64 {
    // A volatile read of the local is an identity that the device MIR lowering
    // cannot see through (it survives as a real local load/store), which is what
    // keeps the caller's `a*b + c` from being rewritten as one `fma.rn.f64`.
    unsafe { core::ptr::read_volatile(&p) }
}

#[inline]
pub fn mul_then_add(a: f64, b: f64, c: f64) -> f64 {
    survive_opt(a * b) + c
}

/// `Config.maxTroops(player)` (`Config.ts:791-823`; `config.rs:363-390`).
///
/// Order of operations is the engine's: `2*(tiles^0.6*1000 + 50000)` first, the
/// city term added, and only then the per-type scale.
/// `tiles ** 0.6`, kept as its own function so the parity harness can compare
/// the device's libdevice `pow` against the host's `powf` bit-for-bit.
#[inline]
pub fn pow_tiles(tiles_owned: f64) -> f64 {
    tiles_owned.powf(0.6)
}

/// `troops ** 0.73`, same purpose as [`pow_tiles`].
#[inline]
pub fn pow_troops(troops: f64) -> f64 {
    troops.powf(0.73)
}

#[inline]
pub fn max_troops(
    player_type: u32,
    tiles_owned: i32,
    city_level_sum: i64,
    difficulty: u32,
    infinite_troops: bool,
) -> f64 {
    // The engine's own order, with the two contractible multiplies forced to
    // stay separate. `mul_then_add` is an identity, but `black_box` is an
    // optimisation barrier: without it the PVT/ltoir lowering fuses
    // `pow*1000 + 50000` and `city*250000 + t` into `fma.rn.f64` (observed in
    // the emitted PTX), which changes `max_troops` on ~10% of the recorded
    // transitions while the host stays two-step. See README "FMA" for the
    // measurement.
    let mut max = 2.0 * mul_then_add(pow_tiles(tiles_owned as f64), 1000.0, 50_000.0)
        + survive_opt(city_level_sum as f64 * CITY_TROOP_INCREASE);
    if player_type == PT_BOT {
        max /= 3.0;
    } else if player_type == PT_HUMAN {
        if infinite_troops {
            return 1_000_000_000.0;
        }
    } else {
        max *= match difficulty {
            DIFF_EASY => 0.5,
            DIFF_MEDIUM => 0.75,
            DIFF_HARD => 1.0,
            DIFF_IMPOSSIBLE => 1.25,
            _ => 0.75,
        };
    }
    max
}

/// `Config.troopIncreaseRate(player)` un-rounded (`Config.ts:825-857`;
/// `config.rs:433-458`). Returns `min(troops + toAdd, max) - troops`, i.e. the
/// raw (possibly fractional, possibly negative) per-tick troop delta.
#[inline]
pub fn troop_increase_rate_raw(
    player_type: u32,
    troops: i32,
    tiles_owned: i32,
    city_level_sum: i64,
    difficulty: u32,
    infinite_troops: bool,
) -> f64 {
    let max = max_troops(player_type, tiles_owned, city_level_sum, difficulty, infinite_troops);
    let mut to_add = 10.0 + pow_troops(troops as f64) / 4.0;
    let ratio = 1.0 - troops as f64 / max;
    to_add *= ratio;
    if player_type == PT_BOT {
        to_add *= 0.5;
    }
    if player_type == PT_NATION {
        to_add *= match difficulty {
            DIFF_EASY => 0.9,
            DIFF_MEDIUM => 0.95,
            DIFF_HARD => 1.0,
            DIFF_IMPOSSIBLE => 1.05,
            _ => 0.95,
        };
    }
    (troops as f64 + to_add).min(max) - troops as f64
}

/// `PlayerImpl.addTroops(amount)` (`PlayerImpl.ts:1177-1182`;
/// `game.rs:1147-1155` + `remove_troops` at `game.rs:1130-1141`).
///
/// A non-negative amount adds `floor(amount)`. A negative amount is routed
/// through `removeTroops(-amount)`, which removes `min(troops, floor(-amount))`
/// - so the pull-back is the floor of the *magnitude*, not the floor of the
/// signed value. That asymmetry is load-bearing: it is the fix for the UK
/// soft-troop fork at tick ~1793 on `curr-b030-s0-pangaea`
/// (`execution/player.rs:130-146`).
#[inline]
pub fn add_troops(troops: i32, amount: f64) -> i32 {
    if amount < 0.0 {
        let to_remove = troops.min(to_int(-amount));
        troops - to_remove.max(0)
    } else {
        troops + to_int(amount)
    }
}

/// `Config.goldAdditionRate(player)` (`Config.ts:859-868`; `config.rs:477-481`).
/// `floor(baseRate * multiplier)`, base 50 for a Bot and 100 otherwise.
#[inline]
pub fn gold_addition_rate(player_type: u32, gold_multiplier: f64) -> i64 {
    let base_rate: f64 = if player_type == PT_BOT { 50.0 } else { 100.0 };
    (base_rate * gold_multiplier).floor() as i64
}

/// The whole per-tick economy step of one player, in one call.
///
/// Returns `(troops_after, gold_after, raw_income)` where `raw_income` is the
/// *pre-truncation* `troopIncreaseRate` float - the same value the RL obs emits
/// (`config.rs:431-432`) - so the GPU's raw float can be compared bit-for-bit
/// against the host's, not just the truncated integer.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn econ_step(
    player_type: u32,
    difficulty: u32,
    troops: i32,
    tiles_owned: i32,
    city_level_sum: i64,
    gold: i64,
    gold_multiplier: f64,
    infinite_troops: bool,
) -> (i32, i64, f64) {
    let raw = troop_increase_rate_raw(
        player_type,
        troops,
        tiles_owned,
        city_level_sum,
        difficulty,
        infinite_troops,
    );
    let troops_after = add_troops(troops, raw);
    let gold_after = gold + gold_addition_rate(player_type, gold_multiplier);
    (troops_after, gold_after, raw)
}