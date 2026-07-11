//! Warship shell flight (`ShellExecution.ts`).

use super::Execution;
use crate::core::schemas::unit_type::{SHELL, TRANSPORT, WARSHIP};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;

fn air_path(game: &Game, seed: i32, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let mut random = PseudoRandom::new(seed);
    let mut path = vec![from];
    let mut current = from;
    while current != to {
        let x = game.x(current) as i64;
        let y = game.y(current) as i64;
        let dst_x = game.x(to) as i64;
        let dst_y = game.y(to) as i64;
        let mut next_x = x;
        let mut next_y = y;
        let ratio = 1 + (dst_y - y).abs() / ((dst_x - x).abs() + 1);
        if x == dst_x {
            next_y += if y < dst_y { 1 } else { -1 };
        } else if y == dst_y {
            next_x += if x < dst_x { 1 } else { -1 };
        } else if random.chance(ratio as i32) {
            next_x += if x < dst_x { 1 } else { -1 };
        } else {
            next_y += if y < dst_y { 1 } else { -1 };
        }
        current = game.ref_xy(next_x as u32, next_y as u32);
        path.push(current);
    }
    path
}

pub struct ShellExecution {
    spawn: TileRef,
    owner_small_id: u16,
    owner_unit_id: i32,
    target_owner_small_id: u16,
    target_unit_id: i32,
    shell_unit_id: Option<i32>,
    seed: i32,
    damage_random: Option<PseudoRandom>,
    destroy_at_tick: Option<u32>,
    last_target_tile: Option<TileRef>,
    path: Vec<TileRef>,
    path_index: usize,
    active: bool,
}

impl ShellExecution {
    pub fn new(
        spawn: TileRef,
        owner_small_id: u16,
        owner_unit_id: i32,
        target_owner_small_id: u16,
        target_unit_id: i32,
    ) -> Self {
        Self {
            spawn,
            owner_small_id,
            owner_unit_id,
            target_owner_small_id,
            target_unit_id,
            shell_unit_id: None,
            seed: 0,
            damage_random: None,
            destroy_at_tick: None,
            last_target_tile: None,
            path: Vec::new(),
            path_index: 0,
            active: true,
        }
    }

    fn remove_shell(&mut self, game: &mut Game) {
        if let Some(shell_id) = self.shell_unit_id {
            game.remove_unit(self.owner_small_id, shell_id);
        }
        self.active = false;
    }

    /// TS `getEffectOnTargetForTesting()` - `WarshipVeterancy.test.ts`'s shell-damage-scaling
    /// case reads this directly rather than through a full `hit_target()` call. Requires
    /// `init()` to have run first (so `damage_random` is seeded), same as TS's `init()`
    /// requirement before calling `effectOnTarget()`.
    pub fn get_effect_on_target_for_testing(&mut self, game: &Game) -> i32 {
        self.effect_on_target(game)
    }

    /// TS `effectOnTarget()`.
    fn effect_on_target(&mut self, game: &Game) -> i32 {
        let roll = self
            .damage_random
            .as_mut()
            .map(|random| random.next_int(1, 6))
            .unwrap_or(1);
        let mut damage_multiplier = (roll - 1) * 25 + 200;
        let veterancy = game.unit_veterancy(self.owner_small_id, self.owner_unit_id);
        if veterancy > 0 {
            let bonus_percent = game.wire.warship_veterancy_shell_damage_bonus();
            damage_multiplier = damage_multiplier * (100 + veterancy * bonus_percent) / 100;
        }
        damage_multiplier
    }

    fn hit_target(&mut self, game: &mut Game) {
        let damage = self.effect_on_target(game);
        let mut destroyed_transport_tile = None;
        let target_type = game.unit_type_of(self.target_owner_small_id, self.target_unit_id);
        let destroyed =
            if let Some(target) = game.unit_mut(self.target_owner_small_id, self.target_unit_id) {
                target.health = (target.health - damage).max(0);
                let died = target.health == 0;
                if died && target.unit_type == TRANSPORT {
                    destroyed_transport_tile = Some(target.tile as TileRef);
                }
                died
            } else {
                false
            };
        if destroyed {
            // TS `UnitImpl.modifyHealth`: `if (this._health === 0n) this.delete(true, attacker)`
            // - record the shooter as destroyer *before* removing the unit, since native (unlike
            // TS's inert-but-still-inspectable deleted `Unit`) has nothing left to query once
            // gone. See `Game::record_transport_kill`'s doc comment.
            if let Some(tile) = destroyed_transport_tile {
                game.record_transport_kill(
                    self.target_unit_id,
                    self.target_owner_small_id,
                    self.owner_small_id,
                    tile,
                );
            }
            // TS: award veterancy to the firing warship on a killing blow, but only if the
            // firing unit is itself still alive (a shell that lands after its own warship was
            // sunk mid-flight awards nothing) and is actually a Warship (this execution is only
            // ever constructed by `WarshipExecution` in this port, but TS's own `ownerUnit.type()
            // === UnitType.Warship` guard is kept for parity/future-proofing).
            if game.unit_type_of(self.owner_small_id, self.owner_unit_id).as_deref() == Some(WARSHIP)
            {
                if let Some(target_type) = target_type {
                    game.record_kill(self.owner_small_id, self.owner_unit_id, &target_type);
                }
            }
            game.remove_unit(self.target_owner_small_id, self.target_unit_id);
        }
        self.remove_shell(game);
    }
}

impl Execution for ShellExecution {
    fn init(&mut self, _game: &mut Game, tick: u32) {
        self.seed = tick as i32;
        self.damage_random = Some(PseudoRandom::new(tick as i32));
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        if self.shell_unit_id.is_none() {
            self.shell_unit_id = Some(game.build_unit(self.owner_small_id, SHELL, self.spawn));
        }
        let Some(shell_id) = self.shell_unit_id else {
            self.active = false;
            return;
        };
        if !game.unit_exists(self.owner_small_id, shell_id)
            || !game.unit_exists(self.target_owner_small_id, self.target_unit_id)
            || self.owner_small_id == self.target_owner_small_id
            || self.destroy_at_tick.is_some_and(|destroy| tick >= destroy)
        {
            self.remove_shell(game);
            return;
        }

        let owner_alive = game.unit_exists(self.owner_small_id, self.owner_unit_id);
        if !owner_alive && self.destroy_at_tick.is_none() {
            self.destroy_at_tick = Some(tick + 50);
        }
        let target_tile = game
            .unit(self.target_owner_small_id, self.target_unit_id)
            .map(|unit| unit.tile as TileRef)
            .unwrap();
        for _ in 0..3 {
            let current = game
                .unit_tile_of(self.owner_small_id, shell_id)
                .unwrap_or(self.spawn);
            if current == target_tile {
                self.hit_target(game);
                return;
            }
            if self.last_target_tile != Some(target_tile) {
                self.path = air_path(game, self.seed, current, target_tile);
                self.path_index = usize::from(self.path.first() == Some(&current));
                self.last_target_tile = Some(target_tile);
            }
            if self.path_index >= self.path.len() {
                self.hit_target(game);
                return;
            }
            let next = self.path[self.path_index];
            self.path_index += 1;
            game.move_unit(self.owner_small_id, shell_id, next);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// Ported from `openfront/tests/WarshipVeterancy.test.ts`'s two `ShellExecution`-facing
// cases (the rest of that file is `Game::record_kill`/etc. tests - see `game.rs`'s
// `veterancy_tests`). Uses `test_util::plains_game` (land is irrelevant here - shells fly
// via `PathFinding.Air`, which doesn't consult terrain) rather than `Game::default()`'s 1x1
// map, since the two units need distinct, non-adjacent tiles.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type::WARSHIP;
    use crate::execution::ExecEnum;
    use crate::game::{Player, PlayerType};

    fn add_player(game: &mut Game, small_id: u16) {
        game.add_player(Player {
            id: small_id.to_string(),
            small_id,
            player_type: PlayerType::Human,
            ..Default::default()
        });
    }

    #[test]
    fn shell_damage_scales_with_the_firing_warships_veterancy() {
        let mut game = crate::test_util::plains_game(20, 20);
        add_player(&mut game, 1);
        add_player(&mut game, 2);
        let max_vet = game.wire.warship_max_veterancy();
        let bonus_percent = game.wire.warship_veterancy_shell_damage_bonus();

        let target = game.build_unit(2, WARSHIP, game.map.ref_xy(10, 10));
        let base_shooter = game.build_unit(1, WARSHIP, game.map.ref_xy(0, 0));
        let vet_shooter = game.build_unit(1, WARSHIP, game.map.ref_xy(1, 0));
        for _ in 0..max_vet {
            game.record_kill(1, vet_shooter, WARSHIP);
        }
        assert_eq!(game.unit_veterancy(1, vet_shooter), max_vet);

        let mut boosted_values = std::collections::HashSet::new();
        for tick in 1..30u32 {
            // Same tick seed -> same PRNG roll for both shells fired this iteration.
            let mut base_shell =
                ShellExecution::new(game.map.ref_xy(0, 0), 1, base_shooter, 2, target);
            let mut vet_shell =
                ShellExecution::new(game.map.ref_xy(1, 0), 1, vet_shooter, 2, target);
            base_shell.init(&mut game, tick);
            vet_shell.init(&mut game, tick);

            let d_base = base_shell.get_effect_on_target_for_testing(&game);
            let d_vet = vet_shell.get_effect_on_target_for_testing(&game);

            // Same roll -> base damage (250 base) equals the rolled multiplier and the
            // veteran's shot is the integer-boosted value.
            assert_eq!(d_vet, d_base * (100 + max_vet * bonus_percent) / 100);
            boosted_values.insert(d_vet);
        }

        assert!(boosted_values.len() > 1, "the roll must vary across ticks");
    }

    #[test]
    fn a_shell_landing_the_killing_blow_awards_veterancy_to_the_firing_warship() {
        let mut game = crate::test_util::plains_game(20, 20);
        add_player(&mut game, 1);
        add_player(&mut game, 2);
        let shooter = game.build_unit(1, WARSHIP, game.map.ref_xy(0, 0));
        let target = game.build_unit(2, WARSHIP, game.map.ref_xy(10, 10));

        // Leave the target on its last sliver of health so any shell finishes it.
        game.unit_mut(2, target).unwrap().health = 1;

        game.add_execution(ExecEnum::Shell(ShellExecution::new(
            game.map.ref_xy(0, 0),
            1,
            shooter,
            2,
            target,
        )));
        for _ in 0..30 {
            if !game.unit_exists(2, target) {
                break;
            }
            game.execute_next_tick();
        }

        assert!(!game.unit_exists(2, target));
        assert_eq!(game.unit_veterancy(1, shooter), 1);
    }
}
