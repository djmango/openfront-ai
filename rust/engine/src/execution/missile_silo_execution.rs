//! Passive Missile Silo reload-cooldown ticking (`MissileSiloExecution.ts`).

use super::Execution;
use crate::game::Game;

pub struct MissileSiloExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl MissileSiloExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for MissileSiloExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        let Some(u) = game.unit(self.small_id, self.unit_id) else {
            self.active = false;
            return;
        };
        if u.under_construction {
            return;
        }
        let Some(&front_time) = u.missile_timer_queue.first() else {
            return;
        };
        let cooldown = game.wire.silo_cooldown() as i64 - (game.ticks() as i64 - front_time as i64);
        if cooldown <= 0 {
            game.unit_reload_missile(self.small_id, self.unit_id);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

// TS `MissileSilo.test.ts`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::execution::nuke_execution::NukeExecution;
    use crate::execution::upgrade_structure::UpgradeStructureExecution;
    use crate::execution::ExecEnum;
    use crate::game::{Player, PlayerType};
    use crate::map::TileRef;

    fn tiny_all_land_map(width: u32, height: u32) -> crate::map::GameMap {
        let n = (width * height) as usize;
        crate::map::GameMap::from_terrain_bytes(
            &crate::map::MapMeta {
                width,
                height,
                num_land_tiles: n as u32,
            },
            &vec![0x80u8; n],
        )
        .unwrap()
    }

    /// A `Game` with an all-land map, one Human player owning a completed
    /// level-1 Missile Silo at `(1, 1)`, and its `MissileSiloExecution`
    /// already running. `spawn_immunity_duration` is forced to 0 - TS's
    /// `TestConfig` (used by every `setup()`-based test, including this
    /// file's TS original) defaults it to 0, unlike production Config's 50.
    fn game_with_silo(width: u32, height: u32) -> (Game, u16, TileRef, i32) {
        let mut game = Game::default();
        game.map = tiny_all_land_map(width, height);
        game.wire = crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "tiny".into(),
                difficulty: "Medium".into(),
                donate_gold: false,
                donate_troops: false,
                game_type: "Singleplayer".into(),
                game_mode: "Free For All".into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: false,
                infinite_troops: false,
                instant_build: false,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units: None,
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: Some(0),
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        game.end_spawn_phase();
        game.add_player(Player {
            id: "attacker".to_string(),
            small_id: 1,
            player_type: PlayerType::Human,
            gold: 1_000_000_000,
            ..Default::default()
        });
        let silo_tile = game.map.ref_xy(1, 1);
        game.conquer(1, silo_tile);
        let silo_id = game.build_unit(1, unit_type::MISSILE_SILO, silo_tile);
        game.add_execution(ExecEnum::MissileSilo(MissileSiloExecution::new(1, silo_id)));
        (game, 1, silo_tile, silo_id)
    }

    fn build_nuke(game: &mut Game, attacker: u16, target: TileRef) {
        game.add_execution(ExecEnum::Nuke(NukeExecution::new(
            unit_type::ATOM_BOMB,
            attacker,
            target,
            None,
            -1.0,
            0,
            true,
        )));
        game.execute_next_tick();
        game.execute_next_tick();
    }

    #[test]
    fn missile_silo_launches_a_nuke_and_it_eventually_detonates() {
        let (mut game, attacker, _silo_tile, _silo_id) = game_with_silo(60, 60);
        let target = game.map.ref_xy(7, 7);

        build_nuke(&mut game, attacker, target);

        assert_eq!(game.unit_count(attacker, unit_type::ATOM_BOMB), 1);
        let bomb_tile = game
            .player_by_small_id(attacker)
            .unwrap()
            .units
            .iter()
            .find(|u| u.unit_type == unit_type::ATOM_BOMB)
            .unwrap()
            .tile as TileRef;
        assert_ne!(bomb_tile, target, "bomb should still be in flight, not already at its target");

        for _ in 0..60 {
            if game.unit_count(attacker, unit_type::ATOM_BOMB) == 0 {
                break;
            }
            game.execute_next_tick();
        }
        assert_eq!(
            game.unit_count(attacker, unit_type::ATOM_BOMB),
            0,
            "nuke should have detonated well within 60 ticks over such a short trajectory"
        );
    }

    #[test]
    fn missile_silo_only_launches_one_nuke_at_a_time() {
        let (mut game, attacker, _silo_tile, _silo_id) = game_with_silo(60, 60);
        // Far enough that the first bomb is still in flight (not yet
        // detonated) by the time the second launch attempt is made, so the
        // count below unambiguously reflects "second launch was rejected"
        // rather than "first bomb also already gone".
        let target = game.map.ref_xy(50, 50);

        build_nuke(&mut game, attacker, target);
        // Second launch attempt: the only silo is now on cooldown, so
        // `nuke_spawn` finds no available silo and this one never builds.
        build_nuke(&mut game, attacker, target);

        assert_eq!(game.unit_count(attacker, unit_type::ATOM_BOMB), 1);
    }

    #[test]
    fn missile_silo_stays_in_cooldown_for_exactly_silo_cooldown_ticks() {
        let (mut game, attacker, _silo_tile, silo_id) = game_with_silo(60, 60);
        assert!(!game.unit_is_in_cooldown(attacker, silo_id));

        // Far enough away that the blast doesn't also destroy the silo.
        let target = game.map.ref_xy(50, 50);
        build_nuke(&mut game, attacker, target);
        assert_eq!(game.unit_count(attacker, unit_type::ATOM_BOMB), 1);

        for _ in 0..(game.wire.silo_cooldown() - 2) {
            game.execute_next_tick();
            assert!(game.unit_is_in_cooldown(attacker, silo_id));
        }

        game.execute_next_tick();
        game.execute_next_tick();

        assert!(!game.unit_is_in_cooldown(attacker, silo_id));
    }

    #[test]
    fn missile_silo_level_increases_after_upgrade() {
        let (mut game, attacker, _silo_tile, silo_id) = game_with_silo(20, 20);
        assert_eq!(game.unit(attacker, silo_id).unwrap().level, 1);

        game.add_execution(ExecEnum::UpgradeStructure(UpgradeStructureExecution::new(
            attacker, silo_id,
        )));
        game.execute_next_tick();
        game.execute_next_tick();

        assert_eq!(game.unit(attacker, silo_id).unwrap().level, 2);
    }
}
