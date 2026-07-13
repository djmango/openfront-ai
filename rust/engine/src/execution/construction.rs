//! Structure construction (`ConstructionExecution.ts` subset for nation cities).

use super::{
    city_execution::CityExecution, defense_post_execution::DefensePostExecution,
    factory_execution::FactoryExecution, mirv_execution::MirvExecution,
    missile_silo_execution::MissileSiloExecution, nuke_execution::NukeExecution,
    port_execution::PortExecution, sam_launcher_execution::SamLauncherExecution, ExecEnum,
    Execution, WarshipExecution,
};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;

fn complete_construction(game: &mut Game, small_id: u16, unit_type: &str, unit_id: i32) {
    game.set_unit_under_construction(small_id, unit_id, false);
    match unit_type {
        unit_type::CITY => {
            game.add_execution(ExecEnum::City(CityExecution::new(small_id, unit_id)));
        }
        unit_type::DEFENSE_POST => {
            game.add_execution(ExecEnum::DefensePost(DefensePostExecution::new(
                small_id, unit_id,
            )));
        }
        unit_type::PORT => {
            game.add_execution(ExecEnum::Port(PortExecution::new(small_id, unit_id)));
        }
        unit_type::FACTORY => {
            game.add_execution(ExecEnum::Factory(FactoryExecution::new(
                small_id, unit_id,
            )));
        }
        unit_type::MISSILE_SILO => {
            game.add_execution(ExecEnum::MissileSilo(MissileSiloExecution::new(
                small_id, unit_id,
            )));
        }
        unit_type::SAM_LAUNCHER => {
            game.add_execution(ExecEnum::SamLauncher(SamLauncherExecution::new(
                small_id, unit_id,
            )));
        }
        _ => {}
    }
}

/// Current owner of `unit_id`, if the unit still exists (may differ from the
/// builder after `capture_unit` / structure-ownership reconciliation).
fn unit_owner_small_id(game: &Game, unit_id: i32) -> Option<u16> {
    game.find_unit_owner(unit_id)
}

/// TS `ConstructionExecution.isStructure()` - Atom Bomb / Hydrogen Bomb / MIRV are *not*
/// structures: they charge gold and resolve their own spawn tile lazily inside
/// `NukeExecution`/`MirvExecution`'s own first tick instead of being built here.
fn is_structure(unit_type: &str) -> bool {
    use crate::core::schemas::unit_type as ut;
    matches!(
        unit_type,
        ut::PORT | ut::MISSILE_SILO | ut::DEFENSE_POST | ut::SAM_LAUNCHER | ut::CITY | ut::FACTORY
    )
}

pub struct ConstructionExecution {
    small_id: u16,
    unit_type: String,
    tile: TileRef,
    rocket_direction_up: bool,
    unit_id: Option<i32>,
    ticks_until_complete: u32,
    active: bool,
    /// Set once the non-structure (nuke/MIRV) delegate execution has been dispatched.
    delegated: bool,
}

impl ConstructionExecution {
    pub fn new(small_id: u16, unit_type: &str, tile: TileRef, rocket_direction_up: bool) -> Self {
        Self {
            small_id,
            unit_type: unit_type.to_string(),
            tile,
            rocket_direction_up,
            unit_id: None,
            ticks_until_complete: 0,
            active: true,
            delegated: false,
        }
    }

    /// Test helper: construction already past `buildUnit`, counting down.
    #[cfg(test)]
    fn mid_build_for_test(small_id: u16, unit_type: &str, unit_id: i32, ticks_left: u32) -> Self {
        Self {
            small_id,
            unit_type: unit_type.to_string(),
            tile: 0,
            rocket_direction_up: true,
            unit_id: Some(unit_id),
            ticks_until_complete: ticks_left,
            active: true,
            delegated: false,
        }
    }
}

impl Execution for ConstructionExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if game.wire.is_unit_disabled(&self.unit_type) {
            self.active = false;
            return;
        }
        let max_tile = game.map.width * game.map.height;
        if self.tile >= max_tile {
            self.active = false;
        }
    }

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }

        if !self.delegated && !is_structure(&self.unit_type) {
            self.delegated = true;
            self.active = false;
            match self.unit_type.as_str() {
                unit_type::ATOM_BOMB | unit_type::HYDROGEN_BOMB => {
                    game.add_execution(ExecEnum::Nuke(NukeExecution::new(
                        &self.unit_type,
                        self.small_id,
                        self.tile,
                        None,
                        -1.0,
                        0,
                        self.rocket_direction_up,
                    )));
                }
                unit_type::MIRV => {
                    game.add_execution(ExecEnum::Mirv(MirvExecution::new(self.small_id, self.tile)));
                }
                unit_type::WARSHIP => {
                    game.add_execution(ExecEnum::Warship(WarshipExecution::new(
                        self.small_id,
                        self.tile,
                    )));
                }
                _ => {}
            }
            return;
        }

        if self.unit_id.is_none() {
            let Some(spawn_tile) = super::nation_structures::resolve_structure_spawn_tile(
                game,
                self.small_id,
                &self.unit_type,
                self.tile,
            ) else {
                self.active = false;
                return;
            };
            let cost = game.structure_cost(self.small_id, &self.unit_type);
            let Some(p) = game.player_by_small_id(self.small_id) else {
                self.active = false;
                return;
            };
            if p.gold < cost {
                self.active = false;
                return;
            }
            let duration = game.wire.construction_ticks(&self.unit_type);
            let id = game.build_unit(self.small_id, &self.unit_type, spawn_tile);
            if duration > 0 {
                game.set_unit_under_construction(self.small_id, id, true);
                self.unit_id = Some(id);
                self.ticks_until_complete = duration;
                return;
            }
            complete_construction(game, self.small_id, &self.unit_type, id);
            self.active = false;
            return;
        }

        let unit_id = self.unit_id.expect("unit_id set above");
        // TS `ConstructionExecution.tick`: when the structure is captured mid-
        // build, retarget to the new owner so completion clears
        // `under_construction` and spawns City/Port/… execs for them — not the
        // original builder (who no longer holds the unit).
        let Some(owner) = unit_owner_small_id(game, unit_id) else {
            self.active = false;
            return;
        };
        self.small_id = owner;

        if self.ticks_until_complete == 0 {
            complete_construction(game, self.small_id, &self.unit_type, unit_id);
            self.active = false;
            return;
        }
        self.ticks_until_complete -= 1;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::execution::ExecEnum;
    use crate::game::{Game, PlayerInfo, PlayerType};
    use crate::test_util::plains_game;

    fn add_nation(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    /// TS retargets ConstructionExecution to the capturer; without that,
    /// `under_construction` stays stuck on the captured city and
    /// `completed_city_level_sum` (hence troop income) permanently lags.
    #[test]
    fn captured_city_under_construction_completes_for_new_owner() {
        let mut game = plains_game(20, 20);
        let builder = add_nation(&mut game, "builder");
        let capturer = add_nation(&mut game, "capturer");
        for sid in [builder, capturer] {
            let tile = game.map.ref_xy(if sid == builder { 0 } else { 5 }, 0);
            game.conquer(sid, tile);
            if let Some(p) = game.player_by_small_id_mut(sid) {
                p.spawn_tile = Some(tile);
                p.alive = true;
                p.gold = 1_000_000;
            }
        }

        let city_tile = game.map.ref_xy(0, 0);
        let city_id = game.build_unit(builder, unit_type::CITY, city_tile);
        game.set_unit_under_construction(builder, city_id, true);
        game.add_execution(ExecEnum::Construction(ConstructionExecution::mid_build_for_test(
            builder,
            unit_type::CITY,
            city_id,
            2,
        )));

        // Capture mid-build (structure ownership reconciliation path).
        game.capture_unit(builder, capturer, city_id);
        assert!(game.player_by_small_id(builder).unwrap().units.is_empty());
        assert_eq!(game.completed_city_level_sum(capturer), 0);

        // ticks_left=2 → decrement to 1, to 0, then complete on the next tick.
        for _ in 0..4 {
            game.execute_next_tick();
        }

        let city = game
            .player_by_small_id(capturer)
            .and_then(|p| p.units.iter().find(|u| u.id == city_id))
            .expect("capturer still owns the city");
        assert!(
            !city.under_construction,
            "construction must clear under_construction on the capturer"
        );
        assert_eq!(game.completed_city_level_sum(capturer), 1);
        assert_eq!(game.completed_city_level_sum(builder), 0);
    }
}
