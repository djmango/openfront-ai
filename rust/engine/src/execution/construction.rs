//! Structure construction (`ConstructionExecution.ts` subset for nation cities).

use super::{
    city_execution::CityExecution, defense_post_execution::DefensePostExecution,
    factory_execution::FactoryExecution, mirv_execution::MirvExecution,
    missile_silo_execution::MissileSiloExecution, nuke_execution::NukeExecution,
    port_execution::PortExecution, sam_launcher_execution::SamLauncherExecution, ExecEnum,
    Execution,
};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;

fn complete_construction(game: &mut Game, small_id: u16, unit_type: &str, unit_id: i32) {
    if let Some(p) = game.player_by_small_id_mut(small_id) {
        if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
            u.under_construction = false;
        }
    }
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
                if let Some(p) = game.player_by_small_id_mut(self.small_id) {
                    if let Some(u) = p.units.iter_mut().find(|u| u.id == id) {
                        u.under_construction = true;
                    }
                }
                self.unit_id = Some(id);
                self.ticks_until_complete = duration;
                return;
            }
            complete_construction(game, self.small_id, &self.unit_type, id);
            self.active = false;
            return;
        }

        if self.ticks_until_complete == 0 {
            if let Some(id) = self.unit_id {
                complete_construction(game, self.small_id, &self.unit_type, id);
            }
            self.active = false;
            return;
        }
        self.ticks_until_complete -= 1;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
