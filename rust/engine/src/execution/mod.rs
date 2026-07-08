//! Execution trait + core tick-side effects (Phase 2).

use crate::game::Game;

#[derive(Debug, Clone)]
pub struct HashUpdate {
    pub tick: u32,
    pub hash: i64,
}

#[derive(Debug, Clone)]
pub struct WinUpdate {
    pub tick: u32,
    pub winner: String,
}

pub trait Execution: Send {
    fn init(&mut self, game: &mut Game, tick: u32);
    fn tick(&mut self, game: &mut Game, tick: u32);
    fn is_active(&self) -> bool;
    fn active_during_spawn(&self) -> bool {
        false
    }
    fn is_spawn_timer(&self) -> bool {
        false
    }
    fn as_attack(&mut self) -> Option<&mut attack::AttackExecution> {
        None
    }
}

pub mod ai_attack;
pub mod alliance_exec;
pub mod attack;
pub mod donate;
pub mod exec_enum;
pub mod flat_heap;
pub mod intent;
pub mod mark_disconnected;
pub mod nation;
pub mod nation_alliance;
pub mod nation_batch;
pub mod nation_emoji;
pub mod city_execution;
pub mod defense_post_execution;
pub mod factory_execution;
pub mod port_execution;
pub mod train_station_execution;
pub mod construction;
pub mod nation_structures;
pub mod nation_tick;
pub mod noop;
pub mod ordered_tiles;
pub mod player;
pub mod player_clusters;
pub mod retreat;
pub mod spawn;
pub mod spawn_timer;
pub mod spawn_util;
pub mod transport_ship;
pub mod upgrade_structure;
pub mod win_check;

pub use attack::AttackExecution;
pub use alliance_exec::{
    AllianceExtensionExecution, AllianceRejectExecution, AllianceRequestExecution,
    BreakAllianceExecution,
};
pub use donate::DonateTroopsExecution;
pub use exec_enum::ExecEnum;
pub use intent::{intent_to_execution, turn_to_executions};
pub use mark_disconnected::MarkDisconnectedExecution;
pub use nation::{NationExecution, NationRuntime};
pub use defense_post_execution::DefensePostExecution;
pub use city_execution::CityExecution;
pub use factory_execution::FactoryExecution;
pub use port_execution::PortExecution;
pub use train_station_execution::TrainStationExecution;
pub use construction::ConstructionExecution;
pub use nation_batch::NationBatch;
pub use noop::NoOpExecution;
pub use player::PlayerExecution;
pub use retreat::RetreatExecution;
pub use spawn::SpawnExecution;
pub use spawn_timer::SpawnTimerExecution;
pub use transport_ship::TransportShipExecution;
pub use upgrade_structure::UpgradeStructureExecution;
pub use win_check::WinCheckExecution;
