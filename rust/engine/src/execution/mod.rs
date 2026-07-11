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
pub mod donate_gold;
pub mod embargo_exec;
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
pub mod doomsday_clock_execution;
pub mod factory_execution;
pub mod mirv_execution;
pub mod missile_silo_execution;
pub mod nuke_ai;
pub mod nuke_execution;
pub mod parabola;
pub mod port_execution;
pub mod recompute_rail_cluster;
pub mod sam_launcher_execution;
pub mod sam_missile_execution;
pub mod shell_execution;
pub mod train_execution;
pub mod train_station_execution;
pub mod construction;
pub mod nation_structures;
pub mod nation_tick;
pub mod noop;
pub mod ordered_map;
pub mod ordered_tiles;
pub mod ordered_units;
pub mod player;
pub mod player_clusters;
pub mod retreat;
pub mod spawn;
pub mod spawn_timer;
pub mod spawn_util;
pub mod target_player;
pub mod trade_ship_execution;
pub mod transport_ship;
pub mod upgrade_structure;
pub mod warship;
pub mod warship_ai;
pub mod win_check;

pub use attack::AttackExecution;
pub use alliance_exec::{
    AllianceExtensionExecution, AllianceRejectExecution, AllianceRequestExecution,
    BreakAllianceExecution,
};
pub use donate::DonateTroopsExecution;
pub use donate_gold::DonateGoldExecution;
pub use embargo_exec::{EmbargoAllExecution, EmbargoExecution};
pub use exec_enum::ExecEnum;
pub use intent::{intent_to_execution, turn_to_executions};
pub use mark_disconnected::MarkDisconnectedExecution;
pub use nation::{NationExecution, NationRuntime};
pub use defense_post_execution::DefensePostExecution;
pub use doomsday_clock_execution::DoomsdayClockExecution;
pub use city_execution::CityExecution;
pub use factory_execution::FactoryExecution;
pub use port_execution::PortExecution;
pub use recompute_rail_cluster::RecomputeRailClusterExecution;
pub use train_execution::TrainExecution;
pub use train_station_execution::TrainStationExecution;
pub use construction::ConstructionExecution;
pub use mirv_execution::MirvExecution;
pub use missile_silo_execution::MissileSiloExecution;
pub use nuke_execution::NukeExecution;
pub use nation_batch::NationBatch;
pub use sam_launcher_execution::SamLauncherExecution;
pub use sam_missile_execution::SamMissileExecution;
pub use shell_execution::ShellExecution;
pub use noop::NoOpExecution;
pub use player::PlayerExecution;
pub use retreat::{BoatRetreatExecution, RetreatExecution};
pub use spawn::SpawnExecution;
pub use spawn_timer::SpawnTimerExecution;
pub use target_player::TargetPlayerExecution;
pub use trade_ship_execution::TradeShipExecution;
pub use transport_ship::TransportShipExecution;
pub use upgrade_structure::UpgradeStructureExecution;
pub use warship::WarshipExecution;
pub use win_check::WinCheckExecution;
