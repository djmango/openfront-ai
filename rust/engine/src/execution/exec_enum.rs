//! Concrete execution enum - avoids `Box<dyn Execution>` vtable dispatch on the tick hot path.

use super::{
    AttackExecution, Execution, MarkDisconnectedExecution, NationExecution, NoOpExecution,
    PlayerExecution, SpawnExecution, SpawnTimerExecution, TransportShipExecution,
    WinCheckExecution,
};
use crate::bot::TribeExecution;
use crate::game::Game;

pub enum ExecEnum {
    Spawn(SpawnExecution),
    Attack(AttackExecution),
    SpawnTimer(SpawnTimerExecution),
    WinCheck(WinCheckExecution),
    NoOp(NoOpExecution),
    MarkDisconnected(MarkDisconnectedExecution),
    TransportShip(TransportShipExecution),
    TribeMassSpawn(crate::bot::TribeMassSpawn),
    Player(PlayerExecution),
    Tribe(TribeExecution),
    Nation(NationExecution),
}

impl Execution for ExecEnum {
    fn init(&mut self, game: &mut Game, tick: u32) {
        match self {
            ExecEnum::Spawn(e) => e.init(game, tick),
            ExecEnum::Attack(e) => e.init(game, tick),
            ExecEnum::SpawnTimer(e) => e.init(game, tick),
            ExecEnum::WinCheck(e) => e.init(game, tick),
            ExecEnum::NoOp(e) => e.init(game, tick),
            ExecEnum::MarkDisconnected(e) => e.init(game, tick),
            ExecEnum::TransportShip(e) => e.init(game, tick),
            ExecEnum::TribeMassSpawn(e) => e.init(game, tick),
            ExecEnum::Player(e) => e.init(game, tick),
            ExecEnum::Tribe(e) => e.init(game, tick),
            ExecEnum::Nation(e) => e.init(game, tick),
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        match self {
            ExecEnum::Spawn(e) => e.tick(game, tick),
            ExecEnum::Attack(e) => e.tick(game, tick),
            ExecEnum::SpawnTimer(e) => e.tick(game, tick),
            ExecEnum::WinCheck(e) => e.tick(game, tick),
            ExecEnum::NoOp(e) => e.tick(game, tick),
            ExecEnum::MarkDisconnected(e) => e.tick(game, tick),
            ExecEnum::TransportShip(e) => e.tick(game, tick),
            ExecEnum::TribeMassSpawn(e) => e.tick(game, tick),
            ExecEnum::Player(e) => e.tick(game, tick),
            ExecEnum::Tribe(e) => e.tick(game, tick),
            ExecEnum::Nation(e) => e.tick(game, tick),
        }
    }

    fn is_active(&self) -> bool {
        match self {
            ExecEnum::Spawn(e) => e.is_active(),
            ExecEnum::Attack(e) => e.is_active(),
            ExecEnum::SpawnTimer(e) => e.is_active(),
            ExecEnum::WinCheck(e) => e.is_active(),
            ExecEnum::NoOp(e) => e.is_active(),
            ExecEnum::MarkDisconnected(e) => e.is_active(),
            ExecEnum::TransportShip(e) => e.is_active(),
            ExecEnum::TribeMassSpawn(e) => e.is_active(),
            ExecEnum::Player(e) => e.is_active(),
            ExecEnum::Tribe(e) => e.is_active(),
            ExecEnum::Nation(e) => e.is_active(),
        }
    }

    fn active_during_spawn(&self) -> bool {
        match self {
            ExecEnum::Spawn(e) => e.active_during_spawn(),
            ExecEnum::Attack(e) => e.active_during_spawn(),
            ExecEnum::SpawnTimer(e) => e.active_during_spawn(),
            ExecEnum::WinCheck(e) => e.active_during_spawn(),
            ExecEnum::NoOp(e) => e.active_during_spawn(),
            ExecEnum::MarkDisconnected(e) => e.active_during_spawn(),
            ExecEnum::TransportShip(e) => e.active_during_spawn(),
            ExecEnum::TribeMassSpawn(e) => e.active_during_spawn(),
            ExecEnum::Player(e) => e.active_during_spawn(),
            ExecEnum::Tribe(e) => e.active_during_spawn(),
            ExecEnum::Nation(e) => e.active_during_spawn(),
        }
    }

    fn is_spawn_timer(&self) -> bool {
        match self {
            ExecEnum::SpawnTimer(e) => e.is_spawn_timer(),
            _ => false,
        }
    }

    fn as_attack(&mut self) -> Option<&mut AttackExecution> {
        match self {
            ExecEnum::Attack(e) => Some(e),
            _ => None,
        }
    }
}
