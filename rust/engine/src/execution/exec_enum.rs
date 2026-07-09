//! Concrete execution enum - avoids `Box<dyn Execution>` vtable dispatch on the tick hot path.

use super::{
    AllianceExtensionExecution, AllianceRejectExecution, AllianceRequestExecution,
    AttackExecution, BreakAllianceExecution, CityExecution, ConstructionExecution,
    DefensePostExecution, DonateGoldExecution, DonateTroopsExecution, FactoryExecution, MirvExecution,
    MissileSiloExecution, NukeExecution, PortExecution, SamLauncherExecution,
    SamMissileExecution, Execution, MarkDisconnectedExecution, NationExecution, NoOpExecution,
    PlayerExecution, RecomputeRailClusterExecution, RetreatExecution, SpawnExecution,
    SpawnTimerExecution, TrainExecution, TrainStationExecution, TransportShipExecution,
    UpgradeStructureExecution, WinCheckExecution,
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
    Construction(ConstructionExecution),
    City(CityExecution),
    DefensePost(DefensePostExecution),
    Port(PortExecution),
    Factory(FactoryExecution),
    TrainStation(TrainStationExecution),
    Train(TrainExecution),
    RecomputeRailCluster(RecomputeRailClusterExecution),
    UpgradeStructure(UpgradeStructureExecution),
    AllianceRequest(AllianceRequestExecution),
    AllianceReject(AllianceRejectExecution),
    BreakAlliance(BreakAllianceExecution),
    AllianceExtension(AllianceExtensionExecution),
    DonateTroops(DonateTroopsExecution),
    DonateGold(DonateGoldExecution),
    Retreat(RetreatExecution),
    TribeMassSpawn(crate::bot::TribeMassSpawn),
    Player(PlayerExecution),
    Tribe(TribeExecution),
    Nation(NationExecution),
    Nuke(NukeExecution),
    Mirv(MirvExecution),
    MissileSilo(MissileSiloExecution),
    SamLauncher(SamLauncherExecution),
    SamMissile(SamMissileExecution),
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
            ExecEnum::Construction(e) => e.init(game, tick),
            ExecEnum::City(e) => e.init(game, tick),
            ExecEnum::DefensePost(e) => e.init(game, tick),
            ExecEnum::Port(e) => e.init(game, tick),
            ExecEnum::Factory(e) => e.init(game, tick),
            ExecEnum::TrainStation(e) => e.init(game, tick),
            ExecEnum::Train(e) => e.init(game, tick),
            ExecEnum::RecomputeRailCluster(e) => e.init(game, tick),
            ExecEnum::UpgradeStructure(e) => e.init(game, tick),
            ExecEnum::AllianceRequest(e) => e.init(game, tick),
            ExecEnum::AllianceReject(e) => e.init(game, tick),
            ExecEnum::BreakAlliance(e) => e.init(game, tick),
            ExecEnum::AllianceExtension(e) => e.init(game, tick),
            ExecEnum::DonateTroops(e) => e.init(game, tick),
            ExecEnum::DonateGold(e) => e.init(game, tick),
            ExecEnum::Retreat(e) => e.init(game, tick),
            ExecEnum::TribeMassSpawn(e) => e.init(game, tick),
            ExecEnum::Player(e) => e.init(game, tick),
            ExecEnum::Tribe(e) => e.init(game, tick),
            ExecEnum::Nation(e) => e.init(game, tick),
            ExecEnum::Nuke(e) => e.init(game, tick),
            ExecEnum::Mirv(e) => e.init(game, tick),
            ExecEnum::MissileSilo(e) => e.init(game, tick),
            ExecEnum::SamLauncher(e) => e.init(game, tick),
            ExecEnum::SamMissile(e) => e.init(game, tick),
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
            ExecEnum::Construction(e) => e.tick(game, tick),
            ExecEnum::City(e) => e.tick(game, tick),
            ExecEnum::DefensePost(e) => e.tick(game, tick),
            ExecEnum::Port(e) => e.tick(game, tick),
            ExecEnum::Factory(e) => e.tick(game, tick),
            ExecEnum::TrainStation(e) => e.tick(game, tick),
            ExecEnum::Train(e) => e.tick(game, tick),
            ExecEnum::RecomputeRailCluster(e) => e.tick(game, tick),
            ExecEnum::UpgradeStructure(e) => e.tick(game, tick),
            ExecEnum::AllianceRequest(e) => e.tick(game, tick),
            ExecEnum::AllianceReject(e) => e.tick(game, tick),
            ExecEnum::BreakAlliance(e) => e.tick(game, tick),
            ExecEnum::AllianceExtension(e) => e.tick(game, tick),
            ExecEnum::DonateTroops(e) => e.tick(game, tick),
            ExecEnum::DonateGold(e) => e.tick(game, tick),
            ExecEnum::Retreat(e) => e.tick(game, tick),
            ExecEnum::TribeMassSpawn(e) => e.tick(game, tick),
            ExecEnum::Player(e) => e.tick(game, tick),
            ExecEnum::Tribe(e) => e.tick(game, tick),
            ExecEnum::Nation(e) => e.tick(game, tick),
            ExecEnum::Nuke(e) => e.tick(game, tick),
            ExecEnum::Mirv(e) => e.tick(game, tick),
            ExecEnum::MissileSilo(e) => e.tick(game, tick),
            ExecEnum::SamLauncher(e) => e.tick(game, tick),
            ExecEnum::SamMissile(e) => e.tick(game, tick),
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
            ExecEnum::Construction(e) => e.is_active(),
            ExecEnum::City(e) => e.is_active(),
            ExecEnum::DefensePost(e) => e.is_active(),
            ExecEnum::Port(e) => e.is_active(),
            ExecEnum::Factory(e) => e.is_active(),
            ExecEnum::TrainStation(e) => e.is_active(),
            ExecEnum::Train(e) => e.is_active(),
            ExecEnum::RecomputeRailCluster(e) => e.is_active(),
            ExecEnum::UpgradeStructure(e) => e.is_active(),
            ExecEnum::AllianceRequest(e) => e.is_active(),
            ExecEnum::AllianceReject(e) => e.is_active(),
            ExecEnum::BreakAlliance(e) => e.is_active(),
            ExecEnum::AllianceExtension(e) => e.is_active(),
            ExecEnum::DonateTroops(e) => e.is_active(),
            ExecEnum::DonateGold(e) => e.is_active(),
            ExecEnum::Retreat(e) => e.is_active(),
            ExecEnum::TribeMassSpawn(e) => e.is_active(),
            ExecEnum::Player(e) => e.is_active(),
            ExecEnum::Tribe(e) => e.is_active(),
            ExecEnum::Nation(e) => e.is_active(),
            ExecEnum::Nuke(e) => e.is_active(),
            ExecEnum::Mirv(e) => e.is_active(),
            ExecEnum::MissileSilo(e) => e.is_active(),
            ExecEnum::SamLauncher(e) => e.is_active(),
            ExecEnum::SamMissile(e) => e.is_active(),
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
            ExecEnum::Construction(e) => e.active_during_spawn(),
            ExecEnum::City(e) => e.active_during_spawn(),
            ExecEnum::DefensePost(e) => e.active_during_spawn(),
            ExecEnum::Port(e) => e.active_during_spawn(),
            ExecEnum::Factory(e) => e.active_during_spawn(),
            ExecEnum::TrainStation(e) => e.active_during_spawn(),
            ExecEnum::Train(e) => e.active_during_spawn(),
            ExecEnum::RecomputeRailCluster(e) => e.active_during_spawn(),
            ExecEnum::UpgradeStructure(e) => e.active_during_spawn(),
            ExecEnum::AllianceRequest(e) => e.active_during_spawn(),
            ExecEnum::AllianceReject(e) => e.active_during_spawn(),
            ExecEnum::BreakAlliance(e) => e.active_during_spawn(),
            ExecEnum::AllianceExtension(e) => e.active_during_spawn(),
            ExecEnum::DonateTroops(e) => e.active_during_spawn(),
            ExecEnum::DonateGold(e) => e.active_during_spawn(),
            ExecEnum::Retreat(e) => e.active_during_spawn(),
            ExecEnum::TribeMassSpawn(e) => e.active_during_spawn(),
            ExecEnum::Player(e) => e.active_during_spawn(),
            ExecEnum::Tribe(e) => e.active_during_spawn(),
            ExecEnum::Nation(e) => e.active_during_spawn(),
            ExecEnum::Nuke(e) => e.active_during_spawn(),
            ExecEnum::Mirv(e) => e.active_during_spawn(),
            ExecEnum::MissileSilo(e) => e.active_during_spawn(),
            ExecEnum::SamLauncher(e) => e.active_during_spawn(),
            ExecEnum::SamMissile(e) => e.active_during_spawn(),
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

impl ExecEnum {
    /// Stable label for parity exec-order dumps.
    pub fn debug_label(&self) -> String {
        match self {
            ExecEnum::Spawn(_) => "Spawn".into(),
            ExecEnum::Attack(e) => {
                format!(
                    "Attack({}->{})",
                    e.owner_small_id(),
                    e.target_small_id()
                )
            }
            ExecEnum::SpawnTimer(_) => "SpawnTimer".into(),
            ExecEnum::WinCheck(_) => "WinCheck".into(),
            ExecEnum::NoOp(_) => "NoOp".into(),
            ExecEnum::MarkDisconnected(_) => "MarkDisconnected".into(),
            ExecEnum::TransportShip(_) => "TransportShipExecution".into(),
            ExecEnum::Construction(_) => "ConstructionExecution".into(),
            ExecEnum::City(_) => "CityExecution".into(),
            ExecEnum::DefensePost(_) => "DefensePostExecution".into(),
            ExecEnum::Port(_) => "PortExecution".into(),
            ExecEnum::Factory(_) => "FactoryExecution".into(),
            ExecEnum::TrainStation(_) => "TrainStationExecution".into(),
            ExecEnum::Train(_) => "TrainExecution".into(),
            ExecEnum::RecomputeRailCluster(_) => "RecomputeRailClusterExecution".into(),
            ExecEnum::UpgradeStructure(_) => "UpgradeStructureExecution".into(),
            ExecEnum::AllianceRequest(e) => {
                format!(
                    "AllianceRequestExecution({}->{})",
                    e.requestor_small_id(),
                    e.recipient_id()
                )
            }
            ExecEnum::AllianceReject(_) => "AllianceReject".into(),
            ExecEnum::BreakAlliance(_) => "BreakAlliance".into(),
            ExecEnum::AllianceExtension(_) => "AllianceExtension".into(),
            ExecEnum::DonateTroops(_) => "DonateTroops".into(),
            ExecEnum::DonateGold(_) => "DonateGold".into(),
            ExecEnum::Retreat(_) => "Retreat".into(),
            ExecEnum::TribeMassSpawn(_) => "TribeMassSpawn".into(),
            ExecEnum::Player(e) => format!("Player({})", e.small_id()),
            ExecEnum::Tribe(e) => format!("Tribe({})", e.small_id()),
            ExecEnum::Nation(_) => "Nation".into(),
            ExecEnum::Nuke(_) => "NukeExecution".into(),
            ExecEnum::Mirv(_) => "MirvExecution".into(),
            ExecEnum::MissileSilo(_) => "MissileSiloExecution".into(),
            ExecEnum::SamLauncher(_) => "SAMLauncherExecution".into(),
            ExecEnum::SamMissile(_) => "SAMMissileExecution".into(),
        }
    }
}
