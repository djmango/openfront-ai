//! Core wire + runtime types (TS `Schemas` / `Config` subset).

pub mod config;
pub mod nation;
pub mod schemas;
pub mod team_assignment;
pub mod terrain;

pub use config::{Config, DoomsdayClockResolved};
pub use nation::{create_nations_for_game, get_compact_map_nation_count, SpawnedNation};
pub use schemas::{
    DoomsdayClockConfig, DoomsdayClockSpeed, GameConfig, NationsConfig, PlayerTeamsConfig,
};
pub use team_assignment::{assign_teams, populate_player_teams, BOT_TEAM, HUMANS_TEAM, NATIONS_TEAM};
