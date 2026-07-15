//! Core game state + tick driver.

use crate::core::config::Config as WireConfig;
use crate::core::team_assignment::{
    assign_teams, populate_player_teams, BOT_TEAM, HUMANS_TEAM, NATIONS_TEAM,
};
use crate::execution::ordered_map::OrderedMap;
use crate::execution::ordered_tiles::OrderedTiles;
use crate::execution::{ExecEnum, Execution, HashUpdate, WinUpdate};
use crate::hash::game_hash;
use crate::map::{GameMap, SpawnArea, TileRef};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;
use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct PlayerInfo {
    pub name: String,
    pub player_type: PlayerType,
    pub client_id: Option<String>,
    pub id: String,
    pub clan_tag: Option<String>,
    pub friends: Vec<String>,
    pub team: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Unit {
    pub id: i32,
    pub unit_type: String,
    pub tile: i32,
    pub under_construction: bool,
    /// TS `UnitImpl._health`; units without configured health use the TS fallback max of 1.
    pub health: i32,
    /// TS `UnitImpl.level()`  -  defaults to 1; affects `maxTroops` for cities.
    pub level: i32,
    /// TS `UnitImpl._hasTrainStation` - set true immediately when a `TrainStationExecution`
    /// is constructed for this unit (City/Port/Factory).
    pub has_train_station: bool,
    /// TS `UnitImpl._missileTimerQueue`  -  ticks at which Missile Silo / SAM Launcher
    /// missiles were launched; used for reload cooldown tracking.
    pub missile_timer_queue: Vec<u32>,
    /// TS `UnitImpl` `targetTile` param  -  destination tile for nukes/MIRVs.
    pub target_tile: Option<TileRef>,
    /// TS `UnitImpl` `trajectory` param  -  precomputed (tile, targetable) pairs for
    /// Atom Bomb / Hydrogen Bomb / MIRV Warhead, consulted by SAM interception logic.
    pub trajectory: Vec<TileRef>,
    pub trajectory_targetable: Vec<bool>,
    pub trajectory_index: u32,
    /// TS `UnitImpl._targetedBySAM`.
    pub targeted_by_sam: bool,
    /// TS `UnitImpl._targetable`  -  live in-range check, recomputed each nuke move tick.
    /// Defaults to `true` in TS; set explicitly on nuke/MIRV-warhead spawn since other unit
    /// types never read this field.
    pub targetable: bool,
    /// Last tick a trade ship moved along shoreline and was protected from piracy.
    pub last_safe_from_pirates_tick: i32,
    /// TS `WarshipState.veterancy` - always 0 for non-warships (nothing ever increments it).
    pub veterancy: i32,
    /// TS `WarshipState.veterancyProgress` - partial progress toward the next veterancy
    /// level from transport kills/trade captures; see `Game::record_kill`'s doc comment.
    pub veterancy_progress: i32,
    /// TS `UnitImpl._deletionAt` - tick at which a voluntary `DeleteUnitExecution` may
    /// finalize this unit's deletion, or `None` if not pending. Cleared on capture
    /// (`Game::capture_unit`, TS `UnitImpl.setOwner`'s `clearPendingDeletion()`).
    pub deletion_at: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerBoundingBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[derive(Debug, Clone)]
pub struct Player {
    pub id: String,
    /// TS `player.name()` - display/tribe name, not the random id.
    pub name: String,
    pub client_id: String,
    pub small_id: u16,
    pub id_hash: i32,
    pub player_type: PlayerType,
    pub troops: i32,
    pub gold: i64,
    pub tiles_owned: i32,
    pub alive: bool,
    pub spawn_tile: Option<TileRef>,
    pub units: Vec<Unit>,
    pub border_tiles: OrderedTiles,
    pub owned_tiles: Vec<TileRef>,
    pub last_cluster_calc: u32,
    pub last_tile_change: u32,
    /// TS `PlayerImpl.largestClusterBoundingBox`; updated by PlayerExecution's
    /// cluster pass and read by island-enemy boat targeting.
    pub largest_cluster_bounding_box: Option<PlayerBoundingBox>,
    pub marked_traitor_tick: i32,
    /// TS `PlayerImpl.markedDoomsdayClockTick` - tick this player's side first
    /// dropped below the Doomsday Clock bar, or -1 if not currently flagged.
    pub marked_doomsday_clock_tick: i32,
    /// TS `PlayerImpl.relations` is a `Map<Player, number>`, which iterates in insertion
    /// order (re-setting an existing key's value does not move it). `allRelationsSorted()`
    /// relies on that insertion order as the tie-break after its stable sort-by-value, so a
    /// plain `HashMap` here (whose iteration order is randomly per-process-seeded) makes
    /// tie-broken attack-target selection genuinely nondeterministic across runs.
    pub relations: OrderedMap<u16, f64>,
    /// TS `PlayerImpl.embargoes` (`Map<PlayerID, Embargo>`) - insertion order doesn't affect
    /// any RNG/numeric outcome (only independent per-target lookups/expiry), but `OrderedMap`
    /// is used anyway for consistency with `relations` above.
    pub embargoes: OrderedMap<u16, EmbargoState>,
    /// TS `PlayerImpl.lastEmbargoAllTick` - defaults to -1 (never).
    pub last_embargo_all_tick: i32,
    pub is_disconnected: bool,
    /// TS `numUnitsConstructed`  -  incremented on build and upgrade.
    pub units_constructed: HashMap<String, u32>,
    /// TS `PlayerImpl._pseudo_random`  -  attack IDs and nation RNG streams.
    pub id_prng: PseudoRandom,
    /// TS `PlayerImpl._team`  -  colored team in Team mode.
    pub team: Option<String>,
    /// TS `PlayerImpl._outgoingAttacks`  -  land attack ids (registry keys).
    pub outgoing_land_attacks: Vec<String>,
    /// TS `PlayerImpl._incomingAttacks`  -  land attack ids targeting this player.
    pub incoming_land_attacks: Vec<String>,
    /// TS `StatsImpl` `attacks[ATTACK_INDEX_SENT]`  -  net troops sent in attacks
    /// (increased on attack init, decreased on genuine order-retreat cancels).
    /// Used only for `GameImpl.conquerPlayer`'s "never attacked" gold-skip check.
    pub attacks_sent_net: i64,
    /// TS `PlayerImpl.targets_`  -  (target small_id, tick painted). Read through
    /// `Game::player_targets` (targetDuration filter) / `Game::can_target`
    /// (targetCooldown filter); entries are never removed, matching TS.
    pub target_marks: Vec<(u16, u32)>,
    /// TS `PlayerImpl.lastDeleteUnitTick` - defaults to -1 (never deleted).
    pub last_delete_unit_tick: i32,
    /// TS `PlayerImpl.outgoingEmojis_` - `(recipient small_id | AllPlayers, createdAt)`.
    /// `None` recipient matches TS `AllPlayers`. Used by `canSendEmoji` cooldown.
    pub outgoing_emoji_sends: Vec<(Option<u16>, u32)>,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            client_id: String::new(),
            small_id: 0,
            id_hash: 0,
            player_type: PlayerType::Human,
            troops: 0,
            gold: 0,
            tiles_owned: 0,
            alive: true,
            spawn_tile: None,
            units: vec![],
            border_tiles: OrderedTiles::default(),
            owned_tiles: Vec::new(),
            last_cluster_calc: 0,
            last_tile_change: 0,
            largest_cluster_bounding_box: None,
            marked_traitor_tick: -1,
            marked_doomsday_clock_tick: -1,
            relations: OrderedMap::new(),
            embargoes: OrderedMap::new(),
            last_embargo_all_tick: -1,
            is_disconnected: false,
            units_constructed: HashMap::new(),
            id_prng: PseudoRandom::new(0),
            team: None,
            outgoing_land_attacks: Vec::new(),
            incoming_land_attacks: Vec::new(),
            attacks_sent_net: 0,
            target_marks: Vec::new(),
            last_delete_unit_tick: -1,
            outgoing_emoji_sends: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    Bot,
    Nation,
}

/// TS `Relation` enum ordering used by nation alliance logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Relation {
    Hostile = 0,
    Distrustful = 1,
    Neutral = 2,
    Friendly = 3,
}

#[derive(Debug, Clone)]
pub struct IncomingAttack {
    pub attacker_small_id: u16,
    pub troops: f64,
}

#[derive(Debug, Clone)]
pub struct IncomingAllianceRequest {
    pub requestor_small_id: u16,
    pub created_at: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllianceRequestStatus {
    Pending,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct AllianceRequestState {
    pub requestor_small_id: u16,
    pub recipient_small_id: u16,
    pub created_at: u32,
    pub status: AllianceRequestStatus,
}

#[derive(Debug, Clone)]
pub struct AllianceState {
    pub id: u32,
    pub requestor_small_id: u16,
    pub recipient_small_id: u16,
    pub created_at: u32,
    pub expires_at: u32,
    pub extension_requested_requestor: bool,
    pub extension_requested_recipient: bool,
}

#[derive(Debug, Clone)]
pub struct AllianceExtensionCandidate {
    pub other_small_id: u16,
}

/// TS `Embargo` (`Game.ts`) - `createdAt`/`isTemporary` stored per-target on `Player.embargoes`.
#[derive(Debug, Clone, Copy)]
pub struct EmbargoState {
    pub created_at: u32,
    pub is_temporary: bool,
}

#[derive(Debug, Clone)]
pub struct GameConfig {
    pub game_map: String,
    pub bots: u32,
    pub num_spawn_phase_turns: u32,
    pub random_spawn: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            game_map: "Onion".into(),
            bots: 100,
            num_spawn_phase_turns: 300,
            random_spawn: false,
        }
    }
}

pub struct Game {
    pub game_id: String,
    pub config: GameConfig,
    pub wire: WireConfig,
    pub map: GameMap,
    pub mini_map: GameMap,
    pub rng: PseudoRandom,
    players: Vec<Player>,
    players_by_client: HashMap<String, usize>,
    players_by_id: HashMap<String, usize>,
    next_player_id: u16,
    execs: Vec<ExecEnum>,
    uninit: Vec<ExecEnum>,
    ticks: u32,
    spawn_phase: bool,
    spawn_end_tick: Option<u32>,
    pub winner: Option<String>,
    pub water_component: Vec<u32>,
    pub(crate) bfs: crate::water::BfsScratch,
    pub(crate) water_astar: crate::water::WaterAstarScratch,
    pub(crate) mini_water_astar: crate::water::WaterAstarScratch,
    pub(crate) mini_water_hpa: Option<crate::water_hpa::WaterHierarchical>,
    water_graph_version: u32,
    path_buf: Vec<TileRef>,
    next_unit_id: i32,
    /// TS `GameImpl.unitGrid` - spatial index for nearbyUnits order.
    unit_grid: crate::unit_grid::UnitGrid,
    pub hash_enabled: bool,
    pub tribe_batch: crate::bot::TribeBatch,
    pub nation_batch: crate::execution::NationBatch,
    pub alliance_requests: Vec<AllianceRequestState>,
    pub alliances: Vec<AllianceState>,
    pub next_alliance_id: u32,
    /// Same-tick inits not yet appended to `execs` (TS `outgoingAttacks()` batch ordering).
    /// SAFETY (`unsafe impl Send` below): only ever `Some` inside the
    /// dynamic extent of `execute_next_tick` (points at a stack local there,
    /// set/cleared in the same call), so a `Game` moved between threads
    /// between ticks never carries a live pointer.
    init_merge_batch: Option<*mut Vec<ExecEnum>>,
    /// TS `GameImpl.playerTeams`  -  colored teams for Team mode spawn areas.
    pub player_teams: Vec<String>,
    /// TS `GameImpl._teamGameSpawnAreas`.
    pub team_spawn_areas: Option<HashMap<String, Vec<SpawnArea>>>,
    /// TS `SharedWaterCache` port: rebuilt at most once every 30 ticks (`None` = never built).
    /// `RefCell` so callers deep in read-only structure-planning code (mirroring TS's lazy
    /// `SharedWaterCache.get`) don't need to thread `&mut Game` through the whole call chain.
    pub(crate) shared_water_cache_tick: std::cell::RefCell<Option<i64>>,
    pub(crate) shared_water_cache:
        std::cell::RefCell<HashMap<u16, Option<std::collections::HashSet<u32>>>>,
    /// TS `GameImpl._railNetwork` (`RailNetworkImpl`) - stations, railroads, clusters.
    pub rail_network: crate::rail::RailNetwork,
    /// TS `StatsImpl._numMirvLaunched`  -  global count feeding escalating MIRV cost.
    pub mirvs_launched: u32,
    /// TS `NationMIRVBehavior.recentMirvTargets` (`static`, shared across every nation's
    /// behavior instance so multiple nations don't pile-on the same freshly-hit target) -
    /// keyed by target `small_id` rather than `PlayerID`, value is the tick of the last hit.
    pub recent_mirv_targets: HashMap<u16, u32>,
    /// TS `UnitImpl._destroyer`/`_wasDestroyedByEnemy`, scoped to transport ships (the only
    /// unit type `NationWarshipBehavior.trackShipsAndRetaliate` queries `destroyer()` for -
    /// see `warship_ai.rs`). Native fully removes a destroyed unit from `Player.units` (no
    /// TS-style inert "deleted but still inspectable" tombstone object), so there is nothing
    /// left to call `.destroyer()` on after the fact; this is a minimal side channel recording
    /// `(unit_id, victim_owner, destroyer)` at the two real "enemy destroys a transport ship"
    /// call sites (`ShellExecution::hit_target`, `NukeExecution`'s blast-radius deletion loop),
    /// consumed (and removed) by the tracker the tick it notices the ship is gone. Entries are
    /// removed on read (`take_transport_kill`) rather than left to grow, and are otherwise
    /// harmless if never read (e.g. a bystander's own transport ship, which nobody is tracking).
    transport_kills: Vec<(i32, u16, u16, TileRef)>,
}

impl Default for Game {
    fn default() -> Self {
        let wire_cfg = crate::core::schemas::GameConfig {
            game_map: "Onion".into(),
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
            spawn_immunity_duration: None,
            starting_gold: None,
            gold_multiplier: None,
            max_timer_value: None,
            ranked_type: None,
        };
        Self {
            game_id: String::new(),
            config: GameConfig::default(),
            wire: WireConfig::new(wire_cfg, false),
            map: GameMap::from_terrain_bytes(
                &crate::map::MapMeta {
                    width: 1,
                    height: 1,
                    num_land_tiles: 0,
                },
                &[0],
            )
            .unwrap(),
            mini_map: GameMap::from_terrain_bytes(
                &crate::map::MapMeta {
                    width: 1,
                    height: 1,
                    num_land_tiles: 0,
                },
                &[0],
            )
            .unwrap(),
            rng: PseudoRandom::new(1),
            players: vec![],
            players_by_client: HashMap::new(),
            players_by_id: HashMap::new(),
            next_player_id: 1,
            execs: vec![],
            uninit: vec![],
            ticks: 0,
            spawn_phase: true,
            spawn_end_tick: None,
            winner: None,
            water_component: vec![],
            bfs: crate::water::BfsScratch::new(1),
            water_astar: crate::water::WaterAstarScratch::new(1),
            mini_water_astar: crate::water::WaterAstarScratch::new(1),
            mini_water_hpa: None,
            water_graph_version: 0,
            path_buf: Vec::new(),
            next_unit_id: 1,
            unit_grid: crate::unit_grid::UnitGrid::new(1, 1),
            hash_enabled: true,
            tribe_batch: crate::bot::TribeBatch::new(),
            nation_batch: crate::execution::NationBatch::new(),
            alliance_requests: Vec::new(),
            alliances: Vec::new(),
            next_alliance_id: 1,
            init_merge_batch: None,
            player_teams: Vec::new(),
            team_spawn_areas: None,
            shared_water_cache_tick: std::cell::RefCell::new(None),
            shared_water_cache: std::cell::RefCell::new(HashMap::new()),
            rail_network: crate::rail::RailNetwork::default(),
            mirvs_launched: 0,
            recent_mirv_targets: HashMap::new(),
            transport_kills: Vec::new(),
        }
    }
}

// SAFETY: see `init_merge_batch` - the only non-Send field is a raw pointer
// that is guaranteed null outside `execute_next_tick`'s own stack frame.
unsafe impl Send for Game {}

impl Game {
    pub fn new(
        game_id: String,
        config: GameConfig,
        wire: WireConfig,
        map: GameMap,
        mini_map: GameMap,
        team_spawn_areas: Option<HashMap<String, Vec<SpawnArea>>>,
    ) -> Self {
        let seed = simple_hash(&game_id);
        let map_width = map.width;
        let map_height = map.height;
        let tile_count = (map_width * map_height) as usize;
        let mini_count = (mini_map.width * mini_map.height) as usize;
        let water_component = crate::water::build_water_components(&map);
        let mini_water_hpa = crate::water_hpa::WaterHierarchical::new(&mini_map, true);
        let water_graph_version = 0u32;
        let game_mode = wire.game_config().game_mode.clone();
        let player_teams_cfg = wire.game_config().player_teams.clone();
        Self {
            game_id,
            config,
            wire,
            map,
            mini_map,
            rng: PseudoRandom::new(seed),
            water_component,
            bfs: crate::water::BfsScratch::new(tile_count),
            water_astar: crate::water::WaterAstarScratch::new(tile_count),
            mini_water_astar: crate::water::WaterAstarScratch::new(mini_count),
            mini_water_hpa: Some(mini_water_hpa),
            water_graph_version,
            path_buf: Vec::with_capacity(256),
            unit_grid: crate::unit_grid::UnitGrid::new(map_width, map_height),
            tribe_batch: crate::bot::TribeBatch::new(),
            nation_batch: crate::execution::NationBatch::new(),
            team_spawn_areas,
            player_teams: populate_player_teams(&game_mode, player_teams_cfg.as_ref(), 0, 0),
            ..Default::default()
        }
    }

    /// Rebuild the UnitGrid for the current map size. Needed when tests swap
    /// `map` after `Game::default()` (which starts with a 1x1 grid).
    pub(crate) fn reinit_unit_grid(&mut self) {
        self.unit_grid = crate::unit_grid::UnitGrid::new(self.map.width, self.map.height);
    }

    pub fn init_player_teams(&mut self, num_humans: usize, num_nations: usize) {
        let game_mode = self.wire.game_config().game_mode.clone();
        let player_teams_cfg = self.wire.game_config().player_teams.clone();
        self.player_teams = populate_player_teams(
            &game_mode,
            player_teams_cfg.as_ref(),
            num_humans,
            num_nations,
        );
    }

    /// TS `GameImpl.teamSpawnArea`.
    pub fn team_spawn_area(&self, team: &str) -> Option<&SpawnArea> {
        let areas = self.team_spawn_areas.as_ref()?;
        let num_teams = self.player_teams.len();
        let team_areas = areas.get(&num_teams.to_string())?;
        let team_index = self.player_teams.iter().position(|t| t == team)?;
        team_areas.get(team_index)
    }

    /// TS `GameImpl.maybeAssignTeam` - the on-the-fly per-player fallback used when a
    /// player joins without an explicit team (`addPlayer(info, team = null)`), as opposed
    /// to `assign_teams_for_players`'s batch clan/friend-aware grouping used at game start.
    /// Was previously a stub that always returned `None` for non-Bot player types even
    /// with a non-empty `player_teams` list - every human/nation joining this way ended up
    /// on no team at all in Team mode instead of TS's `playerTeams[hash(id) % len]` spread.
    fn maybe_assign_team(&self, player_type: PlayerType, id_hash: i32) -> Option<String> {
        if self.wire.game_config().game_mode != "Team" {
            return None;
        }
        if player_type == PlayerType::Bot {
            return Some(BOT_TEAM.to_string());
        }
        let teams = &self.player_teams;
        if teams.is_empty() {
            return None;
        }
        let idx = (id_hash as usize) % teams.len();
        Some(teams[idx].clone())
    }

    pub fn assign_teams_for_players(&mut self, players: &mut [PlayerInfo]) {
        if self.wire.game_config().game_mode != "Team" {
            return;
        }
        if self
            .wire
            .game_config()
            .player_teams
            .as_ref()
            .is_some_and(|c| c.is_humans_vs_nations())
        {
            for p in players.iter_mut() {
                p.team = Some(match p.player_type {
                    PlayerType::Nation => NATIONS_TEAM.into(),
                    _ => HUMANS_TEAM.into(),
                });
            }
            return;
        }
        let assignments = assign_teams(players, &self.player_teams);
        for p in players.iter_mut() {
            if let Some(team) = assignments.0.get(&p.id) {
                if team == "kicked" {
                    p.team = None;
                } else {
                    p.team = Some(team.clone());
                }
            }
        }
    }

    /// Player indices in TS `assignTeams` Map insertion order.
    pub fn assign_teams_insertion_order(&self, players: &[PlayerInfo]) -> Vec<usize> {
        if self.wire.game_config().game_mode != "Team" {
            return (0..players.len()).collect();
        }
        if self
            .wire
            .game_config()
            .player_teams
            .as_ref()
            .is_some_and(|c| c.is_humans_vs_nations())
        {
            let mut nation_idxs = Vec::new();
            let mut other_idxs = Vec::new();
            for (i, p) in players.iter().enumerate() {
                if p.player_type == PlayerType::Nation {
                    nation_idxs.push(i);
                } else {
                    other_idxs.push(i);
                }
            }
            return other_idxs.into_iter().chain(nation_idxs).collect();
        }
        assign_teams(players, &self.player_teams).1
    }

    pub fn register_tribe(&mut self, small_id: u16, player_id: &str) {
        self.tribe_batch.register(small_id, player_id);
    }

    pub fn get_water_component(&self, tile: TileRef) -> Option<u32> {
        let hpa = self.mini_water_hpa.as_ref()?;
        let mini_x = self.map.x(tile) / 2;
        let mini_y = self.map.y(tile) / 2;
        let mini_tile = self.mini_map.ref_xy(mini_x, mini_y);

        if self.mini_map.is_water(mini_tile) {
            return Some(hpa.graph.get_component_id(mini_tile));
        }
        // TS `WaterManager.getWaterComponent`  -  miniMap.neighbors order (N,S,W,E).
        let mut one_hop = [TileRef::MAX; 4];
        let n1 = self.mini_map.neighbors4_ts(mini_tile, &mut one_hop);
        for i in 0..n1 {
            if self.mini_map.is_water(one_hop[i]) {
                return Some(hpa.graph.get_component_id(one_hop[i]));
            }
        }
        for i in 0..n1 {
            let mut two_hop = [TileRef::MAX; 4];
            let n2 = self.mini_map.neighbors4_ts(one_hop[i], &mut two_hop);
            for j in 0..n2 {
                if self.mini_map.is_water(two_hop[j]) {
                    return Some(hpa.graph.get_component_id(two_hop[j]));
                }
            }
        }
        None
    }

    /// TS `WaterManager.getWaterComponentSize` - component size is measured on
    /// the minimap, then scaled by four to approximate full-map tile count.
    pub fn get_water_component_size(&self, component_id: u32) -> Option<u32> {
        let hpa = self.mini_water_hpa.as_ref()?;
        Some(hpa.graph.get_component_size(component_id) * 4)
    }

    /// TS `WaterManager.hasWaterComponent` - like `get_water_component` but a
    /// membership test (used by trade-ship routing to check whether a
    /// candidate port shares any of a source port's adjacent water bodies).
    pub fn has_water_component(&self, tile: TileRef, component: u32) -> bool {
        let Some(hpa) = self.mini_water_hpa.as_ref() else {
            // TS: permissive fallback for tests with disableNavMesh.
            return true;
        };
        let mini_x = self.map.x(tile) / 2;
        let mini_y = self.map.y(tile) / 2;
        let mini_tile = self.mini_map.ref_xy(mini_x, mini_y);

        if self.mini_map.is_water(mini_tile) && hpa.graph.get_component_id(mini_tile) == component {
            return true;
        }
        let mut one_hop = [TileRef::MAX; 4];
        let n1 = self.mini_map.neighbors4_ts(mini_tile, &mut one_hop);
        for i in 0..n1 {
            if self.mini_map.is_water(one_hop[i])
                && hpa.graph.get_component_id(one_hop[i]) == component
            {
                return true;
            }
        }
        for i in 0..n1 {
            let mut two_hop = [TileRef::MAX; 4];
            let n2 = self.mini_map.neighbors4_ts(one_hop[i], &mut two_hop);
            for j in 0..n2 {
                if self.mini_map.is_water(two_hop[j])
                    && hpa.graph.get_component_id(two_hop[j]) == component
                {
                    return true;
                }
            }
        }
        false
    }

    pub fn plan_water_path(&mut self, from: TileRef, to: TileRef) -> bool {
        crate::water::transport_path_into(
            &self.map,
            &self.mini_map,
            &mut self.mini_water_astar,
            self.mini_water_hpa.as_mut(),
            from,
            to,
            &mut self.path_buf,
        )
    }

    pub fn plan_water_path_multi(&mut self, froms: &[TileRef], to: TileRef) -> bool {
        crate::water::transport_path_multi_into(
            &self.map,
            &self.mini_map,
            &mut self.mini_water_astar,
            self.mini_water_hpa.as_mut(),
            froms,
            to,
            &mut self.path_buf,
        )
    }

    pub fn planned_water_path(&self) -> &[TileRef] {
        &self.path_buf
    }

    pub fn width(&self) -> u32 {
        self.map.width
    }

    pub fn height(&self) -> u32 {
        self.map.height
    }

    pub fn ticks(&self) -> u32 {
        self.ticks
    }

    pub fn in_spawn_phase(&self) -> bool {
        self.spawn_phase
    }

    pub fn end_spawn_phase(&mut self) {
        self.spawn_phase = false;
        self.spawn_end_tick = Some(self.ticks);
    }

    pub fn spawn_end_tick(&self) -> Option<u32> {
        self.spawn_end_tick
    }

    pub fn tile_state_buffer(&self) -> &[u16] {
        self.map.tile_state_buffer()
    }

    pub fn terrain_byte(&self, t: TileRef) -> u8 {
        self.map.terrain_byte(t)
    }

    pub fn ref_xy(&self, x: u32, y: u32) -> TileRef {
        self.map.ref_xy(x, y)
    }

    pub fn x(&self, t: TileRef) -> u32 {
        self.map.x(t)
    }

    pub fn y(&self, t: TileRef) -> u32 {
        self.map.y(t)
    }

    pub fn is_land(&self, t: TileRef) -> bool {
        self.map.is_land(t)
    }

    pub fn is_water(&self, t: TileRef) -> bool {
        self.map.is_water(t)
    }

    pub fn is_shore(&self, t: TileRef) -> bool {
        self.map.is_shore(t)
    }

    pub fn has_fallout(&self, t: TileRef) -> bool {
        self.map.has_fallout(t)
    }

    pub fn num_tiles_with_fallout(&self) -> u32 {
        self.map.num_tiles_with_fallout()
    }

    /// TS `Game.nearbyUnits(tile, range, DefensePost)` - through the unit grid,
    /// including its exact cell-window semantics and active/under-construction filters.
    pub fn has_defense_post_nearby(&self, tile: TileRef, owner_small_id: u16) -> bool {
        use crate::core::schemas::unit_type;
        let range = self.wire.defense_post_range();
        self.nearby_structures_any(tile, range, &[unit_type::DEFENSE_POST])
            .into_iter()
            .any(|(owner, ..)| owner == owner_small_id)
    }

    /// TS `Config.attackLogic(gm, attackTroops, attacker, defender, tile)`.
    pub fn attack_logic_at_tile(
        &self,
        attack_troops: f64,
        attacker_small_id: u16,
        defender_small_id: u16,
        tile: TileRef,
        defender_is_player: bool,
    ) -> (f64, f64, f64) {
        use crate::map::TerrainType;
        use crate::util::{sigmoid, within};

        let terrain = self.terrain_type(tile);
        let (mut mag, mut speed) = match terrain {
            TerrainType::Plains => (80.0, 16.5),
            TerrainType::Highland => (100.0, 20.0),
            TerrainType::Mountain => (120.0, 25.0),
            _ => (80.0, 16.5),
        };

        let attacker = self.player_by_small_id(attacker_small_id);
        let attacker_type = attacker.map(|p| p.player_type).unwrap_or(PlayerType::Human);
        let attacker_tiles = attacker.map(|p| p.tiles_owned).unwrap_or(0);

        if defender_is_player {
            if self.has_defense_post_nearby(tile, defender_small_id) {
                mag *= self.wire.defense_post_defense_bonus();
                speed *= self.wire.defense_post_speed_bonus();
            }
        }

        if self.has_fallout(tile) {
            let fallout_ratio =
                self.num_tiles_with_fallout() as f64 / self.num_land_tiles().max(1) as f64;
            let modifier = self.wire.fallout_defense_modifier(fallout_ratio);
            mag *= modifier;
            speed *= modifier;
        }

        if defender_is_player {
            let defender = self.player_by_small_id(defender_small_id);
            let defender_type = defender.map(|p| p.player_type);
            let defender_troops = defender.map(|p| p.troops).unwrap_or(0);
            let defender_tiles = defender.map(|p| p.tiles_owned).unwrap_or(1).max(1);
            let defender_disconnected = defender.is_some_and(|p| p.is_disconnected);

            if defender_disconnected
                && self.players_on_same_team(attacker_small_id, defender_small_id)
            {
                mag = 0.0;
            }

            if matches!(attacker_type, PlayerType::Human | PlayerType::Nation)
                && defender_type == Some(PlayerType::Bot)
            {
                mag *= 0.7;
            }

            const DEFENSE_DEBUFF_MIDPOINT: f64 = 150_000.0;
            const DEFENSE_DEBUFF_DECAY_RATE: f64 = std::f64::consts::LN_2 / 50_000.0;

            let defense_sig = 1.0
                - sigmoid(
                    defender_tiles as f64,
                    DEFENSE_DEBUFF_DECAY_RATE,
                    DEFENSE_DEBUFF_MIDPOINT,
                );
            let large_defender_speed_debuff = 0.7 + 0.3 * defense_sig;
            let large_defender_attack_debuff = 0.7 + 0.3 * defense_sig;

            let mut large_attack_bonus = 1.0;
            if attacker_tiles > 100_000 {
                large_attack_bonus = (100_000.0 / attacker_tiles as f64).sqrt().powf(0.7);
            }
            let mut large_attacker_speed_bonus = 1.0;
            if attacker_tiles > 100_000 {
                large_attacker_speed_bonus = (100_000.0 / attacker_tiles as f64).powf(0.6);
            }

            let defender_troop_loss = defender_troops as f64 / defender_tiles as f64;
            let traitor_mod = if self.is_traitor(defender_small_id) {
                self.wire.traitor_defense_debuff()
            } else {
                1.0
            };
            let current_attacker_loss = within(defender_troops as f64 / attack_troops, 0.6, 2.0)
                * mag
                * 0.8
                * large_defender_attack_debuff
                * large_attack_bonus
                * traitor_mod;
            let alt_attacker_loss = 1.3 * defender_troop_loss * (mag / 100.0) * traitor_mod;
            let attacker_troop_loss = 0.6 * current_attacker_loss + 0.4 * alt_attacker_loss;

            let traitor_speed = if self.is_traitor(defender_small_id) {
                self.wire.traitor_speed_debuff()
            } else {
                1.0
            };
            let tiles_per_tick = within(defender_troops as f64 / (5.0 * attack_troops), 0.2, 1.5)
                * speed
                * large_defender_speed_debuff
                * large_attacker_speed_bonus
                * traitor_speed;

            (attacker_troop_loss, defender_troop_loss, tiles_per_tick)
        } else {
            let attacker_troop_loss = if attacker_type == PlayerType::Bot {
                mag / 10.0
            } else {
                mag / 5.0
            };
            let tiles_per_tick = within((2000.0 * speed.max(10.0)) / attack_troops, 5.0, 100.0);
            (attacker_troop_loss, 0.0, tiles_per_tick)
        }
    }

    pub fn has_owner(&self, t: TileRef) -> bool {
        self.map.has_owner(t)
    }

    pub fn is_border(&self, t: TileRef) -> bool {
        self.map.is_border(t)
    }

    pub fn is_impassable(&self, t: TileRef) -> bool {
        self.map.is_impassable(t)
    }

    pub fn is_valid_coord(&self, x: i32, y: i32) -> bool {
        self.map.is_valid_coord(x, y)
    }

    pub fn manhattan_dist(&self, a: TileRef, b: TileRef) -> u32 {
        self.map.manhattan_dist(a, b)
    }

    pub fn terrain_type(&self, t: TileRef) -> crate::map::TerrainType {
        self.map.terrain_type(t)
    }

    pub fn has_player(&self, id: &str) -> bool {
        self.players_by_id.contains_key(id)
    }

    pub fn add_from_info(&mut self, info: &PlayerInfo) -> u16 {
        let troops = self.wire.start_manpower(info.player_type);
        let gold = self.wire.starting_gold(info.player_type);
        let small_id = self.next_player_id;
        self.next_player_id += 1;
        let client_id = info.client_id.clone().unwrap_or_default();
        let id_hash = simple_hash(&info.id);
        let team = info
            .team
            .clone()
            .or_else(|| self.maybe_assign_team(info.player_type, id_hash));
        let idx = self.players.len();
        self.players.push(Player {
            id: info.id.clone(),
            name: info.name.clone(),
            client_id: client_id.clone(),
            small_id,
            id_hash,
            player_type: info.player_type,
            troops,
            gold,
            border_tiles: OrderedTiles::default(),
            id_prng: PseudoRandom::new(id_hash),
            team,
            ..Default::default()
        });
        if !client_id.is_empty() {
            self.players_by_client.insert(client_id, idx);
        }
        self.players_by_id.insert(info.id.clone(), idx);
        small_id
    }

    pub fn add_player(&mut self, p: Player) {
        let idx = self.players.len();
        if !p.client_id.is_empty() {
            self.players_by_client.insert(p.client_id.clone(), idx);
        }
        self.players_by_id.insert(p.id.clone(), idx);
        self.players.push(p);
    }

    pub fn player_by_client_id(&self, cid: &str) -> Option<&Player> {
        self.players_by_client.get(cid).map(|&i| &self.players[i])
    }

    pub fn player_by_client_id_mut(&mut self, cid: &str) -> Option<&mut Player> {
        let i = *self.players_by_client.get(cid)?;
        Some(&mut self.players[i])
    }

    pub fn player_by_small_id(&self, sid: u16) -> Option<&Player> {
        self.players.iter().find(|p| p.small_id == sid)
    }

    pub fn player_by_small_id_mut(&mut self, sid: u16) -> Option<&mut Player> {
        self.players.iter_mut().find(|p| p.small_id == sid)
    }

    pub fn player_by_id(&self, id: &str) -> Option<&Player> {
        self.players_by_id.get(id).map(|&i| &self.players[i])
    }

    pub fn has_spawned(&self, small_id: u16) -> bool {
        self.player_by_small_id(small_id)
            .is_some_and(|p| p.spawn_tile.is_some())
    }

    pub fn spawn_tile_of(&self, small_id: u16) -> Option<TileRef> {
        self.player_by_small_id(small_id).and_then(|p| p.spawn_tile)
    }

    pub fn set_spawn_tile(&mut self, small_id: u16, tile: TileRef) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.spawn_tile = Some(tile);
        }
    }

    pub fn relinquish_player_tiles(&mut self, small_id: u16) {
        let tiles: Vec<TileRef> = self
            .player_by_small_id_mut(small_id)
            .map(|p| std::mem::take(&mut p.owned_tiles))
            .unwrap_or_default();
        for t in tiles {
            if self.map.owner_id(t) == small_id {
                self.map.set_owner_id(t, 0);
            }
            self.refresh_borders_around(t);
        }
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.tiles_owned = 0;
            p.border_tiles.clear();
            p.owned_tiles.clear();
            // TS `isAlive()` is `_tiles.size > 0` — keep the sticky flag in sync
            // so mid-tick readers match (see conquer_one / relinquish_tile).
            p.alive = false;
        }
    }

    /// TS `GameImpl.relinquish(tile)`  -  drop a single tile back to Terra Nullius
    /// (used by `NukeExecution` detonation). No-op if the tile is unowned.
    pub fn relinquish_tile(&mut self, tile: TileRef) {
        let owner = self.map.owner_id(tile);
        if owner == 0 {
            return;
        }
        let tick = self.ticks;
        if let Some(p) = self.player_by_small_id_mut(owner) {
            p.tiles_owned = (p.tiles_owned - 1).max(0);
            p.border_tiles.remove(tile);
            p.owned_tiles.retain(|&t| t != tile);
            p.last_tile_change = tick;
            if p.tiles_owned == 0 {
                // Match TS `Player.isAlive()` (`_tiles.size > 0`) immediately —
                // do not wait for the next `PlayerExecution` tick.
                p.alive = false;
            }
        }
        self.map.set_owner_id(tile, 0);
        self.refresh_borders_around(tile);
    }

    pub fn players_in_order(&self) -> &[Player] {
        &self.players
    }

    pub fn players_alive(&self) -> impl Iterator<Item = &Player> {
        self.players.iter().filter(|p| p.alive)
    }

    pub fn all_players(&self) -> &[Player] {
        &self.players
    }

    pub fn rebuild_all_borders(&mut self) {
        for p in &mut self.players {
            p.border_tiles.clear();
        }
        let n = (self.map.width * self.map.height) as usize;
        for t in 0..n {
            let t = t as TileRef;
            if self.has_owner(t) {
                self.update_border_status(t);
            }
        }
    }

    fn collect_border_tiles(&self, small_id: u16) -> Vec<TileRef> {
        self.border_tiles_of(small_id)
            .map(|tiles| tiles.as_slice().to_vec())
            .unwrap_or_default()
    }

    pub fn border_tiles_for(&self, small_id: u16) -> Vec<TileRef> {
        self.collect_border_tiles(small_id)
    }

    pub fn for_each_border_tile<F: FnMut(TileRef)>(&self, small_id: u16, mut f: F) {
        let tiles = self.collect_border_tiles(small_id);
        for t in tiles {
            f(t);
        }
    }

    pub fn terra_nullius_id(&self) -> u16 {
        0
    }

    pub fn border_tiles_of(&self, small_id: u16) -> Option<&OrderedTiles> {
        self.player_by_small_id(small_id).map(|p| &p.border_tiles)
    }

    fn update_border_status(&mut self, tile: TileRef) {
        if !self.has_owner(tile) {
            return;
        }
        let owner = self.map.owner_id(tile);
        let is_br = self.is_border(tile);
        if let Some(p) = self.player_by_small_id_mut(owner) {
            if is_br {
                p.border_tiles.insert(tile);
            } else {
                p.border_tiles.remove(tile);
            }
        }
    }

    fn refresh_borders_around(&mut self, tile: TileRef) {
        let mut neighbors = [TileRef::MAX; 4];
        let mut n = 0usize;
        self.map.for_each_neighbor4(tile, |t| {
            if n < 4 {
                neighbors[n] = t;
                n += 1;
            }
        });
        self.update_border_status(tile);
        for i in 0..n {
            self.update_border_status(neighbors[i]);
        }
    }

    pub fn remove_troops(&mut self, small_id: u16, amount: f64) -> i32 {
        // TS `PlayerImpl.removeTroops`: non-positive amounts are a no-op.
        if amount <= 0.0 {
            return 0;
        }
        let Some(p) = self.player_by_small_id_mut(small_id) else {
            return 0;
        };
        let to_remove = p.troops.min(crate::util::to_int(amount));
        p.troops -= to_remove;
        to_remove
    }

    /// TS `PlayerImpl.addTroops`. Negative amounts go through `removeTroops(-amount)`,
    /// so a fractional over-max income pullback of e.g. `-6360.28` removes `floor(6360.28)=6360`
    /// troops rather than applying `floor(-6360.28)=-6361` (which desynced soft troop counts
    /// whenever a player was over `maxTroops`).
    pub fn add_troops(&mut self, small_id: u16, amount: f64) {
        if amount < 0.0 {
            self.remove_troops(small_id, -amount);
            return;
        }
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.troops += crate::util::to_int(amount);
        }
    }

    pub fn conquer(&mut self, small_id: u16, tile: TileRef) {
        self.conquer_one(small_id, tile, true);
    }

    /// TS `GameImpl.conquerPlayer`  -  ship transfer on disconnected teammate wipe,
    /// plus gold transfer from conquered to conqueror (skipped for humans who
    /// never sent an attack, e.g. AFK spawn-camped players).
    pub fn conquer_player(&mut self, conqueror_small_id: u16, conquered_small_id: u16) {
        let Some((disconnected, conquered_team)) = self
            .player_by_small_id(conquered_small_id)
            .map(|p| (p.is_disconnected, p.team.clone()))
        else {
            return;
        };
        let conqueror_team = self
            .player_by_small_id(conqueror_small_id)
            .and_then(|p| p.team.clone());
        let same_team = conqueror_team.is_some() && conqueror_team == conquered_team;

        if disconnected && same_team {
            let ships: Vec<i32> = self
                .player_by_small_id(conquered_small_id)
                .map(|p| {
                    p.units
                        .iter()
                        .filter(|u| {
                            u.unit_type == crate::core::schemas::unit_type::WARSHIP
                                || u.unit_type == crate::core::schemas::unit_type::TRANSPORT
                        })
                        .map(|u| u.id)
                        .collect()
                })
                .unwrap_or_default();
            for unit_id in ships {
                self.capture_unit(conquered_small_id, conqueror_small_id, unit_id);
            }
        }

        let Some((conquered_gold, conquered_type, attacks_sent_net)) = self
            .player_by_small_id(conquered_small_id)
            .map(|p| (p.gold, p.player_type, p.attacks_sent_net))
        else {
            return;
        };
        let skip_gold_transfer = attacks_sent_net == 0 && conquered_type == PlayerType::Human;
        if skip_gold_transfer {
            return;
        }
        let gold_captured = match conquered_type {
            PlayerType::Bot | PlayerType::Nation => conquered_gold,
            PlayerType::Human => conquered_gold / 2,
        };
        if let Some(p) = self.player_by_small_id_mut(conqueror_small_id) {
            p.gold += gold_captured;
        }
        if let Some(p) = self.player_by_small_id_mut(conquered_small_id) {
            p.gold -= conquered_gold;
        }
    }

    /// TS `StatsImpl.attack`/`attackCancel` (SENT bucket only)  -  tracks net
    /// troops a player has sent in attacks, used by `conquerPlayer`'s
    /// never-attacked gold-skip check.
    pub fn adjust_attacks_sent(&mut self, small_id: u16, delta: f64) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.attacks_sent_net += delta.round() as i64;
        }
    }

    /// Bulk conquer for spawn hops - per-tile border refresh (TS `GameImpl.conquer`).
    pub fn conquer_spawn_tiles(&mut self, small_id: u16, tiles: &[TileRef]) {
        for &tile in tiles {
            self.conquer_one(small_id, tile, true);
        }
    }

    fn conquer_one(&mut self, small_id: u16, tile: TileRef, refresh_borders: bool) {
        // TS `GameImpl.conquer`: `if (!this.isLand(tile)) throw ...; if
        // (this.isImpassable(tile)) throw ...;` - impassable tiles (which
        // are land, so the `is_land` check alone doesn't catch them) must
        // never receive a real owner. Native no-ops instead of TS's throw,
        // matching this function's existing silent-reject convention for
        // the `!is_land` case just above.
        if !self.is_land(tile) || self.is_impassable(tile) {
            return;
        }
        let tick = self.ticks;
        let prev = self.map.owner_id(tile);
        if prev > 0 {
            if let Some(p) = self.player_by_small_id_mut(prev) {
                p.tiles_owned = (p.tiles_owned - 1).max(0);
                p.border_tiles.remove(tile);
                p.owned_tiles.retain(|&t| t != tile);
                p.last_tile_change = tick;
                if p.tiles_owned == 0 {
                    // Match TS `Player.isAlive()` (`_tiles.size > 0`) immediately —
                    // do not wait for the next `PlayerExecution` tick. Mid-tick
                    // AI / attack filters that read `.alive` must see the player
                    // as dead the moment their last tile is gone.
                    p.alive = false;
                }
            }
        }
        self.map.set_owner_id(tile, small_id);
        self.map.set_fallout(tile, false);
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.tiles_owned += 1;
            p.owned_tiles.push(tile);
            p.last_tile_change = tick;
            p.alive = true;
        }
        if refresh_borders {
            self.refresh_borders_around(tile);
        }
    }

    fn tick_player_income(&mut self) {
        // Troop income is handled by per-player `PlayerExecution` in exec order.
    }

    pub fn attack_indices_from(&self, owner_small_id: u16) -> Vec<usize> {
        self.execs
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if matches!(
                    e,
                    ExecEnum::Attack(a) if a.owner_small_id() == owner_small_id
                ) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn execs_len(&self) -> usize {
        self.execs.len()
    }

    pub fn player_exec_index(&self, small_id: u16) -> Option<usize> {
        self.execs
            .iter()
            .position(|e| matches!(e, ExecEnum::Player(p) if p.small_id() == small_id))
    }

    pub fn first_attack_index_from(&self, owner_small_id: u16) -> Option<usize> {
        self.execs.iter().position(|e| {
            matches!(
                e,
                ExecEnum::Attack(a) if a.owner_small_id() == owner_small_id
            )
        })
    }

    pub fn count_player_executions(&self, small_id: u16) -> usize {
        self.execs
            .iter()
            .filter(|e| matches!(e, ExecEnum::Player(p) if p.small_id() == small_id))
            .count()
    }

    pub fn count_player_executions_all(&self) -> usize {
        self.execs
            .iter()
            .filter(|e| matches!(e, ExecEnum::Player(_)))
            .count()
    }

    pub fn count_tribe_executions(&self) -> usize {
        self.execs
            .iter()
            .filter(|e| matches!(e, ExecEnum::Tribe(_)))
            .count()
    }

    pub fn count_active_attacks_from(&self, owner_small_id: u16) -> usize {
        self.execs
            .iter()
            .filter(|e| {
                if !e.is_active() {
                    return false;
                }
                matches!(
                    e,
                    ExecEnum::Attack(a) if a.owner_small_id() == owner_small_id
                )
            })
            .count()
    }

    pub fn active_land_attacks_in_order(&self) -> Vec<(usize, u16, u16, f64, Option<String>)> {
        self.execs
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let ExecEnum::Attack(a) = e {
                    if e.is_active() && a.is_initialized() && a.source_tile().is_none() {
                        let owner = self
                            .player_by_small_id(a.owner_small_id())
                            .map(|p| p.id.clone());
                        return Some((
                            i,
                            a.owner_small_id(),
                            a.target_small_id(),
                            a.troops(),
                            owner,
                        ));
                    }
                }
                None
            })
            .collect()
    }

    pub fn exec_labels(&self) -> Vec<String> {
        self.execs.iter().map(|e| e.debug_label()).collect()
    }

    pub fn add_execution(&mut self, exec: ExecEnum) {
        self.uninit.push(exec);
    }

    pub fn order_retreat(&mut self, owner_small_id: u16, attack_id: &str) {
        for exec in &mut self.execs {
            if let ExecEnum::Attack(atk) = exec {
                if atk.owner_small_id() == owner_small_id
                    && atk.attack_id() == attack_id
                    && atk.attack_live()
                {
                    atk.order_retreat();
                    return;
                }
            }
        }
    }

    pub fn execute_retreat(&mut self, owner_small_id: u16, attack_id: &str) {
        for exec in &mut self.execs {
            if let ExecEnum::Attack(atk) = exec {
                if atk.owner_small_id() == owner_small_id && atk.attack_id() == attack_id {
                    atk.execute_retreat();
                    return;
                }
            }
        }
    }

    /// TS `NukeExecution.detonate`'s per-impacted-tile casualty spread to forces
    /// already deployed: the SAME `nukeDeathFactor` rate applied to the impacted
    /// player's home troops (just above this call at each call site) is also applied
    /// to every live outgoing land attack and in-flight transport ship that player
    /// owns, regardless of that attack/ship's own location. Native previously only
    /// reduced the home troop pool, leaving attacks/boats already under way
    /// completely unaffected by a nuke landing on the home territory that spawned them.
    pub fn apply_nuke_deaths_to_deployed_forces(
        &mut self,
        owner_small_id: u16,
        nuke_type: &str,
        num_tiles_left: f64,
        max_troops: f64,
    ) {
        let execs = &mut self.execs;
        let wire = &self.wire;
        for exec in execs {
            match exec {
                ExecEnum::Attack(atk)
                    if atk.owner_small_id() == owner_small_id
                        && atk.attack_live()
                        && atk.is_initialized() =>
                {
                    let troops = atk.troops();
                    let deaths =
                        wire.nuke_death_factor(nuke_type, troops, num_tiles_left, max_troops);
                    atk.set_troops(troops - deaths);
                }
                ExecEnum::TransportShip(t)
                    if t.owner_small_id() == owner_small_id && t.unit_id().is_some() =>
                {
                    let troops = t.carried_troops();
                    let deaths =
                        wire.nuke_death_factor(nuke_type, troops, num_tiles_left, max_troops);
                    t.set_carried_troops(troops - deaths);
                }
                _ => {}
            }
        }
    }

    pub fn order_boat_retreat(&mut self, owner_small_id: u16, unit_id: i32) {
        let owns_unit = self
            .player_by_small_id(owner_small_id)
            .is_some_and(|p| p.units.iter().any(|u| u.id == unit_id));
        if !owns_unit {
            return;
        }
        for exec in &mut self.execs {
            if let ExecEnum::TransportShip(transport) = exec {
                if transport.unit_id() == Some(unit_id) {
                    transport.request_retreat();
                    return;
                }
            }
        }
    }

    pub fn add_land_attack(
        &mut self,
        owner_small_id: u16,
        target_player_id: Option<String>,
        start_troops: Option<f64>,
    ) {
        self.add_land_attack_from(owner_small_id, target_player_id, start_troops, None);
    }

    pub fn add_land_attack_from(
        &mut self,
        owner_small_id: u16,
        target_player_id: Option<String>,
        start_troops: Option<f64>,
        source_tile: Option<TileRef>,
    ) {
        self.add_execution(ExecEnum::Attack(
            crate::execution::AttackExecution::new_with_source(
                owner_small_id,
                target_player_id,
                start_troops,
                source_tile,
            ),
        ));
    }

    pub fn active_attacks_debug(&self) -> Vec<(u16, u16, f64, bool, bool, usize, usize)> {
        use crate::execution::ExecEnum;
        self.execs
            .iter()
            .filter_map(|e| {
                let ExecEnum::Attack(atk) = e else {
                    return None;
                };
                Some((
                    atk.owner_small_id(),
                    atk.target_small_id(),
                    atk.troops(),
                    atk.is_active(),
                    atk.attack_live(),
                    atk.border_tile_count(),
                    atk.to_conquer_len(),
                ))
            })
            .collect()
    }

    /// Test-only scaffolding: appends an already-fully-constructed execution
    /// straight into the live `execs` list, bypassing `add_execution`'s
    /// `uninit` -> `init()` pipeline entirely. Lets tests build an exec (e.g.
    /// a `TransportShipExecution` with a real `unit_id`/carried troops set
    /// directly via its own module's private-field access) that represents
    /// mid-flight state `init()` itself can't reach without real map
    /// geometry `Game::default()` doesn't provide, then exercise later-tick
    /// logic (like nuke blast casualties) against it directly.
    #[cfg(test)]
    pub(crate) fn push_exec_for_test(&mut self, exec: crate::execution::ExecEnum) {
        self.execs.push(exec);
    }

    pub fn add_transport_attack(&mut self, owner_small_id: u16, target_tile: TileRef, troops: f64) {
        self.add_execution(ExecEnum::TransportShip(
            crate::execution::TransportShipExecution::new(owner_small_id, target_tile, troops),
        ));
    }

    pub fn unit_count(&self, small_id: u16, unit_type: &str) -> usize {
        self.player_by_small_id(small_id)
            .map(|p| p.units.iter().filter(|u| u.unit_type == unit_type).count())
            .unwrap_or(0)
    }

    /// TS `PlayerImpl.unitCount(type)` - sums `unit.level()`, not just count.
    pub fn unit_level_sum(&self, small_id: u16, unit_type: &str) -> i32 {
        self.player_by_small_id(small_id)
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.unit_type == unit_type)
                    .map(|u| u.level)
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn unit_tile_of(&self, small_id: u16, unit_id: i32) -> Option<TileRef> {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .map(|u| u.tile as TileRef)
    }

    /// TS `GameImpl.unitCount(type)` - sums `unit.level()` across *all* players
    /// (dead included, matching `this._players.values()`), not just alive ones.
    pub fn unit_level_sum_global(&self, unit_type: &str) -> i32 {
        self.players
            .iter()
            .flat_map(|p| p.units.iter())
            .filter(|u| u.unit_type == unit_type && !u.under_construction)
            .map(|u| u.level)
            .sum()
    }

    /// Find which player (if any) currently owns a unit by id. Used by trade
    /// ships/ports, which track counterpart structures by id across possible
    /// land-conquest ownership changes rather than a frozen owner reference.
    pub fn find_unit_owner(&self, unit_id: i32) -> Option<u16> {
        self.players
            .iter()
            .find(|p| p.units.iter().any(|u| u.id == unit_id))
            .map(|p| p.small_id)
    }

    pub fn unit_level_of(&self, small_id: u16, unit_id: i32) -> i32 {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .map(|u| u.level)
            .unwrap_or(1)
    }

    pub fn unit_exists(&self, small_id: u16, unit_id: i32) -> bool {
        self.player_by_small_id(small_id)
            .is_some_and(|p| p.units.iter().any(|u| u.id == unit_id))
    }

    pub fn unit_type_of(&self, small_id: u16, unit_id: i32) -> Option<String> {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .map(|u| u.unit_type.clone())
    }

    pub fn unit_mut(&mut self, small_id: u16, unit_id: i32) -> Option<&mut Unit> {
        self.player_by_small_id_mut(small_id)
            .and_then(|p| p.units.iter_mut().find(|u| u.id == unit_id))
    }

    pub fn unit(&self, small_id: u16, unit_id: i32) -> Option<&Unit> {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
    }

    /// TS `UnitImpl.veterancy()` - always 0 for non-warships (nothing ever increments
    /// a non-warship's `veterancy` field).
    pub fn unit_veterancy(&self, small_id: u16, unit_id: i32) -> i32 {
        self.unit(small_id, unit_id)
            .map(|u| u.veterancy)
            .unwrap_or(0)
    }

    /// TS `UnitImpl.maxHealth()`. Only `Warship` has a veterancy-scaled bonus (every other
    /// unit type's `veterancy` field is always 0, so `max_health_with_veterancy` is a no-op
    /// for them and this reduces to `base`).
    pub fn unit_max_health(&self, small_id: u16, unit_id: i32) -> i32 {
        use crate::core::schemas::unit_type::WARSHIP;
        let Some(unit) = self.unit(small_id, unit_id) else {
            return 1;
        };
        if unit.unit_type != WARSHIP {
            return 1; // Only warships have configured maxHealth in this port so far.
        }
        crate::core::veterancy::max_health_with_veterancy(
            self.wire.warship_base_max_health(),
            unit.veterancy,
            self.wire.warship_veterancy_health_bonus(),
        )
    }

    /// Raise a warship's veterancy by one level (capped at `warshipMaxVeterancy`), which
    /// raises its max health - TS `UnitImpl.increaseVeterancy()`. The ship is NOT instantly
    /// healed; it heals toward the new, higher cap normally. No-op for non-warships or a
    /// unit already at the cap.
    fn increase_veterancy(&mut self, small_id: u16, unit_id: i32) {
        let max_veterancy = self.wire.warship_max_veterancy();
        let mut updated_veterancy = None;
        if let Some(unit) = self.unit_mut(small_id, unit_id) {
            if unit.veterancy < max_veterancy {
                unit.veterancy += 1;
                updated_veterancy = Some(unit.veterancy);
            }
        }
        if let Some(veterancy) = updated_veterancy {
            self.refresh_shell_owner_veterancy(unit_id, veterancy);
        }
    }

    fn refresh_shell_owner_veterancy(&mut self, unit_id: i32, veterancy: i32) {
        for exec in &mut self.execs {
            if let ExecEnum::Shell(shell) = exec {
                shell.refresh_owner_veterancy(unit_id, veterancy);
            }
        }
        for exec in &mut self.uninit {
            if let ExecEnum::Shell(shell) = exec {
                shell.refresh_owner_veterancy(unit_id, veterancy);
            }
        }
        if let Some(ptr) = self.init_merge_batch {
            let batch = unsafe { &mut *ptr };
            for exec in batch.iter_mut() {
                if let ExecEnum::Shell(shell) = exec {
                    shell.refresh_owner_veterancy(unit_id, veterancy);
                }
            }
        }
    }

    /// TS `UnitImpl.recordKill(targetType)`. A killing blow on an enemy warship is an
    /// instant level (and wipes partial transport/capture progress); a transport kill only
    /// adds partial progress (shared with trade captures - see `add_veterancy_progress`).
    /// No-op for non-warships (mirrors `_warshipState === undefined` in TS) and for any
    /// other `target_type` (TS's `recordKill` only branches on Warship/TransportShip).
    pub fn record_kill(&mut self, small_id: u16, unit_id: i32, target_type: &str) {
        use crate::core::schemas::unit_type::{TRANSPORT, WARSHIP};
        let Some(unit) = self.unit(small_id, unit_id) else {
            return;
        };
        if unit.unit_type != WARSHIP {
            return;
        }
        if target_type == WARSHIP {
            if let Some(unit) = self.unit_mut(small_id, unit_id) {
                unit.veterancy_progress = 0;
            }
            self.increase_veterancy(small_id, unit_id);
        } else if target_type == TRANSPORT {
            self.add_veterancy_progress(small_id, unit_id, TRANSPORT);
        }
    }

    /// TS `UnitImpl.recordTradeCapture()`.
    pub fn record_trade_capture(&mut self, small_id: u16, unit_id: i32) {
        use crate::core::schemas::unit_type::{TRADE_SHIP, WARSHIP};
        let Some(unit) = self.unit(small_id, unit_id) else {
            return;
        };
        if unit.unit_type != WARSHIP {
            return;
        }
        self.add_veterancy_progress(small_id, unit_id, TRADE_SHIP);
    }

    /// TS `UnitImpl.addVeterancyProgress(source)`. Transports and captures share one
    /// integer progress meter: one level = `transportThreshold * captureThreshold` points,
    /// a transport is worth `captureThreshold` points and a capture is worth
    /// `transportThreshold` points, so `transportThreshold` transports OR
    /// `captureThreshold` captures (or any mix) fill exactly one level. Overflow carries
    /// into the next level; only a warship kill (`record_kill`'s Warship arm) resets it.
    fn add_veterancy_progress(&mut self, small_id: u16, unit_id: i32, source: &str) {
        use crate::core::schemas::unit_type::TRANSPORT;
        let max_veterancy = self.wire.warship_max_veterancy();
        let transport_threshold = self.wire.warship_veterancy_transport_kills();
        let capture_threshold = self.wire.warship_veterancy_trade_captures();
        let points_per_level = transport_threshold * capture_threshold;

        let Some(current) = self.unit(small_id, unit_id).map(|u| u.veterancy) else {
            return;
        };
        if current >= max_veterancy {
            return;
        }
        let add = if source == TRANSPORT {
            capture_threshold
        } else {
            transport_threshold
        };
        if let Some(unit) = self.unit_mut(small_id, unit_id) {
            unit.veterancy_progress += add;
        }

        // Each loop iteration re-fetches through `unit()`/`unit_mut()` rather than holding a
        // borrow across `increase_veterancy`'s own `&mut self` call.
        loop {
            let Some(unit) = self.unit(small_id, unit_id) else {
                return;
            };
            if unit.veterancy_progress < points_per_level || unit.veterancy >= max_veterancy {
                break;
            }
            if let Some(unit) = self.unit_mut(small_id, unit_id) {
                unit.veterancy_progress -= points_per_level;
            }
            self.increase_veterancy(small_id, unit_id);
        }

        if self
            .unit(small_id, unit_id)
            .is_some_and(|u| u.veterancy >= max_veterancy)
        {
            if let Some(unit) = self.unit_mut(small_id, unit_id) {
                unit.veterancy_progress = 0;
            }
        }
    }

    /// TS `UnitImpl.launch()`  -  push the current tick onto the missile timer queue.
    pub fn unit_launch(&mut self, small_id: u16, unit_id: i32) {
        let tick = self.ticks;
        if let Some(u) = self.unit_mut(small_id, unit_id) {
            u.missile_timer_queue.push(tick);
        }
    }

    /// TS `UnitImpl.reloadMissile()`  -  pop the oldest queued launch.
    pub fn unit_reload_missile(&mut self, small_id: u16, unit_id: i32) {
        if let Some(u) = self.unit_mut(small_id, unit_id) {
            if !u.missile_timer_queue.is_empty() {
                u.missile_timer_queue.remove(0);
            }
        }
    }

    /// TS `UnitImpl.isInCooldown()`  -  timer queue full at `level()` slots.
    pub fn unit_is_in_cooldown(&self, small_id: u16, unit_id: i32) -> bool {
        self.unit(small_id, unit_id)
            .is_some_and(|u| u.missile_timer_queue.len() as i32 == u.level.max(1))
    }

    pub fn set_has_train_station(&mut self, small_id: u16, unit_id: i32, val: bool) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.has_train_station = val;
            }
        }
    }

    pub fn unit_has_train_station(&self, small_id: u16, unit_id: i32) -> bool {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .is_some_and(|u| u.has_train_station)
    }

    /// TS `UnitGrid.hasUnitNearby` - scans across ALL players (not owner-filtered).
    pub fn has_unit_nearby_any(&self, tile: TileRef, range: u32, unit_type: &str) -> bool {
        let tx = self.map.x(tile);
        let ty = self.map.y(tile);
        self.unit_grid
            .has_unit_nearby(tx, ty, range, unit_type, None, false, |t| {
                (self.map.x(t), self.map.y(t))
            })
    }

    /// TS `UnitGrid.nearbyUnits(tile, range, types)` - scans across ALL players.
    /// Returns `(owner_small_id, unit_id, tile, dist_squared)`.
    ///
    /// Order matters even for callers that re-sort by distance afterward: JS `Array.sort` (and
    /// Rust's `sort_by`) are stable, so ties fall back to this pre-sort order, and some callers
    /// (`FactoryExecution.createStation`) use the raw order with no sort at all. TS's `UnitGrid`
    /// enumerates its 100x100-tile cells row-major (`cy` outer, `cx` inner), then `types` in the
    /// given array order, then Set insertion order within each cell/type. Immobile structures
    /// keep creation order; mobile units that leave and re-enter a cell are re-added at the end.
    pub fn nearby_structures_any(
        &self,
        tile: TileRef,
        range: u32,
        types: &[&str],
    ) -> Vec<(u16, i32, TileRef, f64)> {
        let tx = self.map.x(tile);
        let ty = self.map.y(tile);
        self.unit_grid
            .nearby_units(tx, ty, range, types, false, |t| {
                (self.map.x(t), self.map.y(t))
            })
    }

    /// TS `UnitImpl.setUnderConstruction` - keeps the UnitGrid filter flag in sync.
    pub fn set_unit_under_construction(
        &mut self,
        small_id: u16,
        unit_id: i32,
        under_construction: bool,
    ) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.under_construction = under_construction;
            }
        }
        self.unit_grid
            .set_under_construction(unit_id, under_construction);
    }

    /// TS `unitsOwned(type)`  -  includes construction; completed units count by level.
    pub fn units_owned_count(&self, small_id: u16, unit_type: &str) -> u32 {
        self.player_by_small_id(small_id)
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.unit_type == unit_type)
                    .map(|u| {
                        if u.under_construction {
                            1
                        } else {
                            u.level.max(1) as u32
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn units_constructed_count(&self, small_id: u16, unit_type: &str) -> u32 {
        self.player_by_small_id(small_id)
            .and_then(|p| p.units_constructed.get(unit_type).copied())
            .unwrap_or(0)
    }

    fn record_unit_constructed(&mut self, small_id: u16, unit_type: &str) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            *p.units_constructed
                .entry(unit_type.to_string())
                .or_insert(0) += 1;
        }
    }

    /// TS `costWrapper` unit count for structure pricing.
    pub fn structure_cost_units(&self, small_id: u16, unit_type: &str) -> u32 {
        self.wire
            .cost_types_for(unit_type)
            .iter()
            .map(|t| {
                self.units_owned_count(small_id, t)
                    .min(self.units_constructed_count(small_id, t))
            })
            .sum()
    }

    pub fn structure_cost(&self, small_id: u16, unit_type: &str) -> i64 {
        use crate::core::schemas::unit_type as ut;
        if unit_type == ut::MIRV {
            // TS `unitInfo(MIRV).cost` - escalates with the global launch count.
            return 25_000_000 + self.mirvs_launched as i64 * 15_000_000;
        }
        let n = self.structure_cost_units(small_id, unit_type);
        self.wire.structure_cost(unit_type, n)
    }

    /// Sum of completed city levels for `maxTroops`.
    pub fn completed_city_level_sum(&self, small_id: u16) -> i64 {
        use crate::core::schemas::unit_type::CITY;
        self.player_by_small_id(small_id)
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.unit_type == CITY && !u.under_construction)
                    .map(|u| u.level.max(1) as i64)
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn max_troops_for(&self, small_id: u16) -> f64 {
        let Some(p) = self.player_by_small_id(small_id) else {
            return 0.0;
        };
        let city_levels = self.completed_city_level_sum(small_id);
        self.wire
            .max_troops(p.player_type, p.tiles_owned, city_levels)
    }

    pub fn troop_increase_rate_for(&self, small_id: u16) -> i32 {
        let Some(p) = self.player_by_small_id(small_id) else {
            return 0;
        };
        let city_levels = self.completed_city_level_sum(small_id);
        self.wire
            .troop_increase_rate(p.player_type, p.troops, p.tiles_owned, city_levels)
    }

    /// Unrounded variant for the RL obs (`troopIncome` is a raw float in TS).
    pub fn troop_increase_rate_raw_for(&self, small_id: u16) -> f64 {
        let Some(p) = self.player_by_small_id(small_id) else {
            return 0.0;
        };
        let city_levels = self.completed_city_level_sum(small_id);
        self.wire
            .troop_increase_rate_raw(p.player_type, p.troops, p.tiles_owned, city_levels)
    }

    pub fn can_upgrade_unit(&self, small_id: u16, unit_id: i32) -> bool {
        let Some(p) = self.player_by_small_id(small_id) else {
            return false;
        };
        if !p.alive {
            return false;
        }
        let Some(u) = p.units.iter().find(|u| u.id == unit_id) else {
            return false;
        };
        if u.under_construction {
            return false;
        }
        if !crate::execution::upgrade_structure::is_upgradable_type(&u.unit_type) {
            return false;
        }
        if self.wire.is_unit_disabled(&u.unit_type) {
            return false;
        }
        let cost = self.structure_cost(small_id, &u.unit_type);
        p.gold >= cost
    }

    pub fn upgrade_unit(&mut self, small_id: u16, unit_id: i32) -> bool {
        if !self.can_upgrade_unit(small_id, unit_id) {
            return false;
        }
        let cost = self.structure_cost(small_id, {
            let p = self.player_by_small_id(small_id).unwrap();
            let u = p.units.iter().find(|u| u.id == unit_id).unwrap();
            u.unit_type.as_str()
        });
        let unit_type = {
            let p = self.player_by_small_id(small_id).unwrap();
            p.units
                .iter()
                .find(|u| u.id == unit_id)
                .unwrap()
                .unit_type
                .clone()
        };
        let tick = self.ticks;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold -= cost;
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.level += 1;
                // TS `Unit.increaseLevel`: MissileSilo/SAMLauncher push the current tick
                // onto their missile timer queue on every level-up, not just on launch -
                // the newly-added capacity slot starts in its own cooldown rather than
                // being immediately available. Native previously only bumped `level`,
                // which under-counted the queue relative to `isInCooldown`'s
                // `queue.len() == level` check and let an upgraded silo/SAM fire an extra
                // missile immediately instead of waiting out one cooldown like TS.
                if u.unit_type == crate::core::schemas::unit_type::MISSILE_SILO
                    || u.unit_type == crate::core::schemas::unit_type::SAM_LAUNCHER
                {
                    u.missile_timer_queue.push(tick);
                }
            }
        }
        self.record_unit_constructed(small_id, &unit_type);
        true
    }

    pub fn build_unit(&mut self, small_id: u16, unit_type: &str, tile: TileRef) -> i32 {
        // TS `PlayerImpl.buildUnit` → `removeGold` (clamped); never drive gold negative.
        let cost = self.structure_cost(small_id, unit_type);
        self.remove_gold(small_id, cost);
        let id = self.next_unit_id;
        self.next_unit_id += 1;
        let ticks = self.ticks;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.units.push(Unit {
                id,
                unit_type: unit_type.to_string(),
                tile: tile as i32,
                under_construction: false,
                health: if unit_type == crate::core::schemas::unit_type::WARSHIP {
                    1000
                } else {
                    1
                },
                level: 1,
                has_train_station: false,
                last_safe_from_pirates_tick: if unit_type
                    == crate::core::schemas::unit_type::TRADE_SHIP
                {
                    ticks as i32
                } else {
                    0
                },
                ..Default::default()
            });
        }
        // TS `GameImpl.addUnit` → `unitGrid.addUnit`.
        let (x, y) = (self.map.x(tile), self.map.y(tile));
        self.unit_grid
            .add_unit(small_id, id, unit_type, tile, x, y, false);
        self.record_unit_constructed(small_id, unit_type);
        id
    }

    /// Records that `destroyer_small_id` just destroyed `victim_small_id`'s transport ship
    /// `unit_id` at `tile` - TS `unit.delete(true, destroyer)` (the tile is recorded too since
    /// `maybeRetaliateWithWarship(ship.tile(), ...)` reads it off the - in TS, still
    /// inspectable - dead unit). Call *before* `remove_unit`, which drops the unit immediately.
    pub fn record_transport_kill(
        &mut self,
        unit_id: i32,
        victim_small_id: u16,
        destroyer_small_id: u16,
        tile: TileRef,
    ) {
        self.transport_kills
            .push((unit_id, victim_small_id, destroyer_small_id, tile));
    }

    /// Consumes (removes) a recorded transport-ship kill, if any - TS `ship.wasDestroyedByEnemy()
    /// && ship.destroyer()`. Returns `(destroyer_small_id, tile)`.
    pub fn take_transport_kill(&mut self, unit_id: i32) -> Option<(u16, TileRef)> {
        let idx = self
            .transport_kills
            .iter()
            .position(|&(id, _, _, _)| id == unit_id)?;
        let (_, _, destroyer, tile) = self.transport_kills.remove(idx);
        Some((destroyer, tile))
    }

    pub fn remove_unit(&mut self, small_id: u16, unit_id: i32) {
        let had_station = self
            .player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .is_some_and(|u| u.has_train_station);
        // TS `GameImpl.removeUnit` → `unitGrid.removeUnit` before dropping the unit.
        self.unit_grid.remove_unit(unit_id);
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.units.retain(|u| u.id != unit_id);
        }
        // TS `GameImpl.removeUnit` - drop the train station from the rail network too.
        if had_station {
            crate::rail::remove_station(self, unit_id);
        }
    }

    pub fn capture_unit(&mut self, from_small_id: u16, to_small_id: u16, unit_id: i32) {
        let mut captured = None;
        if let Some(p) = self.player_by_small_id_mut(from_small_id) {
            if let Some(idx) = p.units.iter().position(|u| u.id == unit_id) {
                captured = Some(p.units.remove(idx));
            }
        }
        if let Some(mut unit) = captured {
            // TS `UnitImpl.setOwner()`'s `clearPendingDeletion()` - a captured unit resets
            // any voluntary-deletion mark from its previous owner.
            unit.deletion_at = None;
            if let Some(p) = self.player_by_small_id_mut(to_small_id) {
                p.units.push(unit);
            }
            // TS `setOwner` keeps the same Unit object in the grid; only owner changes.
            self.unit_grid.set_owner(unit_id, to_small_id);
            // TS TrainStation keeps a live Unit ref so owner/isActive track
            // capture automatically — update the cached station owner here.
            crate::rail::update_station_owner_for_unit(
                &mut self.rail_network,
                unit_id,
                to_small_id,
            );
        }
    }

    pub fn move_unit(&mut self, small_id: u16, unit_id: i32, tile: TileRef) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.tile = tile as i32;
            }
        }
        // TS `UnitImpl.move` → `GameImpl.onUnitMoved` → `unitGrid.updateUnitCell`.
        let (x, y) = (self.map.x(tile), self.map.y(tile));
        self.unit_grid.update_unit_tile(unit_id, tile, x, y);
    }

    /// Coalesce pending attacks in the uninit queue only. Merging into initialized
    /// execs must wait until `AttackExecution.init` (after all ticks).
    fn try_merge_land_attack(
        &mut self,
        owner_small_id: u16,
        target_player_id: &Option<String>,
        start_troops: Option<f64>,
    ) -> bool {
        for exec in &mut self.uninit {
            if let Some(atk) = exec.as_attack() {
                if atk.can_merge_with(owner_small_id, target_player_id)
                    && atk.is_active()
                    && !atk.is_initialized()
                    && atk.source_tile().is_none()
                {
                    atk.absorb_pending_troops(start_troops);
                    return true;
                }
            }
        }
        false
    }

    /// TS `PlayerImpl.createAttack`  -  register before cancel/merge in `AttackExecution.init`.
    pub fn register_land_attack(
        &mut self,
        attack_id: &str,
        owner_small_id: u16,
        target_small_id: u16,
    ) {
        if let Some(p) = self.player_by_small_id_mut(owner_small_id) {
            p.outgoing_land_attacks.push(attack_id.to_string());
        }
        if target_small_id != 0
            && target_small_id != owner_small_id
            && target_small_id != self.terra_nullius_id()
        {
            if let Some(p) = self.player_by_small_id_mut(target_small_id) {
                p.incoming_land_attacks.push(attack_id.to_string());
            }
        }
    }

    /// TS `AttackImpl.delete()`  -  drop from player registries.
    pub fn unregister_land_attack(
        &mut self,
        attack_id: &str,
        owner_small_id: u16,
        target_small_id: u16,
    ) {
        if let Some(p) = self.player_by_small_id_mut(owner_small_id) {
            p.outgoing_land_attacks.retain(|id| id != attack_id);
        }
        if let Some(p) = self.player_by_small_id_mut(target_small_id) {
            p.incoming_land_attacks.retain(|id| id != attack_id);
        }
    }

    fn find_land_attack_exec_mut(
        &mut self,
        attack_id: &str,
    ) -> Option<*mut crate::execution::AttackExecution> {
        let batch_ptr = self.init_merge_batch;
        if let Some(ptr) = batch_ptr {
            let batch = unsafe { &mut *ptr };
            for exec in batch.iter_mut() {
                if let ExecEnum::Attack(atk) = exec {
                    if atk.attack_id() == attack_id && atk.attack_live() {
                        return Some(atk as *mut _);
                    }
                }
            }
        }
        for exec in &mut self.execs {
            if let ExecEnum::Attack(atk) = exec {
                if atk.attack_id() == attack_id && atk.attack_live() {
                    return Some(atk as *mut _);
                }
            }
        }
        None
    }

    /// TS `AttackExecution.init` - cancel opposing mutual attacks.
    pub fn cancel_opposing_land_attacks(
        &mut self,
        owner_small_id: u16,
        target_small_id: u16,
        current_attack_id: &str,
        troops: &mut f64,
        active: &mut bool,
    ) {
        if !*active {
            return;
        }
        let incoming_ids: Vec<String> = self
            .player_by_small_id(owner_small_id)
            .map(|p| p.incoming_land_attacks.clone())
            .unwrap_or_default();
        for id in incoming_ids {
            if id == current_attack_id {
                continue;
            }
            let Some(ptr) = self.find_land_attack_exec_mut(&id) else {
                continue;
            };
            let incoming_attacker = unsafe { (*ptr).owner_small_id() };
            if self
                .player_by_small_id(incoming_attacker)
                .is_none_or(|p| p.tiles_owned == 0)
            {
                continue;
            }
            if incoming_attacker != target_small_id {
                continue;
            }
            let incoming_troops = unsafe { (*ptr).troops() };
            if incoming_troops > *troops {
                unsafe {
                    (*ptr).set_troops(incoming_troops - *troops);
                }
                *active = false;
                return;
            }
            *troops -= incoming_troops;
            let game = self as *mut Game;
            unsafe {
                (*ptr).kill_attack(&mut *game);
            }
        }
    }

    /// TS `AttackExecution.init` - merge all same-target outgoing land attacks.
    pub fn merge_outgoing_land_attacks(
        &mut self,
        owner_small_id: u16,
        target_small_id: u16,
        current_attack_id: &str,
        troops: &mut f64,
    ) {
        let outgoing_ids: Vec<String> = self
            .player_by_small_id(owner_small_id)
            .map(|p| p.outgoing_land_attacks.clone())
            .unwrap_or_default();
        for id in outgoing_ids {
            if id == current_attack_id {
                continue;
            }
            let Some(ptr) = self.find_land_attack_exec_mut(&id) else {
                continue;
            };
            let (same_owner, same_target) = unsafe {
                (
                    (*ptr).owner_small_id() == owner_small_id,
                    (*ptr).target_small_id() == target_small_id,
                )
            };
            if !same_owner || !same_target {
                continue;
            }
            let extra = unsafe { (*ptr).troops() };
            *troops += extra;
            let game = self as *mut Game;
            unsafe {
                (*ptr).kill_attack(&mut *game);
            }
        }
    }

    /// Largest incoming land attack from candidates (TS cluster `getCapturingPlayer`).
    /// Includes boat/expansion attacks (`source_tile` set).
    pub fn largest_incoming_land_attack(
        &self,
        defender_small_id: u16,
        attackers: &[u16],
    ) -> Option<u16> {
        let mut largest_attack = 0.0f64;
        let mut largest_attacker: Option<u16> = None;
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if !atk.is_active() || !atk.attack_live() || !atk.is_initialized() {
                continue;
            }
            if atk.target_small_id() != defender_small_id {
                continue;
            }
            let attacker = atk.owner_small_id();
            if !attackers.contains(&attacker) {
                continue;
            }
            if atk.troops() > largest_attack {
                largest_attack = atk.troops();
                largest_attacker = Some(attacker);
            }
        }
        largest_attacker
    }

    /// TS cluster `getCapturingPlayer` attack scan: neighbors Map order, then `outgoingAttacks`.
    pub fn largest_incoming_land_attack_from_neighbors(
        &self,
        defender_small_id: u16,
        attackers: &[u16],
        neighbor_order: &[(u16, u32)],
    ) -> Option<u16> {
        let mut largest_attack = 0.0f64;
        let mut largest_attacker: Option<u16> = None;
        for (attacker, _) in neighbor_order {
            if !attackers.contains(attacker) {
                continue;
            }
            for exec in &self.execs {
                let ExecEnum::Attack(atk) = exec else {
                    continue;
                };
                if !atk.is_active() || !atk.attack_live() || !atk.is_initialized() {
                    continue;
                }
                if atk.owner_small_id() != *attacker || atk.target_small_id() != defender_small_id {
                    continue;
                }
                if atk.troops() > largest_attack {
                    largest_attack = atk.troops();
                    largest_attacker = Some(*attacker);
                }
            }
        }
        largest_attacker
    }

    /// Largest incoming land attack from candidate attackers (TS `AiAttackBehavior.findIncomingAttackPlayer`).
    pub fn largest_land_attack_from(
        &self,
        defender_small_id: u16,
        attackers: &[u16],
    ) -> Option<u16> {
        let mut largest_attack = 0.0f64;
        let mut largest_attacker: Option<u16> = None;
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if !atk.is_active()
                || !atk.attack_live()
                || !atk.is_initialized()
                || atk.source_tile().is_some()
            {
                continue;
            }
            if atk.target_small_id() != defender_small_id {
                continue;
            }
            let attacker = atk.owner_small_id();
            if !attackers.contains(&attacker) {
                continue;
            }
            if atk.troops() > largest_attack {
                largest_attack = atk.troops();
                largest_attacker = Some(attacker);
            }
        }
        largest_attacker
    }

    /// Largest non-friendly incoming attacker, land OR boat-landed (TS
    /// `AiAttackBehavior.findIncomingAttackPlayer`, which reads
    /// `player.incomingAttacks()` - a plain filter over `_incomingAttacks` by
    /// `attacker().isAlive()` with NO `sourceTile()` check - then filters out
    /// friendly attackers, then (only for non-Bot defenders) attackers that
    /// are themselves Bots). Despite the "land" in this function's name (kept
    /// for git-history continuity), it must NOT exclude boat-landed attacks
    /// (`source_tile().is_some()`): TS counts those as ordinary incoming
    /// attacks for retaliation purposes. A dedicated land-only sum exists
    /// separately as `incoming_land_troops` for the one TS call site
    /// (`NationStructureBehavior.defensePostNeeded`) that filters
    /// `sourceTile() === null` explicitly.
    pub fn find_incoming_land_attacker(
        &self,
        defender_small_id: u16,
        defender_type: PlayerType,
    ) -> Option<u16> {
        let mut largest_attack = 0.0f64;
        let mut largest_attacker: Option<u16> = None;
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if !atk.is_active() || !atk.attack_live() || !atk.is_initialized() {
                continue;
            }
            if atk.target_small_id() != defender_small_id {
                continue;
            }
            let attacker = atk.owner_small_id();
            if attacker == defender_small_id || attacker == 0 {
                continue;
            }
            if let Some(p) = self.player_by_small_id(attacker) {
                if p.tiles_owned <= 0 {
                    continue;
                }
            }
            if self.is_friendly(defender_small_id, attacker) {
                continue;
            }
            if defender_type != PlayerType::Bot {
                if let Some(p) = self.player_by_small_id(attacker) {
                    if p.player_type == PlayerType::Bot {
                        continue;
                    }
                }
            }
            if atk.troops() > largest_attack {
                largest_attack = atk.troops();
                largest_attacker = Some(attacker);
            }
        }
        largest_attacker
    }

    pub fn incoming_land_troops(&self, defender_small_id: u16) -> f64 {
        self.execs
            .iter()
            .filter_map(|e| match e {
                ExecEnum::Attack(atk)
                    if atk.is_active()
                        && atk.attack_live()
                        && atk.is_initialized()
                        && atk.target_small_id() == defender_small_id
                        && atk.source_tile().is_none() =>
                {
                    Some(atk.troops())
                }
                _ => None,
            })
            .sum()
    }

    /// TS `PlayerImpl.incomingAttacks()`: all active incoming attacks (land
    /// AND boat-landed) from alive attackers, with `land_only=true` applying
    /// the additional `.filter((a) => a.sourceTile() === null)` some callers
    /// chain on top (e.g. `NationStructureBehavior.tryBuildDefensePost` /
    /// `defensePostNeeded`). Callers that use the plain, unfiltered
    /// `player.incomingAttacks()` (e.g. `NationEmojiBehavior.
    /// checkOverwhelmedByAttacks` / `checkVerySmallAttack`) must pass
    /// `land_only=false`.
    pub fn incoming_attacks(&self, defender_small_id: u16, land_only: bool) -> Vec<IncomingAttack> {
        let mut out = Vec::new();
        let Some(defender) = self.player_by_small_id(defender_small_id) else {
            return out;
        };
        for id in &defender.incoming_land_attacks {
            let Some(attacker) = self.land_attack_owner(id) else {
                continue;
            };
            if attacker == defender_small_id || attacker == 0 {
                continue;
            }
            if let Some(p) = self.player_by_small_id(attacker) {
                if p.tiles_owned <= 0 {
                    continue;
                }
            }
            let Some(troops) = self.land_attack_troops(id) else {
                continue;
            };
            if land_only && self.land_attack_source_tile(id).is_some() {
                continue;
            }
            out.push(IncomingAttack {
                attacker_small_id: attacker,
                troops,
            });
        }
        out
    }

    fn land_attack_owner(&self, attack_id: &str) -> Option<u16> {
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if atk.attack_id() == attack_id && atk.attack_live() && atk.is_initialized() {
                return Some(atk.owner_small_id());
            }
        }
        None
    }

    fn land_attack_troops(&self, attack_id: &str) -> Option<f64> {
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if atk.attack_id() == attack_id && atk.attack_live() && atk.is_initialized() {
                return Some(atk.troops());
            }
        }
        None
    }

    fn land_attack_source_tile(&self, attack_id: &str) -> Option<TileRef> {
        for exec in &self.execs {
            let ExecEnum::Attack(atk) = exec else {
                continue;
            };
            if atk.attack_id() == attack_id && atk.attack_live() && atk.is_initialized() {
                return atk.source_tile();
            }
        }
        None
    }

    /// TS `Player.outgoingAttacks().reduce((sum, a) => sum + a.troops(), 0)`.
    /// Sums troops for *every* outgoing attack, land AND boat-landed -
    /// `PlayerImpl.outgoingAttacks()` returns `_outgoingAttacks` completely
    /// unfiltered, no `sourceTile()` check. Backs
    /// `NationAllianceBehavior.isAlliancePartnerSimilarlyStrong`'s native
    /// mirror (nation_alliance.rs) and the weak-ally betrayal check
    /// (`maybe_betray`); a source-tile filter here would silently undercount
    /// a player's committed troops in both. Independently found and fixed
    /// by two parallel investigations - see docs/bot-ai-parity-double-attack/
    /// and the pr-order-bugs-hunt report for two different call sites this
    /// mattered for.
    pub fn outgoing_attack_troops(&self, attacker_small_id: u16) -> f64 {
        let Some(attacker) = self.player_by_small_id(attacker_small_id) else {
            return 0.0;
        };
        let mut total = 0.0;
        for id in &attacker.outgoing_land_attacks {
            if let Some(troops) = self.land_attack_troops(id) {
                total += troops;
            }
        }
        total
    }

    pub fn num_land_tiles(&self) -> u32 {
        self.map.num_land_tiles
    }

    fn relation_value(&self, a: u16, b: u16) -> f64 {
        self.player_by_small_id(a)
            .and_then(|p| p.relations.get(&b).copied())
            .unwrap_or(0.0)
    }

    /// TS `PlayerImpl.allRelationsSorted()`  -  sort-by-value (stable), tie-broken by
    /// insertion order, filtered to relations with still-alive players.
    pub fn all_relations_sorted(&self, sid: u16) -> Vec<(u16, f64)> {
        let Some(p) = self.player_by_small_id(sid) else {
            return Vec::new();
        };
        let mut out: Vec<(u16, f64)> = p
            .relations
            .iter()
            .filter(|(other, _)| self.player_by_small_id(*other).is_some_and(|o| o.alive))
            .map(|(other, &v)| (other, v))
            .collect();
        out.sort_by(|(_, a), (_, b)| a.total_cmp(b));
        out
    }

    fn relation_from_value(value: f64) -> Relation {
        if value < -50.0 {
            Relation::Hostile
        } else if value < 0.0 {
            Relation::Distrustful
        } else if value < 50.0 {
            Relation::Neutral
        } else {
            Relation::Friendly
        }
    }

    pub fn relation(&self, a: u16, b: u16) -> Relation {
        Self::relation_from_value(self.relation_value(a, b))
    }

    pub fn update_relation(&mut self, a: u16, b: u16, delta: i32) {
        self.update_relation_f64(a, b, delta as f64);
    }

    /// TS `PlayerImpl.updateRelation(other, delta: number)` accepts a float delta - every
    /// existing native call site happens to pass a whole number (embargo malus, attack
    /// retaliation, emoji responses), so `update_relation` above took a plain `i32`, until
    /// `NationWarshipBehavior.maybeRetaliateWithWarship`'s `-7.5` (trade-ship retaliation; the
    /// transport case is `-15`, still whole) needed a non-integer delta for the first time.
    pub fn update_relation_f64(&mut self, a: u16, b: u16, delta: f64) {
        if let Some(p) = self.player_by_small_id_mut(a) {
            let entry = p.relations.entry_or_insert(b, 0.0);
            *entry = clamp_relation(*entry + delta);
        }
    }

    pub fn decay_relations(&mut self, small_id: u16) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            let others: Vec<u16> = p.relations.keys().collect();
            for other in others {
                if let Some(value) = p.relations.get_mut(&other) {
                    *value = decay_relation(*value);
                }
            }
        }
    }

    /// TS `Config.targetDuration()` - how long a target mark stays visible.
    pub const TARGET_DURATION_TICKS: u32 = 10 * 10;
    /// TS `Config.targetCooldown()` - min ticks between a player's own marks.
    pub const TARGET_COOLDOWN_TICKS: u32 = 15 * 10;
    /// TS `Config.deleteUnitCooldown()`.
    pub const DELETE_UNIT_COOLDOWN_TICKS: u32 = 30 * 10;
    /// TS `Config.deletionMarkDuration()`.
    pub const DELETION_MARK_DURATION_TICKS: u32 = 30 * 10;

    /// TS `PlayerImpl.targets()` - marks painted within `targetDuration`.
    pub fn player_targets(&self, small_id: u16) -> Vec<u16> {
        let Some(p) = self.player_by_small_id(small_id) else {
            return Vec::new();
        };
        p.target_marks
            .iter()
            .filter(|(_, tick)| self.ticks - tick < Self::TARGET_DURATION_TICKS)
            .map(|(t, _)| *t)
            .collect()
    }

    /// TS `PlayerImpl.canTarget(other)`.
    pub fn can_target(&self, a: u16, b: u16) -> bool {
        if a == b || self.is_friendly(a, b) {
            return false;
        }
        let Some(p) = self.player_by_small_id(a) else {
            return false;
        };
        !p.target_marks
            .iter()
            .any(|(_, tick)| self.ticks - tick < Self::TARGET_COOLDOWN_TICKS)
    }

    /// TS `PlayerImpl.target(other)` - paint a mark at the current tick.
    pub fn add_target_mark(&mut self, a: u16, b: u16) {
        let tick = self.ticks;
        if let Some(p) = self.player_by_small_id_mut(a) {
            p.target_marks.push((b, tick));
        }
    }

    /// TS `PlayerImpl.canDeleteUnit()`.
    pub fn can_delete_unit(&self, small_id: u16) -> bool {
        self.player_by_small_id(small_id).is_some_and(|p| {
            self.ticks as i64 - p.last_delete_unit_tick as i64
                >= Self::DELETE_UNIT_COOLDOWN_TICKS as i64
        })
    }

    /// TS `PlayerImpl.recordDeleteUnit()`.
    pub fn record_delete_unit(&mut self, small_id: u16) {
        let tick = self.ticks as i32;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.last_delete_unit_tick = tick;
        }
    }

    /// TS `UnitImpl.markForDeletion()`. No-op if the unit doesn't exist (TS's
    /// `!this.isActive()` guard - a nonexistent unit is native's equivalent of inactive).
    pub fn mark_unit_for_deletion(&mut self, small_id: u16, unit_id: i32) {
        let deadline = (self.ticks + Self::DELETION_MARK_DURATION_TICKS) as i32;
        if let Some(u) = self.unit_mut(small_id, unit_id) {
            u.deletion_at = Some(deadline);
        }
    }

    /// TS `UnitImpl.isMarkedForDeletion()`.
    pub fn is_unit_marked_for_deletion(&self, small_id: u16, unit_id: i32) -> bool {
        self.unit(small_id, unit_id)
            .is_some_and(|u| u.deletion_at.is_some())
    }

    /// TS `UnitImpl.isOverdueDeletion()`.
    pub fn is_unit_overdue_deletion(&self, small_id: u16, unit_id: i32) -> bool {
        self.unit(small_id, unit_id).is_some_and(|u| {
            u.deletion_at
                .is_some_and(|at| self.ticks as i64 - at as i64 > 0)
        })
    }

    /// Live land attacks (TS `player.outgoingAttacks()` across all players):
    /// the attack state lives inside `AttackExecution`s on the exec list.
    pub fn live_attacks(&self) -> impl Iterator<Item = &crate::execution::AttackExecution> {
        self.execs.iter().filter_map(|e| match e {
            ExecEnum::Attack(a) if a.attack_live() => Some(a),
            _ => None,
        })
    }

    /// Live transport-ship executions (carried troops for the units obs).
    pub fn live_transports(
        &self,
    ) -> impl Iterator<Item = &crate::execution::TransportShipExecution> {
        self.execs.iter().filter_map(|e| match e {
            ExecEnum::TransportShip(t) => Some(t),
            _ => None,
        })
    }

    /// Live warship executions - mirrors `live_transports()`'s pattern. Backs
    /// `NationWarshipBehavior`'s AI-level queries (`warship_ai.rs`), which need a
    /// warship's `patrolTile` (private to `WarshipExecution`, not stored on the shared
    /// `Unit`).
    pub fn live_warships(&self) -> impl Iterator<Item = &crate::execution::WarshipExecution> {
        self.execs.iter().filter_map(|e| match e {
            ExecEnum::Warship(w) => Some(w),
            _ => None,
        })
    }

    /// TS `NationWarshipBehavior.maybeMoveWarship`'s candidate list: `(unit_id,
    /// current_tile, patrol_tile)` for every live warship owned by `small_id`.
    pub fn warship_patrol_candidates(&self, small_id: u16) -> Vec<(i32, TileRef, TileRef)> {
        self.live_warships()
            .filter(|w| w.owner_small_id() == small_id)
            .filter_map(|w| {
                let unit_id = w.unit_id()?;
                let tile = self.unit_tile_of(small_id, unit_id)?;
                Some((unit_id, tile, w.patrol_tile()))
            })
            .collect()
    }

    /// Redirects one of `small_id`'s existing warships to a new patrol tile - TS
    /// `warship.updateWarshipState({ patrolTile: tile })`.
    pub fn set_warship_patrol_tile(&mut self, small_id: u16, unit_id: i32, tile: TileRef) {
        let tick = self.ticks();
        for exec in &mut self.execs {
            if let ExecEnum::Warship(w) = exec {
                if w.owner_small_id() == small_id && w.unit_id() == Some(unit_id) {
                    w.set_patrol_tile_at_tick(tile, tick);
                    return;
                }
            }
        }
    }

    /// TS `MoveWarshipExecution.init()` - a manual (human-issued) batch move of one or more
    /// of `owner_small_id`'s own warships to `position`. Deduplicates `unit_ids` (TS `new
    /// Set(this.unitIds)`) and, per id, requires: the id names one of `owner_small_id`'s own
    /// `Warship` units (this alone is what makes another player's/another owner's warships
    /// un-movable - TS builds its lookup map from `this.owner.units(Warship)`, so an id the
    /// owner doesn't hold is simply absent from it), an active `WarshipExecution`, and the
    /// warship's current tile sharing a water component with `position`. TS's leading
    /// `isValidRef(position)` bounds check is not ported - no caller in this codebase passes
    /// anything but a `TileRef` already known to be in-bounds.
    pub fn move_warships(&mut self, owner_small_id: u16, unit_ids: &[i32], position: TileRef) {
        let Some(component) = self.get_water_component(position) else {
            return;
        };
        let mut seen = std::collections::HashSet::new();
        for &unit_id in unit_ids {
            if !seen.insert(unit_id) {
                continue;
            }
            let owns_warship = self.player_by_small_id(owner_small_id).is_some_and(|p| {
                p.units.iter().any(|u| {
                    u.id == unit_id && u.unit_type == crate::core::schemas::unit_type::WARSHIP
                })
            });
            if !owns_warship {
                continue;
            }
            let Some(current_tile) = self.unit_tile_of(owner_small_id, unit_id) else {
                continue;
            };
            if !self.has_water_component(current_tile, component) {
                continue;
            }
            for exec in &mut self.execs {
                if let ExecEnum::Warship(w) = exec {
                    if w.owner_small_id() == owner_small_id && w.unit_id() == Some(unit_id) {
                        if w.is_active() {
                            w.retarget_patrol(position);
                        }
                        break;
                    }
                }
            }
        }
    }

    /// TS `UnitGrid.hasUnitNearby(tile, range, type, playerId, includeUnderConstruction)` -
    /// the owner-filtered overload `has_unit_nearby_any` doesn't cover (that one scans all
    /// players and always excludes under-construction units; this one is scoped to a single
    /// owner and always includes them, matching `NationWarshipBehavior`'s only caller:
    /// `hasUnitNearby(target, 90, Warship, this.player.id(), true)`).
    pub fn has_own_unit_nearby(
        &self,
        small_id: u16,
        tile: TileRef,
        range: u32,
        unit_type: &str,
    ) -> bool {
        let Some(p) = self.player_by_small_id(small_id) else {
            return false;
        };
        let range_sq = range * range;
        p.units.iter().any(|u| {
            u.unit_type == unit_type
                && self.map.euclidean_dist_squared(tile, u.tile as TileRef) <= range_sq
        })
    }

    pub fn trade_ship_destination_owner(&self, ship_unit_id: i32) -> Option<u16> {
        let (destination_port, cached_owner) = self
            .execs
            .iter()
            .chain(self.uninit.iter())
            .find_map(|execution| match execution {
                ExecEnum::TradeShip(trade) if trade.ship_unit_id() == Some(ship_unit_id) => Some((
                    trade.destination_port_unit_id(),
                    trade.cached_destination_port_owner_small_id(),
                )),
                _ => None,
            })?;
        self.find_unit_owner(destination_port).or(cached_owner)
    }

    pub fn trade_ship_is_safe_from_pirates(&self, owner: u16, unit_id: i32) -> bool {
        self.unit(owner, unit_id)
            .is_some_and(|unit| self.ticks as i32 - unit.last_safe_from_pirates_tick < 20)
    }

    pub fn warship_is_docked(&self, owner: u16, unit_id: i32) -> bool {
        self.execs.iter().any(|execution| match execution {
            ExecEnum::Warship(warship) => {
                warship.owner_small_id() == owner
                    && warship.unit_id() == Some(unit_id)
                    && warship.is_docked()
            }
            _ => false,
        })
    }

    fn ts_unit_grid_query_includes(&self, center: TileRef, range: i32, unit_tile: TileRef) -> bool {
        // TS `UnitGrid.nearbyUnits` first narrows candidates by 100px grid
        // cells, then applies exact distance. The cell-window formula has
        // edge behavior at exact cell boundaries; mirror it for callers whose
        // semantics go through `nearbyUnits` rather than a direct unit scan.
        const CELL_SIZE: i32 = 100;
        let grid_w = (self.map.width as i32 + CELL_SIZE - 1) / CELL_SIZE;
        let grid_h = (self.map.height as i32 + CELL_SIZE - 1) / CELL_SIZE;
        let x = self.x(center) as i32;
        let y = self.y(center) as i32;
        let grid_x = x / CELL_SIZE;
        let grid_y = y / CELL_SIZE;
        let x_mod = x % CELL_SIZE;
        let y_mod = y % CELL_SIZE;
        let ceil_div = |n: i32, d: i32| (n as f64 / d as f64).ceil() as i32;
        let start_grid_x = (grid_x - ceil_div(range - x_mod, CELL_SIZE)).max(0);
        let end_grid_x =
            (grid_x + ceil_div(range - (CELL_SIZE - x_mod), CELL_SIZE)).min(grid_w - 1);
        let start_grid_y = (grid_y - ceil_div(range - y_mod, CELL_SIZE)).max(0);
        let end_grid_y =
            (grid_y + ceil_div(range - (CELL_SIZE - y_mod), CELL_SIZE)).min(grid_h - 1);
        let unit_grid_x = self.x(unit_tile) as i32 / CELL_SIZE;
        let unit_grid_y = self.y(unit_tile) as i32 / CELL_SIZE;
        unit_grid_x >= start_grid_x
            && unit_grid_x <= end_grid_x
            && unit_grid_y >= start_grid_y
            && unit_grid_y <= end_grid_y
    }

    /// TS `WarshipExecution.dockedShipsAtPort`: non-patrolling own warships
    /// within docking range whose `targetTile` is clear occupy a port's active
    /// healing capacity. `exclude_unit_id` mirrors TS's optional `excludeShip`
    /// used when deciding whether the arriving ship itself may dock.
    pub fn docked_warships_at_port(
        &self,
        owner: u16,
        port_tile: TileRef,
        exclude_unit_id: Option<i32>,
    ) -> usize {
        let docking_radius = 5;
        let docking_radius_sq = docking_radius * docking_radius;
        self.live_warships()
            .filter(|warship| warship.owner_small_id() == owner)
            .filter_map(|warship| {
                let unit_id = warship.unit_id()?;
                if exclude_unit_id == Some(unit_id) {
                    return None;
                }
                let tile = self.unit_tile_of(owner, unit_id)?;
                Some((warship, tile))
            })
            .filter(|(warship, tile)| {
                !warship.is_patrolling()
                    && warship.target_tile().is_none()
                    && self.ts_unit_grid_query_includes(port_tile, docking_radius as i32, *tile)
                    && self.map.euclidean_dist_squared(*tile, port_tile) <= docking_radius_sq
            })
            .count()
    }

    pub fn warship_port_is_full(
        &self,
        owner: u16,
        port_tile: TileRef,
        exclude_unit_id: Option<i32>,
    ) -> bool {
        let Some(level) = self.player_by_small_id(owner).and_then(|player| {
            player
                .units
                .iter()
                .find(|unit| {
                    unit.unit_type == crate::core::schemas::unit_type::PORT
                        && unit.tile as TileRef == port_tile
                })
                .map(|unit| unit.level)
        }) else {
            return false;
        };
        self.docked_warships_at_port(owner, port_tile, exclude_unit_id) >= level as usize
    }

    /// Pending outgoing alliance requests (TS `player.outgoingAllianceRequests()`).
    pub fn outgoing_alliance_requests(&self, small_id: u16) -> Vec<u16> {
        self.alliance_requests
            .iter()
            .filter(|r| {
                r.status == AllianceRequestStatus::Pending && r.requestor_small_id == small_id
            })
            .map(|r| r.recipient_small_id)
            .collect()
    }

    /// Active alliances involving `small_id` (TS `player.alliances()`).
    ///
    /// TS keeps expired-but-not-yet-swept alliances in `_alliances` until
    /// `PlayerExecution` calls `expire()`; do not filter on `expires_at` here.
    pub fn player_alliances(&self, small_id: u16) -> Vec<&AllianceState> {
        self.alliances
            .iter()
            .filter(|al| al.requestor_small_id == small_id || al.recipient_small_id == small_id)
            .collect()
    }

    /// TS `PlayerImpl.hasEmbargoAgainst` - `a` has an active embargo against `b`.
    pub fn has_embargo_against(&self, a: u16, b: u16) -> bool {
        self.player_by_small_id(a)
            .is_some_and(|p| p.embargoes.contains_key(&b))
    }

    /// TS `PlayerImpl.canTrade` - false if either side embargoes the other (checked both
    /// directions), or if `a == b`.
    pub fn can_trade(&self, a: u16, b: u16) -> bool {
        if a == b {
            return false;
        }
        !(self.has_embargo_against(b, a) || self.has_embargo_against(a, b))
    }

    /// TS `PlayerImpl.addEmbargo` - `a` embargoes `b`. A no-op if `a` already has a
    /// non-temporary embargo against `b` (a permanent embargo is never downgraded/refreshed).
    pub fn add_embargo(&mut self, a: u16, b: u16, is_temporary: bool, tick: u32) {
        if let Some(p) = self.player_by_small_id_mut(a) {
            if let Some(existing) = p.embargoes.get(&b) {
                if !existing.is_temporary {
                    return;
                }
            }
            p.embargoes.set(
                b,
                EmbargoState {
                    created_at: tick,
                    is_temporary,
                },
            );
        }
    }

    /// TS `PlayerImpl.stopEmbargo` - `a` removes its embargo against `b`, if any.
    pub fn stop_embargo(&mut self, a: u16, b: u16) {
        if let Some(p) = self.player_by_small_id_mut(a) {
            p.embargoes.remove(&b);
        }
    }

    /// TS `PlayerImpl.endTemporaryEmbargo` - removes `a`'s embargo against `b` unless it's a
    /// permanent (non-temporary) one, which is left in place.
    pub fn end_temporary_embargo(&mut self, a: u16, b: u16) {
        let keep = self
            .player_by_small_id(a)
            .and_then(|p| p.embargoes.get(&b))
            .is_some_and(|e| !e.is_temporary);
        if keep {
            return;
        }
        self.stop_embargo(a, b);
    }

    /// TS `PlayerImpl.canEmbargoAll` - cooldown gate, then requires at least one eligible
    /// (non-self, non-Bot, non-teammate) player to exist.
    pub fn can_embargo_all(&self, small_id: u16) -> bool {
        let Some(p) = self.player_by_small_id(small_id) else {
            return false;
        };
        if (self.ticks as i64) - (p.last_embargo_all_tick as i64)
            < self.wire.embargo_all_cooldown() as i64
        {
            return false;
        }
        self.all_players().iter().any(|other| {
            other.small_id != small_id
                && other.player_type != PlayerType::Bot
                && !self.players_on_same_team(small_id, other.small_id)
        })
    }

    /// TS `PlayerImpl.recordEmbargoAll`.
    pub fn record_embargo_all(&mut self, small_id: u16) {
        let tick = self.ticks;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.last_embargo_all_tick = tick as i32;
        }
    }

    pub fn add_gold(&mut self, small_id: u16, amount: i64) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold += amount;
        }
    }

    /// TS `Player.removeGold` - returns the amount actually removed (clamped
    /// to the sender's balance).
    pub fn remove_gold(&mut self, small_id: u16, amount: i64) -> i64 {
        if amount <= 0 {
            return 0;
        }
        let Some(p) = self.player_by_small_id_mut(small_id) else {
            return 0;
        };
        let actual_removed = p.gold.min(amount);
        p.gold -= actual_removed;
        actual_removed
    }

    pub fn is_allied_with(&self, a: u16, b: u16) -> bool {
        if a == b {
            return false;
        }
        // TS `PlayerImpl.isAlliedWith` / `allianceWith`: membership in
        // `_alliances` only. Expiry is a separate `PlayerExecution` sweep
        // (`expiresAt() <= ticks` → `expire()`); filtering on `expires_at`
        // here made alliances disappear one tick early for every reader
        // (attack bordering-friends, betrayal, donations, …) while TS still
        // treated them as live until that player's `PlayerExecution` ran.
        self.alliances.iter().any(|al| {
            (al.requestor_small_id == a && al.recipient_small_id == b)
                || (al.requestor_small_id == b && al.recipient_small_id == a)
        })
    }

    pub fn pending_alliance_request(
        &self,
        requestor: u16,
        recipient: u16,
    ) -> Option<&AllianceRequestState> {
        self.alliance_requests.iter().find(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == requestor
                && r.recipient_small_id == recipient
        })
    }

    pub fn create_alliance_request(&mut self, requestor: u16, recipient: u16, tick: u32) -> bool {
        if self.wire.disable_alliances() {
            return false;
        }
        if self.is_allied_with(requestor, recipient) {
            return false;
        }
        if self.alliance_requests.iter().any(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == requestor
                && r.recipient_small_id == recipient
        }) {
            return false;
        }
        if self.alliance_requests.iter().any(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == recipient
                && r.recipient_small_id == requestor
        }) {
            self.accept_alliance_pair(recipient, requestor, tick);
            // TS `AllianceRequestExecution.init`: only simultaneous cross-requests
            // improve relations. A later manual `AllianceRequest.accept()` does not.
            self.update_relation(requestor, recipient, 100);
            self.update_relation(recipient, requestor, 100);
            // TS `AllianceRequestExecution.init`'s cross-request auto-accept branch clears
            // any *automatically created* (temporary) embargoes between the pair; this is
            // scoped to this branch specifically - `accept_alliance_request`'s manual accept
            // path (`GameImpl.acceptAllianceRequest`) never touches embargoes in TS.
            self.end_temporary_embargo(requestor, recipient);
            self.end_temporary_embargo(recipient, requestor);
            self.cancel_nukes_between_new_allies(requestor, recipient);
            return true;
        }
        self.alliance_requests.push(AllianceRequestState {
            requestor_small_id: requestor,
            recipient_small_id: recipient,
            created_at: tick,
            status: AllianceRequestStatus::Pending,
        });
        true
    }

    fn cancel_nukes_between_new_allies(&mut self, first: u16, second: u16) {
        use crate::core::schemas::unit_type::{ATOM_BOMB, HYDROGEN_BOMB};

        let mut neutralized = Vec::new();
        for (launcher, ally) in [(first, second), (second, first)] {
            let threatening = self
                .player_by_small_id(launcher)
                .map(|player| {
                    player
                        .units
                        .iter()
                        .filter(|unit| {
                            matches!(unit.unit_type.as_str(), ATOM_BOMB | HYDROGEN_BOMB)
                                && unit.target_tile.is_some_and(|target| {
                                    crate::execution::nuke_execution::would_nuke_break_alliance(
                                        self,
                                        target,
                                        &unit.unit_type,
                                        ally,
                                    )
                                })
                        })
                        .map(|unit| unit.id)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            neutralized.extend(threatening.into_iter().map(|unit_id| (launcher, unit_id)));
        }
        for (launcher, unit_id) in neutralized {
            self.remove_unit(launcher, unit_id);
        }
    }

    fn accept_alliance_pair(&mut self, requestor: u16, recipient: u16, tick: u32) {
        // TS `GameImpl.acceptAllianceRequest` removes the request from the live
        // `allianceRequests` list but pushes it onto the requestor's
        // `pastOutgoingAllianceRequests`, which `canSendAllianceRequest` uses for
        // the alliance-request cooldown. Native previously `retain`-deleted the
        // pending rows, so after accept (and a later break) the sender had no
        // cooldown memory and could re-request immediately - burning extra
        // `getAllianceDecision` PRNG draws TS never makes (NA080 Louisiana→Alabama
        // @1492 accept → @1618 native re-request vs TS cooldown).
        for r in &mut self.alliance_requests {
            if r.status != AllianceRequestStatus::Pending {
                continue;
            }
            if (r.requestor_small_id == requestor && r.recipient_small_id == recipient)
                || (r.requestor_small_id == recipient && r.recipient_small_id == requestor)
            {
                r.status = AllianceRequestStatus::Accepted;
            }
        }
        let id = self.next_alliance_id;
        self.next_alliance_id += 1;
        let duration = self.wire.alliance_duration();
        self.alliances.push(AllianceState {
            id,
            requestor_small_id: requestor,
            recipient_small_id: recipient,
            created_at: tick,
            expires_at: tick + duration,
            extension_requested_requestor: false,
            extension_requested_recipient: false,
        });
    }

    pub fn accept_alliance_request(&mut self, requestor: u16, recipient: u16, tick: u32) {
        if !self.alliance_requests.iter().any(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == requestor
                && r.recipient_small_id == recipient
        }) {
            return;
        }
        self.accept_alliance_pair(requestor, recipient, tick);
    }

    pub fn reject_alliance_request(&mut self, requestor: u16, recipient: u16) {
        for r in &mut self.alliance_requests {
            if r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == requestor
                && r.recipient_small_id == recipient
            {
                r.status = AllianceRequestStatus::Rejected;
            }
        }
    }

    pub fn break_alliance_between(&mut self, breaker: u16, other: u16) {
        if !self.is_allied_with(breaker, other) {
            return;
        }
        if !self.is_traitor(other)
            && !self
                .player_by_small_id(other)
                .is_some_and(|player| player.is_disconnected)
        {
            self.mark_traitor(breaker);
        }
        self.alliances.retain(|al| {
            !((al.requestor_small_id == breaker && al.recipient_small_id == other)
                || (al.requestor_small_id == other && al.recipient_small_id == breaker))
        });
    }

    /// TS `GameImpl.breakAlliance` - marks `breaker` a traitor (unless `other` already is)
    /// and detaches the alliance, WITHOUT any relation change. Used by `NukeExecution`,
    /// which applies its own (differently-directed) relation update afterward.
    pub fn break_alliance_silently(&mut self, breaker: u16, other: u16) -> bool {
        if !self.is_allied_with(breaker, other) {
            return false;
        }
        if !self.is_traitor(other) {
            self.mark_traitor(breaker);
        }
        self.alliances.retain(|al| {
            !((al.requestor_small_id == breaker && al.recipient_small_id == other)
                || (al.requestor_small_id == other && al.recipient_small_id == breaker))
        });
        true
    }

    pub fn expire_alliances_for(&mut self, small_id: u16) {
        let tick = self.ticks;
        let expired: Vec<(u16, u16)> = self
            .alliances
            .iter()
            .filter(|al| {
                al.expires_at <= tick
                    && (al.requestor_small_id == small_id || al.recipient_small_id == small_id)
            })
            .map(|al| (al.requestor_small_id, al.recipient_small_id))
            .collect();
        for (a, b) in expired {
            self.alliances.retain(|al| {
                !((al.requestor_small_id == a && al.recipient_small_id == b)
                    || (al.requestor_small_id == b && al.recipient_small_id == a))
            });
        }
    }

    /// TS `PlayerExecution.tick`'s temporary-embargo expiry loop.
    pub fn expire_temporary_embargoes_for(&mut self, small_id: u16) {
        let tick = self.ticks;
        let duration = self.wire.temporary_embargo_duration();
        let expired: Vec<u16> = self
            .player_by_small_id(small_id)
            .map(|p| {
                p.embargoes
                    .iter()
                    .filter(|(_, e)| e.is_temporary && tick.saturating_sub(e.created_at) > duration)
                    .map(|(target, _)| target)
                    .collect()
            })
            .unwrap_or_default();
        for target in expired {
            self.stop_embargo(small_id, target);
        }
    }

    /// TS `GameImpl.removeAlliancesByPlayerSilently` - drop every alliance
    /// involving `small_id` with no relation/traitor side effects (used on
    /// player death, where the player is gone regardless).
    pub fn remove_all_alliances_for(&mut self, small_id: u16) {
        self.alliances
            .retain(|al| al.requestor_small_id != small_id && al.recipient_small_id != small_id);
    }

    pub fn add_alliance_extension_request(&mut self, from: u16, to: u16) {
        let tick = self.ticks;
        let duration = self.wire.alliance_duration();
        let Some(idx) = self.alliances.iter().position(|al| {
            (al.requestor_small_id == from && al.recipient_small_id == to)
                || (al.requestor_small_id == to && al.recipient_small_id == from)
        }) else {
            return;
        };
        let al = &mut self.alliances[idx];
        if al.requestor_small_id == from {
            al.extension_requested_requestor = true;
        } else {
            al.extension_requested_recipient = true;
        }
        let both = al.extension_requested_requestor && al.extension_requested_recipient;
        if both {
            al.extension_requested_requestor = false;
            al.extension_requested_recipient = false;
            al.expires_at = tick + duration;
        }
    }

    pub fn mark_traitor(&mut self, small_id: u16) {
        let tick = self.ticks as i32;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.marked_traitor_tick = tick;
        }
    }

    /// TS `PlayerImpl.inDoomsdayClock` - a dead player is never "in doomsday
    /// clock": nothing clears the mark on death (`DoomsdayClockExecution` only
    /// ever processes alive contenders), so this gates on `alive` too, to
    /// avoid a stuck skull/panel and per-tick update churn for an eliminated
    /// player.
    pub fn in_doomsday_clock(&self, small_id: u16) -> bool {
        self.player_by_small_id(small_id)
            .is_some_and(|p| p.alive && p.marked_doomsday_clock_tick >= 0)
    }

    /// TS `PlayerImpl.doomsdayClockTicks` - ticks spent continuously below the
    /// bar (0 when not marked or dead).
    pub fn doomsday_clock_ticks(&self, small_id: u16) -> i64 {
        if !self.in_doomsday_clock(small_id) {
            return 0;
        }
        let Some(p) = self.player_by_small_id(small_id) else {
            return 0;
        };
        self.ticks as i64 - p.marked_doomsday_clock_tick as i64
    }

    /// TS `PlayerImpl.enterDoomsdayClock` - only the first drop below the bar
    /// stamps the tick; staying below on later ticks is a no-op so the ticks-
    /// under counter keeps counting from the original drop.
    pub fn enter_doomsday_clock(&mut self, small_id: u16) {
        let tick = self.ticks as i32;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            if p.marked_doomsday_clock_tick < 0 {
                p.marked_doomsday_clock_tick = tick;
            }
        }
    }

    /// TS `PlayerImpl.clearDoomsdayClock`.
    pub fn clear_doomsday_clock(&mut self, small_id: u16) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.marked_doomsday_clock_tick = -1;
        }
    }

    /// TS `GameImpl.elapsedGameSeconds` (`ticksSinceStart() / 10`). Integer
    /// seconds: `doomsday_clock.rs`'s wave-schedule math is integer-only, and
    /// `ticks_since_start()` only ever changes by whole ticks anyway.
    pub fn elapsed_game_seconds(&self) -> i64 {
        self.ticks_since_start() as i64 / 10
    }

    /// TS `Player.isOnSameTeam(other)`, called with the winner's `PlayerID` (not a
    /// `small_id`) at its only call site (`nation_emoji.rs`'s `congratulate_winner`).
    /// Was previously a hardcoded stub always returning `false` - every nation
    /// unconditionally sent the "congratulate winner" emoji even when the winner was
    /// its own teammate, which TS's real `isOnSameTeam` suppresses.
    pub fn is_on_same_team(&self, a: u16, winner_id: &str) -> bool {
        let Some(winner) = self.player_by_id(winner_id) else {
            return false;
        };
        self.players_on_same_team(a, winner.small_id)
    }

    pub fn players_on_same_team(&self, a: u16, b: u16) -> bool {
        let Some(pa) = self.player_by_small_id(a) else {
            return false;
        };
        let Some(pb) = self.player_by_small_id(b) else {
            return false;
        };
        let Some(ta) = pa.team.as_deref() else {
            return false;
        };
        let Some(tb) = pb.team.as_deref() else {
            return false;
        };
        if ta == BOT_TEAM || tb == BOT_TEAM {
            return false;
        }
        ta == tb
    }

    pub fn is_friendly(&self, a: u16, b: u16) -> bool {
        self.is_friendly_ex(a, b, false)
    }

    /// TS `PlayerImpl.isFriendly(other, treatAFKFriendly)`. `treat_afk_friendly`
    /// keeps a disconnected teammate/ally friendly instead of the normal
    /// disconnected-is-never-friendly rule - used by `WarshipExecution`'s
    /// target filter so warships don't shoot at a disconnected team mate's
    /// ships (they're still captured on conquest, just not shelled first).
    pub fn is_friendly_ex(&self, a: u16, b: u16, treat_afk_friendly: bool) -> bool {
        if a == b {
            return true;
        }
        // Match TS `isFriendly`: disconnected players are never considered friendly,
        // even if allied. This keeps `bordering_enemies` and `bordering_friends` lists
        // in sync with TS so that RNG consumption in `maybe_send_alliance_requests`
        // (and attack strategies) is identical between the two engines.
        if !treat_afk_friendly
            && self
                .player_by_small_id(b)
                .map_or(false, |p| p.is_disconnected)
        {
            return false;
        }
        // TS `Player.isFriendly`: teammates count as friendly even without a
        // formal alliance (e.g. lets teammates donate troops/gold to each other).
        self.players_on_same_team(a, b) || self.is_allied_with(a, b)
    }

    pub fn ticks_since_start(&self) -> u32 {
        if self.spawn_phase {
            return 0;
        }
        self.spawn_end_tick
            .map(|start| self.ticks.saturating_sub(start))
            .unwrap_or(0)
    }

    pub fn is_spawn_immunity_active(&self) -> bool {
        self.spawn_phase || self.ticks_since_start() < self.wire.spawn_immunity_duration()
    }

    pub fn is_nation_spawn_immunity_active(&self) -> bool {
        self.spawn_phase || self.ticks_since_start() < self.wire.nation_spawn_immunity_duration()
    }

    pub fn is_player_immune(&self, small_id: u16) -> bool {
        let Some(p) = self.player_by_small_id(small_id) else {
            return false;
        };
        match p.player_type {
            PlayerType::Human => self.is_spawn_immunity_active(),
            PlayerType::Nation => self.is_nation_spawn_immunity_active(),
            _ => false,
        }
    }

    pub fn can_attack_player(&self, attacker_small_id: u16, defender_small_id: u16) -> bool {
        self.can_attack_player_ex(attacker_small_id, defender_small_id, false)
    }

    /// TS `PlayerImpl.canAttackPlayer(player, treatAFKFriendly)`.
    pub fn can_attack_player_ex(
        &self,
        attacker_small_id: u16,
        defender_small_id: u16,
        treat_afk_friendly: bool,
    ) -> bool {
        if self.is_friendly_ex(attacker_small_id, defender_small_id, treat_afk_friendly) {
            return false;
        }
        let attacker_type = self
            .player_by_small_id(attacker_small_id)
            .map(|p| p.player_type)
            .unwrap_or(PlayerType::Human);
        if attacker_type != PlayerType::Human {
            return true;
        }
        !self.is_player_immune(defender_small_id)
    }

    pub fn is_traitor(&self, small_id: u16) -> bool {
        self.traitor_remaining_ticks(small_id) > 0
    }

    fn traitor_remaining_ticks(&self, small_id: u16) -> i32 {
        let Some(p) = self.player_by_small_id(small_id) else {
            return 0;
        };
        if p.marked_traitor_tick < 0 {
            return 0;
        }
        let elapsed = self.ticks as i32 - p.marked_traitor_tick;
        let duration = 300;
        let remaining = duration - elapsed;
        remaining.max(0)
    }

    pub fn allied_small_ids(&self, small_id: u16) -> Vec<u16> {
        // Match TS `PlayerImpl.allies()` / `alliances()`: no `expires_at` filter
        // (see `is_allied_with`).
        self.alliances
            .iter()
            .filter_map(|al| {
                if al.requestor_small_id == small_id {
                    Some(al.recipient_small_id)
                } else if al.recipient_small_id == small_id {
                    Some(al.requestor_small_id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn alliance_count(&self, small_id: u16) -> usize {
        self.allied_small_ids(small_id).len()
    }

    pub fn can_send_emoji(&self, sender_small_id: u16, recipient: Option<u16>) -> bool {
        if recipient.is_some_and(|r| r == sender_small_id) {
            return false;
        }
        // TS `Config.emojiMessageCooldown()` = 5 * 10.
        const EMOJI_MESSAGE_COOLDOWN: u32 = 50;
        let Some(sender) = self.player_by_small_id(sender_small_id) else {
            return false;
        };
        for &(prev_recipient, created_at) in &sender.outgoing_emoji_sends {
            if prev_recipient == recipient && self.ticks.saturating_sub(created_at) < EMOJI_MESSAGE_COOLDOWN
            {
                return false;
            }
        }
        true
    }

    /// Record a sent emoji for `can_send_emoji` cooldown (TS `outgoingEmojis_`).
    ///
    /// Native emoji is hash-neutral (no `EmojiExecution`), but TS records
    /// `createdAt` when that execution *ticks* — always the tick after
    /// `addExecution` (init same tick, tick next). Store `ticks + 1` so the
    /// 50-tick cooldown window matches TS.
    pub fn record_emoji_send(&mut self, sender_small_id: u16, recipient: Option<u16>) {
        let created_at = self.ticks.saturating_add(1);
        if let Some(sender) = self.player_by_small_id_mut(sender_small_id) {
            sender.outgoing_emoji_sends.push((recipient, created_at));
        }
    }

    pub fn can_donate_troops(&self, sender_small_id: u16, recipient_small_id: u16) -> bool {
        if sender_small_id == recipient_small_id {
            return false;
        }
        let Some(sender) = self.player_by_small_id(sender_small_id) else {
            return false;
        };
        let Some(recipient) = self.player_by_small_id(recipient_small_id) else {
            return false;
        };
        if !sender.alive || !recipient.alive || sender.tiles_owned == 0 {
            return false;
        }
        if !self.is_friendly(sender_small_id, recipient_small_id) {
            return false;
        }
        if recipient.player_type == crate::game::PlayerType::Human
            && !self.wire.game_config().donate_troops
        {
            return false;
        }
        true
    }

    pub fn can_donate_gold(&self, sender_small_id: u16, recipient_small_id: u16) -> bool {
        if sender_small_id == recipient_small_id {
            return false;
        }
        let Some(sender) = self.player_by_small_id(sender_small_id) else {
            return false;
        };
        let Some(recipient) = self.player_by_small_id(recipient_small_id) else {
            return false;
        };
        if !sender.alive || !recipient.alive || sender.tiles_owned == 0 {
            return false;
        }
        if !self.is_friendly(sender_small_id, recipient_small_id) {
            return false;
        }
        if recipient.player_type == crate::game::PlayerType::Human
            && !self.wire.game_config().donate_gold
        {
            return false;
        }
        true
    }

    pub fn can_send_alliance_request(&self, sender_small_id: u16, recipient_small_id: u16) -> bool {
        if self.wire.disable_alliances() {
            return false;
        }
        if sender_small_id == recipient_small_id {
            return false;
        }
        let Some(sender) = self.player_by_small_id(sender_small_id) else {
            return false;
        };
        let Some(recipient) = self.player_by_small_id(recipient_small_id) else {
            return false;
        };
        if sender.is_disconnected || recipient.is_disconnected {
            return false;
        }
        // TS `PlayerImpl.canSendAllianceRequest`: `this.isFriendly(other) ||
        // !this.isAlive()` - only the SENDER's aliveness (`isAlive() ==
        // this._tiles.size > 0`) gates the request; a dead recipient (0
        // tiles) is not checked at all, so TS allows requesting an alliance
        // with an eliminated player. Native previously also required
        // `recipient.tiles_owned != 0`, which is stricter than TS.
        if sender.tiles_owned == 0 {
            return false;
        }
        if self.is_friendly(sender_small_id, recipient_small_id) {
            return false;
        }
        if self.alliance_requests.iter().any(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == sender_small_id
                && r.recipient_small_id == recipient_small_id
        }) {
            return false;
        }
        if self.alliance_requests.iter().any(|r| {
            r.status == AllianceRequestStatus::Pending
                && r.requestor_small_id == recipient_small_id
                && r.recipient_small_id == sender_small_id
        }) {
            return true;
        }
        let cooldown = self.wire.alliance_request_cooldown();
        let recent = self
            .alliance_requests
            .iter()
            .filter(|r| {
                r.requestor_small_id == sender_small_id
                    && r.recipient_small_id == recipient_small_id
                    && r.status != AllianceRequestStatus::Pending
            })
            .map(|r| r.created_at)
            .max();
        match recent {
            None => true,
            Some(created) => self.ticks.saturating_sub(created) >= cooldown,
        }
    }

    pub fn incoming_alliance_requests(&self, small_id: u16) -> Vec<IncomingAllianceRequest> {
        self.alliance_requests
            .iter()
            .filter(|r| {
                r.status == AllianceRequestStatus::Pending && r.recipient_small_id == small_id
            })
            .map(|r| IncomingAllianceRequest {
                requestor_small_id: r.requestor_small_id,
                created_at: r.created_at,
            })
            .collect()
    }

    pub fn alliance_extension_candidates(&self, small_id: u16) -> Vec<AllianceExtensionCandidate> {
        let mut out = Vec::new();
        for al in &self.alliances {
            if al.expires_at <= self.ticks {
                continue;
            }
            let other = if al.requestor_small_id == small_id {
                al.recipient_small_id
            } else if al.recipient_small_id == small_id {
                al.requestor_small_id
            } else {
                continue;
            };
            let only_one = al.extension_requested_requestor ^ al.extension_requested_recipient;
            if only_one {
                out.push(AllianceExtensionCandidate {
                    other_small_id: other,
                });
            }
        }
        out
    }

    pub fn too_close_to_existing_spawn(
        &self,
        center: TileRef,
        min_dist: u32,
        except_small_id: Option<u16>,
    ) -> bool {
        for p in &self.players {
            if except_small_id == Some(p.small_id) {
                continue;
            }
            let Some(spawn) = p.spawn_tile else {
                continue;
            };
            if self.manhattan_dist(spawn, center) < min_dist {
                return true;
            }
        }
        false
    }

    pub fn shares_land_border_with(&self, a: u16, b: u16) -> bool {
        let mut found = false;
        self.for_each_border_tile(a, |tile| {
            if found {
                return;
            }
            self.map.for_each_neighbor4(tile, |neighbor| {
                if self.map.owner_id(neighbor) == b {
                    found = true;
                }
            });
        });
        found
    }

    /// TS `GameImpl.executeNextTick`  -  one pass over `execs` in insertion order.
    pub fn execute_next_tick(&mut self) -> TickUpdates {
        let tick = self.ticks;

        let game = self as *mut Game;
        let exec_len = self.execs.len();
        for i in 0..exec_len {
            let exec = &mut self.execs[i];
            if (!self.spawn_phase || exec.active_during_spawn()) && exec.is_active() {
                unsafe {
                    exec.tick(&mut *game, tick);
                }
            }
        }

        let drained = if self.uninit.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut self.uninit)
        };
        let mut inited = Vec::new();
        let mut still_uninit = Vec::new();
        for mut exec in drained {
            if !self.spawn_phase || exec.active_during_spawn() {
                self.init_merge_batch = Some(&mut inited as *mut Vec<ExecEnum>);
                unsafe {
                    exec.init(&mut *game, tick);
                }
                self.init_merge_batch = None;
                inited.push(exec);
            } else {
                still_uninit.push(exec);
            }
        }
        self.uninit = still_uninit;
        let spawn_phase = self.spawn_phase;
        self.execs.retain(|e| {
            if spawn_phase {
                !e.active_during_spawn() || e.is_active()
            } else {
                e.is_active()
            }
        });
        self.execs.append(&mut inited);

        let mut hash = None;
        if self.hash_enabled && tick % 10 == 0 {
            hash = Some(HashUpdate {
                tick,
                hash: game_hash(self),
            });
        }

        self.ticks += 1;
        TickUpdates {
            hash,
            win: self.winner.as_ref().map(|w| WinUpdate {
                tick,
                winner: w.clone(),
            }),
        }
    }
}

fn clamp_relation(value: f64) -> f64 {
    value.clamp(-100.0, 100.0)
}

fn decay_relation(value: f64) -> f64 {
    const DELTA: f64 = 0.05;
    let decayed = value - value.signum() * DELTA;
    if decayed.abs() < DELTA * 2.0 {
        0.0
    } else {
        decayed
    }
}

#[cfg(test)]
mod relation_tests {
    use super::{clamp_relation, decay_relation, Game, PlayerInfo, PlayerType, Relation};
    use crate::core::schemas::unit_type;

    fn add_human(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn relation_updates_are_bounded_like_typescript() {
        assert_eq!(clamp_relation(120.0), 100.0);
        assert_eq!(clamp_relation(-145.0), -100.0);
        assert_eq!(clamp_relation(35.0), 35.0);
    }

    #[test]
    fn relation_decay_moves_scores_toward_neutral() {
        assert_eq!(decay_relation(100.0), 99.95);
        assert_eq!(decay_relation(-100.0), -99.95);
        assert_eq!(decay_relation(0.1), 0.0);
        assert_eq!(decay_relation(-0.1), 0.0);
        assert_eq!(decay_relation(0.0), 0.0);
    }

    #[test]
    fn manual_alliance_accept_does_not_change_relations() {
        let mut game = Game::default();
        let requestor = add_human(&mut game, "requestor");
        let recipient = add_human(&mut game, "recipient");
        game.update_relation(requestor, recipient, -20);
        game.update_relation(recipient, requestor, -30);

        assert!(game.create_alliance_request(requestor, recipient, 10));
        game.accept_alliance_request(requestor, recipient, 11);

        assert!(game.is_allied_with(requestor, recipient));
        assert_eq!(game.relation_value(requestor, recipient), -20.0);
        assert_eq!(game.relation_value(recipient, requestor), -30.0);
    }

    // TS `PlayerImpl.test.ts` "Can't send alliance requests when dead" +
    // the asymmetric case it doesn't cover (see
    // `can_send_alliance_request`'s doc comment on the fix this caught).
    #[test]
    fn dead_sender_cannot_send_alliance_request() {
        let mut game = Game::default();
        let dead = add_human(&mut game, "dead");
        let alive = add_human(&mut game, "alive");
        game.player_by_small_id_mut(alive).unwrap().tiles_owned = 1;
        // `dead` never gained tiles (`tiles_owned == 0`, TS's `!isAlive()`).
        assert!(!game.can_send_alliance_request(dead, alive));
    }

    #[test]
    fn alive_sender_can_send_alliance_request_to_a_dead_recipient() {
        let mut game = Game::default();
        let alive = add_human(&mut game, "alive");
        let dead = add_human(&mut game, "dead");
        game.player_by_small_id_mut(alive).unwrap().tiles_owned = 1;
        assert!(game.can_send_alliance_request(alive, dead));
    }

    #[test]
    fn cross_alliance_requests_improve_relations() {
        let mut game = Game::default();
        let first = add_human(&mut game, "first");
        let second = add_human(&mut game, "second");
        game.update_relation(first, second, -20);
        game.update_relation(second, first, -30);

        assert!(game.create_alliance_request(first, second, 10));
        assert!(game.create_alliance_request(second, first, 11));

        assert!(game.is_allied_with(first, second));
        assert_eq!(game.relation(first, second), Relation::Friendly);
        assert_eq!(game.relation(second, first), Relation::Friendly);
    }

    #[test]
    fn only_cross_request_acceptance_cancels_nukes_threatening_new_allies() {
        fn add_threatening_nuke(game: &mut Game, launcher: u16, target: u16) -> i32 {
            game.build_unit(target, unit_type::CITY, 0);
            let nuke = game.build_unit(launcher, unit_type::ATOM_BOMB, 0);
            game.unit_mut(launcher, nuke).unwrap().target_tile = Some(0);
            nuke
        }

        let mut manual = Game::default();
        let manual_first = add_human(&mut manual, "manual-first");
        let manual_second = add_human(&mut manual, "manual-second");
        let manual_nuke = add_threatening_nuke(&mut manual, manual_first, manual_second);
        assert!(manual.create_alliance_request(manual_first, manual_second, 10));
        manual.accept_alliance_request(manual_first, manual_second, 11);
        assert!(manual.unit_exists(manual_first, manual_nuke));

        let mut reciprocal = Game::default();
        let reciprocal_first = add_human(&mut reciprocal, "reciprocal-first");
        let reciprocal_second = add_human(&mut reciprocal, "reciprocal-second");
        let reciprocal_nuke =
            add_threatening_nuke(&mut reciprocal, reciprocal_first, reciprocal_second);
        assert!(reciprocal.create_alliance_request(reciprocal_first, reciprocal_second, 10));
        assert!(reciprocal.create_alliance_request(reciprocal_second, reciprocal_first, 11));
        assert!(!reciprocal.unit_exists(reciprocal_first, reciprocal_nuke));
    }
}

// Ported from PlayerAllianceList.test.ts. TS maintains a per-player,
// incrementally-updated `Alliance` object list (`Player.alliances()`) as a
// performance optimization over scanning the game's global alliance
// collection; native has no equivalent per-player list at all - it's a flat
// `Vec<AllianceState>` on `Game`, queried on demand. These tests therefore
// verify the same *observable* invariants the TS list is meant to uphold
// (both sides see the same alliance, breaking/expiring updates both sides,
// multiple alliances tracked independently, death clears everything) via
// `Game::player_alliances`/`allied_small_ids`/`alliance_count`/
// `is_allied_with` instead of asserting on a `player.alliances()`-shaped API
// that doesn't exist natively.
#[cfg(test)]
mod player_alliance_list_tests {
    use super::{Game, PlayerInfo, PlayerType};
    use crate::execution::alliance_exec::{AllianceRequestExecution, BreakAllianceExecution};
    use crate::execution::ExecEnum;

    fn add_human(game: &mut Game, id: &str) -> u16 {
        let sid = game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        game.player_by_small_id_mut(sid).unwrap().tiles_owned = 1;
        sid
    }

    fn ally(game: &mut Game, a: u16, a_id: &str, b: u16, b_id: &str) {
        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            a,
            b_id.into(),
        )));
        game.execute_next_tick();
        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            b,
            a_id.into(),
        )));
        game.execute_next_tick();
    }

    fn break_alliance(game: &mut Game, a: u16, b_id: &str) {
        game.add_execution(ExecEnum::BreakAlliance(BreakAllianceExecution::new(
            a,
            b_id.into(),
        )));
        game.execute_next_tick();
        game.execute_next_tick();
    }

    #[test]
    fn forming_an_alliance_is_visible_and_symmetric_for_both_participants() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        assert_eq!(game.alliance_count(player1), 0);
        assert_eq!(game.alliance_count(player2), 0);

        ally(&mut game, player1, "player1", player2, "player2");

        assert_eq!(game.alliance_count(player1), 1);
        assert_eq!(game.alliance_count(player2), 1);
        assert_eq!(game.allied_small_ids(player1), vec![player2]);
        assert_eq!(game.allied_small_ids(player2), vec![player1]);
    }

    #[test]
    fn alliance_state_agrees_with_is_allied_with() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        let player3 = add_human(&mut game, "player3");

        ally(&mut game, player1, "player1", player2, "player2");

        assert!(game.is_allied_with(player1, player2));
        assert!(!game.is_allied_with(player1, player3));
        assert_eq!(game.alliance_count(player3), 0);
    }

    #[test]
    fn breaking_an_alliance_removes_it_from_both_sides() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        ally(&mut game, player1, "player1", player2, "player2");
        assert_eq!(game.alliance_count(player1), 1);

        break_alliance(&mut game, player1, "player2");

        assert_eq!(game.alliance_count(player1), 0);
        assert_eq!(game.alliance_count(player2), 0);
        assert!(!game.is_allied_with(player1, player2));
    }

    #[test]
    fn expiring_an_alliance_removes_it_from_both_sides() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        ally(&mut game, player1, "player1", player2, "player2");
        assert_eq!(game.alliance_count(player1), 1);

        // TS `Alliance.expire()` force-expires a single alliance immediately,
        // bypassing the normal duration wait; native has no per-alliance handle
        // to call, so this simulates it by directly rewinding that alliance's
        // `expires_at` (the same field `expire_alliances_for` checks against
        // `game.ticks`) before running the normal expiry sweep.
        let tick = game.ticks;
        for al in &mut game.alliances {
            if al.requestor_small_id == player1 || al.recipient_small_id == player1 {
                al.expires_at = tick;
            }
        }
        // Until `expire_alliances_for` sweeps, TS still reports the alliance as
        // live (`isAlliedWith` only checks `_alliances` membership).
        assert!(game.is_allied_with(player1, player2));
        assert_eq!(game.alliance_count(player1), 1);

        game.expire_alliances_for(player1);

        assert_eq!(game.alliance_count(player1), 0);
        assert_eq!(game.alliance_count(player2), 0);
        assert!(!game.is_allied_with(player1, player2));
    }

    /// Regression: on the exact expiry tick, before that player's
    /// `PlayerExecution` sweeps, `is_allied_with` must still return true
    /// (TS keeps the alliance in `_alliances` until `expire()`). Found via
    /// curriculum-parity-v4 `curr-b010-s2-pangaea`: Canada/Mexico alliance
    /// expired at 5135; native treated them as non-allied during Canada's
    /// `maybeAttack` that same tick, adding Mexico to bordering enemies and
    /// desyncing alliance-request RNG so Canada skipped the Greenland attack.
    #[test]
    fn is_allied_with_stays_true_on_expiry_tick_until_swept() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        ally(&mut game, player1, "player1", player2, "player2");

        let expires_at = game.alliances[0].expires_at;
        game.ticks = expires_at;
        assert!(
            game.is_allied_with(player1, player2),
            "alliance must remain live on expires_at tick before PlayerExecution sweep"
        );
        assert!(game.is_friendly(player1, player2));

        game.expire_alliances_for(player1);
        assert!(!game.is_allied_with(player1, player2));
        assert!(!game.is_friendly(player1, player2));
    }

    #[test]
    fn a_player_tracks_multiple_alliances_independently() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        let player3 = add_human(&mut game, "player3");
        ally(&mut game, player1, "player1", player2, "player2");
        ally(&mut game, player1, "player1", player3, "player3");

        assert_eq!(game.alliance_count(player1), 2);
        let mut others = game.allied_small_ids(player1);
        others.sort();
        let mut expected = vec![player2, player3];
        expected.sort();
        assert_eq!(others, expected);

        break_alliance(&mut game, player1, "player2");

        assert_eq!(game.alliance_count(player1), 1);
        assert_eq!(game.allied_small_ids(player1), vec![player3]);
        assert_eq!(game.alliance_count(player2), 0);
        assert_eq!(game.alliance_count(player3), 1);
    }

    #[test]
    fn remove_all_alliances_for_clears_the_player_and_every_partner() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        let player3 = add_human(&mut game, "player3");
        ally(&mut game, player1, "player1", player2, "player2");
        ally(&mut game, player1, "player1", player3, "player3");
        assert_eq!(game.alliance_count(player1), 2);

        game.remove_all_alliances_for(player1);

        assert_eq!(game.alliance_count(player1), 0);
        assert_eq!(game.alliance_count(player2), 0);
        assert_eq!(game.alliance_count(player3), 0);
    }
}

// Ported from AllianceAcceptNukes.test.ts. That TS test forces the
// nuke-cancellation weight check to fire regardless of tile-ownership shape
// by monkeypatching `nukeAllianceBreakThreshold` to 0; native hardcodes this
// threshold at TS's own default (100, see `WireConfig::nuke_alliance_break_threshold`)
// with no per-instance override, so these tests instead use the
// threshold-independent "nuke targets a tile with a structure owned by the
// ally" inclusion path in `Util.listNukeBreakAlliance` (mirrored natively by
// `list_nuke_break_alliance`'s `nearby_structures_any` branch) - same
// approach as `nuke_execution.rs`'s
// `nuke_at_a_players_structure_revokes_the_nukers_pending_alliance_request`.
// This also needs >1 distinct tile (to give the "only nukes between allied
// players" test two different nuke targets), so unlike every other test in
// this file it builds its own tiny synthetic map instead of using
// `Game::default()`'s degenerate 1x1 one.
#[cfg(test)]
mod alliance_accept_nukes_tests {
    use super::{Game, GameConfig, PlayerInfo, PlayerType};
    use crate::core::config::Config as WireConfig;
    use crate::core::schemas::unit_type;
    use crate::map::{GameMap, MapMeta};

    // Wide enough that two targets placed far apart don't both fall within a
    // single atom bomb's outer blast radius (30 tiles, see
    // `WireConfig::nuke_magnitudes`) of each other - otherwise a City near one
    // target would spuriously count as "in range" of a nuke aimed at the other.
    const MAP_WIDTH: u32 = 100;

    fn small_land_map_game() -> Game {
        let meta = MapMeta {
            width: MAP_WIDTH,
            height: 1,
            num_land_tiles: MAP_WIDTH,
        };
        let terrain = vec![0x80u8; MAP_WIDTH as usize];
        let map = GameMap::from_terrain_bytes(&meta, &terrain).unwrap();
        let mini_map = GameMap::from_terrain_bytes(&meta, &terrain).unwrap();
        let wire_cfg = crate::core::schemas::GameConfig {
            game_map: "Onion".into(),
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
            spawn_immunity_duration: None,
            starting_gold: None,
            gold_multiplier: None,
            max_timer_value: None,
            ranked_type: None,
        };
        let mut game = Game::new(
            String::new(),
            GameConfig::default(),
            WireConfig::new(wire_cfg, false),
            map,
            mini_map,
            None,
        );
        game.end_spawn_phase();
        game
    }

    fn add_human(game: &mut Game, id: &str) -> u16 {
        let sid = game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        game.player_by_small_id_mut(sid).unwrap().tiles_owned = 1;
        sid
    }

    #[test]
    fn accepting_an_alliance_destroys_in_flight_nukes_between_the_new_allies() {
        let mut game = small_land_map_game();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");

        let target = game.map.ref_xy(1, 0);
        game.build_unit(player2, unit_type::CITY, target);
        let nuke = game.build_unit(player1, unit_type::ATOM_BOMB, game.map.ref_xy(0, 0));
        game.unit_mut(player1, nuke).unwrap().target_tile = Some(target);

        assert!(!game.is_allied_with(player1, player2));

        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));

        assert!(game.is_allied_with(player1, player2));
        assert!(!game.unit_exists(player1, nuke));
    }

    #[test]
    fn accepting_an_alliance_destroys_only_nukes_targeting_the_new_ally() {
        let mut game = small_land_map_game();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        let player3 = add_human(&mut game, "player3");

        let target2 = game.map.ref_xy(1, 0);
        let target3 = game.map.ref_xy(90, 0);
        game.build_unit(player2, unit_type::CITY, target2);
        game.build_unit(player3, unit_type::CITY, target3);

        let nuke_to_2 = game.build_unit(player1, unit_type::ATOM_BOMB, game.map.ref_xy(0, 0));
        game.unit_mut(player1, nuke_to_2).unwrap().target_tile = Some(target2);
        let nuke_to_3 = game.build_unit(player1, unit_type::ATOM_BOMB, game.map.ref_xy(0, 0));
        game.unit_mut(player1, nuke_to_3).unwrap().target_tile = Some(target3);

        assert!(!game.is_allied_with(player1, player2));

        // Both requests created in the same call, mirroring the TS test's "both
        // added in the same tick" setup.
        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));

        assert!(game.is_allied_with(player1, player2));
        assert!(!game.unit_exists(player1, nuke_to_2));
        assert!(game.unit_exists(player1, nuke_to_3));
        assert_eq!(
            game.unit_mut(player1, nuke_to_3).unwrap().target_tile,
            Some(target3)
        );
    }
}

// Ported from Disconnected.test.ts "Conqueror gets conquered disconnected
// team member's transport- and warships": `conquer_player`'s ship-capture
// branch doesn't depend on map/water geometry, so it's exercised directly
// rather than through a full attack simulation.
#[cfg(test)]
mod conquer_player_tests {
    use super::{Game, Player, PlayerType};
    use crate::core::schemas::unit_type;

    #[test]
    fn conquering_a_disconnected_team_mate_captures_their_ships() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "conqueror".to_string(),
            small_id: 1,
            player_type: PlayerType::Human,
            team: Some("CLAN".to_string()),
            ..Default::default()
        });
        game.add_player(Player {
            id: "conquered".to_string(),
            small_id: 2,
            player_type: PlayerType::Human,
            team: Some("CLAN".to_string()),
            is_disconnected: true,
            ..Default::default()
        });
        let warship = game.build_unit(2, unit_type::WARSHIP, 0);
        let transport = game.build_unit(2, unit_type::TRANSPORT, 0);

        game.conquer_player(1, 2);

        assert!(game.unit_exists(1, warship));
        assert!(game.unit_exists(1, transport));
        assert!(!game.unit_exists(2, warship));
        assert!(!game.unit_exists(2, transport));
    }

    #[test]
    fn conquering_a_connected_team_mate_does_not_capture_their_ships() {
        // TS's `conquerPlayer` only transfers ships when the conquered
        // player `isDisconnected()`; capturing a still-connected teammate's
        // ships (e.g. after they're simply eliminated) would be wrong.
        let mut game = Game::default();
        game.add_player(Player {
            id: "conqueror".to_string(),
            small_id: 1,
            player_type: PlayerType::Human,
            team: Some("CLAN".to_string()),
            ..Default::default()
        });
        game.add_player(Player {
            id: "conquered".to_string(),
            small_id: 2,
            player_type: PlayerType::Human,
            team: Some("CLAN".to_string()),
            ..Default::default()
        });
        let warship = game.build_unit(2, unit_type::WARSHIP, 0);

        game.conquer_player(1, 2);

        assert!(!game.unit_exists(1, warship));
        assert!(game.unit_exists(2, warship));
    }

    #[test]
    fn conquering_a_disconnected_non_team_mate_does_not_capture_their_ships() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "conqueror".to_string(),
            small_id: 1,
            player_type: PlayerType::Human,
            ..Default::default()
        });
        game.add_player(Player {
            id: "conquered".to_string(),
            small_id: 2,
            player_type: PlayerType::Human,
            is_disconnected: true,
            ..Default::default()
        });
        let warship = game.build_unit(2, unit_type::WARSHIP, 0);

        game.conquer_player(1, 2);

        assert!(!game.unit_exists(1, warship));
        assert!(game.unit_exists(2, warship));
    }
}

// Ported from ConquerGold.test.ts. TS exposes the gold-share computation as
// a standalone `config().conquerGoldAmount(player)`; native inlines the same
// match directly into `conquer_player`, so both of TS's describe blocks
// ("DefaultConfig.conquerGoldAmount" and "Conquest gold transfer") collapse
// into one set of `conquer_player` assertions here.
#[cfg(test)]
mod conquer_gold_tests {
    use super::{Game, Player, PlayerType};

    fn add_player_with_gold(
        game: &mut Game,
        id: &str,
        small_id: u16,
        player_type: PlayerType,
        gold: i64,
    ) {
        game.add_player(Player {
            id: id.to_string(),
            small_id,
            player_type,
            gold,
            ..Default::default()
        });
    }

    #[test]
    fn conqueror_receives_full_gold_from_a_bot() {
        let mut game = Game::default();
        add_player_with_gold(&mut game, "conqueror", 1, PlayerType::Human, 0);
        add_player_with_gold(&mut game, "bot", 2, PlayerType::Bot, 1000);
        game.conquer_player(1, 2);
        assert_eq!(game.player_by_small_id(1).unwrap().gold, 1000);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 0);
    }

    #[test]
    fn conqueror_receives_full_gold_from_a_nation() {
        let mut game = Game::default();
        add_player_with_gold(&mut game, "conqueror", 1, PlayerType::Human, 0);
        add_player_with_gold(&mut game, "nation", 2, PlayerType::Nation, 800);
        game.conquer_player(1, 2);
        assert_eq!(game.player_by_small_id(1).unwrap().gold, 800);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 0);
    }

    #[test]
    fn conqueror_receives_half_gold_from_a_human_who_has_attacked() {
        let mut game = Game::default();
        add_player_with_gold(&mut game, "conqueror", 1, PlayerType::Human, 0);
        add_player_with_gold(&mut game, "victim", 2, PlayerType::Human, 1000);
        game.adjust_attacks_sent(2, 100.0);
        game.conquer_player(1, 2);
        assert_eq!(game.player_by_small_id(1).unwrap().gold, 500);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 0);
    }

    #[test]
    fn conqueror_receives_no_gold_from_a_human_who_never_attacked() {
        let mut game = Game::default();
        add_player_with_gold(&mut game, "conqueror", 1, PlayerType::Human, 0);
        add_player_with_gold(&mut game, "afk", 2, PlayerType::Human, 1000);
        game.conquer_player(1, 2);
        assert_eq!(game.player_by_small_id(1).unwrap().gold, 0);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 1000);
    }
}

#[cfg(test)]
mod spawn_distance_tests {
    use super::Player;

    #[test]
    fn distance_check_can_ignore_current_player_previous_spawn() {
        let mut game = crate::test_util::plains_game(20, 20);
        game.add_player(Player {
            id: "p1".into(),
            small_id: 1,
            spawn_tile: Some(game.ref_xy(10, 10)),
            ..Default::default()
        });
        game.add_player(Player {
            id: "p2".into(),
            small_id: 2,
            spawn_tile: Some(game.ref_xy(1, 1)),
            ..Default::default()
        });

        let near_p1 = game.ref_xy(11, 10);
        assert!(game.too_close_to_existing_spawn(near_p1, 5, None));
        assert!(!game.too_close_to_existing_spawn(near_p1, 5, Some(1)));
        assert!(game.too_close_to_existing_spawn(near_p1, 5, Some(2)));
    }
}

// TS `TileSet.test.ts` invariants, ported against `Player.owned_tiles`
// (TS `PlayerImpl._tiles`, a `TileSet`) rather than `TileSet` in isolation -
// see `execution/ordered_tiles.rs`'s `tests` module for the direct
// `OrderedTiles`/border-tiles port. `owned_tiles` is a plain `Vec<TileRef>`
// with `push`/`retain` (no dedicated ordered-set type), which a prior
// investigation (`docs/bot-ai-parity-nation-relations/README.md`, Bug 1
// area) argued by manual reasoning preserves the same insertion-order/
// delete-plus-readd-moves-to-end semantics as TS's `TileSet`. These tests
// confirm that directly via `Game::conquer`/`relinquish_tile` instead of
// trusting the argument.
#[cfg(test)]
mod owned_tiles_tests {
    use super::{Game, Player, PlayerType};

    fn add_bot(game: &mut Game, id: &str, small_id: u16) {
        game.add_player(Player {
            id: id.to_string(),
            small_id,
            player_type: PlayerType::Bot,
            ..Default::default()
        });
    }

    #[test]
    fn owned_tiles_preserve_insertion_order() {
        let mut game = crate::test_util::plains_game(10, 10);
        add_bot(&mut game, "p", 1);
        let tiles: Vec<_> = [(1, 1), (3, 3), (0, 0), (5, 5)]
            .iter()
            .map(|&(x, y)| game.map.ref_xy(x, y))
            .collect();
        for &t in &tiles {
            game.conquer(1, t);
        }
        assert_eq!(game.player_by_small_id(1).unwrap().owned_tiles, tiles);
    }

    #[test]
    fn relinquish_then_reconquer_moves_a_tile_to_the_end() {
        let mut game = crate::test_util::plains_game(10, 10);
        add_bot(&mut game, "p", 1);
        let t1 = game.map.ref_xy(1, 1);
        let t2 = game.map.ref_xy(2, 2);
        let t3 = game.map.ref_xy(3, 3);
        for t in [t1, t2, t3] {
            game.conquer(1, t);
        }
        game.relinquish_tile(t1);
        game.conquer(1, t1);
        assert_eq!(
            game.player_by_small_id(1).unwrap().owned_tiles,
            vec![t2, t3, t1]
        );
    }

    /// TS `GameImpl.conquer`: unconditionally deletes-then-re-adds to the
    /// owner's `TileSet` whenever the tile already has a player owner,
    /// without checking whether that owner is the same player conquering it
    /// again - so re-conquering your own already-owned tile also moves it
    /// to the end, same as a genuine delete+re-add. Native's `conquer_one`
    /// mirrors this (unconditional `retain` + `push` whenever `prev > 0`).
    #[test]
    fn reconquering_an_already_owned_tile_moves_it_to_the_end() {
        let mut game = crate::test_util::plains_game(10, 10);
        add_bot(&mut game, "p", 1);
        let t1 = game.map.ref_xy(1, 1);
        let t2 = game.map.ref_xy(2, 2);
        for t in [t1, t2] {
            game.conquer(1, t);
        }
        game.conquer(1, t1);
        assert_eq!(
            game.player_by_small_id(1).unwrap().owned_tiles,
            vec![t2, t1]
        );
    }

    #[test]
    fn owned_tiles_membership_stays_correct_after_relinquish() {
        let mut game = crate::test_util::plains_game(10, 10);
        add_bot(&mut game, "p", 1);
        let t1 = game.map.ref_xy(1, 1);
        let t2 = game.map.ref_xy(2, 2);
        game.conquer(1, t1);
        game.conquer(1, t2);
        game.relinquish_tile(t1);
        let owned = &game.player_by_small_id(1).unwrap().owned_tiles;
        assert!(!owned.contains(&t1));
        assert!(owned.contains(&t2));
        assert_eq!(game.player_by_small_id(1).unwrap().tiles_owned, 1);
    }
}

// TS `PlayerImpl.test.ts`.
#[cfg(test)]
mod player_tests {
    use super::{Game, Player, PlayerType};
    use crate::core::schemas::unit_type;

    fn add_bot(game: &mut Game, id: &str, small_id: u16) {
        game.add_player(Player {
            id: id.to_string(),
            small_id,
            player_type: PlayerType::Bot,
            gold: 1_000_000,
            ..Default::default()
        });
    }

    fn units_of(game: &Game, small_id: u16, types: &[&str]) -> Vec<i32> {
        game.player_by_small_id(small_id)
            .unwrap()
            .units
            .iter()
            .filter(|u| types.contains(&u.unit_type.as_str()))
            .map(|u| u.id)
            .collect()
    }

    /// TS `PlayerImpl.test.ts` "units() type filtering": `Player.units(type1,
    /// type2, ...)` returns the union of matching units in insertion order,
    /// deduplicated, as a fresh array each call. Native has no single shared
    /// helper of that name - every call site filters `Player.units:
    /// Vec<Unit>` inline instead (see `nation_tick.rs`/
    /// `nation_structures.rs`) - so this pins that the same
    /// filter-by-type-set pattern those call sites use preserves the TS
    /// invariants, using the same 4 unit types/order as the TS test.
    #[test]
    fn unit_type_filtering_preserves_insertion_order_and_dedups() {
        let mut game = crate::test_util::plains_game(16, 16);
        add_bot(&mut game, "player", 1);
        game.conquer(1, game.map.ref_xy(0, 0));
        let city1 = game.build_unit(1, unit_type::CITY, game.map.ref_xy(0, 0));
        game.build_unit(1, unit_type::DEFENSE_POST, game.map.ref_xy(11, 0));
        let city2 = game.build_unit(1, unit_type::CITY, game.map.ref_xy(0, 11));
        let silo = game.build_unit(1, unit_type::MISSILE_SILO, game.map.ref_xy(11, 11));

        assert_eq!(units_of(&game, 1, &[unit_type::CITY]), vec![city1, city2]);
        assert_eq!(
            units_of(&game, 1, &[unit_type::CITY, unit_type::MISSILE_SILO]),
            vec![city1, city2, silo]
        );
        // Duplicate types in the filter set don't duplicate results.
        assert_eq!(
            units_of(&game, 1, &[unit_type::CITY, unit_type::CITY]),
            vec![city1, city2]
        );
        assert_eq!(units_of(&game, 1, &[unit_type::PORT]), Vec::<i32>::new());
    }

    /// TS `PlayerImpl.test.ts` "Can't send alliance requests when dead" -
    /// the literal scenario (eliminate a player by conquering every tile it
    /// owns, then check it can no longer send an alliance request),
    /// exercised through the real map/conquer path rather than by directly
    /// setting `tiles_owned` (see `relation_tests::dead_sender_cannot_send_
    /// alliance_request` for the field-level version, and its sibling for
    /// the asymmetric recipient-side bug this same function had).
    #[test]
    fn eliminated_player_cannot_send_alliance_request() {
        let mut game = crate::test_util::plains_game(16, 16);
        add_bot(&mut game, "player", 1);
        add_bot(&mut game, "other", 2);
        game.conquer(2, game.map.ref_xy(5, 5));

        let others_tiles = game.player_by_small_id(2).unwrap().owned_tiles.clone();
        for t in others_tiles {
            game.conquer(1, t);
        }

        assert_eq!(game.player_by_small_id(2).unwrap().tiles_owned, 0);
        assert!(!game.can_send_alliance_request(2, 1));
    }

    /// TS `Game.nearbyUnits()` excludes under-construction units by default.
    /// Defense posts are built immediately as units but stay under construction
    /// for 50 ticks, so they must not apply attack-defense bonuses until
    /// completion.
    #[test]
    fn under_construction_defense_post_does_not_count_as_nearby() {
        let mut game = crate::test_util::plains_game(64, 64);
        add_bot(&mut game, "player", 1);
        let post_tile = game.map.ref_xy(10, 10);
        let check_tile = game.map.ref_xy(12, 10);
        game.conquer(1, post_tile);
        let post_id = game.build_unit(1, unit_type::DEFENSE_POST, post_tile);

        assert!(game.has_defense_post_nearby(check_tile, 1));
        game.set_unit_under_construction(1, post_id, true);
        assert!(!game.has_defense_post_nearby(check_tile, 1));
        game.set_unit_under_construction(1, post_id, false);
        assert!(game.has_defense_post_nearby(check_tile, 1));
    }

    #[test]
    fn defense_post_query_matches_unit_grid_boundary_window() {
        let mut game = crate::test_util::plains_game(128, 128);
        add_bot(&mut game, "player", 1);
        let check_tile = game.map.ref_xy(25, 70);
        let post_tile = game.map.ref_xy(25, 100);
        game.conquer(1, post_tile);
        game.build_unit(1, unit_type::DEFENSE_POST, post_tile);

        // TS UnitGrid.getCellsInRange does not scan the next 100-tile cell
        // when the search range lands exactly on that cell's boundary.
        assert!(!game.has_defense_post_nearby(check_tile, 1));
    }

    /// TS `PlayerImpl.test.ts` "City can be upgraded" / "DefensePost cannot
    /// be upgraded" / "City can be upgraded from another city" / "City
    /// cannot be upgraded when too far away" / "Unit cannot be upgraded when
    /// not enough gold": all five exercise `PlayerImpl.buildableUnits()`/
    /// `findUnitToUpgrade()`, whose only caller is `GameRunner.ts` (feeding
    /// the human build/upgrade UI's clickable-unit hints) - no `Execution`
    /// or other simulation-affecting code path calls them. Native has no
    /// port of either method; the nation-AI upgrade decision (a genuinely
    /// different, already-ported code path) is `find_best_structure_to_
    /// upgrade` in `nation_structures.rs`, covered by that file's own
    /// tests. Building a client-UI-only subsystem from scratch here would
    /// have no hash-parity payoff and is out of scope for this batch.
    #[test]
    #[ignore = "buildableUnits()/findUnitToUpgrade() are human-UI-only build hints (GameRunner.ts); not ported, see doc comment"]
    fn buildable_units_and_find_unit_to_upgrade_are_unported_ui_only_apis() {
        unreachable!(
            "gap marker only - see PlayerImpl.test.ts's City/DefensePost upgrade tests and this test's doc comment"
        );
    }
}

// TS `ImpassableTerrain.test.ts` - terrain classification + ownership
// subset ("Terrain classification" / "Ownership" describe blocks). The
// nuke/attack/nation-AI subsets of the same TS file are ported alongside
// their respective native homes: `execution/nuke_execution.rs`,
// `execution/attack.rs`, `execution/ai_attack.rs`, `execution/
// nation_tick.rs` (the last one `#[ignore]`d - see its doc comment).
#[cfg(test)]
mod impassable_terrain_tests {
    use super::{Game, Player, PlayerType};

    const WALL_X: u32 = 30;
    const WALL_WIDTH: u32 = 2;
    const MAP_W: u32 = 60;
    const MAP_H: u32 = 20;

    fn wall_game() -> Game {
        crate::test_util::walled_game(MAP_W, MAP_H, Some((WALL_X, WALL_WIDTH)))
    }

    #[test]
    fn is_impassable_returns_true_for_impassable_tiles_false_for_plains() {
        let game = wall_game();
        assert!(!game.is_impassable(game.map.ref_xy(20, 10)));
        assert!(game.is_impassable(game.map.ref_xy(WALL_X, 10)));
        assert!(game.is_impassable(game.map.ref_xy(WALL_X + 1, 10)));
    }

    #[test]
    fn terrain_type_returns_impassable_for_impassable_tiles() {
        let game = wall_game();
        assert_eq!(
            game.terrain_type(game.map.ref_xy(WALL_X, 10)),
            crate::map::TerrainType::Impassable
        );
    }

    #[test]
    fn is_land_returns_true_for_impassable_solid_for_pathfinding() {
        let game = wall_game();
        assert!(game.is_land(game.map.ref_xy(WALL_X, 10)));
    }

    #[test]
    fn num_land_tiles_excludes_impassable_tiles() {
        let game = wall_game();
        assert_eq!(game.num_land_tiles(), MAP_W * MAP_H - WALL_WIDTH * MAP_H);
    }

    fn add_bot(game: &mut Game, id: &str, small_id: u16) {
        game.add_player(Player {
            id: id.to_string(),
            small_id,
            player_type: PlayerType::Bot,
            ..Default::default()
        });
    }

    /// TS `GameImpl.conquer` throws on impassable tiles; native no-ops
    /// instead (see `conquer_one`'s doc comment for why) - this pins the
    /// bug that fix caught: before it, `game.conquer(1, impassable_tile)`
    /// would silently succeed and give a player real ownership of
    /// impassable terrain.
    #[test]
    fn conquer_does_not_grant_ownership_of_impassable_tiles() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let tile = game.map.ref_xy(WALL_X, 10);
        game.conquer(1, tile);
        assert_eq!(
            game.map.owner_id(tile),
            0,
            "impassable tile must stay unowned"
        );
        assert_eq!(game.player_by_small_id(1).unwrap().tiles_owned, 0);
    }

    #[test]
    fn conquer_succeeds_on_normal_land() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let tile = game.map.ref_xy(20, 10);
        game.conquer(1, tile);
        assert_eq!(game.map.owner_id(tile), 1);
    }

    /// TS `Player.isAlive()` is `_tiles.size > 0`. Losing the last tile must
    /// clear `alive` immediately (not wait for the next `PlayerExecution`),
    /// otherwise mid-tick AI filters and tick dumps lag TS by one tick.
    #[test]
    fn losing_last_tile_clears_alive_immediately() {
        let mut game = wall_game();
        add_bot(&mut game, "victim", 1);
        add_bot(&mut game, "attacker", 2);
        game.end_spawn_phase();
        let tile = game.map.ref_xy(20, 10);
        game.conquer(1, tile);
        assert!(game.player_by_small_id(1).unwrap().alive);
        assert_eq!(game.player_by_small_id(1).unwrap().tiles_owned, 1);

        game.conquer(2, tile);
        let victim = game.player_by_small_id(1).unwrap();
        assert_eq!(victim.tiles_owned, 0);
        assert!(
            !victim.alive,
            "alive must track tiles like TS isAlive() on the conquer tick"
        );
        assert!(game.player_by_small_id(2).unwrap().alive);
    }

    // TS `PlayerImpl.canAttack(tile)` - only called from `GameRunner.ts`
    // (feeds the "can I click-attack this tile" UI hint); no `Execution`
    // reads it. Native has no port and none is needed for hash parity - see
    // `PlayerImpl.test.ts`'s analogous `buildableUnits`/`findUnitToUpgrade`
    // gap note in `player_tests` above for the same rationale.
    #[test]
    #[ignore = "PlayerImpl.canAttack(tile) is a human-UI-only click-attack hint (GameRunner.ts); not ported"]
    fn can_attack_tile_is_unported_ui_only_api() {
        unreachable!("gap marker only - see doc comment above");
    }

    // TS `GameImpl.setWater`/`queueWaterConversion`'s "water nukes" feature
    // defaults off (`Config.waterNukes() => this._gameConfig.waterNukes ??
    // false`) and has no native port; when off, TS's own
    // `queueWaterConversion` falls back to `setFallout` (which native DOES
    // implement and does guard `!is_impassable` in `NukeExecution::detonate`
    // - see `nuke_execution.rs`'s tests). The low-level `GameMap.setWater`
    // this specific test calls is only reachable via that unported feature
    // (plus a client-only rendering call site), so there is nothing to
    // port it against.
    #[test]
    #[ignore = "GameMap.setWater is only reachable via the unported (default-off) waterNukes feature; native's default fallout path is covered in nuke_execution.rs"]
    fn set_water_does_not_convert_impassable_tiles_is_unported() {
        unreachable!("gap marker only - see doc comment above");
    }
}

#[derive(Debug, Default)]
pub struct TickUpdates {
    pub hash: Option<HashUpdate>,
    pub win: Option<WinUpdate>,
}

// Ported from `openfront/tests/WarshipVeterancy.test.ts`. That file's setup uses a real
// `half_land_half_ocean` map + `WarshipExecution` end-to-end; these tests instead drive
// `Game::record_kill`/`record_trade_capture`/`unit_max_health` directly (this codebase's
// established `Game::default()` idiom - see e.g. `impassable_terrain_tests` above), since
// none of the veterancy math itself depends on geometry/pathfinding.
#[cfg(test)]
mod veterancy_tests {
    use super::{Game, Player, PlayerType};
    use crate::core::schemas::unit_type::{TRADE_SHIP, TRANSPORT, WARSHIP};

    fn add_player(game: &mut Game, small_id: u16) {
        game.add_player(Player {
            id: small_id.to_string(),
            small_id,
            player_type: PlayerType::Human,
            ..Default::default()
        });
    }

    #[test]
    fn killing_an_enemy_warship_grants_one_veterancy_level() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        assert_eq!(game.unit_veterancy(1, ship), 0);

        game.record_kill(1, ship, WARSHIP);
        assert_eq!(game.unit_veterancy(1, ship), 1);
    }

    #[test]
    fn veterancy_is_capped_at_the_configured_maximum() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let max = game.wire.warship_max_veterancy();

        for _ in 0..max + 3 {
            game.record_kill(1, ship, WARSHIP);
        }
        assert_eq!(game.unit_veterancy(1, ship), max);
    }

    #[test]
    fn destroying_transport_ships_alone_fills_a_level_at_the_threshold() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let threshold = game.wire.warship_veterancy_transport_kills();

        for _ in 0..threshold - 1 {
            game.record_kill(1, ship, TRANSPORT);
        }
        assert_eq!(game.unit_veterancy(1, ship), 0);

        game.record_kill(1, ship, TRANSPORT);
        assert_eq!(game.unit_veterancy(1, ship), 1);
    }

    #[test]
    fn capturing_trade_ships_alone_fills_a_level_at_the_threshold() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let threshold = game.wire.warship_veterancy_trade_captures();

        for _ in 0..threshold - 1 {
            game.record_trade_capture(1, ship);
        }
        assert_eq!(game.unit_veterancy(1, ship), 0);

        game.record_trade_capture(1, ship);
        assert_eq!(game.unit_veterancy(1, ship), 1);
    }

    #[test]
    fn transports_and_captures_share_one_progress_meter() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        // Defaults: 10 transports OR 25 captures = 1 level, so a transport is worth 1/10 of
        // a level and a capture 1/25. Mixed progress combines.
        for _ in 0..5 {
            game.record_kill(1, ship, TRANSPORT);
        }
        for _ in 0..12 {
            game.record_trade_capture(1, ship);
        }
        assert_eq!(game.unit_veterancy(1, ship), 0); // 5/10 + 12/25 = 0.98 < 1

        game.record_trade_capture(1, ship);
        assert_eq!(game.unit_veterancy(1, ship), 1); // 5/10 + 13/25 = 1.02 >= 1
    }

    #[test]
    fn a_warship_kill_resets_transport_capture_progress() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let threshold = game.wire.warship_veterancy_transport_kills();

        // Build up 9/10 of a level from transports (no level yet).
        for _ in 0..threshold - 1 {
            game.record_kill(1, ship, TRANSPORT);
        }
        assert_eq!(game.unit_veterancy(1, ship), 0);

        // A warship kill grants a level AND wipes the partial progress.
        game.record_kill(1, ship, WARSHIP);
        assert_eq!(game.unit_veterancy(1, ship), 1);

        // Had progress carried, this transport would have completed level 2. Since it
        // reset, we're still at level 1.
        game.record_kill(1, ship, TRANSPORT);
        assert_eq!(game.unit_veterancy(1, ship), 1);
    }

    #[test]
    fn partial_progress_carries_past_a_level_up() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let threshold = game.wire.warship_veterancy_trade_captures();

        // One past the threshold -> level 1 with 1 capture's worth carried over.
        for _ in 0..threshold + 1 {
            game.record_trade_capture(1, ship);
        }
        assert_eq!(game.unit_veterancy(1, ship), 1);

        // The carried progress means one fewer capture completes level 2.
        for _ in 0..threshold - 1 {
            game.record_trade_capture(1, ship);
        }
        assert_eq!(game.unit_veterancy(1, ship), 2);
    }

    #[test]
    fn veterancy_raises_max_health_but_does_not_instantly_heal() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let ship = game.build_unit(1, WARSHIP, tile);
        let base = game.wire.warship_base_max_health();
        let bonus_percent = game.wire.warship_veterancy_health_bonus();

        // Drop below full so a (removed) instant heal would be observable.
        game.unit_mut(1, ship).unwrap().health -= 100;
        assert_eq!(game.unit_max_health(1, ship), base);
        assert_eq!(game.unit(1, ship).unwrap().health, base - 100);

        game.record_kill(1, ship, WARSHIP); // veterancy 1

        // The cap rises, but current health is unchanged - the ship heals toward the new
        // max normally, it does not jump on level-up.
        assert_eq!(
            game.unit_max_health(1, ship),
            base + base * 1 * bonus_percent / 100
        );
        assert_eq!(game.unit(1, ship).unwrap().health, base - 100);
    }

    #[test]
    fn non_warships_never_gain_veterancy() {
        let mut game = Game::default();
        add_player(&mut game, 1);
        let tile = game.map.ref_xy(0, 0);
        let transport = game.build_unit(1, TRANSPORT, tile);

        game.record_kill(1, transport, WARSHIP);
        game.record_trade_capture(1, transport);

        assert_eq!(game.unit_veterancy(1, transport), 0);
    }
}

// Ported from `Team.test.ts` - covers `GameImpl.addPlayer`'s on-the-fly
// `maybeAssignTeam` fallback (as opposed to `TeamAssignment.test.ts`'s batch
// `assignTeams`, ported separately in `core/team_assignment.rs`).
#[cfg(test)]
mod team_join_tests {
    use super::{Game, GameConfig, PlayerInfo, PlayerType};
    use crate::core::config::Config as WireConfig;
    use crate::core::schemas::{GameConfig as WireGameConfig, NationsConfig, PlayerTeamsConfig};
    use crate::map::{GameMap, MapMeta};

    fn team_mode_game(num_teams: u32) -> Game {
        let meta = MapMeta {
            width: 10,
            height: 1,
            num_land_tiles: 10,
        };
        let terrain = vec![0x80u8; 10];
        let map = GameMap::from_terrain_bytes(&meta, &terrain).unwrap();
        let mini_map = GameMap::from_terrain_bytes(&meta, &terrain).unwrap();
        let wire_cfg = WireGameConfig {
            game_map: "Plains".into(),
            difficulty: "Medium".into(),
            donate_gold: false,
            donate_troops: false,
            game_type: "Singleplayer".into(),
            game_mode: "Team".into(),
            game_map_size: "Normal".into(),
            nations: NationsConfig::Mode("default".into()),
            bots: 0,
            infinite_gold: false,
            infinite_troops: false,
            instant_build: false,
            random_spawn: false,
            doomsday_clock: None,
            disabled_units: None,
            player_teams: Some(PlayerTeamsConfig::Count(num_teams)),
            disable_alliances: None,
            spawn_immunity_duration: None,
            starting_gold: None,
            gold_multiplier: None,
            max_timer_value: None,
            ranked_type: None,
        };
        let mut game = Game::new(
            String::new(),
            GameConfig::default(),
            WireConfig::new(wire_cfg, false),
            map,
            mini_map,
            None,
        );
        game.end_spawn_phase();
        game
    }

    fn add_player(game: &mut Game, id: &str, player_type: PlayerType) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn bots_share_a_team_but_are_not_mutually_friendly() {
        let mut game = team_mode_game(2);
        let bot1 = add_player(&mut game, "bot1", PlayerType::Bot);
        let bot2 = add_player(&mut game, "bot2", PlayerType::Bot);

        assert_eq!(
            game.player_by_small_id(bot1).unwrap().team.as_deref(),
            Some(crate::core::team_assignment::BOT_TEAM)
        );
        assert_eq!(
            game.player_by_small_id(bot2).unwrap().team.as_deref(),
            Some(crate::core::team_assignment::BOT_TEAM)
        );
        // Same nominal team, but `Bot` is explicitly excluded from `players_on_same_team`'s
        // friendliness check - tribes still fight each other.
        assert!(!game.players_on_same_team(bot1, bot2));
    }

    #[test]
    fn humans_joining_without_an_explicit_team_get_hash_spread_across_teams() {
        // Before the fix, `maybe_assign_team` returned `None` for every non-Bot player
        // type regardless of `player_teams`, so both humans ended up on no team at all
        // (`players_on_same_team` treats "no team" as never-same-team, which would have
        // made this assertion pass for the wrong reason). Assert both are actually
        // assigned a real team, then that they land on different ones - matching TS
        // `GameImpl.maybeAssignTeam`'s `simpleHash(id) % playerTeams.length` spread for
        // these specific ids.
        let mut game = team_mode_game(2);
        let human1 = add_player(&mut game, "human1", PlayerType::Human);
        let human2 = add_player(&mut game, "human2", PlayerType::Human);

        assert!(game.player_by_small_id(human1).unwrap().team.is_some());
        assert!(game.player_by_small_id(human2).unwrap().team.is_some());
        assert!(!game.players_on_same_team(human1, human2));
    }

    #[test]
    fn congratulate_winner_recognizes_a_teammate_as_the_winner() {
        // `is_on_same_team` takes the winner's `PlayerID` string (matching its only call
        // site, `nation_emoji.rs`), not a `small_id` - was a hardcoded-`false` stub.
        let mut game = team_mode_game(2);
        let p1 = add_player(&mut game, "p1", PlayerType::Nation);
        let p2 = add_player(&mut game, "p2", PlayerType::Nation);
        game.player_by_small_id_mut(p1).unwrap().team = Some("Red".into());
        game.player_by_small_id_mut(p2).unwrap().team = Some("Red".into());
        let p2_id = game.player_by_small_id(p2).unwrap().id.clone();

        assert!(game.is_on_same_team(p1, &p2_id));
    }

    #[test]
    fn congratulate_winner_does_not_recognize_an_opposing_team_as_the_winner() {
        let mut game = team_mode_game(2);
        let p1 = add_player(&mut game, "p1", PlayerType::Nation);
        let p2 = add_player(&mut game, "p2", PlayerType::Nation);
        game.player_by_small_id_mut(p1).unwrap().team = Some("Red".into());
        game.player_by_small_id_mut(p2).unwrap().team = Some("Blue".into());
        let p2_id = game.player_by_small_id(p2).unwrap().id.clone();

        assert!(!game.is_on_same_team(p1, &p2_id));
    }

    #[test]
    fn is_on_same_team_returns_false_for_an_unknown_winner_id() {
        let mut game = team_mode_game(2);
        let p1 = add_player(&mut game, "p1", PlayerType::Nation);

        assert!(!game.is_on_same_team(p1, "no-such-player"));
    }
}

// Ported from `UnitGrid.test.ts`. Native has no spatial-cell grid (`spatial.rs` is an
// Port of `openfront/tests/UnitGrid.test.ts`. Native now keeps a real UnitGrid
// (`crate::unit_grid`) matching TS cell/Set insertion-order semantics; these
// tests exercise the public `has_unit_nearby_any`/`nearby_structures_any` API.
#[cfg(test)]
mod unit_grid_tests {
    use super::{Game, PlayerInfo, PlayerType};
    use crate::core::schemas::unit_type;
    use crate::test_util::plains_game;

    fn add_human(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    fn check_range(map_size: u32, unit_pos_x: u32, range_check_x: u32, range: u32) -> bool {
        let mut game = plains_game(map_size, map_size);
        let p1 = add_human(&mut game, "test_id");
        let unit_tile = game.map.ref_xy(unit_pos_x, 0);
        game.build_unit(p1, unit_type::DEFENSE_POST, unit_tile);
        let check_tile = game.map.ref_xy(range_check_x, 0);
        game.has_unit_nearby_any(check_tile, range, unit_type::DEFENSE_POST)
    }

    fn nearby_count(
        map_size: u32,
        unit_pos_x: u32,
        range_check_x: u32,
        range: u32,
        unit_types: &[&str],
    ) -> usize {
        let mut game = plains_game(map_size, map_size);
        let p1 = add_human(&mut game, "test_id");
        let unit_tile = game.map.ref_xy(unit_pos_x, 0);
        for &t in unit_types {
            game.build_unit(p1, t, unit_tile);
        }
        let check_tile = game.map.ref_xy(range_check_x, 0);
        game.nearby_structures_any(check_tile, range, unit_types)
            .len()
    }

    #[test]
    fn has_unit_nearby_same_spot() {
        assert!(check_range(100, 0, 0, 10));
    }

    #[test]
    fn has_unit_nearby_exactly_on_the_range() {
        assert!(check_range(100, 0, 10, 10));
    }

    #[test]
    fn has_unit_nearby_exactly_one_outside_the_range() {
        assert!(!check_range(100, 0, 11, 10));
    }

    #[test]
    fn has_unit_nearby_inside_a_huge_range() {
        assert!(check_range(200, 0, 42, 198));
    }

    #[test]
    fn has_unit_nearby_exactly_one_outside_a_huge_range() {
        assert!(!check_range(200, 0, 199, 198));
    }

    #[test]
    fn nearby_units_same_spot() {
        assert_eq!(nearby_count(100, 0, 0, 10, &[unit_type::WARSHIP]), 1);
    }

    #[test]
    fn nearby_units_two_types_in_range() {
        assert_eq!(
            nearby_count(100, 0, 0, 10, &[unit_type::CITY, unit_type::PORT]),
            2
        );
    }

    #[test]
    fn nearby_units_no_types_requested() {
        assert_eq!(nearby_count(100, 0, 0, 10, &[]), 0);
    }

    #[test]
    fn nearby_units_exactly_on_the_range() {
        assert_eq!(nearby_count(100, 0, 10, 10, &[unit_type::CITY]), 1);
    }

    #[test]
    fn nearby_units_one_outside_the_range() {
        assert_eq!(nearby_count(100, 0, 11, 10, &[unit_type::DEFENSE_POST]), 0);
    }

    #[test]
    fn nearby_units_inside_a_huge_range() {
        assert_eq!(nearby_count(200, 0, 42, 198, &[unit_type::TRADE_SHIP]), 1);
    }

    #[test]
    fn nearby_units_one_outside_a_huge_range() {
        assert_eq!(nearby_count(200, 0, 199, 198, &[unit_type::TRANSPORT]), 0);
    }

    #[test]
    fn nearby_units_ignores_a_type_not_present() {
        let mut game = plains_game(100, 100);
        let p1 = add_human(&mut game, "test_id");
        let tile = game.map.ref_xy(0, 0);
        game.build_unit(p1, unit_type::CITY, tile);
        assert_eq!(
            game.nearby_structures_any(tile, 10, &[unit_type::PORT])
                .len(),
            0
        );
    }

    #[test]
    fn nearby_units_one_inside_one_outside_of_range() {
        let mut game = plains_game(100, 100);
        let p1 = add_human(&mut game, "test_id");
        let inside_tile = game.map.ref_xy(0, 0);
        game.build_unit(p1, unit_type::CITY, inside_tile);
        let outside_tile = game.map.ref_xy(99, 0);
        game.build_unit(p1, unit_type::CITY, outside_tile);
        let check_tile = game.map.ref_xy(0, 0);
        assert_eq!(
            game.nearby_structures_any(check_tile, 10, &[unit_type::CITY])
                .len(),
            1
        );
    }
}
