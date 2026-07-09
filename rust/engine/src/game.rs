//! Core game state + tick driver.

use crate::core::config::Config as WireConfig;
use crate::execution::ordered_map::OrderedMap;
use crate::execution::ordered_tiles::OrderedTiles;
use crate::execution::{ExecEnum, Execution, HashUpdate, WinUpdate};
use crate::hash::game_hash;
use crate::core::team_assignment::{assign_teams, populate_player_teams, BOT_TEAM, HUMANS_TEAM, NATIONS_TEAM};
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
    /// TS `UnitImpl.level()`  -  defaults to 1; affects `maxTroops` for cities.
    pub level: i32,
    /// TS `UnitImpl._hasTrainStation` - set true immediately when a `TrainStationExecution`
    /// is constructed for this unit (City/Port/Factory).
    pub has_train_station: bool,
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
    pub marked_traitor_tick: i32,
    /// TS `PlayerImpl.relations` is a `Map<Player, number>`, which iterates in insertion
    /// order (re-setting an existing key's value does not move it). `allRelationsSorted()`
    /// relies on that insertion order as the tie-break after its stable sort-by-value, so a
    /// plain `HashMap` here (whose iteration order is randomly per-process-seeded) makes
    /// tie-broken attack-target selection genuinely nondeterministic across runs.
    pub relations: OrderedMap<u16, i32>,
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
            marked_traitor_tick: -1,
            relations: OrderedMap::new(),
            is_disconnected: false,
            units_constructed: HashMap::new(),
            id_prng: PseudoRandom::new(0),
            team: None,
            outgoing_land_attacks: Vec::new(),
            incoming_land_attacks: Vec::new(),
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
    pub hash_enabled: bool,
    pub tribe_batch: crate::bot::TribeBatch,
    pub nation_batch: crate::execution::NationBatch,
    pub alliance_requests: Vec<AllianceRequestState>,
    pub alliances: Vec<AllianceState>,
    pub next_alliance_id: u32,
    /// Same-tick inits not yet appended to `execs` (TS `outgoingAttacks()` batch ordering).
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
        }
    }
}

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
        let tile_count = (map.width * map.height) as usize;
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
            tribe_batch: crate::bot::TribeBatch::new(),
            nation_batch: crate::execution::NationBatch::new(),
            team_spawn_areas,
            player_teams: populate_player_teams(&game_mode, player_teams_cfg.as_ref(), 0, 0),
            ..Default::default()
        }
    }

    pub fn init_player_teams(&mut self, num_humans: usize, num_nations: usize) {
        let game_mode = self.wire.game_config().game_mode.clone();
        let player_teams_cfg = self.wire.game_config().player_teams.clone();
        self.player_teams =
            populate_player_teams(&game_mode, player_teams_cfg.as_ref(), num_humans, num_nations);
    }

    /// TS `GameImpl.teamSpawnArea`.
    pub fn team_spawn_area(&self, team: &str) -> Option<&SpawnArea> {
        let areas = self.team_spawn_areas.as_ref()?;
        let num_teams = self.player_teams.len();
        let team_areas = areas.get(&num_teams.to_string())?;
        let team_index = self.player_teams.iter().position(|t| t == team)?;
        team_areas.get(team_index)
    }

    fn maybe_assign_team(&self, player_type: PlayerType) -> Option<String> {
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
        None
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

    /// TS `Game.nearbyUnits(tile, range, DefensePost)` - first owned post in range.
    pub fn has_defense_post_nearby(&self, tile: TileRef, owner_small_id: u16) -> bool {
        use crate::core::schemas::unit_type;
        let range = self.wire.defense_post_range();
        let range_sq = (range * range) as u32;
        let tx = self.map.x(tile);
        let ty = self.map.y(tile);
        for p in &self.players {
            if p.small_id != owner_small_id {
                continue;
            }
            for u in &p.units {
                if u.unit_type != unit_type::DEFENSE_POST {
                    continue;
                }
                let ut = u.tile as u32;
                let dx = self.map.x(ut) as i32 - tx as i32;
                let dy = self.map.y(ut) as i32 - ty as i32;
                if (dx * dx + dy * dy) as u32 <= range_sq {
                    return true;
                }
            }
        }
        false
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

            if matches!(
                attacker_type,
                PlayerType::Human | PlayerType::Nation
            ) && defender_type == Some(PlayerType::Bot)
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
                large_attack_bonus =
                    (100_000.0 / attacker_tiles as f64).sqrt().powf(0.7);
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
            let current_attacker_loss = within(
                defender_troops as f64 / attack_troops,
                0.6,
                2.0,
            ) * mag
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
            let tiles_per_tick = within(
                defender_troops as f64 / (5.0 * attack_troops),
                0.2,
                1.5,
            ) * speed
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
            let tiles_per_tick =
                within((2000.0 * speed.max(10.0)) / attack_troops, 5.0, 100.0);
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
            .or_else(|| self.maybe_assign_team(info.player_type));
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
        }
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
        self.player_by_small_id(small_id)
            .map(|p| &p.border_tiles)
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
        let Some(p) = self.player_by_small_id_mut(small_id) else {
            return 0;
        };
        let to_remove = p.troops.min(crate::util::to_int(amount));
        p.troops -= to_remove;
        to_remove
    }

    pub fn add_troops(&mut self, small_id: u16, amount: f64) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.troops += crate::util::to_int(amount);
        }
    }

    pub fn conquer(&mut self, small_id: u16, tile: TileRef) {
        self.conquer_one(small_id, tile, true);
    }

    /// TS `GameImpl.conquerPlayer`  -  ship transfer on disconnected teammate wipe.
    pub fn conquer_player(&mut self, conqueror_small_id: u16, conquered_small_id: u16) {
        let (disconnected, same_team) = {
            let Some(conquered) = self.player_by_small_id(conquered_small_id) else {
                return;
            };
            let conqueror_team = self
                .player_by_small_id(conqueror_small_id)
                .and_then(|p| p.team.clone());
            (
                conquered.is_disconnected,
                conqueror_team.is_some() && conqueror_team == conquered.team.clone(),
            )
        };

        if !disconnected || !same_team {
            return;
        }

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

    /// Bulk conquer for spawn hops - per-tile border refresh (TS `GameImpl.conquer`).
    pub fn conquer_spawn_tiles(&mut self, small_id: u16, tiles: &[TileRef]) {
        for &tile in tiles {
            self.conquer_one(small_id, tile, true);
        }
    }

    fn conquer_one(&mut self, small_id: u16, tile: TileRef, refresh_borders: bool) {
        if !self.is_land(tile) {
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
            }
        }
        self.map.set_owner_id(tile, small_id);
        self.map.set_fallout(tile, false);
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.tiles_owned += 1;
            p.owned_tiles.push(tile);
            p.last_tile_change = tick;
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
        self.execs.iter().position(|e| {
            matches!(e, ExecEnum::Player(p) if p.small_id() == small_id)
        })
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

    pub fn add_transport_attack(&mut self, owner_small_id: u16, target_tile: TileRef, troops: f64) {
        self.add_execution(ExecEnum::TransportShip(
            crate::execution::TransportShipExecution::new(owner_small_id, target_tile, troops),
        ));
    }

    pub fn unit_count(&self, small_id: u16, unit_type: &str) -> usize {
        self.player_by_small_id(small_id)
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.unit_type == unit_type)
                    .count()
            })
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
        let tx = self.map.x(tile) as i64;
        let ty = self.map.y(tile) as i64;
        let range_sq = (range as i64) * (range as i64);
        for p in &self.players {
            for u in &p.units {
                if u.unit_type != unit_type || u.under_construction {
                    continue;
                }
                let ux = self.map.x(u.tile as TileRef) as i64;
                let uy = self.map.y(u.tile as TileRef) as i64;
                let dx = ux - tx;
                let dy = uy - ty;
                if dx * dx + dy * dy <= range_sq {
                    return true;
                }
            }
        }
        false
    }

    /// TS `UnitGrid.nearbyUnits(tile, range, types)` - scans across ALL players.
    /// Returns `(owner_small_id, unit_id, tile, dist_squared)`.
    ///
    /// Order matters even for callers that re-sort by distance afterward: JS `Array.sort` (and
    /// Rust's `sort_by`) are stable, so ties fall back to this pre-sort order, and some callers
    /// (`FactoryExecution.createStation`) use the raw order with no sort at all. TS's `UnitGrid`
    /// enumerates its 100x100-tile cells row-major (`cy` outer, `cx` inner), then `types` in the
    /// given array order, then insertion order within each cell/type `Set` - which for immobile
    /// structures is creation order, i.e. our monotonic global `unit_id`. Replicate that
    /// ordering by sorting on an equivalent key instead of building a real spatial grid.
    pub fn nearby_structures_any(
        &self,
        tile: TileRef,
        range: u32,
        types: &[&str],
    ) -> Vec<(u16, i32, TileRef, f64)> {
        const UNIT_GRID_CELL_SIZE: i64 = 100;
        let tx = self.map.x(tile) as i64;
        let ty = self.map.y(tile) as i64;
        let range_sq = (range as i64) * (range as i64);
        let mut out: Vec<(u16, i32, TileRef, f64, i64, i64, usize)> = Vec::new();
        for p in &self.players {
            for u in &p.units {
                if u.under_construction {
                    continue;
                }
                let Some(type_idx) = types.iter().position(|t| *t == u.unit_type) else {
                    continue;
                };
                let ux = self.map.x(u.tile as TileRef) as i64;
                let uy = self.map.y(u.tile as TileRef) as i64;
                let dx = ux - tx;
                let dy = uy - ty;
                let d2 = dx * dx + dy * dy;
                if d2 > range_sq {
                    continue;
                }
                let cell_x = ux.div_euclid(UNIT_GRID_CELL_SIZE);
                let cell_y = uy.div_euclid(UNIT_GRID_CELL_SIZE);
                out.push((p.small_id, u.id, u.tile as TileRef, d2 as f64, cell_y, cell_x, type_idx));
            }
        }
        out.sort_by(|a, b| {
            (a.4, a.5, a.6, a.1).cmp(&(b.4, b.5, b.6, b.1))
        });
        out.into_iter().map(|(sid, uid, t, d2, ..)| (sid, uid, t, d2)).collect()
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
            *p.units_constructed.entry(unit_type.to_string()).or_insert(0) += 1;
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
        self.wire.troop_increase_rate(
            p.player_type,
            p.troops,
            p.tiles_owned,
            city_levels,
        )
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
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold -= cost;
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.level += 1;
            }
        }
        self.record_unit_constructed(small_id, &unit_type);
        true
    }

    pub fn build_unit(&mut self, small_id: u16, unit_type: &str, tile: TileRef) -> i32 {
        let cost = self.structure_cost(small_id, unit_type);
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold -= cost;
        }
        let id = self.next_unit_id;
        self.next_unit_id += 1;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.units.push(Unit {
                id,
                unit_type: unit_type.to_string(),
                tile: tile as i32,
                under_construction: false,
                level: 1,
                has_train_station: false,
            });
        }
        self.record_unit_constructed(small_id, unit_type);
        id
    }

    pub fn remove_unit(&mut self, small_id: u16, unit_id: i32) {
        let had_station = self
            .player_by_small_id(small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
            .is_some_and(|u| u.has_train_station);
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
        if let Some(unit) = captured {
            if let Some(p) = self.player_by_small_id_mut(to_small_id) {
                p.units.push(unit);
            }
        }
    }

    pub fn move_unit(&mut self, small_id: u16, unit_id: i32, tile: TileRef) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.tile = tile as i32;
            }
        }
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

    fn find_land_attack_exec_mut(&mut self, attack_id: &str) -> Option<*mut crate::execution::AttackExecution> {
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
            if !atk.is_active() || !atk.attack_live() || !atk.is_initialized() || atk.source_tile().is_some() {
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

    /// Largest non-friendly incoming land attacker (TS `AiAttackBehavior.findIncomingAttackPlayer`).
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
            if !atk.is_active() || !atk.attack_live() || !atk.is_initialized() || atk.source_tile().is_some() {
                continue;
            }
            if atk.target_small_id() != defender_small_id {
                continue;
            }
            let attacker = atk.owner_small_id();
            if attacker == defender_small_id || attacker == 0 {
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

    pub fn incoming_attacks(&self, defender_small_id: u16) -> Vec<IncomingAttack> {
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
                if !p.alive {
                    continue;
                }
            }
            let Some(troops) = self.land_attack_troops(id) else {
                continue;
            };
            if self.land_attack_source_tile(id).is_some() {
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

    pub fn outgoing_land_troops(&self, attacker_small_id: u16) -> f64 {
        let Some(attacker) = self.player_by_small_id(attacker_small_id) else {
            return 0.0;
        };
        let mut total = 0.0;
        for id in &attacker.outgoing_land_attacks {
            if let Some(troops) = self.land_attack_troops(id) {
                if self.land_attack_source_tile(id).is_none() {
                    total += troops;
                }
            }
        }
        total
    }

    pub fn num_land_tiles(&self) -> u32 {
        self.map.num_land_tiles
    }

    fn relation_value(&self, a: u16, b: u16) -> i32 {
        self.player_by_small_id(a)
            .and_then(|p| p.relations.get(&b).copied())
            .unwrap_or(0)
    }

    /// TS `PlayerImpl.allRelationsSorted()`  -  sort-by-value (stable), tie-broken by
    /// insertion order, filtered to relations with still-alive players.
    pub fn all_relations_sorted(&self, sid: u16) -> Vec<(u16, i32)> {
        let Some(p) = self.player_by_small_id(sid) else {
            return Vec::new();
        };
        let mut out: Vec<(u16, i32)> = p
            .relations
            .iter()
            .filter(|(other, _)| self.player_by_small_id(*other).is_some_and(|o| o.alive))
            .map(|(other, &v)| (other, v))
            .collect();
        out.sort_by_key(|(_, v)| *v);
        out
    }

    fn relation_from_value(value: i32) -> Relation {
        if value < -50 {
            Relation::Hostile
        } else if value < 0 {
            Relation::Distrustful
        } else if value < 50 {
            Relation::Neutral
        } else {
            Relation::Friendly
        }
    }

    pub fn relation(&self, a: u16, b: u16) -> Relation {
        Self::relation_from_value(self.relation_value(a, b))
    }

    pub fn update_relation(&mut self, a: u16, b: u16, delta: i32) {
        if let Some(p) = self.player_by_small_id_mut(a) {
            let entry = p.relations.entry_or_insert(b, 0);
            *entry += delta;
        }
    }

    pub fn add_gold(&mut self, small_id: u16, amount: i64) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold += amount;
        }
    }

    pub fn is_allied_with(&self, a: u16, b: u16) -> bool {
        if a == b {
            return false;
        }
        self.alliances.iter().any(|al| {
            al.expires_at > self.ticks
                && ((al.requestor_small_id == a && al.recipient_small_id == b)
                    || (al.requestor_small_id == b && al.recipient_small_id == a))
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

    pub fn create_alliance_request(
        &mut self,
        requestor: u16,
        recipient: u16,
        tick: u32,
    ) -> bool {
        if self.wire.disable_alliances() {
            return false;
        }
        if self.is_allied_with(requestor, recipient) {
            return false;
        }
        if self
            .alliance_requests
            .iter()
            .any(|r| {
                r.status == AllianceRequestStatus::Pending
                    && r.requestor_small_id == requestor
                    && r.recipient_small_id == recipient
            })
        {
            return false;
        }
        if self
            .alliance_requests
            .iter()
            .any(|r| {
                r.status == AllianceRequestStatus::Pending
                    && r.requestor_small_id == recipient
                    && r.recipient_small_id == requestor
            })
        {
            self.accept_alliance_pair(recipient, requestor, tick);
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

    fn accept_alliance_pair(&mut self, requestor: u16, recipient: u16, tick: u32) {
        self.alliance_requests.retain(|r| {
            !((r.requestor_small_id == requestor
                && r.recipient_small_id == recipient
                && r.status == AllianceRequestStatus::Pending)
                || (r.requestor_small_id == recipient
                    && r.recipient_small_id == requestor
                    && r.status == AllianceRequestStatus::Pending))
        });
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
        self.update_relation(requestor, recipient, 100);
        self.update_relation(recipient, requestor, 100);
    }

    pub fn accept_alliance_request(&mut self, requestor: u16, recipient: u16, tick: u32) {
        if !self
            .alliance_requests
            .iter()
            .any(|r| {
                r.status == AllianceRequestStatus::Pending
                    && r.requestor_small_id == requestor
                    && r.recipient_small_id == recipient
            })
        {
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
        if !self.is_traitor(other) {
            self.mark_traitor(breaker);
        }
        self.alliances.retain(|al| {
            !((al.requestor_small_id == breaker && al.recipient_small_id == other)
                || (al.requestor_small_id == other && al.recipient_small_id == breaker))
        });
        self.update_relation(breaker, other, -100);
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

    pub fn add_alliance_extension_request(&mut self, from: u16, to: u16) {
        let tick = self.ticks;
        let duration = self.wire.alliance_duration();
        let Some(idx) = self.alliances.iter().position(|al| {
            al.expires_at > tick
                && ((al.requestor_small_id == from && al.recipient_small_id == to)
                    || (al.requestor_small_id == to && al.recipient_small_id == from))
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

    pub fn is_on_same_team(&self, _a: u16, _winner: &str) -> bool {
        false
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
        if a == b {
            return true;
        }
        // Match TS `isFriendly`: disconnected players are never considered friendly,
        // even if allied. This keeps `bordering_enemies` and `bordering_friends` lists
        // in sync with TS so that RNG consumption in `maybe_send_alliance_requests`
        // (and attack strategies) is identical between the two engines.
        if self.player_by_small_id(b).map_or(false, |p| p.is_disconnected) {
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
        self.spawn_phase
            || self.ticks_since_start() < self.wire.spawn_immunity_duration()
    }

    pub fn is_nation_spawn_immunity_active(&self) -> bool {
        self.spawn_phase
            || self.ticks_since_start() < self.wire.nation_spawn_immunity_duration()
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
        if self.is_friendly(attacker_small_id, defender_small_id) {
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
        self.alliances
            .iter()
            .filter(|al| al.expires_at > self.ticks)
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
        true
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
        if sender.tiles_owned == 0 || recipient.tiles_owned == 0 {
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

    pub fn too_close_to_existing_spawn(&self, center: TileRef, min_dist: u32) -> bool {
        for p in &self.players {
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

#[derive(Debug, Default)]
pub struct TickUpdates {
    pub hash: Option<HashUpdate>,
    pub win: Option<WinUpdate>,
}
