//! Core game state + tick driver.

use crate::core::config::Config as WireConfig;
use crate::execution::{ExecEnum, Execution, HashUpdate, WinUpdate};
use crate::hash::game_hash;
use crate::map::{GameMap, TileRef};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;
use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct PlayerInfo {
    pub name: String,
    pub player_type: PlayerType,
    pub client_id: Option<String>,
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct Unit {
    pub id: i32,
    pub unit_type: String,
    pub tile: i32,
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
    pub border_tiles: Vec<TileRef>,
    pub owned_tiles: Vec<TileRef>,
    pub last_cluster_calc: u32,
    pub last_tile_change: u32,
    pub marked_traitor_tick: i32,
    pub relations: HashMap<u16, i32>,
    pub is_disconnected: bool,
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
            border_tiles: Vec::new(),
            owned_tiles: Vec::new(),
            last_cluster_calc: 0,
            last_tile_change: 0,
            marked_traitor_tick: -1,
            relations: HashMap::new(),
            is_disconnected: false,
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
    mini_water_astar: crate::water::WaterAstarScratch,
    path_buf: Vec<TileRef>,
    next_unit_id: i32,
    pub hash_enabled: bool,
    pub tribe_batch: crate::bot::TribeBatch,
    pub nation_batch: crate::execution::NationBatch,
    /// Same-tick inits not yet appended to `execs` (TS `outgoingAttacks()` batch ordering).
    init_merge_batch: Option<*mut Vec<ExecEnum>>,
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
            path_buf: Vec::new(),
            next_unit_id: 1,
            hash_enabled: true,
            tribe_batch: crate::bot::TribeBatch::new(),
            nation_batch: crate::execution::NationBatch::new(),
            init_merge_batch: None,
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
    ) -> Self {
        let seed = simple_hash(&game_id);
        let tile_count = (map.width * map.height) as usize;
        let mini_count = (mini_map.width * mini_map.height) as usize;
        let water_component = crate::water::build_water_components(&map);
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
            path_buf: Vec::with_capacity(256),
            tribe_batch: crate::bot::TribeBatch::new(),
            nation_batch: crate::execution::NationBatch::new(),
            ..Default::default()
        }
    }

    pub fn register_tribe(&mut self, small_id: u16, player_id: &str) {
        self.tribe_batch.register(small_id, player_id);
    }

    pub fn plan_water_path(&mut self, from: TileRef, to: TileRef) -> bool {
        crate::water::transport_path_into(
            &self.map,
            &self.mini_map,
            &mut self.mini_water_astar,
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
        let small_id = self.next_player_id;
        self.next_player_id += 1;
        let client_id = info.client_id.clone().unwrap_or_default();
        let id_hash = simple_hash(&info.id);
        let idx = self.players.len();
        self.players.push(Player {
            id: info.id.clone(),
            name: info.name.clone(),
            client_id: client_id.clone(),
            small_id,
            id_hash,
            player_type: info.player_type,
            troops,
            border_tiles: Vec::new(),
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
        let refresh_borders = !self.spawn_phase;
        for t in tiles {
            if self.map.owner_id(t) == small_id {
                self.map.set_owner_id(t, 0);
            }
            if refresh_borders {
                self.refresh_borders_around(t);
            }
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
            .map(|tiles| tiles.clone())
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

    pub fn border_tiles_of(&self, small_id: u16) -> Option<&Vec<TileRef>> {
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
                if !p.border_tiles.contains(&tile) {
                    p.border_tiles.push(tile);
                }
            } else {
                if let Some(i) = p.border_tiles.iter().position(|&t| t == tile) {
                    p.border_tiles.remove(i);
                }
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
        self.conquer_one(small_id, tile, !self.spawn_phase);
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
                if let Some(i) = p.border_tiles.iter().position(|&t| t == tile) {
                    p.border_tiles.remove(i);
                }
                p.owned_tiles.retain(|&t| t != tile);
                p.last_tile_change = tick;
            }
        }
        self.map.set_owner_id(tile, small_id);
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

    pub fn add_execution(&mut self, exec: ExecEnum) {
        self.uninit.push(exec);
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

    pub fn build_unit(&mut self, small_id: u16, unit_type: &str, tile: TileRef) -> i32 {
        let id = self.next_unit_id;
        self.next_unit_id += 1;
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.units.push(Unit {
                id,
                unit_type: unit_type.to_string(),
                tile: tile as i32,
            });
        }
        id
    }

    pub fn remove_unit(&mut self, small_id: u16, unit_id: i32) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.units.retain(|u| u.id != unit_id);
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

    /// TS `AttackExecution.init` - cancel opposing mutual attacks.
    pub fn cancel_opposing_land_attacks(
        &mut self,
        owner_small_id: u16,
        target_small_id: u16,
        troops: &mut f64,
        active: &mut bool,
    ) {
        if !*active {
            return;
        }
        let exec_count = self.execs.len();
        for i in 0..exec_count {
            let should_cancel = matches!(
                &self.execs[i],
                ExecEnum::Attack(atk)
                    if atk.is_active()
                        && atk.attack_live()
                        && atk.is_initialized()
                        && atk.owner_small_id() == target_small_id
                        && atk.target_small_id() == owner_small_id
                        && atk.source_tile().is_none()
            );
            if !should_cancel {
                continue;
            }
            let game = self as *mut Game;
            unsafe {
                if let ExecEnum::Attack(incoming) = &mut self.execs[i] {
                    let incoming_troops = incoming.troops();
                    if incoming_troops > *troops {
                        incoming.set_troops(incoming_troops - *troops);
                        *active = false;
                        return;
                    }
                    *troops -= incoming_troops;
                    incoming.kill_attack();
                }
            }
        }
    }

    /// TS `AttackExecution.init` - merge all same-target outgoing land attacks.
    pub fn merge_outgoing_land_attacks(
        &mut self,
        owner_small_id: u16,
        target_small_id: u16,
        troops: &mut f64,
    ) {
        if let Some(ptr) = self.init_merge_batch {
            let batch = unsafe { &mut *ptr };
            for exec in batch.iter_mut() {
                let should_merge = matches!(
                    exec,
                    ExecEnum::Attack(atk)
                        if atk.is_active()
                            && atk.attack_live()
                            && atk.is_initialized()
                            && atk.owner_small_id() == owner_small_id
                            && atk.target_small_id() == target_small_id
                            && atk.source_tile().is_none()
                );
                if !should_merge {
                    continue;
                }
                if let ExecEnum::Attack(outgoing) = exec {
                    *troops += outgoing.troops();
                    outgoing.kill_attack();
                }
            }
        }

        let exec_count = self.execs.len();
        for i in 0..exec_count {
            let should_merge = matches!(
                &self.execs[i],
                ExecEnum::Attack(atk)
                    if atk.is_active()
                        && atk.attack_live()
                        && atk.is_initialized()
                        && atk.owner_small_id() == owner_small_id
                        && atk.target_small_id() == target_small_id
                        && atk.source_tile().is_none()
            );
            if !should_merge {
                continue;
            }
            unsafe {
                if let ExecEnum::Attack(outgoing) = &mut self.execs[i] {
                    *troops += outgoing.troops();
                    outgoing.kill_attack();
                }
            }
        }
    }

    /// Largest incoming land attack from candidate attackers (TS cluster `getCapturingPlayer`).
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
            if let Some(p) = self.player_by_small_id(attacker) {
                if !p.alive {
                    continue;
                }
            }
            out.push(IncomingAttack {
                attacker_small_id: attacker,
                troops: atk.troops(),
            });
        }
        out
    }

    pub fn outgoing_land_troops(&self, attacker_small_id: u16) -> f64 {
        self.execs
            .iter()
            .filter_map(|e| match e {
                ExecEnum::Attack(atk)
                    if atk.is_active()
                        && atk.attack_live()
                        && atk.is_initialized()
                        && atk.owner_small_id() == attacker_small_id
                        && atk.source_tile().is_none() =>
                {
                    Some(atk.troops())
                }
                _ => None,
            })
            .sum()
    }

    pub fn num_land_tiles(&self) -> u32 {
        self.map.num_land_tiles
    }

    fn relation_value(&self, a: u16, b: u16) -> i32 {
        self.player_by_small_id(a)
            .and_then(|p| p.relations.get(&b).copied())
            .unwrap_or(0)
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
            let entry = p.relations.entry(b).or_insert(0);
            *entry += delta;
        }
    }

    pub fn add_gold(&mut self, small_id: u16, amount: i64) {
        if let Some(p) = self.player_by_small_id_mut(small_id) {
            p.gold += amount;
        }
    }

    pub fn is_allied_with(&self, _a: u16, _b: u16) -> bool {
        false
    }

    pub fn is_on_same_team(&self, _a: u16, _winner: &str) -> bool {
        false
    }

    pub fn players_on_same_team(&self, _a: u16, _b: u16) -> bool {
        false
    }

    pub fn is_friendly(&self, a: u16, b: u16) -> bool {
        if a == b {
            return true;
        }
        self.is_allied_with(a, b)
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

    pub fn allied_small_ids(&self, _small_id: u16) -> Vec<u16> {
        Vec::new()
    }

    pub fn alliance_count(&self, _small_id: u16) -> usize {
        0
    }

    pub fn can_send_emoji(&self, sender_small_id: u16, recipient: Option<u16>) -> bool {
        if recipient.is_some_and(|r| r == sender_small_id) {
            return false;
        }
        true
    }

    pub fn can_send_alliance_request(&self, sender_small_id: u16, recipient_small_id: u16) -> bool {
        if sender_small_id == recipient_small_id {
            return false;
        }
        if !self
            .player_by_small_id(sender_small_id)
            .is_some_and(|p| p.alive)
        {
            return false;
        }
        if !self
            .player_by_small_id(recipient_small_id)
            .is_some_and(|p| p.alive)
        {
            return false;
        }
        if self.is_friendly(sender_small_id, recipient_small_id) {
            return false;
        }
        true
    }

    pub fn incoming_alliance_requests(&self, _small_id: u16) -> Vec<IncomingAllianceRequest> {
        Vec::new()
    }

    pub fn alliance_extension_candidates(&self, _small_id: u16) -> Vec<AllianceExtensionCandidate> {
        Vec::new()
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

    /// TS appends all inited execs after `removeInactiveExecutions`, so attacks stay
    /// after `PlayerExecution`. Rust pushes attacks mid-init for merge parity, then
    /// `retain` compacts them forward - restore attack-at-end ordering each tick.
    fn move_land_attacks_to_end(execs: &mut Vec<ExecEnum>) {
        let drained: Vec<_> = execs.drain(..).collect();
        let mut attacks = Vec::new();
        for e in drained {
            let is_land_attack = matches!(
                &e,
                ExecEnum::Attack(a) if a.is_active() && a.source_tile().is_none()
            );
            if is_land_attack {
                attacks.push(e);
            } else {
                execs.push(e);
            }
        }
        execs.extend(attacks);
    }

    pub fn execute_next_tick(&mut self) -> TickUpdates {
        let tick = self.ticks;

        let game = self as *mut Game;
        let exec_len = self.execs.len();
        for i in 0..exec_len {
            let exec = &mut self.execs[i];
            if exec.is_spawn_timer()
                && (!self.spawn_phase || exec.active_during_spawn())
                && exec.is_active()
            {
                unsafe {
                    exec.tick(&mut *game, tick);
                }
            }
        }
        for i in 0..exec_len {
            let exec = &mut self.execs[i];
            if exec.is_spawn_timer() {
                continue;
            }
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
                // TS always appends inited execs; `removeInactiveExecutions` compacts later.
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
