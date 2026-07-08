//! Multi-tick land attack (TS `AttackExecution.ts` subset for hash parity).

use super::flat_heap::FlatBinaryHeap;
use super::ordered_tiles::OrderedTiles;
use super::Execution;
use crate::game::{Game, PlayerType};
use crate::map::{TerrainType, TileRef};
use crate::prng::PseudoRandom;

pub struct AttackExecution {
    owner_small_id: u16,
    target_player_id: Option<String>,
    start_troops: Option<f64>,
    troops: f64,
    target_small_id: u16,
    target_is_player: bool,
    to_conquer: FlatBinaryHeap,
    border_tiles: OrderedTiles,
    source_tile: Option<TileRef>,
    remove_troops: bool,
    random: PseudoRandom,
    attack_id: String,
    retreating: bool,
    retreated: bool,
    active: bool,
    /// TS `Attack.isActive()` - false after `delete()` while exec stays active one tick.
    attack_live: bool,
    initialized: bool,
}

impl AttackExecution {
    pub fn new(
        owner_small_id: u16,
        target_player_id: Option<String>,
        start_troops: Option<f64>,
    ) -> Self {
        Self::new_with_source(owner_small_id, target_player_id, start_troops, None)
    }

    pub fn new_with_source(
        owner_small_id: u16,
        target_player_id: Option<String>,
        start_troops: Option<f64>,
        source_tile: Option<TileRef>,
    ) -> Self {
        Self {
            owner_small_id,
            target_player_id,
            start_troops,
            troops: 0.0,
            target_small_id: 0,
            target_is_player: false,
            to_conquer: FlatBinaryHeap::new(1024),
            border_tiles: OrderedTiles::default(),
            source_tile,
            remove_troops: source_tile.is_none(),
            random: PseudoRandom::new(123),
            attack_id: String::new(),
            retreating: false,
            retreated: false,
            active: true,
            attack_live: true,
            initialized: false,
        }
    }

    pub fn from_intent(
        client_id: &str,
        game: &Game,
        troops: f64,
        target_player_id: Option<String>,
    ) -> Option<(u16, Option<String>, Option<f64>)> {
        let owner = game.player_by_client_id(client_id)?;
        Some((
            owner.small_id,
            target_player_id,
            Some(troops),
        ))
    }
}

impl Execution for AttackExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active || self.initialized {
            return;
        }
        self.initialized = true;

        if game.in_spawn_phase() {
            self.active = false;
            return;
        }

        let Some(owner) = game.player_by_small_id(self.owner_small_id) else {
            self.active = false;
            return;
        };
        let owner_type = owner.player_type;
        let owner_troops = owner.troops;

        if !owner.alive {
            self.active = false;
            return;
        }

        if let Some(ref tid) = self.target_player_id {
            if !game.has_player(tid) {
                self.active = false;
                return;
            }
            let def = game.player_by_id(tid).unwrap();
            if def.small_id == self.owner_small_id {
                self.active = false;
                return;
            }
            if game.is_friendly(self.owner_small_id, def.small_id) {
                self.active = false;
                return;
            }
            if !game.can_attack_player(self.owner_small_id, def.small_id) {
                self.active = false;
                return;
            }
            self.target_is_player = true;
            self.target_small_id = def.small_id;
        } else {
            self.target_is_player = false;
            self.target_small_id = game.terra_nullius_id();
        }

        let mut start = self.start_troops.unwrap_or_else(|| {
            game.wire
                .attack_amount(owner_type, owner_troops)
        });
        start = start.min(owner_troops as f64);
        if self.remove_troops {
            game.remove_troops(self.owner_small_id, start);
        }
        self.troops = start;

        if let Some(p) = game.player_by_small_id_mut(self.owner_small_id) {
            self.attack_id = p.id_prng.next_id();
        }
        game.register_land_attack(&self.attack_id, self.owner_small_id, self.target_small_id);

        if let Some(src) = self.source_tile {
            self.add_neighbors(game, src, game.ticks());
        } else {
            self.refresh_to_conquer(game);
        }

        if self.source_tile.is_none() {
            game.cancel_opposing_land_attacks(
                self.owner_small_id,
                self.target_small_id,
                &self.attack_id,
                &mut self.troops,
                &mut self.active,
            );
            if !self.active {
                game.unregister_land_attack(
                    &self.attack_id,
                    self.owner_small_id,
                    self.target_small_id,
                );
                return;
            }
            game.merge_outgoing_land_attacks(
                self.owner_small_id,
                self.target_small_id,
                &self.attack_id,
                &mut self.troops,
            );
        }

        if self.target_is_player {
            let difficulty = game.wire.game_config().difficulty.as_str();
            let delta = match difficulty {
                "Easy" => -60,
                "Medium" => -70,
                "Hard" => -80,
                "Impossible" => -100,
                _ => -70,
            };
            game.update_relation(self.owner_small_id, self.target_small_id, delta);
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || !self.initialized {
            return;
        }
        if !self.attack_live {
            self.active = false;
            return;
        }

        if self.retreated {
            let malus = if self.target_is_player { 25.0 } else { 0.0 };
            self.retreat(game, malus);
            return;
        }
        if self.retreating {
            return;
        }

        if self.target_is_player {
            if game.is_friendly(self.owner_small_id, self.target_small_id) {
                self.retreat(game, 25.0);
                return;
            }
        }

        let mut troop_count = self.troops;
        let owner_type = game
            .player_by_small_id(self.owner_small_id)
            .map(|p| p.player_type)
            .unwrap_or(PlayerType::Human);

        let mut num_tiles_per_tick = {
            let border_size = self.border_tiles.len() as f64;
            let defender_troops = if self.target_is_player {
                game.player_by_small_id(self.target_small_id)
                    .map(|p| p.troops)
                    .unwrap_or(0)
            } else {
                0
            };
            game.wire.attack_tiles_per_tick(
                troop_count,
                owner_type,
                self.target_is_player,
                defender_troops,
                border_size + self.random.next_int(0, 5) as f64,
            )
        };

        while num_tiles_per_tick > 0.0 {
            if troop_count < 1.0 {
                self.kill_attack(game);
                self.active = false;
                return;
            }

            if self.to_conquer.is_empty() {
                self.refresh_to_conquer(game);
                self.troops = troop_count;
                self.retreat(game, 0.0);
                return;
            }

            let Some(tile_to_conquer) = self.to_conquer.dequeue() else {
                break;
            };

            self.remove_border_tile(tile_to_conquer);

            let mut on_border = false;
            game.map.for_each_neighbor4(tile_to_conquer, |n| {
                if game.map.owner_id(n) == self.owner_small_id {
                    on_border = true;
                }
            });

            if game.map.owner_id(tile_to_conquer) != self.target_small_id || !on_border {
                continue;
            }
            if !game.is_land(tile_to_conquer) {
                continue;
            }

            self.add_neighbors(game, tile_to_conquer, tick);

            let terrain = game.terrain_type(tile_to_conquer);
            let (attacker_loss, defender_loss, tiles_used) = if self.target_is_player {
                game.attack_logic_at_tile(
                    troop_count,
                    self.owner_small_id,
                    self.target_small_id,
                    tile_to_conquer,
                    true,
                )
            } else {
                game.attack_logic_at_tile(
                    troop_count,
                    self.owner_small_id,
                    self.target_small_id,
                    tile_to_conquer,
                    false,
                )
            };

            num_tiles_per_tick -= tiles_used;
            troop_count -= attacker_loss;
            self.troops = troop_count;

            if self.target_is_player {
                game.remove_troops(self.target_small_id, defender_loss);
            }
            game.conquer(self.owner_small_id, tile_to_conquer);
            self.handle_dead_defender(game);
        }
        self.troops = troop_count;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }

    fn as_attack(&mut self) -> Option<&mut AttackExecution> {
        Some(self)
    }
}

impl AttackExecution {
    pub fn attack_id(&self) -> &str {
        &self.attack_id
    }

    pub fn order_retreat(&mut self) {
        self.retreating = true;
    }

    pub fn execute_retreat(&mut self) {
        self.retreated = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn owner_small_id(&self) -> u16 {
        self.owner_small_id
    }

    pub fn target_small_id(&self) -> u16 {
        self.target_small_id
    }

    pub fn troops(&self) -> f64 {
        self.troops
    }

    pub fn source_tile(&self) -> Option<TileRef> {
        self.source_tile
    }

    pub fn set_troops(&mut self, troops: f64) {
        self.troops = troops;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn attack_live(&self) -> bool {
        self.attack_live
    }

    pub fn border_tile_count(&self) -> usize {
        self.border_tiles.len()
    }

    pub fn to_conquer_len(&self) -> usize {
        self.to_conquer.len()
    }

    /// TS `Attack.delete()` - remove from outgoing/incoming registries, defer exec cleanup.
    pub fn kill_attack(&mut self, game: &mut Game) {
        if self.attack_live {
            game.unregister_land_attack(
                &self.attack_id,
                self.owner_small_id,
                self.target_small_id,
            );
        }
        self.attack_live = false;
    }

    pub fn can_merge_with(&self, owner_small_id: u16, target_player_id: &Option<String>) -> bool {
        self.owner_small_id == owner_small_id && &self.target_player_id == target_player_id
    }

    pub fn absorb_pending_troops(&mut self, extra: Option<f64>) {
        let add = extra.unwrap_or(0.0);
        if add <= 0.0 {
            return;
        }
        match self.start_troops {
            Some(v) => self.start_troops = Some(v + add),
            None => self.start_troops = Some(add),
        }
    }

    pub fn absorb_troops(&mut self, game: &mut Game, extra: Option<f64>) {
        let Some(owner) = game.player_by_small_id(self.owner_small_id).cloned() else {
            return;
        };
        let add = extra.unwrap_or_else(|| {
            game.wire.attack_amount(owner.player_type, owner.troops)
        });
        let add = add.min(owner.troops as f64);
        if add < 1.0 {
            return;
        }
        game.remove_troops(self.owner_small_id, add);
        self.troops += add;
    }

    fn retreat(&mut self, game: &mut Game, malus_percent: f64) {
        let deaths = self.troops * (malus_percent / 100.0);
        let mut troops = self.troops;
        if !self.remove_troops && self.source_tile.is_none() {
            troops -= self.start_troops.unwrap_or(0.0);
        }
        let survivors = (troops - deaths).max(0.0);
        if survivors >= 1.0 {
            game.add_troops(self.owner_small_id, survivors);
        }
        self.troops = 0.0;
        self.kill_attack(game);
        self.active = false;
    }

    fn refresh_to_conquer(&mut self, game: &Game) {
        self.to_conquer.clear();
        self.border_tiles.clear();
        if let Some(border) = game.border_tiles_of(self.owner_small_id) {
            let tick = game.ticks();
            for tile in border.iter() {
                self.add_neighbors(game, tile, tick);
            }
        }
    }

    fn add_border_tile(&mut self, tile: TileRef) {
        self.border_tiles.insert(tile);
    }

    fn remove_border_tile(&mut self, tile: TileRef) {
        self.border_tiles.remove(tile);
    }

    fn handle_dead_defender(&mut self, game: &mut Game) {
        if !self.target_is_player {
            return;
        }
        let Some(defender) = game.player_by_small_id(self.target_small_id) else {
            return;
        };
        if defender.tiles_owned >= 100 {
            return;
        }
        let target_small = self.target_small_id;
        let owner_small = self.owner_small_id;

        game.conquer_player(owner_small, target_small);

        for _ in 0..10 {
            let tiles: Vec<TileRef> = game
                .player_by_small_id(target_small)
                .map(|p| p.owned_tiles.clone())
                .unwrap_or_default();
            for tile in tiles {
                let mut neighbors = [TileRef::MAX; 4];
                let mut n = 0usize;
                game.map.for_each_neighbor4(tile, |nb| {
                    if n < 4 {
                        neighbors[n] = nb;
                        n += 1;
                    }
                });
                let mut borders_owner = false;
                for i in 0..n {
                    if game.map.owner_id(neighbors[i]) == owner_small {
                        borders_owner = true;
                        break;
                    }
                }
                if borders_owner {
                    game.conquer(owner_small, tile);
                    continue;
                }
                let mut captured = false;
                for i in 0..n {
                    if captured {
                        break;
                    }
                    let oid = game.map.owner_id(neighbors[i]);
                    if oid == 0 || oid == target_small {
                        continue;
                    }
                    if game.is_friendly(oid, target_small) {
                        continue;
                    }
                    game.conquer(oid, tile);
                    captured = true;
                }
            }
        }
    }

    fn add_neighbors(&mut self, game: &Game, tile: TileRef, tick: u32) {
        game.map.for_each_neighbor4(tile, |neighbor| {
            if game.map.is_water(neighbor) {
                return;
            }
            if game.map.owner_id(neighbor) != self.target_small_id {
                return;
            }

            self.add_border_tile(neighbor);

            let mut num_owned_by_me = 0u32;
            game.map.for_each_neighbor4(neighbor, |inner| {
                if game.map.owner_id(inner) == self.owner_small_id {
                    num_owned_by_me += 1;
                }
            });

            let mag = match game.terrain_type(neighbor) {
                TerrainType::Plains => 1.0,
                TerrainType::Highland => 1.5,
                TerrainType::Mountain => 2.0,
                _ => 0.0,
            };

            let priority = {
                let p = (self.random.next_int(0, 7) as f64 + 10.0)
                    * (1.0 - num_owned_by_me as f64 * 0.5 + mag as f64 / 2.0)
                    + tick as f64;
                p as f32
            };
            self.to_conquer.enqueue(neighbor, priority);
        });
    }
}
