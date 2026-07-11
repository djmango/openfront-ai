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
    fn init(&mut self, game: &mut Game, tick: u32) {
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

        if let Some(ref tid) = self.target_player_id {
            if !game.has_player(tid) {
                self.active = false;
                return;
            }
            let def = game.player_by_id(tid).unwrap();
            let def_small_id = def.small_id;
            let def_type = def.player_type;
            if def_small_id == self.owner_small_id {
                self.active = false;
                return;
            }
            if game.is_friendly(self.owner_small_id, def_small_id) {
                self.active = false;
                return;
            }
            // TS `AttackExecution.init`: unconditional automatic temporary embargo (applied
            // before the `canAttackPlayer` gate below, regardless of its outcome) when
            // neither side is a raider Bot ("Don't let bots embargo since they can't trade
            // anyway"). Also rejects any pending alliance request the target has outstanding
            // toward the attacker.
            if owner_type != PlayerType::Bot && def_type != PlayerType::Bot {
                game.add_embargo(def_small_id, self.owner_small_id, true, tick);
                game.reject_alliance_request(def_small_id, self.owner_small_id);
            }
            if !game.can_attack_player(self.owner_small_id, def_small_id) {
                self.active = false;
                return;
            }
            self.target_is_player = true;
            self.target_small_id = def_small_id;
        } else {
            self.target_is_player = false;
            self.target_small_id = game.terra_nullius_id();
        }

        let mut start = self.start_troops.unwrap_or_else(|| {
            game.wire
                .attack_amount(owner_type, owner_troops)
        });
        // TS `AttackExecution.init` only clamps `startTroops` to the owner's
        // current troops inside `if (this.removeTroops)`. Boat-landed attacks
        // (`removeTroops == false`) carry troops already deducted from the
        // owner at boat departure, so clamping against the owner's unrelated
        // current troop pool here would wrongly shrink them.
        if self.remove_troops {
            start = start.min(owner_troops as f64);
            game.remove_troops(self.owner_small_id, start);
        }
        self.troops = start;
        // TS `stats().attack(owner, target, startTroops)` - recorded before
        // opposing-attack cancellation/merge, using the pre-merge start amount.
        game.adjust_attacks_sent(self.owner_small_id, start);

        if let Some(p) = game.player_by_small_id_mut(self.owner_small_id) {
            self.attack_id = p.id_prng.next_id();
        }
        game.register_land_attack(&self.attack_id, self.owner_small_id, self.target_small_id);

        if let Some(src) = self.source_tile {
            self.add_neighbors(game, src, game.ticks());
        } else {
            self.refresh_to_conquer(game);
        }

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
        if self.source_tile.is_none() {
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
            // TS `AttackExecution.init`: `this.target.updateRelation(this._owner,
            // relationChange)` - updates the TARGET's relation toward the OWNER
            // (getting attacked makes the victim more hostile), not the other
            // way around. `update_relation(a, b, ..)` updates a's relation
            // toward b, so the call is `(target, owner, ..)`, not `(owner,
            // target, ..)` (previously swapped - see the bots=5
            // curriculum-parity bisection this was found from).
            game.update_relation(self.target_small_id, self.owner_small_id, delta);
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
                // TS only applies the 25% malus to an explicitly canceled
                // player attack. An alliance formed after launch causes an
                // automatic retreat with every remaining troop returned.
                self.retreat(game, 0.0);
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
            // Equivalent of TS `if (!isLand(t) || isImpassable(t)) continue;`
            // in AttackExecution.tick. Without this, an impassable tile that
            // slipped into `to_conquer` (e.g. via the historical add_neighbors
            // gap) could actually be conquered below, giving it a real owner
            // in violation of the "impassable tiles are always owner 0"
            // invariant the rest of the engine assumes.
            if !game.is_land(tile_to_conquer) || game.is_impassable(tile_to_conquer) {
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
            self.set_troops(troop_count);

            if self.target_is_player {
                game.remove_troops(self.target_small_id, defender_loss);
            }
            game.conquer(self.owner_small_id, tile_to_conquer);
            self.handle_dead_defender(game);
        }
        self.set_troops(troop_count);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    #[test]
    fn alliance_formed_mid_attack_returns_all_remaining_troops() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "attacker".to_string(),
            small_id: 1,
            troops: 1_000,
            team: Some("Blue".to_string()),
            ..Default::default()
        });
        game.add_player(Player {
            id: "target".to_string(),
            small_id: 2,
            team: Some("Blue".to_string()),
            ..Default::default()
        });

        let mut attack = AttackExecution::new(1, Some("target".to_string()), Some(400.0));
        attack.initialized = true;
        attack.target_is_player = true;
        attack.target_small_id = 2;
        attack.troops = 400.0;

        attack.tick(&mut game, 1);

        assert_eq!(game.player_by_small_id(1).unwrap().troops, 1_400);
        assert!(!attack.is_active());
    }

    #[test]
    fn boat_landed_attack_cancels_opposing_land_attack() {
        let mut game = Game::default();
        game.end_spawn_phase();
        game.add_player(Player {
            id: "attacker".to_string(),
            small_id: 1,
            player_type: PlayerType::Bot,
            troops: 1_000,
            tiles_owned: 1,
            id_prng: PseudoRandom::new(1),
            ..Default::default()
        });
        game.add_player(Player {
            id: "defender".to_string(),
            small_id: 2,
            player_type: PlayerType::Bot,
            troops: 1_000,
            tiles_owned: 1,
            id_prng: PseudoRandom::new(2),
            ..Default::default()
        });
        game.add_execution(crate::execution::ExecEnum::Attack(AttackExecution::new(
            1,
            Some("defender".to_string()),
            Some(100.0),
        )));
        game.add_execution(crate::execution::ExecEnum::Attack(
            AttackExecution::new_with_source(
                2,
                Some("attacker".to_string()),
                Some(40.0),
                Some(0),
            ),
        ));

        game.execute_next_tick();

        let attacks = game.active_attacks_debug();
        assert_eq!(attacks.len(), 2);
        assert_eq!(attacks[0].0, 1);
        assert_eq!(attacks[0].1, 2);
        assert!(
            (attacks[0].2 - 60.0).abs() < f64::EPSILON,
            "unexpected attacks: {attacks:?}"
        );
        assert!(attacks[0].4);
        assert_eq!(attacks[1].0, 2);
        assert_eq!(attacks[1].1, 1);
        assert!(!attacks[1].3);
    }

    // Ported from AttackStats.test.ts. TS tracks a detailed per-category gold
    // ledger (`GOLD_INDEX_WAR`/`GOLD_INDEX_WORK` in `StatsImpl`) purely for UI
    // display; native has no such stats subsystem at all (see
    // docs/bot-ai-parity-nation-relations/README.md's documented "dead for
    // this dataset" carve-outs for similar UI-only bookkeeping). What IS
    // gameplay-relevant and ported here is the underlying mechanic those
    // stats are derived from: `Game::conquer_player`'s actual gold transfer,
    // and its "conquered player never attacked anyone" skip condition (TS
    // `GameImpl.conquerPlayer`'s `skipGoldTransfer`, tracked via
    // `attacks_sent_net`/TS's `ATTACK_INDEX_SENT`). TS's third test
    // ("...via territory annexation") exercises this exact same
    // `conquerPlayer` code path with different map geometry leading to the
    // same elimination trigger - not re-ported separately since native's
    // `handle_dead_defender` doesn't distinguish how a defender's tiles
    // dropped below the elimination threshold.
    #[test]
    fn conqueror_receives_half_a_human_defenders_gold_on_elimination() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "attacker".to_string(),
            small_id: 1,
            gold: 0,
            ..Default::default()
        });
        game.add_player(Player {
            id: "defender".to_string(),
            small_id: 2,
            gold: 100,
            ..Default::default()
        });
        // TS: "Player2 must attack to be considered active (otherwise gold
        // won't transfer)".
        game.adjust_attacks_sent(2, 50.0);

        game.conquer_player(1, 2);

        assert_eq!(game.player_by_small_id(1).unwrap().gold, 50);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 0);
    }

    #[test]
    fn no_gold_transfers_when_an_inactive_human_defender_is_eliminated() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "attacker".to_string(),
            small_id: 1,
            gold: 0,
            ..Default::default()
        });
        game.add_player(Player {
            id: "defender".to_string(),
            small_id: 2,
            gold: 100,
            ..Default::default()
        });

        game.conquer_player(1, 2);

        assert_eq!(game.player_by_small_id(1).unwrap().gold, 0);
        assert_eq!(game.player_by_small_id(2).unwrap().gold, 100);
    }
}

impl AttackExecution {
    pub fn attack_id(&self) -> &str {
        &self.attack_id
    }

    pub fn order_retreat(&mut self) {
        self.retreating = true;
    }

    /// TS `Attack.retreating()` - read by the RL obs attacks list.
    pub fn retreating(&self) -> bool {
        self.retreating
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

    /// TS `Attack.setTroops()` clamps to non-negative. Without this, a tick
    /// that overshoots troop_count into negative territory leaves the attack
    /// permanently stuck with negative troops (the top-of-loop `troop_count <
    /// 1.0` guard never re-fires once `num_tiles_per_tick` hits zero), and that
    /// negative balance later gets silently absorbed into a merged attack's
    /// troop count, desyncing from TS which never lets it go negative.
    pub fn set_troops(&mut self, troops: f64) {
        self.troops = troops.max(0.0);
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
        // TS `stats().attackCancel(...)` only fires for genuine order-retreats
        // (`Attack.retreated()`), not the auto-retreats used to end an attack
        // that ran out of tiles/troops or hit a since-allied target.
        if self.retreated {
            game.adjust_attacks_sent(self.owner_small_id, -survivors);
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
            // Equivalent of TS `if (isWater(t) || isImpassable(t) || ownerID(t)
            // !== targetSmallID) continue;` in AttackExecution.addNeighbors.
            // Impassable tiles always have owner 0 (TerraNullius), so when
            // attacking TerraNullius they'd otherwise slip past the owner
            // check below and get added as a phantom border/candidate tile -
            // consuming an extra `self.random.next_int(0, 7)` draw that TS
            // never makes, desyncing the shared per-attack PRNG stream (see
            // the `self.random` usage a few lines down and in
            // `AttackExecution::tick`'s `border_size + random.next_int(0, 5)`).
            if game.map.is_impassable(neighbor) {
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
