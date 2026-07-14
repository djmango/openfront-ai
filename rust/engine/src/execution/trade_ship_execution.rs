//! Trade ship voyage (`TradeShipExecution.ts`).
//!
//! Warship piracy is modeled end-to-end: `WarshipExecution::hunt_trade_ship`
//! calls `Game::capture_unit`, and this execution detects the owner change
//! (`ship_owner != orig_owner_small_id`), sets `was_captured`, redirects the
//! voyage to the capturer's nearest port in the same water component, and
//! pays voyage gold to the pirate on `complete`. Port ends can also change
//! owner via land conquest, so both ends are re-resolved by unit id every
//! tick rather than frozen at spawn time.

use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;

pub struct TradeShipExecution {
    orig_owner_small_id: u16,
    src_port_unit_id: i32,
    src_port_owner_small_id: Option<u16>,
    dst_port_unit_id: i32,
    dst_port_owner_small_id: Option<u16>,
    ship_unit_id: Option<i32>,
    path: Vec<TileRef>,
    path_idx: usize,
    path_dst: Option<TileRef>,
    tiles_traveled: u32,
    was_captured: bool,
    active: bool,
}

impl TradeShipExecution {
    pub fn new(
        orig_owner_small_id: u16,
        src_port_unit_id: i32,
        dst_port_unit_id: i32,
        dst_port_owner_small_id: u16,
    ) -> Self {
        Self {
            orig_owner_small_id,
            src_port_unit_id,
            src_port_owner_small_id: Some(orig_owner_small_id),
            dst_port_unit_id,
            dst_port_owner_small_id: Some(dst_port_owner_small_id),
            ship_unit_id: None,
            path: Vec::new(),
            path_idx: 0,
            path_dst: None,
            tiles_traveled: 0,
            was_captured: false,
            active: true,
        }
    }

    pub fn ship_unit_id(&self) -> Option<i32> {
        self.ship_unit_id
    }

    /// Test-only constructor for a trade ship whose backing `Unit` was already built by
    /// the caller (e.g. via `Game::build_unit`), bypassing `tick()`'s lazy first-tick spawn
    /// (which would otherwise also immediately try to path/complete the voyage). Registers
    /// with `Game::trade_ship_destination_owner` once added via `Game::add_execution` +
    /// `execute_next_tick`, without ever needing this execution's own `tick()` to run.
    /// Mirrors `WarshipExecution::new_for_test`'s naming/rationale.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        orig_owner_small_id: u16,
        dst_port_unit_id: i32,
        ship_unit_id: i32,
    ) -> Self {
        Self {
            orig_owner_small_id,
            src_port_unit_id: 0,
            src_port_owner_small_id: Some(orig_owner_small_id),
            dst_port_unit_id,
            dst_port_owner_small_id: None,
            ship_unit_id: Some(ship_unit_id),
            path: Vec::new(),
            path_idx: 0,
            path_dst: None,
            tiles_traveled: 0,
            was_captured: false,
            active: true,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test_with_destination_owner(
        orig_owner_small_id: u16,
        dst_port_unit_id: i32,
        ship_unit_id: i32,
        dst_port_owner_small_id: u16,
    ) -> Self {
        let mut exec = Self::new_for_test(orig_owner_small_id, dst_port_unit_id, ship_unit_id);
        exec.dst_port_owner_small_id = Some(dst_port_owner_small_id);
        exec
    }

    pub fn destination_port_unit_id(&self) -> i32 {
        self.dst_port_unit_id
    }

    pub fn cached_destination_port_owner_small_id(&self) -> Option<u16> {
        self.dst_port_owner_small_id
    }

    fn refresh_path(&mut self, game: &mut Game, from: TileRef, to: TileRef) -> bool {
        if !game.plan_water_path(from, to) {
            return false;
        }
        self.path.clear();
        self.path.extend_from_slice(game.planned_water_path());
        if self.path.is_empty() || self.path.first() != Some(&from) {
            self.path.insert(0, from);
        }
        self.path_idx = if self.path.first() == Some(&from) {
            1
        } else {
            0
        };
        self.path_dst = Some(to);
        true
    }

    fn next_path_tile(&mut self, game: &mut Game, from: TileRef, to: TileRef) -> Option<TileRef> {
        if self.path_dst != Some(to) || self.path.is_empty() {
            if !self.refresh_path(game, from, to) {
                return None;
            }
        }
        if self.path_idx > 0 {
            let expected = self.path[self.path_idx - 1];
            if from != expected {
                // TS `PathFinderStepper.next` invalidates the cached path on
                // any expected-position mismatch.  Do not scan forward in the
                // stale path: if `from` appears later, skipping to `pos + 1`
                // drops intermediate movement ticks and makes ships arrive or
                // get captured early.
                self.path.clear();
                self.path_idx = 0;
                if !self.refresh_path(game, from, to) {
                    return None;
                }
            }
        }
        if self.path_idx >= self.path.len() {
            return None;
        }
        let next = self.path[self.path_idx];
        self.path_idx += 1;
        Some(next)
    }

    /// TS `TradeShipExecution.complete`.
    fn complete(&mut self, game: &mut Game) {
        self.active = false;
        let Some(uid) = self.ship_unit_id else { return };
        let Some(ship_owner) = game.find_unit_owner(uid) else {
            return;
        };
        game.remove_unit(ship_owner, uid);
        let gold = game.wire.trade_ship_gold(self.tiles_traveled as f64);
        if self.was_captured {
            game.add_gold(ship_owner, gold);
            return;
        }
        if let Some(src_owner) = game
            .find_unit_owner(self.src_port_unit_id)
            .or(self.src_port_owner_small_id)
        {
            game.add_gold(src_owner, gold);
        }
        if let Some(dst_owner) = game
            .find_unit_owner(self.dst_port_unit_id)
            .or(self.dst_port_owner_small_id)
        {
            game.add_gold(dst_owner, gold);
        }
    }
}

impl Execution for TradeShipExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }

        // TS `TradeShipExecution.tick`: the ship is lazily built on the first tick.
        if self.ship_unit_id.is_none() {
            let Some(src_tile) = game.unit_tile_of(self.orig_owner_small_id, self.src_port_unit_id)
            else {
                self.active = false;
                return;
            };
            // TS `origOwner.canBuild(TradeShip, srcPort.tile())` -> `tradeShipSpawn`:
            // owner must still have an (active) Port at that tile.
            let has_port = game
                .player_by_small_id(self.orig_owner_small_id)
                .is_some_and(|p| {
                    p.units.iter().any(|u| {
                        u.id == self.src_port_unit_id
                            && u.unit_type == unit_type::PORT
                            && !u.under_construction
                    })
                });
            if !has_port {
                self.active = false;
                return;
            }
            let uid = game.build_unit(self.orig_owner_small_id, unit_type::TRADE_SHIP, src_tile);
            self.ship_unit_id = Some(uid);
        }
        let uid = self.ship_unit_id.unwrap();

        let Some(ship_owner) = game.find_unit_owner(uid) else {
            self.active = false;
            return;
        };
        if ship_owner != self.orig_owner_small_id {
            self.was_captured = true;
        }
        let Some(cur_tile) = game.unit_tile_of(ship_owner, uid) else {
            self.active = false;
            return;
        };

        if let Some(owner) = game.find_unit_owner(self.src_port_unit_id) {
            self.src_port_owner_small_id = Some(owner);
        }
        // TS: `!this._dstPort.isActive()` (port destroyed/demolished since spawn).
        let mut dst_owner = game.find_unit_owner(self.dst_port_unit_id);
        if let Some(owner) = dst_owner {
            self.dst_port_owner_small_id = Some(owner);
        }

        // TS: `dstPortOwner.id() === srcPort.owner().id()` - the src port (possibly
        // captured via land conquest since spawn) now belongs to the dst owner too.
        if dst_owner.is_some() && game.find_unit_owner(self.src_port_unit_id) == dst_owner {
            game.remove_unit(ship_owner, uid);
            self.active = false;
            return;
        }

        if !self.was_captured && dst_owner.is_none_or(|owner| !game.can_trade(ship_owner, owner)) {
            game.remove_unit(ship_owner, uid);
            self.active = false;
            return;
        }

        if self.was_captured && dst_owner != Some(ship_owner) {
            let component = game.get_water_component(cur_tile);
            let nearest_port = game.player_by_small_id(ship_owner).and_then(|owner| {
                owner
                    .units
                    .iter()
                    .filter(|unit| {
                        unit.unit_type == unit_type::PORT
                            && !unit.under_construction
                            && component.is_some_and(|component| {
                                game.has_water_component(unit.tile as TileRef, component)
                            })
                    })
                    .min_by_key(|unit| game.manhattan_dist(unit.tile as TileRef, cur_tile))
                    .map(|unit| unit.id)
            });
            let Some(port_id) = nearest_port else {
                game.remove_unit(ship_owner, uid);
                self.active = false;
                return;
            };
            if self.dst_port_unit_id != port_id {
                self.dst_port_unit_id = port_id;
                self.dst_port_owner_small_id = Some(ship_owner);
                self.path.clear();
                self.path_idx = 0;
                self.path_dst = None;
            }
            dst_owner = Some(ship_owner);
        }

        let Some(dst_owner) = dst_owner else {
            game.remove_unit(ship_owner, uid);
            self.active = false;
            return;
        };
        let Some(dst_tile) = game.unit_tile_of(dst_owner, self.dst_port_unit_id) else {
            game.remove_unit(ship_owner, uid);
            self.active = false;
            return;
        };

        if cur_tile == dst_tile {
            self.complete(game);
            return;
        }

        let Some(next) = self.next_path_tile(game, cur_tile, dst_tile) else {
            self.complete(game);
            return;
        };
        if game.map.is_water(next) && game.map.is_shoreline(next) {
            let tick = game.ticks() as i32;
            if let Some(ship) = game.unit_mut(ship_owner, uid) {
                ship.last_safe_from_pirates_tick = tick;
            }
        }
        game.move_unit(ship_owner, uid, next);
        self.tiles_traveled += 1;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod piracy_tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::execution::Execution;
    use crate::game::{PlayerInfo, PlayerType};
    use crate::map::{GameMap, MapMeta};

    fn water_game(width: u32, height: u32) -> Game {
        let n = (width * height) as usize;
        let map = GameMap::from_terrain_bytes(
            &MapMeta {
                width,
                height,
                num_land_tiles: 0,
            },
            &vec![0u8; n],
        )
        .expect("all-water test map");
        let mini_w = width.div_ceil(2);
        let mini_h = height.div_ceil(2);
        let mini_n = (mini_w * mini_h) as usize;
        let mini_map = GameMap::from_terrain_bytes(
            &MapMeta {
                width: mini_w,
                height: mini_h,
                num_land_tiles: 0,
            },
            &vec![0u8; mini_n],
        )
        .expect("all-water test mini map");
        let mut game = Game::default();
        game.map = map;
        game.mini_map = mini_map;
        game.bfs = crate::water::BfsScratch::new(n);
        game.water_astar = crate::water::WaterAstarScratch::new(n);
        game.mini_water_astar = crate::water::WaterAstarScratch::new(mini_n);
        game.mini_water_hpa = Some(crate::water_hpa::WaterHierarchical::new(
            &game.mini_map,
            true,
        ));
        game.water_component = crate::water::build_water_components(&game.map);
        game.end_spawn_phase();
        game
    }

    fn add_nation(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn stale_cached_path_replans_instead_of_skipping_ahead() {
        let mut game = water_game(40, 40);
        let from = game.ref_xy(8, 8);
        let dst = game.ref_xy(30, 8);
        let mut exec = TradeShipExecution::new_for_test(1, 0, 123);

        exec.path = vec![
            game.ref_xy(1, 1),
            game.ref_xy(2, 1),
            from,
            game.ref_xy(9, 8),
        ];
        exec.path_idx = 1;
        exec.path_dst = Some(dst);

        exec.next_path_tile(&mut game, from, dst)
            .expect("fresh path from current tile");

        assert_eq!(
            exec.path.first(),
            Some(&from),
            "PathFinderStepper invalidates on expected-position mismatch; it must not scan forward in the stale path"
        );
    }

    /// Warship capture mid-voyage: TradeShipExecution must detect the owner
    /// change, mark `was_captured`, and redirect toward the pirate's port.
    #[test]
    fn captured_trade_ship_redirects_to_pirate_port() {
        let mut game = water_game(40, 40);
        let pirate = add_nation(&mut game, "pirate");
        let merchant = add_nation(&mut game, "merchant");
        let customer = add_nation(&mut game, "customer");

        let pirate_port_tile = game.ref_xy(5, 5);
        let ship_tile = game.ref_xy(20, 20);
        let dst_tile = game.ref_xy(35, 35);

        let pirate_port = game.build_unit(pirate, unit_type::PORT, pirate_port_tile);
        let ship_id = game.build_unit(merchant, unit_type::TRADE_SHIP, ship_tile);
        let dst_port = game.build_unit(customer, unit_type::PORT, dst_tile);

        let mut exec = TradeShipExecution::new_for_test(merchant, dst_port, ship_id);
        assert_eq!(exec.destination_port_unit_id(), dst_port);

        game.capture_unit(merchant, pirate, ship_id);
        assert_eq!(game.find_unit_owner(ship_id), Some(pirate));

        exec.tick(&mut game, 1);
        assert!(exec.was_captured, "owner change must set was_captured");
        assert_eq!(
            exec.destination_port_unit_id(),
            pirate_port,
            "captured ship should redirect to pirate's nearest port"
        );
        assert!(exec.is_active());
    }

    /// Captured voyage gold goes to the pirate (current ship owner), not
    /// the original trade partners — mirrors TS `TradeShipExecution.complete`.
    #[test]
    fn captured_trade_ship_pays_gold_to_pirate_on_complete() {
        let mut game = water_game(20, 20);
        let pirate = add_nation(&mut game, "pirate");
        let merchant = add_nation(&mut game, "merchant");

        let pirate_port_tile = game.ref_xy(2, 2);
        let ship_tile = game.ref_xy(2, 2); // already at pirate port → complete on tick
        let _pirate_port = game.build_unit(pirate, unit_type::PORT, pirate_port_tile);
        let ship_id = game.build_unit(merchant, unit_type::TRADE_SHIP, ship_tile);

        let mut exec = TradeShipExecution::new_for_test(merchant, _pirate_port, ship_id);
        exec.tiles_traveled = 10;
        game.capture_unit(merchant, pirate, ship_id);

        let gold_before = game.player_by_small_id(pirate).unwrap().gold;
        exec.tick(&mut game, 1);
        assert!(!exec.is_active(), "voyage should complete at pirate port");
        let gold_after = game.player_by_small_id(pirate).unwrap().gold;
        assert!(
            gold_after > gold_before,
            "pirate should receive voyage gold ({gold_before} -> {gold_after})"
        );
        assert!(
            game.find_unit_owner(ship_id).is_none(),
            "ship removed on complete"
        );
    }

    /// TS keeps source/destination `Unit` object references in the voyage; if
    /// one endpoint is deleted before arrival, `unit.owner()` still returns
    /// the last owner for payout. Native must cache the last owner too.
    #[test]
    fn deleted_trade_ship_endpoints_still_receive_completion_gold() {
        let mut game = water_game(20, 20);
        let merchant = add_nation(&mut game, "merchant");
        let customer = add_nation(&mut game, "customer");

        let src_port = game.build_unit(merchant, unit_type::PORT, game.ref_xy(2, 2));
        let dst_port = game.build_unit(customer, unit_type::PORT, game.ref_xy(3, 3));
        let ship_id = game.build_unit(merchant, unit_type::TRADE_SHIP, game.ref_xy(4, 4));

        let mut exec = TradeShipExecution::new(merchant, src_port, dst_port, customer);
        exec.ship_unit_id = Some(ship_id);
        exec.tiles_traveled = 10;

        game.remove_unit(merchant, src_port);
        game.remove_unit(customer, dst_port);
        assert_eq!(game.find_unit_owner(src_port), None);
        assert_eq!(game.find_unit_owner(dst_port), None);

        let merchant_before = game.player_by_small_id(merchant).unwrap().gold;
        let customer_before = game.player_by_small_id(customer).unwrap().gold;
        let gold = game.wire.trade_ship_gold(exec.tiles_traveled as f64);
        exec.complete(&mut game);

        assert_eq!(
            game.player_by_small_id(merchant).unwrap().gold,
            merchant_before + gold
        );
        assert_eq!(
            game.player_by_small_id(customer).unwrap().gold,
            customer_before + gold
        );
        assert!(
            game.find_unit_owner(ship_id).is_none(),
            "ship removed on complete"
        );
    }
}
