//! Trade ship voyage (`TradeShipExecution.ts` subset).
//!
//! Piracy/capture (`WarshipExecution` stealing an in-transit trade ship) is not
//! modeled: native has no active Warship AI yet (see `nation_tick::track_ships_and_retaliate`
//! - "no-op until warships are ported"), so a trade ship's owner never changes here.
//! The port counterparts *can* change owner via ordinary land conquest, so both
//! ends are re-resolved by unit id every tick rather than frozen at spawn time.

use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;

pub struct TradeShipExecution {
    orig_owner_small_id: u16,
    src_port_unit_id: i32,
    dst_port_unit_id: i32,
    ship_unit_id: Option<i32>,
    path: Vec<TileRef>,
    path_idx: usize,
    path_dst: Option<TileRef>,
    tiles_traveled: u32,
    active: bool,
}

impl TradeShipExecution {
    pub fn new(orig_owner_small_id: u16, src_port_unit_id: i32, dst_port_unit_id: i32) -> Self {
        Self {
            orig_owner_small_id,
            src_port_unit_id,
            dst_port_unit_id,
            ship_unit_id: None,
            path: Vec::new(),
            path_idx: 0,
            path_dst: None,
            tiles_traveled: 0,
            active: true,
        }
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
        self.path_idx = if self.path.first() == Some(&from) { 1 } else { 0 };
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
                if let Some(pos) = self.path.iter().position(|&t| t == from) {
                    self.path_idx = pos + 1;
                } else {
                    self.path.clear();
                    self.path_idx = 0;
                    return self.next_path_tile(game, from, to);
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

    /// TS `TradeShipExecution.complete` (not-captured branch: both ends get paid).
    fn complete(&mut self, game: &mut Game) {
        self.active = false;
        let Some(uid) = self.ship_unit_id else { return };
        game.remove_unit(self.orig_owner_small_id, uid);
        let gold = game.wire.trade_ship_gold(self.tiles_traveled as f64);
        if let Some(src_owner) = game.find_unit_owner(self.src_port_unit_id) {
            game.add_gold(src_owner, gold);
        }
        if let Some(dst_owner) = game.find_unit_owner(self.dst_port_unit_id) {
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
            let has_port = game.player_by_small_id(self.orig_owner_small_id).is_some_and(|p| {
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

        let Some(cur_tile) = game.unit_tile_of(self.orig_owner_small_id, uid) else {
            self.active = false;
            return;
        };

        // TS: `!this._dstPort.isActive()` (port destroyed/demolished since spawn).
        let Some(dst_owner) = game.find_unit_owner(self.dst_port_unit_id) else {
            game.remove_unit(self.orig_owner_small_id, uid);
            self.active = false;
            return;
        };
        let Some(dst_tile) = game.unit_tile_of(dst_owner, self.dst_port_unit_id) else {
            game.remove_unit(self.orig_owner_small_id, uid);
            self.active = false;
            return;
        };

        // TS: `dstPortOwner.id() === srcPort.owner().id()` - the src port (possibly
        // captured via land conquest since spawn) now belongs to the dst owner too.
        if game.find_unit_owner(self.src_port_unit_id) == Some(dst_owner) {
            game.remove_unit(self.orig_owner_small_id, uid);
            self.active = false;
            return;
        }

        if !game.can_trade(self.orig_owner_small_id, dst_owner) {
            game.remove_unit(self.orig_owner_small_id, uid);
            self.active = false;
            return;
        }

        if cur_tile == dst_tile {
            self.complete(game);
            return;
        }

        let Some(next) = self.next_path_tile(game, cur_tile, dst_tile) else {
            self.complete(game);
            return;
        };
        game.move_unit(self.orig_owner_small_id, uid, next);
        self.tiles_traveled += 1;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
