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
    was_captured: bool,
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
    pub(crate) fn new_for_test(orig_owner_small_id: u16, dst_port_unit_id: i32, ship_unit_id: i32) -> Self {
        let mut exec = Self::new(orig_owner_small_id, 0, dst_port_unit_id);
        exec.ship_unit_id = Some(ship_unit_id);
        exec
    }

    pub fn destination_port_unit_id(&self) -> i32 {
        self.dst_port_unit_id
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

        // TS: `!this._dstPort.isActive()` (port destroyed/demolished since spawn).
        let mut dst_owner = game.find_unit_owner(self.dst_port_unit_id);

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
