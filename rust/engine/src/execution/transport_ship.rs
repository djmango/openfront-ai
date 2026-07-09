//! Transport ship naval invasion (TS `TransportShipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::TRANSPORT;
use crate::game::Game;
use crate::map::TileRef;
use crate::spatial::{can_build_transport_ship, target_transport_tile};

pub struct TransportShipExecution {
    owner_small_id: u16,
    ref_tile: TileRef,
    troops: Option<f64>,
    dst: Option<TileRef>,
    path: Vec<TileRef>,
    path_idx: usize,
    path_dst: Option<TileRef>,
    unit_id: Option<i32>,
    active: bool,
    initialized: bool,
    last_move_tick: u32,
    // TS `TransportShipExecution.target` is snapshotted once in `init()` from
    // `mg.owner(this.ref)` and reused verbatim when the boat lands, even though
    // the actual landing tile (`dst`, chosen by `targetTransportTile` as the
    // closest shore tile of that same owner) can change hands by the time the
    // multi-tick voyage completes. Mirror that by freezing the owner of
    // `ref_tile` here instead of re-reading `dst`'s owner at landing time.
    target_small_id: Option<u16>,
}

impl TransportShipExecution {
    pub fn new(owner_small_id: u16, ref_tile: TileRef, troops: f64) -> Self {
        Self {
            owner_small_id,
            ref_tile,
            troops: Some(troops),
            dst: None,
            path: Vec::with_capacity(64),
            path_idx: 0,
            path_dst: None,
            unit_id: None,
            active: true,
            initialized: false,
            last_move_tick: 0,
            target_small_id: None,
        }
    }

    fn unit_tile(&self, game: &Game) -> Option<TileRef> {
        let uid = self.unit_id?;
        let p = game.player_by_small_id(self.owner_small_id)?;
        let u = p.units.iter().find(|u| u.id == uid)?;
        Some(u.tile as TileRef)
    }

    fn refresh_path(&mut self, game: &mut Game, from: TileRef, to: TileRef) -> bool {
        if !game.plan_water_path(from, to) {
            return false;
        }
        self.path.clear();
        self.path.extend_from_slice(game.planned_water_path());
        // TS `TransportShipExecution.init`: path always starts at `src`.
        if self.path.is_empty() || self.path.first() != Some(&from) {
            self.path.insert(0, from);
        }
        // TS PathFinderStepper keeps duplicate consecutive nodes; one step per tick.
        self.path_idx = 0;
        if self.path.first() == Some(&from) {
            self.path_idx = 1;
        }
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
}

impl Execution for TransportShipExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        if !self.active || self.initialized {
            return;
        }
        self.initialized = true;
        self.last_move_tick = tick;

        if game.wire.is_unit_disabled(TRANSPORT) {
            self.active = false;
            return;
        }
        if game.unit_count(self.owner_small_id, TRANSPORT) >= game.wire.boat_max_number() {
            self.active = false;
            return;
        }

        let ref_owner = game.map.owner_id(self.ref_tile);
        if ref_owner == self.owner_small_id {
            self.active = false;
            return;
        }
        if ref_owner != 0 && ref_owner != game.terra_nullius_id() {
            if game.is_friendly(self.owner_small_id, ref_owner)
                || !game.can_attack_player(self.owner_small_id, ref_owner)
            {
                self.active = false;
                return;
            }
        }
        self.target_small_id = Some(ref_owner);

        let Some(dst) = target_transport_tile(game, self.ref_tile) else {
            self.active = false;
            return;
        };
        self.dst = Some(dst);

        // TS `attacker.canBuild(TransportShip, this.dst)`  -  re-targets from dst tile.
        let Some(src) = can_build_transport_ship(game, self.owner_small_id, dst) else {
            self.active = false;
            return;
        };

        let troops = self.troops.unwrap_or_else(|| {
            game.player_by_small_id(self.owner_small_id)
                .map(|p| game.wire.boat_attack_amount(p.troops))
                .unwrap_or(0.0)
        });
        let troops = troops.min(
            game.player_by_small_id(self.owner_small_id)
                .map(|p| p.troops as f64)
                .unwrap_or(0.0),
        );
        if troops < 1.0 {
            self.active = false;
            return;
        }
        game.remove_troops(self.owner_small_id, troops);
        self.troops = Some(troops);

        let uid = game.build_unit(self.owner_small_id, TRANSPORT, src);
        self.unit_id = Some(uid);

        // TS records a motion plan but does not move the unit in `init`.
        if !self.refresh_path(game, src, dst) {
            game.add_troops(self.owner_small_id, troops);
            game.remove_unit(self.owner_small_id, uid);
            self.active = false;
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || !self.initialized {
            return;
        }
        if tick.saturating_sub(self.last_move_tick) < 1 {
            return;
        }
        self.last_move_tick = tick;

        let Some(dst) = self.dst else {
            self.active = false;
            return;
        };
        let Some(uid) = self.unit_id else {
            self.active = false;
            return;
        };
        let Some(from) = self.unit_tile(game) else {
            self.active = false;
            return;
        };
        let troops = self.troops.unwrap_or(0.0);

        if from == dst {
            self.land(game, dst, uid, troops);
            return;
        }

        let Some(next) = self.next_path_tile(game, from, dst) else {
            self.land(game, dst, uid, troops);
            return;
        };
        game.move_unit(self.owner_small_id, uid, next);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

impl TransportShipExecution {
    fn land(&mut self, game: &mut Game, dst: TileRef, uid: i32, troops: f64) {
        // TS `TransportShipExecution.tick`: the "already own it" check compares
        // against the *current* owner of `dst`, but the subsequent attack target
        // is the `target` snapshotted in `init()` (owner of `ref_tile`), not
        // `dst`'s live owner.
        let live_dst_owner = game.map.owner_id(dst);
        if live_dst_owner == self.owner_small_id {
            game.add_troops(self.owner_small_id, troops * 0.75);
        } else {
            game.conquer(self.owner_small_id, dst);
            let target_owner = self.target_small_id.unwrap_or(live_dst_owner);
            if target_owner == 0 || target_owner == game.terra_nullius_id() {
                game.add_land_attack_from(self.owner_small_id, None, Some(troops), Some(dst));
            } else if let Some(def) = game.player_by_small_id(target_owner) {
                let target_id = def.id.clone();
                game.add_land_attack_from(
                    self.owner_small_id,
                    Some(target_id),
                    Some(troops),
                    Some(dst),
                );
            }
        }
        game.remove_unit(self.owner_small_id, uid);
        self.active = false;
    }
}
