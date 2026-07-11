//! Transport ship naval invasion (TS `TransportShipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::TRANSPORT;
use crate::game::Game;
use crate::map::TileRef;
use crate::spatial::{can_build_transport_ship, closest_shore_by_water, target_transport_tile};

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
    retreating: bool,
    retreat_destination_resolved: bool,
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
            retreating: false,
            retreat_destination_resolved: false,
            target_small_id: None,
        }
    }

    /// Unit id of the in-flight boat (None until init spawns it).
    pub fn unit_id(&self) -> Option<i32> {
        self.unit_id
    }

    pub fn owner_small_id(&self) -> u16 {
        self.owner_small_id
    }

    /// Carried troops (TS `unit.troops()` for transports lives on the unit;
    /// natively it lives here, read by the RL obs units list).
    pub fn carried_troops(&self) -> f64 {
        self.troops.unwrap_or(0.0)
    }

    /// TS `Unit.setTroops()` for a `TransportShip` - used by `NukeExecution::detonate`
    /// to apply blast casualties to an in-flight boat's carried troops.
    pub fn set_carried_troops(&mut self, troops: f64) {
        self.troops = Some(troops.max(0.0));
    }

    pub fn request_retreat(&mut self) {
        self.retreating = true;
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
        // TS `TransportShipExecution.init`: rejects any alliance request the
        // landing tile's owner has outstanding toward the sender - e.g. sending a
        // boat at someone cancels a pending alliance request from them, exactly
        // like attacking them does in `AttackExecution.init`. Native previously
        // never called this, so a boat sent at a would-be ally left their
        // pending alliance request live even as troops were en route to invade them.
        if ref_owner != 0 && ref_owner != game.terra_nullius_id() {
            let owner_is_bot = game
                .player_by_small_id(self.owner_small_id)
                .map_or(false, |p| p.player_type == crate::game::PlayerType::Bot);
            let target_is_bot = game
                .player_by_small_id(ref_owner)
                .map_or(false, |p| p.player_type == crate::game::PlayerType::Bot);
            if !owner_is_bot && !target_is_bot {
                game.reject_alliance_request(ref_owner, self.owner_small_id);
            }
        }
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

        let Some(mut dst) = self.dst else {
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

        if self.retreating && !self.retreat_destination_resolved {
            let Some(retreat_dst) = closest_shore_by_water(game, self.owner_small_id, from) else {
                game.add_troops(self.owner_small_id, troops);
                game.remove_unit(self.owner_small_id, uid);
                self.active = false;
                return;
            };
            self.dst = Some(retreat_dst);
            dst = retreat_dst;
            self.retreat_destination_resolved = true;
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{PlayerInfo, PlayerType};

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

    // Ported from the intent of Attack.test.ts's "Should cancel alliance
    // requests if the recipient sends a transport ship": sending a boat at a
    // player who has a pending alliance request outstanding toward the sender
    // rejects that request, mirroring AttackExecution.init's identical rule for
    // land attacks (see `openfront/src/core/execution/TransportShipExecution.ts`
    // `rejectIncomingAllianceRequests`, called from `init`). `Game::default()`'s
    // 1x1 map is repurposed here (rather than a literal `setup()`+fixture-map
    // port) by directly setting that single tile's owner, since the bug is in
    // `init()`'s alliance bookkeeping, not in boat pathfinding/geometry.
    #[test]
    fn sending_a_transport_ship_rejects_a_pending_alliance_request_from_the_target() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let sender = add_human(&mut game, "sender");
        let target = add_human(&mut game, "target");
        if let Some(p) = game.player_by_small_id_mut(sender) {
            p.troops = 1_000;
        }

        let only_tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(only_tile, target);

        game.create_alliance_request(target, sender, 0);
        assert!(game
            .alliance_requests
            .iter()
            .any(|r| r.requestor_small_id == target
                && r.recipient_small_id == sender
                && r.status == crate::game::AllianceRequestStatus::Pending));

        let mut ship = TransportShipExecution::new(sender, only_tile, 100.0);
        ship.init(&mut game, 1);

        assert!(game.alliance_requests.iter().any(|r| r.requestor_small_id
            == target
            && r.recipient_small_id == sender
            && r.status == crate::game::AllianceRequestStatus::Rejected));
    }
}
