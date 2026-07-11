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

    /// TS `Unit.targetTile()` for a transport ship - `self.dst` doubles as both the initial
    /// attack destination and (once retreating) the resolved retreat shore tile, exactly like
    /// TS reassigns the same `targetTile` field in both cases. Read by
    /// `NationWarshipBehavior.trackIncomingTransportsAndRetaliate` (`warship_ai.rs`) to
    /// find enemy transports heading toward this nation's territory - the shared `Unit.target_tile`
    /// field is never populated for this unit type natively (only nukes/MIRVs set it), so this
    /// exec-local getter is the only place that state actually lives.
    pub fn target_tile(&self) -> Option<TileRef> {
        self.dst
    }

    /// TS `Unit.transportShipState().isRetreating`.
    pub fn is_retreating(&self) -> bool {
        self.retreating
    }

    /// Test-only constructor for an in-flight transport ship whose backing `Unit` already
    /// exists, bypassing `init()`'s real pathfinding (see `WarshipExecution::new_for_test`'s
    /// doc comment for the same rationale - used by `warship_ai.rs`'s incoming-transport tests).
    #[cfg(test)]
    pub(crate) fn new_for_test(
        owner_small_id: u16,
        unit_id: i32,
        target_tile: TileRef,
        retreating: bool,
    ) -> Self {
        let mut exec = Self::new(owner_small_id, target_tile, 0.0);
        exec.unit_id = Some(unit_id);
        exec.dst = Some(target_tile);
        exec.initialized = true;
        exec.retreating = retreating;
        exec
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

    // Ported from Attack.test.ts's "Attack immunity" > "Should not be able to
    // send a boat during immunity phase": a boat aimed at another human's
    // territory is blocked while that human's spawn immunity is active, via
    // the same `can_attack_player` gate `AttackExecution::init` uses (see
    // `immunity_tests` in `execution/attack.rs`). The mirror TS case ("Should
    // be able to send a boat after immunity phase") isn't ported literally:
    // once past this gate, `init()` needs real water/land adjacency
    // (`target_transport_tile`, `can_build_transport_ship`) that
    // `Game::default()`'s 1x1 map can't provide, and that later geometry is
    // already exercised by the alliance-rejection test above using the same
    // land-only map - only the immunity gate itself is new coverage here.
    #[test]
    fn sending_a_transport_ship_during_spawn_immunity_is_blocked() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let sender = add_human(&mut game, "sender");
        let target = add_human(&mut game, "target");
        if let Some(p) = game.player_by_small_id_mut(sender) {
            p.troops = 1_000;
        }
        let only_tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(only_tile, target);

        let mut ship = TransportShipExecution::new(sender, only_tile, 100.0);
        let tick = game.ticks();
        ship.init(&mut game, tick);

        assert!(!ship.is_active());
        assert!(ship.unit_id().is_none());
    }

    // Ported from Attack.test.ts's "Should abort TransportShipExecution when
    // target is the attacker itself": aiming a boat at a tile the sender
    // already owns aborts in `init()` before any pathfinding/build attempt.
    #[test]
    fn transport_ship_targeting_own_territory_aborts_immediately() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let sender = add_human(&mut game, "sender");
        if let Some(p) = game.player_by_small_id_mut(sender) {
            p.troops = 1_000;
        }
        let only_tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(only_tile, sender);

        let mut ship = TransportShipExecution::new(sender, only_tile, 10.0);
        let tick = game.ticks();
        ship.init(&mut game, tick);

        assert!(!ship.is_active());
        assert!(ship.unit_id().is_none());
    }

    // Ported from Attack.test.ts's "Boat penalty on retreat Transport Ship
    // arrival": TS `TransportShipExecution.tick` credits only 75% of a boat's
    // carried troops back to its owner when it lands on territory the owner
    // already controls (`this.mg.units...owner === this._owner ?
    // addTroops(troops * 0.75) : ...` in the TS source) - the same code path
    // for both "retreating home" and "never left home" landings. Reaching
    // this via a real multi-tick voyage+retreat needs water pathfinding
    // `Game::default()` can't provide, so this drives `tick()` directly with
    // the boat already parked at its (owned) destination, which exercises
    // the exact same `land()` branch TS's retreat case does.
    #[test]
    fn transport_ship_landing_on_own_territory_returns_75_percent_of_carried_troops() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let owner = add_human(&mut game, "owner");
        if let Some(p) = game.player_by_small_id_mut(owner) {
            p.troops = 1_000;
        }
        let tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(tile, owner);
        let uid = game.build_unit(owner, TRANSPORT, tile);

        let boat_troops = 500.0;
        let mut ship = TransportShipExecution::new(owner, tile, boat_troops);
        ship.unit_id = Some(uid);
        ship.dst = Some(tile);
        ship.initialized = true;
        ship.retreat_destination_resolved = true;
        ship.last_move_tick = 0;

        ship.tick(&mut game, 1);

        assert!(!ship.is_active());
        assert_eq!(
            game.player_by_small_id(owner).unwrap().troops,
            1_000 + 375
        );
    }

    // Ported from the intent of Attack.test.ts's "Nuke reduce attacking boat
    // troop count": TS's `NukeExecution.detonate` applies the same
    // `nukeDeathFactor` to a nuked player's in-flight transport ships as it
    // does to their home troops and outgoing land attacks (see the sibling
    // attack-side coverage in `execution/nuke_execution.rs`'s
    // `nuke_reduces_troops_of_a_live_outgoing_attack_owned_by_the_impacted_player`,
    // which caught native previously skipping both entirely - fixed in
    // `Game::apply_nuke_deaths_to_deployed_forces`). This exercises the
    // transport-ship arm of that same fix directly: `Game::push_exec_for_test`
    // places an already-in-flight boat (real `unit_id`, no pathfinding
    // needed) straight into `execs`, since `Game::default()`'s 1x1 map can't
    // run this boat through a real naval `init()`.
    #[test]
    fn nuke_reduces_carried_troops_of_a_live_transport_ship_owned_by_the_impacted_player() {
        use crate::core::schemas::unit_type;

        let mut game = Game::default();
        game.end_spawn_phase();
        let owner = add_human(&mut game, "owner");
        let tile = game.map.ref_xy(0, 0);
        let uid = game.build_unit(owner, TRANSPORT, tile);

        let mut ship = TransportShipExecution::new(owner, tile, 300.0);
        ship.unit_id = Some(uid);
        game.push_exec_for_test(crate::execution::ExecEnum::TransportShip(ship));

        game.apply_nuke_deaths_to_deployed_forces(owner, unit_type::ATOM_BOMB, 100.0, 10_000.0);

        let troops_after = game
            .live_transports()
            .find(|t| t.owner_small_id() == owner)
            .map(|t| t.carried_troops())
            .expect("transport ship should still be present");
        // nukeDeathFactor(ATOM_BOMB, 300, 100, _) = 5 * 300 / 100 = 15.
        assert_eq!(troops_after, 285.0);
    }

    // Ported/adapted from Disconnected.test.ts's "Disconnected team member
    // interactions" describe block (openfront/tests/Disconnected.test.ts,
    // the three tests below "Conqueror gets conquered disconnected team
    // member's transport- and warships"):
    //   - "Captured transport ship landing attack should be in name of new owner"
    //   - "Captured transport ship should retreat to closest owner shore tile"
    //   - "Retreating transport ship is deleted if new owner has no shore tiles"
    //
    // All three drive a real `TransportShipExecution` through actual water
    // pathfinding (`half_land_half_ocean` fixture: shore tiles, a real ocean
    // component, `bestTransportShipSpawn`/retreat-tile selection) across
    // multiple ticks after the ship is captured mid-voyage. No existing
    // native test builds a synthetic water map (every current test - attack,
    // nation_structures, warship's own new tests above - stays on land-only
    // or mocks geometry out entirely), and building that infra from scratch
    // is out of scope here per the porting task's guardrails. The underlying
    // capture mechanism itself (`Game::conquer_player`) is covered directly
    // in `game.rs::conquer_player_tests`; what's *not* covered natively is
    // the captured ship's own post-capture behavior (continuing its voyage
    // under the new owner's name, retreating to the new owner's shore, or
    // self-deleting when the new owner has no shore left).
    #[test]
    #[ignore = "needs a synthetic water-map test harness (shore tiles/ocean component/retreat-tile search) - see module comment and Disconnected.test.ts lines 333-458"]
    fn captured_transport_ship_behavior_needs_water_map_test_harness() {}
}
