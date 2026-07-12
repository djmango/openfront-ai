//! Tribe bot expansion (TS `TribeExecution.ts`).

use crate::execution::ai_attack::{send_tn_attack, tribe_maybe_attack};
use crate::execution::alliance_exec::AllianceExtensionExecution;
use crate::execution::exec_enum::ExecEnum;
use crate::execution::{DeleteUnitExecution, Execution};
use crate::game::Game;
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

/// TS `Structures` unit-type group (`Game.ts`).
const STRUCTURE_TYPES: &[&str] = &[
    "City",
    "Defense Post",
    "SAM Launcher",
    "Missile Silo",
    "Port",
    "Factory",
];

pub struct TribeExecution {
    small_id: u16,
    player_id: String,
    random: PseudoRandom,
    attack_rate: i32,
    attack_tick: i32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    attack_behavior_init: bool,
    neighbors_terra_nullius: bool,
    active: bool,
}

impl TribeExecution {
    pub fn new(small_id: u16, player_id: String) -> Self {
        let mut random = PseudoRandom::new(simple_hash(&player_id));
        let attack_rate = random.next_int(40, 80);
        let attack_tick = random.next_int(0, attack_rate);
        let trigger_ratio = random.next_int(50, 60) as f64 / 100.0;
        let reserve_ratio = random.next_int(30, 40) as f64 / 100.0;
        let expand_ratio = random.next_int(10, 20) as f64 / 100.0;
        Self {
            small_id,
            player_id,
            random,
            attack_rate,
            attack_tick,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            attack_behavior_init: false,
            neighbors_terra_nullius: true,
            active: true,
        }
    }
    pub fn small_id(&self) -> u16 {
        self.small_id
    }

    /// TS `TribeExecution.acceptAllAllianceRequests`.
    pub(crate) fn accept_all_alliance_requests(&self, game: &mut Game, tick: u32) {
        let incoming: Vec<u16> = game
            .incoming_alliance_requests(self.small_id)
            .into_iter()
            .map(|r| r.requestor_small_id)
            .collect();
        for requestor in incoming {
            game.accept_alliance_request(requestor, self.small_id, tick);
        }

        // Accept all alliance extension requests (TS: onlyOneAgreedToExtend
        // means the other side already clicked renew - tribe always agrees).
        let extensions = game.alliance_extension_candidates(self.small_id);
        for ext in extensions {
            let Some(other) = game.player_by_small_id(ext.other_small_id) else {
                continue;
            };
            let other_id = other.id.clone();
            game.add_execution(ExecEnum::AllianceExtension(
                AllianceExtensionExecution::new(self.small_id, other_id),
            ));
        }
    }

    /// TS `TribeExecution.deleteNextStructure`: delete one structure per
    /// attack tick when the cooldown allows. Changes structure density →
    /// `attackBots` prioritization and random-boat gating.
    pub(crate) fn delete_next_structure(&self, game: &mut Game) {
        if !game.can_delete_unit(self.small_id) {
            return;
        }
        let unit_id = game.player_by_small_id(self.small_id).and_then(|p| {
            p.units
                .iter()
                .find(|u| {
                    STRUCTURE_TYPES.contains(&u.unit_type.as_str()) && u.deletion_at.is_none()
                })
                .map(|u| u.id)
        });
        let Some(unit_id) = unit_id else {
            return;
        };
        game.add_execution(ExecEnum::DeleteUnit(DeleteUnitExecution::new(
            self.small_id,
            unit_id,
        )));
    }
}

impl Execution for TribeExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || game.in_spawn_phase() {
            return;
        }
        if (tick as i32) % self.attack_rate != self.attack_tick {
            return;
        }

        if game
            .player_by_small_id(self.small_id)
            .is_none_or(|p| !p.alive || p.spawn_tile.is_none())
        {
            self.active = false;
            return;
        }

        if !self.attack_behavior_init {
            self.attack_behavior_init = true;
            send_tn_attack(game, self.small_id, self.expand_ratio);
            return;
        }

        self.accept_all_alliance_requests(game, tick);
        self.delete_next_structure(game);

        tribe_maybe_attack(
            game,
            &mut self.random,
            self.small_id,
            self.trigger_ratio,
            self.reserve_ratio,
            self.expand_ratio,
            &mut self.neighbors_terra_nullius,
        );
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::game::{Game, PlayerInfo, PlayerType};
    use crate::test_util::plains_game;

    fn add_bot(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Bot,
            client_id: None,
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

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

    #[test]
    fn delete_next_structure_marks_one_city_for_deletion() {
        let mut game = plains_game(20, 20);
        let tribe = add_bot(&mut game, "tribe-a");
        let tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(tile, tribe);
        if let Some(p) = game.player_by_small_id_mut(tribe) {
            p.spawn_tile = Some(tile);
            p.alive = true;
        }
        let city = game.build_unit(tribe, unit_type::CITY, tile);
        for _ in 0..Game::DELETE_UNIT_COOLDOWN_TICKS + 1 {
            game.execute_next_tick();
        }
        assert!(game.can_delete_unit(tribe));

        let mut exec = TribeExecution::new(tribe, "tribe-a".into());
        exec.attack_behavior_init = true;
        exec.attack_rate = 1;
        exec.attack_tick = 0;

        let tick = game.ticks();
        exec.tick(&mut game, tick);
        // DeleteUnitExecution lives in `uninit` until the next tick flushes it.
        game.execute_next_tick();
        assert!(
            game.is_unit_marked_for_deletion(tribe, city),
            "tribe should have queued DeleteUnitExecution for its city"
        );
    }

    #[test]
    fn alliance_extension_accepts_when_other_side_already_agreed() {
        let mut game = plains_game(20, 20);
        let tribe = add_bot(&mut game, "tribe-a");
        let human = add_human(&mut game, "human-a");
        for sid in [tribe, human] {
            let tile = game.map.ref_xy(if sid == tribe { 0 } else { 5 }, 0);
            game.map.set_owner_id(tile, sid);
            if let Some(p) = game.player_by_small_id_mut(sid) {
                p.spawn_tile = Some(tile);
                p.alive = true;
            }
        }
        // Form alliance: mutual requests auto-accept in this codebase.
        let now = game.ticks();
        assert!(game.create_alliance_request(human, tribe, now));
        assert!(game.create_alliance_request(tribe, human, now));
        assert!(game.is_allied_with(tribe, human));

        // Advance so a renewal (expires_at = current_tick + duration) is
        // strictly later than the original formation expiry.
        for _ in 0..10 {
            game.execute_next_tick();
        }

        // Human alone requests extension → onlyOneAgreedToExtend.
        game.add_alliance_extension_request(human, tribe);
        let candidates = game.alliance_extension_candidates(tribe);
        assert_eq!(candidates.len(), 1);

        let expires_before = game
            .player_alliances(tribe)
            .first()
            .map(|al| al.expires_at)
            .expect("alliance");

        let mut exec = TribeExecution::new(tribe, "tribe-a".into());
        let tick = game.ticks();
        // Call the alliance step directly — a full `tick()` also runs
        // `tribe_maybe_attack`, which can break the alliance under test.
        exec.accept_all_alliance_requests(&mut game, tick);
        game.execute_next_tick();

        let alliance = game
            .player_alliances(tribe)
            .first()
            .cloned()
            .cloned()
            .expect("alliance still exists");
        // Both sides agreed → duration renewed, flags cleared.
        assert!(alliance.expires_at > expires_before);
        assert!(!alliance.extension_requested_requestor);
        assert!(!alliance.extension_requested_recipient);
    }
}
