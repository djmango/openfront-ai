//! Win detection (TS `WinCheckExecution.ts`).

use super::Execution;
use crate::core::team_assignment::BOT_TEAM;
use crate::game::Game;

/// TS `WinCheckExecution.HARD_TIME_LIMIT_SECONDS` (170 min).
const HARD_TIME_LIMIT_SECONDS: u32 = 170 * 60;
/// TS `Config.percentageTilesOwnedToWin()`.
const PERCENTAGE_TILES_TO_WIN_FFA: f64 = 80.0;
const PERCENTAGE_TILES_TO_WIN_TEAM: f64 = 95.0;

pub struct WinCheckExecution {
    active: bool,
}

impl WinCheckExecution {
    pub fn new() -> Self {
        Self { active: true }
    }

    fn time_limit_reached(game: &Game) -> bool {
        let elapsed_seconds = game.ticks_since_start() / 10;
        game.wire
            .game_config()
            .max_timer_value
            .is_some_and(|minutes| elapsed_seconds >= minutes.saturating_mul(60))
            || elapsed_seconds >= HARD_TIME_LIMIT_SECONDS
    }

    fn percentage_owned(game: &Game, tiles: i32) -> f64 {
        let num_tiles_without_fallout = game
            .num_land_tiles()
            .saturating_sub(game.num_tiles_with_fallout());
        (tiles as f64 / num_tiles_without_fallout as f64) * 100.0
    }

    fn check_winner_ffa(&mut self, game: &mut Game) {
        // TS takes `sorted[0]` of a stable descending sort: the first player
        // in `players()` insertion order wins a tile-count tie.
        let mut leader: Option<(&str, i32)> = None;
        for player in game.players_alive() {
            if leader.is_none() || player.tiles_owned > leader.unwrap().1 {
                leader = Some((&player.id, player.tiles_owned));
            }
        }
        let Some((leader_id, leader_tiles)) = leader else {
            return;
        };

        if game.wire.game_config().ranked_type.as_deref() == Some("1v1") {
            let mut connected_human = None;
            let mut connected_humans = 0;
            for player in game.players_alive() {
                if player.player_type == crate::game::PlayerType::Human && !player.is_disconnected {
                    connected_humans += 1;
                    connected_human = Some(player.id.clone());
                    if connected_humans > 1 {
                        break;
                    }
                }
            }
            if connected_humans == 1 {
                game.winner = connected_human;
                self.active = false;
                return;
            }
        }

        let leader_id = leader_id.to_string();
        if Self::percentage_owned(game, leader_tiles) > PERCENTAGE_TILES_TO_WIN_FFA
            || Self::time_limit_reached(game)
        {
            game.winner = Some(leader_id);
            self.active = false;
        }
    }

    fn check_winner_team(&mut self, game: &mut Game) {
        // A Vec preserves the insertion order of TS's Map<Team, number>, which
        // is also the stable-sort tie break for equally sized teams.
        let mut team_tiles: Vec<(String, i32)> = Vec::new();
        for player in game.players_alive() {
            let Some(team) = player.team.as_ref() else {
                continue;
            };
            if let Some((_, tiles)) = team_tiles.iter_mut().find(|(name, _)| name == team) {
                *tiles += player.tiles_owned;
            } else {
                team_tiles.push((team.clone(), player.tiles_owned));
            }
        }

        let mut leader: Option<(&str, i32)> = None;
        for (team, tiles) in &team_tiles {
            if leader.is_none() || *tiles > leader.unwrap().1 {
                leader = Some((team, *tiles));
            }
        }
        let Some((leader_team, leader_tiles)) = leader else {
            return;
        };
        if Self::percentage_owned(game, leader_tiles) > PERCENTAGE_TILES_TO_WIN_TEAM
            || Self::time_limit_reached(game)
        {
            // TS refuses to finish a team game while the synthetic bot team leads.
            if leader_team == BOT_TEAM {
                return;
            }
            game.winner = Some(leader_team.to_string());
            self.active = false;
        }
    }
}

impl Execution for WinCheckExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, tick: u32) {
        // TS only checks every 10th global tick.
        if tick % 10 != 0 {
            return;
        }
        if game.wire.game_config().game_mode == "Free For All" {
            self.check_winner_ffa(game);
        } else {
            self.check_winner_team(game);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;
    use crate::game::{Player, PlayerType};
    use serde_json::json;

    fn configure(game: &mut Game, mode: &str, extra: serde_json::Value) {
        let mut value = json!({
            "gameMap": "Onion",
            "difficulty": "Medium",
            "donateGold": false,
            "donateTroops": false,
            "gameType": "Public",
            "gameMode": mode,
            "gameMapSize": "Normal",
            "nations": "disabled",
            "bots": 0,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": false
        });
        for (key, item) in extra.as_object().unwrap() {
            value[key] = item.clone();
        }
        game.wire = Config::from_value(&value, true).unwrap();
    }

    fn player(id: &str, tiles: i32) -> Player {
        Player {
            id: id.to_string(),
            name: id.to_string(),
            client_id: format!("{id}-client"),
            tiles_owned: tiles,
            ..Default::default()
        }
    }

    #[test]
    fn ffa_tie_uses_first_player_in_ts_order() {
        let mut game = Game::default();
        game.add_player(player("first", 1));
        game.add_player(player("second", 1));
        let mut execution = WinCheckExecution::new();

        execution.tick(&mut game, 10);

        assert_eq!(game.winner.as_deref(), Some("first"));
        assert!(!execution.is_active());
    }

    #[test]
    fn max_timer_forces_the_current_ffa_leader() {
        let mut game = Game::default();
        configure(&mut game, "Free For All", json!({ "maxTimerValue": 0 }));
        game.add_player(player("leader", 0));
        game.add_player(player("other", 0));
        let mut execution = WinCheckExecution::new();

        execution.tick(&mut game, 10);

        assert_eq!(game.winner.as_deref(), Some("leader"));
    }

    #[test]
    fn one_v_one_awards_the_only_connected_human() {
        let mut game = Game::default();
        configure(&mut game, "Free For All", json!({ "rankedType": "1v1" }));
        let mut disconnected = player("disconnected", 10);
        disconnected.is_disconnected = true;
        game.add_player(disconnected);
        game.add_player(player("connected", 0));
        let mut execution = WinCheckExecution::new();

        execution.tick(&mut game, 10);

        assert_eq!(game.winner.as_deref(), Some("connected"));
    }

    #[test]
    fn team_mode_aggregates_players_and_preserves_team_order() {
        let mut game = Game::default();
        configure(&mut game, "Team", json!({}));
        let mut red_one = player("red-one", 2);
        red_one.team = Some("Red".to_string());
        let mut blue = player("blue", 4);
        blue.team = Some("Blue".to_string());
        let mut red_two = player("red-two", 2);
        red_two.team = Some("Red".to_string());
        game.add_player(red_one);
        game.add_player(blue);
        game.add_player(red_two);
        let mut execution = WinCheckExecution::new();

        execution.tick(&mut game, 10);

        assert_eq!(game.winner.as_deref(), Some("Red"));
    }

    #[test]
    fn bot_team_cannot_win() {
        let mut game = Game::default();
        configure(&mut game, "Team", json!({}));
        let mut bot = player("bot", 1);
        bot.player_type = PlayerType::Bot;
        bot.team = Some(BOT_TEAM.to_string());
        game.add_player(bot);
        let mut execution = WinCheckExecution::new();

        execution.tick(&mut game, 10);

        assert_eq!(game.winner, None);
        assert!(execution.is_active());
    }
}
