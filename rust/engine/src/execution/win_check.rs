//! Win detection (TS `WinCheckExecution.ts`, FFA path).

use super::Execution;
use crate::game::Game;

/// TS `WinCheckExecution.HARD_TIME_LIMIT_SECONDS` (170 min).
const HARD_TIME_LIMIT_SECONDS: u32 = 170 * 60;
/// TS `Config.percentageTilesOwnedToWin()` for FFA.
const PERCENTAGE_TILES_TO_WIN_FFA: f64 = 80.0;

pub struct WinCheckExecution {
    active: bool,
}

impl WinCheckExecution {
    pub fn new() -> Self {
        Self { active: true }
    }
}

impl Execution for WinCheckExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, tick: u32) {
        // TS: only every 10th tick. (Team mode's 95% path is not ported; RL
        // and the record corpus in use are FFA.)
        if tick % 10 != 0 {
            return;
        }
        // TS takes `sorted[0]` of a stable descending sort - i.e. the FIRST
        // player (in players() order) with the maximum tile count.
        let mut max: Option<(&str, i32)> = None;
        for p in game.players_alive() {
            if max.is_none() || p.tiles_owned > max.unwrap().1 {
                max = Some((&p.id, p.tiles_owned));
            }
        }
        let Some((max_id, max_tiles)) = max else {
            return;
        };
        let max_id = max_id.to_string();
        let num_tiles_without_fallout =
            (game.num_land_tiles().saturating_sub(game.num_tiles_with_fallout())).max(1);
        let elapsed_seconds = game.ticks_since_start() / 10;
        if (max_tiles as f64 / num_tiles_without_fallout as f64) * 100.0
            > PERCENTAGE_TILES_TO_WIN_FFA
            || elapsed_seconds >= HARD_TIME_LIMIT_SECONDS
        {
            game.winner = Some(max_id);
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
