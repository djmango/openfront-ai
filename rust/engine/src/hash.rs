//! Lightweight sync checksum matching `GameImpl.hash` / `PlayerImpl.hash` / `UnitImpl.hash`.
//! TS accumulates with IEEE-754 `number`; match that semantics for archived parity.

use crate::game::{Game, Player, Unit};
use crate::util::simple_hash;

pub fn game_hash(game: &Game) -> i64 {
    let mut hash = 1.0_f64;
    for p in game.players_in_order() {
        hash += player_hash_js(p);
    }
    hash as i64
}

pub fn player_hash(p: &Player) -> i64 {
    player_hash_js(p) as i64
}

pub fn player_hash_js(p: &Player) -> f64 {
    p.id_hash as f64 * (p.troops as f64 + p.tiles_owned as f64)
        + p.units.iter().map(unit_hash_js).sum::<f64>()
}

pub fn unit_hash(u: &Unit) -> i64 {
    unit_hash_js(u) as i64
}

pub fn unit_hash_js(u: &Unit) -> f64 {
    u.tile as f64 + simple_hash(&u.unit_type) as f64 * u.id as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    #[test]
    fn empty_game() {
        assert_eq!(game_hash(&Game::default()), 1);
    }

    #[test]
    fn one_player() {
        let mut g = Game::default();
        g.add_player(Player {
            id: "p1".into(),
            client_id: "c1".into(),
            small_id: 1,
            id_hash: simple_hash("p1"),
            troops: 10,
            tiles_owned: 5,
            alive: true,
            ..Default::default()
        });
        assert_eq!(game_hash(&g), (1.0 + simple_hash("p1") as f64 * 15.0) as i64);
    }
}
