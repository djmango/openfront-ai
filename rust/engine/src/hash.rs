//! Lightweight sync checksum matching `GameImpl.hash` / `PlayerImpl.hash` / `UnitImpl.hash`.
//! Uses i64 - summed player hashes exceed i32 on large games.

use crate::game::{Game, Player, Unit};
use crate::util::simple_hash;

pub fn game_hash(game: &Game) -> i64 {
    let mut hash = 1i64;
    for p in game.players_in_order() {
        hash += player_hash(p);
    }
    hash
}

pub fn player_hash(p: &Player) -> i64 {
    p.id_hash as i64 * (p.troops as i64 + p.tiles_owned as i64)
        + p.units.iter().map(unit_hash).sum::<i64>()
}

pub fn unit_hash(u: &Unit) -> i64 {
    u.tile as i64 + simple_hash(&u.unit_type) as i64 * u.id as i64
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
        assert_eq!(game_hash(&g), 1 + simple_hash("p1") as i64 * 15);
    }
}
