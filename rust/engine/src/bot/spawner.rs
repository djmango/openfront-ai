use crate::game::{Game, Player, PlayerType};
use crate::prng::PseudoRandom;

pub fn spawn_tribes(game: &mut Game, n: u32, prng: &mut PseudoRandom) {
    for i in 0..n {
        let client_id = format!("TRIBE{i:03}");
        let id = prng.next_id();
        let id_hash = crate::util::simple_hash(&id);
        game.add_player(Player {
            id,
            client_id: client_id.clone(),
            small_id: (i + 1) as u16,
            id_hash,
            player_type: PlayerType::Bot,
            troops: 0,
            gold: 0,
            tiles_owned: 0,
            alive: true,
            spawn_tile: None,
            units: vec![],
            border_tiles: Default::default(),
            owned_tiles: Default::default(),
            ..Default::default()
        });
    }
}
