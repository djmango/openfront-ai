//! Tribe bot spawner (`execution/TribeSpawner.ts`).

use crate::execution::spawn::SpawnExecution;
use crate::game::{PlayerInfo, PlayerType};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

const PREFIXES: &[&str] = &[
    "Iron", "Stone", "Blood", "Shadow", "Storm", "Fire", "Frost", "Wild",
];
const SUFFIXES: &[&str] = &[
    "Clan", "Tribe", "Horde", "Pack", "Band", "Kin", "Folk", "Brood",
];

pub struct TribeSpawner {
    game_id: String,
    random: PseudoRandom,
}

impl TribeSpawner {
    pub fn new(game_id: &str) -> Self {
        Self {
            game_id: game_id.to_string(),
            random: PseudoRandom::new(simple_hash(game_id) + 2),
        }
    }

    pub fn spawn_tribes(&mut self, n: u32) -> Vec<SpawnExecution> {
        (0..n)
            .map(|_| {
                SpawnExecution::new(
                    self.game_id.clone(),
                    PlayerInfo {
                        name: self.random_tribe_name(),
                        player_type: PlayerType::Bot,
                        client_id: None,
                        id: self.random.next_id(),
                    },
                    None,
                )
            })
            .collect()
    }

    fn random_tribe_name(&mut self) -> String {
        let p = PREFIXES[self.random.next_int(0, PREFIXES.len() as i32) as usize];
        let s = SUFFIXES[self.random.next_int(0, SUFFIXES.len() as i32) as usize];
        format!("{p} {s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::PseudoRandom;
    use crate::util::simple_hash;

    #[test]
    fn first_tribe_ids_match_ts_jby2g() {
        let game_id = "jby2gMJF";
        let mut tribe_rng = PseudoRandom::new(simple_hash(game_id) + 2);
        let expected = ["s8ozfync", "czjwcp3e", "a2dspr0g"];
        for (i, want) in expected.iter().enumerate() {
            tribe_rng.next_int(0, PREFIXES.len() as i32);
            tribe_rng.next_int(0, SUFFIXES.len() as i32);
            assert_eq!(tribe_rng.next_id(), *want, "tribe {i}");
        }
    }
}
