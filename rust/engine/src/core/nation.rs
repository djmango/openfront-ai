//! Nation selection for game bootstrap (TS `NationCreation.ts`).

use crate::core::schemas::{GameConfig, NationsConfig, PlayerTeamsConfig};
use crate::map::Nation;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

/// TS `Game.Nation` - manifest row plus assigned player id.
#[derive(Debug, Clone)]
pub struct SpawnedNation {
    pub nation: Nation,
    pub player_id: String,
}

const GAME_TYPE_PUBLIC: &str = "Public";
const GAME_MODE_TEAM: &str = "Team";
const GAME_MAP_SIZE_COMPACT: &str = "Compact";

/// Creates the nations list for a game (TS `createNationsForGame`).
pub fn create_nations_for_game(
    config: &GameConfig,
    manifest_nations: &[Nation],
    additional_nations: &[Nation],
    num_humans: u32,
    random: &mut PseudoRandom,
) -> Vec<SpawnedNation> {
    let is_compact_map = config.game_map_size == GAME_MAP_SIZE_COMPACT;
    let is_humans_vs_nations = config.game_mode == GAME_MODE_TEAM
        && config
            .player_teams
            .as_ref()
            .is_some_and(PlayerTeamsConfig::is_humans_vs_nations);

    match &config.nations {
        NationsConfig::Mode(s) if s == "disabled" => return vec![],
        NationsConfig::Count(count) => {
            return create_random_nations(
                *count as usize,
                manifest_nations,
                additional_nations,
                random,
            );
        }
        _ => {}
    }

    if config.game_type == GAME_TYPE_PUBLIC {
        if is_humans_vs_nations {
            return create_random_nations(
                num_humans as usize,
                manifest_nations,
                additional_nations,
                random,
            );
        }
        if is_compact_map {
            let target_count = get_compact_map_nation_count(manifest_nations.len(), true);
            let shuffled = random.shuffle_array(manifest_nations);
            return shuffled
                .into_iter()
                .take(target_count)
                .map(|n| to_spawned_nation(&n, random))
                .collect();
        }
    }

    manifest_nations
        .iter()
        .map(|n| to_spawned_nation(n, random))
        .collect()
}

/// Compact maps use 25% of manifest nations (minimum 1).
pub fn get_compact_map_nation_count(manifest_nation_count: usize, is_compact_map: bool) -> usize {
    if manifest_nation_count == 0 {
        return 0;
    }
    if is_compact_map {
        return (manifest_nation_count / 4).max(1);
    }
    manifest_nation_count
}

fn create_random_nations(
    target_count: usize,
    manifest_nations: &[Nation],
    additional_nations: &[Nation],
    random: &mut PseudoRandom,
) -> Vec<SpawnedNation> {
    let shuffled = random.shuffle_array(manifest_nations);
    if target_count <= manifest_nations.len() {
        return shuffled
            .into_iter()
            .take(target_count)
            .map(|n| to_spawned_nation(&n, random))
            .collect();
    }

    let mut nations: Vec<SpawnedNation> = shuffled
        .into_iter()
        .map(|n| to_spawned_nation(&n, random))
        .collect();
    let mut used_names: HashSet<String> = nations.iter().map(|n| n.nation.name.clone()).collect();
    let mut remaining = target_count.saturating_sub(manifest_nations.len());

    if remaining > 0 && !additional_nations.is_empty() {
        let candidates: Vec<&Nation> = additional_nations
            .iter()
            .filter(|n| !used_names.contains(&n.name))
            .collect();
        let shuffled_extras = random.shuffle_array(&candidates);
        let picked = shuffled_extras.into_iter().take(remaining).collect::<Vec<_>>();
        for extra in picked {
            nations.push(to_spawned_nation(extra, random));
            used_names.insert(extra.name.clone());
        }
        remaining = target_count.saturating_sub(nations.len());
    }

    for _ in 0..remaining {
        let name = generate_unique_nation_name(random, &used_names);
        used_names.insert(name.clone());
        nations.push(SpawnedNation {
            nation: Nation {
                name,
                flag: None,
                coordinates: None,
            },
            player_id: random.next_id(),
        });
    }

    nations
}

fn to_spawned_nation(n: &Nation, random: &mut PseudoRandom) -> SpawnedNation {
    SpawnedNation {
        nation: n.clone(),
        player_id: random.next_id(),
    }
}

fn generate_unique_nation_name(random: &mut PseudoRandom, used_names: &HashSet<String>) -> String {
    for _ in 0..1000 {
        let name = generate_nation_name(random);
        if !used_names.contains(&name) {
            return name;
        }
    }
    let base = generate_nation_name(random);
    let mut counter = 1u32;
    loop {
        let candidate = format!("{base} {counter}");
        if !used_names.contains(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

#[derive(Clone, Copy)]
enum TemplatePart {
    Lit(&'static str),
    Noun,
    PluralNoun,
}

fn generate_nation_name(random: &mut PseudoRandom) -> String {
    let template = NAME_TEMPLATES[random.next_int(0, NAME_TEMPLATES.len() as i32) as usize];
    let noun = NOUNS[random.next_int(0, NOUNS.len() as i32) as usize];
    let mut parts = Vec::new();
    for part in template {
        match part {
            TemplatePart::PluralNoun => parts.push(pluralize(noun)),
            TemplatePart::Noun => parts.push(noun.to_string()),
            TemplatePart::Lit(s) => parts.push(s.to_string()),
        }
    }
    parts.join(" ")
}

fn pluralize(noun: &str) -> String {
    for &(key, plural) in SPECIAL_PLURALS {
        if key == noun {
            return plural.to_string();
        }
    }
    if noun.ends_with('s')
        || noun.ends_with("ch")
        || noun.ends_with("sh")
        || noun.ends_with('x')
        || noun.ends_with('z')
    {
        return format!("{noun}es");
    }
    if noun.ends_with('y') {
        let bytes = noun.as_bytes();
        if bytes.len() >= 2 {
            let prev = bytes[bytes.len() - 2] as char;
            if !"aeiou".contains(prev) {
                return format!("{}ies", &noun[..noun.len() - 1]);
            }
        }
    }
    if O_TO_OES.iter().any(|&v| v == noun) {
        return format!("{noun}es");
    }
    format!("{noun}s")
}

const O_TO_OES: &[&str] = &["Potato", "Tomato", "Volcano", "Torpedo"];

const SPECIAL_PLURALS: &[(&str, &str)] = &[
    ("Cactus", "Cacti"),
    ("Platypus", "Platypuses"),
    ("Moose", "Moose"),
    ("Octopus", "Octopi"),
    ("Cyclops", "Cyclopes"),
    ("Samurai", "Samurai"),
    ("Fish", "Fish"),
    ("Salmon", "Salmon"),
    ("Cod", "Cod"),
    ("Enderman", "Endermen"),
    ("Mitochondria", "Mitochondria"),
];

include!("nation_names_data.inc");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::NationsConfig;
    use serde_json::json;

    fn make_manifest_nations(count: usize) -> Vec<Nation> {
        (0..count)
            .map(|i| Nation {
                name: format!("Manifest{i}"),
                flag: Some(String::new()),
                coordinates: Some([i as i32, i as i32]),
            })
            .collect()
    }

    fn make_additional_nations(names: &[&str]) -> Vec<Nation> {
        names
            .iter()
            .map(|name| Nation {
                name: (*name).to_string(),
                flag: None,
                coordinates: None,
            })
            .collect()
    }

    fn make_config(nations: NationsConfig, game_type: &str, game_mode: &str, map_size: &str) -> GameConfig {
        GameConfig {
            game_map: "World".into(),
            difficulty: "Medium".into(),
            donate_gold: false,
            donate_troops: false,
            game_type: game_type.into(),
            game_mode: game_mode.into(),
            game_map_size: map_size.into(),
            nations,
            bots: 0,
            infinite_gold: false,
            infinite_troops: false,
            instant_build: false,
            random_spawn: false,
            doomsday_clock: None,
            disabled_units: None,
            player_teams: None,
            disable_alliances: None,
            spawn_immunity_duration: None,
            starting_gold: None,
            gold_multiplier: None,
        }
    }

    fn nation_names(nations: &[SpawnedNation]) -> Vec<String> {
        nations.iter().map(|n| n.nation.name.clone()).collect()
    }

    #[test]
    fn nations_disabled_returns_empty() {
        let cfg = make_config(
            NationsConfig::Mode("disabled".into()),
            "Public",
            "Free For All",
            "Normal",
        );
        let mut rng = PseudoRandom::new(1);
        let out = create_nations_for_game(&cfg, &make_manifest_nations(5), &[], 3, &mut rng);
        assert!(out.is_empty());
    }

    #[test]
    fn explicit_count_uses_manifest_only_when_enough() {
        let manifest = make_manifest_nations(4);
        let extras = make_additional_nations(&["ExtraA", "ExtraB", "ExtraC"]);
        let cfg = make_config(NationsConfig::Count(3), "Singleplayer", "Free For All", "Normal");
        let mut rng = PseudoRandom::new(1);
        let nations = create_nations_for_game(&cfg, &manifest, &extras, 0, &mut rng);
        assert_eq!(nations.len(), 3);
        for name in nation_names(&nations) {
            assert!(name.starts_with("Manifest"));
        }
    }

    #[test]
    fn fills_deficit_from_additional_pool() {
        let manifest = make_manifest_nations(2);
        let extras = make_additional_nations(&["ExtraA", "ExtraB", "ExtraC", "ExtraD", "ExtraE"]);
        let cfg = make_config(NationsConfig::Count(5), "Singleplayer", "Free For All", "Normal");
        let mut rng = PseudoRandom::new(7);
        let nations = create_nations_for_game(&cfg, &manifest, &extras, 0, &mut rng);
        assert_eq!(nations.len(), 5);
        let names = nation_names(&nations);
        assert_eq!(names.iter().filter(|n| n.starts_with("Manifest")).count(), 2);
        let from_pool: Vec<_> = names.iter().filter(|n| n.starts_with("Extra")).collect();
        assert_eq!(from_pool.len(), 3);
    }

    #[test]
    fn public_compact_uses_quarter_shuffled() {
        let manifest = make_manifest_nations(20);
        let cfg = make_config(
            NationsConfig::Mode("default".into()),
            "Public",
            "Free For All",
            "Compact",
        );
        let mut rng = PseudoRandom::new(42);
        let nations = create_nations_for_game(&cfg, &manifest, &[], 0, &mut rng);
        assert_eq!(nations.len(), 5);
        assert_eq!(get_compact_map_nation_count(20, true), 5);
    }

    #[test]
    fn public_default_uses_all_manifest() {
        let manifest = make_manifest_nations(4);
        let cfg = make_config(
            NationsConfig::Mode("default".into()),
            "Public",
            "Free For All",
            "Normal",
        );
        let mut rng = PseudoRandom::new(1);
        let nations = create_nations_for_game(&cfg, &manifest, &[], 0, &mut rng);
        assert_eq!(nations.len(), 4);
        let names = nation_names(&nations);
        for i in 0..4 {
            assert!(names.contains(&format!("Manifest{i}")));
        }
    }

    #[test]
    fn public_humans_vs_nations_matches_human_count() {
        let manifest = make_manifest_nations(10);
        let cfg = GameConfig {
            player_teams: Some(PlayerTeamsConfig::Mode("Humans Vs Nations".into())),
            ..make_config(
                NationsConfig::Mode("default".into()),
                "Public",
                "Team",
                "Normal",
            )
        };
        let mut rng = PseudoRandom::new(3);
        let nations = create_nations_for_game(&cfg, &manifest, &[], 4, &mut rng);
        assert_eq!(nations.len(), 4);
    }

    #[test]
    fn skips_pool_entries_colliding_with_manifest_names() {
        let manifest = make_manifest_nations(2);
        let extras = make_additional_nations(&["Manifest0", "Manifest1", "UniqueExtra"]);
        let cfg = make_config(NationsConfig::Count(3), "Singleplayer", "Free For All", "Normal");
        let mut rng = PseudoRandom::new(3);
        let nations = create_nations_for_game(&cfg, &manifest, &extras, 0, &mut rng);
        let names = nation_names(&nations);
        assert_eq!(names.len(), 3);
        assert_eq!(names.iter().collect::<HashSet<_>>().len(), 3);
        assert!(names.contains(&"UniqueExtra".to_string()));
    }

    #[test]
    fn parses_wire_config_with_nations_count() {
        let v = json!({
            "gameMap": "World",
            "difficulty": "Medium",
            "donateGold": false,
            "donateTroops": false,
            "gameType": "Public",
            "gameMode": "Free For All",
            "gameMapSize": "Normal",
            "nations": 12,
            "bots": 0,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": false
        });
        let cfg: GameConfig = serde_json::from_value(v).unwrap();
        assert!(matches!(cfg.nations, NationsConfig::Count(12)));
    }
}
