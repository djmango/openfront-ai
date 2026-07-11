//! Team assignment for Team mode (`TeamAssignment.ts` at pinned commit).

use crate::game::{PlayerInfo, PlayerType};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;
use std::collections::{HashMap, HashSet};

pub const BOT_TEAM: &str = "Bot";
pub const HUMANS_TEAM: &str = "Humans";
pub const NATIONS_TEAM: &str = "Nations";

const DUOS: &str = "Duos";
const TRIOS: &str = "Trios";
const QUADS: &str = "Quads";

/// TS `GameImpl.populateTeams`  -  colored team names for spawn areas / alliances.
pub fn populate_player_teams(
    game_mode: &str,
    player_teams: Option<&crate::core::schemas::PlayerTeamsConfig>,
    num_humans: usize,
    num_nations: usize,
) -> Vec<String> {
    if game_mode != "Team" {
        return Vec::new();
    }

    let Some(cfg) = player_teams else {
        return Vec::new();
    };

    if cfg.is_humans_vs_nations() {
        return vec![HUMANS_TEAM.into(), NATIONS_TEAM.into()];
    }

    let num_teams = match cfg {
        crate::core::schemas::PlayerTeamsConfig::Count(n) => *n as usize,
        crate::core::schemas::PlayerTeamsConfig::Mode(s) => {
            let players = num_humans + num_nations;
            match s.as_str() {
                DUOS => (players + 1) / 2,
                TRIOS => (players + 2) / 3,
                QUADS => (players + 3) / 4,
                other => panic!("Unknown TeamCountConfig {other}"),
            }
        }
    };

    if num_teams < 2 {
        panic!("Too few teams: {num_teams}");
    }

    if num_teams < 8 {
        let mut teams = vec!["Red".into(), "Blue".into()];
        if num_teams >= 3 {
            teams.push("Yellow".into());
        }
        if num_teams >= 4 {
            teams.push("Green".into());
        }
        if num_teams >= 5 {
            teams.push("Purple".into());
        }
        if num_teams >= 6 {
            teams.push("Orange".into());
        }
        if num_teams >= 7 {
            teams.push("Teal".into());
        }
        teams
    } else {
        (1..=num_teams).map(|i| format!("Team {i}")).collect()
    }
}

pub fn get_max_team_size(num_players: usize, num_teams: usize) -> usize {
    num_players.div_ceil(num_teams.max(1))
}

/// TS `assignTeams`  -  team per player id (`"kicked"` when benched) plus Map
/// insertion order (clan blocks first, then non-clan humans, then shuffled nations).
pub fn assign_teams(players: &[PlayerInfo], teams: &[String]) -> (HashMap<String, String>, Vec<usize>) {
    assign_teams_with_max_size(players, teams, None)
}

/// TS `assignTeams`'s optional third `maxTeamSize` parameter - every in-game caller
/// (`GameImpl.ts`'s real team assignment) omits it and gets the default
/// `getMaxTeamSize(players.length, teams.length)`; only the client-only lobby-preview
/// helper (`assignTeamsLobbyPreview`, not ported - pure UI) ever overrides it.
pub fn assign_teams_with_max_size(
    players: &[PlayerInfo],
    teams: &[String],
    max_team_size: Option<usize>,
) -> (HashMap<String, String>, Vec<usize>) {
    let max_team_size =
        max_team_size.unwrap_or_else(|| get_max_team_size(players.len(), teams.len()));
    let mut result: HashMap<String, String> = HashMap::new();
    let mut insertion_order: Vec<usize> = Vec::new();
    let mut team_player_count: HashMap<String, usize> = HashMap::new();

    let mut clan_groups: HashMap<String, Vec<usize>> = HashMap::new();
    let mut non_clan: Vec<usize> = Vec::new();
    for (i, p) in players.iter().enumerate() {
        if let Some(clan) = p.clan_tag.as_deref().filter(|c| !c.is_empty()) {
            clan_groups.entry(clan.to_string()).or_default().push(i);
        } else {
            non_clan.push(i);
        }
    }

    let mut sorted_clans: Vec<Vec<usize>> = clan_groups.into_values().collect();
    sorted_clans.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));

    for clan in sorted_clans {
        let mut team: Option<String> = None;
        let mut team_size = 0usize;
        for t in teams {
            let p = *team_player_count.get(t).unwrap_or(&0);
            if team.is_some() && team_size <= p {
                continue;
            }
            team_size = p;
            team = Some(t.clone());
        }
        let Some(team) = team else {
            continue;
        };
        let mut size = team_size;
        for &player_idx in &clan {
            if size < max_team_size {
                size += 1;
                result.insert(players[player_idx].id.clone(), team.clone());
            } else {
                result.insert(players[player_idx].id.clone(), "kicked".into());
            }
            insertion_order.push(player_idx);
        }
        team_player_count.insert(team, size);
    }

    let present_client_ids: HashSet<String> = players
        .iter()
        .filter_map(|p| p.client_id.as_ref().filter(|c| !c.is_empty()).cloned())
        .collect();

    let mut friend_graph: HashMap<String, HashSet<String>> = HashMap::new();
    for p in players {
        let Some(cid) = p.client_id.as_deref().filter(|c| !c.is_empty()) else {
            continue;
        };
        for friend_id in &p.friends {
            if !present_client_ids.contains(friend_id) {
                continue;
            }
            friend_graph
                .entry(cid.to_string())
                .or_default()
                .insert(friend_id.clone());
            friend_graph
                .entry(friend_id.clone())
                .or_default()
                .insert(cid.to_string());
        }
    }

    let mut team_by_client: HashMap<String, String> = HashMap::new();
    for p in players {
        if let Some(team) = result.get(&p.id).filter(|t| *t != "kicked") {
            if let Some(cid) = p.client_id.as_deref().filter(|c| !c.is_empty()) {
                team_by_client.insert(cid.to_string(), team.clone());
            }
        }
    }

    let place_player = |player_idx: usize,
                        result: &mut HashMap<String, String>,
                        team_player_count: &mut HashMap<String, usize>,
                        team_by_client: &mut HashMap<String, String>| {
        let p = &players[player_idx];
        let my_friends = p
            .client_id
            .as_deref()
            .and_then(|cid| friend_graph.get(cid));

        let mut best_team: Option<String> = None;
        let mut best_friend_count = -1i32;
        let mut best_size = usize::MAX;

        for team in teams {
            let size = *team_player_count.get(team).unwrap_or(&0);
            if size >= max_team_size {
                continue;
            }
            let mut friends_on_team = 0;
            if let Some(friends) = my_friends {
                for friend_id in friends {
                    if team_by_client.get(friend_id.as_str()) == Some(team) {
                        friends_on_team += 1;
                    }
                }
            }
            if friends_on_team > best_friend_count
                || (friends_on_team == best_friend_count && size < best_size)
            {
                best_friend_count = friends_on_team;
                best_size = size;
                best_team = Some(team.clone());
            }
        }

        if let Some(team) = best_team {
            *team_player_count.entry(team.clone()).or_default() += 1;
            result.insert(p.id.clone(), team.clone());
            if let Some(cid) = p.client_id.as_deref().filter(|c| !c.is_empty()) {
                team_by_client.insert(cid.to_string(), team);
            }
        } else {
            result.insert(p.id.clone(), "kicked".into());
        }
    };

    let mut nation_idxs: Vec<usize> = non_clan
        .iter()
        .copied()
        .filter(|&i| players[i].player_type == PlayerType::Nation)
        .collect();
    if !nation_idxs.is_empty() {
        let mut random = PseudoRandom::new(simple_hash(&players[nation_idxs[0]].id));
        nation_idxs = random.shuffle_array(&nation_idxs);
    }

    let other_idxs: Vec<usize> = non_clan
        .iter()
        .copied()
        .filter(|&i| players[i].player_type != PlayerType::Nation)
        .collect();

    for idx in other_idxs.into_iter().chain(nation_idxs) {
        place_player(
            idx,
            &mut result,
            &mut team_player_count,
            &mut team_by_client,
        );
        insertion_order.push(idx);
    }

    (result, insertion_order)
}
