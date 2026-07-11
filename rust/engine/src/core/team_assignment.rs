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

// Ported from `TeamAssignment.test.ts`.
#[cfg(test)]
mod tests {
    use super::*;

    fn teams() -> Vec<String> {
        vec!["Red".into(), "Blue".into()]
    }

    fn player(id: &str, clan: Option<&str>) -> PlayerInfo {
        PlayerInfo {
            name: format!("Player {id}"),
            player_type: PlayerType::Human,
            client_id: None,
            id: id.into(),
            clan_tag: clan.map(str::to_string),
            friends: Vec::new(),
            team: None,
        }
    }

    fn player_with_friends(
        id: &str,
        friends: &[&str],
        clan: Option<&str>,
        client_id: &str,
    ) -> PlayerInfo {
        PlayerInfo {
            name: format!("Player {id}"),
            player_type: PlayerType::Human,
            client_id: Some(client_id.into()),
            id: id.into(),
            clan_tag: clan.map(str::to_string),
            friends: friends.iter().map(|s| s.to_string()).collect(),
            team: None,
        }
    }

    fn team_of<'a>(result: &'a HashMap<String, String>, p: &PlayerInfo) -> &'a str {
        result.get(&p.id).map(String::as_str).unwrap_or("<unassigned>")
    }

    #[test]
    fn assigns_players_alternately_when_no_clans_are_present() {
        let players = vec![
            player("1", None),
            player("2", None),
            player("3", None),
            player("4", None),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Blue");
        assert_eq!(team_of(&result, &players[2]), "Red");
        assert_eq!(team_of(&result, &players[3]), "Blue");
    }

    #[test]
    fn keeps_clan_members_together_on_the_same_team() {
        let players = vec![
            player("1", Some("CLANA")),
            player("2", Some("CLANA")),
            player("3", Some("CLANB")),
            player("4", Some("CLANB")),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "Blue");
        assert_eq!(team_of(&result, &players[3]), "Blue");
    }

    #[test]
    fn handles_mixed_clan_and_non_clan_players() {
        let players = vec![
            player("1", Some("CLANA")),
            player("2", Some("CLANA")),
            player("3", None),
            player("4", None),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "Blue");
        assert_eq!(team_of(&result, &players[3]), "Blue");
    }

    #[test]
    fn kicks_players_when_teams_are_full() {
        let players = vec![
            player("1", Some("CLANA")),
            player("2", Some("CLANA")),
            player("3", Some("CLANA")),
            player("4", Some("CLANA")),
            player("5", Some("CLANB")),
            player("6", Some("CLANB")),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "Red");
        assert_eq!(team_of(&result, &players[3]), "kicked");
        assert_eq!(team_of(&result, &players[4]), "Blue");
        assert_eq!(team_of(&result, &players[5]), "Blue");
    }

    #[test]
    fn handles_an_empty_player_list() {
        let (result, _) = assign_teams(&[], &teams());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn handles_a_single_player() {
        let players = vec![player("1", None)];
        let (result, _) = assign_teams(&players, &teams());
        assert_eq!(team_of(&result, &players[0]), "Red");
    }

    #[test]
    fn handles_multiple_clans_with_different_sizes() {
        let players = vec![
            player("1", Some("CLANA")),
            player("2", Some("CLANA")),
            player("3", Some("CLANA")),
            player("4", Some("CLANB")),
            player("5", Some("CLANB")),
            player("6", Some("CLANC")),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "Red");
        assert_eq!(team_of(&result, &players[3]), "Blue");
        assert_eq!(team_of(&result, &players[4]), "Blue");
        assert_eq!(team_of(&result, &players[5]), "Blue");
    }

    #[test]
    fn distributes_players_among_a_larger_number_of_teams() {
        let players = vec![
            player("1", Some("CLANA")),
            player("2", Some("CLANA")),
            player("3", Some("CLANA")),
            player("4", Some("CLANB")),
            player("5", Some("CLANB")),
            player("6", Some("CLANC")),
            player("7", None),
            player("8", None),
            player("9", None),
            player("10", None),
            player("11", None),
            player("12", None),
            player("13", None),
            player("14", None),
        ];
        let big_teams: Vec<String> = ["Red", "Blue", "Yellow", "Green", "Purple", "Orange", "Teal"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (result, _) = assign_teams(&players, &big_teams);

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "kicked");
        assert_eq!(team_of(&result, &players[3]), "Blue");
        assert_eq!(team_of(&result, &players[4]), "Blue");
        assert_eq!(team_of(&result, &players[5]), "Yellow");
        assert_eq!(team_of(&result, &players[6]), "Green");
        assert_eq!(team_of(&result, &players[7]), "Purple");
        assert_eq!(team_of(&result, &players[8]), "Orange");
        assert_eq!(team_of(&result, &players[9]), "Teal");
        assert_eq!(team_of(&result, &players[10]), "Yellow");
        assert_eq!(team_of(&result, &players[11]), "Green");
        assert_eq!(team_of(&result, &players[12]), "Purple");
        assert_eq!(team_of(&result, &players[13]), "Orange");
    }

    #[test]
    fn keeps_two_friends_on_the_same_team() {
        let players = vec![
            player_with_friends("1", &["2"], None, "1"),
            player_with_friends("2", &["1"], None, "2"),
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        let team0 = team_of(&result, &players[0]);
        assert_eq!(team_of(&result, &players[1]), team0);
        assert_ne!(team_of(&result, &players[2]), team0);
        assert_ne!(team_of(&result, &players[3]), team0);
    }

    #[test]
    fn groups_a_chain_of_friends_transitively() {
        // 6 players, 2 teams -> maxTeamSize = 3 (enough room for a 3-friend chain).
        let players = vec![
            player_with_friends("1", &["2"], None, "1"),
            player_with_friends("2", &["3"], None, "2"),
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
            player_with_friends("5", &[], None, "5"),
            player_with_friends("6", &[], None, "6"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        let team0 = team_of(&result, &players[0]);
        assert_eq!(team_of(&result, &players[1]), team0);
        assert_eq!(team_of(&result, &players[2]), team0);
    }

    #[test]
    fn treats_one_directional_friendship_as_a_group() {
        let players = vec![
            player_with_friends("1", &["2"], None, "1"),
            player_with_friends("2", &[], None, "2"), // doesn't list 1 back
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(
            team_of(&result, &players[0]),
            team_of(&result, &players[1])
        );
    }

    #[test]
    fn merges_friend_and_clan_groups_when_they_overlap() {
        // 1 and 2 share clan CLANA, 2 is friends with 3 (no clan) -> all three end up on
        // the same team. 6 players, maxTeamSize = 3.
        let players = vec![
            player_with_friends("1", &[], Some("CLANA"), "1"),
            player_with_friends("2", &["3"], Some("CLANA"), "2"),
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
            player_with_friends("5", &[], None, "5"),
            player_with_friends("6", &[], None, "6"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        let team0 = team_of(&result, &players[0]);
        assert_eq!(team_of(&result, &players[1]), team0);
        assert_eq!(team_of(&result, &players[2]), team0);
    }

    #[test]
    fn spills_friend_group_overflow_to_other_teams_without_kicking() {
        // 4-player friend group + 2 strangers, maxTeamSize = ceil(6/2) = 3. Friend
        // overflow spills to the other team rather than getting kicked.
        let players = vec![
            player_with_friends("1", &["2", "3", "4"], None, "1"),
            player_with_friends("2", &[], None, "2"),
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
            player_with_friends("5", &[], None, "5"),
            player_with_friends("6", &[], None, "6"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        assert_eq!(team_of(&result, &players[0]), "Red");
        assert_eq!(team_of(&result, &players[1]), "Red");
        assert_eq!(team_of(&result, &players[2]), "Red");
        assert_eq!(team_of(&result, &players[3]), "Blue");
        assert_eq!(team_of(&result, &players[4]), "Blue");
        assert_eq!(team_of(&result, &players[5]), "Blue");
    }

    #[test]
    fn keys_friend_grouping_on_client_id_not_player_info_id() {
        // clientID and PlayerInfo.id are distinct. The friends list references clientIDs
        // ("client-2", "client-1"). If grouping ever regressed to keying on
        // PlayerInfo.id ("player-1"/"player-2"), no edges would form and these two would
        // land on opposite teams.
        let players = vec![
            player_with_friends("player-1", &["client-2"], None, "client-1"),
            player_with_friends("player-2", &["client-1"], None, "client-2"),
            player_with_friends("player-3", &[], None, "client-3"),
            player_with_friends("player-4", &[], None, "client-4"),
        ];
        let (result, _) = assign_teams(&players, &teams());

        let team0 = team_of(&result, &players[0]);
        assert_eq!(team_of(&result, &players[1]), team0);
        assert_ne!(team_of(&result, &players[2]), team0);
        assert_ne!(team_of(&result, &players[3]), team0);
    }

    #[test]
    fn still_kicks_when_every_team_is_forced_to_capacity() {
        // 5 friends in a clique, 2 teams. Default maxTeamSize = ceil(5/2) = 3 gives
        // capacity 6 >= 5, so nobody would be kicked - force capacity down via the
        // explicit max_team_size override (TS `assignTeams(players, teams, 2)`; the real
        // engine's only call site never overrides this, see `assign_teams_with_max_size`'s
        // doc comment) to confirm kicks resume once capacity is genuinely insufficient.
        let players = vec![
            player_with_friends("1", &["2", "3", "4", "5"], None, "1"),
            player_with_friends("2", &[], None, "2"),
            player_with_friends("3", &[], None, "3"),
            player_with_friends("4", &[], None, "4"),
            player_with_friends("5", &[], None, "5"),
        ];
        let (result, _) = assign_teams_with_max_size(&players, &teams(), Some(2));

        let kicked = players
            .iter()
            .filter(|p| team_of(&result, p) == "kicked")
            .count();
        assert_eq!(kicked, 1);
    }
}
