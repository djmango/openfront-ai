#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_game_record import (  # noqa: E402
    PlayerInfo,
    analyze_record,
    format_analysis,
    load_sidecars,
    normalize_intent_type,
)


def _record() -> dict:
    return {
        "info": {
            "gameID": "AbCd1234",
            "config": {
                "gameMap": "Caucasus",
                "gameMode": "Team",
                "playerTeams": "Humans Vs Nations",
                "difficulty": "Easy",
                "bots": 0,
                "nations": 1,
            },
            "players": [
                {"clientID": "AGENTRL1", "username": "Agent"},
                {"clientID": "AGENTRL2", "username": "AgentB"},
            ],
            "num_turns": 400,
            "duration": 12,
        },
        "turns": [
            {
                "turnNumber": 1,
                "intents": [
                    {"type": "spawn", "clientID": "AGENTRL1", "tile": 10},
                    {"type": "spawn", "clientID": "AGENTRL2", "tile": 20},
                ],
            },
            {
                "turnNumber": 50,
                "intents": [
                    {
                        "type": "allianceRequest",
                        "clientID": "AGENTRL1",
                        "recipient": "pid-b",
                    }
                ],
            },
            {
                "turnNumber": 60,
                "intents": [
                    {
                        "type": "allianceRequest",
                        "clientID": "AGENTRL2",
                        "recipient": "pid-a",
                    }
                ],
            },
            {
                "turnNumber": 80,
                "intents": [
                    {
                        "type": "donate_gold",
                        "clientID": "AGENTRL1",
                        "recipient": "pid-b",
                        "gold": 1000,
                    }
                ],
            },
            {
                "turnNumber": 90,
                "intents": [
                    {
                        "type": "donate_gold",
                        "clientID": "AGENTRL1",
                        "recipient": "nation-x",
                        "gold": 50,
                    }
                ],
            },
            {
                "turnNumber": 100,
                "intents": [
                    {"type": "attack", "clientID": "AGENTRL2", "targetID": None, "troops": 9}
                ],
            },
        ],
    }


def _roster() -> dict[str, PlayerInfo]:
    return {
        "pid-a": PlayerInfo(
            player_id="pid-a",
            client_id="AGENTRL1",
            name="Agent",
            player_type="Human",
        ),
        "pid-b": PlayerInfo(
            player_id="pid-b",
            client_id="AGENTRL2",
            name="AgentB",
            player_type="Human",
        ),
        "nation-x": PlayerInfo(
            player_id="nation-x",
            name="Rome",
            player_type="Nation",
        ),
    }


def test_normalize_and_cross_pact() -> None:
    assert normalize_intent_type("allianceRequest") == "alliance_request"
    a = analyze_record(_record(), roster=_roster())
    assert a.game_id == "AbCd1234"
    assert a.partner_allied is True
    assert a.partner_allied_ticks == [(60, None)]
    agent = next(s for s in a.agents if s.client_id == "AGENTRL1")
    assert agent.alliance_to_partner == 1
    assert agent.alliance_to_npc == 0
    assert agent.donate_gold_partner == 1000
    assert agent.donate_gold_npc == 50
    buddy = next(s for s in a.agents if s.client_id == "AGENTRL2")
    assert buddy.terra_attacks == 1
    text = format_analysis(a)
    assert "YES" in text
    assert "t=60" in text
    assert "AgentB" in text
    assert "Rome" in text


def test_never_allied_without_cross_request() -> None:
    rec = _record()
    rec["turns"] = rec["turns"][:2]  # spawn + A->B only
    a = analyze_record(rec, roster=_roster())
    assert a.partner_allied is False
    text = format_analysis(a)
    assert "NEVER" in text


def test_thinking_sidecar(tmp_path: Path) -> None:
    rec_path = tmp_path / "game.json"
    rec_path.write_text(json.dumps(_record()))
    (tmp_path / "game.thinking.json").write_text(
        json.dumps(
            {
                "v": 1,
                "o": "timeout",
                "T": 400,
                "n": 2,
                "a": ["noop", "attack"],
                "s": [[10, 1, 12, "attack x"]],
            }
        )
    )
    sides = load_sidecars(rec_path)
    a = analyze_record(_record(), sidecars=sides, roster=_roster())
    assert a.thinking_preview
    assert "attack" in a.thinking_preview[0]


def test_roster_from_info_player_id() -> None:
    rec = _record()
    rec["info"]["players"][0]["playerID"] = "pid-a"
    rec["info"]["players"][1]["playerID"] = "pid-b"
    rec["info"]["roster"] = [
        {"id": "pid-a", "clientID": "AGENTRL1", "name": "Agent", "playerType": "Human"},
        {"id": "pid-b", "clientID": "AGENTRL2", "name": "AgentB", "playerType": "Human"},
        {"id": "nation-x", "name": "Rome", "playerType": "Nation"},
    ]
    from analyze_game_record import roster_from_record

    roster = roster_from_record(rec)
    a = analyze_record(rec, roster=roster)
    assert a.partner_allied is True
    agent = next(s for s in a.agents if s.client_id == "AGENTRL1")
    assert agent.player_id == "pid-a"
    assert agent.donate_gold_partner == 1000


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_normalize_and_cross_pact()
    test_never_allied_without_cross_request()
    test_roster_from_info_player_id()
    with TemporaryDirectory() as d:
        test_thinking_sidecar(Path(d))
    print("analyze_game_record_test: ok")
