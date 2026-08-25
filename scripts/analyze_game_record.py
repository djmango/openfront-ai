#!/usr/bin/env python3
"""Text-first GameRecord analysis: what each agent did, without watching MP4.

Reads a client GameRecord (`info` + sparse `turns`) plus optional sidecars
(`.meta.json`, `.thinking.json`, `.debug.json`). When `tick_dump` is available,
replays the record natively so recipient player-IDs resolve to names and so
gold/tiles are sampled over the match.

Usage:
  uv run python scripts/analyze_game_record.py rust/replay-spool/nK1C3g1M.json
  uv run python scripts/analyze_game_record.py --spool rust/replay-spool --latest 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent

INTENT_ALIAS = {
    "allianceRequest": "alliance_request",
    "allianceReject": "alliance_reject",
    "allianceExtension": "alliance_extension",
    "breakAlliance": "break_alliance",
    "targetPlayer": "target_player",
    "cancel_attack": "retreat",
    "build_unit": "build",
}

# Intents that name another player (player-id string, not clientID).
COUNTERPARTY_KEYS = (
    "recipient",
    "requestor",
    "targetID",
    "target",
    "targetId",
)

NOTABLE_TYPES = {
    "spawn",
    "alliance_request",
    "alliance_reject",
    "alliance_extension",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "build",
    "launch_nuke",
    "boat",
    "embargo",
}


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def normalize_intent_type(raw: str) -> str:
    return INTENT_ALIAS.get(raw, raw)


def intent_counterparty(intent: dict[str, Any]) -> str | None:
    for key in COUNTERPARTY_KEYS:
        value = intent.get(key)
        if isinstance(value, str) and value:
            return value
        if value is not None and value != "":
            return str(value)
    return None


def find_tick_dump() -> Path | None:
    env = os.environ.get("OF_TICK_DUMP", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for candidate in (
        REPO / "rust/target/debug/tick_dump",
        REPO / "rust/target/release/tick_dump",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    which = shutil_which("tick_dump")
    return Path(which) if which else None


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


@dataclass
class PlayerInfo:
    player_id: str
    client_id: str = ""
    name: str = ""
    player_type: str = ""
    team: str | None = None
    identity: str = ""

    def label(self) -> str:
        if self.name and self.client_id:
            return f"{self.name} ({self.client_id})"
        if self.name:
            return self.name
        if self.client_id:
            return self.client_id
        return self.player_id or "?"

    def short(self) -> str:
        return self.name or self.client_id or self.player_id or "?"


@dataclass
class Event:
    tick: int
    client_id: str
    kind: str
    counterparty: str | None = None
    amount: int | None = None
    extra: str = ""
    count: int = 1


@dataclass
class AgentStats:
    client_id: str
    name: str
    player_id: str = ""
    actions: dict[str, int] = field(default_factory=dict)
    spawn_tick: int | None = None
    spawn_tile: int | None = None
    alliance_to_partner: int = 0
    alliance_to_npc: int = 0
    reject: int = 0
    break_alliance: int = 0
    donate_gold_partner: int = 0
    donate_gold_npc: int = 0
    donate_gold_partner_n: int = 0
    donate_gold_npc_n: int = 0
    donate_troops_partner: int = 0
    donate_troops_npc: int = 0
    donate_troops_partner_n: int = 0
    donate_troops_npc_n: int = 0
    attacks: int = 0
    terra_attacks: int = 0
    boats: int = 0
    builds: int = 0
    embargoes: int = 0


@dataclass
class PactEvent:
    tick: int
    kind: str  # formed | broken
    a: str
    b: str


@dataclass
class Analysis:
    game_id: str
    map: str
    mode: str
    teams: str
    difficulty: str
    bots: Any
    nations: Any
    num_turns: int
    duration_s: int | None
    outcome: str
    won: bool | None
    timed_out: bool | None
    engine: str | None
    agents: list[AgentStats]
    roster: list[PlayerInfo]
    pacts: list[PactEvent]
    partner_allied: bool
    partner_allied_ticks: list[tuple[int, int | None]]
    timeline: list[Event]
    tick_series: list[dict[str, Any]] = field(default_factory=list)
    thinking: dict[str, Any] | None = None
    thinking_preview: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    record_path: str = ""


def load_sidecars(record_path: Path) -> dict[str, Any]:
    stem = record_path.with_suffix("")
    out: dict[str, Any] = {}
    for kind in ("meta", "thinking", "debug"):
        # GameRecord is foo.json; sidecars are foo.meta.json (not foo.json.meta).
        path = record_path.parent / f"{record_path.stem}.{kind}.json"
        if path.is_file():
            try:
                out[kind] = _json_load(path)
            except json.JSONDecodeError:
                out[kind] = None
    return out


def _info_players(record: dict[str, Any]) -> list[dict[str, Any]]:
    info = record.get("info") or {}
    players = info.get("players") or []
    return players if isinstance(players, list) else []


def agent_client_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for p in _info_players(record):
        cid = p.get("clientID") or p.get("client_id")
        if isinstance(cid, str) and cid:
            ids.append(cid)
    if not ids:
        ids = ["AGENTRL1"]
    return ids


def replay_tick_dump(
    record_path: Path,
    *,
    every: int = 250,
    max_ticks: int | None = 80,
    timeout_s: int = 90,
    tick_dump: Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any] | None:
    """Native replay → sampled gold/tiles plus player-id roster.

    Default `max_ticks=80` is a roster pass (spawn + names). Pass
    `max_ticks=None` for a full-episode gold/tiles series (slow in debug).
    """
    binary = tick_dump or find_tick_dump()
    if binary is None:
        return None
    repo = repo or REPO
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        cmd = [
            str(binary),
            "--repo",
            str(repo),
            "--record",
            str(record_path),
            "--every",
            str(every),
            "--out",
            str(out),
        ]
        if max_ticks is not None:
            cmd.extend(["--max-ticks", str(max_ticks)])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            print(
                f"[analyze] tick_dump failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-400:]}",
                file=sys.stderr,
            )
            return None
        return _json_load(out)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"[analyze] tick_dump skipped: {e}", file=sys.stderr)
        return None
    finally:
        out.unlink(missing_ok=True)


def roster_from_record(record: dict[str, Any]) -> dict[str, PlayerInfo]:
    """Prefer `info.roster` / `info.players[].playerID` written by native save_record."""
    info = record.get("info") or {}
    by_id: dict[str, PlayerInfo] = {}
    raw = info.get("roster")
    if isinstance(raw, list):
        for p in raw:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or p.get("playerID") or "")
            if not pid:
                continue
            by_id[pid] = PlayerInfo(
                player_id=pid,
                client_id=str(p.get("clientID") or p.get("client_id") or ""),
                name=str(p.get("name") or p.get("username") or ""),
                player_type=str(p.get("playerType") or p.get("player_type") or ""),
                team=p.get("team"),
            )
    for p in _info_players(record):
        pid = p.get("playerID") or p.get("player_id") or p.get("id")
        cid = str(p.get("clientID") or p.get("client_id") or "")
        if not pid:
            continue
        pid = str(pid)
        existing = by_id.get(pid)
        if existing:
            if cid and not existing.client_id:
                existing.client_id = cid
            if p.get("username") and not existing.name:
                existing.name = str(p.get("username"))
        else:
            by_id[pid] = PlayerInfo(
                player_id=pid,
                client_id=cid,
                name=str(p.get("username") or ""),
                player_type="Human",
            )
    return by_id


def roster_from_tick_dump(
    dump: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, PlayerInfo]:
    """Map player-id string → PlayerInfo using the first snapshot with players."""
    by_id: dict[str, PlayerInfo] = {}
    for snap in dump.get("ticks") or []:
        players = snap.get("players") or []
        if not players:
            continue
        for p in players:
            pid = str(p.get("id") or "")
            identity = str(p.get("identity") or "")
            client_id = ""
            if identity.startswith("player:"):
                client_id = identity.split(":", 1)[1]
            info = PlayerInfo(
                player_id=pid,
                client_id=client_id,
                name=str(p.get("name") or ""),
                player_type=str(p.get("playerType") or p.get("player_type") or ""),
                team=p.get("team"),
                identity=identity,
            )
            if pid:
                by_id[pid] = info
        break
    # Fill names from GameRecord when tick dump missed a client_id.
    names = {
        (p.get("clientID") or p.get("client_id")): p.get("username") or ""
        for p in _info_players(record)
    }
    for info in by_id.values():
        if info.client_id and not info.name:
            info.name = str(names.get(info.client_id) or "")
    return by_id


def _classify_target(
    pid: str | None,
    *,
    partner_ids: set[str],
    known_ids: set[str],
) -> str:
    if not pid:
        return "none"
    if pid in partner_ids:
        return "partner"
    if pid in known_ids:
        return "npc"
    return "unknown"


def _collapse_timeline(events: list[Event]) -> list[Event]:
    if not events:
        return []
    out: list[Event] = []
    for ev in events:
        if (
            out
            and out[-1].kind == ev.kind
            and out[-1].client_id == ev.client_id
            and out[-1].counterparty == ev.counterparty
            and out[-1].kind not in ("spawn",)
            and ev.tick - (out[-1].tick) <= 200
        ):
            prev = out[-1]
            prev.count += 1
            if ev.amount is not None:
                prev.amount = (prev.amount or 0) + ev.amount
            continue
        out.append(
            Event(
                tick=ev.tick,
                client_id=ev.client_id,
                kind=ev.kind,
                counterparty=ev.counterparty,
                amount=ev.amount,
                extra=ev.extra,
                count=1,
            )
        )
    return out


def _simulate_pacts(
    events: list[Event],
    *,
    agent_player_ids: set[str],
) -> tuple[list[PactEvent], list[tuple[int, int | None]]]:
    """Cross-request auto-accepts (OpenFront AllianceRequestExecution.init)."""
    pending: dict[tuple[str, str], int] = {}
    allied: set[frozenset[str]] = set()
    pacts: list[PactEvent] = []
    windows: list[tuple[int, int | None]] = []
    open_at: dict[frozenset[str], int] = {}

    def pair_key(a: str, b: str) -> frozenset[str]:
        return frozenset((a, b))

    for ev in events:
        src = ev.extra  # stashed player-id of sender when known
        dst = ev.counterparty
        if ev.kind == "alliance_request" and src and dst:
            reverse = (dst, src)
            if reverse in pending:
                pending.pop(reverse, None)
                key = pair_key(src, dst)
                if key not in allied:
                    allied.add(key)
                    pacts.append(PactEvent(ev.tick, "formed", src, dst))
                    if src in agent_player_ids and dst in agent_player_ids:
                        open_at[key] = ev.tick
            else:
                pending[(src, dst)] = ev.tick
        elif ev.kind == "alliance_reject" and src and dst:
            pending.pop((dst, src), None)
        elif ev.kind == "break_alliance" and src and dst:
            pending.pop((src, dst), None)
            pending.pop((dst, src), None)
            key = pair_key(src, dst)
            if key in allied:
                allied.discard(key)
                pacts.append(PactEvent(ev.tick, "broken", src, dst))
                start = open_at.pop(key, None)
                if start is not None:
                    windows.append((start, ev.tick))
    for key, start in open_at.items():
        windows.append((start, None))
    return pacts, windows


def decode_thinking(thinking: dict[str, Any], *, limit: int = 24) -> list[str]:
    actions = list(thinking.get("a") or thinking.get("actions") or [])
    rows = list(thinking.get("s") or [])
    lines: list[str] = []
    for row in rows[:limit]:
        if not isinstance(row, list) or len(row) < 3:
            continue
        tick = row[0]
        a_idx = row[1]
        value = row[2]
        name = actions[a_idx] if isinstance(a_idx, int) and 0 <= a_idx < len(actions) else str(a_idx)
        desc = ""
        if row and isinstance(row[-1], str):
            desc = row[-1]
        extra = f" {desc}" if desc else ""
        lines.append(f"t={tick:>6}  {name:<18} V={value}{extra}")
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more thinking steps")
    return lines


def analyze_record(
    record: dict[str, Any],
    *,
    sidecars: dict[str, Any] | None = None,
    roster: dict[str, PlayerInfo] | None = None,
    tick_series: list[dict[str, Any]] | None = None,
) -> Analysis:
    sidecars = sidecars or {}
    info = record.get("info") or {}
    cfg = info.get("config") or {}
    clients = agent_client_ids(record)
    names = {
        (p.get("clientID") or p.get("client_id")): str(p.get("username") or "")
        for p in _info_players(record)
    }
    roster = roster or {}
    client_to_pid = {
        info.client_id: pid for pid, info in roster.items() if info.client_id
    }
    partner_pids = {client_to_pid[c] for c in clients if c in client_to_pid}
    known_ids = set(roster)

    stats: dict[str, AgentStats] = {
        cid: AgentStats(client_id=cid, name=str(names.get(cid) or cid), player_id=client_to_pid.get(cid, ""))
        for cid in clients
    }
    raw_events: list[Event] = []

    for turn in record.get("turns") or []:
        tick = int(turn.get("turnNumber") or turn.get("turn_number") or 0)
        for intent in turn.get("intents") or []:
            if not isinstance(intent, dict):
                continue
            cid = str(intent.get("clientID") or intent.get("client_id") or "")
            kind = normalize_intent_type(str(intent.get("type") or "unknown"))
            other = intent_counterparty(intent)
            amount = None
            if kind == "donate_gold" and intent.get("gold") is not None:
                amount = int(intent["gold"])
            elif kind == "donate_troops" and intent.get("troops") is not None:
                amount = int(intent["troops"])
            elif kind == "attack" and intent.get("troops") is not None:
                amount = int(intent["troops"])
            elif kind == "boat" and intent.get("troops") is not None:
                amount = int(intent["troops"])
            if kind == "spawn":
                extra_tile = intent.get("tile")
                if cid in stats:
                    stats[cid].spawn_tick = tick
                    if extra_tile is not None:
                        stats[cid].spawn_tile = int(extra_tile)
            if cid in stats:
                st = stats[cid]
                st.actions[kind] = st.actions.get(kind, 0) + 1
                bucket = _classify_target(
                    other, partner_ids=partner_pids - {st.player_id}, known_ids=known_ids
                )
                if kind == "alliance_request":
                    if bucket == "partner":
                        st.alliance_to_partner += 1
                    else:
                        st.alliance_to_npc += 1
                elif kind == "alliance_reject":
                    st.reject += 1
                elif kind == "break_alliance":
                    st.break_alliance += 1
                elif kind == "donate_gold":
                    if bucket == "partner":
                        st.donate_gold_partner += amount or 0
                        st.donate_gold_partner_n += 1
                    else:
                        st.donate_gold_npc += amount or 0
                        st.donate_gold_npc_n += 1
                elif kind == "donate_troops":
                    if bucket == "partner":
                        st.donate_troops_partner += amount or 0
                        st.donate_troops_partner_n += 1
                    else:
                        st.donate_troops_npc += amount or 0
                        st.donate_troops_npc_n += 1
                elif kind == "attack":
                    st.attacks += 1
                    if other is None:
                        st.terra_attacks += 1
                elif kind == "boat":
                    st.boats += 1
                elif kind == "build":
                    st.builds += 1
                elif kind == "embargo":
                    st.embargoes += 1
            if kind in NOTABLE_TYPES or (kind == "attack" and other is not None):
                extra_note = ""
                if kind == "spawn" and intent.get("tile") is not None:
                    extra_note = f"tile={intent['tile']}"
                elif kind == "build":
                    extra_note = str(intent.get("unit") or intent.get("unitType") or "")
                raw_events.append(
                    Event(
                        tick=tick,
                        client_id=cid,
                        kind=kind,
                        counterparty=other,
                        amount=amount,
                        extra=extra_note,
                    )
                )

    # Pact sim needs sender player-id in Event.extra; restore after spawn notes.
    pact_events: list[Event] = []
    for ev in raw_events:
        if ev.kind in ("alliance_request", "alliance_reject", "break_alliance"):
            sender_pid = client_to_pid.get(ev.client_id, ev.extra if ev.extra in known_ids else "")
            pact_events.append(
                Event(
                    tick=ev.tick,
                    client_id=ev.client_id,
                    kind=ev.kind,
                    counterparty=ev.counterparty,
                    extra=sender_pid,
                )
            )
    pacts, windows = _simulate_pacts(pact_events, agent_player_ids=partner_pids)

    meta = sidecars.get("meta") or {}
    debug = sidecars.get("debug") or {}
    thinking = sidecars.get("thinking")
    outcome = (
        debug.get("outcome")
        or ("timeout" if meta.get("timed_out") else None)
        or ("win" if meta.get("won") else None)
        or ("win" if info.get("winner") else "unknown")
    )
    notes: list[str] = []
    if not roster:
        notes.append(
            "No native replay roster: recipients stay as opaque player-ids. "
            "Pass --replay (tick_dump) to name nations and detect partner gifts."
        )
    elif len(partner_pids) < 2 and len(clients) >= 2:
        notes.append("Could not map both agent clientIDs to player-ids; partner gifts may be undercounted.")

    thinking_preview: list[str] = []
    if isinstance(thinking, dict):
        thinking_preview = decode_thinking(thinking)

    return Analysis(
        game_id=str(info.get("gameID") or info.get("game_id") or ""),
        map=str(cfg.get("gameMap") or ""),
        mode=str(cfg.get("gameMode") or ""),
        teams=str(cfg.get("playerTeams") or ""),
        difficulty=str(cfg.get("difficulty") or ""),
        bots=cfg.get("bots"),
        nations=cfg.get("nations"),
        num_turns=int(info.get("num_turns") or info.get("numTurns") or 0),
        duration_s=info.get("duration") if isinstance(info.get("duration"), int) else None,
        outcome=str(outcome),
        won=meta.get("won") if isinstance(meta.get("won"), bool) else None,
        timed_out=meta.get("timed_out") if isinstance(meta.get("timed_out"), bool) else None,
        engine=meta.get("engine"),
        agents=list(stats.values()),
        roster=list(roster.values()),
        pacts=pacts,
        partner_allied=bool(windows),
        partner_allied_ticks=windows,
        timeline=_collapse_timeline(raw_events),
        tick_series=tick_series or [],
        thinking=thinking if isinstance(thinking, dict) else None,
        thinking_preview=thinking_preview,
        notes=notes,
    )


def _name_of(pid: str | None, roster: dict[str, PlayerInfo], stats: dict[str, AgentStats]) -> str:
    if not pid:
        return "terra"
    if pid in roster:
        return roster[pid].short()
    for st in stats.values():
        if st.player_id == pid or st.client_id == pid:
            return st.name
    return pid


def format_analysis(a: Analysis) -> str:
    roster = {p.player_id: p for p in a.roster}
    stats_by_cid = {s.client_id: s for s in a.agents}
    lines: list[str] = []
    lines.append(f"# Game {a.game_id}")
    bits = [
        f"map={a.map}",
        f"mode={a.mode}",
        f"teams={a.teams}",
        f"difficulty={a.difficulty}",
        f"bots={a.bots}",
        f"nations={a.nations}",
        f"turns={a.num_turns}",
    ]
    if a.duration_s is not None:
        bits.append(f"wall={a.duration_s}s")
    if a.engine:
        bits.append(f"engine={a.engine}")
    lines.append("  " + "  ".join(bits))
    outcome = a.outcome
    if a.timed_out:
        outcome = "timeout"
    elif a.won:
        outcome = "win"
    lines.append(f"  outcome={outcome}")
    if a.notes:
        for n in a.notes:
            lines.append(f"  note: {n}")
    lines.append("")
    lines.append("## Roster")
    if a.roster:
        for p in a.roster:
            team = f" team={p.team}" if p.team else ""
            cid = f" {p.client_id}" if p.client_id else ""
            lines.append(f"  {p.player_id:>10}  {p.player_type:<8}  {p.short():<16}{cid}{team}")
    else:
        for s in a.agents:
            lines.append(f"  {s.client_id}  {s.name}")
    lines.append("")
    lines.append("## Partner pact")
    if len(a.agents) < 2:
        lines.append("  single-agent game (no partner)")
    elif not roster:
        lines.append("  unknown without replay (need player-id map)")
    elif a.partner_allied:
        n = len(a.partner_allied_ticks)
        first = a.partner_allied_ticks[0][0]
        last_start, last_end = a.partner_allied_ticks[-1]
        last = "held to end" if last_end is None else f"broke at t={last_end}"
        lines.append(
            f"  YES - {n} form/break cycle(s); first cross-request t={first}; last {last}"
        )
        show = a.partner_allied_ticks[:6]
        for start, end in show:
            end_s = "end" if end is None else f"t={end}"
            lines.append(f"    t={start} -> {end_s}")
        if n > 6:
            lines.append(f"    ... {n - 6} more cycles")
    else:
        lines.append("  NEVER - agents never cross-requested each other")
        n_req = sum(s.alliance_to_partner for s in a.agents)
        n_npc = sum(s.alliance_to_npc for s in a.agents)
        lines.append(f"  partner requests={n_req}  npc requests={n_npc}")
    if a.pacts:
        partner_pids = {s.player_id for s in a.agents if s.player_id}
        npc_pacts = [
            p
            for p in a.pacts
            if not (p.a in partner_pids and p.b in partner_pids)
        ]
        formed = sum(1 for p in npc_pacts if p.kind == "formed")
        broken = sum(1 for p in npc_pacts if p.kind == "broken")
        lines.append(f"  npc pacts: formed={formed} broken={broken}")
    lines.append("")
    lines.append("## Per-agent")
    for s in a.agents:
        spawn = ""
        if s.spawn_tick is not None:
            spawn = f"  spawn t={s.spawn_tick} tile={s.spawn_tile}"
        lines.append(f"  {s.name} [{s.client_id}] pid={s.player_id or '?'}{spawn}")
        mix = ", ".join(f"{k}={v}" for k, v in sorted(s.actions.items(), key=lambda kv: -kv[1]))
        lines.append(f"    mix: {mix}")
        lines.append(
            f"    alliance: partner={s.alliance_to_partner} npc={s.alliance_to_npc}  "
            f"reject={s.reject} break={s.break_alliance}"
        )
        lines.append(
            f"    donate gold: partner {s.donate_gold_partner_n}x / {s.donate_gold_partner}   "
            f"npc {s.donate_gold_npc_n}x / {s.donate_gold_npc}"
        )
        lines.append(
            f"    donate troops: partner {s.donate_troops_partner_n}x / {s.donate_troops_partner}   "
            f"npc {s.donate_troops_npc_n}x / {s.donate_troops_npc}"
        )
        lines.append(
            f"    fight: attacks={s.attacks} (terra={s.terra_attacks}) boats={s.boats} "
            f"builds={s.builds} embargoes={s.embargoes}"
        )
    if a.tick_series:
        lines.append("")
        lines.append("## Sampled state (native replay)")
        lines.append("  tick   " + "  ".join(f"{s.name:>16}" for s in a.agents) + "   (tiles/troops/gold/alive)")
        agent_pids = {s.player_id for s in a.agents if s.player_id}
        series = a.tick_series
        if len(series) > 16:
            step = max(1, (len(series) - 1) // 12)
            series = series[::step]
            if series[-1] is not a.tick_series[-1]:
                series.append(a.tick_series[-1])
            lines.append(f"  (showing {len(series)}/{len(a.tick_series)} snapshots)")
        for snap in series:
            tick = snap.get("tick")
            by_id = {str(p.get("id")): p for p in snap.get("players") or []}
            parts = []
            for s in a.agents:
                p = by_id.get(s.player_id) or {}
                tiles = p.get("tiles", "?")
                troops = p.get("troops", "?")
                gold = p.get("gold", "?")
                alive = "Y" if p.get("alive") else "n"
                parts.append(f"{tiles}/{troops}/{gold}/{alive}")
            lines.append(f"  {tick:>5}  " + "  ".join(f"{x:>16}" for x in parts))
            # also one-line NPC remaining alive count
            npcs_alive = sum(
                1
                for pid, p in by_id.items()
                if pid not in agent_pids and p.get("alive")
            )
            lines[-1] += f"   npcs_alive={npcs_alive}"
    if a.thinking_preview:
        lines.append("")
        lines.append("## Policy thinking (compact sidecar)")
        o = (a.thinking or {}).get("o")
        n = (a.thinking or {}).get("n")
        lines.append(f"  decisions={n} outcome={o}")
        for row in a.thinking_preview:
            lines.append(f"  {row}")
    lines.append("")
    lines.append("## Timeline")
    for ev in a.timeline:
        who = stats_by_cid.get(ev.client_id)
        name = who.name if who else ev.client_id
        tgt = _name_of(ev.counterparty, roster, stats_by_cid) if ev.counterparty else ""
        amt = ""
        if ev.amount is not None:
            amt = f" {ev.amount}"
        rpt = f" x{ev.count}" if ev.count > 1 else ""
        extra = f" {ev.extra}" if ev.extra and ev.kind == "spawn" else ""
        if ev.kind == "build" and ev.extra:
            extra = f" {ev.extra}"
        arrow = f" -> {tgt}" if tgt and tgt != "terra" else ""
        lines.append(f"  t={ev.tick:>6}  {name:<8}  {ev.kind:<18}{amt}{rpt}{arrow}{extra}")
    lines.append("")
    return "\n".join(lines)


def analysis_to_json(a: Analysis) -> dict[str, Any]:
    payload = asdict(a)
    return payload


def latest_records(spool: Path, n: int) -> list[Path]:
    files = [
        p
        for p in spool.glob("*.json")
        if p.is_file()
        and not p.name.startswith("._")
        and not p.name.endswith(".meta.json")
        and not p.name.endswith(".debug.json")
        and not p.name.endswith(".thinking.json")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def run_one(
    record_path: Path,
    *,
    replay: bool,
    every: int,
    state: bool,
    text_out: Path | None,
    json_out: Path | None,
) -> Analysis:
    record = _json_load(record_path)
    sidecars = load_sidecars(record_path)
    roster = roster_from_record(record)
    series: list[dict[str, Any]] = []
    clients = agent_client_ids(record)
    mapped = {info.client_id for info in roster.values() if info.client_id}
    need_ids = [c for c in clients if c not in mapped]
    if replay and (need_ids or (state and not series)):
        max_ticks = None if state else 80
        timeout_s = 600 if state else 90
        dump = replay_tick_dump(
            record_path,
            every=every if state else 40,
            max_ticks=max_ticks,
            timeout_s=timeout_s,
        )
        if dump:
            dumped = roster_from_tick_dump(dump, record)
            if dumped:
                roster = dumped
            if state:
                series = list(dump.get("ticks") or [])
    analysis = analyze_record(
        record, sidecars=sidecars, roster=roster, tick_series=series
    )
    analysis.record_path = str(record_path)
    text = format_analysis(analysis)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    if text_out:
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(text)
        print(f"[analyze] wrote {text_out}", file=sys.stderr)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(analysis_to_json(analysis), indent=2))
        print(f"[analyze] wrote {json_out}", file=sys.stderr)
    return analysis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("record", nargs="*", type=Path, help="GameRecord JSON path(s)")
    ap.add_argument("--spool", type=Path, help="replay-spool directory")
    ap.add_argument("--latest", type=int, default=0, help="analyze N newest records in --spool")
    ap.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True,
                    help="native tick_dump roster pass when playerIDs are missing (default true)")
    ap.add_argument("--state", action="store_true",
                    help="full-episode tick_dump gold/tiles series (slow in debug builds)")
    ap.add_argument("--every", type=int, default=250, help="tick_dump sample stride with --state")
    ap.add_argument("--text-out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--out-dir", type=Path, help="write <game_id>.txt and .json here")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    paths.extend(args.record)
    if args.spool and args.latest:
        paths.extend(latest_records(args.spool, args.latest))
    if not paths:
        ap.error("pass a GameRecord path or --spool DIR --latest N")

    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        text_out = args.text_out
        json_out = args.json_out
        if args.out_dir:
            gid = path.stem
            text_out = args.out_dir / f"{gid}.txt"
            json_out = args.out_dir / f"{gid}.json"
        print("=" * 72)
        run_one(
            path,
            replay=args.replay,
            every=args.every,
            state=args.state,
            text_out=text_out,
            json_out=json_out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
