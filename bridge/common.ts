/**
 * Shared between bridge/env.ts (offline RL env) and bridge/play.ts (live
 * multiplayer agent client): observation payload construction from a Game.
 */
import * as zlib from "zlib";
import { Game, UnitType } from "../openfront/src/core/game/Game";

export const STRUCTURES = [
  UnitType.City,
  UnitType.Port,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.Factory,
];
export const LAUNCHABLE = [
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
];

export function buildObs(
  game: Game,
  clientID: string,
  winner: unknown,
): object {
  const numTiles = game.width() * game.height();
  const tiles = zlib
    .gzipSync(
      Buffer.from(
        game.tileStateBuffer().buffer,
        game.tileStateBuffer().byteOffset,
        numTiles * 2,
      ),
      { level: 1 },
    )
    .toString("base64");

  const agent = game.playerByClientID(clientID) ?? null;

  return {
    tick: game.ticks(),
    width: game.width(),
    height: game.height(),
    spawnPhase: game.inSpawnPhase(),
    winner,
    me: agent?.smallID() ?? -1,
    alive: agent?.isAlive() ?? false,
    tiles,
    entities: entities(game),
    legal: legality(game, clientID),
  };
}

export function terrainPayload(game: Game): object {
  const numTiles = game.width() * game.height();
  const buf = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) buf[ref] = game.terrainByte(ref);
  return {
    terrain: zlib.gzipSync(Buffer.from(buf), { level: 6 }).toString("base64"),
  };
}

function entities(game: Game): object {
  // allPlayers(): game.players() filters to living players, which would
  // drop the dead agent (and everyone eliminated) from placement math.
  const players = game.allPlayers().map((p) => ({
    id: p.smallID(),
    pid: p.id(), // persistent ID; intents reference players by this
    type: p.type(),
    troops: Math.round(p.troops()),
    gold: p.gold().toString(),
    tiles: p.numTilesOwned(),
    alive: p.isAlive(),
    traitor: p.isTraitor(),
    embargoes: p.getEmbargoes().map((e) => e.target.smallID()),
    reqsIn: p.incomingAllianceRequests().map((r) => r.requestor().smallID()),
    reqsOut: p.outgoingAllianceRequests().map((r) => r.recipient().smallID()),
  }));

  const alliances: number[][] = [];
  const seen = new Set<string>();
  for (const p of game.players()) {
    for (const a of p.alliances()) {
      const x = a.requestor().smallID();
      const y = a.recipient().smallID();
      const key = x < y ? `${x}:${y}` : `${y}:${x}`;
      if (!seen.has(key)) {
        seen.add(key);
        alliances.push([x, y, a.expiresAt()]);
      }
    }
  }

  const units = game.players().flatMap((p) =>
    p.units().map((u) => {
      const tt = u.targetTile();
      return {
        uid: u.id(),
        type: u.type(),
        owner: p.smallID(),
        x: game.x(u.tile()),
        y: game.y(u.tile()),
        tx: tt !== undefined ? game.x(tt) : null,
        ty: tt !== undefined ? game.y(tt) : null,
        samLock: u.targetedBySAM(),
        level: u.level(),
        constructing: u.isUnderConstruction(),
        troops: Math.round(u.troops()),
      };
    }),
  );

  const attacks = game.players().flatMap((p) =>
    p.outgoingAttacks().map((a) => ({
      aid: a.id(),
      from: p.smallID(),
      to: a.target().isPlayer() ? a.target().smallID() : 0,
      troops: Math.round(a.troops()),
      retreating: a.retreating(),
    })),
  );

  return { players, alliances, units, attacks };
}

/** Exact per-action legality from engine calls; Python builds masks. */
export function legality(game: Game, clientID: string): object {
  const agent = game.playerByClientID(clientID) ?? null;
  if (!agent || !agent.isAlive()) {
    return { spawn: game.inSpawnPhase(), actions: {} };
  }
  const others = game.players().filter((p) => p !== agent && p.isAlive());
  const gold = agent.gold();
  const buildable = [...STRUCTURES, ...LAUNCHABLE, UnitType.Warship].filter(
    (t) => gold >= game.unitInfo(t).cost(game, agent),
  );
  return {
    spawn: game.inSpawnPhase(),
    actions: {
      attackable: others
        .filter((p) => agent.sharesBorderWith(p) && !agent.isFriendly(p))
        .map((p) => p.smallID()),
      allianceRequestable: others
        .filter((p) => agent.canSendAllianceRequest(p))
        .map((p) => p.smallID()),
      allianceRejectable: agent
        .incomingAllianceRequests()
        .map((r) => r.requestor().smallID()),
      breakable: agent.alliances().map((a) => a.other(agent).smallID()),
      targetable: others
        .filter((p) => agent.canTarget(p))
        .map((p) => p.smallID()),
      donatableGold: others
        .filter((p) => agent.canDonateGold(p))
        .map((p) => p.smallID()),
      donatableTroops: others
        .filter((p) => agent.canDonateTroops(p))
        .map((p) => p.smallID()),
      embargoable: others
        .filter((p) => !agent.hasEmbargoAgainst(p))
        .map((p) => p.smallID()),
      buildableTypes: buildable,
      hasSilo: agent
        .units(UnitType.MissileSilo)
        .some((u) => !u.isUnderConstruction()),
      troops: Math.round(agent.troops()),
      gold: gold.toString(),
      attacks: agent.outgoingAttacks().map((a) => a.id()),
      boats: agent.units(UnitType.Transport).map((u) => u.id()),
      warships: agent.units(UnitType.Warship).map((u) => u.id()),
      upgradable: agent
        .units()
        .filter((u) => game.unitInfo(u.type()).upgradable)
        .map((u) => u.id()),
    },
  };
}
