/**
 * Replay archived human games (downloaded via scripts/fetch_games.py) through
 * the headless engine and dump snapshots in the same format as generate.ts.
 *
 * The engine is deterministic: feeding the recorded per-turn intents back
 * through the Executor regenerates the full state trajectory. Integrity is
 * verified against the state hashes embedded in the record (clients report
 * game.hash() every 10 ticks during live play); any mismatch marks the game
 * desynced and it is skipped.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx datagen/replay.ts \
 *     --records records --out data-human [--every 10] [--limit N]
 *
 * Output layout per game: data-human/<map>/<gameID>/ with terrain.bin,
 * states/t<tick>.{bin,json}.gz and meta.json (formatVersion 3, plus
 * source/gitCommit/hash-verification fields).
 *
 * --bc additionally writes bc.json.gz: per-snapshot legality for every living
 * human plus their intents in the following window normalized to the policy
 * action space, and final placements - the (state, action) supervision for
 * behavior cloning (rl/bc_data.py).
 *
 * formatVersion 2 adds spawn supervision: every human spawn intent is
 * recorded with a state snapshot taken at the moment the pick was made
 * (pre-execution), so BC can train the spawn-placement head on what the
 * player actually saw. Spawn-phase snapshots share the states/ dir (their
 * ticks precede all regular snapshots).
 *
 * formatVersion 3 (v6 action space) adds labels for upgrade_structure,
 * move_warship, cancel_boat, delete_unit, embargo_stop, target_player,
 * alliance_extension, Warship builds, targeted retreats (t = attacked
 * player, 0 = terra nullius) and the nuke arc flag (up).
 */
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as zlib from "zlib";
import { createHash } from "crypto";
import {
  Game,
  GameMapType,
  GameType,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../openfront/src/core/game/Game";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { Config } from "../openfront/src/core/configuration/Config";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { createRequire } from "module";
import { GameRecord, GameStartInfo } from "../openfront/src/core/Schemas";
import { decompressGameRecord, simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain, snapshotEntities } from "./common";

// The engine submodule gets checked out at whatever commit each game ran on
// (see replay_all.sh), so modules/APIs that only exist on newer commits must
// be loaded dynamically and skipped when absent.
const dynRequire = createRequire(__filename);

interface ReplayResult {
  ok: boolean;
  reason?: string;
  ticks: number;
  snapshots: number;
  hashesChecked: number;
}

const OUTCOME_SCHEMA_VERSION = 1;

interface OutcomeRankingEntry {
  identity: string;
  name: string;
  team: string | null;
  tiles: number;
  alive: boolean;
}

export interface GameOutcome {
  schemaVersion: number;
  gameId: string;
  winner: string | null;
  terminalTick: number | null;
  terminalReason: string | null;
  winnerLandShare: number | null;
  finalTick: number;
  landTilesWithoutFallout: number;
  finalRanking: OutcomeRankingEntry[];
}

function normalizeWinner(winner: unknown): string | null {
  if (!Array.isArray(winner) || winner.length < 2) return null;
  const kind = String(winner[0]);
  if (kind !== "player" && kind !== "team" && kind !== "nation") return null;
  return `${kind}:${String(winner[1])}`;
}

function playerIdentity(player: {
  clientID(): string | null;
  name(): string;
}): string {
  const clientID = player.clientID();
  return clientID === null ? `nation:${player.name()}` : `player:${clientID}`;
}

function terminalReason(game: Game, winner: string): string {
  const config = game.config().gameConfig();
  const players = game.players();
  if (
    config.rankedType === "1v1" &&
    players.filter((p) => p.type() === PlayerType.Human && !p.isDisconnected())
      .length === 1
  ) {
    return "one_v_one";
  }
  const denominator = Math.max(
    1,
    game.numLandTiles() - game.numTilesWithFallout(),
  );
  const winnerTiles = winner.startsWith("team:")
    ? players
        .filter((p) => p.team() === winner.slice("team:".length))
        .reduce((sum, p) => sum + p.numTilesOwned(), 0)
    : (players.find((p) => playerIdentity(p) === winner)?.numTilesOwned() ?? 0);
  if (
    (winnerTiles / denominator) * 100 >
    game.config().percentageTilesOwnedToWin()
  ) {
    return "land_share";
  }
  const elapsedSeconds = game.elapsedGameSeconds();
  if (
    config.maxTimerValue != null &&
    elapsedSeconds >= config.maxTimerValue * 60
  ) {
    return "max_timer";
  }
  return "hard_time_limit";
}

function finalWinnerLandShare(
  game: Game,
  winner: string | null,
): number | null {
  if (winner === null) return null;
  const denominator = Math.max(
    1,
    game.numLandTiles() - game.numTilesWithFallout(),
  );
  if (winner.startsWith("team:")) {
    const team = winner.slice("team:".length);
    return (
      game
        .players()
        .filter((p) => p.team() === team)
        .reduce((sum, p) => sum + p.numTilesOwned(), 0) / denominator
    );
  }
  const player = game.players().find((p) => playerIdentity(p) === winner);
  return player === undefined ? null : player.numTilesOwned() / denominator;
}

/** Replay every recorded intent and capture outcome metrics without treating
 * archived hash mismatches as terminal. */
export async function replayOutcome(record: GameRecord): Promise<GameOutcome> {
  const info: GameStartInfo = record.info;
  const gameConfig = info.config;
  const mapType = gameConfig.gameMap as GameMapType;
  if (!Object.values(GameMapType).includes(mapType)) {
    throw new Error(`unknown map ${gameConfig.gameMap}`);
  }
  const config = new Config(gameConfig, null, false);
  const terrain = await loadFreshTerrain(mapType, gameConfig.gameMapSize);
  const random = new PseudoRandom(simpleHash(info.gameID));
  const humans = info.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
      ),
  );
  const nations = createNationsForGame(
    info,
    terrain.nations,
    terrain.additionalNations,
    humans.length,
    random,
  );
  const game: Game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  const executor = new Executor(game, info.gameID, undefined);
  if (gameConfig.gameType !== GameType.Singleplayer) {
    game.addExecution(new SpawnTimerExecution());
  }
  if (config.spawnNations()) {
    game.addExecution(...executor.nationExecutions());
  }
  if (config.isRandomSpawn()) {
    game.addExecution(...executor.spawnPlayers());
  }
  if (config.bots() > 0) {
    game.addExecution(...executor.spawnTribes(config.bots()));
  }
  game.addExecution(new WinCheckExecution());
  const cfgAny = config as unknown as {
    doomsdayClockConfig?: () => { enabled: boolean };
  };
  if (cfgAny.doomsdayClockConfig?.().enabled) {
    const { DoomsdayClockExecution } = dynRequire(
      "../openfront/src/core/execution/DoomsdayClockExecution",
    );
    game.addExecution(new DoomsdayClockExecution());
  }
  if (!config.isUnitDisabled(UnitType.Factory)) {
    game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
  }

  let winner: string | null = null;
  let terminalTick: number | null = null;
  let reason: string | null = null;
  for (const turn of record.turns) {
    game.addExecution(...executor.createExecs(turn));
    const updates = game.executeNextTick();
    if (winner === null) {
      const winUpdates = updates[GameUpdateType.Win] as
        | { winner: unknown }[]
        | undefined;
      if (winUpdates && winUpdates.length > 0) {
        winner = normalizeWinner(winUpdates[0].winner);
        if (winner !== null) {
          terminalTick = game.ticks();
          reason = terminalReason(game, winner);
        }
      }
    }
  }

  const finalRanking: OutcomeRankingEntry[] = game
    .allPlayers()
    .map((player) => ({
      identity: playerIdentity(player),
      name: player.name(),
      team: player.team(),
      tiles: player.numTilesOwned(),
      alive: player.isAlive(),
    }))
    .sort(
      (a, b) =>
        b.tiles - a.tiles ||
        (a.identity < b.identity ? -1 : a.identity > b.identity ? 1 : 0),
    );
  return {
    schemaVersion: OUTCOME_SCHEMA_VERSION,
    gameId: info.gameID,
    winner,
    terminalTick,
    terminalReason: reason,
    winnerLandShare: finalWinnerLandShare(game, winner),
    finalTick: game.ticks(),
    landTilesWithoutFallout: Math.max(
      1,
      game.numLandTiles() - game.numTilesWithFallout(),
    ),
    finalRanking,
  };
}

/** A human intent normalized to the policy's action space (rl/obs.py
 * ACTIONS). Player references become smallIDs; tiles become x/y. Intents
 * outside the modeled surface (emoji, quick chat, lobby admin) are
 * dropped. Quantity stays a raw absolute amount; the scalar fraction
 * (amt/available) is computed in Python where the actor's troops/gold at
 * the window start are known. Unit-targeted intents (upgrade_structure,
 * cancel_boat, delete_unit) resolve the unit id to its pre-turn x/y so
 * the tile head can be supervised on the unit's location. */
interface BCLabel {
  a: string;
  t?: number; // target smallID
  x?: number;
  y?: number;
  unit?: string;
  amt?: number | null;
  up?: boolean; // nuke arc (rocketDirectionUp); absent for MIRV
}

const NUKE_UNITS = new Set(["Atom Bomb", "Hydrogen Bomb", "MIRV"]);
const STRUCTURE_UNITS = new Set([
  "City",
  "Port",
  "Defense Post",
  "Missile Silo",
  "SAM Launcher",
  "Factory",
  "Warship",
]);

function normalizeIntent(
  game: Game,
  intent: Record<string, unknown>,
): BCLabel | null {
  const smallID = (pid: unknown): number | null => {
    try {
      return game.player(pid as never).smallID();
    } catch {
      return null;
    }
  };
  const xy = (tile: unknown): { x: number; y: number } => ({
    x: game.x(tile as never),
    y: game.y(tile as never),
  });

  // Unit-targeted intents carry a unit id; supervise the tile head with the
  // unit's pre-turn position. Returns null when the unit is already gone.
  const unitXY = (unitId: unknown): { x: number; y: number } | null => {
    try {
      const u = (
        game as unknown as {
          unit(id: number): { tile(): number } | undefined;
        }
      ).unit(unitId as number);
      return u ? xy(u.tile()) : null;
    } catch {
      return null;
    }
  };

  switch (intent.type) {
    case "attack": {
      if (intent.targetID === null) {
        return { a: "expand", amt: intent.troops as number | null };
      }
      const t = smallID(intent.targetID);
      return t === null ? null : { a: "attack", t, amt: intent.troops as number | null };
    }
    case "boat":
      return { a: "boat", ...xy(intent.dst), amt: intent.troops as number };
    case "build_unit": {
      const unit = intent.unit as string;
      if (NUKE_UNITS.has(unit)) {
        const label: BCLabel = { a: "launch_nuke", unit, ...xy(intent.tile) };
        // rocketDirectionUp flips the parabolic flight arc (SAM evasion).
        // Engine default is up when absent; MIRV ignores the flag.
        if (unit !== "MIRV") {
          label.up = (intent.rocketDirectionUp as boolean | undefined) ?? true;
        }
        return label;
      }
      if (STRUCTURE_UNITS.has(unit)) {
        return { a: "build", unit, ...xy(intent.tile) };
      }
      return null;
    }
    case "upgrade_structure": {
      const pos = unitXY(intent.unitId);
      return pos === null
        ? null
        : { a: "upgrade_structure", unit: intent.unit as string, ...pos };
    }
    case "move_warship":
      // Moves every selected warship to one destination; the policy's
      // region->water-tile scheme only needs the destination.
      return { a: "move_warship", ...xy(intent.tile) };
    case "cancel_boat": {
      const pos = unitXY(intent.unitID);
      return pos === null ? null : { a: "cancel_boat", ...pos };
    }
    case "delete_unit": {
      const pos = unitXY(intent.unitId);
      return pos === null ? null : { a: "delete_unit", ...pos };
    }
    case "allianceRequest": {
      const t = smallID(intent.recipient);
      return t === null ? null : { a: "alliance_request", t };
    }
    case "allianceReject": {
      const t = smallID(intent.requestor);
      return t === null ? null : { a: "alliance_reject", t };
    }
    case "breakAlliance": {
      const t = smallID(intent.recipient);
      return t === null ? null : { a: "break_alliance", t };
    }
    case "donate_gold": {
      const t = smallID(intent.recipient);
      return t === null
        ? null
        : { a: "donate_gold", t, amt: intent.gold as number | null };
    }
    case "donate_troops": {
      const t = smallID(intent.recipient);
      return t === null
        ? null
        : { a: "donate_troops", t, amt: intent.troops as number | null };
    }
    case "embargo": {
      const t = smallID(intent.targetID);
      if (t === null) return null;
      return intent.action === "stop"
        ? { a: "embargo_stop", t }
        : { a: "embargo", t };
    }
    case "targetPlayer": {
      const t = smallID(intent.target);
      return t === null ? null : { a: "target_player", t };
    }
    case "allianceExtension": {
      const t = smallID(intent.recipient);
      return t === null ? null : { a: "alliance_extension", t };
    }
    case "cancel_attack": {
      // Targeted retreat: resolve the attack id to the player being
      // attacked (t=0 for terra-nullius expands, matching the entities
      // attacks[].to convention). t is omitted when the attack already
      // resolved by the time the intent arrived.
      const actor = game.playerByClientID(
        intent.clientID as string,
      );
      const attack = actor
        ?.outgoingAttacks()
        .find((a) => a.id() === intent.attackID);
      if (attack === undefined) return { a: "retreat" };
      const target = attack.target();
      return {
        a: "retreat",
        t: target.isPlayer() ? target.smallID() : 0,
      };
    }
    default:
      return null;
  }
}

/** Placement percentile per human: winner is 1.0; survivors rank above the
 * killed (by final tiles); the killed rank by how long they lasted. */
function placements(
  game: Game,
  record: GameRecord,
): Record<string, { smallID: number; placement: number; winner: boolean }> {
  const info = record.info as GameStartInfo & {
    winner?: unknown[];
    players: { clientID: string; stats?: { killedAt?: string } }[];
  };
  const winnerClient =
    Array.isArray(info.winner) && info.winner[0] === "player"
      ? String(info.winner[1])
      : null;

  const rows = info.players.flatMap((p) => {
    const player = game.playerByClientID(p.clientID);
    if (!player) return [];
    const killedAt = p.stats?.killedAt ? Number(p.stats.killedAt) : null;
    return [
      {
        clientID: p.clientID,
        smallID: player.smallID(),
        winner: p.clientID === winnerClient,
        alive: killedAt === null,
        killedAt: killedAt ?? Infinity,
        tiles: player.numTilesOwned(),
      },
    ];
  });
  rows.sort((a, b) => {
    if (a.winner !== b.winner) return a.winner ? -1 : 1;
    if (a.alive !== b.alive) return a.alive ? -1 : 1;
    if (a.alive) return b.tiles - a.tiles;
    return b.killedAt - a.killedAt;
  });
  const out: Record<
    string,
    { smallID: number; placement: number; winner: boolean }
  > = {};
  const n = Math.max(1, rows.length - 1);
  rows.forEach((r, i) => {
    out[r.clientID] = {
      smallID: r.smallID,
      placement: 1 - i / n, // 1.0 best, 0.0 last
      winner: r.winner,
    };
  });
  return out;
}

export async function replayGame(
  record: GameRecord,
  outDir: string,
  snapshotEvery: number,
  bc: boolean,
  writeStates: boolean,
): Promise<ReplayResult> {
  const legality = bc
    ? (await import("../bridge/common")).legality
    : (null as never);
  const info: GameStartInfo = record.info;
  const gameConfig = info.config;

  const mapType = gameConfig.gameMap as GameMapType;
  if (!Object.values(GameMapType).includes(mapType)) {
    return {
      ok: false,
      reason: `unknown map ${gameConfig.gameMap}`,
      ticks: 0,
      snapshots: 0,
      hashesChecked: 0,
    };
  }

  const config = new Config(gameConfig, null, false);
  const terrain = await loadFreshTerrain(mapType, gameConfig.gameMapSize);
  const random = new PseudoRandom(simpleHash(info.gameID));

  // Mirrors createGameRunner(): humans first (consuming random.nextID() in
  // player order), then nations, so IDs match the live game exactly.
  const humans = info.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
      ),
  );
  const nations = createNationsForGame(
    info,
    terrain.nations,
    terrain.additionalNations,
    humans.length,
    random,
  );

  const game: Game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  const executor = new Executor(game, info.gameID, undefined);

  // bridge legality (bordersNeutralLand) calls game.isImpassable(), which
  // only exists on newer engine commits. Older engines have no impassable
  // terrain at all, so a constant-false polyfill is exact.
  const gAny = game as unknown as Record<string, unknown>;
  if (typeof gAny.isImpassable !== "function") {
    gAny.isImpassable = () => false;
  }

  // Mirrors GameRunner.init().
  if (gameConfig.gameType !== GameType.Singleplayer) {
    game.addExecution(new SpawnTimerExecution());
  }
  if (config.spawnNations()) {
    game.addExecution(...executor.nationExecutions());
  }
  if (config.isRandomSpawn()) {
    game.addExecution(...executor.spawnPlayers());
  }
  if (config.bots() > 0) {
    game.addExecution(...executor.spawnTribes(config.bots()));
  }
  game.addExecution(new WinCheckExecution());
  // Doomsday clock only exists on newer engine commits; mirror the init the
  // live game actually ran.
  const cfgAny = config as unknown as {
    doomsdayClockConfig?: () => { enabled: boolean };
  };
  if (cfgAny.doomsdayClockConfig?.().enabled) {
    const { DoomsdayClockExecution } = dynRequire(
      "../openfront/src/core/execution/DoomsdayClockExecution",
    );
    game.addExecution(new DoomsdayClockExecution());
  }
  if (!config.isUnitDisabled(UnitType.Factory)) {
    game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
  }

  fs.mkdirSync(outDir, { recursive: true });
  const w = game.width();
  const h = game.height();
  const numTiles = w * h;

  const terrainBuf = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) {
    terrainBuf[ref] = game.terrainByte(ref);
  }
  fs.writeFileSync(path.join(outDir, "terrain.bin"), terrainBuf);

  const statesDir = path.join(outDir, "states");
  fs.mkdirSync(statesDir, { recursive: true });
  const snapshots: { tick: number }[] = [];
  let winner: unknown = null;
  let hashesChecked = 0;

  // BC: normalized human intents by tick, and per-snapshot legality for every
  // living human. Assembled into bc.json.gz at the end (labels for snapshot T
  // are the intents issued in [T, T+snapshotEvery)).
  const labelsByTick = new Map<number, Record<string, BCLabel[]>>();
  const bcSteps: {
    tick: number;
    legal: Record<string, { me: number; legal: object }>;
  }[] = [];
  // Spawn supervision (formatVersion 2): human spawn picks with a snapshot
  // of the state they were looking at when they picked. smallIDs resolve
  // AFTER the turn executes - the player object is created by
  // SpawnExecution, so it does not exist when the intent is first seen.
  const spawnSteps = new Map<
    number,
    Record<string, { x: number; y: number; me: number }>
  >();
  let pendingSpawns: { cid: string; x: number; y: number; tick: number }[] = [];
  const spawnTicksDumped = new Set<number>();
  const humanClientIDs = new Set(info.players.map((p) => p.clientID));

  const writeSnapshot = (tick: number) => {
    const stem = `t${String(tick).padStart(6, "0")}`;
    const raw = Buffer.from(
      game.tileStateBuffer().buffer,
      game.tileStateBuffer().byteOffset,
      numTiles * 2,
    );
    fs.writeFileSync(
      path.join(statesDir, `${stem}.bin.gz`),
      zlib.gzipSync(raw, { level: 6 }),
    );
    fs.writeFileSync(
      path.join(statesDir, `${stem}.json.gz`),
      zlib.gzipSync(JSON.stringify(snapshotEntities(game)), { level: 6 }),
    );
  };

  for (const turn of record.turns) {
    if (bc) {
      // Normalize against the pre-turn state: the intent was decided by the
      // player looking at this state, and targets may die during the tick.
      for (const si of turn.intents) {
        const cid = (si as { clientID: string }).clientID;
        if (!humanClientIDs.has(cid)) continue;
        const intent = si as Record<string, unknown>;
        if (intent.type === "spawn" && game.inSpawnPhase()) {
          const tick = game.ticks();
          pendingSpawns.push({
            cid,
            x: game.x(intent.tile as never),
            y: game.y(intent.tile as never),
            tick,
          });
          if (!spawnTicksDumped.has(tick)) {
            spawnTicksDumped.add(tick);
            writeSnapshot(tick);
          }
          continue;
        }
        const label = normalizeIntent(game, intent);
        if (label === null) continue;
        const byClient = labelsByTick.get(game.ticks()) ?? {};
        (byClient[cid] ??= []).push(label);
        labelsByTick.set(game.ticks(), byClient);
      }
    }
    game.addExecution(...executor.createExecs(turn));
    const updates = game.executeNextTick();

    if (bc && pendingSpawns.length) {
      const unresolved: typeof pendingSpawns = [];
      for (const ps of pendingSpawns) {
        const player = game.playerByClientID(ps.cid);
        if (!player) {
          unresolved.push(ps);
          continue;
        }
        const byClient = spawnSteps.get(ps.tick) ?? {};
        byClient[ps.cid] = { x: ps.x, y: ps.y, me: player.smallID() };
        spawnSteps.set(ps.tick, byClient);
      }
      pendingSpawns = unresolved;
    }

    // Verify state hashes where the record has them. The engine emits a
    // Hash update every 10 ticks; live clients reported the same value and
    // the server stored it on the matching turn.
    const hashUpdates = updates[GameUpdateType.Hash] as
      | { tick: number; hash: number }[]
      | undefined;
    for (const hu of hashUpdates ?? []) {
      const archived = record.turns[hu.tick]?.hash;
      if (archived === null || archived === undefined) continue;
      hashesChecked++;
      if (archived !== hu.hash) {
        return {
          ok: false,
          reason: `desync at tick ${hu.tick} (ours ${hu.hash} != archived ${archived})`,
          ticks: game.ticks(),
          snapshots: snapshots.length,
          hashesChecked,
        };
      }
    }

    const winUpdates = updates[GameUpdateType.Win];
    if (winUpdates && winUpdates.length > 0) {
      winner = (winUpdates[0] as { winner: unknown }).winner;
    }

    if (!game.inSpawnPhase() && game.ticks() % snapshotEvery === 0) {
      if (writeStates) {
        writeSnapshot(game.ticks());
      }
      snapshots.push({ tick: game.ticks() });

      if (bc) {
        const legal: Record<string, { me: number; legal: object }> = {};
        for (const p of info.players) {
          const player = game.playerByClientID(p.clientID);
          if (!player || !player.isAlive()) continue;
          legal[p.clientID] = {
            me: player.smallID(),
            legal: legality(game, p.clientID),
          };
        }
        bcSteps.push({ tick: game.ticks(), legal });
      }
    }
  }

  if (bc) {
    const steps = bcSteps.map((s) => {
      const labels: Record<string, BCLabel[]> = {};
      for (let t = s.tick; t < s.tick + snapshotEvery; t++) {
        for (const [cid, ls] of Object.entries(labelsByTick.get(t) ?? {})) {
          (labels[cid] ??= []).push(...ls);
        }
      }
      return { ...s, labels };
    });
    const bcDoc = {
      formatVersion: 3,
      snapshotEvery,
      placements: placements(game, record),
      steps,
      spawn_steps: [...spawnSteps.entries()].map(([tick, labels]) => ({
        tick,
        labels,
      })),
    };
    fs.writeFileSync(
      path.join(outDir, "bc.json.gz"),
      zlib.gzipSync(JSON.stringify(bcDoc), { level: 6 }),
    );
  }

  const meta = {
    formatVersion: 3,
    source: "human-replay",
    gitCommit: record.gitCommit,
    gameID: info.gameID,
    map: mapType,
    mapSize: gameConfig.gameMapSize,
    gameMode: gameConfig.gameMode,
    numHumans: humans.length,
    width: w,
    height: h,
    snapshotEvery,
    finalTick: game.ticks(),
    winner: winner ?? (record.info as { winner?: unknown }).winner ?? null,
    hashesChecked,
    dtypeStates: "uint16-le",
    dtypeTerrain: "uint8",
    snapshots,
  };
  fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta));
  return {
    ok: true,
    ticks: game.ticks(),
    snapshots: snapshots.length,
    hashesChecked,
  };
}

// Whether an existing bc.json.gz is already at the current formatVersion, so
// --rebc reruns (e.g. after a crash or restart) skip games instead of
// re-replaying the whole dataset. The field is first in the serialized doc,
// so peeking at the head of the gunzipped stream is enough.
function bcIsCurrent(outDir: string): boolean {
  try {
    const buf = zlib.gunzipSync(
      fs.readFileSync(path.join(outDir, "bc.json.gz")),
    );
    return /"formatVersion"\s*:\s*3\b/.test(
      buf.subarray(0, 200).toString("utf8"),
    );
  } catch {
    return false;
  }
}

function loadRecord(file: string): GameRecord {
  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  // Archived turns are sparse (empty turns dropped); re-expand so index in
  // the array == turn number == tick.
  return decompressGameRecord(JSON.parse(json) as GameRecord);
}

async function writeOutcomeOracle(
  recordsDir: string,
  cacheFile: string,
  parityCommit: string,
): Promise<void> {
  const files = fs
    .readdirSync(recordsDir, { recursive: true })
    .map(String)
    .filter((file) => file.endsWith(".json") || file.endsWith(".json.gz"))
    .map((file) => path.join(recordsDir, file))
    .sort();
  const recordHasher = createHash("sha256");
  for (const file of files) {
    recordHasher.update(path.relative(recordsDir, file));
    recordHasher.update("\0");
    recordHasher.update(fs.readFileSync(file));
    recordHasher.update("\0");
  }
  const recordSetHash = recordHasher.digest("hex");
  const oracleFingerprint = createHash("sha256")
    .update(fs.readFileSync(fileURLToPath(import.meta.url)))
    .update("\0")
    .update(parityCommit)
    .digest("hex");
  try {
    const cached = JSON.parse(fs.readFileSync(cacheFile, "utf8")) as {
      schemaVersion?: number;
      parityCommit?: string;
      recordSetHash?: string;
      oracleFingerprint?: string;
      outcomes?: unknown[];
    };
    if (
      cached.schemaVersion === OUTCOME_SCHEMA_VERSION &&
      cached.parityCommit === parityCommit &&
      cached.recordSetHash === recordSetHash &&
      cached.oracleFingerprint === oracleFingerprint &&
      cached.outcomes?.length === files.length
    ) {
      console.error(
        `[outcome-oracle] cache hit: ${files.length} records at ${cacheFile}`,
      );
      return;
    }
  } catch {
    // A missing or stale cache is regenerated below.
  }

  const originalLog = console.log;
  console.log = () => undefined;
  const outcomes: GameOutcome[] = [];
  try {
    for (const file of files) {
      outcomes.push(await replayOutcome(loadRecord(file)));
    }
  } finally {
    console.log = originalLog;
  }
  outcomes.sort((a, b) =>
    a.gameId < b.gameId ? -1 : a.gameId > b.gameId ? 1 : 0,
  );
  const cache = {
    schemaVersion: OUTCOME_SCHEMA_VERSION,
    parityCommit,
    recordSetHash,
    oracleFingerprint,
    outcomes,
  };
  fs.mkdirSync(path.dirname(cacheFile), { recursive: true });
  const temporary = `${cacheFile}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(cache, null, 2)}\n`);
  fs.renameSync(temporary, cacheFile);
  console.error(
    `[outcome-oracle] cached ${outcomes.length} records at ${cacheFile}`,
  );
}

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  if (args.includes("--outcome-oracle")) {
    const recordsDir = getArg("records", "records");
    const cacheFile = getArg(
      "cache",
      path.join(".cache", "outcome-oracle.json"),
    );
    const parityCommit = getArg("parity-commit", path.basename(recordsDir));
    await writeOutcomeOracle(recordsDir, cacheFile, parityCommit);
    return;
  }

  const recordsDir = getArg("records", "records");
  const outRoot = getArg("out", "data-human");
  const snapshotEvery = parseInt(getArg("every", "10"), 10);
  const limit = parseInt(getArg("limit", "0"), 10);
  const bc = args.includes("--bc");
  // Regenerate sidecars even where bc.json.gz exists (e.g. upgrading
  // formatVersion 2 -> 3 for v6 action labels). States are reused.
  const rebc = args.includes("--rebc");

  const files = fs
    .readdirSync(recordsDir, { recursive: true })
    .map(String)
    .filter((f) => f.endsWith(".json") || f.endsWith(".json.gz"))
    .map((f) => path.join(recordsDir, f))
    .sort();
  const todo = limit > 0 ? files.slice(0, limit) : files;

  let ok = 0;
  let failed = 0;
  for (const file of todo) {
    let record: GameRecord;
    try {
      record = loadRecord(file);
    } catch (e) {
      console.error(`[${path.basename(file)}] unreadable: ${e}`);
      failed++;
      continue;
    }
    const gameID = record.info.gameID;
    const mapKey = String(record.info.config.gameMap)
      .toLowerCase()
      .replace(/\s+/g, "");
    const outDir = path.join(outRoot, mapKey, gameID);
    const hasStates = fs.existsSync(path.join(outDir, "meta.json"));
    const hasBc = fs.existsSync(path.join(outDir, "bc.json.gz"));
    if (hasStates && (!bc || (hasBc && (!rebc || bcIsCurrent(outDir))))) {
      continue; // already replayed (and bc-dumped at the current format)
    }
    const started = Date.now();
    try {
      const res = await replayGame(record, outDir, snapshotEvery, bc, !hasStates);
      const secs = ((Date.now() - started) / 1000).toFixed(1);
      if (res.ok) {
        ok++;
        console.log(
          `[${gameID}] ok: ${res.ticks} ticks, ${res.snapshots} snapshots, ` +
            `${res.hashesChecked} hashes verified, ${secs}s`,
        );
      } else {
        failed++;
        if (!hasStates) fs.rmSync(outDir, { recursive: true, force: true });
        console.error(`[${gameID}] FAILED: ${res.reason} (${secs}s)`);
      }
    } catch (e) {
      failed++;
      if (!hasStates) fs.rmSync(outDir, { recursive: true, force: true });
      console.error(`[${gameID}] crashed: ${e}`);
    }
  }
  console.log(`replay done: ${ok} ok, ${failed} failed, ${todo.length} total`);
}

const isMain =
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);

if (isMain) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
