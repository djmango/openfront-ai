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
 * action space, and final placements — the (state, action) supervision for
 * behavior cloning (rl/bc_data.py).
 */
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
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
import { legality } from "../bridge/common";
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

/** A human intent normalized to the policy's action space (rl/obs.py
 * ACTIONS). Player references become smallIDs; tiles become x/y. Intents
 * outside the modeled surface (emoji, warship moves, upgrades, ...) are
 * dropped. Quantity bucketing happens in Python where the actor's
 * troops/gold at the window start are known. */
interface BCLabel {
  a: string;
  t?: number; // target smallID
  x?: number;
  y?: number;
  unit?: string;
  amt?: number | null;
}

const NUKE_UNITS = new Set(["Atom Bomb", "Hydrogen Bomb", "MIRV"]);
const STRUCTURE_UNITS = new Set([
  "City",
  "Port",
  "Defense Post",
  "Missile Silo",
  "SAM Launcher",
  "Factory",
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
        return { a: "launch_nuke", unit, ...xy(intent.tile) };
      }
      if (STRUCTURE_UNITS.has(unit)) {
        return { a: "build", unit, ...xy(intent.tile) };
      }
      return null;
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
      if (intent.action !== "start") return null;
      const t = smallID(intent.targetID);
      return t === null ? null : { a: "embargo", t };
    }
    case "cancel_attack":
      return { a: "retreat" };
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

async function replayGame(
  record: GameRecord,
  outDir: string,
  snapshotEvery: number,
  bc: boolean,
  writeStates: boolean,
): Promise<ReplayResult> {
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
  const humanClientIDs = new Set(info.players.map((p) => p.clientID));

  for (const turn of record.turns) {
    if (bc) {
      // Normalize against the pre-turn state: the intent was decided by the
      // player looking at this state, and targets may die during the tick.
      for (const si of turn.intents) {
        const cid = (si as { clientID: string }).clientID;
        if (!humanClientIDs.has(cid)) continue;
        const label = normalizeIntent(game, si as Record<string, unknown>);
        if (label === null) continue;
        const byClient = labelsByTick.get(game.ticks()) ?? {};
        (byClient[cid] ??= []).push(label);
        labelsByTick.set(game.ticks(), byClient);
      }
    }
    game.addExecution(...executor.createExecs(turn));
    const updates = game.executeNextTick();

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
        const stem = `t${String(game.ticks()).padStart(6, "0")}`;
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
      formatVersion: 1,
      snapshotEvery,
      placements: placements(game, record),
      steps,
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

function loadRecord(file: string): GameRecord {
  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  // Archived turns are sparse (empty turns dropped); re-expand so index in
  // the array == turn number == tick.
  return decompressGameRecord(JSON.parse(json) as GameRecord);
}

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  const recordsDir = getArg("records", "records");
  const outRoot = getArg("out", "data-human");
  const snapshotEvery = parseInt(getArg("every", "10"), 10);
  const limit = parseInt(getArg("limit", "0"), 10);
  const bc = args.includes("--bc");

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
    if (hasStates && (!bc || fs.existsSync(path.join(outDir, "bc.json.gz")))) {
      continue; // already replayed (and bc-dumped, when requested)
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

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
