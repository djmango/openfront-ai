/**
 * TS-side counterpart to rust/engine/src/bin/tick_dump.rs: replays a
 * GameRecord (bot/nation self-play, see gen_selfplay_record.ts) through
 * the real TypeScript engine and dumps the exact same every-N-tick
 * per-player snapshot schema, so a diff script can compare trajectories
 * tick-for-tick against the native engine.
 *
 * Mirrors createGameRunner()'s init exactly (see verify_record.ts /
 * datagen/replay.ts): humans first (there are none here), then nations,
 * then bots via spawnTribes().
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
 *     records/selfplay/bs50.json.gz 50 /tmp/ts_ticks.json [maxTicks]
 */
import * as fs from "fs";
import * as zlib from "zlib";
import { Config } from "../openfront/src/core/configuration/Config";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import { Game, GameType, Player, PlayerInfo, PlayerType, UnitType } from "../openfront/src/core/game/Game";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { GameRecord } from "../openfront/src/core/Schemas";
import { decompressGameRecord, simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain } from "../datagen/common";

interface UnitSnapshot {
  id: number;
  unitType: string;
  tile: number;
  hash: number;
  level: number;
  underConstruction: boolean;
  health: number;
  veterancy: number;
  veterancyProgress: number;
  targetTile: number | null;
  patrolTile: number | null;
  retreatPort: number | null;
  retreating: boolean;
  docked: boolean;
}

interface PlayerSnapshot {
  identity: string;
  // `id`/`hash`/`numUnits`: added for tick-level bisections that need to
  // match players by their stable game-engine id (`identity` is
  // clientID-keyed and `name` alone is ambiguous - bot names collide) and
  // cross-check against native's own per-player hash contribution without
  // re-deriving it from troops/tiles by hand.
  id: string;
  smallId: number;
  name: string;
  playerType: string;
  team: string | null;
  tiles: number;
  troops: number;
  gold: string;
  alive: boolean;
  hash: number;
  /** IEEE-754 bits of player hash float, as decimal string (bit-exact compare). */
  hashBits: string;
  /** Sum of unit.hash() — cheap layer to isolate unit drift. */
  unitsHash: number;
  numUnits: number;
  units?: UnitSnapshot[];
  ownedTiles?: number[];
  borderOrder?: number[];
  ownedOrder?: number[];
}

interface RailroadSnapshot {
  id: number;
  from: number;
  to: number;
  tiles: number[];
}

interface StationSnapshot {
  id: number;
  unitId: number;
  unitType: string;
  tile: number | null;
  railroads: number[];
  cluster: number | null;
}

interface AttackSnapshot {
  ownerSmallId: number;
  targetSmallId: number;
  troops: number;
  active: boolean;
  attackLive: boolean;
}

interface TickSnapshot {
  tick: number;
  inSpawnPhase: boolean;
  totalLandTiles: number;
  totalOwnedTiles: number;
  gameHash: number;
  /** IEEE-754 bits of game.hash() float, as decimal string. */
  gameHashBits: string;
  players: PlayerSnapshot[];
  railroads?: RailroadSnapshot[];
  stations?: StationSnapshot[];
  attacks?: AttackSnapshot[];
}

function playerIdentity(p: Player): string {
  const clientID = p.clientID();
  return clientID === null ? `nation:${p.name()}` : `player:${clientID}`;
}

/** Decimal string of IEEE-754 bit pattern — stable across JSON number precision loss. */
function f64Bits(n: number): string {
  const bits = new BigUint64Array(new Float64Array([n]).buffer)[0];
  return bits.toString();
}

function snapshot(
  game: Game,
  dumpUnits: boolean,
  dumpOwnedTiles: boolean,
  dumpBorderOrder: boolean,
  dumpOwnedOrder: boolean,
  dumpRails: boolean,
  dumpAttacks: boolean = false,
): TickSnapshot {
  const players: PlayerSnapshot[] = game.allPlayers().map((p) => {
    const base: PlayerSnapshot = {
      identity: playerIdentity(p),
      id: p.id(),
      smallId: p.smallID(),
      name: p.name(),
      playerType: p.type(),
      team: p.team(),
      tiles: p.numTilesOwned(),
      troops: Math.round(p.troops()),
      gold: p.gold().toString(),
      alive: p.isAlive(),
      hash: p.hash(),
      hashBits: f64Bits(p.hash()),
      unitsHash: p.units().reduce((sum, u) => sum + u.hash(), 0),
      numUnits: p.units().length,
    };
    if (dumpUnits) {
      base.units = p.units().map((u) => ({
        id: u.id(),
        unitType: u.type(),
        tile: u.tile(),
        hash: u.hash(),
        level: u.level(),
        underConstruction: u.isUnderConstruction(),
        health: u.health(),
        veterancy: typeof (u as any).veterancy === "function" ? (u as any).veterancy() : 0,
        veterancyProgress:
          u.type() === UnitType.Warship
            ? ((u.warshipState() as any).veterancyProgress ?? 0)
            : 0,
        targetTile: u.targetTile() ?? null,
        patrolTile:
          u.type() === UnitType.Warship
            ? (u.warshipState().patrolTile ?? null)
            : null,
        retreatPort:
          u.type() === UnitType.Warship
            ? (u.warshipState().retreatPort ?? null)
            : null,
        retreating:
          u.type() === UnitType.Warship
            ? u.warshipState().state === "retreating"
            : false,
        docked:
          u.type() === UnitType.Warship
            ? u.warshipState().state === "docked"
            : false,
      }));
    }
    if (dumpOwnedTiles) {
      base.ownedTiles = Array.from(p.tiles() as Iterable<number>).sort(
        (a, b) => a - b,
      );
    }
    if (dumpBorderOrder) {
      base.borderOrder = Array.from(p.borderTiles() as Iterable<number>);
    }
    if (dumpOwnedOrder) {
      base.ownedOrder = Array.from(p.tiles() as Iterable<number>);
    }
    return base;
  });
  const totalOwnedTiles = players.reduce((sum, p) => sum + p.tiles, 0);
  const gameHash = game.hash();
  const out: TickSnapshot = {
    tick: game.ticks(),
    inSpawnPhase: game.inSpawnPhase(),
    totalLandTiles: game.numLandTiles(),
    totalOwnedTiles,
    gameHash,
    gameHashBits: f64Bits(gameHash),
    players,
  };
  if (dumpRails) {
    const rn = game.railNetwork() as any;
    const stations = [...rn.stationManager().getAll()] as any[];
    const railSet = new Map<number, RailroadSnapshot>();
    const stationSnaps: StationSnapshot[] = [];
    for (const st of stations) {
      const rails = [...st.railroads] as any[];
      stationSnaps.push({
        id: st.id,
        unitId: st.unit.id(),
        unitType: st.unit.type(),
        tile: st.tile(),
        railroads: rails.map((r) => r.id),
        cluster: st.getCluster() ? 1 : null,
      });
      for (const r of rails) {
        if (!railSet.has(r.id)) {
          railSet.set(r.id, {
            id: r.id,
            from: r.from.id,
            to: r.to.id,
            tiles: [...r.tiles],
          });
        }
      }
    }
    out.railroads = [...railSet.values()].sort((a, b) => a.id - b.id);
    out.stations = stationSnaps.sort((a, b) => a.id - b.id);
  }
  if (dumpAttacks) {
    const attacks: AttackSnapshot[] = [];
    for (const p of game.allPlayers()) {
      for (const a of p.outgoingAttacks()) {
        const target = a.target();
        attacks.push({
          ownerSmallId: p.smallID(),
          targetSmallId: target.isPlayer() ? target.smallID() : 0,
          troops: Math.round(a.troops()),
          active: a.isActive(),
          attackLive: typeof (a as any).isAlive === "function" ? (a as any).isAlive() : true,
        });
      }
    }
    out.attacks = attacks;
  }
  return out;
}

function pollExpandControl(path: string): number | null {
  try {
    const body = fs.readFileSync(path, "utf8").trim();
    if (!body.startsWith("EXPAND")) return null;
    const m = body.match(/until=(\d+)/);
    return m ? parseInt(m[1], 10) : Number.MAX_SAFE_INTEGER;
  } catch {
    return null;
  }
}

async function bootstrapGame(recordPath: string): Promise<{
  record: GameRecord;
  game: Game;
  executor: Executor;
}> {
  const raw = recordPath.endsWith(".gz")
    ? zlib.gunzipSync(fs.readFileSync(recordPath)).toString("utf8")
    : fs.readFileSync(recordPath, "utf8");
  const record: GameRecord = decompressGameRecord(JSON.parse(raw) as GameRecord);
  const info = record.info;
  const gameConfig = info.config;
  const config = new Config(gameConfig, null, false);
  const terrain = await loadFreshTerrain(gameConfig.gameMap as never, gameConfig.gameMapSize);
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
  if (!config.isUnitDisabled(UnitType.Factory)) {
    game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
  }
  return { record, game, executor };
}

function advanceTo(
  game: Game,
  executor: Executor,
  record: GameRecord,
  target: number,
): void {
  if (target < game.ticks()) {
    throw new Error(`cannot rewind: at tick ${game.ticks()}, requested ${target}`);
  }
  while (game.ticks() < target) {
    const turn = record.turns[game.ticks()];
    if (!turn) break;
    game.addExecution(...executor.createExecs(turn));
    game.executeNextTick();
  }
}

async function runDaemon(recordPath: string): Promise<void> {
  let { record, game, executor } = await bootstrapGame(recordPath);
  const gameId = record.info.gameID;
  const readline = await import("readline");
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  const reply = (s: string) => {
    process.stdout.write(s + "\n");
  };
  reply(`OK tick=${game.ticks()}`);
  for await (const lineRaw of rl) {
    const line = lineRaw.trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    const cmd = (parts[0] || "").toUpperCase();
    try {
      if (cmd === "STATUS") {
        reply(`OK tick=${game.ticks()}`);
      } else if (cmd === "RESET") {
        ({ record, game, executor } = await bootstrapGame(recordPath));
        reply(`OK tick=${game.ticks()}`);
      } else if (cmd === "ADVANCE") {
        const t = parseInt(parts[1], 10);
        if (Number.isNaN(t)) {
          reply("ERR usage ADVANCE <tick>");
          continue;
        }
        advanceTo(game, executor, record, t);
        reply(`OK tick=${game.ticks()}`);
      } else if (cmd === "DUMP") {
        const path = parts[1];
        if (!path) {
          reply("ERR usage DUMP <path> [units]");
          continue;
        }
        const units = parts.slice(2).some((p) => p === "units" || p === "units=1");
        const snap = snapshot(game, units, false, false, false, false);
        fs.writeFileSync(
          path,
          JSON.stringify({
            engine: "ts",
            gameId,
            every: 1,
            finalTick: snap.tick,
            ticks: [snap],
          }),
        );
        reply(`OK tick=${game.ticks()} path=${path}`);
      } else if (cmd === "QUIT" || cmd === "EXIT") {
        reply("OK bye");
        break;
      } else {
        reply(`ERR unknown command ${cmd}`);
      }
    } catch (e: any) {
      reply(`ERR ${e?.message ?? e}`);
    }
  }
}

async function main() {
  if (process.argv[2] === "--daemon" || process.env.OF_DUMP_DAEMON !== undefined) {
    const recordPath = process.argv[2] === "--daemon" ? process.argv[3] : process.argv[2];
    if (!recordPath) {
      console.error("usage: tsx dump_ts_tick_state.ts --daemon <record.gz>");
      process.exit(1);
    }
    await runDaemon(recordPath);
    return;
  }

  const recordPath = process.argv[2];
  const every = parseInt(process.argv[3] ?? "50", 10);
  const outPath = process.argv[4] ?? "/tmp/ts_ticks.json";
  const maxTicks = process.argv[5] ? parseInt(process.argv[5], 10) : undefined;
  if (!recordPath) {
    console.error("usage: tsx dump_ts_tick_state.ts <record.gz> <every> <outPath> [maxTicks]");
    process.exit(1);
  }

  const { record, game, executor } = await bootstrapGame(recordPath);
  const info = record.info;

  let dumpUnits = process.env.OF_DUMP_UNITS !== undefined;
  const dumpOwnedTiles = process.env.OF_DUMP_OWNED_TILES !== undefined;
  const dumpBorderOrder = process.env.OF_DUMP_BORDER_ORDER !== undefined;
  const dumpOwnedOrder = process.env.OF_DUMP_OWNED_ORDER !== undefined;
  const dumpRails = process.env.OF_DUMP_RAILS !== undefined;
  const dumpAttacks = process.env.OF_DUMP_ATTACKS !== undefined;
  const dumpUnitsFrom = process.env.OF_DUMP_UNITS_FROM
    ? parseInt(process.env.OF_DUMP_UNITS_FROM, 10)
    : 0;
  const dumpTicksFrom = process.env.OF_DUMP_TICKS_FROM
    ? parseInt(process.env.OF_DUMP_TICKS_FROM, 10)
    : 0;
  const controlPath = process.env.OF_DUMP_CONTROL;
  const ndjson =
    process.env.OF_DUMP_NDJSON !== undefined || outPath.endsWith(".ndjson");

  let ndjsonFd: number | null = null;
  if (ndjson) {
    ndjsonFd = fs.openSync(outPath, "w");
    fs.writeSync(
      ndjsonFd,
      JSON.stringify({
        type: "header",
        engine: "ts",
        gameId: info.gameID,
        every,
      }) + "\n",
    );
  }

  const out: TickSnapshot[] = [];
  let lastEmittedTick: number | undefined;
  let expandUntil: number | undefined;
  const pushSnap = (snap: TickSnapshot) => {
    lastEmittedTick = snap.tick;
    if (ndjsonFd !== null) {
      fs.writeSync(ndjsonFd, JSON.stringify(snap) + "\n");
    } else {
      out.push(snap);
    }
  };

  for (const turn of record.turns) {
    if (maxTicks !== undefined && turn.turnNumber > maxTicks) break;
    if (expandUntil !== undefined && game.ticks() >= expandUntil) break;
    game.addExecution(...executor.createExecs(turn));
    game.executeNextTick();

    if (controlPath && expandUntil === undefined) {
      const until = pollExpandControl(controlPath);
      if (until !== null) {
        expandUntil =
          until === Number.MAX_SAFE_INTEGER ? game.ticks() + 25 : until;
        dumpUnits = true;
        console.error(
          `[dump_ts_tick_state] EXPAND control → units until tick ${expandUntil} (at ${game.ticks()})`,
        );
      }
    }

    if (game.ticks() < dumpTicksFrom) continue;
    const step = expandUntil !== undefined ? 1 : every;
    if (game.ticks() % step === 0) {
      pushSnap(
        snapshot(
          game,
          dumpUnits && game.ticks() >= dumpUnitsFrom,
          dumpOwnedTiles,
          dumpBorderOrder,
          dumpOwnedOrder,
          dumpRails,
          dumpAttacks,
        ),
      );
    }
    if (expandUntil !== undefined && game.ticks() >= expandUntil) break;
  }
  if (game.ticks() >= dumpTicksFrom && lastEmittedTick !== game.ticks()) {
    pushSnap(
      snapshot(
        game,
        dumpUnits && game.ticks() >= dumpUnitsFrom,
        dumpOwnedTiles,
        dumpBorderOrder,
        dumpOwnedOrder,
        dumpRails,
        dumpAttacks,
      ),
    );
  }

  if (ndjsonFd !== null) {
    fs.closeSync(ndjsonFd);
    console.error(
      `[dump_ts_tick_state] streamed ndjson to ${outPath} (final tick ${game.ticks()})`,
    );
    return;
  }

  fs.writeFileSync(
    outPath,
    JSON.stringify({
      engine: "ts",
      gameId: info.gameID,
      every,
      finalTick: game.ticks(),
      ticks: out,
    }),
  );
  console.error(
    `[dump_ts_tick_state] wrote ${out.length} snapshots to ${outPath} (final tick ${game.ticks()})`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
