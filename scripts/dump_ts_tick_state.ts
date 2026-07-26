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
  name: string;
  playerType: string;
  team: string | null;
  tiles: number;
  troops: number;
  gold: string;
  alive: boolean;
  hash: number;
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

interface TickSnapshot {
  tick: number;
  inSpawnPhase: boolean;
  totalLandTiles: number;
  totalOwnedTiles: number;
  players: PlayerSnapshot[];
  railroads?: RailroadSnapshot[];
  stations?: StationSnapshot[];
}

function playerIdentity(p: Player): string {
  const clientID = p.clientID();
  return clientID === null ? `nation:${p.name()}` : `player:${clientID}`;
}

function snapshot(
  game: Game,
  dumpUnits: boolean,
  dumpOwnedTiles: boolean,
  dumpBorderOrder: boolean,
  dumpOwnedOrder: boolean,
  dumpRails: boolean,
): TickSnapshot {
  const players: PlayerSnapshot[] = game.allPlayers().map((p) => {
    const base: PlayerSnapshot = {
      identity: playerIdentity(p),
      id: p.id(),
      name: p.name(),
      playerType: p.type(),
      team: p.team(),
      tiles: p.numTilesOwned(),
      troops: Math.round(p.troops()),
      gold: p.gold().toString(),
      alive: p.isAlive(),
      hash: p.hash(),
      numUnits: p.units().length,
    };
    if (dumpUnits) {
      base.units = p.units().map((u) => ({
        id: u.id(),
        unitType: u.type(),
        tile: u.tile(),
        hash: u.hash(),
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
  const out: TickSnapshot = {
    tick: game.ticks(),
    inSpawnPhase: game.inSpawnPhase(),
    totalLandTiles: game.numLandTiles(),
    totalOwnedTiles,
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
  return out;
}

async function main() {
  const recordPath = process.argv[2];
  const every = parseInt(process.argv[3] ?? "50", 10);
  const outPath = process.argv[4] ?? "/tmp/ts_ticks.json";
  const maxTicks = process.argv[5] ? parseInt(process.argv[5], 10) : undefined;
  if (!recordPath) {
    console.error("usage: tsx dump_ts_tick_state.ts <record.gz> <every> <outPath> [maxTicks]");
    process.exit(1);
  }

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
    (p) => new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID(), p.isLobbyCreator ?? false, p.clanTag, p.friends ?? []),
  );
  const nations = createNationsForGame(info, terrain.nations, terrain.additionalNations, humans.length, random);
  const game: Game = createGame(humans, nations, terrain.gameMap, terrain.miniGameMap, config, terrain.teamGameSpawnAreas);
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

  const dumpUnits = process.env.OF_DUMP_UNITS !== undefined;
  const dumpOwnedTiles = process.env.OF_DUMP_OWNED_TILES !== undefined;
  const dumpBorderOrder = process.env.OF_DUMP_BORDER_ORDER !== undefined;
  const dumpOwnedOrder = process.env.OF_DUMP_OWNED_ORDER !== undefined;
  const dumpRails = process.env.OF_DUMP_RAILS !== undefined;
  const dumpUnitsFrom = process.env.OF_DUMP_UNITS_FROM
    ? parseInt(process.env.OF_DUMP_UNITS_FROM, 10)
    : 0;
  // Keep snapshots only from this tick onward (still replay from 0). Avoids
  // JSON.stringify blowing past V8's string length on multi-thousand-tick
  // fine dumps of large bot-count curriculum games.
  const dumpTicksFrom = process.env.OF_DUMP_TICKS_FROM
    ? parseInt(process.env.OF_DUMP_TICKS_FROM, 10)
    : 0;

  const out: TickSnapshot[] = [];
  for (const turn of record.turns) {
    if (maxTicks !== undefined && turn.turnNumber > maxTicks) break;
    game.addExecution(...executor.createExecs(turn));
    game.executeNextTick();
    if (game.ticks() < dumpTicksFrom) continue;
    if (game.ticks() % every === 0) {
      out.push(
        snapshot(
          game,
          dumpUnits && game.ticks() >= dumpUnitsFrom,
          dumpOwnedTiles,
          dumpBorderOrder,
          dumpOwnedOrder,
          dumpRails,
        ),
      );
    }
  }
  if (
    game.ticks() >= dumpTicksFrom &&
    (out.length === 0 || out[out.length - 1].tick !== game.ticks())
  ) {
    out.push(
      snapshot(
        game,
        dumpUnits && game.ticks() >= dumpUnitsFrom,
        dumpOwnedTiles,
        dumpBorderOrder,
        dumpOwnedOrder,
        dumpRails,
      ),
    );
  }

  fs.writeFileSync(
    outPath,
    JSON.stringify({ engine: "ts", gameId: info.gameID, every, finalTick: game.ticks(), ticks: out }),
  );
  console.error(`[dump_ts_tick_state] wrote ${out.length} snapshots to ${outPath} (final tick ${game.ticks()})`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
