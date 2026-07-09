/**
 * Debug helper: replicate SpatialQuery.closestShoreByWater's internal steps
 * for a single transport-ship src selection, to compare against native's
 * equivalent debug_transport_src_pick test. Read-only; does not modify the
 * openfront submodule (uses `as any` to reach a private method).
 *
 * Usage: npx tsx scripts/debug_transport_shore.ts <record.json.gz> <tick> <playerId> <refTile>
 */
import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as zlib from "zlib";
import { decompressGameRecord } from "../openfront/src/core/Util";
import type { GameRecord } from "../openfront/src/core/Schemas";
import { Config } from "../openfront/src/core/configuration/Config";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { GameType, PlayerInfo, PlayerType } from "../openfront/src/core/game/Game";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { loadFreshTerrain } from "../datagen/common";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { simpleHash } from "../openfront/src/core/Util";
import { SpatialQuery } from "../openfront/src/core/pathfinding/spatial/SpatialQuery";
import { targetTransportTile } from "../openfront/src/core/game/TransportShipUtils";
import { PathFinding } from "../openfront/src/core/pathfinding/PathFinder";
import { SmoothingWaterTransformer } from "../openfront/src/core/pathfinding/transformers/SmoothingWaterTransformer";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ENGINE_DIR = path.join(REPO, "openfront");
const WEBBOT_ENGINE = "/Users/djmango/github/openfront-ai/openfront";

function commitFromPath(file: string): string | null {
  const parts = path.resolve(file).split(path.sep);
  const i = parts.lastIndexOf("records");
  if (i >= 0 && i + 1 < parts.length) return parts[i + 1];
  return null;
}

function pinEngine(commit: string): () => void {
  if (path.resolve(ENGINE_DIR) === path.resolve(WEBBOT_ENGINE)) {
    throw new Error("refusing to pin webbot openfront");
  }
  const pin = execSync("git rev-parse HEAD", { cwd: ENGINE_DIR, encoding: "utf8" }).trim();
  const checkout = (rev: string) => {
    execSync(`git checkout -q ${rev}`, { cwd: ENGINE_DIR, stdio: "pipe" });
  };
  checkout(commit);
  return () => {
    try {
      checkout(pin);
    } catch {
      /* best effort */
    }
  };
}

async function main() {
  const file = process.argv[2];
  const target = Number(process.argv[3] ?? "310");
  const playerId = process.argv[4];
  const refTile = Number(process.argv[5]);
  if (!file || !playerId || !refTile) {
    throw new Error(
      "usage: debug_transport_shore.ts <record> <tick> <playerId> <refTile>",
    );
  }

  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  const record = decompressGameRecord(JSON.parse(json) as GameRecord);
  const commit =
    record.gitCommit?.slice(0, 12) ?? commitFromPath(file)?.slice(0, 12) ?? null;
  if (!commit) throw new Error("no engine commit on record or path");
  const restore = pinEngine(commit);
  try {
    const info = record.info;
    const gameConfig = info.config;
    const config = new Config(gameConfig, null, false);
    const terrain = await loadFreshTerrain(gameConfig.gameMap, gameConfig.gameMapSize);
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
    const game = createGame(
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

    for (const turn of record.turns) {
      if (turn.turnNumber > target) break;
      game.addExecution(...executor.createExecs(turn));
      game.executeNextTick();
    }

    const gm = game as any;
    const map = gm.map();
    const player = gm.player(playerId);

    const dst = targetTransportTile(gm, refTile as any);
    console.log(
      "dst tile=",
      dst,
      "x=",
      map.x(dst),
      "y=",
      map.y(dst),
    );

    const targetComponent = gm.getWaterComponent(dst);
    const isValidTile = (t: number) => {
      if (!gm.isShore(t) || !gm.isLand(t)) return false;
      return gm.getWaterComponent(t) === targetComponent;
    };
    const shores: number[] = Array.from(player.borderTiles()).filter(isValidTile) as number[];
    console.log("shores.len()=", shores.length);
    for (const s of shores.slice(0, 20)) {
      console.log("  shore=", s, "x=", map.x(s), "y=", map.y(s));
    }

    console.error("=== MARKER before findPath ===");
    const rawPath = PathFinding.Water(gm).findPath(shores as any, dst as any);
    console.log("path.len()=", rawPath?.length);
    const fullDump = process.env.DUMP_FULL_PATH === "1";
    for (const [i, t] of (rawPath ?? []).slice(0, fullDump ? undefined : 10).entries()) {
      console.log(`  path[${i}]=`, t, "x=", map.x(t), "y=", map.y(t));
    }

    const sq: any = new SpatialQuery(gm);
    const refined = sq.refineStartTile(rawPath, shores, gm);
    console.log("refineStartTile ->", refined, "x=", map.x(refined), "y=", map.y(refined));

    const chosen = sq.closestShoreByWater(player, dst);
    console.log("closestShoreByWater ->", chosen, "x=", map.x(chosen), "y=", map.y(chosen));

    if (process.env.DUMP_RAW_MINI_PATH === "1") {
      const hpa: any = gm.miniWaterHPA();
      const miniMap = gm.miniMap();
      const miniShores = shores.map((s: number) => miniMap.ref(Math.floor(map.x(s) / 2), Math.floor(map.y(s) / 2)));
      const miniDst = miniMap.ref(Math.floor(map.x(dst) / 2), Math.floor(map.y(dst) / 2));
      const rawMini = hpa.findPath(miniShores, miniDst);
      console.log("rawMini.len()=", rawMini?.length);
      for (const [i, t] of (rawMini ?? []).entries()) {
        console.log(`  rawMini[${i}]=`, t, "x=", miniMap.x(t), "y=", miniMap.y(t));
      }
      const smoother: any = new SmoothingWaterTransformer(null as any, miniMap);
      if (process.env.DUMP_TRACE) {
        const [from, to] = process.env.DUMP_TRACE.split(",").map(Number);
        const trace = smoother.tracePath(from, to);
        console.log("trace.len()=", trace?.length);
        for (const [i, t] of (trace ?? []).entries()) {
          console.log(`  trace[${i}]=`, t, "x=", miniMap.x(t), "y=", miniMap.y(t));
        }
      }
      const smoothedMini = smoother.smooth(rawMini);
      console.log("smoothedMini.len()=", smoothedMini?.length);
      for (const [i, t] of (smoothedMini ?? []).entries()) {
        console.log(`  smoothedMini[${i}]=`, t, "x=", miniMap.x(t), "y=", miniMap.y(t));
      }
    }

    const terrainDumpArg = process.env.DUMP_TERRAIN;
    if (terrainDumpArg) {
      const terrain = (map as any).terrain as Uint8Array;
      for (const spec of terrainDumpArg.split(",")) {
        const [tx, ty] = spec.split(":").map(Number);
        const t = map.ref(tx, ty);
        console.log(`terrain x=${tx} y=${ty} byte=${terrain[t]} magnitude=${terrain[t] & 0x1f} isWater=${map.isWater(t)}`);
      }
    }
    const miniTerrainDumpArg = process.env.DUMP_MINI_TERRAIN;
    if (miniTerrainDumpArg) {
      const miniMap = gm.miniMap();
      const miniTerrain = (miniMap as any).terrain as Uint8Array;
      for (const spec of miniTerrainDumpArg.split(",")) {
        const [tx, ty] = spec.split(":").map(Number);
        const t = miniMap.ref(tx, ty);
        console.log(`mini terrain x=${tx} y=${ty} byte=${miniTerrain[t]} magnitude=${miniTerrain[t] & 0x1f} isWater=${miniMap.isWater(t)}`);
      }
    }
  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
