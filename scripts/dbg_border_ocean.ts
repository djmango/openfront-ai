/**
 * TEMP debug: dump border tiles + shore/ocean flags for a specific player at a tick.
 * Usage: npx tsx scripts/dbg_border_ocean.ts <record.json.gz> <tick> <playerId>
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
import {
  GameType,
  PlayerInfo,
  PlayerType,
} from "../openfront/src/core/game/Game";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { loadFreshTerrain } from "../datagen/common";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { simpleHash } from "../openfront/src/core/Util";

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
  const pin = execSync("git rev-parse HEAD", {
    cwd: ENGINE_DIR,
    encoding: "utf8",
  }).trim();
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
  const target = Number(process.argv[3] ?? "210");
  const targetId = process.argv[4] ?? "tf6l7nfm";
  if (!file) throw new Error("usage: dbg_border_ocean.ts <record> [tick] [playerId]");

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
    const terrain = await loadFreshTerrain(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
    );
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

    const p = (game as any)
      .allPlayers()
      .find((pl: any) => pl.id() === targetId);
    if (!p) throw new Error("player not found: " + targetId);

    const border: [number, number, boolean, boolean][] = [];
    for (const t of p.borderTiles()) {
      const shore = game.isShore(t);
      let touchesOcean = false;
      for (const n of game.neighbors(t)) {
        if (game.isWater(n) && game.isOcean(n)) touchesOcean = true;
      }
      border.push([game.x(t), game.y(t), shore, touchesOcean]);
    }
    border.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    console.error(
      `player ${targetId} border_count=${border.length}`,
    );
    console.error(
      "ocean_touching_border_tiles=",
      JSON.stringify(border.filter((b) => b[3])),
    );
    console.error("all_border_tiles=", JSON.stringify(border));
  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
