/**
 * Replay to a tick and print hash + tile totals (stdout JSON).
 * Uses openfront-ai-rust-fast/openfront (dedicated parity submodule). Never the webbot checkout.
 * Usage: npx tsx scripts/debug_tick_state.ts <record.json.gz> <tick>
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
  if (i >= 0 && i + 1 < parts.length) {
    return parts[i + 1];
  }
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
  const target = Number(process.argv[3] ?? "310");
  if (!file) throw new Error("usage: debug_tick_state.ts <record> [tick]");

  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  const record = decompressGameRecord(JSON.parse(json) as GameRecord);
  const commit =
    record.gitCommit?.slice(0, 12) ??
    commitFromPath(file)?.slice(0, 12) ??
    null;
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

  const players = [...game.players()]
    .filter((p) => p.isPlayer())
    .map((p) => ({
      id: p.id(),
      playerType: p.type(),
      tilesOwned: p.numTilesOwned(),
      troops: p.troops(),
      units: p.units().length,
      unitHash: p.units().reduce((s, u) => s + u.hash(), 0),
      unitList: p.units().map((u) => ({
        type: u.type(),
        tile: u.tile(),
        id: u.id(),
        hash: u.hash(),
      })),
      idHash: simpleHash(p.id()),
    }))
    .sort((a, b) => b.tilesOwned - a.tilesOwned);

  const totalTiles = players.reduce((s, p) => s + p.tilesOwned, 0);
  const totalTroops = players.reduce((s, p) => s + p.troops, 0);

  process.stdout.write(
    JSON.stringify({
      tick: game.ticks(),
      hash: game.hash(),
      totalTiles,
      totalTroops,
      players,
    }),
  );
  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
