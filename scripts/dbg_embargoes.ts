/**
 * Dump active embargoes and per-player sharedWaterComponents at a given tick, to check
 * whether embargo state (canTrade) affects SharedWaterCache results for 3QNU4eJa.
 * Usage: npx tsx scripts/dbg_embargoes.ts <record.json.gz> <tick>
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

function pinEngine(commit: string): () => void {
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
  if (!file) throw new Error("usage: dbg_embargoes.ts <record> <tick>");
  const target = Number(process.argv[3] ?? "220");

  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  const record = decompressGameRecord(JSON.parse(json) as GameRecord);
  const commit = record.gitCommit?.slice(0, 12) ?? null;
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

    console.log(`tick=${game.ticks()}`);
    console.log("--- active embargoes ---");
    let count = 0;
    for (const p of game.players()) {
      for (const e of (p as any).getEmbargoes?.() ?? []) {
        count++;
        console.log(`  ${p.id()} embargoes ${e.target?.id?.() ?? e.target} isTemporary=${e.isTemporary} createdAt=${e.createdAt}`);
      }
    }
    console.log(`total embargoes: ${count}`);

    const focusIds = [
      "kh71ym4f", "wnep5pzi", "j88scrfi",
      "1zdhjq4c", "567m55cy", "e4wny0kw", "njpi899a", "g2e9nflk",
      "q5dec3u6", "tf6l7nfm", "dx8rfeww", "hrf8g2tf", "16jvij8c", "x9qaj6ai",
    ];
    console.log("\n--- sharedWaterComponents for focus players ---");
    for (const id of focusIds) {
      const p = game.players().find((pp) => pp.id() === id);
      if (!p) {
        console.log(`  ${id}: NOT FOUND`);
        continue;
      }
      const shared = (game as any).sharedWaterComponents?.(p);
      console.log(`  ${id} (small=${p.smallID()}): shared=${shared ? [...shared] : null}`);
    }
  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
