/**
 * Dump exec order at tick 220, focused on AllianceRequest and nearby execs.
 * Usage: npx tsx scripts/dump_exec_220.ts <record.json.gz>
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
  if (!file) throw new Error("usage: dump_exec_220.ts <record>");
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

    const execs = game.executions();
    console.log(`tick=${target} total_execs=${execs.length}`);

    // Print range around the interesting indices (1220-1360)
    const start = Math.max(0, 1220);
    const end = Math.min(execs.length - 1, 1360);
    for (let i = start; i <= end; i++) {
      const e = execs[i];
      const name = (e as any).constructor?.name ?? "unknown";
      // Get more details for alliance requests and attacks
      let details = "";
      if (name === "AllianceRequestExecution") {
        const r = e as any;
        const reqId = r.requestorID ?? r.requestor?.id?.() ?? r.requestorClientID ?? "?";
        const recId = r.recipientID ?? r.recipient?.id?.() ?? r.recipientClientID ?? "?";
        details = ` (${reqId}->${recId} active=${r.active ?? r._active ?? "?"})`;
      } else if (name === "AttackExecution") {
        const a = e as any;
        const src = a.sourceID ?? a.sender?.id?.() ?? "?";
        const dst = a.targetID ?? a.target?.id?.() ?? "?";
        details = ` (${src}->${dst})`;
      }
      console.log(`  [${i}] ${name}${details}`);
    }

    // Also print ALL alliance request execs
    console.log("\n--- All AllianceRequest execs ---");
    execs.forEach((e, i) => {
      const name = (e as any).constructor?.name ?? "unknown";
      if (name === "AllianceRequestExecution") {
        const r = e as any;
        const reqId = r.requestorID ?? r.requestor?.id?.() ?? r.requestorClientID ?? "?";
        const recId = r.recipientID ?? r.recipient?.id?.() ?? r.recipientClientID ?? "?";
        console.log(`  [${i}] ${name} (${reqId}->${recId} active=${r.active ?? r._active ?? "?"})`);
      }
    });

    // Print player 86 (kh71ym4f) info
    const p86 = game.players().find((p) => p.id() === "kh71ym4f");
    const pwnep = game.players().find((p) => p.id() === "wnep5pzi");
    const pj88 = game.players().find((p) => p.id() === "j88scrfi");
    console.log("\n--- Player states ---");
    console.log(`kh71ym4f (p86): tiles=${p86?.numTilesOwned()} troops=${p86?.troops()}`);
    console.log(`wnep5pzi: tiles=${pwnep?.numTilesOwned()} troops=${pwnep?.troops()}`);
    console.log(`j88scrfi: tiles=${pj88?.numTilesOwned()} troops=${pj88?.troops()}`);

    // Print alliance requests involving kh71ym4f
    console.log("\n--- Alliance requests involving kh71ym4f ---");
    const allReqs = (game as any).allianceRequests ?? [];
    allReqs.forEach((ar: any) => {
      const req = ar.requestor?.() ?? ar.requestor;
      const rec = ar.recipient?.() ?? ar.recipient;
      const reqId = typeof req === "object" ? req.id?.() : req;
      const recId = typeof rec === "object" ? rec.id?.() : rec;
      if (reqId === "kh71ym4f" || recId === "kh71ym4f") {
        console.log(`  ${reqId}->${recId} status=${ar.status?.()}`);
      }
    });
  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
