/**
 * Check alliance request state at tick 191 (before the kh71ym4f->wnep5pzi intent is processed).
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
    } catch { /* best effort */ }
  };
}

async function main() {
  const file = process.argv[2];
  if (!file) throw new Error("usage: check_alliance_191.ts <record>");
  const target = Number(process.argv[3] ?? "191");

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

    // Print all alliance requests involving kh71ym4f or wnep5pzi
    const p86 = game.players().find((p) => p.id() === "kh71ym4f");
    const pwnep = game.players().find((p) => p.id() === "wnep5pzi");
    const allReqs = (game as any).allianceRequests ?? [];
    console.log("\n--- All alliance requests involving kh71ym4f or wnep5pzi ---");
    allReqs.forEach((ar: any) => {
      const req = ar.requestor?.() ?? ar.requestor;
      const rec = ar.recipient?.() ?? ar.recipient;
      const reqId = typeof req === "object" ? req.id?.() : req;
      const recId = typeof rec === "object" ? rec.id?.() : rec;
      if (reqId === "kh71ym4f" || recId === "kh71ym4f" || reqId === "wnep5pzi" || recId === "wnep5pzi") {
        console.log(`  ${reqId}->${recId} status=${ar.status?.()}`);
      }
    });

    // Check what alliances kh71ym4f and wnep5pzi have
    console.log("\n--- Alliances of kh71ym4f ---");
    p86?.alliances().forEach(a => {
      const req = a.requestor();
      const rec = a.recipient();
      console.log(`  ${req.id()}<->${rec.id()} expiresAt=${a.expiresAt()}`);
    });
    console.log("\n--- Alliances of wnep5pzi ---");
    pwnep?.alliances().forEach(a => {
      const req = a.requestor();
      const rec = a.recipient();
      console.log(`  ${req.id()}<->${rec.id()} expiresAt=${a.expiresAt()}`);
    });

    // Check can kh71ym4f send alliance request to wnep5pzi
    const canSend = p86?.canSendAllianceRequest(pwnep!);
    console.log(`\ncan kh71ym4f->wnep5pzi: ${canSend}`);
    console.log(`wnep5pzi isDisconnected: ${pwnep?.isDisconnected()}`);
    console.log(`wnep5pzi isAlive: ${pwnep?.isAlive()}`);
    console.log(`kh71ym4f isFriendly(wnep5pzi): ${p86?.isFriendly(pwnep!)}`);

    // Also check what turn 191 contained for kh71ym4f
    console.log("\n--- Turns near 191 involving kh71ym4f or wnep5pzi ---");
    for (const turn of record.turns) {
      if (turn.turnNumber < 185 || turn.turnNumber > 198) continue;
      for (const intent of (turn.intents ?? [])) {
        const parsed = intent as any;
        if (JSON.stringify(parsed).includes("kh71ym4f") || JSON.stringify(parsed).includes("wnep5pzi")) {
          console.log(`  turn ${turn.turnNumber}: ${JSON.stringify(parsed)}`);
        }
      }
    }

    // Check all uninitialized execs (the ones waiting to be initialized next tick)
    const uninitExecs = (game as any).unInitExecs ?? [];
    console.log(`\n--- Uninitialized execs at tick ${target} ---`);
    uninitExecs.forEach((e: any, i: number) => {
      const name = e.constructor?.name ?? "unknown";
      if (name.includes("Alliance")) {
        const reqId = e.requestorID ?? e.requestor?.id?.() ?? e.requestorClientID ?? "?";
        const recId = e.recipientID ?? e.recipient?.id?.() ?? e.recipientClientID ?? "?";
        console.log(`  [${i}] ${name} (${reqId}->${recId})`);
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
