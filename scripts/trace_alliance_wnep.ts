/**
 * Trace the kh71ym4f->wnep5pzi alliance request from tick 192 to 220.
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
  if (!file) throw new Error("usage: trace_alliance_wnep.ts <record>");

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

    // Replay up to tick 192 (process turns 0..191)
    const START_TARGET = 191;
    const END_TARGET = 225;
    
    for (const turn of record.turns) {
      if (turn.turnNumber > END_TARGET) break;
      game.addExecution(...executor.createExecs(turn));
      game.executeNextTick();
      
      const tick = game.ticks();
      if (tick < 193) continue; // Only start checking after tick 192
      
      // Check the status of kh71ym4f->wnep5pzi request
      const allReqs = (game as any).allianceRequests ?? [];
      const wnepReq = allReqs.find((ar: any) => {
        const req = ar.requestor?.();
        const rec = ar.recipient?.();
        return req?.id?.() === "kh71ym4f" && rec?.id?.() === "wnep5pzi";
      });
      
      // Check execs for the alliance request exec
      const execs = game.executions();
      const wnepExec = execs.find((e: any) => {
        return e.constructor?.name === "AllianceRequestExecution" &&
          (e.requestor?.id?.() === "kh71ym4f" || e.requestorClientID === "2G1d6rWF" || e.requestorID === "kh71ym4f");
      });
      
      // Check for wnep5pzi alliances
      const pwnep = game.players().find(p => p.id() === "wnep5pzi");
      const p86 = game.players().find(p => p.id() === "kh71ym4f");
      const allied = p86?.alliances().some(a => a.requestor().id() === "wnep5pzi" || a.recipient().id() === "wnep5pzi");

      if (wnepReq || wnepExec || allied) {
        const status = wnepReq?.status?.() ?? "GONE";
        const execActive = wnepExec ? (wnepExec as any).active ?? (wnepExec as any)._active : "NO_EXEC";
        console.log(`tick=${tick} req=${status} exec_active=${execActive} allied=${allied}`);
      } else if (tick >= 193 && tick <= 225) {
        console.log(`tick=${tick} req=GONE exec=GONE allied=false`);
      }
    }

    // Also get wnep5pzi's NationExecution attack rate and attack tick
    const execs = game.executions();
    const nationExec = execs.find((e: any) => {
      return e.constructor?.name === "NationExecution" &&
        (e.player?.id?.() === "wnep5pzi" || e.playerID === "wnep5pzi");
    });
    if (nationExec) {
      const ne = nationExec as any;
      console.log(`\nwnep5pzi NationExecution: attackRate=${ne.attackRate} attackTick=${ne.attackTick}`);
    }

  } finally {
    restore();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
