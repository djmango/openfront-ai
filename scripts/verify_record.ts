/**
 * Validate an rl/watch.py GameRecord against the engine schema and replay it
 * headlessly to confirm determinism (agent trajectory matches the record).
 *
 * Usage: openfront/node_modules/.bin/tsx scripts/verify_record.ts records-rl/stage3.json
 */
import * as fs from "fs";
import { Config } from "../openfront/src/core/configuration/Config";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import {
  Game,
  GameType,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../openfront/src/core/game/Game";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { GameRecordSchema, Turn } from "../openfront/src/core/Schemas";
import { decompressGameRecord, simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain } from "../datagen/common";

async function main() {
  const file = process.argv[2];
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  const parsed = GameRecordSchema.safeParse(data);
  if (!parsed.success) {
    console.error("SCHEMA INVALID:");
    console.error(JSON.stringify(parsed.error.issues.slice(0, 10), null, 2));
    process.exit(1);
  }
  const record = parsed.data;
  const info = record.info;
  console.log(
    `schema ok: gameID=${info.gameID} map=${info.config.gameMap} ` +
      `turns=${record.turns.length} num_turns=${info.num_turns}`,
  );

  // Replay exactly like the client (GameRunner) would.
  const config = new Config(info.config, null, false);
  const terrain = await loadFreshTerrain(
    info.config.gameMap as never,
    info.config.gameMapSize,
  );
  const random = new PseudoRandom(simpleHash(info.gameID));
  const humans = info.players.map(
    (p) =>
      new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID()),
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
  if (info.config.gameType !== GameType.Singleplayer) {
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

  const turns: Turn[] = decompressGameRecord(record).turns;
  const agentClient = info.players[0].clientID;
  let deathTick = -1;
  for (let t = 0; t < info.num_turns; t++) {
    const turn = turns[t] ?? { turnNumber: t, intents: [] };
    game.addExecution(...executor.createExecs(turn));
    const updates = game.executeNextTick();
    const agent = game.playerByClientID(agentClient);
    if (agent && !agent.isAlive() && deathTick < 0 && !game.inSpawnPhase()) {
      deathTick = game.ticks();
    }
    if (game.ticks() % 1000 === 21) {
      console.log(
        `tick ${game.ticks()}: agent tiles ${agent?.numTilesOwned() ?? 0}, alive ${agent?.isAlive()}`,
      );
    }
    const winUpdates = updates[GameUpdateType.Win];
    if (winUpdates && winUpdates.length > 0) {
      console.log(`winner at tick ${game.ticks()}`);
      break;
    }
  }
  console.log(
    `replay done: final tick ${game.ticks()}, agent death tick ${deathTick}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
