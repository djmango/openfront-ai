#!/usr/bin/env bash
# Dump TS exec order using the dedicated rust-fast openfront submodule.
# Never checks out /Users/djmango/github/openfront-ai/openfront (webbot).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=parity_env.sh
source "$ROOT/scripts/parity_env.sh"
bash "$ROOT/scripts/ensure_parity_openfront.sh"

RECORD="${1:?usage: dump_ts_exec_order.sh <record.gz> <turn>}"
TURN="${2:-302}"

# Prefer a local dump script that imports from this tree's openfront/.
# Fall back to openfront-ai scripts with OPENFRONT_ENGINE_DIR override via NODE_PATH.
export OPENFRONT_ENGINE_DIR="$ROOT/openfront"
export NODE_PATH="$ROOT/openfront/node_modules${NODE_PATH:+:$NODE_PATH}"

cd "$ROOT"
if [[ -f scripts/dump_ts_exec_order_impl.ts ]]; then
  exec npx --yes tsx scripts/dump_ts_exec_order_impl.ts "$RECORD" "$TURN"
fi

# Inline: run against pinned engine without mutating webbot checkout.
exec npx --yes tsx -e "
import * as zlib from 'zlib';
import * as fs from 'fs';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
import * as path from 'path';

const ROOT = process.env.OPENFRONT_REPO!;
const ENGINE = process.env.OPENFRONT_ENGINE_DIR!;
const recordPath = process.argv[1];
const target = Number(process.argv[2] ?? '302');

const require = createRequire(path.join(ENGINE, 'package.json'));
void require;

const util = await import(pathToFileURL(path.join(ENGINE, 'src/core/Util.ts')).href);
const { PseudoRandom } = await import(pathToFileURL(path.join(ENGINE, 'src/core/PseudoRandom.ts')).href);
const { createNationsForGame } = await import(pathToFileURL(path.join(ENGINE, 'src/core/game/NationCreation.ts')).href);
const { createGame } = await import(pathToFileURL(path.join(ENGINE, 'src/core/game/GameImpl.ts')).href);
const { Config } = await import(pathToFileURL(path.join(ENGINE, 'src/core/configuration/Config.ts')).href);
const { loadFreshTerrain } = await import(pathToFileURL(path.join(ROOT, 'datagen/common.ts')).href);
const { PlayerInfo, PlayerType, GameType } = await import(pathToFileURL(path.join(ENGINE, 'src/core/game/Game.ts')).href);
const { SpawnTimerExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/SpawnTimerExecution.ts')).href);
const { WinCheckExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/WinCheckExecution.ts')).href);
const { Executor } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/ExecutionManager.ts')).href);
const { AttackExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/AttackExecution.ts')).href);
const { PlayerExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/PlayerExecution.ts')).href);
const { TribeExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/TribeExecution.ts')).href);
const { NationExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/NationExecution.ts')).href);
const { SpawnExecution } = await import(pathToFileURL(path.join(ENGINE, 'src/core/execution/SpawnExecution.ts')).href);

function execLabel(e) {
  if (e instanceof AttackExecution) {
    const o = e.ownerSmallID ?? e.owner().smallID();
    const t = e.targetSmallID ?? 0;
    return \`Attack(\${o}->\${t})\`;
  }
  if (e instanceof PlayerExecution) return \`Player(\${e.owner().smallID()})\`;
  if (e instanceof TribeExecution) return \`Tribe(\${e.tribe.smallID()})\`;
  if (e instanceof NationExecution) return 'Nation';
  if (e instanceof SpawnExecution) return 'Spawn';
  if (e instanceof SpawnTimerExecution) return 'SpawnTimer';
  if (e instanceof WinCheckExecution) return 'WinCheck';
  return e.constructor.name;
}

const record = util.decompressGameRecord(JSON.parse(zlib.gunzipSync(fs.readFileSync(recordPath)).toString()));
const info = record.info;
const config = new Config(info.config, null, false);
const terrain = await loadFreshTerrain(info.config.gameMap, info.config.gameMapSize);
const random = new PseudoRandom(util.simpleHash(info.gameID));
const humans = info.players.map((p) => new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID(), false, p.clanTag, p.friends ?? []));
const nations = createNationsForGame(info, terrain.nations, terrain.additionalNations, humans.length, random);
const game = createGame(humans, nations, terrain.gameMap, terrain.miniGameMap, config, terrain.teamGameSpawnAreas);
const ex = new Executor(game, info.gameID, undefined);
if (info.config.gameType !== GameType.Singleplayer) game.addExecution(new SpawnTimerExecution());
if (config.spawnNations()) game.addExecution(...ex.nationExecutions());
if (config.isRandomSpawn()) game.addExecution(...ex.spawnPlayers());
if (config.bots() > 0) game.addExecution(...ex.spawnTribes(config.bots()));
game.addExecution(new WinCheckExecution());
for (const turn of record.turns) {
  if (turn.turnNumber > target) break;
  game.addExecution(...ex.createExecs(turn));
  game.executeNextTick();
}
const execs = game.execs;
console.error(\`\${target} execs=\${execs.length}\`);
execs.forEach((e, i) => console.error(\`\${i} \${execLabel(e)}\`));
" -- "$RECORD" "$TURN"
