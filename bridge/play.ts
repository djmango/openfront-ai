/**
 * Live multiplayer agent client: joins a real OpenFront lobby over websocket
 * (the same protocol the browser client speaks), mirrors the game locally
 * from server turns, and plays intents chosen by the Python policy.
 *
 * Protocol with Python (rl/play.py) over stdio, one JSON per line:
 *   -> {"event":"lobby", ...}                    informational
 *   -> {"event":"start", width, height, terrain} once, after game start
 *   -> {"event":"obs", ...buildObs payload}      every N ticks
 *   <- {"intents":[Intent...]}                   reply to each obs
 *   -> {"event":"end", winner}                   game over
 *
 * Usage: tsx bridge/play.ts --game <ID> [--host localhost:9000] [--workers 2]
 */
import * as readline from "readline";
import { randomUUID } from "crypto";
import { Config } from "../openfront/src/core/configuration/Config";
import { DoomsdayClockExecution } from "../openfront/src/core/execution/DoomsdayClockExecution";
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
import type {
  GameStartInfo,
  Intent,
  Turn,
} from "../openfront/src/core/Schemas";
import { simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain } from "../datagen/common";
import { buildObs, terrainPayload } from "./common";

// Engine logs must not corrupt the stdout JSONL stream.
console.log = console.info = console.warn = (...args: unknown[]) =>
  process.stderr.write(args.map(String).join(" ") + "\n");
const log = (...args: unknown[]) =>
  process.stderr.write("[play] " + args.map(String).join(" ") + "\n");

const DECISION_TICKS = 10;

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] !== undefined
    ? process.argv[i + 1]
    : fallback;
}

const write = (obj: object) =>
  process.stdout.write(JSON.stringify(obj) + "\n");

class PlaySession {
  game!: Game;
  executor!: Executor;
  myClientID = "";
  private pendingTurns: Turn[] = [];
  private processing = false;
  private started = false;
  private ended = false;
  private stdin: AsyncIterator<string>;
  private ws!: WebSocket;

  constructor() {
    const rl = readline.createInterface({ input: process.stdin });
    this.stdin = rl[Symbol.asyncIterator]();
  }

  async run(): Promise<void> {
    const gameID = arg("game", "");
    if (!gameID) throw new Error("--game <lobbyID> required");
    const host = arg("host", "localhost:9000");
    const workers = parseInt(arg("workers", "2"), 10);
    const token = randomUUID();

    const workerPath = `w${simpleHash(gameID) % workers}`;
    const url = `ws://${host}/${workerPath}`;
    log(`connecting ${url} (game ${gameID})`);
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.ws.send(
        JSON.stringify({
          type: "join",
          token,
          gameID,
          username: "AgentRL",
          clanTag: null,
          turnstileToken: null,
        }),
      );
      log("join sent; waiting for host to start the game");
      // Server drops connections without heartbeats.
      setInterval(() => {
        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 5000);
    };
    this.ws.onclose = (ev: CloseEvent) =>
      log(`ws closed: ${ev.code} ${ev.reason}`);
    this.ws.onerror = () => log("ws error");
    this.ws.onmessage = (ev: MessageEvent) => {
      void this.onServerMessage(JSON.parse(String(ev.data)));
    };

    // Keep process alive; readline on stdin holds the loop open too.
    await new Promise(() => {});
  }

  private async onServerMessage(msg: {
    type: string;
    [k: string]: unknown;
  }): Promise<void> {
    if (msg.type === "lobby_info") {
      write({ event: "lobby", info: msg });
    } else if (msg.type === "start") {
      if (this.started) return;
      this.started = true;
      const gameStart = msg.gameStartInfo as GameStartInfo;
      this.myClientID = (msg.myClientID as string) ?? "";
      log(
        `game starting: map ${gameStart.config.gameMap}, ` +
          `${gameStart.players.length} humans, me=${this.myClientID}`,
      );
      await this.initGame(gameStart);
      write({
        event: "start",
        width: this.game.width(),
        height: this.game.height(),
        me: this.game.playerByClientID(this.myClientID)?.smallID() ?? -1,
        ...terrainPayload(this.game),
      });
      for (const t of (msg.turns as Turn[]) ?? []) this.enqueue(t);
    } else if (msg.type === "turn") {
      this.enqueue(msg.turn as Turn);
    } else if (msg.type === "desync") {
      log("SERVER REPORTS DESYNC");
    } else if (msg.type === "error") {
      log(`server error: ${JSON.stringify(msg)}`);
    }
  }

  private enqueue(turn: Turn): void {
    this.pendingTurns.push(turn);
    void this.drain();
  }

  /** Process turns strictly in order; pause to ask Python for actions. */
  private async drain(): Promise<void> {
    if (this.processing || this.ended) return;
    this.processing = true;
    while (this.pendingTurns.length > 0) {
      const turn = this.pendingTurns.shift()!;
      this.game.addExecution(...this.executor.createExecs(turn));
      const updates = this.game.executeNextTick();

      let winner: unknown = null;
      const winUpdates = updates[GameUpdateType.Win];
      if (winUpdates && winUpdates.length > 0) {
        winner = (winUpdates[0] as { winner: unknown }).winner;
      }

      const me = this.game.playerByClientID(this.myClientID);
      const dead = !this.game.inSpawnPhase() && me !== null && !me.isAlive();
      if (winner !== null || dead) {
        write({ event: "end", winner, alive: !dead });
        this.ended = true;
        break;
      }

      if (this.game.ticks() % DECISION_TICKS === 0) {
        write({ event: "obs", ...buildObs(this.game, this.myClientID, null) });
        const reply = await this.stdin.next();
        if (reply.done) {
          this.ended = true;
          break;
        }
        const { intents } = JSON.parse(reply.value) as { intents: Intent[] };
        for (const intent of intents ?? []) {
          this.ws.send(JSON.stringify({ type: "intent", intent }));
        }
      }
    }
    this.processing = false;
  }

  /** Mirror createGameRunner() so the local sim matches every client. */
  private async initGame(gameStart: GameStartInfo): Promise<void> {
    const config = new Config(gameStart.config, null, false);
    const terrain = await loadFreshTerrain(
      gameStart.config.gameMap as never,
      gameStart.config.gameMapSize,
    );

    const random = new PseudoRandom(simpleHash(gameStart.gameID));
    const humans = gameStart.players.map(
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
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      humans.length,
      random,
    );
    this.game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      config,
      terrain.teamGameSpawnAreas,
    );
    this.executor = new Executor(this.game, gameStart.gameID, this.myClientID);

    if (gameStart.config.gameType !== GameType.Singleplayer) {
      this.game.addExecution(new SpawnTimerExecution());
    }
    if (config.spawnNations()) {
      this.game.addExecution(...this.executor.nationExecutions());
    }
    if (config.isRandomSpawn()) {
      this.game.addExecution(...this.executor.spawnPlayers());
    }
    if (config.bots() > 0) {
      this.game.addExecution(...this.executor.spawnTribes(config.bots()));
    }
    this.game.addExecution(new WinCheckExecution());
    if (config.doomsdayClockConfig().enabled) {
      this.game.addExecution(new DoomsdayClockExecution());
    }
    if (!config.isUnitDisabled(UnitType.Factory)) {
      this.game.addExecution(
        new RecomputeRailClusterExecution(this.game.railNetwork()),
      );
    }
  }
}

new PlaySession().run().catch((err) => {
  console.error(err);
  process.exit(1);
});
