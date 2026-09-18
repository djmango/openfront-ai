/**
 * Bot-spawn probe for the seed -> game-id fix.
 *
 * `seedToGameID` seeds `PseudoRandom` (the spawn path mixes
 * `simpleHash(playerID)` with `simpleHash(gameID)`), so if the native id
 * derivation disagreed with TS the BOT SPAWN TILES differ and the whole
 * trajectory diverges from the first decision. Tile *counts* agreeing is
 * weak evidence; the spawn tile itself is the thing to compare.
 *
 * This drives the same stage-0 episode as `ts_parity_trace.ts`
 * (bridge/session.ts EnvSession: map / seed / bots / difficulty / nations,
 * n_agents=1) and prints, for every player that has spawned:
 *   small_id, player type, spawn tile, spawn tile (x,y), tiles owned
 * in the exact shape `rust/engine/src/bin/prng_dump.rs` prints its `spawn`
 * lines, so the two can be diffed field by field.
 *
 * Usage:
 *   openfront/node_modules/.bin/tsx scripts/ts_spawn_probe.ts [seed ...]
 * (default: parity alpha bravo charlie, the four seeds prng_dump uses)
 */
import { EnvSession } from "../bridge/session";
import { PlayerType } from "../openfront/src/core/game/Game";

const SEEDS = ["parity", "alpha", "bravo", "charlie"];
const MAP = "Pangaea";
const BOTS = 2;
const DIFFICULTY = "Easy";
const NATIONS: number | "default" | "disabled" = 0;

function typeTag(t: PlayerType): string {
  switch (t) {
    case PlayerType.Human:
      return "H";
    case PlayerType.Bot:
      return "B";
    default:
      return "N";
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const seeds = argv.length > 0 ? argv : SEEDS;
  for (const seed of seeds) {
    const session = new EnvSession();
    await session.reset(MAP, seed, BOTS, DIFFICULTY, NATIONS);
    const game = session.game;

    // Advance tick by tick until every bot has spawned (bounded), mirroring
    // prng_dump's `while ticks < 20 && bots_pending`.
    let ticks = 0;
    while (ticks < 20) {
      const bots = game.allPlayers().filter((p) => p.type() === PlayerType.Bot);
      if (bots.length >= BOTS && bots.every((p) => p.hasSpawned())) break;
      session.step([], 1);
      ticks++;
    }

    const players = game
      .allPlayers()
      .slice()
      .sort((a, b) => a.smallID() - b.smallID());
    console.log(
      JSON.stringify({
        seed,
        game_id: session.gameID,
        ticks_to_spawn: ticks,
        spawned_all: players
          .filter((p) => p.type() === PlayerType.Bot)
          .every((p) => p.hasSpawned()),
        players: players.map((p) => {
          const tile = p.spawnTile();
          return {
            small_id: p.smallID(),
            type: typeTag(p.type()),
            id: p.id(),
            spawned: p.hasSpawned(),
            tile: tile ?? null,
            x: tile === undefined ? null : tile % game.width(),
            y: tile === undefined ? null : Math.floor(tile / game.width()),
            tiles_owned: p.numTilesOwned(),
            troops: Math.round(p.troops()),
          };
        }),
      }),
    );
  }
}

main().catch((e) => {
  process.stderr.write(String(e?.stack ?? e) + "\n");
  process.exit(1);
});