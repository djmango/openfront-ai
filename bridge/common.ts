/**
 * Shared between bridge/env.ts (offline RL env) and bridge/play.ts (live
 * multiplayer agent client): observation payload construction from a Game.
 *
 * The featurization logic itself (entities/legality/border checks) lives in
 * openfront/src/client/webbot/obsCore.ts so the in-browser WebBot can use
 * the identical, single-source-of-truth implementation; this file just adds
 * the Node-only (Buffer/zlib) wire-format wrappers on top.
 */
import * as zlib from "zlib";
import { Game } from "../openfront/src/core/game/Game";
import {
  bordersNeutralLand,
  entities,
  hasShoreBorder,
  LAUNCHABLE,
  legality,
  STRUCTURES,
} from "../openfront/src/client/webbot/obsCore";

export { bordersNeutralLand, entities, hasShoreBorder, LAUNCHABLE, legality, STRUCTURES };

export function buildObsParts(
  game: Game,
  clientID: string,
  winner: unknown,
): { head: Record<string, unknown>; tiles: Buffer } {
  const numTiles = game.width() * game.height();
  const tiles = Buffer.from(
    game.tileStateBuffer().buffer,
    game.tileStateBuffer().byteOffset,
    numTiles * 2,
  );
  const agent = game.playerByClientID(clientID) ?? null;
  return {
    head: {
      tick: game.ticks(),
      width: game.width(),
      height: game.height(),
      spawnPhase: game.inSpawnPhase(),
      winner,
      me: agent?.smallID() ?? -1,
      alive: agent?.isAlive() ?? false,
      entities: entities(game),
      legal: legality(game, clientID),
    },
    tiles,
  };
}

export function buildObs(
  game: Game,
  clientID: string,
  winner: unknown,
): object {
  // JSON-only variant (gzip+base64 tiles) used where the transport is
  // plain JSONL (bridge/play.ts). The training env uses buildObsParts and
  // ships tiles as a raw binary frame instead.
  const { head, tiles } = buildObsParts(game, clientID, winner);
  return {
    ...head,
    tiles: zlib.gzipSync(tiles, { level: 1 }).toString("base64"),
  };
}

export function terrainPayload(game: Game): object {
  const numTiles = game.width() * game.height();
  const buf = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) buf[ref] = game.terrainByte(ref);
  return {
    terrain: zlib.gzipSync(Buffer.from(buf), { level: 6 }).toString("base64"),
  };
}
