/**
 * Shared datagen helpers: map loading and full-entity snapshotting.
 * Used by generate.ts (bot self-play) and replay.ts (archived human games).
 */
import * as fs from "fs";
import * as path from "path";
import {
  Game,
  GameMapSize,
  GameMapType,
  TeamGameSpawnAreas,
} from "../openfront/src/core/game/Game";
import {
  GameMapLoader,
  MapData,
} from "../openfront/src/core/game/GameMapLoader";
import {
  AdditionalNation,
  genTerrainFromBin,
  MapManifest,
  Nation,
} from "../openfront/src/core/game/TerrainMapLoader";
import { GameMap } from "../openfront/src/core/game/GameMap";

const REPO_ROOT = path.join(__dirname, "..");
const MAPS_DIR = path.join(REPO_ROOT, "openfront", "resources", "maps");

export class NodeMapLoader implements GameMapLoader {
  getMapData(map: GameMapType): MapData {
    const key = Object.keys(GameMapType).find(
      (k) => GameMapType[k as keyof typeof GameMapType] === map,
    );
    if (!key) throw new Error(`Unknown map: ${map}`);
    const dir = path.join(MAPS_DIR, key.toLowerCase());
    return {
      mapBin: async () =>
        new Uint8Array(fs.readFileSync(path.join(dir, "map.bin"))),
      map4xBin: async () =>
        new Uint8Array(fs.readFileSync(path.join(dir, "map4x.bin"))),
      map16xBin: async () =>
        new Uint8Array(fs.readFileSync(path.join(dir, "map16x.bin"))),
      manifest: async () =>
        JSON.parse(
          fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
        ) as MapManifest,
      webpPath: "",
    };
  }
}

export interface FreshTerrain {
  nations: Nation[];
  additionalNations: AdditionalNation[];
  gameMap: GameMap;
  miniGameMap: GameMap;
  teamGameSpawnAreas?: TeamGameSpawnAreas;
}

/**
 * Load a fresh (uncached, unowned) terrain map. Deliberately bypasses
 * loadTerrainMap(): it caches the mutable GameMap object across games, so a
 * second game in the same process would inherit the first game's ownership.
 * Replicates its GameMapSize handling: Compact games run on the 4x-downscaled
 * binary with nation coordinates and spawn areas halved.
 */
export async function loadFreshTerrain(
  mapType: GameMapType,
  mapSize: GameMapSize,
): Promise<FreshTerrain> {
  const loader = new NodeMapLoader().getMapData(mapType);
  const manifest = await loader.manifest();

  const gameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map, await loader.mapBin())
      : await genTerrainFromBin(manifest.map4x, await loader.map4xBin());
  const miniGameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map4x, await loader.map4xBin())
      : await genTerrainFromBin(manifest.map16x, await loader.map16xBin());

  if (mapSize === GameMapSize.Compact) {
    for (const nation of [
      ...manifest.nations,
      ...(manifest.additionalNations ?? []),
    ]) {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    }
  }

  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (mapSize === GameMapSize.Compact && teamGameSpawnAreas) {
    const scaled: TeamGameSpawnAreas = {};
    for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
      scaled[key] = areas.map((a) => ({
        ...a,
        x: Math.floor(a.x / 2),
        y: Math.floor(a.y / 2),
      }));
    }
    teamGameSpawnAreas = scaled;
  }

  return {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}

/** Full entity state for one snapshot: everything the observation stack reads. */
export function snapshotEntities(game: Game): object {
  const players = game.players().map((p) => ({
    id: p.smallID(),
    name: p.name(),
    type: p.type(),
    troops: Math.round(p.troops()),
    gold: p.gold().toString(),
    tiles: p.numTilesOwned(),
    alive: p.isAlive(),
    traitor: p.isTraitor(),
    disconnected: p.isDisconnected(),
    targets: p.targets().map((t) => t.smallID()),
    embargoes: p.getEmbargoes().map((e) => e.target.smallID()),
    // Incoming/outgoing alliance requests (pending only).
    reqsIn: p.incomingAllianceRequests().map((r) => r.requestor().smallID()),
    reqsOut: p.outgoingAllianceRequests().map((r) => r.recipient().smallID()),
    // Sparse: only players this one has a stored (non-default) relation with.
    relations: p
      .allRelationsSorted()
      .map((r) => [r.player.smallID(), r.relation]),
  }));

  const alliances: number[][] = [];
  const seenAlliance = new Set<string>();
  for (const p of game.players()) {
    for (const a of p.alliances()) {
      const x = a.requestor().smallID();
      const y = a.recipient().smallID();
      const key = x < y ? `${x}:${y}` : `${y}:${x}`;
      if (!seenAlliance.has(key)) {
        seenAlliance.add(key);
        alliances.push([x, y, a.expiresAt()]);
      }
    }
  }

  const units = game.players().flatMap((p) =>
    p.units().map((u) => {
      const tt = u.targetTile();
      return {
        type: u.type(),
        owner: p.smallID(),
        x: game.x(u.tile()),
        y: game.y(u.tile()),
        // Destination for units in transit (nukes, transports, warships):
        // where it will land matters more than where it currently is.
        tx: tt !== undefined ? game.x(tt) : null,
        ty: tt !== undefined ? game.y(tt) : null,
        samLock: u.targetedBySAM(),
        level: u.level(),
        health: u.hasHealth() ? u.health() : null,
        constructing: u.isUnderConstruction(),
        cooldown: u.isInCooldown(),
        troops: Math.round(u.troops()),
      };
    }),
  );

  const attacks = game.players().flatMap((p) =>
    p.outgoingAttacks().map((a) => {
      const src = a.sourceTile();
      return {
        from: p.smallID(),
        to: a.target().isPlayer() ? a.target().smallID() : 0,
        troops: Math.round(a.troops()),
        retreating: a.retreating(),
        srcX: src !== null ? game.x(src) : null,
        srcY: src !== null ? game.y(src) : null,
      };
    }),
  );

  return { players, alliances, units, attacks };
}
