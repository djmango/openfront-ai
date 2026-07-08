/**
 * Multiplexed engine daemon - one Node process, many RL env sessions.
 *
 * stdin JSONL:
 *   {"op":"new"} → {"sid":"s0"}
 *   {"op":"reset","sid":"s0","map":"Onion","seed":"x","bots":8}
 *   {"op":"step","sid":"s0","intents":[],"ticks":10}
 *   {"op":"drop","sid":"s0"}
 *   {"op":"shutdown"}
 *
 * Module stays loaded; no per-env tsx spawn. Used by ofenv when OPENFRONT_DAEMON=1.
 */
import * as readline from "readline";
import type { Intent } from "../openfront/src/core/Schemas";
import { EnvSession } from "./session";

console.log = console.info = console.warn = (...args: unknown[]) =>
  process.stderr.write(args.map(String).join(" ") + "\n");

async function main() {
  const sessions = new Map<string, EnvSession>();
  let nextId = 0;
  const rl = readline.createInterface({ input: process.stdin });
  const write = (obj: object) => process.stdout.write(JSON.stringify(obj) + "\n");
  const writeObs = (
    parts: { head: Record<string, unknown>; tiles: Buffer },
    extra: object = {},
  ) => {
    process.stdout.write(
      JSON.stringify({ ...parts.head, ...extra, tilesBin: parts.tiles.length }) + "\n",
    );
    process.stdout.write(parts.tiles);
  };

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      op: string;
      sid?: string;
      map?: string;
      seed?: string;
      bots?: number;
      difficulty?: string;
      nations?: number | "default" | "disabled";
      intents?: Intent[];
      ticks?: number;
      path?: string;
    };
    try {
      msg = JSON.parse(line);
    } catch {
      write({ error: "bad json" });
      continue;
    }
    try {
      if (msg.op === "new") {
        const sid = `s${nextId++}`;
        sessions.set(sid, new EnvSession());
        write({ sid });
      } else if (msg.op === "drop") {
        if (msg.sid) sessions.delete(msg.sid);
        write({ ok: true });
      } else if (msg.op === "shutdown") {
        break;
      } else if (msg.op === "reset" || msg.op === "step" || msg.op === "save_record") {
        const sid = msg.sid;
        if (!sid || !sessions.has(sid)) {
          write({ error: `unknown sid ${sid ?? ""}` });
          continue;
        }
        const session = sessions.get(sid)!;
        if (msg.op === "reset") {
          const parts = await session.reset(
            msg.map ?? "Onion",
            msg.seed ?? "0",
            msg.bots ?? 100,
            msg.difficulty ?? "Medium",
            msg.nations ?? "default",
          );
          writeObs(parts, session.terrain());
        } else if (msg.op === "step") {
          writeObs(session.step(msg.intents ?? [], msg.ticks ?? 10));
        } else {
          write(session.saveRecord(msg.path ?? "/tmp/openfront_record.json"));
        }
      } else {
        write({ error: `unknown op ${msg.op}` });
      }
    } catch (err) {
      write({ error: String(err instanceof Error ? err.stack : err) });
    }
  }
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
