/**
 * Hash-verify one GameRecord file (stdout JSON). Used by openfront-replay --backend ts.
 */
import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as zlib from "zlib";
import { decompressGameRecord } from "../openfront/src/core/Util";
import type { GameRecord } from "../openfront/src/core/Schemas";
import { replayGame } from "../datagen/replay";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function loadRecord(file: string): GameRecord {
  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  return decompressGameRecord(JSON.parse(json) as GameRecord);
}

function commitFromPath(file: string): string | null {
  const parts = path.resolve(file).split(path.sep);
  const i = parts.lastIndexOf("records");
  if (i >= 0 && i + 1 < parts.length) {
    return parts[i + 1];
  }
  return null;
}

function pinEngine(commit: string): () => void {
  const pin = execSync("git -C openfront rev-parse HEAD", {
    cwd: REPO,
    encoding: "utf8",
  }).trim();
  const checkout = (rev: string) => {
    try {
      execSync(`git -C openfront checkout -q ${rev}`, { cwd: REPO, stdio: "pipe" });
    } catch {
      execSync("git -C openfront fetch origin --quiet", {
        cwd: REPO,
        stdio: "inherit",
      });
      execSync(`git -C openfront checkout -q ${rev}`, { cwd: REPO, stdio: "pipe" });
    }
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
  if (!file) throw new Error("usage: hash_verify.ts <record>");
  const record = loadRecord(file);
  const commit =
    record.gitCommit?.slice(0, 12) ??
    commitFromPath(file)?.slice(0, 12) ??
    null;
  if (!commit) throw new Error("no engine commit");
  const restore = pinEngine(commit);
  try {
    const outDir = `/tmp/of-hash-verify-${record.info.gameID}`;
    const result = await replayGame(record, outDir, 10_000_000, false, false);
    process.stdout.write(
      JSON.stringify({
        ok: result.ok,
        reason: result.reason ?? null,
        ticks: result.ticks,
        hashes_checked: result.hashesChecked,
        engine_commit: commit,
      }),
    );
  } finally {
    restore();
  }
}

main().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.stdout.write(
    JSON.stringify({
      ok: false,
      reason: String(e),
      ticks: 0,
      hashes_checked: 0,
    }),
  );
  process.exit(1);
});
