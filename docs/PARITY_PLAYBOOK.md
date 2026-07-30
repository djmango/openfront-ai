# Parity work playbook

**Single source of truth for tip full parity (bit / hash):**
`scripts/hash_parity.sh` / `scripts/run_hash_parity_gate.sh`.

A record **passes hash parity** when native and tip TS agree every tick on:

| layer | fields |
|-------|--------|
| **alive / tiles** | exact |
| **units** | `unitsHash`, `numUnits` (expand to per-unit dump on diverge) |
| **player hash** | `hashBits` (IEEE-754), then truncated `hash` |
| **game hash** | `gameHashBits` (IEEE-754) — do not trust JSON `gameHash` ints past 2^53 |
| troops / gold | reported; treated soft under rounding noise |

**Outcome gate** (`scripts/run_outcome_gate.sh`) remains the *merge bar for
shipping a pin* when full hash parity is not yet done: winner + terminal tick
tolerance + land share. Tip full-parity work should measure with
`hash_parity` first, then use outcome only as a coarse smoke.

## Priority order

1. **Hash parity probe** — `scripts/hash_parity.sh <record>`:
   parallel native+TS NDJSON streams, early-stop at first diverge, auto
   unit expand when the layer is units/hash.
2. **Hash parity gate** — `scripts/run_hash_parity_gate.sh` over the tip
   record set (`PARITY_COMMIT=…`). Reports first diverge tick/layer per game.
3. **Outcome gate** — winner-level smoke / pin merge bar when hash gate is
   not yet green.
4. **Functional completeness** — port TS unit tests for silent no-op
   subsystems.
5. **Legacy bisect** — `scripts/bisect_parity.sh` only if you need the old
   coarse/fine JSON dumps; prefer `hash_parity.sh`.

## Why not “bisect over and over”

Old loop (dumb):

1. Coarse dump both engines from tick 0 → JSON
2. Diff → window
3. Fine dump both engines from tick 0 again → JSON
4. Maybe re-run with `OF_DUMP_UNITS` from tick 0 a third time

New loop:

1. `scripts/hash_parity.sh records/<commit>/<game>.json.gz`
2. Native + TS stream NDJSON **in parallel**, compare online, **kill both**
   at first diverge
3. If layer is units/hash: one expand dump near the window with units

**Mid-game resume (dump daemons):** `tick_dump --daemon` and
`dump_ts_tick_state.ts --daemon` accept `ADVANCE` / `RESET` / `DUMP` on stdin.
`ADVANCE` only moves forward (in-memory resume). Use:

- `scripts/hash_bisect.sh` — true binary search with resume (sticky diverge)
- `scripts/hash_parity.sh` expand — daemon `ADVANCE` to diverge tick + unit `DUMP`

Streaming every=1 early-stop remains the fastest way to *find* the first
diverge tick in one pass; bisect helps on long horizons when you want
logarithmic probes with warm engines.

**Join on player `id`, never `identity` alone.** Bot identities
(`nation:Name`) collide; identity-join hides real field diffs and can look
like a mysterious `gameHash`-only miss. Prefer `gameHashBits` / `hashBits`
over JSON int hashes past `2^53`.

## Tooling

- **`scripts/hash_parity.sh`** — primary tip diagnostic (hash/bit, streaming).
- **`scripts/hash_bisect.sh`** — daemon binary-search with mid-game resume.
- **`scripts/run_hash_parity_gate.sh`** — multi-record hash gate (`HASH_PARITY_EVERY`, `HASH_PARITY_USE_BISECT=1`).
- **`scripts/stream_compare_ticks.py`** — online NDJSON comparator.
- **`tick_dump --daemon`** / `dump_ts_tick_state.ts --daemon` — ADVANCE/RESET/DUMP.
- **`scripts/run_outcome_gate.sh`** — winner-level pin merge bar.
- **`scripts/run_curriculum_parity_gate.sh`** — curriculum self-play outcome.
- **`scripts/bisect_parity.sh`** — legacy coarse/fine JSON bisect.
- **`tick_dump --ndjson`** / `OF_DUMP_NDJSON=1` on TS dump — streaming snapshots.
- Always-on dump fields: `hash`, `hashBits`, `unitsHash`, `numUnits`, `gameHash`,
  `gameHashBits` (IEEE-754 bit patterns as decimal strings — required once hashes
  exceed `Number.MAX_SAFE_INTEGER`).
- Comparators join on player **`id`**, not `identity`.

## Dispatch checklist

- [ ] For tip full-parity: measure with `hash_parity` / hash gate, not only outcome.
- [ ] Prefer `hash_parity.sh` over manual bisect loops.
- [ ] Keep tip human games in one commit bucket; don’t mix pins.
- [ ] No `git stash` across concurrent worktrees.
- [ ] Outcome gate alone is not “bit parity.”
