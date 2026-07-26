# Parity work playbook

**Single source of truth for “are we at parity?”:**
`scripts/run_outcome_gate.sh` / the `outcome_gate` binary.

A record **passes** when native and the TS oracle agree on:

| check | tolerance (`compare_outcomes` in `rust/engine/src/replay.rs`) |
|-------|----------------------------------------------------------------|
| **winner** identity | exact match (or agreed stalemate / no-winner with identical rankings) |
| **terminal tick** | within **20%** relative |
| **winner land share** | within **0.10** absolute |

That is the merge bar for engine / pin / human-game work. Do **not** treat
archived-hash bit-exactness (`multi_record_parity_report`,
`find_first_divergence`) or tick-dump bisects as the pass/fail gate — those
are **diagnostics** for *why* an outcome failed (or for functional
completeness while porting a subsystem).

## Priority order

1. **Outcome gate** on the record set that matches the OpenFront pin under
   test (live-tip human bucket when the pin is the live tip; curriculum /
   frozen archive when intentionally testing those).
2. **Functional completeness** — port TS unit tests for silent no-op
   subsystems (nukes, warships, …). Still required for training quality;
   not a substitute for the outcome gate.
3. **Diagnostics** — `bisect_parity.sh`, hash checkpoints, tick dumps.
   Use only after an outcome fail (or while hunting a specific mechanic).

## Tooling

- **`scripts/run_outcome_gate.sh`** — TS outcome oracle cache + native
  compare. Set `PARITY_COMMIT` to the records directory name (and matching
  openfront checkout via `ensure_parity_openfront.sh`).
- **`scripts/run_curriculum_parity_gate.sh`** — same metric on the
  curriculum self-play set (current pin vs current TS).
- **`scripts/fetch_latest_human_games.sh`** — pull Public games hashed on
  the *currently deployed* tip only (see also `fetch_games.py --git-commit`).
- **`scripts/bisect_parity.sh`** — diagnostic only; find first tick/field
  diverge after an outcome fail.
- **`multi_record_parity_report` / `find_first_divergence`** — diagnostic
  hash checks on archived fixtures; **not** the merge bar.

## Dispatch checklist

- [ ] Measure with `outcome_gate` (or curriculum gate) before claiming
      parity or merging engine/pin changes.
- [ ] Do not use bit-exact hash pass rate as the merge criterion.
- [ ] Keep tip human games in one commit bucket; don’t mix pins.
- [ ] No `git stash` across concurrent worktrees.
