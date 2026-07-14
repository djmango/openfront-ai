# Rust workspace

## Architecture

| Layer | Role | Hash-verified |
|-------|------|---------------|
| **TS engine** (`--backend ts`) | Full OpenFront via `hash_verify.ts` | Yes |
| **Engine daemon** (`OPENFRONT_DAEMON=1`, default) | One `tsx` process, multiplexed RL envs | Yes (same engine) |
| **Native port** (`--backend native` / `oftrain --engine native`) | In-process Rust sim | Outcome parity strong at bots≤10; residual gaps at bots=30+ |
| **Stub** (`OPENFRONT_STUB=1`) | Minimal offline tests | No |

## Build

```bash
export OPENFRONT_REPO=/path/to/openfront-ai   # needs openfront/node_modules
export PATH="/workspace/.venv/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

cargo build --release -p openfront-engine -p oftrain -p ofhub   # from rust/
cargo test -p oftrain --bin oftrain

# Parity checks
./scripts/parity_check.sh 5
```

## Versioned curricula

No flag selects the 11-stage legacy schedule. `--v81-curriculum` preserves
the existing 11-stage V8.1 schedule and sizing. The opt-in
`--v811-curriculum` selects schedule identity `v8.1.1`:

| Stage | Maps | Bots | Difficulty | Win gate | Envs/GPU |
|-------|------|------|------------|----------|----------|
| 4 | Pangaea, Caucasus, BlackSea | 30 | Easy | 0.35 | 24 |
| 5 | BlackSea, BetweenTwoSeas, Caucasus | 30 | Easy | 0.30 | 24 |
| 6 | BlackSea, BetweenTwoSeas, Caucasus | 30 | Medium | 0.25 | 24 |
| 7 | World, Asia, BlackSea | 50 | Medium | 0.25 | 12 |
| 8 | World, Asia, BetweenTwoSeas, Caucasus | 80 | Medium | 0.22 | 10 |
| 9–11 | Existing all-map Hard/Impossible progression | 80/120/150 | Hard/Hard/Impossible | 0.20/0.18/0.15 | 8 |

Checkpoint state records the schedule identity and cross-schedule resume is
rejected. The sole migration is V8.1 stage 5 to V8.1.1 stage 5:
`--v811-curriculum --resume PATH --migrate-v81-stage5-to-v811`.

## Native port / training gaps

Native is validated end-to-end for early curriculum stages (bots 0/5/10:
outcome parity ~67–100% vs TS). **Remaining gaps that matter for training:**

| Gap | Notes |
|-----|--------|
| **bots=30+ outcome parity** | Crowded FFA / `nations: default` — wrong narrow leader; see `docs/curriculum-parity-report.md` |
| **UI-only Game APIs** | `buildable_units` / `can_attack_tile` etc. deliberately unported |
| **Archive provenance** | A few `#[ignore]`d replay tests disagree with archived hashes despite matching live TS |

Trade-ship warship piracy is **ported** (`WarshipExecution::hunt_trade_ship` →
`Game::capture_unit`; `TradeShipExecution` detects owner change, sets
`was_captured`, redirects to the capturer's nearest port, and pays gold to
the pirate on voyage complete).

**Hedge for training:** `oftrain --node-fraction 0.2` keeps a fraction of env
workers on the Node/TS engine so ground-truth episodes still flow while most
ticking stays on native (~10× faster).

## oftrain notes

| Item | Status |
|------|--------|
| AE encoders | `scripts/export_safetensors.py` + `fetch_ae_encoders.sh` → `--ckpt` / `--coarse-ckpt` |
| Checkpoints | `.safetensors` + `manifest.json` / `*.state.json` (legacy `.ot` explicit only) |
| Value loss | default `mse`; `--value-loss huber` escape hatch |
| Pipelining | `--pipeline-groups` (default on), `--fp16-rollout` (opt-in) |

Also: dual env-group pipelining, `--resume-warmup-updates` (Adam moments not
restorable in tch).

## Native port progress (engine)

| Milestone | Status |
|-----------|--------|
| PRNG + `nextID` bit-identical to TS | **Done** |
| Record bootstrap (humans + nations + tribes) | **Done** |
| Core executions (attack, nuke, warship, nation AI, …) | **Mostly done** |
| Trade-ship warship piracy | **Done** |
| Curriculum outcome gate bots≤10 | **Pass / strong** |
| Curriculum outcome gate bots=30+ | **Residual gap** — use `--node-fraction` |
| Full 285-game hash suite | Prefer TS oracle; native improving |

Run `./scripts/port_status.sh` for LOC ratio.

## Performance

- `engine_daemon` - eliminates per-env `tsx` spawn (default for TS bridge envs)
- Set `OPENFRONT_DAEMON=0` to use legacy one-subprocess-per-env bridge
- `oftrain --engine native` - in-process tick (~10× vs Node bridge)
