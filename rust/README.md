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

cargo build --release -p openfront-engine -p oftrain -p ofhub -p ofae   # from rust/
cargo test -p oftrain --bin oftrain

# AE train (needs cached games under data/)
cargo run --release -p ofae -- train --data ../data --steps 100 --out /tmp/ofae_smoke

# Parity checks
./scripts/parity_check.sh 5
```

## Versioned curricula

`oftrain` is V10-only. There are no legacy/V8/V9 curriculum selector flags;
stage identity is always persisted as `curriculum_schedule: "v10"` with reward
profile `v10-anti-spiral-v1`.

V10 is a 100-stage anti-death-spiral ladder: a long Onion Easy micro-ramp,
closeout and bridge stages, then broad-map Easy/Medium/Hard/Impossible bands.
The live reward recipe keeps the historical `--v81-*`, `--v84-*`,
`--v85-*`, and `--v86-*` knob names, but those knobs are the V10 defaults.

Normal resume requires a V10 sidecar. The only supported legacy migration is
from a V8.3-schedule checkpoint with V8.6 reward profile:
`--resume PATH --migrate-v86-to-v10`.

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

**Parity hedge (opt-in only):** production training is pure-native
(`--node-fraction 0`). To mix slower Node/TS envs for ground-truth episodes,
pass `--allow-node-mix --node-fraction 0.2` (or `ALLOW_NODE_MIX=1 NODE_FRACTION=0.2`
via `pod_train_v10.sh`). Accidental non-zero mixes are rejected.

## oftrain notes

| Item | Status |
|------|--------|
| AE encoders | `ofae train` / `ofae export-encoder` + `fetch_ae_encoders.sh` → `--ckpt` / `--coarse-ckpt` |
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
| Curriculum outcome gate bots=30+ | **Residual gap** — opt-in `--allow-node-mix --node-fraction` |
| Full 285-game hash suite | Prefer TS oracle; native improving |

Run `./scripts/port_status.sh` for LOC ratio.

## Performance

- `engine_daemon` - eliminates per-env `tsx` spawn (default for TS bridge envs)
- Set `OPENFRONT_DAEMON=0` to use legacy one-subprocess-per-env bridge
- `oftrain --engine native` - in-process tick (~10× vs Node bridge)
