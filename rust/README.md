# Rust workspace (`rust-ofrs-fast` worktree)

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

cargo build --release -p openfront-engine    # from rust/
cargo test -p oftrain --bin oftrain
cd rust/ofenv && maturin develop --release

# Parity checks
PYTHONPATH=. python scripts/env_parity.py
./scripts/parity_check.sh 5
```

## Native port / training gaps

Native is validated end-to-end for early curriculum stages (bots 0/5/10:
outcome parity ~67–100% vs TS). **Remaining gaps that matter for training:**

| Gap | Notes |
|-----|--------|
| **bots=30+ outcome parity** | Crowded FFA / `nations: default` — wrong narrow leader; see `docs/curriculum-parity-report.md` |
| **Trade-ship warship piracy** | Capture of in-transit trade ships by warships not modeled in `trade_ship_execution.rs` (owner never changes mid-voyage via piracy) |
| **UI-only Game APIs** | `buildable_units` / `can_attack_tile` etc. deliberately unported |
| **Archive provenance** | A few `#[ignore]`d replay tests disagree with archived hashes despite matching live TS |

**Hedge for training:** `oftrain --node-fraction 0.2` keeps a fraction of env
workers on the Node/TS engine so ground-truth episodes still flow while most
ticking stays on native (~10× faster).

## oftrain Python-parity plan (phased)

| Phase | Status | What |
|-------|--------|------|
| 0 | Done | `scripts/export_safetensors.py`, `fetch_ae_encoders.sh` |
| 1 | Done | Frozen AE encode (`C_GRID=89`), `--ckpt`/`--coarse-ckpt`, foveate default on |
| 2 | Done | `MAX_UPD_PIX` sub-batches, greedy `--eval-every`, `metrics.jsonl` |
| 3 | Done | `--init path.ot` warm-start (see `scripts/convert_policy_pt_notes.md`) |
| 4 | Docs | Native gaps + `--node-fraction` hedge (this section) |
| **5 (final)** | Partial | `--value-loss huber\|mse` (default **huber** for stability). **Flip default to `mse`** once training is stable, matching Python `F.mse_loss` |

Still skipped (documented in `oftrain` module docs): dual env-group pipelining,
AdamW optimizer-state restore (tch limitation), fp16 host grids.

## Native port progress (engine)

| Milestone | Status |
|-----------|--------|
| PRNG + `nextID` bit-identical to TS | **Done** |
| Record bootstrap (humans + nations + tribes) | **Done** |
| Core executions (attack, nuke, warship, nation AI, …) | **Mostly done** |
| Curriculum outcome gate bots≤10 | **Pass / strong** |
| Curriculum outcome gate bots=30+ | **Residual gap** — use `--node-fraction` |
| Full 285-game hash suite | Prefer TS oracle; native improving |

Run `./scripts/port_status.sh` for LOC ratio.

## Performance

- `ofrs` - GIL-free BC collate/decode (~2.4× collate)
- `engine_daemon` - eliminates per-env `tsx` spawn (default for `ofenv`)
- Set `OPENFRONT_DAEMON=0` to use legacy one-subprocess-per-env bridge
- `oftrain --engine native` - in-process tick (~10× vs Node bridge)
