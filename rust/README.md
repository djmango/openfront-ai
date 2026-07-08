# Rust workspace (`rust-ofrs-fast` worktree)

## Architecture

| Layer | Role | Hash-verified |
|-------|------|---------------|
| **TS engine** (`--backend ts`) | Full OpenFront via `hash_verify.ts` | Yes |
| **Engine daemon** (`OPENFRONT_DAEMON=1`, default) | One `tsx` process, multiplexed RL envs | Yes (same engine) |
| **Native port** (`--backend native`) | Growing Rust sim + record bootstrap | In progress |
| **Stub** (`OPENFRONT_STUB=1`) | Minimal offline tests | No |

## Build

```bash
export OPENFRONT_REPO=/path/to/openfront-ai   # needs openfront/node_modules

cargo build --release -p openfront-engine    # from rust/
cd rust/ofenv && maturin develop --release

# Parity checks
PYTHONPATH=. python scripts/env_parity.py
./scripts/parity_check.sh 5
```

## Native port progress

| Milestone | Status |
|-----------|--------|
| PRNG + `nextID` bit-identical to TS | **Done** |
| Record bootstrap (humans + nations + tribes) | **Done** |
| Hash at tick 0 on `jby2gMJF` | **PASS** |
| Hash at tick 10+ | Desync - executions not yet ported |
| Full 285-game hash suite | TS backend only |

Run `./scripts/port_status.sh` for LOC ratio.

## Performance

- `ofrs` - GIL-free BC collate/decode (~2.4× collate)
- `engine_daemon` - eliminates per-env `tsx` spawn (default for `ofenv`)
- Set `OPENFRONT_DAEMON=0` to use legacy one-subprocess-per-env bridge
