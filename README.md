# openfront-ai

Toward a self-play RL agent for [OpenFront.io](https://openfront.io): headless
data generation on the real game engine, a learned spatial observation
encoder, and PPO self-play over the full action surface.

**Devlog:** [docs/devlog.html](docs/devlog.html) - run ledger, timeline, bugs,
lessons, and the full AE v3.1 bake-off. **Living spec:**
[DESIGN.md](DESIGN.md).

## Status (Jul 6)

- **375k bot + 420k human** full-state snapshots; human games replayed
  deterministically from the public archive.
- **Spatial AE v3.1** concluded: latent *resolution* (1/8, not channel count)
  fixes the human/bot border-accuracy gap. Policy encoder: **`ae_v31_d8c32`**
  (32ch @ 1/8, 88.2% human / 95.5% bot borders).
- **PPO v4** training from scratch on curriculum v2 with the full 1/8 policy
  stack (learned spawns, local owner-crop bypass, GPU featurization). Cleared
  stages 0→3 in ~2h, stage 4 by ~3.6h. **`ppo_v3`** retired after reaching
  stage 4 at 60–70% rolling win rate - first genuine engine wins.
- **BC v4** on 291 cached human games (~56 ex/s GPU-bound); temporal
  transformer experiment (`bc_seq_v4`) running in parallel. Conditional
  **BC→RL** warm start when BC plateaus.
- **5.9M-param policy**, 11 win-gated curriculum stages over 7 maps, restart-proof
  cloud training, full visualization suite (real-client replays, live play).

## Architecture: compress the map, bypass the rest

The observation design went through three iterations (see `DESIGN.md`):

1. **v1** - tile-only autoencoder over ownership + terrain.
2. **v2** - one unified AE compressing *all* state (tiles, players, units,
   diplomacy) into a joint latent. Spatial recon was excellent but tiny exact
   facts fought the bottleneck: alliance pairs peaked at F1 0.67 and relative
   troop strength at 0.81, no matter how losses were weighted.
3. **v3 (current)** - **only compress what is actually big.** The AE
   compresses the map (tile ownership, terrain, fallout, static structures).
   Everything small and exact bypasses the latent: pairwise diplomacy bits,
   per-player scalars, transient units (nukes in flight with impact points,
   transports, warships), attack aggregates, legality masks.

**The lesson: a one-bit fact reconstructed at 95% is strictly worse than
reading the bit.** Autoencoders are for high-dimensional state; exact small
state should never fight the map for latent capacity.

### AE v3.1: border accuracy

Overall tile accuracy saturates near 99% (water inflates it); **border-tile
accuracy** is the honest metric. Benchmarking the bot-trained v3 on human
games exposed a 16-point domain gap (87.5% bot borders vs 71.8% human).
Mixed bot+human retraining helped; the architectural fix was **halving the
latent patch size** (1/8 resolution instead of 1/16):

| model | latent | border (human) | border (bot) |
|---|---|---|---|
| v3 bot-only | 64ch @ 1/16 | 71.8% | 87.5% |
| v3 on bot+human mix | 64ch @ 1/16 | 80.1% | 86.8% |
| v3.1 @ 1/8 res | 64ch @ 1/8 | **89.3%** | **96.1%** |
| **v3.1 d8c32 (policy)** | **32ch @ 1/8** | **88.2%** | **95.5%** |

Structure detection stayed at precision/recall 1.0 per class throughout.
The policy also gets a raw **64×64 local owner-crop** around ego territory for
exact borders where the agent acts; the latent carries global context.

Original v3 training curves and reconstructions (64ch @ 1/16):

![Training curves](assets/loss_curve_v3.png)

![World reconstruction](assets/recon_v3_world.png)

![Latent PCA](assets/latent_pca_v3_world.png)

## RL progress

Curriculum v2: 11 stages over 7 maps (Onion → Pangaea → Caucasus → …),
win-gated advancement (rolling win rate > 0.5 over last 40 on-stage episodes),
25% rehearsal against earlier maps at current difficulty, dense wins from 1v1
stage 0. Strength-index reward (land + military + economy), not raw territory.

![Curriculum progress](docs/graphs/curriculum_progress.png)

![Throughput engineering](docs/graphs/throughput.png)

![BC pipeline](docs/graphs/bc_pipeline.png)

Graphs from `scripts/make_progress_graphs.py`. Highlights:

- **Curriculum:** `ppo_v4` matched `ppo_v3`'s pace despite the heavier 1/8
  stack, learned spawns, and two mid-run restarts. Warm-started `ppo_v2c`
  stalled at stage 3 while from-scratch v3 reached stage 4.
- **Throughput:** fp16 transfers + pinned staging + prefetch took stage-3–4
  game-ticks/s from ~590 → ~2100; v4.1 async rollout/update overlap hides
  the rollout phase inside the update.
- **BC:** prefeaturized cache (`scripts/prefeaturize_bc.py`) cut sample cost
  from ~15–20 ms to ~1.5 ms, moving the bottleneck from CPU to GPU.

Sample agent replay: [assets/replay_v2_stage3.webm](assets/replay_v2_stage3.webm)
(`ppo_v2c` on stage 3 - Onion, 80 Medium bots; peaks ~13k tiles before dying
at tick 3891).

## Key learnings

Condensed from the [devlog](docs/devlog.html#lessons):

- **Make wins reachable before making them valuable.** 1v1 → 1v3 staging turned
  the win bonus from theoretical to dense; win detection was silently broken
  until Jul 6 (checked username, engine emits clientID).
- **Warm starts inherit stale habits.** `ppo_v2c` resumed v2b weights under
  the new curriculum; from-scratch `ppo_v3` overtook it in one day. Retrain
  when reward or curriculum changes materially.
- **Benchmark on the distribution you'll deploy on.** Bot data underrepresents
  human gnarl (naval invasions, enclaves, diplomacy).
- **Spatial precision can't be bought with channels.** Halving the latent patch
  beat +50% channels; don't make the latent re-encode static side-information
  (terrain) the policy already has.
- **Log the metric you care about.** Border accuracy cost one line; overall
  tile accuracy looked done at 87% while human borders were 16 points worse.
- **Pad to the batch, not the maximum.** Most of a "GPU too slow" problem was
  wasted convolution on small-map batches.
- **Watch the agent play.** Replay tooling caught the win-detection bug; curves
  never would have.

## Layout

- `datagen/` - TypeScript headless game runner. Boots the real
  (deterministic) OpenFront engine in Node, plays bot/nation games, dumps
  full-state snapshots every 10 ticks.
- `ae/` - PyTorch: dataset loaders, spatial AE (`model_v3.py`), training
  (`train_v3.py`). Earlier iterations in `model.py`/`model_v2.py`.
- `rl/` - PPO (`ppo.py`), behavior cloning (`bc.py`), obs builder (`obs.py`),
  factorized policy (`policy.py`), curriculum/reward (`curriculum.py`), env
  bridge wrapper (`env.py`), watch/play/replay tooling.
- `bridge/` - persistent Node process wrapping the engine (JSONL reset/step
  over stdio, binary tile IPC, live websocket client for human play).
- `scripts/` - prefeaturization, evals, progress graphs, client replay
  rendering, HF upload, cloud pod supervisors.
- `docs/` - devlog and training graphs.
- `openfront/` - git submodule of
  [openfrontio/OpenFrontIO](https://github.com/openfrontio/OpenFrontIO),
  pinned to a known-good engine commit.

## Setup

```bash
git submodule update --init
(cd openfront && npm install)
uv sync
```

## Generate data

```bash
# single map
openfront/node_modules/.bin/tsx datagen/generate.ts --map Onion --games 20

# the 10-map bot dataset (25 games each, 10 in parallel)
bash datagen/gen_all.sh 25 10

# human archive → deterministic replay → snapshots
bash datagen/replay_all.sh
```

Snapshots are written every 10 ticks (1s of game time). Format details in the
[dataset card](https://huggingface.co/datasets/djmango/openfront-snapshots).

## Train

```bash
# one-time: convert gzip+JSON snapshots to fast zstd caches (~10ms → ~0.5ms/sample)
PYTHONPATH=. uv run python scripts/prefeaturize.py --data data --workers 8

# spatial AE (v3.1)
uv run python -m ae.train_v3 --data data --data-human data-human \
    --steps 40000 --batch-size 64 --latent-down 8 --latent-c 32

# PPO v4 (default encoder: runs/ae_v31_d8c32/ae_v3.pt)
uv run python -m rl.ppo --name ppo_v4 --updates 10000
uv run tensorboard --logdir runs/rl

# behavior cloning on cached human games
PYTHONPATH=. uv run python scripts/prefeaturize_bc.py --data data-human --workers 8
uv run python -m rl.bc --name bc_v4 --steps 60000
```

AE details: owner IDs relabeled to static per-game spawn slots (any player
count, fixed channels); fully convolutional training on border-dense random
crops; rarity-weighted BCE *detection* for structures (count MSE collapses to
all-zeros on 99.9%-empty grids).

## Artifacts

- Bot snapshots: [djmango/openfront-snapshots](https://huggingface.co/datasets/djmango/openfront-snapshots)
  (~375k frames, 250 games, 10 maps)
- Human games: [djmango/openfront-human-games](https://huggingface.co/datasets/djmango/openfront-human-games)
  (285 hash-verified replays + raw intent records)
- Encoders: [djmango/openfront-tile-autoencoder](https://huggingface.co/djmango/openfront-tile-autoencoder)
  (`ae_v31_d8c32.pt`, `ae_v31_d8.pt`, `ae_v3.pt`)
- RL/BC checkpoints on HF under run names (`ppo_v4/policy.pt`, `bc_v4/bc.pt`, …)

## RL stack (v4)

- **`bridge/env.ts`** - persistent Node process: JSONL reset/step, binary tile
  IPC, exact legality masks from engine calls each decision step.
- **`rl/obs.py`** - frozen AE latent (32ch @ 1/8) + ego ownership planes +
  raw 64×64 local owner crop + transient-unit planes + per-player bypass +
  legality masks. 43 channels at H/8 × W/8.
- **`rl/policy.py`** - conv trunk, factorized masked heads: action type,
  player-target pointer, tile-region pointer (8×8 regions), build/nuke type,
  quantity. Learned spawn placement end-to-end.
- **`rl/ppo.py`** - PPO + GAE, entropy anneal, stage LR warmdown, win-gate
  window persisted in checkpoint, periodic fixed-seed greedy eval,
  v4.1 async rollout/update overlap.

### Watching the agent play

`rl/watch.py` runs one greedy episode and saves an engine `GameRecord` - the
same format openfront.io archives - which the **real game client** replays with
the full UI:

```bash
uv run python -m rl.watch --policy runs/rl/ppo_v4/policy.pt --stage 4 \
    --record records-rl/game.json
openfront/node_modules/.bin/tsx scripts/verify_record.ts records-rl/game.json
```

**Client video** - `scripts/render_client_replay.py` replays the record in the
actual OpenFront client (headless Chromium): full game UI, terrain art, units,
boats, nukes, leaderboard. Client hooks give the agent **AGENT** identity on
the leaderboard, gold spawn ring, crown when first, "You Won!" modal. With a
`.debug.json` sidecar (written by `rl.watch --record`), the video also gets a
live MODEL panel synced to the sim tick:

```bash
uv run playwright install chromium   # one-time
uv run python scripts/render_client_replay.py \
    --record records-rl/game.json --out replays/game_client.webm
```

The bridge mirrors the client's `createGameRunner()` init exactly, so records
replay bit-identically: `verify_record.ts` re-simulates from intents alone.

### Playing against the agent

`bridge/play.ts` speaks the real client websocket protocol:

```bash
(cd openfront && npm run dev)      # client :9000 + game server
uv run python -m rl.play --policy runs/rl/ppo_v4/policy.pt --game <LOBBY_ID>
# "AgentRL" appears in the lobby; Start Game and fight it.
```

The MODEL panel tracks the agent in real time on `--debug-port` (default 8988).

## Roadmap

1. ~~Headless datagen + spatial autoencoder~~ (done)
2. ~~Environment bridge + obs builder + PPO scaffold~~ (done)
3. ~~AE v3.1 border-accuracy push + v4 policy stack~~ (done)
4. **In flight:** PPO v4 curriculum climb, BC warm-start bake-off (feedforward
   vs temporal), BC→RL conditional launch
5. Scale PPO: remaining action heads (upgrade/delete/move-warship/cancel-boat),
   reward shaping audit (asymmetric loss aversion), recurrence (LSTM)
6. Self-play league
