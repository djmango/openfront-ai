# Agent design: observation stack & action space

**Decisions (2026-07-05):** (1) the autoencoder compresses ONLY spatial
state (tiles, terrain, static structures); all small exact state (diplomacy,
player scalars, transient units) bypasses the latent and feeds the policy
raw; (2) the FULL action surface ships in v1 (legality masking only, no
curriculum masking).

**Revision note.** The original v2 decision was a single unified AE over
ALL state. We built and trained it; spatial reconstruction was excellent,
but tiny exact facts fought the bottleneck — alliance pairs peaked at
F1 0.67 and troop ordering at 0.81 across four tuning runs (loss weights,
pooled-vector capacity, per-slot latents, feeding the graph to the
encoder). A one-bit fact reconstructed at 95% is strictly worse than
reading the bit. Hence the bypass split below.

Target: a self-play PPO agent over the full OpenFront gameplay surface.
Everything below is grounded in the engine's actual intent schemas
(`openfront/src/core/Schemas.ts`) and `Player`/`Game` interfaces — nothing
invented.

**Prior art:** AlphaFront (josh-freeman/openfront-rl) trains PPO on the
real engine with a scalar-only observation and heuristic tile choice — see
the [devlog](docs/devlog.html#alphafront) for the full comparison and the
ideas worth borrowing (win-rate-gated opponent curriculum, LR warmdown,
live-deployment bot).

## The full action surface

Every gameplay intent the engine accepts (admin/lobby intents excluded):

| # | Intent | Arguments | Notes |
|---|--------|-----------|-------|
| 1 | `spawn` | tile | spawn phase only |
| 2 | `attack` | targetID \| null, troops | null = expand into neutral land; engine spreads along shared border |
| 3 | `boat` | dst tile, troops | naval invasion, targets a specific tile |
| 4 | `cancel_attack` | attackID | retreat |
| 5 | `cancel_boat` | unitID | |
| 6 | `build_unit` | unit type, tile | structures: City, DefensePost, SAMLauncher, MissileSilo, Port, Factory; attacks: Warship, AtomBomb, HydrogenBomb, MIRV (nuke "build" = launch at tile) |
| 7 | `upgrade_structure` | unitId | |
| 8 | `move_warship` | unitIds[], tile | |
| 9 | `delete_unit` | unitId | |
| 10 | `allianceRequest` | recipient | also serves as accept of an incoming request |
| 11 | `allianceReject` | requestor | |
| 12 | `breakAlliance` | recipient | marks you traitor |
| 13 | `allianceExtension` | recipient | |
| 14 | `targetPlayer` | target | paints a target for allies |
| 15 | `donate_gold` | recipient, gold | |
| 16 | `donate_troops` | recipient, troops | |
| 17 | `embargo` / `embargo_all` | targetID, start/stop | trade denial |
| 18 | `emoji`, `quick_chat` | recipient, key | social signaling only |

## Factorized action heads

One flat softmax is impossible (tile arguments alone are up to 8M options).
Standard solution (AlphaStar-style): autoregressive heads with legality
masking, sampled in order. Per decision step:

1. **Action type** (~15-way + no-op). Masked by game state (e.g. `boat`
   masked unless a coastal path exists; nukes masked without a silo).
2. **Player target** (max-players-way, masked): argument for attack,
   alliance ops, donations, embargo, targetPlayer, chat. Masked by validity
   (`canAttackPlayer`, `canSendAllianceRequest`, `canDonate`, ...). Static
   per-game slots, same indexing as the observation.
3. **Tile target** (spatial pointer): argument for spawn, boat, build,
   nuke launch, warship move. Logits over the latent grid (H/16 x W/16),
   i.e. the policy points at a 16x16 region; the bridge snaps to the best
   legal tile in that region (`canBuild`, `bestTransportShipSpawn`,
   valid-shore checks). Snapping keeps the head size map-independent and
   pushes pixel-perfect legality into the engine, which already knows it.
4. **Unit instance** (pointer over own-unit tokens): argument for upgrade,
   delete, cancel_boat, move_warship, cancel_attack (over active attacks).
5. **Quantity** (fraction of available troops/gold): {5, 10, 25, 50, 100}%.
   Applies to attack, boat, donations.

Heads 2-5 are only evaluated/trained where the chosen action type needs
them (masked loss, standard practice).

All heads ship in v1. Masking is *legality-only* (what the engine would
reject), never curricular. Against easy bots most diplomatic/build actions
are simply unrewarding; the policy must learn to ignore them.

## Observation stack

### Spatial autoencoder (v3) + raw bypass

**Compress only what is big.** The spatial AE (`ae/model_v3.py`) consumes
tile ownership + terrain + fallout + static-structure planes (city, port,
defense post, missile silo, SAM launcher, factory) and produces the grid
latent (H/16 x W/16 x 64). Losses: border-weighted CE over owner slots;
rarity-weighted BCE over structure occupancy (detection, not count
regression — count MSE collapses to all-zeros on 99.9%-empty grids).
Measured fidelity: 99.4% tile accuracy, structures at precision/recall 1.0
per class.

**Everything else bypasses the latent** and reaches the policy exactly:

- pairwise diplomacy (alliance/embargo bits, expiry timers, pending
  requests) — targeting exists in the engine but only humans use it, so it
  is absent from bot data
- per-player scalars (alive, troops, gold, tiles, traitor, attack in/out)
- transient units as entity lists: nukes in flight with impact point
  (tx/ty) and SAM-lock status, transports with landing point, warships,
  trade ships
- attack aggregates (from, to, troops, retreating, boat origin)
- globals (tick, phase, players alive) and own-player internals
- **legality masks**, straight to the action heads — a mask reconstructed
  at 99% still yields illegal intents

**Deterministic, not variational.** A VAE's KL term costs capacity;
generative rollouts are not needed for representation learning.

#### v3.1: border-accuracy push

Overall tile accuracy saturates near 99% but border-tile accuracy (the
metric that matters for the policy) lagged at 76.5%/83.5% (human/bot).
v3.1 targets borders specifically:

- **Static-terrain-conditioned decoder.** Land bit + magnitude are static
  side-information the policy gets for free, so the decoder consumes them
  (avg-pooled to each scale, concatenated at every upsampling stage and at
  full res) without cheating; the latent then only encodes ownership
  relative to terrain. Fallout is dynamic state and is never fed to the
  decoder.
- **Stronger decoder head:** nearest-upsample + 3x3 conv stages replace
  ConvTranspose (no checkerboard artifacts), plus a full-resolution 3x3
  refinement block before the classifier (previously a bare 1x1).
- **Border-dense crop sampling:** crops are rejection-sampled with
  acceptance probability proportional to border-edge density (floored at
  0.15 so ocean/interior isn't starved; ≤8 attempts reusing one
  decompressed frame). Eval still samples uniformly.
- **Focal loss** (1 − p_true)^γ, γ=1.5 default, composed with the existing
  border weighting; warmup+cosine LR schedule; native `bacc` logging and
  step-tagged snapshot checkpoints.
- **`--latent-down 8` ablation flag:** latent grid at 1/8 instead of 1/16
  (one fewer stride-2 / upsample stage) for a capacity-vs-resolution
  ablation on cloud GPUs.

**Policy-side decision (approved):** in addition to the AE latent, the
policy will receive a raw local owner-map crop around the agent's own
territory as an exact-borders bypass — consistent with the v3 bypass
philosophy of passing small exact state around the latent instead of
forcing the latent to be pixel-perfect everywhere.

### The streams

### 1. Spatial (the map)

Frozen tile-autoencoder latent: 64 channels at 1/16 resolution (see
README results). Concatenated at the same resolution:

- ownership fractions per region for {self, allies, enemies-at-war,
  neutral players, unowned} (5 ch) — gives the policy an ego view the
  ego-agnostic AE doesn't provide
- structure presence per region, own vs. others, per structure class (2x6 ch)
- active battle intensity (attack tile deltas per region), fallout, and
  incoming-boat/nuke trajectories (3 ch)

Total ~80 channels at H/16 x W/16, any map size.

### 2. Entity tokens (variable-length sets, attention-pooled)

- **Player tokens** (one per alive player): static slot embedding; troops,
  gold, tile count (log-scaled); shared-border length with self; terrain
  mix along that border; relation state (allied / neutral / targeting-me /
  traitor / embargoed); pending alliance request flags; alliance expiry
  countdown; incoming/outgoing attack totals vs. self.
  This is the full player-state representation: every scalar the engine
  exposes about a player that a human can see on the leaderboard/diplomacy
  panels.
- **Unit tokens** (own + visible enemy structures/ships/nukes): type
  embedding, owner slot, level, health fraction, cooldown state,
  under-construction flag, region coordinates.
- **Attack tokens** (active attacks involving self): attacker/defender
  slot, troop count, retreating flag.

### 3. Global scalars

Own troops / max troops / gold / income rate; tick and game phase; players
alive; spawn-phase flag; doomsday-clock state; team-mode flags.

### Fusion

Token sets -> small transformer encoder -> pooled summaries + per-token
embeddings (kept for pointer heads 2 and 4). Spatial stream -> 2-3 conv
layers (kept at grid resolution for pointer head 3, plus a pooled vector).
Concatenate pooled spatial + pooled tokens + scalars -> trunk MLP (LSTM
later if needed) -> action heads.

## Environment bridge contract

The Node bridge must export per step: the three observation streams, plus
legality masks for every head (action types, valid player targets per
type, valid tile regions per type, valid unit instances). The engine
already computes all validity — the bridge's job is serialization, not
game logic. Decision cadence: one policy step per ~10 game ticks.

## Behavior cloning from archived human games

Two BC variants train on the replayed human archive (see `datagen/replay.ts
--bc` and `rl/bc.py`), both on the exact PPO policy architecture so BC
weights double as PPO initialization (`load_state_dict(strict=False)`):

1. **Outcome-conditioned (feedforward, `bc_v1`)** — every player's actions
   are supervision, not just winners'. A final-placement embedding (8
   percentile buckets, zero-init projection added to the trunk output) tells
   the model *whose* behavior it is imitating; at deployment we condition on
   the winner bucket. This is decision-transformer-style
   return-conditioning, collapsed to a single episode-level token.
2. **Temporal (`bc_seq_v1`, `--seq 8`)** — same, plus a 2-layer causal
   transformer over the last 8 decision steps' trunk embeddings.
   AlphaStar's core was a deep LSTM over game steps; OpenFront is fully
   observable so memory is a hypothesis to test, not a given.

Supervision detail: one sample per (snapshot, living human). The label is
the player's intents in the following 10-tick window normalized to the
factorized action space (multiple intents in a window: one sampled per
visit). Idle steps are real supervision too (noop is ~90% of human decision
steps) but are downsampled to ~15% of each batch. Loss is masked CE per
head, sub-heads only where the labeled action uses them. Normal-size human
maps are strided 2/4x down to the Compact grid budget — the engine's own
Compact mode is exactly a downscaled Normal map, so strided games stay
in-distribution for the frozen AE. Games split 90/10 train/holdout by game
ID for eval.
