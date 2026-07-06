# Agent design: observation stack & action space

**Decisions (2026-07-05):** (1) a single unified autoencoder compresses ALL
game state — tiles, players, units, attacks, globals — into one joint latent
that is the policy's primary observation; (2) the FULL action surface ships
in v1 (legality masking only, no curriculum masking).

Target: a self-play PPO agent over the full OpenFront gameplay surface.
Everything below is grounded in the engine's actual intent schemas
(`openfront/src/core/Schemas.ts`) and `Player`/`Game` interfaces — nothing
invented.

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

### Unified state autoencoder (v2, supersedes tile-only AE)

One model consumes every stream below and produces a single joint latent;
the policy reads the latent instead of raw features. Architecture:

- **Encoders**: CNN over the tile grid (as in AE v1); a small transformer
  over player/unit/attack tokens; an MLP over global scalars. Token
  embeddings and grid features cross-attend, then fuse into:
- **Joint latent**: the spatial grid latent (H/16 x W/16 x C) plus a
  vector latent (pooled tokens + scalars). Per-token encoder embeddings
  are retained as pointer keys for action heads 2 and 4.
- **Decoders (reconstruction targets)** — chosen so nothing needs
  set-decoding/Hungarian matching:
  - tile owner slots: per-tile CE, border-weighted (as v1)
  - per-player scalars: regression per *static slot* (slots are bounded
    and fixed per game, so this is a fixed-shape head)
  - units: spatial *presence planes* per unit class at grid resolution
    (unit existence becomes an image, not a set)
  - attacks-in-flight: per-slot-pair troop totals
  - global scalars: direct regression
- **Deterministic first.** Variational (true VAE) only if we later want
  generative rollouts/world-model planning; for representation learning
  the KL term mostly costs capacity.

Two pragmatic skips around the bottleneck (the latent is lossy by design;
these are exact by necessity):

1. **Legality masks** go straight to the action heads, never through the
   latent — a mask reconstructed at 99% accuracy still yields illegal
   intents.
2. **Own-player scalars** (own troops/gold/income) are appended raw to the
   trunk input. Relative strength judgments ("do I outnumber him 1.4x?")
   are too load-bearing to trust to reconstruction blur.

Training data: requires datagen v2 — current snapshots lack units,
attacks, alliances, and relations. Same headless pipeline, richer
`meta.json` + unit/attack dumps per snapshot.

### The streams (all consumed by the unified AE)

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
