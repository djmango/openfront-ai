# OpenFront to PufferLib 5.0 environment specification

Implementation-grade spec for exposing the OpenFront engine as a PufferLib 5.0
environment. Every size, offset, function name and line number below was read
out of the two repositories named here. Anything that could not be read out of
code is marked `not determined from code` or labelled PROPOSAL.

Sources, pinned:

- PufferLib 5.0 clone: `/opt/data/workspaces/skg/PufferLib5`, branch `5.0`, HEAD
  `6ffa5b1` ("Merge pull request #690 from FinlaySanders/nethack-stock-nle").
  `git tag` in this clone returns nothing, so a tag named `5.0-experiments` is
  `not determined from code`.
- OpenFront: `/opt/data/workspaces/skg/openfront-ai`, HEAD `05cee07`
  ("docs(devlog): mark duo ops as V-dead, do not restart"). Game submodule
  `openfront/` at `58c536b7a3c528c125890b40b1b213109b8b7014` (contents of
  `openfront.commit`). Rust workspace `rust/{engine,ofcore,oftrain,ofhub,ofae}`
  (`rust/Cargo.toml`).

No Python exists in PufferLib 5.0; the entire trainer is C/CUDA
(`src/pufferl.cu`, `src/algo.cu`) plus the env header.

---

## 1. The PufferLib 5.0 environment contract

An environment is a single header `ocean/NAME/NAME.h`. It must first
`typedef` an `obs_t`, then include `pufferenv.h`:

`ocean/template/template.h` lines 3-4:

```c
typedef unsigned char obs_t;
#include "pufferenv.h"
```

`ocean/chess/chess.h` lines 11-12 do the same with `typedef uint8_t obs_t;`.
`ocean/go/go.h` line 8 uses `typedef float obs_t;`. `ocean/minimal/minimal.h`
line 6 uses `typedef float obs_t;`.

`build.sh` enforces the typedef. For the CPU play path it greps the header and
aborts if it is missing (`build.sh:288-290`):

```bash
ENV_HEADER="$SRC_DIR/$ENV.h"
if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
    echo "Error: $ENV_HEADER must typedef obs_t for standalone eval"
```

The same grep gates the web build (`build.sh:320-322`) and the native/CUDA
train build (`build.sh:460-462`).

### 1.1 Three required macros

`NUM_ATNS` (number of action heads), `ACT_SIZES` (a brace list, classes per
head) and `OBS_SIZE` (obs elements per agent). `src/algo.cu:1270-1276` makes
two of them hard errors:

```c
// Per-env from ENV_HEADER (ocean/<env>/<env>.h).
#ifndef NUM_ATNS
#error "ENV_HEADER must #define NUM_ATNS (number of action heads)"
#endif
#ifndef ACT_SIZES
#error "ENV_HEADER must #define ACT_SIZES { ... } (classes per head)"
#endif
```

Real examples:

- `ocean/template/template.h:6-8`: `ACT_SIZES {2}`, `OBS_SIZE 1`, `NUM_ATNS 1`.
- `ocean/go/go.h:11-13`: `ACT_SIZES {82}`, `OBS_SIZE 326`, `NUM_ATNS 1`.
- `ocean/chess/chess.h:14-15`: `ACT_SIZES {97}`, `NUM_ATNS 1`; `OBS_SIZE 167`
  at `chess.h:356`; `NUM_ACTIONS 97` and `PASS_ACTION 96` at `chess.h:348-349`.
- `ocean/minimal/minimal.h:10-12`: `ACT_SIZES {9, 5}`, `NUM_ATNS 2`, and
  `OBS_SIZE (2 + 4*(AGENTS + TARGETS))` (a computed macro, which works because
  `OBS_SIZE` is only ever used in arithmetic contexts).

### 1.2 The Agent struct

`src/pufferenv.h:24-30`:

```c
typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;
```

The trainer owns these buffers and assigns the per-agent pointers itself, so
the env must not allocate them. Host allocation is `pufferl.cu:1024-1026`:

```c
size_t obs_bytes = total_agents * OBS_SIZE * sizeof(obs_t);
size_t mask_bytes = total_agents * vec->mask_size * sizeof(unsigned char);
cudaHostAlloc((void**)&vec->observations, obs_bytes, cudaHostAllocPortable);
cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float),
```

Pointer assignment per agent is `pufferl.cu:1079-1084`:

```c
a->observations = vec->observations + (size_t)phys * OBS_SIZE;
a->actions = vec->actions + (size_t)phys * NUM_ATNS;
a->rewards = vec->rewards + phys;
a->terminals = vec->terminals + phys;
a->action_mask = vec->action_mask + (size_t)phys * vec->mask_size;
```

Agent order within one env is slot order 0..num_agents-1; each agent's
`policy` selects which policy plays it (`pufferl.cu:1048`, `pufferl.cu:1073`).

### 1.3 Action mask layout is first class, and contiguous

The mask row width is exactly `act_n = sum(ACT_SIZES)`; there is no per-head
padding. `pufferl.cu:1852-1872`:

```c
// Discrete action layout. Continuous dims are size 1. Mask width is act_n.
int num_action_heads = NUM_ATNS;
int act_sizes[] = ACT_SIZES;
int act_n = 0;
...
vec->mask_size = act_n;
```

The device buffer is `(total_agents, act_n)` (`pufferl.cu:1880`) and is
**memset to 1** at create time (`pufferl.cu:1891`), so an env that wants masked
actions must write 0 for illegal entries every step. `chess.h:1603-1613` is
the reference pattern: only write the mask when
`env->agents[0].action_mask != NULL`, and `memset(my_mask, 0, NUM_ACTIONS)`
before filling.

### 1.4 struct Log and struct Env

`struct Log` must be all floats: `pufferl.cu:922` computes
`constexpr int LOG_NF = (int)(sizeof(Log) / sizeof(float));` and the reduction
at `pufferl.cu:940-951` adds `sizeof(Log)/sizeof(float)` floats per env before
calling `puf_log` at `pufferl.cu:1374`.

`struct Env` must contain at least `Log log; Agent agents[N]; int num_agents;`.
`log_reduce` reads both `envs[i].log` and `envs->num_agents`
(`pufferl.cu:945-946`). `ocean/minimal/minimal.h:25-30` marks the required
members explicitly:

```c
struct Env {
    Log log; int num_agents; unsigned int rng; // Required
    Agent agents[AGENTS]; int tag, boundary_reached; // Required
```

`int tag` (selfplay bank id, 0 = pure selfplay) and `int boundary_reached`
(game-end flag used to swap frozen banks only between games) are required by
contract but only meaningful when `vec.num_policies > 1` or
`selfplay.enabled = 1`. `chess.h:428-529` shows the real shape, including
per-slot alias pointers (`obs_ptr[2]`, `action_mask_ptr[2]`, `action_ptr[2]`,
`reward_ptr[2]`, `terminal_ptr[2]`) that `chess.h:532-540` re-syncs from
`agents[]` at step and reset.

### 1.5 The six functions

`src/pufferenv.h:44-50`:

```c
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
// CPU: host Env*. GPU: device batch base; implementation D2Hs what it needs and draws.
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
```

Optional hooks the header offers:

- `puf_set_bot_policy(Env*, int)` with a default no-op stub, guarded by
  `PUF_HAS_BOT_POLICY` (`pufferenv.h:52-56`). See section 6.
- `MY_VEC_INIT` / `MY_VEC_CLOSE` (`pufferl.cu:996`, `pufferl.cu:1358`), used by
  `chess.h:20-21`.
- `PUF_STEPS_PER_SEC` (render pacing only), e.g. `chess.h:19`.
- `PUFFERCPU_EVAL_MAIN` for the standalone CPU binary, and per-env network
  macros such as `PUF_MINIMAL_NET` (`minimal.h:16-19`). The `PUFFER_<ENV>`
  define is injected by `build.sh:208` (`-DPUFFER_${ENV^^}`).

`puf_init` is called once per env with the `[env]` ini section as `Dict*`
(`pufferl.cu:1004`). It must set `env->num_agents` and each
`agents[i].policy`; the CPU evaluator probes `num_agents` by calling
`puf_init` on a zeroed probe env and then `puf_close`
(`pufferl.cu:2915-2921`).

`puf_log` writes floats into the output dict with `dict_set`, e.g.
`chess.h:3389-3405`. `template.h` shows the minimum:

```c
void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}
```

### 1.6 How build.sh consumes an env header

Native/train build (`build.sh:451-462`): picks `ocean/NAME/NAME.cu` when
`--cu` is passed, otherwise `ocean/NAME/NAME.h`, greps for the `obs_t`
typedef, then (line 462) sets `ENV_COMPILE_FLAGS=(-DENV_HEADER=\"$ENV_HEADER\")`.

`build.sh:506-528` is the training link line:

```bash
$NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
    -I. -Isrc -I$SRC_DIR -Ivendor \
    ...
    "${ENV_COMPILE_FLAGS[@]}" \
    -DENV_NAME=$ENV \
    -DPUFFER_ENV_NAME=\"$ENV\" \
    -DPUFFERLIB_BUILD_MAIN \
    ...
    src/pufferl.cu \
    ...
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
```

The env header is textually included by the trainer at `src/pufferl.cu:64-65`:

```c
// Compile vs a single env: -DENV_HEADER=ocean/<env>/<env>.h or .cu (--cu)
#include ENV_HEADER
```

CUDA discovery and the NCCL include/lib search are at `build.sh:423-443`
(`CUDA_HOME` from `nvcc`, then `/usr/include`, `/usr/local/cuda/include`,
`/usr/lib/x86_64-linux-gnu`, `/usr/local/cuda/lib64`, then the
`nvidia.nccl` wheel as a fallback).

CPU play/eval build (`--cpu`, `build.sh:279-303`) compiles
`src/puffercpu.c` with:

```bash
STANDALONE_DEFINES=(
    -DPUFFERCPU_EVAL_MAIN
    -DENV_HEADER=\"$ENV_HEADER\"
    -DPUFFER_ENV_NAME=\"$ENV\"
)
```

and `src/puffercpu.c:606-612` consumes it:

```c
#ifdef PUFFERCPU_EVAL_MAIN

#ifndef ENV_HEADER
#error "ENV_HEADER required for PUFFERCPU_EVAL_MAIN"
#endif

#include ENV_HEADER
```

`puffercpu.c` then evaluates `OBS_SIZE` and `ACT_SIZES` at
`puffercpu.c:744-745` and `puffercpu.c:762` (weight-count check against a
`.bin` checkpoint). The output binary is `./$OUTPUT_NAME` (default the env
name), and that branch exits before the CUDA path (`build.sh:301-303`).

### 1.7 Custom network hook

A custom encoder/decoder replaces the default PufferNet forward pass through a
vtable. `src/algo.cu:23-32`:

```c
struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int in_dim, out_dim;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};
```

`build_arch` builds the Arch and then calls the override hook
(`algo.cu:986-998`):

```c
    Encoder encoder = {
        .forward = encoder_forward,
        ...
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
    create_custom_encoder(&encoder);
```

`create_custom_encoder` is defined in `src/ocean.cu` (comment at line 55,
definition at line 56) and dispatches on
`PUFFER_<ENV>` defines. This is the insertion point for the
OpenFront spatial encoder discussed in section 4.

A related compile-time fact for a 37,500-wide action head:
`algo.cu:1278-1285` computes `PPO_MAX_HEAD_A` as the max entry of
`ACT_SIZES`, and that value sizes the fused PPO logit cache.

---

## 2. The OpenFront side

### 2.1 RlSession

`rust/engine/src/rl.rs` is the in-process engine surface. Its module doc
(rl.rs:1-4) states it is a native port of `bridge/env.ts`'s `EnvSession`,
"the engine surface the Rust trainer's `--engine native` backend drives
directly, replacing the JSON-over-pipes Node subprocess".

`RlSession::reset` (`rl.rs:55-63`):

```rust
    pub fn reset(
        repo_root: &Path,
        map_key: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
        n_agents: u32,
    ) -> Result<(Self, Value, EntsData, Legal, Vec<u8>, Option<(Value, Legal)>), String> {
```

The five-element tail of the return type is, in order (rl.rs doc, lines 50-54):
the obs head (meta only), typed `EntsData`, typed `Legal`, raw terrain bytes,
and, when `n_agents > 1`, the second agent's `(head, Legal)`.

`RlSession::step` (`rl.rs:246`):

```rust
    pub fn step(&mut self, intents: &[Value], ticks: u32) -> (Value, EntsData, Legal, Option<(Value, Legal)>)
```

Its doc (rl.rs:239-245) states it "count wasted intents against pre-step state,
submit one turn, run `ticks` engine ticks (breaking on a win update), return meta
head + typed ents/legal". The tick loop is `rl.rs:271-279`:

```rust
        for _ in 0..ticks {
            let updates = self.game.execute_next_tick();
            if updates.win.is_some() {
                winner = crate::obs::winner_value(&self.game);
                self.last_winner = winner.clone();
                break;
            }
        }
```

Legality is produced inside `step` at `rl.rs:295` (agent 0) and `rl.rs:299`
(agent 1) via `legality_typed(&self.game, AGENT_CLIENT_ID)`; both are imported
at `rl.rs:21-22`:

```rust
use crate::obs_typed::{entities_typed, legality_typed};
use ofcore::feat::{EntsData, Legal};
```

Agent client ids are `"AGENTRL1"` and `"AGENTRL2"`
(`rust/engine/src/session.rs:11-13`).

`n_agents` is clamped to 1..=2 (`rl.rs:64`) and `team_mode = n_agents > 1`
(`rl.rs:65`).

### 2.2 Map data

`rl.rs:67-69` resolves the map directory:

```rust
        let map_dir = repo_root
            .join("openfront/resources/maps")
            .join(map_key.to_lowercase());
```

Keys are lowercase directory names. `openfront/resources/maps` contains 100
directories (`ls | wc -l`), including `plains`, `achiran`, `aegean`,
`britannia`, `blacksea`, `betweentwoseas`, `caucasus`, `greatlakes`, `onion`,
`europe`, `pangaea`. Each directory holds `manifest.json`, `map.bin`,
`map4x.bin`, `map16x.bin`, `thumbnail.webp`.

Loading is `rust/engine/src/core/terrain.rs:89` `load_fresh_terrain_from_dir`,
called from `rl.rs:90` with `GameMapSize::Normal`. For `Normal` it reads the
manifest plus `map.bin` as the game map and `map4x.bin` as the mini map
(terrain.rs:93-104). The wire config sets `"gameMapSize": "Normal"`
(`rl.rs:72`).

Also on the wire config (`rl.rs:72-85`): `gameMode` is `"Team"` when
`n_agents > 1` else `"Free For All"`, `gameType` is `"Singleplayer"`,
`donateGold` and `donateTroops` are true, `infiniteGold`/`infiniteTroops`/
`instantBuild`/`randomSpawn` are false.

### 2.3 Decision cadence

One policy decision advances the engine by `decision_ticks` engine ticks.
The production value in V10 is **15**, and it is uniform across every stage.

- `rust/ofcore/src/lib.rs:22`: `pub const WATCH_TICKS_PER_DECISION: i64 = 15;`
  with the doc at lib.rs:16-21: "Prefer the curriculum stage's
  `decision_ticks` (V10 uses 15 everywhere)."
- `rust/ofcore/src/curriculum.rs:1963-1973` is the regression test that pins it:

```rust
    #[test]
    fn v10_decision_ticks_are_uniformly_fifteen() {
        let stages = stages_for_schedule(CurriculumSchedule::V10);
        assert_eq!(stages.len(), V10_STAGE_COUNT);
        for (i, stage) in stages.iter().enumerate() {
            assert_eq!(
                stage.decision_ticks, 15,
                "stage {i} decision_ticks={}",
                stage.decision_ticks
            );
        }
    }
```

- The stage struct field is `pub decision_ticks: u32`
  (`curriculum.rs:726-733`), the table is built by `build_v10_stages`
  (`curriculum.rs:1119`), and `stages_for_schedule` currently always returns
  the V10 table (`curriculum.rs:1177-1179`).
- The trainer passes it straight to the engine:
  `rust/oftrain/src/vecenv.rs:1233`
  `let new_obs = self.bridge.step(&intents, self.decision_ticks)?;`
  (also vecenv.rs:1816 and vecenv.rs:2490). vecenv.rs:1107-1111 carries the
  comment "Always use the stage table's decision_ticks (V10 is uniformly 15)".
- Episode budget: `rust/ofcore/src/lib.rs:18`
  `pub const DEFAULT_MAX_EPISODE_TICKS: i64 = 21_000;` with the doc note
  "OpenFront `msPerTick() = 100`, so 21000 ticks ≈ 35 in-game minutes".

So one default episode is `21000 / 15 = 1400` policy decisions. The derived
watch cap is `DEFAULT_WATCH_MAX_STEPS = 21000/15 + 64 = 1464`
(`lib.rs:28-30`, asserted at `lib.rs:43-46`), and the actor step cap is
`(max_ticks / decision_ticks) + 64` (`rust/oftrain/src/train.rs:3932`).

MISMATCH TO FIX: the C ABI shim defaults `ticks_per_decision` to 8, not 15
(`rust/engine/src/puffer_ffi.rs:228`, documented in
`include/openfront_env.h:26`). Any PufferLib env must set
`ticks_per_decision=15` explicitly to reproduce trainer conditions.

---

## 3. Action surface

### 3.1 The 21 action ids

`rust/ofcore/src/feat.rs:32-52` defines the ids and feat.rs:54-76 the names,
in the same order:

```rust
pub const A_NOOP: i64 = 0;
pub const A_ATTACK: i64 = 1;
pub const A_EXPAND: i64 = 2;
pub const A_BOAT: i64 = 3;
pub const A_BUILD: i64 = 4;
pub const A_LAUNCH_NUKE: i64 = 5;
pub const A_ALLIANCE_REQUEST: i64 = 6;
pub const A_ALLIANCE_REJECT: i64 = 7;
pub const A_BREAK_ALLIANCE: i64 = 8;
pub const A_DONATE_GOLD: i64 = 9;
pub const A_DONATE_TROOPS: i64 = 10;
pub const A_EMBARGO: i64 = 11;
pub const A_RETREAT: i64 = 12;
pub const A_SPAWN: i64 = 13;
pub const A_UPGRADE_STRUCTURE: i64 = 14;
pub const A_MOVE_WARSHIP: i64 = 15;
pub const A_CANCEL_BOAT: i64 = 16;
pub const A_DELETE_UNIT: i64 = 17;
pub const A_EMBARGO_STOP: i64 = 18;
pub const A_TARGET_PLAYER: i64 = 19;
pub const A_ALLIANCE_EXTENSION: i64 = 20;
```

`pub const N_ACTIONS: usize = 21;` (feat.rs:18).

Target types come from four predicates in the same file:

| id | name | player target | unit target | tile target | quantity |
|----|------|---------------|-------------|-------------|----------|
| 0 | noop | no | no | no | no |
| 1 | attack | yes | no | no | yes |
| 2 | expand | no | no | no | yes |
| 3 | boat | no | no | yes | yes |
| 4 | build | no | no | yes | no |
| 5 | launch_nuke | no | no | yes | no |
| 6 | alliance_request | yes | no | no | no |
| 7 | alliance_reject | yes | no | no | no |
| 8 | break_alliance | yes | no | no | no |
| 9 | donate_gold | yes | no | no | yes |
| 10 | donate_troops | yes | no | no | yes |
| 11 | embargo | yes | no | no | no |
| 12 | retreat | yes | no | no | no |
| 13 | spawn | no | no | yes | no |
| 14 | upgrade_structure | no | yes | yes | no |
| 15 | move_warship | no | yes | yes | no |
| 16 | cancel_boat | no | yes | yes | no |
| 17 | delete_unit | no | yes | yes | no |
| 18 | embargo_stop | yes | no | no | no |
| 19 | target_player | yes | no | no | no |
| 20 | alliance_extension | yes | no | no | no |

Predicates: `needs_player` feat.rs:126-140, `needs_unit` feat.rs:144-148,
`needs_tile` feat.rs:151-161, `needs_quantity` feat.rs:165-169.
`refine_tile` (feat.rs:173-177) marks the five actions whose tile pick
refines to the fine /8 grid: spawn, build, upgrade_structure, cancel_boat,
delete_unit.

Build and nuke subtype tables:

- `BUILD_TYPES` (feat.rs:78-86), `N_BUILD = 7`: `City`, `Port`,
  `Defense Post`, `Missile Silo`, `SAM Launcher`, `Factory`, `Warship`.
- `NUKE_TYPES` (feat.rs:89-95), `N_NUKE = 5`: Atom Bomb up, Atom Bomb down,
  Hydrogen Bomb up, Hydrogen Bomb down, MIRV (arc flag ignored for MIRV).

### 3.2 ofcore::feat::Legal

`rust/ofcore/src/feat.rs:465-490`:

```rust
#[derive(Default, Clone, Debug)]
pub struct Legal {
    pub present: bool,
    pub attackable: Vec<usize>,
    pub alliance_requestable: Vec<usize>,
    pub alliance_rejectable: Vec<usize>,
    pub breakable: Vec<usize>,
    pub donatable_gold: Vec<usize>,
    pub donatable_troops: Vec<usize>,
    pub embargoable: Vec<usize>,
    pub stop_embargoable: Vec<usize>,
    pub targetable: Vec<usize>,
    pub extendable: Vec<usize>,
    pub can_expand: bool,
    pub can_boat: bool,
    pub troops: f64,
    pub gold: f64,
    pub build_mask: [f32; N_BUILD],
    pub nuke_mask: [f32; N_NUKE],
    pub has_silo: bool,
    pub attacks: Vec<String>, // attack ids (n_attacks = len)
    pub upgradable: Vec<usize>,
    pub warships: Vec<usize>,
    pub boats: Vec<usize>,
    pub deletable: Vec<usize>,
}
```

The `Vec<usize>` fields hold engine small player ids (and engine unit ids for
`upgradable`/`warships`/`boats`/`deletable`); `parse_legal` documents that the
latter "carry engine unit ids (u64), used only for len()/id-membership
downstream" (feat.rs:514-517).

### 3.3 Where legality comes from

`rust/engine/src/obs_typed.rs:187` is `pub fn legality_typed(game: &Game, client_id: &str) -> Legal`,
a typed port of the JSON `legality` builder (obs_typed.rs:185-186:
"Typed port of [`crate::obs::legality`]'s `actions` object (empty when the
agent is dead / missing - same as JSON `actions: {}`)"). It returns
`Legal::default()` immediately when the player is missing or dead
(obs_typed.rs:188-191), so `present == false` is the dead/no-agent sentinel.

Per-field derivation:

- `build_mask` / `nuke_mask`: iterate `STRUCTURES` chained with `LAUNCHABLE`
  chained with `[ut::WARSHIP]` and set the bit when
  `gold >= game.structure_cost(sid, t)` (obs_typed.rs:203-212). Note this is a
  gold test, not a terrain test; map placement legality is separate.
- `attackable`: other living players where `shares_border_with(game, agent, p)`
  and `!game.is_friendly(sid, p)` (obs_typed.rs:214-218).
- `has_silo`: spawn immunity inactive and at least one `MISSILE_SILO` unit
  that is not under construction and not in cooldown (obs_typed.rs:220-226).
- `upgradable`: own units where `game.can_upgrade_unit(sid, u.id)` and
  `gold >= structure_cost` (obs_typed.rs:228-235).
- `deletable`: empty unless `game.can_delete_unit(sid)`; then own land units
  (obs_typed.rs:237-250).
- `alliance_requestable` / `alliance_rejectable` / `breakable` /
  `donatable_gold` / `donatable_troops` / `embargoable` / `stop_embargoable` /
  `targetable` / `extendable`: engine predicates `can_send_alliance_request`,
  `incoming_alliance_requests`, `player_alliances`, `can_donate_gold`,
  `can_donate_troops`, `has_embargo_against`, `agent.embargoes`,
  `game.can_target` (obs_typed.rs:252-300).

### 3.4 Flattened mask layout for the C env

`ofcore::feat::featurize` (feat.rs:650) emits the masks in exactly the layout
below; `ofenv_mask` in the C ABI copies them verbatim
(`rust/engine/src/puffer_ffi.rs:381-407`). Offsets are floats from the start
of one agent's mask block.

| offset | length | contents |
|--------|--------|----------|
| 0 | 21 | `legal_actions`, index = action id 0..20 |
| 21 | 2688 | `legal_ptarget`, `N_ACTIONS * MAX_SLOTS` = 21 * 128, row-major by action: `a*128 + slot` |
| 2709 | 672 | `legal_utarget`, `N_ACTIONS * MAX_UNITS` = 21 * 32, row-major by action: `a*32 + unit_token` |
| 3381 | 7 | `legal_build`, `N_BUILD` |
| 3388 | 5 | `legal_nuke`, `N_NUKE` |
| 3393 | 37500 | `legal_tile` on a fixed `GH_MAX(150) x GW_MAX(250)` row-major plane |
| total | **40893** | |

Formula: `21 + (21*128) + (21*32) + 7 + 5 + (250*150) = 40893`.

These offsets and the total are hard-coded in the C ABI as
`OFENV_MASK_ACTIONS_OFF 0`, `OFENV_MASK_PTARGET_OFF 21`,
`OFENV_MASK_UTARGET_OFF 2709`, `OFENV_MASK_BUILD_OFF 3381`,
`OFENV_MASK_NUKE_OFF 3388`, `OFENV_MASK_TILE_OFF 3393`,
`OFENV_MASK_PER_AGENT 40893` (`include/openfront_env.h:126-138`), with
compile-time asserts in Rust (`puffer_ffi.rs:121-122`).

The tile plane is a fixed 150x250 grid and only rows `0..gh`, cols `0..gw` are
written; the rest is zeroed every step so stale rows cannot look legal
(`puffer_ffi.rs:381-405`). `gh` and `gw` are reported by `ofenv_meta`
(`include/openfront_env.h:94-96`). The tile row stride is `GW_MAX`, so a tile
at fine-grid `(y, x)` lives at `3393 + y*250 + x`.

Note the tile plane is filled from `feat.legal_tile`, which is computed by
`legal_tile_mask` at `feat.rs:1038` (definition at `feat.rs:1392`) from
`land`/`mag`/`owners`/`spawn_phase`. Mask entries are 0.0 or 1.0
(`include/openfront_env.h:80`).

Mapping this onto PufferLib: the trainer's layout is *conditional*
(action first, then target conditioned on the action). PufferLib 5.0 samples
every head independently and masks the flattened row of width
`act_n = sum(ACT_SIZES)`. Two workable encodings:

- PROPOSAL A, one head: `ACT_SIZES {40893}`, `NUM_ATNS 1`. The head is the
  whole conditional block; the env supplies the 40893-long mask and the
  policy learns the conditional structure implicitly. Largest single softmax in
  the repo (the ceiling on head width, `PPO_MAX_HEAD_A`, is the max `ACT_SIZES`
  entry, `algo.cu:1278-1285`).
- PROPOSAL B, six heads: `ACT_SIZES {21, 128, 32, 7, 5, 37500}`,
  `NUM_ATNS 6`, `act_n = 37693` (21 + 128 + 32 + 7 + 5 + 37500). This keeps
  the target dimensions as separate heads but loses the conditioning, so the
  env must only consume the head that the sampled action id requires and must
  maintain the invariant that illegal combinations are unreachable (mask the
  head to a single legal entry when the sampled action needs no target).

Which encoding is correct for training is a modelling decision, not something
readable from either codebase: `not determined from code`.

---

## 4. Observation

### 4.1 Raw pre-AE featurization

Constants, `rust/ofcore/src/feat.rs:8-30`:

```rust
pub const REGION: usize = 8;
pub const MAX_SLOTS: usize = 128;
pub const N_STATIC: usize = 6;
/// V11: ego-split attack fronts add 4 planes (own/ally/enemy × src/retreat).
pub const N_TRANSIENT: usize = 57;
pub const P_FEAT: usize = 30;
pub const N_SCALARS: usize = 12;
pub const N_ACTIONS: usize = 21;
pub const N_BUILD: usize = 7;
pub const N_NUKE: usize = 5;
/// Top-K own (+ visible) unit tokens for the unit pointer head.
pub const MAX_UNITS: usize = 32;
pub const U_FEAT: usize = 12;
pub const GW_MAX: i64 = 250;
pub const GH_MAX: i64 = 150;
```

`GW_MAX` / `GH_MAX` are expressed in /REGION units, i.e. cells of the
/8-resolution grid (comment at `rust/oftrain/src/policy.rs:39-41`: "map sizes
(`GW_MAX`=250, `GH_MAX`=150 in /REGION units)"). `gh = height_rows / REGION`
and `gw = width_cols / REGION` (`rust/oftrain/src/ae.rs:534-535`).

The container is `Feat`, `feat.rs:563-582`:

```rust
pub struct Feat {
    pub stat: Vec<f32>,      // (N_STATIC, gh, gw)
    pub transient: Vec<f32>, // (N_TRANSIENT, gh, gw)
    pub clut: [u8; MAX_SLOTS],
    pub players: Vec<f32>,   // (MAX_SLOTS, P_FEAT)
    pub pmask: [f32; MAX_SLOTS],
    /// Packed unit tokens `(MAX_UNITS, U_FEAT)`.
    pub units: Vec<f32>,
    pub umask: [f32; MAX_UNITS],
    /// Engine uids aligned with `units` rows (`-1` = pad).
    pub unit_uids: [i64; MAX_UNITS],
    /// Per-action legality over unit tokens `(N_ACTIONS, MAX_UNITS)`.
    pub legal_utarget: Vec<f32>,
    pub scalars: [f32; N_SCALARS],
    pub me_slot: i64,
    pub legal_actions: [f32; N_ACTIONS],
    pub legal_ptarget: Vec<f32>, // (N_ACTIONS, MAX_SLOTS)
    pub legal_build: [f32; N_BUILD],
    pub legal_nuke: [f32; N_NUKE],
    pub legal_tile: Vec<f32>, // (gh, gw)
}
```

`featurize` signature (`feat.rs:650-663`):

```rust
pub fn featurize(
    gh: usize,
    gw: usize,
    lut: &[u8],
    land: &[u8],
    mag: &[u8],
    owners: &[u8], // already slotted (uint8) at this resolution
    tick: i64,
    spawn_phase: bool,
    alive: bool,
    me: i64,
    ents: &EntsData,
    legal: &Legal,
) -> Feat {
```

Per-block sizes, all from the code:

| block | shape | element count | source |
|-------|-------|---------------|--------|
| `stat` | (6, gh, gw) | 6 * gh * gw | feat.rs:564 |
| `transient` | (57, gh, gw) | 57 * gh * gw | feat.rs:565 |
| `clut` | (128,) | 128 | feat.rs:566, `make_clut` feat.rs:617 |
| `players` | (128, 30) | 3840 | feat.rs:567 |
| `pmask` | (128,) | 128 | feat.rs:568 |
| `units` | (32, 12) | 384 | feat.rs:570 |
| `umask` | (32,) | 32 | feat.rs:571 |
| `unit_uids` | (32,) | 32 i64 | feat.rs:573 |
| `legal_utarget` | (21, 32) | 672 | feat.rs:575 |
| `scalars` | (12,) | 12 | feat.rs:576 |
| `legal_actions` | (21,) | 21 | feat.rs:578 |
| `legal_ptarget` | (21, 128) | 2688 | feat.rs:579 |
| `legal_build` | (7,) | 7 | feat.rs:580 |
| `legal_nuke` | (5,) | 5 | feat.rs:581 |
| `legal_tile` | (gh, gw) | gh * gw | feat.rs:582 |

The 57 transient planes are laid out as own/ally/enemy triplets with named
bases, `feat.rs:100-120`: `TR_WARSHIP 0`, `TR_TRANSPORT 3`,
`TR_TRANSPORT_DEST 6`, `TR_TRADE 9`, `TR_TRADE_DEST 12`, `TR_NUKE 15`,
`TR_NUKE_IMPACT 18`, `TR_NUKE_SAMLOCK 21`, `TR_CONSTRUCTION 24`,
`TR_SAM_MISSILE 27`, `TR_SAM_MISSILE_IMPACT 30`, `TR_MIRV_WARHEAD 33`,
`TR_MIRV_WARHEAD_IMPACT 36`, `TR_TRAIN 39`, `TR_SILO_COOLDOWN 42`,
`TR_SAM_COOLDOWN 45`, `TR_STATION 48`, `TR_ATTACK_SRC 51`,
`TR_ATTACK_RETREAT 54`. That is 18 triplets = 54 planes plus the V11
ego-split attack front planes, totalling `N_TRANSIENT = 57`.

The 12 scalars are (feat.rs:955-968, matching the C ABI header comment at
`include/openfront_env.h:60-64`):

```
0  tick / 15000
1  spawn_phase
2  alive
3  log_norm(legal.troops)
4  log_norm(legal.gold)
5  n_alive / 128
6  legal.attacks.len() / 8
7  me_slot / MAX_SLOTS
8  log_norm(me_troop_income)
9  log_norm(me_gold_income)
10 team_claimed_share
11 team_map_share        (win fires at >= 0.80; DUO_TEAM_WIN_MAP_SHARE curriculum.rs:31)
```

### 4.2 What the trainer assembles from the raw blocks

`rust/oftrain/src/policy.rs:14-35`:

```rust
pub const N_ACTIONS: i64 = ofcore::feat::N_ACTIONS as i64;
pub const MAX_SLOTS: i64 = ofcore::feat::MAX_SLOTS as i64;

pub const LATENT_C: i64 = 32;
pub const N_STATIC: i64 = 6;
pub const N_TRANSIENT: i64 = ofcore::feat::N_TRANSIENT as i64; // 57
pub const EGO_OWN_CH: i64 = LATENT_C + N_STATIC;
pub const C_GRID: i64 = LATENT_C + N_STATIC + 3 + 1 + N_TRANSIENT; // 99
pub const C_GRID_FINE: i64 = C_GRID + 1; // 100
pub const N_LOCAL: i64 = 5;
pub const LOCAL: i64 = 64;
pub const FOVEATE_SIZE: i64 = 48;
```

So the grid tensor is 99 channels: 32 AE latent + 6 exact static + 3 ego
(own/ally/enemy) + 1 defense bonus + 57 transient. The module doc
(policy.rs:1-8) states: "Grid channels (V11 / AE v3.2): frozen AE latent (32)
+ exact static structures (6) + ego (3) + defense_bonus (1) + transient (57)
= `C_GRID` 99. Buildings bypass the AE."

The non-grid observation blocks are `policy::Obs`
(`policy.rs:145-171`): `players (B, MAX_SLOTS, P_FEAT)`, `pmask (B, MAX_SLOTS)`,
`units (B, MAX_UNITS, U_FEAT)`, `umask`, `legal_utarget (B, N_ACTIONS,
MAX_UNITS)`, `local (B, N_LOCAL, LOCAL, LOCAL)`, `scalars (B, N_SCALARS)`,
`legal_actions (B, N_ACTIONS)`, `legal_ptarget (B, N_ACTIONS, MAX_SLOTS)`,
`legal_build (B, N_BUILD)`, `legal_nuke (B, N_NUKE)`, plus the duo/MAPPO
partner tensors `partner_players`, `partner_pmask`, `partner_scalars`,
`partner_context`.

### 4.3 Where the AE sits, and where it would sit in PufferNet

The autoencoder lives in `rust/oftrain/src/ae.rs`. Its header comment
(ae.rs:13-14) names the two streams:

```
//! - fine:  `ae_v32_nostatic_d8c32`  - 32ch @ 1/8
//! - coarse: `ae_v32_nostatic_d16c32` - 32ch @ 1/16 (optional coarse stream)
```

Constants (ae.rs:23-29): `MAX_SLOTS 128`, `OWNER_EMB_DIM 8`,
`TERRAIN_CHANNELS 3`, `NUM_STATIC 6`, `LATENT_C 32`, `REGION 8`,
`COARSE_REGION 16`. `SpatialAePair` holds `fine: SpatialAE` and
`coarse: Option<SpatialAE>` (ae.rs:287-292), loaded by `SpatialAePair::load`
(ae.rs:296-320) and created untrained by `new_random_pair` (ae.rs:322-330).
`SpatialAE::encode(owners, terrain)` is the forward (ae.rs:268). The
featurization the AE consumes is `pack_static_terrain(land, mag, hr, wr)`
(ae.rs:371) with `packed_fallout_row_bytes` (ae.rs:383), and the fine/coarse
crop sizes are derived at ae.rs:634-639 (`fine_gh = hr / REGION as usize`,
coarse `div_ceil(2)`).

In PufferLib this becomes a custom `Encoder` implementation installed by
`create_custom_encoder` (`algo.cu:986-998`, `ocean.cu:56-70`). What that
requires, stated plainly:

- The `Encoder` vtable carries `forward`/`backward`/`init_weights`/
  `reg_params`/`reg_train`/`reg_rollout`/`create_weights`, plus `in_dim`,
  `out_dim` and `activation_size` (`algo.cu:23-32`). The OpenFront encoder must
  supply all seven function pointers and its own activation struct size.
- `in_dim` is `OBS_SIZE` from the env header. Because the spatial blocks scale
  with map size (6 and 57 planes at `gh x gw`), they cannot be part of a fixed
  `OBS_SIZE`; they must be fetched by the encoder from a side channel, or the
  env must export a fixed-size crop. In PufferLib the encoder receives only the
  obs buffer plus device buffers, so bytes are uploaded through the obs
  vector. `not determined from code` how the AE weights (`.bin` or `.safetensors`,
  checkpoint paths) would be embedded into a PufferLib build;
  `SpatialAE::load(fine_path, coarse_path, device, ae_amp)` exists (ae.rs:296-304)
  but there is no C-side weight loader today.

### 4.4 What the C ABI exposes

Files: `include/openfront_env.h` (203 lines, untracked) and
`rust/engine/src/puffer_ffi.rs` (30670 bytes, untracked, declared as
`pub mod puffer_ffi;` at `rust/engine/src/lib.rs:19`). Both were present in the
working tree when this spec was written and are not committed; treat their
contents as in-flight.

Link target: `rust/target/release/libopenfront_engine.so`, because
`rust/engine/Cargo.toml:14` sets `crate-type = ["lib", "cdylib"]` for lib name
`openfront_engine` (`Cargo.toml:8-9`). The header states this at
`include/openfront_env.h:12-13`.

Exported functions (`include/openfront_env.h:149-197`, implemented at
`puffer_ffi.rs:585-806`):

| function | signature |
|----------|-----------|
| `ofenv_create` | `OFEnv ofenv_create(const char *json_cfg)` |
| `ofenv_destroy` | `void ofenv_destroy(OFEnv env)` |
| `ofenv_reset` | `int ofenv_reset(OFEnv env)` |
| `ofenv_step` | `int ofenv_step(OFEnv env, const char *intents_json)` |
| `ofenv_obs` | `const float *ofenv_obs(OFEnv env, int *n)` |
| `ofenv_mask` | `const float *ofenv_mask(OFEnv env, int *n)` |
| `ofenv_tiles` | `const unsigned short *ofenv_tiles(OFEnv env, int *n)` |
| `ofenv_meta` | `const char *ofenv_meta(OFEnv env)` |
| `ofenv_reward` | `double ofenv_reward(OFEnv env, int agent)` |
| `ofenv_terminal` | `int ofenv_terminal(OFEnv env, int agent)` |
| `ofenv_obs_size` | `int ofenv_obs_size(OFEnv env)` |
| `ofenv_mask_size` | `int ofenv_mask_size(OFEnv env)` |
| `ofenv_last_error` | `const char *ofenv_last_error(void)` |

Config string accepted by `ofenv_create` (header lines 20-26):

```
repo_root=/path/to/openfront-ai, map=plains, seed=s1, bots=3,
difficulty=Easy, n_agents=1, ticks_per_decision=8
```

Defaults: `map=plains`, `seed=s1`, `bots=3`, `difficulty=Easy`, `n_agents=1`
(clamped 1..=2), `ticks_per_decision=8`, `nations=0`; `repo_root` required
(parsed at `puffer_ffi.rs:176-247`, `ticks_per_decision` default at
`puffer_ffi.rs:224-228`).

`intents_json` is a JSON array of engine intent objects,
`[{"type":"spawn","tile":12345}]`; NULL, `""` or `"[]"` means no intents;
`clientID` defaults to `AGENTRL1` (header lines 28-30).

Observation block, `OFENV_OBS_PER_AGENT = 4270` floats per agent:

| offset | length | contents |
|--------|--------|----------|
| 0 | 12 | `Feat::scalars` |
| 12 | 3840 | `Feat::players`, `MAX_SLOTS 128 x P_FEAT 30`, slot-major |
| 3852 | 384 | `Feat::units`, `MAX_UNITS 32 x U_FEAT 12`, token-major |
| 4236 | 7 | `legal_build` |
| 4243 | 5 | `legal_nuke` |
| 4248 | 21 | `legal_actions` |
| 4269 | 1 | `me_slot` (MAX_SLOTS-normalised) |
| total | **4270** | |

Macros at `include/openfront_env.h:112-124`
(`OFENV_OBS_SCALARS_OFF`..`OFENV_OBS_ME_SLOT_OFF`, `OFENV_OBS_PER_AGENT`),
Rust mirror at `puffer_ffi.rs:98` with the assert at `puffer_ffi.rs:121`.

Tile state is packed per `include/openfront_env.h:142-147`: owner id in the low
12 bits (`OFENV_TILE_OWNER_MASK 0x0FFF`), bit 13 fallout
(`OFENV_TILE_FALLOUT_BIT`), bit 14 defense bonus
(`OFENV_TILE_DEFENSE_BONUS_BIT`), row-major at width x height reported in
`ofenv_meta`.

Not exported, and this is the key gap for training: the fine (1/8) stat (6
planes) and transient (57 planes) crops, the coarse (1/16) crops, the AE
latent, the pooled ego (3) / db (1) / local (5 x 64 x 64) planes, and the
full-resolution tile plane at 250 x 150 (header lines 68-75). Their inputs are
recoverable from `ofenv_tiles()` plus the featurizer LUT and the terrain bytes
returned on reset (header lines 72-74), and `puffer_ffi.rs:383-405` already
builds the `legal_tile` plane, but the raw `stat`/`transient` planes are
dropped (`puffer_ffi.rs` module doc lines 32-39: "NOT included (the AE-side
additions live in `oftrain` and need torch)").

---

## 5. Reward and curriculum

### 5.1 Where the reward lives today

Reward is defined in `rust/ofcore/src/curriculum.rs` and assembled per
decision in `rust/oftrain/src/vecenv.rs`. Top-level weights
(curriculum.rs:10-31):

```rust
pub const W_STR: f64 = 0.02;
pub const W_DELTA_GAIN: f64 = 5.0;
pub const W_DELTA_LOSS: f64 = 6.5;
pub const W_PLACE: f64 = 15.0;
pub const W_WIN: f64 = 30.0;
pub const W_DEATH: f64 = 1.0;
pub const W_WASTE: f64 = 0.01;
pub const PLACE_POW: f64 = 1.5;

pub const K_LAND: f64 = 0.40;
pub const K_MIL: f64 = 0.20;
pub const K_ECO: f64 = 0.25;
pub const K_BUILD: f64 = 0.15;
...
pub const V83_CLOSEOUT_SHARE_START: f64 = 0.45;
pub const V83_CLOSEOUT_SHARE_FULL: f64 = 0.80;
```

`RewardConfig` is `curriculum.rs:89`; it gates every shaped term per stage.
Version tags, all in curriculum.rs:

| constant | value | line |
|----------|-------|------|
| `LEGACY_V83_SCHEDULE_ID` | `"v8.3"` | 45 |
| `V86_REWARD_PROFILE` | `"v8.6-attack-fair-v1"` | 46 |
| `V10_REWARD_PROFILE` | `"v10-anti-spiral-v1"` | 48 |
| `V10_DEFAULT_DEATH_PENALTY` | `3.0` | 50 |
| `V10_WIN_AT` | `0.70` | 52 |
| `V10_RAMP_WIN_AT` | `0.90` | 55 |
| `V10_NATION_INTRO_STAGE` | `8` | 57 |
| `V10_ONE_NATION_WIN_AT` | `0.80` | 61 |
| `V10_MULTI_NATION_STAGE` | `12` | 63 |
| `V10_NATION_RAMP_WIN_AT` | `0.75` | 66 |
| `V10_WIN_AT_END` | `0.65` | 69 |
| `V10_STAGE_COUNT` | `68` | 816 |
| `V10_CLOSEOUT_STAGE` | `28` | 818 |
| `V10_BRIDGE_STAGE` | `31` | 820 |
| `V10_MEDIUM_START` | `36` | 822 |
| `V10_HARD_START` | `50` | 824 |
| `V10_IMPOSSIBLE_START` | `60` | 826 |

Named reward functions with file:line:

| function | line | role |
|----------|------|------|
| `terminal_reward(place, won, timed_out)` | 1366 | win / place / timeout terminal |
| `v10_survival_reward(alive, land_share, config)` | 1382 | tapering survival term |
| `v10_timeout_after_closeout_penalty(timed_out, closeout_reached, config)` | 1399 | stick for timing out after 45% land |
| `v10_closeout_entry_bonus(just_entered, config)` | 1413 | one-off closeout entry |
| `v10_win_at_for_stage(index)` | 1021 | per-stage win threshold |
| `v10_diplo_panic_penalty(...)` | 1781 | diplo panic shaping |
| `v10_combat_action_bonus(action, has_target, config)` | 1807 | combat action bonus |
| `v10_empty_action_net_reward(action, config)` | 1824 | no-op accounting |
| `action_churn_penalty(...)` | 636 | anti-churn (legacy path) |
| `v83_action_churn_penalty(inverse_pair, stage, share, config)` | 648 | V8.3 closeout-only churn penalty |
| `stage_learning_rate(base_lr, decay, stage, floor)` | 1039 | per-stage LR |
| `team_map_share(team_tiles, land_total)` | 34 | team land share |
| `team_territory_win(team_tiles, land_total)` | 42 | team win predicate |
| `timeweight(tick)` | 1209 | early-game time weighting |
| `closeout_potential(share)` | 697 | closeout potential |
| `land_share(agent_tiles, land_total)` | 690 | land share |
| `tempo_pressure(...)` | 446 | tempo shaping |
| `fast_win_bonus(won, tick, max_ticks, coef)` | 455 | faster-win bonus |

`terminal_reward` verbatim (curriculum.rs:1366-1375):

```rust
pub fn terminal_reward(place: i64, won: bool, timed_out: bool) -> f64 {
    if timed_out && !won {
        return -W_WIN;
    }
    let mut r = W_PLACE * (place as f64).powf(-PLACE_POW);
    if won {
        r += W_WIN;
    }
    r
}
```

`RewardComponents` is a 16-field struct (`curriculum.rs:334-352`): `strength`,
`strength_delta`, `dominance`, `closeout`, `action_churn`, `boat_outcome`,
`tempo`, `embargo_outcome`, `combat_outcome`, `survival`, `diplo_panic`,
`combat_action`, `waste`, `death`, `terminal`, `duo`.

The per-decision sum is built inside `VecEnv::apply` (`vecenv.rs:1749`) and
`VecEnv::apply_agents` (`vecenv.rs:1757`). The accumulation starts at
vecenv.rs:2091-2097:

```rust
            let mut components = RewardComponents {
                strength: W_STR * mine * tw * solo_scale,
                strength_delta: delta_weight * delta * solo_scale,
                ..RewardComponents::default()
            };
            let mut reward = components.strength + components.strength_delta;
```

then folds in churn (vecenv.rs:2098-2118), embargo and combat outcomes
(2119-2128), PBRS dominance (vecenv.rs:2129-2149), closeout plus entry bonus
(2150-2168), boat outcome (2170-2200), tempo (2200-2210), survival (2211-2215),
diplo panic (2216-2225), combat action bonus (2226-2243), then the waste and
death terms (vecenv.rs:2245-2256):

```rust
            reward -= W_WASTE * wasted as f64;
            components.waste = -W_WASTE * wasted as f64;
            ...
            if !obs_alive && self.was_alive[i] {
                let death = self.reward_config.death_penalty();
                reward -= death;
                components.death = -death;
            }
```

and the terminal block at vecenv.rs:2362-2386:

```rust
            if done {
                let (place, _pn) = placement(self.ents(), obs_me, obs_alive, self.land_total);
                ...
                let no_play = timed_out || !self.ever_alive[i];
                components.terminal = (terminal_reward(place, won, no_play)
                    + fast_win_bonus(
                        won,
                        obs_tick,
                        self.max_episode_ticks,
                        self.reward_config.v84_fast_win_coef,
                    ))
                    * if n > 1 { 1.0 } else { 1.0 };
                if won {
                    components.terminal += self.reward_config.v85_extra_win_bonus;
                }
                components.terminal += v10_timeout_after_closeout_penalty(
                    timed_out,
                    self.closeout_tracker[i].reached(),
                    self.reward_config,
                );
                reward += components.terminal;
            }
```

Duo (team) mode adds eleven Ng-1999 PBRS shapers between
vecenv.rs:2259-2361 (`duo`, `eco`, `boat_commit`, `leftover_continent`,
`port_stand`, `city_stand`, `defense_stand`, `continent_span`, `boat_land`,
`partner_tiles`), each built from a `DominanceShaper` (curriculum.rs:668).
They are skipped when `n == 1`.

The transition object the trainer consumes, `vecenv.rs:631-637`:

```rust
pub struct EnvTransition {
    pub next_obs: PreparedObs,
    pub reward: f64,
    pub done: bool,
    pub info: Option<EpisodeInfo>,
    pub outcome: ActionOutcome,
}
```

So the trainer sees exactly one `f64` reward and one `bool done` per agent per
decision.

### 5.2 How the PufferLib env must surface the same signal

Per `puf_step`, for each agent `i`:

- `env->agents[i].rewards[0] = <reward for this decision>` (a single float,
  `NUM_ATNS` actions but one reward slot; buffer is `total_agents * sizeof(float)`,
  `pufferl.cu:1026-1028`).
- `env->agents[i].terminals[0] = 1.0f` on the decision the episode ends and
  0.0f otherwise.

Contract points that must hold for PufferLib to see the same signal the Rust
trainer sees:

- The reward must be written on every step, including the terminal step, and
  must include the terminal components (`components.terminal`, death) exactly as
  vecenv.rs:2245-2386 does. PufferLib bootstraps value on non-terminal rows, so
  a terminal row that carries only shaped reward under-counts the win.
- `terminals` must be set on the same step as the reward that contains
  `terminal_reward`, matching vecenv.rs:2362-2386.
- Auto-reset is the env's job: `puf_reset` is called by the trainer
  (`build.sh` compiles the env in; `pferl` restarts envs through `env_restart`,
  see `pufferl.cu:3324`), and the template resets inside `puf_step` when a
  terminal fires (`template.h:44-56`). Whichever pattern is chosen, the mask and
  observation must be valid for the state the policy will act on next.
- The Rust trainer's `solo_scale`/`tw` terms (vecenv.rs:2092-2097) depend on
  stage and player count; the C env must reproduce them or explicitly document
  the deviation. `not determined from code` whether `--duo` scaling should be
  replicated in the C env.

The C ABI today exports only a simple default (`include/openfront_env.h:178-184`):

```
   (tiles / map_land now) - (tiles / map_land before)
   + 1.0 if this agent won this decision
   - 1.0 if it was on the map and is no longer
```

That is a tile-share delta, not the shaped V10 reward. It is implemented at
`puffer_ffi.rs:409-440`. For training parity the env must not use
`ofenv_reward` as-is.

---

## 6. Bot policy hook

### 6.1 PufferLib side

`src/pufferenv.h:52-56` documents and defaults the hook:

```c
// Bot ladder writes this between rungs so one PuffeRL can eval a whole ladder.
// Default no-op; envs with scripted opponents #define PUF_HAS_BOT_POLICY and
// assign env->bot_policy.
#ifndef PUF_HAS_BOT_POLICY
static inline void puf_set_bot_policy(Env* env, int bot_policy) {
}
#endif
```

Two envs use it. `ocean/robocode/robocode.h:9` defines `PUF_HAS_BOT_POLICY` and
robocode.h:163-165 assigns `env->bot_policy = bot_policy;`. `ocean/slimevolley/slimevolley.h:7`
and :359-361 do the same, and slimevolley.h:368-369 asserts a single rung:

```c
    assert(env->bot_policy == BOT_ABRANTI
        && "slimevolley ships one bot: env.bot_policy must be 0");
```

The ladder driver is the bot-eval path in `pufferl.cu`. The ini keys are in
`config/default.ini`:

```ini
[selfplay]
# 0 = off. >0 = after train, eval final vs each bot in eval_bots for this many
# episodes each; mean env/perf becomes TrainResult.score (points=1 for Protein).
eval_bot_games = 0
# Ladder rungs, weakest first: the [env] bot_policy ids the env's header
# defines. Required when eval_bot_games > 0, e.g. eval_bots = 3,4,5,6.
eval_bots = 0
eval_bot_envs = 8192
eval_bot_threads = 0

# Per-rung [env] overrides for the bot ladder, as full section.key = value
# lines (e.g. env.dr = 0). Env-owned so core needs no per-env knowledge.
[bot_eval]
```

The driver, `pufferl.cu:3313-3331`:

```c
        double ladder[SELFPLAY_MAX_LADDER];
        int rungs = puf_ini_get_list(ini, "selfplay", "eval_bots", ladder,
            SELFPLAY_MAX_LADDER);
        // One PuffeRL for the whole ladder. close_pufferl frees nothing, so a
        // trainer per rung is a leak. Swap bot_policy and restart instead.
        PuffeRL* ep = eval_make(ini, ctx, EVAL_SCORE, 0);
        float sum = 0;
        for (int i = 0; i < rungs; i++) {
            if (PUF_BACKEND != PUF_GPU) {
                for (int e = 0; e < ep->vec->size; e++) {
                    puf_set_bot_policy(&ep->vec->envs[e], (int)ladder[i]);
                }
            }
            env_restart(ep);
            EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 0, 0, bot_games, NULL, 0);
            sum += r.perf;
            printf("bot_eval policy=%d games=%d perf=%.4f\n",
                (int)ladder[i], r.games, r.perf);
        }
```

Two hard constraints: the bot ladder runs only on the CPU backend
(`PUF_BACKEND != PUF_GPU` guard, pufferl.cu:3321), and
`selfplay.eval_bot_games` requires a non-empty `selfplay.eval_bots`
(assert at `pufferl.cu:3007-3010`).

### 6.2 OpenFront side, and the honest mapping

There is no integer bot difficulty id in the engine. `RlSession::reset` takes
`bots: u32` and `difficulty: &str` (`rl.rs:57-60`) and puts them on the wire as
`"bots"` and `"difficulty"` (rl.rs:77, rl.rs:79).

The difficulty string is consumed for `PlayerType::Nation`, not for
`PlayerType::Bot`. Three consumers, all in
`rust/engine/src/core/config.rs`:

- `start_manpower` (config.rs:215-231): `PlayerType::Bot` is a flat `10_000`;
  `PlayerType::Nation` maps `"Easy" => 12_500`, `"Medium" => 18_750`,
  `"Hard" => 25_000`, `"Impossible" => 31_250`, default `18_750`
  (config.rs:219-227).
- `max_troops` (config.rs:370-392): `PlayerType::Bot` divides by 3.0
  (config.rs:374-375); `PlayerType::Nation` multiplies by `0.5 / 0.75 / 1.0 /
  1.25` for Easy/Medium/Hard/Impossible, default `0.75` (config.rs:379-384).
- `troop_increase_rate` and `troop_increase_rate_raw` (config.rs:394-460):
  `PlayerType::Bot` multiplies by 0.5 (config.rs:407-409); `PlayerType::Nation`
  by `0.9 / 0.95 / 1.0 / 1.05`, default `0.95` (config.rs:411-421, 448-456).

So the ladder is really two knobs: the `difficulty` string (which tunes the
Nations team) and the `bots` count. The curriculum drives both per stage, e.g.
`Stage.difficulty` at curriculum.rs:729 with stage boundaries
`V10_MEDIUM_START 36`, `V10_HARD_START 50`, `V10_IMPOSSIBLE_START 60`
(asserted at curriculum.rs:2495-2499), and bot/nation density from
`V10_BOT_NATION_DENSITY` (curriculum.rs:835).

PROPOSAL for the env header's `bot_policy` contract (this is a design choice,
`not determined from code`): rungs 0..3 map to
`difficulty = Easy | Medium | Hard | Impossible` with the stage's `bots` count;
rungs 4.. map to `(bots, difficulty)` pairs taken from
`ofcore::curriculum::stages_for_schedule(CurriculumSchedule::V10)` so the
PufferLib ladder reproduces the V10 stage ladder. The rung table must be
documented in the header, as `config/default.ini` requires ("the [env]
bot_policy ids the env's header defines").

### 6.3 Single agent versus duo

- `n_agents = 1` (default): one human, `gameMode = "Free For All"`
  (rl.rs:76), one `Agent` in the PufferLib env, `agents[0].policy = 0`.
- `n_agents = 2`: `team_mode = true` (rl.rs:65), `gameMode = "Team"`, and
  `wire_json["playerTeams"] = json!("Humans Vs Nations")` (rl.rs:88). The two
  humans occupy slots 0 and 1 as `Agent` and `AgentB` (rl.rs:107-115), mapped to
  `AGENTRL1` / `AGENTRL2`.
- Team mode is not cosmetic. rl.rs:82-85: "Keep alliances ON. Teammates can
  donate/not-attack without a pact, but train gold is 35k for 'ally' vs 25k for
  'team', so the pair must actually alliance_request each other." Any duo env
  must therefore let `alliance_request` (action id 6) fire.
- `rl.rs:219-228` and `rl.rs:289-296` return the second agent's
  `(head, legal)` only when `n_agents > 1`, so a duo env needs both the step
  tuple and the per-agent legality.

PufferLib mapping: `num_agents` reported by `puf_init` is 2, `agents[0].policy`
0 and `agents[1].policy` 1 (`chess.h:3378-3386` is the reference: slot 0 is the
learner, slot 1 the opponent/historical policy; `vec.num_policies = 2` and
`hist_policy_percent > 0` are then required, `config/chess.ini`). In
`config/default.ini` the default `[env] num_agents = 1` must be overridden to 2
for duo.

One structural difference to state plainly: PufferLib 5.0 has no centralized
critic. The Rust trainer's duo mode uses a MAPPO critic fed by
`partner_players`/`partner_pmask`/`partner_scalars` (`policy.rs:163-172`,
`CENTRALIZED_VALUE_IN` policy.rs:50). In PufferLib the two policies are
independent actors and critics; training a team policy there is single-agent
PPO per slot, `not determined from code` for any value-sharing scheme.

---

## 7. Config and builds

### 7.1 ini keys

`config/default.ini` sections and the keys that matter for a new env:

- `[base]`: `env_name`, `gpu_offset`, `checkpoint_dir`, `log_dir`,
  `checkpoint_interval`, `eval_episodes`, `eval_agents`, `load_model_path`,
  `load_enemy_model_path`, `run_id`, `result_fd`, `cudagraphs`, `seed`,
  `reset_every_horizon`, `async`.
- `[vec]`: `total_agents`, `num_buffers`, `num_threads`, `num_policies`,
  `hist_policy_hidden_size`, `hist_policy_num_layers`, `hist_policy_percent`.
- `[env]`: env-owned kwargs, passed to `puf_init` as `Dict*`. Default content
  is `dr = 0`, `num_agents = 1`, `num_bots = 0`.
- `[policy]`: `hidden_size`, `num_layers`.
- `[train]`: `gpus`, `total_timesteps`, `learning_rate`, `anneal_lr`,
  `min_lr_ratio`, `gamma`, `gae_lambda`, `replay_ratio`, `clip_coef`,
  `vf_coef`, `vf_clip_coef`, `max_grad_norm`, `ent_coef`, `anneal_ent_coef`,
  `min_ent_coef_ratio`, `momentum`, `minibatch_size`, `horizon`, `vtrace`,
  `vtrace_rho_clip`, `vtrace_c_clip`, `verb_eps`, `verb_eps_anneal_start`,
  `verb_eps_anneal_end`.
- `[selfplay]`: `enabled`, `max_size`, `seed`, `opp_timeout_steps`,
  `eval_pool_size`, `eval_games`, `eval_bot_games`, `eval_bots`,
  `eval_bot_envs`, `eval_bot_threads`.
- `[bot_eval]`: per-rung `[env]` override lines.
- `[sweep]` plus `[sweep.<section>.<key>]` distributions.

Real env ini, `config/template.ini` in full:

```ini
[base]
env_name = template

[env]
size = 5

[train]
total_timesteps = 10_000_000
```

Real env ini with a ladder and selfplay, `config/chess.ini`:

```ini
[base]
env_name = chess

[selfplay]
enabled = 1
max_size = 500
opp_timeout_steps = 4_000_000_000

[vec]
total_agents = 8192
num_buffers = 1
num_threads = 2
num_policies = 2
hist_policy_percent = 0.1
hist_policy_hidden_size = 512
hist_policy_num_layers = 3

[env]
max_moves = 5000
reward_draw = 0
...
mode = 1
random_fen = 0
render_fps = 30
fen_curric_pct = 0.9

[policy]
hidden_size = 512
num_layers = 3

[train]
total_timesteps = 1_000_000_000_000
learning_rate = 0.000572786
...
horizon = 64
```

PROPOSAL, `config/openfront.ini` (no such file exists today; keys below are
all from the real files quoted above):

```ini
[base]
env_name = openfront

[env]
ticks_per_decision = 15
n_agents = 1
max_episode_ticks = 21000

[policy]
hidden_size = 768
num_layers = 6

[train]
total_timesteps = 100_000_000_000
learning_rate = 0.0001
gamma = 0.995
gae_lambda = 0.90
horizon = 64
minibatch_size = 8192
```

`hidden_size 768` / `num_layers 6` mirror the Rust trainer's `HIDDEN 768` and
`BLOCKS 6` (`policy.rs:46-52`) and are PROPOSAL only; PufferNet's
hidden/num_layers parameterise a different network shape than the Rust
`PolicyNet`.

### 7.2 (a) CPU eval on this VM

Verified working on this VM (NixOS, no compiler in PATH). The `--cpu` path
builds `src/puffercpu.c` plus `ENV_HEADER`. Two link inputs are missing from
the default Nix shell setup: `libGL` and `libomp5`. The command that was run
and produced `Built: ./template`:

```bash
cd /opt/data/workspaces/skg/PufferLib5
GL=$(nix eval --raw nixpkgs#libglvnd.outPath)
OMP=$(nix eval --raw nixpkgs#llvmPackages.openmp.outPath)
mkdir -p /tmp/ofomp
ln -sf "$OMP/lib/libiomp5.so" /tmp/ofomp/libomp5.so
ln -sf "$OMP/lib/libomp.so"  /tmp/ofomp/libomp.so
nix shell nixpkgs#clang -c bash -c \
  "export LIBRARY_PATH=$GL/lib:/tmp/ofomp; bash build.sh template --cpu"
```

Observations from that run: `build.sh` uses `${CC:-clang}` (build.sh:311) and
downloads `raylib-5.5_linux_amd64` if absent (name at build.sh:84, `download()`
at build.sh:110-118, `RAYLIB_URL` at build.sh:120); it is
already present in this clone, so the build is offline. The resulting binary is
`./template` (1102920 bytes). Running it here
(`timeout 30 ./template --headless eval_episodes=2`) produced a core dump with
no stdout, so the eval binary is not yet exercised end to end on this VM; raylib
window/texture initialisation is the likely cause but that is
`not determined from code`.

For OpenFront the same command shape is:

```bash
nix shell nixpkgs#clang -c bash -c \
  "export LIBRARY_PATH=$GL/lib:/tmp/ofomp; bash build.sh openfront --cpu"
```

This will fail today with `Error: ocean/openfront/openfront.h must typedef obs_t
for standalone eval` (build.sh:288-290) because that header does not exist yet.
`-DPUFFER_ENV_NAME=\"openfront\"` (build.sh:295) selects
`checkpoints/openfront/` for a checkpoint lookup (`puffercpu.c:679-698`).

### 7.3 (b) CUDA training build on a GPU box

The training build compiles `src/pufferl.cu` with `nvcc` and links CUDA, NCCL,
cuBLAS, cuSOLVER, cuRAND and NVML (build.sh:506-527):

```bash
cd /path/to/PufferLib5
./build.sh openfront            # native train/eval -> ./puffer
./build.sh openfront mybin      # native -> ./mybin
```

Requirements and how build.sh finds them:

- `nvcc`: `CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}`
  (build.sh:423). `NVCC="ccache $CUDA_HOME/bin/nvcc"` (build.sh:444).
- `nccl.h` / `libnccl.so`: searched in `/usr/include`, `/usr/local/cuda/include`
  and `/usr/lib/x86_64-linux-gnu`, `/usr/local/cuda/lib64`, then the
  `nvidia.nccl` wheel via `python -c "import nvidia.nccl, ..."` (build.sh:426-443).
  Note that fallback needs a Python with the wheel installed.
- Link line: `-L$CUDA_HOME/lib64 $NCCL_LFLAG ... -lcudart -lnccl -lnvidia-ml
  -lcublas -lcusolver -lcurand -lm -Xlinker=-lpthread $OMP_LIB`
  (build.sh:523-526), with `OMP_LIB=-lomp5` on Linux (build.sh:86).
- Compile flags: `nvcc -O2 --threads 0 -arch=native -std=c++17` by default
  (build.sh:240, 446, 506), `-Xcompiler=-fopenmp`, and
  `-DPUFFERLIB_BUILD_MAIN` (build.sh:515).
- `--float` switches precision with `-DPRECISION_FLOAT` (build.sh:43).

This cannot run on this VM. `nix shell nixpkgs#clang -c 'which nvcc'` returns
nothing on this host; there is no CUDA toolkit and no GPU. The trainer binary
must be built and run on a GPU machine.

---

## 8. What is needed next, and the risks

### 8.1 The C env header

`ocean/openfront/openfront.h` does not exist. It must:

1. `typedef float obs_t;` then `#include "pufferenv.h"` (the `obs_t` typedef is
   grepped by build.sh at three places, so this is mandatory).
2. `#define ACT_SIZES {...}`, `#define OBS_SIZE ...`, `#define NUM_ATNS ...`
   per section 3.4 (encoding choice A or B).
3. `struct Log` of floats only, and `struct Env` containing `Log log;`,
   `Agent agents[2];`, `int num_agents;`, `int tag;`, `int boundary_reached;`
   plus the `OFEnv` handle and scratch.
4. Implement `puf_init`, `puf_reset`, `puf_step`, `puf_render`, `puf_close`,
   `puf_log`, and `#define PUF_HAS_BOT_POLICY` plus `puf_set_bot_policy` for the
   ladder (section 6).
5. Link `libopenfront_engine.so`. `build.sh` has no mechanism for extra link
   inputs for a header-only env other than `EXTRA_LDFLAGS` (`build.sh:145`) and
   `EXTRA_SRC` (`build.sh:144`), and neither is settable from the `[env]` ini
   section, so the env's own link path needs a build.sh touch or a
   pre-built static/shared library placed where `LINK_ARCHIVES` can reach it.
   This is an unresolved build-system gap: `not determined from code` how the
   parallel C ABI work intends to be linked into `./puffer`.

### 8.2 The C ABI shim

Present as untracked work in progress: `rust/engine/src/puffer_ffi.rs` and
`include/openfront_env.h`, wired into the crate at `rust/engine/src/lib.rs:19`.
Remaining work visible from the header itself:

- `ticks_per_decision` defaults to 8 (header line 26, `puffer_ffi.rs:228`) but
  the trainer uses 15 (section 2.3). Set it explicitly or change the default.
- `ofenv_reward` is a tile-share delta plus win/death (header lines 178-184),
  not the V10 shaped reward (curriculum.rs / vecenv.rs). Either export the real
  shaping or keep the shaping in the C env and stop calling `ofenv_reward`.
- The `stat`, `transient`, `grid_coarse`, latent, ego, defense-bonus and local
  planes are not exported (header lines 68-75), so an AE-based encoder has no
  data path from C today.
- Verification status: `rust/engine/src/bin/puffer_smoke.rs` exists, untracked;
  its results were not established here.

### 8.3 The PufferNet Encoder work

A custom `Encoder` (algo.cu:23-32) installed via `create_custom_encoder`
(ocean.cu:56-70) must:

- Run the `ae_v32_nostatic_d8c32` fine encoder at 1/8 and, optionally, the
  `ae_v32_nostatic_d16c32` coarse encoder at 1/16 (ae.rs:13-14), producing the
  32-channel latent that becomes the first 32 of `C_GRID = 99` (policy.rs:26).
- Concatenate the 6 exact static planes, 3 ego planes, 1 defense-bonus plane
  and 57 transient planes (policy.rs:1-8), all at the /8 resolution, and the
  fine tile plane (`C_GRID_FINE = 100`, policy.rs:27).
- Provide `backward`, `reg_params`, `reg_train`, `reg_rollout`,
  `create_weights`, `init_weights` for that stack, and set
  `activation_size` to its own activation struct (algo.cu:26-32).
- Respect `PPO_MAX_HEAD_A` sizing, which is the max `ACT_SIZES` entry
  (algo.cu:1278-1285); with a 37,500-wide head that cache is large.

The `in_dim`/`out_dim` contract is `input_size` to `hidden_size` in
`build_arch` (algo.cu:997), so the encoder must produce the flat hidden vector
the PufferNet trunk expects; mixing a spatial tower into that shape is the
research part, and the code comment at algo.cu:975-978 says as much ("we can't
narrow it yet because fast, general encoder arch is an unsolved research
problem").

### 8.4 Risks

- 21,000-tick horizon. `DEFAULT_MAX_EPISODE_TICKS = 21_000` (lib.rs:18) at 15
  ticks per decision is 1400 decisions per episode. PufferLib's default
  `horizon` is 64 (`config/default.ini`), so a PufferLib episode spans roughly
  22 horizon resets and bootstrap boundaries. Compare the Rust trainer, which
  carries LSTM state across horizons unless `reset_every_horizon = 1`
  (`default.ini` `[base]`), and uses `RECURRENT_HIDDEN 512`
  (policy.rs:57-60). Either the horizon is raised, the episode budget is
  shortened, or a recurrent policy is added; none of that exists in the C path
  today.
- C ABI plus build step. The env needs `cargo build --release -p
  openfront-engine` to produce `libopenfront_engine.so` (crate-type at
  `rust/engine/Cargo.toml:14`), and then a C link step that `build.sh` does not
  currently express for header-only envs (section 8.1). The Rust build is a new
  prerequisite for the PufferLib target, and it must be reproducible on the GPU
  box as well as here.
- 5.0 API churn. PufferLib 5.0 is branch `5.0` at `6ffa5b1` with no tags in
  this clone. Pin the tag `5.0-experiments` if it exists upstream
  (`not determined from code` from this clone: `git tag` is empty). The contract
  is header-only and the mask/Agent layout above was read from this exact commit,
  so any bump needs a re-read of `pufferenv.h`, `pufferl.cu` and `algo.cu`.
- Mask size. At 40,893 floats per agent per decision (163,572 bytes at 4 bytes
  each) and 2 agents in duo mode, the mask is 2 * 40,893 * 4 = 327,144 bytes of
  upload per env
  step, on top of the 4270-float observation. `pufferl.cu:1093-1103` copies the
  whole row every step (`cpu_upload`), and `pufferl.cu:820-826` gathers the
  whole row to device in the rollout path. This should be measured before the
  mask is made this wide.
- Reward parity. The V10 shaping is a large composition
  (vecenv.rs:2085-2386) with stage gates and duo PBRS terms. Reproducing it in
  C is mechanical but error-prone; the `RewardComponents` struct
  (curriculum.rs:334-352) is the checklist to port.

---

## Appendix: file and line index

PufferLib 5.0 (`/opt/data/workspaces/skg/PufferLib5`, `6ffa5b1`):

- `src/pufferenv.h`: 24-30 Agent, 35-39 PUF_CPU/PUF_GPU/PUF_BACKEND, 44-50 six
  functions, 52-56 bot policy hook.
- `src/pufferl.cu`: 64-65 include ENV_HEADER, 922 LOG_NF, 940-951 log_reduce,
  968-969 puf_bind_stream/puf_vec_create default hooks, 996/1358 MY_VEC_INIT/
  CLOSE, 1024-1028 host buffers, 1048/1073 policy tags, 1079-1084 agent
  pointers, 1093-1103 cpu_upload, 1374 puf_log call, 1852-1872 act_n/mask_size,
  1880-1891 mask alloc and memset-to-1, 2915-2921 num_agents probe,
  3007-3010 eval_bots assert, 3313-3331 ladder loop, 3321 CPU-only guard.
- `src/algo.cu`: 23-32 struct Encoder, 975-978 custom arch comment,
  986-998 build_arch, 1270-1276 NUM_ATNS/ACT_SIZES errors, 1278-1285
  PPO_MAX_HEAD_A.
- `src/ocean.cu`: 55-70 create_custom_encoder dispatch.
- `src/puffercpu.c`: 606-612 ENV_HEADER include, 744-745/762 ACT_SIZES and
  OBS_SIZE use, 679-698 checkpoint lookup.
- `build.sh`: 84 raylib name, 86 OMP_LIB=-lomp5, 110-127 raylib download,
  144-145 EXTRA_SRC/EXTRA_LDFLAGS, 240 NVCC_OPT, 279-303 --cpu branch,
  311 CC default clang, 423-446 CUDA/NCCL discovery and nvcc,
  451-462 native env header selection, 506-528 training compile and link.
- `ocean/template/template.h`: 3-4 obs_t and include, 6-8 macros, 15-24 Log and
  Env, 33-61 reset/step/render, 89-93 close, 95-100 init, 102-105 log.
- `ocean/chess/chess.h`: 11-15 obs_t and macros, 19-21 PUF_STEPS_PER_SEC and
  MY_VEC_INIT/CLOSE, 348-356 actions and OBS_SIZE, 379-412 Log, 428-529 Env,
 532-540 pointer sync, 1603-1613 mask write, 3366-3386 puf_init,
 3389+ puf_log.
- `ocean/minimal/minimal.h`: 6-12 obs_t and macros, 25-30 Env required members.
- `ocean/go/go.h`: 8-13 obs_t and macros.
- `config/default.ini`, `config/template.ini`, `config/chess.ini` as quoted.

OpenFront (`/opt/data/workspaces/skg/openfront-ai`, `05cee07`):

- `rust/engine/src/rl.rs`: 1-4 module doc, 55-63 reset, 64-65 n_agents and
  team_mode, 67-69 map dir, 72-89 wire config and playerTeams, 107-115 human
  players, 239-246 step doc and signature, 271-279 tick loop, 295/299 legality.
- `rust/engine/src/session.rs`: 11-13 AGENT_CLIENT_ID / _2 / AGENT_CLIENT_IDS.
- `rust/engine/src/obs_typed.rs`: 29 entities_typed, 187-336 legality_typed.
- `rust/engine/src/puffer_ffi.rs` (untracked): 98 obs stride, 105-122 mask
  offsets and asserts, 176-247 config parse, 224-228 ticks_per_decision,
  316-407 buffer build, 363-407 write_obs/write_mask, 409-440 update_rewards,
  441-485 build_meta, 585-806 exports.
- `include/openfront_env.h` (untracked): 20-30 config and intents, 43-75 obs
  layout, 77-102 mask layout, 104-147 macros and tile packing, 149-197 API.
- `rust/engine/Cargo.toml`: 8-14 lib name and crate-type.
- `rust/engine/src/core/terrain.rs`: 89-104 load_fresh_terrain_from_dir.
- `rust/engine/src/core/config.rs`: 215-231 start_manpower, 370-392 max_troops,
  394-460 troop_increase_rate, 794/535 default "Medium".
- `rust/ofcore/src/lib.rs`: 18 DEFAULT_MAX_EPISODE_TICKS, 22
  WATCH_TICKS_PER_DECISION, 28-30 DEFAULT_WATCH_MAX_STEPS, 43-46 test.
- `rust/ofcore/src/feat.rs`: 8-30 constants, 32-52 action ids, 54-76 names,
  78-95 build/nuke tables, 100-120 transient bases, 126-177 target predicates,
  465-490 Legal, 563-582 Feat, 617 make_clut, 650-663 featurize, 691-1008
  planes and masks, 955-968 scalars, 1010-1056 mask fill, 1038 legal_tile call,
  1392 legal_tile_mask.
- `rust/ofcore/src/curriculum.rs`: 10-31 weights, 34-42 team helpers, 45-48
  version tags, 50-89 V10 constants and RewardConfig, 219-330 stage predicates,
  334-352 RewardComponents, 446-470 shaping helpers, 636-668 churn penalties and
  DominanceShaper, 690-704 share and closeout, 726-733 Stage, 737-826 maps and
  stage constants, 835-915 bot/nation density, 1021-1039 gates and LR, 1119
  build_v10_stages, 1177-1179 stages_for_schedule, 1209 timeweight, 1366-1375
  terminal_reward, 1382-1430 V10 terminal shaping, 1781-1830 V10 action terms.
- `rust/oftrain/src/policy.rs`: 1-8 grid channel doc, 14-65 dims and constants,
  145-172 Obs, 756+ PolicyNet.
- `rust/oftrain/src/ae.rs`: 13-14 AE stream names, 23-29 AE constants, 268-330
  SpatialAE/SpatialAePair, 371-389 terrain packing, 516-551 encode pipeline,
  634-639 crop sizes, 719+ batched encode.
- `rust/oftrain/src/vecenv.rs`: 631-637 EnvTransition, 938-1071 VecEnv fields,
  1107-1111 decision_ticks selection, 1233/1816/2490 bridge.step calls,
  1749/1757 apply and apply_agents, 2085-2386 reward assembly.
- `rust/oftrain/src/train.rs`: 3932 step cap.
- `openfront/resources/maps/`: 100 lowercase map directories, each with
  `manifest.json`, `map.bin`, `map4x.bin`, `map16x.bin`, `thumbnail.webp`.