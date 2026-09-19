# OpenFront RL — experiment plan

Goal: a policy that beats most humans. The curriculum gate needs a rolling
40-episode win rate > 0.90 on a stage before promoting it. **Never lower that
bar.**

The operator job (`openfront-rl-operator`, every 2h) reads this file each run and
advances exactly one step.

## The win condition (read this before trusting any win rate)

`win_check.rs`: `if percentage_owned > 80% || time_limit_reached { game.winner =
leader_id }` — but see D2: in the RL env `rl.rs` sets no `maxTimerValue`, the only
other clock is the 170-minute hard limit, and the env's own 21,000-tick cap fires
first, so the `time_limit_reached` branch is **unreachable in training**. A win
therefore always means `game.winner` was set by a **>80% map takeover**, and an
episode that hits the 21,000-tick cap ends with no winner and counts as a loss
(`died=1`).

**Corrected 2026-09-18:** the older claim here — "the tile leader wins at the
time limit, so a uniform-random expander out-tiles weak bots and wins ~100%" —
was written while the `--env.stage` knob was inert (everyone played the ini's
`bots=3 Easy pangaea`) and while a loss terminal was still positive (D7). With
both fixed, measured all-zero controls win **0.298 at stage 0, 0.600 at stage 1,
0.100 at stage 2** (n=160/160/120) and lose the rest, so a win rate here is
evidence on a config where random play loses. Any win rate is still only
meaningful when read against that matched control.

## How to measure

```
STAGE=<n> bash eval_policy.sh <ckpt|latest> <episodes> <envs>
```

`STAGE` pins the curriculum stage the eval starts on (default 0). `<episodes>` is
the **total** number of episodes, not per env: the eval loop waits until
`env/n >= eval_episodes` and `env/n` is a *sum* over envs (`env_log_sum` in
pufferl.cu), so `--vec.total_agents` only parallelizes the same total. Passing 4
with 8 envs buys 4 episodes, not 32. Size it deliberately: pass 100+ for a number
worth quoting.

**Do not read the `stage=` field on the CUDA_EVAL line as the stage that was
played** (D10, resolved 2026-09-18). `eval_loop` logs with `vec_log(..., clear=0)`
while the trainer clears every interval, and `vec_log` divides *every* channel by
the accumulated episode count, so a level channel like `stage` decays with n
(1.0 / 0.5 / 0.33 / 0.25 for 1 env and 4 episodes; 0.05 for 8 envs and 160). The
pin itself is fine - `OF_TRACE=1` shows `stage=1` on every `OFTRACE-SUM` line, and
stage 0 vs stage 1 give different games. Verify a pinned stage with `OF_TRACE=1`
(or the first dashboard sample, while n is small), never with that field. Win
rates and scores are unaffected: their numerator and denominator accumulate
together.

**A negative result that cost a day: `--env.stage=N` used to be inert.** In the
FFI, `ofenv_create` only *labels* the stage — `new_state` builds the session from
the cfg's own bots/difficulty and `RlSession::reset` takes no stage argument;
`ofenv_set_stage` is what applies a stage's bots/nations/difficulty. So every
per-stage eval played the *same* fixed game (the ini's `bots = 3`, `Easy`,
`pangaea`), which is trivially winnable, and an all-zero policy "won" 100% there.
Proof it was inert: stage 0 and stage 4 returned byte-identical tick counts
(10331 / 10661 / 10771 / 12531) and scores.

Fixed in `openfront.h` (apply `ofenv_set_stage` right after `ofenv_create`). This
also fixes the trainer: its **initial** stage params were never applied, so a run
labelled stage 0 was actually playing the ini's 3-Easy-bot pangaea.

Measured with the fix, 4 episodes/env x 4 envs per cell:

| policy | stage 0 | stage 4 | stage 6 |
|---|---|---|---|
| all-zero control | 0.0 wins, score -6.9 | 0.0, -25.5 | 0.0, -16.8 |
| trained | 0.2 wins, +72.4 | 0.0, -18.4 | 0.0, -16.2 |

Read: stage 0 is the only cell that separates (and n=4 is thin). Stages 4 and 6
are too hard for the current policy — both policies lose every episode, so they
discriminate nothing either. Widen n before drawing conclusions, and prefer the
**return** as the continuous signal when the win rate is 0 for both.

Secondary metric with no headroom problem: **curriculum progress** — the stage
the ladder reaches before it stalls, plus `stage_wins/40` there.

## Hard-won invariants

- **Never launch a trainer outside the unit.** It dies with the gateway and once
  triggered a global OOM that killed the real run.
- **Never build over a running `./puffer`.** Stop, build, start.
- **Always warm-start** and confirm the launch prints `pufferl: resumed schedule
  from ... at step N (epoch E/T, X% into the cosine)`.
- **An eval pauses the trainer** (it is CPU-bound) and stands the watchdog down
  via `WATCHDOG_OFF`. Verify the flag is gone afterwards.
- **Never leave the trainer stopped**; never leave `WATCHDOG_OFF` in place.
- **Every eval restarts the trainer and the curriculum stage is not persisted**,
  so each eval costs the ladder its warm-up. Measure less often, or fix E6.
- Per-experiment overrides go in a runtime drop-in via `run_experiment.sh` (it
  preserves `limits.conf`, StartLimitBurst=30). `/etc` is read-only.
- Reward constants live in `openfront-ai/rust/ofcore/src/curriculum.rs`
  (`W_WIN = 30.0`): changing one needs an engine rebuild, not a config flag.
- Throughput ~100 steps/s (~8.6M/day), and a longer rollout does not raise it.
  Wall-clock is the binding constraint.

## Ladder (do the first unfinished step)

### E2 — SCHED_EPOCHS=600 [CLOSED 2026-09-17, superseded by D8]
Ran 18:22-~19:54 and did complete its anneal: `epoch 300/600 (50%)` -> `600/600`,
LR 0.005 -> 0.00043, ent_coef 0.0985 -> 0.00985. Signal it was written to
produce (entropy falls): **did not happen** - entropy over the e2 segment was
7.997 early vs 8.004 late, i.e. flat at ~ln(legal). Nothing to attribute, because
the same `/run` drop-in that held SCHED_EPOCHS held it against a 20M-step budget:
2.46M steps of 20M, so the cosine hit its floor at **12% of the run** (see D8,
found and removed 21:14). Do not re-run E2; the anneal now spans the whole budget
by declaration.

### E3 — re-measure the ladder now that stage 0 is real [DONE 2026-09-17 22:00]
Fresh stage-0 measurement, n=160 total episodes, 8 envs, on the newest checkpoint
(1789699967701/0000000002458112.bin), against the all-zero control at the same
stage and n:

| policy | wins/n | win rate | score | perf |
|---|---|---|---|---|
| trained | 64/160 | 0.400 | 122.29 | 123.81 |
| all-zero control | 48/161 | 0.298 | 77.53 | 78.52 |

First time the trained policy has separated from the control anywhere: +0.102 win
rate (SE ~0.053, z~1.9 - suggestive, not a settled result) and +44.8 return.
Stage 0 is confirmed **not** saturated: random legal play loses 70% of episodes,
so it is a config where a win means something. Both sit far under the 0.90 gate.
The ladder was not observed stalling anywhere else: no `stage N -> N+1` line has
been logged by any launch since the stage knob was made real, so the ladder has
never moved off stage 0 and the stall point is "stage 0, win rate ~0.4".

### E4 — HORIZON=128 [MEASURED 2026-09-18 03:19, n=160 @ STAGE=0]
Doubles samples per update (8192) at the same step rate. ~16k legal actions
against 4096 samples/rollout means most actions get zero samples per update.
Keep `minibatch_size % horizon == 0` (MB=512 and REPLAY=4 satisfy all three
launch asserts at horizon 128, reverified: 512%128=0, 512<=128*64, 64%4=0).
Delayed 2026-09-17 22:00 because D8 was 45 min old then.

The D8 wait is now resolved by measurement (below): at ~3h of post-D8 training
the policy is back at exact win-rate parity with the all-zero control, so the
D8 LR fix alone did not unlock learning and the horizon is the next variable.
Expect the resume line to read `epoch ~550/2441` (the epoch scale doubles with
horizon; the *fraction* into the cosine must stay 22.5% if the schedule survived).

Result (2026-09-18 03:19, 2.1h / 2.05M steps into E4, ckpt
1789717793731/0000000006554112.bin, STAGE=0, n=160, 8 envs, trainer thawed at the
same PID after):

| policy | wins/n | win rate | score | perf |
|---|---|---|---|---|
| trained (E4) | 64/160 | 0.400 | 104.15 | 105.38 |
| all-zero control | 48/161 | 0.298 | 77.53 | 78.53 |

Read: win rate +0.102 over control (SE ~0.052, z~2.0) - the same reading E3 gave,
so suggestive, not settled. The return separates again (+26.6; E3 +44.8, E3-b
+20.3). E4 changed nothing observable in the update dynamics: entropy over its
segment stayed 8.1-8.9 (flat at ~ln(legal)), kl/clipfrac 0.000 throughout,
stage_advances 0. Doubling the rollout did not make the policy commit, so
samples-per-update is not the binding constraint. E5 (ent_coef) is next, carried
on top of HORIZON=128.

### E3-b — re-measure after D8, trained vs control [DONE 2026-09-18 00:49]
Same protocol as E3 (STAGE=0, n=160 total, 8 envs), on the newest checkpoint
(1789712267631/0000000004506112.bin, 4.51M steps, ~3h of post-D8 training):

| policy | wins/n | win rate | score | perf |
|---|---|---|---|---|
| trained | 48/160 | 0.300 | 97.84 | 99.47 |
| all-zero control | 48/161 | 0.298 | 77.53 | 78.52 |

The control reproduced E3's numbers to the last digit (77.525467 / 78.523750),
so the harness is deterministic and the two trained readings are comparable.
Read: **win rate back at exact parity** (E3's 0.400 was 0.102 above control at
z~1.9, i.e. never settled); only the return still separates, and by less
(+44.8 -> +20.3). Entropy over the run sits at 8.5 and *rises* (8.0 early ->
8.5 late) while `kl`/`clipfrac` print 0.000 - the policy is drifting toward
uniform, not committing. With `ent_coef = 0.0985` (98x PufferLib's default)
the entropy bonus plausibly outweighs the policy gradient, which is E5's
target. Sequence stays E4 then E5, one variable per run.

### E5 — ENT=0.02 [APPLIED 2026-09-18 03:22, carried HORIZON=128]
Force commitment, or take smaller steps. Applied `ENT=0.02` (down from 0.0985).

Why ENT and not LR: entropy sits at ~ln(legal) and *rises* (8.0 early -> 8.5-8.9)
while LR anneals down, which is what an entropy bonus outweighing the policy
gradient looks like; `ent_coef = 0.0985` is ~98x PufferLib's default and was
inherited, not swept against a working run. The per-update policy move has also
collapsed: kl reads 0.000 (|kl| < 5e-4, it is a per-sample mean - see the code
note in eval-log.txt) since ~18:42 yesterday, against 0.001-0.002 earlier. LR is
the *other* candidate and stays untouched - one variable per run.

Launch verified: `pufferl: resumed schedule from 0000000006554112.bin at step
6554112 (epoch 800/2441, 32.8% into the cosine)`; cmdline carries
`--train.ent_coef=0.02 --train.horizon=128`; PID 1276244 in
`/system.slice/openfront-train.service`. Usual restart cost: the curriculum is
unpersisted (E7), so stage 0 restarts with an empty 40-episode window (~1.7M
steps to refill) - do not read that as a regression.

### E5 — ENT=0.02 [CLOSED 2026-09-18 05:36 — the unlock]
Applied 2026-09-18 03:22 (carried HORIZON=128). Measured 2.2h / 1.64M steps in, STAGE=0, n=160, 8 envs,
on ckpt `1789726805008/0000000008192512.bin`:

| policy | wins/n | win rate | score | perf |
|---|---|---|---|---|
| trained (E5) | 144/160 | **0.900** | 281.48 | 282.51 |
| all-zero control | 48/161 | 0.298 | 77.53 | 78.52 |

+0.602 win rate (SE ~0.040, z~15) - the first **decisive** separation in this project, and stage 0 is
now cleared at the 0.90 bar. Return 281.48 vs 77.53 = +203.9 (E3 +44.8, E3-b +20.3, E4 +26.6). Over
the E5 segment: entropy 8.3-8.9 (E4) -> 6.3-7.0, episode_length ~1100 -> ~400 decisions. The trainer's
own gate agrees with the independent eval: 52 `stage 0 -> 1` promotions at win_rate 0.925-0.950 from
~04:55, the first ladder movement this project has recorded. Read: `ent_coef = 0.0985` was the blocker,
not the LR schedule and not the rollout size.

### D10 — a pinned non-zero stage DOES hold; the `stage=` field was the lie [RESOLVED 2026-09-18 09:20 — not a defect]
Discovered 2026-09-18 06:30, diagnosed wrong, resolved 09:20. Original claim: a `STAGE=1` eval reported a
stage field decaying as 1/n_episodes (1 env / 4 episodes -> 1.000, 0.500, 0.333, 0.250; 8 envs / 160
episodes -> 0.05 = 8/160), read as "one stage-1 episode per env, stage 0 for the rest", and the 05:59
stage-1 eval was marked VOID on that basis. **That reading was wrong.**

What is actually true, verified with `OF_TRACE=1` on a 1-env / 4-episode stage-1 eval of ckpt
`1789726805008/0000000009011712.bin`:

- Every `OFTRACE-SUM` line across all four episodes prints `stage=1`. That field is `env->log.stage`,
  written at each episode end as `(float)env->stage`, so `env->stage` holds 1 through every auto-reset.
  The pin is NOT lost. (`of_sync_meta` re-reads it from the engine meta each step and `ofenv_reset`
  rebuilds from `state.cfg`, whose `stage` `ofenv_set_stage` sets - engine and C env both innocent.)
- The pin is *effective*: the same checkpoint, 1 env, 4 episodes, stage 0 vs stage 1 give different
  games - score 301.793640 with episode ticks 9801/13111/5281/14271 (stage 0, 2 bots) vs 305.985260
  with the last episode ending at tick 4421 (stage 1, 4 bots).

The decay is an **aggregation artifact in the eval path**, not a stage the env played:
`eval_loop` -> `trainer_eval_log` -> `vec_log(p->vec, out, 0)` (src/pufferl.cu:2553, clear=0) while the
trainer clears every interval (`vec_log(..., 1)`, :3352). `log_accum` adds the *whole* `Log` of every
env with `log->n != 0`, and `vec_log` then divides every channel by the accumulated episode count. A
level channel (`stage`, and `stage_wins` as a running total) therefore reads
(sum over intervals of stage) / (episodes so far), which shrinks as n grows - exactly 1.000, 0.500,
0.333, 0.250 for one episode per interval, and 8/160 for 8 envs / 160 episodes. Sum-like channels
(`wins`, `losses`, `score`) survive it because both sides accumulate.

Consequences, all of which change earlier conclusions:

- **The 05:59 STAGE=1 eval is valid, not void**: 144/160 = 0.900 on ckpt `.../0000000008192512.bin`.
  Stage 1 was cleared at the 0.90 bar at 2.2h into E5, and the trainer's own 1 -> 2 promotions
  (win_rate 0.925-0.975) agree with it.
- **E6 is unblocked**: it needed a stage-1 baseline and a working pin; both exist.
- To check which stage an eval played, read `OFTRACE-SUM`'s `stage=` (or the first dashboard sample,
  while n is small) - never the `CUDA_EVAL stage=` field, which is meaningless at any n > 1.
- The clean fix is in `pufferl.cu`'s eval path (publish the last env stage, or clear per interval and
  carry the episode count separately). It only needs a `puffer-eval` rebuild, no trainer restart, so it
  can be done while the run is live.


### E5-b — stage-1 baseline (first valid one) [DONE 2026-09-18 09:33, n=160 @ STAGE=1]
Taken on ckpt `1789726805008/0000000009011712.bin` (9.0M steps, 6.2h into E5), 8 envs, with the
all-zero control at the same stage and n:

| policy | wins/n | win rate | score | perf |
|---|---|---|---|---|
| trained (E5) | 144/160 | **0.900** | 279.88 | 280.56 |
| all-zero control | 96/160 | 0.600 | 166.45 | 167.01 |

Read: +0.300 win rate over the control (SE ~0.05, z~6) and +113.4 return - a real separation, and
stage 1 is *not* free (random legal play wins 60% of its games there, against 30% at stage 0). The
policy sits exactly on the 0.90 gate: 144/160, the same count as the 05:36 stage-0 measurement, and
the trainer's own 1 -> 2 promotions fired at window rates of 0.925-0.975. So the ladder is at the
boundary of stage 1, which is the expected shape for a gate that needs > 0.90 strictly.

Cost note: the control at stage 1 is the slow half of this measurement (a zero policy never wins, so
every episode runs to the 21,000-tick cap: ~35 min for 160 episodes vs ~15 min for the trained
policy). Budget an hour for a stage-1 pair, or run the control once and reuse it (it is
deterministic).

### E6 — reward alignment: raise `W_WIN` [CLOSED 2026-09-18 12:20 — the premise was arithmetically wrong]

E6 was written on the claim "a win is worth `W_PLACE*1^-1.5 + W_WIN = 15 + 30 = 45`,
plus up to 40 fast-win, against a return of ~280: only ~20-30% of the return is
about winning". **That sum misses `v85_extra_win_bonus = 200.0`**, a flat terminal
bonus the FFI pays on every win (`rust/engine/src/puffer_ffi.rs:411`, inside
`trainer_default_reward_config()` — the config `ofenv_create` installs at `:1045`),
and it rounds `v84_fast_win_coef = 40.0` down to "up to 40" as if it were small.
The live terminal, term by term:

| outcome | composed terminal | source |
|---|---|---|
| win | `15*place^-1.5 + 30 + 40*(1-tick/max) + 200` = **245..285** | curriculum.rs:1379,455-460; puffer_ffi.rs:858-866 |
| death / elimination | `-30` terminal `-3` death transition = **-33** | puffer_ffi.rs:846-856 |
| timeout after reaching >=45% land | `-30 -20` = **-50** | puffer_ffi.rs:868-872 |
| plain timeout | **-30** | curriculum.rs:1377 |

Corroborated by the live dashboard, not just by reading the file: `episode_return
244.639` at `wins 0.818` is what a ~279 win terminal against a -30 loss predicts
(0.818*279 - 0.182*30 + shaping ~ 235-245); a 45/ -30 pair predicts ~82 and is
flatly inconsistent with it. So **winning already owns ~90% of |return|** — the
misalignment E6 was written to fix does not exist in the live reward.

Raising `W_WIN` to 100 would move the win only +25% (279 -> ~350) while taking
every loss to **-100** (3.3x), i.e. it relativizes losing and pushes the policy
toward safety — the opposite of what a stage whose win requires an 80% map
takeover needs. The one-line source change was made and then reverted (verified
with `git diff`: only the D7 fix remains in `curriculum.rs`); nothing was rebuilt
and **the run was not restarted**, because a restart drops all 32 envs to stage 0
with empty windows (E7) and costs ~6h of ladder re-climb — too much to spend on a
rebalancing whose premise is falsified.

**Lesson, same shape as D10:** a "the reward is misaligned" claim needs the
*composed* terminal, not the named constant. `W_WIN` is the small part; the
`v8x_*` bonuses are the large part, and they live in `puffer_ffi.rs`, not in the
`curriculum.rs` constant the plan quoted.

### E7 — persist the curriculum stage across restarts [DONE 2026-09-18 18:45 — first live restore verified]
`env->stage` and the 40-episode window live nowhere but memory, so every restart
drops all envs to stage 0 with an empty window. With the gate working this is a
pure waste: write the stage to a sidecar next to each checkpoint and restore it
(per-env: the sidecar needs one stage per env, keyed by the env's seed or index).
Land it *at* a restart that is happening anyway, not to trigger one — the restart
itself costs the ~6h of ladder re-climb it is meant to save.

### E8 — more agents (AGENTS=128)
More diverse data per update. Was blocked by RAM; an idle browser held 5.6 GB and
was freed 2026-09-17, so re-check headroom (`free -h`).

## Results log

| exp | config | started | result | notes |
|---|---|---|---|---|
| e1_horizon256_sched400 | HORIZON=256 SCHED_EPOCHS=400 | 18:13 | superseded | 4x data/update but the anneal window would take 16h |
| e2_sched600 | HORIZON=64 SCHED_EPOCHS=600 | 18:22 | closed | anneal ran to the floor but entropy did not fall; the drop-in pinned it to 12% of the 20M budget -> D8 |
| bugfix | ofenv_set_stage at startup | 19:00 | **landed** | stage knob was inert; control "won" 100% before, loses 4/4 after |
| measure | zeros vs latest @ stage 0, n=100 | 19:25 | **parity** | both 0.3 wins; scores 92.67 vs 99.26 (SE ~15) -> indistinguishable |
| bugfix | eval freezes instead of restarting | 19:40 | **landed** | SIGSTOP/SIGCONT keeps the PID and the ladder; verified same PID |
| bugfix | D8: /run drop-in removed, LR off its floor | 21:14 | landed (outside this job) | trainer restarted 21:14:30, `epoch 600/4882, 12.3% into the cosine` |
| e3_stage0_real | measure trained vs zeros, STAGE=0, n=160 | 22:00 | **separation** | 0.400 vs 0.298 wins, 122.29 vs 77.53 return; first separation recorded |
| e3b_post_d8 | re-measure trained vs zeros, STAGE=0, n=160 | 00:49 | **parity** | 0.300 vs 0.298 wins (E3's 0.400 was inside noise), return 97.84 vs 77.53; entropy 8.0 -> 8.5, kl/clipfrac 0.000 -> D8 alone did not unlock learning |
| e4_horizon128 | HORIZON=128 (rollout 8192 samples, 64 updates/rollout) | 00:52 | **measured, no dynamics change** | 03:19, 2.1h in, n=160 @ STAGE=0: 0.400 vs control 0.298, return 104.15 vs 77.53; entropy flat 8.1-8.9, kl/clipfrac 0.000, stage_advances 0 |
| e5_ent002 | ENT=0.02 (was 0.0985), HORIZON=128 carried | 03:22 | **CLOSED: the unlock** | 05:36, 2.2h in, n=160 @ STAGE=0: 144/160 = 0.900 vs control 0.298 (z~15); return 281.48 vs 77.53. Entropy 8.3-8.9 -> 6.3-7.0; 52 `stage 0 -> 1` promotions from ~04:55 |
| d10_resolved | measurement-protocol fix, no run change | 09:20 | **not a defect** | the pinned stage holds (OF_TRACE `stage=1` every episode; stage 0 vs 1 = different games). `CUDA_EVAL stage=` is a `vec_log` aggregation artifact, clear=0 in the eval path. Unblocks E6 |
| e5b_stage1 | stage-1 baseline + control, n=160 | 09:33 | **valid, at the gate** | trained 144/160 = 0.900 (279.88) vs control 96/160 = 0.600 (166.45), z~6. Ladder reached stage 2 for the first time (19 `stage 1 -> 2` promotions in the live segment) |
| e6_closed | reward audit only, **no run change** | 12:20 | **closed: premise arithmetically wrong** | E6 assumed a win pays 45. The live FFI terminal pays `15*place^-1.5 + W_WIN(30) + fast_win(<=40) + v85_extra_win_bonus(200)` = 245..285 (`puffer_ffi.rs:411`, applied `:858-866`), corroborated by the dashboard's own `episode_return 244.6 @ wins 0.818`. Raising `W_WIN` to 100 moves the win +25% and every loss 3.3x (toward safety). Edit made and reverted; no rebuild, no restart |
| e5_stage2 | trained vs control @ STAGE=2, n=160/120 | 12:41 | **the wall is stage 2** | trained 64/160 = 0.400 (92.64) vs control 12/120 = 0.100 (11.21), +0.300 (z~6.3), return +81.4. Stage 0 and stage 1 were both cleared at 0.900 by the same policy family. Ladder: 64 `0 -> 1` + 51 `1 -> 2` promotions, none to 3; trainer windows read 0.39 mixed. Control is not monotone in stage: 0.298 / 0.600 / 0.100 |

## The env goes on the GPU: entire game as Rust CUDA kernels [DECIDED, in progress]

**Decision (user, 2026-09-17): "lets do the entire game in cuda then... i think u can do
cuda kernels in rust now".** The target is Elliot Arledge's `mine.cu` pattern (batched
voxel RL env, pure CUDA, zero-copy GPU tensors, 49.6M env steps/s on an RTX 3090): the
game itself is the kernel, so there is no serialization boundary and no copies.

**Why the whole loop, not just the sim.** `perf record` on the live trainer (28601
samples, live PID) split the CPU as: `rollout_finish` busy-wait 11.5%, our own
`of_region_tile` 10.5%, engine sim (`flood_border_cluster` 9.5% + `conquer_one` 8.6% +
`OrderedUnitSet::remove` 2.3% + `rebuild_buffers` 2.2% + `featurize` 1.6%) ~24%, syscalls
6.9%, OpenMP spin 5.6%, memmove/memset 4.9%, clocksource 3.5%. Game math is only ~18-24%,
so a GPU port of the simulation alone caps out near 1.2x. Only moving sim + featurization
+ masks + obs buffers + reward onto the device with zero copies deletes the other ~80%.

**Toolchain (verified on this VM, not assumed):** RTX 5080, compute cap 12.0 (sm_120),
16 GB; CUDA 12.9.86 (`cuda12.9-cuda_nvcc`, libdevice present) already in the nix store;
rust 1.98.1 stable only (no nightly). Track chosen: **cuda-oxide** (NVlabs rustc codegen
backend, Rust kernels -> PTX, host/device share Rust types). Prerequisite: pinned nightly
2026-08-28 (rustc 1.100, LLVM 23) + `rustc-dev`, `llvm-tools`. Fallback if the alpha
toolchain blocks: `cudarc` + the installed nvcc 12.9, same kernels ported later.

**Gate before any game code:** a hello-world Rust kernel must actually run on this sm_120
card from a Rust host binary. No game code is written until that proves out.

**GATE PASSED 2026-09-17.** `cargo oxide run` on the template project printed
`PASSED: all 1024 elements correct`: a Rust `#[kernel]` fn compiled through cuda-oxide's
rustc backend to PTX, targeting **sm_120a** (detected via nvidia-smi), executed on the
RTX 5080. Four NixOS packaging walls were cleared, none of them fundamental to the
backend: (1) no `cc` on PATH for rustup -> `nix shell nixpkgs#gcc`; (2) `cuda.h` is not
in nix's split `cuda_cudart`/`cuda_nvcc` outputs -> the `cudaPackages_13_3` **merged**
root is the one layout that has it; (3) the host crates require CUDA 13.0+ (Tile 13.2+)
so the VM's 12.9 toolkit is too old -> 13.3; (4) `libcuda.so.1` is not on the default
loader path -> `/run/opengl-driver/lib`. All four live in `ofcuda_env.sh` (repo root of
the skg workspaces) so the setup is repeatable. Hello project: `ofcuda_hello/`.
Still open and optional: `libnvJitLink` is not in the 13.3 merged root (fetch
`cudaPackages_13_3.libnvjitlink` if a kernel needs libdevice math: exp/pow/sin).

### GPU env probe: the batched pattern is real on this card [VERIFIED]
`ofcuda_probe` (cuda-oxide, Rust kernels, sm_120a) steps 4096 independent envs of 4096
tiles for 256 ticks each, one thread per (env,tile), one store to the grid per tick:
**env-steps/s 1.406e8, cell-updates/s 5.757e11**, 0.030s for 4 launches.
VERIFIED: 0 of 8 sampled cells mismatch a CPU reimplementation of the same loop, and
16,776,960 of 16,777,216 cells are nonzero (256 zeros is the expected count for a 16-bit
hash over 16.7M samples). The first run's XOR checksum folded to 0x0 and I did not trust
it: a fold over 16.7M values is a weak check (uniform over 16 bits, so 0 is a 1/65536
event). It was replaced with a CPU-reference comparison.
CAVEAT, stated plainly: this is the SHAPE of an env, not its rules. It proves the
plumbing and the throughput ceiling and says nothing about game fidelity. Fidelity is
what the parity harness must establish against the Rust engine.
Note the rate is L2-bound, not DRAM-bound: 5.757e11 stores/s x 2 bytes exceeds the
card's DRAM bandwidth, because the same lines are rewritten 256 times per launch.

**Acceptance gate, unchanged:** the Rust engine remains the evaluator of record. The CUDA
sim is a *new simulator* until the parity harness matches it against the Rust engine on
identical seeds (obs bytes, terminals, tile shares). No win rate gets quoted off the fast
off the fast env alone.

## Defect ledger (found 2026-09-17, in impact order)

### D1 — the eval restarted the trainer, wiping the curriculum [FIXED]
`eval_policy.sh` did `systemctl stop` + `start`, which is a NEW process. The
curriculum stage and the 40-episode window live only in that process, so every
measurement dropped all envs to stage 0 with an empty window: each one cost
~1.7M steps of ladder warm-up and two win rates were never taken at the same
ladder position. Now SIGSTOP/SIGCONT: the CPU is freed, the process and its
ladder stay. Verified: same PID across freeze/thaw.

### D2 — no game timer in the RL config, so the only win is 80% of the map [BY DESIGN?]
`rl.rs` builds the wire config with no `maxTimerValue`, and the only other timer
is `HARD_TIME_LIMIT_SECONDS = 170 min` (102,000 ticks) versus a 21,000-tick cap.
So `check_winner_ffa`'s `time_limit_reached` branch is unreachable in training:
`game.winner` is set only when the leader owns >80% of the land. Consequence:
`V10_RAMP_WIN_AT = 0.90` means "take 80% of the map in 36 of 40 games", far harder
than the human game where the timer also decides. My earlier explanation
("tile leader at the time limit") was WRONG for this env and is retracted.
SUL decided 2026-09-17: **a timeout with no winner is a loss, hands down.** So the
timer branch stays unreachable on purpose and the 80% takeover stands as the
objective. Consequence to live with: the 0.90 gate means an 80% takeover in 36 of
40 games, and that rung is reached mostly by pure expansion because these bots
offer little resistance.

### D3 — `env.num_agents = 1`, so the duo/team win paths are dead code [RESOLVED: intended]
default.ini sets `num_agents = 1` and openfront.ini does not override it, so every
env is solo FFA. `duo_territory_win()` needs >=2 agents and the Team win check
needs `GameMode::Team`: both are unreachable in training. SUL decided 2026-09-17:
**solo FFA is the product** - the agent must take its 80% alone. So these paths stay
dead *on purpose*; do not "fix" this by enabling 2 agents. (Mark them clearly in the
engine so a future reader does not chase them.)

### D4 — the loss row in the dashboard is unreadable [OPEN, monitoring]
`block_losses[...] = loss * inv_NT` and the reduce averages over threads, so every
loss is displayed scaled down by ~1/NT: `value 0.000`, `policy -0.000` even at
kl 0.003. I cannot see whether the value head is training. A loop you cannot read
is a defect: print the unscaled means.

### D5 — ~2% of decisions are no-ops with the action's credit [OPEN, small]
The mask is refined per head (`mask[id]==1 <=> decodable`) but decodability depends
on the sampled COMBINATION, so ~21/1000 decisions fail `of_decode_action` and
become no-ops while the advantage is credited to the intended action. A joint mask
or a decode-consistency refinement would remove it.

### D6 — tuned hyperparameters are overridden by the unit [FIXED]
The ini holds swept values (`learning_rate = 0.000572786`, `ent_coef = 0.0984801`,
`vf_coef = 4.37465`) but `nixos/openfront.nix` launches with `LR=0.005` (8.7x) and
`ent_coef 0.0985` is ~98x PufferLib's default, which is why the policy stayed near
uniform. FIXED 2026-09-17: `LR=0.005` is deleted from `nixos/openfront.nix`, so the ini
is the single source of truth. Verified live, not just edited: `--train.learning_rate`
is absent from the running trainer's command line, so `0.000572786` applies (8.7x
lower) with the derived `schedule_epochs=600` anneal still decaying it.

### D7 — a loss by death paid a positive placement gift [FIXED]
`terminal_reward(place, won, timed_out)` returned `-W_WIN` only when the caller flagged
a timeout. Every other loss (the agent dies, or another player reaches the 80% win)
fell through to `W_PLACE * place^-1.5`: **up to +15 for dying in first place**, while
`W_DEATH` is only 1.0. So the cheapest way to lose (die early, about +14 net) beat the
path the rule was written to punish (survive to the cap, -30) by up to 45 points,
against a win of +45. The doc comment on that same function already said a death is
"a death/loss, not a placement" - the code applied the rule only to the clock path.
FIXED 2026-09-17: every loss pays `-W_WIN` on every path, and the test that pinned the
old behaviour now asserts the new rule (`every_loss_is_a_loss_not_a_placement_gift`,
passing; the engine cdylib rebuilt, and both `puffer` and `puffer-eval` verified to
resolve it dynamically). Impact is bounded and honestly measured: it touches only
episodes that end without a timeout, turning a gift of +5.3..+15 into -30. The
zero-policy stage-0 score moved from 92.67 (n=100) to 84.52 (n=40), i.e. -8 +/- 23,
consistent with roughly a quarter of episodes taking that path but not resolved at n=40.

## The bar: human baseline, resolved 2026-09-17

SUL's goal is to match published human win rates for the same stage. Researched, with
sources in `/opt/data/workspaces/skg/openfront_human_baseline.json`:

- **There is no published human win rate for a 2-Easy-bot FFA**, i.e. exactly stage 0.
  No official API, wiki, patch note or dataset exposes a win rate conditioned on bot
  count or bot difficulty, so the mapping has to be an explicit equivalence, not a
  citation. Say so in any writeup.
- The closest published setting is humans-vs-AI: OFstats ranks those separately
  because "almost everyone wins those" (no figure given). So the human-equivalent bar
  for beating 2 Easy bots is near-universal victory.
- Public FFA, for scale: typical lobby 29-31 players, so the *median* human wins about
  **3.2%** (1 in ~29); the top 10 humans win **53-71%** (e.g. 71% = 1223/1727 in
  49-player lobbies). These are not comparable to a 3-way stage-0 game.
- These bots are the weakest published tier: Easy = 50% max population, 90% growth,
  10,000 starting troops vs 25,000 for humans, 5% troop commitment vs 20%.

**Stage 0 bar: >= 90% win rate**, reported with a lower 95% confidence bound and a
per-map breakdown. This coincides with the existing `V10_RAMP_WIN_AT = 0.90`, so the
gate is the right bar and stays. Secondary narrative milestone: >= 50%, "beats most
humans in public FFA".

Caveats that make our task harder than the human one: the 0.8x attacker attrition
bonus applies to *humans* attacking bots and does not exist for our agent, and Easy is
not a median human. So a 90% figure here is not the same achievement a human would
claim, and should be stated as "matches the human-vs-bot tier", not "beats humans".

Standing against that bar: **0.400 wins at stage 0** (n=160, 2026-09-17 22:00, vs
0.298 for the all-zero control), versus 0.90. The control losing 70% of its
episodes is what makes that 0.400 a signal rather than a free tile race.

### D8 — a stale /run drop-in pinned the anneal to 12% of the run [FIXED]
`/run/systemd/system/openfront-train.service.d/override.conf`, left over from the E2
experiment, set `Environment=HORIZON=64 SCHED_EPOCHS=600` and silently won over the
declared config. One run epoch is `horizon*agents` = 4096 steps, so 600 epochs is 2.46M
steps of a 20M-step budget: the LR/ent_coef cosine reached its floor (lr 5e-5, ent
0.0086) at **12% of the run**, and the remaining 88%, roughly 17 hours, trained at a
near-frozen LR. That is sufficient on its own to produce a plateau, and it is
consistent with `kl 0.000` plus the unreadably small losses.
FIXED 2026-09-17: both /run drop-ins deleted (they never survived a reboot anyway), so
`nixos/openfront.nix` is the single source of truth again; `StartLimitBurst = 30` was
re-verified as still coming from the nix. Verified live: `--train.schedule_epochs` is
absent from the command line and the resume line reads `epoch 600/4882, 12.3% into the
cosine`, i.e. the LR is lifted back near the swept base and anneals across the whole
remaining budget.
LESSON: a drop-in in `/run/systemd/system/<unit>.d/` or `/etc/systemd/system/...d/`
overrides the declared config and is invisible in both the nix file and the ini. Before
trusting any hyperparameter, run
`systemctl show <unit> -p Environment -p DropInPaths`.

### D9 — the eval's freeze/thaw was a silent no-op under a stripped PATH [FIXED]
`eval_policy.sh` called a bare `sudo`. NixOS's setuid sudo is `/run/wrappers/bin/sudo`;
the `sudo` that the system profile PATH resolves is the raw store binary and is NOT setuid
(`-r-xr-xr-x`), so it fails with "must be owned by uid 0 and have the setuid bit set". Every
call in the harness ended in `|| true`, so on 2026-09-18 05:46 the freeze did nothing (eval and
trainer split the 8 cores, each at ~half rate) and, worse, the **thaw** did nothing either: the
eval exited normally at 05:52:47 and left the trainer SIGSTOPped until it was noticed by hand 13
minutes later at 05:59. Not an environment fault - the operator shell had exported
`PATH=/run/current-system/sw/bin:/usr/bin:/bin` (the unit-PATH warning applied one layer too
high); the cron job's own inherited PATH starts with `/run/wrappers/bin` and is fine.
FIXED in `eval_policy.sh`: `SUDO` resolves `/run/wrappers/bin/sudo` explicitly (overridable),
and `freeze`/`thaw` now read `/proc/<MainPID>/stat` and print a loud warning if the trainer is
not in state `T` after the freeze, or still in `T` after the thaw (with a direct `kill -CONT`
fallback). LESSON: a `|| true` around a privilege call converts "no permission" into "worked",
and the failure mode of a broken thaw is a silently stopped trainer.

## The env goes on the GPU: what "parity" has to mean

DECIDED 2026-09-17 (user-directed): reimplement the game as Rust CUDA kernels
(`cuda-oxide`), because the profile says only ~18% of the loop is game sim and the
other ~80% is C helpers, buffer rebuilds, copies, syscall/clock traffic and a
busy-wait. Only a full on-device loop deletes all of it. The gate passed and the
throughput probe verified (see the `cuda-oxide-nixos` skill); the port itself starts
from the three extracted specs in `experiments/cuda-port/`.

The specs changed the scope in three ways worth writing down before any kernel exists:

1. **The obs is not 99 engine planes.** 32 of the policy's 99 input channels are AE
   latents produced by `oftrain/src/ae.rs`; the engine's own featurizer emits 67. And
   the C ABI does not hand over a grid at all - it hands over 4270 f32 per agent
   (12 scalars + 128 player tokens + 32 unit tokens + masks). The (99, gh, gw) tensor
   is assembled policy-side in `batch.rs`. So "env on the GPU" means the GPU must
   produce the tensor, which is a different interface from the C ABI.
2. **The tick is order-sensitive in ways a GPU kernel has to respect.** Six separate
   sfc32 streams with exact draw order (`Game.rng` is constructed and never drawn;
   `Player.id_prng`; `SpawnExecution.random`; `TribeSpawner.random`;
   `TribeExecution.random`; and `AttackExecution.random = PseudoRandom::new(123)`, a
   constant seed shared by every attack). Executions are initialised the tick they are
   queued and first tick the next; the steady-state dispatch order is
   `[WinCheck, RecomputeRailCluster, Player(bot1), Tribe(bot1), ...]`. The attack
   frontier is a min-heap on an f32 priority that adds `tick` and one `rand(0,7)` per
   candidate neighbour. Reproducing this bit-exactly also means controlling float
   contraction (FMA) in the generated PTX.
3. **Stages 0-7 are the tractable target.** Bridge maps, Easy *tribe* bots, zero
   nations - and the difficulty string barely touches a tribe (only the attack relation
   delta `-60`), so there are almost no difficulty branches to port. Nations, nukes,
   warships, rail clusters, the water manager and alliances are the expensive part and
   are absent or inert at these stages.

### The parity ladder (acceptance bar)

Bit-exact equality on the whole stochastic loop is a multi-week reimplementation with
real float-order risk. The bar is therefore a ladder: exact where exactness is cheap,
statistical where it is not. No number produced by the CUDA env is a result until the
level it depends on has passed.

**CORRECTION 2026-09-18 (user): the reference of record is the TypeScript core, not the
native Rust engine.** The native port never achieved full parity with TS, so gating the
CUDA port against native would bake native's divergences in. Every level below is
re-pointed at TS. L0-L2 as verified against native are necessary but not sufficient, and
they transfer to TS only where native == TS.

The repo already documents exactly where that is. `docs/PARITY_PLAYBOOK.md` is the single
source of truth for TS parity and says a record passes hash parity when native and tip TS
agree every tick on alive/tiles exactly, the units hash and unit count, and the player and
game hash **bit patterns** (IEEE-754, because the JSON ints pass 2^53). Its snapshot at
`dd1277e245b5` records coarse `HASH_PARITY_EVERY=25` at **40/41 games**, with one OPEN
fine-grained divergence: game `YdhKd1j6` tick 2849, layer units, Warship `#15554`
equal-cost water route fork, suspected HPA cache / `refineEndpoints` bounded-A* tie order
after cache warming. Several earlier ones are recorded as fixed.

That matters because the open divergence is in the **naval pathfinder**, and naval units
are inert at curriculum stages 0-7, which is exactly the regime the GPU env targets. So
the working hypothesis was that the early-curriculum regime is clean.

**HYPOTHESIS FALSIFIED 2026-09-18.** It is not clean. A harness was built
(`scripts/run_early_curriculum_parity.sh`) and run over 10 early-curriculum records
(2/4/7 Easy bots, zero nations, bridge maps: Pangaea, Caucasus, Onion, BlackSea, Europe,
GreatLakes, BetweenTwoSeas) at every-tick granularity, max 4500 ticks, 4 jobs. **All 10
games diverge, and every one diverges on the `tiles` layer, at tick 307 to 332:**

| record | first diverge tick | layer |
|---|---|---|
| curr-b002-s1-pangaea | 321 | tiles |
| curr-b002-s4-pangaea | 307 | tiles |
| curr-b002-s7-caucasus | 332 | tiles |
| curr-b004-s2-pangaea | 310 | tiles |
| curr-b004-s5-onion | 332 | tiles |
| curr-b004-s8-blacksea | 311 | tiles |
| curr-b004-s10-greatlakes | 307 | tiles |
| curr-b007-s3-pangaea | 316 | tiles |
| curr-b007-s6-europe | 307 | tiles |
| curr-b007-s9-betweentwoseas | 316 | tiles |

Consequences, in order of severity, with what is established separated from what is not:

1. **ESTABLISHED (I re-ran it myself): native and TS disagree at tick 321 in a
   zero-input early-curriculum game, on tile ownership, by exactly one tile.** The
   harness output is unambiguous: `gameHash native=8159972285026 ts=8160148440203`,
   and `nation:Frankish Duchy(id=5iznss4u): tiles native=77 ts=78`. One player holds one
   tile in native that TS gives away, or vice versa. This is core territory bookkeeping,
   not a naval or float-format artifact.
2. **NOT ESTABLISHED, and I overstated it in an earlier revision of this file: that the
   RL env's own trajectory diverges.** These records are `players: []`, `turns: 0`,
   `bots: 2`, Easy, `nations: disabled`, `randomSpawn: false`, `num_turns: 4500`. They are
   pure bot-vs-bot games with no player input at all, so they exercise the bot AI and the
   core sim, not a policy-driven episode. Two open questions decide severity: (a) is the
   divergence caused by the **bot AI** choosing a different action, or by the **core sim**
   evolving differently given the same actions? (b) does it reproduce on an intent-driven
   trajectory like the RL env's? Until (a) and (b) are answered, "the agent has been
   training on the wrong game" is a hypothesis, not a finding. What is certain is that
   native is not a sound referee for the CUDA port.
3. The uniformity across maps and bot counts (307-332, always `tiles`, never units)
   suggests a timing/scheduling mechanic rather than a map-specific rule. Now that the
   divergence is known to be a single tile for a single player, the cheapest next probe is
   to find *which* tile flips and at what tick in each implementation, then read the TS
   rule governing that tile against the native rule. `scripts/hash_bisect.sh <record>`
   does the resume search; the raw streams from my run are at
   `/tmp/hash_parity.curr-b002-s1-pangaea.{native,ts}.ndjson`.

The repo's own playbook was near-miss on this: it tracks a coarse 40/41 with the naval
divergence open, consistent with an every-25 sampler stepping over a systematic tick-321
tiles divergence, or with the tip having moved since its `dd1277e245b5` snapshot.

**ROOT-CAUSED TO THE MECHANISM 2026-09-18** (3,159 every-tick pairs over 12 records; 10 in
the target regime, 2 nation controls; a native re-run of `curr-b004-s2-pangaea` was
byte-identical, so this is deterministic and not hash-order noise). Three distinct
divergences, descending severity:

**(A) `seedToGameID` disagrees, and it is upstream of everything.** TS `bridge/session.ts`
computes `h*1103515245+12345` in IEEE-754 double, which passes 2^53 and **rounds**, giving
game id `QqkIuyke` for seed `parity`. Native `rust/engine/src/session.rs` does the same
arithmetic in **wrapping u32**, giving `pyr6b8nU`. The game id seeds the PseudoRandom that
assigns player ids and bot spawn tiles, so for one seed string the two engines do not start
the same game at all. Consequence: **the native env has never played the game the real
server would play for a given seed, and no native trace can match a TS trace past the
spawn.** The TS core confirmed this from the other side: pre-spawn terrain
`0xebffa87c2568cc58`, pre-spawn state `0x6334dfb980453d25`, land 420335, spawn tile 8555 are
**byte-identical in TS and native**. The map and the reset state agree exactly; the first
disagreement is the id derivation. *(This does not touch the verified L1 result: the PRNG
itself is bit-exact given the same seed. It is the seed that differs.)*

**(B) The attack/expansion frontier captures a different tile subset per tick.** In
`curr-b004-s2-pangaea` both engines have the same active attack (smallId 1 onto neutral,
targetSmallId 0) and the captured border-tile sets are **disjoint**: TS-only
`(581,195) (576,203) (578,204) (580,204) (581,204)`, native-only
`(584,198) (584,199) (575,201) (583,203)`. The two sets are not adjacent to each other, so
this is not a tie-break on one contested tile but a different per-tick subset of the
frontier. The live troop ledger is already off by one at tick 308 (6447 vs 6446). This is
the divergence that fires at the first post-spawn expansion in all 10 target-regime records.

**ROOT CAUSE OF (B), FOUND 2026-09-18: a neighbour visit order inversion, and it is a
one-line bug.** Native `for_each_neighbor4` (`rust/engine/src/map.rs:319-330`) visits
**W,E,N,S**. The real TS tip visits **N,S,W,E** in both `neighbors4`
(`openfront/src/core/game/GameMap.ts:393-403`) and `forEachNeighbor` (`GameMap.ts:383-391`).
Native's comments at `map.rs:312-318` and `map.rs:501-502` assert TS is W,E,N,S, and that
inverted assumption is how the bug survived. Native already contains the correct order,
parked in `for_each_neighbor_nswe` (`map.rs:360`), **unused by `AttackExecution`**.

`AttackExecution.addNeighbors` draws exactly one PRNG value per enqueued neighbour in visit
order (`priority = (random.nextInt(0,7)+10) * (1 - numOwnedByMe*0.5 + mag/2) + tick`), with
the same PRNG, the same seed 123, the same draw count, and a heap that is algorithmically
identical on both sides. So the only difference is **which tile each jitter draw binds to**.

Verified numerically, not by inspection alone: at tick 319 the native claim order
`[460260,459254,453260,459261,453255,454261,454254,460255]` is monotonically non-decreasing
in priority **only** under W,E,N,S, and the TS order
`[459261,454261,460255,454254,459254,453260,453255,460260]` **only** under N,S,W,E; swapping
the pairings makes both fail. The same inversion reproduces on `curr-b004-s2-pangaea` at its
tick 309. Consequence chain: different pop order draws tiles of different terrain cost, the
per-tick conquest budget exhausts on a different tile, and the tile **count** splits (tick
321: 8 claims native vs 9 TS, 77 vs 78 tiles; attack troops 5015 vs 5005).

**Severity classification: CORE SIM, not bot AI.** The attack is identical in both engines
(owner smallId 2, target TerraNullius smallId 0, created tick 318, troops 5263/5183/5095
through tick 320); the bots chose the same action and the state diverged inside the frontier
ordering. So the native env's conquest order is wrong, not merely its opponents' choices.

**Secondary latent divergence, same area, NOT the cause here:** TS skips impassable
neighbours when enqueuing (`AttackExecution.ts:346-351`) and impassable tiles at pop time
(`AttackExecution.ts:299-304`); native filters water plus wrong owner only
(`attack.rs:1345-1352`) and guards land-only at pop (`attack.rs:311-314`). Verified as not
the cause because there are zero impassable tiles in the 24-tile frontier at ticks 318-319.
Filed as its own defect.

**(C) Nation spawn land-paint is off by one tick.** Control `curr-b005-s1-pangaea` (3
nations): a nation spawn paints 47 tiles in native against 52 in TS at tick 15, with native
tick 15 equal to TS tick 14. Same signature on `curr-b000-s0-onion` at tick 309, which is a
control with 1 nation.

Zero units exist for the entire measured window (`numUnits=0`, no UnitSnapshot before
divergence), so **none of this is the naval pathfinder** the playbook has open, and no
nation-vs-nation logic is involved in the 10 target records. It is land-ownership transfer
during bot expansion.

**Caveat that is inferred, not measured:** these records must use `gameType Public` (spawn
phase 300 turns) because with zero humans the Singleplayer spawn phase never ends, while
the trainer runs Singleplayer (spawn phase 100). The divergence fires at the first
post-spawn expansion in both, so the transfer is expected, but it is not measured.

**Consequence for the port:** porting native's expansion logic into CUDA would port (B), and
porting native's id derivation would port (A). The CUDA tick must be written from the TS
source with the TS oracle as referee, and (A) and (B) must be settled at code level first so
the port has one unambiguous specification.

**USER DIRECTIVE 2026-09-18: "we can kill the current training runs once we get parity and go
from there."** So the sequence is now explicit and authorised: reach TS parity, then kill
`openfront-train`/`openfront-sampler`, then train on the faithful (and, once ported, GPU)
env. No need to preserve the current cosine, and no need to keep the unfaithful run alive for
continuity. The restart is no longer a cost to weigh; it is the plan.

**DEPLOY INCIDENT 2026-09-18, mine, caught and corrected: the first redeploy silently reverted
D7.** The worker was told not to change the reward, and reasonably read that as "exclude the
uncommitted `curriculum.rs` reward change from the build". That change *was* D7 (every loss
pays `-W_WIN`, closing the die-in-first-place-beats-playing-on loophole). It is now restored
byte-identically (`sha256 473fe9f1…`, matching the worker's preserved copy) and **committed as
`522c93d`**, so a future build cannot drop it again. Lesson worth keeping: when a tree is
rebuilt for deployment, every uncommitted file is a landmine and must be triaged explicitly as
either required-in-the-build (then commit it) or deliberately excluded (then stash it).
Honesty note: the preserved old-run log carries no reward columns, so the revert could not be
corroborated from the log; the evidence is the defect ledger plus the fact that the committed
version contradicts its own docstring *and* its own tests.

**REDEPLOYED AND VERIFIED 2026-09-18 15:09.** I confirmed the chain by hand rather than from
the worker's report: on-disk `libopenfront_engine.so` inode `9608558`, sha256 `43c16d48…`, mtime
15:08:15; `openfront-train` MainPID `1907344` started 15:09:46 and maps inode `9608558`, i.e. the
running process executes the new build. HEAD is `522c93d` (D7 reward) over `b14a77d` (N,S,W,E
neighbour order) over `4c0ee39` (IEEE-754 game id), with `git diff HEAD -- curriculum.rs` empty.
The scoped reward test ran green rather than being skipped: `cargo test --release -p ofcore
every_loss` gives `every_loss_is_a_loss_not_a_placement_gift ... ok` (1 passed, 60 filtered).
**The deployed trainer now trains on a world that matches TS and pays every loss `-W_WIN`.**
Resumed from checkpoint `10650112` at epoch 1300/2441 (53.3% into the cosine); old log preserved at
`openfront-train/current.log.pre-d7-restore`. Post-restart SPS 191-327 and Env 91-93% (up from
94-95%, the env being cheaper at a reset stage), Train now 6%.

Caveats worth keeping: the displayed `win_rate 0.000` and the high SPS are both consequences of the
curriculum stage resetting to 0 on resume, not evidence of regression or of a faster env; entropy
is 7.995, so it is exploring. And `rebuild_engine.sh` carries a stale comment claiming the reward
change is "not in the tree" — cosmetic, now false, not edited.

**UNIT-TEST BASELINE ESTABLISHED 2026-09-18: the N,S,W,E fix causes ZERO regressions.** A worker
reported 48 failures from `cargo test --release -p openfront-engine -- --skip
doomsday_clock_execution`, which is meaningless without a baseline, so I built one in a clean
worktree at `4c0ee39` (the commit before the fix) with its own `CARGO_TARGET_DIR` so the live
`.so` was untouched:

| revision | passed | failed |
|---|---|---|
| `4c0ee39` (before the fix) | 436 | 51 |
| HEAD (with the fix) | 439 | **48** |

Set difference: **no new failures**, and 3 failures resolved (`spawn_util::spawn_set_matches_
python_reference_for_1830957`, `spawn_util::spawn_set_tick50_centers`,
`rl::duo_session_tests::duo_reset_is_team_mode_with_alliances_on`). The 48 remaining are
pre-existing and environmental: they panic with `No such file or directory (os error 2)` at
`replay.rs:638`, i.e. they open a recorded fixture absent from this checkout. Two environment
facts to carry: a workspace-wide `cargo test` dies on `torch-sys v0.24.0`, and
`--skip doomsday_clock_execution` is required for the engine package.

**RESUME DEFECT LOCATED CORRECTLY, FIXED, NOT YET DEPLOYED 2026-09-18.** Two of my assumptions
were wrong and a worker corrected both with evidence:

- **I pointed at the wrong file.** I said `rust/oftrain/src/train.rs`; the running trainer is
  **PufferLib5's `./puffer`**, a different program entirely. `oftrain` already persists `stage`,
  `lr_now`, `update`, `recent_wins`, `recent_deaths` and `stage_env_targets` into
  `<ckpt>.state.json` (`train.rs:417-460`, save `1535`, restore `7318+`). Fixing oftrain would
  have changed nothing and left the measured defect in place.
- **`lr_now` was never lost.** `puf_resume_schedule` (`src/pufferl.cu:1834`, called at `3227`)
  reconstructs the cosine position from the checkpoint filename's step count.

The real gap is in the live path (`PufferLib5/src/pufferl.cu` + `ocean/openfront/openfront.h`):
not persisted were the stage index (`openfront.h:233`), the gate's rolling window
`win_hist`/`death_hist`/`hist_n`/`hist_idx` (`openfront.h:239-242`), `stage_wins`/
`stage_advances` (`235-236`) and `stage_win_at` (`234`). So every `LOAD=auto` rebuilt each env at
cfg `stage = 0`, which is the observed reset.

**Fix:** a per-env sidecar `<ckpt>.envNN.state.json` written in the same atomic tmp+rename save as
the weights, with four guards: **identity** (the sidecar names the checkpoint it belongs to and a
mismatch is refused, so a copied or renamed `.bin` cannot smuggle in a stage its weights never
earned); **bounds** (a stage outside `0..stage_count-1`, or a window that is not exactly
`OPENFRONT_WINDOW`, is refused); **missing sidecar is not an error** (keeps the cfg stage and logs
"curriculum progress NOT restored", i.e. today's behaviour); and **`OPENFRONT_HOLD_STAGE`
suppresses the restore**, the established pinned-eval convention, so an eval measures the stage it
was told to. 26/26 checks pass in a purpose-built C harness; a scoped build produced
`./puffer-curriculum-state` with the live `./puffer` untouched.

**Still not proven, stated plainly:** it is **not deployed** (it needs `./puffer` rebuilt) and it
is **not runtime-tested on GPU**, because a second trainer cannot run while the live one owns the
card. Its first real exercise will be a restart. The reset was also observed at the earlier restart
(line 52760 of the preserved log). Cost until it is deployed: every restart re-teaches the early
stages. My original note in this spot pointed at `rust/oftrain/src/train.rs:599` and blamed a
missing `lr_now`; both claims were wrong, as corrected above.

**SECOND DEFECT (open): the run's own progress metric was quoted wrong.** The worker measured
the final window of the old run at `win_rate ~0.148` with `stage ~1.8-2.0`, not the ~0.375 and
~1.4 carried in earlier notes. Trust the measured window over the carried figure.

**FOUR PORT SLICES COMPLETE AND PROVEN 2026-09-18 (18:15).** Recorded after a crash of the
delegation harness killed four workers mid-flight at ~16:28; the re-dispatched workers inventoried
the dead workers' crates and finished them rather than restarting.

**1. The expansion model is now the engine's model, and it took the tick window from 20/32 to 32/32
(`ofcuda_tick`).** The decisive control: with in-tick `add_neighbors` disabled the result falls back
to 20/32 with first mismatch 320, and with the budget draw disabled it also falls back to 20/32. So
the delta is attributable to the fix rather than to incidental edits. Ticks **300..331 all reproduce
the engine's per-tick FNV-1a-64 state hash**, 133/133 claims in engine order, and the contested case
now matches the engine's complete 27-claim order **including 838544 at index 23**. Rules encoded with
`attack.rs` citations: heap persists (`:17,:21`), init is refresh-only (`:160-164,:1265-1272`),
refill-only-when-empty ends the tick (`:264-268`), in-pop `add_neighbors` before the conquer (`:292`),
terra-nullius + attacker-neighbour skip guards (`:284`), a skipped pop costs nothing (`:284-288`), one
extra draw per tick (`:239-254`), per-claim cost (`:313`).

**2. Cluster capture is bit-exact on every case (`ofcuda_cluster`).** GPU == CPU == engine on a clean
269-tile loss at tick 1614 and a contrast case at tick 1910 (1061 tiles), 14/14 checks each, plus two
further cases at 10/10. The dead worker's crate **did not compile** (five in-flight defects) and had
never run. The engine's hash-map trap is resolved rather than ambiguous: `border_tiles` is an
insertion-ordered `OrderedTiles` and `get_capturing_player` uses a first-seen `Vec`.

**3. The economy slice is complete and its per-tile half is a MEASURED NEGATIVE (`ofcuda_econ`).**
Per-player `troops`/`gold` are exact device-vs-host, **7791/7791**. Against the engine, 591/599 and
7102/7192 rows agree and **every one of the 98 differences is `attackActivity > 0`** - attack escrow,
i.e. conquest-side and out of this slice, not an economy bug. Float divergence is confined to f64
diagnostics at 1 ULP of libdevice `__nv_pow`; after the f32 narrowing the obs planes actually use,
7791/7791 agree. **The negative:** `grep population openfront/src/core/` is empty and
`PlayerExecution::tick` mutates only per-player `troops`/`gold` outside unit/alliance/embargo/cluster
bookkeeping, so **there is no per-tile economy to port** - the tile `u16` is written only by
conquest/nuke paths. That shrinks the port. Also resolved: `tick_player_income` (game.rs:1269) and
`try_merge_land_attack` (game.rs:1991) each occur **once, the definition with no caller**; the live
path is `PlayerExecution::tick` (execution/player.rs:82-88) via `troop_increase_rate_raw_for` +
`wire.gold_addition_rate` + `add_troops`, which is also what the obs layer calls.

**4. The curriculum-state RESTORE works on the live run (`ofcuda`-adjacent, PufferLib5).** Verbatim:
`pufferl: curriculum state: 64/64 envs restored from .../0000000011469312.bin sidecars, 0 refused`,
with the negative control in the same log from the earlier sidecar-less checkpoint reading `0/64
restored`. Stage read 1.0 before and 0.0 after, which is **correct, not a silent reset**: that
checkpoint recorded stage 0 because the run only climbed to 1 afterwards. The discriminator that it
came from the file rather than the cfg default is `stage_wins` 8.0 -> ~33 and a `window=32/40` - a
fresh run cannot have either. All 64 sidecars were cross-checked programmatically against their
restore lines with 0 mismatches. Behavioural proof: envs re-advanced `stage 0 -> 1` after ~5 minutes
against ~94 minutes for the first climb on a fresh window.

**THE COMPOSED TICK MATCHES 59 CONSECUTIVE TICKS WITH A COMPUTED BUDGET 2026-09-18 (`ofcuda_env`).**
The four slices now run as one tick in `/opt/data/workspaces/skg/ofcuda_env` with **no changes to the
other four crates** (all four already exposed library entry points). Composition order, cited per call in
`src/lib.rs`: economy -> budget -> persistent-heap expansion -> cluster capture -> plane -> hash.

**The claim budget is COMPUTED, not observed.** `2 * (tracked_border_size + next_int(0,5))` with the
single draw taken before the pop loop (`attack.rs:253`), and it reproduces the engine's claim counts
exactly across ticks **318..376, 59 consecutive ticks**, including the 12-claim ticks and tick 320 = 9.
`ofcuda_tick` deliberately does not carry the border set (its README says the budget is exogenous), so
this crate tracks it: `add_border_tile` on every enqueued candidate (`attack.rs:1363`),
`remove_border_tile` on every pop including skipped ones (`attack.rs:275`). That was the whole gap and no
other crate changed.

**Two named stops, both unported mechanics rather than fudge factors:**
- tick 359 with a truncated `tiles_used` input: six claims byte-identical, engine takes a 7th.
  `sum(tiles_used) = 174.046526` against a budget of `2*(86+1) = 174`, i.e. **0.047 short**. Cause:
  `tick_dump.rs:326` writes `AttackSnapshot.troops = troops as i64`, truncating the true `1363.4`, and
  `tiles_used` divides by it. The missing input is the attack's exact `f64` start troops =
  `attacker.troops - max_troops_for(owner) * expand_ratio` (`ai_attack.rs:9-18`, `tribe.rs:42`), which is
  economy output rather than the record's integer.
- tick 377 with the float input: engine claims 10, the crate 3 - `sid2` has **two concurrent land attacks**
  that merge the following tick. **Multi-attack + merge is unported**, and that is the proven boundary.

**Also found and fixed:** the crate was carrying init priorities at `tick` instead of `tick-1`. An attack
created in tick T first appears in rec[T+1], so its `refresh_to_conquer` runs with `game.ticks() == T-1`.
A uniform one-tick shift that reordered init-enqueued against tick-enqueued candidates. Found by
heap-diffing against a Python mirror rather than by guessing.

**Not claimed, and this matters:** ticks 300..1200 are **not** claimed - the model stops at 377 without
the merge mechanic. The end-to-end FNV hash is also **not** claimed: the engine's `gameHash` is not the
plain owner plane (0/0 on that diagnostic), so the composed hash stage currently rests on `ofcuda_hash`'s
separate 200/200 result rather than on this crate's own plane reconstruction. Cluster capture is wired but
fires only on a player loss, of which there are none in this window.

**THE COMPOSED TICK NOW MATCHES THE WHOLE RECORD 2026-09-18 (`ofcuda_env`).** Ticks **2..1301, no
mismatch: 2598/2598 (tick,player) pairs** reproduce the engine's claim set AND order, 1966/1966 from the
first attack at 318. Three stops were found and closed, and the third only became visible once the first
two were gone:
- **tick 359** - the truncated `tiles_used` input, closed by computing the attack's real f64 start troops
  from the economy instead of the record's `i64`.
- **tick 377** - two concurrent attacks, closed by porting the LIVE merge path:
  `AttackExecution::init` (attack.rs:177-184, land attacks only) -> **`merge_outgoing_land_attacks`
  (game.rs:2122-2156)**, which walks `Player.outgoing_land_attacks` (game.rs:125, pushed :2020, retained
  :2040), adds same-owner + same-target troops, and `kill_attack`s the absorbed exec (attack.rs:215-218).
  `try_merge_land_attack` (game.rs:1991) is confirmed dead: one occurrence in the tree, the definition,
  no caller. Arithmetic: rec377 `start 1353.332073 + absorbed 557.462201 -> 1910.794274`, floor 1910 =
  the record. So it is a real merge, not two racing attacks.
- **tick 557->558** - **the heap was a fixed 256-entry array that DROPS a candidate when full, while the
  engine's `FlatBinaryHeap` is a growable `Vec`** (flat_heap.rs:8,26-27; `Vec::with_capacity(1024)` is only
  a hint). Our heap peaked at exactly 256 and **13180 candidates were dropped**, which is why divergence
  appeared only at the first tick whose heap saturated. Raised to 1024 with `STATE_HEAP_CAP` held at 256 so
  the device layout and `STATE_WORDS`=518 stay unchanged. After the fix: peak 851, **0 drops**.

Per-tick claim counts match everywhere, e.g. 318..333 = `8 9 9 9 9 10 10 12 11 10 12 12 12 12 12 13`.
Accounted: 36 creations (24/24 floor-matched), 12 merges (12/12 floor-matched), 24 starved deaths, 0
heap-dry retreats, 12 lingering no-op ticks, 0 unexplained evictions.

**The hash question is resolved, and my earlier framing of it was wrong in a way worth recording:**
`gameHash` is NOT a plane hash. It is the engine's JS sync checksum `1.0 + sum_p id_hash*(troops+tiles) +
sum_units unit_hash` (hash.rs:7-22, `id_hash=|simple_hash(id)|`). Re-evaluated from the record's own fields
it reproduces **1001/1001**; FNV over the owner plane versus `gameHash` is **0/1001** because they are
different objects, not because FNV is wrong. What the composed tick does reproduce is the plane itself:
composed plane == engine plane **1001/1001 word-for-word**, hence FNV-1a-64(composed) == FNV-1a-64(engine)
**1001/1001**. A `gameHash` from composed state is still not reproduced, since its inputs are player-level
troops/tiles and composed player troops remain record-seeded.

**Assumed:** `ownedOrder` is the claim log, `borderOrder` is the engine's iteration order, one PRNG per
attack at `SEED=123`, and **the record still supplies which attacks exist and when** - so the model aligns
to the record's attack list and a window must begin at or before the first attack (`--t0 548` on the same
dump fails immediately for exactly that reason). **Unported:** player-vs-player budgets, mid-tick
`refresh_to_conquer`, fallout/defense modifiers, boat attacks, and cluster capture, which needs a player
loss and there is none in this window.

**MEASURED SPEED, AND THE HONEST ANSWER 2026-09-18 (`ofcuda_env`).**
The composed tick that reproduces the whole record **launches no kernel**. Grepping `src/main.rs` and
`src/lib.rs` for any device symbol returns nothing: the heap, the budget, the plane update and the hash all
run on the **host**. So 2598/2598 is a HOST result and `k` for the tick is **unmeasurable** - inventing it is
exactly how the retracted ~700x happened.
- **Host composed tick: 142 ticks/s (7.03 ms/tick)** on the proven window, 130 on the later/heavier one, 183
  early, i.e. **9.5 decisions/s** at 15 ticks/decision. The live CPU env on this box in the same minutes runs
  ~115-134 decisions/s (~1.7k-2.0k engine ticks/s across 64 envs), so the composed host tick is currently
  **~12-14x SLOWER per decision** than the thing it is meant to replace. It is a verification harness, not an
  environment.
- **The one stage with both implementations measured on identical input:** `ofcuda_hash` 64.5 ms/tick against
  its CPU reference at 8.3 ms/tick, i.e. **k_hash = 0.13 - the device is 7.8x SLOWER**, and that is harness
  wall time dominated by a 2 MB readback every tick. Amdahl at k=0.13 gives 0.136x, but that is arithmetic on
  one stage and must never be quoted as a whole-env figure. The cap is still 20x as k->inf (Env 94-95%).
- **Batch scaling: not supported.** Per-case 64-256-thread launches only, batched variant unimplemented. No
  scaling point was fabricated.
- **Where the host tick actually goes: ~87% is plane hashing** (FNV over the 2 MB plane = 1.98 ms). The engine
  expansion that the CUDA kernel implements is **0.038 ms/tick = 0.5% of the tick**. The ported mechanic is
  half a percent of the harness, and the verification scaffolding dominates it.
- **Fixed per-iteration cost**, from the live log: Train 2.633 s + Model 2.631 s.
- **Contention at measurement time:** GPU 0-7% util, 51.5 W, 2429 MiB held by the trainer; CPU 7.0-8.1 of 8
  saturated, so the host rates above are lower bounds.

**What this means for the plan:** correctness is proven per slice and end-to-end on the deterministic planes;
**speed is not realized at all yet**. Three things are needed and none of them is the port itself: (1) the tick
must run device-resident with **no per-tick readback**, which is precisely what makes the one measured device
stage 7.8x slower; (2) **hashing must leave the hot loop**, being verification rather than game and currently
87% of the tick; (3) **batched launches across many envs**, the only lever that moves the 20x ceiling, made
mandatory by the fixed 5.26 s/iteration.

**Still unproven after all four:** the `tiles_used` divisor's true `f64` start troops are now wired (see
the composed-tick block above), so what remains is the following: the nonzero-stage `ofenv_set_stage` branch at startup never fired; all
five refusal guards are untested live (0 refused); **optimizer state is not serialized at all**
(`puf_save_weights` writes only the fp32 master weights, 101,933,056 B = 25.5M x 4, so Adam moments
restart every time); the env RNG and in-episode state are not restored; and no save/restore
round-trip of a *new* checkpoint has been observed yet.

**THE VERIFIED FRONTIER SLICE WAS INCOMPLETE, AND PART 1 CAUGHT IT 2026-09-18.** The contested
case (`curr-b007-s3-pangaea`, state tick 581, player 2 Maori Council) closed the conflation
assumption and then exposed a real gap. Do not treat the earlier "frontier PASS" as covering this:

- **The conflation risk was real and is now excluded.** `owned_any` (14332) and `owned_mine`
  (1832) are genuinely distinct sets, not just different counts. Swapping eligibility to
  `owned_mine` moves the enqueue count 211 -> 220 and diverges at claim index 3. The port keeps
  them separate, so CPU == GPU on all 211 pops including priority bits.
- **But kernel != engine.** Claim order matches the engine 0..22 and first differs at **index 23**:
  the engine inserts tile 838544 (a SECOND-RING tile, never popped by a one-pass frontier) between
  839550 and 843564. CPU == GPU on every pop, so this is a **model-vs-engine** gap, not a
  CUDA-lowering bug.

From `rust/engine/src/execution/attack.rs:239-325` (read, not guessed), the engine's tick loop does
three things the one-pass port omits:

1. **The expansion heap PERSISTS across ticks.** `AttackExecution::to_conquer` is kept between
   ticks (`attack.rs:17`) and only refilled when empty (`attack.rs:264`, `refresh_to_conquer` +
   retreat, which also ends the tick). The port rebuilt the frontier from scratch every tick.
2. **`add_neighbors(tile_to_conquer, tick)` is called INSIDE the pop loop**, before the conquer
   (`attack.rs:292`). A tile can therefore enter the queue in the same tick it is claimed.
3. **A pop is skipped unless the tile is still terra nullius AND has an attacker-owned
   neighbour** (`attack.rs:284`).

**And the per-tick claim COUNT is not modelled at all:** `attack.rs:239-254` computes a FLOAT
budget from `attack_tiles_per_tick(troop_count, owner_type, target_is_player, defender_troops,
border_size + random.next_int(0,5))` - note the **one extra PRNG draw per tick**, taken before the
pop loop - then decrements it by the terrain/troop-dependent `tiles_used` per claim
(`attack.rs:313`). The observed 8,9,9,9,10,10,12,11,10,12,12,12 claim counts are that quantity.
So the claim count cannot be complete until troop growth, gold income and `attack_logic_at_tile`
are ported.

**PART 2 INVENTORY: 20 of 32 ticks match.** Ticks 300..319 reproduce the engine's per-tick
FNV-1a-64 state hash exactly (19 no-op ticks plus the first claim tick, 319, with all 8 claims).
First mismatching tick is **320**, the second claim tick, with 6 tile-owner differences. No hash
was relaxed, no tick faked, and the FNV contract is hard-coded on both sides and cross-checked by
reconstructing all 32 recorded hashes engine-side.

**Still unproven, carried forward:** the plane's ground truth is reconstructed from `ownedTiles`
(so the fallout bit 13 is assumed equal, never exercised in this window since no tick sets it);
the exact RANK of 838544 is reproduced in membership but not by any of three models tried, and the
residual is draw accounting; whether in-tick `add_neighbors` is the only in-tick addition
(`add_border_tile`/`remove_border_tile` feeding the next tick's `border_size` is unmodelled); and
the port still lacks troop growth, gold income, the float claim budget, cluster capture,
`handle_dead_defender`, and `refresh_to_conquer`/retreat-on-empty.

**WHAT THE SLICE DID PROVE 2026-09-18 (`ofcuda_tick`).** The cuda-oxide kernel reproduces
the recorded claim order and priorities bit-exactly on two independent cases, compared as raw
f32 bits rather than with an epsilon: 32/32 pops identical across the whole queue including
priority bits, the kernel's own frontier size read back from a device sentinel thread
(`u32::MAX` = heap empty) so it is not inherited from the host, and the heap's tie-breaks
reproduce `flat_heap.rs`'s own vector (`[10,20,30,40]@1.0 -> 10,40,30,20`). The device core is
a verbatim copy of the host core, kept honest by a test that was verified to FAIL when the copy
is deliberately perturbed and pass again after restore. Two earlier MISMATCH lines were the
worker's own harness bugs and were reported rather than tuned away.

**FMA, measured not assumed:** the generated PTX of `frontier_step` contains exactly one
contraction, `fma.rn.f32` for `(r + 10) * f + tick`, while `1 - k*0.5 + mag*0.5` stays
uncontracted. Contracted vs two-step agree on 1219/1219 enumerated (r, owner-count, terrain,
tick) points, so the contraction is inert for this arithmetic, every factor being exact in
binary. The count is re-read from the PTX on every run so it cannot drift silently.

**ORDER OF RECORD CLARIFIED.** My instruction to the worker quoted the tick-319 order
`[460260, 459254, ...]` as the target while also stating the rule is N,S,W,E. Those cannot both
hold. Measured: that order is reproduced ONLY under W,E,N,S, meaning it came from the **pre-fix
native dump**, whereas the `.ts` dump `[459261, 454261, 460255, 454254, 459254, 453260, 453255,
460260]` is reproduced ONLY under N,S,W,E. The worker caught the contradiction, made the order
an explicit input, and printed both with each dump's own recorded order alongside. **The
authoritative order for the CUDA port is N,S,W,E (TS)**; W,E,N,S is carried in the crate only as
a cross-check against pre-fix dumps. My error, not the port's.

**ASSUMPTIONS STILL OPEN in the frontier slice:** (a) that the dump's `ownedOrder` at
`claim_tick-1` is the border set the engine had when it built that frontier; (b) that
`owned_any` (eligibility) and `owned_mine` (owner count) are genuinely two distinct sets. They
carry identical counts in the validated cases because no rival is adjacent, so a conflation bug
would first show on a **contested border**, which is the case that has not been exercised.

**DEFINITIVE ROOT-CAUSE SPEC FOR THE CUDA PORT (both divergences) 2026-09-18.** These are the two
bugs that broke native-vs-TS parity for months. The CUDA port must implement both correctly; each
was measured, not inferred.

**(A) The game id is a DOUBLE rounding FOLLOWED BY a uint32 truncation.** `bridge/session.ts:55-65
seedToGameID` (duplicated verbatim at `bridge/env.ts:74-85`):

1. `h = simpleHash('rl-' + seed)` - djb2 over UTF-16 code units,
   `hash = (hash << 5) - hash + charCode; hash = hash & hash` (the `&` is JS ToInt32 wrapping),
   then `Math.abs(hash)`, giving an integer-valued double in `[0, 2^31]`. The `rl-` prefix is added
   **inside** the function, so the caller passes the raw seed string.
2. Eight times: `h = (h * 1103515245 + 12345) & 0x7fffffff; out += ALPHABET[h % 62]`, with
   `ALPHABET = 'A-Za-z0-9'`, producing an 8-character id.

The multiply and add are **IEEE-754 double**, and that is the precision loss: for `h > 2^22` the
exact product exceeds 2^53 and **rounds** (ulp is 2^9 near the top of the range, so the value moves
by up to ~256), and the add can round again.

**The distinction that matters:** this is a **uint32 truncation applied AFTER the double rounding**,
not pure-double arithmetic. The next iteration receives
`floor(round64(h * 1103515245 + 12345)) mod 2^31`, which is **not** the low 31 bits of the exact
integer product. So a wrapping-u32 port (the original bug) is wrong, and a pure-i128 or pure-f64
port would also be wrong. Verified against the real thing: an inline re-implementation reproduces
the **actual exported TS function for 30/30 seeds** (`scripts/seed_game_ids.ts`), so this is
measured against the live function rather than read off the source.

**(B) The neighbour visit order is the function mapping PRNG-draw-index to tile.** For each frontier
tile popped, iterate its four neighbours **N,S,W,E** and, for every neighbour whose owner differs,
draw **exactly one** PRNG value and push it with
`p = (r + 10) * (1 - 0.5*numOwnedByMe + mag/2) + tick`. The k-th draw therefore binds to whichever
neighbour the visit order reaches k-th, so an inverted order gives every candidate a different key:
a different dequeue order, a different tile's magnitude charged against the live troop ledger, and a
different tile consuming the per-tick budget. That single inversion was the entire divergence,
including an apparently separate tick-308 troop off-by-one.

**Post-fix divergence stack (measured):** with (A) fixed, the first state-plane divergence moved
from decision 0 / tick 2 (0 of 24 decisions equal) to decision 2 / tick 4, with **player ids, bot
spawn tiles and the whole spawn phase now matching TS exactly** on four seeds. The residue at tick 4
is expansion target selection, i.e. bug (B). With (B) fixed, all 10 records pass their full length.

**BOTH FIXED, AND FIRST CLEAN PASSES RECORDED 2026-09-18.** (A) is committed as
`4c0ee39 engine: derive the game id with IEEE-754 double semantics to match the TS core`; the
native game id for seed `parity` is now `QqkIuyke`, matching TS, and the divergence at
decision 0 collapsed from every decision differing to a single tile on a single bot. (B) is
fixed in the working tree at the time of writing (`map.rs` and its callers, plus the
corrected comments in `replay.rs`).

Independently verified by me, not taken from the worker's report:

| record | before | after |
|---|---|---|
| `curr-b002-s1-pangaea` | diverge tick 321 | **PASS, 896 checkpoints, every tick** |
| `curr-b004-s2-pangaea` | diverge tick 310 | **PASS, 596 checkpoints, every tick** |

**SWEEP COMPLETE: 10/10 EARLY-CURRICULUM RECORDS PASS AT EVERY TICK, 2026-09-18.** From the
per-record logs, not the harness summary line (which misleadingly echoes its own max-ticks
cap as `PASS_through_100000`):

| record | before | after |
|---|---|---|
| curr-b002-s1-pangaea | diverge @321 | PASS, 4495 checkpoints |
| curr-b002-s4-pangaea | diverge @307 | PASS, 4495 |
| curr-b002-s7-caucasus | diverge @332 | PASS, 4495 |
| curr-b004-s10-greatlakes | diverge @307 | PASS, 4495 |
| curr-b004-s2-pangaea | diverge @310 | PASS, 4495 |
| curr-b004-s5-onion | diverge @332 | PASS, 4495 |
| curr-b004-s8-blacksea | diverge @311 | PASS, 4495 |
| curr-b007-s3-pangaea | diverge @316 | PASS, 4495 |
| curr-b007-s6-europe | diverge @307 | PASS, 4495 |
| curr-b007-s9-betweentwoseas | diverge @316 | PASS, 4495 |

4495 checkpoints is the record's full length (4500 turns); the horizon is not 100,000. Two of
these I confirmed by hand at 896 and 596 checkpoints before the sweep finished.

**Scope, stated precisely so this is not overclaimed: parity is achieved for the regime the
CUDA port targets (curriculum stages 0-7: bridge maps, 2/4/7 Easy tribe bots, zero nations),
at every tick, over full record length.** Still open: (C) the nation spawn land-paint
off-by-one-tick, which is untouched and fires on the nation controls (tick 15 and tick 309);
the repo's general 40-record corpus with humans and nations has not been re-swept; the
records are `gameType Public` (300-turn spawn) while the trainer runs Singleplayer (100-turn
spawn); and 4495 ticks covers roughly 45% of an RL episode's ~10,000.

| level | claim | how it is checked |
|---|---|---|
| L0 | map/terrain load identical (same bytes, same land mask) | hash of `terrain`/`state` after load |
| L1 | tile plane identical (u16: owner bits 0..=11, fallout bit 13) | per-tick tile-plane hash vs `tick_dump` |
| L2 | featurizer identical given a fixed state | obs hash equality |
| L3 | legality mask identical given a fixed state | mask hash equality |
| L4 | reward identical given a state sequence | term-by-term diff |
| L5 | tick-for-tick trajectory identical on the deterministic subset | `tick_dump` first-divergent-tick |
| L6 | win rate / episode length / tile share within tolerance on the full loop | distributional comparison, same seeds |

L0-L4 are cheap and fully exact, and they cover the parts the profile actually blames
(`of_region_tile` 10.5%, the per-step buffer rebuilds and copies). L5-L6 are the tick.

**L0 PASSED 2026-09-18.** A cuda-oxide kernel on the 5080 loads the engine's own RL map
(Pangaea, 1000x1000) and reproduces, exactly:

| value | engine oracle | CUDA kernel | my independent C verifier |
|---|---|---|---|
| terrain FNV-1a 64 | `0xebffa87c2568cc58` | `0xebffa87c2568cc58` | `0xebffa87c2568cc58` |
| pre-spawn state FNV-1a 64 | `0x6334dfb980453d25` | `0x6334dfb980453d25` | `0x6334dfb980453d25` |
| land tiles | 420335 | 420335 | 420335 |

Three independent implementations agreeing on the same bytes is what makes this a result
rather than an assertion: the engine's own dump, the CUDA kernel, and a throwaway C
program written by a different agent that shares no code with either. The CUDA side also
reproduced the 2000x1000 `world` map the same way (`0xf46bd55d2054841a`, 651569 land) with
`gpu_chunk_digest_mismatches 0` across 489 per-chunk digests. Note for anyone extending
this: FNV-1a cannot be combined across chunks by the usual `h1*P^len2 ^ h2` trick, because
integer multiplication does not distribute over XOR, so the sequential chain is the only
valid reference and the chunk digests are a cross-check, not a substitute.

Also produced and verified: `rust/parity-traces/parity_stage0_seedparity.jsonl`, a
deterministic scripted episode (spawn at tile 8555, then `expand` forever, 15 ticks per
decision, 24 decisions, stage 0, Pangaea, seed `parity`). It drives a direct `RlSession`
and the real `ofenv_*` C ABI in lockstep and hashes the state plane from both after every
decision: 24 decision lines, **zero lockstep failures**. It carries per-decision state
hashes, per-player tiles/troops/gold, the obs hash (`0x9d9227b8ed7152ba`, 4270 f32) and
mask hash (`0x3400cd1de2eb2a28`, 40893 f32) at decision 1, and the itemised reward terms.
That file is the L1-L5 target.

**L1 PASSED 2026-09-18 (PRNG and spawn).** The engine's sfc32 (`prng.rs`) and its spawn
tile selection (`spawn_util.rs`) are ported to a cuda-oxide kernel and agree bit-exactly
with the engine. Three-way agreement again:

| check | result |
|---|---|
| `next()` raw u32, 24 draws x 4 seeds | 96/96 identical (CUDA kernel, my C, engine dump) |
| `next_int` 16 draws x 4 ranges x 4 seeds | 256/256 identical |
| spawn tiles, 4 seeds x 2 bots + scripted human | 12/12 tiles, 24/24 x/y |
| `simple_hash` of the game ids | 4/4 identical to the engine's dumped seeds |

I wrote my own sfc32 and `simple_hash` in C from `prng.rs` and reproduced the engine's
dumped streams exactly, then ran the CUDA binary myself and diffed it against my C: all
four seeds identical. The traps are all pinned by matching rows: the 12 warm-up draws,
`next()` as `(t as u32)/2^32` (a u32 division, not f64), `rand_element` drawing nothing
on an empty list, and `shuffle_array` doing len-1 draws. One real divergence was found
and fixed during the port and is worth knowing: the chance() rows must run on the SAME
generator in sequence, not on a fresh one, or they diverge from draw 1.

**L2 PASSED for the deterministic planes (CPU reference, not yet on GPU).** A new engine
bin dumps the engine's own /8 planes and a comparator diffs them cell by cell against the
standalone reference. I re-ran the comparator myself: **PARITY, 250000/250000 cells
across 16 planes** on the primary dump (stage 40, decision 250, Pangaea, 115 players), LUT
and CLUT byte-identical, exit 0. Across 10 episodes that is 202347 cells per plane and
zero mismatches anywhere. Coverage is real, not vacuous: `ego_enemy` is non-zero in
thousands of cells, four of the six static planes are non-zero somewhere, and a harness
self-test (flipping one cell in a copy) correctly produced a divergence, so a pass means
detection rather than a blind compare.

Findings from this level, each of which changes the port:

1. **The defense-bonus plane is degenerate and cannot be evidence.** Tile-state bit 14 is
   *declared* defense_bonus and is written by the TS client (`TileCodec.ts`), but the
   native Rust engine never sets it. So in the RL env that plane is identically zero and
   both sides matching proves only that both read an always-zero bit. Earlier notes in
   this file said the flag was "declared but never written", and a later audit called that
   "wrong"; the precise truth is the declaration is real and the native setter does not
   exist, which is the version to build against.
2. **Latent bug in the reference, reported not patched:** `offeat_ref/src/pool.rs:183-186`
   applies the slot LUT a second time to an already-slotted owners plane, where the engine
   indexes `clut` directly by the slotted value (`ofcore/src/feat.rs:1225`). It is a no-op
   only because RL small ids are dense `1..N` (verified: the dumps carry ids 1..115). The
   engine side is correct. Fixing it requires re-running the differential, so it is
   recorded here rather than changed under a verified state.
3. The fused walk in `oftrain/src/engine.rs` is unreachable from the engine crate (it
   depends on ofcore, and oftrain depends on the engine: a cycle), so the dump uses
   `ofcore::feat::pool_ego_db`, which is the trainer's own reference for diffing the fused
   walk against. Arithmetic-identical: same clut class counts over the 64 tiles, same /64.
4. Not covered at this level and not claimed: the 32 AE latent channels (not engine
   geometry), the 51 unit-derived transient planes (the reference stubs them with an
   explicit `NeedsUnitState` error rather than silent zeros), the spawn-phase legal_tile
   mask, and the token/scalar blocks.


### D11 — per-decision env cost tripled after the horizon went 64 -> 128 [OPEN, hypothesis]

Measured from the live run on 2026-09-18 (PID 1276244, up 7h7m, 9.5M of 20M steps):

| | horizon 64 (pre-D8-fix, drop-in) | horizon 128 (now, declared value) |
|---|---|---|
| decisions per rollout | 64 x 64 = 4096 | 64 x 128 = 8192 |
| Env phase per rollout | 12.1 s | 60-73 s |
| ms per decision | ~2.9 | **7.7-8.9** |
| SPS | 333 (260-364) | **115 (108-128)** |
| Env share of the loop | 87% | **94-96%** |

The schedule itself is fine and this is NOT a D8 repeat: 20M / 8192 = 2441 epochs, and
epoch 1165 at step 9.5M is 47.7% of the cosine against 47.5% of the step budget, so the
anneal still spans the whole run.

The cause is not CPU contention (no compiler or eval is running; puffer holds 403% of
800% and the GPU sits at 0% with Model at 1%). The leading hypothesis is that cost per
decision scales with how far into the game the rollout runs: horizon 128 covers 1920
ticks instead of 960, and by then players own far more tiles, so `flood_border_cluster`,
`maybe_remove_clusters` and the `add_neighbors` refresh over a player's whole border all
do more work. Stage progression compounds it (stage ~1.4 means 4-7 bots instead of 2,
and more players means more borders and more clusters).

Controlled test if it matters: hold the stage fixed and run the same seed at horizon 64
and 128, comparing Env ms per decision. That costs two trainer restarts, so it is not
free, and it does not block the port.

Consequence for the port: the env is now **95% of the loop**, up from 87%, and the GPU
is idle throughout it. Every argument for putting the env on the device is stronger at
horizon 128 than it was at 64, and the non-game overhead (busy-wait, buffer rebuilds,
copies, syscalls) is a bigger share of a bigger number.


### D10 — a timeout with a lead is punished more than dying [OPEN]

The user's rule is "if it times out without a winner it's a loss, hands down". At the
`terminal_reward` level that holds: the `!won` branch returns `-W_WIN = -30.0`
unconditionally and ignores `timed_out`. But the *composed* loss is not uniform:

| outcome | composed terminal | why |
|---|---|---|
| death / elimination | **-33.0** | `-30` terminal plus the `-3.0` alive->dead transition penalty |
| plain timeout, never reached closeout | **-30.0** | terminal only |
| timeout after reaching >=45% land | **-50.0** | terminal plus `v10_timeout_after_closeout_penalty = -20.0` |

So the ordering is win >> plain timeout > death > timeout-with-a-lead. That last row is
backwards for the behaviour we want: an agent that climbs to 45% and fails to close is
punished 20 points harder than one that dies early, which argues for dying rather than
committing to the closeout. The `-20` also makes a timeout *more* than a loss, which is
not what the rule says. Recommended fix: drop `v10_timeout_after_closeout_penalty` so
every loss prices at `-30`, and keep the `-3.0` death transition as shaping. Caveat
per the standing skill rule: any reward change invalidates the warm start and the value
head has to relearn, so this is a deliberate cost, not a free edit.




### 2026-09-18 16:35 — operator tick: post-parity re-baseline

Run is healthy on the faithful engine. `openfront-train` MainPID 2025707 (started 15:46:29, the 15:41
build), steps 11.0M of 20M, SPS 182-253, Env 88-91% of the loop, entropy 6.7-6.9, `stage_wins` 11.6 and
climbing, `win_rate` 0.000 and `stage` 0 (the window is refilling after the restart). `WATCHDOG_OFF`
absent, `openfront-watchdog.timer` active and enabled (fires every 2 min), StartLimitBurst=30, and the
`/run` drop-in is exactly `HORIZON=128 ENT=0.02` — the declared E5 config, no stale pin (D8 check).

Two restarts landed inside this tick, both resetting the ladder: 15:09 (parity engine redeploy) and
15:46 (E7 build). E7 is now **deployed and executing** — the live segment prints
`pufferl: curriculum state: 0/64 envs restored ... 0 refused`, which is the load path running against a
checkpoint that has no sidecar yet. **E7's outstanding proof is a restart after the first
sidecar-carrying checkpoint** (the 11.47M one this process will write), showing `restored > 0`.

**Measurement (first post-parity number):** STAGE=2, ckpt `10650112`, n=160, 8 envs ->
`wins=0.5 losses=0.5 score=142.603638 perf=143.317627`, i.e. 80/160 = **0.500** against the pre-parity
stage-2 reading (12:41, ckpt 9830912) of 0.400 / 92.64. Two variables moved at once (checkpoint
9.83M -> 10.65M, engine redeployed at 15:09), so this is suggestive, not a settled result, and the
**matched all-zero control at stage 2 has not been re-run on the post-parity engine** — the reference
is still the pre-parity 12/120 = 0.100. The wall is still stage 2: 0.500 against a 0.90 gate, so the
faithful world did not change the ladder's shape.

**Next, in order:** (1) matched all-zero control at STAGE=2 on the post-parity engine (~50 min, cap-bound);
(2) verify E7 restores a non-zero stage count after a restart on a sidecar-carrying checkpoint;
(3) then choose the stage-2 intervention — D10's `v10_timeout_after_closeout_penalty = -20.0` (a failed
closeout prices at -50 vs -33 for dying early, i.e. it argues against committing to the 80% takeover the
stage requires) is the leading candidate, but it is an engine edit and must not be made while the parity
worker is rebuilding that tree. E8 (AGENTS=128) stays blocked: 8.3G available against ~+5G needed.

### E7 — VERIFIED ON THE LIVE RUN 2026-09-18 18:45 (operator tick)

The outstanding proof was "a restart after the first sidecar-carrying checkpoint". That restart
happened: an orderly stop/start at **17:46:34-35** (`systemd[1]: Stopping ... Deactivated
successfully`, MainPID **2316667**) resumed from `1789771591260/0000000011469312.bin` (11,469,312 =
the first checkpoint the E7-enabled process wrote, 17:09, and therefore the first carrying
sidecars). Verbatim from the live segment of `current.log`:

```
pufferl: resumed schedule from 0000000011469312.bin at step 11469312 (epoch 1400/2441, 57.4% into the cosine)
pufferl: curriculum state: 64/64 envs restored from .../0000000011469312.bin sidecars, 0 refused
openfront: curriculum state restored from .../0000000011469312.bin.env000.state.json: stage=0 window=32/40 stage_wins=32.0 advances=0.0
openfront: curriculum state restored ... env003 ... window=38/40 stage_wins=38.0
openfront: curriculum state restored ... env025 ... window=25/40 stage_wins=25.0
```

Evidence that the values came from the **file** rather than the cfg default, in order of strength:

1. the windows are heterogeneous and mostly full (25/40 .. 38/40) and `stage_wins` equals the
   window count per env (env000 32/40 & 32.0). A fresh process reads `window=0/40 stage_wins=0.0`
   for every env - that is what the 15:47 segment's `0/64 restored` produced;
2. **behavioural**: with the window already ~32/40 populated, the envs re-promoted `stage 0 -> 1`
   within ~30 min of the restart - 20 promotions in the live segment at `win_rate 0.950-1.000` -
   against ~94 min for the first climb on a fresh window. The gate can only fire that early if the
   restored window is feeding it;
3. `0 refused` means all 64 sidecars passed the identity check against the checkpoint they sit
   beside (the guard exists and did not reject a legitimate restore).

Cost of the fix, measured: the ~1.7M-step / ~94-minute ladder re-warm after every restart is gone.
`stage` itself restored as 0 for every env because that checkpoint recorded stage 0 (the run only
climbed to 1 afterwards) - expected, not a silent reset.

**Still not proven, carried forward:** none of the four refusal guards has ever fired live
(identity mismatch, out-of-range stage, window-length mismatch, `OPENFRONT_HOLD_STAGE`
suppression) - they need a deliberate negative probe, not an inference from "0 refused".
**Optimizer state is still not serialized** (`puf_save_weights` writes only the fp32 master
weights, 101,933,056 B = 25.5M x 4), so Adam's moments restart at every launch, as do the env RNG
and in-episode state. That is the next durability gap after this one.

### E5-c — matched stage-2 control on the faithful engine [DONE 2026-09-18 19:20, n=160]

The 16:32 stage-2 reading (80/160 = 0.500) was taken against a **pre-parity** control. This is the
matched one, same engine (post-parity cdylib, sha256 `43c16d48…`, inode 9608558), same stage,
n=160, 8 envs:

| policy | wins/n | win rate | score |
|---|---|---|---|
| trained (ckpt `10650112`, 16:32) | 80/160 | **0.500** | 142.60 |
| all-zero control (post-parity, 19:20) | 16/161 | **0.099** | -5.68 |
| all-zero control (pre-parity, 13:22) | 12/120 | 0.100 | +11.21 |

Read: **+0.401 win rate (SE ~0.043, z~9) and +148.3 return** - the policy is decisive over random
legal play at stage 2, and still 0.40 below the 0.90 gate. The wall is a policy/curriculum
problem, neither a stale control nor a broken gate. Note which channel is comparable across the
two engines: the **win rate reproduces (0.099 vs 0.100)** while the return does not (+11.21 ->
-5.68), because the parity fixes change the games a zero policy plays (different game ids, spawn
tiles, expansion order). Post-parity control by stage: 0.298 (s0) / 0.600 (s1) / 0.099 (s2).

### D12 — the watchdog raced the eval's thaw [FIXED in `eval_policy.sh`, NOT yet exercised]

`stall-report.txt`, 2026-09-18 19:20:43: `watchdog: frozen: log untouched for 2416s -> systemctl
restart openfront-train.service`, i.e. **48 s after the eval's SIGCONT** at 19:19:55. The stall
check compares `current.log`'s mtime against now, and a SIGSTOP freeze leaves that mtime at the
moment of the freeze (18:40:27) - so the instant `WATCHDOG_OFF` is removed, every eval that
outlasted `STALL=900s` (all of them) hands the watchdog a "frozen" trainer. The SIGSTOP design
that keeps the ladder alive was thus defeated by the watchdog on the way out.

E7 turned the damage into a cheap restart (64/64 sidecars restored, envs re-promoted `0 -> 1`
inside a minute) but the in-flight epoch and the Adam moments were still lost. Fix, in
`eval_policy.sh`: after the thaw, wait (bounded, `FRESH_LOG_TIMEOUT=180s`) for a fresh log write
before removing `WATCHDOG_OFF`, using pure-bash `-nt` so there is no `stat` dependency on the cron
PATH; on timeout it says so and releases the watchdog anyway, because a genuinely dead trainer must
still be restartable. Not exercised yet - the next eval is its first test.
Independent hardening still available on the watchdog side (treat a stale log as frozen only if
`/proc/<pid>/stat` CPU time is *also* not advancing); that lives in the flake and needs a
`nixos-rebuild`, so it was deliberately not done from an unattended tick.

### Next step, ranked (2026-09-18 19:30)

1. **Do not change the reward yet.** D10's `v10_timeout_after_closeout_penalty = -20.0` is still the
   only stage-2 lever with a stated mechanism, but its premise is unmeasured: nothing counts how
   often an episode actually takes the timeout-after-closeout path at stage 2, and E6's lesson is
   that a reward-misalignment claim built on a named constant rather than the composed terminal is
   how a day gets spent. Instrument the path (a counter through `puf_log`) before pricing it.
2. **E8 (AGENTS=128) stays blocked**: `free -h` reads 4.8G available with 5.8G swap already in use,
   against the ~+5G the doubled env state needs, on a box whose trainer has already been
   OOM-killed once at 6.7G peak RSS + 5.5G swap peak. Do not run it until RAM is freed.
3. **A fresh run is not indicated yet.** The user directive ("kill the runs once we have parity")
   authorises it, and parity for stages 0-7 is in hand, but a from-random-init run would re-climb
   the ~11.5M steps of progress this run already has while the current LR (57% into the cosine) is
   still learning. Revisit if stage 2 is still at ~0.50 when the cosine ends (8.5M steps, ~18h).
