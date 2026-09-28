# The stage-0 reward, measured as a stream

Late in the project we stopped trusting the reward function and measured it. This
is the report. It explains why the stage-0 policy learned nothing for a long time. It is also
the reason we no longer treat a reward curve as evidence on its own.

## What we did

A harness drove the OpenFront environment, which is the PufferLib 5 port. It
played 24 full episodes at stage 0 and logged every reward component at every
decision. Four scripted behaviours played six episodes each:

| behaviour | what it does |
|---|---|
| `random` | seeded uniform choice among the actions the legality mask allows |
| `grow` | expand into neutral land whenever that is legal |
| `attack` | attack an adjacent enemy whenever that is legal, else expand |
| `camp` | spawn once, then send no-op forever |

- Environment: pangaea, seed `s1`, 3 bots, Easy, stage 0, 1 agent,
  `ticks_per_decision=15`, `max_episode_ticks=21000`, `quantity_frac=0.25`.
- Engine: `libopenfront_engine.so`, sha256 `4051ebd3...f96fd`.
- Scale: 12,696 decisions, 24 terminal events, 299 s of wall time.

## The per-decision reward

| | mean | std |
|---|---|---|
| all decisions | +0.14729 | 5.98471 |
| non-terminal | +0.01931 | 0.54817 |
| terminal | +67.72 | 121.78 |

The terminal term fires on 0.19% of decisions and carries 86.9% of the absolute
return. Everything the agent can do between terminals is worth almost nothing.

## Where the return sits

![share of total return per component](graphs/stage0_reward_shares.png)

| component | mean | sum | share of total return |
|---|---|---|---|
| terminal | +0.13456 | +1708.42 | 91.4% |
| closeout | +0.01178 | +149.58 | 8.0% |
| strength | +0.00301 | +38.23 | 2.0% |
| strength_delta | +0.00174 | +22.07 | 1.2% |
| death | -0.00402 | -51.00 | -2.7% |
| survival | +0.00068 | +8.61 | 0.5% |
| tempo | -0.00043 | -5.41 | -0.3% |
| waste | -0.00006 | -0.82 | 0.0% |
| dominance | +0.00003 | +0.37 | 0.0% |

Seven further terms were identically zero because nothing read them:
`action_churn`, `boat_outcome`, `embargo_outcome`, `combat_outcome`,
`diplo_panic`, `combat_action`, `duo`.

## Terminal versus shaping, per behaviour

| behaviour | shaping sum | terminal sum | mean episode return |
|---|---|---|---|
| random | +181.45 | +1671.25 | +308.78 ± 0.41 |
| grow | -0.44 | +14.68 | +2.37 ± 3.78 |
| attack | -7.38 | +11.25 | +0.64 ± 0.43 |
| camp | -12.00 | +11.25 | -0.13 ± 0.50 |

A win paid 278.94. A loss paid **+1.875, which is positive**, so 17 of the 24
episodes ended on the same positive constant. The agent was paid for dying.

## The stream is inverted

- Non-terminal reward, by behaviour: random +0.15351, grow +0.00347,
  attack +0.00350, camp +0.00276. Pooled within-behaviour std 0.54647.
- Random against the rest: `d = 0.276`, a spread of 0.1508.
- Grow against attack against camp: spread 0.00074, `d < 0.02`. The deliberate
  behaviours are indistinguishable to the learner.
- 100% of episode-return variance is between behaviours, and 0% is within.
- Wins against deaths: random 6/6 and 0/6, grow 0/6 and 5/6, attack 0/6 and 6/6,
  camp 0/6 and 6/6.

So the only policy that won was the uniform random one, and it won every game.
Every policy that tried to play the game died. A policy-gradient learner
maximising advantage on that stream is pushed toward maximum entropy.

The live run agreed with the measurement. At stage 0 its entropy sat at 7.4 to
8.3, its policy loss was about 0.000, and it never advanced a stage.

## What we took from it

A reward profile is a distribution over decisions, not a scalar. Log the
components, and check which behaviour wins under each one. No reward curve would
have shown this: the stage-0 curve was flat, and it stayed flat. The harness is
the only reason we can state what the agent was optimising at all.
