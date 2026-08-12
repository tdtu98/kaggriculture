# Orbit Wars — Game Rules and the PPO Agent in This Reference

This file summarizes two things:

1. **The rules of Orbit Wars**, a past Kaggle simulation competition (distilled from `README.md` / `agents.md`).
2. **How the reference PPO agent in `src/` actually works** — its observation encoding, action space, network, training loop, and the specific simplifications it makes.

Everything in section 2 is read off the code in this folder, not from the notebook prose alone, so where the notebook description and the code differ, the code wins.

---

# Part 1 — The Game

## 1.1 One-paragraph summary

Orbit Wars is a real-time strategy game in **continuous 2D space**. Each player starts with one home planet and sends fleets of ships at chosen *angles* to capture neutral and enemy planets. Planets near the center **orbit a sun**; the sun destroys any fleet that crosses it. The game runs **500 turns**, and the winner is whoever ends with the most total ships (garrisoned + in flight). It supports 2 or 4 players; the reference agent trains 2-player.

The defining twist vs. classic "planet wars" games: you don't pick a target planet, **you pick a direction**. Since planets rotate and fleets take many turns to arrive, aiming correctly means leading a moving target.

## 1.2 Board

| Element | Value |
|---|---|
| Board | 100 × 100 continuous space, origin top-left |
| Sun | center `(50, 50)`, radius `10` — fleets crossing it are destroyed |
| Planets | 20–40, in 5–10 symmetric groups of 4 |
| Symmetry | every object mirrored 4-fold: `(x,y)`, `(100-x,y)`, `(x,100-y)`, `(100-x,100-y)` |
| Episode length | 500 turns |
| Time budget | 1 s per turn (`actTimeout`), plus an overage pool |

## 1.3 Planets

A planet is `[id, owner, x, y, radius, ships, production]`.

- `owner`: player `0–3`, or `-1` for neutral.
- `production`: integer `1–5`. An owned planet adds this many ships every turn.
- `radius = 1 + ln(production)` — higher-production planets are physically bigger, so they're both more valuable *and* easier to hit.
- `ships`: garrison, initially 5–99 skewed low.

**Orbiting vs. static.** A planet rotates around the sun iff `orbital_radius + planet_radius < 50`, at a per-game constant angular velocity of 0.025–0.05 rad/turn. Outer planets are static. At least 3 groups are guaranteed static, at least 1 guaranteed orbiting. You can reconstruct any orbiting planet's future position from `initial_planets` + `angular_velocity`.

**Home planets.** One symmetric group is chosen as the starting group; each player gets one planet from it with 10 ships. In 2-player games the two homes are diagonally opposite.

## 1.4 Fleets

A fleet is `[id, owner, x, y, angle, from_planet_id, ships]`. Fleet size is fixed at launch and never changes in flight.

**Speed scales with size:**

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ** 1.5     # maxSpeed = 6.0
```

1 ship → 1.0 units/turn; ~500 ships → ~5; ~1000 ships → max. **Big fleets are strictly faster**, which is a real strategic lever: splitting into many small fleets makes them slow *and* individually weak.

**Movement and death.** Each turn a fleet advances in a straight line. It is removed if it leaves the board, if its path segment comes within the sun radius, or if it hits a planet (→ combat). Collision is **continuous** — the whole segment from old to new position is tested, not just the endpoint. So a fast fleet can't tunnel through a planet.

**Launching.** Each turn your agent returns `[[from_planet_id, angle_radians, num_ships], ...]`. You may only launch from planets you own, not more ships than the garrison holds, and you may issue multiple launches from the same planet in one turn. The fleet spawns just outside the source planet's radius along the given angle. `angle = 0` is +x (right), `pi/2` is +y (down).

## 1.5 Comets

Comets are temporary planets that fly through on highly elliptical solar orbits. They spawn in symmetric groups of 4 at steps **50, 150, 250, 350, 450**.

- radius 1.0, production 1/turn, starting ships random and skewed low (min of 4 rolls of 1–99, shared by the group).
- Speed `cometSpeed` = 4.0 units/turn.
- They appear inside the normal `planets` array and follow all normal planet rules — you can capture them, they produce, you can launch from them.
- `comet_planet_ids` tells you which IDs are comets; `comets[].paths` + `path_index` give the full future trajectory, so comet interception is fully predictable.
- When a comet leaves the board it is deleted **along with its garrison**. Comet removal happens *before* fleet launch, so you can't evacuate a departing comet.

Comets are a pure tempo resource: cheap free production for ~100 turns, but any ships you park there are eventually lost.

## 1.6 Turn order

Every turn resolves in this fixed order:

1. Comet expiration (departed comets removed).
2. Comet spawning (at the 5 designated steps).
3. **Fleet launch** — all player actions processed.
4. **Production** — every owned planet/comet gains `production` ships.
5. **Fleet movement** — advance, check out-of-bounds / sun / planet collisions; collisions queue combat.
6. **Planet rotation & comet movement** — orbiting bodies move; a fleet swept over by a moving planet is dragged into combat with it.
7. **Combat resolution.**

Note step 4 happens *after* launch: ships produced this turn cannot be launched until next turn.

## 1.7 Combat

When fleets arrive at a planet:

1. Arriving fleets are grouped by owner and summed.
2. The **largest attacking force fights the second largest; the difference survives.** If two attackers tie, everything is annihilated.
3. The surviving attacker then:
   - if it's the planet's owner → ships join the garrison;
   - if it's a different owner → it fights the garrison. If attackers > garrison, **ownership flips** and the new garrison is the surplus.

Consequences worth internalizing: capturing needs `garrison + 1`; multi-way attacks on the same tick cancel each other out first; and reinforcing your own planet is just addition.

## 1.8 Ending and scoring

The game ends at **500 steps**, or early by **elimination** (only one or zero players still have any planet or fleet). Final score = ships on owned planets + ships in owned fleets; highest wins.

## 1.9 Observation reference

| Field | Shape | Notes |
|---|---|---|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | includes comets |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | all players' fleets are visible |
| `player` | `int` | your ID |
| `angular_velocity` | `float` | rad/turn for orbiting planets |
| `initial_planets` | same as `planets` | positions at t=0, for orbit reconstruction |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | full comet trajectories |
| `comet_planet_ids` | `[int, ...]` | which planet IDs are comets |
| `remainingOverageTime` | `float` | seconds of overage budget left |

The game is **fully observable** — there's no fog of war, so any modeling weakness is a modeling choice, not an information limit.

---

# Part 2 — The Reference PPO Agent

## 2.1 Headline design

The agent in `src/` is a deliberately minimal PPO baseline whose only job is to beat the "Nearest Planet Sniper" tutorial bot. Its central design decision:

> **Each owned planet, on each turn, is an independent decision unit.**

The policy network is evaluated once per owned planet per turn. It emits a categorical distribution over a small set of **candidate targets** (plus a no-op), and the chosen target is converted into an `[id, angle, ships]` move by a *hand-written* rule. There is no inter-planet coordination, no learned ship count, and no learned aiming.

So the learned function is narrow: *given this planet and this board, which of ~7 nearby planets should I shoot at, or should I hold?*

## 2.2 File map

| File | Role |
|---|---|
| `src/game_types.py` | `PlanetState` / `FleetState` / `GameState` dataclasses + `parse_observation` (dict-or-attr tolerant) |
| `src/config.py` | YAML → nested dataclasses (`EnvConfig`, `ModelConfig`, `PPOConfig`, `TrainConfig`) |
| `src/features.py` | The interesting file: candidate construction, feature encoding, action masking |
| `src/policy.py` | `PlanetPolicy` — 3-encoder MLP, target logits + value |
| `src/ppo.py` | Action sampling, clipped-surrogate PPO update |
| `src/opponents.py` | `KaggleRandomOpponent`, `SelfPlayOpponent` |
| `src/env.py` | Wraps `kaggle_environments.make("orbit_wars")` into a single-learner env |
| `src/train.py` | Rollout collection, return/advantage computation, training loop, checkpointing |
| `eval_vs_sniper.py` | Win-rate evaluation vs. the sniper baseline |
| `play_vs_sniper.py` | Plays one game and dumps an HTML replay |
| `default_cfg.yaml` | The config used for the notebook run |

## 2.3 Action space

Per source planet, a single categorical over `candidate_count = 8` slots:

- **index 0 = no-op** (always unmasked — the policy can always choose to hold).
- **indices 1..7 = candidate target planets.**

When index `k > 0` is selected, the move is emitted as:

```python
[src.id, atan2(tgt.y - src.y, tgt.x - src.x), max(tgt.ships + 1, 20)]
```

Two things are *not* learned:

- **Aim.** The angle points at the target's **current** position. There is no lead/intercept computation, so shots at orbiting planets and comets systematically miss unless they happen to be close.
- **Ship count.** `fixed_ship_count()` in `features.py:229` returns `max(tgt.ships + 1, 20)` — the exact same rule the sniper baseline uses. `EnvConfig.ship_bucket_count = 8` exists in the config but is **unused**; it's a leftover hook for a future ship-count head.

### Candidate construction (`build_candidates`, `features.py:107`)

For each source planet, other planets are bucketed and taken nearest-first with fixed quotas:

```
enemy_quota    = candidate_count // 3      = 2
neutral_quota  = candidate_count // 3      = 2
friendly_quota = candidate_count - 4       = 4
```

then padded with the nearest remaining planets if a bucket is short.

⚠️ Two quirks worth knowing before you copy this pattern:

- `build_candidate_features` enumerates `start=1` and breaks at `idx >= candidate_count`, so with `candidate_count = 8` only **7 of the 8 candidates ever reach the feature tensor** — the last one is silently dropped. Effective mix: 2 enemy, 2 neutral, 3 friendly.
- The quota skews **friendly-heavy** (4 of 8 slots), which spends capacity on reinforcement moves rather than expansion.

### Action masking (`features.py:201`)

Candidate `k` is valid iff:

```python
ships_needed > 0  and  not crosses_sun  and  src.ships >= ships_needed
```

`shot_crosses_sun` does a point-to-segment distance test from the sun center to the segment `launch_point → target`, so straight shots through the sun are pruned. Invalid logits are set to `finfo.min` before the softmax; `safe_target_logits` in `ppo.py:51` rescues fully-masked rows by re-enabling the no-op.

Note the mask only checks the *sun*, not intervening planets — a shot can still be intercepted by a third planet in the path.

## 2.4 Observations

All features are hand-engineered scalars, normalized by constants in `EnvConfig` (`board_size=100`, `max_ships=400`, `max_production=5`, `max_planets=48`). Three groups per decision row:

**`self_features` — 11 dims** (`features.py:139`)

`[bias=1, x/100, y/100, radius/5, ships/400, production/5, is_rotating, my_planet_count/48, enemy_planet_count/48, my_total_ships/19200, enemy_total_ships/19200]`

**`candidate_features` — 8 × 14 dims** (`features.py:160`)

Per candidate: `[valid_flag, is_neutral, is_mine, is_enemy, tgt.x/100, tgt.y/100, dx/100, dy/100, distance/100, tgt.ships/400, tgt.production/5, tgt_is_rotating, crosses_sun, src.ships/400]`. Slot 0 (no-op) is left as all zeros.

**`global_features` — 8 dims** (`features.py:208`)

`[step/500, my_planets/48, enemy_planets/48, neutral_planets/48, my_ships, enemy_ships, my_fleet_ships, enemy_fleet_ships]` (last four normalized by `48*400`).

### What the observation deliberately omits

This is where most of the headroom lives:

- **Individual fleets.** Only aggregate in-flight ship totals appear. The agent cannot see that 200 enemy ships are 3 turns from one of its planets, so it cannot defend or dodge.
- **`angular_velocity` and orbital phase.** `is_rotating_planet` is recomputed geometrically from the current position; the actual rotation rate is never fed in and never used to predict.
- **Comets.** No comet flag, no `path`/`path_index`, no time-to-departure. Comets are treated as ordinary planets, so the agent will happily invest in one that's about to leave and take the garrison with it.
- **Time-to-arrival.** Distance is present but fleet travel time (which depends on fleet size via the log-speed curve) is not.
- **Third-planet occlusion** along the shot line.

## 2.5 Network (`src/policy.py`)

A plain MLP with three separate encoders and two heads, `hidden_size = 128`:

```
self_features   (B, 11)      → self_encoder    → (B, 128)
global_features (B, 8)       → global_encoder  → (B, 128)
candidate_feats (B, 8, 14)   → cand_encoder    → (B, 8, 128)     [shared weights per candidate]

target_logits = target_head( concat[self⊗8, global⊗8, cand] )     → (B, 8)  → mask → softmax
value         = value_head ( concat[self, global, mean(cand)] )   → (B,)
```

Each encoder is `Linear → ReLU → Linear → ReLU`; each head is `Linear(384→128) → ReLU → Linear(128→1)`.

Properties: the candidate encoder is **weight-shared and permutation-equivariant** across slots (a pointer-network-style scoring head), which is the one genuinely good architectural choice here. The value head pools candidates by mean, so it estimates the value *of a single planet's decision row*, not of the board — see the next section for why that matters.

Total size ≈ 460k parameters (~1.8 MB checkpoints).

## 2.6 Environment wrapper (`src/env.py`)

- Builds a fresh `kaggle_environments` env per episode with `num_agents=2`, seeded per episode.
- One side is the learner; the other is driven by an `OpponentPolicy`. With `alternate_player_sides: true`, `learner_player = (env_index + episode_index) % 2`, so the learner sees both starting corners.
- `step()` calls the opponent on its own observation, assembles the joint action in the right order, and returns `StepResult(batch, reward, done, info)`.

**Reward is terminal-only and sparse:** `0.0` on every non-terminal step; at the end, the Kaggle reward (`+1` win / `-1` loss). `terminal_reward` returns `0.0` if both sides report positive reward (draw guard). There is **no reward shaping at all** — no credit for captures, production, or ship differential. Over a 500-turn episode with dozens of decisions per turn, that's an extremely thin learning signal, and it's the main reason the notebook needs 2000 updates.

## 2.7 Training loop (`src/train.py`)

```
for update in 1..total_updates:
    rollout  = collect_rollout(envs, policy, rollout_steps)    # 64 steps × 2 envs
    metrics  = ppo_update(policy, optimizer, rollout, ...)     # 4 epochs, minibatch 256
    if update % 50 == 0: opponent.sync_from(policy)            # self-play refresh
    if update % 50 == 0: save_checkpoint(...)
```

**Rollout collection** (`collect_rollout`, `train.py:48`). Each of the `num_envs` envs produces a `TurnBatch` with one row per owned planet. Rows from all envs are concatenated into one forward pass, actions are sampled for every row simultaneously, then split back per env and assembled into that env's move list. Every row is stored as an independent PPO transition. A rollout step is therefore *one game turn*, but contributes *N rows* where N = number of owned planets — so sample count grows as the agent expands.

**Returns and advantages** (`train.py:131`). Rows produced on the same turn form a `StepGroup` and all share one return:

```python
future_return = group.reward + gamma * future_return * (1 - done)
returns[i]    = future_return
advantages[i] = future_return - values[i]
```

Two consequences:

- This is **plain discounted-return minus baseline (TD(∞) / Monte-Carlo advantage), not GAE** — high variance, no `lambda` knob.
- Every planet on a turn gets the *identical* return, so there is **no per-planet credit assignment**. A good move and a terrible move made on the same turn receive the same reward signal; only the value baseline differentiates them.

Truncated rollouts are bootstrapped by `bootstrap_values`, which averages the value predictions across the env's current planet rows — a rough stand-in for a board-level value.

**PPO update** (`ppo.py:60`). Standard clipped surrogate:

```
adv       = (adv - adv.mean()) / (adv.std() + 1e-8)
ratio     = exp(new_logp - old_logp)
L_policy  = max(-adv*ratio, -adv*clip(ratio, 1-ε, 1+ε)).mean()
L_value   = 0.5 * (returns - value)²
loss      = L_policy + 0.5*L_value - 0.01*entropy
```
with `clip_grad_norm_(0.5)`, Adam at `3e-4`, 4 epochs over the buffer at minibatch 256. There is no value clipping and no LR annealing.

**Self-play** (`opponents.py:33`). `opponent: self` instantiates a second `PlanetPolicy` that is **hard-synced from the learner every `self_play_update_interval` updates** (50). It is not a pool of historical opponents, so the agent trains against a single lagging copy of itself — cheap, but it invites strategy cycling and can drift away from beating fixed baselines. `self_play_deterministic: false` means the opponent samples rather than argmaxes, which adds useful noise.

`opponent: random` swaps in the environment's built-in `random_agent` instead.

## 2.8 Configuration used

`default_cfg.yaml`:

| Key | Value |
|---|---|
| `opponent` | `self` |
| `self_play_update_interval` | 50 |
| `alternate_player_sides` | true |
| `env.candidate_count` | 8 |
| `model.hidden_size` | 128 |
| `ppo.rollout_steps` / `num_envs` | 64 / 2 |
| `ppo.total_updates` | 100 (notebook); **2000 for the real run** |
| `ppo.epochs` / `minibatch_size` | 4 / 256 |
| `gamma`, `clip_coef`, `ent_coef`, `vf_coef` | 0.99, 0.2, 0.01, 0.5 |
| `lr`, `max_grad_norm` | 3e-4, 0.5 |

## 2.9 Reported results

Win rate over 20 games vs. the Nearest Planet Sniper (`eval_vs_sniper.py`, deterministic policy):

| Checkpoint | Win rate |
|---|---|
| untrained | 0% |
| 500 updates (25%) | 30% |
| 1000 updates (50%) | 85% |
| 2000 updates (100%) | **100%** |

Wins typically arrive by **elimination well before turn 500** (game lengths of 70–400 steps in the logs), i.e. the trained policy snowballs rather than winning on final count.

Read the 100% with the right caveats: the opponent is a single fixed heuristic, the ship-count rule is identical to that opponent's, and self-play never exposed the agent to anything qualitatively different. It is evidence the pipeline learns, not evidence of a strong agent.

## 2.10 Running it locally

```bash
pip install "kaggle-environments>=1.28.0" torch pyyaml

cd reference/orbit_war
python -m src.train --config default_cfg.yaml

python eval_vs_sniper.py --config default_cfg.yaml \
    --checkpoint kaggle/orbit_wars_ppo/ckpt_last.pt --games 20 --deterministic

python play_vs_sniper.py --config default_cfg.yaml \
    --checkpoint kaggle/orbit_wars_ppo/ckpt_last.pt --deterministic --output result.html
```

Practical notes on this checkout:

- `default_cfg.yaml` has `save_dir: /Users/tu/Desktop/kaggriculture/reference/kaggle`, which no longer exists — the checkpoints actually here live in `reference/orbit_war/kaggle/orbit_wars_ppo/` (`ckpt_000050.pt`, `ckpt_000100.pt`, `ckpt_last.pt`, from a short 100-update run, **not** the 2000-update weights behind the results table). Fix `save_dir` before retraining.
- `default_train_config_path()` points at `src/configs/default.yaml`, which doesn't exist — always pass `--config` explicitly.
- `eval_vs_sniper.py` / `play_vs_sniper.py` compute `REPO_ROOT = parents[1]`, which assumes they sit one level below the repo root. Here they're at the same level as `src/`, so run them **from inside `reference/orbit_war/`** and rely on cwd for the `src` import.
- Checkpoints are pickled with `weights_only=False` and reference `src.*` module paths; `register_checkpoint_module_aliases()` in the eval scripts installs aliases so old `src.rl_template.*` checkpoints still load.

---

# Part 3 — Where the headroom is

Ranked roughly by expected value if you were to extend this agent:

1. **Learn the aim, or at least lead the target.** Firing at a rotating planet's current position is a guaranteed miss at range. A closed-form intercept (solve for the angle where fleet-arrival time meets orbital position, using `angular_velocity` and the size-dependent speed curve) is pure engineering and probably the single biggest win.
2. **Learn the ship count.** `ship_bucket_count` is already stubbed. `max(garrison+1, 20)` overpays on weak targets, underpays against a planet that will produce during the fleet's transit, and ignores that bigger fleets fly faster.
3. **Reward shaping.** Terminal ±1 over 500 turns is brutally sparse. A potential-based shaping term on ship/production differential would cut training time dramatically without changing the optimal policy.
4. **Per-planet credit assignment.** Giving every planet the same return is the loop's weakest link. Options: a counterfactual baseline, or treating the turn as one joint action with an autoregressive head.
5. **Put fleets in the observation.** Without in-flight enemy fleets the agent literally cannot defend. Even a small per-planet "incoming enemy ships within N turns" feature would help.
6. **Model comets explicitly.** Free production with a known expiry date is a well-defined, exploitable resource — and one the current agent actively misplays.
7. **Better self-play.** Keep a pool of past checkpoints and sample opponents from it rather than always facing the latest lagging copy.
8. **GAE(λ)** instead of raw discounted returns, plus value clipping and LR annealing — cheap standard PPO hygiene.
9. Fix the off-by-one that drops the 8th candidate, and revisit the friendly-heavy quota split.

## Transfer notes for Kaggriculture

The parts of this template that generalize to a different Kaggle sim competition:

- **The decomposition trick.** Turning "one turn = one huge combinatorial action" into "one decision per controllable unit, each a small categorical over masked candidates" is what makes a tiny MLP workable at all. For Kaggriculture the natural unit is the farmer / farm hand rather than the planet.
- **Candidate + mask instead of a flat action space.** Enumerate legal options, featurize each, score with a shared per-candidate encoder, mask illegal ones to `-inf`. This keeps the head size fixed while the legal set changes turn to turn, and it makes illegal actions structurally impossible.
- **Hand-rules for the parts you don't want to learn yet.** Fixing the ship count let this baseline learn only target selection. The same staging works elsewhere: freeze the hard continuous sub-decision, learn the discrete one, then unfreeze.
- **Self-play with side alternation** and a lagging opponent copy, which needs no external baseline to train against.
- **The reusable skeleton itself:** `game_types` (parse) → `features` (encode + mask) → `policy` (score) → `ppo` (update) → `env` (wrap) → `train` (loop) is a clean split worth copying verbatim.

What to *not* copy: the terminal-only reward, the shared per-turn return, and the missing GAE.
