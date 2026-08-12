# CLAUDE.md

Kaggle **Kaggriculture** competition agent. Two-player farming sim; reward is final bank balance.

## Read first

- `PLAN2.md` — **current plan. Start here.** Written after the first external opponent beat our
  champion 0-40 and invalidated v1's sequencing. Carries the measured causal chain, the refuted v1
  claims, and a roadmap where every phase has a kill criterion fixed before the work starts.
- `PLAN.md` — v1. Superseded for strategy and sequencing; its environment facts and market
  economics remain valid except where marked [REFUTED].
- `TASKS2.md` — **current task list.** Executable breakdown of `PLAN2.md`: per-task build detail,
  verification method, binary done-when, and a kill criterion fixed before the work starts. Check
  the task ID before starting.
- `TASKS.md` — v1. Kept for the completed simulator/arena/search work, which still stands.
- `docs/README.md`, `docs/AGENTS.md` — the competition's own rules. Verified accurate against source.
- `reference/orbit_war/OVERVIEW.md` — analysis of a past competition's PPO agent, kept as an
  architectural reference. Its per-unit-decision + masked-candidate-scoring pattern is what we borrow.

## Environment

**Use `/opt/miniconda3/bin/python`.** `/usr/bin/python3` has no `kaggle_environments`.

Env source (the ground truth for all mechanics):
`/opt/miniconda3/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`

A full 720-step episode runs in ~0.85 s single-core (~860 steps/s), so large-scale simulation and
parameter search are cheap. Prefer measuring over reasoning about mechanics.

```bash
/opt/miniconda3/bin/python -c "
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 7}, debug=True)
env.run(['main.py', 'starter'])
print([(i, s['reward'], s['status']) for i, s in enumerate(env.steps[-1])])
"
```

Built-in opponents: `"pass"`, `"random"`, `"starter"`. Baselines: `starter` = $3,496, `random` = $0.

## Working notes

- **`make verify` is the parity gate.** It runs the whole suite under `coverage.py` against the
  reference `kaggriculture.py` and fails if any simulation line is unexecuted or any state
  diverges. Run it after *any* change to `kagsim/`.
- kagsim supports every documented config knob, including `marketParams`. Do not assume defaults —
  Kaggle can change the settings, and training across varied settings is deliberate.
- **Verify against the surface the runner actually uses, not the one that is easy to reach.**
  Two shipped defects, same root cause, both hidden behind green tests (E21):
  - `obs["step"]` **does** reach seat 1, correct on all 719 turns. The old rule here said it did
    not, because the test read `env.state[1].observation` — the *stored replay state*, which has
    shared fields stripped. Agents are handed `__get_shared_state(position)` (`core.py:754-767`)
    instead, and that carries `step` to both seats. kagsim had been suppressing `step` for player 1
    to "reproduce" the omission, so it genuinely diverged. Use `tests/.../delivered(env, p)`.
  - Kaggle never imports `main.py`. It `exec`s the source into an **empty globals dict** and calls
    **the last module-level callable** (`agent.py:47-63`). So `__file__` does not exist, and the
    smoke test's `import main` could not see it. The submission raised `NameError`, never loaded,
    and scored the $3,000 starting bank — measured 0-40 against a real opponent.
  `make submission` now goes through `env.run(["main.py", ...])` in both seats and fails if money
  is not above the starting bank.
- Submission must be `main.py` at repo root exposing `agent(obs)`.
- Scratch files, benchmarks, and one-off probes go in a temp dir, not the repo.
- When a mechanic matters, verify it against `kaggriculture.py` line numbers and cite them —
  several rules (watering every *other* day suffices, unconditional animal fertilizer, atomic
  PLANT validation) are easy to get wrong from prose alone.
- **The deliverable is a learned model.** The scripted engine is scaffolding on the critical path
  to it — training opponent, BC expert, macro-action-space definition, fallback submission — not an
  alternative to it. See `PLAN.md` §0 and §3.
- The farm half is analytically solvable; hand-code it. The market half is an adversarial timing
  game with public supply and private inventory (`PLAN.md` §2.5) — that is where the model goes.
- `PLAN.md` §2.6 summarizes all six published Orbit Wars top-10 writeups. Non-obvious consensus:
  **rewrite the env** (all six did), **terminal reward only — shaping measurably hurt**, **semantic
  action spaces beat raw ones**, **never train against a live copy of yourself**, and **2.5–9M
  params on one consumer GPU was enough for places 2/3/6/7.** Check it before proposing a design.
- Accept or reject changes by **local arena tournament winrate** (`make arena`), never by whether
  the change looks sound. A bundle of sensible-looking fixes scored 39% against its own baseline
  for 7th place.
- **Name the opponent explicitly; never inherit it from a default.** Twice now a dataclass default
  has silently defined an experiment: `buy_land=True` made every config buy land (E6), and
  `Params()` being wheat-based made CEM optimize against a weak agent while reporting a +$34k
  improvement worth $207 (E8).
- **A search result is a hypothesis** until it wins in the arena on seeds the search never saw.
- **Nothing below ~80 games is believable since 1.32.6.** Shop draws vary per game (E33), and three
  separate results this session looked real at 16-48 games and vanished at 80+: the equal-land gap
  (E37 -> E41), optimal assignment (E39 -> E40), and the scaling config (E42). A promising number
  *is* the signal to re-run on fresh seeds — before building on it or writing it into a plan.
- **Every conclusion is scoped to the system that produced it, and expires when that system
  changes.** This has now happened four times: land was correct about our engine and false about
  the game (E26); melon was correct under the `elif` bug that killed every other crop (E24);
  steps-per-action was confounded by task density (E29); and the whole land/servicing chain was
  rewritten by Kaggle's 1.32.6 demand cut (E34) — a claim that sat in `PLAN2.md` as prose for one
  hour before measurement refuted it. When anything changes underneath a result — engine, rules,
  opponent — **re-run it rather than reasoning about whether it still holds.** Ten minutes of
  measurement has beaten an hour of argument every time.
- **Promote only through `make promote`** (D19). Five straight promotions were wrong because a
  3-8pp difference was read off a 24-64 game sample that can only resolve 12pp+. The gate needs
  500 games vs the incumbent, a clean gauntlet over every registered agent, and a neighbourhood
  sweep — CEM's optimum is not necessarily a *local* optimum. `make audit-champion` re-checks the
  sitting champion at any time, and its first run rejected it.
- **Held-out seeds do not detect a single-opponent optimum.** The T2.1 champion was correctly
  seed-validated and still lost **0/80** to a naive dumper it had never faced (E10). Vary the
  *opponent*, not just the seeds: search against a pool, and keep the exploiters in the arena.
- **`BUY_LAND` is the mechanic that decides the matchup (E26).** The only external opponent we have
  beats our champion 24/0 with it and loses 24/0 without it, and its expanded production also cuts
  *our* revenue by $14k through the shared market. The four experiments that rejected land (E1, E6,
  E14, E20) were all correct about our engine -- which plants 76 tiles and loses 65 to thirst -- and
  all wrong as claims about the game. Land is gated on unit throughput: 28% productive turns for us
  against 42% for them.
- **Rank a product by how many shops demand it, never by its price curve.** Seasonal capacity is
  `one-shot cap + drain/day x days` and the drain term dominates (D17). Melon: zero shops, cannot
  regenerate. Milk: three. Judging by the price-curve integral ranked the markets almost inversely
  and cost a 2.4x improvement — it is the single most expensive mistake in this project.
- **Never evaluate against a weak fixed opponent.** Measured: `melon-wheat` earns 7% *more* than
  `melon` against `starter`, and then loses to it **32/32** head-to-head, its earnings collapsing
  60%. They compete for the same melon market. Rankings taken against `starter` are wrong
  (`docs/experiments.md` E5). Mean money is only comparable within a fixed opponent field —
  **skill and pairwise winrate are the ranking**.
