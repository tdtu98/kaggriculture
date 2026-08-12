# Decision Record

Pinned choices for the Kaggriculture agent. Each entry states the decision, the reasoning, what was
rejected, and what would change our mind. Do not relitigate these without new evidence.

Referenced from `TASKS.md` (T0.0) and `PLAN.md`.

---

## D1 — The deliverable is a learned model, not a scripted agent

**Decision.** Target a trained policy. The scripted engine (Phase 1) is scaffolding on the critical
path to it, not an alternative.

**Reasoning.** The Orbit Wars top-10 were all learned models. A hand-tuned script is a ceiling. But
the script is still required, for four independent reasons: it is the opponent to train against
(self-play from random will not bootstrap over a 720-step horizon with 30-day payoffs), the expert
for behavior cloning, the definition of the macro-action space, and the fallback submission.

**Changes our mind.** If the game turns out to be close to a solved optimization — a tuned script
sitting within a few percent of the analytic ceiling and beating the front-runner exploit — then
the model's remaining headroom is only the market game, and Phase 3 should shrink to the market
head.

**Status: this trigger is now live and unresolved.** The scripted engine passed its $20k Phase 1
target by a wide margin (champion: 100% over the field, suppressing the previous champion to
$3,190 against a $3,000 start), and every lever that moved results was a *market* lever — the
supply forecast (E7) and the reserve/forecast knobs CEM tuned (E9) — while routing turned out not
to matter at all (E6). **T2.2 and T2.3 exist to resolve it**: whether an exploiter beats the
champion, and what share of the contested ceiling the champion already realizes. Do not scope
Phase 3 until both have answers.

---

## D2 — Simulator language: Rust + PyO3, built with maturin

**Decision.** Rewrite the environment as a Rust crate (`kagsim/`) exposed to Python via PyO3, built
with maturin.

**Reasoning.**

1. **The logic is branchy, not arithmetic.** Per-tile type dispatch, per-unit action dispatch, and
   the market lockstep loop are control-flow heavy. This is the shape that vectorizes worst.
2. **NumPy batching would be a redesign, not a port,** and would still be bounded by the branchy
   paths it cannot express. A compiled port is closer to a transcription and gets 100×+.
3. **Rust releases the GIL,** so `rayon` gives true multicore rollout across the 8 available cores.
4. **It matches the reference class.** Of the six published Orbit Wars top-10 solutions, all six
   rewrote the environment: 1st Rust, 2nd Rust→C, 5th C++, 6th C++, 3rd and 7th JAX. Both JAX users
   reported problems; 3rd wrote: *"In the future I would probably do this part in Rust or C++ …
   compilation on the Kaggle servers just seems risky."*

**Rejected.**

- **Numba** — fragile on the dict/object state the reference uses; would require the same
  array-of-structs redesign first, at which point Rust is barely more work and much faster.
- **Cython** — comparable effort, weaker tooling and ergonomics.
- **JAX** — static shapes fight a variable number of hired hands; see 3rd place's compile-time
  problems and their own retraction.
- **Staying on the Python env** — 860 steps/s vs. the 15k–40k the field trained at. Not viable.

**Changes our mind.** If Rust setup blocks for more than a day, fall back to the array-of-structs
redesign in Python and port later. The state design in T0.2 and the parity harness in T0.4 are
language-agnostic — **the harness is the durable asset, not the language.**

---

## D3 — Simulator scope

**Decision.** `kagsim` models both farms, the shared market, and the town. It does **not** implement
the kaggle-environments wrapper, replay JSON, the HTML renderer, or the observation dicts.

**Reasoning.** Those are submission-time and debugging concerns, served by the reference env. The
simulator exists only to produce training transitions fast.

---

## D4 — Verification is by parity, in three layers

**Decision.** Correctness of `kagsim` is established by step-by-step canonical-state equality
against the reference Python env, split into: (A) deterministic core with `weedSpawnChance: 0` and
shop unlocks disabled, (B) the RNG in isolation against a golden file, (C) the full default config.

**Reasoning.** Layering localizes failures. A single full-config parity run that diverges at step
300 tells you nothing about *which* subsystem is wrong; layer A isolates all deterministic rules
from the RNG, and layer B isolates the RNG from the rules.

5th place, on their C++ port: *"a faster environment is only useful if it produces exactly the same
game transitions and model inputs."*

---

## D4b — Parity is gated on reference *branch coverage*, not on episode count

**Decision.** The parity claim is measured by running `coverage.py` over the reference
`kaggriculture.py` while its state is compared against kagsim step-by-step. `make verify` fails if
any simulation-logic line is uncovered or any state diverges.

**Reasoning.** Episode counts and hand-written rule lists both measure the wrong thing — they can
only report on behaviours someone thought to name. Branch coverage of the reference is objective:
**a covered line was differentially tested; an uncovered line is unverified**, regardless of how
many million steps ran.

This immediately paid for itself. With 14 full episodes passing and a hand-written 34-rule
checklist reporting full coverage, the audit still found that:

- the **terminal step was never executed** — the loop stopped one call short of
  `episodeSteps - 2`, so the DONE/reward path, which produces the score we train on, was untested;
- `FEED` on an already-fed animal and `PLACE` into a full shed were never exercised;
- Python `int()` coercion of quantity arguments differed (see D4c).

**Excused lines** carry a written reason in `tools/coverage_audit.py` and are either display code
(renderer), bundled agents (tested separately), or provably unreachable through the public API —
for example PLANT's own `crop not in CROPS` guard, which the atomic-PLANT precheck shadows.

---

## D4c — kagsim reproduces the reference's *failures*, not just its successes

**Decision.** Where the reference raises, kagsim raises; where it coerces, kagsim coerces
identically.

**Reasoning.** `_apply_unit_action` coerces quantities with a bare `int()` (`:347`, `:465`) and so
throws on a bad value, while `_parse_order` (`:619`) catches it. A strict integer extract in Rust
diverged three ways: `["BUY_SEED","WHEAT","5"]` bought 5 seeds in the reference and 0 in kagsim;
`3.9` truncated to 3 versus being dropped; and a non-numeric quantity crashed the episode versus
being silently treated as 1. None of these are reachable from our own agents, but a training run
that silently disagrees with the submission environment is exactly the failure this project cannot
afford.

**`marketParams` is now implemented** (was previously refused). Kaggle can change the environment
configuration between now and scoring, and training across varied settings is worth having, so
kagsim honours every documented knob rather than assuming defaults. Three details had to be
replicated exactly:

- the merge is **sparse** — unspecified fields, unknown product names, and non-dict patches all
  fall through to the defaults;
- `market["params"]` is only stored when the value is **truthy**, so `{"BOGUS": {...}}` exposes a
  resolved table even though nothing matched, while `{}` is indistinguishable from omitting it;
- the resolved table lands in the **shared observation**, making it state an agent can read — so
  the parity harness compares it, and `Sim.observation` emits it under the same condition.

---

## D4d — `Sim.observation()` is compared against the reference directly

**Decision.** The observation surface is diffed field-by-field against
`env.state[p].observation` every step, for both seats, not merely inferred from closed-loop agent
behaviour.

**Reasoning.** The main harness compares a canonical state of my own design, so anything it does
not look at is unverified — and the observation is the exact surface a policy consumes. Adding the
direct comparison immediately found one difference:

**~~`step` is delivered only to player 0.~~ [REFUTED E21]** This was wrong, and acting on it
introduced the very divergence it claimed to prevent.

`step` is delivered to **both** seats, correct on all 719 turns. The error was comparing against
`env.state[1].observation` — the *stored replay state*, where shared fields really are stripped
for seats above 0. That is not what an agent receives: the framework rebuilds the observation
per-agent in `Environment.__get_shared_state` (`core.py:754-767`) and passes that to `agent.act`,
and `step` survives the rebuild.

kagsim had it **right**, emitting `step` for both seats, and was changed to suppress it for
player 1 in order to "reproduce the omission". So a parity check that read the wrong object turned
a correct simulator into a diverging one, then certified the result. Restored in E21, and the test
now goes through a `delivered()` helper that can only see the runner's surface.

`remainingOverageTime` is excluded from the comparison: it is the framework's wall-clock budget,
not simulation state.

There are now **no known behavioural differences** between kagsim and the reference.

---

## D5 — Bit-exact CPython RNG is required

**Decision.** Reimplement CPython's MT19937 (`random()`, `getrandbits`, `_randbelow`, `choice`) in
Rust rather than approximating or precomputing the stochastic stream.

**Reasoning.** `_end_of_day` creates `random.Random((seed * 1_000_003) ^ day)` and then calls
`rng.random()` **once per empty unlocked tile**. Which tiles are empty depends on gameplay, so the
number of draws consumed varies per episode and the stream cannot be precomputed from `(seed, day)`
alone. Without bit-exact RNG, layer C parity is impossible.

**Note.** Training itself does not need bit-exact weeds (p = 0.005 is strategically minor). The RNG
exists to make the parity harness total, which is what validates everything else.

---

## D6 — Reward: terminal only, no shaping

**Decision.** Train on the game outcome (money margin vs. opponent). Do not shape with per-day
`Δmoney`.

**Reasoning.** This reverses an earlier recommendation. 7th place ablated it directly: "Dominance
reward shaping" scored **34.6%** against its own baseline over 512 games. Terminal-only reward was
unanimous across 1st, 3rd, 5th, 6th, and 7th.

**Escape hatch.** If a dense signal proves necessary to bootstrap, use 6th place's pattern instead —
train a small fast model *with* dense reward, then transfer into the real model via teacher-KL —
rather than shaping the main run.

---

## D7 — Action space: semantic, per unit

**Decision.** The policy selects `(target tile, intent)` per unit, where intent is a small semantic
set. The executor computes the path and the exact op sequence. **The model never emits `NORTH`.**

**Reasoning.** The clearest empirical lesson from the field. 1st place began with raw launch angles,
got "a far cry from competitive," and only became competitive after switching to target selection.
3rd swapped fixed ship fractions for four *intents* and it "increased learning speed by a lot." 2nd
place reached second with an action space of literally two options: no-op and all-in.

**Corollary to explore early.** Find the Kaggriculture equivalent of "no-op and all-in" — the two or
three decisions carrying most of the value — and consider making that the entire initial action
space.

---

## D8 — Model size target: 2–10M parameters

**Decision.** Design for ~5M parameters on a single GPU. Do not design for anything larger.

**Reasoning.** 6th place used 2.5M, 2nd used 4.3M, 3rd used 6.2M, 7th used 9M. Only 1st place was
large (200M), and he trained on four 8×B200 nodes from a lab cluster. 3rd, 5th, 6th, and 7th all
trained on one or two consumer GPUs. Asked whether someone with spare time and a 4090 could compete,
the 1st-place author answered: *"For sure! A number of top-10 competitors did so."*

---

## D9 — Opponent pool from day one

**Decision.** Never train against only a live copy of the current policy. The rollout opponent is
sampled from a pool: the tuned scripted engine plus checkpoints drawn across the whole run history.

**Reasoning.** 7th measured training against a live copy at **20.7%** winrate versus their baseline.
Adding a full-history pool turned their 4p model "into one of the better 4P models in the
competition." 1st place names omitting league play as his main regret.

---

## D10 — Acceptance is by arena winrate, never by inspection

**Decision.** Any change that could affect agent strength is accepted or rejected by a local arena
tournament with confidence intervals, not by whether the change looks correct.

**Reasoning.** 7th place ran ~200 experiments this way. A bundle of "small, sensible-looking changes
to the network and training code" — each of which they believed sound — scored **39%** against its
own baseline. Their conclusion, which we adopt: *always trust the tournament.*

**Operationally.** 512 games gives roughly ±4pp, so **do not act on a 52%**.

---

## D11 — Compute: develop locally, train remotely

**Decision.** Simulator, scripted agent, arena, and CEM search run on the local machine. Billion-step
PPO runs on a rented GPU.

**Reasoning.** The local machine is Apple Silicon arm64, 8 cores, 16 GB, **MPS only — no CUDA**. MPS
is adequate for shape-debugging and behavior cloning with tiny models, not for the reference-class
training budget. 3rd and 6th both rented from vast.ai; ~1 week on a single 4090/5090 is the
reference-class spend.

**Prerequisite.** Cloud checkpointing of weights, optimizer state, opponent pool, and win rates must
exist *before* the first long run — 3rd place's rented boxes crashed repeatedly and this saved them.


---

## D15 — Claims carry provenance; computed is not measured

**Decision.** Every strategic claim in `PLAN.md` is tagged **[MEASURED Ex]**, **[COMPUTED]**,
**[ASSUMED]** or **[REFUTED Ex]**. A claim with no tag is not a finding.

**Reasoning.** Reasoning-derived numbers have been wrong here repeatedly, and convincingly:

| claim | source | outcome |
|---|---|---|
| "strawberry is the worst crop, skip it" | price-curve integral | **refuted** — it exceeds that cap in play (E13); CEM independently chose a 0.30 weight |
| "buy all three quadrants ASAP" | $/tile-day arithmetic | **refuted** — not buying land doubled money (E1) |
| "labour is nearly free" | fib cost at 6 hands | **refuted past ~8 hands** — 12 bankrupts (E1) |
| "geese are catastrophic" | a run with a cash bug | **refuted** — +47%, then a further +38% once ungated (E1, E12) |
| "buy animals first, they compound" | payback ratio | **refuted** — 52.6% vs 100% (E13) |
| "milk and wool are traps; geese are the only animal worth owning" | one-shot price integral | **refuted** — cows beat the goose champion **48/48**; champion mean money $37k -> $66k (E15) |

Each was plausible, quantitative, and wrong. **[COMPUTED] describes the market; only
[MEASURED] describes the game.**

---

## D16 — The arena is circular until an external opponent exists

**Decision.** Treat every current ranking as provisional. The exploiter field is the champion with
knobs changed, so "no known exploit" means "none I thought to build". Two remedies, both scheduled:
submit (V1) and independently designed opponents written from a design brief rather than by
mutating champion parameters (V2).

**Reasoning.** E10 measured a two-line change beating a fully tuned champion **0/80** — a strategy
one step outside the search field. There is no reason to think the field's *edge* has been found,
only that its *interior* has been searched.

**Consequence.** No claim of the form "the champion is robust" is supportable today. It is
"unbeaten by six variants of itself".


---

## D17 — Market capacity is set by shop demand, not by the price curve

**Decision.** Rank any product by **how many town shops demand it**, then by drain rate. The price
curve tells you how fast a price falls; it does not tell you how much the season will absorb.

**Reasoning.** The town removes inventory continuously, so
`seasonal capacity ~= one-shot cap + drain/day x days`, and the second term dominates:

| product | shops | one-shot cap | seasonal | original verdict |
|---|---:|---:|---:|---|
| MILK | 3 | $6,181 | **~$52k** | "trap" |
| WOOL | 1 (x2) | $7,928 | **~$51k** | "trap" |
| STRAWBERRY | 4 | $3,809 | **~$47k** | "worst in game" |
| MELON | **0** | $26,485 | ~$44k, *unrenewable* | "best revenue density" |

**Melon is demanded by no shop at all.** Every anomaly that accumulated over the project follows
from that single fact: melon saturating at 82% of its cap, an agent extracting 99% of it and still
losing, and the entire melon-first strategy resting on the one market that cannot recover.

**What it cost.** Champion mean money went $37k -> $66k when livestock and the corrected ranking
were tested — a 2.4x improvement that was available the whole time and was blocked by a
plausible-looking integral.

**Changes our mind.** Nothing measured so far; this *is* the measurement. But the drain figures
assume default `townShopSellInterval` and `townCenterSellInterval`, so a non-default configuration
would shift the ranking. kagsim supports those knobs (D4c) and the ranking should be recomputed
rather than assumed if the competition changes them.


---

## D18 — Sell immediately: because cash compounds, not because the price will crash

**Decision.** Release every unit as soon as it exists — **except on products our own production
can flood**, which currently means wool.

> **Amended by E19.** The original decision pinned every reserve to zero. That was measured before
> sheep existed. With 8 sheep per side, wool is the one shop-demanded market two farms can
> outproduce, and the promoted champion carries a `0.35 x base` reserve on it — a variant with that
> reserve beat the reserve-free champion 59%. The reserve is a **floor against our own glut**, not
> market timing. Which products qualify depends on our herd and crop mix, so it must be
> re-measured when the mix changes rather than carried forward.

**Reasoning (for the products still sold immediately) — corrected.** E11 justified this as *"a reserve is a bet that prices recover, and the
opponent decides whether they do."* Measured (E16), that is true only for melon, the one product no
shop demands. For everything else the town drains faster than two players can supply, so inventory
sits **below** `I0` all season and prices *rise*.

Holding still loses, monotonically, in that rising market:

| reserve | winrate | mean $ |
|---|---|---:|
| none (sell now) | **95.1%** | 68,177 |
| 0.9x base | 79.9% | 65,015 |
| 1.1x base | 50.0% | 60,376 |
| 1.3x base | 25.0% | 38,637 |

The real mechanism: **cash compounds and inventory does not.** The shed caps at 100 items, unsold
stock scores zero, and money released now buys seeds and animals that yield every remaining day. A
10-20% price drift cannot compete with reinvestment.

**Why the distinction matters.** The original reasoning implied the policy was contingent on facing
an aggressive opponent. It is not — it holds against a passive one too, and for a reason that does
not depend on the opponent at all. Pinned by `tests/test_market_regime.py`.

**Changes our mind.** A configuration where the town drains far less (`townShopSellInterval` or
`townCenterSellInterval` raised) would move the game into glut, where prices fall and the original
melon-style logic applies to everything. kagsim supports those knobs; the regime should be
re-measured, not assumed, if the competition changes them.


---

## D19 — Champions are promoted by a gate, not by a search result

**Decision.** `search/champion.json` may only change via `tools/promote.py`, which refuses to
promote unless three stages pass on **fresh seeds**:

1. **Beat the incumbent** over 500 games, with the Wilson interval clear of 50%.
2. **Survive the full registry gauntlet** — screened at 24 games, every close call escalated to
   500 — with no resolved loss to any of the ~65 agents.
3. **Survive a neighbourhood sweep** — perturb the discrete knobs (herd sizes, hire count) and
   confirm no small variation beats it.

`make audit-champion` runs stages 2–3 against the sitting champion at any time.

**Reasoning — this is a fix for a repeated, measured failure.** Five consecutive promotions
installed an agent that a larger later measurement showed was not the best available:

| promotion | what a bigger sample found |
|---|---|
| E8 | +$34,207 held-out was worth **$207** — wrong opponent, inherited from a default |
| E10 | lost **0/80** to a naive dumper never in its search field |
| E12 | `goose_target = 0` chosen because another knob gated it off |
| E15 | an entire species (cows) was unimplemented and beat it 48/48 |
| E17 | CEM's herd of 8 cows + 4 sheep lost **64.4%** to 6 + 8 |

The arithmetic behind all five:

| games | ± at 50% | smallest edge resolvable |
|---|---|---|
| 24 | 18.6pp | 68.6% |
| 64 | 11.9pp | 61.9% |
| 500 | 4.4pp | 54.4% |

**Champions were promoted on 24–64 games, from differences of 3–8pp** — inside the noise of the
sample that produced them. Every "improvement" was a coin flip dressed as a result.

**Why a search result is not sufficient evidence.** CEM's held-out score has the same sample-size
problem *and* a second one: it only ever sees the pool it trained against. Both failure modes are
invisible from inside the search, and neither is fixed by more generations.

**Why stage 3 exists.** CEM optimises a 33-dimensional continuous vector and samples each dimension
independently, so it can converge to a point that a one-step change in a *discrete* knob beats
outright — measured at 64.4% for the herd mix (E17). A search optimum is not necessarily a local
optimum.

**Cost.** Screening is cheap; only close calls escalate. A full gate run is a few minutes, against
promotions that have each cost hours of work built on the wrong base.

**Changes our mind.** If stage 2 starts passing trivially for every candidate, the registry has
stopped being adversarial and needs new opponents (D16), not a weaker gate.


---

## D20 — Do not start Phase 3 (the model) until V1 has returned a leaderboard placement

**Decision.** The model is not begun on local evidence alone, however good that evidence looks.

**Why.** Two facts now point the same way. First, the scripted agent is at a structural ceiling
(E20: farm 25/25 full, herd at target, land dead on a fourth test, 18% idle) — so the *cheap* work
is nearly exhausted and Phase 3 looks like the natural next move. Second, per D16 every number here
was measured against 74 agents I wrote, and this project has already had one shared blind spot
(the one-shot market model) survive **nine consecutive confirming experiments**.

Those two facts together are the dangerous combination: strong local evidence that it is time to
train, and no way from the inside to tell whether the game we would be training against is the real
one. A model trained on a misunderstood game optimises the misunderstanding, at GPU cost, with the
same arena reporting success throughout.

**What unblocks it.** One leaderboard placement (V1). Near the top -> build as designed. Mid-field
-> find the structural gap first. Far down -> a core assumption is wrong; do not train yet.

**Cost of the delay.** Low. V1 is a browser action plus a submission, and the local loop
(CEM -> `make promote`) keeps running meanwhile.

**Status.** Live. Blocked on the user joining the competition. See `PLAN.md` §3.5.


---

## D21 — Do not submit to the leaderboard until we beat boatlee offline

**Decision (user, this session).** V1 is no longer "blocked on joining the competition". It is
gated on a local result: **beat `boatlee` head-to-head in the arena first.**

**What this changes.** D20 said the model was gated on a leaderboard placement, and V1 had been
carried as the critical path since PLAN2 was written. It is now downstream, not upstream. The
objective is unambiguous and entirely local: `boatlee` is the target, the arena is the measurement,
and no external signal is needed to make progress.

**What it does not change.** D16 still holds -- every agent in the registry except `boatlee` is
self-referential, so a win against the rest of the field means little. `boatlee` is the only
opponent whose result carries information.
