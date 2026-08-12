# Kaggriculture — Task Breakdown v2

Executable breakdown of `PLAN2.md`. Every task states **what to build**, **how to verify it**, and
a binary **done-when**. "It runs" is not done; done means the stated verification passed.

`TASKS.md` (v1) is kept for the completed simulator/arena/search work, which still stands. This
file covers everything from the v2 roadmap onward.

## Status

| Task | State |
|---|---|
| P0.1–P0.4 farm forward model | **done (E36)** — 31 parity tests, 14/14 mutations caught, 208k steps/s, ships in the bundle |
| P1.1 value function | **done (E38)** — 75% pairwise ranking accuracy; the pre-set gate was mis-specified and is recorded as failed |
| P1.2 assignment | **built, NOT promoted (E39, E40)** — solves assignment exactly; +4-11% money vs boatlee but a resolved winrate *loss* in the mirror, so the gate refused. Available as `assign_mode="optimal"`, default off |
| ~~P1.3 rollout over assignment candidates~~ | **superseded (E39)** — the search targeted an exactly-solvable problem |
| **P1.5 gate** | **next** — `assign_mode=optimal` through `make promote`, then find the rest of the servicing gap |
| ~~P1.5-A shop-adaptive mix~~ | **killed (E35)** — lost on its own kill subset; mix is downstream of servicing, not independent of it. Knobs kept off by default; retest after P1 |
| P2 land re-test | blocked on P1 — and its +$106,397 premise is now $69,475 (E34), still large |
| P3 learned model | gated on V1 (D20) |
| P4 trajectory-optimisation probe | fallback if P1 is killed |
| **V1 submit to leaderboard** | **blocked on you** — the largest single unknown |
| H1 competition deadline | **unknown, raised 4x** |
| H4 re-baseline champion under 1.32.6 | **open** — every mix conclusion in E1-E32 is void (E33) |
| H2 v1 claim audit (was V7) | open |

Standing gates, unchanged: `make verify` after any `kagsim/` change; `make promote` before any
champion install (D19); never rank on money vs `starter`.

---

## P0 — Farm forward model  *(DONE — E36)*

`agent/forward.py` + `tests/test_forward_parity.py`. 31 parity tests, **14/14 mutations caught**,
**208,003 farm-steps/s**, ships in the bundle, `make submission` green in both seats.

**Read E36 before writing another parity test.** Mutation testing found four tests that passed
while proving nothing — a fuzz that never fertilized, a scenario that killed its own plant before
the branch ran, a harness that never passed `shedCapacity`, and a guard made invisible by a `min()`
clamp. Run mutations against a scratchpad copy (never the repo file), and `cmp` after each `sed` so
a non-matching pattern reports NO-OP instead of a misleading MISS.

Original specification follows.

A pure-Python model of **our own farm**: tiles, crops, animals, unit positions, unit inventories,
shed. Not the market, not the opponent. It exists so P1 can roll forward inside Kaggle's turn
budget, where `kagsim` is not importable.

Speed is already settled — a skeleton runs at 572k–836k steps/s, 286x the kill criterion
[MEASURED E32]. **The risk is correctness.**

### P0.1 — `agent/forward.py`

**Build.** A `FarmModel` with:

```python
FarmModel.from_obs(farm: dict, priv: dict, day: int, hour: int) -> FarmModel
model.step(unit_actions: list[list]) -> None      # one turn, our farm only
model.clone() -> FarmModel                        # cheap; rollouts branch constantly
model.score(params) -> float                      # standing value, see P1.1
```

Mirror these, citing line numbers in comments as `CLAUDE.md` requires:

| mechanic | source |
|---|---|
| unit ops: MOVE/WATER/HARVEST/PLANT/FERTILIZE/DIG/FEED/CARE/COLLECT_FERTILIZER/PICKUP/DROP/PLACE/BUILD_* | `kaggriculture.py` `_apply_unit_action` |
| watering + one-shot yield window, `bonus = 2 if fertilized` | `:378-388` |
| fertilizer duration `day..day+2` | `:419-426` |
| daily plant refresh, death at 2 dry days, ongoing accrual | `:760-780` |
| daily animal refresh, escape at 2 unfed days | `:783-810` |
| shed capacity + **insertion-ordered** inventories | v1 parity notes — dict order is observable |

**Constraints.** stdlib only, plus `CROPS`/`ANIMALS` imported from `kaggle_environments`. No numpy,
no kagsim. Must survive `tools/build_submission.py`'s AST check.

**Done when:** P0.2 passes.

### P0.2 — Differential parity against kagsim

**Build.** `tests/test_forward_parity.py`, same shape as `tests/test_parity.py`: drive kagsim with
fuzzed unit actions, step `FarmModel` alongside, compare the full farm state every turn.

**Determinism.** The only stochastic term is weed spawning, and it can be removed *and* forced:

| config | behaviour | verified |
|---|---|---|
| `weedSpawnChance: 0.0` | no weeds ever — clean determinism | yes, 0 weeds/100 turns |
| `weedSpawnChance: 1.0` | every empty tile weeds — the weed branch, still deterministic | yes, 25 weeds |

Run the suite at **both**, so the weed path is covered rather than excused.

**Cover explicitly** (each of these was a real v1 bug, in the simulator or the engine):
- shed overflow with insertion-ordered inventories
- `int()` coercion of string quantities (`["PICKUP","WHEAT","5"]`)
- ops that must no-op: harvesting an empty tile, feeding a structure, planting on occupied ground
- animals escaping and structures surviving
- fertilizer applied outside its window (must change nothing)

**Done when:** 0 divergences over ≥200 fuzzed episodes at both weed settings, across ≥3 board/config
variants, and the branch-coverage audit shows every farm-related line of `_apply_unit_action` and
both daily refreshes was executed under comparison.

**Kill criterion:** none — if this cannot be made exact, P1 is unsafe and the plan reverts to P4.

### P0.3 — Speed on the real model

**Verify.** Re-run the E32 probe against the finished model.

**Done when:** ≥2,000 farm-steps/s at 75 tiles / 15 units. **Kill criterion:** below that, P1 dies
and P4 becomes the main line. (Skeleton measured 572k; a 10x-heavier real model still clears by 28x.)

### P0.4 — Submission safety

**Verify.** `make submission` — the AST import check plus the smoke test through Kaggle's real
loader in both seats (E21).

**Done when:** bundle builds, both seats DONE, money above the starting bank.

---

## P1 — Rollout-based assignment  *(the main line)*

Replace hand-written assignment rules with short-horizon search. Justification: every local rule
correction attempted has cost money [E23, E24, E29]; a rollout needs a forward model and a scoring
function, not rules.

### P1.1 — Value function

**Build.** `score(state)` — standing value of a farm: expected revenue from crops in the ground
(yield already accrued + what it will accrue if serviced), animals, and shed stock, minus nothing.
Keep it simple and explainable; the rollout supplies the intelligence.

**Verify — gate replaced after measurement [E38].**

The original gate was *Spearman ρ ≥ 0.6 between score at day 15 and final money, over ≥200
episodes*. It **failed** (0.584 at best). It was also the wrong test, and demonstrably so: a
rollout never compares two strategies fifteen days apart, it compares candidate assignments from
one common state a few days ahead. Measured both ways, the two metrics rank the `unripe` parameter
in **opposite directions** — tuning to the old gate would have chosen `unripe = 1.5`, which is
economically incoherent and the *worst* of six settings for ranking (66.7% vs 76.0%).

**Current gate:** pairwise ranking accuracy — same farm, same seed, two action sequences, score
both at +3 days; does the higher score end the season richer? **Wilson interval must exclude 50%**,
target ≥70%. A value function that cannot beat a coin flip at ranking nearby futures is useless to
a rollout; one that can is usable regardless of its cross-strategy forecasting.

**Status: PASSED** — 75.0% [65.5, 82.6] at `unripe=0.5`, n=96.

Changing a pre-set gate is the move this project's rules exist to prevent, so the original is
recorded as failed rather than quietly dropped, and the replacement was measured before adoption.

### P1.2 — Candidate assignments

**Build.** Generate K candidate unit→task assignments per turn: the current greedy one, plus
perturbations (swap two units' targets, reassign the idlest unit, prefer the largest cluster).

**Done when:** K is a `Params` knob, and K=1 reproduces today's champion **bit-identically** — the
same discipline that kept `water_mode`/`assign_mode` safe.

**Constraint on the candidate set [E38].** `score()` values goods at base price, which ignores that
the market cannot absorb them — MELON has no shop buyer in any game, WOOL none in 36% (E33). That
cancels out when candidates differ only in *which worker does which job*, and does not when they
differ in *what to plant or breed*. So: keep candidates to assignment permutations, or switch
`score()` to a demand-aware price table first and re-measure the pairwise gate.

### P1.3 — Rollout and selection  *(SUPERSEDED — E39)*

Not built. It was aimed at choosing among candidate assignments, and assignment turned out to be
exactly solvable at this size (66 us for 12 units x 30 tasks). `FarmModel` and `score()` are not
wasted — they remain the tools for decisions that are *not* exactly solvable — but they are off the
critical path for assignment.

Original specification follows.

### P1.3 — Rollout and selection

**Build.** For each candidate: clone the model, apply the assignment, roll forward H days holding a
simple continuation policy, score, keep the best.

**Budget [E36]:** a 3-day rollout costs **0.35 ms**, so ~2,800 fit in the 1000 ms turn budget and
~280 inside a 100 ms self-imposed cap. K and H can be far more ambitious than "a handful of
candidates" — start wide and cut back if P1.4 trips, rather than assuming scarcity.

**Verify.** `tools/routing_bench.py --games 12`, ~5s per configuration.

**Done when:** a sweep over (K, H) is recorded in `docs/experiments.md` with winrate vs `boatlee`.

### P1.4 — Turn budget guard

**Build.** A wall-clock check that degrades to K=1 when the turn is running long.

**Done when:** p99 turn time <100 ms of the 1000 ms budget, measured through the reference env in
both seats. (Today: 2.1 ms.)

### P1.5 — The gate

**Revised by [E37].** The original gate (≥40 tiles) mixed up what P1 can move with what P2
unlocks: tile *count* is gated on land, which P1 does not buy. At equal land we already hold 19
tiles to their 15 — the difference is what happens to them afterwards. So P1 is gated on
**servicing quality at the land we have**, and scale is P2's gate.

**Done when**, on one quadrant, measured against boatlee-with-land-stripped [E37]:

| metric | ours today | target | theirs |
|---|---:|---:|---:|
| crop work, share of unit-turns | 5.8% | **≥20%** | 25.2% |
| plants lost to weeds/season | 7 | **≤2** | 1 |
| hauling share | 9.5% | **<5%** | 2.9% |

Formerly, and still the P2 gate once land is in play: ≥40 crop tiles serviced with ≤12 plants lost.

Reference points re-measured under 1.32.6 [E34]: **boatlee 63 tiles, 10 weeds, $109k–$135k**;
**us 15 tiles, 7–9 weeds, $8k–$53k**. Their tile count and earnings survived the demand cut, so a
large serviced crop area still pays and this gate is still the right one. Their weed loss rose
5 → 10, hence the threshold moved to ≤12.

**Note the variance:** our money swung $7,965 → $53,256 across two adjacent seeds. Per-game shop
draws now dominate single-seed results, so **never read this gate off fewer than ~16 seeds**.

**Kill criterion — decided now, before the work:** if P1 cannot reach ≥20% crop work with ≤2
plants lost at equal land, it has failed at the thing it was built for. **Abandon it regardless of what it does to money**, record
the result, and move to P4. Do not patch it into a third routing heuristic.

---

## P1.5-A — Shop-adaptive product selection  *(KILLED — E35)*

**Status: dropped**, by the kill criterion fixed before the work started. Every adaptive variant
lost, and lost hardest on the zero-yarn-store subset where it was supposed to win most clearly:
$11k–$21k against the fixed mix's $32,636. Two corrections were tried (revenue-capacity weighting,
and ruling out the `ongoing`-crop water interaction); neither changed the verdict.

**Why, and the lesson:** selection does not matter while servicing is the constraint. We service
10–15 crop tiles; *which* crop sits on them is second-order when the first-order problem is that we
cannot occupy more. E34 promoted this task on the reasoning that the mix deficit was *independent*
of servicing — that reasoning was wrong. It is downstream of it.

**Retest after P1**, not before: the hypothesis is untested for a 40-tile farm, only disproven for
a 15-tile one.

**Kept from the work:** the `town_drain_per_day` correctness fix (it still used the deleted
`TOWN_CENTER_DEMAND_SCHEDULE`, overstating absorption 2–8x) and `tests/test_demand_model.py`.
`adaptive_mix` / `demand_exponent` / `animal_demand_floor` / `animal_min_target` remain as `Params`,
off by default.

### Original specification (retained for the retest)

**Why now.** Under 1.32.4 all 8 shops unlocked in every game, so "which products sell" was a
constant and could be baked into `crop_mix` / animal targets. Under 1.32.6 shops draw **with
replacement**, so it is a per-game random variable — and it is fully observable at runtime in
`obs["town"]["unlocked_shops"]`.

| product | mean shop demand | P(zero shops) |
|---|---:|---:|
| WHEAT | 5.0 | **0%** |
| STRAWBERRY | 3.7 | **0%** |
| CARROT / MILK / EGG / TOMATO | 2.0–3.2 | 2–12% |
| **WOOL** | 2.0 | **36%** |
| MELON | 0.0 | 100% |

The champion breeds **8 sheep for wool** and leads on **melon**. [E33]

### P1.5-A.1 — Runtime demand model

**Build.** In `agent/engine.py`, derive per-product drain from the observed shop list each day:
`demand[item] = sum(2 if len(products)==1 else 1 for each unlocked instance demanding item)`, plus
the flat town-centre term (1 per non-fertilizer product per `townCenterSellInterval`).

**Verify.** Unit-test against `SHOPS` for hand-built shop lists including duplicates and the
empty list. Assert it matches a brute-force count over `obs["town"]["unlocked_shops"]`.

**Done when:** exact on ≥100 random shop draws taken from kagsim.

### P1.5-A.2 — Wire demand into crop and animal choice

**Build.** Weight `crop_mix` and the animal targets by observed demand rather than constants.
Gate behind a `Params` flag (`adaptive_mix`), **default off**, so the champion stays
bit-identical — the discipline that kept `water_mode`, `assign_mode` and `fertilize` from
regressing anything.

**Uncertainty is real:** shops unlock every 3 days, so early decisions are made blind. The policy
must be able to revise — prefer short-cycle crops early, commit later. Sheep bought on day 4 for a
yarn store that never appears are dead capital.

**Verify.** `tools/routing_bench.py` vs `boatlee`, **split by shop draw**: report the
zero-yarn-store subset (~36% of seeds) separately from the rest.

**Done when:** a sweep is recorded in `docs/experiments.md` with winrate vs `boatlee` on both
subsets.

**Kill criterion — fixed now:** if adapting to the observed draw does not beat the fixed mix **on
the zero-yarn-store subset**, the effect is not there and the idea is dropped. That subset is where
it must show up most clearly; no signal there means no signal anywhere.

---

## P2 — Land, re-tested properly  *(gated on P1.5)*

**Hypothesis.** With servicing fixed, `buy_land=True` turns positive — worth up to the **+$69,475**
it is worth to them under 1.32.6 [E34] (it was +$106,397 under 1.32.4 [E26]; the demand cut reduced
it). **[ASSUMED]**: that is an upper bound for *their* execution, not a promise for ours.

**Test.** Re-run the E20 land variants and a fresh CEM over the new engine, gated on winrate vs
`boatlee`, then `make promote`.

**Done when:** land either wins through the full promotion gate, or is refuted again *with the
servicing constraint removed* — which would be a genuinely new result.

**Kill criterion:** if land still loses after P1 passes, the causal chain in `PLAN2.md` §2 is wrong.
Stop and revise the plan; do not continue patching.

**Note [E34]:** land is no longer *sufficient* to explain the matchup — stripping it moves us only
0% -> 25%, not to 100% as under 1.32.4. So P2 passing would no longer be expected to close the gap
on its own; P1.5-A is the other half.

---

## P3 — Learned policy  *(gated on V1, per D20)*

Design unchanged from `PLAN.md` §3. Two things are new: **boatlee is a behaviour-cloning target
that verifiably scores on the leaderboard**, and P0 gives a fast pure-Python rollout for
inference-time search.

**Do not start before V1 returns a placement.** Local evidence has been wrong about the shape of
this game four times; training against a misunderstood game is the most expensive mistake available.

---

## P4 — Trajectory optimisation  *(fallback, or parallel if P1 is killed)*

**P4.0 — Cheap probe first.** Take boatlee's own 719-step plan and re-optimise it in kagsim. An
open-loop plan needs no observation built, so rollouts run at ~500/s/core — ~100x the throughput
the CEM run used [E31].

**Done when:** we know whether their trajectory can be improved at all. If it cannot be improved
with 100x their likely budget, the route is weaker than it looks and this is cheap to learn.

**P4.1+** — only if the probe is positive: structured plan parameterisation (crop/route schedule
compiled to actions, not raw ops), seed-generalisation testing, closed-loop repair overlays.

**Risk, stated up front, and now larger [E33]:** an open-loop plan cannot adapt. It already lost
21% of its money to supply competition and could not respond [E22, E26]; since 1.32.6 it must also
commit to a product mix *before seeing which shops spawn*, when wool has no buyer in 36% of games
and demand varies per game. boatlee already carries a hand-written `YARN_STORE` check, which
suggests its author saw this coming. Anything built this way inherits the problem.

---

## V1 — Submit to the leaderboard  *(blocked on you)*

The only non-self-referential measurement. `boatlee` is worth far more than the 78 agents I wrote,
but a field of one cannot show whether its strategy is dominant or merely good.

**Needs:** you to join the competition at `kaggle.com/competitions/kaggriculture` in a browser, and
`pip install kaggle`. The bundle builds and is verified through Kaggle's real loader (E21). I will
not submit unattended — it is outward-facing and uses your account.

---

## Hygiene

**H1 — Competition deadline.** Still unknown, raised four times. Everything about how much to
invest in P3 vs P4 depends on it.

**H2 — v1 claim audit** (was V7). Six claims already refuted (`PLAN2.md` §3). Walk the rest of
`PLAN.md` and tag each [MEASURED] / [COMPUTED] / [ASSUMED] / [REFUTED].

**H4 — Re-baseline under 1.32.6.** Every conclusion about crop and animal mix in E1–E32 was
measured under a demand model that no longer exists [E33]. The servicing/land evidence (E23–E30)
survives, revised by [E34]. Re-run the champion's neighbourhood and `make audit-champion` before
treating any mix number as current. **Do not run a fresh CEM before P1.5-A** — searching a fixed
mix over a game whose demand is now random re-finds the same trap E30 documented.

**H3 — Environment drift.** `tests/test_env_version.py` hashes the env source and fails on change.
If it fires, re-verify parity **before trusting any result** — kagsim mirrors one pinned version.
