# Kaggriculture — Plan v2

Supersedes `PLAN.md` for strategy and sequencing. v1 is kept as the record of what was believed
and why; its environment facts and market notes remain valid except where marked [REFUTED].

**Why v2 exists.** v1 was written before we had any opponent we did not write ourselves. The first
external agent (`reference/kaggriculture/1/submission.py`, "boatlee", a scoring leaderboard entry)
beat our champion **0–40** and invalidated the sequencing of the whole plan. v1's roadmap assumed
the scripted engine was near its ceiling and the remaining work was a model for the market game.
Both halves were wrong.

---

## 1. The standard of evidence

Unchanged from v1 in principle, tightened in practice, because v1's discipline still let five wrong
champions through and let four separate conclusions survive that were true of our engine and false
about the game.

Every claim carries a tag: **[MEASURED Ex]** — an experiment number in `docs/experiments.md`.
**[COMPUTED]** — derived from cited environment source lines. **[ASSUMED]** — believed, untested,
and therefore *a task, not a fact*. **[REFUTED Ex]** — was believed, is now disproven.

Three rules earned the hard way this session:

1. **Measure the surface the runner uses.** Three separate defects came from measuring something
   adjacent to it: the submission that scored $3,000 because Kaggle never imports `main.py`; the
   `step` field we believed seat 1 never received; the arena that ran 718 turns where the real
   environment runs 719. [MEASURED E21, E27]
2. **A conclusion is scoped to the system that produced it.** "Land loses" was measured four times
   and was correct every time *about our engine*, and wrong as a claim about the game.
   [REFUTED E26]
3. **A plausible fix is a hypothesis with a cost.** Five principled corrections were implemented
   this session — water policy, global assignment, in-flight fetch accounting, fertilizer, the
   combined strategy — and **every one measured worse**. [MEASURED E23, E24, E29]

**Every phase below states its kill criterion before the work starts.** If a phase hits its
criterion, it is abandoned and recorded, not patched.

---

## 1.5 Rule change, 1.32.6 — read this before trusting any number below

Kaggle changed the environment mid-project [MEASURED E33]. Town-centre demand was cut ~4.7x, shop
unlocks now draw **with replacement**, and shed operations moved ahead of the LOCKED guard. kagsim
was updated and re-verified (0 divergences, bit-exact against the third-party agent); the drift
guard built one session earlier fired on all three assertions and forced the sequence.

**What survives — tested, not assumed [MEASURED E34].** An earlier draft of this section claimed
the whole §2 chain survived "because production mechanics did not change." That was refuted within
the hour:

* **Servicing capacity holds.** E24/E29 are about converting area into harvests. Untouched.
* **"Land is necessary *and sufficient*" is dead.** Stripping `BUY_LAND` used to flip the match
  100%; it now leaves us losing 3 games in 4. Land is still worth $69,475 to them — it is no longer
  the whole story.
* **Product mix is now an independent deficit of comparable size.** Stripped of land *and*
  fertilizer they still out-earn us; the same crippled agent lost to us by $23,734 under 1.32.4.
  A ~$29,000 swing that cannot be attributed to either mechanic.

**What does not:** every conclusion about *crop and animal mix*. Demand used to be a constant —
all 8 shops unlocked in every game — so the best mix could be baked into a parameter vector. It is
now a per-game random variable, and it is **fully observable at runtime** in
`obs["town"]["unlocked_shops"]`.

| product | mean shop demand | P(zero shops) |
|---|---:|---:|
| WHEAT | 5.0 | **0%** |
| STRAWBERRY | 3.7 | **0%** |
| CARROT / MILK / EGG / TOMATO | 2.0–3.2 | 2–12% |
| **WOOL** | 2.0 | **36%** |
| MELON | 0.0 | 100% |

Our champion runs 8 sheep for wool and a melon-led crop mix. Measured cost of the change: our money
fell **54%** against boatlee's **25%**, widening the gap from 2.2x to 3.5x.

This creates a new opportunity that did not exist before — **P1.5-A below** — and weakens P4, since
an open-loop plan cannot condition on a draw it never sees.

---

## 2. Where we actually are

### The result that defines the project

All figures below are 1.32.6 [E33/E34]. The 1.32.4 numbers this section first carried
($69,804 vs $152,120) are superseded — the rule change cost us 54% and them 25%.

| | ours | boatlee |
|---|---:|---:|
| head-to-head, 32 games both seats | **0 wins** | 32 |
| money | $26,464 | $122,028 |
| **at equal land** (their `BUY_LAND` stripped) | **$32,208** | **$52,926** |

[MEASURED E34, E37]

### The causal chain, each link measured

1. **Land is a large part of the matchup, no longer all of it.** Under 1.32.4, stripping
   `BUY_LAND` flipped the result 24/0 to us [E26]. Under 1.32.6 it moves us only 0% -> 25%, and is
   worth $69,475 to them. Stripping fertilizer instead still loses 0/32. **[REVISED E34]**
1b. **Product mix matters only at scale, and is not independent** [REVISED E41]. At 15 tiles it is
   worth little — which is why E35's adaptive-mix experiment found nothing. With land bought and
   ~59 crops it is worth **$14,426 -> $32,665**, recovering most of the land penalty. Mix and scale
   interact; neither alone pays.
1c. **Our marginal crop is negative beyond ~15 tiles** [MEASURED E41]. Capping planting recovers
   money monotonically ($7,566 at 77 crops -> $32,541 at 15), while boatlee farms 63 profitably.
   With the mix fixed, land roughly breaks even — it no longer loses, but it does not yet win.
2. **Land pays them twice.** Their extra supply floods the shared market: our own revenue rises
   $69,804 → $83,756 when they stop buying it. This term is invisible to any single-player
   estimate, which is why four experiments missed it. [MEASURED E26]
3. **Land is worth having because they can farm it.** 63 crop tiles against our 10.
   [MEASURED E23]
4. **We cannot service that area.** Given land our engine plants 76 tiles, loses **65 plants to
   thirst**, leaves 20–32 ripe and unharvested, and earns $60k instead of $107k. [MEASURED E24]
5. **The waste is hauling.** Crop work 416 turns vs their 1,671; animal work identical (889 vs
   897); **hauling 698 (34.8%) vs 192 (6.9%)**. 1,359 wheat collected to service 290 FEED ops.
   [MEASURED E29]
5b. **~~At equal land the gap is still servicing~~ [REFUTED E41].** E37 measured this on an
   8-seed block and it does not replicate: on 80 games at equal land we are at **45.0%
   [34.6, 55.9]** — statistical parity. Shop draws vary per game since 1.32.6 (E33), so small seed
   blocks are worthless for this comparison, a caution already written into the P1.5 gate and then
   not applied. **Land is the dominant factor**, which restores E26 and walks back E34.
6. **This is not a tuning problem.** CEM over 38 knobs — including five behaviours that did not
   exist when the champion was fitted — trained directly against boatlee for 14 generations, scored
   **0% in every generation**, returned a vector *worse* than the incumbent, and set
   `buy_land=False`, `fertilize=False`, `water_mode=elif`, `assign_mode=sequential`.
   [MEASURED E30]

**Conclusion: the executor is the binding constraint, and parameter search is exhausted as a route
to fixing it.**

### What boatlee is, and its weakness

A **719-step hardcoded action table** (zlib+base85), replayed by index, with four small closed-loop
patches: weed repair, a conditional cow→sheep swap, a wool release controller, and a fertilizer
relay. So the strategy that beats us was produced by **offline trajectory optimisation**, not by
policy engineering or parameter search — which is what we have been doing.

Its structural weakness: **it cannot adapt.** Same plan regardless of seed, opponent or market
state. Measured cost — it earns $193k against `starter` but $152k against our champion; supply
competition degrades it 21% and it cannot respond. [MEASURED E22, E26]

### The asset

| what runs | ms/episode | episodes/s |
|---|---:|---:|
| kagsim alone, no observation built | **2.0** | **497** |
| + building the Python observation | 15.6 | 64 |
| + boatlee's policy | 110.4 | 9 |
| + our engine's policy | 78.6 | 13 |
| CEM as actually run (8 workers) | — | ~38 |

[MEASURED E31, under 1.32.4 — the rule change does not touch the simulation loop, but these
figures have not been re-measured since.]

The simulator is 2ms and **bit-exact against the reference environment when driven by third-party
code** [MEASURED E27]. Everything above 2ms is Python. There is roughly **100× more search
throughput available** than the CEM run used.

Second asset, unexploited: **our submission uses 2.1ms of Kaggle's 1000ms per-turn budget.**

---

## 3. What v1 got wrong

Recorded because the plan should carry its own error history.

| v1 claim | status |
|---|---|
| Buying land loses money | **[REFUTED E26]** — true of our engine, false of the game |
| The scripted engine is near its ceiling; remaining headroom is the market game | **[REFUTED E22, E30]** — we are at 45% of a leaderboard agent's money |
| `obs["step"]` never reaches seat 1 | **[REFUTED E21]** — reaches both seats; kagsim was *made wrong* to match the false claim |
| Sale timing is near-trivial, so little is left to learn | **[REFUTED E23]** — crop *production* is 4× behind; selling was never the gap |
| Melon is the right crop core | **[REFUTED E24]** — zero shop demand (D17); CEM chose it because it is the only crop the engine's `elif` bug did not kill |
| The arena is the ranking authority | **narrowed** — it ran 718 turns [E27] and every agent in it but one was written by me (D16) |

---

## 4. Roadmap

Ordered by what unblocks what. Each phase: **hypothesis → test → kill criterion.**

### P0 — A forward model of our own farm  *(DONE — E36)*

`agent/forward.py`, 31 parity tests, **14/14 deliberate mutations caught**, **208,003
farm-steps/s** (104x the criterion), shipping in the submission bundle. A 3-day lookahead costs
**0.35 ms** of the 1000 ms turn budget — so P1 can afford hundreds of rollouts per turn, not the
handful originally assumed. Mutation testing found four tests that passed while proving nothing;
see E36 before writing any new parity test.

Original specification follows.

A pure-Python, dependency-free simulator of **our own farm only** — tiles, growth, watering,
harvest, animals, unit positions. Not the market, not the opponent. It must run on the Kaggle
runner, where `kagsim` does not exist.

- **Why it is possible:** farm dynamics are deterministic given actions. The only stochastic term
  is weed spawning (`weedSpawnChance`, default 0.005/empty tile/day, `kaggriculture.py:814`).
- **Test:** step it alongside kagsim for full episodes under fuzzed actions and assert the farm
  state matches exactly, weeds excluded. Same differential harness as `tests/test_parity.py`.
- **Kill criterion:** ≥2,000 farm-steps/second in pure Python, or P1 dies with it.
  **[MEASURED E32] — PASSED by 286x.** A faithful skeleton of the hot loop runs at 572k–836k
  steps/s depending on farm size; a 3-day lookahead costs ~0.13 ms of the 1000 ms turn budget.
  Even a real model 10x heavier clears the bar by ~28x.
- **The risk is therefore correctness, not speed.** The parity test against kagsim is the work:
  step both under fuzzed actions, assert farm state matches exactly, weeds excluded.

### P1 — Rollout-based assignment  *(the main line)*

Replace the hand-written assignment rules with short-horizon search: propose a few candidate
assignments, roll each forward with P0, keep the best by a simple value estimate.

- **Why this and not more rules:** every local rule correction attempted has cost money
  [E23, E24, E29]. A rollout does not need rules; it needs a forward model and a scoring function.
- **Why it should help:** the failure is measurable and specific — 65 plants lost to thirst,
  20–32 tiles ripe and unharvested [E24]. A 2–3 day rollout sees a plant about to die and a crop
  about to rot; the greedy rule cannot.
- **Precedent:** Orbit Wars 6th place gained +30–40 leaderboard points from a greedy 2-step rollout
  (`PLAN.md` §2.6).
- **Test:** `tools/routing_bench.py` — 24 games vs `boatlee` in ~5s. Primary metric **winrate vs
  boatlee**, then money. Diagnostics: plants lost to weeds, ripe-unharvested backlog, hauling share.
- **Kill criterion:** if it cannot service **≥40 crop tiles with <10 plants lost per season**, it
  has not fixed the thing it was built to fix — abandon regardless of money. Their figures: 63
  tiles, 5 weeds.
- **Do not** optimise steps-per-productive-action directly: it is confounded by task density, and
  our land configuration scores *better* on it (1.62 vs 1.84) while earning far less. [E29]

### P2 — Land, re-tested properly  *(gated on P1)*

Only meaningful once P1 passes its kill criterion.

- **Hypothesis:** with servicing fixed, `buy_land=True` becomes positive, worth up to the
  +$106,397 it is worth to them. **[ASSUMED]** — their number is an upper bound for their
  execution, not a promise for ours.
- **Test:** re-run the E20 land variants, plus a fresh CEM over the (now different) engine, gated
  on winrate vs `boatlee`.
- **Kill criterion:** if land still loses after P1, the diagnosis in §2 is wrong and this plan
  needs revision before more work — do not patch onward.

### P3 — Learned policy  *(the v1 deliverable, now with an expert)*

Unchanged in design from `PLAN.md` §3 — entity-token transformer, semantic action space, terminal
reward, opponent pool. Two things are new: **boatlee is a behaviour-cloning target that verifiably
scores on the leaderboard**, and P0 gives a differentiable-free fast rollout for inference search.

- **Gate (from D20, still live):** do not start until V1 returns a placement. Local evidence has
  been wrong about the shape of this game four times; training a model against a misunderstood game
  is the most expensive available mistake.

### P4 — Trajectory optimisation  *(the alternative line, if P1 fails)*

Do what boatlee did: optimise an action sequence offline rather than write a policy.

- **Why it is credible:** it is what the agent beating us actually is, and our simulator is 2ms per
  episode with no observation built [E31] — ~100× the search throughput CEM used.
- **Why it is not first:** an open-loop plan cannot adapt (their measured weakness), the action
  space is enormous so it needs a structured parameterisation rather than raw ops, and seed
  overfitting is severe. **[ASSUMED]** that a structured plan generalises across seeds — untested.
- **Cheap probe before committing:** take boatlee's own plan and re-optimise it in kagsim. If their
  trajectory cannot be improved with 100× their likely search budget, the route is weaker than it
  looks.

### V1 — Submit to the leaderboard  *(blocked on the user, unchanged, and now more important)*

The only non-self-referential measurement. `boatlee` is one opponent; it tells us a great deal more
than the 78 agents I wrote, but it cannot tell us whether "service a large crop area" is *the*
winning shape or merely *a* winning shape.

Requires the user to join the competition in a browser. The submission bundle is built and verified
through Kaggle's real loader [E21].

---

## 5. Metrics

**Ranking authority:** winrate vs `boatlee`, then the full gauntlet through `make promote`
(500 games vs incumbent, clean gauntlet, neighbourhood sweep — D19). Never money vs `starter`.

**Diagnostics that actually track the constraint:**

Measured at **equal land** [E37], because the 63-vs-10 tile comparison is land-confounded and
flatters the diagnosis. These are the part of the gap P1 can actually move:

**These targets came from E37 and are withdrawn** — that comparison does not replicate (E41). At
equal land we are at parity, so there is no equal-land servicing gap to close.

The measurement that matters now is at **scale**, because that is where we fail:

| metric | ours (land, best mix) | boatlee | note |
|---|---:|---:|---|
| crop tiles held | 59 | 63 | comparable |
| money | **$32,665** | **$130,000** | 4x gap at comparable scale |
| harvests/season | ~137 | 390 | the leading candidate |

**Never read any of these off fewer than ~30 seeds** (E41): an 8-seed block produced a 64% gap
where 80 games show parity.

**Deliberately not a target:** steps per productive action. Confounded by density [E29].

---

## 6. Open risks

1. **One opponent.** Every conclusion in §2 rests on a single external agent. It is leaderboard-
   validated, which makes it far better than self-play, but a field of one cannot show whether its
   strategy is dominant or merely good. V1 is the fix and it is blocked.
2. **Overfitting to boatlee.** It is a *fixed script*, so anything we tune against it can exploit
   its exact plan without generalising. Keep the engine-agent pool and the exploiters in every
   evaluation; never gate solely on it.
3. **Environment drift.** kagsim mirrors one pinned version. `tests/test_env_version.py` hashes the
   source and fails on any change (V6, done) — if it fires, parity must be re-verified before any
   result is trusted.
4. **P0 may be too slow in pure Python**, which kills P1. Tested first, cheaply, by design.
5. **The competition deadline is still unknown.** Raised three times, never established.
