# Kaggriculture — Plan v4: Plan → Compile → Execute, with online re-planning

*Synthesises the Claude-session executor line, the kagsim/CEM/relay line (E1–E53), the decoded Boatlee table, and the env source (1.32.7).*

## 0. Objective and how we'll know

Beat Boatlee head-to-head **≥ 80% on both seats over ≥ 80 games on fresh seeds**, while not regressing against a pool of non-Boatlee agents (starter, our old executor, the kagsim champion, exploiters). Money vs `starter` is a diagnostic, never the target. Every claim: ≥ 80 games, both seats, fresh seed block, effect counter checked before the money is read.

## 1. The facts this plan is built on

- Boatlee = a 719-step open-loop table (100% replay). It wins on **per-unit efficiency, not scale**: 1.01 steps walked per useful action (ours 1.3–1.84), 37 strawberry plants at 7.7/plant (fertilize at ages 9 and 13 → all four yields doubled), cows cared daily (3 milk per cycle), 3 quadrants, up to 14 hands/day, plants die of old age not thirst.
- Every reactive engine we've built plateaus (30–75k solo, 0–43k contested) for the same measured reason: units decide turn-by-turn, so they walk too much and start plants they can't finish. Matching Boatlee's farm shape with a reactive executor still loses 50 plants to thirst (E44). Assignment optimisation moved it ~10% (E39). Parameter search over such an engine finds nothing (E30).
- Boatlee's own weakness is rigidity: it commits its whole herd by day 8 and its crop plan by day 12, before most shop draws are known (1.32.6: shops drawn with replacement; wool has no buyer 36% of games), and it cannot react to an opponent's flooding. Its schedule is public and readable — every one of its dumps is front-runnable (relay-sell: +$300, 90% vs Boatlee, mirror-margin only).
- Market: sell immediately, except products *we* flood; opponent tiles = their future supply.
- Compute: reference env ~2 s/game; kagsim 2 ms/episode (Mac); pure-Python farm forward model 208k steps/s. Turn budget 1 s + 60 s overage; we use ~2 ms.

## 2. The central bet: a plan compiler, not a policy

Boatlee's author found the right *structure* — decide the whole season offline, then just execute — and paid for it with zero adaptivity. Reactive engines have adaptivity and no structure. The unbuilt middle is:

**Plan (small, structured, searchable) → Compiler (turns it into per-unit, per-turn actions with a real routing solver) → Executor (replays the compiled day, repairs weeds) — and re-compile at every day boundary from the true state.**

### 2.1 The plan (genome), ~60–120 numbers
- Layout: which tiles are pasture (clustered on the shed-access side), which rows are strawberry / wheat / melon per quadrant, purchase day for NE / SW / (SE).
- Herd: number of cows, sheep (and geese as an option), purchase days.
- Crop schedule: strawberry cohorts (count, plant day), wheat cycles, melon count.
- Labour curve: hands per day (or "hire until marginal task value < wage", computed by the compiler).
- Policy constants: fertilize strawberry at ages 9 and 13 (fixed — it's arithmetic, not a knob), care every animal daily, harvest ongoing crops the day they tick, sell floors only for self-flooded goods, front-run lead (steps).
- Branch points (see 2.4): thresholds on observed shop counts that switch herd/crop sub-plans.

### 2.2 The compiler (the new component)
At the start of a day, given farm state + plan:
1. Enumerate the day's tasks with deadlines and values: waters due (survival: `consecutive_unwatered==1`; bonus-window waters for one-time crops), harvests (ripe now, or ticking tonight → harvest tomorrow), fertilize (tiles at age 9/13 today), plant (cohort due today, gated on seeds and *sustainable watering capacity*), feed/care/collect per animal, pickups (wheat/fertilizer at the shed at spawn — all four centre tiles are shed-adjacent since 1.32.6), sells.
2. Decide hands for the day: add a hand while the marginal task value it can service (≈ its route's value) exceeds `fib(n)`.
3. **Route**: assign tasks to units and order them — a small vehicle-routing problem (≤ 15 units, ≤ 100 tasks, 24 turns, Manhattan grid, all units start at the shed centre). Solve with cluster-first/route-second (k-means by tile position, role-pure units per E46: keepers stay in the pasture cluster, field hands in field rows), then 2-opt / insertion. Target: ≤ 1.1 steps per useful action, 0 survival waters missed. Verify with the forward model before committing; if the plan overcommits (a water would be missed), drop the lowest-value *plant* tasks first — never a survival water. This is the "sustaining cost" the kagsim docs identified as unmodelled.
4. Emit the 24-turn action script for each unit.

Cost: the routing is milliseconds; the forward-model check of a day is < 1 ms. This runs online inside the turn budget.

### 2.3 Offline optimisation (where "learning" actually goes)
Search the plan genome with a simulator in the loop:
- Fitness = **margin and win-rate against a pool** (Boatlee, our previous best, starter, an exploiter that floods our top product), both seats, fixed seed set per generation, held-out block for acceptance. Not solo money.
- Method: structured GA / evolution strategy for discrete/layout genes, CMA-ES for the continuous ones (full-covariance — it *does* search knob combinations; the E30 failure was the executor, not the optimiser). Write safe ranges next to knobs known to have cliffs.
- Warm start: **compile Boatlee's own plan through our compiler** as a calibration case — if our compiler can't reproduce ~its score from its farm shape, the compiler is the bug. This is imitation used correctly, and the shipped table is ours (no provenance issue).
- Speed: kagsim (Mac) at ~500 eps/s/core for open-loop candidates; in the cloud, a stripped pure-Python two-farm sim (their forward model + market/town, ~50–100 eps/s/core) is enough for the genome search.

### 2.4 Online re-planning (the adaptivity Boatlee lacks)
Because compilation is cheap, re-run it at every day boundary from the *actual* state:
- Shop draws (day 3, 6, 9, …): the branch thresholds switch sub-plans (e.g. yarn store present by day 6 → sheep cohort; ≥ 2 strawberry shops → extra strawberry cohort; no milk shops → cap cows). Decisions are made when the information exists, not on day 0.
- Weeds/misses: the next day's plan simply includes the DIG and replant — no replay hacks.
- Opponent (2.5): their forecast supply feeds the sell rules and the crop/herd choice for cohorts not yet planted.
- Late season: horizon-aware — no cohort whose payoff lands after step 719; final-day liquidation spread over turns.

### 2.5 Opponent module (new)
- **Fingerprint**: at steps ~24/48/72, compare the opponent's public tile census to known scripts (Boatlee's table, our own past submissions). A match tells us their entire future supply *and* their sell steps.
- If matched: front-run **every** scheduled dump of a product we also hold by ~10 steps (relay-sell generalised across the season, not just the end), and steer un-committed cohorts away from the products they flood mid-season (strawberry ~day 20–24 is where our contested games crashed to $1).
- If unmatched: forecast supply from their tiles (planted_day + crop → harvest day), same rules with less confidence.
- Denial/withholding: off — measured to lose (E11, denial −40k).

### 2.7 Scarcity hunter (added by the source audit — see plan_v4_audit.md; verified in E54)
1.32.7 gives CARROT, TOMATO and EGG a `hinge` below-I0 price shape: calm until the market is drained past `T`, then a quadratic runaway (tomato $300 at −2T, $552 at −2.5T; carrot $385 at −2T). Measured in real games: tomato peaks $247/$399/$445 with Pizza-Shop/Farmers-Market draws, carrot $202 with three Pet Cafés — and nobody supplies them (Boatlee plants zero of both). A fertilized tomato tile yields 8 units in ~12 days: $1,200–3,200 per cycle in those seeds. Every daily re-plan projects each product's inventory ~10 days ahead (shop instances × drain + town centre − opponent forecast + our plan); when a projection crosses −T (hinge products) or −0.5T (strawberry/milk), the next cohort goes to that product and sells into the spike in metered batches. Genes: thresholds, max tiles per scarce product, batch size. Boatlee's vendored price table is stale for these three products — we vendor the 1.32.7 table and assert it against the env in I0.

### 2.6 Market rules (kept simple)
Sell on sight; floor only for goods we self-flood (wool with many sheep, melon at scale); shed-pressure valve at ~70; buy wheat/fertilizer on demand only when a unit is about to pick it up.

## 3. Why this can beat Boatlee where the previous lines couldn't
- Same efficiency class as Boatlee (planned routes, role purity, correct fertilize/care timing) — the thing E44 showed reactive engines can't reach.
- Plus four edges Boatlee structurally lacks: scarcity hunting (tomato/carrot/egg hinge spikes, §2.7), shop-draw adaptivity (herd/crops decided when known), opponent-aware production and season-long front-running (incl. market slot alignment against a known script), and a fitness function that optimises *margin vs a pool* rather than one solo trajectory.
- Provenance-clean: our compiler, our plan, our table.

## 4. Roadmap (each phase has a kill criterion fixed now)

**Phase 0 — infra + fallback (2–3 days).** Fast two-farm Python sim (or kagsim on the Mac) with parity check vs reference on the seeds we use; harness upgraded to ≥ 80 games / both seats / fresh blocks / effect+desync counters; opponent pool registered. In parallel, ship the cheap executor fixes (4-tile SHED_ADJ, fertilize at 9/13, daily care) as an improved fallback submission. *Kill: none — infra.*

**Phase 1 — compiler v0, fixed plan (3–5 days).** Hand-write one plan close to Boatlee's shape (3 quads, 9 cows/4 sheep, ~40 strawberry, wheat cycles) and compile it. Gate: **≤ 1.15 steps per useful action, ≤ 10 plants lost/season on 3 quads, strawberry ≥ 6.5/plant, solo money ≥ 110k.** *Kill: if after two routing iterations plants-lost > 20 or steps/useful > 1.4, the compiler idea is wrong — fall back to relay-sell + executor fixes.*

**Phase 2 — offline plan search (3–5 days).** GA/CMA-ES over the genome vs the pool. Gate: solo ≥ Boatlee's; head-to-head vs Boatlee ≥ 60% both seats on a fresh block. *Kill: if search cannot beat the hand-written plan by the CI, the genome is mis-parameterised — revisit representation, not the optimiser.*

**Phase 3 — online re-planning + opponent module (3–5 days).** Branch points on shop draws; fingerprint + front-run; opponent-conditional cohorts. Gate: **≥ 80% vs Boatlee both seats, ≥ 80 games; no pool regression; p99 turn time < 100 ms.**

**Phase 4 — ladder.** Submit; read episodes and logs; fingerprint the field (how many Boatlee-clones, what the others do); add their scripts to the pool; iterate the genome against the real distribution.

## 5. What we deliberately don't do
Deep RL from scratch (env speed, action space, and the kagsim line's own survey say no); pure behaviour cloning (caps at a tie); more per-turn routing heuristics or per-turn rollouts (E29, E39/E40, my resident/anchored/route routers — all measured flat); market timing beyond front-running and self-flood floors; mutating Boatlee's table as the shipped agent (only as a calibration case, unless you clear provenance — then relay-sell stays a valid overlay on whatever we ship).

## 6. Risks and their answers
- *Compiler is a real piece of software* — the VRP is small (grid, ≤ 100 tasks) and the forward model already exists; day-granularity keeps it simple. Phase 1's gate catches failure early.
- *Overfitting the search to Boatlee* — pool fitness + held-out seeds; keep an exploiter in the pool.
- *Rule drift* — pin the env version, hash the source (their V6 guard); re-verify parity on change.
- *Deadline unknown* — Phase 0's fallback submission exists so we're never empty-handed; please find the date.

## 7. First three actions
1. Build the two-farm fast sim (or wire kagsim) and the 80-game/both-seat harness with pool + counters.
2. Write the compiler v0 (day tasks → hands → cluster-route → 24-turn script → forward-model check) and run Phase 1's gate on the hand-written plan.
3. While that runs: apply the executor precision fixes and re-measure it properly (≥ 80 games) as the fallback.
