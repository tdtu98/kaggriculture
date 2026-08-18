# Kaggriculture — Tasks v4 (executable breakdown of PLAN_v4)

Every task states **build**, **verify**, and a binary **done-when**. "It runs" is not done. Measurement rules apply to every number: ≥ 80 games, both seats, a fresh seed block never used for tuning, effect counter checked before the money is read. Sizes: S ≈ half day, M ≈ 1–2 days, L ≈ 3–5 days.

Where a component already exists in the kagsim repo (`agent/forward.py`, `kagsim`, `arena/`, `tools/promote.py`), reuse it; the tasks say so. Where we are in the cloud without it, the task gives the pure-Python fallback.

## Status

| Task | State |
|---|---|
| I0 env pin + facts test | **done** — `tests/test_env_facts.py` (29 assertions), env pinned to 1.32.7, `hinge` ported to kagsim + `agent/relay.py`, `make verify` 0 divergences (E54) |
| I1 fast two-farm sim + parity | **partly** — kagsim re-verified against 1.32.7 (700/755 statements, 0 divergences, 269k steps/s). The pure-Python cloud `FastSim` is still todo |
| I2 harness v2 (pool, blocks, counters) | **done** — `harness/{run,registry,counters}.py`, 12 tests. Both I2 verifications reproduce (mirror wobble −96/+771/+843; `executor_v7` $72,979 ≈ 74k). Pool baseline on 21000:21040 in `results/games.jsonl` (E55) |
| F1–F5 executor precision fixes + scarcity redirect (fallback submission) | todo — **source now present**: `session_line/executor.py` (+ its `planner-2.py`), loaded via `session_line.load`; smoke 67–82k vs starter |
| C0 plan representation | **done** — `agent/plan.py` (49 genes), 24 tests. 5,000 random genomes all decode valid and round-trip. `Plan.boatlee_like()` built from a measured season, not prose (E56) |
| C1 day task generator | **done** — `agent/tasks.py`, 29 tests. 0.31 ms worst case on a real season (budget 2 ms). C1′ deadlines/cadence included; `price_fn` is the seam C6 plugs into (E57). **Tick-day water bug fixed (E67)**: `_water_bonus` gated on `fertilized_until_day >= day`, false on fertilize-today ages (strawberry 9/13, tomato 7/10) so **no** water task was emitted on half of tick days; new `_fertilizes_today` predicate is the single source for both tasks and tick-day water is **survival-class** — tick-day watered 64.3% → 96.5%, doubled 2.17 → 3.10/plant. **`price_fn` seam now projection-capable (E72), off by default** (`projected_pricing=0`) |
| C2 router (VRP) | **done** — `agent/router.py`, 34 tests. 0.85 is the *planned* steps/useful; **in-play 1.02** (gate ≤ 1.15, Boatlee 1.02); `route()` 0.3 ms, `decide_hands` 14–28 ms/day (E58). Audit (E61): logistics DROP now implemented and **gated to the last day** (+$8,604/season, CI [7,640, 9,568]) — dropping daily as spec'd loses $5–9k. `partition()` **fixed** (E64): position-aware, **mean 1.015x exact** (was 1.492x), 5%-of-exact now a passing test, guard 1.8x → 1.15x; **+$28,728 in play**, CI [19,760, 37,696], steps/useful 1.05 → 0.78. Last-day DROP regression **closed (E65)**: +$6,664, CI [5,923, 7,404], better 80/80, buzzer holdings 29/39 → **0/39**; the day-29 hour-23 phantom turn (framework plays `episodeSteps−1`) also fixed |
| C3 day verifier + overcommit pruning | **done** — `agent/verify.py`, 15 tests. 0 thirst / 0 overcommit days are the *shadow-model* numbers; 5.2 blocked ops/season (Boatlee ~10), 15 ms of a 100 ms budget (E59). Counters now wired (E62): live play loses ~2.6 plants/game to thirst (E60), and `boatlee_like` prunes **~200 tasks/season** with 1.4 overcommit days. Stock-ledger double-claim fixed |
| C4 agent shell (compile at dawn, execute, market) | **done** — `agent/main_v4.py`, registered as `compiler`, 12 tests. **1.01 steps/useful (= Boatlee)**, blocked_ops 1–3 (bar 12), p99 6.7–7.6 ms, 0 fallbacks both seats through the reference env (fallback counter proven live by injected exception, E61). Money $52k — plan-bound, see E60. Effects ledger + `release_pressure` valve + dawn order-truncation fix landed (E62); terminal spread still dump-all, front-run awaits O3. Conjunction gate E63. **Herd 1:1 pacing + purchasable-wheat throttle (E67)**: `_paced_plan` trims herd to `min(due, owned+lead)` (empty pastures day 8 13 → 6.4), `_feedable_animals` gains `(shed + budget//price)//FEED_BUFFER_DAYS(=4)` — cow-days 95 → 130, +$22,879 CI [+14,857, +30,900]; buying **animals before seed** (E66's literal cash-floor reading) is **−$22,296** and refuted. Wheat-wave stagger (E68). **Cow-first ordering + payback cutoff (E70)**: `_purchase_order`/`_yield_per_dollar` rank COW first ($0.60/day/$ vs SHEEP $0.53, and cheaper at $400), `_last_buy_day` derives COW 20 / SHEEP 22 / GOOSE 22 from env arithmetic (`kaggriculture.py:822-829`) so `animals_ordered_d25` 5.8 → 0; plan `cow_start` 5 → 1. First cow day 11.7 → **2.01**, +$7.7k solo / +$10.5k vs boatlee. **Open: HIRE starves the dawn market queue** — `maxMarketOrdersPerTurn=10` and `_dawn` appends HIREs before buys, so a 10-hire day buys nothing (E68). **CLOSED (E71)**: the env fact was wrong — `maxMarketOrdersPerTurn` caps a **turn**, not a day (`_process_market` runs every step, `kaggriculture.py:941`), so a day carries 240 slots and land is buyable at any hour. `_truncate` → `_dispatch` ranks BUY_LAND → BUY_SEED/BUY_PRODUCT → BUY_ANIMAL → HIRE → SELL (E62 sell rule kept) and serves **every turn**; overflow hires **dropped** (deferring/spreading measured worse: +$22.5k/+$25.4k/+$26.4k vs +$28.9k). `dawn_starved` **0** (naive 5.5/game), `hires_dropped` 16.0. **Value is substrate-dependent**: +$29,365 [+25.4k,+33.4k] 76/80 on a frozen pre-C6 tree, **≈$0 on the final tree** (+$1,574 / +$402 vs starter, −$1,541 vs boatlee, n=50; `pruned_tasks` unchanged) — the outage stopped binding. Kept as a tripwire; guardrails pass (feed 0.870, care 0.916, thirst 3.05, fallbacks 0). **NW-wheat/LOCKED dawn bug fixed (E74)**: `_dawn` derived the shopping list and crew size **before** the same-turn `BUY_LAND`, and `agent/tasks.py:567` tested `tiles[y][x] is None` while a locked tile is the **string `"LOCKED"`** — so the arriving quadrant (unlocked that same turn, `kaggriculture.py:724`) contributed no seed order and no hands, and hour 1 pruned its cohort against an empty shed. A one-day tax normally; **a season when land is late** — it produced a **2× money cliff** (18-pasture NW-full genome $46.5k vs 20-pasture $91.5k) that would have made the S2 landscape unsearchable. Fix (~40 lines): land decision moved **above** task derivation + `_unlocked_view` gated on the arriving quadrant's oldest cohort being ≥6 days overdue (`LATE_COHORT_DAYS=6`, measured — always-on costs incumbents −$1.6k/−$2.8k, threshold 3 worse); counter `land_day_reseed` is 0 over a boatlee_like season (test-asserted) and fires on every late dawn of the cliff genome. Bad arm $44.3k → **$75.8k**, worst wheat-ladder step 55% → 3.5%, fresh-seed sweep (17 genomes) worst step **14.4%** (<30% property S2 needs); incumbent invariance 480 games × 2 blocks, +$373 [+32,+714] / −$5 [−39,+29], 946/960 byte-identical. Ruled out by measurement: thirst, pruning/blocked/overcommit, herd/structures, and "NW exactly full" per se. **Residual: `int(0.8·wheat_tiles)` in `_feedable_animals` is still a discontinuity** — bounded in cost now, but the noisiest axis in the sweep |
| C5 Phase-1 gate on hand-written plan | **measured (E66)** — gate run on 45000:45040, 80 games both seats. PASS steps/useful **0.79**, thirst **2.5**, blocked_ops **1**; FAIL straw/plant 5.6, milk/cow-day 0.60, ripe@end 28, solo money **$89.6k**. **Kill does not fire** (bars 20 / 1.4) — compiler idea alive, F-gate fallback not needed. **E60's "money is plan-bound" refuted**: ~all of the −$66.8k gap is execution — herd arrival **day 18.6** vs plan day 5 (≈$45k; `_feedable_animals` wheat-tile throttle `agent/main_v4.py:342` + empty-pasture cash floor days 2–11) + **tick-day watering 67% vs 98.6%** (≈$7k, `kaggriculture.py:797`) + wheat base oscillating **1–10 vs steady 13** (≈$17k). **Fixed through E67/E68, re-gated E69** (50000:50040, fingerprint `ab86c6175b7b`): **5/7 PASS** — steps/useful 0.76 · thirst 1.14 · straw/plant **6.77** · blocked_ops 1.6 · **solo money $114,302 ≥ 110k** (+$24,681 vs E66), 80/80 vs starter; **kill still does not fire**. Remaining FAILs: milk/cow-day 0.73 awaits the **cow-first fix** (first COW day 11 vs boatlee day 1; in flight, expected +$15–17k); ripe@end 26.7 is a **counter artifact** (17.4 wheat, worth ~$0 — the compiler eats its wheat, E68). Then C6/S: the residual boatlee gap (0/80, $39.4k/$129.0k) is **plan-bound** — market closure, not execution (E69). **Cow-first landed (E70)**: first cow day **2.0**, d25 waste **0**, **+$7.7k solo / +$10.5k vs boatlee** (the E67 boatlee regression is gone; gap −$93.5k → −$84.0k), solo money ≈ **$120k**. milk/cow-day 0.90 is now **plan-bound** — wallet $4–$1,000 on days 4–9, no early-cash crop in `Plan.boatlee_like()` (boatlee plants CARROT/TOMATO day 0) → **S-track gene**. **C5 arc CLOSED, proceed to C6/S** |
| C6 product projection + scarcity hunter | **done (E72)** — `agent/projection.py` (588 lines), every mechanic derived from env source (drain 6/day at `step%4==0`, x2 single-product `:741`; town centre 1/day `:734,745-747`; instances `min(8, day//3)`; partial-day hour ordering `:941-942`, a real 6-unit bug). Prediction on deviations ≥0.25T: CARROT 2.1% / EGG 1.5% / TOMATO 3.5% median error (WOOL 32.5% worst); the spec's ±15%-on-raw-inventory bar is unusable (everything within 1% of I0). Hunter done-when **met**: 6/7 hinge seeds fire, 0/7 calm. **Every money lever null-or-negative over 3 blocks / 240 paired games**: projected pricing −$2,675 [−4,602,−749] vs starter, strawberry redirect −$4,215, metering null both settings; single-block 52000's +$3,898 did **not** survive two more blocks (E37/E39/E42 pattern). **Ships as 5 genes (49 → 54), all off** — shipped tree outcome-identical to pre-C6 (40 games) and $0 different from all-off (160 games). `test_projection.py` 32 tests, 9/9 mutations caught. **S2's job to turn on**; the hunter can never fire vs `boatlee_like` (pending melon $1,420 / strawberry $1,660 per tile outvalue any tomato) — it needs a **cheap pending cohort**, i.e. plan shape, not execution |
| S1 fitness + pool | **done (E73)** — `search/fitness.py`, fitness `0.7·weighted_win_rate + 0.3·tanh(margin/50k)`, pool = boatlee ×2 / flooder / tomato_rusher / executor_v7, both seats; `search_seeds(gen)` rotates 6 seeds in 60000:64000 and **raises** on the reserved blocks (54000:54080 acceptance, 50000/51000 gates). Reproduces the knowns (solo $124–126k vs E70's ≈$120k; 0% vs boatlee at −$74/83k vs −$84k). **The exploiters had to be measured into existence**: the spec-literal flooder (75 strawberry tiles) soloed $75k and **lost 8/8 to the compiler by +$42k** (`animals_past_payback` 149, strawberry never in the ground) → zero selection pressure; rebuilt to 20 pastures / 67 strawberry tiles / 14 cows, solo $98.5k, **beats the compiler 16/0 by $59.7k**. `tomato_rusher`'s day-6 plant is **advisory** — the cash floor delays seed to ~day 13 (same E70 wallet cliff) — solo $117.2k, loses 14/16 but tightest margin (+$14.6k), kept. 3 of 4 pool members apply real pressure. **Win-rate is coarse**: the spec's "no pastures" bad plan scores 0.164 vs baseline 0.172 (a dead plan scores 0.0024) — **the 30% margin term carries the gradient**. Cost measured: 7.4–8 games/s (7 workers), 48-game eval ≈6 s, **generation ≈5–6 min = 12× the spec's 25 s/gen estimate** |
| S2 genome search | **done (E76)** — **68 gens** (plateau, best unchanged gens 25→67), 4,320 evals (3,232 GA + 1,088 CMA), guard clean, fallbacks 0, best score 0.6844. Winner beats the incumbent **80/80 on 54000:54040, CI [95.4%, 100%]**, margin +$123,644 — but **that margin is an artifact**: the incumbent hit the E70/E74 wallet cliff under the candidate's aggressive early selling (`release_pressure` 70→16), earning <$1,000 in 74/80 games (median $2, `plants_started` 12 vs 87, `structures_deferred` 259, no crash) while the candidate soloed $122.6k — genuine market denial, not skill. **Differs in both herd and cohorts** (S2's must-differ): 4 sheep+9 cows → 5 sheep+7 cows (`animals_per_day` 3→4), cohorts 6→5 (NE wheat replant deleted), strawberry NE 22→25 tiles day 7→10 / SW 14→11 day 10→15, melon SW 11→10, pastures 13→12, land NE 6→9 SW 10→9, `hands_start` 4→3; C6 `projected_pricing` **OFF** (the E72 −$2,675 gene did not survive, unlike the smoke's gen-2 winner), scarcity thetas moved (`theta_sqrt` 0.5→1.15, `max_scarce_tiles` 8→19), market consts hard (`sell_floor_wool` 0.35→0.131, `sell_floor_melon` 0.35→0.695, `frontrun_lead` 10→20). Pool no-regression (62000:62012) **strict domination**: flooder/tomato_rusher/executor_v7 100% at +$83k/+$118k/+$94k, thirst 3.0-7.0 → 0.08-0.83. **The search's own boatlee signal (0.167 at 6 seeds) was noise — refuted at 80 games** (E37/E39/E42 pattern, caught by the gate). Ledger `search/accept_ledger.json` created, 54000:54040 spent, **54040:54080 reserved**; candidate vec at `results/s2_prod/state.json['best']['vec']`. Nothing promoted; `agent/plan.py` defaults unchanged. Prior build detail (E75): `search/ga.py` + `run.py` + `accept.py` + `tests/test_search_run.py` (49 tests; name avoids v1 CEM's `test_search.py`). Uniform **per-BLOCK** crossover (land/herd/hands/cohort-slots as units — per-gene shreds cohorts), CMA genes taken whole from the fitter parent, GA/CMA **partition the 54 genes** (asserted at import), **CMA budget ≥8 enforced-with-warning** (budget 4 = one ask/tell round = a random perturbation that looks like a working inner loop — a real config bug in earlier runs, now pinned by test), guard penalty **2.0** (at 1.0 a perfect illegal genome ties the worst legal one), atomic state JSON, CMA seeded on (run,gen,rank) so resume is stream-independent. Smoke (pop 48, 3 gens, 208 evals, 9,648 games): best **0.5193 → 0.5272 → 0.5879** monotone, margin −$9,241 → **+$5,984**, **diversity RISES 0.077 → 0.137** (no collapse), repair 0.28–0.30, guard violations 0, counters inside C5 bars (steps/useful ≤1.051, thirst ≤8.06, fallbacks 0). Gen-2 winner differs from `boatlee_like` in layout (14 pastures, NE day 8, melon 11) and turned `projected_pricing` **ON** — 6 seeds only, and **E72 measured that gene at −$2,675**; watch, don't believe. Resume determinism: killed-after-gen-2 vs uninterrupted **bit-identical across all 3 gens**; 9 mutations, 9 caught (one a real gap — RNG state unused by the first resumed gen). ~5.2 s/eval → 50 gens ≈4.7 h. Suite **710 passed**, `make verify` 0 divergences, acceptance block **54000:54080 untouched**. **Lessons**: the "compiler stall" was the agent's own `;` chaining masking a killed process (no defect — verify before diagnosing); background children get a **9% duty cycle** on this Mac, so production searches run **detached** (double-fork+setsid) under `caffeinate` with wall-time in every log row. **Acceptance via `search/accept.py` on 54000:54080; S3 gate next**. **Genome widened 54 → 80 genes (E76 → E77)** — 10 cohort slots, 8 per-product sell floors, per-quadrant row operators; **explored (round 2 used two new slots, a new crop, and all eight floors), and insufficient** — see S3 |
| S3 Phase-2 gate | **failed both rounds (E76, E77); track CLOSED — pivot to Track O** — **round 2 (E77)**: the widened 80-gene space was searched (67 gens, plateau gens 38→66, 5,351 evals, best 0.6866) and produced a genuinely third point (13 pastures, 4 sheep + 8 cows, 7 cohorts incl. two new slots and the **first TOMATO cohort in any plan**, all 8 per-product floors non-zero, `release_pressure` 70→4, `projected_pricing` OFF for the third time). Accepted 98.8% (79/80) on the **last reserved sub-block 54040:54080 — the acceptance reserve is now EXHAUSTED** (`accept.remaining() == []`; any further block is a **logged decision to extend**), margin +$124,680 again diagnosed as the E70/E74 wallet-cliff artifact (incumbent <$1,000 in 73/80, median $2). **Decisive 64000:64040** (freshness verified, max search seed 60407, overlap empty), three-way same-seed paired vs boatlee, 80 games both seats: R2 **0/80** [0, 4.6] at −$60,209 · R1 0/80 at −$67,096 · incumbent 0/80 at −$82,189 — **gap monotone −82.2k → −67.1k (+15.1k) → −60.2k (+6.9k)**, so **E76's gain reproduces on an independent block**. But **the mechanism turned**: solo (64000:64020) boatlee $159,188 · R1 $129,142 · incumbent $122,786 · **R2 $119,799 — worst of the three**. R2's head-to-head gain is **denial, not production** (gives up ~$9.3k of its own output to take ~$6.2k off boatlee) — denial beats our wallet-cliff incumbent and does nothing to boatlee, who has no cliff. **[SETTLED]** `projected_pricing` is **inert** on this plan shape: ON vs OFF, 80 paired games vs boatlee, **$79** apart — the launch-report +$15k probe was noise *and* E72's −$2,675 does not reproduce (third scope-expiry of this gene). **Kill criterion re-read**: "mis-parameterised genome" **no longer fits** — the widened space was demonstrably explored (new cohorts, new crop, all floors live) and still tops out **$30–39k of solo production short**; the gap is **structural to the choreography**. **[LESSON, third occurrence]** the fitness's 6-seed boatlee sub-evaluation produced the **identical 0.167** signal in both rounds (E37/E39/E42 → E76 → here) — **any future genome round requires ≥40 boatlee seeds in fitness plus a logged reserve extension**. Artifacts: `results/s2_r2/state.json['best']['vec']` (gen 38), copy `/private/tmp/r2_cand_vec.json`, raw `/private/tmp/decisive_all.json`. Nothing promoted. **Round 1 (E76)**, kept for the record — vs boatlee on a **fresh** 80-game block (63000:63040; freshness verified from the search log — all 408 search seeds in 60000:60407, overlap empty): **0/80**, CI [0%, 4.6%], $62,764 vs $130,472. Gate needs ≥60%. Solo (63000:63020) candidate $132,700 > incumbent $122,558 but **boatlee $150,792** — fails "solo ≥ boatlee's" by −$18k. **But the gap closed**: margin −$67,708 vs the incumbent's **−$82,768** on the identical block, **+$15,061**. **Kill criterion invoked as written** — the optimiser found a strictly better point in the space it was given, so the **genome is mis-parameterised, not the optimiser**: ~$18k of solo production the representation cannot express. Iteration prescribed: **per-quadrant crop rows**, **more cohort slots**, **per-product sell floors**; then **ONE re-gate on 54040:54080** (last reserved block) — **all of it done and spent in round 2 above (E77); insufficient** |
| O1 shop-draw branch points | todo |
| O2 opponent fingerprint + forecast | todo |
| O3 front-run + counter-mix | todo |
| O4 Phase-3 gate + turn budget | todo |
| L1 submission bundle + smoke | todo |
| L2 ladder read-back | todo |

---

# Track I — Infrastructure (Phase 0)

## I0 — Pin the env and assert the facts we depend on · S

**Build.** `tests/test_env_facts.py`: hash `kaggriculture.py`, record `kaggle-environments==1.32.7`, and assert by *running the reference env* (not by reading docs):
- `FERTILIZE` on day d → `fertilized_until_day == d+2`; strawberry planted day 0, watered daily, fertilized at ages 9 and 13 → `yield_units` accrues 2,2,2,2 (harvest each tick and count 8 total).
- Cow fed+cared daily → 3 milk on the 2nd production; unfed production day pays 1 and clears the bank.
- All four centre tiles accept `PICKUP`/`DROP` while NE/SW/SE are LOCKED.
- Hands hired at step s appear at step s+1 on the least-occupied centre tile; farmer respawns at (4,4) at dawn; end-of-day inventories drop to the shed (cap 100, overflow lost).
- Unit actions resolve before market orders in a step (a `DROP` at step s can be sold at step s).
- Shops draw with replacement (`MAX_SHOP_INSTANCES = 8`), `townCenterSellInterval` default 24.

- (audit adds) vendored `market_price` equals `kaggle_environments.envs.kaggriculture.kaggriculture.market_price` for every product at inventories I0±{0.5T, T, 2T, 2.5T} — catches Boatlee's stale table (carrot/tomato/egg are `hinge` below I0 in 1.32.7); melon `max_yield_day == 12`; ongoing `yield_units` cap 4; wheat decays from dawn of age 5; a newly placed animal survives day 1 unfed; empty pasture tiles never spawn weeds.

**Done when:** all assertions pass; the test fails loudly if the source hash changes (message: re-verify I1 parity before trusting anything).

## I1 — Fast two-farm simulator · M (or S if kagsim available)

**Build.** If on the Mac: use `kagsim` (already bit-exact under 1.32.6; re-run `make verify` against 1.32.7 first). In the cloud: `sim/fastsim.py` — pure Python, both farms + market + town + weeds RNG, cloned from `kaggriculture.py`'s functions with the framework/observation-building removed (import its `CROPS`, `ANIMALS`, `MARKET_PARAMS`, `SHOPS`, `_market_price`, `_apply_unit_action`, `_process_market`-equivalent directly where the signatures allow — copy where they don't). API:
```python
sim = FastSim(seed, cfg=None)
obs0, obs1 = sim.observe()          # same dict shape as the reference (both seats)
sim.step(action0, action1)          # one turn, both players
sim.done, sim.money -> (m0, m1)
```
Also `sim.snapshot()` for compile-time state cloning.

**Verify.** `tests/test_fastsim_parity.py`: drive reference env and FastSim in lockstep with (a) `starter` vs `starter`, (b) Boatlee vs our executor, (c) a fuzzer (legal-ish, illegal, market storms), on ≥ 20 seeds × full 719 steps; assert canonical state equality every step (money, tiles, hands, shed, seeds, inventories, market inventory/prices, town shops). Weeds must match bit-exact (CPython `random.Random((seed*1_000_003)^day)` — reuse the reference's own call so no RNG port is needed).

**Done when:** 0 divergences on all three drivers; ≥ 100 episodes/s per core for open-loop scripts (measure with `tools/bench.py`). *Kill:* if parity can't be reached, run everything on the reference env at ~0.5 games/s/core and shrink search budgets accordingly — slower, not blocked.

## I2 — Harness v2: pool, blocks, counters · M

**Build.** `harness/run.py`:
- Agents by name from `harness/registry.py` (module path + kwargs). Pool: `boatlee` (main.py, read-only), `executor_v7` (session executor), `starter`, `kagsim_champion` if available, `flooder` and `tomato_rusher` (see S1).
- `--seeds 21000:21040 --both-seats --games 80` → per matchup: win/loss/tie, mean margin, Wilson 95% CI, per-seed rows.
- **Counters** printed with every result: `steps_per_useful` (moves ÷ non-move non-PASS ops), `idle_pct`, `plants_lost` (weeds created from PLANT tiles, per season), `strawberry_per_plant`, `milk_per_cow_day`, `unharvested_ripe_at_end`, `shed_overflow_discarded`, `blocked_ops` (ops that were no-ops at their tile — the desync counter), `fertilize_hits` (fertilizes landing at ages 9/13), and per-overlay `effect_count`. These come from a per-step observer, so every agent gets them for free.
- Multiprocessing across cores; results appended to `results/*.jsonl` keyed by agent hash + seed block.

**Verify.** `boatlee` vs `boatlee` on 20 seeds reproduces exact ties on seeds 2–6 and ±96/±771/±843 on 0/1/7 (the known seat wobble). `executor_v7` vs `starter` reproduces ~74k ± noise.

**Done when:** one command produces the table with CIs and counters for any pair in the registry.

---

# Track F — Executor precision fixes (fallback submission, Phase 0)

These are the cheapest measurable wins on the existing `executor.py`; they also make it a stronger pool member. Each is a separate flag, measured alone then together (conjunctions matter — E43).

## F1 — Shed adjacency = 4 tiles · S
`SHED_ADJ = {(4,4),(5,4),(4,5),(5,5)}`; keepers path to the nearest of the four. Effect counter: pickups per day by tile. Verify feeding slips don't rise (audit tool). Done when: `underfed_animal_days` ≤ baseline and pickups occur on ≥ 2 distinct tiles.

## F2 — Strawberry fertilize at ages 9 and 13, pre-assigned at dawn · M
At hour 0: list strawberry tiles with age ∈ {9,13} today, grouped by the zone that owns them; that zone's worker gets `need_fert = n`; if `shed.FERTILIZER < total_need`, add `BUY_PRODUCT FERTILIZER k` at hour 0 (buy-on-demand); the worker `PICKUP FERTILIZER need_fert` at spawn (it is on a shed tile), and in `_zone_op` a tile at age 9/13 that is watered gets `FERTILIZE` at priority just below survival water. Never fertilize other ages. Effect counters: `fertilize_hits`, `fertilizer_wasted` (bought/collected but unapplied). Verify with a scripted single-tile test through the reference env: 8 units harvested from one plant. Done when: `strawberry_per_plant ≥ 6.0` on 3-quad games and `plants_lost` not above baseline.

## F3 — Care every animal every day · S
In the keeper ladder: FEED > CARE > HARVEST (harvest can wait a turn; the care bank can't be back-filled). Effect counter: `care_days / animal_days`. Done when: ≥ 0.9 and `milk_per_cow_day ≥ 1.2` (base is 0.5).

## F4 — Harvest before decay; hire like it's cheap · S
Harvest ongoing crops the day they tick (priority above optional water); raise hand cap toward 12 with the cash guard kept. Done when: `unharvested_ripe_at_end` halves and solo money not lower.

## F5 — Cheap scarcity redirect · S
When C6's `scarcity_signal(TOMATO|CARROT) ≥ 1.0` (or the live price ≥ 2× base), the next N empty crop tiles take that crop, harvest on tick, fertilize tomato at 7/10. Effect counter: scarce units sold and realised price. Done when: on hinge seeds solo money rises by CI; on non-hinge seeds unchanged.

**F-gate:** F1–F5 together vs `starter` and vs `boatlee`, 80 games each. Record as fallback. Expected: solo 90–110k, vs Boatlee still < 50%. This is not the plan; it's the parachute.

---

# Track C — The compiler (Phase 1)

## C0 — Plan representation · S

**Build.** `agent/plan.py`:
```python
@dataclass
class Plan:
    pasture_tiles: list[tuple]           # ordered; keepers' cluster (default: ring around (4,4))
    land_days: dict[str,int]             # {"NE":6,"SW":10,"SE":99}
    herd: list[tuple[str,int]]           # [("SHEEP",0),("COW",5),("COW",6)...] (species, buy day)
    cohorts: list[Cohort]                # crop, tiles (explicit list or "rows of quadrant"), plant_day, replant: bool
    hands: list[int] | "auto"            # per-day count, or auto from marginal task value
    consts: dict                         # fert_ages={STRAWBERRY:(9,13), MELON:(5,)}, care=True, sell_floor={WOOL:0.35,MELON:0.35}, release_pressure=70, frontrun_lead=10
    branches: list[Branch]               # (day, condition on shop counts, plan patch)  -- used from O1
```
`encode(plan) -> np.array` / `decode(vec) -> Plan` with bounds, for search. `Plan.boatlee_like()` = a hand-written plan matching Boatlee's shape (3 quads at days 6/10, 4 sheep day 0 + 9 cows days 5–15, ~37 strawberry in 2 cohorts, wheat cycling on the rest, melon 19).

**Verify.** Round-trip encode/decode; every generated plan passes `plan.validate()` (tiles unique, inside unlocked land by plant day, herd ≤ pastures).

**Done when:** tests pass and `Plan.boatlee_like()` prints as a readable table.

## C1 — Day task generator · M

**Build.** `agent/tasks.py`: `daily_tasks(state, plan, day) -> list[Task]` where `Task(tile, op, args, value, deadline_turn, needs)`:
- Survival water: every plant with `consecutive_unwatered == 1` and not watered → value = plant's remaining expected value (must not miss). Optional water: `consecutive_unwatered == 0` → value = 0 for ongoing crops on non-tick days, small for one-time crops *inside* their bonus window (each watered day = +1 unit).
- Harvest: `yield_units > 0` (ongoing: harvest today; one-time: at max yield day or when `yield_units == max`); value = units × current price. Ripe tiles near `max_lifespan_step` get deadline = now.
- Fertilize: strawberry age ∈ fert_ages today (needs FERTILIZER in inventory) value = +2 units × price; melon age 5 (+1/day for 3 days).
- Plant: cohorts due today; needs seeds; value = cohort NPV; **only admitted if the sustaining cost fits** (C3).
- Animals: FEED (needs WHEAT in inventory), CARE, HARVEST if `yield_units > 0`, COLLECT_FERTILIZER only if fertilizer will be used or sold (value = price − nothing; low).
- Logistics: PICKUP WHEAT/FERTILIZER at a centre tile at spawn for units that carry those tasks; DROP for units ending near the shed with inventory (so sales happen today, not tomorrow — the shed cap discards overflow at dusk).
- Weeds: DIG on tiles the plan wants.

**Verify.** `tests/test_tasks.py`: hand-built states → expected task lists (a strawberry at age 9 unwatered yields WATER then FERTILIZE; a cow unfed yields PICKUP-then-FEED for its keeper; a plant at `consecutive_unwatered==1` is always a survival task).

**Done when:** tests pass and the generator runs < 2 ms on a 100-tile farm.

**C1′ — corrections from the source audit (mandatory):** task values price each harvest at the *projected* inventory when it will be sold (C6), not base price; deadlines from the exact rules — wheat harvest by age 4 (or ≤ hour 7 of age 5), melon window ages 10–13 (`max_yield_day` is 12 in code), fertilized strawberry harvested every ≤ 2 ticks and tomato every ≤ 2 days (cap 4), cows every ≤ 2 productions when cared (cap 6); tomato fertilize at ages 7 and 10, melon at 6; one-time HARVEST clears the tile → replant task next day; DROP tasks whenever projected dusk shed total > 90.

## C6 — Product-projection module · M (new; shared by C1′, O1, O3, scarcity hunter)

**Build.** `agent/projection.py`: `project(product, days, state) -> [inv_t]` using the 1.32.7 curves: current inventory − Σ_instances(drain: 6/day, ×2 single-product) − town centre (1/day) − opponent forecast (O2, from their tiles) + our planned harvests; `scarcity_signal(product) = (I0 − min projected)/T`. Scarcity hunter: if signal ≥ `θ_hinge` (tomato/carrot/egg, gene, ~1.0) or ≥ `θ_sqrt` (strawberry/milk, ~0.5), the next cohort(s) up to `max_scarce_tiles` go to that product; sells into a spike are metered in `batch` units so each batch stays above `floor × spike price`.

**Verify.** Replay 40 recorded games: prediction of day-t inventory from day t−5 within ±15% for shop-demanded products; on seeds 3 and 5 (tomato $399/$445) a tomato cohort of 12 tiles planted at day 8–12 sells ≥ 80 units at ≥ $150 in the forward model. Effect counters: `scarce_cohorts`, `scarce_units_sold`, realised price.

**Done when:** tests pass and the counters are non-zero on hinge seeds and zero on non-hinge seeds.

## C2 — Router · L

**Build.** `agent/router.py`: `route(tasks, units, turns_left) -> {unit: [op per turn]}`.
1. Roles (E46): keepers = units assigned to the pasture cluster (count = ceil(animals/6)); the rest field hands. Keepers never take field tasks unless idle with ≥ 6 turns left.
2. Cluster field tasks into `n_field_hands` groups by k-means on tile coordinates (seeded by row bands, quadrant-local), balancing total *time* (1 turn per op + Manhattan moves) not counts.
3. Per unit: build a route by nearest-neighbour from its spawn tile respecting deadlines (survival waters first-class), then 2-opt on the visit order; expand to per-turn ops (moves then ops, multiple ops on the same tile back-to-back, e.g. WATER then FERTILIZE then HARVEST).
4. Fill leftover turns with optional waters in the bonus window, then DROP at a centre tile if inventory > 0 and the shed is reachable in time.
5. Hands for the day (when `plan.hands == "auto"`): start at 0, add a hand while the marginal task value routed by the last-added hand > `fib(n)`; also add a hand if any survival water is unrouted.

**Verify.** `tests/test_router.py`: synthetic farms (25/50/75 tiles), asserts: every survival water is scheduled ≤ its deadline; `steps_per_useful ≤ 1.15` on the 3-quad Boatlee-like layout; no unit's script exceeds `turns_left`; PICKUP precedes FEED/FERTILIZE for the same unit; solve time < 20 ms for 12 units × 100 tasks. Compare against exact assignment on small instances (Hungarian on 4 units × 8 tasks) — within 5% of optimal.

**Done when:** all asserts pass on 50 random farms.

**Audit corrections (E61):** item 4's DROP is **last-day only** — dropping every day costs $5–9k
(milk/cow-day 0.48 → 0.23, thirst 1.4 → 3.4); the leak it fixes is that the episode ends at hour 22
of day 29, so goods left in hands never meet a market turn. The 5%-of-exact assert **fails**
(1.26–1.72x on Hungarian 4x8, cause: `partition()` sweep cuts are unit-blind) and is pinned as a
strict xfail with a 1.8x regression guard — not a passing gate.

**Superseded (E64):** `partition()` is now position-aware (cheapest insertion from unit starts +
boundary repair + soft load penalty). Against a Held-Karp exact oracle on 4x8: **mean 1.015x, worst
1.105x, 34/40 within 5%** — the spec'd assert is a normal passing test and the guard is 1.15x. Worth
**+$28,728/season** vs `starter` (CI [19,760, 37,696], 80 paired games, block 43000:43040).
**Was open, now resolved (E65):** the denser day crowded out the last-day DROP above (`hand_drops`
2.7 → 1.0, ripe@end 10 → 22). Fixed by budgeting the return leg *during* last-day routing (the
suggested approach) **plus** a second defect it exposed: day 29 ends at hour 22 because the
framework plays `episodeSteps−1` turns, so DROPs scheduled at hour 23 were never emitted. Worth
**+$6,664** vs `starter` (CI [5,923, 7,404], 80 paired games, block 44000:44040, better 80/80);
`hand_drops` 0.79 → 5.13, units still holding at the buzzer 29/39 → **0/39**.

## C3 — Day verifier and overcommit pruning · M

**Build.** Reuse `agent/forward.py` (`FarmModel.from_obs`, `.step`, `.clone`) if available; else `agent/forward.py` in the cloud = FastSim's own-farm subset. `verify_day(state, scripts) -> DayReport{missed_survival: [...], blocked_ops: [...], end_state}`: step the model through the 23 turns and check every task landed. If a survival water was missed or a PLANT would land on a tile with no water in the next 2 days, **drop the lowest-value plant tasks and re-route** (bounded to 3 iterations); if still infeasible, raise hands (C2.5) once; if still infeasible, log `overcommit` and accept.

**Verify.** `tests/test_verify.py`: a deliberately over-planted plan (75 tiles, 3 hands) → verifier prunes plants until 0 survival misses; a feasible plan → 0 pruning, `blocked_ops == 0`. Mutation check: break the router (skip last task) and confirm the verifier reports it.

**Done when:** on the Boatlee-like plan, `verify_day` reports 0 missed survival waters on every day of 20 seeds, and compile+verify ≤ 100 ms per day.

**Audit correction (E62):** that holds in the shadow model only. With the pruned/overcommit counters
wired to the harness, live `boatlee_like` prunes ~200 tasks/season over 1.4 overcommit days and
loses ~2.6 plants/game to thirst (E60). Carried stock is now decremented per claim (two tasks could
previously spend the same unit in one hand).

## C4 — Agent shell · M

**Build.** `agent/main_v4.py`: `agent(obs)`:
- State reset on `step == 0`. At **hour 0**: farmer script for hour 0 only (usually PICKUP/DROP/PASS), market: HIRE × n, BUY_LAND if `day == plan.land_days[q]`, BUY_SEED for cohorts due (bulk, ahead), BUY_ANIMAL per herd schedule (only if a pasture is empty or being built today), BUY_PRODUCT WHEAT/FERTILIZER = today's need − shed, SELL per market rules.
- At **hour 1** (hands exist, positions known): C1 → C2 → C3 → per-unit 23-turn scripts cached; each subsequent hour emits the cached op. Cheap **mid-day repair**: if a scripted op is invalid at its tile (weed appeared/plant died — cannot happen mid-day; but a PLANT may have failed on atomic seed rule) log `blocked_ops` and PASS; the next dawn re-plans anyway.
- Market rules: sell all shed stock at hour 0 (and after DROPs at hour 23), except goods in `sell_floor` (binary-search quantity to keep price ≥ floor × base) and except a `frontrun` reservation (O3); shed-pressure valve at `release_pressure`; terminal spread from step 700.
- Fallback: any exception → all PASS + empty market (never crash), counted.

**Verify.** `harness` with `plan=Plan.boatlee_like()`, 20 seeds vs starter: runs to 719 without exceptions; `blocked_ops` per season ≤ 12 (Boatlee's own is ~10, from weeds); p99 per-turn latency < 100 ms through the reference env, both seats.

**Done when:** all three hold.

**Audit correction (E62):** the market rules were partly prose. Now live: `release_pressure` valve
(dormant on `boatlee_like` vs starter — shed peaks 67 < 70, $0 ± $0 — but +$3,431 ± $1,427 on a
hoard plan, 1.2 fires/game vs real boatlee), and the dawn `orders[:10]` truncation that was silently
dropping sells (`maxMarketOrdersPerTurn=10` is real, `kaggriculture.py:551,560`; ~6.9 truncated
dawns, ~10 sells deferred per season). Still simplified: terminal spread is dump-all, and `BUY_SEED`
is on-demand rather than bulk. Front-run reservation awaits O3.

## C5 — Phase-1 gate · S (measurement only)

vs `starter`, 80 games both seats, Boatlee-like plan:

| metric | target | Boatlee reference (estimated) | Boatlee **measured** (I2, 240 games) |
|---|---:|---:|---:|
| steps per useful action | ≤ 1.15 | 1.01 | 1.02 |
| plants lost / season — **thirst** | ≤ 10 | 6–7 | **0.0** |
| plants lost / season — old age | (not a defect) | — | 10.0 |
| strawberry / plant | ≥ 6.5 | 7.7 | 7.9 |
| milk / cow-day | ≥ 1.2 | ~1.3 | 1.11 |
| unharvested ripe at end | ≤ 10 | — | 2 |
| solo money | ≥ 110k | 128–150k | 142,555 (vs pool) |

**Read "plants lost" as thirst deaths.** The harness separates them because they mean opposite
things: an old-age loss is a plant that paid out and expired, a thirst loss is one the routing
failed. Boatlee loses ~10/season and *all* of it is old age — so a gate of "≤ 10 lost" passes for an
agent that kills ten plants through bad routing, which is the failure the gate exists to catch. The
milk figure is measured over every cow-day including the seven before first production; a
productive-days denominator would give the plan's ~1.3.

**Kill (fixed now):** two routing iterations and still `plants_lost > 20` or `steps_per_useful > 1.4` → the compiler idea failed; ship F-gate fallback (+ relay-sell if provenance is cleared) and stop this track. Otherwise proceed to S.

**Measured (E66)** — 45000:45040, 80 games both seats, fingerprint `1dafaa205072`. PASS steps/useful 0.79 · thirst 2.5 · blocked_ops 1; FAIL straw/plant 5.6 · milk/cow-day 0.60 · ripe@end 28 · money $89,621. vs boatlee 0/80 ($50,036 / $139,465). **Kill does not fire.** The shortfall is **execution, not the plan** — E60 refuted. Four mechanisms, in fix order: (1) herd placed day 18.6 vs 7.7 ≈ **$45k** (milk/wool/fertilizer share one cause; husbandry is clean) — empty-pasture cash floor + `_feedable_animals` throttle (`agent/main_v4.py:342`); (2) strawberry **tick-day** watering 67.0% vs 98.6% voids the +2 bonus (`kaggriculture.py:797`) ≈ **$7k** — wrong days, not too little water; (3) wheat base 1–10 vs steady 13 ≈ **$17k**; (4) ripe@end is 16.6 wheat (harmless) + 9.2 day-29 strawberry (~$2k). Only then re-ask the plan question in S.

**Re-gated (E69)** — 50000:50040, 80 games both seats, fingerprint `ab86c6175b7b`, after the E67 (tick-day water, herd 1:1 pacing) and E68 (wheat wave) fixes. **5/7 PASS**: steps/useful **0.76** · thirst **1.14** · straw/plant **6.77** (was 5.6 FAIL) · blocked_ops **1.6** · **solo money $114,302** (was $89,621 FAIL, **+$24,681**); **80/80 vs starter**; **kill still does not fire** (1.14 / 0.76 vs bars 20 / 1.4). Counters checked first: tick-day watered 97.6% (boatlee 98.6), doubled 95.2%, first animal placed_day 2.0, cow-days 128 (boatlee 174), wheat steady 6–7 tiles, fallbacks 0. Residual FAILs: **milk/cow-day 0.73** is a **cow-first defect** — first COW day 11 (7 by d16) vs boatlee day 1 (7 by d8), because 1:1 pacing takes the `plan.herd` prefix (4 sheep) and the cash gate binds on the $400 cow, plus a dead day-25 batch (6.9 ordered, 0 productions); fix in flight, expected +$15–17k. **ripe@end 26.7** is largely a counter artifact: 17.4 wheat at ~$0 (the compiler eats its wheat, E68) + 7.2 day-29 strawberry (~$1.5–2k genuine). vs boatlee $39,405 / $129,021, 0/80 — **not an execution regression**: boatlee's presence closes the markets (shed discards 78.5 units/game, valve saturated 10.1 fires/168 units, plants_started 97.6 vs 195.9). **Verdict: one more execution round (cow-first), then proceed to C6/S** — the remaining head-to-head deficit is plan-bound (cohort count, wheat scale, sell pressure) and those are **S-track genes**; do not spend more execution rounds on it.

**Closed (E70)** — the cow-first round landed: payback cutoff (COW 20 / SHEEP 22 / GOOSE 22, derived from `kaggriculture.py:822-829`) kills the day-25 batch (`animals_ordered_d25` 5.8 → 0), `_yield_per_dollar` ranks COW first ($0.60/day/$, $400), `cow_start` 5 → 1 (boatlee buys COW 1 + SHEEP 4 at day 0 hour 0). First cow `placed_day` 11.7 → **2.01** (target ≤4 PASS), cow-days 126.8 → 152.9, milk/cow-day 0.77 → **0.90**. Held-out 51000: **+$6,651 ± 5,629** vs starter (≈**+$7.7k ± 2.5k** over 400 games), **+$10,508 ± 7,038 vs boatlee — no regression**, gap −$93.5k → −$84.0k, still 0/80. Guardrails clean (blocked_ops 1.9, thirst 0.9, steps/useful 0.75, fallbacks 0); 601 tests, `make verify` 0 divergences. Refuted along the way: `cow_start=0`+sheep day 2 **−$96k** (`plants_started` 53), no-animals-day-0 **−$30.7k**, 2/day pacing −$18k…−$47k — the herd failure mode is buying too **many** early, not cows before sheep. **milk/cow-day 0.90 vs gate 1.2 is now plan-bound, not execution**: wallet $4–$1,000 on days 4–9 gates cows 2–9, and the plan has no early-cash crop → S-track gene. **The C5 execution arc is CLOSED** (arc lift ≈ +$24.7k + $7.7k, solo ≈ $120k vs the 110k gate); **proceed to C6/S**. Only open execution item: the HIRE/dawn-queue defect (E68) — **closed by E71** (kept as `_dispatch` ranking + `dawn_starved` tripwire; value was substrate-dependent, +$29.4k pre-C6 → ≈$0 on the final tree).

---

# Track S — Search (Phase 2)

## S1 — Fitness and pool · S

**Build.** `search/fitness.py`: `evaluate(vec, seeds, opponents) -> dict(win_rate, mean_margin, solo, counters)`. Fitness = `0.7·win_rate_vs_pool + 0.3·normalised margin`, pool = {boatlee ×2 weight, flooder, executor_v7}, both seats, `seeds` = 6 per candidate during search, rotated per generation (never the acceptance block). `flooder` = our own compiler with a plan that over-plants strawberry and cows (built once from `Plan.boatlee_like()` with 2× strawberry cohorts); `tomato_rusher` = a plan that always plants 15 tomato tiles at day 6 — in-family exploiters so the search can't overfit to Boatlee alone or assume the hinge products are ours by default.

**Verify.** `evaluate(encode(Plan.boatlee_like()))` reproduces C5's numbers ± noise; a deliberately bad plan (no pastures) scores near 0.

## S2 — Genome search · M

**Build.** `search/run.py`: two-level — outer GA over discrete/layout genes (population 48, tournament selection, mutation ops = move a cohort by ±1–2 days, ±3 tiles, swap a pasture tile, ±1 animal, ±1 hand on a day, toggle SE), inner CMA-ES (`pip install cma`, full covariance) over the continuous consts (sell floors, release pressure, front-run lead, hand-value threshold) for the top-8 genomes each generation. Log every evaluation to `search/log.jsonl`. Elitism 4. Acceptance: a candidate replaces the incumbent only if it beats it on a **fresh 80-game block** with the CI clear of 50%.

Budget: 48 × 12 games (6 seeds × 2 seats) × 4 opponents ≈ 2,300 games/generation → ~25 s/generation on FastSim (8 cores) or ~10 min on the reference env. Run 50–100 generations.

**Verify.** Search finds a plan that beats `Plan.boatlee_like()` on the acceptance block; the winning genome is *different* in at least herd or cohorts (else it's noise); counters stay inside C5's targets (a plan that "wins" by killing plants is rejected).

## S3 — Phase-2 gate · S
Best plan vs `boatlee`: ≥ 60% both seats, 80 games, fresh block; solo ≥ Boatlee's on the same seeds; no regression vs pool. **Kill:** search cannot beat the hand plan by CI → the genome is mis-parameterised (add/remove genes, e.g. per-quadrant crop rows), not the optimiser; iterate once, then re-gate.

---

# Track O — Online adaptivity (Phase 3)

**Next (E77): the O-track is the active path to the boatlee matchup.** Two full genome rounds closed
0/80 with the gap only narrowing (−82k → −67k → −60k) and round 2's narrowing coming from *denial*
rather than production; the deficit is structural to the choreography, so the remaining lever is the
one that changes the **shape** of the matchup — **O2** fingerprint, **O3** front-run / counter-mix /
slot alignment, **O4** gate.

## O1 — Shop-draw branch points · M

**Build.** Because C4 re-compiles daily, adaptivity is just plan *patches* applied when a condition first holds: `Branch(day_from, cond, patch)`, e.g. `("YARN_STORE" in shops, +2 sheep instead of cows on the next 2 buys)`, `(count(strawberry shops) ≥ 2 by day 9, +1 strawberry cohort in NE)`, `(no milk shop by day 12, cap cows at 6)`. Conditions read `obs.town.unlocked_shops` counts (with replacement — count instances). Patches must be *forward-only* (never dig a live plant; never sell an animal). Search these thresholds in S2 (they're genes).

**Verify.** Effect counter `branches_fired` per game; on a seed set split by draw (yarn/no-yarn, ≥2 strawberry shops/other), the branch fires only where it should. Gate: on the **disagreement subset** the branched plan beats the fixed plan by CI (this is E35's kill criterion, now at scale where it's untested).

## O2 — Opponent fingerprint and forecast · M

**Build.** `agent/opponent.py`:
- `fingerprint(opp_farm, step)`: at steps 24, 48, 72 compare tile census (crop counts, pasture count, unlocked quads, hands) with a library of known scripts (Boatlee's table simulated forward; our own past submissions). Distance ≤ threshold on all three checkpoints → `known = "boatlee"` (Boatlee's own RC2 relay does exactly this to detect us).
- `forecast_supply(opp_farm)`: per product, units by day-to-maturity from `planted_day`/`crop`/animal type — the E7 forecaster.
- If known: `sell_schedule = script's SELL steps per product`.

**Verify.** vs Boatlee: fingerprint locks by step 72 in ≥ 95% of games and never locks vs starter/executor (0 false positives on 40 games).

## O3 — Front-run and counter-mix · M

**Build.** Two overlays on C4's market and plan:
- **Front-run**: for each product P we hold and the opponent is scheduled (known) or forecast (unknown, ±1 day) to sell at step s, our SELL of P moves to `s − lead` (lead ∈ 6–12, searched) — season-long, not just terminal. Effect counter: sells moved / game.
- **Counter-mix**: cohorts not yet planted avoid products where the opponent's forecast supply for our harvest window exceeds a threshold; prefer wheat/melon-cap/eggs. Effect counter: cohorts redirected.
- **Slot alignment** (audit): our market list is processed slot-by-slot in lockstep with theirs; against a fingerprinted script we know their slot order at each step, so same-product sells go in an earlier slot than theirs. Effect counter: sells re-slotted; verify by replaying a Boatlee step where they sell strawberry in slot 2 and asserting our slot-0 units realise higher quotes.

**Verify.** vs Boatlee, 80 games: front-run alone ≥ +$300/game (relay-sell's ceiling is the floor here, since we hit every dump not just the last); counter-mix alone must not lower solo by > 3%. Together: O4.

## O4 — Phase-3 gate · S
vs `boatlee` **≥ 80% both seats, ≥ 80 games, fresh block**; vs pool no regression (each CI not below the S3 result); p99 turn < 100 ms in both seats on the reference env; `blocked_ops` ≤ 12; every overlay's effect counter non-zero in the combined run.

---

# Track L — Ladder

## L1 — Submission bundle · S
`main.py` (agent shell) + `agent/*.py` + `plan.json` in a tar.gz; no `__file__`, stdlib-only imports; smoke test through the *reference loader* (`env.run(["submission.tar.gz"...])` equivalent used in E21) on both seats. Done when both seats finish with money > 3000 and no timeout on a throttled CPU (run under `taskset`/nice).

## L2 — Read the ladder · ongoing
`kaggle competitions episodes/replay/logs`: for each opponent, run `fingerprint` offline; count Boatlee-clones vs others; add distinct opponents' replays to the pool as `replay_agents` (open-loop replays of their unit actions are a fine stand-in for pool diversity); re-run S2 against the enlarged pool.

---

# Measurement rules (carried from the kagsim line, applied here)
- ≥ 80 games, both seats, fresh block; act on nothing whose CI includes 50%.
- Print counters with every money number; a zero effect counter means "not implemented", not "refuted".
- `blocked_ops` is the desync/health counter for the compiler; it must stay near Boatlee's ~10.
- One change per run; conjunction tests explicitly (F1–F4 together; O3's two overlays together).
- Pin the env; re-verify I1 parity on any hash change.
- Registry names, never throwaway scripts; every result appended to `results/*.jsonl`.

# Order of work (first two weeks)
Days 1–2: I0, I2, F1–F4 (fallback in hand). Days 2–3: I1. Days 3–7: C0–C4, C5 gate. Days 8–10: S1–S3. Days 10–14: O1–O4, L1. Then L2 loop.
