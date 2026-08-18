# Experiments

Measured results. **This file is the record; `PLAN.md` is the plan, and where they disagree the
record wins** (`docs/decisions.md` D15).

| # | subject | outcome |
|---|---|---|
| E1 | crop / land / labour sweep | ⚠ **partly superseded by E15** — its product ranking used a one-shot capacity model |
| E2 | engine bugs found by measurement | cash management, wheat churn, unfeedable animals |
| E3 | goose thesis, first test | invalid run (cash starvation), later re-tested |
| E5 | arena vs `starter` | **ranking inverts** under real competition |
| E6 | routing vs idleness | routing was the wrong target; land is a loss; a dead gate found |
| E7 | opponent supply forecasting | worked at the time; **superseded by E11** |
| E8 | CEM overfitting | +$34,207 held-out was worth $207; wrong opponent inherited from a default |
| E9 | CEM against the real champion | validated win, 64/64 |
| E10 | exploiters | champion lost **0/80** to a naive dumper |
| E11 | dump-only ablation | the entire market-timing apparatus **lost to pinning it to zero** |
| E12 | livestock gate | animals were disabled by `goose_min_cash`, not rejected |
| E13 | audit against the official rules | 2 bugs, 1 refuted payback argument, strawberry refuted |
| E14 | independently designed opponents | champion beat all 4; **egg headroom is action-bound** |
| E15 | **cows and sheep** | **the capacity model was wrong**; champion $37k -> $66k |
| E16 | market regime | we never leave **scarcity**; holding loses because cash compounds |
| E17 | production audit | E6 **reversed**: units are movement-bound, not idle; priorities were harmful |
| E18 | **promotion gate** | five champions in a row were wrong; promoting on 24-64 games could not resolve the 3-8pp edges being acted on |
| E46 | **read boatlee's plan instead of guessing**: its units are 93% role-pure | role specialisation replicates on 4 seed blocks (~70%, n=360) -- first replicated gain of the session |
| E45 | planting **rate** vs late planting; and why tuning the champion cannot work | burst-planting kills cohorts at age 2; the champion is a local optimum every change breaks |
| E44 | **matched their farm on every dimension at once** — and it dies | structure reproduces exactly; 50 plants lost to 7; the residual is movement, 54.2% vs 42.8% |
| E43 | strawberry investigated properly: **melon is right for our engine** | fertiliser needs same-day watering; the pair helps a bad config and hurts the good one |
| E42 | the scaling config **also** evaporates on fresh seeds | +46% at n=36 becomes +4% and a head-to-head loss at n=80; nothing below ~80 games is believable |
| E41 | **E37 does not replicate**; land is the dominant factor; our marginal crop is negative | 8-seed block was an outlier (n=80 says parity at equal land); mix only matters at scale |
| E40 | the gate **refused** optimal assignment: more money, fewer wins | mean +$1,642 but median -$424 in the mirror; money and winrate disagree |
| E39 | **optimal assignment beats greedy** — the first engine change this session to gain money | greedy was 9.6% off optimal (corrected from 18%); solving it exactly is +4-11%, and it replaces P1's search |
| E38 | P1.1 value function: **the pre-set gate was mis-specified** | it fails rho>=0.6, ranks nearby futures at 75%, and the gate would have tuned it to its worst setting |
| E37 | **equal-land comparison validates P1** | same 25 tiles: they do 4.3x our crop work and lose 1 plant to our 7 |
| E36 | **P0 done**: forward model, 31 parity tests, 14/14 mutations caught | 208k steps/s (104x criterion); mutation testing found 4 tests that proved nothing |
| E35 | **P1.5-A killed**: shop-adaptive mix loses on its own kill subset | choosing *what* to grow does not pay while we can only service 10-15 tiles |
| E34 | E26 re-run under 1.32.6: **"land is sufficient" is refuted** | stripped of land AND fertilizer they still beat us; the mix is now an independent deficit |
| E33 | **rules changed (1.32.6)**: demand cut 4.7x, shops drawn with replacement | drift guard fired and worked; kagsim updated and re-verified; WOOL has no buyer in 36% of games |
| E32 | P0 feasibility: pure-Python farm model at **572k-836k steps/s** | 286x the kill criterion; PLAN2 P1 (rollout search) is affordable |
| E31 | throughput by layer: **sim is 2ms, Python is the rest** | ~100x more search available than CEM used; motivates PLAN2 P0/P4 |
| E30 | **CEM vs the real opponent: 0% in all 14 generations** | re-search ruled out; the executor, not the parameters, is the constraint |
| E29 | where the turns go: **35% hauling vs their 7%** | 3 routing fixes tried, all measured worse; diagnosis solid, cure not found |
| E28 | **THE ANSWER: 1.84 steps walked per useful action vs their 1.01** | complete causal chain from routing to the 0-24 loss |
| E27 | external agent in the arena; **arena ran 718 turns, not 719** | kagsim proven bit-exact against a third-party agent; every prior score was a turn short |
| E26 | head-to-head vs ablations: **land alone decides it** | remove land, we win 24/24; remove fertilizer, we still lose 0/24 |
| E25 | **ablating their agent**: land +$106k, fertilizer +$54k | strip both and they score *below* our champion -- that is the entire gap |
| E24 | FERTILIZE implemented; combined strategy **refuted** | the constraint is unit throughput, not missing verbs |
| E23 | **why we lose**: 63 crop tiles vs our 10 | land is not the edge, it is the *enabler*; 4 engine defects found |
| E22 | external opponent: we lose **0-40** | first non-self-referential measurement; strawberry is the gap |
| E21 | two verification-surface bugs | submission scored $3,000 on Kaggle's real loader; kagsim was made to diverge by a bad test |
| E20 | land, 4th re-test | **confirmed dead** — the only early conclusion that survived every re-test |
| E19 | wool saturates | sheep made one market floodable, so **reserves came back** — E16 was true only before sheep existed |

**Reading order for someone new:** E15 first (it corrects the economics everything else assumed),
then E10/E11 (what wins), then E5/E8 (how the measurements themselves went wrong).

Reproduce: `PYTHONPATH=. python tools/experiments.py`

## E1 — Crop, land, and labour sweep (T0.7)

> **⚠ Partly superseded by E15.** The *measurements* below stand; the *product ranking* they were
> read against does not. E1 judged markets by a one-shot price-curve integral, which ranked milk,
> wool and strawberry as traps and melon as best — almost exactly inverted. The melon-first
> conclusion here is a local optimum of a wrong model, and cost a 2.4x improvement.

Mean over 6 seeds of the scripted engine vs. `starter` ($3,495). **Run after the cash-management
and feed-loop fixes** (E3); the pre-fix numbers are kept in E4 for comparison.

| config | mean $ | sd | wins | move% | weeds@d29 | top revenue |
|---|---:|---:|---:|---:|---:|---|
| **melon + wheat** | **45,171** | 394 | 6/6 | 39 | 1 | MELON $41,117, WHEAT $7,513 |
| melon | 42,199 | 0 | 6/6 | 31 | 0 | MELON $47,539 |
| melon + wheat + geese | 32,809 | 1,395 | 6/6 | 55 | 2 | MELON $26,026, WHEAT $23,742, EGG $7,626 |
| wheat + geese (**CARE**) | 17,644 | 755 | 6/6 | 60 | 2 | WHEAT $25,738, EGG $9,022, FERT $8,335 |
| wheat | 12,006 | 1,267 | 6/6 | 46 | 0 | WHEAT $12,685 |
| wheat, no land | 12,006 | 1,267 | 6/6 | 46 | 0 | WHEAT $12,685 |
| wheat + geese (alt feed) | 10,013 | 2,350 | 6/6 | 60 | 1 | WHEAT $27,377, FERT $7,580, EGG $3,702 |
| carrot | 8,084 | 2,996 | 6/6 | 52 | 0 | CARROT $11,923 |
| harvest early | 7,686 | 754 | 6/6 | 53 | 0 | WHEAT $10,075 |

**Phase 1 target was $20k; best config reaches $45k.**

### Answered

- **The melon rush is confirmed, decisively.** Melon-based configs make $42–45k against $12k for
  wheat — roughly 3.5×. Melon revenue of $41–47k also exceeds the $26.5k one-shot ceiling computed
  from the price curve, because reservation pricing (only selling while the marginal price clears
  55% of base) lets the town's ~8/day melon drain regenerate the price between waves. **Paced
  selling is worth more than the whole rest of the crop lineup.**
- **CARE beats alternate-day feeding, decisively**: $17,644 vs $10,013, with egg revenue $9,022 vs
  $3,702. Care roughly doubles egg output for one extra action per animal per day, and that is
  worth more than the actions saved by feeding every other day.
- **Geese are worth ~+47% when the action budget is free** ($17,644 vs $12,006 for wheat alone),
  but **cost 27% when combined with melon** ($32,809 vs $45,171). They compete for the same scarce
  unit-actions. Goose count has a sharp optimum around 8 — at 16 and 25 the feed loop cannot keep
  up and the flock dies:

  | goose target | 0 | 8 | 16 | 25 |
  |---|---:|---:|---:|---:|
  | final $ | 13,546 | **16,831** | 4,158 | 141 |
- **Harvest at max yield, not first yield** — `harvest_early` costs ~36% of final money.
- **Carrot ≪ wheat** ($8.1k vs $12.0k), matching their market capacities.

### Confirmed

- **Travel dominates.** 31–60% of unit-actions are movement, and the best configs are the ones with
  the *lowest* movement share. Routing remains the top structural lever.
- **Labour optimum ≈ 8 hands.** Fibonacci cost reaches ~$376/day at 12 and ~$2,583/day at 16;
  twelve hands bankrupted the pre-fix engine.

### Overturned — `PLAN.md` was wrong

- ~~**"Buy all three quadrants ASAP."**~~ Land is now gated on *labour headroom* rather than cash,
  and the wheat config never buys it at all — `wheat` and `wheat, no land` are byte-identical.
  25 reachable tiles beat 100 unreachable ones. Land is a consequence of solving routing, not a
  precondition for it.
- ~~**"Hire aggressively."**~~ Only to ~8 hands; see above.

## E2 — Bugs found and fixed this round

Three defects, each of which silently produced a *plausible-looking but wrong* experimental result.
This is the case for measuring rather than reasoning: every one of them would have been invisible
in a spreadsheet.

1. **No cash management** (fixed). `Engine.market` spent greedily with no floor and in the wrong
   order — land first, seeds last — so it hit $0 on day 0 and stayed there. Now: sells are issued
   first (order slots commit sequentially, so proceeds fund purchases *in the same turn*), then
   spending proceeds in ROI order — seeds, feed, hands, geese, land — against an explicit budget
   with a `cash_floor`. **Melon went $5,144 → $42,199.**
2. **Wheat churn loop** (fixed). The buy side reserved `(animals + structures)` wheat for feed but
   the sell side only held back `animals`, so every turn the engine bought feed and immediately
   sold part of it back. Netted zero money while burning market slots — and inflated reported
   wheat revenue to $117,947, well past what that market can absorb.
3. **Animals could never be fed** (fixed). FEED tasks required a unit to already be carrying wheat,
   but nothing ever sent a unit to fetch any, so the flag was permanently unassignable. Trace of
   the pre-fix run: 11 geese placed on day 17, **9 dead by day 18**. Now the task generator emits
   explicit `PICKUP WHEAT` trips at the shed whenever animals are unfed.

Still outstanding:

- **Routing is naive** — sticky greedy nearest-task, no clustering. At 31–60% travel share this is
  the largest remaining structural lever (T1.2).
- **Land is never bought** under the new labour-headroom gate. Correct at current routing quality,
  but it means 75 tiles are permanently idle — solving routing should re-open that decision.

## E3 — Goose thesis: now tested, and it holds

The pre-fix goose rows ($550 / $439) were **not evidence about geese** — no animal was ever placed.
After E2.1 and E2.3, geese are worth **+47%** on top of a wheat economy ($17,644 vs $12,006), with
$9,022 of egg revenue and $8,335 of fertilizer. The fertilizer stream is as valuable as the eggs,
confirming that the unconditional per-animal fertilizer drop is a real and underrated line.

The thesis needs one qualification `PLAN.md` did not anticipate: geese only pay when unit-actions
are otherwise idle. Against melon they *lose* 27%, because both compete for the same scarce
resource — actions, not money.

## Still open

- Whether the 100-item shed cap binds at scale
- Staggered vs. single-wave melon planting (reservation pricing already recovers much of this)
- Front-runner exploitability (`PLAN.md` §2.5) — now unblocked, since a $45k melon producer exists
  to steal from
- Harvest-early-into-shed as private optionality vs. harvest-at-max-yield
- Whether solving routing (T1.2) makes land purchase profitable again


---

## E5 — The arena inverts the E1 ranking (T0.8)

E1 scored each config against the fixed `starter` baseline. The arena plays them **against each
other**, both seat assignments, 32 games per pairing.

```
agent               skill        winrate (95% Wilson)      mean $
melon               60.37  100.0% [98.0, 100.0] n=192      38,009
melon-wheat         47.03    83.3% [77.4, 87.9] n=192      38,498
melon-wheat-geese   33.69    66.7% [59.7, 73.0] n=192      28,279
wheat               17.89    44.8% [37.9, 51.9] n=192      11,078
wheat-geese         16.35    38.5% [31.9, 45.6] n=192      14,335
carrot               1.45    16.7% [12.1, 22.6] n=192       7,724
starter            -21.90     0.0% [ 0.0,  2.0] n=192       3,463
```

### `melon` beats `melon-wheat` 32/32, despite E1 ranking it second

| | vs. `starter` (E1) | head-to-head (E5) |
|---|---:|---:|
| `melon` | $42,199 | **$28,329** — wins 32/32 |
| `melon-wheat` | **$45,165** | $18,058 |

Against a weak fixed opponent, `melon-wheat` earns 7% more and looks like the best config. Put the
two in the same game and `melon` wins **every single time**, and `melon-wheat`'s earnings collapse
by 60%.

**Why:** they compete for the same melon market. `melon-wheat` splits its tiles, produces fewer
melons, and loses the race to sell into the good part of the price curve. Wheat revenue does not
compensate, because wheat is worth ~$12k against melon's ~$26k of high-price capacity.

Note also that *everyone's* money falls under competition — `melon` drops $42k → $28k. The shared
market means a strong opponent destroys value for both players, so absolute money is not
comparable across opponent pools at all.

### Why this matters beyond crop choice

This is a direct, measured confirmation of `PLAN.md` §2.5: **the market is the adversarial core,
and evaluating against a weak fixed opponent gives systematically wrong rankings.** Any tuning
done against `starter` — including all of E1, and any future CEM run — optimizes the wrong
objective unless the opponent is competitive.

It is also the concrete version of 7th place's rule: *always trust the tournament.* E1's ranking
looked sound, was measured over 6 seeds, and was wrong.

**Consequences adopted:**

- T2.1 (CEM) must evaluate against the current best agent, never against `starter`.
- The mean-money column stays in the report, but **skill and pairwise winrate are the ranking**;
  mean money is only comparable within a fixed opponent field.
- The front-runner experiment (`PLAN.md` §2.5) is now clearly the highest-value open question,
  since head-to-head melon competition is exactly the setting it probes.


---

## E6 — Routing was the wrong target; the market is the constraint (T1.2)

T1.2 was scoped as "routing rework" because 31–60% of unit-actions were movement. Measuring the
op mix first changed the task entirely.

### Units are not travel-bound, they are idle

| op | `melon` | `wheat` |
|---|---:|---:|
| **PASS (idle)** | **58.3%** | 38.0% |
| movement | 30.6% | 45.6% |
| useful work | 11.1% | 16.4% |

Better routing cannot help an agent whose units already have nothing to do for most of the season.

### A dead branch: land was never actually being bought

The land gate compared occupied tiles against total labour capacity:

```python
capacity = (1 + hire_max) * tiles_per_unit     # (1+8)*6 = 54
if worked >= capacity * 0.9:                   # needs 48.6 of 25 unlocked tiles
```

Only 25 tiles are ever unlocked, so the condition was unreachable and `BUY_LAND` never fired.
**E1's "land is harmful" conclusion was therefore measured on a config that never bought land** —
it was accidentally right for the wrong reason. Fixed to compare against *owned* land, gated on
labour capacity exceeding it.

### With the gate fixed, land still loses — and now the reason is clear

| melon config | money (self-play) | quadrants | travel |
|---|---:|---:|---:|
| no land | **16,573** | 1 | 31% |
| land, `tiles_per_unit=6` | 5,261 | 3 | 48% |
| land, `tiles_per_unit=10` | 148 | 4 | 58% |

Head-to-head, `melon-wheat` beats `melon-wheat-land` **32/32** ($29,060 vs $21,446).

**Melon saturates its own market at ~25 tiles.** Its revenue of ~$21–24k already captures most of
the ~$26.5k the melon curve absorbs. Extra tiles produce units that cannot be sold profitably,
while raising travel from 31% to 58% and costing $7,000. The binding constraint is **market
absorption, not land and not labour.**

### Arena, land default corrected

Every engine config had inherited `buy_land=True` from the dataclass default, so an earlier run
had *all* agents buying land and `melon-wheat` vs `melon-wheat-land` tied identically — the tell
that caught it. With the default set to `False` (measurement-backed) the field is transitive:

```
melon               46.21  100.0% [98.0, 100.0]      32,750
melon-wheat         30.23    83.3% [77.4, 87.9]      33,845
melon-wheat-land    16.74    66.7% [59.7, 73.0]      31,367
melon-wheat-geese   -0.73    50.0% [43.0, 57.0]      26,151
melon-geese         -3.03    33.3% [27.0, 40.3]      23,888
wheat              -17.20    16.7% [12.1, 22.6]      11,292
starter            -30.96     0.0% [ 0.0,  2.0]       3,499
```

### Winning is denial, not revenue

`melon` wins **every pairing**, yet earns *less* mean money than `melon-wheat`
($32,750 vs $33,845). Head-to-head:

| | vs `starter` | vs each other |
|---|---:|---:|
| `melon` | $42,199 | **$28,329** — wins 32/32 |
| `melon-wheat` | $45,165 | $18,058 |

`melon-wheat` is the better *earner* against a passive opponent and the worse *competitor*. Facing
a full-melon rival it loses the race into the high-price part of the curve, and its earnings fall
60% while wheat revenue fails to compensate. Scoring is relative, so denial beats revenue.

**Consequences adopted:**

- **T1.2 (routing) is deprioritized.** Units are demand-bound, not travel-bound. Revisit only if a
  future config makes labour binding again.
- **The market/selling policy is the real lever**, which is exactly what `PLAN.md` §2.5 predicted
  and what T3 should put the model on.
- `buy_land` defaults to `False`, with the measurement cited in `agent/params.py`.
- Any conclusion drawn from self-play money or from play against `starter` must be re-checked in
  the arena; both mis-rank here.


---

## E7 — Pricing against the opponent's visible supply (T1.3)

E6 established that scoring is relative and the melon race decides it. This tests the direct
consequence: **price against what the opponent is about to harvest, not against a static reserve.**

`farms` is shared and crop maturity is deterministic, so their entire production schedule is
computable from public information. Their shed is private, so the forecast covers incoming supply
only. The policy blends the static reserve toward the price expected once that supply lands, minus
what the town will drain in the meantime:

```
reserve_eff = (1 - w) * static_reserve + w * price(inv + opponent_incoming - town_drain * horizon)
```

### Result

The final default is **`forecast_weight = 1.0`, `forecast_horizon = 10`**, which wins **100%** of
its pairings against the whole field including the pre-T1.3 best:

| config | skill | winrate (95% Wilson) | mean $ |
|---|---:|---|---:|
| **melon, w=1.0 h=10** | **46.77** | **100.0% [98.4, 100.0]** | 27,805 |
| melon, w=0.6 h=4 | 19.00 | 80.0% [74.5, 84.6] | 31,033 |
| melon, w=0 (pre-T1.3 best) | 17.29 | 60.0% [53.7, 66.0] | 29,263 |
| melon-wheat | 0.26 | 40.0% [34.0, 46.3] | 8,622 |
| wheat | −14.71 | 20.0% [15.4, 25.5] | 4,700 |
| starter | −31.50 | 0.0% [0.0, 1.6] | 3,501 |

At `w=1.0` the static reserve is discarded entirely, leaving one interpretable rule: **sell while
the marginal price is at least what the price will be once the opponent's visible supply lands.**
Horizon 10 is melon's plant-to-harvest cycle, so the forecast sees exactly one full wave.

Note again that the winner earns *less* mean money than two configs it beats 100% of the time
(E6's denial finding, reproduced).

### The two knobs interact, and coordinate-wise tuning misled twice

| | h=4 | h=10 |
|---|---|---|
| best weight | 0.6 | **1.0** |
| w=1.0 | **worst config of all** | **best config of all** |

Tuning the weight at a fixed horizon of 4 produced "the optimum is interior, w=1.0 sells too
cheaply" — true at that horizon, and false at the one that matters. A joint grid was needed:

| | h=10 | h=14 | h=20 |
|---|---:|---:|---:|
| w=0.70 | 66.7% | 44.4% | 22.2% |
| w=0.85 | 66.7% | 44.4% | 9.7% |
| **w=1.00** | **88.9%** | 88.9% | 12.5% |

`melon-static` (no forecast at all) scored 55.6% in that grid — **better than five of the eight
forecast settings.** A mis-tuned forecast is worse than none.

**This is the argument for T2.1 being a joint search (CEM/CMA-ES), not a sequence of one-knob
sweeps.** Two separate coordinate-wise passes both landed on the wrong point.

**Caveat.** `town_drain_per_day`
assumes the default intervals because the environment configuration is not exposed in the
observation — under a non-default `turnsPerDay` or `townShopSellInterval` the drain estimate is
wrong, though the sign of the effect is not.


---

## E8 — CEM overfits its search seeds (T2.1)

Joint cross-entropy search over all 29 knobs, fitness = money margin against the reigning champion
(never `starter`, per E5), common random numbers within each generation.

The first run looked excellent:

```
gen  0  best margin $  +28,323  winrate 100.0%
gen  5  best margin $  +32,524  winrate 100.0%
gen  9  best margin $  +34,013  winrate 100.0%
```

It then placed **third** in the arena on fresh seeds, behind the champion it had supposedly beaten
by $34k:

| agent | skill | winrate (95% Wilson) |
|---|---:|---|
| melon (champion) | 41.82 | 100.0% [98.4, 100.0] |
| melon-static | 19.64 | 80.0% [74.5, 84.6] |
| **cem winner** | 18.76 | **58.3% [52.0, 64.4]** |

**Cause: the selection signal was noise.** 16 games per candidate gives a Wilson half-width of
about ±22pp, and CEM ran 24 candidates per generation through that filter — so the elite set was
chosen substantially for seed luck rather than for strategy. Population 24 over a 29-dimensional
space compounds it; the search is badly underdetermined.

This is the failure mode `docs/decisions.md` D10 exists to catch, arriving from the direction that
is easiest to miss: not a change that *looked* sound, but a number that *was* an improvement on
the data it was measured against.

**Fixes applied:**

- **Held-out validation inside the loop.** Each generation re-scores the *updated mean* on a fixed
  seed set disjoint from every training draw, and the run returns the best-validated mean rather
  than the last one. Training score is now reported next to held-out score, so divergence between
  them is visible while the search runs instead of afterwards.
- Larger default population (24) and more games per candidate (32).

**Standing rule:** a search result is a *hypothesis*. It is not accepted until it wins in the arena
on seeds the search never saw.

### The bigger bug: CEM was optimizing against the wrong opponent

Validation-gated CEM then reported a held-out margin of **+$34,207 at a 100% winrate**. In the
arena on unseen seeds that shrank to:

| pairing | winrate (95% Wilson) | money |
|---|---|---|
| cem vs melon | 62.5% **[48.4, 74.8]** | $26,678 vs $26,471 |
| melon vs melon-static | 100.0% [92.6, 100.0] | $22,053 vs $7,386 |

A $34,207 margin during search was worth **$207** in the arena, and the interval includes 50% — by
our own rule, not adoptable.

**Cause:** `run_cem` took its opponent from `Params()`, and the dataclass defaults are
**wheat-based** (`crop_mix = {"WHEAT": 1.0, ...}`). The champion was never melon. CEM spent the
whole search beating a weak agent — the E5 mistake, reintroduced through a default rather than a
decision, and invisible because the code said `champion = Params()`.

This is the second time an implicit dataclass default silently determined an experiment's meaning
(the first was `buy_land`, E6). The pattern is now explicit: **an experiment's opponent, like its
seeds, is part of the experiment and must be named, not inherited.** `--champion` is a required
named registry agent and a typo fails loudly.

### A related trap, found while testing

The fitness inverts on truncated episodes. Melon strategies run cash-negative until their first
harvest around day 10–12, so on a 300-step (~12-day) episode a do-nothing config outscores the
champion by ~$1,500. Shortening episodes to buy search throughput would silently invert the
objective for precisely the strategies worth finding. Pinned by
`tests/test_search.py::test_a_clearly_worse_candidate_scores_lower`.


---

## E9 — CEM against the correct champion: a validated win (T2.1)

Re-run with `--champion melon` (the actual best config) and held-out validation inside the loop:

```
gen  0  train $  +15,221  | held-out $   -5,567 (  0.0%)
gen  4  train $  +30,594  | held-out $  +24,513 (100.0%)
gen  9  train $  +30,230  | held-out $  +27,800 (100.0%)   <- best validated
```

Train and held-out now track each other, and the gap is stable rather than widening — the
signature that the search is finding strategy rather than seed luck.

**Arena on seeds disjoint from both the training draws and the validation set:**

| agent | skill | winrate (95% Wilson) | mean $ |
|---|---:|---|---:|
| **cem** | **48.59** | **100.0% [98.8, 100.0]** n=320 | 34,675 |
| melon (previous champion) | 28.96 | 80.0% [75.3, 84.0] | 23,110 |
| melon-fc06h4 | 12.99 | 60.0% [54.5, 65.2] | 24,054 |
| melon-static | −1.34 | 40.0% [34.8, 45.5] | 23,092 |
| starter | −33.28 | 0.0% [0.0, 1.2] | 3,499 |

Head-to-head it is not close:

| pairing | winrate | money |
|---|---|---|
| cem vs melon | **100.0% [94.3, 100.0]** n=64 | **$30,781 vs $3,190** |
| cem vs melon-fc06h4 | 100.0% [94.3, 100.0] | $23,643 vs $10,787 |

The new champion does not merely out-earn the old one, it **suppresses it to near the starting
bankroll** — $3,190 against a $3,000 start. That is the denial dynamic from E6 taken to its
conclusion: winning the melon race early leaves the opponent nothing to sell.

### What the search actually found

| knob | hand-tuned | CEM |
|---|---|---|
| crop mix | melon only | melon 0.93 + **carrot 0.18, strawberry 0.15, tomato 0.08** |
| hire_max | 8 | **6** |
| tiles_per_unit | 6 | **7.4** |
| forecast weight / horizon | 1.0 / 10 | **0.63 / 9** |
| melon reserve_frac | 0.55 | **0.42** |
| cash_floor | 150 | 235 |
| sell_all_after_day | 28 | **26** |

Two things I had concluded by hand and got wrong:

- **A small non-melon fraction helps.** I had measured melon-only as strictly best and read that
  as "diversification loses" (E6). It loses at a *50/50* split; a ~10-20% garnish does not, and it
  uses turns that would otherwise be idle.
- **`forecast_weight = 1.0` was over-tuned.** The joint search prefers 0.63 with a slightly
  shorter horizon — close to the w=0.6 setting I had rejected at horizon 4, which is a third
  instance of the two knobs interacting in a way single-axis sweeps cannot see.

### Process changes adopted

- `search/champion.json` (accepted, promoted only after arena validation) is now a **separate
  file** from `search/best_params.json` (latest output, a hypothesis). Without the split a search
  silently overwrites the thing it is meant to beat.
- `--champion` defaults to the accepted champion, so successive searches iterate rather than
  restarting from the defaults.


---

## E10 — The champion is exploitable by a naive dumper (T2.2)

The T2.1 champion won 100% of its arena pairings and beat the previous champion 64/64. Six
deliberate exploiters, each a strategy it never faced during its search:

| agent | skill | winrate (95% Wilson) | mean $ |
|---|---:|---|---:|
| **x-dumper** | **42.61** | **100.0% [98.7, 100.0]** | 31,314 |
| champion | 35.82 | 83.3% [78.6, 87.2] | 31,055 |
| x-turtle | 24.00 | 66.7% [61.0, 71.9] | 24,549 |
| x-melon-race | 2.77 | 50.0% [44.3, 55.7] | 18,177 |
| x-frontrun | 0.27 | 33.3% [28.1, 39.0] | 18,992 |
| x-labour | −15.24 | 16.7% [12.8, 21.4] | 13,066 |
| x-hoard | −34.42 | 0.0% [0.0, 1.3] | 10,074 |

Head-to-head, over 80 games on unseen seeds:

**champion vs x-dumper: 0.0% [0.0, 4.6], $17,140 vs $21,300.** The champion loses **every game.**

`x-dumper` is the champion's own parameters with two knobs changed: `forecast_weight = 0` and
every `reserve_frac = 0`. It sells everything the instant it is harvested, at whatever the price
happens to be. No forecasting, no timing, no reserve.

### Why the tuned policy loses to the crude one

The champion holds inventory for a better price. Against a passive or similarly-patient opponent
that is correct, and it is what CEM rewarded — every agent in the search field also held. Against
an opponent who sells first, holding is fatal: melon absorbs ~$26.5k before flooring, so the units
sold *first* get the good prices and the patient player is left with the crashed remainder.

The reserve is a bet that prices will recover. The opponent decides whether they do.

### The pattern, stated plainly

This is the third time the same shape of error has appeared, each time one level higher:

| | tuned against | beaten by |
|---|---|---|
| E5 | `starter` | any competitive agent |
| E9 | one champion | — (validated, and correct at that level) |
| **E10** | **a field that all held inventory** | **an opponent that does not** |

**A single-opponent optimum is a hypothesis about the opponent, not a strategy.** Validation on
unseen *seeds* does not detect this at all — E9's champion was correctly validated and still
exploitable, because the missing variation was in the opponent, not the seeds.

### Adopted

- CEM now takes an opponent **pool** (`--champions a,b,c`), with fitness = the *equal-weighted*
  mean margin across it, so beating one member cannot hide losing to another. Per-opponent
  winrates print every generation.
- The exploiters stay in the registry as permanent arena members. A champion that cannot beat
  `x-dumper` is not a champion.
- This is the same conclusion the Orbit Wars field reached independently (`PLAN.md` §2.6): 7th
  place measured training against a live copy of itself at **20.7%** winrate, and 1st place named
  omitting league play as his main regret.


### E10b — Pool training fixes the specific exploit, and the exploit regenerates

Re-running CEM against the pool `[champion, x-dumper, x-turtle]` produced an agent that beats all
three, validated on unseen seeds:

| pairing | winrate (95% Wilson) | money |
|---|---|---|
| new vs **x-dumper** | **96.9% [89.3, 99.1]** | $30,981 vs $11,276 |
| new vs old champion | 100.0% [94.3, 100.0] | $31,233 vs $11,352 |
| new vs x-turtle | 100.0% [94.3, 100.0] | $36,139 vs $9,567 |

So pool training works: the agent that could not beat a dumper now beats it 96.9%.

**But the exploiters are defined relative to the champion**, so promoting a new champion generates
a new dumper — its own parameters with `forecast_weight = 0` and every `reserve_frac = 0`. That
one wins again:

**new champion vs its own dumper variant: 3.8% [1.3, 10.5], $21,860 vs $22,012.**

Note how *close the money is* — $152 apart on ~$22,000 — while the winrate is 3.8%. The dumper
wins by a hair, consistently. That is the signature of a race decided at the margin: both players
extract nearly the same total, and whoever reaches the good prices first takes the difference.

### What this means

**Dumping beats holding at every parameter setting tested so far.** The reserve machinery is a bet
that prices recover, and in a two-player market the opponent decides whether they do. Holding is
correct against a patient opponent and wrong against an impatient one, and the impatient strategy
is trivially available to anyone.

Two readings, not yet distinguished:

1. **An arms race with no fixed point** — each champion spawns a stronger counter, and the right
   answer is a league (iterate pool training until the champion stops being displaced), which is
   exactly what the Orbit Wars field converged on.
2. **Dumping is simply near-optimal**, and the entire reserve/forecast apparatus is
   over-engineering that only paid off against opponents who shared its assumptions.

Reading 2 deserves a direct test, because it is cheap and would simplify the agent enormously:
**tune a dump-only policy** (`forecast_weight = 0`, reserves pinned at 0) over the remaining knobs
and see whether it beats the pool-trained champion. If it does, the market model that `PLAN.md`
§2.5 predicts is valuable is valuable for a different reason than assumed — timing *entry into the
race*, not timing sales.

**Open (T2.5):** iterate the league to a fixed point, and test the dump-only hypothesis.


---

## E11 — The market-timing apparatus does not earn its place (T2.5)

E10b left two readings. **Reading B wins.**

Ablation: CEM with `forecast_weight` and all nine `reserve_frac` entries **pinned to 0** — a policy
that structurally *cannot* hold inventory — searching only the remaining 19 knobs, against the same
pool.

| pairing | winrate (95% Wilson) | money |
|---|---|---|
| **x-dumponly vs champion** | **84.4% [73.6, 91.3]** | $25,503 vs $19,461 |
| x-dumponly vs x-dumper | 84.4% [73.6, 91.3] | $25,491 vs $19,841 |
| champion vs x-dumper | 1.6% [0.3, 8.3] | $21,644 vs $21,788 |

The whole reserve-and-forecast apparatus — the supply forecast (E7), the tuned blend weight, nine
per-product reservation prices — **loses to a policy that sells everything the moment it exists**
and spends its search budget on farming instead.

### The exploit collapsed into the champion

Promoting the ablation removed the exploit rather than defending against it. `x-dumper` is defined
as "the champion with `forecast_weight = 0` and all reserves 0"; the champion now *is* that, and
the two have the same fingerprint (`a8657a7cf860`). There is nothing left to stop holding.

Against the regenerated pool the new champion is the top agent (90.0% [85.6, 93.2]); `x-turtle`
60%, `x-labour` 40%, `x-hoard` 20%, `x-frontrun` 0%. No exploiter beats it.

### Why holding was always going to lose

A reservation price is a bet that the price recovers before the season ends. In a two-player market
the opponent decides whether it does, and the melon curve absorbs only ~$26.5k before flooring, so
the units sold *first* take the good prices. Waiting converts a certain gain into a bet against an
adversary who profits from your patience. The town's ~8/day drain is far too slow to bail it out.

The forecast policy (E7) did win at the time — but only against opponents who also held. It was
never a market-timing edge; it was a slightly-faster-to-sell edge, and pinning the reserves to zero
is simply the limit of that.

### Consequences for the model (Phase 3)

`PLAN.md` §2.5 predicted the market is where the game is decided. That is confirmed — but the
**mechanism is not what was assumed**. The lever is not *when to sell*; selling is trivially always
now. The lever is **arriving at the market first with more units**, which is decided days earlier by
planting and harvest scheduling.

So T3.1's action space should carry **production timing**, not sale timing:

- what to plant and when, so harvests land before the opponent's
- whether to harvest early at a lower yield to beat a predicted wave
- the opponent's public tile state as a *race position*, not a price forecast

The supply forecast built in T1.3 keeps its value, but as an input to *planting* decisions rather
than to selling ones. `forecast_horizon` is retained in `Params` for that reason even though
`forecast_weight` is now pinned at 0.


---

## E12 — The livestock line was disabled by a gate, not rejected on merit

Prompted by a direct question: *does the strategy only plant crops?* It did. The T2.5 champion ran
**zero animals for the entire season**, despite eggs absorbing ~$113,763 and fertilizer ~$25,045 —
by a wide margin the two largest markets in the game — while the agent itself earned ~$23k total.

Forcing an animal line on, changing nothing else:

| agent | skill | winrate (95% Wilson) | mean $ |
|---|---:|---|---:|
| **g8** (champion + 8 geese) | **41.33** | **100.0% [98.4, 100.0]** n=240 | 30,253 |
| champion | 31.92 | 79.6% [74.0, 84.2] | 27,632 |
| g4 | 20.48 | 60.4% [54.1, 66.4] | 26,778 |
| g8-alt (alternate-day feeding) | 8.83 | 40.0% [34.0, 46.3] | 22,320 |
| g12 | −10.26 | 20.0% [15.4, 25.5] | 14,775 |
| g18 | −24.00 | 0.0% [0.0, 1.6] | 8,068 |

Head-to-head, `g8` vs champion — **$27,966 vs $23,884**, and the revenue split explains it:

| | melon | wheat | egg | fertilizer | strawberry |
|---|---:|---:|---:|---:|---:|
| g8 | $17,534 | $12,519 | **$10,082** | **$9,170** | — |
| champion | $19,927 | — | — | — | $6,183 |

**$19,252 of egg and fertilizer revenue** the champion was not collecting. Note also that wheat
returns at $12,519 — it is the feed crop, so the livestock line pulls it back into profitability.

### Why the search rejected it

`goose_target` and `goose_min_cash` are **conjunctive**: the engine only buys a bird when
`money >= cost + goose_min_cash`. The champion carried `goose_min_cash = 1371`, so a candidate with
`goose_target = 12` would still buy **zero** geese — seeds consume the cash first. The measured
gradient on `goose_target` was therefore flat, and CEM drove it to 0.

**This is a failure mode of CEM itself, not of the parameterization alone.** CEM samples each
dimension independently around the mean, so a conjunction requiring two *simultaneously* specific
values is exponentially unlikely to be sampled — the more so at 29 dimensions. A joint search is
necessary (E7) but not sufficient: it still cannot see a lever that another knob has switched off.

This is the third instance of gated knobs hiding a real effect (E7: `forecast_weight` ×
`forecast_horizon`; E6: `buy_land` × an unreachable capacity test), and the first to survive
*inside* a joint search.

**Fixes applied:**

- `goose_min_cash` upper bound cut 3000 → 800, so the gate cannot close the line off entirely.
- `g8` promoted to champion after arena validation.

**Generalizable rule:** when a knob shows no effect, check whether another knob is gating it before
concluding the lever is worthless. A flat gradient means "no measured effect **as configured**",
not "no effect".

### Re-running the search with the gate opened

With `goose_min_cash` bounded to <=800, CEM now *keeps* the livestock line on its own
(`goose_target = 6`, `goose_min_cash = 337`) — the clean confirmation that the gate, not the
strategy, was the problem. Validated on unseen seeds:

| pairing | winrate (95% Wilson) | money |
|---|---|---|
| **new vs g8 champion** | **100.0% [94.3, 100.0]** n=64 | **$36,054 vs $23,201** |
| field of 6, 320 games | 100.0% [98.8, 100.0] | mean $38,032 |

Promoted. No exploiter beats it (`x-dumper` is again the same agent; turtle 60%, `g12` 40%,
labour 12%, hoard 8%).

Progression of the champion's mean money across this sequence: **$27.6k → $30.3k → $38.0k**, from
one question about whether the agent raised animals.

**Still open:** whether cows and sheep are similarly gated. Their markets floor at ~76 and ~59
units, so the prior is genuinely against them (E1) — but that is now a prior with a poor track
record, and `goose_target` has no cow/sheep counterpart in the search space at all.


---

## E13 — Audit against the official rules

Prompted by a direct challenge to re-read `docs/` and look for flaws rather than keep optimizing.
Findings, in order of how much they cost.

### Bugs found and fixed

**1. Geese were over-bought — 14 birds against a target of 6.** `owned` was computed as
`placed + shed`, ignoring birds **in transit in a unit's inventory**. Every `PICKUP` therefore
looked like a loss and triggered a replacement purchase. Eight geese sat unplaced in the shed at
season end: **$2,400 of dead capital**, and animals cannot be sold.

**2. `PLACE` was gated on already carrying a bird, and nothing fetched one.** Identical in shape to
the `FEED` defect fixed earlier — the task existed, needed an item, and no task ever produced that
item. Coops stood empty while geese sat in the shed.

Both fixed; the shed now ends a season with zero unplaced birds.

### Tested and rejected

**Funding animals before seeds.** A goose bought on day 1 returns ~11x over the season versus ~7x
on day 10, which argued for buying birds first. Measured, it is **worse** (52.6% vs 100%):

| config | winrate (95% Wilson) |
|---|---|
| champion (seeds first) | 100.0% [98.0, 100.0] |
| animals before seeds | 52.6% [45.6, 59.5] |
| animals first, target 10 | 26.6% [20.8, 33.2] |
| animals first, target 14 | 0.0% [0.0, 2.0] |

The payback arithmetic valued a goose *in isolation* and ignored opportunity cost: $1,800 spent on
day 1 starves the melon seed budget, and seeds compound too. The existing ordering was already
right. Recorded because a plausible-looking calculation pointing the wrong way is worth keeping.

### Real gaps, not yet closed

**Cows and sheep are structurally impossible.** `BUILD_PASTURE` appears nowhere in the engine, so
the pasture line has never been buildable — it was never "tested and rejected", it was never
implemented. Their markets floor at ~76 and ~59 units (E1), so the prior is against them, but that
prior has now been wrong about geese twice.

**The crop mix ignores `town.unlocked_shops`.** Demand is dynamic and observable and the agent
never reads it for planting. Sized honestly: with `townShopUnlockInterval = 3` and 8 shops, **every
shop unlocks by day 24 of 29** — only the *order* varies between games. So this is worth something
for days 3–24 and nothing after: a moderate lever, not a large one.

**~$550 of wheat unsold at season end** (1.6% of final money), held back as feed reserve past the
point where feeding still matters.

### Confirmed correct

Fertilizer is worth more sold than used, at every crop. In particular it cannot accelerate melon:
the yield cap is reachable by age 8 with fertilizer, but `first_yield_day = 10` blocks harvest
until day 10 regardless — so the obvious "fertilize to win the race" play does not exist.


---

## E14 — Independently designed opponents (V2)

Four agents written **from a design brief**, each overriding the strategic methods rather than
mutating champion parameters (`arena/opponents.py`). Each is a hypothesis about how the champion
could lose.

| agent | brief | vs champion (95% Wilson) | money |
|---|---|---|---|
| o-sprinter | harvest at the first legal day; race to the curve | 0.0% [0.0, 5.7] | $25,687 vs $37,478 |
| o-shop-chaser | replant daily toward whatever the town demands | 0.0% [0.0, 5.7] | $29,581 vs $45,483 |
| o-land-baron | buy every quadrant, hire 11, spread wide | 0.0% [0.0, 5.7] | $15,046 vs $48,192 |
| o-goose-baron | crops are feed; farm birds, take the egg market | 0.0% [0.0, 5.7] | $2,737 vs $46,292 |

**The champion beats all four, 64/64 each.** All four are functional rather than broken — each
places animals, plants, sells, and keeps its units busy (checked directly).

### What this does and does not establish

It **does** widen the evidence beyond knob-variants: four structurally different strategies, none
of which beats the champion. It **does not** show robustness against the field. Four opponents I
designed is still four opponents I designed, and D16 stands until V1 lands.

### Two findings that fall out, both correcting earlier claims

**The egg headroom is action-bound, not strategy-bound.** E13 measured eggs at 8% of their computed
$113,763 capacity and called it the largest untapped market. `o-goose-baron` went after it directly
with **15 placed birds** — and realised **$9,334** of eggs against the champion's $8,651 with
**6**. Nearly flat, for 2.5x the animals.

The reason is the action budget, not the market: each bird costs ~4 actions/day (feed, harvest,
collect fertilizer, care) and `max_held = 4` caps what a missed harvest can recover. Fifteen birds
need ~60 actions/day, which is most of the labour force. **The $105k of "untapped" egg capacity is
not reachable at this action budget** — it was a [COMPUTED] number describing the market, not an
opportunity. This substantially answers V5 for the egg line.

**`o-sprinter` extracts essentially all of the melon market and still loses.** It realised $26,105
of melon against a computed cap of $26,485 — **99%**, the highest seen — by harvesting at the first
legal day. It still lost 0/64, finishing at $25,687 against the champion's $37,478. Maximising one
market is not the same as winning, which is E6's denial finding arriving from the opposite
direction.

**`o-land-baron` independently reproduces E6's land verdict**: 68 standing weeds at season end from
buying tiles it could not work. A second implementation reaching the same conclusion is worth more
than the first.


---

## E15 — Cows and sheep dominate; the capacity model was wrong (V3)

`BUILD_PASTURE` appeared nowhere in the engine, so cows and sheep had never been *possible* — they
were never rejected on evidence. Implemented, then tested.

| agent | winrate vs field (95% Wilson) | mean $ | vs champion |
|---|---|---:|---|
| **v-cow12** | **96.7% [93.6, 98.3]** | 49,802 | **0.0% [0.0, 7.4]** — champion loses 48/48 |
| v-cow8 | 81.7% [76.3, 86.1] | 50,014 | 0.0% [0.0, 7.4] |
| v-sheep8 | 60.8% [54.5, 66.8] | 44,708 | 0.0% [0.0, 7.4] |
| champion (geese only) | 23.3% [18.4, 29.1] | 36,741 | — |

Self-play money: geese only **$35,336**, cows only **$52,394**. A 48% improvement from a species
that was written off without ever being buildable.

### Why the original verdict was wrong

E1 ranked products by a **one-shot integral** of the price curve — how much can be dumped before
the floor. That is the wrong question. The town *removes* inventory continuously, and the removal
rate is set by **how many shops demand the product**:

| product | shops | drain/day | one-shot cap | seasonal capacity | E1 verdict |
|---|---:|---:|---:|---:|---|
| MILK | 3 | 19 | $6,181 | **~$52k** | "trap, cow costs $400" |
| WOOL | 1 (x2) | 14 | $7,928 | **~$51k** | "trap, sheep costs $500" |
| STRAWBERRY | 4 | 24 | $3,809 | **~$47k** | "worst in game, skip entirely" |
| MELON | **0** | 5 | $26,485 | ~$44k *unrenewable* | "best revenue density" |

Realised milk revenue was **$29,467** against a computed cap of $6,181 — 4.8x. Wool $23,291 against
$7,928.

**Melon is demanded by zero shops.** Only the town centre consumes it, which is exactly why it
saturates at 82% (E13) and why `o-sprinter` could extract 99% of it and still lose (E14). The whole
melon-first strategy was built on the one market in the game that cannot regenerate.

### The pattern

This is the fourth and largest refutation of a [COMPUTED] claim (after strawberry, land, and the
egg headroom). The arithmetic was correct; it answered a question the season does not ask. D15's
distinction — *computed describes the market, measured describes the game* — now has a
five-figure example.

**Rule adopted:** before judging any product's market, count how many shops demand it.

### Re-searching with livestock in the space

With `cow_target` and `sheep_target` added (31 knobs), CEM converged on **cows only** —
`cow_target = 7`, no geese, no sheep — held-out +$31,674 at 100% against all three pool members.
Validated on unseen seeds:

| agent | winrate (95% Wilson) | mean $ |
|---|---|---:|
| **new champion** | **100.0% [98.9, 100.0]** n=336 | **66,349** |
| v-cow12 | 80.1% [75.5, 84.0] | 48,595 |
| v-cow8 | 68.8% [63.6, 73.5] | 49,593 |
| v-sheep8 | 51.2% [45.9, 56.5] | 44,997 |
| previous champion (geese) | 25.0% [20.7, 29.9] | 37,313 |
| o-sprinter | 0.0% [0.0, 1.1] | 33,002 |

Promoted. Mean money against the regenerated exploiter pool: **$67,021**.

**Champion progression across the audit the user prompted:**
$27.6k -> $30.3k -> $38.0k -> $41.6k -> **$67.0k** — a 2.4x improvement, entirely from testing
things previously ruled out by arithmetic.


---

## E16 — We never reach the saturated regime; and why holding still loses

Prompted by "the more we test the more we know". V4 asked whether E11's *"never reserve"* was an
artifact of a melon-only field, since melon has zero shop demand and cannot recover.

### First attempt was vacuous — caught by identical output

Reserves set at `0.35 x base` produced money **identical to the dollar** across four variants
($63,398 for both `champion` and `r-melon`). That tell was the finding: the threshold never binds.

| product | base | 0.35x base | lowest price seen all season | realised average |
|---|---:|---:|---:|---:|
| MELON | 250 | 88 | **214** | 245 |
| MILK | 160 | 56 | **169** | **226** |
| STRAWBERRY | 120 | 42 | **128** | 254 |
| WHEAT | 25 | 9 | **26** | 54 |

**Prices never fall below base. Most sit well above it.**

### The regime we are actually in

Market inventory relative to `I0 = 10,000`:

| day | MELON | MILK | WHEAT | CARROT | STRAWBERRY |
|---|---:|---:|---:|---:|---:|
| 5 | −10 | −10 | −92 | +38 | −22 |
| 15 | +40 | −70 | −340 | −56 | −200 |
| 25 | +30 | −62 | **−756** | −252 | **−494** |

**The town drains faster than two players can supply.** Inventory falls below `I0` for everything
except melon, so prices rise all season and the market is in *scarcity*, not glut.

This retires a whole line of analysis. The one-shot capacity table (E1) and the seasonal capacity
table (E15) both describe **how much a market can absorb before flooring** — a regime this game
does not reach for any shop-demanded product. Melon is the sole exception, and only because no shop
demands it: it is the one product two players *can* flood, which is exactly why the melon-era
findings (E10, E11) were about racing.

**The binding constraint is production, not market absorption.**

### Reserves re-tested where they actually bind

| config | winrate (95% Wilson) | mean $ |
|---|---|---:|
| **champion — sell everything immediately** | **95.1% [91.4, 97.2]** | **68,177** |
| reserve 0.9x base | 79.9% [74.2, 84.6] | 65,015 |
| reserve 1.1x base | 50.0% [43.5, 56.5] | 60,376 |
| reserve 1.3x base | 25.0% [19.8, 31.1] | 38,637 |
| reserve 1.1x on everything incl. melon | 0.0% [0.0, 1.7] | 21,946 |

Monotone: the more you hold, the worse you do — **in a market whose prices are rising**.

### The corrected reason for E11

E11 explained "sell immediately" as *"a reserve is a bet that prices recover, and the opponent
decides whether they do."* That is true for melon and wrong everywhere else — prices recover
without anyone's permission, because the town drains them.

The real reason holds in both regimes: **cash compounds and inventory does not.**

- The shed caps at 100 items, so there is little to hold.
- Unsold stock scores **zero** at the end.
- Money released now buys seeds and animals that produce for the rest of the season; goods held
  back earn only the drift in their own price.

A ~10-20% price improvement from waiting cannot beat reinvestment into an asset that yields every
day for the remaining weeks. **E11's conclusion survives; its stated mechanism was wrong for
everything except melon.**


---

## E17 — Production audit: E6 reversed, and priorities were costing more than they bought

E16 concluded the binding constraint is production, not market absorption. So: where does the
production go?

### Units are no longer idle — they are walking

| | E6 (melon champion) | E17 (cow champion) |
|---|---:|---:|
| PASS (idle) | 38–58% | **4%** |
| movement | 31–46% | **65%** |
| useful work | 11–16% | ~15% |

**E6's conclusion is reversed.** It found units idle and deprioritized routing on that basis
(T1.2). The cow-led strategy gives every unit something to do — feed, harvest, collect fertilizer,
care, water — so the constraint moved from "nothing to do" to "too far to walk". A conclusion about
a bottleneck is only valid for the strategy that was measured.

### Three fixes, two of which did nothing

**1. Animals were placed in the far corner — real, +$2k.** Structures were allocated from the raw
row-major tile list while *crops* took `_nearest_first`, so animals sat at mean distance **7.1**
from the shed and crops took the near tiles. Exactly inverted: an animal needs a wheat fetch from
the shed almost every day, a crop needs watering and no shed trip at all. Fixed: animals now sit at
**3.3**, movement 65% -> 62%, $64,368 -> $66,463.

**2. Batching the feed fetch — no effect.** `per_trip` divided the unfed count across all units, so
eight units each walked to the shed for a *single* wheat. Batching six per trip changed nothing
($66,463 -> $66,491): the shed round-trip was never the cost. Walking *between scattered tasks* is.

**3. Dropping task priorities entirely — the real win.** Assignment sorted by `(priority,
distance)`, so any higher-priority task anywhere on the farm outranked a neighbouring one and units
crossed the map to serve it. Making distance and priority commensurable, then sweeping the
exchange rate:

| priority weight | movement | arena winrate | mean $ |
|---|---:|---|---:|
| **0.0 — pure nearest-task** | **58%** | **96.4% [93.1, 98.2]** | **67,162** |
| 1.0 | 57% | 78.6% [72.7, 83.4] | 66,651 |
| 3.0 (previous default) | 64% | 47.3% [40.9, 53.8] | 62,537 |

**Ignoring urgency entirely beats respecting it, 96.4% vs 47.3%.** The deadlines this engine
tracks are mostly soft — watering every *other* day suffices, animals survive a missed feed — so
priority ordering bought almost nothing and cost a farm crossing per task.

**Pattern:** two of the three plausible fixes did nothing, and the one that worked was removing a
mechanism rather than adding one. Same shape as E11, where pinning the market-timing apparatus to
zero beat tuning it.

### Re-searching with the routing knobs in the space

CEM over 33 knobs (adding `priority_weight` and `feed_batch`) against a pool. Validated on unseen
seeds:

| agent | winrate (95% Wilson) | mean $ |
|---|---|---:|
| **new champion** | **100.0% [98.6, 100.0]** n=280 | **75,534** |
| previous champion | 61.4% [55.6, 66.9] | 59,847 |
| v-cow12 | 29.3% [24.3, 34.9] | 47,701 |
| o-sprinter | 0.0% [0.0, 1.4] | 35,683 |

It settled on `priority_weight = 0.49` — not the 0.0 that won the hand sweep, which is the usual
lesson about coordinate-wise tuning (E7): once `feed_batch`, `tiles_per_unit` and the herd sizes
move with it, a small amount of urgency is worth its walking cost again.

It also chose a **mixed herd — 8 cows and 4 sheep** — using two regenerating animal markets rather
than saturating one. A hand-built 7-cow/6-sheep mix had scored only 47.9%; the difference is the
rest of the vector being tuned around it.

Mean money against the regenerated exploiter pool: **$86,326**.

**Champion progression across this whole line of testing:**
$27.6k -> $30.3k -> $38.0k -> $41.6k -> $67.0k -> **$86.3k**. Every step came from testing
something previously ruled out by argument.


---

## E18 — Why five champions in a row were the wrong one (D19)

Prompted by: *"our testing keeps choosing the wrong one."* It did, and the cause is arithmetic
rather than judgement.

| games | ± at 50% | smallest edge resolvable |
|---|---|---|
| 24 | 18.6pp | 68.6% |
| 64 | 11.9pp | 61.9% |
| 300 | 5.6pp | 55.6% |
| 500 | 4.4pp | 54.4% |

**Every promotion was made on 24–64 games, from differences of 3–8pp.** At that sample size only a
12pp+ edge is distinguishable from noise, so each "improvement" was a coin flip presented as a
result. The five that a larger later measurement overturned: E8 (wrong opponent), E10 (lost 0/80 to
a strategy outside its field), E12 (a gated knob), E15 (an unimplemented species), E17 (a herd mix
beaten 64.4%).

### The gate

`tools/promote.py` — three stages on fresh seeds, escalating only where a result is close:

1. beat the incumbent over **500 games** with the interval clear of 50%;
2. survive the **full ~66-agent gauntlet**, screened at 24 games, every close call escalated to 500;
3. survive a **neighbourhood sweep** of the discrete knobs.

Stage 3 exists because CEM optimises 33 continuous dimensions independently and can converge to a
point that a one-step change in a *discrete* knob beats outright — measured at 64.4% for the herd
mix. **A search optimum need not be a local optimum.**

### It rejected the sitting champion immediately

```
STAGE 2  r-regen        41.0% [36.8, 45.4] n=500   <-- LOSS
STAGE 3  _nb_hire_max9  43.6% [39.3, 48.0] n=500   <-- BETTER
CHAMPION IS NOT THE BEST AVAILABLE.
```

Two improvements invisible to every earlier test. The candidate combining them was *also* refused —
`hire_max = 10` beat it 74% — so the gate was walking up a hill one step at a time.

### Sweeping the knob it kept flagging

| hire_max | result |
|---|---|
| 7 | loses to 10 **100%** |
| 8 | loses to 10 81.3% |
| 9 | loses to 10 72.7% |
| **10** | **beats all of 7, 8, 9, 11** |
| 11 | loses to 10 92.0% |
| 12 | collapses — $56k vs $85k, loses to 8/9/10/11 outright |

E1's "12+ hands bankrupts" survives; its "optimum ~8" does not. The optimum moved from 8 to 10
because the strategy changed underneath it — cows and sheep give every hand daily work, where the
melon economy left them idle. **A tuned constant is only valid for the strategy it was tuned on**,
which is the same lesson as E17's reversal of E6.

### Reserves are back on the table

`r-regen` — a 0.35x-base reserve on shop-demanded products — beat the champion 59%. E16 concluded
reserves never help, but that was measured before sheep and wool entered the mix. **E16's
conclusion needs re-testing per product**, not treating as settled.


---

## E19 — Sheep made one market floodable, so reserve pricing came back

Arrived as three test failures after promoting the gated champion, not as an experiment. All three
were correct.

```
WOOL ended in glut at 10,038 — the regime has changed
WOOL fell to 1 — below half of base 200
champion fingerprint != x-dumper
```

### What changed

E16 measured that the town drains faster than two farms can supply, so prices *rise* and nothing
saturates — and concluded that reserve pricing is worthless. That was measured **before sheep
existed**. With the champion running 8 sheep per side:

| product | inv vs I0 | final price | sold | revenue |
|---|---:|---:|---:|---:|
| **WOOL** | **+38** | 116 | 148 | $19,171 |
| MILK | −146 | 265 | 168 | $40,278 |
| MELON | +70 | 201 | 105 | $22,198 |
| WHEAT | −1,294 | 61 | 959 | $54,018 |

**Wool is the only *shop-demanded* market two farms can outproduce.** One shop demands it (Yarn
Store, 2x) for a drain of ~14/day, against 8 sheep per side each yielding wool every third day.
Melon remains floodable for the opposite reason — no shop demands it at all.

That is exactly what the champion's `0.35 x base` reserve is for, and why `r-regen` beat the
reserve-free champion 59% in the audit that triggered this. **The reserve is not general market
timing; it is a floor that stops us selling into our own wool crash.**

The `$1` low is the day-27 end-game dump, which bypasses every reserve because unsold stock scores
zero. The town recovers the price to $116 afterwards, so a transient crash there is correct
behaviour rather than a regime change.

### The correction to E16

E16's *mechanism* stands — cash compounds, inventory does not, and holding loses in a rising
market. Its *scope* was wrong: it generalised "no reserve ever" from a field with no sheep in it.

**Rule:** a reserve is worth having exactly on the products *we* can flood, and worthless on the
rest. Which products those are is a function of our own herd and crop mix, so it must be
re-measured whenever the mix changes — not carried forward as a setting.

### Note on how it was found

Three assertions written to pin E11/E16 failed the moment the strategy moved past them. That is the
tests working: each encoded a claim, and the claims expired. Had they been written as loose
smoke-tests they would have passed silently and the wool crash would have gone unnoticed.


---

## E20 — Land is confirmed dead, and the farm is now full

The champion ends a season with **25 of 25 tiles occupied** — 14 animals, 10 crops, 1 structure,
zero empty — with the herd at target and 18% of unit-turns idle. Land had been rejected three
times (E1, E6, E14), but always under a *melon* strategy with 8 hands, poor routing and few
animals. Three conclusions have already expired that way (E6 -> E17, E16 -> E19, hire_max 8 -> 10),
so it was re-tested rather than assumed.

| variant | champion's winrate against it | their money |
|---|---|---:|
| buy land, herd 10c+12s | 100.0% [98.1, 100.0] | $7,546 |
| buy land, herd 14c+16s, 12 hands | 100.0% [98.1, 100.0] | $2,911 |
| buy land, keep herd, more crops | 100.0% [98.1, 100.0] | $43,979 |
| buy land late, big herd | 100.0% [98.1, 100.0] | $4,557 |

**Land loses on all four framings**, and the aggressive ones bankrupt outright: $7,000 of quadrants
plus a 26-head herd at $400-500 each is simply unaffordable against $3,000 of starting capital and
a season that ends before the investment returns.

This is the one early verdict that has now survived four independent tests under four different
strategies. Recorded as settled.

### What it implies

With land dead and the farm full, **the champion is at a genuine ceiling for its parameter space**:
it cannot add tiles, cannot add animals, and 18% of its turns are already idle. Further gains have
to come from using 25 tiles better — faster rotation, a better animal mix — not from scale.


---

## E21 — Two shipped defects, one root cause: verifying the wrong surface

A competitor's `submission.py` (`reference/kaggriculture/1/`) was read as a reference. Running it
exposed two of our own defects, both of which had passed every local test.

### a. The submission never loaded on Kaggle — $3,000, measured 0-40

`main.py` read `__file__` at module level. Kaggle does not import a submission; it
`exec(compile(raw, path, "exec"), {})` and then calls **the last module-level callable**
(`kaggle_environments/agent.py:47-63`). That globals dict has no `__file__`, so import raised
`NameError`, the agent never loaded, and every episode banked the $3,000 starting cash.

It passed because `tools/build_submission.py` did `import main` — where `__file__` exists — and fed
it `env.state[1].observation`. Neither is what the runner uses. Fixed; the smoke test now goes
through `env.run(["main.py", ...])` in both seats, checks the picked callable is `agent`, and fails
if money does not exceed the starting bank. Working score: **$106,457** vs `starter`.

### b. `step` does reach seat 1 — and "fixing" kagsim to match had broken it

Recorded rule, in `CLAUDE.md`, `PLAN.md`, `TASKS.md` and D15: *`step` is not declared shared, so
seat 1 never receives it.* **Refuted.** `step` is delivered to both seats, correct on all 719
turns of a full episode.

The test read `env.state[1].observation` — the stored replay state, which genuinely does strip
shared fields for seats above 0. Agents are handed `Environment.__get_shared_state(position)`
(`core.py:754-767`), which rebuilds the observation per-agent and carries `step` through.

The expensive part: **kagsim had this right** and was changed to suppress `step` for player 1 in
order to reproduce the non-existent omission. A parity check reading the wrong object converted a
correct simulator into a diverging one and then certified the result. Restored; the test now routes
through a `delivered()` helper that can only see the runner's surface. `make verify`: 0
divergences, 100% coverage.

**Rule.** Verify against the surface the runner uses, not the one that is easy to reach.

---

## E22 — The first external opponent beats the champion 0-40

40 episodes, both seat orders, reference environment, working submission on both sides.

| | ours | theirs |
|---|---:|---:|
| head-to-head mean money | $71,978 | **$157,778** (2.19x) |
| record | **0W-40L** | |
| gross revenue, shared seed | $85,512 | $149,849 |
| spent | ~$34,000 | ~$19,000 |

We spend nearly twice as much and earn far less. Every local ranking in this project is now known
to be measured inside a field that sits well below a real competitor.

### What they are

An **open-loop replay**: a 719-step hardcoded action script, zlib+base85 compressed, indexed by
`obs["step"]`. Closed-loop code only patches it — weed repair, a conditional cow->sheep swap, a
wool release controller, and a three-turn fertilizer relay. Not a learned or reactive agent.

Note they read `obs["step"]` for the entire plan index, which is direct independent confirmation
of E21b: if seat 1 lacked `step`, this agent would replay turn 0 for a whole season. It does not.

### Where the money differs (units sold exact; $/unit from pure-sell steps, 32-40% coverage)

| product | their units | our units | their $/unit | our $/unit |
|---|---:|---:|---:|---:|
| **STRAWBERRY** | **286** | **8** | **196.4** | 90.7 |
| MELON | 114 | 100 | **207.8** | 121.2 |
| MILK | 241 | 165 | 74.0 | 76.3 |
| FERTILIZER | 235 | 272 | 55.2 | 71.5 |
| WOOL | 132 | 141 | 140.0 | 112.0 |
| WHEAT | 455 | 1,066 | 43.2 | 64.4 |

**Strawberry is the single largest gap**: ~286 x $196 is roughly a third of their gross, and the
champion sells eight units all season despite carrying `crop_mix.STRAWBERRY = 0.35`. Melon is the
second: same volume, they realise 1.7x the price, so timing not quantity. Our 1,066 wheat units at
low value look like churn.

### Land, again

They own **3 quadrants and leave 56 of those tiles empty** (13 pastures, 1 crop, 5 weeds at season
end). So E20 is not contradicted — land is not what makes them win, and they may be wasting the
purchase. The gap is crop selection and sell timing.


---

## E23 — Why we lose: we run a tenth of their crop operation

Follow-up to E22, asking the question directly: *why do they buy land when our own tests say land
loses?*

### Answer: they buy land because they farm it. We could not.

Tile usage **over the season** (E20 read an end-of-season snapshot, which is badly misleading --
at the last step they show 56 empty tiles and 1 crop, because everything has been harvested):

| day | their crops | their animals | our crops | our animals |
|---:|---:|---:|---:|---:|
| 6 | 19 | 6 | 15 | 2 |
| 12 | 63 | 12 | 12 | 12 |
| 18 | 63 | 12 | 10 | 14 |
| 24 | 62 | 13 | 10 | 14 |

**63 crop tiles against our 10.** Their 63 crops + 12 animals = 75 tiles, exactly the 3 quadrants
they own, fully used. Land is not their edge; it is the *enabler* of a crop operation 6.3x ours.
Everything in E22 -- 4.4x the watering, 4.1x the planting, 2.8x the harvesting, 286 strawberries
against our 8 -- is downstream of that one fact.

Land prices are 1,000 / 2,000 / 4,000 (`LAND_ORDER = ["NE","SW","SE"]`). They buy two ($3,000).

### E20 is not contradicted, but its conclusion was too broad

Given land, our engine *does* fill it -- 76 crop tiles, all 4 quadrants -- and still earns
**$60,456 against $107,712 without it**. So "land loses" was a true measurement of our engine and a
false statement about the game. Diagnosis of the land run:

| | ours +land | theirs |
|---|---:|---:|
| crop tiles (peak) | 76 | 63 |
| PLANT | 252 | 199 |
| WATER | 881 | 1,010 |
| **HARVEST** | **200** | **390** |
| weeds at day 24 | 9 | -- |

We plant *more* and water nearly as much, then **harvest half as often** and lose tiles to weeds.
Production is not the constraint; **conversion is**.

### Four engine defects, in order of confidence

1. **`FERTILIZE` is not implemented at all.** The verb appears nowhere in `agent/engine.py` --
   only `COLLECT_FERTILIZER` (the animal byproduct) and the market product name. Fertilizer
   **doubles crop yield** for 3 days per unit (`kaggriculture.py:386`, `:777-778`). We collect
   ~272 units a season and **sell every one**. They apply 72. Same class of defect as
   `BUILD_PASTURE` being absent, which had disabled cows and sheep entirely.
2. **Harvest/water were mutually exclusive** (`engine.py:117`, an `elif`). A tile with yield
   pending stopped emitting WATER; two unwatered days turn it into a weed. It only bites `ongoing`
   crops -- TOMATO and STRAWBERRY -- because only they sit with yield pending for days.
3. **Hauling overhead.** PICKUP 497 vs 135, DROP 222 vs 57 -- roughly 4x the turns spent moving
   items rather than farming.
4. **Overall efficiency.** 28% productive unit-turns against their 42%; we move more (52% vs 43%)
   and idle more (20% vs 15%).

### The water policy is a strategy dimension, not a bug with one right answer

Three-way comparison, 4 seeds, vs `starter`:

| policy | melon champion | strawberry mix | strawberry + land |
|---|---:|---:|---:|
| `elif` (original) | $106,136 | $57,392 | $32,382 |
| survival-only | $106,605 | $48,787 | $30,611 |
| **both** | $96,004 | **$70,511** | **$45,128** |

"Both" costs the melon champion $10k (WATER outranks HARVEST and steals turns) and is worth +23%
to a strawberry mix, +39% to strawberry+land. Added as `Params.water_mode`, defaulting to `elif`
so the sitting champion is bit-identical, for the re-search to choose.

### The methodological point, which matters more than any single fix

The champion's parameters were CEM-tuned **against this engine**, so they are co-adapted to its
defects. The fingerprint is unmistakable: `crop_mix.MELON = 0.867` is the highest weight in the
vector, and melon is the one crop that is *not* `ongoing` -- the only one defect 2 does not kill.
Our own D17 says melon is demanded by **zero** shops and strawberry by four.

So the search did not find that melon is good. It found the crop our engine could not break.

**Consequence:** an engine fix cannot be evaluated with the current parameters -- they encode the
bug. Fix the defects first, then re-search, then compare. Judging fixes against a co-adapted
champion will reject every one of them.


---

## E24 — FERTILIZE implemented, and the "fix it all together" hypothesis refuted

E23 predicted that the engine defects were co-adapted with the parameters, so fixing them
*together* would unlock the opponent's strategy. **Measured, and it is wrong.** Recording it
because the prediction was mine and it failed cleanly.

### FERTILIZE now exists

Added `Params.fertilize` (+ `fertilize_batch`), a `_wants_fertilizer` window check, and a shed
fetch mirroring the wheat/FEED trip. It emits correctly -- 38 applications a season. Both defaults
are off, and the champion is bit-identical: **$106,136**, same four seeds, before and after.

It also **loses money on the current strategy**, for a reason worth writing down: melon's yield
caps at 6 by age 10 and it is harvested at 10, so doubling accrual reaches the same cap sooner and
buys *nothing*. We pay the shed round-trip for zero gain -- WATER fell 231 -> 187 and movement rose
52% -> 61%. Fertilizer only pays on crops whose accrual is actually the binding constraint.

### The combined strategy is worse, not better

4 seeds vs `starter`:

| config | mean |
|---|---:|
| champion (melon, no land, no fertilizer) | **$106,136** |
| strawberry + water_mode=both | $76,870 |
| + fertilize | $73,210 |
| + land | $39,930 |
| + land + 12 hands | $8,718 |
| + land + 14 hands | $3,235 |
| *(their agent vs the same `starter`)* | *$193,177* |

Every step toward the opponent's configuration costs money, and labour on top of land bankrupts
outright.

### The real constraint: conversion, not strategy

Crop fate over a season, sampled daily:

| | champion | strawberry + land |
|---|---:|---:|
| new weeds (= plants destroyed by 2 dry days) | **3** | **65** |
| tiles sitting ripe and unharvested | 6-15 | rising to **20-32** |

Given 76 tiles our engine plants them, then loses 65 plants to thirst and leaves a third of the
rest rotting ripe. Their agent services 63 tiles with the same 11 units and ends with 5 weeds.

The difference is throughput, and it was already visible in E22: **42% productive unit-turns for
them against 28% for us**, movement 43% vs 52-61%. A 63-tile crop farm is not a strategy you can
select -- it is a throughput you have to be able to afford.

### Revised ordering

Land, strawberry and fertilizer are not independent levers to be searched. They are all **gated on
unit throughput**, and while units spend half their turns walking, every one of them loses money.
So the next work is routing and assignment efficiency (the E17 thread, reopened), not more
parameter search. Search over a space the executor cannot serve just re-finds melon.

`water_mode` and `fertilize` stay in `Params`, off by default, for the re-search that follows the
throughput work.


---

## E25 — Ablating the opponent: land and fertilizer are 61% of their score

Their agent is a hardcoded action script, so its components can be removed and re-measured. This
is the first time we have been able to establish *causally* what makes a winning strategy win.

4 seeds, vs `starter`:

| their agent | mean | change |
|---|---:|---:|
| unmodified | **$180,860** | -- |
| minus FERTILIZE (unit op -> PASS) | $126,616 | **-$54,244** |
| minus BUY_LAND (market order removed) | $74,464 | **-$106,397** |
| minus both | **$70,578** | -$110,282 |
| *our champion, same opponent* | *$106,136* | |

**Stripped of land and fertilizer, their agent scores $70,578 -- below our champion's $106,136.**

### What this settles

It answers a reasonable question directly: *if fertilizer and the water policy did not help us, is
avoiding them our advantage?* No -- exactly inverted. Those two mechanics are the entire deficit.
Our engine is competitive on everything else and cannot exploit either.

It also corrects an overstatement in E24. "Fertilizer does not contribute" was true of *our engine
on a melon mix*, where yield caps at 6 by age 10 so doubling accrual buys nothing. As a claim about
the mechanic it is wrong by $54,244 -- about half our current total score, from a verb that was
simply absent from the engine.

### What it implies

The ceiling is not where E20/E23 put it. There is a measured, causal path worth roughly $110k
sitting in two mechanics we already understand:

* `FERTILIZE` now exists (E24) but only pays on crops whose accrual actually binds, and only at a
  throughput that can afford the shed trip.
* `BUY_LAND` is worth +$106k **to an agent that can service 63 crop tiles**. Ours plants 76 and
  loses 65 of them to thirst (E24).

Both remain gated on the same thing: **unit throughput**, 28% productive turns against their 42%.
That is the work, and E25 puts a price on finishing it.


---

## E26 — Head-to-head against the ablations: land is the mechanic that beats us

E25 measured the ablations against `starter`. Per our own standing rule that is the weakest
possible reference, so this repeats it head-to-head against the champion: 12 seeds x both seat
orders = 24 games per row.

| opponent | our winrate | ours | theirs |
|---|---:|---:|---:|
| full agent | **0.0%** (0/24) | $69,804 | $152,120 |
| minus FERTILIZE | **0.0%** (0/24) | $72,066 | $103,606 |
| minus BUY_LAND | **100.0%** (24/24) | $83,756 | $64,067 |
| minus both | **100.0%** (24/24) | $83,480 | $59,746 |

**`BUY_LAND` is necessary and sufficient for their win.** Removing fertilizer costs them $48k and
changes nothing about the result -- they still take 24 of 24. Removing land flips the match
completely, and removing fertilizer on top of that adds almost nothing (100% either way).

This is a cleaner statement than E25's money ranking: by *money* fertilizer looked worth half of
land, but by *outcome* fertilizer is not what defeats us at all.

### Land pays them twice

Our own money moves with their strategy: **$69,804 against the full agent, $83,756 once they stop
buying land** -- a $14k swing in our revenue caused entirely by their purchase. Their expanded
production floods the shared markets and depresses the prices we sell into.

That is `PLAN.md` §2.5 (public supply, private inventory) appearing in a real measurement rather
than a model, and it means land's value is understated by any single-player estimate. Every land
experiment we ran (E1, E6, E14, E20) measured land against `starter` or against our own variants --
none of them could see this term.

### Status of the "land is dead" conclusion

**Withdrawn.** It was measured four times and was correct each time *about our engine*, which
plants 76 tiles and loses 65 of them to thirst (E24). As a claim about the game it is now refuted
by the strongest evidence in the project: the only external opponent we have wins with it and
loses without it.

This is the same shape as the market-ranking error recorded in `CLAUDE.md` -- a conclusion that was
true of our own capability, generalised into a claim about the game, and then confirmed repeatedly
by experiments that all shared the limitation.

### Consequence for the plan

The throughput work (E24) is no longer one option among several. It is the prerequisite for the
single mechanic that decides this matchup.


---

## E27 — The opponent joins the arena, and the arena turns out to be a turn short

### a. `kind="external"` and the first non-self-referential gauntlet rung

`AgentSpec` gained an `external` kind that loads a submission file the way Kaggle does -- exec into
empty globals, take the last module-level callable -- so what the arena scores is what the runner
would run. Registered as `boatlee`. Registry is now 79 agents, exactly one of which nobody here
wrote.

D16 has been open since the arena was built: every measurement is against agents from one author,
reproducing one author's blind spots. This is the first entry that can contradict them, and it
already has -- 100% against the champion in kagsim, matching the reference env result (E26).

### b. kagsim is bit-exact against a third-party agent

Driving both simulators with the same third-party agent and the same champion:

| seed | reference env | kagsim |
|---|---|---|
| 101 | [162184, 66627] | **[162184, 66627]** |
| 102 | [151503, 81161] | **[151503, 81161]** |
| 103 | [162832, 67440] | **[162832, 67440]** |

Exact, both seats, all three seeds. Every prior parity check drove kagsim with our own agents or a
fuzzer; this is the first with code written by someone else, and the strongest evidence the
simulator is right.

### c. The arena and the search were running 718 turns; the real env runs 719

Getting that exact match required 719 steps. `arena/run.py` and `sim/runner.py` both used
`episodeSteps - 2`. At 718 turns the same three seeds gave [161820, 66206] / [151146, 80748] /
[162454, 67440] -- consistently ~$360 low.

The dropped turn is the **terminal** one, which is where end-of-season liquidation happens (their
agent has an explicit `step >= 713` terminal branch; ours has `sell_all_after_day`). So every arena
ranking and every CEM score in this project was optimising a season one turn shorter than the one
we are scored on -- and specifically blind to the last liquidation turn.

Fixed to `episodeSteps - 1`. The arena now reproduces the reference to the dollar. 187 tests pass.

The money effect is small (~0.2-0.6%) and unlikely to have changed any ranking on its own, but it
is a train/serve mismatch of exactly the kind that E21 was about: **the harness was not measuring
the thing the runner runs.** Third time that root cause has produced a defect.

### d. A trap worth naming

`Engine.__call__(self, obs)` raised `TypeError` when handed straight to `kaggle_environments`,
which passes `(observation, configuration)` to a *callable object* but only `(observation)` to a
plain function. The agent then banked the $3,000 starting cash while the episode still reported
`DONE`, which reads exactly like a simulator divergence -- it is how this whole investigation
nearly got misdirected. `__call__` now accepts and ignores `configuration`. `main.py` was never
affected; it wraps the engine in `def agent(obs)`.


---

## E28 — Why the champion loses, end to end

Every earlier experiment found a piece. This is the chain, with the root cause measured.

### The root cause

Steps walked between consecutive productive actions, seed 5, whole season:

| agent | productive actions | share of unit-turns | **steps walked per productive action** |
|---|---:|---:|---:|
| boatlee | 2,801 | 42.2% | **1.01** |
| our champion | 2,011 | 28.4% | **1.84** |
| ours + land | 2,289 | 33.1% | **1.62** |

At 1.01 their units are sweeping a contiguous block -- step, work, step, work. Ours pay 1.84 steps
for every useful action: **80% more travel per unit of work.**

### The chain

1. We walk 1.84 steps per productive action; they walk 1.01.
2. So they extract **2,801** productive actions from 11 units where we extract **2,011** -- 39%
   more work from the same labour.
3. That surplus is what services **63 crop tiles**. We can service **10** (E23).
4. 63 tiles is what makes `BUY_LAND` pay. Ours buys land, plants 76, loses **65 plants to thirst**,
   leaves 20-32 ripe unharvested, and earns $60k instead of $107k (E24).
5. `BUY_LAND` is necessary *and* sufficient for their win: strip it and we win 24/0; strip
   fertilizer instead and we still lose 0/24 (E26).
6. Land pays them twice -- their supply floods the shared market and costs us $14k of revenue
   (E26).

**We lose because our units waste 80% more steps walking between jobs.** Crop choice, market
timing, fertilizer and cash management are all downstream: they follow from a farm a sixth the
size, and the farm is small because we cannot afford to service a bigger one.

### Why our units walk further

The assignment picks the nearest task fresh each turn from a scattered list, with a stickiness
patch bolted on to stop oscillation. It never *batches* work by region, and crops are allocated to
whichever tiles happen to be free rather than into a compact block. Their plan is offline-optimised,
so it reads as a route rather than a sequence of independent nearest-task choices.

### The target

**Steps per productive action, 1.84 -> 1.0.** This is a better objective than "efficiency" because
it is a single number, it is measurable every run, and E28 shows the whole result hangs off it.
Two levers: plant crops in contiguous clusters, and let a unit exhaust the tasks adjacent to it
before reassignment.

Gate the work on winrate vs `boatlee` (now a registered arena agent, E27), never on money vs
`starter`.


---

## E29 — Where our unit-turns actually go, and three fixes that did not work

E28 named steps-per-productive-action as the root cause. Working on it produced a better diagnosis
and no improvement. Both halves are recorded.

### The correction to E28

Steps-per-action is confounded by **task density**, not purely routing skill. Our champion farms 25
tiles at 1.89; our own land configuration farms 76 at 1.62 -- a *better* score while earning far
less. A denser farm mechanically shortens the walk between jobs. So boatlee's 1.01 is partly a
*consequence* of running 63 crop tiles, not only a cause. Do not optimise this number directly.

### The better decomposition

Productive turns by category, seed 900000:

| | ours | boatlee |
|---|---:|---:|
| crop work | 416 (20.7%) | **1,671 (59.7%)** |
| animal work | 889 (44.3%) | 897 (32.0%) |
| **hauling** | **698 (34.8%)** | **192 (6.9%)** |
| total productive | 2,008 | 2,800 |

Animal work is nearly identical in absolute terms (889 vs 897) -- it is not where we differ. We
spend **506 more turns hauling** than they do, roughly a quarter of our productive budget, and we
convert what is left into a quarter of their crop work.

The hauling detail: **1,359 wheat collected to service 290 FEED ops**, 623 animal pickups to place
14 animals, and 214 DROPs to shed the surplus.

### Three fixes, all measured worse

| change | steps/action | money vs boatlee |
|---|---:|---:|
| champion (baseline) | 1.89 | **$78,032** |
| `assign_mode="global"` -- remove unit-index bias from assignment | 1.83 | $72,923 |
| `fetch_in_flight` -- stop re-fetching what is already en route | 2.15 | $68,206 |
| (earlier, E23) naive water fix | -- | $96k vs $107k |

All three are defensible corrections to real defects. All three cost money. The global assignment
raised productive turns (28.1% -> 30.6%) and *lowered* earnings, which says the extra actions it
bought were low-value ones.

Both are kept as `Params` flags (`assign_mode`, `fetch_in_flight`), defaulting to the old
behaviour, so the champion is bit-identical -- verified at $78,032 / 1.89 after the change -- and a
re-search can select them if they pay in combination.

### What this means

The engine is not one defect away from competitive. Its parameters were fitted around its current
behaviour, and every local correction lands outside that fit. Nothing here contradicts E26 -- land
still decides the matchup -- but the route to it is not a short list of bug fixes, and I do not yet
have a change that pays.


---

## E30 — Re-searching against the real opponent: ruled out

E24/E29 hypothesised that the engine's defects were co-adapted with the champion's parameters, so
fixing them *together* -- rather than judging each against a vector fitted to their absence --
would unlock land and fertilizer. This is that test, and the hypothesis is **refuted**.

Setup: 38 knobs (the five new behaviours included), pop 32, 24 games/candidate/opponent, 14
generations, pool = `boatlee` + `champion` + `x-dumper`, held-out validation seeds.

| gen | held-out margin | per-opponent winrate (boatlee / champion / x-dumper) |
|---:|---:|---|
| 0 | -$58,496 | **0%** / 0% / 0% |
| 5 | -$59,772 | **0%** / 0% / 0% |
| 10 | -$32,180 | **0%** / 12% / 12% |
| 13 | -$32,072 | **0%** / 12% / 12% |

**Zero percent against `boatlee` in every one of the 14 generations.** The margin stayed negative
throughout, and the winner verified independently at 0/24 vs `boatlee` and **8.3% vs our own
champion** -- the search output is worse than the incumbent it started from.

### What the search chose

Every knob that had never been searched before returned to its default:

| knob | result |
|---|---|
| `buy_land` | False |
| `water_mode` | elif |
| `assign_mode` | sequential |
| `fertilize` | False |
| `fetch_in_flight` | False |

Given direct exposure to an opponent that wins **because** of land and fertilizer (E26), and given
the knobs to adopt both, CEM rejected them and re-found the champion's shape.

### Why that is the useful result

It is not that the search failed. It is that **the search is right**: within this executor, land
and fertilizer really do lose, because the executor cannot service what they produce -- 76 tiles
planted, 65 plants lost to thirst, harvests halved (E24). CEM correctly reports that the best
available configuration of a bad executor does not buy land.

So the co-adaptation story is dead as an explanation. The parameters were never the binding
constraint; they were a faithful fit to an executor that cannot convert area into money.

### Consequence

**Parameter search is exhausted as a route.** It has now been given the full knob set, the real
opponent, and held-out validation, and returns the incumbent. Further CEM runs on this engine are
not worth the compute.

The remaining work is the executor itself -- how tasks are generated, assigned and sequenced -- and
E29 is a warning about how that goes: three principled corrections there each measured worse in
isolation. The thing to change is the *structure* of assignment, not another local rule, and the
gate is winrate vs `boatlee` (`tools/routing_bench.py`, 24 games in ~5s).


---

## E31 — Where the wall-clock actually goes

Measured after E30 ruled out parameter search, to find out what search budget is available for the
alternatives. 20 full 719-turn episodes per row, single core.

| what is running | ms/episode | episodes/s |
|---|---:|---:|
| kagsim alone, constant action, **no `observation()` built** | **2.0** | **497** |
| + building the Python observation dict each turn | 15.6 | 64 |
| + boatlee's policy (deep-copies its action table per turn) | 110.4 | 9 |
| + our engine's policy | 78.6 | 13 |
| CEM as actually run, 8 workers | — | ~38 |

**The simulator is 2ms; everything else is Python.** Building the observation dict costs 7x the
simulation it describes, and the policies cost 4-6x again.

Two consequences for PLAN2:

* An **open-loop plan needs no observation at all** -- it indexes a table. So trajectory
  optimisation (P4) can run at ~500 episodes/s/core, roughly **100x the throughput the CEM run
  used**, which is the strongest argument that boatlee's approach was affordable to whoever built
  it and is more affordable to us.
* Our submission uses **2.1 ms of Kaggle's 1000 ms per-turn budget**, so inference-time search (P1)
  has ~500x headroom -- but only if a forward model exists in pure Python, since kagsim is not
  importable on the runner. That is P0, and its kill criterion is a speed measurement.


---

## E32 — P0 feasibility probe: a pure-Python farm model is fast enough, by 286x

PLAN2 P0 carried the plan's first `[ASSUMED]`: that a farm forward model can reach **>=2,000
farm-steps/s in pure Python**. Below that, inference-time search (P1) does not fit in Kaggle's turn
budget and the main line dies. Tested immediately, before any of P0 was built.

Method: a faithful *skeleton* of the model's hot loop -- per-unit op application plus the daily
tile refresh (watering, death at 2 dry days, ongoing-crop accrual with the fertilizer doubling,
animal feeding and escape). Plain dicts and lists, no numpy, since the submission bundle forbids it.
20 reps x 719 turns per row, single core.

| farm | farm-steps/s | 719-turn rollouts/s |
|---|---:|---:|
| ours today (10 crops, 14 animals, 11 units) | **836,190** | 1,163 |
| boatlee-scale (63 crops, 12 animals, 11 units) | **701,693** | 976 |
| full board (75 tiles, 15 units) | **572,652** | 797 |

**Passes by 286x at the worst size.**

### What it buys inside the turn budget

Our submission currently uses 2.1 ms of Kaggle's 1000 ms per turn (E21). At ~570k steps/s, a
3-day lookahead (72 turns) costs ~0.13 ms, so the budget allows on the order of **thousands of
short-horizon rollouts per turn** -- far more than the handful P1 needs to compare a few candidate
assignments.

### Honest limits of this probe

* It is a skeleton, not the model. The real one adds inventories, structures, seeds, more ops, and
  bounds checking -- call it 3-10x heavier. At 10x it still clears the criterion by ~28x.
* It does not prove *correctness*, only speed. P0's parity test against kagsim is still required
  and is the real work.
* Weed spawning is stochastic and deliberately excluded; rollouts will have to treat it as noise.

**Verdict: P0's kill criterion is cleared with margin large enough that model weight is not the
risk. Correctness is.**


---

## E33 — Kaggle changed the rules (1.32.4 -> 1.32.6)

Announced as two balance changes. The diff contains **three**, and the undocumented one is a
mechanics fix.

### What actually changed

1. **Town-centre demand cut ~4.7x.** `TOWN_CENTER_DEMAND_SCHEDULE = [(20,4),(10,2),(0,1)]` is
   deleted -- the centre now buys exactly 1 of each non-fertilizer product per tick, flat -- and
   `townCenterSellInterval` moved 12 -> 24. Seasonal centre demand per product goes
   **140 -> 30 units**.
2. **Shops drawn WITH replacement**, capped at `MAX_SHOP_INSTANCES = 8`. Previously each of the 8
   shops unlocked at most once, so by late game every product's demand was known and identical in
   every game. Now a product can have zero shops or four.
3. **Undocumented: shed operations moved ahead of the LOCKED guard.** DROP / PICKUP / PLACE now
   resolve before the tile-ownership check, because three of the four shed-access tiles start
   LOCKED. A genuine bug fix.

### The guard fired, and it worked

`tests/test_env_version.py` (V6, built one session earlier) failed on all three assertions -- both
source hashes and the version string -- with the message telling us to read the diff and re-verify
parity **before** moving the pin. That is exactly the sequence that followed. Without it, kagsim
would have kept producing confident numbers under obsolete rules with nothing failing.

### kagsim updated and re-verified

All three changes ported. `make verify`: **0 divergences, every simulation line covered, suite
passed.** Bit-exact against the reference driven by the third-party agent on all three probe seeds
(152706/62873, 147473/79588, 144811/62413). Baseline constant updated 3495 -> 3488, confirmed
against the reference.

### Measured impact

| | 1.32.4 | 1.32.6 | change |
|---|---:|---:|---:|
| our money vs boatlee | $69,804 | **$32,269** | **-54%** |
| their money vs us | $152,120 | $113,419 | -25% |
| our money vs `starter` | $106,457 | $89,857 | -16% |
| their money vs `starter` | $193,177 | $152,889 | -21% |
| head-to-head record | 0/40 | **0/16** | unchanged |

**The change hurt us roughly twice as hard as it hurt them**, widening the money gap from 2.2x to
3.5x.

### Why: our product mix is now badly exposed

Shop demand per product across 200 seeds (units drained per shop tick):

| product | mean | min | max | **P(zero shops)** |
|---|---:|---:|---:|---:|
| WHEAT | 5.0 | 1 | 8 | **0%** |
| STRAWBERRY | 3.7 | 0 | 8 | **0%** |
| CARROT | 3.2 | 0 | 12 | 12% |
| MILK | 2.8 | 0 | 7 | 2% |
| EGG | 2.1 | 0 | 7 | 8% |
| TOMATO | 2.0 | 0 | 5 | 8% |
| **WOOL** | 2.0 | 0 | 8 | **36%** |
| MELON | 0.0 | 0 | 0 | 100% |

The champion runs **8 sheep for wool (no buyer in 36% of games)** and a **melon-led crop mix (no
buyer ever)**, on top of a town centre whose demand just fell 4.7x. Only WHEAT and STRAWBERRY are
guaranteed a shop buyer in every game.

### The strategic consequence

D17 said *rank a product by how many shops demand it*. Under 1.32.4 that ranking was a **constant**
-- every game unlocked the same 8 shops -- so it could be baked into a parameter vector. Under
1.32.6 it is a **per-game random variable that is fully observable at runtime** in
`obs["town"]["unlocked_shops"]`.

That is a new, cheap edge and it favours adaptive agents over fixed plans. A static mix must bet on
the average; an agent that reads the shop list can stop breeding sheep when no yarn store exists.
It also weakens `PLAN2` P4 (trajectory optimisation), since an open-loop plan cannot condition on a
draw it has not seen -- boatlee already carries a hand-written `YARN_STORE` check, which suggests
its author saw the same problem coming.

**No conclusion from E1-E32 about crop or animal mix survives this change.** The causal chain about
*land and servicing capacity* (E23-E30) is untouched -- it is about production, not demand.


---

## E34 — The causal chain, re-tested under 1.32.6

`PLAN2` §2 asserted that the E23–E30 chain survived the rule change "because it is about
production, and production mechanics did not change." That was an assumption written into a plan.
Tested, and **it is wrong**.

E26 re-run in kagsim (verified bit-exact under 1.32.6), 16 seeds x 2 seats:

| opponent | our winrate | ours | theirs | 1.32.4 winrate |
|---|---:|---:|---:|---:|
| full agent | 0% (0/32) | $26,464 | $122,028 | 0% |
| minus FERTILIZE | 0% (0/32) | $28,372 | $84,806 | 0% |
| **minus BUY_LAND** | **25%** (8/32) | $42,471 | $52,553 | **100%** |
| minus both | 25% (8/32) | $42,166 | $47,368 | 100% |

### What changed

**`BUY_LAND` is no longer sufficient to explain their win.** Removing it used to flip the match
completely (100% to us). It now leaves us losing 3 games in 4.

More telling: stripped of **both** land and fertilizer, they earn $47,368 against our $42,166. The
same crippled agent under 1.32.4 lost to us by $23,734. **A ~$29,000 swing against us, with land
and fertilizer both removed from the comparison** — so it cannot be attributed to either.

### What it is instead

The demand cut (E33) punishes product mix, and ours is the exposed one: 8 sheep for **WOOL, which
has no shop buyer in 36% of games**, and a melon-led crop mix with **no shop buyer ever**, against
a town centre whose demand fell 4.7x. Their mix is wheat- and strawberry-led — the only two
products guaranteed a buyer in every game (E33).

### Revised chain

1. Land is **still worth a lot** — removing it costs them $69,475 and moves us 0% -> 25%. Not
   refuted, just no longer the whole story.
2. **Product mix is now an independent, comparable deficit**, and it did not exist as a separate
   term under 1.32.4 because demand was a constant every game.
3. Servicing capacity (E24, E29) is untouched — that evidence is about converting area into
   harvests, and it stands.

### Process note

This is the fourth time a conclusion has been correct about the system that produced it and wrong
once that system changed — land under our engine (E26), melon under the `elif` bug (E24),
steps-per-action under task density (E29), and now the whole chain under 1.32.4's demand model.
The plan had it as prose within an hour of the rule change; the measurement took ten minutes.


---

## E35 — P1.5-A: shop-adaptive product selection, killed

E33/E34 made this look like the highest-value task: shops now draw with replacement, wool has no
buyer in 36% of games, the champion breeds 8 sheep for it, and the draw is fully observable. The
kill criterion was fixed before the work — *beat the fixed mix on the zero-yarn-store subset, or it
is dropped.*

### Result

24 seeds x 2 seats vs `boatlee`; 7/24 seeds had no yarn store.

| config | all seeds | no-yarn subset |
|---|---:|---:|
| **fixed mix, `water_mode=elif`** (the champion) | **$33,706** | **$32,636** |
| fixed mix, `water_mode=both` | $28,644 | $24,462 |
| adaptive, `water_mode=both` | $24,949 | $21,260 |
| adaptive by revenue capacity, `elif` | $18,870 | $11,121 |
| adaptive by unit demand, `elif` | $28,904 | $23,420 |

Every adaptive variant loses, and loses hardest on the subset where it was supposed to win.
Winrate stayed 0% throughout.

### Two corrections tried, both recorded

1. **Weighting by unit demand** was wrong on its face — it gives wheat ~31x melon's weight while
   wheat sells at $25 and melon at $250. Replaced with **revenue capacity** (units/day x base
   price), which ranks strawberry > wheat > melon and matches the leaderboard agent's implied
   ordering. It scored *worse*, not better.
2. **The `ongoing`-crop interaction** was a real mechanism worth ruling out: adaptive weighting
   pushes toward strawberry, and `water_mode="elif"` kills ongoing crops by thirst (E24). Setting
   `water_mode="both"` recovered ~$6k of the penalty and did not change the verdict.

### Why it fails, and what it tells us

**Selection does not matter while servicing is the constraint.** We service 10-15 crop tiles.
Choosing *which* crop occupies those tiles is a second-order decision when the first-order problem
is that we cannot occupy more of them. Market capacity only binds an agent producing near it —
boatlee, at 63 tiles, is that agent; we are not.

Note the champion sells ~100 melon into a market that absorbs ~30, so it *is* over-saturating its
chosen crop. Fixing that still loses, which is the point: the loss is not coming from the mix.

### What is kept

* **The demand-model fix stays** — it is a correctness fix independent of this idea.
  `town_drain_per_day` still used the deleted `TOWN_CENTER_DEMAND_SCHEDULE` and the old 12-turn
  interval, overstating town-centre absorption by **2-8x**. Dead code for the champion
  (`forecast_weight=0`), live for anything that forecasts. Covered by
  `tests/test_demand_model.py` (11 tests, incl. >=100 real kagsim draws).
* `adaptive_mix`, `demand_exponent`, `animal_demand_floor`, `animal_min_target` remain as
  `Params`, **off by default**, champion bit-identical. Worth re-testing *after* P1 — the
  hypothesis is not disproven for a 40-tile farm, only for a 15-tile one.

### Plan consequence

**P1.5-A is un-promoted and re-gated behind P1.** E34 moved it ahead of P0/P1 on the reasoning
that the mix deficit was independent of servicing. That reasoning was wrong: it is downstream of
servicing. P0 -> P1 is the main line again.


---

## E36 — P0: the farm forward model, and what mutation testing found

`agent/forward.py` — pure-Python model of our own farm (tiles, crops, animals, unit positions,
inventories, shed). No numpy, no kagsim, ships in the submission bundle. Not the market, opponent
or money: those need the other player.

### Verification

`tests/test_forward_parity.py`, 31 tests:

* multi-unit fuzz within a day, and 220-turn runs across many day boundaries, at
  `weedSpawnChance` **0.0 and 1.0** — the two deterministic extremes, so the weed branch is
  covered rather than excused as noise
* directed scenarios for branches fuzzing never reaches: fertilizer bonus, fertilizer expiry
  boundary, ongoing-crop bonus requiring water, `count > max_yield`, animal escape, FEED without
  wheat, care bonus, shed capacity, decay cadence, atomic PLANT validation, shed ops from LOCKED
  tiles

### Mutation testing: 14/14 caught, and four tests that proved nothing

A passing parity test that cannot fail is worse than none — it licenses a rollout built on a wrong
model. So 14 deliberate bugs were introduced into a **scratchpad copy** of the model (never the
repo file) and the suite had to catch each.

The first run caught **5 of 10**. The gaps it exposed were all tests that looked fine:

| what mutation testing found | why the test proved nothing |
|---|---|
| fertilizer branches unreachable | the fuzz never fertilized: `FERTILIZE` needs FERTILIZER in hand, which the random walk essentially never arranged |
| the fertilizer scenario never fertilized either | it planted a crop and did not water it on the planting day — `_new_plant` starts `consecutive_unwatered` at 1 (`:209`), so the plant was a weed by nightfall |
| the fertilizer *duration* boundary invisible | applying on day 2 covers WHEAT's whole window (days 2-4), and the yield cap of 6 hid the difference; applying on **day 1** puts the expiry inside the window |
| shed capacity never clamped | the harness never passed `shedCapacity` to the model, so it clamped at 100 while the simulator clamped at 2 |
| `count > max_yield` unobservable | `min(max_yield, ...)` clamps a stray accrual to the same number — only visible if the tile is **harvested** first |

Two apparent misses were **false**: the `sed` patterns did not match the source, so nothing was
mutated. The harness now `cmp`s against a pristine copy and reports `NO-OP` instead of `MISSED`.

### Speed (P0.3)

| | |
|---|---|
| real model | **208,003 farm-steps/s** — 104x the 2,000/s kill criterion |
| 3-day lookahead (72 turns) | **0.346 ms** of the 1000 ms turn budget |

The E32 skeleton predicted 572k-836k; the real model is ~3x heavier, exactly inside the "even 10x
heavier still clears it" margin recorded then.

### Status

P0.1-P0.4 complete. `make submission` builds with `agent/forward.py` included, both seats DONE,
$101,646, worst turn 2.0 ms. Full suite 233 passing. **P1 is unblocked.**


---

## E37 — At equal land, the gap is still servicing

E34 left a loose end: stripped of **both** land and fertilizer, boatlee still out-earned us. That
is a comparison on *equal land*, so it cannot be explained by the land deficit — and P1's entire
premise is that servicing capacity explains the gap. Worth testing before building on it.

8 seeds, 1.32.6, boatlee's `BUY_LAND` orders removed so both farm one quadrant:

| | boatlee (no land) | our champion |
|---|---:|---:|
| money | **$52,926** | $32,208 |
| peak crop tiles | 19 | 15 |
| **plants lost to weeds** | **1** | **7** |
| **crop work (share of unit-turns)** | **25.2%** | **5.8%** |
| hauling | 2.9% | 9.5% |
| movement | 42.8% | 56.3% |
| productive | 42.2% | 28.0% |

On the same 25 tiles they do **4.3x our crop work** and lose **one plant to our seven**. Tile
*count* is nearly equal (19 vs 15) — what differs is what happens to those tiles once planted.

**P1's premise holds and is now directly evidenced rather than inferred.** The earlier
decomposition (E29) was taken at unequal land, so servicing and scale were confounded; this
separates them.

### A better set of reference points

The 63-vs-10 tile comparison is land-confounded and flatters the diagnosis. These equal-land
numbers are the ones P1 should be measured against, because they are the part of the gap P1 can
actually move:

| metric | ours | target (boatlee at equal land) |
|---|---:|---:|
| crop work, share of turns | 5.8% | **25%** |
| plants lost per season | 7 | **<=2** |
| hauling share | 9.5% | **<5%** |
| productive share | 28.0% | **42%** |


---

## E38 — The value function, and a gate that would have made it worse

P1.1 added `FarmModel.score()` — goods in hand, yield already accrued, and yield a tile can still
produce before the season ends, discounted by `unripe`. Deliberately simple: the rollout is meant
to supply the intelligence, and a clever value function is a second thing that can be wrong.

### The pre-set gate fails

`TASKS2` required **Spearman rho >= 0.6** between score at day 15 and realised final money, over
>=200 episodes with varied parameters:

| `unripe` | day 10 | day 15 | day 20 |
|---:|---:|---:|---:|
| 0.0 | +0.414 | +0.384 | +0.668 |
| 0.5 | +0.416 | +0.521 | +0.715 |
| 1.0 | +0.396 | +0.570 | +0.725 |
| 1.5 | +0.374 | **+0.584** | +0.720 |

Day 15 tops out at **0.584**, below the 0.6 bar. Day 20 clears it comfortably. (A 60-episode run
had reached 0.616 at `unripe=1.5`; it did not survive the specified sample size — which is the
reason the spec asked for >=200.)

### But the gate measures the wrong thing, and measurably so

A rollout never compares two strategies fifteen days apart. It compares candidate assignments from
**one common state, a few days ahead**. So the relevant test is pairwise: same farm, same seed, two
different action sequences, score both at +3 days — does the higher score end the season richer?

| `unripe` | pairwise accuracy at day 15, n=96 |
|---:|---|
| 0.0 | **76.0%** [66.6, 83.5] |
| 0.25 | 75.0% [65.5, 82.6] |
| 0.5 | 75.0% [65.5, 82.6] |
| 0.75 | 72.9% [63.3, 80.8] |
| 1.0 | 70.8% [61.1, 79.0] |
| 1.5 | **66.7%** [56.8, 75.3] |

Every interval excludes 50%, so the function is genuinely informative for the job it has.

**The two metrics disagree about the parameter, and in opposite directions.** Correlation rises
monotonically with `unripe` and peaks at 1.5; pairwise accuracy *falls* monotonically and is worst
at 1.5. Tuning to the original gate would have set `unripe = 1.5` — a value that is also
economically incoherent (future yield worth more than banked yield) and the **worst** of the six
for ranking, at 66.7% against 76.0%.

### Gate replaced, deliberately and on the record

New P1.1 gate: **pairwise ranking accuracy with the Wilson interval excluding 50%**, target >=70%.
That is the principled bar — a value function that cannot beat a coin flip at ranking nearby
futures is useless to a rollout, and one that can is usable regardless of how it does at
cross-strategy forecasting.

Changing a pre-set gate is exactly the move this project's rules exist to prevent, so: the original
is recorded as failed above, the replacement was measured before being adopted, and the reason is
that the old gate demonstrably tunes the parameter the wrong way. Not because the number was
inconvenient.

`unripe` stays at **0.5**. 0.0 and 0.5 are statistically indistinguishable here (76.0% vs 75.0%),
and the perturbation used varies *assignments* rather than planting decisions, so this experiment
cannot separate them. Re-tune once P1.2 defines the real candidate set.

### Known limitation, recorded now rather than when it bites

**Goods are valued at `BASE_PRICE`, which is not what they sell for.** Market price falls as
inventory rises, and 1.32.6 cut absorption hard (E33): the town centre takes ~30 units per product
per season, shops draw with replacement, MELON has no shop buyer in any game and WOOL none in 36%.
So `score()` prices a field of melons at $250/unit into a market that cannot take them.

Why it is acceptable *here*: P1 compares candidate **worker assignments** from a common state, and
the alternatives involve the same crops, so a shared mispricing largely cancels in the ranking.
The 75% pairwise accuracy above is measured with exactly that mispricing present.

Why it is not acceptable elsewhere: anything choosing **what to plant or breed** would be misled,
which is one reason P1.5-A failed (E35). If P1.2's candidate set ever includes planting decisions,
this must be revisited -- pass a demand-aware price table built from `Engine.town_drain_per_day`,
and **measure it**, since three plausible corrections have already cost money.

### Also added

`seeds` now count toward score, at purchase cost. They are purchased capacity that cannot be sold,
and a farm holding seeds is not the same as one that has spent everything. It did not rescue the
correlation gate.


---

## E39 — Solve the assignment instead of searching it

P1.2 was specified as *generate K candidate assignments, roll them out, pick the best*. Building it
surfaced two things that redirected the phase.

### The candidate space was nearly empty

At day 10 the engine had **5 tasks for 11 units**, three of them needing an item nobody carried, so
every legal permutation collapsed back to the greedy plan. Task supply turns out to be bimodal:

| period | doable tasks vs units |
|---|---|
| days 3-6 | **5-7 tasks for 10 units** (~4 units idle) |
| days 12+ | 20-23 tasks for 10 units |

**53% of turns have fewer doable tasks than units** (mean 16.4 vs 9.8). Testing the search in the
early lull measures an empty task list rather than the search.

### Greedy is 9.6% off optimal — and optimal is exactly solvable

**[CORRECTED — see the note below. The first version of this section said 18.0%.]**

Comparing the engine's real plan against a true min-cost assignment on the same tasks, five seeds,
~620 tasked turns each:

| seed | greedy | optimal | penalty |
|---:|---:|---:|---:|
| 3 | 5,224 | 4,691 | +11.4% |
| 5 | 5,045 | 4,798 | +5.1% |
| 7 | 5,390 | 4,933 | +9.3% |
| 11 | 5,245 | 4,724 | +11.0% |
| 13 | 5,224 | 4,691 | +11.4% |
| **pooled** | **26,128** | **23,837** | **+9.6%** |

**Correction, and the methodology lesson.** The original +18.0% came from a single seed and, worse,
from a **re-implementation of the greedy rule inside the analysis script** rather than the engine's
own plan. That re-implementation ignored sticky assignments and the priority weight, so it was
worse than the real engine and the gap it "measured" was largely its own. The engine now records
`_last_plan` for diagnostics, because routing quality cannot be recovered from the emitted actions
(they are direction steps) and re-deriving it in a script measures the script.

Same family of error as the `sed` patterns that silently did not match during mutation testing
(E36) and the `grep` that reported a documentation edit had failed when it had not: **the
measurement apparatus is as capable of being wrong as the thing measured.**

The end-to-end money numbers below are unaffected -- they were always measured by playing whole
episodes, never by re-deriving anything.

There is no reason to *search* for something exactly solvable at this size. A pure-Python
Jonker-Volgenant solver (no numpy — it must run on the Kaggle runner) does 12 units x 30 tasks in
**66 us**, against a 1000 ms turn budget, and matches `scipy.optimize.linear_sum_assignment` on
300 random matrices.

### Measured end to end: the first gain this session

`assign_mode="optimal"`, vs `boatlee`, three independent seed sets:

| seeds | sequential | optimal | change |
|---|---:|---:|---:|
| 900000 | $35,412 | **$39,266** | +10.9% |
| 400000 | $39,051 | **$40,651** | +4.1% |
| 700000 | $59,329 | **$61,690** | +4.0% |

Movement falls **56.3% -> 52.9%** in every set. Winrate is still 0% — they earn $124-134k — so
this closes part of the gap, not the gap.

Worth stating plainly: **five engine changes this session measured worse** (water policy, global
assignment, in-flight fetch accounting, fertilizer, adaptive mix). This is the first that measured
better, and the difference is that it replaced a heuristic with an exact solution to a
well-specified sub-problem rather than adding another rule.

### Consequence for P1

**P1.2/P1.3 are superseded for assignment.** The rollout search was aimed at a problem that can be
solved outright. `FarmModel` (P0) and `score()` (P1.1) are not wasted — they remain the tools for
decisions that are *not* exactly solvable — but they are no longer on the critical path for
assignment.

The remaining servicing gap is therefore **not** about matching units to tasks. Optimal matching is
now in hand and the gap is still large, which points at the task list itself: 5-7 doable tasks for
10 units through the early game, and crop work at 5.8% of turns against boatlee's 25.2% (E37).


---

## E40 — The promotion gate refuses optimal assignment, correctly

E39 measured `assign_mode="optimal"` at +4-11% money against `boatlee` across three seed sets.
Put through `make promote` it **failed stage 1**:

```
39.6% [35.4, 44.0] n=500   $49,474 vs $48,729   -> loss
NOT PROMOTED
```

**It earns more money and wins fewer games.** Mean margin +$745 across 500 games; the interval on
winrate is clear of 50% in the wrong direction.

### Where it helps and where it does not

24 seeds x both seats:

| opponent | greedy | optimal |
|---|---|---|
| `boatlee` | 0% win, $28,642 | 0% win, **$32,213** |
| mirror (`champion`) | tie, self-play | **45.8% win**, mean **+$1,642**, median **-$424** |
| `x-dumper` | 8.3% win, $38,515 | **12.5% win**, $40,088 |
| `starter` | 100% win, $84,716 | 100% win, $83,683 |

The distribution explains the contradiction: in the mirror it **usually loses by a little and
occasionally wins by a lot**. Mean money says improvement; median margin and winrate say
regression.

### Why this is the gate working

`CLAUDE.md` already records the rule — *mean money is only comparable within a fixed opponent
field; skill and pairwise winrate are the ranking*. E5 established it across different strategies.
E40 shows it holds **inside one strategy family**, where the only difference is how units are
matched to tasks.

A plausible mechanism: both sides share one market. Producing more in a mirror floods it for both,
so part of the gain is competed away — the same coupling E26 measured when boatlee's land purchase
cut *our* revenue by $14k.

### Not promoted, and not argued around

`champion.json` is unchanged and `assign_mode` stays `"sequential"` by default. The measured facts
are: better money against every non-mirror opponent, better winrate against `x-dumper`, no change
against `starter`, and a resolved winrate loss in the mirror.

That is a genuinely mixed result, and the mirror match is a proxy for the leaderboard rather than
the leaderboard itself — we will never actually face ourselves. But stage 1 exists because five
straight promotions were made on evidence weaker than this, and "the gate is measuring the wrong
thing" is exactly what a wrong promotion sounds like from the inside. Recorded as refused.

**What would settle it:** a second external opponent, or V1. Against the one real agent we have,
optimal assignment is strictly better on money and neither version wins a single game — so the
matchup that decides this is one we cannot yet run.


---

## E41 — Step-by-step comparison: E37 does not replicate, and the real failure is scaling

Prompted by a request to compare money **day by day** rather than in season totals. That is a
better instrument than anything used so far, and it overturned one of this session's own results.

### The day-by-day picture (equal land)

| day | boatlee $ | crops | ours $ | crops |
|---:|---:|---:|---:|---:|
| 8 | 1,696 | 19 | 185 | 15 |
| 12 | 12,421 | 19 | 8,442 | 11 |
| 18 | 26,347 | 19 | 23,270 | 10 |
| 22 | 37,298 | 18 | 25,512 | **9** |
| 28 | 47,409 | 9 | 31,818 | **9** |

**Their crop count holds at ~19 all season; ours decays 15 -> 9.** Not through death or idle
land -- our farm has **zero empty tiles all season**. The herd displaces the crops: 14 animals take
14 of 25 tiles.

Late game, they sell strawberry repeatedly (28, 34, 32 units on days 18/22/24) while we sell wheat
and fertilizer and finish with a **262-unit wheat dump** on day 28 -- bought feed being liquidated.

### E37 does not replicate

E37 claimed *"at equal land they earn 64% more"* and set P1's gate targets from it. Re-measured:

| seeds | n | our winrate | ours | theirs |
|---|---:|---|---:|---:|
| 900-907 (**E37's block**) | 16 | 12.5% | $32,208 | $52,926 |
| 950-961 | 24 | 50.0% | $53,544 | $56,067 |
| **1000-1039** | **80** | **45.0% [34.6, 55.9]** | **$49,363** | $54,397 |

**At equal land we are at statistical parity.** E37 drew an unlucky 8-seed block and a whole
framing was built on it -- including PLAN2 §2's claim that product mix is an independent deficit,
and the P1.5 gate numbers. Corrected below.

Shop draws now vary per game (E33), so 8-16 seeds is simply not enough for this comparison; that
was already recorded in the P1.5 gate note ("never read this off fewer than ~16 seeds") and then
not applied.

### So land is the dominant factor after all

This restores E26 and walks back part of E34. Their advantage is land, and ours is that we cannot
use it. Testing land on the current engine, 30 seeds x 2 seats vs the full boatlee:

| config | our money | peak crops |
|---|---:|---:|
| champion, no land | **$34,140** | 15 |
| + land | $4,439 | 76 |
| + land + optimal assignment | $7,566 | 77 |

### Our marginal crop is negative

Capping how much we plant, with land bought:

| `tiles_per_unit` | peak crops | money |
|---:|---:|---:|
| 12 (current) | 77 | $7,566 |
| 6 | 55 | $14,426 |
| 4 | 30 | $23,723 |
| 2 | 15 | $32,541 |
| *(no land)* | 15 | **$34,140** |

Monotonic: **every crop beyond ~15 costs us money**, while boatlee farms 63 profitably.

### Because we scale the wrong crop

MELON is our largest holding and the town absorbs ~30 units of it a season (E33). Sixty melon tiles
produce roughly 360. Changing the mix at scale:

| config (land, tpu=6) | money | peak crops |
|---|---:|---:|
| melon mix | $14,426 | 55 |
| wheat + strawberry | $23,971 | 58 |
| **wheat + strawberry, `water_mode=both`** | **$32,665** | 59 |
| wheat only | $29,895 | 57 |

The mix recovers **most of the land penalty** ($14k -> $33k). This is why E35 saw nothing: at 15
tiles the mix barely matters, and E35 tested it at 15 tiles. **Mix and scale interact, and neither
alone is worth anything.**

### What is still unexplained

At 59 crops with a sensible mix we earn **$32,665**; boatlee at 63 crops earns **$130,000**. Same
scale, comparable crops, 4x the money. Land now pays for itself but nothing more.

The candidate is conversion: they harvest 390 times a season to our ~137. Holding a tile is not the
same as collecting from it. That is the next thing to measure, and it is a sharper question than
anything the servicing framing produced.


---

## E42 — The scaling configuration does not survive a fresh sample either

E41 ended with land roughly breaking even once the crop mix was fixed. Pushing further found what
looked like a real gain, and it evaporated on more seeds — an hour after E41 documented exactly
that failure mode.

### What looked promising

At scale (land, wheat+strawberry, `water_mode=both`, optimal assignment, `tiles_per_unit=6`),
sweeping the herd on 18 seeds:

| config | money |
|---|---:|
| champion (no land) | $31,787 |
| scale, herd 6c+8s | $29,071 |
| scale, herd 4c+4s | $41,950 |
| **scale, herd 4c+0s** | **$46,466** |

Dropping sheep entirely looked best, which fitted E33 nicely — wool has no shop buyer in 36% of
games. **+46% over the champion.** Adding labour was catastrophic ($5,177 at 14 hands: hiring costs
compound and bankrupt the farm) and fertilizer still lost money.

### On 40 fresh seeds it is not there

| | 18-seed block | **80 fresh games** |
|---|---:|---:|
| candidate vs `boatlee` | $46,466 | $41,774 |
| champion vs `boatlee` | $31,787 | $40,131 |
| apparent advantage | **+46%** | **+4%** |

And head-to-head, which is the gate's stage 1:

```
candidate vs champion:  17.5% [10.7, 27.3] n=80   $60,732 vs $74,516
```

A decisive loss. **No promotable improvement.** `champion.json` unchanged.

### The pattern, now three times in one session

| claim | small sample | large sample |
|---|---|---|
| equal-land gap (E37) | 64% gap, n=16 | parity, n=80 |
| optimal assignment (E39/E40) | +4-11% money, n=24-48 | resolved winrate loss, n=500 |
| scaling config (E42) | +46%, n=36 | +4%, and a head-to-head loss, n=80 |

Every one looked like a real effect at 16-48 games. **Since 1.32.6 made shop draws vary per game
(E33), the between-seed variance is large enough that nothing below ~80 games should be believed**,
and the P1.5 gate's own note ("never read this off fewer than ~16 seeds") was too lenient by a
factor of five.

The habit to fix is not "run more seeds when it matters" — it is that a promising number *is* the
signal to re-run on fresh seeds, before building anything on it or writing it into a plan.

### What does survive

The diagnostic findings from E41 are unaffected, because they are structural rather than marginal:
our crop count decays as the herd displaces it, our farm holds zero empty tiles, melon cannot be
sold at scale, and at comparable tile counts we harvest 281 times to their 390 while losing 37
plants to their 9. Those are large, repeatable differences. What has failed is every attempt to
convert them into money.


---

## E43 — Following the crops, and why melon is not the mistake

Prompted by a simple question: *why not save the actions and field state and diff them against
boatlee?* Every diagnostic until now had been a throwaway script printing to a terminal. Recording
the episodes to files (`tools/trace.py`, output in `traces/`) answered in twenty minutes what
several rounds of aggregate statistics had not.

### What the trace showed

The champion plants **3 strawberries a season**, first one on day 10. boatlee has 20 in the ground
by day 9 and 36 by day 12, identically on every seed -- it is a fixed script, so it plants the same
farm regardless of which shops spawn.

Forcing a strawberry-only mix and following individual plants:

| | boatlee | ours (strawberry-only) |
|---|---:|---:|
| strawberry planted | 37 | 29 |
| strawberry sold | **286** | 62 |
| **units per plant** | **7.7** | **2.1** |
| FERTILIZE ops | 72 | 0 |

Strawberry allows **4 productions per plant, of 1 unit -- or 2 if fertilised**. So the ceiling is 4
unfertilised and **8 fertilised**. boatlee is at 7.7: essentially the fertilised maximum. We are at
2.1, roughly half the *unfertilised* maximum.

### The interaction that explains every earlier failure

Fertiliser alone changes nothing: 60.5 strawberry sold against 60.3 without it, and it costs money.
Eager watering alone was worth ~2 units. **Together they are worth +52% strawberry and +24%
money.**

The mechanism is in the environment: `fertilized = was_watered and fertilized_until_day >= day`
(`kaggriculture.py:786`). The bonus requires the tile to have been **watered that same day**. Our
default only waters an ongoing crop when it is one day from death, so on most days the tile is dry
and any fertiliser on it is wasted.

That is why E24 and E29 and every other single-knob test failed: **these two knobs are worthless
apart and only pay together.** A one-factor-at-a-time search cannot find them.

### And it still does not help

All against the same passive opponent, 30 seeds:

| config | money |
|---|---:|
| **champion (melon mix)** | **$84,517** |
| champion + fertiliser + eager water | $74,425 |
| strawberry-only + fertiliser + eager water | $52,641 |
| strawberry-only | $42,563 |

The pair improves a strawberry farm by 24% and that farm is still **half** the champion. Applied to
the champion it costs $10k, because melon is not `ongoing` (eager watering does nothing for it) and
melon caps its yield anyway (fertiliser does nothing for it) -- so both knobs are pure overhead.

Head-to-head on 90 fresh games the pair loses to the champion **$25,972 vs $59,728**.

### The conclusion that matters

**Melon is not our mistake. It is the best crop our engine can grow**, and the melon mix is the
strongest configuration we have. Earlier framing in E41 -- "we grow the crop nothing buys" -- was
right that melon cannot *scale*, and wrong to imply the champion should not be growing it.

boatlee's advantage is not that strawberry beats melon. It is that they run **57 wheat + 36
strawberry tiles productively** and we cannot run more than ~15 tiles of anything. Every attempt to
change *what* we grow has failed; the constraint is *how much* we can grow, which is where E41 also
landed from the other direction.


---

## E44 — Match every dimension at once, then check the match before reading the result

Every investigation so far changed one knob. E43 showed why that cannot work here: fertiliser and
eager watering are each worth ~nothing alone and +24% together, because the environment grants the
fertiliser bonus only on a day the tile was also watered. If the opponent's advantage is a
*conjunction*, single-knob tests cannot find it -- and neither can CEM, which samples each
dimension independently around a mean (the failure already recorded for `goose_min_cash`, E12).

`tools/profile.py` was written to compare everything at once, and `mimic` to match it.

### The method that makes the result meaningful

**Verify the configuration produced the intended farm before interpreting its money.** Otherwise a
failure means "our engine cannot express this", which is a completely different finding from "this
strategy does not work". Today alone, three measurements were wrong in ways that looked plausible:
a comparison that measured a re-implementation instead of the engine (E39, 2x error), and a profile
tool that sampled the roster at hour 0 -- just after `_end_of_day` clears it -- and reported the
workforce three times too small.

### The configuration reproduces their farm

| dimension | boatlee | `mimic` |
|---|---:|---:|
| wheat tiles | 50 | 49 |
| strawberry tiles | 36 | 42 |
| plant ops | 199 | 196 |
| fertilise ops | 72 | 74 |
| seeds bought | 204 | 210 |
| cows / sheep | 9 / 4 | 9 / 4 |
| mean hands | 8.7 | 8.9 |
| quadrants | 3 | 4 |

Eight dimensions matched. The engine *can* build their farm.

### And the farm dies

| | boatlee | `mimic` |
|---|---:|---:|
| **plants lost to weeds** | **7** | **50** |
| harvests | 390 | 194 |
| waterings | 1,010 | 812 |
| **movement** | **42.8%** | **54.2%** |
| hauling | 2.9% | 7.3% |
| money | $110,262 | $14,563 |

91 crop tiles need ~1,140 waterings a season (every other day for ~25 days). We manage **812**, 29%
short, and **50 plants die of thirst**.

The shortfall and the movement gap are the same quantity: 11 percentage points of ~7,000 unit-turns
is ~780 turns, against a watering deficit of ~330 and a hauling excess of ~300.

### What this establishes

Not "we grow the wrong crops" (E43 already killed that), not "we have too few tiles", not "our herd
is wrong", not "our labour is short" -- all matched here. **The residual is that our units spend a
quarter more of their time walking, and that is exactly what the farm cannot afford at scale.**

This is the same constraint E39/E41 reached from two other directions, now isolated with everything
else held equal. It is also the one thing optimal assignment only moved 56.3% -> 52.9% (E39), which
suggests the remaining travel is not a matching problem but a *layout* problem: which tiles the
farm uses, and how far apart they are.

`mimic` is registered and `champion.json` is unchanged -- at $14,563 it is far worse than the
champion and is a diagnostic, not a candidate.

### Following the deaths in the trace (`traces/boatlee_vs_mimic_s*.json`)

"Under-watering" was too vague. The traces record `consecutive_unwatered` per tile per day, so the
deaths can be attributed rather than inferred:

| | boatlee | `mimic` |
|---|---:|---:|
| crop deaths over the season | **6** | **56** |
| of which at `consecutive_unwatered = 1` | 5 | **46** |
| age at death | 4, 5, **17, 17, 17, 17** | **2, 2, 2, 2, 2, ...** |

**Their plants die of old age; ours die of neglect.** Boatlee's age-17 deaths are strawberry that
has completed its four productions and expired (`max_lifespan_step`, `:789`). Ours die at **age 2**,
46 times out of 56.

A new plant is created with `consecutive_unwatered = 1` -- the planting day counts as unwatered
(`:209`) -- so it must be watered almost immediately or it is a weed within two days. We plant 196
times and then abandon them. Every death costs a seed, a planting turn and the walk to the tile.

Meanwhile ripe tiles accumulate to **57 simultaneously** against 194 harvests all season.

So the failure is not that we water too little in aggregate. It is that **we keep starting plants we
have no capacity to finish** -- planting is cheap to *order* and expensive to *sustain*, and nothing
in the engine ties the two together. `tiles_per_unit` exists to cap this and is set to 12, which
permits 120 tiles for 10 hands and therefore never binds.


---

## E45 — Planting rate, late planting, and why the champion cannot be improved by nudging

E44 left the residual as "movement". Following the traces further showed movement is a *symptom*.

### They plant every day; we plant in bursts

Planting schedules from `traces/boatlee_vs_mimic_s5.json`:

* **boatlee**: day 0 (5 melon + 5 wheat), day 3 (3+1), day 4 (5), day 5 (5), day 7 (14), day 8 (10),
  day 10 (24), day 11 (11), day 12 (8) -- a handful **every single day**.
* **ours**: day 0 (14), day 4 (13), **days 5-14 nothing**, days 15-17 (**46**), ... day 27 (**42**).

A burst creates a wall of tiles that all need water on the same day, all at
`consecutive_unwatered = 1`, and the units cannot cover it -- so the cohort dies together at age 2.
That is 46 of our 56 crop deaths (E44). Their farm reaches 63 crops by day 12 and holds it; ours
oscillates 14 -> 59 -> 34 -> 55.

**Movement is downstream of this.** A burst scatters simultaneous demand across the whole farm so
units criss-cross; a trickle keeps work local. It is why optimal assignment recovered only 3
points (E39) -- it was optimising the routing of an impossible workload.

### The economics behind their trickle

| crop | seed | first yield | max yield | seed cost per unit |
|---|---:|---:|---:|---:|
| **WHEAT** | **$10** | **day 2** | 6 | **$1.7** |
| CARROT | $20 | day 2 | 4 | $5.0 |
| MELON | $80 | day 10 | 6 | $13.3 |
| STRAWBERRY | $100 | day 10 | 4 | $25.0 |

Wheat is 8x cheaper per unit than melon and pays back in **2 days instead of 10**, so it compounds:
plant cheap wheat, harvest on day 2-4, buy more seed, plant again. Their 143 wheat plantings are
the engine of the whole farm. We plant melon and wait ten days for one payday.

### Two caps built, and they are not the same thing

`tiles_per_unit` caps the **stock** of crops. Nothing capped the **rate**. Added
`plant_rate_per_day`, and separately `plant_stop_late` (refuse a crop that cannot reach first yield
before the season ends -- we plant 42 wheat on day 27).

On `mimic`, the rate cap does exactly what the diagnosis predicts:

| rate cap | money | plants lost | peak crops |
|---|---:|---:|---:|
| none | $18,656 | 45.1 | 65.9 |
| 12/day | $23,060 | 33.2 | 50.5 |
| **8/day** | **$32,092** | 21.6 | 39.5 |
| 3/day | $26,211 | **5.8** | 15.6 |

Deaths fall monotonically; money peaks at 8/day, **+72%**.

### But on the champion the two caps go opposite ways

90 fresh games, head-to-head vs the champion:

| change | winrate |
|---|---|
| **`plant_stop_late`** | **62.2% [51.9, 71.5]** -- a win |
| rate cap 8/day | 0.0% |
| rate cap 4/day | 0.0% |

The champion plants 53 times a season, so a rate cap is pure restriction. `plant_stop_late` passed
**gate stage 1 at 61.2% [56.9, 65.4] over 500 games** and was then **refused at stage 2**, losing to
`x-dumper`, `r-melon`, `herd5c8s` and `boatlee` -- opponents the champion also loses to, but by
less. Not promoted.

### The conclusion that reframes the session

Every change tried today lost: water policy, global assignment, in-flight fetching, fertiliser,
adaptive mix, optimal assignment (more money, fewer wins), the scaling config, and now the rate cap.
`plant_stop_late` beat the champion head-to-head and still failed the gauntlet.

That is not eight bad ideas. **It is what a tuned local optimum looks like from the inside**: CEM
fitted 33 knobs to a melon farm, and any single perturbation breaks the fit long before it can reach
a different basin.

So the champion should be left alone and a *different* configuration built and tuned in its own
right, then compared. `mimic` is that starting point -- it reproduces boatlee's farm structurally
(E44) while carrying the champion's parameters everywhere else, which is exactly the mismatch a
search should resolve. `search/cem.py --start` was added for this.

### A knob the search could not reach

Wiring that search exposed a real defect: **`assign_mode="optimal"` was never added to the search
space.** It was added to the engine in E39 and to `Params`, but the `Knob` listed only
`("sequential", "global")`, so CEM could not select it at any point. A knob the engine has and the
search cannot express is worse than no knob at all. Space is now 42 dimensions and every registered
configuration round-trips through encode/decode.


---

## E46 — Read the opponent's plan rather than inferring its mechanism

Every explanation of boatlee's execution so far had been inferred from aggregate statistics. Its
719-step action table is decompressed in the repo; reading it directly answered in one query what
several rounds of statistics had not.

### What its units actually do

| unit | dominant ops | reads as |
|---|---|---|
| 0 | CARE 66 | animal carer |
| 1 | COLLECT_FERTILIZER 78 | fertiliser collector |
| 4 | FEED 82, COLLECT 74 | animal handler |
| 5 | FEED 73, CARE 66, COLLECT 66, **y range 0-4** | animal handler, restricted area |
| 2, 3, 6-11 | WATER 45-169 | field hands |

**Role specialisation.** Measured across the season:

| | role purity | switches between crop and animal work |
|---|---:|---:|
| boatlee | **93.1%** | **10.2%** |
| our champion | 68.6% | **33.4%** |

Pastures and fields are in different places, so a unit that alternates pays a crossing each time.

### Implementation, and three defects the tests caught

`Params.role_penalty` adds a cost, in tiles, for taking work outside a unit's role. Roles are
derived from the farm's composition and applied in all three assignment modes. Writing
`tests/test_roles.py` exposed three problems in the first version, none of which the aggregate
money numbers would have revealed:

1. **Roles were recomputed every turn** -- a unit could be a livestock hand one turn and a field
   hand the next, which is the churn the penalty exists to prevent. Now sticky per day.
2. **`self._day` was dead state**, initialised to -1 in `__init__` and never assigned. The role
   cache was keyed on it, so it would never have rotated. Now maintained.
3. **The split was sized from the momentary task list.** At dawn that list is almost entirely
   "feed everything", so most of the roster was locked to livestock, finished by mid-morning, then
   *barred* from the fields. This is why purity peaked at 0.68 and **fell** at higher penalties.
   Now sized from tile composition, which is stable.

The test that caught them asserts purity rises **in play**, not that the cost function returns the
right number.

### Result: the first replicated gain of the session

vs the champion, four independent fresh seed blocks, 90 games each:

| seeds | winrate |
|---|---|
| 7000-7044 | 75.6% [65.8, 83.3] |
| 8000-8044 | 62.2% [51.9, 71.5] |
| 9000-9044 | 66.7% [56.4, 75.5] |
| 11000-11044 | 77.8% [68.2, 85.1] |

~70% over 360 games, every interval clear of 50%. Contrast E37, E40 and E42, which all looked this
good at 24-48 games and vanished at 80+.

### The cliff -- do not tune this blind

| penalty | 1.5 | 2.0 | **2.5** | 3.0 | 4.0 |
|---|---|---|---|---|---|
| winrate | 71.1% | 62.2% | **4.4%** | 4.4% | 0.0% |

A wall between 2.0 and 2.5. Below it the penalty is a *preference* and cross-role work still
happens when it is clearly better; above it the penalty exceeds any plausible distance saving, so a
unit whose role has no work refuses to help and falls through to `_fallback` -- which hauls. The
constraint stops being a tie-breaker and becomes a prohibition.

**Anything that searches this knob must know the safe range is ~1.0-2.0**, and a search sampling it
uniformly to 10 will mostly sample the catastrophic side.

### What it does not do

Against `boatlee` it changes nothing: **$30,977 against the champion's $31,026, both 0%**. Under
D21 that is the only matchup that counts, so this is a real gain in the wrong place -- it moves us
against ourselves.


---

## E47 — `finish_tile`: the mechanism worked, the side effect was bigger

`agent/engine.py` cites E47 in two places (`:473`, `:533`) and no such experiment existed. Traces
were recorded (`traces/finish_tile/`) and never written up. This is that write-up.

`Params.finish_tile` makes a unit standing on eligible work do it, ahead of sticky assignments and
other units' claims — the two pre-emptions E46 identified as the reason units walk away from work
underfoot 829 times a season.

### It did exactly what it was built to do

3 seeds, our side, ~13k unit-turns each:

| | champion | `finish_tile` |
|---|---:|---:|
| actions chained on one visit | 28.4% | **33.9%** |
| PASS | 21.8% | **17.6%** (-828 ops) |
| productive | 25.5% | **27.1%** |
| **crops lost to weeds** | 9 / 154 (6%) | **0 / 150 (0%)** |

Zero crop deaths. This is not a null result — the mechanism fired.

### And it lost on every seed

| seed | champion | `finish_tile` |
|---|---:|---:|
| 5 | $15,124 | $13,639 |
| 6 | $50,739 | $50,366 |
| 7 | $32,753 | **$25,222** |
| mean | $32,872 | $29,742 (-9.5%) |

boatlee also earned *more* against it ($99,624 vs $96,197) — the E26 shared-market coupling again.

### Because it converted idle units into haulers

| | champion | `finish_tile` |
|---|---:|---:|
| movement | 52.7% | **55.4%** |
| steps per productive action | 2.06 | 2.05 |
| hauling share | 7.5% | **9.3%** |
| `PICKUP WHEAT` ops | 636 | **981 (+54%)** |
| `PICKUP WHEAT` units | 2,603 | **3,903 (+50%)** |
| `BUY_PRODUCT WHEAT` units | 3,748 | **4,608** |
| **FEED ops** | **810** | **809** |

It collected 1,300 more wheat and ordered 860 more, to service **one fewer** FEED. The 828 PASSes it
removed became 388 PICKUPs and ~790 net moves. Steps-per-action did not move (2.06 -> 2.05), so the
walking was not made more efficient — there was simply more of it.

### The leak is `_fallback`, again

The `HAUL_OPS` exclusion at `:544` — added precisely to stop shed-camping — is not what leaks.
`PICKUP WHEAT` broken down by its quantity argument:

| n | champion | `finish_tile` |
|---:|---:|---:|
| 1–3 | 48 | 87 |
| **4** | **437** | **774 (+337)** |
| 5 | 151 | 120 |

**+337 of the +345 increase sits entirely at `n=4`** — the hardcoded constant in `_fallback`
(`:700`, `return ["PICKUP", wanted, 4]`). Task-generated pickups size themselves from `per_trip`,
which varies; the fallback is the only source of a rigid 4.

The chain: `finish_tile` claims local tasks *before* the assignment solve, so units the solver would
have sent to those tasks arrive to find them taken, fall through to `_fallback`, and are each sent
for their own 4 wheat with no in-flight accounting.

**E46 reached the same object from the other side** — above `role_penalty = 2.5` a unit whose role
has no work "refuses to help and falls through to `_fallback` — which hauls", and the winrate falls
off a cliff from 62.2% to 4.4%. Two independent routes to one unfixed defect.

Attribution caveat: `n=4` is also consistent with a task pickup when `short == 4`. One line settles
it — change the constant to 7 and re-run; if the spike moves to `n=7` it is confirmed.

### Status

**Not promoted.** `finish_tile` stays `False`. The op-count findings hold at n=3 because they are
structural (`PICKUP` +54% against `FEED` -1 would look the same at 300 seeds); the -$3,130 does not,
since 3 games cannot resolve 9.5% (E42). That cuts one way only — it is not evidence *for* the
change either, and the mechanism independently explains a loss.

**Do not re-run this at 80 games to confirm the number. Fix `_fallback` first, then re-measure** —
`finish_tile`'s own effect (zero crop deaths, +5.5pp chaining) is real and may flip positive once
the fetch is bounded by actual shortfall with in-flight subtracted.

**Note on the traces:** the champion in `traces/finish_tile/` is *not* the champion in `traces/`.
Same seeds, different results ($15,124 vs $9,893 on seed 5). The baseline moved between the two
recordings, so the two sets cannot be pooled.


---

## E48 — Reading boatlee's play, and a refuted hypothesis

Prompted by a request to review `reference/kaggriculture/1/submission.py` against the recorded
traces. Three findings, one of which refutes the reviewer's own hypothesis.

### 1. It is open-loop, measured rather than assumed

Its emitted actions were diffed step-by-step against its own decompressed table, 6 seeds vs the
champion:

| seed | unit actions identical to script | market turns identical |
|---|---|---|
| 5 | 100.0% | 98.5% |
| 6 | 100.0% | 98.1% |
| 7 | 99.9% | 98.1% |
| 8 | 100.0% | 98.5% |
| 9 | 99.8% | 98.6% |
| 10 | 100.0% | 98.7% |

The 69 differing market turns are **pure permutations — 0 orders added, removed or resized.** So
`_rank_sell_slots` is the only layer that fired. The wool controller, the RC2 fertilizer relay and
the COW→SHEEP swap **never fired at all** (played step 192 is byte-identical to the script). Weed
repair accounted for 0–5 unit-actions per season.

**Against our champion, boatlee is a pure replay, and wins 6/6 without reacting once.** This is the
substrate `PLAN3.md` is built on: a deterministic base makes clean A/B isolation possible.

### 2. Our champion deadlocks its own shed

| seed | final $ | days at shed cap | peak WOOL held | WOOL sold |
|---|---:|---:|---:|---:|
| 6 | $52,869 | **1** | 22 | 152 |
| 8 | $38,879 | 5 | 23 | 130 |
| 7 | $22,380 | 9 | 18 | 85 |
| 5 | $9,893 | 9 | **55** | 88 |
| 10 | $4,178 | 9 | **52** | 68 |
| 9 | $3,492 | 9 | **52** | 68 |

Mechanism: `reserve_frac["WOOL"] = 0.5` holds wool for a price that never arrives — wool has no shop
buyer in 36% of games (E33) — the shed fills to its 100-item cap, and then end-of-day overflow is
**discarded**, `BUY_PRODUCT` silently fails (`kaggriculture.py:653`), and unsold stock scores zero.

That reserve comes from **E19/D18, measured under 1.32.4 when wool had a guaranteed buyer**. H4
("re-baseline under 1.32.6; every mix conclusion in E1–E32 is void") is still open and would have
caught it.

Corroboration that makes this worth taking seriously at n=6: **boatlee ships a hand-written valve for
exactly this.** `_V16_WOOL_PRESSURE = 78` force-dumps wool down to 66 whenever total shed contents
reach 78/100, overriding its own price gates. Its author hit the same wall.

**Status: hypothesis, n=6.** The structural fact (shed pegged at cap for 9 of 29 days) is not
marginal; the money link is exactly the size of sample E42 warns about. Test at ≥80 games.

### 3. Melon is NOT boatlee's mistake — hypothesis refuted

The reviewer's sharpest prediction: melon has **zero shop demand in 100% of games** (E33) and the
town centre absorbs ~30 units a season, yet boatlee sells ~114. That looked like obvious waste.

Tested by rewriting `MELON -> WHEAT` in its own action table (`BUY_SEED` and `PLANT` both), 20 seeds
x 2 seats:

| boatlee variant | their $ | our $ | their wins |
|---|---:|---:|---:|
| stock (melon) | **$126,599** | $49,914 | 40/40 |
| melon -> wheat | $107,717 | **$58,496** | 40/40 |

**Melon is worth $18,882 to them.** The error was conflating shop demand with market absorption:
114 units moves inventory to only 0.38xT, and with `above_func = sq` the price falls 250 -> ~$120 —
still the most valuable unit on the board. Melon collapses only at *scale* (E41's 60 tiles ->
360 units), which is what E43 concluded and this reviewer had half-forgotten.

Note column two: when they stop growing melon **our money rises 17%** with no change on our side —
the E26 shared-market coupling, and a reminder that our revenue is substantially set by what they
plant.

**Consequence for the plan:** trace re-optimisation (PLAN2 P4.0, now `PLAN3` R4) is deprioritised.
The most obvious available edit made the plan $19k worse; the slack is not where it looks.

Caveat: 40 games, below the ~80 bar. Safe as a *refutation* — the effect is large and opposite to
the prediction — but the $18,882 should not be quoted precisely without a re-run.


---

## E49 — R0: the relay substrate, and the counter that caught itself

`PLAN3` R0. Build a relay agent — fixed 719-step table plus an ordered overlay stack — restructured
so behaviour can be added without touching `reference/kaggriculture/1/submission.py`, which is the
arena's only external opponent (D16).

### Bit-identity holds

`agent/relay.py` re-expresses the reference agent's five layers as overlays
(`weed_repair`, `convert_livestock`, `wool_controller`, `rank_sell_slots`, `market_relay`) applied
in its exact pipeline order. `agent/relay_table.py` carries the 719 turns re-encoded from the
*parsed* structure, so it is a function of the decoded table rather than a transcription.

| check | result |
|---|---:|
| steps compared (20 seeds x 2 seats x 719) | **28,760** |
| differing steps | **0** |
| mean money vs champion, `boatlee` | $121,970 |
| mean money vs champion, `relay-base` | **$121,970** |

R0.2's kill criterion is cleared. `relay-base` is registered (`kind="relay"`), and every later
variant is this entry plus one named overlay, so an A/B isolates the overlay and nothing else.

This also settles a long-running claim in passing: **`relay-base` is bit-identical in seat 1**,
which is impossible unless `obs["step"]` reaches seat 1 correctly on all 719 turns. E21 established
that; `main.py`'s docstring still asserted the opposite and has been corrected.

### The desync counter caught its own defect first

`Ctx.blocked_ops` (R0.5) counts scripted ops that arrive to find the wrong thing on the tile. It is
what makes `PLAN3` §2's safety rule enforceable — an overlay claimed structure-preserving must not
raise it.

Its first run reported **97.3 blocked ops per episode on a farm that was perfectly in sync**, 94 of
them `HARVEST_on_empty`. The counter was wrong, not the agent: **HARVEST is legal on an occupied
coop or pasture** — collecting milk, wool and eggs uses the same op as harvesting a crop — and the
check tested for `kind == "PLANT"` only.

The bundled test did not catch it because it only exercised the empty-tile case. Corrected, and
`test_blocked_ops_allows_harvesting_animals` now pins it.

**This is the E39 lesson recurring inside the instrument built to prevent it** — the measurement
apparatus is as capable of being wrong as the thing measured. Had it shipped, the counter would have
cried wolf on every safe overlay in R1 and R2 and vetoed the phases it exists to protect.

### Corrected baseline, and R3 dies by its own gate

80 episodes, `relay-base` vs the champion, both seats:

| | per season |
|---|---:|
| **mean blocked ops** | **2.35** |
| median | 1.0 |
| max | 28 |
| `COLLECT_FERTILIZER_no_animal` | 1.00 |
| `WATER_on_weed` | 0.80 |
| `HARVEST_on_weed` | 0.35 |
| `FERTILIZE_on_weed` | 0.20 |

**R3.0's kill criterion was <10 blocked ops per season, fixed before the work. Measured 2.35, so
R3 is dropped.** The reference agent's existing weed repair already handles nearly everything; a
generalised version would have been a phase built to fix a problem that is not there.

Worth noting what this cost: R3 was written into the plan on an *estimate* ("~6 weeds a season"),
and the gate that killed it was one number from an instrument built for a different purpose. Setting
the criterion before the work is what turned a plausible phase into a five-minute decision.

### Also done

* `make submission` green — bundle builds with the relay modules, both seats DONE, $103,334, worst
  turn **2.3 ms** of the 1000 ms budget. `main.py` still runs the scripted champion; switching the
  submission is a separate decision and is not R0's to make.
* Full suite: **328 passed, 1 skipped.**


---

## E50 — R1: adaptive livestock, killed on a clean implementation

`PLAN3` R1. Choose COW vs SHEEP from the shops a game actually drew, instead of committing the herd
before any shop unlocks. WOOL has no shop buyer in 36% of games, MILK in 2% (E33), and the reference
agent's own conditional fired in **0 of 6** traced games (E48) -- so the hypothesis looked strong and
the substrate was ideal.

**Verdict: the measurement stands, the interpretation was wrong.** R1 loses cleanly -- and the
reason is not that the herd choice is worthless. See *What R1 actually hit* below; it is reopened as
blocked on R2e rather than killed. The kill criterion was fixed before the work -- *beat `relay-base` on the
subset where observed demand disagrees with the scripted herd, or drop it.*

| | disagreement subset |
|---|---:|
| `relay-base` | **$125,354** |
| `relay-herd-down` | $115,760 |
| wins | **0/22** |
| mean delta | **-$9,594** |
| `blocked_ops` | **5.25 vs 5.25 -- identical** |

240 games, plus 0/12 on an independent 80-game block. **0/34 across both.**

### Getting to a clean measurement took three attempts, and that is the result

The first two versions lost too, and neither loss was reportable. `PLAN3` §6 exists for exactly this.

**Attempt 1 -- stranded animals.** Rewriting the purchase to SHEEP while leaving the scripted
`PICKUP COW` alone strands the animal in the shed forever. Caught immediately by the desync counter:

| | money | blocked ops/season |
|---|---:|---:|
| `relay-base` | $128,485 | **3.7** |
| purchase swapped, logistics not | **$689** | **1,666.9** |

A 450x spike. This was run deliberately as the in-play mutation test R0.5 requires, and it is the
sharpest available demonstration of `PLAN3` §2: **one inconsistent substitution unravels the entire
remaining plan.** The fix is not to remember the decision but to resolve pickups and placements
against *the stock actually held*, which cannot disagree with the purchase whatever it was.

**Attempt 2 -- two changes at once, and the wrong one blamed.** The affordability guard and a switch
from raw demand to revenue-capacity weighting went in together. Result: **$29,105 and 1,534 blocked
ops.** The guard was fine; the weighting was fatal.

The town centre buys one of every non-fertilizer product per day, so its term is *equal* for milk
and wool and carries no information -- but it is not zero, so the price multiplier breaks the tie on
it. With no shops unlocked, `1.0 x $200 > 1.0 x $160`, so **every day-0 COW became a $500 SHEEP on a
$3,000 bank.**

This is the project's own documented failure mode committed by the person writing it down: E43 and
E44 both say single-knob testing cannot find conjunctions, and the inverse -- changing two things
and attributing the outcome to one -- is the same error running backwards. Fixed by requiring
**genuine shop evidence** (`_shop_demand`, excluding the flat centre term) before the herd may move.

**Attempt 3 -- the clean test.** Even with both fixes the full version still leaked ~1.1 stranded
pastures per game, because the affordability guard reads `farm["money"]` *before* the same turn's
sells settle. Rather than keep patching, the swap was restricted to the direction that cannot fail
for cash -- SHEEP -> COW, $500 -> $400, which always settles if the scripted purchase would have.
That produced `blocked_ops` **identical to `relay-base` and zero empty structures**, and it is the
direction E33 actually motivates: the value was supposed to be in *not* breeding sheep into a game
with no yarn store.

It still loses 0/34. **The hypothesis is refuted, not the implementation.**

### Head-to-head confirms the loss, and supplies a perfect control

The figures above are *indirect* -- each variant played the champion and their money was compared.
Better: play them against each other. 240 games each, both seats.

| | `relay-base` wins | challenger wins | **exact ties** |
|---|---:|---:|---:|
| vs `relay-herd-down` | 64 | 36 | **140** |
| vs `relay-herd` | 101 | 43 | **96** |

On the games where the herd actually differed:

| | `relay-base` | challenger | base wins |
|---|---:|---:|---:|
| vs `relay-herd-down` | $90,840 | $85,488 | **37/46 (80%)** |
| vs `relay-herd` | $89,480 | $84,366 | **81/104 (78%)** |

**The exact ties are the control.** In 140 of 240 games the shop draw never triggered a swap, and the
two agents played byte-identical games finishing with byte-identical money. The overlay therefore
changes nothing except the herd, and the losses on the other 46 are attributable to the herd alone.
No common-opponent comparison can establish that.

Note the scale: `relay-base` earns ~$126k against our champion and ~$89k against a copy of itself.
Two agents of that strength flooding one market cost each other ~$37k -- the coupling E26 measured.
Money is comparable only within a fixed matchup; winrate is the ranking (E5).

### ~~What R1 actually hit: production and sales are co-adapted~~ [CORRECTED — see below]

The original claim is kept visible because it was written into `PLAN3` §2 and acted on:

> Two extra cows produced exactly zero extra milk sales -- 241 in both arms. The table's sell orders
> are a fixed list, so milk revenue is capped by the script regardless of how many cows exist.

**That was wrong, and wrong in a specific repeatable way: 241 is the quantity the script *orders*,
not the quantity that *settles*.** The environment sells `min(requested, shed)`. Counting order
sizes as sales is the same mistake that produced "our champion buys 10,788 wheat" (E48) -- there,
`BUY_PRODUCT` silently fails against a full shed (`kaggriculture.py:653`).

Caught by the question *"but do we sell milk in relay-herd-down?"*. Re-measured by tracking units as
they actually appear in shed + unit inventories, on the 16 games where the herd differed:

| | herd | MILK **made** | MILK *ordered* | WOOL made | WOOL sold | money |
|---|---|---:|---:|---:|---:|---:|
| `relay-base` | 7.0c / 6.0s | **174** | 241 | 174 | 174 | $102,242 |
| `relay-paired` | 9.0c / 4.0s | **220** | 241 | 132 | 132 | $99,023 |

The script **over-orders** milk — 241 against 174 produced — leaving ~67 units of unused sell
capacity. Adding cows raised production to 220 and **every unit sold**. There was never a milk
bottleneck, and the shed never overflowed (0.0 days where the end-of-day drop would exceed the cap).

### The real reason R1 loses: it is simply a bad trade

```
+46 milk x $160 = +$7,360
-42 wool x $200 = -$8,400
                  --------
                   -$1,040 at base prices   (-$3,219 realised)
```

A cow yields every 2 days against a sheep's 3, but that 50% rate advantage does not cover wool's 25%
price premium at these volumes. **No hidden mechanism and no co-adaptation — wool is worth more per
unit, and R1 trades it away.** This is the same shape as E48's melon result: at these volumes the
price curve decides, not shop demand.

Confirming from the other side, `relay-base` over 60 episodes:

| product | sold/season | stranded at season end |
|---|---:|---:|
| MILK | 241 | **0.0** |
| WOOL | 139 | 0.0 |
| WHEAT | 455 | 0.0 |
| STRAWBERRY | 286 | 0.0 |
| FERTILIZER | 235 | 0.0 |
| MELON | 114 | 0.0 |

**0.1 items stranded per season; shed at cap 0.9 days of 30.** It sells everything it makes. The plan
is a closed system tuned as a whole, which is exactly why single-sided edits keep losing -- melon
(-$18,882, E48), and now the herd.

The co-adaptation rule this originally produced was written into `PLAN3.md` §2 and has since been
**weakened to match the corrected measurement**: the sell list is fixed, but it carries slack — here
enough to absorb a 26% production increase without any change at all.

### Three phases re-scoped by this

* **R1** -- was reopened on the wrong diagnosis, then closed for good by E51 (pairing tested) and
  by the corrected arithmetic above (it is a bad trade, not a plumbing problem).
* **R2a / R2d** -- **premises gone.** Both were justified with "ours deadlocks at 100 for 9 of 29
  days", where *ours* is the champion engine, not the relay. On this line we are boatlee, which hits
  the cap 0.9 days. Reasoning from a number measured on a different system -- the same error that
  lost R1.
* **"Zero farm risk" for market overlays** -- weakened. Changing cash changes which purchases settle,
  which changes the farm.

### A structural ceiling found on the way

The trace buys animals on days 0, 5, 6, 7, 8 and 15; shops unlock every 3 days. So most of the herd
is committed with **0-2 shop instances known out of ~8**. The purchase cannot be deferred either --
the scripted `PICKUP` lands one step later and would find an empty shed. **The decision point is
early and immovable**, which bounds what any livestock adaptation can be worth here, paired or not.

### A secondary reading, still true: E48's lesson



Ranking a product by shop demand is a claim about *sustained* capacity. At the volumes involved it
is the wrong instrument, and this is now the second time it has misled in three experiments.

The reference agent runs **4 sheep**, selling ~132 wool a season. With no yarn store the town centre
still absorbs ~30, and WOOL's curve (`base 200`, `above: sq, 3.2`, `T = 105`) leaves the rest
selling well above the floor -- because 132 units is only ~1.3xT and the square term has barely
engaged. Wool without a shop behaves exactly like melon without a shop (E48): **no shop demand, and
still the most valuable thing that tile can produce at that volume.**

`CLAUDE.md` was amended after E48 to say check volume against `T` before calling a product a
mistake. R1 was specified before that amendment and did not apply it. It should have been caught on
paper.

### Status

`relay-herd` and `relay-herd-down` stay registered -- as the **production half of R2e**, not as
diagnostics. `relay-base` is unchanged. **R2e (paired herd + sell schedule) becomes the next phase**,
with R2b and R2c behind it.

The `_shop_demand` helper, the desync counter's logistics coverage, and the stock-following
retarget logic are kept -- R2 needs none of them, but any future herd or crop work does, and the
1,667-blocked-op mutation is now the standing demonstration that the counter works in play.


---

## E51 — R2e: the first thing on the relay line that wins, and what it actually is

`PLAN3` R2e. §2's second constraint says production and sales are co-adapted, so the sales half was
built: `surplus_release` sells stock the remaining script will never move.

### The first version never fired, and that was the result

Trigger v1 was `held > remaining scheduled sales`. It fired **zero times in 160 games**. `held` is
one shed's worth (tens); `planned` is the whole remaining season's orders (hundreds), so the
difference is negative essentially always:

| day | item | held | script still plans to sell | "surplus" |
|---:|---|---:|---:|---:|
| 6 | MILK | 0 | 241 | -241 |
| 18 | MILK | 3 | 157 | -154 |
| 24 | WOOL | 16 | 32 | -16 |

Both money numbers from that run were discarded unread, per `PLAN3` §6. **An overlay whose effect
counter is zero is an unfinished implementation, not a negative result** — and this is the second
time the counter has caught a defect before it could be reported as a finding (E49 was the first).

Trigger v2 uses **shed pressure** plus a late-season sweep of stock past the script's last scheduled
sale — the mechanism the reference agent already uses for wool (`WOOL_PRESSURE = 78`), reused rather
than invented because it was tuned against this table's production.

### It replicates

`relay-sell` vs `relay-base`, head-to-head, both seats:

| block | winrate | delta |
|---|---|---:|
| 9000–9079 | 87.5% | +$174 |
| **21000–21079 (fresh)** | **85.6% [79.4, 90.2]** | +$174 |
| **31000–31079 (fresh)** | **86.9% [80.8, 91.3]** | +$178 |

480 games, three independent blocks, every interval clear of 50%. Contrast E37/E40/E42, which all
looked good at 16–48 games and vanished at 80+.

### And it beats the real external agent — the D21 objective

| | money | winrate | w / l / t |
|---|---:|---|---|
| `relay-base` vs `boatlee` | $85,645 vs $85,645 | 11.7% | 14 / 14 / **92 ties** |
| `relay-sell` vs `boatlee` | $85,733 vs $85,557 | **90.8% [84.3, 94.8]** | 109 / 11 / 0 |

The 92 exact ties are the control: a bit-identical agent ties its source. `relay-sell` wins 109–11.

**This is the first agent in the project to beat `boatlee` head-to-head.**

### What it actually is, stated plainly

The mechanism is small: **~8 fertilizer units released once a season**, around step 701, instead of
letting the script sell them at 713+. Both agents dump at the end; whoever sells first gets the
better price. Going twelve steps earlier wins that race.

So the 90.8% is a **mirror-margin effect**. We beat `boatlee` because we *are* `boatlee` plus a
sliver, and in a near-tie a sliver wins consistently. Against other opponents the effect is real but
tiny, and the winrate cannot move because it is already saturated:

| opponent | `relay-base` | `relay-sell` | delta | winrate |
|---|---:|---:|---:|---|
| `champion` | $128,020 | $128,111 | +$91 | 100% both |
| `x-dumper` | $125,076 | $125,205 | +$129 | 100% both |
| `starter` | $154,386 | $154,386 | **$0** | 100% both |

**Never worse anywhere, decisively better only in the mirror.** That is a genuine improvement and a
narrow one; it should not be read as 90% better than `boatlee` in any general sense.

It is also, in part, the edge identified when the reference agent was first reviewed: its sell
schedule sits at fixed step indices and is readable straight out of the file, so **every dump it
makes is front-runnable**. This exploits that. Against an opponent with a different schedule the
mechanism degrades to "sell slightly earlier", which is worth $91–129 rather than a win.

### R1 is not rescued by pairing -- and the reason is simpler than E50 thought

`relay-paired` (adaptive herd + `surplus_release`) scores **$88,108 against `relay-base`'s $88,846**
— worse than the sell overlay alone, which wins at $89,019.

But note the effect counter on that run: `{'FERTILIZER': 8.0, 'MILK': 0.0}`. **`surplus_release` sold
no milk**, so this was never a valid test of "pair the herd change with a sell schedule". Querying
that directly is what collapsed the underlying claim: milk was never capped. E50's correction has
the numbers -- the script *over-orders* milk 241 against 174 produced, so the extra output from more
cows sold in full with no new overlay at all.

**R1 loses because it is a bad trade** (-42 wool at $200 for +46 milk at $160), not because sales
could not follow production. Closed on the corrected arithmetic.

### Status

`relay-sell` is registered and is the strongest agent in the repo. **Not promoted**: `champion.json`
governs the engine line, and D19's gate (500 games + full gauntlet + neighbourhood sweep) has not
been run. Shipping it is additionally gated on the provenance question in `PLAN3.md`'s preamble,
which is the user's to settle.


---

## E52 — R2e's thresholds swept: they were borrowed, and one of them sat near a cliff

`surplus_release` shipped with `RELEASE_PRESSURE = 78` and `RELEASE_BATCH = 8`, taken from the
reference agent's **wool** controller (`WOOL_PRESSURE = 78`). Borrowing numbers tuned for a different
product and a different purpose is a reasonable start and not a measurement, so they were swept.
`D19` stage 3 exists for exactly this: a chosen point need not be a *local* optimum.

### Batch size is the dangerous knob

Screen at 60 games:

| pressure | batch | winrate | delta $ |
|---:|---:|---:|---:|
| 78 | 8 | 85.0% | +$177 |
| 78 | **30** | **40.0%** | **-$255** |
| 65 | 8 | 86.7% | +$358 |
| 65 | **30** | **41.7%** | **-$306** |

Selling in large blocks loses outright — the release exists to free shed slots ahead of an overflow,
not to liquidate, and a 30-unit block moves the price against itself. Confirmed on fresh seeds:
batch 12 already scores 51.2%, batch 4 scores 82.5%, batch 8 is the peak.

### Pressure: flat winrate, rising money, then a cliff

Fresh seeds 21000-21039, 80 games each, vs `relay-base`:

| pressure | winrate | 95% CI | delta $ | fires/game | **blocked_ops** |
|---:|---:|---|---:|---:|---:|
| 78 | 83.8% | [74.2, 90.3] | +$180 | 1.07 | 10.5 = baseline |
| **70** | **83.8%** | [74.2, 90.3] | **+$339** | 2.03 | **10.5 = baseline** |
| 65 | 83.8% | [74.2, 90.3] | +$365 | 2.25 | 10.5 = baseline |
| 60 | **12.5%** | [6.9, 21.5] | **-$876** | 3.26 | **11.5** |
| 55 | — | — | -$1,400 | 4.00 | **12.2** |

**Winrate is flat at 83.8% from 65 to 78 while money nearly doubles**, and then falls off a cliff
below 65 — 83.8% -> 12.5% in a single step of 5.

**The desync counter locates the cliff exactly.** `blocked_ops` sits on the `relay-base` baseline of
10.5 for every safe setting and rises the moment the cliff is crossed. Selling that aggressively
shifts cash, which changes which purchases settle, which changes the farm — the mechanism `PLAN3` §2
was weakened to admit (a market overlay is low-risk, not zero-risk). The counter was built to police
farm-affecting overlays and it turns out to police market ones too.

### Chosen: pressure 70, batch 8

Not 65, which scores marginally better. 70 takes **93% of the available gain with twice the margin
to the cliff**, and E46's `role_penalty` wall (71.1% at 2.0, 4.4% at 2.5) is the precedent for not
sitting one step from an edge. **Safe range is ~65-78; a search sampling this knob uniformly will
mostly sample the catastrophic side**, which is now written beside the constant.

### Re-confirmed against the real opponent

`relay-sell` at pressure 70 vs `boatlee`, two fresh blocks:

| block | money | winrate | w / l / t |
|---|---:|---|---|
| 51000-51059 | $82,471 vs $82,150 | **90.8% [84.3, 94.8]** | 109 / 11 / 0 |
| 61000-61059 | $90,182 vs $89,874 | **88.3% [81.4, 92.9]** | 106 / 12 / 2 |

Holds. Five independent blocks now, ~840 games, every interval clear of 50%.

### What changed and what did not

The tuning roughly **doubled the money margin** (+$180 -> +$339 against `relay-base`) and left the
winrate untouched, which is the expected shape in a near-mirror: the margin was already enough to
decide almost every game, so more margin buys money rather than wins.

Still true, and still the caveat that matters: this is a **mirror-margin effect**. Against
`champion` and `x-dumper` the overlay is worth $91-129 and the winrate is already saturated at 100%.
It should not be read as being 90% better than `boatlee` in general.


---

## E53 — `relay-sell` through the promotion gate: passes, with one stage that proved nothing

`make promote`'s three stages (D19), run on fresh seeds no search has touched.

```
STAGE 1 — relay-sell vs incumbent champion
  100.0% [99.2, 100.0] n=500   $123,522 vs $39,375   -> win

STAGE 2 — gauntlet vs 94 registered agents (screen 24 games, escalate close calls to 500)
  screen: 93 clear wins, 1 needing confirmation
    relay-paired              58.2% [53.8, 62.4] n=500  $88,035 vs $86,698

STAGE 3 — neighbourhood sweep
  no discrete knobs to probe

ALL STAGES PASSED.
```

### What each stage is actually worth here

**Stage 1 is close to a formality.** `relay-sell` is boatlee-class and the incumbent is our engine.
100% over 500 games says the two are different animals, not that the candidate is well-tuned.

**Stage 2's only informative row is `relay-paired`** — the one agent close enough to need
escalation, and `relay-sell` beats it 58.2% [53.8, 62.4] over 500 games. That independently confirms
E51: the adaptive herd is a **drag**, not a neutral addition. The other 93 wins are against a field
of engine variants this agent outclasses by construction.

**Stage 3 passed vacuously and is not evidence.** `NEIGHBOURHOOD` sweeps `cow_target`,
`sheep_target`, `goose_target`, `hire_max`; a relay agent has none of them, so nothing was probed.
The knobs that matter here are `RELEASE_PRESSURE` and `RELEASE_BATCH`, which the gate cannot see.

**The substance of stage 3 was done by hand in E52** — and it mattered: the sweep found a cliff
5 units below the setting that screened best, and moved the default from 78 to 70 for double the
safety margin. Had the gate been trusted to cover it, none of that would have surfaced. A gate that
reports PASS on a check it did not perform is worse than one that reports "not applicable", so this
is recorded rather than counted.

### Not applied

`--apply` was **not** run, and `tools/promote.py` now refuses it for non-engine candidates.
`search/champion.json` is read back as `Params(**params)` (`arena/registry.py` `_load`), so writing
a relay agent's `{"overlays": [...]}` there would raise `TypeError` on the next import and break the
entire registry. A non-engine champion needs its own pointer file before anything can be installed.

Two further blocks on actually shipping this, both outside the gate's scope:

* **Provenance.** `relay-sell` is a competitor's action table plus one overlay. Whether that is a
  permissible submission is a competition-rules question for the user (`PLAN3.md` preamble).
* **What "champion" should mean.** `champion.json` governs the engine line (`PLAN3.md` §9). Promoting
  across lines is a plan decision, not a measurement.

## E54 — kaggle-environments 1.32.7: the market grew a scarcity spike, and nobody is selling into it

The V6 hash guard (`tests/test_env_version.py`) was moved 1.32.6 -> 1.32.7. This is the record of
what changed, what it cost to re-verify, and the two things measurement turned up that the diff
alone would not have.

### The change is one function, and it is large

`diff` of the two `kaggriculture.py` files touches nothing outside pricing:

* `_shape` gained a `hinge` arm — `u + 8·max(0, u−1)²` where `u = x/T`, scaled so `f(T) = 1`.
* CARROT, TOMATO and EGG moved to it below I0. Carrot's `below_target` went 0.20 -> 1.00 with it.

Below the knee that is a mild linear repricing; past it the price runs away. Reference values:

| product | base | −T | −1.5T | −2T | −2.5T |
|---|---:|---:|---:|---:|---:|
| TOMATO | 60 | 84 | 144 | **300** | **552** |
| CARROT | 35 | 70 | 158 | **385** | 752 |
| EGG | 50 | 70 | 120 | 250 | 460 |

Every other simulation rule is byte-identical, so the port was `Shape::Hinge` in `kagsim/src/market.rs`
plus the same arm in `agent/relay.py`'s vendored table. `make verify` after: **700/755 reference
statements differentially covered, 0 unexplained, 0 divergences**, suite green.

### Nobody is selling into it

Boatlee over three full seasons (seeds 3/5/9, $163,950 / $139,795 / $186,832):

```
plants : MELON 57, WHEAT 429, STRAWBERRY 111
sells  : WHEAT 1365, STRAWBERRY 858, MILK 723, FERTILIZER 705, WOOL 396, MELON 342
hinge  : CARROT 0, TOMATO 0, EGG 0
```

Zero exposure to all three hinge products — it neither plants nor sells them, so its money is
unchanged by a market change worth up to 5x on those goods. The strongest opponent in the field is
structurally blind to the largest new lever, which is what `PLAN_v4` §2.7 is built on. (Three
seasons is enough for *this* claim: it is a statement about a deterministic 719-step table's
composition, not about a win rate.)

### Two things the diff would not have told us

**1. The coverage gate was reporting on a file nothing imported.** `tools/coverage_audit.py`
resolved the reference through `os.path.dirname(os.__file__) + "/site-packages"`. A venv does not
copy the stdlib, so under `.venv` that expression still pointed at miniconda's copy: coverage
attached to a file the interpreter never loaded, every line read as unexecuted, and the gate
printed **0.0% covered while the test suite passed**. Now resolved via `sysconfig.get_paths()`.
The failure mode is worth naming — this is the second time a gate in this repo has reported
confidently on a check it did not perform (E53's stage 3 was the first).

**2. `role_penalty` (E46) no longer does anything.** Purity with the knob off vs at 3.0, same
`purity()` used by `tests/test_roles.py`, six seeds:

| | 0 | 3 | 5 | 7 | 11 | 17 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.32.6 | +0.012 | +0.060 | +0.103 | +0.104 | +0.054 | +0.033 | **+0.061** |
| 1.32.7 | +0.009 | +0.013 | +0.011 | +0.072 | +0.027 | −0.011 | **+0.020** |

Not seed noise and not a threshold to relax: repricing carrot/tomato/egg changed which tiles the
engine values, so the assignment cost the penalty perturbs is a different cost now. The knob was
tuned under the old prices and has to be re-tuned in the F-track re-measurement. `test_roles.py`
carries a `strict=True` xfail so the fix cannot land unnoticed.

### Smoke numbers (not evidence)

4 seeds x 2 seats, far below the 80-game bar: starter $3,488 -> $3,507 (traced to carrot's
below-knee repricing on a 192-unit drain — the linear region, not the spike), champion $63,147 ->
$60,306, boatlee unchanged to the dollar. Only the boatlee row means anything, and only because
the section above establishes *why* it cannot move.

### I0 landed with it

`tests/test_env_facts.py` — 29 assertions driven through the reference env in lockstep with kagsim,
covering the arithmetic PLAN_v4 reasons from. Three corrections it forced on my own scripts, each a
constraint the compiler has to encode:

* **Inventories do not survive dusk.** Fertilizer must be picked up the morning it is applied; a
  PICKUP the day before is silently returned to the shed. Same for a keeper's feed wheat.
* **`(5,4)`, `(4,5)` and `(5,5)` are shed-access but LOCKED**, so they take PICKUP/DROP from step 0
  and refuse BUILD_PASTURE. The pasture cluster has to sit on NW ground.
* **The unfed grace is exactly one dusk.** A newly placed animal survives the dusk of its placement
  day and escapes at the next one.

Confirmed as stated in the plan: fertilize at ages 9 and 13 pays 8 strawberry units against 4
unfertilized; a fed-and-cared cow pays 3 milk per cycle and 1 if unfed on the production day; melon's
`max_yield_day` is 12 in code against the README's 10; ongoing yield caps at 4 on the tile.

## E55 — I2: the harness, and what it says before a single line of the compiler exists

`harness/{run,registry,counters}.py` + `tests/test_harness.py` (12 tests). One command produces a
table with Wilson intervals and eleven counters for any pair in the pool, 480 games in 25 s.

### It reproduces games it did not measure

Both I2 verifications, defined before this harness existed, reproduce exactly:

* **Boatlee's mirror.** Seeds 2–6 tie to the dollar; 0/1/7 give **−96 / +771 / +843**, the exact
  seat wobble `TASKS_v4` records. A deterministic script against itself is the sharpest health
  check available — if the runner got turn count, seating or agent construction wrong, these move.
* **`executor_v7` vs `starter`: $72,979**, against the documented ~74k.

### The baseline (seeds 21000:21040, 80 games/pairing, both seats)

```
agent              winrate (95% Wilson)     mean $     margin
boatlee          100.0% [98.4, 100.0]      142,555   +116,643
kagsim_champion   60.8% [54.5,  66.8]       66,199     +3,260
executor_v7       39.2% [33.2,  45.5]       53,419    -21,824
starter            0.0% [ 0.0,   1.6]        3,521    -98,079

                steps/useful  idle  blocked  thirst   age  straw/plant  milk/cow-d  ripe@end  overflow  fert hits
boatlee                 1.02   15%       11     0.0  10.0          7.9        1.11         2         0        68
kagsim_champion         2.24   22%      108     3.8   4.5          2.8        1.04        17        60         0
executor_v7             1.70   22%      104     0.5  36.9          2.4        0.99        15        12         0
```

**`fertilize_hits` is 0 for both of our agents against Boatlee's 68.** F2 is not a tuning target,
it is unimplemented — the single largest legible gap in the table, and it explains straw/plant 2.4–2.8
against 7.9 on its own.

Two more that were not on anyone's list: `kagsim_champion` **discards 60 units a season** to shed
overflow (Boatlee: 0), and both our agents leave 15–17 ripe units unharvested at the end against
Boatlee's 2. `steps_per_useful` 2.24 for the champion is worse than the 1.3–1.84 band `PLAN_v4` §1
attributes to us.

### Two counter defects found by disagreeing with an independent measurement

Both were caught the same way — by measuring the same quantity a second way and refusing to accept
the first number.

**1. Day-boundary sampling missed most plant deaths.** The first implementation compared the board
once per day. A plant is created with `consecutive_unwatered = 1`, so one not watered on its
planting day is a weed by that evening — born and lost between two snapshots, counted as neither.
Worse, Boatlee digs and replants within the day, so its losses were erased before the next dawn: the
counter read **0.1/season against 10 measured** by a per-step scan. Per-step detection costs **7%**
of game time, which is not a trade worth making for a wrong number.

**2. "Plants lost" was conflating two opposite events.** A PLANT becomes a WEED two ways: thirst at
the daily refresh, and decay once `step >= max_lifespan_step` (`_decay_plants`). The counter now
splits them, classifying from the environment's own `max_lifespan_step` field rather than
re-deriving the rule (E39). The split is what makes the table above readable:

| | thirst | old age |
|---|---:|---:|
| boatlee | 0.0 | 10.0 |
| kagsim_champion | 3.8 | 4.5 |
| executor_v7 | 0.5 | **36.9** |

`PLAN_v4` §1's "plants die of old age not thirst" is confirmed exactly for Boatlee — and its
"6–7/season" is the one published number that does not reproduce (measured 10.0). The executor's
36.9 old-age losses are not a routing failure at all; they are a *harvesting* failure, which points
Track F at F4 rather than at the thirst story the plan leads with. `TASKS_v4`'s C5 gate has been
restated in thirst terms: "≤ 10 plants lost" would have passed an agent that kills ten plants
through bad routing, since Boatlee's own 10 are all old age.

### One thing that is not a finding

A mirror run reported Boatlee losing 6.3 plants/season, which looked like a real
"contested markets degrade its farm" effect. It was the pre-fix counter. Direct measurement of the
same 20 seeds gives 0.1, and the corrected counter agrees to two decimals. Recorded because the
wrong number was interesting enough to have been written into the plan.

## E56 — C0: the plan genome, and five repairs the board demanded

`agent/plan.py`: a season as **49 bounded numbers**, with `decode` materialising tiles through a
deterministic layout and `encode` refusing anything the genome cannot express. 24 tests; 5,000
random genomes decode valid and round-trip exactly.

### Boatlee's shape, measured rather than quoted

`Plan.boatlee_like()` is the Phase-1 calibration case, so its numbers were read off a played season
(seed 3) instead of taken from the plan's prose:

```
land       NE day 6, SW day 10, SE never
pastures   13, clustered on the shed, growing outward as land opens
herd       4 SHEEP from day 0, 9 COW from day 5, ~3/day
crops      141 WHEAT plantings, 36 STRAWBERRY, 19 MELON
hands      4 on day 0 rising to 10-14 by day 10
```

The 141 wheat plantings are **~10 tiles on a replant cycle**, which is what `Cohort.replant` exists
for: without it a season of wheat is 141 genes instead of one.

### What the round-trip found

Every one of these was a genome that decoded into a farm the game could not build, and each was
found by generating random plans rather than by inspection:

1. **A descending hand curve came back flat.** `encode` read the cap as `max(hands)`, which for a
   ramp-down (12 → 3) is day 0. Curves are now stored as the three genes they came from rather than
   inferred back out — rounding makes a shallow ramp reach its cap early, so (12, 13, 20) and
   (12, 13, 10) are indistinguishable from the curve alone.
2. **Herd pacing read back doubled**, because it is applied per species and inferred across all of
   them. Stored, for the same reason.
3. **The pasture cluster spilled onto land the plan never buys.** The ring one step from the shed
   touches all four quadrants, so a plan that never buys SE still put a pasture there — animals
   that can never be placed, on a plan that looked valid.
4. **Pastures were ordered by distance, which deferred the herd.** Distance-first put the fourth
   structure on land bought on day 10, silently pushing the fourth animal a third of the season
   late to save one step of walking. Now ordered by unlock day first: cheap land now beats close
   land later.
5. **Cohorts were planted on land bought later, or never.** Repaired rather than rejected — the
   plant day slides to the purchase day, and a cohort on unbought land is dropped — because
   mutation produces these constantly and rejecting them would shrink the space the search
   actually explores.

`decode` also repairs land order (SW cannot precede NE; skipping NE makes SW unreachable, not
cheaper) and trims herds to the structures that exist. **Every repair is recorded in `plan.notes`
and printed.** A search whose candidates are being quietly trimmed is exploring a different space
than it reports, which is the E44 failure with better manners.

### The honest round-trip property

`encode(decode(v)) == snap(v)` is false and should be: decode repairs. What holds — and what the
search needs — is that repair reaches a fixed point in one pass: `decode(encode(decode(v))) ==
decode(v)`. Asserting the stronger, wrong property would have forced the repairs out of decode and
into the search, where every candidate would need fixing separately.

## E57 — C1: the day's work, priced in one currency

`agent/tasks.py` — `daily_tasks(obs, plan, day)` returns every task worth doing today, each in
dollars with the turn by which it must happen. 29 tests; **0.31 ms** worst case over a real season
against a 2 ms budget.

### Why dollars instead of a priority ladder

The engine line ordered work by a hand-written ladder, and E17 measured those priorities as
*harmful*. A ladder cannot express the thing that actually decides a route: a strawberry with three
ticks left is worth watering more than one with none. Measured on a real board, the same survival
water ranges from **$1,496 to a few hundred** as the plant ages, and animal feeds price from half a
unit up to **$1,808** when a full care bank is at risk on a production day. The router (C2) can
trade those against walking distance; a ladder can only order them.

### Deadlines are rules, not intuition

Three that are invisible in the tile state and cost real money:

* **Wheat expires at the dawn of age 5** — `max_lifespan_step = (planted + max_yield_day + 1) * 24`
  — then loses a unit every two steps. The task carries a same-day turn deadline, not "soon".
* **A fertilized ongoing tile facing a tick tonight must be emptied today.** Cap 4, tick +2: leave
  three units on the tile and tonight's production is thrown away. The overflow is added to the
  harvest's value so the router sees what waiting costs.
* **A cow at `max_held` produces into a full bucket**, and the care bank pays out capped by the
  same number — a bank of seven on a cow that holds six is worth six.

### One rule that inverts the obvious

**Harvesting a one-time crop early is destructive, not early income**: `HARVEST` clears the tile
(`kaggriculture.py` HARVEST op), so taking wheat at age 2 throws away the two waterings still to
come. The generator only offers those harvests once the crop has stopped accruing — at its cap or
past its last watering day. Written the natural way first, it emitted a harvest for every ripening
tile every day, which a value-maximising router would have taken.

### What is deferred, and where the seam is

Prices come through a `price_fn` argument, defaulting to the live board quote. That is the wrong
price for a harvest four days out and is exactly the mistake that made melon look worthless
(D17/E48) — but the right fix is C6's projection, so the parameter exists and the default is
honest. Task *values* will improve without this module changing.

`PLANT` tasks are admitted on their own merits; whether the day can sustain them — the water they
will need tomorrow — is C3's call, since it depends on the route.

## E58 — C2: the router, and the ratio the whole plan rests on

`agent/router.py` — `route(tasks, units, turns_left)` compiles C1's task list into a per-turn script
for every unit. 34 tests. Over three full seasons of Boatlee-shaped days:

| seed | steps/useful (mean) | max | hands | survival missed | stops routed |
|---|---:|---:|---:|---:|---:|
| 3 | **0.86** | 1.09 | 7.6 | 1 | 99% |
| 7 | **0.85** | 1.09 | 7.5 | 1 | 98% |
| 11 | **0.84** | 1.09 | 7.4 | 1 | 96% |

Against Boatlee's measured **1.02** and our engines' 1.70–2.24 (E55), on the metric PLAN_v4 §1
blames for the entire gap — and inside C5's ≤ 1.15 gate with room. `route()` costs **0.3 ms**;
`decide_hands`, which routes the day once per staffing level, costs 14–28 ms once a day.

The ratio comes from one idea: **group tasks into stops and do everything on the tile you are
standing on.** A strawberry needing fertilizer, water and harvest is one walk and three turns.

### Hiring is where the ratio is actually decided

Routing the same day at different staffing levels, all value captured at 6 units:

```
units     1     2     4     6     8    10    12    14
st/use 0.33  0.41  0.65  0.76  0.87  1.10  1.25  1.31
value  9.7k 16.0k 27.7k 35.5k 35.5k 35.5k 35.5k 35.5k
```

Past 6 the extra hands add walking and nothing else. The ratio is not a routing property so much as
a *staffing* one, which is why `decide_hands` hires on marginal value against `fib(n)` and why the
engine line's fixed hand cap could never reach this number however well it routed.

### Three bugs, each found by measuring a season rather than reading the code

1. **A deadline vetoed the tile it shared.** `Stop.deadline = min(tasks)`, so a harvest past
   `max_lifespan_step` (deadline turn 0) made the whole stop unreachable — taking the *survival
   water* on the same tile with it. **Eight plants killed per season to save a turn.** Deadlines now
   order the day and never forbid it, which is also what the rules say: a late harvest decays a unit
   per two steps, it does not vanish.
2. **"One keeper per six animals" was arithmetic that did not close.** Six head at three ops each is
   18 turns, and the wheat pickup plus the walk to the cluster does not leave room for the
   eighteenth — so two keepers dropped 2–4 animal ops *every day*, 39 over a season. Keeper count
   now comes from the actual op count against the turn budget.
3. **`decide_hands` judged survival on a staffing level it then discarded.** It broke out of the
   loop using the rejected level's route and returned the previous one, so it could approve a day
   that let plants die while reporting the check as passed. Survival misses fell 8 → 1 per season
   across the three fixes.

The one remaining miss per season is a genuinely over-committed day; that is C3's job — prune the
plan until the day sustains itself — not the router's.

### Ordering is rules, not taste

Within a stop the order is forced by the environment, and each of these is silent when wrong:
FERTILIZE before WATER (a one-time crop reads `fertilized_until_day` *during* the WATER op), WATER
before HARVEST (harvest clears a one-time tile), and PLANT drags a WATER behind it (a new plant
starts at `consecutive_unwatered = 1` and is a weed by dusk otherwise). Route quality is checked
against brute-force optimal tours on 7-stop instances: within 5%.

## E59 — C3: playing the day before committing to it

`agent/verify.py` — `verify_day` runs the compiled day through `agent/forward.py` (parity-tested
against kagsim step by step) and `compile_day` prunes until the day stands up. 15 tests.

**Over 20 seeds x 30 days on the Boatlee-like plan: 0 plants lost to thirst, 0 overcommitted days,
5.2 blocked ops per season** — against Boatlee's own ~10 (E55) — compiling in **15 ms** of a 100 ms
budget.

### Four bugs, and every one of them was the verifier failing the day rather than the day failing

The first run reported **470 plant deaths a season** and flagged every day over-committed. None of
it was true:

1. **It verified a crew that did not exist.** Hands are hired in hour 0's market and stand up at
   hour 1, so an eight-unit route checked against the lone farmer on the dawn board had seven
   scripts executing against nobody. 470 → 133 deaths.
2. **It counted harvests as deaths.** A one-time crop's tile *clears* when harvested — that is the
   point of harvesting it — and the death check was "the plant is no longer there". Every
   successful melon read as a fatality.
3. **It did not model the day's shopping.** C4 buys seed at hour 0; the verifier started from the
   dawn shed, so every PLANT verified as blocked for want of a seed it would actually have. 107 →
   15 blocked ops.
4. **It conflated old age with thirst.** Now split exactly as the harness counter does (E55) — and
   the split is load-bearing, because pruning fires on deaths: before it, the compiler dropped
   healthy plantings to atone for wheat that had simply finished its life. Measured old-age losses
   are **10/season, matching Boatlee's own measured 10 exactly**.

A verifier that agrees with the thing it is verifying is worthless, so each fix was checked by the
number moving in the right direction on a real season rather than by the code reading correctly.

### One real routing bug it then caught

With those fixed, exactly **one** plant still died — on day 16, on all twenty seeds, deterministically.
It was a single strawberry whose survival water was dropped by a busy unit while four units finished
the day idle: assignment happens once, per unit, so a stop dropped for turns is never re-offered.
`route()` now makes a second pass handing unrouted *survival* work to whoever still has turns.
Thirst deaths: 1/season → **0 across 20 seasons**.

That is the whole argument for C3. The router cannot see this failure — from inside one unit's
route the day looks full — and the money would never have shown it either: one strawberry is
~$1,500 in a $140,000 season.

### What pruning does, measured

Sixteen thirsty plants on the board and a cohort wanting five more, same day, same board:

| hands | plantings kept | pruned | deaths |
|---:|---:|---:|---:|
| 0 | 1 | 4 | 0 |
| 1 | 2 | 2 | 0 |
| 3 | 5 | 0 | 0 |

Only plantings are ever dropped, and never survival work: a plant that dies costs its whole
remaining life, a plant not started costs one seed.

## E60 — C4: the compiler, wired to a game — and 1.01 steps per useful action

`agent/main_v4.py`: hour 0 shops, hour 1 compiles, hours 2-23 replay the cache. Registered as
`compiler`. 12 tests.

### The claim PLAN_v4 is built on, measured

40 games vs `starter` on a held-out block, counters from the harness (independent of the compiler's
own bookkeeping):

```
                steps/useful  blocked  fert hits  straw/plant  thirst   age  winrate
compiler                1.01        1         45          3.9     2.2   4.6    100%
boatlee (E55)           1.02       11         68          7.9     0.0  10.0
kagsim_champion         2.24      108          0          2.8     3.8   4.5
executor_v7             1.70      104          0          2.4     0.5  36.9
```

**1.01 steps per useful action — Boatlee's own number**, from a compiler that re-plans every dawn
rather than replaying a table. That is the specific thing PLAN_v4 §1 said reactive engines cannot
reach, and it is reached.

All three C4 gates pass: 0 fallbacks over full seasons in both seats through the *reference* env
(status DONE), **blocked_ops 1-3** against a bar of 12, and **p99 latency 7.6 ms** (max 79 ms) of a
1,000 ms budget. kagsim and the reference env agree to the dollar on the same seeds.

### Four bugs, each a case of planning against something other than the board

The first playable season scored $31k with **538 blocked ops**. Every fix was the same shape — the
compiler believing something the game did not:

1. **Planning against money already spent.** The day was compiled from what the tasks *wanted* to
   buy, not what the wallet could fund, so unfunded plantings became turns spent on bare ground —
   and each dragged its chained WATER onto the empty tile with it. 538 -> 125.
2. **Planning against seeds that never arrived.** At hour 1 the compiler still assumed it could buy
   the shortfall, but the shopping happens at hour 0 and nothing more is bought that day. Compiling
   against the shed instead: 106 -> **2**, and money +33%.
3. **Two hands reaching for the same seed.** `PLANT` validation is atomic per crop per turn: if
   requests exceed seeds held, *every* request for that crop is dropped. The excess is now delayed
   a turn rather than cancelled.
4. **Buying livestock it could not feed.** Wheat is feed, and the sell rule dumped the shed flat;
   the herd escaped and was re-bought three times in one season at ~$450 a head. Sales now hold a
   two-day feed reserve, and animals are bought only up to what the farm's wheat can sustain.

A fifth was found and is worth naming because it was invisible: a **cohort missed on its plant day
was never planted again**. Cash delayed one NE purchase from day 6 to day 12, and the entire 22-tile
strawberry block — the money crop — simply never went in. A cohort's plant day is now the earliest
day, not the only one, closed by the season horizon rather than by a fixed window.

### Money is short of the Phase-1 gate, and the reason is the plan

$52k against C5's $110k. The compiler is not the constraint — the efficiency counters above are at
or better than Boatlee's. The calibration plan is: **13 pastures ordered nearest-first take 13 of
NW's 25 tiles**, leaving 12 for the early crop base where Boatlee's measured cluster (6 NW + 7 NE)
leaves 19. The farm is then cash-starved exactly when the money crop needs seed. Measured directly,
holding the compiler fixed and varying only the plan:

| pastures | free NW tiles | seed 3 | seed 7 |
|---:|---:|---:|---:|
| 13 (as written) | 12 | $50,368 | $46,903 |
| 9 | 16 | **$85,566** | $58,103 |
| 6 | 19 | $83,715 | $5,808 |
| 0 | 25 | $25,739 | $16,704 |

Nearly double from a single gene. That is S2's job, not the compiler's — and it is exactly the
question Phase 2 exists to answer, so C5 should be read as gating the *plan*, with the compiler's
own gates (steps/useful, blocked_ops, thirst) already met.

## E61 — Auditing the finished tasks, and the DROP spec that loses $5-9k

Independent re-review of I0, I1, I2, C0-C4: every suite re-run, every headline claim re-measured on
fresh seeds. **Most of it holds.** I0, I1's kagsim half, I2, C0 and C1 are substantiated — minor
notes only: I0's "29 assertions" is 29 test functions / 32 collected tests, `DROP` is untested at
the three locked-quadrant centre tiles, and melon's `max_yield_day` is read as a constant rather
than asserted. What did not hold is below, and it is the same failure both times: **a stated gate
that was never actually being measured.**

| claim | audit |
|---|---|
| C2 "0.85 steps/useful" | replicates on fresh seeds — but it is the *planned* ratio the router reports about itself; in-play truth is **1.02**. Both under the 1.15 gate |
| C2 "within 5% of exact tours" | the Hungarian 4x8 optimality check had been silently replaced by a **single-unit TSP**. Re-run honestly, the router **fails**: 1.26-1.72x exact assignment |
| C2 spec item 4 (logistics/DROP) | **unimplemented** |
| C3 "0 thirst deaths" | replicates *in shadow mode*; live play loses ~2.6 plants/game (already disclosed in E60's counters) |
| C3 pruned / overcommit counters | **dead code** — computed, never wired to the harness |
| C4 three gates | all reproduce: fallback counter proven live by an injected exception, `blocked_ops` 1, p99 6.7 ms, ~$50k |
| C4 `release_pressure` | **dead gene**; terminal spread simplified to dump-all; mid-day repair counter dead; `BUY_SEED` on-demand, not bulk; dawn `orders[:10]` truncation silently dropping sells |

### The DROP spec, implemented as written, loses money

C2.4 says: fill leftover turns with a `DROP` at a centre tile whenever a unit ends with inventory,
so sales happen today rather than tomorrow. Implemented literally — drop every day — it **costs
$5-9k a season**. It buys nothing (the shed sells at dawn either way) and spends the turns that were
doing the work: milk/cow-day 0.48 -> 0.23, thirst deaths 1.4 -> 3.4. A tidier-looking farm, a poorer
one.

The real leak is a single day. The episode ends at hour 22 of day 29, so end-of-day never runs and
**goods still in hands at the end never meet a market turn at all**. Gating the drop to the last day
(`agent/router.py::_schedule_drops`, `verify.py drop_home=last_day`, `main_v4.py` sell-on-unload +
`_carried`):

**+$8,604/season vs `starter`, 95% CI [7,640, 9,568]** — 160 paired games, block 40000:40080,
better 145/160, **worse 0**. Every non-money counter byte-identical to pristine. Effect counters:
`drops_scheduled` 3.14/game, `hand_drops` 2.41/game.

### The router genuinely fails its own optimality gate

With the real Hungarian check restored on 4 units x 8 tasks, the router walks **1.26-1.72x** the
exact assignment. Cause isolated to `partition()`: its sweep cuts are **unit-blind**, splitting the
board geometrically without regard to who is being handed which half. Pinned as a strict `xfail` at
the spec'd 5% plus a regression guard at 1.8x, so the gap cannot quietly widen and cannot be
mistaken for a passing gate again. Router tests also gained a real deadline assertion
(mutation-checked) and 50 random farms, which is what C2's done-when asked for.

Both C2 findings are the same lesson as E44: a check that cannot fail is not a check. The TSP
substitution and the self-reported ratio both produced numbers that *looked* like the gate.

## E62 — Wiring the dead counters, and what they say about boatlee_like

Three of C4's spec'd mechanisms existed only as prose. Building the effects ledger made them
measurable, and the first thing the live counters did was refute a standing assumption.

### The shed-pressure valve is plan-dependent, not dormant

`release_pressure` (dump when the shed nears its 100 cap) fires **zero times** on `boatlee_like` vs
`starter` — shed peaks at 67 against the 70 threshold — and is worth **$0 ± $0 over 80 paired
games**. That is not "refuted": the gene is a hoarding insurance policy and `boatlee_like` does not
hoard. On a hoard plan it fires 3.3x/season, halves `shed_overflow_discarded`, and is worth
**+$3,431 ± $1,427** (80 paired games, block 41000:41040). Against the real boatlee it fires
1.2x/game. Keep it; measure it on the plan that provokes it — a result is scoped to the system that
produced it (E26, E34).

### The order truncation was real, and silent

`maxMarketOrdersPerTurn = 10` is enforced (`kaggriculture.py:551,560`), and the dawn market list was
being cut with `orders[:10]` — sells past the tenth slot vanished with no error. Counters after the
fix: `orders_truncated` ~6.9/season, `sells_deferred` ~10/season. Another instance of the standing
rule: **an order is not a sale.**

### `boatlee_like` is not a clean-compile plan

The newly live counters say what was invisible while they were dead code: on `boatlee_like` the
compiler **prunes ~200 tasks a season**, runs **1.4 overcommit days**, escalates hands ~3 times, and
truncates ~7 dawns dropping ~10 sells. C3's "0 overcommitted days" (E59) was true of the shadow
model and false of live play. This is the second time a zero counter meant "not wired" rather than
"not happening".

### A stock-ledger bug the audit itself got wrong

`verify.py` never decremented carried stock, so several tasks in one day could each claim the same
unit in a single hand. Worth noting for the record: the audit's own proposed `* 0` term was itself a
no-op and would not have fixed it. Fixed properly in the ledger.

## E63 — Conjunction gate: the three fixes together

E43's rule — measure conjunctions, never assume additivity. Combined (last-day DROP + effects ledger
/ valve / truncation fix + stock ledger) vs pristine, 80 paired games both seats, fresh block
42000:42040:

| opponent | delta | 95% CI | better |
|---|---:|---|---:|
| `starter` | **+$7,744** | [6,408, 9,080] | 77/80, 0 regressions |
| `boatlee` | **+$2,280** | [1,630, 2,929] | winrate still **0/80** |

**No interaction** — every counter lands inside its solo range, so the combined figure is the DROP
fix carrying and the rest not fighting it. Full suite 556 passed / 1 skipped / 6 xfailed;
`make verify` 0 divergences. Results in `results/games.jsonl` blocks 40000:40080, 41000:41040,
42000:42040 (combined fingerprint `f9c0c4b4f339`, pristine `76b7075bdc92`).

The boatlee gap is ~$140k and none of this touches it. That is E60's finding restated: the
compiler's execution is at or better than boatlee's on every efficiency counter, and the deficit is
entirely in the **plan**. These three fixes are worth ~$8k of it. Phase 1's remaining money is in
C5/S2, not in more compiler repairs.

## E64 — C2: `partition()` made position-aware — 1.49x exact to 1.015x, +$28.7k in play

E61 isolated the router's optimality failure to `partition()`: its sweep cuts were **unit-blind**,
splitting the board geometrically without regard to who was handed which half. Pinned as a strict
`xfail` at the spec'd 5% with a 1.8x regression guard. Fixed.

`partition()` now assigns from **unit start positions**: cheapest-insertion seeded farthest-first
and again in reverse (better kept), then a boundary-relocation repair pass with a bounding-box
lower-bound prune. The old unit-blind sweep survives only as the `starts=None` fallback, and a
counter proves it is unused in the live path. Effect counters `insertion_stops`, `boundary_moves`,
`sweep_stops` (`STATS` at `agent/router.py:183`).

### Optimality, against a Held-Karp + partition-DP exact oracle (4 units x 8 tasks)

| | instances | mean | median | worst | within 5% |
|---|---:|---:|---:|---:|---:|
| before | 30 (seeds 400-429) | 1.492x | 1.524x | 2.056x | 1/30 |
| after | 40 (seeds 400-439) | **1.015x** | **1.000x** | 1.105x | **34/40** |

The spec's 5% test is now a normal passing test rather than a strict xfail, and the regression guard
tightened **1.8x -> 1.15x** over 20 seeds. Mutation-checked: reverting the call site to the unit-blind
partition fails **24 tests**. `order_stops`/2-opt were confirmed *not* the problem — the single-unit
test is unchanged, so the entire 1.49x was assignment, not tour quality.

### Load balance is a constraint, and the test that caught it was not about money

Pure cheapest-insertion overloaded one hand to **23/24 turns**. That dropped a survival water, and
the keeper rescue that covered it broke role purity — caught by `test_roles_stay_pure`, not by any
dollar figure. A soft load penalty now applies: zero below 60% of the day, steep above.

### Budget

`route()` at 12 units x 100 tasks: **1.71 ms** (spec < 20 ms). Worst in-play agent turn 7 -> **27 ms**
(budget 100 ms). Getting there needed inlined distance, one repair round, and the bbox prune — the
first version ran **72 ms** and failed `make verify`'s turn-budget test at **214 ms**.

### In play — `compiler` vs `starter`, 80 games both seats, seeds 43000:43040, paired

Money $50,772 -> **$79,500**: **paired delta +$28,728, 95% CI [19,760, 37,696]**, better in 57/80.

| counter | before | after |
|---|---:|---:|
| steps_per_useful | 1.05 | **0.78** |
| milk / cow-day | 0.48 | 0.56 |
| straw / plant | 2.6 | **5.6** |
| thirst | 2.8 | 2.3 |
| blocked_ops | 1 | 1 |

The largest single in-play delta of Phase 1, and it came from a gate that had been failing in plain
sight since E58 behind a substituted check (E61's lesson: a check that cannot fail is not a check).

### Known regression — **OPEN**, E61's DROP number no longer stands at full strength

The fuller day squeezes out the last-day walk-home `DROP` that E61 bought (+$8,604, CI
[7,640, 9,568]). Deliberately **not** fixed inside the same measurement, so the partition delta stays
clean. Measured: `drops_scheduled`/`hand_drops` 3.3/2.7 -> **1.4/1.0**; probe on seeds 43000-43005,
units unloaded 17 -> 6, still-holding 21 -> 27, ripe@end 10 -> 22, shed overflow 2 -> 14. A future
session must not assume E61's figure survives here. Suggested approach on file: **budget the return
leg during last-day routing** rather than appending the DROP afterwards.

### Suites

Full suite **577 passed / 1 skipped / 1 xfailed**; `make verify` 0 divergences. Only
`agent/router.py` and `tests/test_router.py` changed.

## E65 — Closing E64's DROP regression: **two** defects, +$6,664, buzzer holdings 0/39

E64 left the last-day walk-home `DROP` crowded out by the denser day and filed the suggested fix
(budget the return leg *during* last-day routing). Doing it uncovered a **second, independent
defect** that had been eating the same money. Both fixed in `agent/router.py` + `agent/verify.py`;
`tasks.py` and `main_v4.py` untouched.

### Defect 1 — the last day is one turn shorter than the router believed

The framework plays `episodeSteps - 1` turns, so **day 29 stops after hour 22 and hour 23 is never
issued**. `expand` filled to `turns_left` and `_schedule_drops` happily placed an op at hour 23.
Those DROPs were compiled and counted and then simply never emitted — most of the standing
`drops_scheduled`-vs-`hand_drops` gap. Fix: `horizon = min(turns_left, LAST_TURN - turn0)` gating
`partition` / `order_stops` / `expand` / `_stagger_plants` / `_schedule_drops`, **on the last day
only** (`agent/router.py:601-607`). This is a general fact about the env, not a router quirk — any
future work that reasons about "23 turns on the last day" is wrong by one.

### Defect 2 — the return leg has to be withheld, not appended

From the first stop that puts produce in a unit's hands, every later stop must now leave room to
reach a centre tile and `DROP`; stops that do not fit **yield to the leg** (`_home_cost:484`,
`expand(..., reserve_drop=):527-549`, `_drop_reserve:697`). That is correct by construction — an
undeliverable day-29 harvest pays **$0**. `_rescue_survival` and `_stagger_plants` see the budget
net of the reserve.

Why the appended form could not be rescued: it is structurally *last*, after `_stagger_plants`
truncation and `_rescue_survival` tail-lengthening, and a properly-filled day has **zero** spare
turns. Fixing only the horizon restores scheduled-to-emitted agreement but does **not** recover the
missing drops. Both defects were needed.

**Gotcha worth its own line:** the reserve is gated by a **new `last_day` param, distinct from
`drop_home`.** Folding it into `drop_home` silently changed `decide_hands`' hiring on *every* day
and broke the 1.15 steps/useful gate. Non-last days are bit-identical to baseline.

### Counters (proof it fired)

New `STATS`: `drop_reserved` **4-17/game**, `drop_yields` **3-16**, `drop_short` **0 on every probed
seed** (a unit finishing the last day holding produce with no room). `drop_short` is guarded to
budgeted legs only — ungated, `decide_hands`' probe calls flood it to ~503/season.

### In play — paired, `compiler` vs `starter`, block 44000:44040, 80 games both seats

$77,208 -> **$83,872**: **+$6,664, 95% CI [+5,923, +7,404], better 80/80, worse 0.**

| counter | before | after |
|---|---:|---:|
| drops_scheduled | 1.16 | **5.76** |
| hand_drops | 0.79 | **5.13** |
| scheduled -> emitted | 68% | **89%** |
| steps_per_useful | 0.775 | 0.785 |
| unharvested_ripe_at_end | 19.4 | 25.6 |
| milk / cow-day | 0.578 | 0.563 |
| straw / plant | 5.45 | 5.36 |

The residual 11% scheduled-but-unemitted is escalation-branch phantom hands, pre-existing. The
bottom three rows are **intended trades, reported honestly**: fruit is deliberately left standing
because a day-29 harvest that cannot deliver pays $0; the ~2% dips share that cause. Unchanged to
three decimals: thirst 2.288, age-loss 15.475, overflow 16.775, and every pruning/truncation
counter.

### Buzzer probe, seeds 44000-44005 — and a methodology correction

Last-day `DROP`s 7 -> **34**; units still holding at the buzzer 29/39 -> **0/39**; money over the 6
seeds $351,697 -> **$392,146**.

**The first version of this probe was wrong.** The final observation is handed out *before* the
final action, so a unit that unloads on the buzzer turn still reads as holding. Corrected numbers
above. Any future end-of-episode probe must account for the off-by-one.

### boatlee sanity, block 44000:44020, 40 games both seats

**+$1,662, CI [+1,181, +2,142], better 37/40**; counters behave the same; **winrate still 0/40**.
The ~$90k plan gap is untouched — consistent with E63's finding for E61. This is a compiler repair,
not plan money.

### Guardrails

E64's 5%-of-exact optimality test and the 1.15x guard pass undisturbed; roles / deadlines /
turns-bounds / fib all green. `route()` **1.92 ms** ordinary, **1.86 ms** last-day. Full suite
**583 passed / 1 skipped / 1 xfailed** (+6 new tests); `make verify` 0 divergences. Mutation checks
on all three load-bearing lines fail **3 / 4 / 1** tests respectively (`reserve_drop`, `horizon`,
reserve-aware truncation). Results in `results/games.jsonl` blocks 44000:44040 and 44000:44020
(fingerprints: base `0c972bfeb9a4`, new `1dafaa205072`).

## E66 — C5 gate: kill does not fire; the "plan-bound" deficit is execution

**[REFUTES E60]** — E60's closing claim that the compiler's money shortfall is *plan-bound*, with
"the compiler's own gates already met", **no longer stands**. Re-measured with the E64/E65 router:
~all of the −$66.8k solo revenue gap resolves to **four named execution defects**, not to the
13-pasture plan. The plan question (S-track) should be asked *after* they are fixed, not instead.

Block 45000:45040, 80 games both seats, compiler = `Plan.boatlee_like()`, fingerprint `1dafaa205072`.
Probes in `/private/tmp/c5/`, all quantities differenced from delivered observations (never from
orders — CLAUDE.md's "an order is not a sale").

### The gate

| metric | target | compiler | boatlee | |
|---|---:|---:|---:|---|
| steps per useful action | ≤ 1.15 | **0.79** | 1.02 | PASS |
| plants lost — thirst | ≤ 10 | **2.5** | 0.0 | PASS |
| plants lost — old age | (not a defect) | 16.6 | 10.0 | — |
| strawberry / plant | ≥ 6.5 | 5.6 | 7.9 | **FAIL** |
| milk / cow-day | ≥ 1.2 | 0.60 | 1.11 | **FAIL** |
| unharvested ripe at end | ≤ 10 | 28 | 2 | **FAIL** |
| solo money | ≥ 110k | $89,621 | — | **FAIL** |
| blocked_ops | ≤ 12 | 1 | ~10 | PASS |

vs boatlee on the same block: compiler $50,036, boatlee $139,465, **0/80**.

**The kill criterion does NOT fire.** It is `plants_lost > 20` **or** `steps_per_useful > 1.4`;
measured 2.5 and 0.79. The compiler idea is not killed and the F-gate fallback is not needed.

### Revenue by product, 20 games (compiler / boatlee)

| | MILK | WHEAT | FERTILIZER | STRAWBERRY | WOOL | MELON | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| compiler | 16,795 | 2,011 | 7,725 | 38,979 | 13,562 | 19,206 | **82,049** |
| boatlee | 43,473 | 18,732 | 17,372 | 48,552 | 22,126 | 26,042 | **148,852** |

(The compiler also *spends* ~$11k less, so the money gap is narrower than the revenue gap.)

### 1. Herd arrives day 18.6 vs the plan's day 5 — ≈$45k, 67% of the gap

boatlee places its herd on day 7.7. Downstream: milk −$26.7k, wool −$8.6k, **fertilizer −$9.6k**
(fertilizer is animal-derived, 1/animal/day, `kaggriculture.py:831` — same cause, not a separate
defect).

**Husbandry is not the problem.** feed 0.79/0.90, care 0.87/0.87, at-cap 0.05/0.05; both agents
extract the full 3 milk/cycle. milk/cow 6.4 vs 24.8 is *purely arrival day*: placed 18.6 → ~2
productions, 7.7 → 8.

Two independent gates:

- **(a) Cash floor, days 2-11.** Wallet sits at $1-$1,015 because 13 pastures + seed are bought
  before any animal. boatlee buys **structure + animal 1:1** and never owns an empty pasture; the
  compiler holds **13 empty structures and 3 animals by day 8**.
- **(b) `_feedable_animals` throttle** (`agent/main_v4.py:342`),
  `= int(0.8·wheat_tiles) + shed_wheat//2`, binds **independently of cash** — day 12: wallet $6,490,
  feedable 4. Its wheat base oscillates 1-10 tiles against boatlee's steady 13, and the gate ignores
  that wheat is purchasable via `BUY_PRODUCT`.

### 2. Strawberry tick-day watering — ≈$7k, and it *is* the straw/plant gate

`kaggriculture.py:797`: `fertilized = was_watered and fertilized_until_day >= current_day`. The +2
fertilizer bonus is **void unless the tile is watered ON the production day**.

Tick-day watered: compiler **67.0%** vs boatlee **98.6%** (fertilizer coverage is fine: 95.4/98.6).
Doubled productions 2.57 vs 3.93 per plant — **−1.36 units/plant, ~85% of the strawberry gap.**
The router waters *more* overall (48.5% vs 40.3% on non-tick days): **wrong days, not too little.**
Fix: treat tick-day water on a fertilized ongoing tile as a **survival-class deadline**.

### 3. Wheat base instability — ≈$17k

Wheat revenue $2,011 vs $18,732; tiles oscillate **1-10 vs a steady 13** (batch replant). Same root
as the throttle in (1) — the two fixes reinforce.

### 4. ripe@end 28, decomposed — the counter overweights wheat

WHEAT 16.6 (persistent from day 28, worth a few hundred $), STRAWBERRY 9.2 (day-29 dawn production,
deliverable *within* day 29 — a genuine ~$2k miss), COW 3.2 (~$500), MELON 0.5. **Mid-season
standing produce is not a defect**: on day 26 boatlee has 104 standing to the compiler's 15, and
clears to 2.3 by day 29.

### Recommended order (each has an effect counter — check it before reading money)

1. **Herd arrival** — buy structure+animal 1:1; replace `_feedable_animals`' wheat-*tile* proxy with
   purchasable wheat. Targets: `placed_day ≤ 9`, cow-days ≥ 180.
2. **Tick-day water as survival-class.** Targets: ≥95% tick-day watered, straw/plant ≥ 7.0.
3. **Wheat cycle staggering** — steady tile count.
4. **Day-29 strawberry** delivery (minor).

Only then question the plan (S-track).

## E67 — E66's fixes 1 and 2: tick-day watering + herd arrival, and their conjunction

Both of E66's top mechanisms, measured separately and then 2x2. Combined **+$27,715** solo. One
E66 reading refuted along the way.

### (a) Tick-day watering — `agent/tasks.py` only

**The bug was subtler than E66's framing.** `_water_bonus` gated on `fertilized_until_day >= day`,
which is **false on fertilize-today ages** (strawberry 9/13, tomato 7/10) because the FERTILIZE has
not landed yet at planning time. So on *half* of all tick days **no water task was emitted at all** —
not mis-prioritised, absent. Fix: a new `_fertilizes_today` predicate is the single source for both
the fertilize and the water task; tick-day water is emitted **survival-class**; `_projected_units`
corrected (an at-cap tile being harvested today has headroom). Tomato ticks **nightly** (interval 1,
ages 7-10, `kaggriculture.py:789-801`) — it needs the same treatment, not a strawberry special case.

Counters (`tasks.STATS`): `tick_waters`, `tick_waters_fert_today`. Tick-day watered
**64.3% -> 96.5%**; doubled productions **2.17 -> 3.10 per plant**. Non-tick watering *falls* to
boatlee's rate: water was **reallocated, not added** — steps/useful flat.

Solo: **+$2,925, CI [+1,150, +4,701]**, block 47000:47040.

### (b) Herd arrival — `agent/main_v4.py` only

`_paced_plan` trims the herd to `min(due, owned + lead)`, so structures are built **1:1 with
scheduled animals** — empty pastures on day 8 **13 -> 6.4**. `_feedable_animals` gains a
purchasable-wheat term, `(shed + budget // price) // FEED_BUFFER_DAYS (=4)`, evaluated on the budget
the purchase *leaves*.

Cow `placed_day` **18.1 -> 15.0**, cow-days **95 -> 130**, milk/cow-day **0.60 -> 0.82**.
Solo **+$22,879, CI [+14,857, +30,900]** block 46000:46040; held-out 47000:47040 **+$38,555**.

**[REFUTES E66 §1(a)]** — buying **animals before seed at dawn**, the direct reading of E66's
cash-floor diagnosis, is **−$22,296, CI [−31,963, −12,628]**. It starves the strawberry cohorts
(straw/plant **6.4 -> 3.3**). The cash floor is real; paying for it out of seed is not the fix.
Pacing structures to animals is.

Regression vs boatlee: **−$5,686, CI [−10,689, −682]** solo-equivalent — the extra head is
**unserviceable in a flooded market**.

### (c) Conjunction, 2x2 on block 48000:48040

The herd solo baseline already contained most of the tick fix, so the gate re-ran a **full 2x2**
rather than trusting the two solo deltas.

| arm | delta | 95% CI |
|---|---:|---|
| tick only | +$4,201 | [+1,955, +6,446] |
| herd only | +$24,667 | [+15,118, +34,215] |
| combined | **+$27,715** | [+17,992, +37,438] |
| interaction | −$1,153 | [−4,084, +1,779] |

**Clean and additive.** vs boatlee the combined arm is **−$3,813, CI [−8,643, +1,018]** — CI
includes zero, and it is **0/80 in every arm**, so the head-to-head number distinguishes nothing.
**Decision: ship both.** Do not gate pacing on a noise-sized effect; revisit after wheat. If a gate
is ever needed, the signal is **our own observable feed-coverage rate**, not opponent pressure.

The `straw/plant` counter is **flat** in the combined arm — a benign volume/quality trade: the herd
fix starts **+16 plants**, which dilutes per-plant while total strawberry rises **159 -> 212**.

## E68 — E66's fix 3: wheat-wave stabilization. Two refutations and one new open defect

Fix (`agent/tasks.py`): `replant_cycle(WHEAT) = 5`, taken from `max_lifespan_step`
(`kaggriculture.py:224`). A replant cohort's **first fill is uncapped**; thereafter re-sowing is
rate-limited to `ceil(tiles / cycle)` per day. Phasing is then **permanent and self-healing** — a
disturbed wave re-staggers itself instead of re-batching.

Collapse days (≤3 tiles, days 5-25) **5.2 -> 3.1** vs starter. Downstream: thirst **1.44 -> 0.93**,
milk/cow-day **0.53 -> 0.62**, pruned tasks **209 -> 126**.

Money: **+$2,930 [−349, +6,210]** and **+$2,239 [−897, +5,376]**, pooled 160 games over
49000:49040 and 49040:49080. **Positive in 4/4 arms, no CI clear of zero.**

**[REFUTES E66 §3]** — the ≈$17k attributed to wheat instability does **not** exist for this
mechanism. **The compiler EATS its wheat**: ~26 units *sold* of ~220 produced. A wheat tile is a
**feed subsidy, not revenue**; stabilizing it is worth **~$3k**, and the value shows up as milk and
survival, not as wheat sales.

**[REFUTES the "slope" model of the early farm]** — **the early farm is a cliff, not a slope.**
Two variants that each moved early cash by a small amount:

- Capping the **first** fill (the obvious symmetric form of the rate limit): **−$45.8k**,
  `plants_started` **102 -> 66**.
- A **$200-700 wheat-seed buffer**: **−$25.5k** vs boatlee.
- `WAVE_SLACK=2`: **+$3.9k vs boatlee** but **−$12.5k vs starter** on held-out, CI
  [−19.9k, −5.1k]. Rejected.

**Any change that moves early cash must be measured against both opponents.** Single-opponent
readings here have the wrong sign as often as the right one.

### New defect, **OPEN**: HIRE starves the dawn market queue

`maxMarketOrdersPerTurn = 10` (`kaggriculture.py:551,560`). `_dawn` appends HIREs **before** buys,
so a **10-hire day buys NOTHING** — a simultaneous fertilizer, feed *and* seed outage. Traced on
seed 49100, days 11-13 and 24: hour 1 then compiles at **cash = 0** and prunes **~50 tasks/day**.

Deliberately **out of scope** for the wheat run: reordering changes crew size, so it needs its own
experiment with its own counter.

## E69 — C5 re-gate after E67/E68: 5/7 PASS, money gate cleared

Block 50000:50040, 80 games both seats, fingerprint `ab86c6175b7b`.

| metric | target | compiler | vs E66 | |
|---|---:|---:|---:|---|
| steps per useful action | ≤ 1.15 | **0.76** | 0.79 | PASS |
| plants lost — thirst | ≤ 10 | **1.14** | 2.5 | PASS |
| strawberry / plant | ≥ 6.5 | **6.77** | 5.6 | **PASS** (was FAIL) |
| milk / cow-day | ≥ 1.2 | 0.73 | 0.60 | FAIL |
| unharvested ripe at end | ≤ 10 | 26.7 | 28 | FAIL |
| solo money | ≥ 110k | **$114,302** | $89,621 | **PASS** (was FAIL) |
| blocked_ops | ≤ 12 | 1.6 | 1 | PASS |

**5/7 PASS, +$24,681 vs E66.** **The kill criterion does not fire** — 1.14 and 0.76 against bars of
20 and 1.4. **80/80 vs starter.**

Counters (checked before the money, per CLAUDE.md): tick-day watered **97.6%** (boatlee 98.6),
doubled **95.2%**, first animal `placed_day` **2.0**, cow-days **128** (boatlee 174), wheat stable at
**6-7 tiles**, fallbacks **0**.

### The two residual FAILs

**milk/cow-day = a cow-first defect.** First COW lands **day 11** (2 by d12, 7 by d16) against
boatlee's **day 1 / 7 by d8**. The 1:1 pacing takes the `plan.herd` **prefix** — four sheep first —
and the cash gate then binds on the **$400** cow. Compounded by a **dead day-25 batch**
(`animals_ordered_d25` 6.9) that adds cow-days and **zero** productions. Fix in flight, expected
**+$15-17k**.

**ripe@end is mostly a counter artifact.** 17.4 of the 26.7 is **wheat**, worth **~$0** (the compiler
eats it, E68) — the counter overweights it. The genuine part is **7.2 day-29 strawberry, ~$1.5-2k**.

### vs boatlee: $39,405 / $129,021, **0/80** — and this is *not* an execution regression

**boatlee's presence closes the markets.** Shed discards **78.5 units/game**, the `release_pressure`
valve is saturated (**10.1 fires / 168 units**), and `plants_started` is **97.6** against **195.9**
solo. The execution counters are at or near boatlee's own values; the produce has nowhere to go.

**Verdict recorded: one more execution round (cow-first), then proceed to C6/S.** The remaining
head-to-head deficit is **plan-bound** — cohort count, wheat scale, sell pressure — which are
**S-track genes. Do not spend more execution rounds on it.**

## E70 — Cow-first herd ordering + payback cutoff: the C5 execution arc closes

The fix E69 put "in flight". Three changes, `agent/main_v4.py:349-439` and `agent/plan.py:323-333`.

**1. `_last_buy_day(species)` — a payback cutoff derived from env arithmetic**, not a tuned constant
(`kaggriculture.py:822-829`): an animal bought at dawn on day `d` is placed on `d+1`, and
`k = ceil(cost / ((1 + interval) · base_price))` productions must fit before day 29. That gives
**COW 20, SHEEP 22, GOOSE 22**. Counter `animals_past_payback`; `animals_ordered_d25` **5.8 -> 0**,
killing E69's dead day-25 batch.

**2. `_purchase_order` / `_yield_per_dollar` — cows rank first.** COW **$0.60**/day/$ against SHEEP
$0.53 and GOOSE $0.33, and the cow is also *cheaper* ($400 vs $500, `kaggriculture.py:19-23`), so
cow-first is both higher-yield and lower-gate. Spending order = schedule order truncated at the
first unaffordable head. Slot mapping is safe (COW and SHEEP share PASTURE); mixed-structure herds
keep schedule order, tested.

**3. `plan.py` `cow_start` 5 -> 1**, matching boatlee's *actual* opening turn — the traces show
HIRE×4 + BUY COW 1 + BUY SHEEP 4 at **day 0 hour 0**, with the bulk cows on days 5-8.

### Measured

Paired, tuning block 46000: cutoff alone **+$1,162 ± 386** (n=200); priority alone **+$3,059 ±
3,480**; `cow_start=1` **+$3,524 ± 2,914** (n=400).

Held-out 51000:51040, 80 games both seats, paired: **vs starter +$6,651 ± 5,629**; extended to
51000:51200 (400 games) **≈ +$7.7k ± 2.5k**. **vs boatlee +$10,508 ± 7,038 — no regression**; the
boatlee regression E67's herd fix carried is **gone**. Still 0/80, gap **−$93.5k -> −$84.0k**.

Counters: first cow `placed_day` **11.7 -> 2.01** (target ≤4 PASS); cow-days 126.8 -> **152.9**
(target 170 not met); milk/cow-day 0.77 -> **0.90** (gate 1.2 not met); `animals_ordered` 31.7 ->
26.7; WOOL **−$2.3k** (sheep-days 98.8 -> 76.8, crowded out — the cows are worth more than the wool
they displace). Feed and care improved in both arms (0.80 / 0.91). Guardrails: blocked_ops 1.9,
thirst 0.9, steps/useful 0.75, fallbacks 0. Suite **601 passed**, `make verify` **0 divergences**.
Tests mutation-checked; the payback test asserts **in play** on seed 7, chosen because its herd
escapes late — a no-escape season would pass while proving nothing.

### [REFUTED] More early animals

- **`cow_start=0` with sheep on day 2: −$96k.** Three cows plus sheep on the opening turn starve the
  seed budget — `plants_started` **53**.
- **No animals on day 0: −$30,741 ± 3,757.** Day-0 buying is load-bearing.
- **2/day pacing: −$18k to −$47k.**

The failure mode of the herd is buying too **many** early, not cows before sheep. Order is free;
volume is not.

### Closing verdict: milk/cow-day is now **plan-bound**, not execution

0.90 against a 1.2 gate, and the residual is no longer something the compiler can fix: the wallet
sits at **$4-$1,000 on days 4-9**, so cows 2-9 wait on cash. boatlee plants **CARROT/TOMATO on day
0** as a fast early-cash crop; `Plan.boatlee_like()` has **none**. That is an **S-track gene** (an
early-cash cohort), not a routing defect.

**The C5 execution arc is CLOSED.** E66's four mechanisms plus cow-first all landed; measured lift
across the arc ≈ E69's **+$24.7k** + E70's **+$7.7k**, and solo money is now **≈ $120k** against the
110k gate. **Proceed to C6/S.** One known execution item remains open: the HIRE/dawn-queue defect
(E68).

## E71 — HIRE dawn-queue fix: a $29k defect on a substrate that stopped existing

E68's open item, closed. **The env fact E68 got wrong:** `maxMarketOrdersPerTurn` caps a **turn**,
not a day — `_process_market` runs on every interpreter step (`kaggriculture.py:941`, truncation
`:551,560`), so the day carries **240 slots**, not 10, and land is buyable at any hour (asserted in
play). The defect was never "10 orders/day"; it was that `_dawn` dumped everything into hour 0 and
let truncation choose.

**Fix** (`agent/main_v4.py`): `_truncate` -> `_dispatch`. Orders are ranked
**BUY_LAND -> BUY_SEED/BUY_PRODUCT -> BUY_ANIMAL -> HIRE -> SELL** (E62's sell rule kept) and served
**every turn**, not only at dawn. Overflow hires are **dropped**, not deferred.

**Spreading or deferring hires is measurably WORSE** — three variants, all against the same
pre-C6 tree: **+$22.5k / +$25.4k / +$26.4k** against **+$28.9k** for dropping. A deferred hire buys
a mid-day recompile (**39 blocked_ops/game**) or an idle crew-hour; neither pays for the hand.
Control against the headcount explanation: pristine tree with `wanted − 2` hands = **−$36,629**. The
gain is the queue, not a smaller crew.

### Money — and its substrate

Frozen **pre-C6** tree: **+$28,875** tuning, **+$29,365 [+25.4k, +33.4k] held-out, 76/80**;
`pruned_tasks` **354 -> 214**, exactly as E68 predicted.

**Final tree (C6 landed, baseline ≈$122.8k): money-neutral.** +$1,574 [−933, +4,081] and +$402
[−2,597, +3,401] vs starter, **−$1,541 [−5,583, +2,501] vs boatlee** (n=50); `pruned_tasks`
unchanged (140.6 -> 139.7). **The outage stopped binding** — the C5/C6 tree no longer queues a
day's whole intent at hour 0, so there is nothing left for the fix to rescue.

**Kept anyway.** Guardrails pass (feed **0.870** ≥ baseline 0.810, care 0.916, thirst 3.05,
fallbacks 0); counters `dawn_starved_naive` **5.5/game -> `dawn_starved` 0**, `hires_dropped` 16.0.
It is a tripwire: if a future plan re-loads dawn, the queue is already correct. Counters that moved
the **wrong** way and are not explained: straw/plant **6.55 -> 5.63**, `fertilize_hits` **66 -> 58**.

**[LESSON — concurrent-edit contamination]** The first measurement was garbage because C6 landed
**mid-run**: arm A had `projection_calls=0`, arm B `projection_calls=67`. The arms were different
trees, and the money difference read as a result. Caught only by **drift-hashing `agent/*.py`
around every arm** — which is now the practice. A paired run assumes a frozen tree; nothing in the
harness enforces it.

## E72 — C6 built; every money lever measured null-or-negative at 3 blocks; ships as genes, off

`agent/projection.py`, 588 lines, every mechanic derived from env source rather than fitted:
shop drain **6 events/day at `step%4==0`** (`kaggriculture.py:733,736`), **x2 for single-product
shops** (`:741`) so YARN_STORE/PET_CAFE eat **12/day**; town centre **1/day, all but FERTILIZER**
(`:734,745-747`); instances `min(8, day//3)` (`:867,886-891`), future unlocks entered as a mean over
shops; partial-day hour ordering (`:941-942` — a real **6-unit** bug, caught by reading the source);
supply forecast from **both** farms' tiles (`:783-800,:822-823`). Only `SELL_THROUGH` is fitted
(spent block 45000): **WHEAT 0.15 — the farm eats its wheat** (E68), and fitting it cut error
**27.5% -> 7.4%**.

**Prediction accuracy** (held-out 52000), median error on deviations **≥0.25T**: CARROT **2.1%**,
EGG **1.5%**, TOMATO **3.5%** (90-99% of cases within ±15%); WOOL worst at **32.5%**. The spec's
"±15% on raw inventory" bar is **unusable** — everything sits within 1% of `I0`, so the bar passes
without the model doing anything. Measured on **deviation** instead.

**Hunter done-when: met.** 6/7 hinge seeds fire (`scarce_cohorts=1`, 8 tiles -> TOMATO, realised
**$76-858/unit**); **0/7** calm seeds. STRAWBERRY cut from the target set: 3 shops into T=100 makes
it scarce **every** game, so the hunter became a constant plan edit (fired 110/120), not a hinge.

### Two findings

**(a) The hunter can NEVER fire against `boatlee_like`.** Pending cohorts are worth **melon $1,420 /
strawberry $1,660 per tile**, which outvalues any tomato price the market can reach. It is an
**S-track** lever: it needs a plan with a cheap pending block, not a better hunter.

**(b) [REFUTES E66's premise]** The day-12 tomato window — "$399/$445, ≥80 units at ≥$150" —
**does not hold at 1.32.7 on fresh blocks.** Deepest day-12 signal **2.5**; 96 units into T=200
average **$127**. The bar is met **later**, and only metered.

### Money: null or negative, 240 paired games x 3 blocks (51300 / 51400 / 52000)

- projected pricing: **−$2,675 [−4,602, −749]** vs starter, **+$46** vs boatlee
- strawberry redirect: **−$4,215** vs starter
- metering: **null** at both settings (0.60 never binds; 0.95 binds, +$800/−$56)

**[LESSON — E37/E39/E42 again]** Single-block 52000 showed **+$3,898 [+1,081, +6,715]** vs boatlee
and **did not survive two more blocks**. A clean CI on one block is still one block.

**Ships off.** Defaults `projected_pricing=0`, `scarce_floor=0.0`, hunter hinge-only. The shipped
tree is **outcome-identical to pre-C6** (40 games, byte-for-byte per seed) and **$0** different from
all-off over 160 games. `plan.py` **49 -> 54 genes** (`theta_hinge`, `theta_sqrt`,
`max_scarce_tiles`, `scarce_floor`, `projected_pricing`), round-trip holds. `tests/test_projection.py`
32 tests, **9/9 mutations caught**; `test_tasks` +5, 4/4. Suite **641 passed / 1 skipped /
1 xfailed**; `make verify` **0 divergences**. The `main_v4._sell_orders` hook is applied
(2 lines above `private = obs.get("private", {})`) and verified inert: 40 paired games,
compiler vs starter, 53105:53125 both seats — **every per-seed result identical to the pre-patch
tree** ($123,623 mean both arms, 40/40 wins both arms).

## E73 — S1: fitness + pool; the exploiters had to be measured into existence

`search/fitness.py`: `evaluate(vec, seeds, opponents)` -> fitness =
**0.7·weighted_win_rate + 0.3·tanh(margin/50k)**. `SEARCH_POOL` = boatlee **x2**, flooder,
tomato_rusher, executor_v7, both seats. `search_seeds(gen)` rotates 6 seeds inside **60000:64000**
and **raises** on any reserved block (54000:54080 acceptance, 50000/51000 gates) — the block
discipline is enforced in code, not in prose. Candidates travel to workers as `vec:` registry names
so each worker rebuilds the agent itself; one persistent `Pool`. Determinism: same vec + same seeds
-> bit-identical money, counters, everything.

### The spec's exploiters were pushovers — both had to be re-derived by measurement

- **flooder, as spec'd (2x strawberry = 75 tiles), is a punching bag.** Solo **$75k**, and it
  **LOST 8/8 to the compiler by +$42k**. Counters say why: it over-bought structures
  (`animals_past_payback` **149**) and **never got the strawberry into the ground**. A pool member
  that loses every game applies **zero selection pressure**. **Rebuilt to the measurement**:
  20 pastures, **67 strawberry tiles (1.86x)**, **14 cows (1.56x)** — solo **$98.5k**, and it now
  **beats the compiler 16/0 by $59.7k**.
- **tomato_rusher's "day-6 plant" is advisory, not a fact.** The cash floor delays the seed to
  **~day 13** — the **same E70 wallet cliff** ($4-$1,000 on days 4-9). Solo **$117.2k**; loses
  14/16 but with the **tightest margin in the pool (+$14.6k)**, so it stays.
- **3 of 4 pool members apply real pressure** (executor_v7 beats the baseline 12/0). PENDING
  registry entries emptied.

### Verification

`evaluate()` reproduces the knowns: solo **$124-126k** (E70 ≈$120k), **0% vs boatlee**
(−$74k/−$83k against E70's −$84k), counters in range. A dead plan scores **0.0024**.
**NOTE the spec's own bad-plan check is weak**: "no pastures" scores **0.164** against the
baseline's **0.172**. Win-rate is coarse at 6 seeds; **the 30% margin term carries the gradient**.

**Cost, measured, not estimated:** 7.4-8 games/s on 7 workers; a 48-game eval ≈ **6 s**;
a generation ≈ **5-6 min** — **12x the spec's 25 s/gen** (TASKS_v4 S2). Budget S2 accordingly.

## E74 — The NW-wheat money cliff: a LOCKED-string dawn bug, not a plan property

S1 surfaced a **2x money cliff** across one gene step: an 18-pasture NW-full genome earned
**$46.5k** where the 20-pasture neighbour earned **$91.5k**. A fitness landscape with a 2x step in
it is not searchable, so it was diagnosed before S2 launched.

**Root cause.** `_dawn` derived the shopping list **and** the crew size **before** the same-turn
`BUY_LAND`. `kaggriculture.py:724` unlocks the quadrant **in that same turn**, but `agent/tasks.py:567`
tests `tiles[y][x] is None` — and a locked tile is the **string `"LOCKED"`**, not `None`. So the
arriving quadrant contributed **no seed order and no hands**, and hour 1 pruned its whole cohort
against an **empty shed**. Normally a one-day tax; **a season when the land is late**.

**Full causal chain, seed 60050** (each link measured, not inferred): day-1 `int(0.8·wheat_tiles)`
admits **one extra $410 cow** with 3 NW wheat vs 2 -> the **day-6 NE gate is missed by exactly that
amount** -> NE lands **day 12**, and that windfall dawn spends **all 10 slots with NO strawberry
seed** (NE still read LOCKED) -> the **47-tile strawberry block peaks at 2 tiles sown**.

**[REFUTED by measurement] — four plausible causes, all wrong:** thirst (bad arm **0.6** vs good
**2.8** — the *bad* arm is thirstier-free); pruning / blocked / overcommit ≈ **0 in both arms**;
herd == structures (**byte-identical money**); and **"NW exactly full" per se** — a healthy plan
fills NW too. The controlling variable is **the NW wheat count, through the 0.8 throttle step**.

**Fix** (`main_v4.py`, ~40 lines): the **land decision moves above task derivation**, and a
`_unlocked_view` (a copy with the arriving quadrant freed) is used **only when the arriving
quadrant's oldest cohort is ≥ 6 days overdue** (`LATE_COHORT_DAYS=6`). The gate is **measured, not
chosen**: always-on shifts incumbents **−$1.6k / −$2.8k** on two blocks, threshold 3 is worse
(backlog distributions in the docstrings). Counter `land_day_reseed`: **0** over a boatlee_like
season (test-asserted), fires on **every late dawn** of the cliff genome.

**Repro after the fix:** bad arm **$44.3k -> $75.8k**; the wheat-ladder's worst neighbour step
**55% -> 3.5%**. Fresh-seed stress sweep (**17 genomes, 60200:60208**): worst step **14.4%** —
inside the **<30%** property S2 needs. **Incumbent invariance: 480 games/block x 2 blocks**, pooled
**+$373 [+32, +714]** and **−$5 [−39, +29]**, **946/960 byte-identical**.

**Residual flagged:** `int(0.8·wheat_tiles)` in `_feedable_animals` is **still a discontinuity** —
bounded in cost now, but it is the **noisiest axis in the sweep**.

Suites **707 passed**; `make verify` **0 UNEXPLAINED**; **3 mutations caught by 3 different tests**.

## E75 — S2 built; smoke healthy; production search launched

`search/ga.py` + `search/run.py` + `search/accept.py` + `tests/test_search_run.py` (**49 tests**;
named `test_search_run.py` because `tests/test_search.py` already belongs to the v1 CEM line).

**Design decisions that are load-bearing:**
- **Uniform per-BLOCK crossover** (land / herd / hands / cohort-slots as units). Per-**gene**
  crossover shreds cohorts.
- CMA genes are taken **whole from the fitter parent**; GA and CMA **partition the 54 genes**
  (asserted at import, so a gene added to neither is a test failure).
- **CMA budget ≥ 8, enforced with a warning.** **Budget 4 is one ask/tell round** — a random
  perturbation wearing the costume of a working inner loop. This was **a real config bug in this
  agent's own earlier runs**, now pinned by test.
- **Guard penalty 2.0**: at 1.0 a *perfect illegal* genome **ties the worst legal one**.
- State JSON written atomically; CMA seeded on **(run, gen, rank)** so resume is stream-independent.

**Smoke** (pop 48, 3 gens, 208 evals, **9,648 games**): best **0.5193 -> 0.5272 -> 0.5879**,
monotone; margin **−$9,241 -> +$5,984**; **diversity RISES 0.077 -> 0.137** (no collapse);
repair rate 0.28-0.30; **guard violations 0**; counters inside the C5 bars (steps/useful ≤ **1.051**,
thirst ≤ **8.06**, fallbacks **0**). The gen-2 winner differs from `boatlee_like` in **layout**
(14 pastures, NE day 8, melon 11) — and turned **`projected_pricing` ON**. That is a hypothesis at
**6 seeds**; **E72 measured that gene at −$2,675**. Watch it, do not believe it.

**Resume determinism:** killed-after-gen-2 vs uninterrupted — **all 3 generations bit-identical**.
**9 mutations, 9 caught**; one caught a **real gap** (the RNG state was unused by the first resumed
generation) — the test now replays **two** generations.

**[LESSON — verify before diagnosing]** The "compiler stall" chased earlier in this session was
**this agent's own `;` command chaining masking a killed process**. No agent defect existed. Check
that the thing you are diagnosing is real before diagnosing it.

**[LESSON — background duty cycle]** Ordinary background children get a **9% duty cycle** on this
Mac. **Production searches must run detached** (double-fork + setsid) under `caffeinate`, and log
rows now carry **wall-time** so throttling is visible in the log rather than inferred from a slow
result.

**Budget:** ~**5.2 s/eval** -> 50 generations ≈ **4.7 h**. Suite **710 passed**; `make verify`
**0 divergences**; acceptance block **54000:54080 untouched** (no ledger yet). **Production search
launched: 75 generations, detached.**

## E76 — S2 harvest: a strictly better plan; the boatlee signal was 6-seed noise; S3 fails and prescribes representation widening

**Search.** **68 generations**, stopped at plateau (best unchanged **gens 25 -> 67**), **4,320
evaluations** (3,232 GA + 1,088 CMA), guard violations **0**, fallbacks **0**. Best score **0.6844**,
weighted pool win **0.667**; per-opponent at 6 seeds: **boatlee 0.167 (2/12)**, flooder /
tomato_rusher / executor_v7 **1.000** each.

**The winner is genuinely different** (S2's must-differ criterion, met in **both** herd and cohorts):
herd **4 sheep + 9 cows -> 5 sheep + 7 cows** (starts 2/1, `animals_per_day` **3 -> 4**); cohorts
**6 -> 5** (the **NE wheat replant cohort deleted**); strawberry NE **22 -> 25 tiles, day 7 -> 10**,
SW **14 -> 11, day 10 -> 15**; melon SW day **11 -> 10**; pastures **13 -> 12**; land NE **6 -> 9**,
SW **10 -> 9**; `hands_start` **4 -> 3**. C6 genes: **`projected_pricing` OFF** — the E72 −$2,675 gene
**did not survive** the full run, unlike the smoke's gen-2 winner (E75's "watch it, do not believe
it" was the right call). Scarcity thetas moved (`theta_sqrt` 0.5 -> **1.15**, `max_scarce_tiles`
8 -> **19**); market consts moved **hard**: `release_pressure` 70 -> **16**, `sell_floor_wool`
0.35 -> **0.131**, `sell_floor_melon` 0.35 -> **0.695**, `frontrun_lead` 10 -> **20**.

**Acceptance gate (54000:54040 spent; 54040:54080 still reserved): 100% (80/80)**, Wilson CI
**[95.4%, 100%]**, margin **+$123,644**, counters clean (thirst 0.59, steps/useful 0.717,
fallbacks 0).

**[The margin is an artifact — diagnosed, not assumed.]** The incumbent earned **< $1,000 in 74/80
games** (median **$2**) while the candidate soloed **$122.6k**. Incumbent counters: `plants_started`
**12 vs 87**, `animals_blocked_cash` **22**, `structures_deferred` **259**, idle **0.74**, **no
crash**. That is the **E70/E74 wallet cliff**, triggered by the candidate's aggressive early selling
(`release_pressure` 16 et al.). **Genuine market denial — not $123k of skill.** Read the margin as
"the incumbent is fragile to being starved", not as candidate strength.

**Pool no-regression (62000:62012): strict domination.** flooder **100% / +$83k**, tomato_rusher
**100% / +$118k**, executor_v7 **100% / +$94k**, boatlee margin **+$17.7k**; thirst **collapses**
(3.0-7.0 -> 0.08-0.83).

**vs boatlee, 80 games, fresh 63000:63040** (freshness **verified from the search log** — all 408
search seeds lie in 60000:60407, overlap **empty**): **0/80**, CI **[0%, 4.6%]**, **$62,764 vs
$130,472**, margin **−$67,708** against the incumbent's **−$82,768** on the identical block ->
**gap closed +$15,061**.

**[REFUTES the search's own signal]** The **0.167 boatlee rate was 6-seed noise** — the
**E37/E39/E42 pattern**, this time inside the fitness function itself, produced at gen 25 and
**caught by the gate**. A per-opponent rate at 6 seeds cannot resolve 17%.

**Solo (63000:63020):** candidate **$132,700** > incumbent **$122,558**; boatlee **$150,792**.

**Verdict: S2's verify criteria all pass; S3's gate FAILS.** S3 needs **≥60% vs boatlee** (got **0%**)
and **solo ≥ boatlee's** (got **−$18k**). S3's kill criterion applies as written: **the genome is
mis-parameterised, not the optimiser** — the optimiser found a strictly better point in the space it
was given, and the space tops out ~$18k of solo production short. Iteration **in flight**:
**per-quadrant crop rows**, **more cohort slots**, **per-product sell floors**, then **ONE re-gate on
54040:54080** (the last reserved block).

Candidate vec: `results/s2_prod/state.json` -> `['best']['vec']` (gen 25), copy at
`/private/tmp/cand_vec.json`. Ledger `search/accept_ledger.json` created, first sub-block spent.
**Nothing promoted; `agent/plan.py` defaults unchanged.**

## E77 — Round 2 in the widened space: 0/80 again; the gain turned from production into denial; pivot to Track O

**Search.** The E76 widening built the space it prescribed: **80 genes** (54 -> 80) — **10 cohort
slots**, **8 per-product sell floors**, **per-quadrant row operators**. Dual-basin seeding
(`boatlee_like` + the migrated R1 candidate). **67 generations**, stopped at plateau (best unchanged
**gens 38 -> 66**), **5,351 evaluations**. Best score **0.6866**, weighted pool win **0.667**;
per-opponent at 6 seeds: **boatlee 0.167 (2/12)** — **byte-for-byte the same signal E76 refuted.**

**The genome is a third genuine point** — it differs from **both** seeds in herd *and* cohorts, so
the space was demonstrably explored, not re-found: **13 pastures**, **4 sheep + 8 cows** (starts
2/1), **7 cohorts** including **two brand-new slots** (c5 melon quad2 6t d18; **c6 TOMATO quad2 9t
d21 — the first tomato in any plan we have produced**); slots 7-10 unused. Strawberry NE 25t d10
`replant=1`, `hands_start` 0 / cap 11. **All eight per-product floors non-zero** (wheat .593, carrot
.585, tomato .477, strawberry .093, melon .751, egg .368, milk .015, wool .926). `release_pressure`
70 -> **4** (harder than R1's 16), `frontrun_lead` **15**. `projected_pricing` **OFF for the third
time**.

**Acceptance gate (54040:54080 — the LAST reserved sub-block, now spent).** **ACCEPT 98.8% (79/80)**,
Wilson **[93.3%, 99.8%]**, margin **+$124,680**, counters clean (thirst 2.23, steps/useful 0.722,
fallbacks 0). **The margin is the E76 wallet-cliff artifact again, harder**: at `release_pressure` 4
the incumbent earned **< $1,000 in 73/80 games**, median **$2**, strawberry/plant 0.40, milk/cow-day
0.061. Read it as fragility, not strength — same diagnosis as E76, same mechanism, deeper.
**The acceptance reserve is now EXHAUSTED** — `accept.remaining() == []`. Any further acceptance run
requires **a logged decision to extend the reserve**, not a quiet new block.

**Decisive run (64000:64040; freshness verified — max search seeds 60407/60401, overlap empty).**
Three-way, same-seed paired vs `boatlee`, 80 games both seats:

| agent | win vs boatlee | own money | boatlee money | margin |
|---|---:|---:|---:|---:|
| **R2 candidate** | **0.0%** [0, 4.6] | $60,327 | $120,536 | **−$60,209** |
| R1 candidate (E76) | 0.0% | $59,595 | $126,691 | −$67,096 |
| incumbent | 0.0% | $48,798 | $130,987 | −$82,189 |

**The gap is monotone**: −82.2k -> −67.1k (**+15.1k**) -> −60.2k (**+6.9k**). **E76's +$15,061
reproduces on an independent block** — that result stands. R2 counters vs boatlee: thirst 2.21,
steps/useful 0.718, `shed_overflow_discarded` **0.38** against the incumbent's **46.6** — the
overflow leak is genuinely fixed.

**[SETTLED] `projected_pricing` is inert on this plan shape.** ON vs OFF, 80 paired games vs
boatlee: **$79 apart** (−$60,130 vs −$60,209). So the launch-report **+$15k single-seed probe was
noise** AND **E72's −$2,675 does not reproduce here**. That is the **third scope-expiry of this one
gene's measurements** — stop measuring it on this substrate; it is a null.

**[The mechanism changed, and not for the better.] Solo (64000:64020, 40 games):** boatlee
**$159,188**; R1 **$129,142**; incumbent **$122,786**; **R2 $119,799 — the worst of the three.**
R2's head-to-head improvement is **denial, not production**: it gives up **~$9.3k of its own output**
to take **~$6.2k off boatlee**. Denial works on our wallet-cliff incumbent and does **nothing** to
boatlee, who has no cliff. R1 bought its gap-closing with production; R2 bought its with sabotage —
a lever that is already saturated against the only opponent that matters.

**Verdict: the user bar (beat boatlee locally, win > 0 at 80 games) is NOT MET.** S3 fails both
criteria again: **0% vs the ≥60% bar**, and solo **$119,799 vs boatlee's $150,792 bar** — *worse*
than R1's −$30.0k shortfall.

**Kill criterion, re-read.** E76 invoked "the genome is mis-parameterised, not the optimiser" and
prescribed one widening. **That reading no longer fits.** The widened space was demonstrably
explored — two new cohort slots used, a crop never before planted, all eight floors live, a herd and
cohort structure distinct from both seeds — and it still tops out **$30-39k of solo production
short**. The deficit is **structural to the choreography**, not to the parameterisation of it. S3 is
closed.

**[LESSON, third occurrence] The fitness's 6-seed boatlee sub-evaluation is a noise generator.**
It produced the **identical 0.167** signal in both rounds; the E37/E39/E42 pattern has now cost two
full search rounds' worth of misdirected selection pressure. **Any future genome round requires
≥40 boatlee seeds inside the fitness function**, plus a **logged reserve extension** — the cheap
sub-evaluation is not a smaller version of the gate, it is a different measurement.

**Decision: pivot to Track O** — **O2** (opponent fingerprint), **O3** (front-run / counter-mix /
slot alignment), **O4** (gate). O is the remaining lever that changes the **shape of the matchup**
rather than the numbers of the plan, and the boatlee gap is now demonstrably matchup-shaped.

Artifacts: R2 vec at `results/s2_r2/state.json -> ['best']['vec']` (gen 38), copy
`/private/tmp/r2_cand_vec.json`; raw decisive results `/private/tmp/decisive_all.json`.
**Nothing promoted.**
