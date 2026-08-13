# Audit: why the Claude implementation loses to `another_work/00_baseline`

Date: 2026-08-13  
Runtime: Python 3.12.9, `kaggle-environments` 1.32.6, 720 turns

## Scope

I treated these as the requested design record:

- `tu/CLAUDE.md`
- `tu/PLAN2.md`
- `tu/TASKS2.md` (the repository's match for `task2.md`)
- `tu/docs/experiments.md` (the repository's match for `expermiment.md`)

I compared the shipped Claude agent, `tu/main.py` plus `tu/search/champion.json`, with
`duy/another_work/00_baseline/main.py`. I did not modify anything under `tu`.

## Verdict

The failure is not one bad crop coefficient. The shipped Claude policy is a reactive,
single-quadrant, fixed-herd policy whose executor cannot efficiently service a larger farm. The
baseline is a pre-optimized season plan with a compact route and closed-loop repair. Those are
different strategy classes.

Claude's current champion compounds that architectural disadvantage by always targeting 8 sheep
and 6 cows, even when the random shop draw has no Yarn Store. Its nominal demand adaptation does
not fix this because the task builder and the market purchaser use different animal targets. In
the worst seeds, repeated wheat buying and shed traffic consume cash while useful goods are also
discarded through shed overflow.

The causal chain is:

```text
one quadrant + fixed 14-animal target
             |
             +--> about 10 crop tiles --> far fewer harvests
             |
random shops +--> 8 sheep even with no wool shop --> feed/haul/cash drain

reactive task regeneration + local assignment
             |
             +--> excessive walking and shed trips
             +--> planting bursts exceed daily servicing capacity
             +--> unlocking more land makes the executor worse, not competitive
```

## Controlled benchmark

The canonical run used seeds 0–49 in both seat orders:

```bash
cd duy
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py ../tu/main.py \
  --seed-start 0 --seed-count 50 --steps 720 \
  --output-dir ../duy_audit/benchmark_results
```

| Metric | Baseline | Claude |
|---|---:|---:|
| Wins | 100 | 0 |
| Mean final money | $116,576.30 | $32,992.58 |
| Median final money | $115,848.00 | $26,998.50 |
| Minimum | $47,305 | $35 |
| Maximum | $174,731 | $83,637 |

The mean margin was $83,583.72. Results were identical when the seats were reversed: 50/50 wins
for the baseline from each seat. This rules out seat order and the old `obs["step"]` theory as the
cause of this comparison.

Canonical outputs are in
`benchmark_results/20260812T165954Z_00_baseline_vs_tu/`. Exact replays for representative seeds 0,
33, and 46 are in `traces/`.

## Findings

### 1. The shipped strategy imposes a hard production ceiling

`tu/search/champion.json` has `buy_land: false`, `cow_target: 6`, and `sheep_target: 8`. A single
unlocked quadrant has 25 tiles. Once the 14-animal herd is built, the agent normally maintains only
about 10 crops. The baseline unlocks three quadrants and operates about 62–63 crops plus 13 animals.

Representative replay totals were stable across seeds:

| Per-game workload | Baseline | Claude |
|---|---:|---:|
| Peak crop tiles | 62–63 | 15 initially, then about 10 |
| Peak animals | 13 | 14 |
| Harvest actions | 390 | 137–139 |
| Water actions | 1,010 | 224–227 |
| Plant actions | 199 | 44–49 |
| Unlocked quadrants | 3 | 1 |

This is the largest direct explanation for the score gap: Claude simply creates much less sellable
output.

### 2. Matching the baseline's farm dimensions does not fix the executor

I ran a focused configuration matching the documented mimic experiment: land buying enabled,
baseline-like crop proportions, eager watering, fertilizer, optimal assignment, 9 cows, and 4
sheep. Across seeds 0–9 it lost all 10 games:

| Configuration | Mean final money |
|---|---:|
| Baseline | $107,924.00 |
| Claude mimic | $16,982.10 |

On seed 0, the mimic reached 69 crops and 13 animals but finished at $27,784 versus $128,703. It
walked during 53.5% of unit turns, produced only 228 harvest actions, and allowed 33 crops to turn
into weeds. It planted 20 crops on day 14 and 26 on day 16, then continued planting on days 28 and
29. The larger layout therefore exposed an execution-capacity problem; it did not remove one.

The stock cap controls how many crop tiles exist, but it does not constrain new planting by the
number of plants that can actually be watered and harvested. The champion leaves both the daily
plant-rate guard and late-season planting stop disabled.

### 3. Claude spends much more labor on motion and hauling

Across the representative exact replays:

| Share of unit turns | Baseline | Claude |
|---|---:|---:|
| Movement | about 42.8% | 56.2–56.8% |
| Crop work | about 25.8% | 5.8–5.9% |
| Animal work | about 13.5% | 12.5–12.6% |
| Shed hauling | about 3.1% | 9.7–9.8% |
| Pass | about 15.0% | 15–16% |

The baseline encodes a season plan and repeatable routes, then repairs exceptional state such as
weeds. Claude rebuilds the current task set each turn and performs sticky nearest-cost assignment.
Even its `optimal` mode optimizes units against today's task locations; it does not optimize task
generation, planting cadence, layout, multi-stop routes, or future servicing load.

This agrees with the design record's own E28 observation: Claude needed 1.84 walking steps per
productive action versus 1.01 for boatlee. Later parameter search improved a local policy but did
not change this underlying execution model.

### 4. The forward model is not part of the shipped decision path

`tu/agent/forward.py` contains `FarmModel`, but there is no construction or use of `FarmModel` in
the shipped agent. `Engine.__call__` in `tu/agent/engine.py` does only three things: build current
tasks, assign units to those tasks, and emit current market orders.

So the implementation described as planning in `PLAN2.md` did produce simulator/parity machinery,
but the final policy does not use forward rollout to choose a season plan. The later exact-matching
work improved one-turn assignment only. This is why parameter optimization could converge while
still remaining in a weak, reactive policy class.

### 5. The fixed sheep policy is badly exposed to stochastic shop draws

Version 1.32.6 draws shops with replacement. Claude nevertheless targets 8 sheep in every game and
has `adaptive_mix` disabled in the champion.

In a 50-seed, one-seat replay sample using the same opponent and rules:

| Yarn Store copies drawn | Seeds | Claude mean final money |
|---:|---:|---:|
| 0 | 18 | $12,897.30 |
| 1 | 20 | $43,186.90 |
| 2 | 8 | $46,984.40 |
| 3 | 4 | $44,466.00 |

Claude money had a +0.520 correlation with Yarn Store count. On seven selected low/no-Yarn seeds,
the current policy averaged only $1,027. Removing sheep raised that subset to $20,832.60; removing
sheep and reducing the wheat reserve raised it to $22,998.40. Conversely, removing sheep reduced
the mean on five Yarn-rich/high-score seeds from $75,796.80 to $50,461.20.

That is causal evidence, not a recommendation to always remove sheep: sheep are valuable when wool
demand exists. The defect is the static target in a stochastic market.

### 6. `adaptive_mix` is incomplete across the task and market layers

When `adaptive_mix` is enabled, `build_tasks()` reduces local animal targets based on observed
demand (`tu/agent/engine.py`, around lines 300–316). But `market().buy_animals()` independently
uses the raw `Params.cow_target` and `Params.sheep_target` (around lines 756–782). It never uses the
demand-adjusted targets.

End-to-end confirmation on seed 33, which had no Yarn Store: the adaptive variant still bought and
placed all 8 sheep and finished at only $5,127. On the selected 12-seed stress sample, existing
`adaptive_mix` averaged $18,563.60 versus $33,809.60 for the much smaller change of lowering the
wheat reserve.

`tu/tests/test_demand_model.py` validates demand-count arithmetic, but it does not test the behavior
that matters: that a no-demand observation changes purchasing, placement, and the final herd. This
is an integration-test gap between two individually plausible components.

### 7. Feed accounting creates repeated purchases and shed churn

The market code computes required feed from all animals and empty structures, but counts only wheat
currently in the shed. It ignores wheat already carried by workers and wheat pickups already in
flight. The task builder does account for carried/in-flight wheat, so the two subsystems disagree.

On catastrophic seed 33, Claude submitted 275 `BUY_PRODUCT WHEAT` orders requesting 2,484 units in
total. Not every requested unit executes—the market can reject unaffordable orders—but the repeated
requests expose the loop. Money fell from $13,859 on day 15 to $9,235 on day 21, $3,879 on day 24,
$514 on day 27, and $35 at the finish while the agent continued supporting 14 animals and about 10
crops.

Claude also performed 416–447 pickups and 236–265 drops in representative games, compared with 135
pickups and 57 drops for the baseline.

### 8. The fallback can destroy inventory through shed overflow

When no task is assignable, `_fallback()` sends any carrying unit to the shed and emits a bare
`DROP`. The game accepts only the remaining shed capacity and discards overflow from the unit's
inventory. There is no capacity check or item selection before this fallback.

Replaying the unit actions and shed occupancy on seed 33 found 69 overflow events. They discarded at
least 158 wheat, 56 wool, 75 fertilizer, 42 milk, and smaller quantities of crops. The baseline had
no shed overflow in the same replay. Seed 0 had four smaller Claude overflow events; seed 46 had
none, which is consistent with overflow being an amplifier of the bad-shop failure rather than the
sole cause of all losses.

### 9. Two secondary correctness/process issues remain

- Hire budgeting uses a fixed estimated cost of `$4` for each `HIRE`, while the environment charges
  the Fibonacci sequence 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 for the first ten hires of a day. The
  internal cash projection can therefore approve later orders using money that is no longer
  available. This is not the main score gap, but it is another market-model mismatch.
- `tu/main.py` still says `obs["step"]` is absent for seat 1, while the later experiments in the same
  repository say E21 disproved that claim. The current engine does not use `step`, so this stale
  comment is documentation drift, not a benchmark cause.

## What did *not* cause the result

- **Seat bias:** reversing seats produced exactly the same scores and outcomes.
- **A single unlucky seed:** the baseline won all 100 controlled games over 50 seeds.
- **Only crop mix:** the land/crop/animal mimic still lost all 10 focused games and scored worse than
  the current champion on those seeds.
- **A missing simulator:** the parity/forward work is useful infrastructure, but the forward model is
  not wired into the live policy.
- **One isolated bug:** fixing sheep, wheat reserve, or assignment parameters individually changes
  subsets, but none bridges the production and routing gap.

## Why the documented optimization approach converged to the wrong place

The experiment log repeatedly shows small-sample reversals: E37 was corrected by E41, E42's scaling
result evaporated, E43 found melon was merely the best crop the current executor could keep alive,
and E44/E45 showed that baseline-like scale overloads the engine. Those later findings are the key
ones.

CEM and promotion gates tuned parameters around the current executor and opponent pool. They could
select a better point within that local basin, but could not invent the baseline's route, planting
schedule, inventory choreography, or capacity model. Several sensible safeguards remain disabled
because they hurt the already-tuned local champion, which is further evidence of coupling to that
local optimum.

## Recommended repair order

1. Add end-to-end tests that a no-Yarn observation changes sheep purchases and final herd size; make
   one demand-adjusted target object feed task generation, buying, structures, and placement.
2. Use a single inventory ledger for market and worker planning: shed + carried + reserved/in-flight,
   and prohibit `DROP` quantities beyond shed capacity.
3. Add a daily service-capacity budget. Admit new crops/animals only if the planned route can water,
   feed, care for, and harvest them before deadlines; stop late planting by expected payoff.
4. Replace independent per-turn nearest-task assignment with route/block scheduling across several
   turns. Preserve closed-loop repair for weeds, blocked actions, and stochastic state.
5. Make land and herd choices conditional on shop demand and remaining season horizon.
6. Only after those changes, retune crop mix and economic thresholds against the external baseline
   over many seeds and both seats.

Parameter tuning should be last. The benchmark evidence says the current limiting factor is the
policy/executor architecture, followed by demand and inventory integration bugs.
