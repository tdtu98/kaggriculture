# 01_baseline3k vs 00_baseline: Benchmark Findings

## Conclusion

`01_baseline3k` should replace `00_baseline` as the working baseline.

The main improvement is the reconstructed production route: it keeps livestock
alive and completes more useful feed, care, collection, and harvest work. The
one-turn premium market lead is a smaller, matchup-dependent improvement that
is most valuable against an opponent with a similar production and sale
calendar.

## Artifacts and Protocol

- Candidate: [`main.py`](main.py), extracted byte-for-byte from the notebook's
  `%%writefile main.py` cell
- Source notebook:
  [`v16-rc5-high-score-8c-4s-premium-market-lead.ipynb`](v16-rc5-high-score-8c-4s-premium-market-lead.ipynb)
- Candidate SHA-256:
  `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`
- Reference: `../00_baseline/main.py`, SHA-256
  `3c9b6e75d1bb9cc1f23b6bf5d8821c84193d1306d5bcb74ada1628359e3fb025`
- Environment: `kaggle-environments==1.32.6`, Python 3.12.9
- Configuration: 720 steps, 24 turns/day, and
  `townCenterSellInterval=24` in the installed environment
- Canonical schedule: seeds 0 through 49, with both seat orders for every seed

The full machine-readable result is in
[`summary.json`](../../benchmarks/results/20260813T055449Z_01_baseline3k_vs_00_baseline/summary.json),
and every game is recorded in
[`games.csv`](../../benchmarks/results/20260813T055449Z_01_baseline3k_vs_00_baseline/games.csv).

## Canonical Head-to-Head Result

| Metric | `01_baseline3k` | `00_baseline` |
|---|---:|---:|
| Wins | **100** | 0 |
| Losses | 0 | **100** |
| Mean final money | **92,296.01** | 75,123.93 |
| Median final money | **92,061.00** | 75,017.00 |
| Minimum final money | **47,996.00** | 38,455.00 |
| Maximum final money | **152,551.00** | 130,477.00 |

`01_baseline3k` had a mean margin of **+17,172.08**, a median margin of
**+17,068.50**, and a 22.9% advantage in mean final money. Its smallest margin
was still **+8,860**, and all 50 paired seed results were positive.

Seat order did not explain the result:

- Candidate in seat 0: 50/50 wins, mean margin **+17,151.62**
- Candidate in seat 1: 50/50 wins, mean margin **+17,192.54**

## Strategy Differences

### Production route

Both raw routes plant the same crop totals:

| Crop | Both routes |
|---|---:|
| Wheat | 143 |
| Strawberry | 37 |
| Melon | 19 |

Their coordination and livestock plans differ:

| Scheduled feature | `01_baseline3k` | `00_baseline` |
|---|---:|---:|
| Cows purchased | 8 | 9 |
| Sheep purchased | 4 | 4 |
| Hires | 264 | 262 |
| Wheat bought for feed | 189 | 212 |
| `CARE` actions | 967 | 285 |
| `PASS` actions | 324 | 994 |
| Scheduled sale units | 1,679 | 1,463 |

`01_baseline3k` uses the fixed 8-cow/4-sheep route reconstructed from three
public high-ranking replays. It replaces many idle actions with animal care
and uses a different movement and maintenance schedule even though its crop
targets are unchanged.

`00_baseline` nominally targets 9 cows and 4 sheep. It can switch two cows to
sheep when the Yarn Store is favorable, producing a 7-cow/6-sheep branch. It
also has price-gated wool release and same-turn sale-impact ordering.

### Market timing

`01_baseline3k` examines next-turn sales of `MELON`, `MILK`, `STRAWBERRY`, and
`WOOL`. When the current turn has no matching town demand and enough stock is
available, it sells part of the batch one turn early and removes exactly that
quantity from the original next-turn order. The two-turn intended quantity is
unchanged.

`00_baseline` instead has a three-turn fertilizer relay. It activates only
when the public farms remain near-mirrors at steps 216, 240, and 264. Once the
production routes diverge, this controller usually provides no advantage.

Both agents have narrow weed recovery for a weed blocking a scheduled plant
or pasture build. Neither is a general observation-driven planner.

## Controlled Ablations

Two additional benchmarks used seeds 0 through 11 in both seats, for 24 games
per comparison. The no-lead version imported the exact `01_baseline3k` agent
and disabled only `_FR_ITEMS`; its production route was unchanged.

| Comparison | Record | Mean margin |
|---|---:|---:|
| `01` production core without premium lead vs `00` | 24-0 | **+16,141.46** |
| Full `01` vs identical `01` core without premium lead | 24-0 | **+1,917.50** |
| Full `01` vs `00` on the same 12 seeds | 24-0 | **+16,333.63** |

These margins are not additive because both players affect the shared market.
Against `00`, enabling the premium lead increased the mean margin by only
about **192** coins on this panel. Against the otherwise identical production
core, it created a much larger **+1,917.50** competitive margin. The lead is
therefore a strong close-match or mirror-match tie-breaker, while the
production route supplies the broad improvement.

## Live Seed-0 Trace

The following is a single-game diagnostic, not an aggregate over all 100
games. It counted field actions whose live pre-action observation satisfied
the relevant success conditions.

| Work completed or ready to complete | `01_baseline3k` | `00_baseline` |
|---|---:|---:|
| Valid feed actions | **283** | 225 |
| Valid care actions | **301** | 222 |
| Valid fertilizer collections | **287** | 231 |
| Valid harvests | **383** | 362 |
| Visible units on harvested tiles | **1,244** | 1,124 |
| Surviving animals at the finish | **12/12** | 10/13 |
| Final money | **59,495** | 47,357 |

The candidate kept all 8 cows and 4 sheep alive. The reference finished with
7 cows, 3 sheep, and 3 empty pastures. Because livestock produces milk, wool,
and fertilizer throughout the season, avoiding animal loss has a compounding
effect.

In this game, the premium controller moved 227 scheduled premium units one
turn earlier without changing total planned volume. The reference's
fertilizer relay did not activate because its final near-mirror checkpoint
failed.

## What Helps Most

1. **Production and route coordination.** This is the dominant, general gain.
   The core alone retained a roughly +16k margin over `00` in the ablation.
2. **Animal survival and useful maintenance.** More successful feed, care,
   collection, and harvest actions compound into additional premium products
   and fertilizer.
3. **Premium market lead.** It consistently breaks close or mirror matchups,
   but contributes less when the opponent follows a different sale calendar.
4. **Weed recovery.** It protects critical scripted setup actions, although it
   does not repair arbitrary route drift.

The conditional near-mirror fertilizer relay in `00_baseline` is useful only
in a narrow matchup and should not be treated as the source of a general
performance improvement.

## Recommended Next Improvements

Use `01_baseline3k` as the base and keep the premium lead, then make its field
maintenance more state-aware:

- Prioritize feed and animal survival from the live observation.
- Issue `CARE` only when the animal has not been cared for and a future payout
  remains in the season.
- Stop late care that cannot pay back. The seed-0 candidate ended with seven
  pending care bonuses.
- Reuse freed duplicate/no-op action slots for harvest, fertilizer collection,
  inventory movement, or route recovery.
- Extend recovery beyond weeds to detect missed purchases, placements, feed,
  and other route drift.
- Consider combining the premium lead with `00_baseline`'s same-turn sale
  ranking, but benchmark that interaction before adopting it.

The high raw `CARE` count is not itself the goal: in the seed-0 trace only 301
of 967 submitted care actions had useful preconditions. The valuable lesson is
better livestock maintenance and route alignment, not blindly issuing more
care commands.
