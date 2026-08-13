# Kaggriculture Agent Summary

## Current Best

| Version | Status | Agent | Strategy |
|---|---|---|---|
| `00_baseline` | **Current best** | [`main.py`](another_work/00_baseline/main.py) | Boatlee V16-RC2 Market Relay |

`00_baseline` is the only qualified local version so far. It is proven against
`demo_agent.py`, but has not yet faced another competitive version.

## Version Comparison

| Version | Strategy change | Opponent | Record | Average money | Average margin | Minimum money | Status |
|---|---|---|---:|---:|---:|---:|---|
| `00_baseline` | Initial Market Relay baseline | `demo_agent` | 100-0-0 | $151,986.04 | +$148,617.10 | $79,225.00 | **Current best** |

Add one row after testing each new version with the same 50 seeds, both seats,
and 720-turn games. A new version becomes the current best only after beating
the current best in a 100-game head-to-head benchmark.

## Versions

### `00_baseline`

#### Strategy

- Open aggressively with farm hands, cows, sheep, wheat, and melons.
- Add strawberries and unlock two more land quadrants.
- Hire a large daily workforce to run crops, animals, fertilizer collection,
  and inventory logistics in parallel.
- Earn from wheat, melons, strawberries, milk, wool, and fertilizer.
- Repair route-blocking weeds and adapt livestock for the Yarn Store.
- Time high-impact sales and relay fertilizer early against near-mirror farms.
- Finish with wheat-heavy production and final inventory liquidation.

Its strength is very high production throughput. Its limitation is that most
field actions follow a prerecorded route rather than a general planner. See the
[full strategy](another_work/00_baseline/STRATEGY.md).

#### Performance

Benchmark: `00_baseline` versus `demo_agent.py`, seeds `0–49`, both seats, 720
turns per game.

| Metric | Result |
|---|---:|
| Games | 100 |
| Record | **100 wins / 0 losses / 0 ties** |
| Win rate | **100.00%** |
| Average money | **$151,986.04** |
| Median money | $154,871.00 |
| Minimum money | $79,225.00 |
| Maximum money | $192,508.00 |
| Average demo money | $3,368.94 |
| Average margin | **+$148,617.10** |
| Seat 0 average margin | +$148,422.00 |
| Seat 1 average margin | +$148,812.20 |

[Summary](benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/summary.txt)
· [All 100 games](benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/games.csv)
· [Full metrics](benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/summary.json)
