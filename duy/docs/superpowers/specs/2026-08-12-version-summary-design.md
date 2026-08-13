# Kaggriculture Version Summary Design

## Goal

Create `SUMMARY.md` at the repository root as the single, easy-to-find record
of the current best agent, its key strategy, its verified performance, and the
process for evaluating and promoting future versions.

## Audience and Scope

The summary is for developers iterating on Kaggriculture agents. It must answer
four questions without requiring them to inspect notebooks, source code, or raw
benchmark data:

1. Which version is currently best?
2. What strategy does it use?
3. What evidence supports its status?
4. How should a new version be tested and compared?

The document is a concise registry, not a replacement for detailed per-version
strategy analysis. It links to the longer strategy and raw benchmark artifacts
instead of duplicating them.

## Document Structure

### Current Best

Name `00_baseline` as the current champion and only qualified version. Record
the relative agent path and SHA-256 hash so the title is tied to the exact code
that produced the benchmark result. State explicitly that its verified result
establishes superiority over `demo_agent`, not global optimality.

### Version Comparison

Use one append-friendly Markdown table. Each version row records status,
strategy change, benchmark opponent, record, mean money, mean margin, minimum
money, and evidence. `00_baseline` begins as the sole qualified champion row.

Future results must append new rows; existing historical rows should not be
silently rewritten. If a version's code changes, it becomes a new version or a
new recorded hash/result.

### Versions

Keep all details for a version inside one self-contained subsection. Every
version uses the same shape:

```markdown
### `<version_name>`

#### Strategy

Concise bullets describing only this version's approach and changes.

#### Performance

The benchmark protocol, core metrics, and links to this version's evidence.
```

The initial `00_baseline` section summarizes:

- aggressive livestock, wheat, and melon opening;
- strawberry addition and two land expansions;
- large daily labor force for crop and animal throughput;
- mixed wheat, premium-crop, milk, wool, and fertilizer production;
- weed repair and Yarn Store adaptation;
- price-aware sale ordering, wool release thresholds, and near-mirror
  fertilizer relay;
- wheat-heavy late game and final liquidation.

Its performance subsection records:

- opponent: `demo_agent.py`;
- environment: `kaggle-environments` 1.32.6 and Python 3.12.9;
- 50 contiguous seeds, `0` through `49`;
- both seat orientations per seed, 100 games total;
- 720 turns per game;
- 100 wins, 0 losses, 0 ties;
- baseline mean/median/minimum/maximum money;
- demo mean money;
- mean margin and separate seat-0/seat-1 mean margins.

Link the `00_baseline` strategy to
`another_work/00_baseline/STRATEGY.md`. Link its performance to the saved
`summary.txt`, `summary.json`, and `games.csv` rather than copying all 100 rows.
Future versions append parallel `### <version>` sections rather than adding
strategy or performance to generic shared sections.

### Promotion Rule

The current champion is selected by a canonical head-to-head suite between a
candidate and the current champion using seeds `0` through `49`, both seats,
and 720 turns. Ranking criteria are applied in this order:

1. more head-to-head wins;
2. higher average head-to-head margin;
3. higher average final money;
4. higher minimum final money.

A candidate must complete all 100 games with `DONE` status. Testing against
`demo_agent` remains a regression/sanity benchmark, but does not replace the
champion head-to-head comparison.

### Testing Future Versions

Provide exact commands for:

1. candidate versus `demo_agent.py`;
2. candidate versus `another_work/00_baseline/main.py` or the current
   champion path;
3. viewing the generated summary and all game rows;
4. generating one replay on demand for a selected seed and seat.

Include a short checklist of fields to copy into the leaderboard after a run:
version path/hash, opponent, protocol, record, money statistics, margins,
per-seat figures, and artifact path.

## Source of Truth and Consistency

All numeric claims must come from
`benchmarks/results/20260812T012722Z_00_baseline_vs_demo_agent/summary.json`.
The exact 100 game records remain in the adjacent `games.csv`. The strategy
summary must be grounded in `another_work/00_baseline/STRATEGY.md`.

Relative links in `SUMMARY.md` are rooted at the repository root. The document
must not claim `00_baseline` is globally optimal; it is the current best among
locally qualified versions and is proven only against the opponents actually
recorded.

## Verification

Before delivery:

- confirm every linked local file exists;
- confirm all copied metrics exactly match `summary.json`;
- confirm the stated hash matches the benchmark metadata and current baseline
  file;
- confirm the benchmark command uses the stable runner defaults or explicitly
  supplies seeds `0` through `49`, both seats, and 720 turns;
- scan for placeholders, ambiguous champion language, and stale paths.
