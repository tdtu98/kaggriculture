# Reproducible Two-Agent Benchmark Design

## Goal

Create one centralized benchmark runner that compares any two Kaggriculture
agents under an identical, repeatable protocol. Use it immediately to benchmark
`another_work/00_baseline/main.py` against `demo_agent.py`, and keep it stable so
future agent versions can be compared without copying or changing benchmark
logic.

## Benchmark Protocol

The default suite uses 50 contiguous seeds, starting at zero: `0` through `49`.
Each seed is played twice with the seats reversed:

1. agent A in seat 0 and agent B in seat 1;
2. agent B in seat 0 and agent A in seat 1.

This produces 100 games. Both orientations for a seed use the same environment
seed. Games run sequentially to keep execution and reporting deterministic.
The game rules are symmetric, but seat index can still affect a result: seeded
farm events consume randomness in player order, some atomic market operations
are iterated in player order, and agents can inspect `obs["player"]`. Pairing
and separately reporting both seats controls for and exposes those effects.

The seed range is controlled by `--seed-start` and `--seed-count`, defaulting to
`0` and `50`. A later independent batch can therefore use, for example,
`--seed-start 50 --seed-count 50` for the contiguous range `50` through `99`.
The runner rejects a negative start, a non-positive count, or duplicate agent
identifiers.

The default season length is 720 turns and is recorded in every output. A
`--steps` option exists for fast integration checks, but published comparisons
must use 720 turns.

## Command-Line Interface

The central entry point is `benchmarks/benchmark.py`. Its two positional
arguments select the agents:

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py \
  demo_agent.py
```

An agent identifier may be a Python file containing an `agent(obs)` function or
one of Kaggriculture's built-in agents: `starter`, `random`, or `pass`. Python
paths are resolved from the current working directory and validated before any
games start. Selecting the built-in `random` agent emits a reproducibility
warning because its own action randomness is not controlled by the environment
seed.

Useful options are:

```text
--seed-start INTEGER   first seed in the contiguous range (default: 0)
--seed-count INTEGER   number of seeds (default: 50)
--steps INTEGER        turns per game (default: 720)
--output-dir PATH      result root (default: benchmarks/results)
```

The CLI prints progress and a final human-readable summary. It returns a
non-zero exit status if validation fails or any match does not finish normally.

## Match Isolation and Scoring

Every match creates a fresh Kaggriculture environment. File agents are imported
under a fresh unique module name for that episode, and only their `agent`
function is supplied to `env.run`; built-in names are supplied directly. This
prevents module-level state from leaking between games without triggering a
file's command-line entry point.

The primary score is each player's final bank balance read from the final
observation. Kaggriculture currently sets terminal `reward` to that same bank
balance; the runner records it as a cross-check and fails clearly if the two
values differ. Reading the observation explicitly keeps the benchmark tied to
the documented win condition. A win, loss, or tie is derived by comparing final
bank balances.

The runner requires both final statuses to be `DONE`. If a match errors or ends
with another status, it records the failure context, stops the suite, and exits
non-zero rather than producing a misleading aggregate.

## Result Artifacts

Each invocation creates a separate result directory named with a UTC timestamp
and filesystem-safe agent labels. It contains:

- `games.csv`: one row per game with seed, orientation, seat assignment, final
  money, rewards, statuses, outcome from agent A's perspective, and margin;
- `summary.json`: protocol metadata and aggregate statistics;
- `summary.txt`: the same core aggregate in a compact human-readable form.

Metadata includes the exact contiguous seed list, seed start/count, game count,
episode length, both original agent identifiers, resolved file paths where
applicable, SHA-256 hashes for file agents, Python version, and installed
`kaggle-environments` version.

Aggregate statistics are always from agent A's perspective and include games,
wins, losses, ties, win rate, mean/median/minimum/maximum money for each agent,
mean/median/minimum/maximum margin, and separate summaries for agent A in seat 0
and seat 1.

## Components

`benchmarks/benchmark.py` owns CLI parsing and orchestration. Focused helpers
handle:

- agent identifier validation and metadata;
- construction of the fixed two-orientation match schedule;
- one isolated environment match and final-money extraction;
- normalization of every result into agent A's perspective;
- aggregate calculation;
- CSV, JSON, and text output.

`benchmarks/test_benchmark.py` tests these helpers without running 100 full
seasons. It uses a fake environment for schedule, seat normalization, status,
score extraction, summary, and metadata tests, plus a minimal real-environment
integration test when Kaggriculture is installed.

## Verification and Initial Run

Implementation follows test-driven development: helper and CLI tests fail
first, then the smallest runner implementation makes them pass. Verification
includes unit tests, Python compilation, and a two-game smoke run using one seed
in both orientations.

After verification, run the required full benchmark:

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py \
  demo_agent.py \
  --seed-start 0 \
  --seed-count 50 \
  --steps 720
```

The generated artifact directory is the canonical baseline result. Future
versions must use the same runner and the same `0` through `49` suite for direct
comparison.
