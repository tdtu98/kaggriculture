# Kaggriculture Benchmarks

`benchmark.py` compares any two agents using a stable paired-seat protocol.
The default suite uses the 50 contiguous seeds `0` through `49`. Each seed is
played twice with the agents swapping seats, producing 100 full 720-turn games.

## Canonical benchmark

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py \
  demo_agent.py
```

The first positional argument is agent A, whose perspective is used for wins,
margins, and seat summaries. The second is agent B. Either argument may be a
Python file containing `agent(obs)` or a built-in name: `starter`, `random`, or
`pass`.

The built-in `random` agent uses action randomness that is not controlled by
the environment seed. It is accepted for convenience but is not suitable for a
strictly reproducible comparison.

## Options

```text
--seed-start INTEGER   first seed in the contiguous range (default: 0)
--seed-count INTEGER   number of seeds (default: 50)
--steps INTEGER        turns per game (default: 720)
--output-dir PATH      result root (default: benchmarks/results)
```

Keep `--seed-start 0 --seed-count 50 --steps 720` unchanged when comparing
versions. To run the next independent contiguous batch, use
`--seed-start 50 --seed-count 50`.

For example, compare a candidate with the baseline using the canonical suite:

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/01_candidate/main.py \
  another_work/00_baseline/main.py \
  --seed-start 0 --seed-count 50 --steps 720
```

## Results

Each invocation writes a timestamped directory under `benchmarks/results/`:

- `games.csv` contains every seed, seat assignment, final score, status,
  outcome, and agent-A margin.
- `summary.json` contains the exact protocol, agent paths and hashes,
  environment/Python versions, and overall/per-seat aggregates.
- `summary.txt` is a compact human-readable report.

Final bank money from the terminal observation is the canonical score. The
runner also checks that terminal reward matches that value and requires both
players to finish with `DONE`; otherwise it exits non-zero without producing an
aggregate.

## Generate one replay on demand

The benchmark does not store 100 large replay files. Rerun one recorded seed
and seat orientation only when it is useful:

```python
from pathlib import Path

from benchmarks.benchmark import generate_replay, resolve_agent

result = generate_replay(
    resolve_agent("another_work/00_baseline/main.py"),
    resolve_agent("demo_agent.py"),
    seed=15,
    agent_a_seat=1,
    output_path=Path("replays/seed-15-baseline-seat-1.json"),
    steps=720,
)
print(result)
```

`agent_a_seat=0` reproduces the row where agent A occupied seat 0;
`agent_a_seat=1` reproduces the swapped row. The function refuses to overwrite
an existing replay path.
