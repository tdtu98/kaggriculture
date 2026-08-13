# Reproducible Two-Agent Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a central runner that selects any two Kaggriculture agents, plays 50 contiguous seeds in both seats, saves reproducible artifacts, and produces the initial `00_baseline` versus `demo_agent` result.

**Architecture:** Keep protocol and orchestration in `benchmarks/benchmark.py`, with small dataclasses and pure helpers for agent resolution, schedules, normalization, and statistics. Each episode creates a fresh environment and freshly imports file agents; output is normalized to agent A's perspective and saved as CSV, JSON, and text.

**Tech Stack:** Python 3.12 standard library (`argparse`, `csv`, `dataclasses`, `hashlib`, `importlib`, `json`, `statistics`, `unittest`) plus installed `kaggle-environments` 1.32.6.

## Global Constraints

- Default to the contiguous seeds `0` through `49` via `--seed-start 0 --seed-count 50`.
- Play every seed twice, agent A in seat 0 and then agent A in seat 1, for 100 games.
- Default to 720 episode steps; only smoke tests may use fewer turns.
- Run games sequentially and create a fresh environment for every game.
- Freshly import file agents for every game; do not share module globals across episodes.
- Accept Python agent files and built-ins `starter`, `random`, and `pass`; warn that `random` is not action-deterministic.
- Use final observed bank money as the canonical score and require terminal reward to match it.
- Require both player statuses to be `DONE`; abort the aggregate on a failed match.
- Report overall and per-seat results from agent A's perspective.
- Preserve all existing user changes under `another_work/` and do not stage them in benchmark code commits.

---

## File Structure

- Create `benchmarks/__init__.py`: marks the benchmark directory as an importable package.
- Create `benchmarks/benchmark.py`: protocol, match execution, statistics, artifacts, and CLI.
- Create `benchmarks/test_benchmark.py`: isolated unit tests and one short environment integration test.
- Create `benchmarks/README.md`: stable usage and interpretation notes for future versions.
- Generate `benchmarks/results/<timestamp>_<agent-a>_vs_<agent-b>/`: canonical initial CSV/JSON/text result.

---

### Task 1: Agent resolution and paired-seat schedule

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/benchmark.py`
- Create: `benchmarks/test_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkError`, `AgentRef`, `MatchSpec`, `resolve_agent(identifier, cwd=None)`, `load_file_agent(agent_ref)`, and `build_schedule(seed_start, seed_count)`.
- `AgentRef.runner_value()` returns a freshly imported callable for a file agent or a built-in string.
- `build_schedule()` returns seed-major `MatchSpec` values with agent A in seat 0 before seat 1.

- [ ] **Step 1: Write failing resolution and schedule tests**

```python
import tempfile
import unittest
from pathlib import Path

from benchmarks import benchmark


class ResolutionAndScheduleTests(unittest.TestCase):
    def test_resolves_file_agent_with_hash_and_fresh_callable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_one.py"
            path.write_text("def agent(obs):\n    return {'farmer': ['PASS']}\n")

            ref = benchmark.resolve_agent(str(path))

            self.assertEqual(ref.label, "agent_one")
            self.assertEqual(ref.resolved_path, path.resolve())
            self.assertEqual(len(ref.sha256), 64)
            self.assertIsNot(ref.runner_value(), ref.runner_value())

    def test_accepts_supported_builtin(self):
        ref = benchmark.resolve_agent("starter")
        self.assertEqual(ref.runner_value(), "starter")
        self.assertIsNone(ref.resolved_path)

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "not found"):
            benchmark.resolve_agent("missing-agent.py")

    def test_builds_contiguous_seed_major_seat_pairs(self):
        self.assertEqual(
            benchmark.build_schedule(7, 2),
            [
                benchmark.MatchSpec(seed=7, agent_a_seat=0),
                benchmark.MatchSpec(seed=7, agent_a_seat=1),
                benchmark.MatchSpec(seed=8, agent_a_seat=0),
                benchmark.MatchSpec(seed=8, agent_a_seat=1),
            ],
        )

    def test_rejects_invalid_seed_ranges(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "non-negative"):
            benchmark.build_schedule(-1, 50)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "positive"):
            benchmark.build_schedule(0, 0)
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.ResolutionAndScheduleTests`

Expected: `ImportError` because `benchmarks/benchmark.py` does not exist.

- [ ] **Step 3: Implement agent references and schedules**

```python
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import platform
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class BenchmarkError(RuntimeError):
    pass


BUILT_INS = frozenset({"pass", "random", "starter"})
_IMPORT_COUNTER = itertools.count()


@dataclass(frozen=True)
class AgentRef:
    identifier: str
    label: str
    resolved_path: Path | None
    sha256: str | None

    def runner_value(self):
        if self.resolved_path is None:
            return self.identifier
        return load_file_agent(self)

    def metadata(self):
        return {
            "identifier": self.identifier,
            "label": self.label,
            "resolved_path": str(self.resolved_path) if self.resolved_path else None,
            "sha256": self.sha256,
            "builtin": self.resolved_path is None,
        }


@dataclass(frozen=True)
class MatchSpec:
    seed: int
    agent_a_seat: int


def load_file_agent(agent_ref):
    module_name = f"benchmark_agent_{next(_IMPORT_COUNTER)}_{agent_ref.sha256[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, agent_ref.resolved_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"unable to load agent: {agent_ref.identifier}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "agent", None)
    if not callable(candidate):
        raise BenchmarkError(f"agent function not found: {agent_ref.identifier}")
    return candidate


def resolve_agent(identifier, cwd=None):
    if identifier in BUILT_INS:
        return AgentRef(identifier, identifier, None, None)
    base = Path.cwd() if cwd is None else Path(cwd)
    path = Path(identifier)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise BenchmarkError(f"agent file not found: {identifier}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    label = path.parent.name if path.name == "main.py" else path.stem
    ref = AgentRef(identifier, label, path, digest)
    load_file_agent(ref)
    return ref


def build_schedule(seed_start, seed_count):
    if seed_start < 0:
        raise BenchmarkError("seed start must be non-negative")
    if seed_count <= 0:
        raise BenchmarkError("seed count must be positive")
    return [
        MatchSpec(seed=seed, agent_a_seat=seat)
        for seed in range(seed_start, seed_start + seed_count)
        for seat in (0, 1)
    ]
```

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.ResolutionAndScheduleTests`

Expected: five tests pass.

- [ ] **Step 5: Commit Task 1 without staging user files**

```bash
git add benchmarks/__init__.py benchmarks/benchmark.py benchmarks/test_benchmark.py
git commit -m "feat: add deterministic benchmark schedule"
```

---

### Task 2: Isolated match execution and seat normalization

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `benchmarks/test_benchmark.py`

**Interfaces:**
- Consumes: `AgentRef`, `MatchSpec`.
- Produces: `run_match(agent_a, agent_b, match, steps=720, make_environment=None) -> dict` and `run_suite(agent_a, agent_b, schedule, steps=720, make_environment=None, progress=None) -> list[dict]`.
- Each result is normalized to agent A and contains `seed`, `agent_a_seat`, both monetary scores, rewards, statuses, `outcome`, and `margin`.

- [ ] **Step 1: Write failing execution tests with a fake environment**

```python
from types import SimpleNamespace


class MatchExecutionTests(unittest.TestCase):
    @staticmethod
    def fake_make(name, configuration, debug):
        class FakeEnvironment:
            def run(self, runners):
                self.runners = runners
                self.steps = [[
                    SimpleNamespace(
                        status="DONE",
                        reward=4100.0,
                        observation={"farms": [{"money": 4100.0}, {"money": 3600.0}]},
                    ),
                    SimpleNamespace(
                        status="DONE",
                        reward=3600.0,
                        observation={"farms": [{"money": 4100.0}, {"money": 3600.0}]},
                    ),
                ]]
        self = FakeEnvironment()
        self.configuration = configuration
        return self

    def test_normalizes_agent_a_from_seat_zero(self):
        a = benchmark.resolve_agent("pass")
        b = benchmark.resolve_agent("starter")
        result = benchmark.run_match(
            a, b, benchmark.MatchSpec(12, 0), make_environment=self.fake_make
        )
        self.assertEqual(result["agent_a_money"], 4100.0)
        self.assertEqual(result["agent_b_money"], 3600.0)
        self.assertEqual(result["outcome"], "win")
        self.assertEqual(result["margin"], 500.0)

    def test_normalizes_agent_a_from_seat_one(self):
        a = benchmark.resolve_agent("pass")
        b = benchmark.resolve_agent("starter")
        result = benchmark.run_match(
            a, b, benchmark.MatchSpec(12, 1), make_environment=self.fake_make
        )
        self.assertEqual(result["agent_a_money"], 3600.0)
        self.assertEqual(result["agent_b_money"], 4100.0)
        self.assertEqual(result["outcome"], "loss")
        self.assertEqual(result["margin"], -500.0)

    def test_rejects_non_done_status(self):
        def failed_make(name, configuration, debug):
            environment = self.fake_make(name, configuration, debug)
            environment.steps[-1][1].status = "ERROR"
            return environment
        with self.assertRaisesRegex(benchmark.BenchmarkError, "statuses"):
            benchmark.run_match(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                benchmark.MatchSpec(2, 0),
                make_environment=failed_make,
            )

    def test_rejects_reward_money_mismatch(self):
        def mismatch_make(name, configuration, debug):
            environment = self.fake_make(name, configuration, debug)
            environment.steps[-1][0].reward = 1.0
            return environment
        with self.assertRaisesRegex(benchmark.BenchmarkError, "reward"):
            benchmark.run_match(
                benchmark.resolve_agent("pass"),
                benchmark.resolve_agent("starter"),
                benchmark.MatchSpec(2, 0),
                make_environment=mismatch_make,
            )
```

- [ ] **Step 2: Run execution tests to verify RED**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.MatchExecutionTests`

Expected: four errors because `run_match` is not defined.

- [ ] **Step 3: Implement isolated execution and normalization**

```python
def run_match(agent_a, agent_b, match, steps=720, make_environment=None):
    if steps <= 0:
        raise BenchmarkError("steps must be positive")
    if make_environment is None:
        from kaggle_environments import make as make_environment
    seats = [agent_a, agent_b] if match.agent_a_seat == 0 else [agent_b, agent_a]
    environment = make_environment(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": match.seed},
        debug=False,
    )
    environment.run([ref.runner_value() for ref in seats])
    final = environment.steps[-1]
    statuses = [state.status for state in final]
    if statuses != ["DONE", "DONE"]:
        raise BenchmarkError(f"seed {match.seed} seat {match.agent_a_seat} statuses: {statuses}")
    seat_money = [float(final[seat].observation["farms"][seat]["money"]) for seat in (0, 1)]
    rewards = [float(final[seat].reward) for seat in (0, 1)]
    if any(not math.isclose(rewards[seat], seat_money[seat]) for seat in (0, 1)):
        raise BenchmarkError(
            f"seed {match.seed} reward/money mismatch: {rewards} != {seat_money}"
        )
    a_seat = match.agent_a_seat
    b_seat = 1 - a_seat
    a_money, b_money = seat_money[a_seat], seat_money[b_seat]
    return {
        "seed": match.seed,
        "agent_a_seat": a_seat,
        "agent_b_seat": b_seat,
        "seat_0_agent": seats[0].label,
        "seat_1_agent": seats[1].label,
        "seat_0_money": seat_money[0],
        "seat_1_money": seat_money[1],
        "seat_0_reward": rewards[0],
        "seat_1_reward": rewards[1],
        "seat_0_status": statuses[0],
        "seat_1_status": statuses[1],
        "agent_a_money": a_money,
        "agent_b_money": b_money,
        "margin": a_money - b_money,
        "outcome": "win" if a_money > b_money else "loss" if a_money < b_money else "tie",
    }


def run_suite(agent_a, agent_b, schedule, steps=720, make_environment=None, progress=None):
    results = []
    total = len(schedule)
    for index, match in enumerate(schedule, start=1):
        result = run_match(agent_a, agent_b, match, steps, make_environment)
        results.append(result)
        if progress is not None:
            progress(index, total, result)
    return results
```

- [ ] **Step 4: Run execution tests to verify GREEN**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.MatchExecutionTests`

Expected: four tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add benchmarks/benchmark.py benchmarks/test_benchmark.py
git commit -m "feat: execute paired benchmark matches"
```

---

### Task 3: Overall and per-seat statistics

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `benchmarks/test_benchmark.py`

**Interfaces:**
- Consumes: normalized result dictionaries from `run_match`.
- Produces: `summarize(results) -> dict` with overall and `by_agent_a_seat` summaries.

- [ ] **Step 1: Write a failing summary test**

```python
class SummaryTests(unittest.TestCase):
    def test_summarizes_overall_and_each_agent_a_seat(self):
        results = [
            {"agent_a_seat": 0, "agent_a_money": 5000.0, "agent_b_money": 4000.0, "margin": 1000.0, "outcome": "win"},
            {"agent_a_seat": 1, "agent_a_money": 3000.0, "agent_b_money": 3500.0, "margin": -500.0, "outcome": "loss"},
            {"agent_a_seat": 0, "agent_a_money": 4200.0, "agent_b_money": 4200.0, "margin": 0.0, "outcome": "tie"},
        ]
        summary = benchmark.summarize(results)
        self.assertEqual(summary["games"], 3)
        self.assertEqual((summary["wins"], summary["losses"], summary["ties"]), (1, 1, 1))
        self.assertAlmostEqual(summary["win_rate"], 1 / 3)
        self.assertEqual(summary["agent_a_money"]["median"], 4200.0)
        self.assertEqual(summary["margin"]["mean"], 500.0 / 3)
        self.assertEqual(summary["by_agent_a_seat"]["0"]["games"], 2)
        self.assertEqual(summary["by_agent_a_seat"]["1"]["losses"], 1)
```

- [ ] **Step 2: Run the summary test to verify RED**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.SummaryTests`

Expected: error because `summarize` is not defined.

- [ ] **Step 3: Implement statistics**

```python
def _stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summary_slice(results):
    if not results:
        raise BenchmarkError("cannot summarize an empty result set")
    wins = sum(row["outcome"] == "win" for row in results)
    losses = sum(row["outcome"] == "loss" for row in results)
    ties = sum(row["outcome"] == "tie" for row in results)
    return {
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(results),
        "agent_a_money": _stats([row["agent_a_money"] for row in results]),
        "agent_b_money": _stats([row["agent_b_money"] for row in results]),
        "margin": _stats([row["margin"] for row in results]),
    }


def summarize(results):
    summary = _summary_slice(results)
    summary["by_agent_a_seat"] = {
        str(seat): _summary_slice([row for row in results if row["agent_a_seat"] == seat])
        for seat in (0, 1)
    }
    return summary
```

- [ ] **Step 4: Run all unit tests to verify GREEN**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark`

Expected: all resolution, execution, and summary tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add benchmarks/benchmark.py benchmarks/test_benchmark.py
git commit -m "feat: summarize benchmark seat effects"
```

---

### Task 4: Reproducible artifacts and two-agent CLI

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `benchmarks/test_benchmark.py`
- Create: `benchmarks/README.md`

**Interfaces:**
- Consumes: `AgentRef`, schedule, normalized results, and summary.
- Produces: `build_metadata(...)`, `write_artifacts(...)`, `format_summary(...)`, and `main(argv=None)`.
- CLI: `benchmark.py AGENT_A AGENT_B [--seed-start N] [--seed-count N] [--steps N] [--output-dir PATH]`.

- [ ] **Step 1: Write failing artifact tests**

```python
import csv
import json


class ArtifactTests(unittest.TestCase):
    def test_writes_csv_json_and_text_with_protocol_metadata(self):
        results = [
            {
                "seed": 4, "agent_a_seat": 0, "agent_b_seat": 1,
                "seat_0_agent": "pass", "seat_1_agent": "starter",
                "seat_0_money": 4000.0, "seat_1_money": 3500.0,
                "seat_0_reward": 4000.0, "seat_1_reward": 3500.0,
                "seat_0_status": "DONE", "seat_1_status": "DONE",
                "agent_a_money": 4000.0, "agent_b_money": 3500.0,
                "margin": 500.0, "outcome": "win",
            },
            {
                "seed": 4, "agent_a_seat": 1, "agent_b_seat": 0,
                "seat_0_agent": "starter", "seat_1_agent": "pass",
                "seat_0_money": 3500.0, "seat_1_money": 4000.0,
                "seat_0_reward": 3500.0, "seat_1_reward": 4000.0,
                "seat_0_status": "DONE", "seat_1_status": "DONE",
                "agent_a_money": 4000.0, "agent_b_money": 3500.0,
                "margin": 500.0, "outcome": "win",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = benchmark.build_metadata(
                benchmark.resolve_agent("pass"), benchmark.resolve_agent("starter"),
                seed_start=4, seed_count=1, steps=720,
            )
            benchmark.write_artifacts(output, metadata, results, benchmark.summarize(results))
            payload = json.loads((output / "summary.json").read_text())
            rows = list(csv.DictReader((output / "games.csv").open()))
            self.assertEqual(payload["metadata"]["seeds"], [4])
            self.assertEqual(payload["summary"]["games"], 2)
            self.assertEqual(len(rows), 2)
            self.assertIn("Agent A in seat 0", (output / "summary.txt").read_text())
```

- [ ] **Step 2: Run the artifact test to verify RED**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark.ArtifactTests`

Expected: errors because metadata and artifact helpers are undefined.

- [ ] **Step 3: Implement metadata, artifacts, CLI parsing, progress, and failures**

Implementation requirements:

```python
CSV_FIELDS = (
    "seed", "agent_a_seat", "agent_b_seat", "seat_0_agent", "seat_1_agent",
    "seat_0_money", "seat_1_money", "seat_0_reward", "seat_1_reward",
    "seat_0_status", "seat_1_status", "agent_a_money", "agent_b_money",
    "margin", "outcome",
)


def build_metadata(agent_a, agent_b, seed_start, seed_count, steps):
    return {
        "protocol_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "seeds": list(range(seed_start, seed_start + seed_count)),
        "games": seed_count * 2,
        "steps": steps,
        "agent_a": agent_a.metadata(),
        "agent_b": agent_b.metadata(),
        "python_version": platform.python_version(),
        "kaggle_environments_version": importlib.metadata.version("kaggle-environments"),
    }


def write_artifacts(output, metadata, results, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "games.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    (output / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.txt").write_text(format_summary(metadata, summary))


def safe_label(label):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    return cleaned or "agent"


def print_progress(index, total, result):
    print(
        f"[{index:>3}/{total}] seed={result['seed']} "
        f"A-seat={result['agent_a_seat']} {result['outcome']} "
        f"A=${result['agent_a_money']:.0f} B=${result['agent_b_money']:.0f} "
        f"margin={result['margin']:+.0f}",
        flush=True,
    )


def format_summary(metadata, summary):
    a_label = metadata["agent_a"]["label"]
    b_label = metadata["agent_b"]["label"]
    a_money = summary["agent_a_money"]
    b_money = summary["agent_b_money"]
    margin = summary["margin"]
    seat_zero = summary["by_agent_a_seat"]["0"]
    seat_one = summary["by_agent_a_seat"]["1"]
    return "\n".join([
        f"Agent A: {a_label}",
        f"Agent B: {b_label}",
        f"Seeds: {metadata['seed_start']}..{metadata['seed_start'] + metadata['seed_count'] - 1}",
        f"Games: {summary['games']}",
        f"Outcomes: {summary['wins']} wins / {summary['losses']} losses / {summary['ties']} ties",
        f"Win rate: {summary['win_rate']:.2%}",
        f"Agent A money: mean={a_money['mean']:.2f} median={a_money['median']:.2f} min={a_money['minimum']:.2f} max={a_money['maximum']:.2f}",
        f"Agent B money: mean={b_money['mean']:.2f} median={b_money['median']:.2f} min={b_money['minimum']:.2f} max={b_money['maximum']:.2f}",
        f"Margin: mean={margin['mean']:.2f} median={margin['median']:.2f} min={margin['minimum']:.2f} max={margin['maximum']:.2f}",
        f"Agent A in seat 0: games={seat_zero['games']} win_rate={seat_zero['win_rate']:.2%} mean_margin={seat_zero['margin']['mean']:.2f}",
        f"Agent A in seat 1: games={seat_one['games']} win_rate={seat_one['win_rate']:.2%} mean_margin={seat_one['margin']['mean']:.2f}",
        "",
    ])


def same_agent(agent_a, agent_b):
    if agent_a.resolved_path is not None or agent_b.resolved_path is not None:
        return agent_a.resolved_path is not None and agent_a.resolved_path == agent_b.resolved_path
    return agent_a.identifier == agent_b.identifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark two Kaggriculture agents.")
    parser.add_argument("agent_a")
    parser.add_argument("agent_b")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args(argv)
    try:
        agent_a = resolve_agent(args.agent_a)
        agent_b = resolve_agent(args.agent_b)
        if same_agent(agent_a, agent_b):
            raise BenchmarkError("agent A and agent B must be different")
        schedule = build_schedule(args.seed_start, args.seed_count)
        if args.steps <= 0:
            raise BenchmarkError("steps must be positive")
        if "random" in (agent_a.identifier, agent_b.identifier):
            print("WARNING: built-in random actions are not controlled by the environment seed", file=sys.stderr)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir / f"{stamp}_{safe_label(agent_a.label)}_vs_{safe_label(agent_b.label)}"
        results = run_suite(agent_a, agent_b, schedule, args.steps, progress=print_progress)
        summary = summarize(results)
        metadata = build_metadata(agent_a, agent_b, args.seed_start, args.seed_count, args.steps)
        write_artifacts(run_dir, metadata, results, summary)
        print(format_summary(metadata, summary), end="")
        print(f"Results: {run_dir}")
        return 0
    except BenchmarkError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
```

`safe_label` permits ASCII letters, digits, dots, underscores, and hyphens and replaces other character runs with `-`. Match exceptions already include the seed and agent-A seat; `main` prints that context to stderr and returns non-zero without writing a misleading aggregate.

- [ ] **Step 4: Document the stable protocol**

Create `benchmarks/README.md` with the canonical command, the definition of 50 contiguous seeds and 100 paired-seat games, option descriptions, output-file meanings, the `random` warning, and this future-version example:

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/01_candidate/main.py \
  another_work/00_baseline/main.py \
  --seed-start 0 --seed-count 50 --steps 720
```

- [ ] **Step 5: Run all benchmark tests and compilation**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark`

Expected: all tests pass.

Run: `.venv/bin/python -m py_compile benchmarks/benchmark.py benchmarks/test_benchmark.py`

Expected: exit code 0 and no output.

- [ ] **Step 6: Commit Task 4**

```bash
git add benchmarks/benchmark.py benchmarks/test_benchmark.py benchmarks/README.md
git commit -m "feat: add reproducible benchmark artifacts"
```

---

### Task 5: Real-environment smoke test and full baseline benchmark

**Files:**
- Modify: `benchmarks/test_benchmark.py`
- Generate: `benchmarks/results/<timestamp>_00_baseline_vs_demo_agent/games.csv`
- Generate: `benchmarks/results/<timestamp>_00_baseline_vs_demo_agent/summary.json`
- Generate: `benchmarks/results/<timestamp>_00_baseline_vs_demo_agent/summary.txt`

**Interfaces:**
- Consumes: completed CLI and `another_work/00_baseline/main.py`, `demo_agent.py`.
- Produces: a verified 100-game canonical benchmark with exactly 50 games in each agent-A seat.

- [ ] **Step 1: Add a short installed-environment integration test**

```python
class EnvironmentIntegrationTests(unittest.TestCase):
    def test_pass_agents_finish_one_short_seed_in_both_seats(self):
        agent_a = benchmark.resolve_agent("pass")
        agent_b = benchmark.resolve_agent("starter")
        results = benchmark.run_suite(
            agent_a,
            agent_b,
            benchmark.build_schedule(0, 1),
            steps=2,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual({row["agent_a_seat"] for row in results}, {0, 1})
        self.assertTrue(all(row["seat_0_status"] == "DONE" for row in results))
        self.assertTrue(all(row["seat_1_status"] == "DONE" for row in results))
```

- [ ] **Step 2: Run the complete suite and smoke CLI**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark test_demo_agent.py test_observer_agent.py`

Expected: all tests pass.

Run:

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py demo_agent.py \
  --seed-start 0 --seed-count 1 --steps 24 \
  --output-dir /tmp/kaggriculture-benchmark-smoke
```

Expected: two `DONE` games, one for each seat, and CSV/JSON/text artifacts.

- [ ] **Step 3: Run the canonical 50-seed, 100-game benchmark**

```bash
.venv/bin/python benchmarks/benchmark.py \
  another_work/00_baseline/main.py \
  demo_agent.py \
  --seed-start 0 \
  --seed-count 50 \
  --steps 720 \
  --output-dir benchmarks/results
```

Expected: progress reaches `100/100`, all games finish with `DONE`, and the CLI prints the timestamped result directory.

- [ ] **Step 4: Verify the generated benchmark invariants**

Run:

```bash
.venv/bin/python -c "import csv,json,pathlib; roots=sorted(pathlib.Path('benchmarks/results').glob('*_00_baseline_vs_demo_agent')); root=roots[-1]; rows=list(csv.DictReader((root/'games.csv').open())); data=json.loads((root/'summary.json').read_text()); assert len(rows)==100; assert data['metadata']['seeds']==list(range(50)); assert data['summary']['games']==100; assert data['summary']['by_agent_a_seat']['0']['games']==50; assert data['summary']['by_agent_a_seat']['1']['games']==50; assert all(r['seat_0_status']=='DONE' and r['seat_1_status']=='DONE' for r in rows); print(root); print((root/'summary.txt').read_text())"
```

Expected: assertions pass, then the exact result directory and summary print.

- [ ] **Step 5: Run final regression verification**

Run: `.venv/bin/python -m unittest -v benchmarks.test_benchmark test_demo_agent.py test_observer_agent.py`

Expected: all tests pass.

Run: `.venv/bin/python -m py_compile benchmarks/benchmark.py benchmarks/test_benchmark.py another_work/00_baseline/main.py demo_agent.py`

Expected: exit code 0 and no output.

- [ ] **Step 6: Commit the integration test and canonical result without staging unrelated files**

```bash
git add benchmarks/test_benchmark.py benchmarks/results
git commit -m "test: record baseline demo benchmark"
```
