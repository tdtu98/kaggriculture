"""Benchmark two Kaggriculture agents under a deterministic paired-seat protocol."""

from __future__ import annotations

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
from typing import Callable


BUILT_INS = frozenset({"pass", "random", "starter"})
_IMPORT_COUNTER = itertools.count()
CSV_FIELDS = (
    "seed",
    "agent_a_seat",
    "agent_b_seat",
    "seat_0_agent",
    "seat_1_agent",
    "seat_0_money",
    "seat_1_money",
    "seat_0_reward",
    "seat_1_reward",
    "seat_0_status",
    "seat_1_status",
    "agent_a_money",
    "agent_b_money",
    "margin",
    "outcome",
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce trustworthy results."""


@dataclass(frozen=True)
class AgentRef:
    """A validated file agent or Kaggriculture built-in agent."""

    identifier: str
    label: str
    resolved_path: Path | None
    sha256: str | None

    def runner_value(self) -> Callable | str:
        if self.resolved_path is None:
            return self.identifier
        return load_file_agent(self)

    def metadata(self) -> dict:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "resolved_path": str(self.resolved_path) if self.resolved_path else None,
            "sha256": self.sha256,
            "builtin": self.resolved_path is None,
        }


@dataclass(frozen=True)
class MatchSpec:
    """One seed and one seat assignment from agent A's perspective."""

    seed: int
    agent_a_seat: int


def load_file_agent(agent_ref: AgentRef) -> Callable:
    """Freshly import and return a file agent's public callable."""
    assert agent_ref.resolved_path is not None
    assert agent_ref.sha256 is not None
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


def resolve_agent(identifier: str, cwd: str | Path | None = None) -> AgentRef:
    """Validate an agent identifier and capture reproducibility metadata."""
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
    agent_ref = AgentRef(identifier, label, path, digest)
    load_file_agent(agent_ref)
    return agent_ref


def build_schedule(seed_start: int, seed_count: int) -> list[MatchSpec]:
    """Build contiguous seed-major pairs with agent A in each seat."""
    if seed_start < 0:
        raise BenchmarkError("seed start must be non-negative")
    if seed_count <= 0:
        raise BenchmarkError("seed count must be positive")
    return [
        MatchSpec(seed=seed, agent_a_seat=seat)
        for seed in range(seed_start, seed_start + seed_count)
        for seat in (0, 1)
    ]


def run_match(
    agent_a: AgentRef,
    agent_b: AgentRef,
    match: MatchSpec,
    steps: int = 720,
    make_environment=None,
) -> dict:
    """Run one fresh episode and normalize its result to agent A."""
    result, _environment = _execute_match(
        agent_a,
        agent_b,
        match,
        steps=steps,
        make_environment=make_environment,
    )
    return result


def _execute_match(
    agent_a: AgentRef,
    agent_b: AgentRef,
    match: MatchSpec,
    steps: int = 720,
    make_environment=None,
):
    if steps <= 0:
        raise BenchmarkError("steps must be positive")
    if match.agent_a_seat not in (0, 1):
        raise BenchmarkError("agent A seat must be 0 or 1")
    if make_environment is None:
        from kaggle_environments import make as make_environment

    seats = [agent_a, agent_b] if match.agent_a_seat == 0 else [agent_b, agent_a]
    environment = make_environment(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": match.seed},
        debug=False,
    )
    environment.run([agent_ref.runner_value() for agent_ref in seats])
    final = environment.steps[-1]
    statuses = [state.status for state in final]
    if statuses != ["DONE", "DONE"]:
        raise BenchmarkError(
            f"seed {match.seed} agent-A-seat {match.agent_a_seat} "
            f"statuses: {statuses}"
        )

    seat_money = [
        float(final[seat].observation["farms"][seat]["money"])
        for seat in (0, 1)
    ]
    rewards = [float(final[seat].reward) for seat in (0, 1)]
    if any(
        not math.isclose(rewards[seat], seat_money[seat])
        for seat in (0, 1)
    ):
        raise BenchmarkError(
            f"seed {match.seed} reward/money mismatch: "
            f"{rewards} != {seat_money}"
        )

    agent_a_seat = match.agent_a_seat
    agent_b_seat = 1 - agent_a_seat
    agent_a_money = seat_money[agent_a_seat]
    agent_b_money = seat_money[agent_b_seat]
    result = {
        "seed": match.seed,
        "agent_a_seat": agent_a_seat,
        "agent_b_seat": agent_b_seat,
        "seat_0_agent": seats[0].label,
        "seat_1_agent": seats[1].label,
        "seat_0_money": seat_money[0],
        "seat_1_money": seat_money[1],
        "seat_0_reward": rewards[0],
        "seat_1_reward": rewards[1],
        "seat_0_status": statuses[0],
        "seat_1_status": statuses[1],
        "agent_a_money": agent_a_money,
        "agent_b_money": agent_b_money,
        "margin": agent_a_money - agent_b_money,
        "outcome": (
            "win"
            if agent_a_money > agent_b_money
            else "loss"
            if agent_a_money < agent_b_money
            else "tie"
        ),
    }
    return result, environment


def generate_replay(
    agent_a: AgentRef,
    agent_b: AgentRef,
    seed: int,
    agent_a_seat: int,
    output_path: str | Path,
    steps: int = 720,
    make_environment=None,
) -> dict:
    """Rerun one seed/seat matchup and save its replay JSON on demand."""
    result, environment = _execute_match(
        agent_a,
        agent_b,
        MatchSpec(seed=seed, agent_a_seat=agent_a_seat),
        steps=steps,
        make_environment=make_environment,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(environment.toJSON(), stream)
    return result


def run_suite(
    agent_a: AgentRef,
    agent_b: AgentRef,
    schedule: list[MatchSpec],
    steps: int = 720,
    make_environment=None,
    progress=None,
) -> list[dict]:
    """Run a schedule sequentially and return normalized game records."""
    results = []
    total = len(schedule)
    for index, match in enumerate(schedule, start=1):
        result = run_match(
            agent_a,
            agent_b,
            match,
            steps=steps,
            make_environment=make_environment,
        )
        results.append(result)
        if progress is not None:
            progress(index, total, result)
    return results


def _stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summary_slice(results: list[dict]) -> dict:
    if not results:
        raise BenchmarkError("cannot summarize an empty result set")
    wins = sum(result["outcome"] == "win" for result in results)
    losses = sum(result["outcome"] == "loss" for result in results)
    ties = sum(result["outcome"] == "tie" for result in results)
    return {
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(results),
        "agent_a_money": _stats(
            [result["agent_a_money"] for result in results]
        ),
        "agent_b_money": _stats(
            [result["agent_b_money"] for result in results]
        ),
        "margin": _stats([result["margin"] for result in results]),
    }


def summarize(results: list[dict]) -> dict:
    """Summarize all games and each of agent A's seat assignments."""
    summary = _summary_slice(results)
    summary["by_agent_a_seat"] = {
        str(seat): _summary_slice(
            [result for result in results if result["agent_a_seat"] == seat]
        )
        for seat in (0, 1)
    }
    return summary


def build_metadata(
    agent_a: AgentRef,
    agent_b: AgentRef,
    seed_start: int,
    seed_count: int,
    steps: int,
) -> dict:
    """Capture the complete protocol and software identity for one run."""
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
        "kaggle_environments_version": importlib.metadata.version(
            "kaggle-environments"
        ),
    }


def format_summary(metadata: dict, summary: dict) -> str:
    """Render a compact overall and per-seat benchmark summary."""
    agent_a_label = metadata["agent_a"]["label"]
    agent_b_label = metadata["agent_b"]["label"]
    agent_a_money = summary["agent_a_money"]
    agent_b_money = summary["agent_b_money"]
    margin = summary["margin"]
    seat_zero = summary["by_agent_a_seat"]["0"]
    seat_one = summary["by_agent_a_seat"]["1"]
    seed_end = metadata["seed_start"] + metadata["seed_count"] - 1
    return "\n".join(
        [
            f"Agent A: {agent_a_label}",
            f"Agent B: {agent_b_label}",
            f"Seeds: {metadata['seed_start']}..{seed_end}",
            f"Games: {summary['games']}",
            (
                f"Outcomes: {summary['wins']} wins / "
                f"{summary['losses']} losses / {summary['ties']} ties"
            ),
            f"Win rate: {summary['win_rate']:.2%}",
            (
                "Agent A money: "
                f"mean={agent_a_money['mean']:.2f} "
                f"median={agent_a_money['median']:.2f} "
                f"min={agent_a_money['minimum']:.2f} "
                f"max={agent_a_money['maximum']:.2f}"
            ),
            (
                "Agent B money: "
                f"mean={agent_b_money['mean']:.2f} "
                f"median={agent_b_money['median']:.2f} "
                f"min={agent_b_money['minimum']:.2f} "
                f"max={agent_b_money['maximum']:.2f}"
            ),
            (
                "Margin: "
                f"mean={margin['mean']:.2f} "
                f"median={margin['median']:.2f} "
                f"min={margin['minimum']:.2f} "
                f"max={margin['maximum']:.2f}"
            ),
            (
                "Agent A in seat 0: "
                f"games={seat_zero['games']} "
                f"win_rate={seat_zero['win_rate']:.2%} "
                f"mean_margin={seat_zero['margin']['mean']:.2f}"
            ),
            (
                "Agent A in seat 1: "
                f"games={seat_one['games']} "
                f"win_rate={seat_one['win_rate']:.2%} "
                f"mean_margin={seat_one['margin']['mean']:.2f}"
            ),
            "",
        ]
    )


def write_artifacts(
    output: Path,
    metadata: dict,
    results: list[dict],
    summary: dict,
) -> None:
    """Write per-game data and human/machine-readable summaries."""
    output.mkdir(parents=True, exist_ok=False)
    with (output / "games.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    (output / "summary.json").write_text(
        json.dumps(
            {"metadata": metadata, "summary": summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "summary.txt").write_text(format_summary(metadata, summary))


def safe_label(label: str) -> str:
    """Make a stable, filesystem-safe label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    return cleaned or "agent"


def print_progress(index: int, total: int, result: dict) -> None:
    """Print one deterministic progress line."""
    print(
        f"[{index:>3}/{total}] seed={result['seed']} "
        f"A-seat={result['agent_a_seat']} {result['outcome']} "
        f"A=${result['agent_a_money']:.0f} "
        f"B=${result['agent_b_money']:.0f} "
        f"margin={result['margin']:+.0f}",
        flush=True,
    )


def same_agent(agent_a: AgentRef, agent_b: AgentRef) -> bool:
    """Return whether two identifiers select the same agent implementation."""
    if agent_a.resolved_path is not None or agent_b.resolved_path is not None:
        return (
            agent_a.resolved_path is not None
            and agent_a.resolved_path == agent_b.resolved_path
        )
    return agent_a.identifier == agent_b.identifier


def main(argv: list[str] | None = None) -> int:
    """Run the two-agent benchmark command."""
    parser = argparse.ArgumentParser(
        description="Benchmark two Kaggriculture agents."
    )
    parser.add_argument("agent_a")
    parser.add_argument("agent_b")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results")
    )
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
            print(
                "WARNING: built-in random actions are not controlled by "
                "the environment seed",
                file=sys.stderr,
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir / (
            f"{timestamp}_{safe_label(agent_a.label)}_vs_"
            f"{safe_label(agent_b.label)}"
        )
        results = run_suite(
            agent_a,
            agent_b,
            schedule,
            steps=args.steps,
            progress=print_progress,
        )
        summary = summarize(results)
        metadata = build_metadata(
            agent_a,
            agent_b,
            args.seed_start,
            args.seed_count,
            args.steps,
        )
        write_artifacts(run_dir, metadata, results, summary)
        print(format_summary(metadata, summary), end="")
        print(f"Results: {run_dir}")
        return 0
    except BenchmarkError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
