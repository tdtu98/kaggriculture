"""Benchmark isolated controller variants and profile standalone candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path


DUY_ROOT = Path(__file__).resolve().parents[2]
if str(DUY_ROOT) not in sys.path:
    sys.path.insert(0, str(DUY_ROOT))

from benchmarks import benchmark


FLAG_NAMES = (
    "_ENABLE_FIELD_GUARDS",
    "_ENABLE_PURCHASE_RECOVERY",
    "_ENABLE_SALE_CAP",
    "_ENABLE_FRONT_RUN",
)
VARIANTS = {
    "route_only": (False, False, False, False),
    "field_guards": (True, False, False, False),
    "purchase_recovery": (True, True, False, False),
    "sale_cap": (True, False, True, False),
    "front_run": (True, False, False, True),
}


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot produce trustworthy evidence."""


def _flag_pattern(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)} = (True|False)$", re.MULTILINE)


def render_variant(source: str, flags: dict[str, bool]) -> str:
    """Replace requested feature constants, each present as one full line."""
    unknown = set(flags) - set(FLAG_NAMES)
    if unknown:
        raise ValueError(f"unknown feature flags: {sorted(unknown)}")

    rendered = source
    for name, enabled in flags.items():
        if type(enabled) is not bool:
            raise ValueError(f"feature flag {name} must be bool")
        pattern = _flag_pattern(name)
        matches = pattern.findall(rendered)
        if len(matches) != 1:
            raise ValueError(
                f"feature flag {name} must occur exactly once as a full line"
            )
        rendered = pattern.sub(f"{name} = {enabled}", rendered, count=1)
    return rendered


def _source_flags(source: str) -> dict[str, bool]:
    flags = {}
    for name in FLAG_NAMES:
        matches = _flag_pattern(name).findall(source)
        if len(matches) != 1:
            raise ValueError(
                f"feature flag {name} must occur exactly once as a full line"
            )
        flags[name] = matches[0] == "True"
    return flags


def _render_named_variant(name: str, source: str) -> tuple[dict[str, bool], str]:
    if name == "frozen":
        return _source_flags(source), source
    try:
        values = VARIANTS[name]
    except KeyError as exc:
        raise EvaluationError(f"unknown variant: {name}") from exc
    flags = dict(zip(FLAG_NAMES, values))
    return flags, render_variant(source, flags)


def promotion_failures(summary: dict, expected_games: int = 200) -> list[str]:
    """Return all promotion threshold failures in stable decision order."""
    failures = []
    if summary["games"] != expected_games:
        failures.append("unexpected_game_count")
    if summary["paired_seeds"]["margin"]["mean"] <= 0:
        failures.append("paired_mean_not_positive")
    if summary["paired_seeds"]["margin"]["median"] <= 0:
        failures.append("paired_median_not_positive")
    if summary["wins"] / summary["games"] <= 0.55:
        failures.append("win_rate_not_above_55_percent")
    if summary["by_agent_a_seat"]["0"]["margin"]["mean"] <= 0:
        failures.append("seat_zero_mean_not_positive")
    if summary["by_agent_a_seat"]["1"]["margin"]["mean"] <= 0:
        failures.append("seat_one_mean_not_positive")
    if summary["paired_seeds"]["bootstrap_mean_95ci"]["lower"] <= 0:
        failures.append("bootstrap_lower_not_positive")
    return failures


def _load_replay_support():
    path = Path(__file__).with_name("inspect_replays.py")
    spec = importlib.util.spec_from_file_location(
        "evaluation_inspect_replays", path
    )
    if spec is None or spec.loader is None:
        raise EvaluationError(f"unable to load replay support: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_candidate_once(candidate_path: Path):
    candidate_path = Path(candidate_path).resolve()
    if not candidate_path.is_file():
        raise EvaluationError(f"candidate file not found: {candidate_path}")
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"profile_candidate_{digest[:16]}", candidate_path
    )
    if spec is None or spec.loader is None:
        raise EvaluationError(f"unable to import candidate: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    started = time.perf_counter_ns()
    spec.loader.exec_module(module)
    finished = time.perf_counter_ns()
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise EvaluationError(f"agent function not found: {candidate_path}")
    return agent, finished - started


def _nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def profile_candidate(candidate_path, replay_path, team_name) -> dict:
    """Time one candidate call for every target observation in a replay."""
    agent, import_ns = _load_candidate_once(Path(candidate_path))
    replay_support = _load_replay_support()
    try:
        accepted, _rejected = replay_support.load_compatible_replays(
            [Path(replay_path)]
        )
        if len(accepted) != 1:
            raise EvaluationError("profile replay is not module-version compatible")
        _path, replay = accepted[0]
        seat = replay_support.find_seat(replay, team_name)
    except replay_support.ReplayError as exc:
        raise EvaluationError(str(exc)) from exc

    durations = []
    for states in replay["steps"]:
        observation = states[seat]["observation"]
        started = time.perf_counter_ns()
        agent(observation)
        durations.append(time.perf_counter_ns() - started)

    return {
        "import_ms": import_ns / 1_000_000,
        "calls": len(durations),
        "mean_ms": statistics.mean(durations) / 1_000_000,
        "p50_ms": statistics.median(durations) / 1_000_000,
        "p95_ms": _nearest_rank(durations, 0.95) / 1_000_000,
        "maximum_ms": max(durations) / 1_000_000,
    }


def _latency_failures(profile: dict) -> list[str]:
    failures = []
    if profile["mean_ms"] >= 1.0:
        failures.append("mean_latency_not_below_1_ms")
    if profile["p95_ms"] >= 2.0:
        failures.append("p95_latency_not_below_2_ms")
    return failures


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_variant_metadata(
    candidate: benchmark.AgentRef,
    baseline: benchmark.AgentRef,
    source_candidate: benchmark.AgentRef,
    name: str,
    flags: dict[str, bool],
    seed_start: int,
    seed_count: int,
) -> dict:
    metadata = benchmark.build_metadata(
        candidate,
        baseline,
        seed_start,
        seed_count,
        720,
    )
    metadata.pop("created_at_utc")
    metadata["creation_marker"] = "deterministic-generated-variant-v1"
    metadata["agent_a"] = {
        "identifier": f"generated-variant:{name}@sha256:{candidate.sha256}",
        "label": name,
        "resolved_path": None,
        "sha256": candidate.sha256,
        "builtin": False,
    }
    metadata["generated_variant"] = {
        "name": name,
        "flags": flags,
        "source_candidate": source_candidate.metadata(),
    }
    return metadata


def _run_profile(args) -> int:
    required = {
        "--profile-candidate": args.profile_candidate,
        "--profile-replay": args.profile_replay,
        "--profile-team-name": args.profile_team_name,
        "--profile-output": args.profile_output,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        raise EvaluationError(
            "profile mode requires " + ", ".join(sorted(missing))
        )
    profile = profile_candidate(
        args.profile_candidate,
        args.profile_replay,
        args.profile_team_name,
    )
    _write_json(args.profile_output, profile)
    print(json.dumps(profile, sort_keys=True))
    failures = _latency_failures(profile)
    if failures:
        print("Latency gate failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def _run_benchmarks(args) -> int:
    if args.candidate is None or args.baseline is None:
        raise EvaluationError("benchmark mode requires --candidate and --baseline")
    if not args.variants:
        raise EvaluationError("benchmark mode requires at least one --variant")
    if len(set(args.variants)) != len(args.variants):
        raise EvaluationError("variant names must not be repeated")
    if args.output_dir.exists():
        raise EvaluationError(f"output directory already exists: {args.output_dir}")

    source_candidate = benchmark.resolve_agent(str(args.candidate))
    assert source_candidate.resolved_path is not None
    source = source_candidate.resolved_path.read_text()
    baseline = benchmark.resolve_agent(str(args.baseline))
    schedule = benchmark.build_schedule(args.seed_start, args.seed_count)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ablations = {"variants": {}}
    gate_failed = False

    with tempfile.TemporaryDirectory(prefix="kaggriculture-variants-") as temp:
        temp_root = Path(temp)
        for name in args.variants:
            flags, rendered = _render_named_variant(name, source)
            isolated_dir = temp_root / name
            isolated_dir.mkdir()
            isolated = isolated_dir / "main.py"
            isolated.write_text(rendered)
            candidate = benchmark.resolve_agent(str(isolated))
            results = benchmark.run_suite(
                candidate,
                baseline,
                schedule,
                steps=720,
                progress=benchmark.print_progress,
            )
            summary = benchmark.summarize(results)
            metadata = _build_variant_metadata(
                candidate,
                baseline,
                source_candidate,
                name,
                flags,
                args.seed_start,
                args.seed_count,
            )
            benchmark.write_artifacts(
                args.output_dir / name,
                metadata,
                results,
                summary,
            )
            failures = promotion_failures(summary) if args.promotion_gate else []
            gate_failed = gate_failed or bool(failures)
            ablations["variants"][name] = {
                "flags": flags,
                "hashes": {
                    "candidate": candidate.sha256,
                    "baseline": baseline.sha256,
                },
                "summary": summary,
                "gate_failures": failures,
            }

    _write_json(args.output_dir / "ablations.json", ablations)
    print(f"Results: {args.output_dir}")
    return 1 if gate_failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark feature variants or profile one candidate."
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=(*VARIANTS, "frozen"),
    )
    parser.add_argument("--promotion-gate", action="store_true")
    parser.add_argument("--profile-candidate", type=Path)
    parser.add_argument("--profile-replay", type=Path)
    parser.add_argument("--profile-team-name")
    parser.add_argument("--profile-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmark mode or replay-derived latency profile mode."""
    args = _parser().parse_args(argv)
    try:
        if args.profile_candidate is not None:
            return _run_profile(args)
        if args.output_dir is None:
            raise EvaluationError("benchmark mode requires --output-dir")
        return _run_benchmarks(args)
    except (
        EvaluationError,
        ValueError,
        OSError,
        benchmark.BenchmarkError,
    ) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
