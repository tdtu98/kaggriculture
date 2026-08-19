"""Screen and confirm isolated live-market overlays against baseline3k."""

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
    "_ENABLE_DEMAND_DEFERRAL",
    "_ENABLE_ADAPTIVE_FRONT_RUN",
)
VARIANTS = {
    "control": (False, False),
    "demand_defer": (True, False),
    "adaptive_front_run": (False, True),
}
PROFILE_REPLAY = (
    DUY_ROOT.parent
    / "duy_explore"
    / "kaggriculture-episodes-2026-08-15"
    / "top-100"
    / "93232089.json"
)
PROFILE_TEAM = "カワシギ"
SCREEN_SEED_START = 0
SCREEN_SEED_COUNT = 10
CONFIRM_SEED_START = 50
CONFIRM_SEED_COUNT = 50
EPISODE_STEPS = 720


class EvaluationError(RuntimeError):
    """Raised when evaluation evidence cannot be trusted."""


def _flag_pattern(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)} = (True|False)$", re.MULTILINE)


def render_variant(source: str, flags: dict[str, bool]) -> str:
    """Replace each requested full-line Boolean feature flag exactly once."""
    unknown = set(flags) - set(FLAG_NAMES)
    if unknown:
        raise ValueError(f"unknown feature flags: {sorted(unknown)}")
    rendered = source
    for name, enabled in flags.items():
        if type(enabled) is not bool:
            raise ValueError(f"feature flag {name} must be bool")
        pattern = _flag_pattern(name)
        if len(pattern.findall(rendered)) != 1:
            raise ValueError(f"feature flag {name} must occur exactly once")
        rendered = pattern.sub(f"{name} = {enabled}", rendered, count=1)
    return rendered


def _variant_flags(name: str) -> dict[str, bool]:
    try:
        values = VARIANTS[name]
    except KeyError as exc:
        raise EvaluationError(f"unknown variant: {name}") from exc
    return dict(zip(FLAG_NAMES, values))


def promotion_failures(summary: dict, expected_games: int = 100) -> list[str]:
    """Return binding promotion failures in stable order."""
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


def latency_failures(profile: dict) -> list[str]:
    """Return strict decision-latency failures."""
    failures = []
    if profile["mean_ms"] >= 1.0:
        failures.append("mean_latency_not_below_1_ms")
    if profile["p95_ms"] >= 2.0:
        failures.append("p95_latency_not_below_2_ms")
    return failures


def select_screen_winner(variants: dict) -> str | None:
    """Select a positive non-control policy by mean, median, then name."""
    qualified = []
    for name, evidence in variants.items():
        if name == "control":
            continue
        margin = evidence["summary"]["paired_seeds"]["margin"]
        if margin["mean"] > 0:
            qualified.append((name, margin["mean"], margin["median"]))
    if not qualified:
        return None
    return sorted(qualified, key=lambda row: (-row[1], -row[2], row[0]))[0][0]


def control_pair_failures(results: list[dict]) -> list[int]:
    """Return seeds whose two seat-swapped control margins do not cancel."""
    return [
        int(row["seed"])
        for row in benchmark.build_paired_rows(results)
        if not math.isclose(float(row["paired_margin"]), 0.0, abs_tol=1e-9)
    ]


def build_variant_metadata(
    candidate: benchmark.AgentRef,
    baseline: benchmark.AgentRef,
    source_candidate: benchmark.AgentRef,
    name: str,
    flags: dict[str, bool],
    seed_start: int,
    seed_count: int,
) -> dict:
    """Build deterministic benchmark metadata without temporary paths."""
    metadata = benchmark.build_metadata(
        candidate,
        baseline,
        seed_start,
        seed_count,
        EPISODE_STEPS,
    )
    metadata.pop("created_at_utc")
    metadata["creation_marker"] = "deterministic-market-overlay-v1"
    metadata["agent_a"] = {
        "identifier": f"generated-market-overlay:{name}@sha256:{candidate.sha256}",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _load_candidate_once(candidate_path: Path):
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        f"market_profile_{digest[:16]}", candidate_path
    )
    if spec is None or spec.loader is None:
        raise EvaluationError(f"unable to import candidate: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    started = time.perf_counter_ns()
    spec.loader.exec_module(module)
    import_ns = time.perf_counter_ns() - started
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise EvaluationError(f"agent function not found: {candidate_path}")
    return agent, import_ns


def _nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def profile_candidate(candidate_path: Path, replay_path: Path = PROFILE_REPLAY) -> dict:
    """Profile one call for every target observation in a compatible replay."""
    replay = json.loads(Path(replay_path).read_text())
    if replay.get("module_version") != "1.32.7":
        raise EvaluationError("profile replay must use module version 1.32.7")
    names = replay.get("info", {}).get("TeamNames", [])
    matches = [index for index, name in enumerate(names) if name == PROFILE_TEAM]
    if len(matches) != 1:
        raise EvaluationError("profile replay must contain target team once")
    seat = matches[0]
    observations = [states[seat]["observation"] for states in replay["steps"]]
    if len(observations) != EPISODE_STEPS:
        raise EvaluationError("profile replay must contain 720 observations")
    agent, import_ns = _load_candidate_once(Path(candidate_path))
    durations = []
    for observation in observations:
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


def _resolve_sources(args):
    source_candidate = benchmark.resolve_agent(str(args.candidate))
    baseline = benchmark.resolve_agent(str(args.baseline))
    if source_candidate.resolved_path is None:
        raise EvaluationError("candidate must be a Python file")
    return source_candidate, baseline, source_candidate.resolved_path.read_text()


def _run_variant(
    *,
    name: str,
    source: str,
    source_candidate: benchmark.AgentRef,
    baseline: benchmark.AgentRef,
    seed_start: int,
    seed_count: int,
    output_dir: Path,
    temp_root: Path,
) -> dict:
    flags = _variant_flags(name)
    rendered = render_variant(source, flags)
    isolated_dir = temp_root / name
    isolated_dir.mkdir()
    isolated = isolated_dir / "main.py"
    isolated.write_text(rendered)
    candidate = benchmark.resolve_agent(str(isolated))
    results = benchmark.run_suite(
        candidate,
        baseline,
        benchmark.build_schedule(seed_start, seed_count),
        steps=EPISODE_STEPS,
        progress=benchmark.print_progress,
    )
    summary = benchmark.summarize(results)
    metadata = build_variant_metadata(
        candidate,
        baseline,
        source_candidate,
        name,
        flags,
        seed_start,
        seed_count,
    )
    benchmark.write_artifacts(output_dir, metadata, results, summary)
    (output_dir / "agent.py").write_text(rendered)
    evidence = {
        "flags": flags,
        "hashes": {
            "candidate": candidate.sha256,
            "source_candidate": source_candidate.sha256,
            "baseline": baseline.sha256,
        },
        "summary": summary,
    }
    if name == "control":
        evidence["control_pair_failures"] = control_pair_failures(results)
    return evidence


def _run_screen(args) -> int:
    if args.screening_json is not None:
        raise EvaluationError("screen phase does not accept --screening-json")
    if args.output_dir.exists():
        raise EvaluationError(f"output directory already exists: {args.output_dir}")
    source_candidate, baseline, source = _resolve_sources(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    variants = {}
    with tempfile.TemporaryDirectory(prefix="market-overlay-screen-") as temp:
        temp_root = Path(temp)
        for name in VARIANTS:
            variants[name] = _run_variant(
                name=name,
                source=source,
                source_candidate=source_candidate,
                baseline=baseline,
                seed_start=SCREEN_SEED_START,
                seed_count=SCREEN_SEED_COUNT,
                output_dir=args.output_dir / name,
                temp_root=temp_root,
            )
    failed_control_seeds = variants["control"].get("control_pair_failures", [])
    if failed_control_seeds:
        raise EvaluationError(
            f"control paired margins did not cancel: {failed_control_seeds}"
        )
    winner_name = select_screen_winner(variants)
    winner = None
    if winner_name is not None:
        evidence = variants[winner_name]
        winner = {
            "name": winner_name,
            "flags": evidence["flags"],
            "candidate_sha256": evidence["hashes"]["candidate"],
        }
    screening = {
        "phase": "screen",
        "protocol": {
            "seed_start": SCREEN_SEED_START,
            "seed_count": SCREEN_SEED_COUNT,
            "games_per_variant": SCREEN_SEED_COUNT * 2,
            "steps": EPISODE_STEPS,
        },
        "source_candidate_sha256": source_candidate.sha256,
        "baseline_sha256": baseline.sha256,
        "variants": variants,
        "winner": winner,
    }
    _write_json(args.output_dir / "screening.json", screening)
    print(f"Results: {args.output_dir}")
    return 0 if winner is not None else 1


def _load_frozen_screen(path: Path) -> dict:
    screening = json.loads(path.read_text())
    if screening.get("phase") != "screen" or not screening.get("winner"):
        raise EvaluationError("screening record has no frozen winner")
    return screening


def _run_confirm(args) -> int:
    if args.screening_json is None:
        raise EvaluationError("confirm phase requires --screening-json")
    if args.output_dir.exists():
        raise EvaluationError(f"output directory already exists: {args.output_dir}")
    screening = _load_frozen_screen(args.screening_json)
    source_candidate, baseline, source = _resolve_sources(args)
    if source_candidate.sha256 != screening["source_candidate_sha256"]:
        raise EvaluationError("candidate source changed after screening")
    if baseline.sha256 != screening["baseline_sha256"]:
        raise EvaluationError("baseline changed after screening")
    winner = screening["winner"]
    name = winner["name"]
    flags = _variant_flags(name)
    if flags != winner["flags"]:
        raise EvaluationError("frozen winner flags do not match evaluator")
    rendered = render_variant(source, flags)
    if _sha256_text(rendered) != winner["candidate_sha256"]:
        raise EvaluationError("frozen winner digest does not match rendered source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="market-overlay-confirm-") as temp:
        evidence = _run_variant(
            name=name,
            source=source,
            source_candidate=source_candidate,
            baseline=baseline,
            seed_start=CONFIRM_SEED_START,
            seed_count=CONFIRM_SEED_COUNT,
            output_dir=args.output_dir / "benchmark",
            temp_root=Path(temp),
        )
        confirmed = args.output_dir / "confirmed_candidate.py"
        confirmed.write_text(rendered)
        if hashlib.sha256(confirmed.read_bytes()).hexdigest() != winner["candidate_sha256"]:
            raise EvaluationError("confirmed candidate write changed frozen bytes")
        profile = profile_candidate(confirmed)
    failures = promotion_failures(evidence["summary"])
    failures.extend(latency_failures(profile))
    promotion = {
        "phase": "confirm",
        "promoted": not failures,
        "variant": name,
        "flags": flags,
        "candidate_sha256": winner["candidate_sha256"],
        "source_candidate_sha256": source_candidate.sha256,
        "baseline_sha256": baseline.sha256,
        "protocol": {
            "seed_start": CONFIRM_SEED_START,
            "seed_count": CONFIRM_SEED_COUNT,
            "games": CONFIRM_SEED_COUNT * 2,
            "steps": EPISODE_STEPS,
        },
        "summary": evidence["summary"],
        "latency": profile,
        "failures": failures,
    }
    _write_json(args.output_dir / "promotion.json", promotion)
    print(f"Results: {args.output_dir}")
    return 0 if not failures else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen or confirm baseline3k live-market overlays."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--screening-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.phase == "screen":
            return _run_screen(args)
        return _run_confirm(args)
    except (
        EvaluationError,
        ValueError,
        OSError,
        KeyError,
        benchmark.BenchmarkError,
    ) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
