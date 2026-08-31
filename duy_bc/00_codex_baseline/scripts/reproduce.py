#!/usr/bin/env python3
"""Reproduce the complete prepared, trained, and frozen-test BC v0 run."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bc_core.evaluate import evaluate_frozen_run, verify_selection
from bc_core.paths import baseline_path
from bc_core.prepare import prepare
from scripts.train_v0 import train_run


def reproduce(
    config_path: Path,
    data_root: Path,
    runs_root: Path,
    run_id: str,
    *,
    prepare_fn: Callable[..., Any] = prepare,
    train_fn: Callable[..., Any] = train_run,
    verify_fn: Callable[..., Any] = verify_selection,
    evaluate_fn: Callable[..., dict[str, Any]] = evaluate_frozen_run,
) -> dict[str, Any]:
    """Run every authenticated stage, resuming only a published training prefix."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run-id must be one non-empty path component")

    prepare_fn(config_path, data_root)
    run_dir = runs_root / run_id
    resume = (run_dir / "train_artifacts.npz").is_file()
    train_fn(config_path, run_id, data_root, runs_root, resume=resume)
    verify_fn(run_dir)
    return evaluate_fn(run_dir, data_root, split="test")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id", default="ryo-v0")
    arguments = parser.parse_args(argv)
    runs_root = baseline_path(arguments.runs_root)
    try:
        report = reproduce(
            baseline_path(arguments.config),
            baseline_path(arguments.data_root),
            runs_root,
            arguments.run_id,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"report={runs_root / arguments.run_id / 'evaluation.test.json'} "
        f"decision={report['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
