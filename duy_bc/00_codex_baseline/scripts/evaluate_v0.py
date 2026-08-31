#!/usr/bin/env python3
"""Evaluate a frozen Ryo v0 run and publish its go/no-go report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from bc_core.evaluate import evaluate_frozen_run, verify_selection
from bc_core.paths import baseline_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    arguments = parser.parse_args(argv)
    if (
        not arguments.run_id
        or arguments.run_id in {".", ".."}
        or Path(arguments.run_id).name != arguments.run_id
    ):
        print("error: run-id must be one non-empty path component", file=sys.stderr)
        return 1

    run_dir = baseline_path(arguments.runs_root) / arguments.run_id
    data_root = baseline_path(arguments.data_root)
    try:
        if arguments.split == "test":
            verify_selection(run_dir)
        report = evaluate_frozen_run(run_dir, data_root, split=arguments.split)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    section = report["test"] if arguments.split == "test" else report["validation"]
    bootstrap = report["bootstrap"]
    report_name = (
        "evaluation.test.json" if arguments.split == "test" else "evaluation.val.json"
    )
    print(
        f"report={run_dir / report_name} "
        f"state_macro_f1={section['state']['macro_f1']:.6f} "
        f"clock_macro_f1={section['clock']['macro_f1']:.6f} "
        f"top1_delta_ci95=[{bootstrap['ci95_low']:.6f},{bootstrap['ci95_high']:.6f}] "
        f"decision={report['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
