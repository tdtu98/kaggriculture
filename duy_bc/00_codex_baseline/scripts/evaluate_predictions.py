#!/usr/bin/env python3
"""Evaluate one external model's saved actions on authenticated validation rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from bc_core.evaluate import evaluate_external_predictions
from bc_core.paths import baseline_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    arguments = parser.parse_args(argv)
    if (
        not arguments.reference_run_id
        or arguments.reference_run_id in {".", ".."}
        or Path(arguments.reference_run_id).name != arguments.reference_run_id
    ):
        print(
            "error: reference-run-id must be one non-empty path component",
            file=sys.stderr,
        )
        return 1

    run_dir = baseline_path(arguments.runs_root) / arguments.reference_run_id
    try:
        report, json_path, markdown_path = evaluate_external_predictions(
            baseline_path(arguments.predictions),
            run_dir,
            baseline_path(arguments.data_root),
            arguments.model_name,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    core = report["core_cloning_metrics"]
    print(
        f"report={json_path} markdown={markdown_path} "
        f"step_prefix_auc_at_24={core['step_prefix_auc_at_24']:.6f} "
        f"daily_gated_prefix_auc={core['daily_gated_prefix_auc']:.6f} "
        f"action_macro_f1={core['action_macro_f1']:.6f} "
        f"raw_accuracy={core['raw_accuracy']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
