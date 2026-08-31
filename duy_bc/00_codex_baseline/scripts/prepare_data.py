#!/usr/bin/env python3
"""Prepare deterministic behavior-cloning shards from the fixed replay corpus."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from bc_core.paths import baseline_path
from bc_core.prepare import prepare


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.json")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--limit-per-split", type=_positive_integer)
    arguments = parser.parse_args(argv)
    try:
        audit = prepare(
            baseline_path(arguments.config),
            baseline_path(arguments.output_root),
            limit_per_split=arguments.limit_per_split,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = " ".join(
        f"{split}={audit['splits'][split]['games']} games/{audit['splits'][split]['samples']} samples"
        for split in ("train", "val", "test")
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
