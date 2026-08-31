"""Behavioral checks for the standalone handoff package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]


class PackageBoundaryTest(unittest.TestCase):
    def test_core_modules_import_outside_the_repository_without_duy_rl(self) -> None:
        """Catch copied modules that still depend on the external Duy package."""
        probe = """
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
modules = (
    "bc_core.constants",
    "bc_core.dataset",
    "bc_core.evaluate",
    "bc_core.features",
    "bc_core.metrics",
    "bc_core.checkpoints",
    "bc_core.prepare",
    "bc_core.replay",
    "bc_core.train",
    "bc_core.training_audit",
    "model.clock",
    "model.majority",
    "model.state",
)
loaded = [importlib.import_module(name) for name in modules]
assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root / "src") for module in loaded)
assert not any(name == "duy_rl" or name.startswith("duy_rl.") for name in sys.modules)
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(BASELINE_ROOT / "src")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    probe,
                    str(BASELINE_ROOT),
                ],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
