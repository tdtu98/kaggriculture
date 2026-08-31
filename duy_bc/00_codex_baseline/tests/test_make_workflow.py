"""Executable contracts for the two-command Make workflow."""

import subprocess
import unittest
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]


class MakeWorkflowTest(unittest.TestCase):
    def _make_dry_run(self, target: str) -> str:
        result = subprocess.run(
            ["make", "-n", target],
            cwd=BASELINE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_setup_uses_python_312_and_installs_the_development_extra(self) -> None:
        """Catch setup silently selecting an incompatible Python environment."""
        output = self._make_dry_run("setup")
        self.assertIn("python3.12", output)
        self.assertIn("pip install -e '.[dev]'", output)

    def test_reproduce_runs_the_single_restart_safe_entrypoint(self) -> None:
        """Catch Make bypassing automatic resume or frozen evaluation."""
        output = self._make_dry_run("reproduce")
        self.assertIn("-m scripts.reproduce", output)
        self.assertIn("--config configs/v0.json", output)
        self.assertIn("--run-id ryo-v0", output)

    def test_git_ignores_local_environment_data_runs_and_caches(self) -> None:
        """Catch generated or machine-local content entering the handoff."""
        for relative in (
            ".venv/bin/python",
            "data/train/game.npz",
            "runs/ryo-v0/model.pt",
            "src/bc_core/__pycache__/paths.pyc",
            ".pytest_cache/state",
            "src/duy_bc_baseline.egg-info/PKG-INFO",
            "docs/superpowers/plans/example-plan.md",
            "duy_bc/00_codex_baseline/docs/superpowers/plans/example-plan.md",
        ):
            result = subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", relative],
                cwd=BASELINE_ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, relative)


if __name__ == "__main__":
    unittest.main()
