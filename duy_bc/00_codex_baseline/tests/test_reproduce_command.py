"""Behavioral tests for the complete restart-safe command."""

import tempfile
import unittest
from pathlib import Path

from scripts.reproduce import reproduce


class ReproduceCommandTest(unittest.TestCase):
    def test_reproduce_runs_the_authenticated_stages_in_order(self) -> None:
        """Catch evaluation before selection verification or omitted preparation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "v0.json"
            data = root / "data"
            runs = root / "runs"
            calls: list[str] = []

            def prepare_stage(config_path: Path, data_root: Path) -> None:
                self.assertEqual((config_path, data_root), (config, data))
                data_root.mkdir()
                calls.append("prepare")

            def train_stage(
                config_path: Path,
                run_id: str,
                data_root: Path,
                runs_root: Path,
                *,
                resume: bool,
            ) -> None:
                self.assertEqual(
                    (config_path, run_id, data_root, runs_root, resume),
                    (config, "ryo-v0", data, runs, False),
                )
                run_dir = runs_root / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "selection.json").write_text("{}\n", encoding="utf-8")
                calls.append("train")

            def verify_stage(run_dir: Path) -> None:
                self.assertTrue((run_dir / "selection.json").is_file())
                calls.append("verify")

            def evaluate_stage(
                run_dir: Path, data_root: Path, *, split: str
            ) -> dict[str, str]:
                self.assertEqual((data_root, split), (data, "test"))
                (run_dir / "evaluation.test.json").write_text(
                    '{"decision":"GO"}\n', encoding="utf-8"
                )
                (run_dir / "REPORT.md").write_text("GO\n", encoding="utf-8")
                calls.append("evaluate")
                return {"decision": "GO"}

            report = reproduce(
                config,
                data,
                runs,
                "ryo-v0",
                prepare_fn=prepare_stage,
                train_fn=train_stage,
                verify_fn=verify_stage,
                evaluate_fn=evaluate_stage,
            )

            self.assertEqual(calls, ["prepare", "train", "verify", "evaluate"])
            self.assertEqual(report, {"decision": "GO"})
            self.assertTrue((runs / "ryo-v0/evaluation.test.json").is_file())
            self.assertTrue((runs / "ryo-v0/REPORT.md").is_file())

    def test_reproduce_resumes_only_after_train_artifacts_exist(self) -> None:
        """Catch a rerun that starts over or resumes an empty partial directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "ryo-v0"
            run_dir.mkdir(parents=True)
            (run_dir / "train_artifacts.npz").write_bytes(b"frozen")
            observed: list[bool] = []

            def train_stage(*_args, resume: bool, **_kwargs) -> None:
                observed.append(resume)

            reproduce(
                root / "v0.json",
                root / "data",
                root / "runs",
                "ryo-v0",
                prepare_fn=lambda *_args: None,
                train_fn=train_stage,
                verify_fn=lambda *_args: None,
                evaluate_fn=lambda *_args, **_kwargs: {},
            )

            self.assertEqual(observed, [True])

    def test_reproduce_rejects_unsafe_run_id_before_preparation(self) -> None:
        """Catch path traversal reaching the preparation or run directories."""
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "one non-empty path component"):
            reproduce(
                Path("v0.json"),
                Path("data"),
                Path("runs"),
                "../escape",
                prepare_fn=lambda *_args: calls.append("prepare"),
                train_fn=lambda *_args, **_kwargs: None,
                verify_fn=lambda *_args: None,
                evaluate_fn=lambda *_args, **_kwargs: {},
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
