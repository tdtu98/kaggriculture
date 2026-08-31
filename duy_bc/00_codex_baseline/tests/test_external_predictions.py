"""Model-independent prediction evaluation contracts."""

from __future__ import annotations

import unittest
import tempfile
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS
from bc_core.evaluate import (
    EvaluationError,
    external_prediction_report,
    load_external_prediction_archive,
)
from bc_core.features import EncodedGame


def _game(
    episode_id: str,
    labels: list[int],
    steps: list[int],
    actors: list[int],
) -> EncodedGame:
    rows = len(labels)
    environment_steps = max(steps) + 1
    actor_features = np.zeros((rows, ACTOR_DIM), dtype=np.float32)
    actor_features[:, 1] = np.asarray(actors, dtype=np.float32) / 8.0
    actor_features[:, 0] = np.asarray(
        [1.0 if actor == 0 else 0.0 for actor in actors], dtype=np.float32
    )
    arrays = {
        "grid": np.zeros(
            (environment_steps, GRID_CHANNELS, 10, 10), dtype=np.float32
        ),
        "global_features": np.zeros(
            (environment_steps, GLOBAL_DIM), dtype=np.float32
        ),
        "actor_features": actor_features,
        "step_index": np.asarray(steps, dtype=np.int32),
        "label": np.asarray(labels, dtype=np.int64),
        "argument_item": np.full(rows, -1, dtype=np.int16),
        "argument_quantity": np.zeros(rows, dtype=np.int16),
    }
    for array in arrays.values():
        array.flags.writeable = False
    return EncodedGame(
        **arrays,
        metadata={
            "episode_id": episode_id,
            "split": "val",
            "source_date": "21-08",
            "route_family": "route-a",
        },
    )


class ExternalPredictionReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.games = [
            _game("game-a", [0, 1, 2], [0, 0, 1], [0, 1, 0]),
            _game("game-b", [3, 4], [0, 1], [0, 0]),
        ]

    def test_predictions_are_aligned_by_row_identity(self) -> None:
        # Catches treating an external file's arbitrary row order as canonical.
        report = external_prediction_report(
            predictions=np.asarray([4, 3, 2, 1, 0], dtype=np.int64),
            prediction_game_ids=np.asarray(
                ["game-b", "game-b", "game-a", "game-a", "game-a"]
            ),
            prediction_step_indices=np.asarray([1, 0, 1, 0, 0], dtype=np.int64),
            prediction_actor_ids=np.asarray([0, 0, 0, 1, 0], dtype=np.int64),
            games=self.games,
            model_name="tu-transformer-v1",
        )

        self.assertEqual(report["model_name"], "tu-transformer-v1")
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["games"], 2)
        self.assertEqual(report["operations"], list(OPERATIONS))
        metrics = report["core_cloning_metrics"]
        self.assertEqual(metrics["step_prefix_auc_at_24"], 1.0)
        self.assertEqual(metrics["daily_gated_prefix_auc"], 1.0)
        self.assertEqual(metrics["action_macro_f1"], 1.0)
        self.assertEqual(metrics["raw_accuracy"], 1.0)
        self.assertEqual(metrics["joint_farm_step_prefix_auc_at_24"], 1.0)
        self.assertEqual(metrics["joint_farm_daily_gated_prefix_auc"], 1.0)
        self.assertEqual(
            [item["operation"] for item in report["per_action"]],
            list(OPERATIONS),
        )

    def test_duplicate_prediction_identity_is_rejected(self) -> None:
        # Catches silently keeping one of two predictions for the same actor-step.
        with self.assertRaisesRegex(EvaluationError, "duplicate prediction row"):
            external_prediction_report(
                predictions=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                prediction_game_ids=np.asarray(
                    ["game-a", "game-a", "game-a", "game-b", "game-b"]
                ),
                prediction_step_indices=np.asarray([0, 0, 1, 0, 0], dtype=np.int64),
                prediction_actor_ids=np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
                games=self.games,
                model_name="tu-model",
            )

    def test_prediction_identity_set_must_match_validation_rows(self) -> None:
        # Catches scoring a subset or an unrelated row in place of held-out rows.
        with self.assertRaisesRegex(
            EvaluationError, "prediction rows do not match validation rows"
        ):
            external_prediction_report(
                predictions=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                prediction_game_ids=np.asarray(
                    ["game-a", "game-a", "game-a", "game-b", "other-game"]
                ),
                prediction_step_indices=np.asarray([0, 0, 1, 0, 1], dtype=np.int64),
                prediction_actor_ids=np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
                games=self.games,
                model_name="tu-model",
            )

    def test_prediction_operation_ids_must_use_fixed_vocabulary(self) -> None:
        # Catches NumPy negative indexing or silently accepting a new class ID.
        for invalid in (-1, len(OPERATIONS)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                EvaluationError, "prediction operation IDs"
            ):
                external_prediction_report(
                    predictions=np.asarray([invalid, 1, 2, 3, 4], dtype=np.int64),
                    prediction_game_ids=np.asarray(
                        ["game-a", "game-a", "game-a", "game-b", "game-b"]
                    ),
                    prediction_step_indices=np.asarray([0, 0, 1, 0, 1], dtype=np.int64),
                    prediction_actor_ids=np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
                    games=self.games,
                    model_name="tu-model",
                )

    def test_model_name_must_be_safe_for_report_filename(self) -> None:
        # Catches traversal or ambiguous names reaching the output path.
        for invalid in ("", "../tu-model", "tu model", "."):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                EvaluationError, "model name"
            ):
                external_prediction_report(
                    predictions=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                    prediction_game_ids=np.asarray(
                        ["game-a", "game-a", "game-a", "game-b", "game-b"]
                    ),
                    prediction_step_indices=np.asarray([0, 0, 1, 0, 1], dtype=np.int64),
                    prediction_actor_ids=np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
                    games=self.games,
                    model_name=invalid,
                )


class ExternalPredictionArchiveTest(unittest.TestCase):
    def _write(self, path: Path, **overrides: np.ndarray) -> None:
        arrays = {
            "predictions": np.asarray([0, 1], dtype=np.int64),
            "game_ids": np.asarray(["game-a", "game-a"]),
            "step_indices": np.asarray([0, 0], dtype=np.int64),
            "actor_ids": np.asarray([0, 1], dtype=np.int64),
        }
        np.savez_compressed(path, **(arrays | overrides))

    def test_valid_archive_loads_without_expert_labels(self) -> None:
        # Catches changing the collaboration format or requiring label leakage.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tu-model.val.npz"
            self._write(path)

            archive = load_external_prediction_archive(path)

        self.assertEqual(set(archive), {
            "predictions",
            "game_ids",
            "step_indices",
            "actor_ids",
        })
        np.testing.assert_array_equal(archive["predictions"], [0, 1])
        np.testing.assert_array_equal(archive["game_ids"], ["game-a", "game-a"])

    def test_archive_rejects_labels_or_unknown_fields(self) -> None:
        # Catches trusting model-supplied truth instead of evaluator-owned labels.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "with-labels.npz"
            self._write(path, labels=np.asarray([0, 1], dtype=np.int64))

            with self.assertRaisesRegex(EvaluationError, "exactly these fields"):
                load_external_prediction_archive(path)

    def test_archive_rejects_object_game_ids(self) -> None:
        # Catches enabling pickle while loading an untrusted collaborator artifact.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-ids.npz"
            self._write(path, game_ids=np.asarray(["game-a", "game-a"], dtype=object))

            with self.assertRaisesRegex(EvaluationError, "without pickle"):
                load_external_prediction_archive(path)

    def test_archive_rejects_misaligned_or_wrong_dtype_arrays(self) -> None:
        # Catches delaying basic shape/type failures until metric calculation.
        variants = {
            "short-steps": {"step_indices": np.asarray([0], dtype=np.int64)},
            "float-predictions": {
                "predictions": np.asarray([0.0, 1.0], dtype=np.float32)
            },
            "byte-game-ids": {"game_ids": np.asarray([b"game-a", b"game-a"])},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, overrides in variants.items():
                with self.subTest(name=name):
                    path = root / f"{name}.npz"
                    self._write(path, **overrides)
                    with self.assertRaises(EvaluationError):
                        load_external_prediction_archive(path)

    def test_archive_must_be_a_regular_non_symlink_file(self) -> None:
        # Catches path swaps at the prediction-file trust boundary.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.npz"
            link = root / "link.npz"
            self._write(target)
            link.symlink_to(target)

            with self.assertRaisesRegex(EvaluationError, "regular non-symlink"):
                load_external_prediction_archive(link)


class ExternalPredictionCliTest(unittest.TestCase):
    def test_help_exposes_complete_collaborator_contract(self) -> None:
        # Catches shipping the evaluator logic without a usable command for Tu.
        baseline = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(baseline / "src")

        result = subprocess.run(
            [
                sys.executable,
                str(baseline / "scripts" / "evaluate_predictions.py"),
                "--help",
            ],
            cwd=baseline,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in (
            "--predictions",
            "--model-name",
            "--reference-run-id",
            "--data-root",
            "--runs-root",
        ):
            self.assertIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main()
