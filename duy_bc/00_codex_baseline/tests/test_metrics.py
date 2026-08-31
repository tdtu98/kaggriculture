import json
import unittest

import numpy as np

from bc_core.metrics import (
    classification_report,
    core_cloning_metrics,
    paired_game_bootstrap,
    slice_reports,
)


def _logits_for(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    logits = np.full((len(labels), 17), -10.0)
    for row, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
        logits[row, prediction] = 10.0
        if label != prediction:
            logits[row, label] = 9.0
    return logits


class MetricsTest(unittest.TestCase):
    def test_core_cloning_metrics_recovers_after_a_bad_step(self) -> None:
        # Catches truncating the whole game after the first disagreement.
        labels = np.zeros(8, dtype=np.int64)
        predictions = np.asarray([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64)

        report = core_cloning_metrics(
            _logits_for(labels, predictions),
            labels,
            ["game-a"] * 8,
            np.arange(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            step_horizon=4,
            turns_per_day=4,
        )

        self.assertAlmostEqual(
            report["step_prefix_auc_at_4"],
            (7 / 8 + 5 / 7 + 3 / 6 + 1 / 5) / 4,
        )
        self.assertEqual(
            report["step_prefix_survival"],
            [
                {"horizon": 1, "perfect_windows": 7, "windows": 8, "rate": 7 / 8},
                {"horizon": 2, "perfect_windows": 5, "windows": 7, "rate": 5 / 7},
                {"horizon": 3, "perfect_windows": 3, "windows": 6, "rate": 3 / 6},
                {"horizon": 4, "perfect_windows": 1, "windows": 5, "rate": 1 / 5},
            ],
        )

    def test_core_cloning_metrics_isolates_actor_gates_and_retains_joint_farm(self) -> None:
        # Catches one actor's miss erasing other actors from the primary daily metric.
        labels = np.zeros(10, dtype=np.int64)
        predictions = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
        step_indices = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 5], dtype=np.int64)
        actor_ids = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)

        report = core_cloning_metrics(
            _logits_for(labels, predictions),
            labels,
            ["game-a"] * 10,
            step_indices,
            actor_ids,
            step_horizon=4,
            turns_per_day=4,
        )

        # Actor 0 earns 2/4 on day 0 and 2/2 on day 1; actor 1 earns 4/4.
        # The strict joint-farm diagnostic is [T, T, F, T] then [T, T].
        self.assertAlmostEqual(report["daily_gated_prefix_auc"], 5 / 6)
        self.assertAlmostEqual(report["joint_farm_daily_gated_prefix_auc"], 0.75)
        self.assertEqual(report["actor_trajectories"], 2)
        self.assertEqual(report["actor_days"], 3)
        self.assertEqual(report["environment_steps"], 6)
        self.assertEqual(report["days"], 2)

    def test_core_cloning_metrics_macro_f1_uses_observed_actions_only(self) -> None:
        # Catches diluting action balance with the thirteen unsupported operations.
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        predictions = np.asarray([0, 1, 1, 1], dtype=np.int64)

        report = core_cloning_metrics(
            _logits_for(labels, predictions),
            labels,
            ["game-a"] * 4,
            np.arange(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
        )

        self.assertAlmostEqual(report["action_macro_f1"], ((2 / 3) + 0.8) / 2)
        self.assertEqual(report["raw_accuracy"], 0.75)
        self.assertEqual(report["supported_actions"], 2)

    def test_core_cloning_metrics_allows_inactive_gaps_in_an_actor_slot(self) -> None:
        # Catches rejecting normal turns where a hand has no observed action row.
        labels = np.zeros(6, dtype=np.int64)
        steps = np.asarray([0, 1, 1, 2, 3, 3], dtype=np.int64)
        actors = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.int64)

        report = core_cloning_metrics(
            _logits_for(labels, labels),
            labels,
            ["game-a"] * 6,
            steps,
            actors,
            step_horizon=3,
        )

        self.assertEqual(report["step_prefix_auc_at_3"], 1.0)
        self.assertEqual(report["actor_trajectories"], 2)

    def test_core_cloning_metrics_reports_short_fixture_horizon_transparently(self) -> None:
        # Catches silently inventing windows when an evaluation game is shorter than 24.
        labels = np.zeros(3, dtype=np.int64)

        report = core_cloning_metrics(
            _logits_for(labels, labels),
            labels,
            ["game-a"] * 3,
            np.arange(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        )

        self.assertEqual(report["requested_step_horizon"], 24)
        self.assertEqual(report["evaluated_step_horizon"], 3)
        self.assertEqual(report["step_prefix_auc_at_24"], 1.0)

    def test_core_cloning_metrics_rejects_misaligned_sequence_metadata(self) -> None:
        # Catches zip truncation and ambiguous temporal order in sequence metrics.
        labels = np.zeros(2, dtype=np.int64)
        logits = _logits_for(labels, labels)
        invalid = (
            (["game-a"], np.asarray([0, 1]), np.asarray([0, 0])),
            (["game-a", "game-a"], np.asarray([0]), np.asarray([0, 0])),
            (["game-a", "game-a"], np.asarray([0, 1]), np.asarray([0])),
            (["game-a", 2], np.asarray([0, 1]), np.asarray([0, 0])),
            (["game-a", "game-a"], np.asarray([0.0, 1.0]), np.asarray([0, 0])),
            (["game-a", "game-a"], np.asarray([0, 1]), np.asarray([0.0, 0.0])),
            (["game-a", "game-a"], np.asarray([0, 2]), np.asarray([0, 0])),
            (["game-a", "game-a"], np.asarray([0, 0]), np.asarray([0, 0])),
        )
        for game_ids, steps, actors in invalid:
            with self.subTest(
                game_ids=game_ids, steps=steps.tolist(), actors=actors.tolist()
            ):
                with self.assertRaises(ValueError):
                    core_cloning_metrics(logits, labels, game_ids, steps, actors)

    def test_classification_report_uses_all_seventeen_classes(self) -> None:
        # Catches deriving the vocabulary from only the classes present in the batch.
        logits = np.full((3, 17), -10.0)
        logits[0, 0], logits[1, 1], logits[2, 2] = 10.0, 10.0, 10.0
        report = classification_report(logits, np.array([0, 2, 2]))
        self.assertAlmostEqual(report["top1"], 2 / 3)
        self.assertEqual(len(report["per_class"]), 17)
        self.assertEqual(np.asarray(report["confusion_matrix"]).shape, (17, 17))

    def test_classification_report_has_exact_confusion_and_observed_class_metrics(
        self,
    ) -> None:
        # Catches transposed confusion axes and incorrect precision/recall denominators.
        labels = np.array([0, 0, 1, 1, 2, 3])
        predictions = np.array([0, 1, 1, 2, 2, 2])
        report = classification_report(_logits_for(labels, predictions), labels)
        expected_confusion = np.zeros((17, 17), dtype=np.int64)
        expected_confusion[0, 0] = 1
        expected_confusion[0, 1] = 1
        expected_confusion[1, 1] = 1
        expected_confusion[1, 2] = 1
        expected_confusion[2, 2] = 1
        expected_confusion[3, 2] = 1
        np.testing.assert_array_equal(report["confusion_matrix"], expected_confusion)
        expected_metrics = (
            (1.0, 0.5, 2.0 / 3.0, 2),
            (0.5, 0.5, 0.5, 2),
            (1.0 / 3.0, 1.0, 0.5, 1),
            (0.0, 0.0, 0.0, 1),
        )
        for class_id, (precision, recall, f1, support) in enumerate(expected_metrics):
            with self.subTest(class_id=class_id):
                actual = report["per_class"][class_id]
                self.assertAlmostEqual(actual["precision"], precision)
                self.assertAlmostEqual(actual["recall"], recall)
                self.assertAlmostEqual(actual["f1"], f1)
                self.assertEqual(actual["support"], support)
        self.assertEqual(report["top1"], 0.5)
        self.assertEqual(report["top3"], 1.0)
        self.assertAlmostEqual(report["macro_f1"], 5.0 / 51.0)

    def test_classification_report_computes_top_three_membership(self) -> None:
        # Catches treating top-3 as an ordered prediction or using the wrong axis.
        logits = np.zeros((2, 17), dtype=np.float64)
        logits[0, [1, 5, 9]] = [3.0, 2.0, 1.0]
        logits[1, [3, 4, 6, 8]] = [4.0, 3.0, 2.0, 1.0]
        report = classification_report(logits, np.array([9, 8]))
        self.assertEqual(report["top1"], 0.0)
        self.assertEqual(report["top3"], 0.5)

    def test_top_three_breaks_boundary_ties_by_lowest_class_id(self) -> None:
        # Catches platform-dependent argpartition membership at a tied boundary.
        logits = np.zeros((2, 17), dtype=np.float64)
        report = classification_report(logits, np.array([2, 3]))
        self.assertEqual(report["top3"], 0.5)

    def test_zero_support_metrics_are_zero_and_macro_averages_all_classes(self) -> None:
        # Catches NaNs or averaging macro-F1 over only observed operations.
        logits = np.full((2, 17), -1.0)
        logits[:, 0] = 1.0
        report = classification_report(logits, np.array([0, 0]))
        self.assertEqual(report["per_class"][1]["support"], 0)
        self.assertEqual(report["per_class"][1]["precision"], 0.0)
        self.assertEqual(report["per_class"][1]["recall"], 0.0)
        self.assertEqual(report["per_class"][1]["f1"], 0.0)
        self.assertAlmostEqual(report["macro_f1"], 1.0 / 17.0)

    def test_classification_report_is_json_serializable(self) -> None:
        # Catches NumPy arrays and scalar values leaking into persisted reports.
        logits = np.eye(17, dtype=np.float32)
        report = classification_report(logits, np.arange(17, dtype=np.int64))
        encoded = json.dumps(report, allow_nan=False)
        self.assertIsInstance(encoded, str)

    def test_classification_report_rejects_malformed_inputs(self) -> None:
        # Catches silent broadcasting, invalid labels, and non-finite score handling.
        valid_logits = np.zeros((2, 17), dtype=np.float64)
        bad_inputs = (
            (np.zeros((2, 16)), np.array([0, 1])),
            (valid_logits, np.array([[0, 1]])),
            (valid_logits, np.array([0])),
            (valid_logits, np.array([0.0, 1.0])),
            (valid_logits, np.array([0, 17])),
            (np.where(np.arange(34).reshape(2, 17) == 0, np.nan, valid_logits),
             np.array([0, 1])),
        )
        for logits, labels in bad_inputs:
            with self.subTest(logits_shape=logits.shape, labels_shape=labels.shape):
                with self.assertRaises(ValueError):
                    classification_report(logits, labels)

    def test_slice_reports_groups_every_required_dimension_and_exact_value(self) -> None:
        # Catches shifted masks, first-N masks, omitted families, and collapsed groups.
        labels = np.array([0, 0, 1, 1, 2, 3])
        predictions = np.array([0, 1, 1, 2, 2, 2])
        slices = (
            {
                "actor_type": "farmer",
                "seat": "0",
                "day_band": "days-1-7",
                "source_date": "2026-08-21",
                "route_family": "route-a",
            },
            {
                "actor_type": "hand",
                "seat": "1",
                "day_band": "days-1-7",
                "source_date": "2026-08-22",
                "route_family": "route-b",
            },
            {
                "actor_type": "hand",
                "seat": "0",
                "day_band": "days-8-14",
                "source_date": "2026-08-22",
                "route_family": "route-a",
            },
            {
                "actor_type": "farmer",
                "seat": "1",
                "day_band": "days-8-14",
                "source_date": "2026-08-21",
                "route_family": "route-a",
            },
            {
                "actor_type": "farmer",
                "seat": "1",
                "day_band": "days-1-7",
                "source_date": "2026-08-22",
                "route_family": "route-b",
            },
            {
                "actor_type": "hand",
                "seat": "0",
                "day_band": "days-8-14",
                "source_date": "2026-08-21",
                "route_family": "route-b",
            },
        )
        reports = slice_reports(_logits_for(labels, predictions), labels, slices)
        self.assertEqual(
            tuple(reports),
            ("actor_type", "seat", "day_band", "source_date", "route_family"),
        )
        self.assertEqual(tuple(reports["actor_type"]), ("farmer", "hand"))
        self.assertEqual(tuple(reports["seat"]), ("0", "1"))
        self.assertEqual(tuple(reports["day_band"]), ("days-1-7", "days-8-14"))
        self.assertEqual(
            tuple(reports["source_date"]), ("2026-08-21", "2026-08-22")
        )
        self.assertEqual(tuple(reports["route_family"]), ("route-a", "route-b"))
        expected = {
            "actor_type": {
                "farmer": (2.0 / 3.0, [1, 1, 1] + [0] * 14,
                           ((0, 0), (1, 2), (2, 2))),
                "hand": (1.0 / 3.0, [1, 1, 0, 1] + [0] * 13,
                        ((0, 1), (1, 1), (3, 2))),
            },
            "seat": {
                "0": (2.0 / 3.0, [1, 1, 0, 1] + [0] * 13,
                      ((0, 0), (1, 1), (3, 2))),
                "1": (1.0 / 3.0, [1, 1, 1] + [0] * 14,
                      ((0, 1), (1, 2), (2, 2))),
            },
            "day_band": {
                "days-1-7": (2.0 / 3.0, [2, 0, 1] + [0] * 14,
                             ((0, 0), (0, 1), (2, 2))),
                "days-8-14": (1.0 / 3.0, [0, 2, 0, 1] + [0] * 13,
                              ((1, 1), (1, 2), (3, 2))),
            },
            "source_date": {
                "2026-08-21": (1.0 / 3.0, [1, 1, 0, 1] + [0] * 13,
                               ((0, 0), (1, 2), (3, 2))),
                "2026-08-22": (2.0 / 3.0, [1, 1, 1] + [0] * 14,
                               ((0, 1), (1, 1), (2, 2))),
            },
            "route_family": {
                "route-a": (2.0 / 3.0, [1, 2] + [0] * 15,
                            ((0, 0), (1, 1), (1, 2))),
                "route-b": (1.0 / 3.0, [1, 0, 1, 1] + [0] * 13,
                            ((0, 1), (2, 2), (3, 2))),
            },
        }
        for dimension, values in expected.items():
            for value, (top1, support, entries) in values.items():
                with self.subTest(dimension=dimension, value=value):
                    report = reports[dimension][value]
                    expected_confusion = np.zeros((17, 17), dtype=np.int64)
                    for label, prediction in entries:
                        expected_confusion[label, prediction] += 1
                    np.testing.assert_array_equal(
                        report["confusion_matrix"], expected_confusion
                    )
                    self.assertEqual(
                        [item["support"] for item in report["per_class"]], support
                    )
                    self.assertAlmostEqual(report["top1"], top1)
                    self.assertEqual(report["top3"], 1.0)

    def test_slice_reports_rejects_bad_row_metadata(self) -> None:
        # Catches misaligned rows and missing/non-string Batch slice values.
        logits = np.zeros((1, 17), dtype=np.float64)
        labels = np.array([0])
        valid = {
            "actor_type": "farmer",
            "seat": "0",
            "day_band": "days-1-7",
            "source_date": "2026-08-21",
            "route_family": "route-a",
        }
        with self.assertRaises(ValueError):
            slice_reports(logits, labels, ())
        for field in valid:
            row = dict(valid)
            row.pop(field)
            with self.subTest(missing=field):
                with self.assertRaises(ValueError):
                    slice_reports(logits, labels, (row,))
        row = dict(valid)
        row["seat"] = 0  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            slice_reports(logits, labels, (row,))

    def test_bootstrap_resamples_games_not_rows(self) -> None:
        # Catches weighting games by their row counts instead of equally.
        games = np.array(["large"] * 100 + ["small"])
        state = np.array([1] * 100 + [0], dtype=bool)
        clock = np.array([0] * 100 + [1], dtype=bool)
        result = paired_game_bootstrap(state, clock, games, resamples=1000, seed=7)
        self.assertEqual(result["games"], 2)
        self.assertAlmostEqual(result["point_delta"], 0.0)
        self.assertEqual(result["ci95_low"], -1.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_bootstrap_keeps_models_paired_within_each_sampled_game(self) -> None:
        # Catches independently drawing state and clock games in each replicate.
        games = np.array(["accurate", "inaccurate"])
        state = np.array([True, False])
        clock = np.array([True, False])
        result = paired_game_bootstrap(state, clock, games, resamples=100, seed=11)
        self.assertEqual(result["point_delta"], 0.0)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 0.0)

    def test_bootstrap_matches_seeded_unweighted_per_game_delta_draws(self) -> None:
        # Catches pooling unequal row counts after drawing game IDs.
        games = np.array(["large"] * 4 + ["medium"] * 2 + ["small"])
        state = np.array([1, 1, 1, 0, 0, 1, 1], dtype=bool)
        clock = np.array([1, 0, 0, 0, 1, 1, 0], dtype=bool)
        resamples = 40
        seed = 23
        game_deltas = np.array([0.5, -0.5, 1.0])
        draws = np.random.default_rng(seed).integers(0, 3, size=(resamples, 3))
        expected_deltas = game_deltas[draws].mean(axis=1)
        expected_low, expected_high = np.percentile(expected_deltas, [2.5, 97.5])

        result = paired_game_bootstrap(state, clock, games, resamples, seed)

        self.assertEqual(result["point_delta"], 1.0 / 3.0)
        self.assertEqual(result["ci95_low"], float(expected_low))
        self.assertEqual(result["ci95_high"], float(expected_high))

    def test_bootstrap_is_paired_deterministic_and_json_serializable(self) -> None:
        # Catches independent resampling, global RNG use, or NumPy scalar leakage.
        games = np.array(["a", "a", "b", "b", "c", "c"])
        state = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        clock = np.array([1, 0, 0, 0, 1, 1], dtype=bool)
        first = paired_game_bootstrap(state, clock, games, resamples=200, seed=19)
        second = paired_game_bootstrap(state, clock, games, resamples=200, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 19)
        self.assertEqual(first["resamples"], 200)
        self.assertLessEqual(first["ci95_low"], first["point_delta"])
        self.assertGreaterEqual(first["ci95_high"], first["point_delta"])
        self.assertIsInstance(json.dumps(first, allow_nan=False), str)

    def test_bootstrap_rejects_malformed_paired_inputs(self) -> None:
        # Catches truncating mismatched pairs or accepting invalid correctness flags.
        valid = np.array([True, False])
        invalid_calls = (
            (valid[:1], valid, ["a", "b"], 10, 1),
            (valid, valid[:1], ["a", "b"], 10, 1),
            (valid, valid, ["a"], 10, 1),
            (np.array([0, 2]), valid, ["a", "b"], 10, 1),
            (valid, valid, ["a", 2], 10, 1),
            (valid, valid, ["a", "b"], 0, 1),
            (valid, valid, ["a", "b"], 10, -1),
        )
        for state, clock, games, resamples, seed in invalid_calls:
            with self.subTest(games=games, resamples=resamples, seed=seed):
                with self.assertRaises(ValueError):
                    paired_game_bootstrap(state, clock, games, resamples, seed)


if __name__ == "__main__":
    unittest.main()
