import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS
from bc_core.dataset import (
    EncodedGameDataset,
    NormalizationStats,
    ShardDataset,
    collate_examples,
    fit_array_statistics,
    fit_train_artifacts,
    inverse_sqrt_class_weights,
    load_train_artifacts,
    save_train_artifacts,
)
from bc_core.features import EncodedGame, FeatureError, read_shard, write_shard
from model.majority import MajorityRules, fit_majority_rules, ranking_to_logits


class DatasetTest(unittest.TestCase):
    def test_statistics_use_only_supplied_training_rows(self) -> None:
        # Catches fitting to anything beyond the arrays explicitly supplied.
        train = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        mean, std = fit_array_statistics([train])
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(std, [1.0, 2.0])

    def test_statistics_replace_zero_variance_with_one(self) -> None:
        # Catches division by zero when a training feature is constant.
        rows = np.array([[4.0, 1.0], [4.0, 3.0]], dtype=np.float32)
        mean, std = fit_array_statistics([rows])
        np.testing.assert_array_equal(mean, np.array([4.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(std, np.array([1.0, 1.0], dtype=np.float32))

    def test_class_weights_are_exactly_mean_one_when_cap_is_active(self) -> None:
        # Catches clipping after normalization, which destroys the mean-one invariant.
        weights = inverse_sqrt_class_weights(
            np.array([10**12, 10**12, 1], dtype=np.int64), cap=1.2
        )
        np.testing.assert_allclose(weights, [0.9, 0.9, 1.2], rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 1.2)

    def test_majority_rules_have_complete_deterministic_rankings(self) -> None:
        # Catches incomplete top-k rules and nondeterministic tie handling.
        rules = fit_majority_rules(
            labels=np.array([1, 1, 2, 2, 2]),
            actor_is_farmer=np.array([1, 1, 0, 0, 0]),
        )
        self.assertEqual((rules.global_label, rules.farmer_label, rules.hand_label), (2, 1, 2))
        self.assertEqual(rules.global_ranking[:3], (2, 1, 0))
        self.assertEqual(tuple(sorted(rules.global_ranking)), tuple(range(17)))
        np.testing.assert_array_equal(rules.predict(np.array([1, 0, 0])), [1, 2, 2])
        logits = ranking_to_logits(rules.global_ranking)
        self.assertEqual((logits[2], logits[1], logits[0]), (17.0, 16.0, 15.0))

    def test_train_artifacts_require_all_operations_in_train(self) -> None:
        # Catches deferring missing train support until validation or model training.
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write_game(Path(directory), "train", "missing", labels=np.arange(16))
            with self.assertRaisesRegex(ValueError, "positive train count"):
                fit_train_artifacts([shard])

    def test_train_artifacts_reject_non_train_shards(self) -> None:
        # Catches validation/test leakage into fitted statistics or baselines.
        for split in ("val", "test"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                shard = self._write_game(
                    Path(directory), split, f"{split}-game", labels=np.arange(17)
                )
                with self.assertRaisesRegex(ValueError, "split.*train"):
                    fit_train_artifacts([shard])

    def test_fit_train_artifacts_uses_sample_indexed_global_rows(self) -> None:
        # Catches weighting every stored step equally instead of each unit example.
        with tempfile.TemporaryDirectory() as directory:
            game = self._game("train", "weighted", labels=np.arange(17))
            game.global_features[0, 0] = 1.0
            game.global_features[1, 0] = 9.0
            game.step_index[:] = 0
            game.step_index[-1] = 1
            shard = Path(directory) / "weighted.npz"
            write_shard(game, shard)
            stats, counts, weights, majority = fit_train_artifacts([shard])
        self.assertAlmostEqual(float(stats.global_mean[0]), 25.0 / 17.0, places=6)
        np.testing.assert_array_equal(counts, np.ones(17, dtype=np.int64))
        np.testing.assert_array_equal(weights, np.ones(17, dtype=np.float32))
        self.assertEqual(majority.global_ranking, tuple(range(17)))

    def test_artifact_round_trip_is_logical_and_refuses_mismatched_overwrite(self) -> None:
        # Catches zip-container identity and silent replacement of frozen train artifacts.
        stats = self._stats()
        counts = np.arange(1, 18, dtype=np.int64)
        weights = inverse_sqrt_class_weights(counts, cap=4.0)
        labels = np.repeat(np.arange(17), counts)
        actors = np.resize(np.array([1, 0], dtype=np.int64), labels.shape)
        majority = fit_majority_rules(labels, actors)
        metadata = self._artifact_metadata(weight_cap=4.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-artifacts.npz"
            first = save_train_artifacts(path, stats, counts, weights, majority, metadata)
            second = save_train_artifacts(path, stats, counts, weights, majority, metadata)
            loaded = load_train_artifacts(path)
            changed_counts = counts.copy()
            changed_counts[0] += 1
            with self.assertRaisesRegex(ValueError, "non-identical"):
                save_train_artifacts(
                    path, stats, changed_counts, weights, majority, metadata
                )
        self.assertEqual(first, second)
        loaded_stats, loaded_counts, loaded_weights, loaded_majority, loaded_metadata = loaded
        np.testing.assert_array_equal(loaded_stats.global_mean, stats.global_mean)
        np.testing.assert_array_equal(loaded_counts, counts)
        np.testing.assert_array_equal(loaded_weights, weights)
        self.assertEqual(loaded_majority, majority)
        self.assertEqual(loaded_metadata, metadata)

    def test_load_train_artifacts_rejects_wrong_vocabulary(self) -> None:
        # Catches applying normalization and label rules to a reordered vocabulary.
        stats = self._stats()
        counts = np.arange(1, 18, dtype=np.int64)
        weights = inverse_sqrt_class_weights(counts, cap=4.0)
        majority = fit_majority_rules(np.arange(17), np.resize([1, 0], 17))
        metadata = self._artifact_metadata(weight_cap=4.0)
        metadata["operations"] = list(reversed(OPERATIONS))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-vocab.npz"
            arrays = self._artifact_arrays(stats, counts, weights, majority, metadata)
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "operations"):
                load_train_artifacts(path)

    def test_save_train_artifacts_rejects_nested_nan_and_infinity(self) -> None:
        # Catches non-standard NaN/Infinity tokens entering supposedly canonical JSON.
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                stats, counts, weights, majority = self._artifact_values()
                metadata = self._artifact_metadata(weight_cap=4.0)
                metadata["nested"] = {"bad": [value]}
                with self.assertRaisesRegex(ValueError, "JSON-compatible"):
                    save_train_artifacts(
                        Path(directory) / "artifacts.npz",
                        stats,
                        counts,
                        weights,
                        majority,
                        metadata,
                    )

    def test_save_train_artifacts_rejects_non_json_keys_and_values(self) -> None:
        # Catches json.dumps coercing object keys or accepting Python-only structures.
        invalid_nested_values = ({1: "coerced-key"}, {"bad": Path("not-json")})
        for nested in invalid_nested_values:
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as directory:
                stats, counts, weights, majority = self._artifact_values()
                metadata = self._artifact_metadata(weight_cap=4.0)
                metadata["nested"] = nested
                with self.assertRaisesRegex(ValueError, "JSON-compatible"):
                    save_train_artifacts(
                        Path(directory) / "artifacts.npz",
                        stats,
                        counts,
                        weights,
                        majority,
                        metadata,
                    )

    def test_load_train_artifacts_rejects_noncanonical_metadata_text(self) -> None:
        # Catches accepting semantically equal JSON with unstable spacing or key order.
        stats, counts, weights, majority = self._artifact_values()
        metadata = self._artifact_metadata(weight_cap=4.0)
        arrays = self._artifact_arrays(stats, counts, weights, majority, metadata)
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noncanonical.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_train_artifacts(path)

    def test_dataset_uses_step_grid_injects_only_self_actor_and_preserves_shard(self) -> None:
        # Catches indexing grids by sample row or mutating shared on-disk state.
        with tempfile.TemporaryDirectory() as directory:
            game = self._game("train", "lookup", labels=np.arange(17))
            game.step_index[0] = 5
            game.grid[5, 0, :, :] = 5.0
            game.grid[0, 0, :, :] = 9.0
            game.actor_features[0, 2:4] = (2.0 / 9.0, 3.0 / 9.0)
            shard = Path(directory) / "lookup.npz"
            write_shard(game, shard)
            row = ShardDataset([shard], self._stats())[0]
            stored = read_shard(shard)
        self.assertEqual(float(row["grid"][0, 0, 0]), 5.0)
        self.assertEqual(float(row["grid"][21, 3, 2]), 1.0)
        self.assertEqual(int(np.count_nonzero(row["grid"][21])), 1)
        self.assertEqual(int(np.count_nonzero(row["grid"][43])), 0)
        self.assertEqual(int(np.count_nonzero(stored.grid[:, 21])), 0)
        self.assertEqual(int(np.count_nonzero(stored.grid[:, 43])), 0)

    def test_dataset_clock_is_only_fixed_normalized_feature_slices(self) -> None:
        # Catches raw clock values, wrong source columns, or leaking other state.
        with tempfile.TemporaryDirectory() as directory:
            game = self._game("train", "clock", labels=np.arange(17))
            game.global_features[0, :7] = np.array(
                [1, 1, 3, 4, 5, 1, 7], dtype=np.float32
            )
            game.actor_features[0, :3] = np.array([1.0, 0.5, 0.9], dtype=np.float32)
            stats = NormalizationStats(
                global_mean=np.ones(GLOBAL_DIM, dtype=np.float32),
                global_std=np.full(GLOBAL_DIM, 2.0, dtype=np.float32),
                actor_mean=np.full(ACTOR_DIM, 0.25, dtype=np.float32),
                actor_std=np.full(ACTOR_DIM, 0.5, dtype=np.float32),
            )
            shard = Path(directory) / "clock.npz"
            write_shard(game, shard)
            row = ShardDataset([shard], stats)[0]
        expected = np.array([0, 0, 1, 1.5, 2, 0, 1.5, 0.5], dtype=np.float32)
        self.assertEqual(row["clock_features"].shape, (8,))
        np.testing.assert_array_equal(row["clock_features"], expected)

    def test_encoded_game_dataset_matches_shard_materialization_without_reopening(
        self,
    ) -> None:
        # Catches snapshot evaluation drifting from normalization, injection, or slices.
        with tempfile.TemporaryDirectory() as directory:
            game = self._game("test", "snapshot", labels=np.arange(17))
            game.step_index[0] = 5
            game.grid[5, 0, :, :] = 5.0
            game.actor_features[0, 2:4] = (2.0 / 9.0, 3.0 / 9.0)
            shard = Path(directory) / "snapshot.npz"
            write_shard(game, shard)
            snapshot = read_shard(shard)
            for name in (
                "grid",
                "global_features",
                "actor_features",
                "step_index",
                "label",
                "argument_item",
                "argument_quantity",
            ):
                getattr(snapshot, name).setflags(write=False)
            expected = ShardDataset([shard], self._stats())[0]
            shard.unlink()
            actual = EncodedGameDataset([snapshot], self._stats())[0]

        for name in (
            "grid",
            "global_features",
            "actor_features",
            "clock_features",
        ):
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(actual["label"], expected["label"])
        self.assertEqual(actual["game_id"], expected["game_id"])
        self.assertEqual(actual["slices"], expected["slices"])

    def test_dataset_cache_evicts_least_recent_shard_at_size_two(self) -> None:
        # Catches unbounded shard retention or FIFO behavior in the lazy cache.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self._write_game(root, "train", f"game-{index}", labels=np.arange(17))
                for index in range(3)
            ]
            dataset = ShardDataset(paths, self._stats())
            dataset[0]
            dataset[17]
            dataset[0]  # First shard becomes most recently used.
            dataset[34]  # Second shard must be evicted.
            paths[1].unlink()
            dataset[0]  # Still cached.
            with self.assertRaises(FeatureError):
                dataset[17]

    def test_dataset_rejects_cache_capacity_above_two(self) -> None:
        # Catches callers overriding the binding two-shard resident-memory ceiling.
        with self.assertRaisesRegex(ValueError, "at most two"):
            ShardDataset([], self._stats(), cache_size=3)

    def test_dataset_evicts_before_loader_reads_a_cache_miss(self) -> None:
        # Catches a transient third resident EncodedGame while loading an LRU miss.
        paths = tuple(Path(f"tiny-{index}.npz") for index in range(3))
        games = {path: self._tiny_game(f"tiny-{index}") for index, path in enumerate(paths)}
        holder: dict[str, ShardDataset] = {}
        cache_at_load: list[tuple[int, ...]] = []

        def load_tiny(path: Path) -> EncodedGame:
            if "dataset" in holder:
                cache_at_load.append(tuple(holder["dataset"]._cache))
            return games[path]

        with patch("bc_core.dataset.read_shard", side_effect=load_tiny):
            dataset = ShardDataset(paths, self._stats())
            holder["dataset"] = dataset
            dataset[0]
            dataset[1]
            dataset[2]
        self.assertEqual(cache_at_load, [(), (0,), (1,)])

    def test_dataset_preserves_slices_and_zero_based_day_bands(self) -> None:
        # Catches one-based day conversion and lost provenance slice fields.
        days = [0, 6, 7, 13, 14, 20, 21, 29]
        with tempfile.TemporaryDirectory() as directory:
            game = self._game("train", "slices", labels=np.arange(len(days)))
            game.metadata["ryo_seat"] = 1
            game.metadata["source_date"] = "2026-08-21"
            game.metadata["route_family"] = "route-z"
            for row_index, day in enumerate(days):
                game.step_index[row_index] = row_index
                game.global_features[row_index, 1] = day / 29.0
                game.global_features[row_index, 5] = 1.0
            game.actor_features[0, 0] = 1.0
            shard = Path(directory) / "slices.npz"
            write_shard(game, shard)
            dataset = ShardDataset([shard], self._stats())
            slices = [dataset[index]["slices"] for index in range(len(days))]
        self.assertEqual(
            [item["day_band"] for item in slices],
            ["days-1-7", "days-1-7", "days-8-14", "days-8-14",
             "days-15-21", "days-15-21", "days-22-plus", "days-22-plus"],
        )
        self.assertEqual(
            slices[0],
            {
                "actor_type": "farmer",
                "seat": "1",
                "day_band": "days-1-7",
                "source_date": "2026-08-21",
                "route_family": "route-z",
            },
        )

    def test_collate_examples_returns_typed_batch_and_provenance(self) -> None:
        # Catches dropping sample identity or emitting tensors with unusable dtypes.
        with tempfile.TemporaryDirectory() as directory:
            shard = self._write_game(Path(directory), "train", "batch", labels=np.arange(17))
            dataset = ShardDataset([shard], self._stats())
            batch = collate_examples([dataset[0], dataset[1]])
        self.assertEqual(batch.grid.shape, (2, GRID_CHANNELS, 10, 10))
        self.assertEqual(batch.global_features.shape, (2, GLOBAL_DIM))
        self.assertEqual(batch.actor_features.shape, (2, ACTOR_DIM))
        self.assertEqual(batch.clock_features.shape, (2, 8))
        self.assertEqual(batch.label.dtype, torch.int64)
        self.assertEqual(batch.game_id, ("batch", "batch"))
        self.assertEqual(len(batch.slices), 2)

    def _write_game(
        self, root: Path, split: str, episode_id: str, *, labels: np.ndarray
    ) -> Path:
        path = root / f"{episode_id}.npz"
        write_shard(self._game(split, episode_id, labels=labels), path)
        return path

    def _game(self, split: str, episode_id: str, *, labels: np.ndarray) -> EncodedGame:
        labels = np.asarray(labels, dtype=np.int64)
        sample_count = labels.shape[0]
        actor_features = np.zeros((sample_count, ACTOR_DIM), dtype=np.float32)
        actor_features[::2, 0] = 1.0
        step_index = np.arange(sample_count, dtype=np.int32)
        shapes = {
            "grid": [719, GRID_CHANNELS, 10, 10],
            "global_features": [719, GLOBAL_DIM],
            "actor_features": [sample_count, ACTOR_DIM],
            "step_index": [sample_count],
            "label": [sample_count],
        }
        return EncodedGame(
            grid=np.zeros((719, GRID_CHANNELS, 10, 10), dtype=np.float32),
            global_features=np.zeros((719, GLOBAL_DIM), dtype=np.float32),
            actor_features=actor_features,
            step_index=step_index,
            label=labels,
            argument_item=np.full(sample_count, -1, dtype=np.int32),
            argument_quantity=np.full(sample_count, -1, dtype=np.int32),
            metadata={
                "schema_version": "ryo-features-v0",
                "split": split,
                "episode_id": episode_id,
                "ryo_seat": 0,
                "source_path": f"{episode_id}.json",
                "source_sha256": "a" * 64,
                "source_date": "21-08",
                "route_family": "route-a",
                "sample_count": sample_count,
                "shapes": shapes,
            },
        )

    def _stats(self) -> NormalizationStats:
        return NormalizationStats(
            global_mean=np.zeros(GLOBAL_DIM, dtype=np.float32),
            global_std=np.ones(GLOBAL_DIM, dtype=np.float32),
            actor_mean=np.zeros(ACTOR_DIM, dtype=np.float32),
            actor_std=np.ones(ACTOR_DIM, dtype=np.float32),
        )

    def _tiny_game(self, episode_id: str) -> EncodedGame:
        return EncodedGame(
            grid=np.zeros((1, GRID_CHANNELS, 1, 1), dtype=np.float32),
            global_features=np.zeros((1, GLOBAL_DIM), dtype=np.float32),
            actor_features=np.zeros((1, ACTOR_DIM), dtype=np.float32),
            step_index=np.zeros(1, dtype=np.int32),
            label=np.zeros(1, dtype=np.int64),
            argument_item=np.full(1, -1, dtype=np.int32),
            argument_quantity=np.full(1, -1, dtype=np.int32),
            metadata={
                "episode_id": episode_id,
                "source_date": "21-08",
                "route_family": "route-a",
            },
        )

    def _artifact_values(
        self,
    ) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules]:
        stats = self._stats()
        counts = np.arange(1, 18, dtype=np.int64)
        weights = inverse_sqrt_class_weights(counts, cap=4.0)
        labels = np.repeat(np.arange(17), counts)
        actors = np.resize(np.array([1, 0], dtype=np.int64), labels.shape)
        return stats, counts, weights, fit_majority_rules(labels, actors)

    def _artifact_metadata(self, *, weight_cap: float) -> dict[str, object]:
        return {
            "schema_version": "ryo-bc-v0",
            "feature_schema_version": "ryo-features-v0",
            "operations": list(OPERATIONS),
            "train_shard_identities": ["a" * 64],
            "preparation_manifest_sha256": "b" * 64,
            "weight_cap": weight_cap,
        }

    def _artifact_arrays(
        self,
        stats: NormalizationStats,
        counts: np.ndarray,
        weights: np.ndarray,
        majority: MajorityRules,
        metadata: dict[str, object],
    ) -> dict[str, np.ndarray]:
        return {
            "global_mean": stats.global_mean,
            "global_std": stats.global_std,
            "actor_mean": stats.actor_mean,
            "actor_std": stats.actor_std,
            "class_counts": counts,
            "class_weights": weights,
            "majority_labels": np.asarray(
                [majority.global_label, majority.farmer_label, majority.hand_label],
                dtype=np.int64,
            ),
            "global_ranking": np.asarray(majority.global_ranking, dtype=np.int64),
            "farmer_ranking": np.asarray(majority.farmer_ranking, dtype=np.int64),
            "hand_ranking": np.asarray(majority.hand_ranking, dtype=np.int64),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        }


if __name__ == "__main__":
    unittest.main()
