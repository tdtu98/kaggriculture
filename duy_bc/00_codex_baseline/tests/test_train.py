import hashlib
import contextlib
import importlib.util
import io
import json
import random
import shutil
import tempfile
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS, load_config
from bc_core.dataset import (
    Batch,
    NormalizationStats,
    ShardDataset,
    load_train_artifacts,
    save_train_artifacts,
)
from bc_core.features import EncodedGame, logical_shard_identity, write_shard
from bc_core.checkpoints import (
    architecture_metadata,
    choose_device,
    save_checkpoint,
)
from bc_core.metrics import classification_report, slice_reports
from bc_core.train import (
    _append_history,
    _write_identity,
    evaluate_loader,
    evaluate_majority,
    fit_model,
    freeze_selection,
    is_better_epoch,
    make_loaders,
    seed_everything,
    train_one_epoch,
)
from model.clock import ClockOnlyModel
from model.majority import MajorityRules, fit_majority_rules
from model.state import StateAwareModel


def _batch(labels: tuple[int, ...] = (0, 1)) -> Batch:
    rows = len(labels)
    slices = tuple(
        {
            "actor_type": "farmer" if index % 2 == 0 else "hand",
            "seat": str(index % 2),
            "day_band": "days-1-7",
            "source_date": "21-08",
            "route_family": "route-a",
        }
        for index in range(rows)
    )
    return Batch(
        grid=torch.zeros(rows, 44, 10, 10),
        global_features=torch.zeros(rows, GLOBAL_DIM),
        actor_features=torch.zeros(rows, ACTOR_DIM),
        clock_features=torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / 10,
        label=torch.tensor(labels, dtype=torch.int64),
        game_id=tuple(f"game-{index}" for index in range(rows)),
        slices=slices,
    )


def _validation_report(loss: float = 1.0, correct_rows: int = 1) -> dict[str, object]:
    labels = np.arange(len(OPERATIONS), dtype=np.int64)
    predictions = (labels + 1) % len(OPERATIONS)
    predictions[:correct_rows] = labels[:correct_rows]
    logits = np.full((len(OPERATIONS), len(OPERATIONS)), -10.0, dtype=np.float32)
    logits[np.arange(len(OPERATIONS)), predictions] = 10.0
    rows = [
        {
            "actor_type": "farmer" if index % 2 == 0 else "hand",
            "seat": str(index % 2),
            "day_band": "days-1-7",
            "source_date": "21-08",
            "route_family": "route-a",
        }
        for index in range(len(OPERATIONS))
    ]
    return {
        "loss": loss,
        "metrics": classification_report(logits, labels),
        "slice_reports": slice_reports(logits, labels, rows),
        "logits": logits,
        "labels": labels,
        "game_ids": [f"game-{index}" for index in range(len(OPERATIONS))],
        "slices": rows,
    }


def _checkpoint_metadata(model: ClockOnlyModel) -> dict[str, object]:
    return {
        "schema_version": "ryo-bc-v0",
        "feature_schema_version": "ryo-features-v0",
        "vocabularies": {"operations": list(OPERATIONS)},
        "normalization": {
            "global_mean": [0.0] * GLOBAL_DIM,
            "global_std": [1.0] * GLOBAL_DIM,
            "actor_mean": [0.0] * ACTOR_DIM,
            "actor_std": [1.0] * ACTOR_DIM,
        },
        "class_weights": [1.0] * len(OPERATIONS),
        "manifest_sha256": "a" * 64,
        "architecture": architecture_metadata(model),
        "preparation_identity": "b" * 64,
        "train_artifact_identity": "c" * 64,
    }


def _config(max_epochs: int = 50) -> dict[str, object]:
    return {
        "schema_version": "ryo-bc-v0",
        "feature_schema_version": "ryo-features-v0",
        "seed": 20260824,
        "corpus_root": "unused",
        "module_version": "1.32.7",
        "training": {
            "learning_rate": 0.001,
            "batch_size": 2,
            "max_epochs": max_epochs,
            "patience": 5,
            "weight_cap": 4.0,
        },
        "bootstrap_resamples": 10000,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _rewrite_training_audit(path: Path, audit: dict[str, object]) -> None:
    audit.pop("training_identity", None)
    audit["training_identity"] = hashlib.sha256(
        _canonical_json_bytes(audit)
    ).hexdigest()
    path.write_text(json.dumps(audit))


def _filesystem_snapshot(root: Path) -> tuple[tuple[str, str, bytes], ...]:
    if not root.exists():
        return ()
    entries: list[tuple[str, str, bytes]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append((relative, "symlink", str(path.readlink()).encode()))
        elif path.is_dir():
            entries.append((relative, "directory", b""))
        else:
            entries.append((relative, "file", path.read_bytes()))
    return tuple(entries)


def _rewrite_epoch_record(
    model_dir: Path,
    epoch: int,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    history_path = model_dir / "epochs.jsonl"
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    mutate(records[epoch - 1])
    history_path.write_bytes(
        b"".join(_canonical_json_bytes(record) + b"\n" for record in records)
    )
    (model_dir / f"epoch-{epoch:03d}.json").write_bytes(
        _canonical_json_bytes(records[epoch - 1]) + b"\n"
    )


def _prune_model_history(model_dir: Path, completed_epochs: int) -> None:
    records = [
        json.loads(line)
        for line in (model_dir / "epochs.jsonl").read_text().splitlines()
    ][:completed_epochs]
    for epoch in range(completed_epochs + 1, 7):
        for suffix in (".pt", ".pt.sha256", ".json"):
            (model_dir / f"epoch-{epoch:03d}{suffix}").unlink()
    (model_dir / "epochs.jsonl").write_bytes(
        b"".join(_canonical_json_bytes(record) + b"\n" for record in records)
    )


def _encoded_game(
    split: str, episode_id: str, labels: np.ndarray, source_sha256: str
) -> EncodedGame:
    sample_count = int(labels.shape[0])
    shapes = {
        "grid": [719, GRID_CHANNELS, 10, 10],
        "global_features": [719, GLOBAL_DIM],
        "actor_features": [sample_count, ACTOR_DIM],
        "step_index": [sample_count],
        "label": [sample_count],
    }
    return EncodedGame(
        grid=np.zeros(tuple(shapes["grid"]), dtype=np.float32),
        global_features=np.zeros(tuple(shapes["global_features"]), dtype=np.float32),
        actor_features=np.zeros(tuple(shapes["actor_features"]), dtype=np.float32),
        step_index=np.arange(sample_count, dtype=np.int32),
        label=labels.astype(np.int64, copy=False),
        argument_item=np.full(sample_count, -1, dtype=np.int32),
        argument_quantity=np.full(sample_count, -1, dtype=np.int32),
        metadata={
            "schema_version": "ryo-features-v0",
            "split": split,
            "episode_id": episode_id,
            "ryo_seat": 0,
            "source_path": f"sources/{episode_id}.json",
            "source_sha256": source_sha256,
            "source_date": "21-08",
            "route_family": "route-a",
            "sample_count": sample_count,
            "shapes": shapes,
        },
    )


def _locality_game(episode_id: str, sample_count: int) -> EncodedGame:
    actor_features = np.zeros((sample_count, ACTOR_DIM), dtype=np.float32)
    actor_features[:, 4] = np.arange(sample_count, dtype=np.float32)
    return EncodedGame(
        grid=np.zeros((1, GRID_CHANNELS, 1, 1), dtype=np.float32),
        global_features=np.zeros((1, GLOBAL_DIM), dtype=np.float32),
        actor_features=actor_features,
        step_index=np.zeros(sample_count, dtype=np.int32),
        label=np.arange(sample_count, dtype=np.int64) % len(OPERATIONS),
        argument_item=np.full(sample_count, -1, dtype=np.int32),
        argument_quantity=np.full(sample_count, -1, dtype=np.int32),
        metadata={
            "episode_id": episode_id,
            "source_date": "21-08",
            "route_family": "route-a",
        },
    )


class _Rows(torch.utils.data.Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        batch = _batch((index % 2,))
        return {
            "grid": batch.grid[0].numpy(),
            "global_features": batch.global_features[0].numpy(),
            "actor_features": batch.actor_features[0].numpy(),
            "clock_features": batch.clock_features[0].numpy(),
            "label": np.int64(batch.label[0]),
            "game_id": batch.game_id[0],
            "slices": batch.slices[0],
        }


class _NaNLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits.sum() * torch.tensor(float("nan"), device=logits.device)


class TrainTest(unittest.TestCase):
    def test_macro_f1_wins_and_loss_breaks_ties(self) -> None:
        self.assertTrue(is_better_epoch(0.31, 1.2, 0.30, 0.8))
        self.assertTrue(is_better_epoch(0.30, 0.7, 0.30, 0.8))
        self.assertFalse(is_better_epoch(0.30, 0.9, 0.30, 0.8))

    def test_one_epoch_updates_clock_model(self) -> None:
        model = ClockOnlyModel()
        before = [value.detach().clone() for value in model.parameters()]
        loss = train_one_epoch(
            model,
            [_batch()],
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            torch.nn.CrossEntropyLoss(weight=torch.ones(len(OPERATIONS))),
            torch.device("cpu"),
            "clock",
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        )

    def test_nonfinite_training_loss_raises_before_optimizer_step(self) -> None:
        model = ClockOnlyModel()
        before = [value.detach().clone() for value in model.parameters()]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            train_one_epoch(
                model,
                [_batch()],
                torch.optim.AdamW(model.parameters(), lr=1e-3),
                _NaNLoss(),
                torch.device("cpu"),
                "clock",
            )
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_evaluation_returns_weighted_loss_metrics_slices_and_game_ids(self) -> None:
        model = ClockOnlyModel()
        report = evaluate_loader(
            model,
            [_batch((0,)), _batch((1, 2, 3))],
            torch.nn.CrossEntropyLoss(weight=torch.ones(len(OPERATIONS))),
            torch.device("cpu"),
            "clock",
        )
        self.assertEqual(report["labels"].tolist(), [0, 1, 2, 3])
        self.assertEqual(report["game_ids"], ["game-0", "game-0", "game-1", "game-2"])
        self.assertEqual(set(report["slice_reports"]), {"actor_type", "seat", "day_band", "source_date", "route_family"})
        self.assertEqual(report["logits"].shape, (4, len(OPERATIONS)))
        self.assertTrue(np.isfinite(report["loss"]))

    def test_seeded_loaders_shuffle_only_training_with_zero_workers(self) -> None:
        train_a, val_a = make_loaders(_Rows(), _Rows(), batch_size=2, seed=91)
        train_b, val_b = make_loaders(_Rows(), _Rows(), batch_size=2, seed=91)
        self.assertEqual(train_a.num_workers, 0)
        self.assertEqual(val_a.num_workers, 0)
        self.assertEqual(type(train_a.sampler).__name__, "RandomSampler")
        self.assertEqual(type(val_a.sampler).__name__, "SequentialSampler")
        self.assertEqual(
            [batch.label.tolist() for batch in train_a],
            [batch.label.tolist() for batch in train_b],
        )
        self.assertEqual(
            [batch.label.tolist() for batch in val_a],
            [[0, 1], [0, 1]],
        )

    def test_shard_local_training_shuffle_is_deterministic_complete_and_cache_bounded(
        self,
    ) -> None:
        # Catches global row shuffling thrashing the binding two-game shard cache.
        shard_count = 8
        rows_per_shard = 128
        paths = tuple(Path(f"locality-{index}.npz") for index in range(shard_count))
        games = {
            path: _locality_game(f"game-{index}", rows_per_shard)
            for index, path in enumerate(paths)
        }
        stats = NormalizationStats(
            global_mean=np.zeros(GLOBAL_DIM, dtype=np.float32),
            global_std=np.ones(GLOBAL_DIM, dtype=np.float32),
            actor_mean=np.zeros(ACTOR_DIM, dtype=np.float32),
            actor_std=np.ones(ACTOR_DIM, dtype=np.float32),
        )

        def epoch(
            train_loader: Any, seed: int
        ) -> tuple[list[tuple[str, int]], int, list[int]]:
            self.assertIsInstance(train_loader.generator, torch.Generator)
            train_loader.generator.manual_seed(seed)
            reader.reset_mock()
            order: list[tuple[str, int]] = []
            batch_sizes: list[int] = []
            for batch in train_loader:
                batch_sizes.append(int(batch.label.shape[0]))
                order.extend(
                    (game_id, int(local_index))
                    for game_id, local_index in zip(
                        batch.game_id, batch.actor_features[:, 4].tolist()
                    )
                )
            return order, reader.call_count, batch_sizes

        with patch(
            "bc_core.dataset.read_shard", side_effect=lambda path: games[path]
        ) as reader:
            dataset = ShardDataset(paths, stats)
            train_loader, _ = make_loaders(
                dataset, _Rows(), batch_size=32, seed=91
            )
            order_a, loads_a, batch_sizes_a = epoch(train_loader, 91)
            order_b, loads_b, batch_sizes_b = epoch(train_loader, 91)
            order_c, loads_c, batch_sizes_c = epoch(train_loader, 92)

        expected = {
            (f"game-{shard}", row)
            for shard in range(shard_count)
            for row in range(rows_per_shard)
        }
        natural = [
            (f"game-{shard}", row)
            for shard in range(shard_count)
            for row in range(rows_per_shard)
        ]
        self.assertEqual(order_a, order_b)
        self.assertNotEqual(order_a, order_c)
        self.assertNotEqual(order_a, natural)
        self.assertEqual(set(order_a), expected)
        self.assertEqual(len(order_a), len(expected))
        for shard in range(shard_count):
            local_rows = [
                row for game_id, row in order_a if game_id == f"game-{shard}"
            ]
            self.assertEqual(sorted(local_rows), list(range(rows_per_shard)))
            self.assertNotEqual(local_rows, list(range(rows_per_shard)))
        self.assertEqual(
            sum(left != right for left, right in zip(
                (game_id for game_id, _ in order_a),
                (game_id for game_id, _ in order_a[1:]),
            )),
            shard_count - 1,
        )
        self.assertGreater(min(loads_a, loads_b, loads_c), 0)
        self.assertLessEqual(max(loads_a, loads_b, loads_c), shard_count)
        expected_batch_sizes = [32] * (len(expected) // 32)
        self.assertEqual(batch_sizes_a, expected_batch_sizes)
        self.assertEqual(batch_sizes_b, expected_batch_sizes)
        self.assertEqual(batch_sizes_c, expected_batch_sizes)
        self.assertEqual(train_loader.num_workers, 0)

    def test_majority_validation_reports_global_and_actor_stratified_rules(self) -> None:
        ascending = tuple(range(17))
        farmer = (1, 0, *range(2, 17))
        hand = (2, 0, 1, *range(3, 17))
        rules = MajorityRules(0, 1, 2, ascending, farmer, hand)
        report = evaluate_majority([_batch((1, 2))], rules)
        self.assertEqual(set(report), {"global", "actor"})
        self.assertEqual(report["actor"]["metrics"]["top1"], 1.0)
        self.assertEqual(report["global"]["metrics"]["top1"], 0.0)
        self.assertEqual(set(report["actor"]["slice_reports"]), {"actor_type", "seat", "day_band", "source_date", "route_family"})

    def test_cli_rejects_nontrainable_audit_before_creating_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            runs_root = root / "runs"
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            audit = self._training_fixture(data_root, config_path)
            audit["trainable"] = False
            audit["all_checks_passed"] = False
            audit["checks"]["operation_support_covered"] = False
            (data_root / "training_audit.json").write_text(json.dumps(audit))
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                status = self._load_cli().main([
                    "--config", str(config_path), "--run-id", "bad-run",
                    "--data-root", str(data_root), "--runs-root", str(runs_root),
                ])
            self.assertNotEqual(status, 0)
            self.assertIn("not trainable", error.getvalue())
            self.assertFalse((runs_root / "bad-run").exists())

    def test_training_uses_safe_audit_without_opening_full_audit(self) -> None:
        # Catches opening audit.json or allowing test-only changes into training identity.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            runs_root = root / "runs"
            audit = self._training_fixture(data_root, config_path)
            (data_root / "audit.json").mkdir()
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=RuntimeError("reached train-only fitting"),
            ), self.assertRaisesRegex(RuntimeError, "reached train-only fitting"):
                cli.train_run(config_path, "safe", data_root, runs_root)
            first_identity = audit["training_identity"]
            (data_root / "audit.json" / "test-only").write_text("changed")
            loaded = cli._load_training_audit(
                data_root / "training_audit.json", load_config(config_path)
            )
        self.assertEqual(loaded["training_identity"], first_identity)

    def test_training_audit_rejects_even_self_consistent_test_fields(self) -> None:
        # Catches accepting a recomputed audit identity that structurally embeds test data.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            audit = self._training_fixture(data_root, config_path)
            audit["test"] = {"shard_path": "test/hidden.npz", "label_counts": [1]}
            audit.pop("training_identity")
            audit["training_identity"] = hashlib.sha256(
                json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            (data_root / "training_audit.json").write_text(json.dumps(audit))
            cli = self._load_cli()
            with self.assertRaisesRegex(ValueError, "training audit"):
                cli._load_training_audit(
                    data_root / "training_audit.json", load_config(config_path)
                )

    def test_training_audit_rejects_self_consistent_nested_test_check_everywhere(self) -> None:
        # Catches open nested audit dictionaries at both training consumers.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            run_dir, selected, _, artifacts, audit_path = self._selection_fixture(root)
            audit = json.loads(audit_path.read_text())
            audit["checks"]["test_counts_validated"] = True
            _rewrite_training_audit(audit_path, audit)
            cli = self._load_cli()
            with self.assertRaisesRegex(ValueError, "training audit"):
                cli._load_training_audit(audit_path, load_config(config_path))
            with self.assertRaisesRegex(ValueError, "training audit"):
                freeze_selection(run_dir, selected, config_path, artifacts, audit_path)
            self.assertFalse((run_dir / "selection.json").exists())

    def test_training_audit_rejects_self_consistent_test_source_path(self) -> None:
        # Catches a safe record being relabeled while retaining a test-derived source path.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            audit = self._training_fixture(data_root, config_path)
            audit["shards"][0]["source_path"] = "test/hidden.json"
            safe_manifest = {
                "schema_version": "ryo-training-preparation-v0",
                "records": audit["shards"],
            }
            audit["safe_manifest_sha256"] = hashlib.sha256(
                _canonical_json_bytes(safe_manifest)
            ).hexdigest()
            audit_path = data_root / "training_audit.json"
            _rewrite_training_audit(audit_path, audit)
            with self.assertRaisesRegex(ValueError, "training audit.*test"):
                self._load_cli()._load_training_audit(
                    audit_path, load_config(config_path)
                )

    def test_actual_shards_must_exactly_match_safe_audit_before_run_creation(self) -> None:
        # Catches substituted, misplaced, missing, or extra valid shard files.
        config_path = Path(__file__).parents[1] / "configs" / "v0.json"
        cli = self._load_cli()
        mutations = ("substitute", "train-at-val", "missing", "extra", "nested-extra")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / "data"
                runs_root = root / "runs"
                self._training_fixture(data_root, config_path)
                if mutation == "substitute":
                    replacement = _encoded_game(
                        "train", "train-game", np.arange(17)[::-1], "a" * 64
                    )
                    candidate = data_root / "replacement.npz"
                    write_shard(replacement, candidate)
                    candidate.replace(data_root / "train" / "train-game.npz")
                elif mutation == "train-at-val":
                    replacement = _encoded_game(
                        "train", "val-game", np.array([0]), "b" * 64
                    )
                    candidate = data_root / "replacement.npz"
                    write_shard(replacement, candidate)
                    candidate.replace(data_root / "val" / "val-game.npz")
                elif mutation == "missing":
                    (data_root / "val" / "val-game.npz").unlink()
                elif mutation == "extra":
                    write_shard(
                        _encoded_game("val", "extra", np.array([0]), "c" * 64),
                        data_root / "val" / "extra.npz",
                    )
                else:
                    (data_root / "val" / "nested").mkdir()
                    write_shard(
                        _encoded_game("val", "extra", np.array([0]), "c" * 64),
                        data_root / "val" / "nested" / "extra.npz",
                    )
                with self.assertRaisesRegex(ValueError, "shard"):
                    cli.train_run(config_path, "rejected", data_root, runs_root)
                self.assertFalse((runs_root / "rejected").exists())

    def test_actual_shards_reject_extra_symlink_alias_without_collapsing_paths(self) -> None:
        # Catches resolving an extra val path onto an audited train path before set comparison.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            audit = self._training_fixture(data_root, config_path)
            (data_root / "val" / "train-alias.npz").symlink_to(
                Path("..") / "train" / "train-game.npz"
            )
            with self.assertRaisesRegex(ValueError, "symlink|shard set"):
                self._load_cli()._verified_shards(audit, data_root)

    def test_actual_shards_reject_nested_symlink_directory_without_following_it(self) -> None:
        # Catches relying on rglob's platform-specific traversal of directory symlinks.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            audit = self._training_fixture(data_root, config_path)
            cli = self._load_cli()
            verified = cli._verified_shards(audit, data_root)
            self.assertEqual(
                {
                    path.relative_to(data_root.resolve()).as_posix()
                    for path in verified["val"]
                },
                {"val/val-game.npz"},
            )
            (data_root / "val" / "aliasdir").symlink_to(Path("..") / "train")
            with self.assertRaisesRegex(ValueError, "symlink"):
                cli._verified_shards(audit, data_root)

    def test_fit_orphan_preflight_leaves_fresh_run_snapshot_unchanged(self) -> None:
        # Catches publishing a model directory identity before orphan validation.
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            model_dir = run_dir / "clock"
            model_dir.mkdir(parents=True)
            (model_dir / "epoch-099.json").write_text('{"epoch":99}\n')
            before = _filesystem_snapshot(run_dir)
            with self.assertRaisesRegex(ValueError, "artifact|epoch"):
                fit_model(
                    "clock",
                    [_batch()],
                    [_batch()],
                    np.ones(17, np.float32),
                    _config(2),
                    run_dir,
                    _checkpoint_metadata(ClockOnlyModel()),
                )
            self.assertEqual(_filesystem_snapshot(run_dir), before)

    def test_cli_orphan_model_preflight_leaves_run_snapshot_unchanged(self) -> None:
        # Catches fitting/saving global artifacts before either model subrun is preflighted.
        config_path = Path(__file__).parents[1] / "configs" / "v0.json"
        for model_name in ("clock", "state"):
            with self.subTest(model_name=model_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / "data"
                runs_root = root / "runs"
                self._training_fixture(data_root, config_path)
                run_dir = runs_root / "orphan"
                model_dir = run_dir / model_name
                model_dir.mkdir(parents=True)
                (model_dir / "epoch-099.json").write_text('{"epoch":99}\n')
                before = _filesystem_snapshot(run_dir)
                cli = self._load_cli()
                real_fit_model = cli.fit_model

                def bypass_clock(name: str, *args: object, **kwargs: object) -> object:
                    if name == "clock":
                        return {"best_checkpoint": run_dir / "unused-clock.pt"}
                    return real_fit_model(name, *args, **kwargs)

                fit_patch = (
                    patch.object(cli, "fit_model", side_effect=bypass_clock)
                    if model_name == "state"
                    else contextlib.nullcontext()
                )
                with fit_patch, self.assertRaisesRegex(ValueError, "artifact|epoch"):
                    cli.train_run(config_path, "orphan", data_root, runs_root)
                self.assertEqual(_filesystem_snapshot(run_dir), before)

    def test_cli_fresh_run_compatibility_preflight_happens_before_artifact_fit(self) -> None:
        # Catches discovering an incompatible overall run only after train-only fitting.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).parents[1] / "configs" / "v0.json"
            data_root = root / "data"
            runs_root = root / "runs"
            self._training_fixture(data_root, config_path)
            run_dir = runs_root / "occupied"
            run_dir.mkdir(parents=True)
            (run_dir / "train_artifacts.npz").write_bytes(b"orphan")
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("artifact fitting reached before preflight"),
            ), self.assertRaisesRegex(ValueError, "run|artifact"):
                cli.train_run(config_path, "occupied", data_root, runs_root)
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_resume_rejects_wrong_state_identity_before_any_fit(self) -> None:
        # Catches validating the second model identity only after the clock fit mutates state.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(
                root, model_identities=True
            )
            state_identity = run_dir / "state" / "run-identity.json"
            identity = json.loads(state_identity.read_text())
            identity["checkpoint_metadata"]["preparation_identity"] = "f" * 64
            state_identity.write_bytes(_canonical_json_bytes(identity) + b"\n")
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), patch.object(
                cli,
                "preflight_resumed_model",
                side_effect=AssertionError(
                    "model history evaluation started before both identities passed"
                ),
            ), self.assertRaisesRegex(ValueError, "identity|configuration"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_resume_rejects_state_prefix_before_clock_completion(self) -> None:
        # Catches accepting an unreachable phase where state exists before clock completes.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(
                root, model_identities=True
            )
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaisesRegex(ValueError, "state machine|clock"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_resume_rejects_semantic_train_artifact_before_any_fit(self) -> None:
        # Catches using the fitting/saving path to discover a plausible but wrong archive.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(root)
            artifacts_path = run_dir / "train_artifacts.npz"
            stats, counts, weights, majority, metadata = load_train_artifacts(
                artifacts_path
            )
            wrong_mean = stats.global_mean.copy()
            wrong_mean[0] = 1.0
            candidate = run_dir / "candidate.npz"
            save_train_artifacts(
                candidate,
                NormalizationStats(
                    wrong_mean,
                    stats.global_std,
                    stats.actor_mean,
                    stats.actor_std,
                ),
                counts,
                weights,
                majority,
                metadata,
            )
            candidate.replace(artifacts_path)
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaisesRegex(ValueError, "train artifact"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_resume_rejects_corrupt_majority_report_before_any_fit(self) -> None:
        # Catches accepting any canonical JSON at the majority-report topology slot.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(root)
            majority_path = run_dir / "majority.validation.json"
            majority = json.loads(majority_path.read_text())
            majority["global"]["metrics"]["top1"] = 0.5
            majority_path.write_text(json.dumps(majority, indent=2, sort_keys=True) + "\n")
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaisesRegex(ValueError, "majority"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_completed_resume_is_idempotent_and_rejects_bad_selection_before_fit(
        self,
    ) -> None:
        # Catches fitting both models before validating an already-published selection.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, selection = (
                self._cli_resume_fixture(root, complete_models=True, selection=True)
            )
            assert selection is not None
            cli = self._load_cli()
            before = _filesystem_snapshot(root)
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ):
                resumed = cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(resumed, selection)
            self.assertEqual(_filesystem_snapshot(root), before)

            selection_path = run_dir / "selection.json"
            canonical_selection = selection_path.read_bytes()
            selection_path.write_bytes(canonical_selection + b"\n")
            before_noncanonical = _filesystem_snapshot(root)
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaisesRegex(
                RuntimeError, "selection.*canonical|canonical.*selection"
            ):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before_noncanonical)
            selection_path.write_bytes(canonical_selection)

            corrupted = json.loads(selection_path.read_text())
            corrupted["models"]["state"]["epoch"] = 99
            identity_payload = dict(corrupted)
            identity_payload.pop("created_at")
            identity_payload.pop("selection_identity")
            corrupted["selection_identity"] = hashlib.sha256(
                _canonical_json_bytes(identity_payload)
            ).hexdigest()
            selection_path.write_text(
                json.dumps(corrupted, indent=2, sort_keys=True) + "\n"
            )
            before_corrupt = _filesystem_snapshot(root)
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaisesRegex((ValueError, RuntimeError), "selection|identity"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before_corrupt)

    def test_cli_valid_baseline_resume_uses_frozen_artifacts_before_model_fit(self) -> None:
        # Catches refitting train artifacts on the coherent baseline-only resume phase.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(root)
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=RuntimeError("model fitting reached"),
            ), self.assertRaisesRegex(RuntimeError, "model fitting reached"):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)

    def test_cli_active_resume_rejects_negative_train_loss_in_read_only_preflight(
        self,
    ) -> None:
        # Catches accepting a corrupt active prefix until after a new epoch is published.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(
                root, complete_models=True
            )
            shutil.rmtree(run_dir / "state")
            _prune_model_history(run_dir / "clock", 1)
            _rewrite_epoch_record(
                run_dir / "clock",
                1,
                lambda record: record.__setitem__("train_loss", -1.0),
            )
            before = _filesystem_snapshot(root)
            cli = self._load_cli()
            with patch.object(
                cli,
                "fit_train_artifacts",
                side_effect=AssertionError("train artifact fitting must not run"),
            ), patch.object(
                cli,
                "fit_model",
                side_effect=AssertionError("model fitting must not run"),
            ), self.assertRaises(ValueError):
                cli.train_run(
                    config_path, "resume", data_root, runs_root, resume=True
                )
            self.assertEqual(_filesystem_snapshot(root), before)
            self.assertFalse((run_dir / "clock" / "epoch-002.pt").exists())

    def test_cli_completed_histories_reject_invalid_numbers_and_reports_in_preflight(
        self,
    ) -> None:
        # Catches reaching fit_model for a corrupt models-complete/no-selection state.
        mutations = {
            "negative-train-loss": lambda record: record.__setitem__(
                "train_loss", -1.0
            ),
            "nonfinite-train-loss": lambda record: record.__setitem__(
                "train_loss", float("inf")
            ),
            "boolean-train-loss": lambda record: record.__setitem__(
                "train_loss", True
            ),
            "negative-validation-loss": lambda record: record.__setitem__(
                "validation_loss", -1.0
            ),
            "out-of-range-top1": lambda record: record["validation_metrics"].__setitem__(  # type: ignore[index,union-attr]
                "top1", 1.1
            ),
            "negative-support": lambda record: record["validation_metrics"][  # type: ignore[index]
                "per_class"
            ][0].__setitem__("support", -1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path, data_root, runs_root, run_dir, _ = (
                    self._cli_resume_fixture(root, complete_models=True)
                )
                _rewrite_epoch_record(run_dir / "clock", 1, mutate)
                before = _filesystem_snapshot(root)
                cli = self._load_cli()
                with patch.object(
                    cli,
                    "fit_train_artifacts",
                    side_effect=AssertionError("train artifact fitting must not run"),
                ), patch.object(
                    cli,
                    "fit_model",
                    side_effect=AssertionError("model fitting must not run"),
                ), self.assertRaises(ValueError):
                    cli.train_run(
                        config_path, "resume", data_root, runs_root, resume=True
                    )
                self.assertEqual(_filesystem_snapshot(root), before)
                self.assertFalse((run_dir / "selection.json").exists())

    def test_cli_valid_models_complete_resume_freezes_selection_without_new_epochs(
        self,
    ) -> None:
        # Catches a strict preflight rejecting its own canonical completed histories.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, data_root, runs_root, run_dir, _ = self._cli_resume_fixture(
                root, complete_models=True
            )
            checkpoint_bytes = {
                path.relative_to(run_dir): path.read_bytes()
                for path in run_dir.glob("*/epoch-*.pt")
            }
            selection = self._load_cli().train_run(
                config_path, "resume", data_root, runs_root, resume=True
            )
            self.assertEqual(
                selection,
                json.loads((run_dir / "selection.json").read_text()),
            )
            self.assertEqual(
                checkpoint_bytes,
                {
                    path.relative_to(run_dir): path.read_bytes()
                    for path in run_dir.glob("*/epoch-*.pt")
                },
            )

    def test_fit_stops_after_exactly_five_completed_non_improving_epochs(self) -> None:
        fixed_report = _validation_report()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bc_core.train.evaluate_loader", return_value=fixed_report
        ):
            run_dir = Path(temporary) / "run-a"
            result = fit_model(
                "clock",
                [_batch()],
                [_batch()],
                np.ones(len(OPERATIONS), dtype=np.float32),
                _config(),
                run_dir,
                _checkpoint_metadata(ClockOnlyModel()),
            )
            lines = (run_dir / "clock" / "epochs.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_completed"], 6)
        for index, line in enumerate(lines, 1):
            record = json.loads(line)
            self.assertEqual(record["epoch"], index)
            self.assertEqual(line, json.dumps(record, sort_keys=True, separators=(",", ":")))

    def test_resume_restores_epoch_optimizer_and_rng_and_appends_history(self) -> None:
        reports = [
            _validation_report(loss=1.0 - index / 100, correct_rows=index)
            for index in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run-a"
            real_train_one_epoch = train_one_epoch
            train_calls = 0

            def interrupt_after_two(*args: object, **kwargs: object) -> float:
                nonlocal train_calls
                train_calls += 1
                if train_calls == 3:
                    raise RuntimeError("simulated interruption")
                return real_train_one_epoch(*args, **kwargs)

            with patch("bc_core.train.evaluate_loader", side_effect=reports[:2]), patch(
                "bc_core.train.train_one_epoch", side_effect=interrupt_after_two
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    fit_model(
                        "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                        _config(max_epochs=3), run_dir, _checkpoint_metadata(ClockOnlyModel())
                    )
            saved = torch.load(
                run_dir / "clock" / "epoch-002.pt",
                map_location="cpu",
                weights_only=False,
            )
            python_probe = random.Random()
            python_probe.setstate(saved["rng_state"]["python"])
            expected_python = python_probe.random()
            numpy_probe = np.random.RandomState()
            numpy_probe.set_state(saved["rng_state"]["numpy"])
            expected_numpy = float(numpy_probe.random_sample())
            torch_probe = torch.Generator(device="cpu")
            torch_probe.set_state(saved["rng_state"]["torch"])
            expected_torch = float(torch.rand((), generator=torch_probe))
            expected_optimizer = saved["optimizer_state_dict"]
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)

            def observe_restored_state(
                model: torch.nn.Module,
                loader: object,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                device: torch.device,
                model_name: str,
            ) -> float:
                actual_optimizer = optimizer.state_dict()
                first_key = next(iter(expected_optimizer["state"]))
                self.assertEqual(
                    float(actual_optimizer["state"][first_key]["step"]),
                    float(expected_optimizer["state"][first_key]["step"]),
                )
                torch.testing.assert_close(
                    actual_optimizer["state"][first_key]["exp_avg"],
                    expected_optimizer["state"][first_key]["exp_avg"],
                )
                self.assertEqual(random.random(), expected_python)
                self.assertEqual(float(np.random.random()), expected_numpy)
                self.assertEqual(float(torch.rand(())), expected_torch)
                return real_train_one_epoch(
                    model, loader, optimizer, criterion, device, model_name
                )

            with patch("bc_core.train.evaluate_loader", side_effect=reports), patch(
                "bc_core.train.train_one_epoch", side_effect=observe_restored_state
            ):
                resumed = fit_model(
                    "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                    _config(max_epochs=3), run_dir, _checkpoint_metadata(ClockOnlyModel()),
                    resume=True,
                )
            records = [json.loads(line) for line in (run_dir / "clock" / "epochs.jsonl").read_text().splitlines()]
        self.assertEqual(resumed["epochs_completed"], 3)
        self.assertEqual([record["epoch"] for record in records], [1, 2, 3])
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["resume_epoch"], 2)

    def test_resume_rejects_a_different_run_configuration(self) -> None:
        fixed_report = _validation_report()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bc_core.train.evaluate_loader", return_value=fixed_report
        ):
            run_dir = Path(temporary) / "run-a"
            fit_model("clock", [_batch()], [_batch()], np.ones(17, np.float32), _config(1), run_dir, _checkpoint_metadata(ClockOnlyModel()))
            changed = _config(2)
            changed["seed"] = 7
            with self.assertRaisesRegex(ValueError, "configuration"):
                fit_model("clock", [_batch()], [_batch()], np.ones(17, np.float32), changed, run_dir, _checkpoint_metadata(ClockOnlyModel()), resume=True)

    def test_fit_model_resume_rejects_negative_train_loss_without_writes(self) -> None:
        # Catches a direct resume bypassing the strict whole-run CLI preflight.
        report = _validation_report()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bc_core.train.evaluate_loader", return_value=report
        ):
            run_dir = Path(temporary) / "run"
            fit_model(
                "clock",
                [_batch()],
                [_batch()],
                np.ones(17, np.float32),
                _config(1),
                run_dir,
                _checkpoint_metadata(ClockOnlyModel()),
            )
            _rewrite_epoch_record(
                run_dir / "clock",
                1,
                lambda record: record.__setitem__("train_loss", -1.0),
            )
            before = _filesystem_snapshot(run_dir)
            with patch(
                "bc_core.train.train_one_epoch",
                side_effect=AssertionError("training must not start"),
            ), self.assertRaises(ValueError):
                fit_model(
                    "clock",
                    [_batch()],
                    [_batch()],
                    np.ones(17, np.float32),
                    _config(1),
                    run_dir,
                    _checkpoint_metadata(ClockOnlyModel()),
                    resume=True,
                )
            self.assertEqual(_filesystem_snapshot(run_dir), before)

    def test_resume_rejects_tampered_metric_counter_path_and_hash_before_append(self) -> None:
        # Catches deriving resume selection/patience from editable JSONL fields.
        report = _validation_report()
        mutations = {
            "metric": lambda record: record["validation_metrics"].__setitem__("macro_f1", 0.9),
            "counter": lambda record: record.__setitem__("non_improving_epochs", 4),
            "path": lambda record: record.__setitem__("checkpoint_path", "/tmp/other.pt"),
            "hash": lambda record: record.__setitem__("checkpoint_sha256", "f" * 64),
        }
        real_train = train_one_epoch
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                calls = 0

                def interrupt(*args: object, **kwargs: object) -> float:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("stop after epoch one")
                    return real_train(*args, **kwargs)

                with patch("bc_core.train.evaluate_loader", return_value=report), patch(
                    "bc_core.train.train_one_epoch", side_effect=interrupt
                ), self.assertRaisesRegex(RuntimeError, "stop after epoch one"):
                    fit_model(
                        "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                        _config(2), run_dir, _checkpoint_metadata(ClockOnlyModel())
                    )
                history = run_dir / "clock" / "epochs.jsonl"
                record = json.loads(history.read_text().strip())
                mutate(record)
                history.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                before = history.read_bytes()
                with patch("bc_core.train.evaluate_loader", return_value=report), self.assertRaisesRegex(
                    ValueError, "history|checkpoint"
                ):
                    fit_model(
                        "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                        _config(2), run_dir, _checkpoint_metadata(ClockOnlyModel()), resume=True
                    )
                self.assertEqual(history.read_bytes(), before)
                self.assertFalse((run_dir / "clock" / "epoch-002.pt").exists())

    def test_resume_rejects_future_epoch_record_before_creating_checkpoint(self) -> None:
        # Catches preflight that checks checkpoint files but overlooks immutable epoch records.
        report = _validation_report()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            calls = 0
            real_train = train_one_epoch

            def interrupt(*args: object, **kwargs: object) -> float:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("stop after epoch one")
                return real_train(*args, **kwargs)

            with patch("bc_core.train.evaluate_loader", return_value=report), patch(
                "bc_core.train.train_one_epoch", side_effect=interrupt
            ), self.assertRaisesRegex(RuntimeError, "stop after epoch one"):
                fit_model(
                    "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                    _config(2), run_dir, _checkpoint_metadata(ClockOnlyModel())
                )
            future_record = run_dir / "clock" / "epoch-099.json"
            future_record.write_text('{"epoch":99}\n')
            before = _filesystem_snapshot(run_dir)
            with patch(
                "bc_core.train.train_one_epoch",
                side_effect=AssertionError("training started before preflight"),
            ), patch("bc_core.train.evaluate_loader", return_value=report), self.assertRaisesRegex(
                ValueError, "artifact|epoch"
            ):
                fit_model(
                    "clock", [_batch()], [_batch()], np.ones(17, np.float32),
                    _config(2), run_dir, _checkpoint_metadata(ClockOnlyModel()), resume=True
                )
            self.assertEqual(_filesystem_snapshot(run_dir), before)
            self.assertFalse((run_dir / "clock" / "epoch-002.pt").exists())
            self.assertFalse((run_dir / "clock" / "epoch-002.pt.sha256").exists())

    def test_freeze_selection_hashes_all_inputs_and_accepts_only_identical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, selected, config, artifacts, audit = self._selection_fixture(root)
            clock, state = selected["clock"], selected["state"]
            selected = {"clock": clock, "state": state}
            frozen = freeze_selection(run_dir, selected, config, artifacts, audit)
            rerun = freeze_selection(run_dir, selected, config, artifacts, audit)
            self.assertEqual(frozen, rerun)
            self.assertEqual(frozen["run_id"], "run-a")
            self.assertEqual(frozen["models"]["clock"]["sha256"], hashlib.sha256(clock.read_bytes()).hexdigest())
            self.assertEqual(frozen["models"]["state"]["sha256"], hashlib.sha256(state.read_bytes()).hexdigest())
            self.assertEqual(frozen["models"]["clock"]["sidecar"]["canonical"], (clock.with_name(clock.name + ".sha256")).read_text().strip())
            self.assertEqual(frozen["config"]["sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertEqual(frozen["train_artifacts"]["sha256"], hashlib.sha256(artifacts.read_bytes()).hexdigest())
            self.assertEqual(frozen["training_audit"]["sha256"], hashlib.sha256(audit.read_bytes()).hexdigest())
            timestamp = frozen["created_at"]
            identity = frozen["selection_identity"]
            frozen_without_timestamp = dict(frozen)
            frozen_without_timestamp.pop("created_at")
            frozen_without_timestamp.pop("selection_identity")
            self.assertEqual(identity, hashlib.sha256(json.dumps(frozen_without_timestamp, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(rerun["created_at"], timestamp)
            state.write_bytes(state.read_bytes() + b"changed")
            with self.assertRaises((ValueError, RuntimeError)):
                freeze_selection(run_dir, selected, config, artifacts, audit)

    def test_freeze_selection_rejects_arbitrary_wrong_model_and_bad_sidecar(self) -> None:
        # Catches selection hashing bytes without Task 6 checkpoint validation.
        for mutation in ("arbitrary", "wrong-model", "missing-sidecar", "corrupt-sidecar"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                run_dir, selected, config, artifacts, audit = self._selection_fixture(Path(temporary))
                if mutation == "arbitrary":
                    selected["clock"].write_bytes(b"not a checkpoint")
                elif mutation == "wrong-model":
                    selected["state"] = selected["clock"]
                elif mutation == "missing-sidecar":
                    selected["clock"].with_name(selected["clock"].name + ".sha256").unlink()
                else:
                    selected["clock"].with_name(selected["clock"].name + ".sha256").write_text("f" * 64 + "\n")
                with self.assertRaisesRegex(
                    ValueError, "checkpoint|SHA-256|sidecar|architecture|deserialize"
                ):
                    freeze_selection(run_dir, selected, config, artifacts, audit)
                self.assertFalse((run_dir / "selection.json").exists())

    def test_freeze_selection_rejects_unrecorded_and_recorded_nonwinner_checkpoints(self) -> None:
        # Catches selecting a valid checkpoint without proving it is the recorded val winner.
        for mutation in ("unrecorded", "recorded-nonwinner"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                run_dir, selected, config, artifacts, audit = self._selection_fixture(
                    Path(temporary)
                )
                if mutation == "recorded-nonwinner":
                    selected["clock"] = run_dir / "clock" / "epoch-002.pt"
                else:
                    winner_payload = torch.load(
                        selected["clock"], map_location="cpu", weights_only=False
                    )
                    model = ClockOnlyModel()
                    unrecorded = run_dir / "clock" / "epoch-099.pt"
                    save_checkpoint(
                        unrecorded,
                        model,
                        torch.optim.AdamW(model.parameters(), lr=1e-3),
                        winner_payload["metadata"],
                        99,
                    )
                    selected["clock"] = unrecorded
                with self.assertRaisesRegex(ValueError, "winner|history|artifact|selected"):
                    freeze_selection(run_dir, selected, config, artifacts, audit)
                self.assertFalse((run_dir / "selection.json").exists())

    def test_selection_rejects_malformed_epoch_and_nested_report_contracts(self) -> None:
        # Catches winner reconstruction validating only macro-F1 and a few scalar fields.
        def remove_slice_dimension(record: dict[str, object]) -> None:
            reports = record["validation_slice_reports"]
            assert isinstance(reports, dict)
            reports.pop("seat")

        def add_metric_field(record: dict[str, object]) -> None:
            metrics = record["validation_metrics"]
            assert isinstance(metrics, dict)
            metrics["unexpected"] = 0.0

        mutations = {
            "boolean-epoch": lambda record: record.__setitem__("epoch", True),
            "boolean-best-epoch": lambda record: record.__setitem__(
                "best_epoch", True
            ),
            "negative-validation-loss": lambda record: record.__setitem__(
                "validation_loss", -0.5
            ),
            "integer-validation-loss": lambda record: record.__setitem__(
                "validation_loss", 0
            ),
            "out-of-range-macro-f1": lambda record: record[
                "validation_metrics"
            ].__setitem__("macro_f1", 1.1),  # type: ignore[index,union-attr]
            "fractional-support": lambda record: record["validation_metrics"][
                "per_class"
            ][0].__setitem__("support", 0.5),  # type: ignore[index]
            "boolean-confusion-count": lambda record: record[
                "validation_metrics"
            ]["confusion_matrix"][0].__setitem__(0, True),  # type: ignore[index]
            "missing-slice-dimension": remove_slice_dimension,
            "extra-metric-field": add_metric_field,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run_dir, selected, config, artifacts, audit = self._selection_fixture(
                    root
                )
                _rewrite_epoch_record(run_dir / "clock", 1, mutate)
                before = _filesystem_snapshot(root)
                with self.assertRaises(ValueError):
                    freeze_selection(
                        run_dir, selected, config, artifacts, audit
                    )
                self.assertEqual(_filesystem_snapshot(root), before)
                self.assertFalse((run_dir / "selection.json").exists())

    def test_concurrent_identity_and_history_publication_never_forks(self) -> None:
        # Catches check-then-write races and duplicate JSONL records for one epoch.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity_path = root / "identity.json"
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_write_identity, identity_path, {"worker": value}, False)
                    for value in (1, 2)
                ]
            errors = [future.exception() for future in futures if future.exception()]
            self.assertEqual(len(errors), 1)
            self.assertIn(json.loads(identity_path.read_text())["worker"], (1, 2))

            history = root / "epochs.jsonl"
            record = {"epoch": 1, "value": "same"}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(_append_history, history, record) for _ in range(2)]
            self.assertTrue(all(future.exception() is None for future in futures))
            self.assertEqual(history.read_text().splitlines(), [json.dumps(record, sort_keys=True, separators=(",", ":"))])

            conflict = root / "conflict.jsonl"
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_append_history, conflict, {"epoch": 1, "value": value})
                    for value in ("a", "b")
                ]
            self.assertEqual(sum(future.exception() is not None for future in futures), 1)
            self.assertEqual(len(conflict.read_text().splitlines()), 1)

    def test_concurrent_identical_selection_has_one_identity_and_timestamp(self) -> None:
        # Catches racing selection writers treating descriptive timestamps as conflicts.
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, selected, config, artifacts, audit = self._selection_fixture(Path(temporary))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        freeze_selection, run_dir, selected, config, artifacts, audit
                    )
                    for _ in range(2)
                ]
            self.assertTrue(all(future.exception() is None for future in futures))
            results = [future.result() for future in futures]
            self.assertEqual(results[0], results[1])
            self.assertEqual(
                json.loads((run_dir / "selection.json").read_text()), results[0]
            )

    def test_concurrent_conflicting_selections_publish_exactly_one_winner(self) -> None:
        # Catches a loser overwriting or merging a concurrently frozen selection.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, selected, config, artifacts, audit = self._selection_fixture(root)
            alternate: dict[str, Path] = {}
            for name, model in (("clock", ClockOnlyModel()), ("state", StateAwareModel())):
                original = torch.load(selected[name], map_location="cpu", weights_only=False)
                with torch.no_grad():
                    next(model.parameters()).add_(1.0)
                path = run_dir / f"alternate-{name}.pt"
                save_checkpoint(
                    path,
                    model,
                    torch.optim.AdamW(model.parameters(), lr=1e-3),
                    original["metadata"],
                    1,
                )
                alternate[name] = path
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        freeze_selection, run_dir, choice, config, artifacts, audit
                    )
                    for choice in (selected, alternate)
                ]
            self.assertEqual(sum(future.exception() is not None for future in futures), 1)
            winner = next(future.result() for future in futures if future.exception() is None)
            self.assertEqual(json.loads((run_dir / "selection.json").read_text()), winner)

    def test_selection_identity_ignores_full_audit_and_test_shard_state(self) -> None:
        # Catches selection opening or hashing any test-only input after validation choice.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, selected, config, artifacts, audit = self._selection_fixture(root)
            full_audit = audit.with_name("audit.json")
            full_audit.write_text('{"test": "first"}\n')
            test_dir = audit.parent / "test"
            test_dir.mkdir(exist_ok=True)
            test_shard = test_dir / "test-only.npz"
            test_shard.write_bytes(b"first")
            first = freeze_selection(run_dir, selected, config, artifacts, audit)
            full_audit.unlink()
            full_audit.mkdir()
            test_shard.write_bytes(b"changed")
            second = freeze_selection(run_dir, selected, config, artifacts, audit)
        self.assertEqual(first, second)

    def _load_cli(self) -> object:
        script = Path(__file__).parents[1] / "scripts" / "train_v0.py"
        specification = importlib.util.spec_from_file_location("train_v0_cli", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def _training_fixture(
        self, data_root: Path, config_path: Path
    ) -> dict[str, object]:
        data_root.mkdir(parents=True)
        (data_root / "train").mkdir()
        (data_root / "val").mkdir()
        games = (
            _encoded_game("train", "train-game", np.arange(17), "a" * 64),
            _encoded_game("val", "val-game", np.array([0]), "b" * 64),
        )
        records: list[dict[str, object]] = []
        split_counts: dict[str, dict[str, object]] = {}
        for game in games:
            split = str(game.metadata["split"])
            episode_id = str(game.metadata["episode_id"])
            relative = Path(split) / f"{episode_id}.npz"
            identity = write_shard(game, data_root / relative)
            counts = np.bincount(game.label, minlength=17).tolist()
            records.append({
                "episode_id": episode_id,
                "label_counts": counts,
                "sample_count": int(game.label.shape[0]),
                "shard_identity": identity,
                "shard_path": relative.as_posix(),
                "source_path": game.metadata["source_path"],
                "shard_source_path": game.metadata["source_path"],
                "source_sha256": game.metadata["source_sha256"],
                "source_date": game.metadata["source_date"],
                "route_family": game.metadata["route_family"],
                "split": split,
                "tensor_shapes": {
                    "actor_features": list(game.actor_features.shape),
                    "argument_item": list(game.argument_item.shape),
                    "argument_quantity": list(game.argument_quantity.shape),
                    "global_features": list(game.global_features.shape),
                    "grid": list(game.grid.shape),
                    "label": list(game.label.shape),
                    "step_index": list(game.step_index.shape),
                },
            })
            split_counts[split] = {
                "games": 1,
                "samples": int(game.label.shape[0]),
                "label_counts": counts,
            }
        config = load_config(config_path)
        checks = {
            "operation_support_covered": True,
            "safe_records_validated": True,
            "shard_identities_verified": True,
            "source_hashes_unique": True,
            "tensor_shapes_validated": True,
            "train_val_episode_ids_unique": True,
        }
        safe_manifest = {
            "schema_version": "ryo-training-preparation-v0",
            "records": records,
        }
        audit: dict[str, object] = {
            "all_checks_passed": True,
            "checks": checks,
            "config": {
                "canonical": config,
                "sha256": hashlib.sha256(_canonical_json_bytes(config)).hexdigest(),
            },
            "feature_schema_version": "ryo-features-v0",
            "model_schema_version": "ryo-bc-v0",
            "operations": list(OPERATIONS),
            "safe_manifest_sha256": hashlib.sha256(
                _canonical_json_bytes(safe_manifest)
            ).hexdigest(),
            "schema_version": "ryo-training-preparation-v0",
            "shard_identities": sorted(record["shard_identity"] for record in records),
            "shards": records,
            "smoke_mode": False,
            "source_hashes": sorted(record["source_sha256"] for record in records),
            "splits": split_counts,
            "totals": {"games": 2, "samples": 18},
            "trainable": True,
        }
        audit["training_identity"] = hashlib.sha256(
            _canonical_json_bytes(audit)
        ).hexdigest()
        (data_root / "training_audit.json").write_text(json.dumps(audit))
        return audit

    def _cli_resume_fixture(
        self,
        root: Path,
        *,
        model_identities: bool = False,
        complete_models: bool = False,
        selection: bool = False,
    ) -> tuple[Path, Path, Path, Path, dict[str, object] | None]:
        config_path = Path(__file__).parents[1] / "configs" / "v0.json"
        config = load_config(config_path)
        data_root = root / "data"
        runs_root = root / "runs"
        audit = self._training_fixture(data_root, config_path)
        audit_path = data_root / "training_audit.json"
        run_dir = runs_root / "resume"
        run_dir.mkdir(parents=True)
        stats = NormalizationStats(
            np.zeros(GLOBAL_DIM, np.float32),
            np.ones(GLOBAL_DIM, np.float32),
            np.zeros(ACTOR_DIM, np.float32),
            np.ones(ACTOR_DIM, np.float32),
        )
        counts = np.ones(len(OPERATIONS), np.int64)
        weights = np.ones(len(OPERATIONS), np.float32)
        majority = fit_majority_rules(
            np.arange(len(OPERATIONS)), np.zeros(len(OPERATIONS), dtype=np.int64)
        )
        artifact_metadata = {
            "schema_version": config["schema_version"],
            "feature_schema_version": config["feature_schema_version"],
            "operations": list(OPERATIONS),
            "train_shard_identities": [
                record["shard_identity"]
                for record in audit["shards"]
                if record["split"] == "train"
            ],
            "preparation_manifest_sha256": audit["safe_manifest_sha256"],
            "weight_cap": config["training"]["weight_cap"],
            "training_identity": audit["training_identity"],
            "preparation_identity": audit["training_identity"],
        }
        artifacts_path = run_dir / "train_artifacts.npz"
        artifact_identity = save_train_artifacts(
            artifacts_path,
            stats,
            counts,
            weights,
            majority,
            artifact_metadata,
        )
        artifact_sha256 = hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
        train_dataset = ShardDataset([data_root / "train" / "train-game.npz"], stats)
        val_dataset = ShardDataset([data_root / "val" / "val-game.npz"], stats)
        _, val_loader = make_loaders(
            train_dataset,
            val_dataset,
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["seed"]),
        )
        majority_report = evaluate_majority(val_loader, majority)
        (run_dir / "majority.validation.json").write_text(
            json.dumps(majority_report, indent=2, sort_keys=True) + "\n"
        )

        selected: dict[str, Path] = {}
        if model_identities or complete_models:
            device = choose_device()
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights.copy()).to(device)
            )
            cli = self._load_cli()
            for name, model in (
                ("clock", ClockOnlyModel()),
                ("state", StateAwareModel()),
            ):
                checkpoint_metadata = cli._checkpoint_metadata(
                    config,
                    audit,
                    stats,
                    weights,
                    artifact_identity,
                    artifact_sha256,
                    model,
                )
                model_dir = run_dir / name
                model_dir.mkdir()
                _write_identity(
                    model_dir / "run-identity.json",
                    {
                        "model_name": name,
                        "config": config,
                        "checkpoint_metadata": checkpoint_metadata,
                    },
                    False,
                )
                if not complete_models:
                    continue
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                validation = evaluate_loader(
                    model, val_loader, criterion, device, name
                )
                records: list[dict[str, object]] = []
                for epoch in range(1, 7):
                    checkpoint_path = model_dir / f"epoch-{epoch:03d}.pt"
                    checkpoint_sha256 = save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        checkpoint_metadata,
                        epoch,
                    )
                    record: dict[str, object] = {
                        "best_epoch": 1,
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "device": str(device),
                        "elapsed_seconds": float(epoch),
                        "epoch": epoch,
                        "model_name": name,
                        "non_improving_epochs": epoch - 1,
                        "seed": config["seed"],
                        "train_artifact_identity": artifact_identity,
                        "train_loss": 1.0,
                        "preparation_identity": audit["training_identity"],
                        "validation_loss": validation["loss"],
                        "validation_metrics": validation["metrics"],
                        "validation_slice_reports": validation["slice_reports"],
                    }
                    encoded = _canonical_json_bytes(record) + b"\n"
                    (model_dir / f"epoch-{epoch:03d}.json").write_bytes(encoded)
                    records.append(record)
                (model_dir / "epochs.jsonl").write_bytes(
                    b"".join(_canonical_json_bytes(record) + b"\n" for record in records)
                )
                selected[name] = model_dir / "epoch-001.pt"

        frozen: dict[str, object] | None = None
        if selection:
            if not complete_models:
                raise AssertionError("selection fixture requires complete models")
            frozen = freeze_selection(
                run_dir, selected, config_path, artifacts_path, audit_path
            )
        return config_path, data_root, runs_root, run_dir, frozen

    def _selection_fixture(
        self, root: Path
    ) -> tuple[Path, dict[str, Path], Path, Path, Path]:
        config_path = Path(__file__).parents[1] / "configs" / "v0.json"
        data_root = root / "data"
        audit = self._training_fixture(data_root, config_path)
        audit_path = data_root / "training_audit.json"
        run_dir = root / "runs" / "run-a"
        run_dir.mkdir(parents=True)
        stats = NormalizationStats(
            np.zeros(GLOBAL_DIM, np.float32),
            np.ones(GLOBAL_DIM, np.float32),
            np.zeros(ACTOR_DIM, np.float32),
            np.ones(ACTOR_DIM, np.float32),
        )
        counts = np.ones(17, np.int64)
        weights = np.ones(17, np.float32)
        majority = fit_majority_rules(np.arange(17), np.resize(np.array([0, 1]), 17))
        artifact_metadata = {
            "schema_version": "ryo-bc-v0",
            "feature_schema_version": "ryo-features-v0",
            "operations": list(OPERATIONS),
            "train_shard_identities": [
                record["shard_identity"]
                for record in audit["shards"]
                if record["split"] == "train"
            ],
            "preparation_manifest_sha256": audit["safe_manifest_sha256"],
            "weight_cap": 4.0,
            "training_identity": audit["training_identity"],
            "preparation_identity": audit["training_identity"],
        }
        artifacts = run_dir / "train_artifacts.npz"
        artifact_identity = save_train_artifacts(
            artifacts, stats, counts, weights, majority, artifact_metadata
        )
        artifact_sha256 = hashlib.sha256(artifacts.read_bytes()).hexdigest()
        selected: dict[str, Path] = {}
        for name, model in (("clock", ClockOnlyModel()), ("state", StateAwareModel())):
            metadata = {
                "schema_version": "ryo-bc-v0",
                "feature_schema_version": "ryo-features-v0",
                "vocabularies": {"operations": list(OPERATIONS)},
                "normalization": {
                    "global_mean": stats.global_mean,
                    "global_std": stats.global_std,
                    "actor_mean": stats.actor_mean,
                    "actor_std": stats.actor_std,
                },
                "class_weights": weights,
                "manifest_sha256": audit["safe_manifest_sha256"],
                "architecture": architecture_metadata(model),
                "training_identity": audit["training_identity"],
                "preparation_identity": audit["training_identity"],
                "train_artifact_identity": artifact_identity,
                "train_artifact_sha256": artifact_sha256,
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            model_dir = run_dir / name
            model_dir.mkdir()
            records: list[dict[str, object]] = []
            for epoch, (correct_rows, validation_loss) in enumerate(
                ((2, 0.5), (1, 0.4)), 1
            ):
                path = model_dir / f"epoch-{epoch:03d}.pt"
                checkpoint_sha256 = save_checkpoint(
                    path, model, optimizer, metadata, epoch
                )
                validation = _validation_report(
                    loss=validation_loss, correct_rows=correct_rows
                )
                record: dict[str, object] = {
                    "best_epoch": 1,
                    "checkpoint_path": str(path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "device": "cpu",
                    "elapsed_seconds": float(epoch),
                    "epoch": epoch,
                    "model_name": name,
                    "non_improving_epochs": epoch - 1,
                    "seed": 20260824,
                    "train_artifact_identity": artifact_identity,
                    "train_loss": 1.0,
                    "preparation_identity": audit["training_identity"],
                    "validation_loss": validation_loss,
                    "validation_metrics": validation["metrics"],
                    "validation_slice_reports": validation["slice_reports"],
                }
                encoded = _canonical_json_bytes(record).decode() + "\n"
                (model_dir / f"epoch-{epoch:03d}.json").write_text(encoded)
                records.append(record)
            (model_dir / "epochs.jsonl").write_text(
                "".join(_canonical_json_bytes(record).decode() + "\n" for record in records)
            )
            selected[name] = model_dir / "epoch-001.pt"
        return run_dir, selected, config_path, artifacts, audit_path


if __name__ == "__main__":
    unittest.main()
