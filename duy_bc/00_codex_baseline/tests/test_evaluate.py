import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import test_train
from fixtures import make_replay
from bc_core.constants import (
    ACTOR_DIM,
    GLOBAL_DIM,
    GRID_CHANNELS,
    OPERATIONS,
    load_config,
)
from bc_core.dataset import (
    ShardDataset,
    collate_examples,
    fit_train_artifacts,
    save_train_artifacts,
)
from bc_core.evaluate import (
    EvaluationError,
    evaluate_frozen_run,
    success_gate,
    verify_selection,
)
from bc_core.features import logical_shard_identity, write_shard
from bc_core.checkpoints import (
    choose_device,
    save_checkpoint,
)
from bc_core.prepare import _training_audit
from bc_core.replay import SourceReplay
from bc_core.train import evaluate_loader, evaluate_majority
from model.clock import ClockOnlyModel
from model.state import StateAwareModel


class SourceReplayAuthenticationTest(unittest.TestCase):
    def test_real_validated_replay_is_encoded_by_the_authentication_boundary(self) -> None:
        # Catches replacing the required load-and-encode path with audit-only metadata.
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-game.json"
            path.write_text(json.dumps(make_replay()), encoding="utf-8")
            source = SourceReplay(
                "test",
                "fixture-game",
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "21-08",
                "route-a",
                "archive/fixture-game.json",
            )

            game = evaluate_module._reencode_source_game(
                evaluate_module._snapshot_source_replay(source), "1.32.7"
            )

        self.assertEqual(game.metadata["episode_id"], "fixture-game")
        self.assertEqual(game.metadata["source_sha256"], source.sha256)
        self.assertEqual(game.label.shape, (1438,))
        self.assertEqual(len(logical_shard_identity(game)), 64)

    def test_reencoding_uses_hashed_snapshot_after_repeated_path_swaps(self) -> None:
        # Catches hashing one path version and decoding a later replacement version.
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-game.json"
            path.write_text(json.dumps(make_replay()), encoding="utf-8")
            source = SourceReplay(
                "test",
                "fixture-game",
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "21-08",
                "route-a",
                "archive/fixture-game.json",
            )
            snapshot = evaluate_module._snapshot_source_replay(source)
            expected = logical_shard_identity(
                evaluate_module._reencode_source_game(snapshot, "1.32.7")
            )

            for replacement in (
                b'{"malformed":"first"}\n',
                b'{"malformed":"second","steps":[]}\n',
            ):
                path.write_bytes(replacement)
                game = evaluate_module._reencode_source_game(snapshot, "1.32.7")
                self.assertEqual(logical_shard_identity(game), expected)


class RowIdentityValidationTest(unittest.TestCase):
    def test_canonical_step_accepts_actor_indices_above_normalization_scale(self) -> None:
        # The actor-index feature is divided by eight, but farms may have more hands.
        import bc_core.evaluate as evaluate_module

        actor_count = 13
        game = test_train._encoded_game(
            "val",
            "many-hands",
            np.arange(actor_count, dtype=np.int64) % len(OPERATIONS),
            "a" * 64,
        )
        game.step_index[:] = 0
        game.actor_features[:, 0] = 0.0
        game.actor_features[0, 0] = 1.0
        game.actor_features[:, 1] = (
            np.arange(actor_count, dtype=np.float32) / 8.0
        )

        evaluate_module._validate_row_identities(game, Path("many-hands.npz"))


class SourceCorpusResolutionTest(unittest.TestCase):
    def test_repository_relative_corpus_may_use_a_resolved_directory_link(self) -> None:
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository_root = root / "worktree"
            config_path = (
                repository_root
                / "duy_bc"
                / "00_codex_baseline"
                / "configs"
                / "v0.json"
            )
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}\n", encoding="utf-8")
            source_home = root / "source-home"
            corpus_root = source_home / "ryo_hasegawa_100_stratified"
            corpus_root.mkdir(parents=True)
            (repository_root / "duy_explore").symlink_to(
                source_home, target_is_directory=True
            )
            selection = {
                "config": {
                    "path": str(config_path),
                    "canonical": {
                        "corpus_root": "duy_explore/ryo_hasegawa_100_stratified"
                    },
                }
            }

            resolved = evaluate_module._source_corpus_root(selection)

        self.assertEqual(resolved, corpus_root.resolve())


class EvaluateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        source_config = Path(__file__).parents[1] / "configs" / "v0.json"
        cls.repository_root = cls.root / "repository"
        cls.config_path = (
            cls.repository_root
            / "duy_bc"
            / "00_codex_baseline"
            / "configs"
            / "v0.json"
        )
        cls.config_path.parent.mkdir(parents=True)
        shutil.copyfile(source_config, cls.config_path)
        cls.data_root = cls.root / "data"
        cls.runs_root = cls.root / "runs"
        cls.run_dir = cls.runs_root / "resume"
        cls._write_corpus_and_prepared_data()
        cls._write_training_run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        for name in (
            "evaluation.test.json",
            "evaluation.val.json",
            "REPORT.md",
            "REPORT.val.md",
        ):
            path = self.run_dir / name
            if path.exists() or path.is_symlink():
                path.unlink()
        import bc_core.evaluate as evaluate_module

        replay_patch = patch.object(
            evaluate_module,
            "load_validated_replay_bytes",
            return_value={"validated": True},
        )
        encode_patch = patch.object(
            evaluate_module,
            "encode_game",
            side_effect=lambda source, replay: self._source_game(source),
        )
        replay_patch.start()
        encode_patch.start()
        self.addCleanup(encode_patch.stop)
        self.addCleanup(replay_patch.stop)

    def test_unfrozen_run_is_rejected(self) -> None:
        # Catches evaluating a run without the immutable selection authorization.
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(EvaluationError, "selection.json"):
                verify_selection(Path(temporary))

    def test_gate_requires_every_condition(self) -> None:
        # Catches a gate implementation that ignores a non-positive bootstrap bound.
        report = self._passing_report()
        self.assertTrue(success_gate(report)["pass"])
        report["bootstrap"]["ci95_low"] = 0.0

        result = success_gate(report)

        self.assertFalse(result["pass"])
        self.assertIn("bootstrap_lower_bound_positive", result["failed"])

    def test_each_of_the_six_gate_conditions_can_stop_the_run(self) -> None:
        # Catches collapsing the gate into a subset of the six fixed conditions.
        cases = {
            "preparation_and_leakage_valid": lambda report: report["audit"].update(
                all_checks_passed=False
            ),
            "operation_support_covered": lambda report: report["audit"].update(
                operation_support_covered=False
            ),
            "validation_state_macro_f1_gt_clock": lambda report: report[
                "validation"
            ]["state"].update(macro_f1=0.3),
            "test_state_macro_f1_gt_clock": lambda report: report["test"][
                "state"
            ].update(macro_f1=0.3),
            "bootstrap_lower_bound_positive": lambda report: report[
                "bootstrap"
            ].update(ci95_low=0.0),
            "complete_diagnostics": lambda report: report.update(
                complete_diagnostics=False
            ),
        }
        for expected_failure, mutate in cases.items():
            with self.subTest(expected_failure):
                report = self._passing_report()
                mutate(report)
                result = success_gate(report)
                self.assertFalse(result["pass"])
                self.assertEqual(result["failed"], [expected_failure])

    def test_test_evaluation_has_identical_primary_rows_and_complete_diagnostics(self) -> None:
        # Catches evaluating systems on different rows or omitting fixed diagnostics.
        report = evaluate_frozen_run(self.run_dir, self.data_root, split="test")

        self.assertEqual(report["split"], "test")
        self.assertTrue((self.run_dir / "evaluation.test.json").is_file())
        self.assertTrue((self.run_dir / "REPORT.md").is_file())
        self.assertEqual(report["row_order"]["systems"], [
            "majority_actor", "clock", "state"
        ])
        self.assertEqual(report["row_order"]["rows"], 15)
        for name in ("majority_actor", "clock", "state"):
            system = report["test"][name]
            self.assertEqual(len(system["per_class"]), len(OPERATIONS))
            self.assertEqual(
                set(system["slice_reports"]),
                {"actor_type", "seat", "day_band", "source_date", "route_family"},
            )
            self.assertEqual(
                np.asarray(system["confusion_matrix"]).shape,
                (len(OPERATIONS), len(OPERATIONS)),
            )
        self.assertEqual(report["bootstrap"]["resamples"], 10000)
        self.assertTrue(report["complete_diagnostics"])
        for name in ("clock", "state"):
            self.assertEqual(
                report["artifacts"]["checkpoints"][name]["architecture"],
                self.selection["models"][name]["architecture"],
            )

    def test_selection_verifies_before_full_audit_or_test_data_is_opened(self) -> None:
        # Catches reading even one test label before checkpoint authorization succeeds.
        import bc_core.evaluate as evaluate_module

        checkpoint = Path(self.selection["models"]["clock"]["checkpoint_path"])
        checkpoint_bytes = checkpoint.read_bytes()
        real_reader = evaluate_module.read_shard
        test_reads = 0

        def reject_test_read(path: Path) -> object:
            nonlocal test_reads
            if Path(path).parent.name == "test":
                test_reads += 1
                raise AssertionError(f"test label read before selection path={path}")
            return real_reader(path)

        try:
            checkpoint.write_bytes(checkpoint_bytes + b"forged")
            with patch.object(
                evaluate_module, "read_shard", side_effect=reject_test_read
            ), patch(
                "bc_core.dataset.read_shard", side_effect=reject_test_read
            ), self.assertRaisesRegex(EvaluationError, "checkpoint|SHA-256"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            self.assertEqual(test_reads, 0)
        finally:
            checkpoint.write_bytes(checkpoint_bytes)

    def test_full_audit_train_val_projection_must_equal_frozen_training_audit(self) -> None:
        # Catches authorizing test data from a full audit with a substituted safe history.
        path = self.data_root / "audit.json"
        original = path.read_bytes()
        shard_path = self.data_root / "test" / "test-game-000.npz"
        shard_bytes = shard_path.read_bytes()
        try:
            audit = json.loads(original)
            train_record = next(
                record for record in audit["shards"] if record["split"] == "train"
            )
            train_record["source_date"] = "substituted"
            path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shard_path.unlink()
            shard_path.mkdir()

            with self.assertRaisesRegex(EvaluationError, "train/val projection"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            path.write_bytes(original)
            if shard_path.is_dir():
                shard_path.rmdir()
            shard_path.write_bytes(shard_bytes)

    def test_full_audit_preparation_identity_is_recomputed(self) -> None:
        # Catches treating the recorded full-audit identity as an unchecked label.
        path = self.data_root / "audit.json"
        original = path.read_bytes()
        try:
            audit = json.loads(original)
            audit["preparation_identity"] = "f" * 64
            path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EvaluationError, "preparation_identity"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            path.write_bytes(original)

    def test_changed_test_shard_identity_is_rejected(self) -> None:
        # Catches accepting a valid NPZ at the audited path with different logical rows.
        path = self.data_root / "test" / "test-game-000.npz"
        original = path.read_bytes()
        replacement = self.data_root / "replacement.npz"
        try:
            game = test_train._encoded_game(
                "test", "test-game-000", np.asarray([0, 1, 3]), "c" * 64
            )
            game.actor_features[:, 0] = np.asarray([1.0, 0.0, 0.0])
            game.actor_features[:, 1] = np.asarray([0.0, 1.0 / 8.0, 2.0 / 8.0])
            write_shard(game, replacement)
            path.write_bytes(replacement.read_bytes())

            with self.assertRaisesRegex(EvaluationError, "does not match full audit"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
        finally:
            path.write_bytes(original)
            replacement.unlink(missing_ok=True)

    def test_self_consistent_test_shard_and_audit_forgery_mismatches_source(self) -> None:
        # Catches a full audit that authenticates only its own forged shard projection.
        audit_path = self.data_root / "audit.json"
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        record = next(item for item in audit["shards"] if item["split"] == "test")
        shard_path = self.data_root / str(record["shard_path"])
        shard_bytes = shard_path.read_bytes()
        replacement = self.data_root / "source-forgery.npz"
        try:
            forged_source_sha256 = "f" * 64
            game = test_train._encoded_game(
                "test",
                str(record["episode_id"]),
                np.asarray([1, 2], dtype=np.int64),
                forged_source_sha256,
            )
            game.metadata.update(
                {
                    "source_path": record["shard_source_path"],
                    "source_date": record["source_date"],
                    "route_family": record["route_family"],
                }
            )
            identity = write_shard(game, replacement)
            shard_path.write_bytes(replacement.read_bytes())
            record.update(
                {
                    "label_counts": np.bincount(
                        game.label, minlength=len(OPERATIONS)
                    ).tolist(),
                    "sample_count": int(game.label.shape[0]),
                    "shard_identity": identity,
                    "source_sha256": forged_source_sha256,
                    "tensor_shapes": {
                        "actor_features": list(game.actor_features.shape),
                        "argument_item": list(game.argument_item.shape),
                        "argument_quantity": list(game.argument_quantity.shape),
                        "global_features": list(game.global_features.shape),
                        "grid": list(game.grid.shape),
                        "label": list(game.label.shape),
                        "step_index": list(game.step_index.shape),
                    },
                }
            )
            self._refresh_full_audit(audit)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationError, "corpus|manifest|source"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
        finally:
            shard_path.write_bytes(shard_bytes)
            audit_path.write_bytes(audit_bytes)
            replacement.unlink(missing_ok=True)

    def test_coordinated_prepared_label_and_audit_forgery_mismatches_reencoded_source(
        self,
    ) -> None:
        # Catches authenticating replay provenance without deriving its tensors/labels.
        import bc_core.evaluate as evaluate_module

        audit_path = self.data_root / "audit.json"
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        record = next(
            item
            for item in audit["shards"]
            if item["split"] == "test" and item["episode_id"] == "test-game-000"
        )
        original_provenance = {
            name: record[name]
            for name in (
                "source_sha256",
                "source_path",
                "shard_source_path",
                "source_date",
                "route_family",
            )
        }
        shard_path = self.data_root / str(record["shard_path"])
        shard_bytes = shard_path.read_bytes()
        replacement = self.data_root / "coordinated-label-forgery.npz"
        try:
            game = test_train._encoded_game(
                "test",
                str(record["episode_id"]),
                np.asarray([1, 2], dtype=np.int64),
                str(record["source_sha256"]),
            )
            game.metadata.update(
                {
                    "source_path": record["shard_source_path"],
                    "source_date": record["source_date"],
                    "route_family": record["route_family"],
                }
            )
            identity = write_shard(game, replacement)
            shard_path.write_bytes(replacement.read_bytes())
            record.update(
                {
                    "label_counts": np.bincount(
                        game.label, minlength=len(OPERATIONS)
                    ).tolist(),
                    "sample_count": int(game.label.shape[0]),
                    "shard_identity": identity,
                    "tensor_shapes": self._tensor_shapes(game),
                }
            )
            self.assertEqual(
                {name: record[name] for name in original_provenance},
                original_provenance,
            )
            self._refresh_full_audit(audit)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with patch.object(
                evaluate_module,
                "load_validated_replay_bytes",
                return_value={"validated": True},
                create=True,
            ), patch.object(
                evaluate_module,
                "encode_game",
                side_effect=lambda source, replay: self._source_game(source),
                create=True,
            ), self.assertRaisesRegex(
                EvaluationError, "re-encoded|source replay|prepared shard"
            ):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
            self.assertFalse((self.run_dir / "REPORT.md").exists())
        finally:
            shard_path.write_bytes(shard_bytes)
            audit_path.write_bytes(audit_bytes)
            replacement.unlink(missing_ok=True)
            (self.run_dir / "evaluation.test.json").unlink(missing_ok=True)
            (self.run_dir / "REPORT.md").unlink(missing_ok=True)

    def test_extra_test_path_is_rejected(self) -> None:
        # Catches ignoring an unaudited substitute or sidecar in the test tree.
        extra = self.data_root / "test" / "extra.txt"
        try:
            extra.write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "extra"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            extra.unlink(missing_ok=True)

    def test_missing_test_shard_is_rejected(self) -> None:
        # Catches silently evaluating a strict subset of the audited test corpus.
        path = self.data_root / "test" / "test-game-000.npz"
        backup = self.data_root / "test-game.backup"
        try:
            path.rename(backup)
            with self.assertRaisesRegex(EvaluationError, "missing"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            if backup.exists():
                backup.rename(path)

    def test_test_symlink_alias_is_rejected(self) -> None:
        # Catches traversal that collapses a second path onto an audited test shard.
        alias = self.data_root / "test" / "alias.npz"
        try:
            alias.symlink_to(self.data_root / "test" / "test-game-000.npz")
            with self.assertRaisesRegex(EvaluationError, "symlink"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            alias.unlink(missing_ok=True)

    def test_test_shard_mutation_between_verification_and_consumption_rejects(self) -> None:
        # Catches ShardDataset reopening a path whose verified bytes were replaced.
        import bc_core.evaluate as evaluate_module

        audit = json.loads((self.data_root / "audit.json").read_text(encoding="utf-8"))
        record = next(item for item in audit["shards"] if item["split"] == "test")
        shard_path = self.data_root / str(record["shard_path"])
        shard_bytes = shard_path.read_bytes()
        replacement = self.data_root / "toctou-replacement.npz"
        game = test_train._encoded_game(
            "test",
            str(record["episode_id"]),
            np.asarray([4], dtype=np.int64),
            str(record["source_sha256"]),
        )
        game.metadata.update(
            {
                "source_path": record["shard_source_path"],
                "source_date": record["source_date"],
                "route_family": record["route_family"],
            }
        )
        write_shard(game, replacement)
        original_verify = evaluate_module._verified_split_shards
        mutated = False

        def verify_then_replace(
            full_audit: dict[str, object],
            root: Path,
            split: str,
            **kwargs: object,
        ) -> object:
            nonlocal mutated
            result = original_verify(full_audit, root, split, **kwargs)
            if split == "test" and not mutated:
                shard_path.write_bytes(replacement.read_bytes())
                mutated = True
            return result

        try:
            with patch.object(
                evaluate_module,
                "_verified_split_shards",
                side_effect=verify_then_replace,
            ), self.assertRaisesRegex(EvaluationError, "changed during evaluation"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            self.assertTrue(mutated)
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
            self.assertFalse((self.run_dir / "REPORT.md").exists())
        finally:
            shard_path.write_bytes(shard_bytes)
            replacement.unlink(missing_ok=True)
            (self.run_dir / "evaluation.test.json").unlink(missing_ok=True)
            (self.run_dir / "REPORT.md").unlink(missing_ok=True)

    def test_transient_test_shard_swap_cannot_change_evaluated_labels(self) -> None:
        # Catches model evaluation reopening a verified shard path after authentication.
        import bc_core.evaluate as evaluate_module

        audit = json.loads((self.data_root / "audit.json").read_text(encoding="utf-8"))
        record = next(
            item
            for item in audit["shards"]
            if item["split"] == "test" and item["episode_id"] == "test-game-000"
        )
        shard_path = self.data_root / str(record["shard_path"])
        shard_bytes = shard_path.read_bytes()
        replacement = self.data_root / "transient-test-swap.npz"
        game = test_train._encoded_game(
            "test",
            str(record["episode_id"]),
            np.asarray([4], dtype=np.int64),
            str(record["source_sha256"]),
        )
        game.metadata.update(
            {
                "source_path": record["shard_source_path"],
                "source_date": record["source_date"],
                "route_family": record["route_family"],
            }
        )
        write_shard(game, replacement)
        original_evaluate = evaluate_module._evaluation_systems
        swapped = False

        def evaluate_while_swapped(inputs: object, selection: object) -> object:
            nonlocal swapped
            shard_path.write_bytes(replacement.read_bytes())
            swapped = True
            try:
                return original_evaluate(inputs, selection)
            finally:
                shard_path.write_bytes(shard_bytes)

        try:
            with patch.object(
                evaluate_module,
                "load_validated_replay_bytes",
                return_value={"validated": True},
            ), patch.object(
                evaluate_module,
                "encode_game",
                side_effect=lambda source, replay: self._source_game(source),
            ), patch.object(
                evaluate_module,
                "_evaluation_systems",
                side_effect=evaluate_while_swapped,
            ):
                report = evaluate_frozen_run(
                    self.run_dir, self.data_root, split="test"
                )

            self.assertTrue(swapped)
            expected_counts = audit["splits"]["test"]["label_counts"]
            for system in ("majority_actor", "clock", "state"):
                self.assertEqual(
                    [
                        int(item["support"])
                        for item in report["test"][system]["per_class"]
                    ],
                    expected_counts,
                )
        finally:
            shard_path.write_bytes(shard_bytes)
            replacement.unlink(missing_ok=True)
            (self.run_dir / "evaluation.test.json").unlink(missing_ok=True)
            (self.run_dir / "REPORT.md").unlink(missing_ok=True)

    def test_transient_source_swap_after_authentication_cannot_change_test_snapshot(
        self,
    ) -> None:
        # Catches reopening a replay path after its authenticated bytes were captured.
        import bc_core.evaluate as evaluate_module

        record = next(
            item
            for item in self.full_audit["shards"]
            if item["split"] == "test" and item["episode_id"] == "test-game-000"
        )
        source_path = Path(str(record["shard_source_path"]))
        source_bytes = source_path.read_bytes()
        forged_bytes = b'{"episode_id":"test-game-000","fixture":"forged"}\n'
        original_authenticate = evaluate_module._authenticate_source_corpus
        original_verified = evaluate_module._verified_split_shards
        swapped = False

        def authenticate_then_swap(*args: object, **kwargs: object) -> object:
            nonlocal swapped
            result = original_authenticate(*args, **kwargs)
            if not swapped:
                source_path.write_bytes(forged_bytes)
                swapped = True
            return result

        def verify_then_restore(*args: object, **kwargs: object) -> object:
            try:
                return original_verified(*args, **kwargs)
            finally:
                if swapped:
                    source_path.write_bytes(source_bytes)

        def decode_snapshot(
            source: object, content: bytes, expected_module_version: str
        ) -> object:
            del source, expected_module_version
            return json.loads(content.decode("utf-8"))

        def encode_snapshot(source: object, replay: dict[str, object]) -> object:
            if replay.get("fixture") == "forged":
                game = test_train._encoded_game(
                    str(source.split),
                    str(source.episode_id),
                    np.asarray([4], dtype=np.int64),
                    str(source.sha256),
                )
                game.metadata.update(
                    {
                        "source_path": str(source.path),
                        "source_date": str(source.source_date),
                        "route_family": str(source.route_family),
                    }
                )
                return game
            return self._source_game(source)

        try:
            with patch.object(
                evaluate_module,
                "load_validated_replay_bytes",
                side_effect=decode_snapshot,
                create=True,
            ), patch.object(
                evaluate_module,
                "encode_game",
                side_effect=encode_snapshot,
            ), patch.object(
                evaluate_module,
                "_authenticate_source_corpus",
                side_effect=authenticate_then_swap,
            ), patch.object(
                evaluate_module,
                "_verified_split_shards",
                side_effect=verify_then_restore,
            ), self.assertRaisesRegex(EvaluationError, "frozen input changed"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")

            self.assertTrue(swapped)
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
            self.assertFalse((self.run_dir / "REPORT.md").exists())
        finally:
            source_path.write_bytes(source_bytes)
            (self.run_dir / "evaluation.test.json").unlink(missing_ok=True)
            (self.run_dir / "REPORT.md").unlink(missing_ok=True)

    def test_nonfinite_model_logits_are_rejected_without_publication(self) -> None:
        # Catches publication after a selected model emits NaN in val replay or test.
        def nonfinite_forward(
            model: ClockOnlyModel, clock_features: torch.Tensor
        ) -> torch.Tensor:
            del model
            return torch.full(
                (clock_features.shape[0], len(OPERATIONS)), float("nan")
            )

        with patch.object(ClockOnlyModel, "forward", nonfinite_forward):
            with self.assertRaisesRegex(EvaluationError, "non-finite (loss|logits)"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")

        self.assertFalse((self.run_dir / "evaluation.test.json").exists())

    def test_duplicate_test_row_identity_is_rejected_even_when_audit_is_consistent(self) -> None:
        # Catches duplicate (game, step, actor) decisions hidden by self-consistent hashes.
        shard_path = self.data_root / "test" / "test-game-000.npz"
        shard_bytes = shard_path.read_bytes()
        audit_path = self.data_root / "audit.json"
        audit_bytes = audit_path.read_bytes()
        replacement = self.data_root / "duplicate-rows.npz"
        try:
            audit = json.loads(audit_bytes)
            record = next(
                item
                for item in audit["shards"]
                if item["split"] == "test" and item["episode_id"] == "test-game-000"
            )
            game = test_train._encoded_game(
                "test",
                "test-game-000",
                np.asarray([0, 1]),
                str(record["source_sha256"]),
            )
            game.metadata.update(
                {
                    "source_path": record["shard_source_path"],
                    "source_date": record["source_date"],
                    "route_family": record["route_family"],
                }
            )
            game.step_index[:] = 0
            game.actor_features[:, 0] = 1.0
            game.actor_features[:, 1] = 0.0
            identity = write_shard(game, replacement)
            shard_path.write_bytes(replacement.read_bytes())

            counts = np.bincount(game.label, minlength=len(OPERATIONS)).tolist()
            record.update(
                {
                    "label_counts": counts,
                    "sample_count": 2,
                    "shard_identity": identity,
                    "tensor_shapes": {
                        "actor_features": list(game.actor_features.shape),
                        "argument_item": list(game.argument_item.shape),
                        "argument_quantity": list(game.argument_quantity.shape),
                        "global_features": list(game.global_features.shape),
                        "grid": list(game.grid.shape),
                        "label": list(game.label.shape),
                        "step_index": list(game.step_index.shape),
                    },
                }
            )
            self._refresh_full_audit(audit)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(EvaluationError, "duplicate evaluation rows"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            shard_path.write_bytes(shard_bytes)
            audit_path.write_bytes(audit_bytes)
            replacement.unlink(missing_ok=True)

    def test_identical_frozen_test_rerun_is_idempotent(self) -> None:
        # Catches nondeterministic reports or replacement-style output publication.
        first = evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        json_bytes = (self.run_dir / "evaluation.test.json").read_bytes()
        markdown_bytes = (self.run_dir / "REPORT.md").read_bytes()

        second = evaluate_frozen_run(self.run_dir, self.data_root, split="test")

        self.assertEqual(first, second)
        self.assertEqual((self.run_dir / "evaluation.test.json").read_bytes(), json_bytes)
        self.assertEqual((self.run_dir / "REPORT.md").read_bytes(), markdown_bytes)

    def test_input_mutation_inside_publication_rolls_back_new_outputs(self) -> None:
        # Catches returning after publication without sealing inputs a final time.
        import bc_core.evaluate as evaluate_module

        audit_path = self.data_root / "audit.json"
        audit_bytes = audit_path.read_bytes()
        original_publish = evaluate_module._publish_outputs
        mutated = False

        def publish_then_mutate(outputs: dict[Path, bytes]) -> object:
            nonlocal mutated
            result = original_publish(outputs)
            audit_path.write_bytes(audit_bytes + b"\n")
            mutated = True
            return result

        try:
            with patch.object(
                evaluate_module,
                "load_validated_replay_bytes",
                return_value={"validated": True},
            ), patch.object(
                evaluate_module,
                "encode_game",
                side_effect=lambda source, replay: self._source_game(source),
            ), patch.object(
                evaluate_module,
                "_publish_outputs",
                side_effect=publish_then_mutate,
            ), self.assertRaisesRegex(EvaluationError, "changed during evaluation"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")

            self.assertTrue(mutated)
            self.assertFalse((self.run_dir / "evaluation.test.json").exists())
            self.assertFalse((self.run_dir / "REPORT.md").exists())
        finally:
            audit_path.write_bytes(audit_bytes)
            (self.run_dir / "evaluation.test.json").unlink(missing_ok=True)
            (self.run_dir / "REPORT.md").unlink(missing_ok=True)

    def test_post_publication_seal_rejects_changed_test_and_val_output_bytes(
        self,
    ) -> None:
        # Catches returning a report whose canonical JSON or Markdown changed on disk.
        import bc_core.evaluate as evaluate_module

        cases = (
            ("test", "evaluation.test.json", ("evaluation.test.json", "REPORT.md")),
            ("val", "REPORT.val.md", ("evaluation.val.json", "REPORT.val.md")),
        )
        for split, changed_name, output_names in cases:
            with self.subTest(split=split, changed_name=changed_name):
                for name in output_names:
                    (self.run_dir / name).unlink(missing_ok=True)
                original_publish = evaluate_module._publish_outputs

                def publish_then_change(outputs: dict[Path, bytes]) -> object:
                    created = original_publish(outputs)
                    (self.run_dir / changed_name).write_bytes(b"changed output\n")
                    return created

                try:
                    with patch.object(
                        evaluate_module,
                        "_publish_outputs",
                        side_effect=publish_then_change,
                    ), self.assertRaisesRegex(
                        EvaluationError, "output.*changed|changed.*output"
                    ):
                        evaluate_frozen_run(
                            self.run_dir, self.data_root, split=split
                        )
                finally:
                    for name in output_names:
                        (self.run_dir / name).unlink(missing_ok=True)

    def test_post_publication_failure_preserves_identical_existing_outputs(self) -> None:
        # Catches rollback deleting idempotent outputs that predated this invocation.
        import bc_core.evaluate as evaluate_module

        audit_path = self.data_root / "audit.json"
        audit_bytes = audit_path.read_bytes()
        original_publish = evaluate_module._publish_outputs
        with patch.object(
            evaluate_module,
            "load_validated_replay_bytes",
            return_value={"validated": True},
        ), patch.object(
            evaluate_module,
            "encode_game",
            side_effect=lambda source, replay: self._source_game(source),
        ):
            evaluate_frozen_run(self.run_dir, self.data_root, split="test")
            json_bytes = (self.run_dir / "evaluation.test.json").read_bytes()
            markdown_bytes = (self.run_dir / "REPORT.md").read_bytes()

            def publish_then_mutate(outputs: dict[Path, bytes]) -> object:
                result = original_publish(outputs)
                audit_path.write_bytes(audit_bytes + b"\n")
                return result

            try:
                with patch.object(
                    evaluate_module,
                    "_publish_outputs",
                    side_effect=publish_then_mutate,
                ), self.assertRaisesRegex(
                    EvaluationError, "changed during evaluation"
                ):
                    evaluate_frozen_run(
                        self.run_dir, self.data_root, split="test"
                    )

                self.assertEqual(
                    (self.run_dir / "evaluation.test.json").read_bytes(), json_bytes
                )
                self.assertEqual(
                    (self.run_dir / "REPORT.md").read_bytes(), markdown_bytes
                )
            finally:
                audit_path.write_bytes(audit_bytes)

    def test_publication_ledger_excludes_concurrent_identical_winner(self) -> None:
        # Catches rollback claiming and deleting an identical output another writer won.
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.test.json"
            content = b'{"result":"identical"}\n'

            def concurrent_link(source: object, destination: object) -> None:
                del source
                Path(destination).write_bytes(content)
                raise FileExistsError

            with patch.object(
                evaluate_module.os, "link", side_effect=concurrent_link
            ):
                created = evaluate_module._publish_outputs({output: content})

            self.assertEqual(created, {})
            self.assertEqual(output.read_bytes(), content)

    def test_rollback_preserves_replacement_inode_even_with_identical_bytes(self) -> None:
        # Catches byte equality being mistaken for exclusive file ownership.
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evaluation.test.json"
            content = b'{"result":"owned"}\n'
            created = evaluate_module._publish_outputs({output: content})
            owned_inode = (output.stat().st_dev, output.stat().st_ino)
            replacement = root / "replacement.json"
            replacement.write_bytes(content)
            replacement.replace(output)
            replacement_inode = (output.stat().st_dev, output.stat().st_ino)
            self.assertNotEqual(replacement_inode, owned_inode)

            evaluate_module._rollback_new_outputs(created)

            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes(), content)
            self.assertEqual(
                (output.stat().st_dev, output.stat().st_ino), replacement_inode
            )

    def test_rollback_removes_owned_inode_modified_in_place(self) -> None:
        # Catches refusing to remove an owned output merely because its bytes changed.
        import bc_core.evaluate as evaluate_module

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.test.json"
            created = evaluate_module._publish_outputs({output: b"original\n"})
            owned_inode = (output.stat().st_dev, output.stat().st_ino)
            output.write_bytes(b"modified in place\n")
            self.assertEqual(
                (output.stat().st_dev, output.stat().st_ino), owned_inode
            )

            evaluate_module._rollback_new_outputs(created)

            self.assertFalse(output.exists())

    def test_evaluation_never_modifies_selection_or_training_inputs(self) -> None:
        # Catches evaluation code writing into any frozen authorization/training file.
        paths = [
            self.run_dir / "selection.json",
            self.config_path,
            self.run_dir / "train_artifacts.npz",
            self.data_root / "training_audit.json",
            self.data_root / "audit.json",
            *(path for path in (self.run_dir / "clock").iterdir() if path.is_file()),
            *(path for path in (self.run_dir / "state").iterdir() if path.is_file()),
        ]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

        evaluate_frozen_run(self.run_dir, self.data_root, split="test")

        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(after, before)
        output = self.run_dir / "evaluation.test.json"
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            output.read_bytes(),
            (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def test_conflicting_test_rerun_is_rejected_before_any_other_output(self) -> None:
        # Catches overwriting a prior result or partially publishing around a conflict.
        output = self.run_dir / "evaluation.test.json"
        output.write_text("conflict\n", encoding="utf-8")

        with self.assertRaisesRegex(EvaluationError, "conflicting"):
            evaluate_frozen_run(self.run_dir, self.data_root, split="test")

        self.assertEqual(output.read_text(encoding="utf-8"), "conflict\n")
        self.assertFalse((self.run_dir / "REPORT.md").exists())

    def test_markdown_lists_all_six_gates_and_exact_decision(self) -> None:
        # Catches a human report that hides diagnostics, a gate, or the exact decision.
        report = evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        markdown = (self.run_dir / "REPORT.md").read_text(encoding="utf-8")

        self.assertEqual(len(report["gate"]["checks"]), 6)
        for name in report["gate"]["checks"]:
            self.assertIn(name, markdown)
        self.assertIn(report["decision"], markdown)
        self.assertEqual(markdown.rstrip().splitlines()[-1], report["decision"])
        self.assertIn(
            report["decision"],
            {"PROCEED TO MULTI-HEAD CLONING", "STOP AND DIAGNOSE V0"},
        )
        self.assertIn("| majority_actor | hand | 15 |", markdown)
        for family in (
            "actor_type",
            "seat",
            "day_band",
            "source_date",
            "route_family",
        ):
            self.assertIn(f"### {family}", markdown)
            for system in ("majority_actor", "clock", "state"):
                for value, metrics in report["test"][system]["slice_reports"][
                    family
                ].items():
                    support = sum(
                        int(class_report["support"])
                        for class_report in metrics["per_class"]
                    )
                    expected = (
                        f"| {system} | {value} | {support} | "
                        f"{metrics['top1']:.6f} | {metrics['top3']:.6f} | "
                        f"{metrics['macro_f1']:.6f} |"
                    )
                    self.assertIn(expected, markdown)

    def test_validation_split_uses_distinct_non_test_outputs(self) -> None:
        # Catches a validation rerun masquerading as the one-shot frozen test result.
        report = evaluate_frozen_run(self.run_dir, self.data_root, split="val")

        self.assertEqual(report["split"], "val")
        self.assertEqual(
            report["decision"], "VALIDATION ONLY — NO FROZEN TEST DECISION"
        )
        self.assertNotIn("test", report)
        self.assertNotIn("gate", report)
        self.assertTrue((self.run_dir / "evaluation.val.json").is_file())
        self.assertTrue((self.run_dir / "REPORT.val.md").is_file())
        self.assertFalse((self.run_dir / "evaluation.test.json").exists())
        self.assertFalse((self.run_dir / "REPORT.md").exists())

    def test_cli_prints_report_metrics_interval_and_decision(self) -> None:
        # Catches a CLI that evaluates but withholds the required operator summary.
        cli = self._load_evaluate_cli()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main(
                [
                    "--run-id",
                    self.run_dir.name,
                    "--data-root",
                    str(self.data_root),
                    "--runs-root",
                    str(self.runs_root),
                    "--split",
                    "test",
                ]
            )

        self.assertEqual(status, 0)
        summary = output.getvalue()
        self.assertIn("evaluation.test.json", summary)
        self.assertIn("state_macro_f1=", summary)
        self.assertIn("clock_macro_f1=", summary)
        self.assertIn("top1_delta_ci95=", summary)
        self.assertTrue(
            "PROCEED TO MULTI-HEAD CLONING" in summary
            or "STOP AND DIAGNOSE V0" in summary
        )

    def test_supported_training_cli_selection_verifies(self) -> None:
        # Catches accepting only the presence of selection.json without its provenance.
        verified = verify_selection(self.run_dir)

        self.assertEqual(
            verified["selection_identity"], self.selection["selection_identity"]
        )
        self.assertEqual(set(verified["models"]), {"clock", "state"})

    def test_selection_schema_is_rejected_before_opening_training_audit(self) -> None:
        # Catches following forged selection paths before validating the authorization.
        selection_path = self.run_dir / "selection.json"
        selection_bytes = selection_path.read_bytes()
        audit_path = self.data_root / "training_audit.json"
        audit_backup = self.data_root / "training_audit.backup"
        try:
            selection = json.loads(selection_bytes)
            selection["operations"][0] = "FORGED"
            selection_path.write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            audit_path.rename(audit_backup)
            audit_path.mkdir()

            with self.assertRaisesRegex(EvaluationError, "operations"):
                verify_selection(self.run_dir)
        finally:
            selection_path.write_bytes(selection_bytes)
            if audit_path.is_dir():
                audit_path.rmdir()
            if audit_backup.exists():
                audit_backup.rename(audit_path)

    def test_modified_checkpoint_is_rejected(self) -> None:
        # Catches trusting the selected path while ignoring its exact checkpoint bytes.
        path = Path(self.selection["models"]["clock"]["checkpoint_path"])
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"forged")
            with self.assertRaisesRegex(EvaluationError, "checkpoint|SHA-256"):
                verify_selection(self.run_dir)
        finally:
            path.write_bytes(original)

    def test_modified_config_is_rejected(self) -> None:
        # Catches using configuration content that differs from the frozen selection.
        original = self.config_path.read_bytes()
        try:
            self.config_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "configuration|selection"):
                verify_selection(self.run_dir)
        finally:
            self.config_path.write_bytes(original)

    def test_modified_train_artifact_is_rejected(self) -> None:
        # Catches accepting a byte-different train artifact with the same path.
        path = self.run_dir / "train_artifacts.npz"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"forged")
            with self.assertRaisesRegex(EvaluationError, "artifact|selection"):
                verify_selection(self.run_dir)
        finally:
            path.write_bytes(original)

    def test_modified_training_audit_is_rejected(self) -> None:
        # Catches accepting a semantically parseable but byte-different frozen audit.
        path = self.data_root / "training_audit.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(EvaluationError, "audit|selection"):
                verify_selection(self.run_dir)
        finally:
            path.write_bytes(original)

    def test_modified_training_history_is_rejected(self) -> None:
        # Catches trusting only the winner checkpoint while its selection history changed.
        path = self.run_dir / "clock" / "epochs.jsonl"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(EvaluationError, "history|JSON"):
                verify_selection(self.run_dir)
        finally:
            path.write_bytes(original)

    def test_transient_validation_shard_swap_cannot_change_history_replay(self) -> None:
        # Catches checkpoint-history validation reopening prepared validation paths.
        import bc_core.evaluate as evaluate_module

        record = next(
            item
            for item in self.training_audit["shards"]
            if item["split"] == "val" and item["episode_id"] == "val-game-000"
        )
        shard_path = self.data_root / str(record["shard_path"])
        shard_bytes = shard_path.read_bytes()
        replacement = self.data_root / "transient-val-swap.npz"
        game = test_train._encoded_game(
            "val",
            str(record["episode_id"]),
            np.asarray([4], dtype=np.int64),
            str(record["source_sha256"]),
        )
        game.metadata.update(
            {
                "source_path": record["shard_source_path"],
                "source_date": record["source_date"],
                "route_family": record["route_family"],
            }
        )
        write_shard(game, replacement)
        original_preflight = evaluate_module.preflight_resumed_model
        swapped = False

        def preflight_while_swapped(*args: object, **kwargs: object) -> object:
            nonlocal swapped
            if not swapped:
                shard_path.write_bytes(replacement.read_bytes())
                swapped = True
                try:
                    return original_preflight(*args, **kwargs)
                finally:
                    shard_path.write_bytes(shard_bytes)
            return original_preflight(*args, **kwargs)

        try:
            with patch.object(
                evaluate_module,
                "load_validated_replay_bytes",
                return_value={"validated": True},
            ), patch.object(
                evaluate_module,
                "encode_game",
                side_effect=lambda source, replay: self._source_game(source),
            ), patch.object(
                evaluate_module,
                "preflight_resumed_model",
                side_effect=preflight_while_swapped,
            ):
                verified = verify_selection(self.run_dir)

            self.assertTrue(swapped)
            self.assertEqual(
                verified["selection_identity"], self.selection["selection_identity"]
            )
        finally:
            shard_path.write_bytes(shard_bytes)
            replacement.unlink(missing_ok=True)

    def test_forged_consistent_history_is_rejected_before_full_audit(self) -> None:
        # Catches validating only internally consistent history instead of replaying val.
        model_dir = self.run_dir / "clock"
        history_path = model_dir / "epochs.jsonl"
        history_bytes = history_path.read_bytes()
        record_paths = sorted(model_dir.glob("epoch-*.json"))
        record_bytes = {path: path.read_bytes() for path in record_paths}
        try:
            records = [json.loads(line) for line in history_bytes.splitlines()]
            for record in records:
                record["validation_loss"] = float(record["validation_loss"]) + 0.25
                epoch_path = model_dir / f"epoch-{record['epoch']:03d}.json"
                epoch_path.write_bytes(self._canonical_bytes(record) + b"\n")
            history_path.write_bytes(
                b"".join(self._canonical_bytes(record) + b"\n" for record in records)
            )

            with patch(
                "bc_core.evaluate._validate_full_audit",
                side_effect=AssertionError("full audit opened before validation evidence"),
            ), self.assertRaisesRegex(EvaluationError, "validation evidence"):
                evaluate_frozen_run(self.run_dir, self.data_root, split="test")
        finally:
            history_path.write_bytes(history_bytes)
            for path, content in record_bytes.items():
                path.write_bytes(content)

    def test_modified_model_run_identity_is_rejected(self) -> None:
        # Catches validating histories without binding their model/config identity file.
        path = self.run_dir / "clock" / "run-identity.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(EvaluationError, "identity|configuration"):
                verify_selection(self.run_dir)
        finally:
            path.write_bytes(original)

    @staticmethod
    def _passing_report() -> dict[str, object]:
        return {
            "audit": {
                "all_checks_passed": True,
                "operation_support_covered": True,
            },
            "validation": {
                "state": {"macro_f1": 0.4},
                "clock": {"macro_f1": 0.3},
            },
            "test": {
                "state": {"macro_f1": 0.4},
                "clock": {"macro_f1": 0.3},
            },
            "bootstrap": {"ci95_low": 0.01},
            "complete_diagnostics": True,
        }

    @classmethod
    def _write_corpus_and_prepared_data(cls) -> None:
        config = load_config(cls.config_path)
        corpus_root = cls.repository_root / str(config["corpus_root"])
        cls.corpus_root = corpus_root
        cls.data_root.mkdir()
        for split in ("train", "val", "test"):
            (corpus_root / split).mkdir(parents=True)
            (corpus_root.parent / "source-archive" / split).mkdir(parents=True)
            (cls.data_root / split).mkdir()

        manifest_rows: list[dict[str, str]] = []
        records: list[dict[str, object]] = []
        split_counts = {
            split: {"games": 0, "samples": 0, "label_counts": [0] * len(OPERATIONS)}
            for split in ("train", "val", "test")
        }
        for split, game_count in (("train", 70), ("val", 15), ("test", 15)):
            for index in range(game_count):
                episode_id = f"{split}-game-{index:03d}"
                replay_path = corpus_root / split / f"{episode_id}.json"
                archive_path = (
                    corpus_root.parent / "source-archive" / split / f"{episode_id}.json"
                )
                replay_bytes = (
                    json.dumps(
                        {"episode_id": episode_id, "fixture": "immutable-source"},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                archive_path.write_bytes(replay_bytes)
                replay_path.symlink_to(archive_path)
                source_sha256 = hashlib.sha256(replay_bytes).hexdigest()
                source_date = f"2026-08-{19 + index % 3:02d}"
                route_family = f"route-{index % 2}"
                manifest_rows.append(
                    {
                        "episode_id": episode_id,
                        "split": split,
                        "source_date": source_date,
                        "source_path": f"archive/{episode_id}.json",
                        "source_sha256": source_sha256,
                        "route_family": route_family,
                    }
                )

                labels = (
                    np.arange(len(OPERATIONS), dtype=np.int64)
                    if split == "train" and index == 0
                    else np.asarray([index % len(OPERATIONS)], dtype=np.int64)
                )
                game = test_train._encoded_game(
                    split, episode_id, labels, source_sha256
                )
                game.metadata.update(
                    {
                        "source_path": str(replay_path.resolve()),
                        "source_date": source_date,
                        "route_family": route_family,
                    }
                )
                relative = Path(split) / f"{episode_id}.npz"
                identity = write_shard(game, cls.data_root / relative)
                counts = np.bincount(
                    game.label, minlength=len(OPERATIONS)
                ).astype(np.int64).tolist()
                records.append(
                    {
                        "episode_id": episode_id,
                        "label_counts": counts,
                        "sample_count": int(game.label.shape[0]),
                        "shard_identity": identity,
                        "shard_path": relative.as_posix(),
                        "source_path": f"archive/{episode_id}.json",
                        "shard_source_path": str(replay_path.resolve()),
                        "source_sha256": source_sha256,
                        "source_date": source_date,
                        "route_family": route_family,
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
                    }
                )
                summary = split_counts[split]
                summary["games"] = int(summary["games"]) + 1
                summary["samples"] = int(summary["samples"]) + int(
                    game.label.shape[0]
                )
                summary["label_counts"] = (
                    np.asarray(summary["label_counts"], dtype=np.int64)
                    + np.asarray(counts, dtype=np.int64)
                ).tolist()

        manifest_path = corpus_root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=(
                    "episode_id",
                    "split",
                    "source_date",
                    "source_path",
                    "source_sha256",
                    "route_family",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        split_summary = {
            "schema_version": 1,
            "selected_win_count": 100,
            "unique_episode_ids": 100,
            "unique_source_hashes": 100,
            "split_counts": {"train": 70, "val": 15, "test": 15},
            "stratify_fields": [
                "source_date",
                "opponent",
                "ryo_seat",
                "margin_quartile",
                "shop_profile",
                "route_family",
            ],
        }
        (corpus_root / "split_summary.json").write_text(
            json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        cls.prepared_records = sorted(
            records, key=lambda record: (record["split"], record["episode_id"])
        )
        cls.split_counts = split_counts
        training = _training_audit(
            config,
            cls._canonical_bytes(config),
            cls.prepared_records,
            split_counts,
            smoke_mode=False,
        )
        cls.training_audit = training
        (cls.data_root / "training_audit.json").write_text(
            json.dumps(training, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cls._write_full_audit()

    @classmethod
    def _write_full_audit(cls) -> None:
        training = cls.training_audit
        records = cls.prepared_records
        splits = cls.split_counts
        shard_identities = sorted(str(record["shard_identity"]) for record in records)
        source_hashes = sorted(str(record["source_sha256"]) for record in records)
        manifest = {
            "manifest_csv_sha256": hashlib.sha256(
                (cls.corpus_root / "manifest.csv").read_bytes()
            ).hexdigest(),
            "split_summary_json_sha256": hashlib.sha256(
                (cls.corpus_root / "split_summary.json").read_bytes()
            ).hexdigest(),
        }
        preparation_payload = {
            "config": training["config"]["canonical"],
            **manifest,
            "shard_identities": shard_identities,
            "source_hashes": source_hashes,
        }
        total_counts = np.sum(
            np.asarray(
                [splits[name]["label_counts"] for name in ("train", "val", "test")]
            ),
            axis=0,
        ).tolist()
        checks = {
            "cross_split_leakage_absent": True,
            "episode_ids_unique": True,
            "manifest_validated": True,
            "operation_support_covered": True,
            "selected_replays_validated": True,
            "shard_identities_verified": True,
            "source_hashes_unique": True,
            "tensor_shapes_validated": True,
        }
        total_games = sum(int(summary["games"]) for summary in splits.values())
        total_samples = sum(int(summary["samples"]) for summary in splits.values())
        audit = {
            "all_checks_passed": True,
            "checks": checks,
            "config": training["config"],
            "label_counts": total_counts,
            "manifest": manifest,
            "operations": list(OPERATIONS),
            "preparation_identity": hashlib.sha256(
                cls._canonical_bytes(preparation_payload)
            ).hexdigest(),
            "schema_version": "ryo-preparation-v0",
            "shard_identities": shard_identities,
            "shards": records,
            "smoke_mode": False,
            "source_hashes": source_hashes,
            "splits": splits,
            "tensor_shapes": {
                "actor_features": ["samples", ACTOR_DIM],
                "argument_item": ["samples"],
                "argument_quantity": ["samples"],
                "global_features": [719, GLOBAL_DIM],
                "grid": [719, GRID_CHANNELS, 10, 10],
                "label": ["samples"],
                "step_index": ["samples"],
            },
            "totals": {"games": total_games, "samples": total_samples},
            "trainable": True,
            "validation_counts": {
                "encoded_games": total_games,
                "manifest_sources": total_games,
                "selected_replays": total_games,
                "written_shards": total_games,
            },
        }
        cls.full_audit = audit
        (cls.data_root / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def _write_training_run(cls) -> None:
        config = load_config(cls.config_path)
        cli = test_train.TrainTest()._load_cli()
        cls.run_dir.mkdir(parents=True)
        train_paths = [
            cls.data_root / str(record["shard_path"])
            for record in cls.training_audit["shards"]
            if record["split"] == "train"
        ]
        val_paths = [
            cls.data_root / str(record["shard_path"])
            for record in cls.training_audit["shards"]
            if record["split"] == "val"
        ]
        stats, class_counts, class_weights, majority = fit_train_artifacts(
            train_paths, weight_cap=float(config["training"]["weight_cap"])
        )
        artifacts_path = cls.run_dir / "train_artifacts.npz"
        artifact_identity = save_train_artifacts(
            artifacts_path,
            stats,
            class_counts,
            class_weights,
            majority,
            cli._artifact_metadata(config, cls.training_audit),
        )
        artifact_sha256 = hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
        validation_loader = torch.utils.data.DataLoader(
            ShardDataset(val_paths, stats),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_examples,
            generator=torch.Generator(device="cpu").manual_seed(int(config["seed"])),
        )
        majority_report = evaluate_majority(validation_loader, majority)
        (cls.run_dir / "majority.validation.json").write_text(
            json.dumps(majority_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        device = choose_device()
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights.copy()).to(device)
        )
        for name, model in (
            ("clock", ClockOnlyModel()),
            ("state", StateAwareModel()),
        ):
            checkpoint_metadata = cli._checkpoint_metadata(
                config,
                cls.training_audit,
                stats,
                class_weights,
                artifact_identity,
                artifact_sha256,
                model,
            )
            model_dir = cls.run_dir / name
            model_dir.mkdir()
            test_train._write_identity(
                model_dir / "run-identity.json",
                {
                    "model_name": name,
                    "config": config,
                    "checkpoint_metadata": checkpoint_metadata,
                },
                False,
            )
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            validation = evaluate_loader(
                model, validation_loader, criterion, device, name
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
                    "preparation_identity": cls.training_audit["training_identity"],
                    "validation_loss": validation["loss"],
                    "validation_metrics": validation["metrics"],
                    "validation_slice_reports": validation["slice_reports"],
                }
                encoded = cls._canonical_bytes(record) + b"\n"
                (model_dir / f"epoch-{epoch:03d}.json").write_bytes(encoded)
                records.append(record)
            (model_dir / "epochs.jsonl").write_bytes(
                b"".join(cls._canonical_bytes(record) + b"\n" for record in records)
            )

        cls.selection = cli.train_run(
            cls.config_path,
            cls.run_dir.name,
            cls.data_root,
            cls.runs_root,
            resume=True,
        )

    @staticmethod
    def _canonical_bytes(value: object) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @classmethod
    def _refresh_full_audit(cls, audit: dict[str, object]) -> None:
        records = audit["shards"]
        assert isinstance(records, list)
        splits: dict[str, dict[str, object]] = {}
        for split in ("train", "val", "test"):
            selected = [record for record in records if record["split"] == split]
            splits[split] = {
                "games": len(selected),
                "samples": sum(int(record["sample_count"]) for record in selected),
                "label_counts": np.sum(
                    np.asarray(
                        [record["label_counts"] for record in selected],
                        dtype=np.int64,
                    ),
                    axis=0,
                ).tolist(),
            }
        audit["splits"] = splits
        audit["label_counts"] = np.sum(
            np.asarray(
                [splits[name]["label_counts"] for name in ("train", "val", "test")],
                dtype=np.int64,
            ),
            axis=0,
        ).tolist()
        audit["totals"] = {
            "games": sum(int(summary["games"]) for summary in splits.values()),
            "samples": sum(int(summary["samples"]) for summary in splits.values()),
        }
        audit["shard_identities"] = sorted(
            str(record["shard_identity"]) for record in records
        )
        audit["source_hashes"] = sorted(
            str(record["source_sha256"]) for record in records
        )
        validation_counts = audit["validation_counts"]
        assert isinstance(validation_counts, dict)
        for name in validation_counts:
            validation_counts[name] = int(audit["totals"]["games"])
        manifest = audit["manifest"]
        assert isinstance(manifest, dict)
        preparation_payload = {
            "config": audit["config"]["canonical"],
            "manifest_csv_sha256": manifest["manifest_csv_sha256"],
            "shard_identities": audit["shard_identities"],
            "source_hashes": audit["source_hashes"],
            "split_summary_json_sha256": manifest["split_summary_json_sha256"],
        }
        audit["preparation_identity"] = hashlib.sha256(
            cls._canonical_bytes(preparation_payload)
        ).hexdigest()

    @staticmethod
    def _tensor_shapes(game: object) -> dict[str, list[int]]:
        return {
            "actor_features": list(game.actor_features.shape),
            "argument_item": list(game.argument_item.shape),
            "argument_quantity": list(game.argument_quantity.shape),
            "global_features": list(game.global_features.shape),
            "grid": list(game.grid.shape),
            "label": list(game.label.shape),
            "step_index": list(game.step_index.shape),
        }

    @staticmethod
    def _source_game(source: object) -> object:
        index = int(str(source.episode_id).rsplit("-", 1)[1])
        labels = (
            np.arange(len(OPERATIONS), dtype=np.int64)
            if source.split == "train" and index == 0
            else np.asarray([index % len(OPERATIONS)], dtype=np.int64)
        )
        game = test_train._encoded_game(
            str(source.split), str(source.episode_id), labels, str(source.sha256)
        )
        game.metadata.update(
            {
                "source_path": str(source.path),
                "source_date": str(source.source_date),
                "route_family": str(source.route_family),
            }
        )
        return game

    @staticmethod
    def _load_evaluate_cli() -> object:
        script = Path(__file__).parents[1] / "scripts" / "evaluate_v0.py"
        specification = importlib.util.spec_from_file_location(
            "evaluate_v0_cli", script
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
