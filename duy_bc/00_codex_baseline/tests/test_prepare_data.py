import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bc_core.constants import ACTOR_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATION_TO_ID
from bc_core.features import EncodedGame
from bc_core.prepare import prepare
from bc_core.replay import SourceReplay
from bc_core.scripts_support import atomic_json_write


class PrepareAuditTest(unittest.TestCase):
    def test_atomic_json_write_refuses_mismatched_existing_content(self) -> None:
        # Catches a rerun silently replacing an audit from different inputs.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            atomic_json_write(path, {"games": 1})
            atomic_json_write(path, {"games": 1})
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                atomic_json_write(path, {"games": 2})
            self.assertEqual(json.loads(path.read_text()), {"games": 1})

    def test_atomic_json_write_concurrent_identical_writers_share_one_winner(self) -> None:
        # Catches shared temporary names and check-then-replace publication races.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(atomic_json_write, path, {"games": 1})
                    for _ in range(4)
                ]
            self.assertTrue(all(future.exception() is None for future in futures))
            self.assertEqual(json.loads(path.read_text()), {"games": 1})
            self.assertEqual(list(path.parent.glob(".audit.json.*.tmp")), [])

    def test_prepare_publishes_deterministic_complete_audit(self) -> None:
        # Catches omitted counts/hashes/identities and source-order-dependent audits.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            sources = (
                self._source(corpus_root, "val", "episode-v", "b"),
                self._source(corpus_root, "train", "episode-z", "c"),
                self._source(corpus_root, "test", "episode-a", "a"),
            )
            output_root = root / "prepared"
            audit = self._prepare_with_labels(
                config_path,
                output_root,
                sources,
                {split: [OPERATION_TO_ID["PASS"]] for split in ("train", "val", "test")},
            )

            self.assertEqual(audit, json.loads((output_root / "audit.json").read_text()))
            self.assertFalse(audit["smoke_mode"])
            self.assertTrue(audit["trainable"])
            self.assertTrue(audit["all_checks_passed"])
            self.assertTrue(all(audit["checks"].values()))
            self.assertEqual(
                audit["manifest"],
                {
                    "manifest_csv_sha256": self._sha256(corpus_root / "manifest.csv"),
                    "split_summary_json_sha256": self._sha256(corpus_root / "split_summary.json"),
                },
            )
            self.assertEqual(audit["source_hashes"], sorted(source.sha256 for source in sources))
            self.assertEqual(audit["shard_identities"], sorted(audit["shard_identities"]))
            self.assertEqual(
                [(item["split"], item["episode_id"]) for item in audit["shards"]],
                [("test", "episode-a"), ("train", "episode-z"), ("val", "episode-v")],
            )
            episode_by_split = {
                "train": "episode-z",
                "val": "episode-v",
                "test": "episode-a",
            }
            for split in ("train", "val", "test"):
                self.assertEqual(audit["splits"][split]["games"], 1)
                self.assertEqual(audit["splits"][split]["samples"], 1)
                self.assertEqual(
                    audit["splits"][split]["label_counts"][OPERATION_TO_ID["PASS"]], 1
                )
                self.assertTrue(
                    (output_root / split / f"{episode_by_split[split]}.npz").exists()
                )
            self.assertEqual(audit["totals"], {"games": 3, "samples": 3})
            self.assertEqual(audit["label_counts"][OPERATION_TO_ID["PASS"]], 3)
            self.assertEqual(
                audit["tensor_shapes"],
                {
                    "actor_features": ["samples", ACTOR_DIM],
                    "argument_item": ["samples"],
                    "argument_quantity": ["samples"],
                    "global_features": [719, GLOBAL_DIM],
                    "grid": [719, GRID_CHANNELS, 10, 10],
                    "label": ["samples"],
                    "step_index": ["samples"],
                },
            )
            self.assertEqual(len(audit["preparation_identity"]), 64)

    def test_training_audit_contains_only_train_and_validation_projection(self) -> None:
        # Catches training authorization inheriting any test path, identity, or count.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            sources = tuple(
                self._source(corpus_root, split, f"episode-{split}", character)
                for split, character in (("train", "a"), ("val", "b"), ("test", "c"))
            )
            output_root = root / "prepared"
            self._prepare_with_labels(
                config_path,
                output_root,
                sources,
                {split: [OPERATION_TO_ID["PASS"]] for split in ("train", "val", "test")},
            )
            training = json.loads((output_root / "training_audit.json").read_text())

        self.assertEqual(training["schema_version"], "ryo-training-preparation-v0")
        self.assertEqual(set(training["splits"]), {"train", "val"})
        self.assertEqual({record["split"] for record in training["shards"]}, {"train", "val"})
        self.assertNotIn("manifest_csv_sha256", training)
        self.assertNotIn("split_summary_json_sha256", training)
        self.assertNotIn("test", json.dumps(training, sort_keys=True))
        self.assertEqual(len(training["safe_manifest_sha256"]), 64)
        self.assertEqual(len(training["training_identity"]), 64)

    def test_training_audit_identity_is_independent_of_test_records(self) -> None:
        # Catches a safe identity that hashes the full manifest or any test projection.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            safe = (
                self._source(corpus_root, "train", "episode-train", "a"),
                self._source(corpus_root, "val", "episode-val", "b"),
            )
            first_sources = safe + (self._source(corpus_root, "test", "episode-test-a", "c"),)
            second_sources = safe + (self._source(corpus_root, "test", "episode-test-b", "d"),)
            labels = {split: [OPERATION_TO_ID["PASS"]] for split in ("train", "val", "test")}
            self._prepare_with_labels(config_path, root / "first", first_sources, labels)
            self._prepare_with_labels(config_path, root / "second", second_sources, labels)
            first = json.loads((root / "first" / "training_audit.json").read_text())
            second = json.loads((root / "second" / "training_audit.json").read_text())

        self.assertEqual(first, second)

    def test_cli_returns_nonzero_when_eval_operation_is_absent_from_train(self) -> None:
        # Catches publishing trainable data whose validation labels have no train support.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            sources = tuple(
                self._source(corpus_root, split, f"episode-{split[0]}", chr(97 + index))
                for index, split in enumerate(("train", "val", "test"))
            )
            cli = self._load_cli()
            error_output = io.StringIO()
            with (
                patch("bc_core.prepare.load_split_manifest", return_value=sources),
                patch("bc_core.prepare.load_validated_replay", return_value={}),
                patch(
                    "bc_core.prepare.encode_game",
                    side_effect=lambda source, replay: self._game(
                        source,
                        [
                            OPERATION_TO_ID["NORTH"]
                            if source.split == "val"
                            else OPERATION_TO_ID["PASS"]
                        ],
                    ),
                ),
                contextlib.redirect_stderr(error_output),
            ):
                status = cli.main(
                    ["--config", str(config_path), "--output-root", str(root / "prepared")]
                )
            self.assertNotEqual(status, 0)
            self.assertIn("operation support", error_output.getvalue())

    def test_limited_prepare_is_non_trainable_and_does_not_claim_support(self) -> None:
        # Catches treating a small smoke sample as a full-corpus training input.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            sources = tuple(
                self._source(corpus_root, split, f"episode-{split[0]}", chr(97 + index))
                for index, split in enumerate(("train", "val", "test"))
            )
            audit = self._prepare_with_labels(
                config_path,
                root / "prepared",
                sources,
                {split: [OPERATION_TO_ID["PASS"]] for split in ("train", "val", "test")},
                limit_per_split=1,
            )
            self.assertTrue(audit["smoke_mode"])
            self.assertFalse(audit["trainable"])
            self.assertFalse(audit["all_checks_passed"])
            self.assertFalse(audit["checks"]["operation_support_covered"])
            self.assertTrue(audit["checks"]["manifest_validated"])
            self.assertTrue(audit["checks"]["selected_replays_validated"])

    def test_prepare_refuses_output_root_resolving_to_corpus_root(self) -> None:
        # Catches writing generated shards into the immutable source corpus via a symlink.
        with tempfile.TemporaryDirectory() as directory:
            root, config_path, corpus_root = self._repository(Path(directory))
            output_alias = root / "corpus-output-alias"
            output_alias.symlink_to(corpus_root, target_is_directory=True)
            before = sorted(path.relative_to(corpus_root) for path in corpus_root.rglob("*"))

            with (
                patch("bc_core.prepare.load_split_manifest", return_value=()),
                self.assertRaisesRegex(
                    ValueError,
                    rf"output_root={corpus_root.resolve()}.*corpus_root={corpus_root.resolve()}",
                ),
            ):
                prepare(config_path, output_alias)

            after = sorted(path.relative_to(corpus_root) for path in corpus_root.rglob("*"))
            self.assertEqual(after, before)
            self.assertFalse((corpus_root / "audit.json").exists())
            for split in ("train", "val", "test"):
                self.assertFalse((corpus_root / split).exists())

    def test_prepare_refuses_output_root_nested_beneath_corpus_root(self) -> None:
        # Catches creating any generated-data directory inside the immutable corpus.
        with tempfile.TemporaryDirectory() as directory:
            _, config_path, corpus_root = self._repository(Path(directory))
            output_root = corpus_root / "generated" / "nested"
            self.assertFalse(output_root.exists())

            with (
                patch("bc_core.prepare.load_split_manifest", return_value=()),
                self.assertRaisesRegex(
                    ValueError,
                    rf"output_root={output_root.resolve()}.*corpus_root={corpus_root.resolve()}",
                ),
            ):
                prepare(config_path, output_root)

            self.assertFalse(output_root.exists())
            self.assertFalse((corpus_root / "generated").exists())
            self.assertFalse((corpus_root / "audit.json").exists())

    def _repository(self, temporary_root: Path) -> tuple[Path, Path, Path]:
        root = temporary_root / "repository"
        config_path = root / "duy_bc" / "00_codex_baseline" / "configs" / "v0.json"
        config_path.parent.mkdir(parents=True)
        source_config = Path(__file__).parents[1] / "configs" / "v0.json"
        config_path.write_text(source_config.read_text())
        corpus_root = root / "duy_explore" / "ryo_hasegawa_100_stratified"
        corpus_root.mkdir(parents=True)
        (corpus_root / "manifest.csv").write_text("three-game fixture\n")
        (corpus_root / "split_summary.json").write_text('{"fixture": true}\n')
        return root, config_path, corpus_root

    def _source(
        self, corpus_root: Path, split: str, episode_id: str, hash_character: str
    ) -> SourceReplay:
        return SourceReplay(
            split,
            episode_id,
            corpus_root / split / f"{episode_id}.json",
            hash_character * 64,
            "21-08",
            f"route-{split}",
            f"audit/{episode_id}.json",
        )

    def _game(self, source: SourceReplay, labels: list[int]) -> EncodedGame:
        sample_count = len(labels)
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
            step_index=np.zeros(sample_count, dtype=np.int32),
            label=np.asarray(labels, dtype=np.int64),
            argument_item=np.full(sample_count, -1, dtype=np.int32),
            argument_quantity=np.full(sample_count, -1, dtype=np.int32),
            metadata={
                "schema_version": "ryo-features-v0",
                "split": source.split,
                "episode_id": source.episode_id,
                "ryo_seat": 0,
                "source_path": str(source.path),
                "source_sha256": source.sha256,
                "source_date": source.source_date,
                "route_family": source.route_family,
                "sample_count": sample_count,
                "shapes": shapes,
            },
        )

    def _prepare_with_labels(
        self,
        config_path: Path,
        output_root: Path,
        sources: tuple[SourceReplay, ...],
        labels: dict[str, list[int]],
        *,
        limit_per_split: int | None = None,
    ) -> dict:
        with (
            patch("bc_core.prepare.load_split_manifest", return_value=sources),
            patch("bc_core.prepare.load_validated_replay", return_value={}),
            patch(
                "bc_core.prepare.encode_game",
                side_effect=lambda source, replay: self._game(source, labels[source.split]),
            ),
        ):
            return prepare(config_path, output_root, limit_per_split=limit_per_split)

    def _load_cli(self):
        path = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
        spec = importlib.util.spec_from_file_location("prepare_data_cli_test", path)
        if spec is None or spec.loader is None:
            self.fail("could not load preparation CLI")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
