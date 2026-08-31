import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fixtures import make_replay
from bc_core.features import (
    FeatureError,
    encode_game,
    logical_shard_identity,
    read_shard,
    write_shard,
)
from bc_core.replay import SourceReplay


class ShardTest(unittest.TestCase):
    def source(self) -> SourceReplay:
        return SourceReplay(
            "train", "fixture-game", Path("fixture.json"), "a" * 64, "21-08", "route-a"
        )

    def test_round_trip_preserves_arrays_metadata_and_identity(self) -> None:
        # Catches omissions or coercions in the on-disk shard representation.
        game = encode_game(self.source(), make_replay())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.npz"
            first = write_shard(game, path)
            loaded = read_shard(path)
            second = logical_shard_identity(loaded)
        self.assertEqual(first, second)
        self.assertEqual(game.metadata, loaded.metadata)
        for field in (
            "grid",
            "global_features",
            "actor_features",
            "step_index",
            "label",
            "argument_item",
            "argument_quantity",
        ):
            np.testing.assert_array_equal(getattr(game, field), getattr(loaded, field))

    def test_identical_existing_shard_is_accepted(self) -> None:
        # Catches treating a deterministic rerun as a collision.
        game = encode_game(self.source(), make_replay())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.npz"
            first = write_shard(game, path)
            second = write_shard(game, path)
            loaded = read_shard(path)
        self.assertEqual(first, second)
        self.assertEqual(second, logical_shard_identity(loaded))

    def test_nonidentical_existing_shard_is_rejected_without_overwrite(self) -> None:
        # Catches silently overwriting a shard from different logical contents.
        game = encode_game(self.source(), make_replay())
        changed_label = game.label.copy()
        changed_label[0] = 0
        changed = replace(game, label=changed_label)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.npz"
            original_identity = write_shard(game, path)
            with self.assertRaisesRegex(FeatureError, "non-identical"):
                write_shard(changed, path)
            retained_identity = logical_shard_identity(read_shard(path))
        self.assertEqual(original_identity, retained_identity)

    def test_separately_encoded_games_have_the_same_logical_identity(self) -> None:
        # Catches using zip timestamps or object identity in the shard digest.
        first = encode_game(self.source(), make_replay())
        second = encode_game(self.source(), make_replay())
        self.assertEqual(logical_shard_identity(first), logical_shard_identity(second))

    def test_read_rejects_corrupt_shape_and_dtype(self) -> None:
        # Catches accepting arrays that downstream indexing cannot safely consume.
        game = encode_game(self.source(), make_replay())
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            metadata = np.asarray(
                json.dumps(game.metadata, sort_keys=True, separators=(",", ":"))
            )
            arrays = {
                "grid": game.grid,
                "global_features": game.global_features,
                "actor_features": game.actor_features,
                "step_index": game.step_index,
                "label": game.label,
                "argument_item": game.argument_item,
                "argument_quantity": game.argument_quantity,
                "metadata": metadata,
            }
            bad_shape = directory_path / "bad-shape.npz"
            np.savez_compressed(bad_shape, **(arrays | {"grid": game.grid[:, :43]}))
            with self.assertRaisesRegex(FeatureError, "grid shape"):
                read_shard(bad_shape)

            bad_dtype = directory_path / "bad-dtype.npz"
            np.savez_compressed(bad_dtype, **(arrays | {"step_index": game.step_index.astype(np.int64)}))
            with self.assertRaisesRegex(FeatureError, "step_index dtype"):
                read_shard(bad_dtype)


if __name__ == "__main__":
    unittest.main()
