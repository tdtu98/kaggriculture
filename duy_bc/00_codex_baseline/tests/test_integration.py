import tempfile
import unittest
from pathlib import Path

import torch

from fixtures import make_replay
from bc_core.constants import OPERATIONS
from bc_core.dataset import ShardDataset, collate_examples, fit_train_artifacts
from bc_core.features import encode_game, write_shard
from bc_core.checkpoints import architecture_metadata, load_checkpoint, save_checkpoint
from bc_core.replay import SourceReplay
from bc_core.train import train_one_epoch
from model.state import StateAwareModel


class IntegrationTest(unittest.TestCase):
    def source(self) -> SourceReplay:
        return SourceReplay(
            "train",
            "fixture-game",
            Path("fixture.json"),
            "a" * 64,
            "2026-08-21",
            "route-a",
        )

    def metadata(self, stats, weights, model: StateAwareModel) -> dict:
        return {
            "schema_version": "ryo-bc-v0",
            "feature_schema_version": "ryo-features-v0",
            "vocabularies": {"operations": list(OPERATIONS)},
            "normalization": stats,
            "class_weights": weights,
            "manifest_sha256": "b" * 64,
            "architecture": architecture_metadata(model),
        }

    def test_replay_to_checkpoint_reload_has_identical_logits(self) -> None:
        # Catches a broken contract anywhere from shifted replay encoding through
        # one optimizer update and immutable checkpoint reload.
        replay = make_replay(hands=1)
        for state in range(1, 720):
            operation = OPERATIONS[(state - 1) % len(OPERATIONS)]
            replay["steps"][state][0]["action"]["farmer"] = [operation]
            replay["steps"][state][0]["action"]["hands"] = [[operation]]
        encoded = encode_game(self.source(), replay)
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "train" / "fixture.npz"
            shard.parent.mkdir()
            write_shard(encoded, shard)
            stats, _counts, weights, _majority = fit_train_artifacts([shard])
            dataset = ShardDataset([shard], stats)
            batch = collate_examples([dataset[0], dataset[1]])
            model = StateAwareModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.from_numpy(weights).float()
            )
            train_one_epoch(
                model,
                [batch],
                optimizer,
                criterion,
                torch.device("cpu"),
                "state",
            )
            model.eval()
            before = model(batch.grid, batch.global_features, batch.actor_features)
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                self.metadata(stats, weights, model),
                1,
            )
            restored = StateAwareModel()
            load_checkpoint(checkpoint, restored)
            restored.eval()
            after = restored(batch.grid, batch.global_features, batch.actor_features)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
