import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fixtures import make_replay
from bc_core.replay import (
    ReplayError,
    SourceReplay,
    iter_decisions,
    load_split_manifest,
    load_validated_replay,
    operation_and_arguments,
)


class ReplayTest(unittest.TestCase):
    def source(self) -> SourceReplay:
        return SourceReplay("train", "fixture-game", Path("fixture.json"), "a" * 64, "21-08", "route-a")

    def write_replay(self, replay: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.json"
        path.write_text(json.dumps(replay), encoding="utf-8")
        return path

    def test_shift_excludes_terminal_state(self) -> None:
        replay = make_replay(hands=2)
        replay["steps"][1][0]["action"] = {
            "farmer": ["NORTH"],
            "hands": [["PLANT", "WHEAT", 2], ["WATER"]],
            "market": [],
        }
        decisions = tuple(iter_decisions(self.source(), replay))
        self.assertEqual(len(decisions), 719)
        self.assertEqual(decisions[0].step, 0)
        self.assertEqual(decisions[-1].step, 718)
        self.assertEqual(decisions[0].action["farmer"], ["NORTH"])
        self.assertNotIn("reward", decisions[0].observation)

    def test_unknown_operation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReplayError, "UNKNOWN"):
            operation_and_arguments(["UNKNOWN"])

    def test_operation_arguments_preserve_item_and_quantity(self) -> None:
        self.assertEqual(operation_and_arguments(["PLANT", "WHEAT", 2]), (7, 0, 2))
        self.assertEqual(operation_and_arguments("PASS"), (4, -1, -1))

    def test_hand_action_mismatch_names_step_and_seat(self) -> None:
        replay = make_replay(hands=2)
        replay["steps"][1][0]["action"]["hands"] = []
        with self.assertRaisesRegex(ReplayError, "step=0.*seat=0"):
            tuple(iter_decisions(self.source(), replay))

    def test_load_rejects_wrong_version(self) -> None:
        replay = make_replay()
        replay["module_version"] = "wrong"
        with self.assertRaisesRegex(ReplayError, "module_version"):
            load_validated_replay(self.source_for(self.write_replay(replay)), "1.32.7")

    def test_extraction_rejects_wrong_version_without_loader(self) -> None:
        replay = make_replay()
        replay["module_version"] = "0.0.0"
        with self.assertRaisesRegex(ReplayError, "module_version"):
            tuple(iter_decisions(self.source(), replay))

    def test_source_exposes_contract_sha256_name(self) -> None:
        self.assertEqual(self.source().sha256, "a" * 64)

    def test_load_rejects_wrong_state_count(self) -> None:
        replay = make_replay()
        replay["steps"].pop()
        with self.assertRaisesRegex(ReplayError, "720"):
            load_validated_replay(self.source_for(self.write_replay(replay)), "1.32.7")

    def test_load_rejects_missing_or_duplicate_ryo(self) -> None:
        for names in (["Opponent", "Opponent 2"], ["Ryo Hasegawa", "Ryo Hasegawa"]):
            with self.subTest(names=names):
                replay = make_replay()
                replay["info"]["TeamNames"] = names
                replay["info"]["Agents"] = [{"Name": name} for name in names]
                with self.assertRaisesRegex(ReplayError, "Ryo Hasegawa"):
                    load_validated_replay(self.source_for(self.write_replay(replay)), "1.32.7")

    def test_load_rejects_nonwinning_ryo(self) -> None:
        replay = make_replay()
        replay["rewards"] = [0, 1]
        with self.assertRaisesRegex(ReplayError, "winning"):
            load_validated_replay(self.source_for(self.write_replay(replay)), "1.32.7")

    def test_load_rejects_non_done_terminal_status(self) -> None:
        replay = make_replay()
        replay["statuses"] = ["ACTIVE", "DONE"]
        with self.assertRaisesRegex(ReplayError, "DONE"):
            load_validated_replay(self.source_for(self.write_replay(replay)), "1.32.7")

    def test_iter_rejects_missing_observation_or_action(self) -> None:
        for key, step in (("observation", 0), ("action", 1)):
            with self.subTest(key=key):
                replay = make_replay()
                del replay["steps"][step][0][key]
                with self.assertRaisesRegex(ReplayError, key):
                    tuple(iter_decisions(self.source(), replay))

    def source_for(self, path: Path) -> SourceReplay:
        return SourceReplay("train", "fixture-game", path, "a" * 64, "21-08", "route-a")

    def test_manifest_rejects_duplicate_episode_hash_and_cross_split_reuse(self) -> None:
        for mutation in ("episode", "hash", "split"):
            with self.subTest(mutation=mutation):
                with self.corpus() as root:
                    rows = self.manifest_rows(root)
                    if mutation == "episode":
                        rows[1]["episode_id"] = rows[0]["episode_id"]
                    elif mutation == "hash":
                        rows[1]["source_sha256"] = rows[0]["source_sha256"]
                    else:
                        rows[1]["episode_id"] = rows[0]["episode_id"]
                        rows[1]["split"] = "val"
                    self.write_manifest(root, rows)
                    with self.assertRaisesRegex(ReplayError, "row 1.*row 2"):
                        load_split_manifest(Path(root))

    def test_manifest_rejects_hash_mismatch(self) -> None:
        with self.corpus() as root:
            rows = self.manifest_rows(root)
            rows[0]["source_sha256"] = "0" * 64
            self.write_manifest(root, rows)
            with self.assertRaisesRegex(ReplayError, "hash mismatch"):
                load_split_manifest(Path(root))

    def corpus(self):
        return tempfile.TemporaryDirectory()

    def manifest_rows(self, root: str) -> list[dict[str, str]]:
        root_path = Path(root)
        rows = []
        for index in range(100):
            split = "train" if index < 70 else "val" if index < 85 else "test"
            episode = f"episode-{index}"
            directory = root_path / split
            directory.mkdir(exist_ok=True)
            payload = f"replay-{index}".encode()
            (directory / f"{episode}.json").write_bytes(payload)
            rows.append({
                "episode_id": episode, "split": split, "source_date": "21-08",
                "source_path": f"audit/{episode}.json",
                "source_sha256": hashlib.sha256(payload).hexdigest(), "route_family": "route-a",
            })
        return rows

    def write_manifest(self, root: str, rows: list[dict[str, str]]) -> None:
        root_path = Path(root)
        columns = ["episode_id", "split", "source_date", "source_path", "source_sha256", "route_family"]
        with (root_path / "manifest.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (root_path / "split_summary.json").write_text(json.dumps({
            "schema_version": 1, "selected_win_count": 100, "unique_episode_ids": 100,
            "unique_source_hashes": 100, "split_counts": {"train": 70, "val": 15, "test": 15},
            "stratify_fields": ["source_date", "opponent", "ryo_seat", "margin_quartile", "shop_profile", "route_family"],
        }), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
