import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
INSPECTOR_PATH = PROJECT_DIR / "replay_inspector.py"
TOP100_DIR = (
    Path(__file__).resolve().parents[4]
    / "duy_explore"
    / "kaggriculture-episodes-2026-08-15"
    / "top-100"
)


def load_inspector():
    spec = importlib.util.spec_from_file_location(
        "boatlee_replay_inspector", INSPECTOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action(farmer=None, hands=None, market=None):
    return {
        "farmer": list(farmer or ["PASS"]),
        "hands": copy.deepcopy(hands or []),
        "market": copy.deepcopy(market or []),
    }


def observation(player, step=0):
    farm = {
        "money": 3000,
        "farmer": [4, 4],
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW"],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    return {
        "player": player,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def fake_replay(module_version="1.32.7"):
    steps = []
    for step in range(720):
        states = []
        for seat in (0, 1):
            states.append(
                {
                    "action": action(),
                    "observation": observation(seat, step),
                    "reward": 0,
                    "status": "ACTIVE" if step < 719 else "DONE",
                }
            )
        steps.append(states)
    return {
        "module_version": module_version,
        "configuration": {"episodeSteps": 720, "turnsPerDay": 24},
        "info": {"TeamNames": ["alpha", "beta"], "seed": 7},
        "rewards": [120, 100],
        "steps": steps,
    }


class ReplayValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = load_inspector()

    def test_loads_valid_replay_and_rejects_wrong_version(self):
        replay = fake_replay()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(replay))
            loaded = self.inspector.load_replay(path)
        self.assertEqual(loaded["info"]["seed"], 7)

        with self.assertRaisesRegex(
            self.inspector.ReplayError, "module_version"
        ):
            self.inspector.validate_replay(
                fake_replay(module_version="1.32.6"), "old.json"
            )

    def test_rejects_truncated_episode_and_missing_action(self):
        short = fake_replay()
        short["steps"].pop()
        with self.assertRaisesRegex(self.inspector.ReplayError, "720"):
            self.inspector.validate_replay(short, "short.json")

        missing = fake_replay()
        del missing["steps"][22][1]["action"]
        with self.assertRaisesRegex(self.inspector.ReplayError, "action"):
            self.inspector.validate_replay(missing, "missing.json")

    def test_shifted_actions_aligns_state_one_action_to_observation_zero(self):
        replay = fake_replay()
        replay["steps"][1][0]["action"] = action(
            ["NORTH"], market=[["HIRE"]]
        )

        aligned = self.inspector.shifted_actions(replay, 0)

        self.assertEqual(aligned[0]["farmer"], ["NORTH"])
        self.assertEqual(aligned[0]["market"], [["HIRE"]])
        self.assertEqual(len(aligned), 720)
        self.assertEqual(
            aligned[-1], {"farmer": ["PASS"], "hands": [], "market": []}
        )

    def test_extracts_both_seats_and_strategy_features(self):
        replay = fake_replay()
        replay["steps"][1][0]["action"] = action(
            ["HARVEST"],
            market=[
                ["HIRE"],
                ["BUY_ANIMAL", "COW", 1],
                ["SELL", "MILK", 3],
            ],
        )
        replay["steps"][1][1]["action"] = action(
            ["BUILD_COOP"], market=[["BUY_ANIMAL", "GOOSE", 2]]
        )

        rows = [
            self.inspector.extract_seat_record(
                replay, "fixture.json", seat
            )
            for seat in (0, 1)
        ]

        self.assertEqual([row["seat"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["team"], "alpha")
        self.assertEqual(rows[0]["opponent"], "beta")
        self.assertTrue(rows[0]["won"])
        self.assertEqual(rows[0]["final_margin"], 20)
        self.assertEqual(
            rows[0]["features"]["purchase_totals"]["COW"], 1
        )
        self.assertEqual(rows[0]["features"]["hire_count"], 1)
        self.assertEqual(
            rows[0]["features"]["operation_counts"]["HARVEST"], 1
        )
        self.assertEqual(rows[0]["features"]["sale_totals"]["MILK"], 3)
        self.assertEqual(
            rows[1]["features"]["purchase_totals"]["GOOSE"], 2
        )
        self.assertEqual(len(rows[0]["actions"]), 720)
        self.assertEqual(len(rows[0]["canonical_states"]), 720)

    def test_canonical_state_excludes_same_decision_purchase_attempt(self):
        replay = fake_replay()
        replay["steps"][1][0]["action"] = action(
            market=[["BUY_ANIMAL", "COW", 1]]
        )

        row = self.inspector.extract_seat_record(replay, "fixture.json", 0)

        self.assertEqual(row["canonical_states"]["0"]["purchases"], {})
        self.assertEqual(
            row["canonical_states"]["1"]["purchases"], {"COW": 1}
        )

    def test_seed_and_product_purchases_have_distinct_feature_keys(self):
        replay = fake_replay()
        replay["steps"][1][0]["action"] = action(
            market=[
                ["BUY_SEED", "WHEAT", 7],
                ["BUY_PRODUCT", "WHEAT", 5],
            ]
        )

        row = self.inspector.extract_seat_record(replay, "fixture.json", 0)

        self.assertEqual(
            row["features"]["purchase_totals"],
            {"WHEAT_PRODUCT": 5, "WHEAT_SEED": 7},
        )

    def test_stable_json_sorts_keys_and_ends_with_newline(self):
        rendered = self.inspector.stable_json({"z": 1, "a": {"b": 2}})
        self.assertEqual(rendered, '{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n')


class DeterministicSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = load_inspector()

    def test_split_is_exact_stable_and_source_order_independent(self):
        rows = []
        for index in range(90):
            rows.append(
                {
                    "source": f"{index:03d}.json",
                    "winner_team": f"team-{index % 7}",
                    "winner_seat": index % 2,
                    "opponent_pair": f"pair-{index % 11}",
                    "shop_signature": f"shops-{index % 5}",
                    "core_family": f"core-{index % 4}",
                }
            )

        forward = self.inspector.stratified_split(rows)
        reverse = self.inspector.stratified_split(list(reversed(rows)))

        self.assertEqual(forward, reverse)
        self.assertEqual(
            Counter(forward.values()),
            {"discovery": 60, "holdout": 30},
        )
        self.assertEqual(set(forward), {row["source"] for row in rows})

    def test_split_rejects_duplicate_sources_or_wrong_count(self):
        rows = [
            {
                "source": f"{index:03d}.json",
                "winner_team": "a",
                "winner_seat": 0,
                "opponent_pair": "a-v-b",
                "shop_signature": "s",
                "core_family": "c",
            }
            for index in range(90)
        ]
        rows[-1]["source"] = rows[0]["source"]
        with self.assertRaisesRegex(self.inspector.ReplayError, "unique"):
            self.inspector.stratified_split(rows)
        with self.assertRaisesRegex(self.inspector.ReplayError, "90"):
            self.inspector.stratified_split(rows[:-1])

    def test_real_corpus_catalog_has_90_files_and_180_seat_records(self):
        catalog = self.inspector.build_catalog(TOP100_DIR)

        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(len(catalog["replays"]), 90)
        self.assertEqual(len(catalog["seat_records"]), 180)
        self.assertEqual(
            Counter(row["split"] for row in catalog["replays"]),
            {"discovery": 60, "holdout": 30},
        )
        self.assertEqual(
            {row["module_version"] for row in catalog["replays"]},
            {"1.32.7"},
        )
        self.assertTrue(
            all(record["trace_payload"] for record in catalog["seat_records"])
        )


if __name__ == "__main__":
    unittest.main()
