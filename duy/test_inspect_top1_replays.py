import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


INSPECTOR_PATH = (
    Path(__file__).parent
    / "another_work"
    / "02_inspect_top1"
    / "inspect_replays.py"
)
TOP100_DIR = (
    Path(__file__).parent.parent
    / "duy_explore"
    / "kaggriculture-episodes-2026-08-15"
    / "top-100"
)


def load_inspector():
    spec = importlib.util.spec_from_file_location(
        "inspect_top1_replays", INSPECTOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_replay(names=None, module_version="1.32.7"):
    names = names or ["top", "opponent"]
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    observation = {
        "day": 0,
        "hour": 0,
        "farms": [
            {"farmer": [4, 4], "hands": [], "tiles": []},
            {"farmer": [4, 4], "hands": [], "tiles": []},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "town": {"unlocked_shops": []},
    }
    player = {
        "action": action,
        "observation": observation,
        "reward": 0,
        "status": "ACTIVE",
    }
    return {
        "module_version": module_version,
        "configuration": {
            "startingMoney": 3000,
            "episodeSteps": 720,
            "turnsPerDay": 24,
            "townCenterSellInterval": 24,
        },
        "info": {"TeamNames": names, "seed": 7},
        "rewards": [0, 0],
        "steps": [copy.deepcopy([player, player]) for _ in range(720)],
    }


def pass_action():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def synthetic_record(actions, decision_step=5):
    state = {
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    return {
        "actions": copy.deepcopy(actions),
        "comparison_actions": [
            [copy.deepcopy(action["farmer"]),
             *copy.deepcopy(action.get("hands", []))]
            for action in actions
        ],
        "comparison_timeline": [
            {
                "field": [copy.deepcopy(action["farmer"]),
                          *copy.deepcopy(action.get("hands", []))],
                "market": [
                    list(order[:2]) if order[0] == "SELL" else list(order)
                    for order in action.get("market", [])
                ],
            }
            for action in actions
        ],
        "canonical_states": {str(decision_step): state},
    }


class ReplayValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = load_inspector()

    def test_loads_a_valid_720_step_replay(self):
        replay = fake_replay()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(replay))

            loaded = self.inspector.load_replay(path)

        self.assertEqual(loaded["info"]["seed"], 7)
        self.assertEqual(len(loaded["steps"]), 720)

    def test_extracts_required_root_module_version(self):
        self.assertEqual(
            self.inspector.replay_module_version(fake_replay()), "1.32.7"
        )

    def test_mixed_versions_are_partitioned_not_aborted(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, version in (("new.json", "1.32.7"), ("old.json", "1.32.6")):
                path = Path(directory) / name
                path.write_text(json.dumps(fake_replay(module_version=version)))
                paths.append(path)
            accepted, rejected = self.inspector.load_compatible_replays(paths)

        self.assertEqual([path.name for path, _ in accepted], ["new.json"])
        self.assertEqual(
            rejected,
            [{
                "source": "old.json",
                "module_version": "1.32.6",
                "reason": "module_version_mismatch",
            }],
        )

    def test_top100_version_and_team_counts_are_stable(self):
        paths = sorted(TOP100_DIR.glob("*.json"))
        accepted, rejected = self.inspector.load_compatible_replays(paths)
        records = self.inspector.collect_team_records(accepted, "カワシギ")

        self.assertEqual(len(paths), 100)
        self.assertEqual(len(accepted), 90)
        self.assertEqual(len(rejected), 10)
        self.assertEqual(
            {row["module_version"] for row in rejected}, {"1.32.6"}
        )
        self.assertEqual(len(records), 69)

    def test_rejects_wrong_competition_configuration(self):
        replay = fake_replay()
        replay["configuration"]["startingMoney"] = 5000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(replay))

            with self.assertRaisesRegex(
                self.inspector.ReplayError, "startingMoney"
            ):
                self.inspector.load_replay(path)

    def test_rejects_truncated_episode(self):
        replay = fake_replay()
        replay["steps"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(replay))

            with self.assertRaisesRegex(
                self.inspector.ReplayError, "720 states"
            ):
                self.inspector.load_replay(path)

    def test_shifted_actions_discards_initial_placeholder(self):
        replay = fake_replay()
        replay["steps"][1][0]["action"] = {
            "farmer": ["BUILD_PASTURE"],
            "hands": [],
            "market": [["HIRE"]],
        }

        actions = self.inspector.shifted_actions(replay, 0)

        self.assertEqual(actions[0]["farmer"], ["BUILD_PASTURE"])
        self.assertEqual(actions[0]["market"], [["HIRE"]])
        self.assertEqual(len(actions), 720)
        self.assertEqual(
            actions[-1], {"farmer": ["PASS"], "hands": [], "market": []}
        )

    def test_self_play_requires_explicit_seat(self):
        replay = fake_replay(names=["top", "top"])

        with self.assertRaisesRegex(self.inspector.ReplayError, "self-play"):
            self.inspector.find_seat(replay, "top")

        self.assertEqual(
            self.inspector.find_seat(replay, "top", self_seat=1), 1
        )

    def test_rejects_missing_team_name(self):
        with self.assertRaisesRegex(self.inspector.ReplayError, "not found"):
            self.inspector.find_seat(fake_replay(), "unknown")


class StrategyExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = load_inspector()

    def test_branch_key_uses_route_defining_purchases(self):
        evidence = {
            "purchases": {
                "COW": 10,
                "SHEEP": 4,
                "STRAWBERRY_SEED": 34,
                "MELON_SEED": 20,
            }
        }

        self.assertEqual(
            self.inspector.branch_key(evidence),
            "c10-s4-straw34-melon20",
        )

    def test_medoid_minimizes_disagreement_and_breaks_ties_by_name(self):
        records = [
            {
                "source": "b.json",
                "comparison_actions": [["NORTH"], ["PASS"]],
            },
            {
                "source": "a.json",
                "comparison_actions": [["NORTH"], ["PASS"]],
            },
            {
                "source": "c.json",
                "comparison_actions": [["SOUTH"], ["PASS"]],
            },
        ]

        self.assertEqual(
            self.inspector.select_medoid(records)["source"], "a.json"
        )

    def test_actor_disagreement_counts_missing_and_changed_orders(self):
        left = {
            "comparison_actions": [
                [["NORTH"], ["PASS"]],
                [["WATER"]],
            ]
        }
        right = {
            "comparison_actions": [
                [["SOUTH"]],
                [["WATER"]],
            ]
        }

        self.assertEqual(self.inspector.actor_disagreement(left, right), 2)

    def test_market_comparison_normalizes_only_sell_quantity(self):
        normalize = self.inspector.normalize_market_order

        self.assertEqual(normalize(["SELL", "MILK", 99]), ["SELL", "MILK"])
        self.assertEqual(
            normalize(["BUY_ANIMAL", "COW", 2]),
            ["BUY_ANIMAL", "COW", 2],
        )

    def test_comparison_timeline_keeps_field_and_normalized_market_orders(self):
        record = {
            "comparison_actions": [[["NORTH"]]],
            "actions": [{
                "farmer": ["NORTH"],
                "hands": [],
                "market": [["SELL", "MILK", 99], ["BUY_ANIMAL", "COW", 2]],
            }],
        }

        self.assertEqual(
            self.inspector.comparison_timeline(record),
            [{
                "field": [["NORTH"]],
                "market": [["SELL", "MILK"], ["BUY_ANIMAL", "COW", 2]],
            }],
        )

    def test_opening_fingerprint_is_stable_and_prefix_bounded(self):
        record = {
            "comparison_timeline": [
                {"field": [["PASS"]], "market": []}
                for _ in range(73)
            ]
        }
        first = self.inspector.opening_fingerprint(record, stop=72)
        changed = copy.deepcopy(record)
        changed["comparison_timeline"][72]["field"] = [["NORTH"]]

        self.assertEqual(first, self.inspector.opening_fingerprint(changed, stop=72))
        self.assertEqual(len(first), 64)

    def test_route_distance_counts_field_and_market_disagreements(self):
        left = {
            "comparison_actions": [["NORTH"], ["PASS"]],
            "comparison_timeline": [
                {"field": [["NORTH"]], "market": [["SELL", "MILK"]]},
                {"field": [["PASS"]], "market": []},
            ],
        }
        right = {
            "comparison_actions": [["SOUTH"], ["PASS"]],
            "comparison_timeline": [
                {"field": [["SOUTH"]], "market": [["SELL", "MILK"]]},
                {"field": [["PASS"]], "market": [["HIRE"]]},
            ],
        }

        self.assertEqual(self.inspector.route_distance(left, right), 2)

    def test_opening_family_requires_all_supported_branches_then_fingerprint(self):
        supported = (
            "c10-s4-straw42-melon12",
            "c8-s6-straw42-melon12",
            "c6-s8-straw42-melon12",
            "c6-s12-straw42-melon12",
        )

        def record(source, branch, marker):
            return {
                "source": source,
                "branch": branch,
                "comparison_timeline": [
                    {"field": [[marker]], "market": []}
                ],
            }

        records = [
            *[record(f"incomplete-{index}.json", supported[index % 3], "FAMILY_A")
              for index in range(5)],
            *[record(f"eligible-b-{index}.json", branch, "FAMILY_B")
              for index, branch in enumerate(supported)],
            *[record(f"eligible-c-{index}.json", branch, "FAMILY_C")
              for index, branch in enumerate(supported)],
        ]

        fingerprint, family = self.inspector.select_opening_family(records)

        self.assertEqual(
            fingerprint,
            "1f202de437712d48c4c8cb17669ee2d0a9db3266b65c0e19beeb3fd52db81d49",
        )
        self.assertEqual(
            [record["source"] for record in family],
            [
                "eligible-b-0.json",
                "eligible-b-1.json",
                "eligible-b-2.json",
                "eligible-b-3.json",
            ],
        )

    def test_top100_family_and_medoids_are_stable(self):
        accepted, _ = self.inspector.load_compatible_replays(
            sorted(TOP100_DIR.glob("*.json"))
        )
        records = self.inspector.collect_team_records(accepted, "カワシギ")
        fingerprint, family = self.inspector.select_opening_family(records)
        medoids = self.inspector.select_branch_medoids(family)
        counts = Counter(record["branch"] for record in family)

        self.assertEqual(fingerprint[:12], "c860b6d9f00f")
        self.assertEqual(len(family), 35)
        self.assertEqual(
            {record["branch"] for record in family},
            set(self.inspector.SUPPORTED_BRANCHES),
        )
        self.assertEqual(
            counts,
            Counter({
                "c10-s4-straw42-melon12": 23,
                "c8-s6-straw42-melon12": 4,
                "c6-s8-straw42-melon12": 4,
                "c6-s12-straw42-melon12": 4,
            }),
        )
        self.assertEqual(
            {branch: record["source"] for branch, record in medoids.items()},
            {
                "c10-s4-straw42-melon12": "93232089.json",
                "c8-s6-straw42-melon12": "93316226.json",
                "c6-s8-straw42-melon12": "93339617.json",
                "c6-s12-straw42-melon12": "93399364.json",
            },
        )

class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inspector = load_inspector()

    def test_handoff_rejects_a_changed_predecision_action(self):
        base = synthetic_record(actions=[pass_action() for _ in range(6)])
        branch = copy.deepcopy(base)
        branch["actions"][4]["farmer"] = ["NORTH"]
        report = self.inspector.build_handoff_report(base, branch, 5)
        self.assertFalse(report["safe"])
        self.assertEqual(report["first_field_difference"], 4)

    def test_handoff_accepts_a_postdecision_difference(self):
        base = synthetic_record(actions=[pass_action() for _ in range(6)])
        branch = copy.deepcopy(base)
        branch["actions"][5]["farmer"] = ["NORTH"]
        report = self.inspector.build_handoff_report(base, branch, 5)
        self.assertTrue(report["safe"])

    def test_handoff_rejects_a_decision_state_difference(self):
        base = synthetic_record(actions=[pass_action() for _ in range(6)])
        branch = copy.deepcopy(base)
        branch["canonical_states"]["5"]["farmer"] = [5, 4]

        report = self.inspector.build_handoff_report(base, branch, 5)

        self.assertFalse(report["safe"])
        self.assertFalse(report["farm_state_equal"])

    def test_handoff_rejects_missing_decision_state(self):
        base = synthetic_record(actions=[pass_action() for _ in range(6)])
        branch = copy.deepcopy(base)
        del base["canonical_states"]["5"]
        del branch["canonical_states"]["5"]

        report = self.inspector.build_handoff_report(base, branch, 5)

        self.assertFalse(report["safe"])
        self.assertFalse(report["farm_state_equal"])

    def test_required_handoff_generation_rejects_an_unsafe_branch(self):
        actions = [pass_action() for _ in range(217)]
        base = synthetic_record(actions, decision_step=216)
        base["canonical_states"]["144"] = copy.deepcopy(
            base["canonical_states"]["216"]
        )
        medoids = {
            branch: copy.deepcopy(base)
            for branch in self.inspector.SUPPORTED_BRANCHES
        }
        medoids["c6-s12-straw42-melon12"]["actions"][4]["farmer"] = [
            "NORTH"
        ]

        with self.assertRaisesRegex(self.inspector.ReplayError, "unsafe"):
            self.inspector.build_required_handoff_reports(medoids)

    def test_selected_branch_handoffs_are_safe(self):
        analysis, routes = self.inspector.build_outputs(
            sorted(TOP100_DIR.glob("*.json")),
            team_name="カワシギ",
            self_seat=None,
        )
        reports = routes["handoffs"]
        self.assertTrue(reports["c6-s12-straw42-melon12"]["safe"])
        self.assertEqual(
            reports["c6-s12-straw42-melon12"]["decision_step"], 144
        )
        for branch in (
            "c6-s8-straw42-melon12",
            "c8-s6-straw42-melon12",
        ):
            self.assertTrue(reports[branch]["safe"])
            self.assertEqual(reports[branch]["decision_step"], 216)

        self.assertEqual(analysis["schema_version"], 2)
        self.assertEqual(analysis["required_module_version"], "1.32.7")
        self.assertEqual(len(analysis["accepted_sources"]), 90)
        self.assertEqual(len(analysis["rejected_records"]), 10)
        self.assertEqual(analysis["target_record_count"], 69)
        self.assertEqual(analysis["family_member_count"], 35)
        self.assertTrue(
            all(
                "canonical_states" not in record
                for record in analysis["records"]
            )
        )


if __name__ == "__main__":
    unittest.main()
