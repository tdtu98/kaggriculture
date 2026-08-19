import ast
import base64
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock


DEFAULT_AGENT_PATH = (
    Path(__file__).parent / "another_work" / "02_inspect_top1" / "main.py"
)
REPLAY_ROUTE_CANDIDATE_PATH = (
    Path(__file__).parent
    / "another_work"
    / "02_inspect_top1"
    / "candidate_main.py"
)
AGENT_PATH = Path(
    os.environ.get("INSPECT_TOP1_AGENT_PATH", REPLAY_ROUTE_CANDIDATE_PATH)
)
BUILDER_PATH = (
    Path(__file__).parent
    / "another_work"
    / "02_inspect_top1"
    / "build_agent.py"
)
EVALUATOR_PATH = (
    Path(__file__).parent
    / "another_work"
    / "02_inspect_top1"
    / "evaluate_variants.py"
)


def load_agent_module(path=AGENT_PATH, name="inspect_top1_agent"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_builder_module(name="inspect_top1_builder"):
    spec = importlib.util.spec_from_file_location(name, BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluator_module(name="inspect_top1_evaluator"):
    spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_summary():
    return {
        "games": 200,
        "wins": 112,
        "losses": 88,
        "ties": 0,
        "by_agent_a_seat": {
            "0": {"margin": {"mean": 100.0}},
            "1": {"margin": {"mean": 80.0}},
        },
        "paired_seeds": {
            "margin": {"mean": 90.0, "median": 75.0},
            "bootstrap_mean_95ci": {"lower": 10.0, "upper": 170.0},
        },
    }


def profile_replay(team_name="target", seat=1):
    names = ["other", "other"]
    names[seat] = team_name
    steps = []
    for step in range(720):
        states = []
        for player in range(2):
            states.append(
                {
                    "action": {
                        "farmer": ["PASS"],
                        "hands": [],
                        "market": [],
                    },
                    "observation": {"player": player, "step": step},
                    "reward": 0,
                    "status": "DONE" if step == 719 else "ACTIVE",
                }
            )
        steps.append(states)
    return {
        "module_version": "1.32.7",
        "configuration": {
            "startingMoney": 3000,
            "episodeSteps": 720,
            "turnsPerDay": 24,
            "townCenterSellInterval": 24,
        },
        "info": {"TeamNames": names},
        "rewards": [0, 0],
        "steps": steps,
    }


class EvaluationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator_module()

    def test_renderer_changes_only_named_boolean_constants(self):
        source = "_ENABLE_FIELD_GUARDS = True\n_ENABLE_SALE_CAP = False\n"

        rendered = self.evaluator.render_variant(
            source,
            {"_ENABLE_FIELD_GUARDS": False, "_ENABLE_SALE_CAP": True},
        )

        self.assertEqual(
            rendered,
            "_ENABLE_FIELD_GUARDS = False\n_ENABLE_SALE_CAP = True\n",
        )

    def test_renderer_rejects_unknown_missing_and_duplicate_constants(self):
        with self.assertRaises(ValueError):
            self.evaluator.render_variant("", {"_ENABLE_UNKNOWN": True})
        with self.assertRaises(ValueError):
            self.evaluator.render_variant(
                "_ENABLE_FIELD_GUARDS = True # not a full line\n",
                {"_ENABLE_FIELD_GUARDS": False},
            )
        with self.assertRaises(ValueError):
            self.evaluator.render_variant(
                "_ENABLE_FIELD_GUARDS = True\n"
                "_ENABLE_FIELD_GUARDS = False\n",
                {"_ENABLE_FIELD_GUARDS": False},
            )

    def test_frozen_variant_preserves_candidate_constants_unchanged(self):
        source = (
            "_ENABLE_FIELD_GUARDS = True\n"
            "_ENABLE_PURCHASE_RECOVERY = True\n"
            "_ENABLE_SALE_CAP = False\n"
            "_ENABLE_FRONT_RUN = True\n"
            "candidate body\n"
        )

        flags, rendered = self.evaluator._render_named_variant(
            "frozen", source
        )

        self.assertEqual(rendered, source)
        self.assertEqual(
            flags,
            {
                "_ENABLE_FIELD_GUARDS": True,
                "_ENABLE_PURCHASE_RECOVERY": True,
                "_ENABLE_SALE_CAP": False,
                "_ENABLE_FRONT_RUN": True,
            },
        )

    def test_promotion_gate_reports_every_failed_threshold_in_order(self):
        summary = passing_summary()
        summary["games"] = 199
        summary["paired_seeds"]["margin"] = {
            "mean": 0.0,
            "median": 0.0,
        }
        summary["wins"] = 109
        summary["by_agent_a_seat"]["0"]["margin"]["mean"] = 0.0
        summary["by_agent_a_seat"]["1"]["margin"]["mean"] = 0.0
        summary["paired_seeds"]["bootstrap_mean_95ci"]["lower"] = 0.0

        self.assertEqual(
            self.evaluator.promotion_failures(summary),
            [
                "unexpected_game_count",
                "paired_mean_not_positive",
                "paired_median_not_positive",
                "win_rate_not_above_55_percent",
                "seat_zero_mean_not_positive",
                "seat_one_mean_not_positive",
                "bootstrap_lower_not_positive",
            ],
        )

    def test_promotion_gate_accepts_passing_summary(self):
        self.assertEqual(self.evaluator.promotion_failures(passing_summary()), [])

    def test_promotion_gate_rejects_only_unexpected_game_count(self):
        summary = passing_summary()
        summary["games"] = 199
        summary["wins"] = 112

        self.assertEqual(
            self.evaluator.promotion_failures(summary),
            ["unexpected_game_count"],
        )

    def _profile_fixture(self, directory):
        directory = Path(directory)
        candidate = directory / "candidate.py"
        candidate.write_text(
            "_expected_step = 0\n"
            "def agent(obs):\n"
            "    global _expected_step\n"
            "    assert obs['player'] == 1\n"
            "    assert obs['step'] == _expected_step\n"
            "    _expected_step += 1\n"
            "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
        )
        replay = directory / "replay.json"
        replay.write_text(json.dumps(profile_replay()))
        return candidate, replay

    @staticmethod
    def _clock_values(import_ns, call_ns):
        values = [0, import_ns]
        current = import_ns
        for _ in range(720):
            values.extend((current, current + call_ns))
            current += call_ns
        return values

    def test_profile_candidate_times_all_replay_observations_in_step_order(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, replay = self._profile_fixture(directory)
            clock = self._clock_values(2_000_000, 100_000)
            with mock.patch.object(
                self.evaluator.time,
                "perf_counter_ns",
                side_effect=clock,
            ):
                profile = self.evaluator.profile_candidate(
                    candidate, replay, "target"
                )

        self.assertEqual(
            set(profile),
            {
                "import_ms",
                "calls",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "maximum_ms",
            },
        )
        self.assertEqual(profile["calls"], 720)
        self.assertAlmostEqual(profile["import_ms"], 2.0)
        self.assertAlmostEqual(profile["mean_ms"], 0.1)
        self.assertAlmostEqual(profile["p50_ms"], 0.1)
        self.assertAlmostEqual(profile["p95_ms"], 0.1)
        self.assertAlmostEqual(profile["maximum_ms"], 0.1)

    def test_profile_cli_writes_evidence_before_failing_latency_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, replay = self._profile_fixture(directory)
            output = Path(directory) / "latency.json"
            clock = self._clock_values(2_000_000, 2_000_000)
            with mock.patch.object(
                self.evaluator.time,
                "perf_counter_ns",
                side_effect=clock,
            ):
                result = self.evaluator.main(
                    [
                        "--profile-candidate",
                        str(candidate),
                        "--profile-replay",
                        str(replay),
                        "--profile-team-name",
                        "target",
                        "--profile-output",
                        str(output),
                    ]
                )
            profile = json.loads(output.read_text())

        self.assertEqual(result, 1)
        self.assertEqual(profile["calls"], 720)
        self.assertEqual(profile["mean_ms"], 2.0)
        self.assertEqual(profile["p95_ms"], 2.0)

    def test_benchmark_cli_refuses_an_existing_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "already-present"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep")

            result = self.evaluator.main(
                [
                    "--candidate",
                    str(Path(directory) / "missing-candidate.py"),
                    "--baseline",
                    str(Path(directory) / "missing-baseline.py"),
                    "--seed-count",
                    "1",
                    "--variant",
                    "frozen",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(result, 1)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_benchmark_summary_artifacts_are_deterministic_across_runs(self):
        results = [
            {
                "seed": 7,
                "agent_a_seat": 0,
                "agent_b_seat": 1,
                "seat_0_agent": "route_only",
                "seat_1_agent": "baseline",
                "seat_0_money": 4000.0,
                "seat_1_money": 3000.0,
                "seat_0_reward": 4000.0,
                "seat_1_reward": 3000.0,
                "seat_0_status": "DONE",
                "seat_1_status": "DONE",
                "agent_a_money": 4000.0,
                "agent_b_money": 3000.0,
                "margin": 1000.0,
                "outcome": "win",
            },
            {
                "seed": 7,
                "agent_a_seat": 1,
                "agent_b_seat": 0,
                "seat_0_agent": "baseline",
                "seat_1_agent": "route_only",
                "seat_0_money": 3200.0,
                "seat_1_money": 4200.0,
                "seat_0_reward": 3200.0,
                "seat_1_reward": 4200.0,
                "seat_0_status": "DONE",
                "seat_1_status": "DONE",
                "agent_a_money": 4200.0,
                "agent_b_money": 3200.0,
                "margin": 1000.0,
                "outcome": "win",
            },
        ]
        candidate_source = (
            "_ENABLE_FIELD_GUARDS = True\n"
            "_ENABLE_PURCHASE_RECOVERY = False\n"
            "_ENABLE_SALE_CAP = False\n"
            "_ENABLE_FRONT_RUN = False\n"
            "def agent(obs):\n"
            "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
        )
        baseline_source = (
            "def agent(obs):\n"
            "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            baseline = root / "baseline.py"
            candidate.write_text(candidate_source)
            baseline.write_text(baseline_source)

            def evaluate(output, timestamp):
                fake_datetime = mock.Mock()
                fake_datetime.now.return_value.isoformat.return_value = timestamp
                with mock.patch.object(
                    self.evaluator.benchmark,
                    "run_suite",
                    return_value=results,
                ), mock.patch.object(
                    self.evaluator.benchmark,
                    "datetime",
                    fake_datetime,
                ):
                    return self.evaluator.main(
                        [
                            "--candidate",
                            str(candidate),
                            "--baseline",
                            str(baseline),
                            "--seed-start",
                            "7",
                            "--seed-count",
                            "1",
                            "--variant",
                            "route_only",
                            "--output-dir",
                            str(output),
                        ]
                    )

            output_a = root / "results-a"
            output_b = root / "results-b"
            self.assertEqual(evaluate(output_a, "2026-01-01T00:00:00Z"), 0)
            self.assertEqual(evaluate(output_b, "2027-02-02T00:00:00Z"), 0)
            summary_a = (output_a / "route_only" / "summary.json").read_bytes()
            summary_b = (output_b / "route_only" / "summary.json").read_bytes()

        self.assertEqual(summary_a, summary_b)
        metadata = json.loads(summary_a)["metadata"]
        rendered_hash = hashlib.sha256(
            self.evaluator.render_variant(
                candidate_source,
                dict.fromkeys(self.evaluator.FLAG_NAMES, False),
            ).encode()
        ).hexdigest()
        self.assertEqual(metadata["agent_a"]["sha256"], rendered_hash)
        self.assertEqual(
            metadata["generated_variant"]["flags"],
            dict.fromkeys(self.evaluator.FLAG_NAMES, False),
        )
        self.assertEqual(
            metadata["generated_variant"]["source_candidate"]["sha256"],
            hashlib.sha256(candidate_source.encode()).hexdigest(),
        )


def make_observation(
    *,
    step=0,
    player=0,
    hands=None,
    farmer=None,
    tile=None,
    shed=None,
    seeds=None,
    inventories=None,
    money=3000,
    shops=None,
    town_present=True,
):
    hands = copy.deepcopy(hands or [])
    farmer = list(farmer or [4, 4])
    own_tiles = [[None for _ in range(10)] for _ in range(10)]
    own_tiles[farmer[1]][farmer[0]] = copy.deepcopy(tile)
    other_tiles = [[None for _ in range(10)] for _ in range(10)]
    own = {
        "money": money,
        "farmer": farmer,
        "hands": hands,
        "hires_today": 0,
        "tiles": own_tiles,
        "unlocked_quadrants": ["NW"],
    }
    other = {
        "money": 3000,
        "farmer": [4, 4],
        "hands": [],
        "hires_today": 0,
        "tiles": other_tiles,
        "unlocked_quadrants": ["NW"],
    }
    farms = [own, other] if player == 0 else [other, own]
    if inventories is None:
        inventories = [{} for _ in range(1 + len(hands))]
    observation = {
        "player": player,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": farms,
        "private": {
            "shed": copy.deepcopy(shed or {}),
            "seeds": copy.deepcopy(seeds or {}),
            "inventories": copy.deepcopy(inventories),
        },
        "market": {"prices": {}, "inventory": {}},
    }
    if town_present:
        observation["town"] = {
            "unlocked_shops": copy.deepcopy([] if shops is None else shops)
        }
    return observation


class AgentShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agent_module()

    def setUp(self):
        self.module = load_agent_module(name=f"inspect_top1_agent_{id(self)}")

    def test_embedded_route_starts_with_top1_opening(self):
        action = self.module._route_action(0)

        self.assertEqual(action["farmer"], ["BUILD_PASTURE"])
        self.assertEqual(
            action["market"][:5], [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]
        )
        self.assertIn(["BUY_ANIMAL", "COW", 2], action["market"])
        self.assertIn(["BUY_SEED", "MELON", 12], action["market"])

    def test_aligns_hands_to_live_count(self):
        obs = make_observation(hands=[[4, 4], [3, 4]])
        action = {
            "farmer": ["PASS"],
            "hands": [["NORTH"]],
            "market": [],
        }

        aligned = self.module._align_hands(action, obs)

        self.assertEqual(aligned["hands"], [["NORTH"], ["PASS"]])

    def test_suppresses_replay_specific_dig_without_live_weed(self):
        obs = make_observation(step=10, farmer=[1, 1], tile=None)
        action = {"farmer": ["DIG"], "hands": [], "market": []}

        guarded = self.module._guard_field_actions(
            obs, action, step=10, weed_only={"farmer"}
        )

        self.assertEqual(guarded["farmer"], ["PASS"])

    def test_keeps_replay_specific_dig_on_live_weed(self):
        obs = make_observation(
            step=10, farmer=[1, 1], tile={"kind": "WEED"}
        )
        action = {"farmer": ["DIG"], "hands": [], "market": []}

        guarded = self.module._guard_field_actions(
            obs, action, step=10, weed_only={"farmer"}
        )

        self.assertEqual(guarded["farmer"], ["DIG"])

    def test_invalid_care_becomes_pass(self):
        obs = make_observation(step=10, tile=None)
        action = {"farmer": ["CARE"], "hands": [], "market": []}

        guarded = self.module._guard_field_actions(
            obs, action, step=10, weed_only=set()
        )

        self.assertEqual(guarded["farmer"], ["PASS"])

    def test_invalid_collect_fertilizer_becomes_pass(self):
        obs = make_observation(
            step=10,
            tile={
                "kind": "PASTURE",
                "animal": "COW",
                "fertilizer_available": False,
            },
        )
        action = {
            "farmer": ["COLLECT_FERTILIZER"],
            "hands": [],
            "market": [],
        }

        guarded = self.module._guard_field_actions(
            obs, action, step=10, weed_only=set()
        )

        self.assertEqual(guarded["farmer"], ["PASS"])

    def test_feed_requires_animal_and_carried_wheat(self):
        tile = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": False,
        }
        action = {"farmer": ["FEED"], "hands": [], "market": []}

        without_wheat = self.module._guard_field_actions(
            make_observation(tile=tile), action, step=10, weed_only=set()
        )
        with_wheat = self.module._guard_field_actions(
            make_observation(tile=tile, inventories=[{"WHEAT": 1}]),
            action,
            step=10,
            weed_only=set(),
        )

        self.assertEqual(without_wheat["farmer"], ["PASS"])
        self.assertEqual(with_wheat["farmer"], ["FEED"])

    def test_weed_repair_digs_then_retries_intended_action(self):
        state = {"last_step": 5, "active": {}}
        intended = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}

        dug = self.module._weed_repair_action(
            make_observation(step=5, tile={"kind": "WEED"}),
            intended,
            5,
            state,
        )
        retried = self.module._weed_repair_action(
            make_observation(step=6, tile=None),
            {"farmer": ["PASS"], "hands": [], "market": []},
            6,
            state,
        )

        self.assertEqual(dug["farmer"], ["DIG"])
        self.assertEqual(retried["farmer"], ["PLANT", "WHEAT"])

    def test_step_moving_backwards_resets_seat_state(self):
        state = self.module._reset_if_needed(make_observation(step=20), 20)
        state["synthetic"] = True

        reset = self.module._reset_if_needed(make_observation(step=5), 5)

        self.assertNotIn("synthetic", reset)
        self.assertEqual(reset["last_step"], 5)

    def test_exception_fallback_passes_every_live_hand(self):
        original = self.module._route_action
        self.module._route_action = lambda *_: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            result = self.module.agent(
                make_observation(hands=[[1, 1], [2, 2]])
            )
        finally:
            self.module._route_action = original

        self.assertEqual(
            result,
            {
                "farmer": ["PASS"],
                "hands": [["PASS"], ["PASS"]],
                "market": [],
            },
        )

    def test_public_agent_declares_only_observation_argument(self):
        self.assertEqual(str(inspect.signature(self.module.agent)), "(obs)")

    def test_imports_only_python_standard_library_modules(self):
        tree = ast.parse(AGENT_PATH.read_text())
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )

        self.assertLessEqual(imported, {"base64", "copy", "json", "zlib"})

    def test_module_import_does_not_need_route_json(self):
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / "main.py"
            isolated.write_bytes(AGENT_PATH.read_bytes())

            module = load_agent_module(isolated, "isolated_top1_agent")

        self.assertEqual(module._route_action(0)["farmer"], ["BUILD_PASTURE"])

    def test_live_controllers_have_independent_feature_flags(self):
        self.assertIs(self.module._ENABLE_FIELD_GUARDS, True)
        self.assertIs(self.module._ENABLE_PURCHASE_RECOVERY, False)
        self.assertIs(self.module._ENABLE_SALE_CAP, False)
        self.assertIs(self.module._ENABLE_FRONT_RUN, False)


@unittest.skipIf(
    AGENT_PATH.resolve() == DEFAULT_AGENT_PATH.resolve(),
    "shop-adaptive behavior belongs to the isolated candidate",
)
class ShopBranchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_agent_module(name=f"shop_branch_{id(self)}")

    def test_early_yarn_selects_six_cow_twelve_sheep_at_144(self):
        state = {"last_step": 143, "active": {}}
        obs = make_observation(step=144, shops=["YARN_STORE", "YARN_STORE"])
        self.assertEqual(
            self.module._select_branch(obs, state, 144),
            "c6-s12-straw42-melon12",
        )

    def test_late_yarn_selects_six_cow_eight_sheep(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}
        obs = make_observation(step=216, shops=["BAKERY", "YARN_STORE"])
        self.assertEqual(
            self.module._select_branch(obs, state, 216),
            "c6-s8-straw42-melon12",
        )

    def test_milk_or_strawberry_demand_selects_default(self):
        for shop in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"):
            state = {"last_step": 215, "active": {}, "early_yarn": False}
            self.assertEqual(
                self.module._select_branch(
                    make_observation(step=216, shops=[shop]), state, 216
                ),
                "c10-s4-straw42-melon12",
            )

    def test_low_demand_selects_eight_cow_six_sheep(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}
        self.assertEqual(
            self.module._select_branch(
                make_observation(step=216, shops=["BAKERY", "PET_CAFE"]),
                state,
                216,
            ),
            "c8-s6-straw42-melon12",
        )

    def test_missing_or_malformed_town_falls_back_to_default_and_freezes(self):
        malformed_observations = (
            make_observation(step=216, town_present=False),
            make_observation(step=216, shops="YARN_STORE"),
        )
        for obs in malformed_observations:
            state = {"last_step": 215, "active": {}, "early_yarn": False}
            first = self.module._select_branch(obs, state, 216)
            second = self.module._select_branch(
                make_observation(step=300, shops=["YARN_STORE"]), state, 300
            )
            self.assertEqual(first, "c10-s4-straw42-melon12")
            self.assertEqual(second, first)

    def test_frozen_low_demand_branch_ignores_later_shop_changes(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}

        first = self.module._select_branch(
            make_observation(step=216, shops=["BAKERY", "PET_CAFE"]),
            state,
            216,
        )
        second = self.module._select_branch(
            make_observation(step=300, shops=["YARN_STORE"]), state, 300
        )

        self.assertEqual(first, "c8-s6-straw42-melon12")
        self.assertEqual(second, first)

    def test_backwards_step_clears_frozen_branch_for_that_seat(self):
        other_seat = self.module._reset_if_needed(
            make_observation(step=216, player=0), 216
        )
        other_seat.update(
            {
                "branch": "c8-s6-straw42-melon12",
                "branch_frozen": True,
            }
        )
        state = self.module._reset_if_needed(
            make_observation(step=216, player=1), 216
        )
        state.update(
            {
                "branch": "c6-s8-straw42-melon12",
                "branch_frozen": True,
                "early_yarn": False,
            }
        )

        reset = self.module._reset_if_needed(
            make_observation(step=100, player=1), 100
        )

        self.assertEqual(reset["branch"], "c10-s4-straw42-melon12")
        self.assertNotIn("branch_frozen", reset)
        self.assertNotIn("early_yarn", reset)
        self.assertEqual(
            self.module._STATE[0]["branch"], "c8-s6-straw42-melon12"
        )
        self.assertIs(self.module._STATE[0]["branch_frozen"], True)

    def test_shop_adaptive_candidate_imports_without_route_json(self):
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / "main.py"
            isolated.write_bytes(AGENT_PATH.read_bytes())

            module = load_agent_module(isolated, "isolated_shop_adaptive")

        action = module._route_action(216, "c6-s8-straw42-melon12")
        self.assertIn(["BUY_ANIMAL", "SHEEP", 2], action["market"])
        self.assertEqual(
            set(module._BRANCHES),
            {
                "c10-s4-straw42-melon12",
                "c6-s12-straw42-melon12",
                "c6-s8-straw42-melon12",
                "c8-s6-straw42-melon12",
            },
        )

    def test_selected_branch_drives_actions_annotations_and_purchase_targets(self):
        default = "c10-s4-straw42-melon12"
        early_yarn = "c6-s12-straw42-melon12"

        self.assertIn(
            ["BUY_ANIMAL", "COW", 2],
            self.module._route_action(216, default)["market"],
        )
        self.assertIn(
            ["BUY_ANIMAL", "SHEEP", 2],
            self.module._route_action(216, early_yarn)["market"],
        )
        self.assertEqual(self.module._weed_annotations(455, default), set())
        self.assertEqual(
            self.module._weed_annotations(455, early_yarn), {"hand:3"}
        )
        self.assertEqual(
            (
                self.module._ROUTE_TARGETS[default][216]["COW"],
                self.module._ROUTE_TARGETS[default][216]["SHEEP"],
            ),
            (8, 4),
        )
        self.assertEqual(
            (
                self.module._ROUTE_TARGETS[early_yarn][216]["COW"],
                self.module._ROUTE_TARGETS[early_yarn][216]["SHEEP"],
            ),
            (6, 6),
        )

    def test_malformed_branch_keys_fall_back_to_default_route_data(self):
        default = "c10-s4-straw42-melon12"

        self.assertEqual(
            self.module._route_action(216, "not-a-branch"),
            self.module._route_action(216, default),
        )
        self.assertEqual(
            self.module._weed_annotations(451, "not-a-branch"),
            {"hand:6"},
        )

    def test_agent_routes_late_yarn_through_selected_branch(self):
        self.module.agent(make_observation(step=144, shops=["BAKERY"]))

        action = self.module.agent(
            make_observation(step=216, shops=["BAKERY", "YARN_STORE"])
        )

        self.assertEqual(
            self.module._STATE[0]["branch"], "c6-s8-straw42-melon12"
        )
        self.assertIn(["BUY_ANIMAL", "SHEEP", 2], action["market"])


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.module = load_builder_module(name=f"route_builder_{id(self)}")

    def test_encode_payload_is_deterministic_and_round_trips(self):
        payload = {"z": 1, "a": {"label": "mélon"}}

        encoded = self.module.encode_payload(payload)
        reordered = self.module.encode_payload(
            {"a": {"label": "mélon"}, "z": 1}
        )
        decoded = json.loads(
            zlib.decompress(base64.b85decode(encoded)).decode("utf-8")
        )

        self.assertEqual(encoded, reordered)
        self.assertEqual(decoded, payload)

    def test_replace_payload_replaces_only_generated_marker_block(self):
        source = (
            "before\n"
            "# BEGIN GENERATED ROUTES\n"
            "stale payload\n"
            "# END GENERATED ROUTES\n"
            "after\n"
        )

        replaced = self.module.replace_payload(source, "encoded")

        self.assertEqual(
            replaced,
            "before\n"
            "# BEGIN GENERATED ROUTES\n"
            "_PAYLOAD = json.loads(\n"
            "    zlib.decompress(base64.b85decode('encoded')).decode('utf-8')\n"
            ")\n"
            "# END GENERATED ROUTES\n"
            "after\n",
        )

    def test_replace_payload_refuses_source_without_markers(self):
        with self.assertRaises(ValueError):
            self.module.replace_payload("no generated block", "encoded")

    def test_cli_filters_routes_to_submission_payload_fields(self):
        routes_payload = {
            "schema_version": 2,
            "selector": {"default_branch": "alpha"},
            "ignored_root": "drop me",
            "branches": {
                "alpha": {
                    "source": "replay.json",
                    "actions": [{"farmer": ["PASS"]}],
                    "weed_only": {},
                    "ignored_branch": "drop me",
                }
            },
        }
        expected = {
            "schema_version": 2,
            "selector": {"default_branch": "alpha"},
            "branches": {
                "alpha": {
                    "source": "replay.json",
                    "actions": [{"farmer": ["PASS"]}],
                    "weed_only": {},
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            routes = root / "routes.json"
            template = root / "template.py"
            output = root / "output.py"
            routes.write_text(json.dumps(routes_payload), encoding="utf-8")
            template.write_text(
                "import base64\nimport json\nimport zlib\n"
                "# BEGIN GENERATED ROUTES\n"
                "stale payload\n"
                "# END GENERATED ROUTES\n",
                encoding="utf-8",
            )

            self.module.main(
                [
                    "--routes",
                    str(routes),
                    "--template",
                    str(template),
                    "--output",
                    str(output),
                ]
            )
            namespace = {}
            exec(output.read_text(encoding="utf-8"), namespace)

        self.assertEqual(namespace["_PAYLOAD"], expected)


class MarketControllerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_agent_module(name=f"inspect_top1_market_{id(self)}")

    def test_sale_cap_counts_shed_and_place_but_reserves_pickups(self):
        obs = make_observation(
            hands=[[3, 4]],
            shed={"MILK": 7},
            inventories=[{}, {"MILK": 3}],
        )
        action = {
            "farmer": ["PICKUP", "MILK", 2],
            "hands": [["PLACE", "MILK", 3]],
            "market": [["SELL", "MILK", 20]],
        }

        capped = self.module._cap_sales(action, obs)

        self.assertEqual(capped["market"], [["SELL", "MILK", 8]])

    def test_sale_cap_removes_zero_quantity_orders(self):
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WOOL", 4]],
        }

        capped = self.module._cap_sales(action, make_observation())

        self.assertEqual(capped["market"], [])

    def test_recovery_never_exceeds_cumulative_cow_target(self):
        state = {"purchased": {"COW": 9}, "pending": {"COW": 2}}
        targets = {"COW": 10}
        action = {"farmer": ["PASS"], "hands": [], "market": []}

        recovered = self.module._recover_purchases(
            action, make_observation(money=100000), state, targets
        )

        self.assertEqual(recovered["market"], [["BUY_ANIMAL", "COW", 1]])

    def test_recovery_respects_ten_order_capacity(self):
        state = {"purchased": {}, "pending": {"COW": 1}}
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WHEAT", 1] for _ in range(10)],
        }

        recovered = self.module._recover_purchases(
            action, make_observation(money=100000), state, {"COW": 1}
        )

        self.assertEqual(recovered["market"], action["market"])

    def test_recovery_does_not_add_an_unaffordable_purchase(self):
        state = {"purchased": {}, "pending": {"COW": 1}}
        action = {"farmer": ["PASS"], "hands": [], "market": []}

        recovered = self.module._recover_purchases(
            action, make_observation(money=399), state, {"COW": 1}
        )

        self.assertEqual(recovered["market"], [])

    def test_recovery_prioritizes_feed_wheat_before_animals(self):
        state = {
            "purchased": {},
            "pending": {"COW": 1, "WHEAT_PRODUCT": 3},
        }
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "CARROT", 1] for _ in range(9)],
        }

        recovered = self.module._recover_purchases(
            action,
            make_observation(money=100000),
            state,
            {"COW": 1, "WHEAT_PRODUCT": 3},
        )

        self.assertEqual(
            recovered["market"][-1], ["BUY_PRODUCT", "WHEAT", 3]
        )
        self.assertNotIn(["BUY_ANIMAL", "COW", 1], recovered["market"])

    def test_purchase_state_resets_with_episode(self):
        state = self.module._purchase_state(make_observation(step=20), 20)
        state["pending"]["COW"] = 1

        reset = self.module._purchase_state(make_observation(step=0), 0)

        self.assertEqual(reset["pending"], {})

    def test_front_run_reserves_scheduled_pickups(self):
        original = self.module._future_quantity
        self.module._future_quantity = (
            lambda _step, item: 5 if item == "MILK" else 0
        )
        try:
            action = {
                "farmer": ["PICKUP", "MILK", 2],
                "hands": [],
                "market": [],
            }
            state = {"due_step": -1, "due": {}}
            moved = self.module._front_run(
                action,
                make_observation(step=1, shed={"MILK": 7}),
                state,
                1,
            )
        finally:
            self.module._future_quantity = original

        self.assertEqual(moved["market"], [["SELL", "MILK", 5]])
        self.assertEqual(state, {"due_step": 2, "due": {"MILK": 5}})

    def test_repay_subtracts_exact_front_run_debt(self):
        state = {"due_step": 10, "due": {"MILK": 3}}
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "MILK", 5], ["SELL", "WOOL", 2]],
        }

        repaid = self.module._repay(action, state, 10)

        self.assertEqual(
            repaid["market"],
            [["SELL", "MILK", 2], ["SELL", "WOOL", 2]],
        )
        self.assertEqual(state, {"due_step": -1, "due": {}})


if __name__ == "__main__":
    unittest.main()
