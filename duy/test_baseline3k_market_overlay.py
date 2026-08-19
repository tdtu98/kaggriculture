import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


DUY_ROOT = Path(__file__).resolve().parent
BASELINE_PATH = DUY_ROOT / "another_work" / "01_baseline3k" / "main.py"
CANDIDATE_PATH = (
    DUY_ROOT / "another_work" / "02_inspect_top1" / "market_candidate.py"
)
EVALUATOR_PATH = (
    DUY_ROOT
    / "another_work"
    / "02_inspect_top1"
    / "evaluate_market_overlay.py"
)
PROMOTED_MAIN_PATH = DUY_ROOT / "another_work" / "02_inspect_top1" / "main.py"
PROMOTION_RECORD_PATH = (
    DUY_ROOT
    / "another_work"
    / "02_inspect_top1"
    / "market_overlay_promotion.json"
)
PROFILE_REPLAY_PATH = (
    DUY_ROOT.parent
    / "duy_explore"
    / "kaggriculture-episodes-2026-08-15"
    / "top-100"
    / "93232089.json"
)


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_summary(expected_games=100):
    return {
        "games": expected_games,
        "wins": 56,
        "losses": expected_games - 56,
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


def make_observation(
    *,
    step=101,
    seat=0,
    shed=None,
    prices=None,
    shops=None,
    hands=0,
):
    farms = [
        {
            "money": 3000,
            "tiles": [[None for _ in range(10)] for _ in range(10)],
            "farmer": [4, 4],
            "hands": [[4, 4] for _ in range(hands if index == seat else 0)],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }
        for index in range(2)
    ]
    return {
        "step": step,
        "player": seat,
        "farms": farms,
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": {"inventory": {}, "prices": dict(prices or {})},
        "town": {"unlocked_shops": list(shops or [])},
    }


def pass_action(*, farmer=None, hands=None, market=None):
    return {
        "farmer": list(farmer or ["PASS"]),
        "hands": [list(order) for order in (hands or [])],
        "market": [list(order) for order in (market or [])],
    }


class CandidateParityTests(unittest.TestCase):
    def test_candidate_embeds_exact_baseline_schedule(self):
        baseline = load_agent(BASELINE_PATH, "overlay_baseline")
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate")

        self.assertEqual(candidate._ACTIONS, baseline._ACTIONS)
        self.assertFalse(candidate._ENABLE_DEMAND_DEFERRAL)
        self.assertFalse(candidate._ENABLE_ADAPTIVE_FRONT_RUN)

    def test_disabled_candidate_matches_720_replay_observations(self):
        baseline = load_agent(BASELINE_PATH, "overlay_baseline_replay")
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_replay")
        replay = json.loads(PROFILE_REPLAY_PATH.read_text())
        seat = replay["info"]["TeamNames"].index("カワシギ")

        for states in replay["steps"]:
            observation = states[seat]["observation"]
            self.assertEqual(candidate.agent(observation), baseline.agent(observation))

    def test_malformed_observation_preserves_baseline_fallback_shape(self):
        baseline = load_agent(BASELINE_PATH, "overlay_baseline_bad_obs")
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_bad_obs")
        malformed = {"step": "not-an-integer", "farms": [{}, {}]}

        self.assertEqual(candidate.agent(malformed), baseline.agent(malformed))

    def test_candidate_call_has_no_runtime_file_dependency(self):
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_no_files")
        replay = json.loads(PROFILE_REPLAY_PATH.read_text())
        seat = replay["info"]["TeamNames"].index("カワシギ")
        observation = replay["steps"][0][seat]["observation"]

        with mock.patch(
            "builtins.open", side_effect=AssertionError("runtime file read")
        ):
            action = candidate.agent(observation)

        self.assertEqual(set(action), {"farmer", "hands", "market"})


class DemandDeferralTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent(CANDIDATE_PATH, f"demand_agent_{self._testMethodName}")

    def test_duplicate_yarn_stores_preserve_instance_demand(self):
        observation = make_observation(
            step=100, shops=["YARN_STORE", "YARN_STORE"]
        )

        self.assertEqual(self.agent._demand_units(observation, "WOOL", 100), 4)

    def test_depressed_sale_waits_until_after_next_demand_tick(self):
        observation = make_observation(
            step=101,
            shed={"MILK": 8},
            prices={"MILK": 40},
            shops=["SMOOTHIE_SHOP"],
        )
        state = self.agent._market_state(observation, 101)

        result = self.agent._apply_demand_deferral(
            pass_action(market=[["SELL", "MILK", 6]]),
            observation,
            state,
            101,
        )

        self.assertEqual(result["market"], [])
        self.assertEqual(
            state["deferred"]["MILK"],
            {"quantity": 6, "release_step": 105, "deadline": 105},
        )

    def test_due_sale_is_live_stock_capped_and_released_once(self):
        state = {
            "last_step": 104,
            "deferred": {
                "MILK": {"quantity": 6, "release_step": 105, "deadline": 105}
            },
            "front_debt": {},
        }
        observation = make_observation(
            step=105,
            shed={"MILK": 5},
            prices={"MILK": 80},
            shops=["SMOOTHIE_SHOP"],
        )

        first = self.agent._apply_demand_deferral(
            pass_action(), observation, state, 105
        )
        second = self.agent._apply_demand_deferral(
            pass_action(), observation, state, 105
        )

        self.assertEqual(first["market"], [["SELL", "MILK", 5]])
        self.assertEqual(second["market"], [])
        self.assertEqual(state["deferred"], {})

    def test_pickup_reserve_caps_deferred_quantity(self):
        observation = make_observation(
            step=101,
            shed={"WOOL": 8},
            prices={"WOOL": 20},
            shops=["YARN_STORE"],
        )
        state = self.agent._market_state(observation, 101)
        action = pass_action(
            farmer=["PICKUP", "WOOL", 3],
            market=[["SELL", "WOOL", 20]],
        )

        result = self.agent._apply_demand_deferral(
            action, observation, state, 101
        )

        self.assertEqual(result["farmer"], ["PICKUP", "WOOL", 3])
        self.assertEqual(result["market"], [])
        self.assertEqual(state["deferred"]["WOOL"]["quantity"], 5)

    def test_sale_is_not_deferred_without_a_nearby_demand_tick(self):
        observation = make_observation(
            step=101,
            shed={"MELON": 4},
            prices={"MELON": 1},
            shops=[],
        )
        state = self.agent._market_state(observation, 101)
        action = pass_action(market=[["SELL", "MELON", 4]])

        result = self.agent._apply_demand_deferral(
            action, observation, state, 101
        )

        self.assertEqual(result["market"], [["SELL", "MELON", 4]])
        self.assertEqual(state["deferred"], {})

    def test_sale_is_not_deferred_at_or_above_base_price(self):
        observation = make_observation(
            step=101,
            shed={"STRAWBERRY": 4},
            prices={"STRAWBERRY": 120},
            shops=["BRUNCH_SPOT"],
        )
        state = self.agent._market_state(observation, 101)
        action = pass_action(market=[["SELL", "STRAWBERRY", 4]])

        result = self.agent._apply_demand_deferral(
            action, observation, state, 101
        )

        self.assertEqual(result["market"], [["SELL", "STRAWBERRY", 4]])

    def test_high_shed_occupancy_and_final_window_force_sale(self):
        high_shed = make_observation(
            step=101,
            shed={"MILK": 8, "WHEAT": 83},
            prices={"MILK": 1},
            shops=["SMOOTHIE_SHOP"],
        )
        final_window = make_observation(
            step=715,
            shed={"MILK": 8},
            prices={"MILK": 1},
            shops=["SMOOTHIE_SHOP"],
        )
        sale = pass_action(market=[["SELL", "MILK", 6]])

        high_result = self.agent._apply_demand_deferral(
            sale, high_shed, self.agent._market_state(high_shed, 101), 101
        )
        final_result = self.agent._apply_demand_deferral(
            sale,
            final_window,
            self.agent._market_state(final_window, 715),
            715,
        )

        self.assertEqual(high_result["market"], [["SELL", "MILK", 6]])
        self.assertEqual(final_result["market"], [["SELL", "MILK", 6]])

    def test_final_window_flushes_pending_quantity_before_its_deadline(self):
        state = {
            "last_step": 714,
            "deferred": {
                "WOOL": {"quantity": 3, "release_step": 719, "deadline": 719}
            },
            "front_debt": {},
        }
        observation = make_observation(step=715, shed={"WOOL": 3})

        result = self.agent._apply_demand_deferral(
            pass_action(), observation, state, 715
        )

        self.assertEqual(result["market"], [["SELL", "WOOL", 3]])
        self.assertEqual(state["deferred"], {})

    def test_state_is_isolated_by_seat_and_resets_when_step_moves_back(self):
        seat_zero = make_observation(step=30, seat=0)
        seat_one = make_observation(step=30, seat=1)
        state_zero = self.agent._market_state(seat_zero, 30)
        state_one = self.agent._market_state(seat_one, 30)
        state_zero["deferred"]["MILK"] = {
            "quantity": 2,
            "release_step": 33,
            "deadline": 33,
        }

        self.assertEqual(state_one["deferred"], {})
        reset = self.agent._market_state(make_observation(step=0, seat=0), 0)
        self.assertEqual(reset["deferred"], {})

    def test_nonpremium_orders_and_field_actions_are_unchanged(self):
        observation = make_observation(
            step=101,
            shed={"WHEAT": 7},
            prices={"WHEAT": 1},
            shops=["BAKERY"],
            hands=1,
        )
        state = self.agent._market_state(observation, 101)
        action = pass_action(
            farmer=["NORTH"],
            hands=[["SOUTH"]],
            market=[["HIRE"], ["BUY_SEED", "WHEAT", 2], ["SELL", "WHEAT", 7]],
        )

        result = self.agent._apply_demand_deferral(
            action, observation, state, 101
        )

        self.assertEqual(result, action)

    def test_due_sale_waits_when_market_order_queue_is_full(self):
        state = {
            "last_step": 104,
            "deferred": {
                "WOOL": {"quantity": 3, "release_step": 105, "deadline": 105}
            },
            "front_debt": {},
        }
        observation = make_observation(step=105, shed={"WOOL": 3})
        full_queue = [["HIRE"] for _ in range(10)]

        result = self.agent._apply_demand_deferral(
            pass_action(market=full_queue), observation, state, 105
        )

        self.assertEqual(len(result["market"]), 10)
        self.assertEqual(state["deferred"]["WOOL"]["quantity"], 3)


class AdaptiveFrontRunTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent(
            CANDIDATE_PATH, f"front_agent_{self._testMethodName}"
        )

    def replace_route_sale(self, *, step, item, quantity):
        self.agent._ACTIONS[step] = pass_action(
            market=[["SELL", item, quantity]]
        )

    def test_available_stock_moves_four_turns_early(self):
        self.replace_route_sale(step=104, item="MILK", quantity=6)
        observation = make_observation(
            step=100,
            shed={"MILK": 5},
            prices={"MILK": 160},
            shops=[],
        )
        state = self.agent._market_state(observation, 100)

        result = self.agent._adaptive_front_run(
            pass_action(), observation, state, 100
        )

        self.assertEqual(result["market"], [["SELL", "MILK", 5]])
        self.assertEqual(state["front_debt"], {104: {"MILK": 5}})

    def test_sale_does_not_cross_intervening_shop_demand(self):
        self.replace_route_sale(step=104, item="MILK", quantity=6)
        observation = make_observation(
            step=100,
            shed={"MILK": 6},
            prices={"MILK": 160},
            shops=["SMOOTHIE_SHOP"],
        )
        state = self.agent._market_state(observation, 100)

        result = self.agent._adaptive_front_run(
            pass_action(), observation, state, 100
        )

        self.assertEqual(result["market"], [])
        self.assertEqual(state["front_debt"], {})

    def test_repayment_reduces_future_sale_exactly_once(self):
        state = {
            "last_step": 103,
            "deferred": {},
            "front_debt": {104: {"MILK": 5}},
        }
        action = pass_action(market=[["SELL", "MILK", 8]])

        first = self.agent._repay_front_debt(action, state, 104)
        second = self.agent._repay_front_debt(action, state, 104)

        self.assertEqual(first["market"], [["SELL", "MILK", 3]])
        self.assertEqual(second["market"], [["SELL", "MILK", 8]])
        self.assertEqual(state["front_debt"], {})

    def test_lookahead_stops_after_four_turns(self):
        self.replace_route_sale(step=105, item="WOOL", quantity=4)
        observation = make_observation(step=100, shed={"WOOL": 4}, shops=[])
        state = self.agent._market_state(observation, 100)

        result = self.agent._adaptive_front_run(
            pass_action(), observation, state, 100
        )

        self.assertEqual(result["market"], [])
        self.assertEqual(state["front_debt"], {})

    def test_nonpremium_future_sale_is_not_moved(self):
        self.replace_route_sale(step=102, item="WHEAT", quantity=7)
        observation = make_observation(step=100, shed={"WHEAT": 7}, shops=[])
        state = self.agent._market_state(observation, 100)

        result = self.agent._adaptive_front_run(
            pass_action(), observation, state, 100
        )

        self.assertEqual(result["market"], [])

    def test_pickup_and_current_sale_reserves_cap_moved_quantity(self):
        self.replace_route_sale(step=104, item="STRAWBERRY", quantity=10)
        observation = make_observation(
            step=100,
            shed={"STRAWBERRY": 8},
            prices={"STRAWBERRY": 120},
            shops=[],
        )
        state = self.agent._market_state(observation, 100)
        action = pass_action(
            farmer=["PICKUP", "STRAWBERRY", 3],
            market=[["SELL", "STRAWBERRY", 2]],
        )

        result = self.agent._adaptive_front_run(
            action, observation, state, 100
        )

        self.assertEqual(result["farmer"], ["PICKUP", "STRAWBERRY", 3])
        self.assertEqual(result["market"], [["SELL", "STRAWBERRY", 5]])
        self.assertEqual(state["front_debt"], {104: {"STRAWBERRY": 3}})

    def test_full_market_queue_prevents_move_and_debt(self):
        self.replace_route_sale(step=104, item="MELON", quantity=4)
        observation = make_observation(step=100, shed={"MELON": 4}, shops=[])
        state = self.agent._market_state(observation, 100)
        full_queue = pass_action(market=[["HIRE"] for _ in range(10)])

        result = self.agent._adaptive_front_run(
            full_queue, observation, state, 100
        )

        self.assertEqual(len(result["market"]), 10)
        self.assertEqual(state["front_debt"], {})

    def test_repayment_handles_multiple_orders_and_items(self):
        state = {
            "last_step": 103,
            "deferred": {},
            "front_debt": {104: {"MILK": 5, "WOOL": 2}},
        }
        action = pass_action(
            market=[
                ["SELL", "MILK", 3],
                ["SELL", "WHEAT", 7],
                ["SELL", "MILK", 4],
                ["SELL", "WOOL", 5],
            ]
        )

        result = self.agent._repay_front_debt(action, state, 104)

        self.assertEqual(
            result["market"],
            [
                ["SELL", "WHEAT", 7],
                ["SELL", "MILK", 2],
                ["SELL", "WOOL", 3],
            ],
        )


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_agent(EVALUATOR_PATH, "market_overlay_evaluator")

    def test_variant_matrix_keeps_policies_isolated(self):
        self.assertEqual(
            self.evaluator.VARIANTS,
            {
                "control": (False, False),
                "demand_defer": (True, False),
                "adaptive_front_run": (False, True),
            },
        )

    def test_renderer_changes_each_requested_flag_once(self):
        source = (
            "_ENABLE_DEMAND_DEFERRAL = False\n"
            "_ENABLE_ADAPTIVE_FRONT_RUN = False\n"
        )

        rendered = self.evaluator.render_variant(
            source,
            {
                "_ENABLE_DEMAND_DEFERRAL": True,
                "_ENABLE_ADAPTIVE_FRONT_RUN": False,
            },
        )

        self.assertEqual(
            rendered,
            "_ENABLE_DEMAND_DEFERRAL = True\n"
            "_ENABLE_ADAPTIVE_FRONT_RUN = False\n",
        )

    def test_renderer_rejects_unknown_missing_and_duplicate_flags(self):
        valid = (
            "_ENABLE_DEMAND_DEFERRAL = False\n"
            "_ENABLE_ADAPTIVE_FRONT_RUN = False\n"
        )
        with self.assertRaises(ValueError):
            self.evaluator.render_variant(valid, {"_UNKNOWN": True})
        with self.assertRaises(ValueError):
            self.evaluator.render_variant(
                "_ENABLE_DEMAND_DEFERRAL = False\n",
                {"_ENABLE_ADAPTIVE_FRONT_RUN": True},
            )
        with self.assertRaises(ValueError):
            self.evaluator.render_variant(
                valid + "_ENABLE_DEMAND_DEFERRAL = True\n",
                {"_ENABLE_DEMAND_DEFERRAL": True},
            )

    def test_promotion_gate_reports_every_binding_failure(self):
        self.assertEqual(self.evaluator.promotion_failures(passing_summary()), [])
        cases = {
            "unexpected_game_count": lambda summary: summary.update(games=99),
            "paired_mean_not_positive": lambda summary: summary["paired_seeds"][
                "margin"
            ].update(mean=0),
            "paired_median_not_positive": lambda summary: summary[
                "paired_seeds"
            ]["margin"].update(median=0),
            "win_rate_not_above_55_percent": lambda summary: summary.update(
                wins=55
            ),
            "seat_zero_mean_not_positive": lambda summary: summary[
                "by_agent_a_seat"
            ]["0"]["margin"].update(mean=0),
            "seat_one_mean_not_positive": lambda summary: summary[
                "by_agent_a_seat"
            ]["1"]["margin"].update(mean=0),
            "bootstrap_lower_not_positive": lambda summary: summary[
                "paired_seeds"
            ]["bootstrap_mean_95ci"].update(lower=0),
        }
        for expected, mutate in cases.items():
            with self.subTest(expected=expected):
                summary = passing_summary()
                mutate(summary)
                self.assertIn(expected, self.evaluator.promotion_failures(summary))

    def test_screen_winner_requires_positive_mean_and_stable_tiebreak(self):
        variants = {
            "control": {"summary": passing_summary(20)},
            "demand_defer": {"summary": passing_summary(20)},
            "adaptive_front_run": {"summary": passing_summary(20)},
        }
        variants["control"]["summary"]["paired_seeds"]["margin"].update(
            mean=0, median=0
        )
        variants["demand_defer"]["summary"]["paired_seeds"]["margin"].update(
            mean=12, median=3
        )
        variants["adaptive_front_run"]["summary"]["paired_seeds"][
            "margin"
        ].update(mean=12, median=3)

        winner = self.evaluator.select_screen_winner(variants)

        self.assertEqual(winner, "adaptive_front_run")
        variants["demand_defer"]["summary"]["paired_seeds"]["margin"][
            "mean"
        ] = -1
        variants["adaptive_front_run"]["summary"]["paired_seeds"]["margin"][
            "mean"
        ] = 0
        self.assertIsNone(self.evaluator.select_screen_winner(variants))

    def test_control_validation_accepts_seat_bias_only_when_pairs_cancel(self):
        valid = [
            {"seed": 0, "agent_a_seat": 0, "margin": -239.0},
            {"seed": 0, "agent_a_seat": 1, "margin": 239.0},
            {"seed": 1, "agent_a_seat": 0, "margin": 1006.0},
            {"seed": 1, "agent_a_seat": 1, "margin": -1006.0},
        ]
        broken = [dict(row) for row in valid]
        broken[-1]["margin"] = -1005.0

        self.assertEqual(self.evaluator.control_pair_failures(valid), [])
        self.assertEqual(self.evaluator.control_pair_failures(broken), [1])

    def test_variant_metadata_hides_temporary_path_and_timestamp(self):
        candidate = self.evaluator.benchmark.AgentRef(
            "temporary.py", "temporary", Path("/tmp/generated/main.py"), "abc"
        )
        baseline = self.evaluator.benchmark.AgentRef(
            "baseline.py", "baseline", BASELINE_PATH, "def"
        )
        source = self.evaluator.benchmark.AgentRef(
            "candidate.py", "candidate", CANDIDATE_PATH, "ghi"
        )

        metadata = self.evaluator.build_variant_metadata(
            candidate,
            baseline,
            source,
            "demand_defer",
            {
                "_ENABLE_DEMAND_DEFERRAL": True,
                "_ENABLE_ADAPTIVE_FRONT_RUN": False,
            },
            0,
            10,
        )

        self.assertNotIn("created_at_utc", metadata)
        self.assertIsNone(metadata["agent_a"]["resolved_path"])
        self.assertNotIn("/tmp/generated", json.dumps(metadata))

    def test_latency_gate_uses_strict_mean_and_p95_limits(self):
        passing = {"mean_ms": 0.999, "p95_ms": 1.999}
        self.assertEqual(self.evaluator.latency_failures(passing), [])
        self.assertEqual(
            self.evaluator.latency_failures({"mean_ms": 1.0, "p95_ms": 2.0}),
            ["mean_latency_not_below_1_ms", "p95_latency_not_below_2_ms"],
        )


class PromotionIntegrityTests(unittest.TestCase):
    def test_promoted_main_matches_committed_winner_identity(self):
        promotion = json.loads(PROMOTION_RECORD_PATH.read_text())
        actual = hashlib.sha256(PROMOTED_MAIN_PATH.read_bytes()).hexdigest()

        self.assertTrue(promotion["promoted"])
        self.assertEqual(promotion["failures"], [])
        self.assertEqual(actual, promotion["candidate_sha256"])

    def test_promoted_main_is_exact_render_of_committed_source_and_flags(self):
        promotion = json.loads(PROMOTION_RECORD_PATH.read_text())
        source_bytes = CANDIDATE_PATH.read_bytes()
        evaluator = load_agent(EVALUATOR_PATH, "promotion_source_renderer")
        rendered = evaluator.render_variant(
            source_bytes.decode("utf-8"), promotion["flags"]
        ).encode("utf-8")

        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            promotion["source_candidate_sha256"],
        )
        self.assertEqual(rendered, PROMOTED_MAIN_PATH.read_bytes())
        self.assertEqual(
            hashlib.sha256(rendered).hexdigest(), promotion["candidate_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
