import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace

from demo_agent import agent, main, run_demo


def observation(tile=None, *, seeds=0, shed_wheat=0, money=3000, day=0, hands=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[4][4] = tile
    hands = [] if hands is None else hands
    return {
        "player": 0,
        "day": day,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": hands,
            }
        ],
        "private": {
            "shed": {"WHEAT": shed_wheat},
            "seeds": {"WHEAT": seeds},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
    }


class AgentTests(unittest.TestCase):
    def test_buys_seed_and_passes_when_none_available(self):
        action = agent(observation())
        self.assertEqual(action["farmer"], ["PASS"])
        self.assertEqual(action["market"], [["BUY_SEED", "WHEAT", 1]])

    def test_plants_available_seed(self):
        self.assertEqual(
            agent(observation(seeds=1))["farmer"], ["PLANT", "WHEAT"]
        )

    def test_waters_young_wheat(self):
        tile = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "watered_today": False,
        }
        self.assertEqual(agent(observation(tile, seeds=1, day=1))["farmer"], ["WATER"])

    def test_harvests_mature_wheat(self):
        tile = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "watered_today": True,
        }
        self.assertEqual(
            agent(observation(tile, seeds=1, day=2))["farmer"], ["HARVEST"]
        )

    def test_sells_shed_wheat(self):
        self.assertIn(
            ["SELL", "WHEAT", 3],
            agent(observation(seeds=1, shed_wheat=3))["market"],
        )

    def test_passes_each_hired_hand(self):
        action = agent(observation(seeds=1, hands=[[4, 4], [3, 4]]))
        self.assertEqual(action["hands"], [["PASS"], ["PASS"]])


class RunnerTests(unittest.TestCase):
    def test_rejects_nonpositive_steps(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            run_demo(0, "random", make_environment=lambda *args, **kwargs: None)

    def test_runs_and_summarizes_match(self):
        calls = {}

        class FakeEnvironment:
            steps = [
                [
                    SimpleNamespace(
                        reward=1,
                        status="DONE",
                        observation={
                            "player": 0,
                            "farms": [{"money": 3025}, {"money": 3000}],
                        },
                    ),
                    SimpleNamespace(
                        reward=0,
                        status="DONE",
                        observation={
                            "player": 1,
                            "farms": [{"money": 3025}, {"money": 3000}],
                        },
                    ),
                ]
            ]

            def run(self, agents):
                calls["agents"] = agents

        def fake_make(name, configuration, debug):
            calls.update(name=name, configuration=configuration, debug=debug)
            return FakeEnvironment()

        result = run_demo(200, "random", make_environment=fake_make)

        self.assertEqual(calls["name"], "kaggriculture")
        self.assertEqual(calls["configuration"], {"episodeSteps": 200, "seed": 7})
        self.assertIs(calls["agents"][0], agent)
        self.assertEqual(calls["agents"][1], "random")
        self.assertEqual(result[0]["money"], 3025)
        self.assertEqual(result[1]["status"], "DONE")

    def test_main_prints_match_summary(self):
        def fake_runner(steps, opponent):
            self.assertEqual((steps, opponent), (12, "pass"))
            return [
                {
                    "agent": "demo_agent",
                    "reward": 1,
                    "status": "DONE",
                    "money": 3025,
                }
            ]

        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["--steps", "12", "--opponent", "pass"], runner=fake_runner)

        self.assertEqual(exit_code, 0)
        self.assertIn("demo_agent: reward=1 status=DONE money=3025", output.getvalue())

    def test_main_reports_runner_error_without_traceback(self):
        def failing_runner(steps, opponent):
            raise RuntimeError("Unknown Environment Specification")

        errors = StringIO()
        with redirect_stderr(errors):
            exit_code = main([], runner=failing_runner)

        self.assertEqual(exit_code, 1)
        self.assertIn("Unable to run Kaggriculture", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
