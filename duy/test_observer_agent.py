import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from observer_agent import agent


class ObserverAgentTests(unittest.TestCase):
    def test_prints_the_complete_observation_as_json(self):
        obs = {
            "player": 0,
            "step": 7,
            "farms": [{"hands": [[5, 4]], "money": 3000}],
            "private": {"seeds": {"WHEAT": 2}},
        }
        output = StringIO()

        with redirect_stdout(output):
            agent(obs)

        self.assertEqual(json.loads(output.getvalue()), obs)
        self.assertIn('\n  "farms":', output.getvalue())

    def test_returns_pass_actions_for_farmer_and_hands(self):
        obs = {
            "player": 1,
            "farms": [
                {"hands": []},
                {"hands": [[5, 4], [4, 5]]},
            ],
        }

        with redirect_stdout(StringIO()):
            action = agent(obs)

        self.assertEqual(
            action,
            {
                "farmer": ["PASS"],
                "hands": [["PASS"], ["PASS"]],
                "market": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
