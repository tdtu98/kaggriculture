import json
import unittest
from pathlib import Path

import numpy as np

from fixtures import make_replay
from bc_core.features import FeatureError, encode_game
from bc_core.replay import SourceReplay


class FeatureTest(unittest.TestCase):
    def source(self, seat: int = 0) -> SourceReplay:
        return SourceReplay(
            "train", "fixture-game", Path("fixture.json"), "a" * 64, "21-08", "route-a"
        )

    def test_shapes_actor_count_and_current_state_only(self) -> None:
        # Catches reading action[t]'s next observation instead of observation[t].
        replay = make_replay(ryo_seat=0, hands=2)
        replay["steps"][1][0]["observation"]["farms"][0]["money"] = 999999
        game = encode_game(self.source(), replay)
        self.assertEqual(game.grid.shape, (719, 44, 10, 10))
        self.assertEqual(game.global_features.shape, (719, 62))
        self.assertEqual(game.actor_features.shape, (719 * 3, 38))
        self.assertEqual(game.step_index.shape, (719 * 3,))
        self.assertEqual(game.label.shape, (719 * 3,))
        self.assertTrue(np.all(game.step_index[:3] == 0))
        self.assertAlmostEqual(game.global_features[0, 6], np.log1p(100), places=5)
        self.assertAlmostEqual(game.global_features[1, 6], np.log1p(999999), places=5)
        self.assertEqual(np.count_nonzero(game.grid[:, 21]), 0)
        self.assertEqual(np.count_nonzero(game.grid[:, 43]), 0)
        self.assertNotIn("reward", json.dumps(game.metadata).lower())

    def test_self_farm_is_first_for_ryo_in_seat_one(self) -> None:
        # Catches concatenating farms by physical seat instead of self/opponent.
        seat_zero = encode_game(self.source(0), make_replay(ryo_seat=0))
        seat_one = encode_game(self.source(1), make_replay(ryo_seat=1))
        np.testing.assert_allclose(seat_zero.grid[:, :22], seat_one.grid[:, :22])

    def test_unexpected_category_fails(self) -> None:
        # Catches silently treating an unknown tile as empty/all-zero.
        replay = make_replay()
        replay["steps"][0][0]["observation"]["farms"][0]["tiles"][0][0] = {
            "kind": "PORTAL"
        }
        with self.assertRaisesRegex(FeatureError, "PORTAL.*step=0"):
            encode_game(self.source(), replay)

    def test_exact_channel_and_feature_order(self) -> None:
        # Catches reordered channels/groups and incorrect fixed normalizations.
        replay = make_replay(hands=1)
        replay["steps"][1][0]["action"] = {
            "farmer": ["DROP", "WHEAT", 3],
            "hands": [["NORTH"]],
            "market": [["IGNORED"]],
        }
        game = encode_game(self.source(), replay)

        grid = game.grid[0, :22]
        self.assertEqual(grid[0, 0, 0], 1.0)  # EMPTY
        self.assertEqual(grid[2, 0, 1], 1.0)  # WEED
        self.assertEqual(grid[4, 0, 2], 1.0)  # COOP
        self.assertEqual(grid[5, 0, 3], 1.0)  # PASTURE
        self.assertEqual(grid[3, 0, 4], 1.0)  # PLANT
        self.assertEqual(grid[6, 0, 4], 1.0)  # WHEAT
        self.assertEqual(grid[12, 0, 3], 1.0)  # COW
        self.assertAlmostEqual(grid[14, 0, 3], 0.5)
        self.assertAlmostEqual(grid[14, 0, 4], 1.0 / 3.0)
        np.testing.assert_array_equal(grid[15:21, 0, 4], [1, 0, 0, 1, 0.5, 0])

        np.testing.assert_allclose(game.global_features[0, :6], [0, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(
            game.global_features[0, 6:12],
            [np.log1p(100), np.log1p(200), 1, 1, 1, 2],
        )
        np.testing.assert_array_equal(
            game.global_features[0, 12:20], [1, 0, 0, 1, 0, 1, 1, 0]
        )
        np.testing.assert_allclose(game.global_features[0, 20:32], np.log1p(range(1, 13)))
        np.testing.assert_allclose(game.global_features[0, 32:37], np.log1p(range(1, 6)))
        np.testing.assert_allclose(
            game.global_features[0, 37:41],
            [np.log1p(100), np.log1p(10), np.log1p(101), np.log1p(11)],
        )
        np.testing.assert_array_equal(game.global_features[0, 55:62], [2, 0, 0, 0, 0, 0, 0])

        np.testing.assert_allclose(game.actor_features[0, :4], [1, 0, 4 / 9, 4 / 9])
        np.testing.assert_allclose(game.actor_features[0, 4:16], np.log1p(range(1, 13)))
        self.assertEqual(game.actor_features[0, 16], 1)
        self.assertEqual(game.actor_features[0, 18], 1)  # current tile is LOCKED
        np.testing.assert_allclose(game.actor_features[1, :4], [0, 1 / 8, 1 / 9, 2 / 9])
        np.testing.assert_allclose(game.actor_features[1, 4:16], np.log1p(range(2, 14)))
        self.assertEqual(game.actor_features[1, 16], 0)
        np.testing.assert_array_equal(game.label[:2], [6, 0])
        np.testing.assert_array_equal(game.argument_item[:2], [0, -1])
        np.testing.assert_array_equal(game.argument_quantity[:2], [3, -1])

    def test_explicit_fertilizer_flag_overrides_plant_day_fallback(self) -> None:
        # Catches applying the day fallback when an explicit false flag exists.
        replay = make_replay()
        plant = replay["steps"][0][0]["observation"]["farms"][0]["tiles"][0][4]
        plant["fertilizer_available"] = False
        game = encode_game(self.source(), replay)
        self.assertEqual(game.grid[0, 18, 0, 4], 0)
        self.assertEqual(game.grid[0, 18, 0, 3], 1)
        self.assertEqual(game.grid[0, 18, 0, 2], 0)

    def test_unknown_shop_and_inventory_length_fail_with_context(self) -> None:
        # Catches accepting categories that cannot be represented or misaligning actors.
        replay = make_replay()
        replay["steps"][0][0]["observation"]["town"]["unlocked_shops"].append("ARCADE")
        with self.assertRaisesRegex(FeatureError, "ARCADE.*step=0"):
            encode_game(self.source(), replay)

        replay = make_replay(hands=1)
        replay["steps"][0][0]["observation"]["private"]["inventories"].pop()
        with self.assertRaisesRegex(FeatureError, "inventories.*step=0"):
            encode_game(self.source(), replay)

    def test_invalid_coordinate_and_negative_count_fail_with_context(self) -> None:
        # Catches indexing with invalid positions and silently clipping impossible counts.
        replay = make_replay()
        replay["steps"][0][0]["observation"]["farms"][0]["farmer"] = [-1, 4]
        with self.assertRaisesRegex(FeatureError, "coordinate.*step=0"):
            encode_game(self.source(), replay)

        replay = make_replay()
        replay["steps"][0][0]["observation"]["private"]["shed"]["WHEAT"] = -1
        with self.assertRaisesRegex(FeatureError, "negative.*step=0"):
            encode_game(self.source(), replay)

    def test_metadata_is_canonical_source_provenance(self) -> None:
        # Catches metadata drift, including accidental terminal outcome leakage.
        game = encode_game(self.source(), make_replay())
        self.assertEqual(
            game.metadata,
            {
                "schema_version": "ryo-features-v0",
                "split": "train",
                "episode_id": "fixture-game",
                "ryo_seat": 0,
                "source_path": "fixture.json",
                "source_sha256": "a" * 64,
                "source_date": "21-08",
                "route_family": "route-a",
                "sample_count": 1438,
                "shapes": {
                    "grid": [719, 44, 10, 10],
                    "global_features": [719, 62],
                    "actor_features": [1438, 38],
                    "step_index": [1438],
                    "label": [1438],
                },
            },
        )

    def test_shifted_decision_step_does_not_require_observation_step_field(self) -> None:
        # Catches depending on an optional replay field instead of Decision.step.
        replay = make_replay()
        for agents in replay["steps"]:
            agents[0]["observation"].pop("step")
        game = encode_game(self.source(), replay)
        self.assertEqual(game.global_features[0, 0], 0)
        self.assertAlmostEqual(game.global_features[718, 0], 718 / 719)

    def test_unknown_actor_action_has_complete_sample_context(self) -> None:
        # Catches leaking a context-free ReplayError from actor action parsing.
        replay = make_replay(hands=1)
        replay["steps"][1][0]["action"]["hands"][0] = ["TELEPORT"]
        with self.assertRaises(FeatureError) as raised:
            encode_game(self.source(), replay)
        message = str(raised.exception)
        for expected in (
            "unknown operation TELEPORT",
            "split=train",
            "episode=fixture-game",
            "step=0",
            "seat=0",
            "actor=1",
        ):
            self.assertIn(expected, message)

    def test_invalid_carried_count_has_complete_sample_context(self) -> None:
        # Catches inventory validation that cannot identify the failing actor.
        replay = make_replay(hands=1)
        replay["steps"][0][0]["observation"]["private"]["inventories"][1]["WHEAT"] = -1
        with self.assertRaises(FeatureError) as raised:
            encode_game(self.source(), replay)
        message = str(raised.exception)
        for expected in (
            "negative count inventory.WHEAT=-1",
            "split=train",
            "episode=fixture-game",
            "step=0",
            "seat=0",
            "actor=1",
        ):
            self.assertIn(expected, message)


if __name__ == "__main__":
    unittest.main()
