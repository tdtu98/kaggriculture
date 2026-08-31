"""Public behavior contracts for the three baseline model families."""

import unittest

import numpy as np
import torch

from bc_core.constants import OPERATIONS
from model.clock import ClockOnlyModel
from model.majority import fit_majority_rules, ranking_to_logits
from model.state import StateAwareModel


class PublicModelTest(unittest.TestCase):
    def test_majority_rules_use_actor_specific_train_rankings(self) -> None:
        """Catch a baseline that ignores actor type or unstable tie ordering."""
        labels = np.array([2, 2, 4, 4, 4, 6], dtype=np.int64)
        actors = np.array([1, 1, 0, 0, 0, 0], dtype=np.int64)

        rules = fit_majority_rules(labels, actors)

        np.testing.assert_array_equal(
            rules.predict(np.array([1, 0])), np.array([2, 4])
        )
        self.assertEqual(rules.farmer_ranking[:2], (2, 0))
        self.assertEqual(rules.hand_ranking[:3], (4, 6, 0))
        self.assertEqual(int(np.argmax(ranking_to_logits(rules.global_ranking))), 4)

    def test_clock_model_owns_the_eight_feature_forward_path(self) -> None:
        """Catch a public clock model with a changed input or output contract."""
        logits = ClockOnlyModel()(torch.zeros(3, 8))
        self.assertEqual(tuple(logits.shape), (3, len(OPERATIONS)))

    def test_state_model_owns_the_grid_global_actor_forward_path(self) -> None:
        """Catch a public state model missing one of the fixed feature branches."""
        logits = StateAwareModel()(
            torch.zeros(2, 44, 10, 10),
            torch.zeros(2, 62),
            torch.zeros(2, 38),
        )
        self.assertEqual(tuple(logits.shape), (2, len(OPERATIONS)))


if __name__ == "__main__":
    unittest.main()
