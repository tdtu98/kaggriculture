import json
import unittest
from pathlib import Path

from bc_core.constants import (
    ACTOR_DIM, CLOCK_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS, load_config,
)

ROOT = Path(__file__).resolve().parents[1]


class ConstantsTest(unittest.TestCase):
    def test_v0_contract_is_fixed(self) -> None:
        config = load_config(ROOT / "configs" / "v0.json")
        self.assertEqual(len(OPERATIONS), 17)
        self.assertEqual((GRID_CHANNELS, ACTOR_DIM, GLOBAL_DIM, CLOCK_DIM), (44, 38, 62, 8))
        self.assertEqual(config["schema_version"], "ryo-bc-v0")
        self.assertEqual(config["seed"], 20260824)
        self.assertEqual(config["training"]["batch_size"], 512)
        self.assertEqual(config["training"]["max_epochs"], 50)
        json.dumps(config, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
