"""Fixed v0 vocabulary, dimensions, and configuration contract."""

import json
from pathlib import Path
from typing import Any

OPERATIONS = (
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP", "PLANT",
    "WATER", "HARVEST", "FERTILIZE", "BUILD_PASTURE", "DIG", "PLACE", "FEED",
    "COLLECT_FERTILIZER", "CARE",
)
OPERATION_TO_ID = {operation: index for index, operation in enumerate(OPERATIONS)}

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP",
    "SMOOTHIE_SHOP", "YARN_STORE",
)
KNOWN_SHOPS = SHOPS + ("FARMERS_MARKET",)
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
ARGUMENT_ITEMS = PRODUCTS + ANIMALS

GRID_CHANNELS = 44
ACTOR_DIM = 38
GLOBAL_DIM = 62
CLOCK_DIM = 8

EXPECTED_REPLAY_CONFIGURATION = {
    "actTimeout": 1,
    "boardSize": 10,
    "episodeSteps": 720,
    "farmHandCostMult": 1,
    "marketParams": {},
    "maxMarketOrdersPerTurn": 10,
    "runTimeout": 1200,
    "seed": None,
    "shedCapacity": 100,
    "startingMoney": 3000,
    "townCenterSellInterval": 24,
    "townShopSellInterval": 4,
    "townShopUnlockInterval": 3,
    "turnsPerDay": 24,
    "weedSpawnChance": 0.005,
}

EXPECTED_MODULE_VERSION = "1.32.7"

_REQUIRED_CONFIG = {
    "schema_version": "ryo-bc-v0",
    "feature_schema_version": "ryo-features-v0",
    "seed": 20260824,
    "corpus_root": "duy_explore/ryo_hasegawa_100_stratified",
    "module_version": EXPECTED_MODULE_VERSION,
    "training": {
        "learning_rate": 0.001,
        "batch_size": 512,
        "max_epochs": 50,
        "patience": 5,
        "weight_cap": 4.0,
    },
    "bootstrap_resamples": 10000,
}


def load_config(path: Path) -> dict[str, Any]:
    """Load fixed v0 settings while allowing the local corpus path to vary."""
    with path.open(encoding="utf-8") as config_file:
        config: Any = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("v0 configuration must be a JSON object")
    corpus_root = config.get("corpus_root")
    if (
        not isinstance(corpus_root, str)
        or not corpus_root.strip()
        or "\x00" in corpus_root
    ):
        raise ValueError("configuration corpus_root must be a non-empty path string")
    fixed = dict(config)
    fixed.pop("corpus_root")
    required_fixed = dict(_REQUIRED_CONFIG)
    required_fixed.pop("corpus_root")
    if fixed != required_fixed:
        raise ValueError("configuration does not match the ryo-bc-v0 contract")
    return config
