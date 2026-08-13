"""A Kaggriculture agent that prints the observation visible to it."""

import json


def agent(obs):
    """Print the complete observation and return valid no-op actions."""
    print(json.dumps(obs, indent=2, sort_keys=True), flush=True)

    player = obs["player"]
    hands = obs["farms"][player].get("hands", [])
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }
