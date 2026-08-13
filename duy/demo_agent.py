"""A small Kaggriculture wheat-loop agent."""

import argparse
import sys
from collections.abc import Callable
from typing import Any


def agent(obs):
    """Choose one simple wheat-farming action for the current turn."""
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    seeds = private.get("seeds", {}).get("WHEAT", 0)
    shed_wheat = private.get("shed", {}).get("WHEAT", 0)

    market = []
    if seeds == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
    if shed_wheat > 0:
        market.append(["SELL", "WHEAT", shed_wheat])

    farmer = ["PASS"]
    if tile is None and seeds > 0:
        farmer = ["PLANT", "WHEAT"]
    elif (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "WHEAT"
    ):
        age = obs["day"] - tile["planted_day"]
        if age >= 2:
            farmer = ["HARVEST"]
        elif not tile.get("watered_today", False):
            farmer = ["WATER"]

    return {
        "farmer": farmer,
        "hands": [["PASS"] for _ in me.get("hands", [])],
        "market": market,
    }


def run_demo(
    steps: int = 200,
    opponent: str = "random",
    *,
    make_environment: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Run a deterministic match and return a summary for both players."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    if make_environment is None:
        from kaggle_environments import make as make_environment

    environment = make_environment(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": 7},
        debug=True,
    )
    environment.run([agent, opponent])

    names = ["demo_agent", opponent]
    summaries = []
    for index, state in enumerate(environment.steps[-1]):
        observation = state.observation
        player = observation.get("player", index)
        summaries.append(
            {
                "agent": names[index],
                "reward": state.reward,
                "status": state.status,
                "money": observation["farms"][player]["money"],
            }
        )
    return summaries


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[[int, str], list[dict[str, Any]]] = run_demo,
) -> int:
    """Run the command-line demo and return its process exit code."""
    parser = argparse.ArgumentParser(description="Run the Kaggriculture demo agent.")
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of episode turns (default: 200)",
    )
    parser.add_argument(
        "--opponent",
        default="random",
        help="built-in opponent name (default: random)",
    )
    args = parser.parse_args(argv)

    try:
        summaries = runner(args.steps, args.opponent)
    except Exception as exc:
        print(f"Unable to run Kaggriculture: {exc}", file=sys.stderr)
        print(
            "Use Python 3.11+ and install a current release with "
            "`python3.12 -m pip install -U kaggle-environments`.",
            file=sys.stderr,
        )
        return 1

    for result in summaries:
        print(
            f"{result['agent']}: reward={result['reward']} "
            f"status={result['status']} money={result['money']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
