"""Small, structurally valid replays used by replay and feature tests."""

from bc_core.constants import ARGUMENT_ITEMS, CROPS, EXPECTED_REPLAY_CONFIGURATION, PRODUCTS


def _tiles(*, day: int, offset: int) -> list[list[object]]:
    tiles: list[list[object]] = [["LOCKED" for _ in range(10)] for _ in range(10)]
    tiles[0][0] = None
    tiles[0][1] = "WEED"
    tiles[0][2] = {"kind": "COOP"}
    tiles[0][3] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 3 + offset,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": True,
        "consecutive_unfed": 0.25,
    }
    tiles[0][4] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "yield_units": 2 + offset,
        "watered_today": offset == 0,
        "fertilized_until_day": day,
        "consecutive_unwatered": 0.5,
    }
    return tiles


def make_observation(step: int, player: int, hands: int) -> dict:
    """Return a complete current-state observation with seat-relative values."""
    day, hour = divmod(step, 24)
    farms = [None, None]
    farms[player] = {
        "farmer": [4, 4],
        "hands": [[index + 1, index + 2] for index in range(hands)],
        "hires_today": 1,
        "money": 100 if step == 0 else 999999 if step == 1 else 100 + step,
        "tiles": _tiles(day=day, offset=0),
        "unlocked_quadrants": ["NW", "SE"],
    }
    farms[1 - player] = {
        "farmer": [9, 9],
        "hands": [[8 - index, 7 - index] for index in range(hands)],
        "hires_today": 2,
        "money": 200 + step,
        "tiles": _tiles(day=day, offset=1),
        "unlocked_quadrants": ["NE", "SW"],
    }
    inventories = [
        {item: actor + item_index + 1 for item_index, item in enumerate(ARGUMENT_ITEMS)}
        for actor in range(hands + 1)
    ]
    return {
        "day": day,
        "farms": farms,
        "hour": hour,
        "market": {
            "inventory": {product: 100 + index for index, product in enumerate(PRODUCTS)},
            "prices": {product: 10 + index for index, product in enumerate(PRODUCTS)},
        },
        "player": player,
        "private": {
            "inventories": inventories,
            "seeds": {crop: index + 1 for index, crop in enumerate(CROPS)},
            "shed": {item: index + 1 for index, item in enumerate(ARGUMENT_ITEMS)},
        },
        "step": step,
        "town": {"unlocked_shops": ["BAKERY", "BAKERY", "FARMERS_MARKET"]},
    }


def make_replay(*, ryo_seat: int = 0, hands: int = 1) -> dict:
    names = ["Ryo Hasegawa", "Opponent"] if ryo_seat == 0 else ["Opponent", "Ryo Hasegawa"]
    steps = []
    for state in range(720):
        observations = [make_observation(state, seat, hands) for seat in range(2)]
        action = {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}
        agents = [
            {"observation": observations[0], "action": action, "reward": 0, "status": "ACTIVE"},
            {"observation": observations[1], "action": action, "reward": 0, "status": "ACTIVE"},
        ]
        steps.append(agents)
    steps[-1][ryo_seat].update(reward=1, status="DONE")
    steps[-1][1 - ryo_seat].update(reward=0, status="DONE")
    return {
        "id": "fixture-uuid",
        "info": {
            "EpisodeId": "fixture-game",
            "Agents": [{"Name": name} for name in names],
            "TeamNames": names,
        },
        "module_version": "1.32.7",
        "configuration": dict(EXPECTED_REPLAY_CONFIGURATION),
        "rewards": [1, 0] if ryo_seat == 0 else [0, 1],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }
