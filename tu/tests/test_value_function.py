"""`FarmModel.score()` — the ranking signal P1's rollout depends on (P1.1).

Behavioural tests only. The gate that matters is empirical (75% pairwise ranking accuracy, E38);
these pin the properties that make the number meaningful, so a refactor cannot quietly invert a
sign or drop a term while still "passing" somewhere downstream.
"""

from __future__ import annotations

import kagsim

from agent.forward import BASE_PRICE, FarmModel

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def farm_after(steps, market=None, seed=3):
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.0})
    for i in range(steps):
        act = {"farmer": ["PASS"], "hands": [], "market": (market or {}).get(i, [])}
        sim.step([act, PASS])
    return sim


def test_empty_farm_is_worth_its_cash():
    sim = farm_after(1)
    m = FarmModel.from_obs(sim.observation(0), player=0)
    assert m.score(cash=3000.0) == 3000.0


def test_goods_in_the_shed_count_at_base_price():
    sim = farm_after(2, {0: [["BUY_PRODUCT", "WHEAT", 4]]})
    m = FarmModel.from_obs(sim.observation(0), player=0)
    assert m.score(cash=0.0) == 4 * BASE_PRICE["WHEAT"]


def test_seeds_count_as_purchased_capacity():
    """A farm holding seeds is not the same as one that has spent everything (E38)."""
    bare = FarmModel.from_obs(farm_after(1).observation(0), player=0)
    seeded = FarmModel.from_obs(
        farm_after(2, {0: [["BUY_SEED", "MELON", 2]]}).observation(0), player=0)
    assert seeded.score(cash=0.0) > bare.score(cash=0.0)


def test_unripe_zero_ignores_future_yield():
    """`unripe` scales only the not-yet-grown term, so at 0 a young crop adds nothing."""
    sim = farm_after(4, {0: [["BUY_SEED", "MELON", 1]]})
    m = FarmModel.from_obs(sim.observation(0), player=0)
    m.tiles[4][4] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "watered_today": False,
                     "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": 312,
                     "fertilized_until_day": -1}
    assert m.score(cash=0.0, unripe=0.0) < m.score(cash=0.0, unripe=1.0)


def test_accrued_yield_is_valued_above_potential():
    """Yield in the tile is worth full price; yield that still needs work is discounted."""
    sim = farm_after(4, {0: [["BUY_SEED", "MELON", 1]]})
    base = FarmModel.from_obs(sim.observation(0), player=0)
    tile = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "watered_today": False,
            "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": 312,
            "fertilized_until_day": -1}
    empty_plant = base.clone(); empty_plant.tiles[4][4] = dict(tile)
    ripe = base.clone(); ripe.tiles[4][4] = {**tile, "yield_units": 3}
    gain = ripe.score(cash=0.0, unripe=0.5) - empty_plant.score(cash=0.0, unripe=0.5)
    assert gain > 0, "accrued yield must raise the score"
    # 3 units gained at full price, minus the 3 units of discounted potential they consumed.
    assert gain == 3 * BASE_PRICE["MELON"] - 0.5 * 3 * BASE_PRICE["MELON"]


def test_a_dead_season_has_no_future_value():
    """Past the end of the season nothing can still grow, so only what exists counts."""
    sim = farm_after(4, {0: [["BUY_SEED", "MELON", 1]]})
    m = FarmModel.from_obs(sim.observation(0), player=0)
    m.tiles[4][4] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "watered_today": False,
                     "consecutive_unwatered": 0, "yield_units": 2, "max_lifespan_step": 312,
                     "fertilized_until_day": -1}
    m.day = 40
    m.seeds = {}          # the setup seed is itself capacity and would otherwise be counted
    assert m.score(cash=0.0, unripe=1.0, season_end_day=30) == 2 * BASE_PRICE["MELON"]
