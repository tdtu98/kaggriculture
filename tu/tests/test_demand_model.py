"""`Engine.town_drain_per_day` is the demand model (P1.5-A.1).

Before kaggle-environments 1.32.6 this only fed price forecasting, and being a bit wrong was
survivable. Now shops are drawn WITH replacement, so which products have a buyer is a per-game
random variable — and it is the only signal available for choosing what to grow. Wool has no shop
buyer in 36% of games and melon in none (E33), while the champion breeds 8 sheep and leads on melon.

So this is tested directly against the environment's own tables and against real draws from kagsim,
rather than inferred from agent behaviour.
"""

from __future__ import annotations

import collections

import kagsim
import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS, TOWN_CENTER_PRODUCTS

from agent.engine import Engine

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
SHOP_TICKS_PER_DAY = 6      # turnsPerDay 24 / townShopSellInterval 4
CENTRE_TICKS_PER_DAY = 1    # turnsPerDay 24 / townCenterSellInterval 24, since 1.32.6


def brute_force(shops: list[str]) -> dict[str, int]:
    """Independent reimplementation, written from `_town_consume` rather than from the engine."""
    drain: collections.Counter = collections.Counter()
    for shop in shops:                       # duplicates consume independently
        products = SHOPS[shop]
        mult = 2 if len(products) == 1 else 1
        for item in products:
            drain[item] += mult * SHOP_TICKS_PER_DAY
    for item in TOWN_CENTER_PRODUCTS:
        drain[item] += CENTRE_TICKS_PER_DAY
    return dict(drain)


def obs_with(shops: list[str], day: int = 12) -> dict:
    return {"town": {"unlocked_shops": list(shops)}, "day": day}


@pytest.mark.parametrize("shops", [
    [],                                             # day 0-2: nothing unlocked yet
    ["YARN_STORE"],
    ["PET_CAFE", "PET_CAFE"],                       # duplicate single-product shop -> 2x each
    ["BRUNCH_SPOT"] * 3,                            # the seed-6 draw: three of one shop
    ["BAKERY", "BAKERY", "ICE_CREAM_SHOP", "ICE_CREAM_SHOP"],
    sorted(SHOPS) * 1,                              # one of everything, the old 1.32.4 endgame
])
def test_matches_brute_force_on_handbuilt_draws(shops):
    assert Engine.town_drain_per_day(obs_with(shops)) == brute_force(shops)


def test_duplicates_are_counted_once_per_instance():
    """The 1.32.6 change in one assertion: N copies of a shop drain N times."""
    one = Engine.town_drain_per_day(obs_with(["YARN_STORE"]))["WOOL"]
    three = Engine.town_drain_per_day(obs_with(["YARN_STORE"] * 3))["WOOL"]
    assert three - CENTRE_TICKS_PER_DAY == 3 * (one - CENTRE_TICKS_PER_DAY)


def test_centre_term_is_flat_all_season():
    """1.32.6 deleted TOWN_CENTER_DEMAND_SCHEDULE; day must no longer change the centre term.

    The previous implementation returned 2/4/8 per day by day threshold, which overstated absorption
    by up to 8x late in the season -- exactly when a crop plan commits.
    """
    early = Engine.town_drain_per_day(obs_with([], day=0))
    late = Engine.town_drain_per_day(obs_with([], day=29))
    assert early == late
    assert all(early[item] == CENTRE_TICKS_PER_DAY for item in TOWN_CENTER_PRODUCTS)


def test_fertilizer_is_never_town_centre_demand():
    """FERTILIZER is excluded from TOWN_CENTER_PRODUCTS and no shop buys it."""
    assert "FERTILIZER" not in Engine.town_drain_per_day(obs_with(sorted(SHOPS)))


def test_matches_brute_force_on_real_kagsim_draws():
    """>=100 draws taken from the simulator itself, so the shape of real draws is covered."""
    seen: collections.Counter = collections.Counter()
    for seed in range(100):
        sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
        for _ in range(719):
            sim.step([PASS, PASS])
        shops = list(sim.observation(0)["town"]["unlocked_shops"])
        assert Engine.town_drain_per_day(obs_with(shops)) == brute_force(shops)
        seen[len(set(shops))] += 1
    # Guard the guard: if draws stopped producing duplicates, this test would pass vacuously.
    assert min(seen) < 8, f"expected duplicate draws, saw distinct-count histogram {dict(seen)}"


def test_wool_really_does_go_unbought():
    """The fact that motivates P1.5-A, asserted so a rules change cannot silently retire it."""
    zero = sum(
        1 for seed in range(100)
        if "WOOL" not in brute_force(_shops_for(seed)) or _shops_for(seed).count("YARN_STORE") == 0
    )
    assert zero > 15, f"only {zero}/100 seeds lacked a yarn store"


_SHOP_CACHE: dict[int, list[str]] = {}


def _shops_for(seed: int) -> list[str]:
    if seed not in _SHOP_CACHE:
        sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
        for _ in range(719):
            sim.step([PASS, PASS])
        _SHOP_CACHE[seed] = list(sim.observation(0)["town"]["unlocked_shops"])
    return _SHOP_CACHE[seed]
