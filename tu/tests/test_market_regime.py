"""E16 — the market regime, pinned as a test.

Two findings here are load-bearing for the whole strategy, and both are the kind that could be
silently reversed by an engine change or an environment update:

* the game is played in **scarcity**, not glut — prices rise, nothing saturates except melon;
* selling immediately beats holding, and it does so *because cash compounds*, not because the
  opponent crashes the price.

If either flips, the strategy needs rethinking rather than retuning, so they get assertions.
"""

from __future__ import annotations

import pytest

import kagsim
from agent import Params, make_agent
from arena.registry import REGISTRY

SHOP_DEMANDED = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MILK", "EGG", "WOOL"]

# Products two players can actually flood, and therefore where selling policy matters:
#   MELON — no shop demands it at all (drain ~5/day).
#   WOOL  — one shop (2x), drain ~14/day, and the champion runs 8 sheep. Measured (E19): wool
#           ends slightly above I0 and is the reason a reserve returned to the champion.
# Everything else stays in scarcity: the town outpaces two farms.
SATURABLE = {"MELON", "WOOL"}
ALWAYS_SCARCE = [p for p in SHOP_DEMANDED if p not in SATURABLE]


def _run(params, seed=1, steps=718):
    a = [make_agent(params), make_agent(params)]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    lows = {}
    for _ in range(steps):
        sim.step([a[0](sim.observation(0)), a[1](sim.observation(1))])
        for k, v in sim.observation(0)["market"]["prices"].items():
            lows[k] = min(lows.get(k, 10**9), v)
    return sim, lows


@pytest.fixture(scope="module")
def champion_run():
    return _run(Params(**REGISTRY["champion"].params))


def test_most_products_stay_in_scarcity(champion_run):
    """Inventory below I0 means prices *above* base — the opposite of the glut the capacity tables
    model. Wool and melon are excluded by measurement, not by assumption: those two are the only
    markets two farms can outproduce (E19)."""
    sim, _ = champion_run
    inv = sim.observation(0)["market"]["inventory"]
    for p in ALWAYS_SCARCE:
        assert inv[p] <= 10_000, f"{p} ended in glut at {inv[p]:,} — the regime has changed"


def test_wool_is_saturable_which_is_why_a_reserve_returned(champion_run):
    """The champion carries a reserve again, and this is what it is for.

    E16 concluded reserves never help — measured before sheep existed. With 8 sheep per side, wool
    is the one shop-demanded market two farms can flood, so a reservation price stops us selling
    into our own crash. Asserted so that removing the reserve, or the sheep, is a visible decision.
    """
    from agent.params import Params

    champ = Params(**REGISTRY["champion"].params)
    if champ.sheep_target == 0:
        pytest.skip("champion no longer keeps sheep; wool saturation does not apply")
    assert champ.reserve_frac["WOOL"] > 0, (
        "champion runs sheep but sells wool with no reserve — E19 says that floods the market"
    )


def test_melon_is_the_one_floodable_product(champion_run):
    """No shop demands melon, so it is the only market two players can saturate."""
    from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS

    assert not any("MELON" in prods for prods in SHOPS.values())
    for p in SHOP_DEMANDED:
        assert any(p in prods for prods in SHOPS.values()), f"{p} should have shop demand"


def test_prices_never_collapse_during_the_season(champion_run):
    """Excludes the end-game dump.

    `sell_all_after_day` bypasses every reserve near the end, because unsold stock scores zero — so
    a transient crash on the final days is correct behaviour, not a regime change. The invariant
    worth holding is that prices stay healthy *while it still matters*.
    """
    from agent.engine import BASE_PRICE
    from agent.params import Params

    champ = Params(**REGISTRY["champion"].params)
    sim, _ = _run(champ, seed=1, steps=24 * champ.sell_all_after_day)
    prices = sim.observation(0)["market"]["prices"]
    for p in ALWAYS_SCARCE:
        assert prices[p] > 0.5 * BASE_PRICE[p], (
            f"{p} sat at {prices[p]} before the end-game dump — below half of base "
            f"{BASE_PRICE[p]}; the scarcity finding may have reversed"
        )


@pytest.mark.parametrize("reserve", [0.9, 1.3])
def test_holding_inventory_loses_when_the_reserve_actually_binds(reserve):
    """Cash compounds; inventory does not. The shed caps at 100 and unsold stock scores zero.

    Guarded against the trap that produced this test's first false alarm: a reserve threshold the
    price never reaches makes the "holding" agent *identical* to the champion, and the comparison
    becomes vacuous rather than reassuring. Identical money is the tell — assert on it explicitly
    instead of reading it as a reversal.
    """
    base = dict(REGISTRY["champion"].params)
    hold = dict(base)
    hold["reserve_frac"] = {**{k: 0.0 for k in hold["reserve_frac"]},
                            **{k: reserve for k in SHOP_DEMANDED}}
    sell_now, _ = _run(Params(**base), seed=4)
    holding, _ = _run(Params(**hold), seed=4)

    if sell_now.money(0) == holding.money(0):
        pytest.skip(f"reserve at {reserve}x base never binds — prices stay above it all season, "
                    "so this comparison is vacuous (itself a confirmation of the scarcity regime)")
    assert sell_now.money(0) > holding.money(0), (
        f"holding at {reserve}x base earned ${holding.money(0):,.0f} vs "
        f"${sell_now.money(0):,.0f} — E16 has reversed"
    )


def test_at_least_one_reserve_level_binds():
    """Stops the suite degrading into all-skips.

    If no reserve level changes behaviour, the tests above prove nothing and the claim
    "holding loses" would be unsupported rather than confirmed.
    """
    base = dict(REGISTRY["champion"].params)
    sell_now, _ = _run(Params(**base), seed=4)
    for reserve in (1.1, 1.3, 1.6):
        hold = dict(base)
        hold["reserve_frac"] = {**{k: 0.0 for k in hold["reserve_frac"]},
                                **{k: reserve for k in SHOP_DEMANDED}}
        holding, _ = _run(Params(**hold), seed=4)
        if holding.money(0) != sell_now.money(0):
            assert holding.money(0) < sell_now.money(0), (
                f"reserve {reserve}x base *helped* (${holding.money(0):,.0f} vs "
                f"${sell_now.money(0):,.0f}) — E16/D18 need revisiting"
            )
            return
    pytest.fail("no reserve level up to 1.6x base changed behaviour; the holding tests are vacuous")
