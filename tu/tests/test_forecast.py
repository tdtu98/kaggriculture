"""T1.3 — opponent supply forecasting.

`farms` is shared and crop maturity is deterministic, so the opponent's entire production schedule
is computable from public information (`PLAN.md` §2.5). Their *shed* is private, so this is
incoming supply only, never stock they may already hold.
"""

from __future__ import annotations

import pytest

import kagsim
from agent import Params, make_agent

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
MIX = {"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0, "STRAWBERRY": 0.0, "MELON": 0.0}


def sim_with(script_p1, steps, cfg=None):
    """Drive player 1 with a script; player 0 idles. Returns the sim after `steps`."""
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 0, "turnsPerDay": 4, **(cfg or {})})
    for i in range(steps):
        a1 = script_p1[i] if i < len(script_p1) else PASS
        sim.step([PASS, a1])
    return sim


def farmer(op, *args, market=None):
    return {"farmer": [op, *args], "hands": [], "market": market or []}


def test_sees_an_opponent_melon_before_it_ripens():
    """A melon planted on day 0 is visible, and its harvest date is deterministic."""
    script = [farmer("PASS", market=[["BUY_SEED", "MELON", 1]]),
              farmer("PLANT", "MELON"), farmer("WATER"), PASS]
    for _ in range(6):                       # water through day 6
        script += [farmer("WATER"), PASS, PASS, PASS]
    sim = sim_with(script, len(script))
    agent = make_agent(Params(forecast_weight=0.5, forecast_horizon=4))
    obs = sim.observation(0)

    # Day ~7, melon harvests at age 10 -> 3 days out, inside a 4-day horizon.
    supply = agent.opponent_supply(obs, 0)
    assert supply.get("MELON", 0) > 0, f"should see the incoming melon, got {supply}"

    # A 1-day horizon should not see it yet.
    near = make_agent(Params(forecast_weight=0.5, forecast_horizon=1))
    assert near.opponent_supply(obs, 0).get("MELON", 0) == 0


def test_ignores_our_own_farm():
    """The forecast is about the *opponent*; our own crops are not incoming competition."""
    script = [farmer("PASS", market=[["BUY_SEED", "MELON", 1]]),
              farmer("PLANT", "MELON"), farmer("WATER"), PASS]
    sim = sim_with(script, 4)
    agent = make_agent(Params(forecast_weight=0.5))
    # Player 1 planted; from player 1's own perspective there is no opponent supply.
    assert agent.opponent_supply(sim.observation(1), 1) == {}


def test_town_drain_grows_with_shops_but_no_longer_with_day():
    """Rewritten for 1.32.6, which deleted the town centre's day multiplier (E33).

    The previous version asserted `d1["WHEAT"] > d0["WHEAT"]` partly *because* "the centre scales
    after day 20". It does not any more: the centre buys 1 of each product per tick and the default
    interval moved 12 -> 24, so its contribution is a flat 1/day all season. Shops remain the only
    thing that makes demand grow -- and since they now draw with replacement, the only thing that
    makes it vary between games.
    """
    agent = make_agent(Params())
    early = {"town": {"unlocked_shops": []}, "day": 0}
    late = {"town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]}, "day": 25}
    d0, d1 = agent.town_drain_per_day(early), agent.town_drain_per_day(late)
    assert d0["WHEAT"] == 1, "town centre only: 1 per tick x 1 tick/day since 1.32.6"
    assert d1["WHEAT"] > d0["WHEAT"], "the bakery adds wheat demand"
    assert d1["WOOL"] > d0["WOOL"], "yarn store is single-product, so it consumes 2x"
    assert "FERTILIZER" not in d0, "the town centre excludes fertilizer"
    # The day term is gone: same shops, different day, identical drain.
    same_shops_later = {"town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]}, "day": 0}
    assert agent.town_drain_per_day(same_shops_later) == d1


def test_forecast_price_falls_when_a_wave_is_coming():
    agent = make_agent(Params(forecast_horizon=4))
    inv = {p: 10_000 for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                               "EGG", "MILK", "WOOL", "FERTILIZER"]}
    quiet = agent.forecast_price("MELON", inv, {}, {})
    flooded = agent.forecast_price("MELON", inv, {"MELON": 120}, {})
    assert flooded < quiet, "incoming supply must lower the expected price"
    assert quiet == pytest.approx(kagsim.market_price(4, 10_000))


def test_forecast_price_rises_when_the_town_will_drain():
    agent = make_agent(Params(forecast_horizon=4))
    inv = {p: 10_000 for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                               "EGG", "MILK", "WOOL", "FERTILIZER"]}
    assert agent.forecast_price("MELON", inv, {}, {"MELON": 8}) > agent.forecast_price(
        "MELON", inv, {}, {})


def test_weight_zero_bypasses_the_forecast_entirely():
    """With the forecast off, the opponent's farm must not be consulted at all.

    Written against the behaviour rather than against `Params()`, because the default weight is
    itself a tuned value that will keep moving (it has already gone 0.0 -> 0.6 -> 1.0).
    """
    agent = make_agent(Params(crop_mix={**MIX, "MELON": 1.0}, forecast_weight=0.0))

    def explode(*_args, **_kw):
        raise AssertionError("opponent_supply must not be called when forecast_weight == 0")

    agent.opponent_supply = explode
    sim = kagsim.Sim({"episodeSteps": 200, "seed": 3})
    for _ in range(198):
        sim.step([agent(sim.observation(0)), PASS])
    assert sim.money(0) > 0


def test_a_nonzero_weight_does_consult_the_opponent():
    """The mirror: otherwise the test above would pass on a forecast that never runs."""
    agent = make_agent(Params(crop_mix={**MIX, "MELON": 1.0}, forecast_weight=1.0))
    calls = []
    real = agent.opponent_supply
    agent.opponent_supply = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    sim = kagsim.Sim({"episodeSteps": 200, "seed": 3})
    for _ in range(198):
        sim.step([agent(sim.observation(0)), PASS])
    assert calls, "forecast path never ran"
