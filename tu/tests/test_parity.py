"""T0.4 — parity suite.

Layer A isolates the deterministic rules (weeds and shop unlocks disabled); layer C runs the full
default config. Scenario tests target rules the fuzzer reaches only rarely, and assert on the
resulting state as well as on parity.
"""

from __future__ import annotations

import pytest

from parity import LAYER_A, PASS, run_parity, run_scripted

# --------------------------------------------------------------------------- fuzz parity


@pytest.mark.parametrize("env_seed", range(4))
@pytest.mark.parametrize("fuzz_seed", range(3))
def test_layer_a_deterministic_core(env_seed, fuzz_seed):
    run_parity(env_seed, fuzz_seed, steps=250, config=LAYER_A)


@pytest.mark.parametrize("env_seed", range(3))
@pytest.mark.parametrize("fuzz_seed", range(2))
def test_layer_c_full_config(env_seed, fuzz_seed):
    run_parity(env_seed, fuzz_seed, steps=400, config={})


def test_full_length_episode():
    """720 steps — the real season length, including the terminal step."""
    run_parity(env_seed=7, fuzz_seed=7, steps=718, config={})


# --------------------------------------------------------------------------- scenarios

TPD = 4  # short days keep multi-day scenarios fast; parity covers non-default turnsPerDay


def farmer(op, *args, market=None):
    return {"farmer": [op, *args], "hands": [], "market": market or []}


def test_wheat_yield_curve():
    """WHEAT: bonus window is ages 2..4, +1 per watered day on top of the starting unit."""
    script = [
        farmer("PASS", market=[["BUY_SEED", "WHEAT", 1]]),  # step 0, day 0
        farmer("PLANT", "WHEAT"),                            # step 1
        farmer("WATER"),                                     # step 2 — age 0, survival only
        PASS, PASS, PASS, PASS, PASS,                        # day 1 unwatered (survives at 1)
        farmer("WATER"),                                     # step 8  — age 2 -> yield 2
        PASS, PASS, PASS,
        farmer("WATER"),                                     # step 12 — age 3 -> yield 3
        PASS, PASS, PASS,
        farmer("WATER"),                                     # step 16 — age 4 -> yield 4
        farmer("HARVEST"),                                   # step 17
    ]
    st = run_scripted({"turnsPerDay": TPD}, script)
    inv = st["privates"][0]["inventories"][0]
    assert inv == [["WHEAT", 4]], inv
    assert st["farms"][0]["tiles"][44] == ["EMPTY"], "one-time crops vacate the tile on harvest"


def test_planting_day_counts_as_unwatered():
    """A seed planted and never watered weeds that same night (`consecutive_unwatered` starts 1)."""
    script = [
        farmer("PASS", market=[["BUY_SEED", "WHEAT", 1]]),
        farmer("PLANT", "WHEAT"),
        PASS, PASS,  # end of day 0 arrives with watered_today False -> 2 -> weed
    ]
    st = run_scripted({"turnsPerDay": TPD}, script)
    assert st["farms"][0]["tiles"][44] == ["WEED"]


def test_atomic_plant_blocks_all_requests_for_that_crop():
    """One seed, two units both requesting PLANT WHEAT -> neither plants, seed not consumed."""
    script = [
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1], ["HIRE"]]},
        {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": []},
    ]
    st = run_scripted({"turnsPerDay": TPD}, script)
    assert st["privates"][0]["seeds"] == {"WHEAT": 1}, "seed must not be consumed"
    tiles = st["farms"][0]["tiles"]
    assert not any(t[0] == "PLANT" for t in tiles), "no tile may be planted"
    assert tiles[44] == ["EMPTY"], "the farmer's own tile stays empty"
    # The first hire of the day spawns on (5,4), which is in the still-locked NE quadrant
    # (`:159`) — spawn placement deliberately ignores locking.
    assert tiles[45] == ["LOCKED"]


def test_buy_then_sell_round_trip_nets_zero():
    """BUY_PRODUCT quotes at post-buy inventory, so the round trip is exactly free."""
    st = run_scripted(
        {"turnsPerDay": TPD},
        [farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 1], ["SELL", "WHEAT", 1]])],
    )
    assert st["farms"][0]["money"] == 3000.0
    # Market inventory is back to 10000 from the trade, then the town centre consumes 1 of every
    # non-fertilizer product on step 0 (`step % townCenterSellInterval == 0`).
    assert st["market_inv"]["WHEAT"] == 9999
    assert st["privates"][0]["shed"] == {}


def test_hire_cost_is_fibonacci():
    """Eight hires in one turn cost 1+1+2+3+5+8+13+21 = 54."""
    st = run_scripted({"turnsPerDay": TPD}, [farmer("PASS", market=[["HIRE"]] * 8)])
    assert st["farms"][0]["money"] == 3000.0 - 54
    assert st["farms"][0]["hires_today"] == 8
    assert len(st["farms"][0]["hands"]) == 8


def test_buy_land_prices_and_exhaustion():
    """$1k / $2k / $4k, then further BUY_LAND is a no-op."""
    st = run_scripted(
        {"turnsPerDay": TPD, "startingMoney": 10000},
        [farmer("PASS", market=[["BUY_LAND"]] * 4)],
    )
    assert st["farms"][0]["money"] == 10000 - 7000
    assert st["farms"][0]["unlocked"] == ["NW", "NE", "SW", "SE"]
    assert not any(t == ["LOCKED"] for t in st["farms"][0]["tiles"])


def test_animal_survives_alternate_day_feeding():
    """Death is at `consecutive_unfed >= 2`, so feeding every other day keeps the animal alive.

    Two details make this fiddly, and both are real constraints on any feeding strategy:
    a newly placed animal starts at `consecutive_unfed = 0` so it survives day 0 unfed, but it
    then *must* be fed on day 1; and the end-of-day drop empties unit inventories, so the wheat
    has to be picked up again every single day before FEED can consume it.
    """
    script = [
        farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 20]]),
        farmer("PICKUP", "GOOSE", 1),
        farmer("PLACE", "GOOSE"),
        farmer("PASS"),
    ]
    # The farmer respawns at (4,4) each day, which is both the coop tile and shed-adjacent.
    for day in range(1, 10):
        if day % 2 == 1:
            script += [farmer("PICKUP", "WHEAT", 1), farmer("FEED"), PASS, PASS]
        else:
            script += [PASS] * 4
    st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000}, script)
    tile = st["farms"][0]["tiles"][44]
    assert tile[0] == "COOP_A", f"goose should have survived, got {tile}"
    assert tile[1] == "GOOSE"


def test_animal_dies_if_first_feed_slips_to_day_two():
    """The mirror of the above: unfed on days 0 and 1 is already fatal."""
    script = [
        farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 20]]),
        farmer("PICKUP", "GOOSE", 1),
        farmer("PLACE", "GOOSE"),
        farmer("PASS"),
    ] + [PASS] * 4 + [farmer("PICKUP", "WHEAT", 1), farmer("FEED"), PASS, PASS]
    st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000}, script)
    assert st["farms"][0]["tiles"][44] == ["COOP"], "escaped at the end of day 1"


def test_animal_escapes_after_two_consecutive_unfed_days():
    script = [
        farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1]]),
        farmer("PICKUP", "GOOSE", 1),
        farmer("PLACE", "GOOSE"),
        farmer("PASS"),
    ] + [PASS] * 12  # three full days with no feeding
    st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000}, script)
    assert st["farms"][0]["tiles"][44] == ["COOP"], "structure remains, animal is gone"


def test_shed_overflow_is_discarded():
    """The shed caps at `shedCapacity`; anything over it at the end-of-day drop is destroyed."""
    script = [
        farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 8]]),
        farmer("PICKUP", "WHEAT", 8),
        PASS, PASS,  # end of day drops the 8 back, but the shed only has room for 5
    ]
    st = run_scripted({"turnsPerDay": TPD, "shedCapacity": 5}, script)
    assert st["privates"][0]["shed"] == {"WHEAT": 5}
    assert st["privates"][0]["inventories"] == [[]], "inventory is cleared regardless"


def test_dig_cannot_remove_an_occupied_structure():
    script = [
        farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1]]),
        farmer("PICKUP", "GOOSE", 1),
        farmer("PLACE", "GOOSE"),
        farmer("DIG"),
    ]
    st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000}, script)
    assert st["farms"][0]["tiles"][44][0] == "COOP_A", "DIG must no-op on a placed animal"


def test_melon_price_crashes_and_floors():
    """Both players quote against the same pre-commit inventory; the $1 floor stops adding supply."""
    import kagsim

    assert kagsim.market_price(4, 10000) == 250
    assert kagsim.market_price(4, 10000 + 158) == 1
    assert kagsim.market_price(5, 10000 + 3000) > 30  # EGG is effectively a bottomless sink


def test_runs_through_termination():
    """The DONE/reward path (`:937`) and the post-done early return (`:880`).

    `run_parity`'s normal loop stops one call short of `episodeSteps - 2`, so termination — the
    code that produces the score we train on — was previously never exercised.
    """
    run_parity(env_seed=3, fuzz_seed=11, steps=120, config={}, to_termination=True)


def test_terminal_reward_matches_reference():
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

    import kagsim

    cfg = {"episodeSteps": 100, "seed": 5}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)
    sim = kagsim.Sim(dict(cfg))
    for _ in range(cfg["episodeSteps"] - 1):
        a = [starter_agent(env.state[p].observation) for p in range(2)]
        env.step(a)
        sim.step(a)

    assert [s["status"] for s in env.state] == ["DONE", "DONE"]
    assert sim.done is True
    for p in range(2):
        assert env.state[p]["reward"] == sim.money(p), f"player {p} reward"
