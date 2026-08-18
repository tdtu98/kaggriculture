"""I0 — the rules PLAN_v4 is built on, asserted by running the reference environment.

Every other test file asks "does kagsim match the reference?". This one asks a different and
narrower question: **are the facts the plan reasons from actually true?** PLAN_v4 budgets tiles,
hands and cohorts off specific arithmetic -- fertilized strawberry pays 8 units, a cared cow pays
3 milk, all four centre tiles reach the shed, a DROP at step s is sellable at step s -- and if any
of those is wrong the compiler will be optimising a fiction with every downstream measurement
looking healthy.

So each test here scripts the situation and reads the outcome. `run_scripted` drives the reference
env and kagsim in lockstep and raises on any divergence, so these double as parity tests, but the
assertions are about the *game*, not about kagsim.

Pinned to the source hash by `test_env_version.py`: when that guard fires, re-verify I1 parity
before trusting anything measured here.
"""

from __future__ import annotations

import math

import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_I0,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    market_price as ref_price,
)

import kagsim
from agent.relay import MARKET_PARAMS as RELAY_PARAMS, market_price as relay_price
from parity import PASS, PRODUCTS, run_scripted

TPD = 4  # short days keep the multi-week scripts fast; parity covers the default 24


def farmer(op, *args, market=None):
    return {"farmer": [op, *args], "hands": [], "market": market or []}


def day_of(*ops, tpd: int = TPD):
    return list(ops) + [PASS] * (tpd - len(ops))


def held(st, item, player=0):
    """Inventory + shed: the dusk refresh moves one into the other."""
    n = st["privates"][player]["shed"].get(item, 0)
    for inv in st["privates"][player]["inventories"]:
        n += dict(inv).get(item, 0)
    return n


def tile(st, x, y, player=0):
    return st["farms"][player]["tiles"][y * 10 + x]


# --------------------------------------------------------------- fertilizer arithmetic

def test_fertilize_covers_the_day_it_is_applied_and_the_two_after():
    """`fertilized_until_day == day + 2`, so one application spans three tick-days.

    This is the whole reason the plan fertilizes strawberry at ages 9 and 13 rather than at every
    tick: two applications cover ages 9-11 and 13-15, which is all four productions.
    """
    script = day_of(farmer("PASS", market=[["BUY_SEED", "WHEAT", 1],
                                           ["BUY_PRODUCT", "FERTILIZER", 1]]),
                    farmer("PLANT", "WHEAT"), farmer("WATER"))
    # PICKUP has to happen on the day the fertilizer is used: dusk empties every unit inventory
    # back into the shed, so nothing is carried overnight. That is a C1 logistics constraint, not
    # an incidental detail of this script.
    script += day_of(farmer("PICKUP", "FERTILIZER", 1), farmer("FERTILIZE"), farmer("WATER"))
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert tile(st, 4, 4)[0] == "PLANT"
    assert tile(st, 4, 4)[7] == 3, "fertilized_until_day must be day+2 (applied on day 1)"


def test_fertilized_strawberry_pays_eight_units_over_its_life():
    """PLAN_v4 1: 'fertilize at ages 9 and 13 -> all four yields doubled', 7.7 units/plant.

    STRAWBERRY is `first_yield_day 10, interval 2, max_yield 4, ongoing` -- productions land at
    the end of ages 9, 11, 13 and 15. Fertilizer doubles a production only if the tile is also
    watered that day, and `max_yield` caps the *tile*, so the units have to be harvested as they
    come; a plan that fertilizes correctly but harvests lazily throws half of this away.
    """
    buy = [["BUY_SEED", "STRAWBERRY", 1], ["BUY_PRODUCT", "FERTILIZER", 2]]
    script = day_of(farmer("PASS", market=buy), farmer("PLANT", "STRAWBERRY"), farmer("WATER"))
    for age in range(1, 17):
        ops = []
        if age in (9, 13):
            # fetched the same morning it is applied — inventories do not survive dusk
            ops += [farmer("PICKUP", "FERTILIZER", 1), farmer("FERTILIZE")]
        if age in (10, 12, 14, 16):
            ops.append(farmer("HARVEST"))
        ops.append(farmer("WATER"))
        script += day_of(*ops)
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert held(st, "STRAWBERRY") == 8, "2 units x 4 productions when fertilized at ages 9 and 13"


def test_unfertilized_strawberry_pays_half_that():
    """The counterfactual, so the test above is measuring fertilizer and not something else."""
    script = day_of(farmer("PASS", market=[["BUY_SEED", "STRAWBERRY", 1]]),
                    farmer("PLANT", "STRAWBERRY"), farmer("WATER"))
    for age in range(1, 17):
        ops = [farmer("HARVEST")] if age in (10, 12, 14, 16) else []
        script += day_of(*ops, farmer("WATER"))
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert held(st, "STRAWBERRY") == 4, "1 unit x 4 productions unfertilized"


def test_ongoing_yield_units_cap_at_max_yield():
    """Never harvesting is not equivalent to harvesting late: the tile stops accruing at 4."""
    script = day_of(farmer("PASS", market=[["BUY_SEED", "STRAWBERRY", 1]]),
                    farmer("PLANT", "STRAWBERRY"), farmer("WATER"))
    for _ in range(1, 17):
        script += day_of(farmer("WATER"))
    script += day_of(farmer("HARVEST"))
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert CROPS["STRAWBERRY"]["max_yield"] == 4
    assert held(st, "STRAWBERRY") == 4, "four productions accrued but the tile capped at 4"


def test_melon_max_yield_day_is_twelve_in_code():
    """The audit's correction: the README table says 10, the code says 12.

    The task generator has to read the code table -- a melon harvested on the README's schedule
    leaves two days of yield on the tile.
    """
    assert CROPS["MELON"]["max_yield_day"] == 12
    assert CROPS["MELON"]["first_yield_day"] == 10


def test_wheat_is_ripe_at_age_four_and_decays_from_dawn_of_age_five():
    """Deadline arithmetic for C1: wheat's harvest window closes hard."""
    script = day_of(farmer("PASS", market=[["BUY_SEED", "WHEAT", 2]]),
                    farmer("PLANT", "WHEAT"), farmer("WATER"))
    for _ in range(4):
        script += day_of(farmer("WATER"))
    at_four = run_scripted({"turnsPerDay": TPD}, script + day_of(farmer("HARVEST")))

    late = script + day_of(farmer("WATER")) + day_of(farmer("WATER")) + day_of(farmer("HARVEST"))
    at_six = run_scripted({"turnsPerDay": TPD}, late)

    assert held(at_four, "WHEAT") == 4, "window ages 2-4, +1 per watered day, starts at 1"
    assert held(at_six, "WHEAT") < held(at_four, "WHEAT"), "decay must bite after the window"


# --------------------------------------------------------------- animals

ATPD = 6  # the animal days need six turns: fetch wheat, walk, feed, care, harvest


def _pasture_script(feed_days, care_days, harvest_days, last_day: int):
    """Build a pasture at (4,3), place a cow on day 0, then work it through `last_day`.

    (5,4) and (4,5) are shed-access tiles but sit in LOCKED quadrants, so the pasture goes one
    tile north of the spawn instead — the same NW-cluster shape the plan's keepers use.
    """
    buy = [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 20]]
    script = day_of(farmer("PASS", market=buy), farmer("PICKUP", "COW", 1), farmer("NORTH"),
                    farmer("BUILD_PASTURE"), farmer("PLACE", "COW"), tpd=ATPD)
    for day in range(1, last_day + 1):
        ops = [farmer("PICKUP", "WHEAT", 1), farmer("NORTH")]
        if day in feed_days:
            ops.append(farmer("FEED"))
        if day in care_days:
            ops.append(farmer("CARE"))
        if day in harvest_days:
            ops.append(farmer("HARVEST"))
        script += day_of(*ops, tpd=ATPD)
    return script


def _milk_after(last_day, feed_days, care_days, harvest_days):
    st = run_scripted({"turnsPerDay": ATPD},
                      _pasture_script(feed_days, care_days, harvest_days, last_day))
    return held(st, "MILK")


def test_cow_fed_and_cared_daily_pays_three_milk_per_cycle():
    """PLAN_v4 1: 'cows cared daily (3 milk per cycle)'.

    A COW is `first_yield_day 8, interval 2`, placed on day 0 -> productions at the end of days 7,
    9, 11. The care bank adds +1 for every fed-*and*-cared day and is paid out on a production day
    only if the animal was fed that day, so a daily-cared cow yields 1 + 2 = 3 per cycle. The
    first production is bigger (the bank has been filling since placement) and clips at
    `max_held` = 6, which is exactly why the plan harvests every cycle instead of hoarding.

    Measured as a difference between two runs, so it is the *cycle* that is asserted, not a total.
    """
    all_days = set(range(1, 12))
    after_first = _milk_after(8, all_days, all_days, {8})
    after_second = _milk_after(10, all_days, all_days, {8, 10})

    assert ANIMALS["COW"] == {"cost": 400, "structure": "PASTURE", "first_yield_day": 8,
                              "interval": 2, "max_held": 6, "product": "MILK"}
    assert after_first == 6, "first production banks 7 cared days and clips at max_held"
    assert after_second - after_first == 3, "1 base + 2 banked care days per cycle"


def test_unfed_production_day_pays_one_and_clears_the_bank():
    """The bank cannot be back-filled: skip the feed on the production day and it is gone."""
    all_days = set(range(1, 12))
    fed_except_production = all_days - {9}
    after_first = _milk_after(8, all_days, all_days, {8})
    after_second = _milk_after(10, fed_except_production, all_days, {8, 10})

    assert after_second - after_first == 1, "unfed on the production day: base only, bank cleared"


def test_a_newly_placed_animal_survives_its_first_day_unfed_then_escapes():
    """`consecutive_unfed` starts at 0, so buy-and-place does not need same-day wheat — but the
    grace is exactly one day, which sets the deadline for the keeper's first feeding round."""
    none_: set[int] = set()
    script = _pasture_script(none_, none_, none_, 2)
    # one turn into day 1: the placement day's dusk has been taken, day 1's has not
    after_first_dusk = run_scripted({"turnsPerDay": ATPD}, script, steps=ATPD + 1)
    after_second_dusk = run_scripted({"turnsPerDay": ATPD}, script)

    assert after_first_dusk["farms"][0]["tiles"][3 * 10 + 4][0] == "PASTURE_A", \
        "survives the day it was placed with no wheat at all"
    assert tile(after_first_dusk, 4, 3)[4] == 1, "…carrying consecutive_unfed = 1"
    assert tile(after_second_dusk, 4, 3) == ["PASTURE"], \
        "escapes on the second unfed dusk, leaving the structure behind"


# --------------------------------------------------------------- units, shed, day cycle

def test_all_four_centre_tiles_reach_the_shed_while_the_land_is_locked():
    """PLAN_v4 2.2 / F1: SHED_ADJ is four tiles, not one.

    Shed operations resolve ahead of the LOCKED guard, so (4,4), (5,4), (4,5) and (5,5) all take
    PICKUP/DROP from turn 0 -- which is what lets several units load in the same turn instead of
    queueing on the spawn tile.
    """
    moves = {(4, 4): [], (5, 4): [farmer("EAST")], (4, 5): [farmer("SOUTH")],
             (5, 5): [farmer("EAST"), farmer("SOUTH")]}
    for (x, y), walk in moves.items():
        # one long day: the dawn respawn would undo the walk, and dusk would undo the PICKUP
        script = [farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 1]]), *walk,
                  farmer("PICKUP", "WHEAT", 1)]
        st = run_scripted({"turnsPerDay": 12}, script)
        inv = dict(st["privates"][0]["inventories"][0])
        assert st["farms"][0]["farmer"] == [x, y]
        assert inv.get("WHEAT") == 1, f"PICKUP failed at centre tile {(x, y)}"
        if (x, y) != (4, 4):
            assert tile(st, x, y) == ["LOCKED"], "…and it worked while the quadrant was LOCKED"


def test_a_drop_at_step_s_is_sellable_at_step_s():
    """Unit actions resolve before market orders inside one step (`:935-941`).

    C4's market rules depend on this: the day's last DROP can be sold the same turn instead of
    waiting for dawn, which is what keeps the shed under its cap at dusk.
    """
    script = day_of(farmer("PASS", market=[["BUY_SEED", "WHEAT", 1]]),
                    farmer("PLANT", "WHEAT"), farmer("WATER"))
    for _ in range(4):
        script += day_of(farmer("WATER"))
    script += day_of(farmer("HARVEST"),
                     farmer("DROP", "WHEAT", 4, market=[["SELL", "WHEAT", 4]]))
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert held(st, "WHEAT") == 0, "the DROP landed and the same step's SELL took it"
    assert st["market_inv"]["WHEAT"] > MARKET_I0, "the sale reached the market this step"


def test_hands_spawn_on_the_least_occupied_centre_tile_and_act_the_next_step():
    """HIRE resolves inside step s's market phase, so the hand exists at the end of step s and
    takes its first action at s+1. It spawns on the least-occupied shed tile in NWSE order, which
    is why the router can assume hands start spread across the centre rather than stacked."""
    # HIRE is atomic per order slot — the count is ignored, so two hands need two orders.
    script = [{"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"]]},
              {"farmer": ["PASS"], "hands": [["EAST"], ["SOUTH"]], "market": []}]
    st = run_scripted({"turnsPerDay": TPD}, script)

    hands = st["farms"][0]["hands"]
    assert len(hands) == 2, "both hands exist"
    # they moved on step 1, so their spawn tiles were one step back the way they walked
    spawned = [[hands[0][0] - 1, hands[0][1]], [hands[1][0], hands[1][1] - 1]]
    centre = [[4, 4], [5, 4], [4, 5], [5, 5]]
    assert all(p in centre for p in spawned), f"hands spawned off the centre: {spawned}"
    assert spawned[0] != [4, 4] and spawned[0] != spawned[1], "least-occupied, not stacked"


def test_the_farmer_respawns_at_the_centre_every_dawn():
    """Routing assumes every unit starts the day at the shed; the farmer is teleported back."""
    script = day_of(farmer("NORTH"), farmer("NORTH"), farmer("WEST"), farmer("WEST"))
    script += day_of(farmer("PASS"))
    st = run_scripted({"turnsPerDay": TPD}, script)
    assert st["farms"][0]["farmer"] == [4, 4], "dawn respawn at the centre"


def test_dusk_moves_inventories_to_the_shed_and_discards_the_overflow():
    """The shed is capped and the excess is *lost*, not left in the unit's hands.

    Run at `shedCapacity: 10` so the discard is reachable in a few turns; the rule is the same one
    that costs a real season money at 100, and it is why C4 sells during the day and why C1 emits
    DROP tasks before dusk rather than after.
    """
    cfg = {"turnsPerDay": TPD, "shedCapacity": 10}
    script = day_of(farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 10]]),
                    farmer("PICKUP", "WHEAT", 10),
                    farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 10]]))
    script += day_of(farmer("PASS"))
    st = run_scripted(cfg, script)

    assert st["privates"][0]["shed"].get("WHEAT") == 10, "shed full at capacity"
    assert dict(st["privates"][0]["inventories"][0]) == {}, "dusk empties the unit"
    assert held(st, "WHEAT") == 10, "the 10 the farmer was holding at dusk were discarded"


# --------------------------------------------------------------- town and market

def test_shops_draw_with_replacement_up_to_eight_instances():
    """1.32.6 change the plan leans on: a product can have zero shops or four.

    `MAX_SHOP_INSTANCES` draws are uniform *with replacement*, so wool has no buyer in a third of
    games and three Pet Cafes (carrot) is a normal draw, not a freak one. Branch points (O1) and
    the scarcity hunter (2.7) both exist because of this.
    """
    assert MAX_SHOP_INSTANCES == 8
    seen_duplicate = False
    for seed in range(12):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.reset(num_agents=2)
        for _ in range(24 * 24):
            env.step([PASS, PASS])
        shops = list(env.state[0].observation["town"]["unlocked_shops"])
        assert len(shops) == MAX_SHOP_INSTANCES, "all eight draws land by day 24"
        if len(set(shops)) < len(shops):
            seen_duplicate = True
    assert seen_duplicate, "with replacement means duplicates must occur across seeds"


def test_town_centre_sell_interval_default_is_24():
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0})
    assert env.configuration["townCenterSellInterval"] == 24


@pytest.mark.parametrize("product", PRODUCTS)
def test_every_vendored_price_table_matches_the_reference(product):
    """The audit's headline correction, pinned in both directions.

    Boatlee ships a stale `_MARKET_PARAMS`, and so did we: carrot/tomato/egg went `hinge` below I0
    in 1.32.7 and anything still pricing them `log`/`linear` under-values scarcity by up to 5x --
    exactly when selling is most valuable. Checked at the offsets the hinge is defined by, so a
    table that is right at the base price and wrong past the knee still fails.
    """
    i = PRODUCTS.index(product)
    T = MARKET_PARAMS[product]["T"]
    offsets = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    points = {MARKET_I0 + int(sign * k * T) for k in offsets for sign in (-1, 1)}
    for inv in sorted(points):
        want = ref_price(product, inv)
        assert kagsim.market_price(i, inv) == want, f"kagsim {product} @ {inv}"
        assert relay_price(product, inv) == want, f"agent.relay {product} @ {inv}"


def test_the_hinge_products_are_the_three_the_plan_targets():
    """PLAN_v4 2.7 names CARROT, TOMATO and EGG. If Kaggle moves a fourth product onto the hinge,
    the scarcity hunter should be told about it rather than quietly missing it."""
    hinged = {p for p, v in MARKET_PARAMS.items() if v["below_func"] == "hinge"}
    assert hinged == {"CARROT", "TOMATO", "EGG"}
    assert RELAY_PARAMS["CARROT"][3] == "hinge" and RELAY_PARAMS["CARROT"][4] == 1.0


def test_the_hinge_runs_away_past_T():
    """The magnitude the plan's economics rest on: ~5x base at -2T, ~9x at -2.5T for tomato."""
    T = MARKET_PARAMS["TOMATO"]["T"]
    at_T = ref_price("TOMATO", MARKET_I0 - T)
    at_2T = ref_price("TOMATO", MARKET_I0 - 2 * T)
    assert at_T == 84 and at_2T == 300, (at_T, at_2T)
    assert ref_price("CARROT", MARKET_I0 - 2 * 450) == 385
    assert ref_price("EGG", MARKET_I0 - 2 * 332) == 250


def test_hinge_is_continuous_and_equals_one_at_T():
    """`f(T) == 1` by construction, which is what keeps `target` comparable across shapes."""
    for product in ("CARROT", "TOMATO", "EGG"):
        p = MARKET_PARAMS[product]
        T, base, target = p["T"], p["base"], p["below_target"]
        assert ref_price(product, MARKET_I0 - T) == max(1, round(base + target * base))
        below = ref_price(product, MARKET_I0 - T + 1)
        above = ref_price(product, MARKET_I0 - T - 1)
        assert below <= ref_price(product, MARKET_I0 - T) <= above, "monotone through the knee"


def test_empty_structures_never_spawn_weeds():
    """Weeds only land on `None` tiles, so a built pasture is permanently weed-free -- the plan's
    pasture cluster costs nothing in DIG work, while a dug-out plant tile does."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3,
                                               "weedSpawnChance": 1.0})
    env.reset(num_agents=2)
    build = {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
    env.step([build, PASS])
    for _ in range(24 * 5):
        env.step([PASS, PASS])
    tiles = env.state[0].observation["farms"][0]["tiles"]
    assert tiles[4][4]["kind"] == "PASTURE", "the structure survives a 100% weed chance"
    assert any(t == {"kind": "WEED"} or (isinstance(t, dict) and t.get("kind") == "WEED")
               for row in tiles for t in row), "…while bare ground is covered in them"
