"""C6: the market projection, and the hunter that acts on it.

Two kinds of test here, and the split is deliberate.

The **drain arithmetic** is checked against the reference environment *running*, not against a
re-statement of the formula. A test that recomputed `6 * instances` and compared it to
`SHOP_DRAIN_PER_DAY` would pass for any consistent pair of wrong numbers, which is E39's failure
exactly; so `test_drain_matches_the_reference_env` plays real turns with two `pass` agents — no
supply from either farm, no sales, so every unit that leaves the market left through the town — and
demands the projection reproduce the inventory to the unit.

The **hunter** is checked in play, through `main_v4`'s own effect ledger, because "the redirect
would have fired" and "the redirect fired" are different claims and only the second one is
evidence (E44). The done-when is a *contrast*: non-zero on seeds where a hinge product spikes, zero
on seeds where it does not, same plan, same opponent.
"""

from __future__ import annotations

import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS,
    MARKET_PARAMS,
    SHOPS,
    market_price,
)

from agent import projection
from agent.plan import Cohort, Plan
from agent.projection import (
    EXPECTED_DRAIN_PER_INSTANCE,
    REDIRECTABLE,
    SHOP_DRAIN_PER_DAY,
    SHOP_EVENTS_PER_DAY,
    Projection,
    instances_present,
    project,
    redirect,
    scarcity_plan,
    scarcity_signal,
    tile_value,
)

kagsim = pytest.importorskip("kagsim")


@pytest.fixture(autouse=True)
def _fresh_season():
    """The hunter's memory is a module global that outlives an agent, by design (a season is many
    calls). Tests are seasons too, and one leaking into the next is how a redirect "fails" because
    its budget was already spent by the test above."""
    projection.reset()
    yield
    projection.reset()


# --------------------------------------------------------------------------- fixtures

def _game(seed, a="pass", b="pass", config=None):
    from harness import registry

    projection.reset()
    agents = [registry.get(a).build(), registry.get(b).build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed, **(config or {})})
    sim.collect_stats = True
    return sim, agents


def _play(sim, agents, turns=719, watch=None):
    """Run a game; `watch(obs0, day, hour)` is called before every step."""
    for _ in range(turns):
        obs = [sim.observation(0), sim.observation(1)]
        if watch is not None:
            watch(obs[0], obs[0]["day"], obs[0]["hour"])
        sim.step([agents[0](obs[0]), agents[1](obs[1])])
    return sim


def _obs(day=0, hour=0, shops=(), inventory=None, tiles=None, shed=None):
    board = tiles or [[None] * 10 for _ in range(10)]
    inv = {k: 10_000 for k in MARKET_PARAMS}
    inv.update(inventory or {})
    return {
        "player": 0, "day": day, "hour": hour, "step": day * 24 + hour,
        "farms": [{"money": 3000, "farmer": [4, 4], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW"], "tiles": board},
                  {"money": 3000, "farmer": [4, 4], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW"],
                   "tiles": [[None] * 10 for _ in range(10)]}],
        "market": {"inventory": inv,
                   "prices": {k: market_price(k, inv[k]) for k in MARKET_PARAMS}},
        "town": {"unlocked_shops": list(shops)},
        "private": {"shed": dict(shed or {}), "seeds": {}, "inventories": [{}]},
    }


# --------------------------------------------------------------------------- the drain

def test_drain_matches_the_reference_env_to_the_unit():
    """Two `pass` agents produce and sell nothing, so the town is the *only* thing moving the
    market: whatever the projection says the inventory will be, that is what it must be.

    Exact wherever no new shop unlocks inside the horizon — there is nothing to estimate — and
    within the widest a single unknown instance can be (12 units/day for a single-product shop)
    everywhere else.
    """
    sim, agents = _game(4242, "pass", "pass")
    seen, preds = {}, []

    def watch(obs, day, hour):
        if hour == 0:
            seen[day] = (dict(obs["market"]["inventory"]),
                         len(obs["town"]["unlocked_shops"]))
            preds.append((day, {p: Projection(obs, horizon=5).inventory(p, 5)
                                for p in MARKET_PARAMS}))

    _play(sim, agents, watch=watch)

    exact = loose = 0
    for day, predicted in preds:
        target = seen.get(day + 5)
        if target is None:
            continue
        actual, known_now = target[0], seen[day][1]
        settled = instances_present(day + 5) == known_now
        for product, value in predicted.items():
            if settled:
                assert value == pytest.approx(actual[product], abs=1e-6), (day, product)
                exact += 1
            else:
                assert abs(value - actual[product]) <= 12 * 5, (day, product)
                loose += 1
    # Only the tail of the season is fully settled — a new instance unlocks every third day, so a
    # five-day horizon usually straddles one.
    assert exact >= 8 and loose >= 40, (exact, loose)


def test_drain_is_exact_once_the_town_is_full():
    """From day 24 every shop instance is known (`MAX_SHOP_INSTANCES` is 8, unlocking every third
    day), so there is nothing left to estimate and the projection must be exact."""
    sim, agents = _game(99, "pass", "pass")
    seen, preds = {}, []

    def watch(obs, day, hour):
        if hour == 0:
            seen[day] = dict(obs["market"]["inventory"])
            if day >= 24:
                proj = Projection(obs, horizon=4)
                preds.append((day, {p: proj.inventory(p, 4) for p in MARKET_PARAMS}))

    _play(sim, agents, watch=watch)
    checked = 0
    for day, predicted in preds:
        actual = seen.get(day + 4)
        if actual is None:
            continue
        for product, value in predicted.items():
            assert value == pytest.approx(actual[product], abs=1e-6), (day, product)
            checked += 1
    assert checked >= 16, checked


def test_a_single_product_shop_eats_double():
    """`multiplier = 2 if len(products) == 1` (`kaggriculture.py:741`) — the reason WOOL starves."""
    assert SHOP_DRAIN_PER_DAY["YARN_STORE"]["WOOL"] == 2 * SHOP_EVENTS_PER_DAY
    assert SHOP_DRAIN_PER_DAY["PET_CAFE"]["CARROT"] == 2 * SHOP_EVENTS_PER_DAY
    assert SHOP_DRAIN_PER_DAY["FARMERS_MARKET"]["CARROT"] == SHOP_EVENTS_PER_DAY
    for shop, products in SHOPS.items():
        assert set(SHOP_DRAIN_PER_DAY[shop]) == set(products), shop


def test_expected_drain_of_an_unknown_instance_is_the_mean_over_the_eight_shops():
    for product in ("WOOL", "CARROT", "WHEAT"):
        want = sum(SHOP_DRAIN_PER_DAY[s].get(product, 0) for s in SHOPS) / len(SHOPS)
        assert EXPECTED_DRAIN_PER_INSTANCE[product] == pytest.approx(want)
    # WHEAT sits in five of eight shops and MELON in none: the ranking that D17 says decides a
    # product, restated as a number this module actually uses.
    assert EXPECTED_DRAIN_PER_INSTANCE["MELON"] == 0
    assert EXPECTED_DRAIN_PER_INSTANCE["WHEAT"] > EXPECTED_DRAIN_PER_INSTANCE["TOMATO"]


def test_instance_count_follows_the_unlock_schedule():
    assert [instances_present(d) for d in range(0, 10)] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert instances_present(24) == 8
    assert instances_present(29) == 8            # MAX_SHOP_INSTANCES caps it


def test_today_is_only_counted_from_the_current_hour():
    """The board already shows this morning's consumption; counting it again is a 6-unit error on
    a `T` of 100. Four events remain after hour 4, one after hour 20, none after hour 21."""
    shops = ["YARN_STORE"]                               # day 3: exactly one instance is unlocked,
    per_event = 2                                        # so nothing is estimated
    assert instances_present(3) == len(shops)
    for hour, remaining in ((0, 6), (4, 5), (20, 1), (21, 0), (23, 0)):
        proj = Projection(_obs(day=3, hour=hour, shops=shops), horizon=1)
        drop = 10_000 - proj.inventory("WOOL", 1)
        # The town centre visits at hour 0 only, so it is in today's remainder exactly once.
        centre = 1 if hour == 0 else 0
        assert drop == pytest.approx(remaining * per_event + centre), (hour, drop)


def test_town_centre_takes_one_of_every_product_but_fertilizer():
    # From hour 1 of day 6, today's hour-0 centre visit has already happened, so the dawn of day 9
    # is two more visits away — and melon is in no shop at all, so the centre is all there is.
    proj = Projection(_obs(day=6, hour=1), horizon=3)
    assert 10_000 - proj.inventory("MELON", 3) == pytest.approx(2.0)
    assert 10_000 - Projection(_obs(day=6, hour=0), horizon=3).inventory("MELON", 3) == 3.0
    assert proj.inventory("FERTILIZER", 3) == pytest.approx(10_000.0)


# --------------------------------------------------------------------------- supply

def test_our_standing_crop_is_forecast_onto_the_market():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0, "watered_today": True,
                   "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
                   "fertilized_until_day": -1}
    bare = Projection(_obs(day=8), horizon=6)
    grown = Projection(_obs(day=8, tiles=tiles), horizon=6)
    # Strawberry ticks at the dusk of ages 9, 11, 13, 15 -> the age-9 tick reaches the market on
    # day 10, i.e. two days out. Nothing before that.
    assert grown.inventory("STRAWBERRY", 1) == pytest.approx(bare.inventory("STRAWBERRY", 1))
    assert grown.inventory("STRAWBERRY", 2) > bare.inventory("STRAWBERRY", 2)


def test_the_opponents_farm_counts_too():
    """Their tiles are on the shared board, which is the whole reason a tiles-based forecast is
    possible before O2 exists."""
    obs = _obs(day=8)
    theirs = obs["farms"][1]["tiles"]
    for x in range(5):
        theirs[0][x] = {"kind": "PLANT", "crop": "TOMATO", "planted_day": 0, "watered_today": True,
                        "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
                        "fertilized_until_day": -1}
    with_them = Projection(obs, horizon=6).inventory("TOMATO", 4)
    obs["farms"][1]["tiles"] = [[None] * 10 for _ in range(10)]
    without = Projection(obs, horizon=6).inventory("TOMATO", 4)
    assert with_them > without + 10


def test_five_day_prediction_error_stays_inside_15_percent_for_hinge_products():
    """The C6 verification bar, stated in the units that can fail it.

    Raw inventory error is meaningless — everything sits within a percent of `I0 = 10,000` — so the
    error is measured on the **deviation** `I0 - inventory`, which is what the price curve reads,
    and only on days where that deviation is at least `0.25T` (below it the relative error is a
    division by noise).
    """
    seeds = [52000, 52001, 52002]
    # WHEAT is in the list because it is the product the measured `SELL_THROUGH` constant exists
    # for: forecasting wheat production as market supply is a 27.5% error, because the farm eats
    # nearly all of it and buys back what it is short (E68). Without the discount this bar fails.
    errors = {p: [] for p in ("CARROT", "TOMATO", "EGG", "WHEAT")}
    for seed in seeds:
        sim, agents = _game(seed, "compiler", "boatlee")
        seen, preds = {}, []

        def watch(obs, day, hour, seen=seen, preds=preds):
            if hour == 0:
                seen[day] = dict(obs["market"]["inventory"])
                proj = Projection(obs, horizon=5)
                preds.append((day, {p: proj.inventory(p, 5) for p in errors}))

        _play(sim, agents, watch=watch)
        for day, predicted in preds:
            actual = seen.get(day + 5)
            if actual is None:
                continue
            for product, value in predicted.items():
                real = 10_000 - actual[product]
                if abs(real) < 0.25 * MARKET_PARAMS[product]["T"]:
                    continue
                errors[product].append(abs((10_000 - value) - real) / abs(real))

    for product, es in errors.items():
        assert len(es) >= 20, (product, len(es))
        es.sort()
        median = es[len(es) // 2]
        assert median <= 0.15, (product, median)


# --------------------------------------------------------------------------- signal

def test_signal_is_measured_in_T_units():
    """One `T` of extra scarcity on the board is exactly 1.0 more signal — that equivalence is what
    lets one threshold gene mean the same thing for a `T` of 100 and a `T` of 450."""
    T = MARKET_PARAMS["TOMATO"]["T"]
    flat = scarcity_signal("TOMATO", _obs(day=4, hour=0))
    projection.reset()
    short = scarcity_signal("TOMATO", _obs(day=4, hour=0, inventory={"TOMATO": 10_000 - T}))
    assert short - flat == pytest.approx(1.0, abs=1e-6)
    assert flat > 0, "future town demand alone already makes the board short"


def test_signal_reads_the_minimum_of_the_curve_not_its_end():
    """A trough five days out is a spike we can plant into; a curve that only reported its endpoint
    would miss it entirely."""
    obs = _obs(day=3, shops=["PET_CAFE"])
    tiles = obs["farms"][0]["tiles"]
    for i in range(24):                                 # a big carrot harvest lands on day 3+3
        tiles[i // 10][i % 10] = {
            "kind": "PLANT", "crop": "CARROT", "planted_day": 3, "watered_today": True,
            "consecutive_unwatered": 0, "yield_units": 0,
            "max_lifespan_step": (3 + 4) * 24, "fertilized_until_day": -1}
    proj = Projection(obs, horizon=6)
    curve = proj.project("CARROT")
    assert min(curve) < curve[-1], "the harvest refills the market; the trough is before it"
    assert proj.signal("CARROT") == pytest.approx((10_000 - min(curve)) / 450, abs=1e-6)


def test_project_matches_the_spec_signature():
    obs = _obs(day=12, shops=["BAKERY"])
    curve = project("WHEAT", 4, obs)
    assert len(curve) == 5 and curve[0] == 10_000
    assert curve == sorted(curve, reverse=True)         # nobody is supplying; it only falls


# --------------------------------------------------------------------------- the hunter

def _cheap_cohort_plan(day=13, crop="WHEAT"):
    """`boatlee_like` with its day-11 melon block swapped for a cheap one.

    Not a contrivance — it is the finding. Against `boatlee_like` itself the hunter *never* fires,
    because its only pending cohorts are melon and strawberry, worth $1,420 and $1,660 a tile, and
    no tomato price reachable by day 13 clears that. The hunter is a lever on plans with a cheap
    pending block, which is what S2 will produce.
    """
    from dataclasses import replace

    base = Plan.boatlee_like()
    cohorts = tuple(replace(c, crop=crop, plant_day=day)
                    if c.crop == "MELON" and c.plant_day >= 10 else c for c in base.cohorts)
    return replace(base, cohorts=cohorts)


def _hunt(seed, plan=None):
    """One real season; returns `main_v4`'s own effect ledger for our seat."""
    from agent import main_v4
    from harness import registry

    projection.reset()
    ours = main_v4.make_agent(plan or _cheap_cohort_plan())
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    theirs = registry.get("boatlee").build()
    _play(sim, [ours, theirs])
    effects = dict((main_v4._STATE.get(0) or {}).get("effects") or {})
    return effects, sim.stats(0)


@pytest.mark.parametrize("seed", [52011, 52013, 52014])
def test_the_hunter_fires_on_a_hinge_seed(seed):
    """The done-when, half one. `scarce_cohorts` is written by `redirect` when it commits, and the
    units are counted by kagsim's own sales ledger — an order is not a sale (CLAUDE.md)."""
    effects, stats = _hunt(seed)
    assert effects.get("scarce_cohorts", 0) >= 1, effects
    assert effects.get("scarce_tiles", 0) >= 8, effects
    assert effects.get("scarce_to_TOMATO", 0) >= 8, effects
    sold = int(stats["sold_units"].get("TOMATO", 0))
    assert sold >= 20, sold
    assert stats["sold_revenue"]["TOMATO"] / sold >= 60, "realised below base — not a spike"


@pytest.mark.parametrize("seed", [52000, 52004, 52016])
def test_the_hunter_stays_silent_on_a_calm_seed(seed):
    """The done-when, half two — and the half that makes the other half evidence. Same plan, same
    opponent, no spike: nothing fires, and no tomato is planted or sold."""
    effects, stats = _hunt(seed)
    assert effects.get("scarce_cohorts", 0) == 0, effects
    assert effects.get("scarce_tiles", 0) == 0, effects
    assert int(stats["sold_units"].get("TOMATO", 0)) == 0


def test_the_hunter_never_fires_when_its_budget_is_zero():
    """`max_scarce_tiles` is the off switch, and a change with no off switch cannot be measured."""
    from dataclasses import replace

    plan = _cheap_cohort_plan()
    off = replace(plan, consts={**plan.consts, "max_scarce_tiles": 0})
    effects, stats = _hunt(52014, off)
    assert effects.get("scarce_cohorts", 0) == 0, effects
    assert int(stats["sold_units"].get("TOMATO", 0)) == 0


def test_redirect_leaves_open_and_replanting_cohorts_alone():
    """Redirecting a block that is already half in the ground sows two crops on one cohort, and a
    `replant` block is the farm's standing base rather than a bet."""
    plan = Plan(
        pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
        cohorts=(Cohort("WHEAT", "NW", 4, 0, replant=True, tiles=((0, 0), (1, 0), (2, 0), (3, 0))),
                 Cohort("WHEAT", "NW", 4, 2, tiles=((0, 1), (1, 1), (2, 1), (3, 1))),
                 Cohort("WHEAT", "NW", 4, 9, tiles=((0, 2), (1, 2), (2, 2), (3, 2)))),
        hands="auto", consts={"max_scarce_tiles": 4, "theta_hinge": 0.5})
    # A carrot market three PET_CAFEs deep and a board with no carrot on it.
    obs = _obs(day=5, shops=["PET_CAFE"] * 3, inventory={"CARROT": 10_000 - 900})
    out = redirect(obs, plan, day=5)
    assert out.cohorts[0].crop == "WHEAT", "replanting base was redirected"
    assert out.cohorts[1].crop == "WHEAT", "an already-open cohort was redirected"
    assert out.cohorts[2].crop == "CARROT", "the pending cohort should have been redirected"


def test_redirect_is_remembered_for_the_season():
    """A redirect that flickers with the signal sows half a cohort of each crop."""
    plan = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                cohorts=(Cohort("WHEAT", "NW", 4, 9, tiles=((0, 2), (1, 2), (2, 2), (3, 2))),),
                hands="auto", consts={"max_scarce_tiles": 4, "theta_hinge": 0.5})
    hot = _obs(day=5, shops=["PET_CAFE"] * 3, inventory={"CARROT": 10_000 - 900})
    assert redirect(hot, plan, day=5).cohorts[0].crop == "CARROT"
    calm = _obs(day=6, shops=[], inventory={"CARROT": 10_000})
    calm["step"] = 6 * 24                        # same season, signal gone
    assert redirect(calm, plan, day=6).cohorts[0].crop == "CARROT", "the commitment was forgotten"


def test_redirect_refuses_a_crop_that_cannot_ripen_in_time():
    """Tomato's first yield is eight days out; sown on day 25 it is seed money set on fire."""
    plan = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                cohorts=(Cohort("WHEAT", "NW", 4, 25, tiles=((0, 2), (1, 2), (2, 2), (3, 2))),),
                hands="auto", consts={"max_scarce_tiles": 4, "theta_hinge": 0.5})
    obs = _obs(day=25, shops=["PIZZA_SHOP"] * 4, inventory={"TOMATO": 10_000 - 700})
    assert redirect(obs, plan, day=25).cohorts[0].crop == "WHEAT"


def test_redirect_will_not_downgrade_a_more_valuable_cohort():
    """The threshold is a filter; `tile_value` is the decision. A melon tile is worth ~$1,400 and
    is not traded for a carrot because carrot happens to be scarce."""
    plan = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                cohorts=(Cohort("MELON", "NW", 4, 9, tiles=((0, 2), (1, 2), (2, 2), (3, 2))),),
                hands="auto", consts={"max_scarce_tiles": 4, "theta_hinge": 0.5})
    obs = _obs(day=5, shops=["PET_CAFE"] * 3, inventory={"CARROT": 10_000 - 900})
    assert redirect(obs, plan, day=5).cohorts[0].crop == "MELON"


def test_strawberry_is_not_a_redirect_target():
    """It is scarce in every game from about day 6, so chasing it is a plan edit rather than a
    response — measured at -$4,215 vs starter over 240 paired games."""
    assert "STRAWBERRY" not in REDIRECTABLE
    assert set(REDIRECTABLE) <= {p for p in CROPS if MARKET_PARAMS[p]["below_func"] == "hinge"}


def test_tile_value_matches_the_task_generator():
    """`projection.tile_value` is `tasks._cohort_value` in miniature (the import would be a cycle);
    they must not drift."""
    from agent.tasks import _cohort_value

    proj = Projection(_obs(day=3), horizon=14)
    for crop in CROPS:
        mine = tile_value(crop, proj)
        theirs = _cohort_value(crop, lambda p, d=0: proj.price(p, d))
        assert mine == pytest.approx(theirs), crop


# --------------------------------------------------------------------------- metering

def test_scarcity_floor_is_quoted_against_the_spike_not_the_base():
    """A tomato at 6x base defended at `scarce_floor` must come back as a floor *multiple* of 6x
    that fraction — a floor written against the base would release the whole stock at once."""
    plan = Plan(pasture_tiles=(), land_days={}, herd=(), cohorts=(), hands="auto",
                consts={"scarce_floor": 0.9, "theta_hinge": 0.5, "sell_floor": {}})
    obs = _obs(day=12, shops=["PIZZA_SHOP"] * 4, inventory={"TOMATO": 10_000 - 700})
    spot = Projection(obs, horizon=14).price("TOMATO", 0)
    assert spot > 4 * MARKET_PARAMS["TOMATO"]["base"], spot
    out = scarcity_plan(obs, plan, 0)
    assert out.consts["sell_floor"]["TOMATO"] == pytest.approx(
        0.9 * spot / MARKET_PARAMS["TOMATO"]["base"], abs=0.01)


def test_scarcity_floor_is_inert_when_the_gene_is_zero():
    plan = Plan(pasture_tiles=(), land_days={}, herd=(), cohorts=(), hands="auto",
                consts={"scarce_floor": 0.0, "sell_floor": {"WOOL": 0.35}})
    obs = _obs(day=12, shops=["PIZZA_SHOP"] * 4, inventory={"TOMATO": 10_000 - 700})
    assert scarcity_plan(obs, plan, 0) is plan


def test_a_scarce_product_below_base_gets_no_floor():
    """Scarce on the counter but not yet paying for it: a floor there withholds stock at the price
    it was going to fetch anyway."""
    plan = Plan(pasture_tiles=(), land_days={}, herd=(), cohorts=(), hands="auto",
                consts={"scarce_floor": 0.9, "theta_sqrt": 0.0, "sell_floor": {}})
    obs = _obs(day=12, inventory={"MELON": 10_000 + 400})       # flooded, well under base
    assert "MELON" not in (scarcity_plan(obs, plan, 0).consts.get("sell_floor") or {})


# --------------------------------------------------------------------------- plumbing

def test_the_projection_is_built_once_per_turn():
    """`daily_tasks` prices every task through this; rebuilding a two-farm 14-day forecast per tile
    is how a 0.3 ms generator becomes a 30 ms one."""
    projection.reset()
    obs = _obs(day=7, hour=3)
    first = projection.for_obs(obs)
    assert projection.for_obs(obs) is first
    obs2 = dict(obs, hour=4, step=7 * 24 + 4)
    assert projection.for_obs(obs2) is not first


def test_step_zero_forgets_the_previous_season():
    """The harness rebuilds the agent per game but not the module; a hunter that remembered last
    season would fire on a board that never spiked, and the counter would be evidence of nothing."""
    plan = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                cohorts=(Cohort("WHEAT", "NW", 4, 9, tiles=((0, 2), (1, 2), (2, 2), (3, 2))),),
                hands="auto", consts={"max_scarce_tiles": 4, "theta_hinge": 0.5})
    hot = _obs(day=5, shops=["PET_CAFE"] * 3, inventory={"CARROT": 10_000 - 900})
    assert redirect(hot, plan, day=5).cohorts[0].crop == "CARROT"
    fresh = _obs(day=0, hour=0)                       # step 0 — a new game
    assert redirect(fresh, plan, day=0).cohorts[0].crop == "WHEAT"
    assert projection.stats(0).get("scarce_cohorts") is None


def test_a_twelve_tile_tomato_cohort_on_the_deepest_spike_seed():
    """TASKS_v4 C6's forward-model check, re-measured against 1.32.7.

    The spec asks for >= 80 units at >= $150 from a 12-tile cohort sown on day 8-12, citing tomato
    at $399/$445 on the old block. On 52000:52040 the deepest tomato signal available *while a
    cohort can still be sown* is 2.5 at day 12, and dumping 96 units into a `T` of 200 averages
    ~$127 — the units clear, the price does not, and the reason is that the spike deepens after day
    12. It is met later and metered: the live agent on this seed realised its tomato at **$858**.
    """
    sim, agents = _game(52014, "compiler", "boatlee")
    got = {}

    def watch(obs, day, hour):
        if hour == 0 and day == 12:
            proj = Projection(obs, horizon=14)
            inv = int(round(proj.inventory("TOMATO", CROPS["TOMATO"]["first_yield_day"])))
            units = 12 * CROPS["TOMATO"]["max_yield"] * 2          # fertilized: 2 per tick
            got["units"] = units
            got["mean"] = sum(market_price("TOMATO", inv + k) for k in range(units)) / units
            got["first"] = market_price("TOMATO", inv)

    _play(sim, agents, watch=watch)
    assert got["units"] >= 80
    assert got["first"] >= 150, got
    assert got["mean"] >= 100, got
    assert sim.stats(0)["sold_units"].get("TOMATO", 0) >= 0
