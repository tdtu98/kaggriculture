"""P1 — the day-granular season model (`agent/season.py`).

Four kinds of test, and the split is the point.

* **Arithmetic against hand-computed fixtures.** A wheat tile, a strawberry tile and a cow, each on
  an otherwise empty board, each with the units it must produce worked out from `CROPS`/`ANIMALS` in
  the test itself. These are the tests that fail when a curve constant moves, which is what makes the
  mutation check below meaningful.
* **A rollout regression pin.** One fixed state, one exact terminal bank. Any change to the model
  that moves money has to move this number deliberately.
* **The decision hooks fire.** A `DayDecisions` that suppresses a cohort, adds a planting or changes
  the crew must change the terminal bank. A hook that is wired but inert is the E44 failure with a
  green test on top, so each one is asserted to *move the money*, not merely to be accepted.
* **Fidelity, in play.** A small seed set of real games, the model built from the **delivered**
  observation and compared against **settled** money (E44/E21: never from orders, never from the
  stored replay state). Deliberately loose bounds — the gate is quoted from the 80-game block in the
  report, not from four games here; what this test defends is that the model has not silently
  stopped tracking the game at all.
"""

from __future__ import annotations

import statistics

import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, market_price

from agent import projection, season
from agent.plan import Cohort, Plan
from agent.season import DayDecisions, SeasonState, rollout, step_day

BOARD = 10
HALF = BOARD // 2


# --------------------------------------------------------------------------- fixtures

def _obs(tiles=None, money=10_000.0, day=0, shed=None, seeds=None, shops=()):
    """A delivered-shaped observation with NW unlocked and everything else LOCKED.

    Shaped like `__get_shared_state(position)` (`kaggriculture.py:754-767`) — the surface the runner
    actually hands an agent — rather than like `env.state[p].observation`, which has the shared
    fields stripped (E21).
    """
    grid = [["LOCKED"] * BOARD for _ in range(BOARD)]
    for y in range(HALF):
        for x in range(HALF):
            grid[y][x] = None
    for (x, y), tile in (tiles or {}).items():
        grid[y][x] = tile
    farm = {"tiles": grid, "farmer": [HALF - 1, HALF - 1], "hands": [], "money": money,
            "hires_today": 0, "unlocked_quadrants": ["NW"]}
    empty = {"tiles": [["LOCKED"] * BOARD for _ in range(BOARD)], "farmer": [0, 0], "hands": [],
             "money": 3000.0, "hires_today": 0, "unlocked_quadrants": ["NW"]}
    return {
        "player": 0, "day": day, "hour": 0, "step": day * 24,
        "farms": [farm, empty],
        "private": {"shed": dict(shed or {}), "seeds": dict(seeds or {}), "inventories": [{}]},
        "market": {"inventory": {p: 10_000 for p in season.PRODUCTS},
                   "prices": {p: season.BASE_PRICE[p] for p in season.PRODUCTS}},
        "town": {"unlocked_shops": list(shops)},
    }


def _plan(cohorts=(), herd=(), pastures=(), consts=None):
    return Plan(pasture_tiles=tuple(pastures), land_days={"NE": 99, "SW": 99, "SE": 99},
                herd=tuple(herd), cohorts=tuple(cohorts), hands="auto",
                consts=dict(consts or {}))


def _standing(crop, planted_day):
    """A plant already in the ground, shaped like `kaggriculture.py:202`'s `_new_plant`."""
    from agent.forward import _new_plant

    return _new_plant(crop, planted_day, 24)


def _standing_cow(placed_day):
    from agent.forward import _new_animal

    return _new_animal("COW", placed_day)


def _state(**kw):
    plan = kw.pop("plan", None)
    return SeasonState.from_obs(_obs(**kw), seat=0, plan=plan, known="starter")


# --------------------------------------------------------------------------- crop arithmetic

def test_wheat_tile_yields_six_units_with_the_plan_s_fertilizer():
    """WHEAT: `max_yield_day` 4, water window ages 2-4, `max_yield` 6.

    Three watered days inside the window, doubled by a fertilizer applied at age 2 (which covers
    ages 2, 3 and 4 — `kaggriculture.py:468`), is `1 + 2 + 2 + 2 = 7`, capped at 6. Harvested on
    age 4, which is `max_yield_day`.
    """
    cohort = Cohort("WHEAT", "NW", 1, 0, tiles=((0, 0),))
    # A floor above 1.0 on FERTILIZER keeps the four units in the shed for the fertilize task; without
    # it dawn sells everything above the reserve, which is what the shell also does.
    plan = _plan(cohorts=(cohort,), consts={"fert_ages": {"WHEAT": (2,)},
                                            "sell_floor": {"FERTILIZER": 2.0}})
    st = _state(plan=plan, shed={"FERTILIZER": 4})
    for _ in range(5):                       # days 0..4
        step_day(st)
    assert st.sold["WHEAT"] == 6
    assert st.day == 5
    # And the tile is free again on the dawn after the harvest, which is `replant_cycle` = 5.
    assert st.farm.tiles[0][0] is None


def test_wheat_without_fertilizer_yields_four():
    cohort = Cohort("WHEAT", "NW", 1, 0, tiles=((0, 0),))
    st = _state(plan=_plan(cohorts=(cohort,)), shed={})
    for _ in range(5):
        step_day(st)
    assert st.sold["WHEAT"] == 1 + 3         # base unit plus one per watered day in the window


def test_strawberry_ticks_four_times_at_ages_9_11_13_15():
    """STRAWBERRY: `first_yield_day` 10, `interval` 2, `max_yield` 4 productions.

    `days_since_first = next_day - planted - first_yield_day` (`kaggriculture.py:783`), so the
    productions close the ends of ages 9, 11, 13 and 15, and one fertilizer at 9 and one at 13 cover
    all four. Two units a tick, harvested the following day, is 8 — times `YIELD_REALISED` 0.9.
    """
    cohort = Cohort("STRAWBERRY", "NW", 1, 0, tiles=((0, 0),))
    plan = _plan(cohorts=(cohort,), consts={"fert_ages": {"STRAWBERRY": (9, 13)},
                                            "sell_floor": {"FERTILIZER": 2.0}})
    st = _state(plan=plan, shed={"FERTILIZER": 4})
    days = []
    for _ in range(20):
        before = st.sold.get("STRAWBERRY", 0)
        step_day(st)
        if st.sold.get("STRAWBERRY", 0) > before:
            days.append(st.day - 1)
    assert days == [10, 12, 14, 16]          # the tick's units reach the market the next day
    # Four ticks of 2 units, each scaled by the strawberry realisation factor and rounded.
    assert st.sold["STRAWBERRY"] == 4 * round(2 * season.YIELD_REALISED["STRAWBERRY"])


def test_an_ongoing_crop_dies_after_its_last_production_and_frees_its_tile():
    """`_daily_refresh_plants` stamps `max_lifespan_step` on the last production (`:786-800`)."""
    # No cohort: a plan that names the tile would simply re-sow it the same dawn, which is what
    # `_planting_tasks` does and is not what this test is about.
    st = _state(tiles={(0, 0): _standing("STRAWBERRY", 0)}, plan=_plan())
    for _ in range(17):                       # days 0..16; the last tick closed the dusk of day 15
        step_day(st)
    assert isinstance(st.farm.tiles[0][0], dict)
    for _ in range(2):
        step_day(st)
    assert st.farm.tiles[0][0] is None


def test_a_cow_produces_one_plus_interval_units_per_production():
    """COW: cost 400, `first_yield_day` 8, `interval` 2, `max_held` 6.

    A fed-and-cared animal banks one `pending_care_bonus` per day and spends it on the production
    day (`kaggriculture.py:822-830`), so a production is `1 + interval` = 3 units.
    """
    spec = ANIMALS["COW"]
    plan = _plan(herd=(("COW", 0),), pastures=((0, 0),))
    st = _state(tiles={(0, 0): {"kind": "PASTURE"}}, plan=plan)
    paid = st.money
    step_day(st)
    assert paid - st.money >= spec["cost"], "the head was not actually paid for"
    got = []
    for _ in range(15):
        before = st.sold.get("MILK", 0)
        step_day(st)
        if st.sold.get("MILK", 0) > before:
            got.append((st.day - 1, st.sold["MILK"] - before))
    first_day, first_units = got[0]
    # `days_since_first = next_day - placed_day - first_yield_day` (`:822`), so a head placed on
    # day 0 produces at the dusk of day 7 and the units are collected on day 8.
    assert first_day == spec["first_yield_day"]
    assert all(u == 1 + spec["interval"] for _d, u in got[1:])
    assert first_units >= 1 + spec["interval"]


def test_an_animal_eats_one_wheat_a_day():
    st = _state(tiles={(0, 0): _standing_cow(0)}, plan=_plan(), shed={"WHEAT": 10})
    step_day(st)
    # One unit was eaten; the two-day feed reserve stayed in the shed and the rest was sold.
    eaten = 10 - st.farm.shed.get("WHEAT", 0) - st.sold.get("WHEAT", 0)
    assert eaten == season.FEED_PER_ANIMAL
    assert st.farm.shed.get("WHEAT", 0) == 2


# --------------------------------------------------------------------------- market arithmetic

def test_selling_moves_the_price_one_unit_at_a_time():
    """`_commit_unit` adds one to the inventory per unit and re-quotes inside the loop (`:597,660`).

    So the proceeds of `n` units are the *sum* of `n` successive quotes, not `n x` the first one.
    """
    st = _state(shed={"STRAWBERRY": 30}, plan=_plan())
    inv0 = st.market_inv["STRAWBERRY"]
    got = season._sell_units(st, "STRAWBERRY", 30)
    assert got == sum(market_price("STRAWBERRY", inv0 + k) for k in range(30))
    assert st.market_inv["STRAWBERRY"] == inv0 + 30
    assert got < 30 * market_price("STRAWBERRY", inv0)       # the quote fell as the line executed


def test_a_sell_floor_withholds_the_units_below_it():
    st = _state(shed={"STRAWBERRY": 60},
                plan=_plan(consts={"sell_floor": {"STRAWBERRY": 0.95}}))
    step_day(st)
    assert 0 < st.sold["STRAWBERRY"] < 60
    limit = 0.95 * season.BASE_PRICE["STRAWBERRY"]
    assert market_price("STRAWBERRY", st.market_inv["STRAWBERRY"] - 1) >= limit


def test_the_town_drain_is_projection_s_own_arithmetic():
    """`season` must not carry a second copy of C6's drain table (E39)."""
    shops = ["YARN_STORE", "BAKERY"]
    st = _state(shops=shops, plan=_plan())
    inv0 = dict(st.market_inv)
    season._town_day(st)
    for item in ("WOOL", "WHEAT", "MELON"):
        assert inv0[item] - st.market_inv[item] == projection.drain_per_day(item, 0, shops)


def test_projection_drain_per_day_and_the_method_agree():
    """The refactor pin: `Projection._drain` delegates to the module function, byte-identically."""
    obs = _obs(shops=["PET_CAFE", "PET_CAFE", "SMOOTHIE_SHOP"])
    proj = projection.Projection(obs)
    for item in season.PRODUCTS:
        for day in (0, 3, 9, 27):
            assert proj._drain(item, day) == projection.drain_per_day(item, day, proj.shops)


def test_the_opponent_s_measured_schedule_lands_in_the_market():
    """A fingerprinted boatlee contributes its settled sell table, not its orders (E48/E50)."""
    from agent import opponent

    obs = _obs()
    obs["farms"][1]["tiles"] = [[None] * BOARD for _ in range(BOARD)]
    st = SeasonState.from_obs(obs, seat=0, plan=_plan(), known="boatlee")
    assert st.opp_supply, "boatlee's table produced no supply at all"
    day_of_first_wool = min(d for d, row in st.opp_supply.items() if "WOOL" in row)
    settled = opponent.settled_schedule("boatlee")
    assert day_of_first_wool == min(s for s, _n in settled["WOOL"]) // 24


# --------------------------------------------------------------------------- costs

def test_hiring_is_the_fibonacci_wage_and_is_paid_every_dawn():
    """`_hire_cost = mult * fib(n_already_today)` (`:690-698`), and the roster clears daily (`:867`)."""
    cohort = Cohort("WHEAT", "NW", 20, 0, replant=True,
                    tiles=tuple((x, y) for y in range(4) for x in range(5)))
    st = _state(plan=_plan(cohorts=(cohort,)), money=10_000.0)
    before = st.money
    step_day(st)
    wages = sum(season.HIRE_FIB[:st.hands])
    seeds = 20 * CROPS["WHEAT"]["seed"]
    assert st.hands > 0
    assert before - st.money == pytest.approx(wages + seeds)


def test_land_is_bought_on_the_plan_s_day_at_the_env_s_price():
    plan = Plan(pasture_tiles=(), land_days={"NE": 2, "SW": 99, "SE": 99}, herd=(), cohorts=(),
                hands="auto", consts={})
    st = _state(plan=plan, money=5_000.0)
    step_day(st)
    assert "NE" not in st.unlocked
    step_day(st)
    step_day(st)
    assert "NE" in st.unlocked
    assert st.money == pytest.approx(5_000.0 - 1_000.0)
    assert st.farm.tiles[0][HALF] is None                    # the quadrant's tiles are free now


# --------------------------------------------------------------------------- decisions

def _decision_plan():
    return _plan(
        cohorts=(Cohort("STRAWBERRY", "NW", 6, 1, tiles=((0, 0), (1, 0), (2, 0),
                                                         (0, 1), (1, 1), (2, 1))),),
        consts={"fert_ages": {"STRAWBERRY": (9, 13)}})


def test_holding_a_cohort_changes_the_bank():
    plan = _decision_plan()
    base = rollout(_state(plan=plan))
    held = rollout(_state(plan=plan), overrides={1: DayDecisions(hold_cohorts=frozenset({0}))})
    assert held != base, "hold_cohorts is wired but inert (E44)"
    assert held < base


def test_an_extra_planting_changes_the_bank():
    plan = _decision_plan()
    base = rollout(_state(plan=plan))
    more = rollout(_state(plan=plan),
                   overrides={1: DayDecisions(plant=(("MELON", "NW", 5),))})
    assert more != base


def test_hire_delta_costs_wages():
    plan = _decision_plan()
    base = rollout(_state(plan=plan))
    richer = rollout(_state(plan=plan),
                     overrides={d: DayDecisions(hire_delta=6) for d in range(1, 30)})
    assert richer < base, "six extra hands a day for a season must cost money"


def test_a_sell_floor_override_changes_the_bank():
    st = _state(shed={"MELON": 40}, plan=_plan())
    base = rollout(st)
    st2 = _state(shed={"MELON": 40}, plan=_plan())
    floored = rollout(st2, overrides={0: DayDecisions(sell_floor={"MELON": 0.99})})
    assert floored != base


def test_buying_land_can_be_forced_and_suppressed():
    plan = Plan(pasture_tiles=(), land_days={"NE": 0, "SW": 99, "SE": 99}, herd=(), cohorts=(),
                hands="auto", consts={})
    st = _state(plan=plan, money=5_000.0)
    step_day(st, DayDecisions(buy_land=False))
    assert "NE" not in st.unlocked
    step_day(st, DayDecisions(buy_land=True))
    assert "NE" in st.unlocked


# --------------------------------------------------------------------------- rollout

def test_rollout_regression_pin():
    """One fixed state, one exact bank. Moves only when the model is changed deliberately."""
    plan = _plan(
        cohorts=(Cohort("STRAWBERRY", "NW", 8, 1,
                        tiles=tuple((x, y) for y in range(2) for x in range(4))),
                 Cohort("WHEAT", "NW", 4, 0, replant=True, tiles=((0, 2), (1, 2), (2, 2), (3, 2))),),
        herd=(("COW", 0), ("COW", 1)),
        pastures=((0, 3), (1, 3)),
        consts={"fert_ages": {"STRAWBERRY": (9, 13), "WHEAT": (2,)}, "release_pressure": 70})
    st = _state(plan=plan, money=3_000.0)
    assert rollout(st) == pytest.approx(35_768.0)
    assert st.day == season.LAST_DAY + 1


def test_rollout_is_deterministic_and_does_not_mutate_the_source_state():
    plan = _decision_plan()
    st = _state(plan=plan)
    a = rollout(st.clone())
    b = rollout(st.clone())
    assert a == b
    assert st.day == 0, "clone() must not share the board with its parent"


def test_rollout_never_raises():
    """A rollout is called from a turn that must not forfeit the episode (E21)."""
    st = _state(plan=_plan())
    st.plan = object()                        # a plan-shaped hole; anything may go wrong downstream
    assert isinstance(rollout(st), float)


def test_rollout_from_day_three_is_inside_the_p2_latency_budget():
    """P2 needs thousands of these per dawn; the task's bar is <= 5 ms, the target ~1 ms."""
    import time

    plan = _decision_plan()
    st = _state(plan=plan, day=3)
    rollout(st.clone())                                        # warm the caches
    t0 = time.perf_counter()
    for _ in range(50):
        rollout(st.clone())
    assert (time.perf_counter() - t0) / 50 < 0.005


# --------------------------------------------------------------------------- fidelity, in play

@pytest.fixture(scope="module")
def played_games():
    """Four real games, snapshotted at each checkpoint dawn from the **delivered** observation.

    Small on purpose: the gate is quoted from the 80-game 72000 block in the P1 report. What this
    fixture defends is that the model still tracks a real season at all.
    """
    import kagsim
    from harness import registry

    out = []
    for seed in (72100, 72101):
        for seat in (0, 1):
            names = ["starter", "starter"]
            names[seat] = "compiler"
            agents = [registry.get(n).build() for n in names]
            sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
            snaps = {}
            for step in range(719):
                obs = [sim.observation(p) for p in range(2)]
                if step % 24 == 0 and step // 24 in (8, 15, 22):
                    snaps[step // 24] = obs[seat]
                sim.step([agents[p](obs[p]) for p in range(2)])
            out.append((snaps, getattr(agents[seat], "plan", None), sim.money(seat), seat))
    return out


@pytest.mark.parametrize("day,bound", [(8, 0.30), (15, 0.25), (22, 0.15)])
def test_fidelity_tracks_the_settled_bank(played_games, day, bound):
    """Median absolute terminal-bank error, from the delivered obs against **settled** money.

    Never from orders and never from `env.state[p].observation` — both are how a fidelity number
    stops meaning anything (E21/E48). The bounds are the gate's, loosened for a four-game sample.
    """
    assert statistics.median(_fidelity(played_games, day)) <= bound


def _fidelity(games, day):
    out = []
    for snaps, plan, final, seat in games:
        st = SeasonState.from_obs(snaps[day], seat=seat, plan=plan, known="starter")
        out.append(abs(rollout(st) - final) / max(1.0, final))
    return out


def test_mutating_a_curve_constant_breaks_fidelity(played_games, monkeypatch):
    """The mutation check: if the strawberry tick cadence is wrong, the fidelity test must fail.

    Cheap to write and the only thing that says the fidelity bound above is load-bearing rather than
    a bound wide enough to pass on any model at all.
    """
    intact_errors = _fidelity(played_games, 8)
    broken = dict(CROPS["STRAWBERRY"])
    broken["max_yield"] = 1                   # one production a tile instead of four
    monkeypatch.setitem(CROPS, "STRAWBERRY", broken)
    monkeypatch.setattr(season, "TICK_DAYS", {c: season.tick_days(c) for c in CROPS})
    broken_errors = _fidelity(played_games, 8)
    assert statistics.median(broken_errors) > 2 * statistics.median(intact_errors)


# --------------------------------------------------------------------------- the shop draw

def test_future_shops_is_the_seam_for_averaging_over_the_draw():
    """`shops_on` reveals a sampled continuation only as fast as the town can unlock it.

    The draw is the model's dominant error term (module docstring), so the thing P2 will do about
    it — roll out under several sampled continuations and average — has to be expressible without
    reaching inside `_town_day`.
    """
    st = _state(shops=["BAKERY"], plan=_plan())
    st.future_shops = ["YARN_STORE", "PET_CAFE"]
    assert st.shops_on(0) == ["BAKERY"]                 # no instance has unlocked yet
    assert st.shops_on(6) == ["BAKERY", "YARN_STORE"]   # two instances by day 6
    assert st.shops_on(24) == ["BAKERY", "YARN_STORE", "PET_CAFE"]
    assert st.clone().future_shops == st.future_shops


def test_a_revealed_draw_changes_the_bank():
    # A floor just above par holds the wool until the town's demand lifts the quote, which is the
    # only way a *drain* difference can show up in the bank at all.
    held = _plan(consts={"sell_floor": {"WOOL": 1.05}})
    base = rollout(_state(shed={"WOOL": 40}, shops=[], plan=held))
    st2 = _state(shed={"WOOL": 40}, shops=[], plan=held)
    st2.future_shops = ["YARN_STORE"] * 8               # the only shop that eats WOOL, x8
    assert rollout(st2) > base
