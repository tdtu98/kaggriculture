"""Directed scenario tests for rules that random and competent play never reach.

`tools/audit.py` measures which rules actually fire during fuzz and engine-driven parity runs.
Everything here is a rule that showed a count of 0 there, or whose exact arithmetic is worth
pinning even though it does fire. Each test runs through `run_scripted`, so it verifies
kagsim == reference step by step *and* asserts the resulting game state.
"""

from __future__ import annotations

import pytest

from parity import PASS, run_scripted

TPD = 4  # short days keep multi-day scripts fast; parity covers non-default turnsPerDay


def farmer(op, *args, market=None):
    return {"farmer": [op, *args], "hands": [], "market": market or []}


def day_of(*ops, tpd: int = TPD):
    """Pad a day out to `tpd` turns."""
    return list(ops) + [PASS] * (tpd - len(ops))


def held(st, item, player=0):
    """Total held, inventory + shed.

    The end-of-day refresh empties unit inventories into the shed, so a test whose script runs
    past a day boundary must not look at `inventories` alone.
    """
    n = st["privates"][player]["shed"].get(item, 0)
    for inv in st["privates"][player]["inventories"]:
        n += dict(inv).get(item, 0)
    return n


# --------------------------------------------------------------- ongoing crops

def test_tomato_production_schedule():
    """TOMATO: first_yield_day 8, interval 1, capped at 4 *productions* — so 4 units, not 4/day.

    Productions land at the end of days 7,8,9,10 (`days_since_first = next_day - planted - 8`).
    """
    script = day_of(farmer("PASS", market=[["BUY_SEED", "TOMATO", 1]]),
                    farmer("PLANT", "TOMATO"), farmer("WATER"))
    for _ in range(10):                       # days 1..10, water once per day
        script += day_of(farmer("WATER"))
    script += day_of(farmer("HARVEST"))    # day 11
    st = run_scripted({"turnsPerDay": TPD}, script)

    assert held(st, "TOMATO") == 4, "4 productions x 1 unit, capped at max_yield"
    tile = st["farms"][0]["tiles"][44]
    assert tile[0] == "PLANT", "ongoing crops stay on the tile after harvest"
    assert tile[5] == 0, "yield consumed"


def test_strawberry_every_other_day_schedule():
    """STRAWBERRY: first_yield_day 10, interval 2 — productions at days 10, 12, 14, 16."""
    script = day_of(farmer("PASS", market=[["BUY_SEED", "STRAWBERRY", 1]]),
                    farmer("PLANT", "STRAWBERRY"), farmer("WATER"))
    for _ in range(15):
        script += day_of(farmer("WATER"))
    script += day_of(farmer("HARVEST"))    # day 16
    st = run_scripted({"turnsPerDay": TPD}, script)
    assert held(st, "STRAWBERRY") == 4


def test_ongoing_crop_gets_a_lifespan_once_capped():
    """Hitting the production cap sets max_lifespan_step; before that it is -1 (immortal)."""
    script = day_of(farmer("PASS", market=[["BUY_SEED", "TOMATO", 1]]),
                    farmer("PLANT", "TOMATO"), farmer("WATER"))
    for _ in range(9):
        script += day_of(farmer("WATER"))
    st = run_scripted({"turnsPerDay": TPD}, script)   # through day 9
    assert st["farms"][0]["tiles"][44][6] == -1, "not yet capped"

    for _ in range(2):
        script += day_of(farmer("WATER"))
    st = run_scripted({"turnsPerDay": TPD}, script)   # through day 11
    assert st["farms"][0]["tiles"][44][6] == (11 + 1) * TPD


# ------------------------------------------------------------------- decay

def _wheat_to_maturity() -> list:
    script = day_of(farmer("PASS", market=[["BUY_SEED", "WHEAT", 1]]),
                    farmer("PLANT", "WHEAT"), farmer("WATER"))
    for _ in range(4):                        # days 1..4
        script += day_of(farmer("WATER"))
    return script


def test_onetime_crop_decays_every_other_step_past_lifespan():
    """max_lifespan_step = (planted + max_yield_day + 1) * turnsPerDay; then -1 unit per 2 steps."""
    base = _wheat_to_maturity()               # 20 steps; yield 4, mls = 20
    st = run_scripted({"turnsPerDay": TPD}, base, steps=20)
    assert st["farms"][0]["tiles"][44][5] == 4, "watered days 2,3,4 on top of the starting unit"
    assert st["farms"][0]["tiles"][44][6] == 20

    script = base + [PASS] * 12
    st = run_scripted({"turnsPerDay": TPD}, script, steps=21)   # step 20 fires once
    assert st["farms"][0]["tiles"][44][5] == 3

    st = run_scripted({"turnsPerDay": TPD}, script, steps=23)   # steps 20, 22
    assert st["farms"][0]["tiles"][44][5] == 2

    st = run_scripted({"turnsPerDay": TPD}, script, steps=27)   # 20,22,24,26 -> 0 -> weed
    assert st["farms"][0]["tiles"][44] == ["WEED"]


# ---------------------------------------------------------------- fertilizer

@pytest.mark.parametrize("fertilize,expected", [(False, 4), (True, 5)])
def test_fertilizer_doubles_the_in_window_water_bonus(fertilize, expected):
    """FERTILIZE covers day, day+1, day+2 and doubles the per-watered-day bonus in that span.

    Applied on day 0 it only overlaps the bonus window (WHEAT ages 2..4) on day 2, so it buys
    exactly one extra unit — which is why selling fertilizer beats using it
    (docs/experiments.md E1). Uses turnsPerDay=6: day 0 needs PLANT + PICKUP + FERTILIZE + WATER,
    and watering on the planting day is mandatory.
    """
    tpd = 6
    market = [["BUY_SEED", "WHEAT", 1]]
    if fertilize:
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])
    day0 = [farmer("PASS", market=market), farmer("PLANT", "WHEAT")]
    day0 += [farmer("PICKUP", "FERTILIZER", 1), farmer("FERTILIZE")] if fertilize else [PASS, PASS]
    day0 += [farmer("WATER")]
    script = day_of(*day0, tpd=tpd)
    for _ in range(4):                       # days 1..4; day 2,3,4 are the bonus window
        script += day_of(farmer("WATER"), tpd=tpd)
    script += day_of(farmer("HARVEST"), tpd=tpd)
    st = run_scripted({"turnsPerDay": tpd}, script)
    assert held(st, "WHEAT") == expected


# --------------------------------------------------------------------- care

def _goose_script(days: int, care: bool) -> list:
    script = [farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1],
                                           ["BUY_PRODUCT", "WHEAT", 40]]),
              farmer("PICKUP", "GOOSE", 1),
              farmer("PLACE", "GOOSE"),
              PASS]
    for _ in range(days):
        ops = [farmer("PICKUP", "WHEAT", 1), farmer("FEED"), farmer("HARVEST")]
        ops.append(farmer("CARE") if care else PASS)
        script += ops
    return script


def test_care_bonus_is_banked_then_paid_on_the_next_production():
    """CARE + FEED banks +1 at end of day; it is paid out on the next production day, if fed."""
    eggs = {}
    for care in (False, True):
        st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000},
                          _goose_script(10, care))
        shed = st["privates"][0]["shed"]
        eggs[care] = shed.get("EGG", 0) + dict(st["privates"][0]["inventories"][0]).get("EGG", 0)
    assert eggs[True] > eggs[False], f"care should raise egg output: {eggs}"
    # One extra action per animal per day roughly doubles output (base 1 -> 1 + banked 1).
    assert eggs[True] >= 1.6 * eggs[False], eggs


def test_care_bank_is_destroyed_on_an_unfed_production_day():
    """Skipping the feed on a production day forfeits the whole banked bonus, not just that day."""
    script = [farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1],
                                           ["BUY_PRODUCT", "WHEAT", 40]]),
              farmer("PICKUP", "GOOSE", 1), farmer("PLACE", "GOOSE"), PASS]
    for d in range(1, 8):
        if d == 5:                      # bank exists, but no feed on this production day
            script += [PASS] * TPD
        else:
            script += [farmer("PICKUP", "WHEAT", 1), farmer("FEED"),
                       farmer("CARE"), farmer("HARVEST")]
    st = run_scripted({"turnsPerDay": TPD, "startingMoney": 10000}, script)
    tile = st["farms"][0]["tiles"][44]
    assert tile[0] == "COOP_A", "one skipped day is survivable"
    assert tile[8] == 0 or tile[8] == 1, f"bank reset after the unfed production, got {tile[8]}"


# ------------------------------------------------------------- locked tiles

def test_unit_can_cross_but_not_act_on_a_locked_tile():
    """Locked tiles are passable (`:314`) but every tile op no-ops there (`:324`)."""
    script = [
        farmer("PASS", market=[["BUY_SEED", "WHEAT", 5]]),
        farmer("EAST"),                 # (4,4) -> (5,4), which is in the locked NE quadrant
        farmer("PLANT", "WHEAT"),       # no-op: locked
        farmer("BUILD_COOP"),           # no-op: locked
    ]
    # Stop before the day boundary: end_of_day respawns the farmer back at (4,4).
    st = run_scripted({"turnsPerDay": TPD}, script, steps=3)
    assert st["farms"][0]["farmer"] == [5, 4], "movement onto locked land is allowed"
    assert st["farms"][0]["tiles"][45] == ["LOCKED"], "no tile op took effect"
    assert st["privates"][0]["seeds"] == {"WHEAT": 5}, "no seed consumed"


# ------------------------------------------------- action argument coercion

def test_quantity_arguments_use_python_int_semantics():
    """`_apply_unit_action` and `_parse_order` both coerce the quantity with a bare `int()`.

    That accepts numeric strings and truncates floats, and *raises* on anything else — unlike a
    strict integer extract, which would silently drop orders the reference honours.
    """
    from kaggle_environments import make

    import kagsim

    def both(arg):
        act = [{"farmer": ["PICKUP", "WHEAT", arg], "hands": [], "market": []},
               {"farmer": ["PASS"], "hands": [], "market": []}]
        env = make("kaggriculture", configuration={"episodeSteps": 10, "seed": 1})
        env.reset(num_agents=2)
        try:
            env.step(act)
            ref_ok = True
        except (ValueError, TypeError):
            ref_ok = False
        sim = kagsim.Sim({"episodeSteps": 10, "seed": 1})
        try:
            sim.step(act)
            sim_ok = True
        except ValueError:
            sim_ok = False
        return ref_ok, sim_ok

    for arg in ["abc", 3.7, -2.9, None, "5", " 7 ", "5.5", True, 4]:
        ref_ok, sim_ok = both(arg)
        assert ref_ok == sim_ok, f"{arg!r}: reference ok={ref_ok}, kagsim ok={sim_ok}"


@pytest.mark.parametrize("arg,expected", [("5", 5), (3.9, 3), ("abc", 0), (2, 2), (" 3 ", 3)])
def test_market_order_quantity_coercion(arg, expected):
    st = run_scripted({"turnsPerDay": TPD},
                      [{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", arg]]}])
    assert st["privates"][0]["seeds"].get("WHEAT", 0) == expected


# ------------------------------------------------- guards found by coverage audit

def test_feeding_an_already_fed_animal_is_a_no_op():
    """`FEED` returns early on `fed_today` (`:485`) *before* taking the wheat.

    So a double-feed must not consume a second unit — which matters, because a task planner that
    re-issues FEED after a state refresh would otherwise silently burn the flock's feed.
    """
    script = [
        farmer("BUILD_COOP", market=[["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 10]]),
        farmer("PICKUP", "GOOSE", 1),
        farmer("PLACE", "GOOSE"),
        PASS,
        farmer("PICKUP", "WHEAT", 2),
        farmer("FEED"),
        farmer("FEED"),        # already fed today -> no-op, no second wheat consumed
        PASS,
    ]
    st = run_scripted({"turnsPerDay": 4, "startingMoney": 10000}, script, steps=7)
    inv = dict(st["privates"][0]["inventories"][0])
    assert inv.get("WHEAT", 0) == 1, f"exactly one wheat consumed, got {inv}"
    assert st["farms"][0]["tiles"][44][5] is True, "fed_today set"


def test_place_into_a_full_shed_is_a_no_op():
    """PLACE's shed-drop clamps to remaining room and returns when there is none (`:474`)."""
    script = [
        farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 2]]),   # shed 2/2 -> full
        farmer("PICKUP", "WHEAT", 1),                            # shed 1, inv 1
        farmer("PASS", market=[["BUY_PRODUCT", "WHEAT", 1]]),    # shed 2/2 -> full again
        farmer("PLACE", "WHEAT", 1),                             # no room -> no-op
    ]
    st = run_scripted({"turnsPerDay": 6, "shedCapacity": 2}, script, steps=4)
    assert st["privates"][0]["shed"] == {"WHEAT": 2}, "shed unchanged at capacity"
    assert dict(st["privates"][0]["inventories"][0]) == {"WHEAT": 1}, "item stays in inventory"


def test_plant_guards_are_shadowed_by_the_atomic_precheck():
    """PLANT's own `crop not in CROPS` and `seeds <= 0` guards (`:365`, `:369`) are unreachable.

    The interpreter tallies PLANT demand per crop *before* applying anything and drops every
    request for a crop whose demand exceeds the seeds on hand (`:905`). An unknown crop has
    `seeds.get(crop, 0) == 0`, so demand 1 > 0 blocks it; a known crop with no seeds blocks the
    same way. Either request becomes PASS and never reaches the guards.
    """
    st = run_scripted({"turnsPerDay": TPD}, [
        farmer("PLANT", "BANANA"),      # unknown crop -> blocked upstream
        farmer("PLANT", "WHEAT"),       # no seeds     -> blocked upstream
    ], steps=2)
    assert st["privates"][0]["seeds"] == {}
    assert st["farms"][0]["tiles"][44] == ["EMPTY"]
