"""C3: play the day before committing to it, and prune what the farm cannot sustain.

The verifier exists because the router grades its own homework. It can only report what it
*scheduled*; whether those ops land on a live plant, with the right thing in hand, on a tile that
still exists at that turn, is a question only the rules can answer — so this runs the compiled day
through `agent/forward.py`, which is parity-tested against kagsim step by step.

Two failure classes are separated throughout, because conflating them made the first version prune
healthy plantings to atone for wheat that had simply finished its life:

* **thirst** — a plant the routing failed. A compiler bug, and the thing pruning exists to prevent.
* **old age** — a crop past `max_lifespan_step` that already paid out. Not a failure at all.
"""

from __future__ import annotations

import pytest

from agent.forward import FarmModel
from agent.plan import Plan
from agent.router import UnitScript, route, spawn_positions
from agent.tasks import LAST_DAY, Task, daily_tasks

from agent.verify import compile_day, verify_day
from tests.test_tasks import EMPTY_PLAN, PRICES, animal, board, obs_for, plant


def script(*ops) -> UnitScript:
    return UnitScript(unit=0, ops=[list(o) for o in ops], stops=[])


def scripts(*op_lists) -> dict:
    return {i: script(*ops) for i, ops in enumerate(op_lists)}


# --------------------------------------------------------------------- blocked ops

def test_an_op_that_changes_nothing_is_reported_blocked():
    """A HARVEST on bare ground costs a turn and does nothing — the compiler's health counter, and
    the one number that tells a desynced script from a bad plan (E55)."""
    obs = obs_for(board(), day=3)
    report = verify_day(obs, scripts([["HARVEST"], ["HARVEST"]]), turns=4)
    assert len(report.blocked_ops) == 2
    assert {op for _u, _t, op in report.blocked_ops} == {"HARVEST"}


def test_real_work_is_not_reported_blocked():
    tiles = board({(4, 4): plant("STRAWBERRY", 0, unwatered=1)})
    report = verify_day(obs_for(tiles, day=3), scripts([["WATER"]]), turns=2)
    assert report.blocked_ops == []


def test_moves_and_passes_are_never_counted_as_blocked():
    obs = obs_for(board(), day=1)
    report = verify_day(obs, scripts([["NORTH"], ["PASS"], ["SOUTH"]]), turns=4)
    assert report.blocked_ops == []


# --------------------------------------------------------------------- deaths

def test_a_plant_left_unwatered_is_reported_as_a_thirst_death():
    tiles = board({(4, 4): plant("STRAWBERRY", 0, unwatered=1)})
    report = verify_day(obs_for(tiles, day=5), scripts([["PASS"]]), turns=24)
    assert len(report.deaths) == 1
    assert report.expired == []
    assert not report.ok


def test_watering_it_saves_it():
    tiles = board({(4, 4): plant("STRAWBERRY", 0, unwatered=1)})
    report = verify_day(obs_for(tiles, day=5), scripts([["WATER"]]), turns=24)
    assert report.deaths == []


def test_a_crop_that_expires_after_paying_out_is_not_a_death():
    """Wheat's `max_lifespan_step` is the dawn of age 5; it then sheds a unit every two steps. That
    is the crop finishing, not the router failing, and pruning plantings over it would be a
    compiler apologising for the rules."""
    tiles = board({(4, 4): plant("WHEAT", 0, units=1, watered=True)})
    report = verify_day(obs_for(tiles, day=5), scripts([["PASS"]]), turns=24)
    assert report.deaths == [], "not a thirst death"
    assert len(report.expired) == 1
    assert report.ok, "a day is not broken by a crop reaching the end of its life"


# --------------------------------------------------------------------- crew and supplies

def test_the_crew_the_route_assumed_is_the_crew_that_is_verified():
    """At dawn the hands do not exist — they are hired in hour 0's market and stand up at hour 1.
    Verifying an eight-unit day against the lone farmer had seven scripts execute against nobody,
    which read as 470 plant deaths a season."""
    tiles = board({(4, 3): plant("STRAWBERRY", 0, unwatered=1),
                   (4, 5): plant("STRAWBERRY", 0, unwatered=1)})
    obs = obs_for(tiles, day=5)                      # one farmer on the board, no hands
    # unit 1 stands up on (5, 4), the next free centre tile — so its walk starts from there
    both = {0: script(["NORTH"], ["WATER"]),
            1: script(["WEST"], ["SOUTH"], ["WATER"])}

    assert len(verify_day(obs, both, turns=24).deaths) == 1, "unit 1 does not exist yet"
    assert verify_day(obs, both, turns=24, crew=2).deaths == [], "…but it will by hour 1"


def test_the_days_shopping_is_modelled_before_the_units_act():
    """C4 buys seed at hour 0 and the PLANT happens at hour 2. A verifier that skipped the purchase
    reported every planting as blocked — failing a step it had declined to model."""
    plan = Plan.boatlee_like()
    obs = obs_for(board(), day=0)
    compiled = compile_day(obs, plan)
    planted = [op for s in compiled.scripts.values() for op in s.ops if op[0] == "PLANT"]
    assert planted, "day 0 should be a planting day"
    assert not [b for b in compiled.report.blocked_ops if b[2] == "PLANT"]


# --------------------------------------------------------------------- pruning

def _busy_board_and_plan():
    """Sixteen thirsty plants already on the board, and a cohort wanting five more today.

    The shape that makes planting a real decision: the waters must happen or plants die, and every
    planting spends two of the same turns.
    """
    from agent.plan import Cohort

    tiles = board({(x, y): plant("STRAWBERRY", 0, unwatered=1)
                   for x in range(8) for y in (0, 1)})
    base = Plan.boatlee_like()
    plan = Plan(**{**base.__dict__, "pasture_tiles": (), "herd": (),
                   "land_days": {"NE": 0, "SW": 99, "SE": 99},
                   "cohorts": (Cohort(crop="STRAWBERRY", quadrant="NW", n_tiles=5, plant_day=5,
                                      tiles=tuple((x, 3) for x in range(5))),)})
    return obs_for(tiles, day=5), plan


def test_a_day_that_cannot_carry_its_plantings_loses_them():
    obs, plan = _busy_board_and_plan()
    compiled = compile_day(obs, plan, hands=0)

    assert compiled.pruned, "one unit cannot water sixteen plants and start five more"
    assert all(t.op == "PLANT" for t in compiled.pruned), "only plantings are ever dropped"
    assert compiled.report.deaths == [], "and the point of dropping them is that nothing dies"


def test_pruning_is_proportionate_to_the_crew():
    """More hands, fewer sacrifices — and with enough hands, none."""
    obs, plan = _busy_board_and_plan()
    pruned = [len(compile_day(obs, plan, hands=n).pruned) for n in (0, 1, 3)]
    assert pruned[0] > pruned[1] > pruned[2] == 0, pruned


def test_a_feasible_day_is_left_alone():
    tiles = board({(4, 3): plant("STRAWBERRY", 0, unwatered=1)})
    empty = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                 cohorts=(), hands="auto", consts={})
    compiled = compile_day(obs_for(tiles, day=5), empty)
    assert compiled.pruned == []
    assert compiled.report.ok
    assert not compiled.overcommit


def test_survival_work_is_never_what_gets_pruned():
    """The ordering that matters: a plant that dies costs its whole remaining life, a plant not
    started costs a seed."""
    obs, plan = _busy_board_and_plan()
    compiled = compile_day(obs, plan, hands=0)
    assert {t.kind for t in compiled.pruned} == {"plant"}


# --------------------------------------------------------------------- the gate

def test_the_boatlee_like_plan_compiles_a_clean_day_on_a_real_board():
    """C3's done-when, in miniature: a real season board, compiled and verified, with no plant lost
    to thirst. The full check is 20 seeds x 30 days, run in E59."""
    import kagsim

    from harness.registry import get

    plan = Plan.boatlee_like()
    boat, starter = get("boatlee").build(), get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 5})
    thirst = 0
    for _ in range(24 * 12):
        o = sim.observation(0)
        if o["hour"] == 0:
            compiled = compile_day(o, plan)
            thirst += len(compiled.report.deaths)
        sim.step([boat(o), starter(sim.observation(1))])
    assert thirst == 0, f"{thirst} plants lost to thirst in twelve days"


def test_compiling_a_day_fits_the_turn_budget():
    """C3's budget is 100 ms per day: tasks, hiring, routing, verification and any pruning."""
    import time

    import kagsim

    from harness.registry import get

    plan = Plan.boatlee_like()
    boat, starter = get("boatlee").build(), get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 5})
    worst = 0.0
    for _ in range(24 * 10):
        o = sim.observation(0)
        if o["hour"] == 0:
            t0 = time.perf_counter()
            compile_day(o, plan)
            worst = max(worst, (time.perf_counter() - t0) * 1000)
        sim.step([boat(o), starter(sim.observation(1))])
    assert worst < 100, f"{worst:.0f} ms to compile a day"


# --------------------------------------------------------------------- mutation check

def test_a_router_that_drops_its_last_task_is_caught():
    """The discipline from CLAUDE.md: break the thing on purpose and confirm the check fails.

    A verifier that only re-read the router's own claims would pass this happily.
    """
    tiles = board({(4, 3): plant("STRAWBERRY", 0, unwatered=1),
                   (3, 4): plant("STRAWBERRY", 0, unwatered=1)})
    obs = obs_for(tiles, day=5)
    tasks = daily_tasks(obs, Plan.boatlee_like())
    tasks = [t for t in tasks if t.tile in ((4, 3), (3, 4))]

    full = route(tasks, spawn_positions(0), turns_left=24)
    assert verify_day(obs, full.scripts, turns=24, crew=1).deaths == []

    sabotaged = {u: UnitScript(unit=u, ops=s.ops[:-1], stops=s.stops)
                 for u, s in full.scripts.items()}
    assert verify_day(obs, sabotaged, turns=24, crew=1).deaths, "dropping a water must show up"


# --------------------------------------------------------------------- _afford's stock ledger

def test_stock_already_held_is_spent_once_and_only_once():
    """`_afford` walks tasks in value order against a running ledger of what the farm holds.

    The ledger had two dead lines — `seeds.get(crop) - supplies.get(item) * 0` and, after a
    purchase, `shed[item] = shed.get(item, 0)`. Both were no-ops, so a unit of stock stayed on the
    books after the task that consumed it and the next task ordered nothing to replace it. The
    shopping list then under-buys, which at hour 1 is a planting pruned for want of a seed the
    agent decided it already had.
    """
    from agent.verify import _afford

    obs = obs_for(board(), day=3, seeds={"WHEAT": 1})
    tasks = [Task((x, 0), "PLANT", value=100 - x, needs={"SEED:WHEAT": 1}, kind="plant")
             for x in range(3)]

    kept, supplies, dropped = _afford(tasks, obs, cash=10_000)
    assert len(kept) == 3 and not dropped
    assert supplies == {"SEED:WHEAT": 2}, (
        f"one seed in stock and three plantings needs two bought, got {supplies}")


def test_stock_in_a_hand_counts_once_too():
    """Carried stock was added to `have` and never decremented, so every FERTILIZE task in the day
    saw the same single unit in the same hand and none of them ordered a second."""
    from agent.verify import _afford

    obs = obs_for(board(), day=3, inventories=[{"FERTILIZER": 1}])
    tasks = [Task((x, 0), "FERTILIZE", value=100 - x, needs={"FERTILIZER": 1}, kind="fertilize")
             for x in range(3)]

    _kept, supplies, _dropped = _afford(tasks, obs, cash=10_000)
    assert supplies == {"FERTILIZER": 2}, supplies


def test_stock_that_covers_the_day_is_not_re_bought():
    """The other direction: the ledger must not over-buy either."""
    from agent.verify import _afford

    obs = obs_for(board(), day=3, seeds={"WHEAT": 5})
    tasks = [Task((x, 0), "PLANT", value=100 - x, needs={"SEED:WHEAT": 1}, kind="plant")
             for x in range(3)]
    _kept, supplies, _dropped = _afford(tasks, obs, cash=10_000)
    assert supplies == {}, supplies


# --------------------------------------------------------------------- last-day unloading

def _ripe_farm(day):
    """A board with four ripe strawberry tiles, so the day contains real HARVEST work."""
    tiles = board({(x, 1): plant("STRAWBERRY", day - 10, watered=True, units=2)
                   for x in range(4)})
    return obs_for(tiles, day=day, hands=[(4, 4), (5, 4), (4, 5)])


def _drops(compiled) -> int:
    return sum(1 for s in compiled.scripts.values() for o in s.ops if o and o[0] == "DROP")


def test_the_last_day_sends_the_hands_home_to_unload():
    """Day 29 stops at hour 22 — the framework plays `episodeSteps - 1` turns — so the dusk drop is
    never followed by a market turn and anything still in a hand is never sold. Measured at +$7.6k
    and +$7.8k on two 80-game blocks vs `starter`."""
    compiled = compile_day(_ripe_farm(LAST_DAY), EMPTY_PLAN, hands=3, start_turn=1)
    assert _drops(compiled) > 0, "the season's last harvest is being left in the hands"


def test_an_ordinary_day_does_not_spend_turns_walking_home():
    """The same walk on day 12 is pure cost: dusk banks the goods for free and dawn sells them.
    Running it daily measured -$5k to -$9k a season, with milk per cow-day halving (0.48 -> 0.23) as
    the keepers spent their last turns walking instead of working.

    The day is deliberately near-empty, so the drop would comfortably *fit* if it were enabled —
    a busy day has no spare turns and would pass this whatever the gate said."""
    compiled = compile_day(_ripe_farm(12), EMPTY_PLAN, hands=3, start_turn=1)
    assert _drops(compiled) == 0, "the walk home is only worth its turns on the last day"
