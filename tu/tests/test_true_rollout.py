"""P2b — the compiler-in-the-loop rollout.

Written to the failure modes this line can actually produce, not to the API:

* **a rollout that quietly stops being a compiled day.** `test_breaking_the_router_step_changes_the
  _value` and `test_a_day_with_no_script_earns_nothing` drive that directly — if the C2/C3 step can
  be removed without the value moving, the whole build is `season.py` wearing a new name.
* **a dream that writes into the waking season** (E44 with the sign flipped).
  `test_rollout_does_not_leak_module_state` and `test_rollout_does_not_touch_shell_counters` read
  the real module globals across a rollout.
* **a knob that does nothing.** `test_true_mode_is_selected_by_the_const` and
  `test_horizon_truncation_changes_cost_not_correctness`.
"""

from __future__ import annotations

import json
import os

from agent import main_v4, planner, projection, season, true_rollout
from agent import tasks as tasks_module
from agent.plan import Cohort, Plan

VEC_PATH = "/private/tmp/r2_cand_vec.json"


def _plan() -> Plan:
    """The champion plan when it is on this machine, `boatlee_like` otherwise (see test_planner)."""
    if os.path.exists(VEC_PATH):
        from agent.plan import decode

        with open(VEC_PATH) as fh:
            return decode([float(x) for x in json.load(fh)])
    return Plan.boatlee_like()


_OBS: dict = {}


def _dawn_obs(day: int = 8, seed: int = 74000):
    """A real delivered observation at a real dawn, played by the real shell.

    Cached per day because the fixture is a whole game and every test in this file wants one; the
    JSON round-trip is what makes it a *snapshot* rather than a live alias into the simulator.
    """
    if day in _OBS:
        return _OBS[day]
    import kagsim

    from harness.registry import get

    plan = _plan()
    main_v4._STATE.clear()
    ours, theirs = main_v4.make_agent(plan), get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    for _ in range(719):
        a, b = sim.observation(0), sim.observation(1)
        if int(a["day"]) == day and int(a["hour"]) == 0:
            _OBS[day] = json.loads(json.dumps(a))
            return _OBS[day]
        sim.step([ours(a), theirs(b)])
    raise AssertionError(f"never reached day {day}")


def _world(day: int = 8):
    return true_rollout.World.from_obs(_dawn_obs(day), seat=0, plan=_plan())


# --------------------------------------------------------------------------- it runs, and it works

def test_rollout_runs_clean_and_is_deterministic():
    """A zero here is the whole build: `rollout_errors` is the swallowed-exception counter, and an
    exception on day one of the rollout returns the starting bank and looks like a bad candidate."""
    plan = _plan()
    a, b = _world(), _world()
    va, vb = true_rollout.rollout(a, plan), true_rollout.rollout(b, plan)
    assert a.counters.get("rollout_errors", 0) == 0, a.counters
    assert va == vb
    assert va > _dawn_obs()["farms"][0]["money"]
    assert a.state.day == season.LAST_DAY + 1


def test_the_rollout_really_compiles_every_day():
    """C1 -> C2 -> C3 ran on each imagined day, and the day was *played*, not scored.

    The counters are the E44 proof: `rollout_pruned` can only be non-zero if `verify.compile_day`
    ran its prune loop, and the shed/market movement can only happen if the compiled script was
    stepped through `FarmModel`.
    """
    plan = _plan()
    world = _world()
    true_rollout.rollout(world, plan)
    assert world.counters.get("rollout_pruned", 0) > 0
    assert "rollout_blocked_ops" in world.counters
    # Goods actually moved: harvested by a routed unit, dropped in the shed, sold on a hook. Read
    # off `sold`/`revenue`, which count units that *arrived* — market inventory is the wrong meter
    # because the town drains it faster than we fill it (CLAUDE.md: an order is not a sale).
    assert sum(world.state.sold.values()) > 50
    assert sum(world.state.revenue.values()) > 10_000
    assert {"STRAWBERRY", "MILK"} <= set(world.state.sold)


def test_all_three_selling_hooks_fire():
    """`main_v4._act` sells at dawn, on the turns a unit DROPs, and at hour 23. A rollout that
    quietly kept only the first still returns a plausible number, so the hooks are counted rather
    than assumed. Mutation: disable the hour-23 branch and `rollout_sell_hook_dusk` goes to zero
    while every money assertion in this file stays green — measured, it survived the first pass."""
    world = _world()
    true_rollout.rollout(world, _plan())
    days = season.LAST_DAY - _dawn_obs()["day"] + 1
    assert world.counters.get("rollout_sell_hook_dawn", 0) == days
    assert world.counters.get("rollout_sell_hook_dusk", 0) == days
    assert world.counters.get("rollout_sell_hook_drop", 0) > 0


def test_breaking_the_router_step_changes_the_value(monkeypatch):
    """The mutation check the design turns on: sabotage C2/C3 and the money must move.

    If a rollout whose compiled day is replaced by an empty script scores the *same* as one that
    routes properly, then nothing in here is reading the router and P2b is P2 with extra latency.
    """
    plan = _plan()
    honest = true_rollout.rollout(_world(), plan)

    real = true_rollout.compile_day

    def empty(obs, plan_, **kw):
        out = real(obs, plan_, **kw)
        for script in out.scripts.values():
            script.ops = [["PASS"] for _ in script.ops]
        return out

    monkeypatch.setattr(true_rollout, "compile_day", empty)
    crippled = true_rollout.rollout(_world(), plan)
    assert crippled < honest - 5_000, (crippled, honest)


def test_labour_is_routed_not_assumed():
    """A plan that asks for far more ground than a day can work must not be free.

    `season.py` prices this as a throughput fraction; here the extra tiles are planted, not watered,
    and die — which is exactly E83's uncontained quantity. Mutation: skip the replay loop and this
    fails, because the plantings would never be verified against a route.
    """
    plan = _plan()
    obs = _dawn_obs()
    free = [(x, y) for x in range(5) for y in range(5)
            if obs["farms"][0]["tiles"][y][x] is None]
    if len(free) < 5:
        # A board with no free ground cannot express the test; use the SW quadrant's plan tiles.
        free = [(x, y) for x in range(5) for y in range(5, 10)
                if obs["farms"][0]["tiles"][y][x] is None][:10]
    assert free, "no free ground on the fixture board"
    greedy = plan.__class__(**{**plan.__dict__, "cohorts": plan.cohorts + (
        Cohort(crop="STRAWBERRY", quadrant="NW", n_tiles=len(free), plant_day=9,
               replant=False, tiles=tuple(free)),)})
    base = true_rollout.rollout(_world(), plan)
    piled = true_rollout.rollout(_world(), greedy)
    assert base != piled


# --------------------------------------------------------------------------- the sandbox

def test_rollout_does_not_leak_module_state():
    """The dream must not write into the waking season.

    `daily_tasks` commits `projection.redirect` / `counter_mix` decisions to per-seat module dicts so
    they are idempotent for the season. A rollout that ran them unguarded would have an imagined
    day 14 decide a redirect the real day 7 then treats as already made — E44's failure with the
    sign flipped. Mutation: delete `_sandbox` from `rollout` and this fails on `_CACHE` alone.
    """
    plan = _plan()
    _dawn_obs()                                   # play a real game first, so the globals are warm
    before = (repr(projection._REDIRECTS), repr(projection._CONTESTED),
              repr(projection._CACHE), repr(tasks_module.STATS))
    true_rollout.rollout(_world(), plan)
    after = (repr(projection._REDIRECTS), repr(projection._CONTESTED),
             repr(projection._CACHE), repr(tasks_module.STATS))
    assert before == after


def test_rollout_does_not_touch_shell_counters():
    """`projection._count` mirrors into `main_v4._STATE`; an imagined day must not be counted."""
    plan = _plan()
    _dawn_obs()
    before = repr(main_v4._STATE)
    true_rollout.rollout(_world(), plan)
    assert repr(main_v4._STATE) == before


def test_rollout_does_not_reset_the_planners_season():
    """The bug this cost: an *empty* sandboxed `_STATE` makes `main_v4._state` think a new season
    has started, and it then calls `branches.reset` / `planner.reset` on the live modules. Measured
    before the fix: `planner_max_moves=1` committed **11** deviations while `planner.log()` returned
    `()`. Mutation: seed the sandbox with `{}` and this fails."""
    from agent import branches

    plan = _plan()
    planner.reset()
    planner._SEASON[0] = {"plan": plan, "base": plan, "moves": 3, "log": [(6, "x", "y", 1.0)],
                          "day": 6, "locks": {}}
    branches._SEASON[0] = {"marker": True}
    try:
        true_rollout.rollout(_world(), plan)
        assert planner._SEASON.get(0, {}).get("moves") == 3
        assert planner.log(0) == ((6, "x", "y", 1.0),)
        assert branches._SEASON.get(0) == {"marker": True}
    finally:
        planner.reset()
        branches.reset()


def test_the_sandbox_carries_committed_decisions_in():
    """Isolation is not amnesia: a redirect the season has really made is visible to the rollout."""
    projection._REDIRECTS[(0, "crop")] = {"MARKER": "SEEN"}
    try:
        with true_rollout._sandbox():
            assert projection._REDIRECTS[(0, "crop")] == {"MARKER": "SEEN"}
            projection._REDIRECTS[(0, "crop")]["MARKER"] = "MUTATED"
        assert projection._REDIRECTS[(0, "crop")] == {"MARKER": "SEEN"}
    finally:
        projection._REDIRECTS.pop((0, "crop"), None)


def test_tasks_stats_survive_the_sandbox():
    """`tasks.STATS` is indexed directly, so an emptied dict is a KeyError rather than a clean slate
    — measured: it was the first thing the rollout crashed on."""
    plan = _plan()
    world = _world()
    true_rollout.rollout(world, plan)
    assert world.counters.get("rollout_errors", 0) == 0
    assert set(tasks_module.STATS) == {"tick_waters", "tick_waters_fert_today"}


# --------------------------------------------------------------------------- the knobs

def test_true_mode_is_selected_by_the_const():
    """Mutation: return `value` unconditionally and the whole of P2b is dead code."""
    assert planner.valuer({}) is planner.value
    assert planner.valuer({"planner_value": "fast"}) is planner.value
    assert planner.valuer({"planner_value": "true"}) is not planner.value
    assert isinstance(planner.base_state(_dawn_obs(), 0, _plan(), {}), season.SeasonState)
    assert isinstance(planner.base_state(_dawn_obs(), 0, _plan(), {"planner_value": "true"}),
                      true_rollout.World)


def test_the_true_value_is_not_the_fast_one():
    """Two models of the same plan from the same dawn must disagree; if they agree exactly, one of
    them is not being called."""
    plan = _plan()
    obs = _dawn_obs()
    sample = planner.draws(8, 0, 1)
    fast = planner.value(season.SeasonState.from_obs(obs, seat=0, plan=plan), plan, sample)
    true = planner.valuer({"planner_value": "true"})(
        true_rollout.World.from_obs(obs, seat=0, plan=plan), plan, sample)
    assert fast != true


def test_horizon_truncation_hands_off_to_the_fast_model():
    """`planner_true_days` compiles a prefix and hands the tail to P1 — and the hand-off tells P1
    which cohorts are standing. Mutation: drop `_sync_cohort_state` and the tail re-fills every
    cohort at full width on its first day, which shows up as a strictly richer season."""
    plan = _plan()
    short = _world()
    truncated = true_rollout.rollout(short, plan, days=6)
    assert short.counters.get("rollout_fast_tail_days", 0) > 0
    assert short.counters.get("rollout_errors", 0) == 0

    # The hand-off, behaviourally. `cohort_state` is what tells P1's tail which cohorts are already
    # standing; the true days never touch it, so a rollout handed a *cleared* one must rebuild it
    # from the board and land on exactly the same money. Without the rebuild the tail re-fills every
    # cohort at full width on its first day — a free harvest the search would learn to buy.
    #
    # Asserting that `_sync_cohort_state` populates a dict is not enough and it was the first
    # version: `SeasonState.from_obs` has already populated it at this dawn, so deleting the *call
    # site* changed nothing and the mutant survived.
    cleared = _world()
    cleared.state.cohort_state = {}
    assert true_rollout.rollout(cleared, plan, days=6) == truncated

    full = _world()
    true_rollout.rollout(full, plan)
    assert full.counters.get("rollout_fast_tail_days", 0) == 0


def test_the_view_is_a_delivered_observation():
    """The rollout's obs shim carries every field the daily stack reads, at the delivered surface
    (CLAUDE.md: verify against the surface the runner uses, not the one that is easy to reach)."""
    world = _world()
    view = world.view(1)
    assert view["player"] == 0 and view["hour"] == 1
    assert view["step"] == view["day"] * 24 + 1
    farm = view["farms"][0]
    assert set(farm) >= {"tiles", "farmer", "hands", "money", "unlocked_quadrants"}
    assert len(view["farms"]) == 2 and view["farms"][1] is not farm
    assert set(view["private"]) == {"shed", "seeds", "inventories"}
    assert view["market"]["prices"] and view["market"]["inventory"]
    # The task generator must run against it without a shim of its own.
    assert tasks_module.daily_tasks(view, _plan(), day=view["day"], turn=1)


def test_crew_size_proxy_is_bounded():
    """The one scalar-labour assumption left (see the module's assumption 4)."""
    assert true_rollout._hands_for([], 10_000) == 0
    class T:                                          # a task-shaped stub; only `.op` is read
        op = "WATER"
    assert true_rollout._hands_for([T()] * 200, 10_000) == 13
    assert true_rollout._hands_for([T()] * 200, 0.0) == 0


# --------------------------------------------------------------------------- the E83 worked example

def test_e83_worked_example_is_priced_negative():
    """E83's named counter-example, re-priced. This is the regression the whole module exists for.

    E83: seed 74000, seat 1, day 6 — the fast model offered `cohort1_+3` (a strawberry cohort moved
    three days) at **+$2,446** and it cost **−$16,200** in play, with `strawberry_per_plant`
    6.33 -> 4.73. The true rollout must price the same candidate at the same dawn **negative**.

    Scoped honestly: this pins the sign on *this* move, not the model's calibration — the full
    20-move set still comes back corr −0.61 (see the module docstring). A version of this rollout
    that stopped reading the router would price it positive again, which is what the test is for.
    """
    import kagsim
    import pytest

    from harness.registry import get

    if not os.path.exists(VEC_PATH):
        pytest.skip("the champion genome is not on this machine; see test_planner._plan")

    plan = _plan()
    main_v4._STATE.clear()
    ours, theirs = main_v4.make_agent(plan), get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 74000})
    obs = None
    for _ in range(719):
        o0, o1 = sim.observation(0), sim.observation(1)
        if int(o1["day"]) == 6 and int(o1["hour"]) == 0:
            obs = json.loads(json.dumps(o1))
            break
        sim.step([theirs(o0), ours(o1)])
    assert obs is not None

    consts = {**plan.consts, "planner": 1, "planner_value": "true", "planner_menu_cap": 30}
    ctx = planner.ctx_from_obs(obs, 1)
    cand = next(c for c in planner.menu(plan, ctx, consts) if c.name == "cohort1_+3")
    evaluate = planner.valuer(consts)
    base = planner.base_state(obs, 1, plan, consts)
    sample = planner.draws(6, 1, 1)
    gain = evaluate(base, cand.plan, sample) - evaluate(base, plan, sample)
    assert gain < 0, f"E83's −$16,200 move priced at {gain:+,.0f}"

    fast = planner.value(season.SeasonState.from_obs(obs, seat=1, plan=plan), cand.plan, sample) \
        - planner.value(season.SeasonState.from_obs(obs, seat=1, plan=plan), plan, sample)
    assert fast > 0, "the fast model is supposed to get this one wrong; the contrast is the point"
