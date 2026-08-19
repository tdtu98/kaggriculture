"""P2 — the dawn planner.

The tests are written to the two failure modes this line has actually produced, not to the API:

* **a change that never fired** (E44). `test_planner_fires_in_play` reads counters off a *played*
  season, not off a helper that returned the right value, and `test_planner_off_is_identical`
  proves the flag-off path by object identity rather than by equal money.
* **a search that chooses something it cannot execute.** `test_never_digs_live_ground` and
  `test_never_retimes_owned_land` drive the gate with hostile patches rather than asserting that a
  legal one passes.

Each mutation the module could suffer is named next to the test that catches it.
"""

from __future__ import annotations

import json
import os

from agent import main_v4, planner, season
from agent.plan import NEVER, Cohort, Plan, quadrant_tiles

VEC_PATH = "/private/tmp/r2_cand_vec.json"


def _plan() -> Plan:
    """The champion plan when it is on this machine, `boatlee_like` otherwise.

    The genome lives outside the repo (E77 recorded the path), so the suite must not depend on it;
    what it must not do is *silently* test something else, hence the explicit fallback.
    """
    if os.path.exists(VEC_PATH):
        from agent.plan import decode

        with open(VEC_PATH) as fh:
            return decode([float(x) for x in json.load(fh)])
    return Plan.boatlee_like()


def _play(plan: Plan, seed: int = 74000, opponent: str = "starter", turns: int = 719):
    """One real season through `kagsim`, returning `(money, effects, planner log)`."""
    import kagsim

    from harness.registry import get

    # Deliberately *not* `planner.reset()`: clearing `main_v4._STATE` is what a new game does, and
    # the reset has to ride on that edge for `test_season_state_is_cleared_between_games` to be
    # testing the shipped path rather than the test's own housekeeping.
    main_v4._STATE.clear()
    ours = main_v4.make_agent(plan)
    theirs = get(opponent).build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    for _ in range(turns):
        a, b = sim.observation(0), sim.observation(1)
        sim.step([ours(a), theirs(b)])
    effects = dict((main_v4._STATE.get(0) or {}).get("effects") or {})
    return sim.money(0), effects, planner.log(0)


# --------------------------------------------------------------------------- the flag

def test_planner_off_is_identical_object():
    """Flag off returns the *same* plan object and spends nothing.

    Both halves are load-bearing. Identity catches `replace(plan)`, which every money-based test
    would miss. The empty counter dict catches a flag that is read but not obeyed: the first version
    asserted identity alone, and a mutant that ignored the flag *still passed it* — the search ran,
    failed on the stub observation, counted `planner_errors` and handed back the same object.
    """
    plan = _plan()
    obs = {"farms": [{"money": 5000, "tiles": [[None] * 10 for _ in range(10)],
                      "unlocked_quadrants": ["NW"]}, {}], "day": 9, "hour": 0}
    seen: dict = {}
    planner.reset()
    assert planner.apply(obs, plan, 0, 9, 0,
                         lambda k, n=1: seen.__setitem__(k, seen.get(k, 0) + n)) is plan
    assert seen == {}


def test_planner_off_is_byte_identical_in_play(monkeypatch):
    """Ten paired seasons, flag off against a build with the hook stubbed out entirely.

    The comparison has to be against a *hookless* agent, not against another flag-off agent. The
    first version compared `planner=0` with an implicit default, i.e. two runs of the same code, and
    a mutant that ignored the flag altogether passed it — both arms deviated identically. Stubbing
    `planner.apply` to the identity is the only arm that is genuinely the pre-P2 agent.
    """
    plan = _plan()
    off = [_play(plan.with_consts(planner=0), seed=s)[0] for s in range(74000, 74010)]
    monkeypatch.setattr(main_v4.season_planner, "apply",
                        lambda obs, plan, *a, **k: plan)
    hookless = [_play(plan, seed=s)[0] for s in range(74000, 74010)]
    assert off == hookless
    assert all(m > 3000 for m in off)


def test_planner_fires_in_play():
    """Counters, read off a played season. A zero here is an unfinished wiring, not a null result."""
    money, effects, log = _play(_plan().with_consts(planner=1))
    assert effects.get("planner_dawns", 0) >= 10
    assert effects.get("planner_candidates", 0) > 50
    assert effects.get("planner_deviations", 0) >= 1
    assert effects.get("fallbacks", 0) == 0
    assert effects.get("planner_errors", 0) == 0
    assert money > 3000
    assert log and all(kind != "follow" for _d, _n, kind, _g in log)
    # Every committed deviation is attributed to exactly one decision-type counter.
    kinds = sum(v for k, v in effects.items() if k.startswith("planner_kind_"))
    assert kinds == effects["planner_deviations"]


# --------------------------------------------------------------------------- the menu

def test_menu_always_offers_follow_first():
    plan = _plan()
    ctx = planner.Ctx(day=8, money=20_000, unlocked=frozenset({"NW"}), animals=4, seat=0)
    items = planner.menu(plan, ctx, plan.consts)
    assert items[0].name == "follow" and items[0].plan is plan
    assert len(items) <= planner.DEFAULTS["planner_menu_cap"]


def test_menu_prunes_by_cash():
    """A candidate the wallet cannot pay for is not offered. Mutation: drop the `ctx.money` guards
    and the menu grows — which is the mode where the search picks a plan the dawn cannot buy."""
    plan = _plan()
    rich = planner.Ctx(day=8, money=50_000, unlocked=frozenset({"NW", "NE", "SW"}), animals=4,
                       seat=0)
    poor = planner.Ctx(day=8, money=1.0, unlocked=frozenset({"NW", "NE", "SW"}), animals=4, seat=0)
    assert len(planner.menu(plan, rich, plan.consts)) > len(planner.menu(plan, poor, plan.consts))


def test_menu_direction_lock_drops_the_reverse():
    """A target already pushed one way may not be pushed back. Mutation: ignore `locks` and both
    directions reappear — the oscillation the module docstring measures."""
    plan = _plan()
    ctx = planner.Ctx(day=8, money=50_000, unlocked=frozenset({"NW", "NE", "SW"}), animals=4,
                      seat=0)
    free = planner.menu(plan, ctx, plan.consts)
    target = next(c.target for c in free if c.kind == "cohort_shift")
    locked = planner.menu(plan, ctx, plan.consts, {target: -1})
    assert {c.name for c in free if c.target == target and c.direction == 1}
    assert not {c.name for c in locked if c.target == target and c.direction == 1}
    assert {c.name for c in locked if c.target == target and c.direction == -1}


def test_menu_never_plants_past_the_buzzer():
    plan = _plan()
    ctx = planner.Ctx(day=27, money=90_000, unlocked=frozenset({"NW", "NE", "SW", "SE"}),
                      animals=8, seat=0)
    for cand in planner.menu(plan, ctx, plan.consts):
        for cohort in cand.plan.cohorts:
            assert cohort.plant_day <= season.LAST_DAY


# --------------------------------------------------------------------------- the forward-only gate

def _tiny_plan(**kw) -> Plan:
    base = dict(pasture_tiles=((0, 0),), land_days={"NE": 6, "SW": 10, "SE": NEVER},
                herd=(("COW", 1),),
                cohorts=(Cohort("WHEAT", "NW", 3, 2, tiles=((1, 0), (2, 0), (3, 0))),
                         Cohort("MELON", "NW", 2, 20, tiles=((4, 0), (0, 1)))),
                hands="auto", consts={})
    base.update(kw)
    return Plan(**base)


def test_never_digs_live_ground():
    """A patch that re-tiles a cohort already in the ground is rejected. Mutation: swap
    `_forward_only` for `after.validate()` alone and this passes a plan that digs a live crop."""
    before = _tiny_plan()
    stolen = Cohort("MELON", "NW", 2, 20, tiles=((1, 0), (2, 0)))   # cohort0's live tiles
    after = Plan(**{**before.__dict__, "cohorts": (before.cohorts[0], stolen)})
    assert planner._forward_only(before, after, 9, 2, frozenset({"NW", "NE"}))


def test_never_edits_a_sown_cohort():
    before = _tiny_plan()
    edited = list(before.cohorts)
    edited[0] = Cohort("WHEAT", "NW", 3, 25, tiles=edited[0].tiles)
    after = Plan(**{**before.__dict__, "cohorts": tuple(edited)})
    assert planner._forward_only(before, after, 9, 2, frozenset({"NW"}))


def test_never_retimes_owned_land():
    """The one rule this module adds to O1's gate, in both directions."""
    before = _tiny_plan()
    # A *future* day, so the "moved into the past" rule cannot mask the ownership rule — with both
    # firing at once, deleting the ownership check leaves the test green (measured: mutant 6 of
    # `/private/tmp/p2/mutants.sh` survived exactly this way).
    owned = Plan(**{**before.__dict__, "land_days": {"NE": 12, "SW": 12, "SE": NEVER}})
    assert planner._forward_only(before, owned, 9, 2, frozenset({"NW", "NE"}))       # NE is bought
    assert not planner._forward_only(before, owned, 9, 2, frozenset({"NW"}))         # NE is not
    past = Plan(**{**before.__dict__, "land_days": {"NE": 6, "SW": 3, "SE": NEVER}})
    assert planner._forward_only(before, past, 9, 2, frozenset({"NW"}))              # day 3 < 9


def test_gate_admits_a_pending_shift():
    before = _tiny_plan()
    moved = list(before.cohorts)
    moved[1] = Cohort("MELON", "NW", 2, 18, tiles=moved[1].tiles)
    after = Plan(**{**before.__dict__, "cohorts": tuple(moved)})
    assert planner._forward_only(before, after, 9, 2, frozenset({"NW"})) == []


# --------------------------------------------------------------------------- draws and value

def test_draws_are_common_and_reproducible():
    """The same dawn gives the same sample twice, a different dawn a different one, and the
    held-out sample is disjoint in construction from the selection one."""
    a = planner.draws(8, 0, 4)
    assert a == planner.draws(8, 0, 4)
    assert a != planner.draws(9, 0, 4)
    assert a != planner.draws(8, 1, 4)
    assert a != planner.draws(8, 0, 4, planner.CONFIRM_SALT)
    assert all(len(d) == planner.DRAW_LENGTH for d in a)


def test_value_is_deterministic_and_uses_every_draw():
    """Mutation: make `value` read only `sample[0]` and the second assertion fails — which is the
    silent way `planner_draws` becomes a knob that does nothing."""
    import kagsim

    from harness.registry import get

    plan = _plan()
    ours, theirs = main_v4.make_agent(plan), get("starter").build()
    main_v4._STATE.clear()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 74000})
    obs = None
    for _ in range(719):
        a, b = sim.observation(0), sim.observation(1)
        if int(a["day"]) == 8 and int(a["hour"]) == 0:
            obs = a
            break
        sim.step([ours(a), theirs(b)])
    assert obs is not None
    base = season.SeasonState.from_obs(obs, seat=0, plan=plan)
    sample = planner.draws(8, 0, 4)
    assert planner.value(base, plan, sample) == planner.value(base, plan, sample)
    # The mean of the four draws, computed one draw at a time. Comparing against `sample[:1]`
    # instead is not enough: a `value` that iterates `sample[:1]` but still divides by `len(sample)`
    # returns a *different* number and passes that check while using one draw (mutant 7 survived
    # exactly this). The per-draw spread is what proves the extra draws carry information.
    per = [planner.value(base, plan, [d]) for d in sample]
    assert planner.value(base, plan, sample) == sum(per) / len(per)
    assert len(set(per)) > 1
    assert len(set(planner.value(base, plan, [d]) for d in planner.draws(8, 0, 4)[:2])) > 0


def test_search_returns_follow_when_nothing_clears_the_bar():
    """A prohibitive `planner_min_gain` must produce the incumbent plan, by identity.

    This is the test that catches a broken rollout value: mutation — make `value` return a constant
    — and `search` can no longer distinguish anything, so with a normal bar it also returns follow,
    while `test_planner_fires_in_play`'s `planner_deviations >= 1` fails.
    """
    import kagsim

    from harness.registry import get

    plan = _plan()
    ours, theirs = main_v4.make_agent(plan), get("starter").build()
    main_v4._STATE.clear()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 74000})
    obs = None
    for _ in range(719):
        a, b = sim.observation(0), sim.observation(1)
        if int(a["day"]) == 8 and int(a["hour"]) == 0:
            obs = a
            break
        sim.step([ours(a), theirs(b)])
    ctx = planner.ctx_from_obs(obs, 0)
    base = season.SeasonState.from_obs(obs, seat=0, plan=plan)
    consts = {**plan.consts, "planner_min_gain": 10 ** 9}
    chosen, gain = planner.search(base, plan, ctx, consts, lambda *_a, **_k: None)
    assert chosen.plan is plan and gain == 0.0


# --------------------------------------------------------------------------- safety

def test_a_broken_search_never_crashes_the_turn(monkeypatch):
    """C4's fallback shape: a planner that raises loses the dawn's search, not the episode."""
    plan = _plan().with_consts(planner=1)
    monkeypatch.setattr(planner, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    seen: dict = {}
    obs = {"farms": [{"money": 5000, "tiles": [[None] * 10 for _ in range(10)],
                      "unlocked_quadrants": ["NW"], "hands": []}, {}], "day": 9, "hour": 0,
           "market": {"inventory": {}}, "town": {"unlocked_shops": []}, "private": {}}
    planner.reset()
    out = planner.apply(obs, plan, 0, 9, 0, lambda k, n=1: seen.__setitem__(k, seen.get(k, 0) + n))
    assert out is plan
    assert seen.get("planner_errors") == 1


def test_season_state_is_cleared_between_games():
    """Two games in one process. Mutation: drop `planner.reset(seat)` from `main_v4._state` and the
    second season inherits the first's deviations — the `projection._CONTESTED` defect (E79) in a
    new module."""
    plan = _plan().with_consts(planner=1)
    _m1, _e1, log1 = _play(plan, seed=74000, turns=719)
    assert max(day for day, _n, _k, _g in log1) > 9
    _m2, _e2, log2 = _play(plan, seed=74001, turns=240)     # ten days
    assert all(day <= 10 for day, _n, _k, _g in log2), log2


def test_moves_are_capped():
    plan = _plan().with_consts(planner=1, planner_max_moves=1, planner_min_gain=1)
    _money, effects, log = _play(plan, seed=74000)
    assert effects.get("planner_deviations", 0) <= 1
    assert len(log) <= 1
