"""C4: the shell that turns the compiler into an agent.

Three properties, in descending order of what they cost when broken:

1. **It never raises.** One exception is an ERROR for the whole episode and scores the $3,000
   starting bank — measured, 0-40 (E21). This is why every test here runs a real season rather than
   a hand-built observation.
2. **Its ops land.** `blocked_ops` is the compiler's health counter: a script that has drifted from
   the board spends turns on bare ground and looks exactly like a bad strategy in the money.
3. **It fits the turn.** 1 s per turn, with an ERROR if exceeded.

Money is deliberately *not* asserted here. That is C5's gate and belongs to the plan, not the shell.
"""

from __future__ import annotations

import time

import pytest

import kagsim
from agent.main_v4 import (LATE_COHORT_DAYS, _feed_reserve, _sell_orders, agent,
                           counters, make_agent)
from agent.plan import Plan
from harness.registry import get
from agent.plan import quadrant_of
from tests.test_tasks import PRICES, animal, board, obs_for, plant

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def play(seed: int, plan=None, steps: int = 719):
    """A full season against `starter`, returning (money, blocked_ops, agent)."""
    ours = make_agent(plan or Plan.boatlee_like())
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    previous = blocked = 0
    for _ in range(steps):
        obs = sim.observation(0)
        action = ours(obs)
        passes = sum(1 for op in [action["farmer"]] + list(action["hands"])
                     if op and op[0] == "PASS")
        sim.step([action, other(sim.observation(1))])
        noop = sim.stats(0)["actions_noop"]
        blocked += max(0, (noop - previous) - passes)
        previous = noop
    return sim.money(0), blocked, ours


# --------------------------------------------------------------------- it plays

def test_a_full_season_runs_without_a_single_fallback():
    money, _blocked, _ours = play(seed=3)
    assert counters(0)["fallbacks"] == 0, "an exception forfeits the episode"
    assert money > 3_000, f"${money:,.0f} is the starting bank — the agent did nothing"


def test_it_plays_both_seats():
    """Seat 1 sees the same shared observation with `player` set; a shell that assumed seat 0 would
    farm the opponent's board."""
    ours = make_agent()
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 4})
    for _ in range(719):
        sim.step([other(sim.observation(0)), ours(sim.observation(1))])
    assert counters(1)["fallbacks"] == 0
    assert sim.money(1) > 3_000


def test_an_exception_inside_the_shell_becomes_a_pass_not_a_crash():
    """The fallback has to survive a broken plan, because the alternative is scoring $3,000."""
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    obs = obs_for(board(), day=2)
    obs["farms"][0]["hands"] = [[5, 4], [4, 5]]
    action = agent(obs, Exploding())

    assert action["farmer"] == ["PASS"]
    assert action["hands"] == [["PASS"], ["PASS"]], "one op per hand, or the turn is malformed"
    assert action["market"] == []


# --------------------------------------------------------------------- its ops land

@pytest.mark.parametrize("seed", [3, 11])
def test_blocked_ops_stay_inside_the_gate(seed):
    """C4's bar is <= 12 a season, against Boatlee's own ~10 (E55).

    This is the test that caught the expensive ones: 538 blocked ops a season from planning against
    money that was already spent, and a further 39 from two hands reaching for the same seed.
    """
    _money, blocked, _ours = play(seed=seed)
    assert blocked <= 12, f"{blocked} blocked ops"


def test_two_hands_never_reach_for_the_same_seed():
    """`PLANT` validation is atomic per crop per turn: if requests exceed seeds held, *every*
    request for that crop is dropped. Two hands planting wheat with one seed plant nothing at
    all — and each wastes the WATER chained behind it."""
    ours = make_agent()
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 3})
    collisions = 0
    for _ in range(719):
        obs = sim.observation(0)
        action = ours(obs)
        seeds = dict(obs["private"]["seeds"])
        wanted: dict = {}
        for op in [action["farmer"]] + list(action["hands"]):
            if op and op[0] == "PLANT":
                wanted[op[1]] = wanted.get(op[1], 0) + 1
        for crop, n in wanted.items():
            if n > int(seeds.get(crop, 0)):
                collisions += 1
        sim.step([action, other(sim.observation(1))])
    assert collisions == 0, f"{collisions} turns asked for more seed than the farm held"


# --------------------------------------------------------------------- market rules

def test_wheat_needed_for_feed_is_not_sold():
    """Wheat is also feed. Selling the shed flat left the keepers with nothing to carry and the
    herd escaped — twice in a measured season, at ~$450 a head."""
    tiles = board({(4, 3): animal("COW", 0), (3, 4): animal("COW", 0)})
    obs = obs_for(tiles, day=5, shed={"WHEAT": 6, "MILK": 3})
    orders = _sell_orders(obs, Plan.boatlee_like(), 0)

    wheat = [o for o in orders if o[1] == "WHEAT"]
    assert _feed_reserve(obs, 0)["WHEAT"] == 4, "two days of feed for two cows"
    assert not wheat or wheat[0][2] <= 2, f"sold into the feed reserve: {wheat}"
    assert any(o[1] == "MILK" for o in orders), "…but produce is still sold"


def test_everything_goes_at_the_buzzer():
    """Stock left unsold at the end is worth nothing, floors included."""
    obs = obs_for(board(), day=29, shed={"MELON": 20, "WOOL": 10})
    obs["step"] = 705
    orders = _sell_orders(obs, Plan.boatlee_like(), 0)
    assert {o[1] for o in orders} == {"MELON", "WOOL"}
    assert all(o[2] == 20 or o[2] == 10 for o in orders)


def test_the_order_list_never_exceeds_what_the_market_reads():
    """`maxMarketOrdersPerTurn` is 10 and the queue is *truncated* past it, silently — so an
    over-long list does not error, it just loses its tail."""
    ours = make_agent()
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 3})
    for _ in range(719):
        action = ours(sim.observation(0))
        assert len(action["market"]) <= 10, action["market"]
        sim.step([action, other(sim.observation(1))])


def test_livestock_is_not_bought_faster_than_it_can_be_fed():
    """An animal eats a wheat a day and escapes after two missed dusks. Buying beyond the feed
    supply re-bought a herd three times in one season and read as a strategy failure.

    "Beyond the feed supply" counts the wheat the farm can *buy* as well as the wheat it grows —
    `BUY_PRODUCT` takes WHEAT, and a rule that ignored that refused animals on days the wallet held
    thousands and landed the herd on day 18.6 against a plan that said day 5 (E66). So a barren farm
    with no money still buys nothing, and a barren farm with money buys what its money can feed.
    """
    from agent.main_v4 import FEED_BUFFER_DAYS, _animal_orders, _feedable_animals

    broke = obs_for(board(), day=6, shed={})
    assert _feedable_animals(broke, 0) == 0
    assert _animal_orders(broke, Plan.boatlee_like(), 0, 6, budget=0) == []

    # $400 buys the cow and leaves nothing to feed it: the guard is on the money *left over*.
    assert _animal_orders(broke, Plan.boatlee_like(), 0, 6, budget=400) == []

    # Enough over the purchase price to cover the buffer, and the same barren farm buys — a COW,
    # not the sheep the schedule lists first: `_purchase_order` spends short money on the best
    # head per dollar.
    wheat = PRICES["WHEAT"]
    funded = 400 + wheat * FEED_BUFFER_DAYS
    assert _animal_orders(broke, Plan.boatlee_like(), 0, 6, budget=funded) == \
        [["BUY_ANIMAL", "COW", 1]]

    tiles = board({(x, 0): plant("WHEAT", 0) for x in range(10)})
    fed = obs_for(tiles, day=6, shed={"WHEAT": 10})
    assert _feedable_animals(fed, 0) >= 8
    assert _animal_orders(fed, Plan.boatlee_like(), 0, 6, budget=5_000)


def test_the_wheat_cohort_is_resown_at_a_rate_in_a_real_season():
    """E66 mechanism 3, proved *in play* rather than on a task list.

    The counter that matters is the one the change is supposed to move: after its first cycle, the
    cycling block is never re-sown faster than a wave a day, whatever the day looks like. Asserting
    the resulting tile count instead would be asserting the whole farm — the block is emptied by
    harvests, weeds, thirst and C3's pruning, and on 49000:49020 the stagger moves collapse days
    (5-25 with three tiles or fewer) 5.2 -> 3.1 vs `starter` and 2.6 -> 1.1 vs `boatlee` while the
    mean count barely moves. The rate is the mechanism; the tile count is a season away from it.
    """
    from agent.tasks import cohort_opens, replant_cycle, wave_size

    plan = Plan.boatlee_like()
    wheat = [c for c in plan.cohorts if c.replant and c.crop == "WHEAT"]
    assert wheat, "the calibration plan must have a cycling wheat block"
    caps = {id(c): wave_size(c, 99, cohort_opens(plan, c)) for c in wheat}
    assert all(0 < caps[id(c)] < len(c.tiles) for c in wheat), caps

    me = make_agent(plan)
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 49113})
    other = get("starter").build()
    per_day: dict = {}
    for _ in range(719):
        o0, o1 = sim.observation(0), sim.observation(1)
        action = me(o0)
        day = int(o0["day"])
        for op, pos in zip([action["farmer"]] + list(action["hands"]),
                           [tuple(o0["farms"][0]["farmer"])]
                           + [tuple(h) for h in o0["farms"][0]["hands"]]):
            if op and op[0] == "PLANT" and len(op) > 1 and op[1] == "WHEAT":
                for c in wheat:
                    if pos in c.tiles:
                        per_day.setdefault((day, id(c)), 0)
                        per_day[(day, id(c))] += 1
        sim.step([action, other(o1)])

    over = {k: v for k, v in per_day.items()
            if v > caps[k[1]] and k[0] >= min(cohort_opens(plan, c) + replant_cycle(c.crop)
                                              for c in wheat if id(c) == k[1])}
    assert not over, f"a cycling block was re-sown as a batch: {over}"
    assert sum(per_day.values()) >= 25, "the block was not actually being cycled"


def test_short_money_buys_the_cow_before_the_sheep():
    """The cow is the cheapest head on the board and the best earner per dollar (`$0.60/day/$`
    against the sheep's `$0.53`), and the buy loop stops at the first head it cannot afford. Taking
    the plan's schedule order literally therefore spent the early cash on sheep and left the cows
    waiting on the wallet: first animal day 2, first COW day 11 (E66 re-gate)."""
    from agent.main_v4 import _animal_orders, _purchase_order

    tiles = board({(x, 0): plant("WHEAT", 0) for x in range(10)})
    fed = obs_for(tiles, day=6, shed={"WHEAT": 40})
    plan = Plan.boatlee_like()
    assert plan.herd[0][0] == "SHEEP" and any(s == "COW" for s, _d in plan.herd)
    assert [s for s, _d in _purchase_order(plan, 6)][0] == "COW"

    # $450 is a cow and not a sheep, and the sheep standing first in the schedule does not block it.
    assert _animal_orders(fed, plan, 0, 6, budget=450) == [["BUY_ANIMAL", "COW", 1]]
    # Nine cows before the first sheep, then the sheep — the plan's own count, in value order.
    got = [o[1] for o in _animal_orders(fed, plan, 0, 6, budget=100_000)]
    assert got[:9] == ["COW"] * 9 and set(got[9:]) == {"SHEEP"}


def test_a_mixed_herd_keeps_the_schedule_order_so_the_slots_still_match():
    """Slot k of the herd is served by structure k, and `_structure_tasks` picks that structure's
    *kind* from the plan's own order. Reordering is only safe while every due head wants the same
    kind: a GOOSE bought against a PASTURE slot has no COOP to stand in and never leaves the shed.
    So a mixed herd keeps schedule order, and an unaffordable head stops the queue there."""
    from agent.main_v4 import _animal_orders, _purchase_order
    from agent.plan import decode, encode_fields, NEVER

    plan = decode(encode_fields(
        n_pastures=6, land_days={"NE": NEVER, "SW": NEVER, "SE": NEVER},
        n_sheep=2, sheep_start=0, n_cows=2, cow_start=0, n_geese=2, geese_start=0,
        animals_per_day=3, cohorts=[], hands_mode="curve", hands_start=4, hands_cap=12,
        hands_ramp=10, sell_floor_wool=0.35, sell_floor_melon=0.35,
        release_pressure=70, frontrun_lead=10))
    assert {s for s, _d in plan.herd} == {"COW", "SHEEP", "GOOSE"}
    assert _purchase_order(plan, 6) == [(s, d) for s, d in plan.herd if d <= 6]

    tiles = board({(x, 0): plant("WHEAT", 0) for x in range(10)})
    fed = obs_for(tiles, day=6, shed={"WHEAT": 40})
    # The schedule opens with COW (sorted by day then name); $350 cannot pay for it, and nothing
    # further down the list is bought instead.
    assert plan.herd[0][0] == "COW"
    assert _animal_orders(fed, plan, 0, 6, budget=350) == []


def test_no_animal_is_bought_after_it_can_no_longer_repay_its_price():
    """A cow bought on day 25 is placed on 26 and would first produce on day 34: the $400 buys
    nothing at all. E66's re-gate measured 7 such purchases a season, so this is checked **in play**
    and not only on the arithmetic — the arithmetic was right before and the orders still went out.

    `kaggriculture.py:822-823` sets the warm-up (`next_day - placed_day - first_yield_day`), and
    `:824-829` the `1 + interval` units a fed-and-cared head yields per production.
    """
    from agent.main_v4 import _animal_orders, _last_buy_day

    assert (_last_buy_day("COW"), _last_buy_day("SHEEP"), _last_buy_day("GOOSE")) == (20, 22, 22)

    tiles = board({(x, 0): plant("WHEAT", 0) for x in range(10)})
    rich = obs_for(tiles, day=21, shed={"WHEAT": 40})
    plan = Plan.boatlee_like()
    # A herd that has escaped is re-bought all season — until the day it cannot pay for itself.
    assert _animal_orders(rich, plan, 0, 20, budget=100_000)
    assert [o[1] for o in _animal_orders(rich, plan, 0, 21, budget=100_000)] == ["SHEEP"] * 4
    assert _animal_orders(rich, plan, 0, 23, budget=100_000) == []

    # In play: the counter fires, and nothing is ordered past the cutoff.
    #
    # A season only *reaches* the cutoff if its herd escapes after day 20, so a quiet season passes
    # this while proving nothing — which is why the counter is asserted nonzero. Which seeds are
    # late-escaping seasons is a property of the *plan*, not of this rule, and it moves whenever the
    # plan does: seed 7 was the original choice and stopped escaping late, and so did seed 29 one
    # planner change later. So the trigger is taken over a small set and only the set has to fire,
    # while the "nothing past the cutoff" half is asserted on every season in it.
    fired = 0
    for seed in (17, 23, 37):
        _money, _blocked, _ours = play(seed=seed)
        effects = counters(0)
        fired += effects.get("animals_past_payback", 0)
        assert effects.get("animals_ordered_d25", 0) == 0, "a purchase that cannot produce went out"
        assert effects.get("animals_ordered", 0) > 0, "no animals bought at all"
    assert fired > 0, "the cutoff never fired on any of the three late-escaping seasons"


def test_structures_track_the_animals_rather_than_the_calendar():
    """`_structure_tasks` builds one structure per *scheduled* animal, which put 13 pastures on the
    board by day 8 with three animals in them (E66). The paced plan trims the herd to what the farm
    has actually got hold of, so a structure is built for an animal that exists."""
    from agent.main_v4 import _paced_plan, _structure_lead

    plan = Plan.boatlee_like()
    lead = _structure_lead(plan)
    day = max(d for _s, d in plan.herd)               # every animal is due

    empty = obs_for(board(), day=day, shed={})
    assert len(_paced_plan(empty, plan, 0, day, None).herd) == lead

    tiles = board({tuple(plan.pasture_tiles[0]): animal("SHEEP", placed_day=1)})
    one = obs_for(tiles, day=day, shed={"COW": 1})
    paced = _paced_plan(one, plan, 0, day, None)
    assert len(paced.herd) == 2 + lead
    assert paced.herd == plan.herd[:2 + lead], "the schedule order is intent, not ours to reorder"

    # Nothing is ever brought forward: on every day the paced herd is a prefix of the plan's, and
    # never has more due than the plan's own schedule does.
    for d in range(30):
        paced = _paced_plan(empty, plan, 0, d, None)
        assert paced.herd == plan.herd[:len(paced.herd)]
        assert sum(1 for _s, day_ in paced.herd if day_ <= d) <= \
            sum(1 for _s, day_ in plan.herd if day_ <= d)

    st: dict = {}
    _paced_plan(empty, plan, 0, day, st)
    assert st["effects"]["herd_paced_days"] == 1
    assert st["effects"]["structures_deferred"] == len(plan.herd) - lead


# --------------------------------------------------------------------- the budget

def test_a_turn_fits_the_budget_with_room():
    """1 s per turn, or the episode ERRORs. The compile happens on one turn a day, so that turn is
    the one that matters."""
    ours = make_agent()
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 3})
    worst = 0.0
    for _ in range(24 * 20):
        obs = sim.observation(0)
        t0 = time.perf_counter()
        action = ours(obs)
        worst = max(worst, (time.perf_counter() - t0) * 1000)
        sim.step([action, other(sim.observation(1))])
    assert worst < 200, f"worst turn {worst:.0f} ms"


def test_the_day_is_compiled_once_and_replayed():
    """Hours 2-23 are cache reads. If the shell recompiled every hour it would still be correct and
    twenty times slower, so this is a performance invariant with a cheap test."""
    ours = make_agent()
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 3})
    for _ in range(24 * 3):
        sim.step([ours(sim.observation(0)), other(sim.observation(1))])

    obs = sim.observation(0)
    if obs["hour"] in (0, 1):                       # land mid-day, where the cache is in use
        sim.step([ours(obs), other(sim.observation(1))])
        obs = sim.observation(0)
    t0 = time.perf_counter()
    ours(obs)
    assert (time.perf_counter() - t0) * 1000 < 5, "a cached hour should be nearly free"


# --------------------------------------------------------------------- effect counters

def _pressured(**consts):
    base = Plan.boatlee_like()
    return Plan(**{**base.__dict__, "consts": {**base.consts, **consts}})


def test_the_compilers_repair_record_is_a_season_total_and_reaches_the_harness():
    """C3 prunes and escalates all season; `counters()` used to report only the last compiled day.

    Worse, nothing consumed it: 0 of 488 rows in `results/games.jsonl` carried `pruned`. This asserts
    the wire end to end — the agent accumulates, `ctx.effects` carries it, and the harness Observer
    lands it in `effect_count`, which is what the jsonl rows and the printed table read.
    """
    from harness.counters import Observer

    ours = make_agent(Plan.boatlee_like())
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 41001})
    sim.collect_stats = True
    observer = Observer(player=0)
    for _ in range(719):
        obs = sim.observation(0)
        action = ours(obs)
        observer.observe(obs, action)
        sim.step([action, other(sim.observation(1))])

    effects = observer.finish(sim, ours)["effect_count"]
    assert effects.get("compiled_days") == 30, effects
    assert effects.get("pruned_tasks", 0) > 0, (
        "the Boatlee-like plan does prune — a zero here is a dead wire, not a clean season")
    assert effects.get("pruned_tasks") == counters(0)["pruned"]
    assert effects.get("overcommit_days", 0) == counters(0)["overcommit"]


def test_a_second_season_does_not_inherit_the_first_ones_totals():
    """`_STATE` is a module global and outlives the agent, so the ctx has to be rebound, not copied."""
    other = get("starter").build()
    totals = []
    for _ in range(2):
        ours = make_agent(Plan.boatlee_like())
        sim = kagsim.Sim({"episodeSteps": 720, "seed": 41002})
        for _ in range(48):
            sim.step([ours(sim.observation(0)), other(sim.observation(1))])
        totals.append(dict(ours.ctx.effects))
    assert totals[0] == totals[1], totals


# --------------------------------------------------------------------- the shed-pressure valve

def test_the_release_valve_opens_the_floors_when_the_shed_is_the_binding_constraint():
    """A floor defended past the shed cap does not protect a price, it discards the goods.

    Dusk drops every carried inventory into a 100-unit shed and **throws away the overflow**
    (`kaggriculture.py:843-857`), so stock held back at 85/100 is stock about to be lost for $0.
    Above `release_pressure` the withheld stock is released, and only down to the threshold.
    """
    plan = _pressured(sell_floor={"MELON": 0.95, "WOOL": 0.95}, release_pressure=70)
    obs = obs_for(board(), day=10, shed={"MELON": 60, "WOOL": 25})
    obs["step"] = 240
    st: dict = {"effects": {}}

    orders = _sell_orders(obs, plan, 0, st)
    released = st["effects"]["release_valve_units"]
    assert st["effects"]["release_valve_fires"] == 1
    assert released == 85 - 70, f"released {released}, wanted exactly the excess"
    # Most valuable first: melon quotes above wool, so the melon order is the one that grows.
    sold = {o[1]: o[2] for o in orders}
    assert sold.get("MELON", 0) > 0 and sold.get("MELON", 0) + sold.get("WOOL", 0) >= released


def test_the_valve_stays_shut_below_the_threshold():
    plan = _pressured(sell_floor={"MELON": 0.95, "WOOL": 0.95}, release_pressure=70)
    obs = obs_for(board(), day=10, shed={"MELON": 30, "WOOL": 10})
    obs["step"] = 240
    st: dict = {"effects": {}}
    _sell_orders(obs, plan, 0, st)
    assert "release_valve_fires" not in st["effects"], st["effects"]


def test_release_pressure_zero_disables_the_valve_entirely():
    """The gene is the off switch, which is what makes an A/B of the valve alone possible."""
    plan = _pressured(sell_floor={"MELON": 0.95, "WOOL": 0.95}, release_pressure=0)
    obs = obs_for(board(), day=10, shed={"MELON": 60, "WOOL": 25})
    obs["step"] = 240
    st: dict = {"effects": {}}
    shut = {o[1]: o[2] for o in _sell_orders(obs, plan, 0, st)}
    # The floors still bind (that is what `shut` is), so the floor counters are expected; what
    # `release_pressure=0` must not do is fire the valve.
    assert "release_valve_fires" not in st["effects"], st["effects"]
    assert "release_valve_units" not in st["effects"], st["effects"]

    open_plan = _pressured(sell_floor={"MELON": 0.95, "WOOL": 0.95}, release_pressure=70)
    opened = {o[1]: o[2] for o in _sell_orders(obs, open_plan, 0, {"effects": {}})}
    assert sum(opened.values()) - sum(shut.values()) == 85 - 70, (shut, opened)


# --------------------------------------------------------------------- order truncation

def test_a_busy_dawn_drops_the_cheapest_sale_not_the_dearest():
    """`_process_market` truncates the queue at `maxMarketOrdersPerTurn`
    (`kaggriculture.py:551`, `q[:max_orders]`). Sells were appended last in `SELLABLE` order, so
    the WHEAT sale survived and the MELON one did not."""
    from agent.main_v4 import _dispatch

    obs = obs_for(board(), day=10)
    orders = ([["HIRE"]] * 6 + [["BUY_SEED", "WHEAT", 3]]
              + [["SELL", "WHEAT", 4], ["SELL", "EGG", 6], ["SELL", "MELON", 8],
                 ["SELL", "MILK", 5]])
    st: dict = {"effects": {}}
    kept = _dispatch(obs, orders, st)

    assert len(kept) == 10
    assert st["effects"]["orders_truncated"] == 1
    assert st["effects"]["orders_dropped"] == 1
    assert st["effects"]["sells_deferred"] == 1
    sold = [o[1] for o in kept if o[0] == "SELL"]
    assert "MELON" in sold and "WHEAT" not in sold, sold
    assert all(o in kept for o in orders if o[0] != "SELL"), "the gates and buys are never dropped"


def test_a_short_queue_is_passed_through_untouched():
    from agent.main_v4 import _dispatch

    obs = obs_for(board(), day=10)
    orders = [["HIRE"], ["SELL", "WHEAT", 4], ["SELL", "MELON", 8]]
    st: dict = {"effects": {}}
    assert _dispatch(obs, list(orders), st) == orders
    assert st["effects"] == {}


# ------------------------------------------------------------- HIRE vs the day's supplies (E68)

def test_market_orders_are_accepted_on_every_turn_not_only_at_dawn():
    """The load-bearing env fact behind `_dispatch`'s docstring, asserted in play rather than read.

    `maxMarketOrdersPerTurn` caps a *turn*, not a day: `_process_market` runs unconditionally on
    every step of the interpreter (`kaggriculture.py:941`, `kagsim/src/rules.rs:585`). If this ever
    stopped being true, the "spreading the queue over the day is possible, and measured worse"
    record in `_dispatch` would be about a game that does not exist.
    """
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 11})
    idle = {"farmer": ["PASS"], "hands": [], "market": []}
    for _ in range(7):                                    # run to hour 7 — nowhere near dawn
        sim.step([idle, idle])
    assert sim.observation(0)["hour"] == 7

    before = dict(sim.observation(0)["private"]["seeds"])
    sim.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 4]]}, idle])
    after = dict(sim.observation(0)["private"]["seeds"])
    assert after.get("WHEAT", 0) - before.get("WHEAT", 0) == 4, (before, after)


def test_hires_never_starve_the_days_seed_feed_and_fertilizer():
    """E68's defect: `_dawn` appended the HIREs first, so a ten-hire day bought *nothing*.

    Every essential buy has to survive a full-crew dawn, because hour 1 compiles against `cash=0`
    and prunes every task whose supplies never arrived. The hires are what gives way.
    """
    from agent.main_v4 import _dispatch

    obs = obs_for(board(), day=10)
    essentials = [["BUY_SEED", "WHEAT", 3], ["BUY_SEED", "MELON", 2],
                  ["BUY_PRODUCT", "WHEAT", 8], ["BUY_PRODUCT", "FERTILIZER", 5]]
    orders = [["BUY_LAND"]] + [["HIRE"]] * 10 + essentials + [["SELL", "MILK", 4]]
    st: dict = {"effects": {}}
    kept = _dispatch(obs, orders, st)

    assert len(kept) == 10
    assert all(o in kept for o in essentials), kept
    assert kept[0] == ["BUY_LAND"], "the land gate still goes first"
    assert st["effects"]["dawn_starved_naive"] == 1, "the disease is counted in the old order"
    assert "dawn_starved" not in st["effects"], "and it does not survive the fix"
    assert st["effects"]["hires_dropped"] == 5, kept


def test_an_animal_outranks_a_marginal_hire_at_dawn():
    """Measured, not assumed: ranking BUY_ANIMAL above HIRE is worth +$2.5k over the reverse on
    53000:53040 (see `_dispatch`). The herd compounds for the rest of the season; the eighth hand
    does not."""
    from agent.main_v4 import _dispatch

    obs = obs_for(board(), day=10)
    orders = [["HIRE"]] * 9 + [["BUY_ANIMAL", "COW", 1], ["BUY_ANIMAL", "SHEEP", 1]]
    kept = _dispatch(obs, orders, {"effects": {}})
    assert ["BUY_ANIMAL", "COW", 1] in kept and ["BUY_ANIMAL", "SHEEP", 1] in kept, kept
    assert sum(1 for o in kept if o[0] == "HIRE") == 8


def test_no_real_season_ever_drops_a_buy_it_needed_that_day():
    """In play, not in a unit fixture (E44). The season must both *have* the disease — dawns whose
    naive order would have cut an essential buy — and never actually cut one. Seed 49100 is the
    season E68 traced the outage on."""
    from harness.counters import Observer

    ours = make_agent(Plan.boatlee_like())
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 49100})
    sim.collect_stats = True
    observer = Observer(player=0)
    for _ in range(719):
        obs = sim.observation(0)
        action = ours(obs)
        observer.observe(obs, action)
        if obs["hour"] == 0:
            # Everything the shell asks for that is not a sell has to fit inside the ten slots the
            # market reads: a sell that does not fit is deferred and re-derived, a buy that does not
            # is a planting, a feeding or a fertilizing lost for the whole day.
            buys = [o for o in action["market"] if o[0] in ("BUY_SEED", "BUY_PRODUCT")]
            assert len(buys) <= 10 and len(action["market"]) <= 10
            assert all(o in action["market"] for o in buys)
        sim.step([action, other(sim.observation(1))])

    effects = observer.finish(sim, ours)["effect_count"]
    assert effects.get("dawn_starved_naive", 0) > 0, (
        "a zero here means the counter is dead, not that the dawns were quiet")
    assert effects.get("dawn_starved", 0) == 0, effects
    assert effects.get("hires_dropped", 0) > 0, "and the hires are what gave way"


# ------------------------------------------------- the arriving quadrant is shopped for (E73)

def _late_land_plan():
    """S1's cliff genome: NW packed so tight that the season's cash misses NE's day-6 gate, and
    the two 25/22-tile strawberry cohorts arrive six days behind their quadrant."""
    from agent.plan import NEVER, decode, encode_fields

    return decode(encode_fields(
        n_pastures=10, n_cows=14, cow_start=1, n_sheep=4, sheep_start=0, n_geese=0, geese_start=0,
        animals_per_day=3, hands_mode="curve", hands_start=4, hands_ramp=10, hands_cap=14,
        land_days={"NE": 6, "SW": 10, "SE": NEVER},
        cohorts=[("STRAWBERRY", "NE", 25, 6, False), ("STRAWBERRY", "SW", 22, 10, False),
                 ("MELON", "NW", 4, 0, False), ("WHEAT", "NW", 3, 0, True)]))


def _land_dawns(plan, seed: int):
    """Every dawn that issues `BUY_LAND`, as (backlog, arriving quadrant, market orders)."""
    from agent.main_v4 import _quadrant_backlog
    from agent.plan import LAND_ORDER

    ours = make_agent(plan)
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    out = []
    for _ in range(719):
        obs = sim.observation(0)
        action = ours(obs)
        if obs["hour"] == 0 and ["BUY_LAND"] in action["market"]:
            owned = set(obs["farms"][0].get("unlocked_quadrants") or [])
            quad = next(q for q in LAND_ORDER if q not in owned)
            out.append((_quadrant_backlog(plan, quad, int(obs["day"])), quad, action["market"]))
        sim.step([action, other(sim.observation(1))])
    return out, counters(0)


def test_a_badly_late_quadrant_is_shopped_for_on_the_dawn_it_is_bought():
    """`BUY_LAND` settles in the *same* market turn dawn issues it, but every other decision dawn
    makes used to be sized against tiles that still read `"LOCKED"` — so on the morning a 25-tile
    cohort's quadrant finally arrives, `_planting_tasks` found no free tiles in it and none of its
    seed was ordered. Hour 1 then compiled the unlocked block against an empty shed and pruned
    every PLANT in it.

    Driven through `_dawn` on a board built to the cliff's shape, because the affordability of the
    seed on any particular morning is a separate question from whether dawn asked for it: the fix
    is that the arriving quadrant is *on the shopping list*, and that is what is asserted.
    """
    from agent.main_v4 import _dawn

    plan = _late_land_plan()
    tiles = board()
    for (x, y) in [(x, y) for y in range(10) for x in range(10)]:
        if quadrant_of(x, y) != "NW":
            tiles[y][x] = "LOCKED"
    obs = obs_for(tiles, day=12)                  # NE is due on 6: its cohort is six days late
    obs["farms"][0]["unlocked_quadrants"] = ["NW"]
    obs["farms"][0]["money"] = 8000

    st: dict = {"compiled": None, "compiled_day": -1, "effects": {}}
    market = _dawn(obs, st, plan, 0, 12)["market"]

    assert ["BUY_LAND"] in market, market
    assert st["effects"]["land_day_reseed"] == 1
    assert any(o[:2] == ["BUY_SEED", "STRAWBERRY"] for o in market), (
        f"bought NE six days late and asked for none of its strawberry: {market}")
    assert obs["farms"][0]["tiles"][0][5] == "LOCKED", "the live observation was written to"


def test_the_cliff_genome_gets_its_strawberry_into_the_ground():
    """The same defect at season scale — S1's repro, which is the reason any of this is here.

    The cliff genome misses NE's day-6 gate on cash, so NE and SW arrive back to back around day 12
    on a melon windfall; the herd took the first morning and the second quadrant took the next, and
    the 47-tile strawberry block was never sown. `plants_started` 66 -> 43, **$91.5k -> $46.5k**.
    """
    plan = _late_land_plan()
    ours = make_agent(plan)
    other = get("starter").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 60050})
    peak = 0
    for _ in range(719):
        obs = sim.observation(0)
        peak = max(peak, sum(1 for row in obs["farms"][0]["tiles"] for t in row
                             if isinstance(t, dict) and t.get("crop") == "STRAWBERRY"))
        sim.step([ours(obs), other(sim.observation(1))])

    assert counters(0).get("land_day_reseed", 0) > 0, "the fix never fired — this proves nothing"
    assert peak >= 20, f"the strawberry block never went in: {peak} tiles at its peak"
    assert sim.money(0) > 60_000, f"${sim.money(0):,.0f} — the cliff is back"


def test_a_plan_whose_land_is_on_time_is_left_alone():
    """The other half of the gate, and the reason it exists. Shopping a day late for a quadrant
    that arrives on its plan day is a one-day tax the incumbent plans are already tuned around, so
    the fix must not fire on them: re-sizing every land dawn measures **-$1.6k** on 60100:60140 and
    **-$2.8k** on 61000:61040 across `boatlee_like`/`flooder`/`tomato_rusher` (480 paired games
    each). Asserted as the counter staying at zero over a real season."""
    dawns, effects = _land_dawns(Plan.boatlee_like(), 60100)

    assert dawns, "a zero here means the season bought no land, not that it was punctual"
    assert max(b for b, _q, _m in dawns) < LATE_COHORT_DAYS, dawns
    assert effects.get("land_day_reseed", 0) == 0, effects


def test_the_unlocked_view_is_a_copy_and_never_touches_the_live_observation():
    """`_unlocked_view` is handed the observation the opponent's seat and the rest of the turn
    still read from, so it may not write to it — and the tiles it frees are exactly the arriving
    quadrant's locked ones, never another quadrant's."""
    from agent.main_v4 import _unlocked_view
    from agent.plan import quadrant_of

    tiles = board()
    for y in range(10):
        for x in range(10):
            if quadrant_of(x, y) != "NW":
                tiles[y][x] = "LOCKED"
    tiles[6][1] = plant("WHEAT", 0)               # an SW tile that is not locked
    obs = obs_for(tiles, day=6)
    obs["farms"][0]["unlocked_quadrants"] = ["NW"]

    view = _unlocked_view(obs, 0, "NE")

    assert obs["farms"][0]["tiles"] is tiles, "the live observation was rebound"
    assert all(tiles[y][x] == "LOCKED" for y in range(5) for x in range(5, 10)), "NE was mutated"
    assert view["farms"][0]["unlocked_quadrants"] == ["NW", "NE"]
    freed = {(x, y) for y in range(10) for x in range(10)
             if tiles[y][x] != view["farms"][0]["tiles"][y][x]}
    assert freed == {(x, y) for y in range(5) for x in range(5, 10)}, sorted(freed)
    assert all(view["farms"][0]["tiles"][y][x] is None for x, y in freed)
    assert view["private"] is obs["private"], "everything else is shared, not copied"


# ------------------------------------------------------ S3: the per-product floors, in play

def test_every_product_floor_binds_on_its_own_product():
    """One floor per sellable product (S3, E76), and each one reaching only its own goods.

    The counter is per product for the reason CLAUDE.md's "prove the change fired" rule exists: a
    floor that never withholds a unit is a gene the search will tune against nothing, and an inert
    floor and a transposed floor table produce the same money.
    """
    from agent.plan import FLOOR_PRODUCTS

    for product in FLOOR_PRODUCTS:
        plan = _pressured(sell_floor={product: 0.99}, release_pressure=0)
        obs = obs_for(board(), day=10, shed={product: 60})
        obs["step"] = 240
        st: dict = {"effects": {}}
        orders = _sell_orders(obs, plan, 0, st)
        assert st["effects"].get(f"floor_withheld_{product}", 0) > 0, (product, orders)
        assert st["effects"]["floor_withheld_units"] == st["effects"][f"floor_withheld_{product}"]
        assert st["effects"]["floor_binds"] == 1


def test_a_zero_floor_withholds_nothing_and_says_so():
    """The off switch, per product: 0.0 is what every product that had no floor gene already did."""
    plan = _pressured(sell_floor={"MILK": 0.0, "WOOL": 0.0, "MELON": 0.0}, release_pressure=0)
    obs = obs_for(board(), day=10, shed={"MILK": 40, "WOOL": 40, "MELON": 40})
    obs["step"] = 240
    st: dict = {"effects": {}}
    sold = {o[1]: o[2] for o in _sell_orders(obs, plan, 0, st)}
    assert sold == {"MILK": 40, "WOOL": 40, "MELON": 40}, sold
    assert "floor_withheld_units" not in st["effects"], st["effects"]


def test_the_new_floors_reach_a_real_season():
    """In play, not in a fixture (E44). A milk floor on `boatlee_like` — nine cows, no geese.

    The opponent is `boatlee` and that is the measurement, not a detail: against `starter` the milk
    quote never falls to 0.99 x base and this floor withholds **nothing** (measured, 719 steps).
    Boatlee sells milk into the same three shops we do, the quote drops, and the floor starts
    binding — 664 units on this seed. A floor is a gene about the *opponent's* supply, so a fixture
    or a weak opponent cannot tell a working one from an inert one.
    """
    from agent.plan import GENE_INDEX, decode, encode

    v = encode(Plan.boatlee_like())
    v[GENE_INDEX["sell_floor_milk"]] = 0.99
    ours = make_agent(decode(v))
    other = get("boatlee").build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 60300})
    for _ in range(719):
        sim.step([ours(sim.observation(0)), other(sim.observation(1))])
    effects = dict(ours.ctx.effects)
    assert effects.get("floor_withheld_MILK", 0) > 0, effects
    assert effects.get("floor_withheld_EGG", 0) == 0, "no geese, so no egg floor can bind"
    assert effects.get("floor_withheld_CARROT", 0) == 0, "no carrots planted, no carrot floor"
