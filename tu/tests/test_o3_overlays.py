"""O3: front-run, counter-mix and slot alignment.

Written to the same rule as `test_opponent.py` — a helper that returns the right value proves
nothing (E44), so every overlay is asserted **in play**: the game is run through kagsim and the
answer read off the effects ledger `main_v4` keeps during the episode. The two properties that
matter most are the ones a money number would never reveal:

* each overlay is **off by default** — with the flag clear, the agent's whole action stream is
  byte-identical to the incumbent's, so the incumbent measurements are not silently re-based;
* each overlay's counter is **non-zero when it is on**, which is what makes a subsequent money
  number evidence rather than noise.

The env facts the overlays are priced off are asserted here too (price impact, slot lockstep),
against the reference `kaggriculture.py` rather than against prose.
"""

from __future__ import annotations

import time

import pytest

from agent import main_v4, opponent, projection
from agent.plan import Plan


def _play(a_name, b_name, seed, steps=719, collect=False):
    """One real game. Returns (effects of seat 0, money, per-step market lists)."""
    import kagsim

    from harness import registry

    agents = [registry.get(a_name).build(), registry.get(b_name).build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    markets = []
    for _ in range(steps):
        acts = [agents[p](sim.observation(p)) for p in (0, 1)]
        if collect:
            markets.append([[list(o) for o in (a.get("market") or [])] for a in acts])
        sim.step(acts)
    effects = dict(getattr(agents[0], "ctx").effects)
    effects.update({f"proj_{k}": v for k, v in (projection.stats(0) or {}).items()})
    return effects, (sim.money(0), sim.money(1)), markets


def _actions(a_name, b_name, seed, steps):
    import kagsim

    from harness import registry

    agents = [registry.get(a_name).build(), registry.get(b_name).build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    out = []
    for _ in range(steps):
        acts = [agents[p](sim.observation(p)) for p in (0, 1)]
        out.append(repr(acts[0]))
        sim.step(acts)
    return out


# ------------------------------------------------------------------ the env rules being exploited

def test_a_sale_moves_the_price_one_unit_at_a_time():
    """`_commit_unit` adds 1 per unit (`kaggriculture.py:660`) and the next unit is re-quoted at
    the new inventory inside the same loop (`:597`). The thin markets are the point."""
    from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

    assert market_price("STRAWBERRY", 10_000) == 120
    # boatlee's 34-unit strawberry dump at step 528: over half the quote, in one turn.
    after = market_price("STRAWBERRY", 10_000 + 34)
    assert after < 0.5 * 120, after
    # and it is monotone, so getting there first is strictly better
    quotes = [market_price("MILK", 10_000 + k) for k in range(0, 22)]
    assert quotes == sorted(quotes, reverse=True)
    assert quotes[-1] < quotes[0]


def test_orders_are_processed_slot_by_slot_across_both_players():
    """The claim `slot_align` is built on, asserted against the reference source rather than prose.

    `_process_market`'s outer loop is over the order **index**; slot `i` is drained for both
    players and re-priced before slot `i+1` is quoted. So a slot-0 sell executes against the
    untouched inventory and a slot-2 sell of the same product does not — while *within* a slot both
    players are quoted identically, which is why there is no seat advantage to take.
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        _new_market, _process_market,
    )

    def _seat(shed, orders):
        farm = {"money": 0.0}
        return farm, {"shed": dict(shed), "seeds": {}}, [list(o) for o in orders]

    def _run(p0_orders, p1_orders):
        market = _new_market()
        states, farms = [], []
        for orders, shed in ((p0_orders, {"STRAWBERRY": 20}), (p1_orders, {"STRAWBERRY": 20})):
            farm, private, os_ = _seat(shed, orders)
            farms.append(farm)
            states.append(type("S", (), {"observation": type("O", (), {
                "market": market, "farms": farms, "private": private})(),
                "action": {"market": os_}})())
        # both seats must share one `farms`/`market` object, as the interpreter does
        for s in states:
            s.observation.farms = farms
            s.observation.market = market
        env = type("E", (), {"configuration": {}})()
        states[0].observation.farms = farms
        _process_market(states, env)
        return farms[0]["money"], farms[1]["money"]

    # Same slot: identical quotes, so the money splits evenly — no seat advantage.
    a, b = _run([["SELL", "STRAWBERRY", 10]], [["SELL", "STRAWBERRY", 10]])
    assert a == b, (a, b)

    # Different slots: player 0 at slot 0, player 1 behind a filler order at slot 1.
    a, b = _run([["SELL", "STRAWBERRY", 10]],
                [["BUY_SEED", "WHEAT", 0], ["SELL", "STRAWBERRY", 10]])
    assert a > b, (a, b)


# ------------------------------------------------------------------ off by default

@pytest.mark.parametrize("flag", ["frontrun", "counter_mix", "slot_align"])
def test_every_overlay_is_off_by_default(flag):
    assert not (Plan.boatlee_like().consts or {}).get(flag)


def test_the_incumbent_is_byte_identical_with_the_flags_clear():
    """The flags default off, so C4's measured season must not move by a single action.

    This is the guard that lets O3 land without re-basing every incumbent number in TASKS_v4.
    """
    base = _actions("compiler", "boatlee", 67101, 240)
    again = _actions("compiler#frontrun=0,counter_mix=0,slot_align=0", "boatlee", 67101, 240)
    assert base == again


# ------------------------------------------------------------------ front-run

def test_frontrun_windows_straddle_the_dump_and_skip_the_dribbles():
    rows = {"MILK": [(400, 21), (410, 3)]}
    w = opponent.windows_from_rows(rows, lead=10, min_units=8)
    assert set(w) == set(range(390, 401))               # [s - lead, s], inclusive of s
    assert w[400] == {"MILK": 21}
    assert 410 not in w and 405 not in w                # the 3-unit dribble is not worth a shed
    # ±1 day for a forecast rather than a measured table
    spread = opponent.windows_from_rows(rows, lead=10, min_units=8, spread=24)
    assert min(spread) == 366 and max(spread) == 424


def test_frontrun_settles_the_schedule_before_sizing_it():
    """An order is not a sale (CLAUDE.md): boatlee asks for 241 milk and moves 209."""
    settled = opponent.settled_schedule("boatlee")
    assert sum(n for _s, n in settled["MILK"]) < sum(n for _s, n in opponent.SELL_SCHEDULE["MILK"])
    assert opponent.settled_schedule("starter") == {}
    assert opponent.settled_schedule(None) == {}


def test_the_lead_is_clamped_into_a_usable_range():
    """`frontrun_lead` is a 0-24 gene. A lead under six turns cannot straddle a dump at 24 turns a
    day; the clamp is what keeps an un-searched genome from disabling the overlay by accident."""
    st = {"opp_known": "boatlee"}
    zero = main_v4._frontrun_map(st, Plan.boatlee_like().with_consts(frontrun_lead=0))
    st = {"opp_known": "boatlee"}
    six = main_v4._frontrun_map(st, Plan.boatlee_like().with_consts(frontrun_lead=6))
    assert zero == six and zero
    st = {"opp_known": "boatlee"}
    wide = main_v4._frontrun_map(st, Plan.boatlee_like().with_consts(frontrun_lead=99))
    assert len(wide) > len(six)


def test_frontrun_never_front_runs_itself():
    for name in sorted(opponent.SELF_PROFILES):
        assert main_v4._frontrun_map({"opp_known": name}, Plan.boatlee_like()) == {}


def test_frontrun_fires_in_play_and_moves_units_that_the_floor_was_holding():
    effects, _money, _m = _play("compiler#frontrun=1", "boatlee", 67101)
    assert effects.get("sells_frontrun", 0) > 0
    assert effects.get("frontrun_units", 0) > 0
    assert effects.get("frontrun_value", 0) > 0
    # the overlay also buys market turns the routing calendar would not have given it
    assert effects.get("frontrun_turns", 0) > 0
    assert effects.get("fallbacks", 0) == 0
    off, _money, _m = _play("compiler", "boatlee", 67101)
    assert off.get("sells_frontrun", 0) == 0


def test_frontrun_suspends_the_floor_rather_than_fighting_it():
    """Precedence, asserted on the sell path directly: a floored product inside a front-run window
    goes out whole, and the units the floor would have kept are counted."""
    plan = Plan.boatlee_like().with_consts(frontrun=1, sell_floor={"WOOL": 0.95})
    obs = {"step": 300, "day": 12, "hour": 5, "player": 0,
           "farms": [{"tiles": [[None] * 10 for _ in range(10)], "money": 5000.0,
                      "hands": [], "farmer": [4, 4], "unlocked_quadrants": ["NW"]}],
           "private": {"shed": {"WOOL": 30}, "inventories": []},
           "market": {"inventory": {"WOOL": 10_400}, "prices": {"WOOL": 40}},
           "town": {"unlocked_shops": []}}
    st = {"effects": {}}
    held = main_v4._sell_orders(obs, plan, 0, st)
    assert held == [] or held[0][2] < 30                       # the floor binds at $40 vs 0.95x200

    st = {"effects": {}}
    ran = main_v4._sell_orders(obs, plan, 0, st, frontrun={"WOOL": 16}, forced=True)
    assert ran == [["SELL", "WOOL", 30]]
    assert st["effects"]["frontrun_units"] == 30
    assert st["effects"]["frontrun_floor_override"] > 0


def test_terminal_liquidation_still_outranks_the_front_run():
    plan = Plan.boatlee_like().with_consts(frontrun=1, sell_floor={"WOOL": 0.95})
    obs = {"step": 710, "day": 29, "hour": 14, "player": 0,
           "farms": [{"tiles": [[None] * 10 for _ in range(10)], "money": 5000.0,
                      "hands": [], "farmer": [4, 4], "unlocked_quadrants": ["NW"]}],
           "private": {"shed": {"WOOL": 30, "MELON": 4}, "inventories": []},
           "market": {"inventory": {"WOOL": 10_400, "MELON": 10_000}, "prices": {"WOOL": 40}},
           "town": {"unlocked_shops": []}}
    st = {"effects": {}}
    orders = main_v4._sell_orders(obs, plan, 0, st, frontrun={"WOOL": 16})
    assert sorted(orders) == sorted([["SELL", "MELON", 4], ["SELL", "WOOL", 30]])
    # liquidation, not a front-run: the counter must not claim credit for it
    assert "frontrun_units" not in st["effects"]


# ------------------------------------------------------------------ counter-mix

def test_contest_is_measured_over_the_cohorts_own_harvest_window():
    lo, hi = projection._window("STRAWBERRY", 7)
    assert (lo, hi) == (17 * 24, 30 * 24 - 1)          # sown day 7, first yield day 10, ongoing
    supply = {"STRAWBERRY": [(lo + 24, 50), (lo - 24, 500)]}
    assert projection.contest("STRAWBERRY", 7, supply) == 0.5      # T = 100; the early row is out
    assert projection.contest("MELON", 7, supply) == 0.0


def test_the_value_gate_prices_the_contest_rather_than_ignoring_it():
    """The bug the first implementation had: `tile_value` reads a fourteen-day projection, so a
    cohort sown on day 7 was priced against a board where the opponent's day-13 strawberry did not
    exist, STRAWBERRY ranked first every time, and the veto never vetoed."""
    obs = {"step": 0, "day": 0, "hour": 0, "player": 0,
           "farms": [{"tiles": [[None] * 10 for _ in range(10)], "money": 3000.0, "hands": [],
                      "farmer": [4, 4], "unlocked_quadrants": ["NW"]}] * 2,
           "private": {"shed": {}, "inventories": []},
           "market": {"inventory": {}, "prices": {}}, "town": {"unlocked_shops": []}}
    proj = projection.for_obs(obs)
    clean = projection.contested_tile_value("STRAWBERRY", 7, 0, proj, {})
    flooded = projection.contested_tile_value(
        "STRAWBERRY", 7, 0, proj, {"STRAWBERRY": [(408, 300)]})
    assert flooded < clean
    assert flooded < projection.contested_tile_value("MELON", 7, 0, proj, {})


def test_counter_mix_fires_in_play_and_does_not_flee_into_one_market():
    effects, _money, _m = _play("compiler#counter_mix=1", "boatlee", 67101)
    assert effects.get("cohorts_redirected", 0) > 0
    assert effects.get("counter_mix_tiles", 0) > 0
    assert effects.get("fallbacks", 0) == 0
    # both contested strawberry cohorts moving to the same crop is E41's collapse rebuilt by hand
    targets = {k for k in effects if k.startswith("counter_mix_to_")}
    assert len(targets) >= 2, effects
    off, _money, _m = _play("compiler", "boatlee", 67101)
    assert off.get("cohorts_redirected", 0) == 0


def test_a_redirected_cohort_stays_redirected_and_the_memory_does_not_outlive_the_season():
    """Two properties in one game pair, because they failed together.

    A veto that flickers with the forecast sows half a cohort of each crop — the same argument
    `projection._REDIRECTS` makes for the scarcity hunter — so a season commits at most once per
    cohort. And `_CONTESTED` is a **module global**: the harness rebuilds the agent per game but
    not the module, so without a step-0 reset the second game in a worker process finds every
    cohort already committed, re-evaluates nothing, and reports `cohorts_redirected == 0`. That is
    a change that never fired wearing the counters of one that did (E44), and it is invisible from
    one game — hence two.
    """
    first, _m, _mk = _play("compiler#counter_mix=1", "boatlee", 67101)
    second, _m, _mk = _play("compiler#counter_mix=1", "boatlee", 67102)
    for effects in (first, second):
        assert 0 < effects["cohorts_redirected"] <= len(Plan.boatlee_like().cohorts)


def test_counter_mix_is_inert_without_an_opponent_to_read():
    plan = Plan.boatlee_like().with_consts(counter_mix=1)
    obs = {"step": 0, "day": 0, "hour": 0, "player": 0,
           "farms": [{"tiles": [[None] * 10 for _ in range(10)], "money": 3000.0, "hands": [],
                      "farmer": [4, 4], "unlocked_quadrants": ["NW"]}],
           "private": {"shed": {}, "inventories": []},
           "market": {"inventory": {}, "prices": {}}, "town": {"unlocked_shops": []}}
    assert projection.counter_mix(obs, plan, 0) is plan


# ------------------------------------------------------------------ slot alignment

def test_hoist_is_a_permutation_and_only_counts_when_it_moves_something():
    st = {"effects": {}}
    orders = [["BUY_SEED", "WHEAT", 3], ["SELL", "MILK", 4], ["SELL", "WOOL", 2]]
    out = main_v4._hoist(list(orders), frozenset({"WOOL"}), st)
    assert out == [["SELL", "WOOL", 2], ["BUY_SEED", "WHEAT", 3], ["SELL", "MILK", 4]]
    assert sorted(map(tuple, out)) == sorted(map(tuple, orders))
    assert st["effects"]["sells_reslotted"] == 1

    st = {"effects": {}}
    already = [["SELL", "WOOL", 2], ["BUY_SEED", "WHEAT", 3]]
    assert main_v4._hoist(list(already), frozenset({"WOOL"}), st) == already
    assert "sells_reslotted" not in st["effects"]                 # nothing moved, nothing counted


def test_the_hoist_runs_after_truncation_so_it_cannot_starve_a_buy():
    """E68's failure mode, guarded: a sell hoisted into the ten-slot queue must never displace an
    essential buy. The hoist is applied to the *kept* list, so it can reorder but never evict."""
    obs = {"market": {"prices": {}}, "day": 3, "hour": 0, "player": 0}
    orders = ([["BUY_SEED", "WHEAT", 1]] * 9 + [["BUY_SEED", "MELON", 1]] +
              [["SELL", "WOOL", 5], ["SELL", "MILK", 2]])
    st = {"effects": {}}
    out = main_v4._dispatch(obs, orders, st, hoist=frozenset({"WOOL"}))
    assert len(out) == main_v4.MAX_ORDERS
    assert all(o[0] == "BUY_SEED" for o in out)
    assert "sells_reslotted" not in st["effects"]


def test_slot_align_only_names_a_step_a_measured_table_knows():
    assert main_v4._slot_align_targets({"opp_known": None}, Plan.boatlee_like().with_consts(
        slot_align=1), 528) == frozenset()
    hit = main_v4._slot_align_targets({"opp_known": "boatlee"},
                                      Plan.boatlee_like().with_consts(slot_align=1), 528)
    assert hit == frozenset({"STRAWBERRY", "MELON"}), hit
    # off by default
    assert main_v4._slot_align_targets({"opp_known": "boatlee"}, Plan.boatlee_like(),
                                       528) == frozenset()


def test_slot_align_fires_in_play():
    effects, _money, _m = _play("compiler#slot_align=1", "boatlee", 67101)
    assert effects.get("sells_reslotted", 0) > 0
    assert effects.get("fallbacks", 0) == 0
    off, _money, _m = _play("compiler", "boatlee", 67101)
    assert off.get("sells_reslotted", 0) == 0


def test_the_hoisted_sell_really_lands_in_an_earlier_slot_than_theirs():
    """The spec's own verification: replay a step where boatlee sells a product from a later slot
    and assert our order for it is ahead of theirs once the overlay is on."""
    _e, _m, markets = _play("compiler#slot_align=1", "boatlee", 67101, collect=True)
    checked = 0
    for us, them in markets:
        ours = {o[1]: i for i, o in enumerate(us) if o[0] == "SELL"}
        theirs = {o[1]: i for i, o in enumerate(them) if o[0] == "SELL"}
        for item in set(ours) & set(theirs):
            if theirs[item] > 0:
                assert ours[item] <= theirs[item], (item, ours[item], theirs[item])
                checked += 1
    assert checked > 0


# ------------------------------------------------------------------ budget

def test_the_overlays_fit_the_turn_budget():
    """p99 < 100 ms is the C4 gate; all three overlays together must not eat into it."""
    import kagsim

    from harness import registry

    agents = [registry.get("compiler#frontrun=1,counter_mix=1,slot_align=1").build(),
              registry.get("boatlee").build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 67101})
    times = []
    for _ in range(719):
        obs = sim.observation(0)
        t0 = time.perf_counter()
        a0 = agents[0](obs)
        times.append(time.perf_counter() - t0)
        sim.step([a0, agents[1](sim.observation(1))])
    times.sort()
    assert times[int(0.99 * len(times))] < 0.100, times[-1]


def test_the_keep_floor_arm_leaves_the_floor_sizing_the_order():
    """The other half of the precedence question, kept switchable because the measurement wanted it.

    With `frontrun_keep_floor` set the overlay contributes only the *turn*: the floor still decides
    the quantity, and whatever it withholds stays available to `release_pressure` exactly as it
    would without the overlay.
    """
    plan = Plan.boatlee_like().with_consts(frontrun=1, frontrun_keep_floor=1,
                                           sell_floor={"WOOL": 0.95}, release_pressure=0)
    obs = {"step": 300, "day": 12, "hour": 5, "player": 0,
           "farms": [{"tiles": [[None] * 10 for _ in range(10)], "money": 5000.0,
                      "hands": [], "farmer": [4, 4], "unlocked_quadrants": ["NW"]}],
           "private": {"shed": {"WOOL": 30}, "inventories": []},
           "market": {"inventory": {"WOOL": 10_400}, "prices": {"WOOL": 40}},
           "town": {"unlocked_shops": []}}
    st = {"effects": {}}
    orders = main_v4._sell_orders(obs, plan, 0, st, frontrun={"WOOL": 16}, forced=True)
    assert orders == [] or orders[0][2] < 30
    assert st["effects"].get("frontrun_floor_override", 0) == 0
