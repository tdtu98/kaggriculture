"""Harness v2 (I2): the runner reproduces known games, and the counters count.

The harness is what every v4 decision will be read off, so it is verified against outcomes that
were established before it existed — Boatlee's mirror wobble and the session executor's band — and
its counters are checked against agents whose behaviour is known by construction.

The counter tests matter more than they look. A counter that silently reads zero is worse than no
counter: it turns "the change never fired" into "the change did not work" (E44), and it is exactly
what a harness cannot detect about itself. So two agents here are deliberately pathological.
"""

from __future__ import annotations

import json
import os

import pytest

import kagsim
from harness.counters import Observer
from harness.registry import PENDING, get
from harness.run import GameResult, parse_seeds, run

CONFIG = {"episodeSteps": 720}
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _margins(results: list[GameResult]) -> dict[int, float]:
    return {r.match.seed: r.money_a - r.money_b for r in results}


# --------------------------------------------------------------- reproduces known games

def test_boatlee_mirror_ties_exactly_and_wobbles_where_it_always_has():
    """The sharpest health check available: a deterministic script against itself.

    Seeds 2-6 tie to the dollar. Seeds 0/1/7 do not, and their margins are fixed quantities —
    seat-order effects in the shared market, since both farms play the identical 719-step table.
    These three numbers were measured before this harness existed; reproducing them is what says
    the runner drives the game correctly (turn count, seating, agent construction).
    """
    results = run(["boatlee"], list(range(0, 8)), games=8, both_seats=False, config=CONFIG,
                  workers=1, jsonl=None, mirror=True)
    margins = _margins(results)

    for seed in (2, 3, 4, 5, 6):
        assert margins[seed] == 0, f"seed {seed} must tie exactly, got {margins[seed]:+,.0f}"
    assert margins[0] == -96, margins[0]
    assert margins[1] == 771, margins[1]
    assert margins[7] == 843, margins[7]


def test_executor_reproduces_its_documented_band():
    """`executor_v7` vs `starter` is ~74k (TASKS_v4 I2). A band, not a point: the claim being
    checked is that the session executor is wired up and playing, not that 74k is meaningful."""
    results = run(["executor_v7", "starter"], list(range(0, 3)), games=6, both_seats=True,
                  config=CONFIG, workers=1, jsonl=None)
    money = [r.money_a for r in results]
    mean = sum(money) / len(money)

    assert 55_000 < mean < 95_000, (
        f"executor mean ${mean:,.0f} is outside its band. At ~$3,000 it is playing PASS — see "
        f"session_line/README.md before treating this as a result."
    )


# --------------------------------------------------------------- counters count

def test_counters_reproduce_boatlees_published_profile():
    """PLAN_v4 §1 describes Boatlee by four numbers. The observer must find them independently."""
    results = run(["boatlee", "starter"], [900, 901], games=4, both_seats=True, config=CONFIG,
                  workers=1, jsonl=None)
    c = [r.counters_a for r in results]
    mean = lambda k: sum(x[k] for x in c) / len(c)  # noqa: E731

    assert mean("steps_per_useful") < 1.10, "plan: 1.01 steps walked per useful action"
    assert mean("strawberry_per_plant") > 7.0, "plan: 37 plants at 7.7 units each"
    assert mean("fertilize_hits") > 30, "plan: fertilized at ages 9 and 13, so ~2 per plant"
    assert mean("blocked_ops") < 40, "plan: boatlee's own desync is ~10 (weeds)"

    # "plants die of old age not thirst" (PLAN_v4 §1) is the claim, and it holds exactly: every
    # loss is a plant that expired after paying out, none is a plant the routing failed to water.
    # The plan's "6-7 per season" is the one number here that does not reproduce — measured 10,
    # confirmed by an independent per-step scan of PLANT->WEED transitions.
    # Over 20 seeds x 2 seats: thirst 0.1, age 10.0 — roughly one thirst death per ten seasons.
    assert mean("plants_lost_thirst") <= 0.5, "boatlee almost never loses a plant to thirst"
    assert 8 <= mean("plants_lost_age") <= 12, "…it loses ~10/season to old age"


def _observe_scripted(action_for, steps: int = 240, seed: int = 5):
    """Drive one seat with a hand-written policy and return the counters."""
    obs_ = Observer(player=0)
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    for _ in range(steps):
        o = sim.observation(0)
        a = action_for(o)
        obs_.observe(o, a)
        sim.step([a, PASS_ACTION])
    return obs_.finish(sim)


def test_an_inert_agent_reads_as_inert():
    """All-PASS: the failure mode `session_line/executor.py` produces when miswired."""
    c = _observe_scripted(lambda obs: PASS_ACTION)

    assert c["idle_pct"] == pytest.approx(1.0), "every op was a PASS"
    assert c["blocked_ops"] == 0, "a PASS is not a blocked op — it is a chosen no-op"
    assert c["steps_per_useful"] == 0.0
    assert c["produced"] == {}


def test_blocked_ops_catches_a_deliberate_desync():
    """An agent that harvests bare ground all season. Every op is well-formed and every op does
    nothing — which is precisely what a compiled plan looks like when it has desynced from the
    board, and the reason `blocked_ops` is the compiler's health counter."""
    c = _observe_scripted(lambda obs: {"farmer": ["HARVEST"], "hands": [], "market": []})

    assert c["blocked_ops"] > 200, f"expected a wall of blocked ops, got {c['blocked_ops']}"
    assert c["idle_pct"] == 0.0, "it never passed — it acted, uselessly"


def test_plants_lost_counts_a_plant_that_dies_the_day_it_was_planted():
    """Plant once, never water. A new plant carries `consecutive_unwatered = 1`, so it reaches 2
    at that same evening's refresh and is a WEED before the next day starts — born and dead inside
    one day. The counter has to see both, or "started plants it cannot finish" is invisible."""
    def policy(obs):
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}
        if step == 1:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
        return PASS_ACTION

    c = _observe_scripted(policy, steps=24 * 4)
    assert c["plants_started"] == 1
    assert c["plants_lost"] == 1, "one plant, unwatered, must be counted lost"
    assert c["plants_lost_thirst"] == 1, "…and attributed to thirst, not to old age"
    assert c["plants_lost_age"] == 0


def _carrot_policy(sell_on_day: int | None):
    """Grow one carrot tile, harvest it on day 3, optionally sell the lot on `sell_on_day`."""
    def policy(obs):
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "CARROT", 1]]}
        if step == 1:
            return {"farmer": ["PLANT", "CARROT"], "hands": [], "market": []}
        if day == 3 and hour == 3:
            return {"farmer": ["HARVEST"], "hands": [], "market": []}
        if sell_on_day is not None and day == sell_on_day and hour == 1:
            return {"farmer": ["PASS"], "hands": [], "market": [["SELL", "CARROT", 99]]}
        if day <= 3 and hour in (1, 2):     # watered on the planting day too, or it dies that dusk
            return {"farmer": ["WATER"], "hands": [], "market": []}
        return PASS_ACTION
    return policy


def test_production_counts_units_that_arrived_and_survives_selling_them():
    """`produced` is held + sold, never what was ordered (E48/E50).

    The invariant that matters is that production does not vanish when the goods do: the same farm,
    played twice, must report the same production whether or not the crop was sold off the books.
    A counter that read the shed alone would say the seller produced nothing.
    """
    kept = _observe_scripted(_carrot_policy(sell_on_day=None), steps=24 * 6)
    sold = _observe_scripted(_carrot_policy(sell_on_day=4), steps=24 * 6)

    assert kept["produced"].get("CARROT", 0) > 0, "the farm grew carrots"
    assert sold["sold_units"].get("CARROT", 0) > 0, "…and the second run really sold them"
    assert sold["produced"] == kept["produced"], (
        f"selling changed reported production: {sold['produced']} vs {kept['produced']}"
    )


# --------------------------------------------------------------- runner discipline

def test_a_seed_block_too_small_for_the_game_count_is_refused():
    """Silently recycling seeds is how a held-out block stops being held out."""
    with pytest.raises(SystemExit, match="Widen --seeds"):
        run(["boatlee", "starter"], [1, 2], games=80, both_seats=True, config=CONFIG,
            workers=1, jsonl=None)


def test_nothing_is_pending_and_the_whole_pool_resolves():
    """`flooder`/`tomato_rusher` were PENDING until S1 gave them plans; the compiler landed at C4.

    The PENDING mechanism itself is kept (L2 will add replay agents the same way), so this asserts
    the invariant rather than the empty dict: every name the harness advertises must build, and
    anything that cannot must say which task creates it.
    """
    for name in ("flooder", "tomato_rusher", "compiler"):
        assert name not in PENDING
        assert get(name).build(), f"{name} is registered but does not build"
    for name, why in PENDING.items():
        assert why.split(":")[0].strip(), f"{name} is pending without a task ID"
        with pytest.raises(NotImplementedError):
            get(name)


def test_unknown_agents_fail_with_the_pool_listed():
    with pytest.raises(KeyError, match="unknown agent"):
        get("no_such_agent")


def test_results_rows_carry_the_fingerprint_and_the_seed_block(tmp_path):
    """Two runs can never be pooled by name alone: the row says which build and which block."""
    path = os.path.join(tmp_path, "games.jsonl")
    run(["boatlee", "starter"], [700, 701], games=2, both_seats=False, config=CONFIG,
        workers=1, jsonl=path)

    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 2
    row = rows[0]
    assert row["seed_block"] == "700:702"
    assert len(row["a_fingerprint"]) == 12 and row["a_fingerprint"] != row["b_fingerprint"]
    assert row["counters_a"]["steps_per_useful"] > 0
    assert row["config"] == CONFIG


def test_seed_blocks_parse_as_half_open_ranges():
    assert parse_seeds("21000:21004") == [21000, 21001, 21002, 21003]
    with pytest.raises(SystemExit):
        parse_seeds("21000")
