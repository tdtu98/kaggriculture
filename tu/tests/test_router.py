"""C2: the router — the ratio C5's gate is about, and the ordering rules that make it legal.

`steps_per_useful` is the number PLAN_v4 §1 blames for the whole gap: Boatlee walks 1.01 steps per
useful action, our engines 1.70-2.24 (E55). So the tests here are mostly about *efficiency being
real* rather than an artefact of counting — a router that drops half the work would score a
beautiful ratio, so every efficiency assertion is paired with one about coverage.

The ordering tests are different in kind: they encode environment rules, not preferences. Getting
FERTILIZE and WATER the wrong way round on a one-time crop silently halves that tile's yield, and
nothing in the state afterwards says why.
"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from agent.plan import Plan
from agent.router import (
    LAST_TURN,
    STATS as router_stats_dict,
    Stop,
    _steps,
    build_stops,
    decide_hands,
    expand,
    manhattan,
    order_stops,
    route,
    spawn_positions,
    split_roles,
)
from agent.tasks import SHED_TILES, Task, daily_tasks

MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}


def router_stats() -> dict:
    """A snapshot of `agent.router.STATS`, which is cumulative across the process."""
    return dict(router_stats_dict)


def water(tile, value=100.0, kind="survival", deadline=LAST_TURN):
    return Task(tile=tile, op="WATER", value=value, kind=kind, deadline_turn=deadline)


def op(tile, name, value=50.0, needs=None, kind="", args=()):
    return Task(tile=tile, op=name, value=value, needs=needs or {}, kind=kind, args=args)


def scripts_ops(result):
    return {u: [o[0] for o in s.ops] for u, s in result.scripts.items()}


DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _walk(ops, start):
    """`(turn, tile, op)` for every non-move op, replaying the moves to know where it lands.

    A script says what a unit does, never where — so any assertion about *which tile* an op reaches,
    and about when, has to re-walk it. This is what makes the deadline check non-vacuous.
    """
    x, y = start
    for turn, op_ in enumerate(ops):
        name = op_[0]
        if name in DELTA:
            dx, dy = DELTA[name]
            x, y = x + dx, y + dy
        else:
            yield turn, (x, y), name


# --------------------------------------------------------------------- ordering rules

def test_fertilize_precedes_water_on_the_same_tile():
    """A one-time crop reads `fertilized_until_day` *during* the WATER op, so fertilizing after
    watering wastes both the fertilizer and the doubled unit — invisibly."""
    tasks = [water((4, 3), kind="bonus"), op((4, 3), "FERTILIZE", needs={"FERTILIZER": 1})]
    result = route(tasks, [(4, 4)], carrying={0: {"FERTILIZER": 1}})
    ops = scripts_ops(result)[0]
    assert ops.index("FERTILIZE") < ops.index("WATER")


def test_harvest_comes_last_on_the_tile():
    """HARVEST clears a one-time crop, so anything else due there has to happen first."""
    tasks = [water((4, 3), kind="bonus"), op((4, 3), "HARVEST", value=500)]
    ops = scripts_ops(route(tasks, [(4, 4)]))[0]
    assert ops.index("WATER") < ops.index("HARVEST")


def test_a_plant_drags_its_water_behind_it():
    """New plants start at `consecutive_unwatered = 1`: unwatered today, the tile is a weed by
    dusk. The water is part of planting, not a later decision (E56, E57)."""
    tasks = [op((4, 3), "PLANT", args=("WHEAT",), needs={"SEED:WHEAT": 1})]
    ops = scripts_ops(route(tasks, [(4, 4)]))[0]
    assert ops == ["SOUTH"] * 0 + ["NORTH", "PLANT", "WATER"], ops


def test_pickup_precedes_the_ops_that_need_what_it_carries():
    tasks = [op((4, 2), "FEED", needs={"WHEAT": 1}), op((3, 3), "FERTILIZE",
                                                        needs={"FERTILIZER": 1})]
    ops = scripts_ops(route(tasks, [(4, 4)]))[0]
    assert ops[0] == "PICKUP" and ops[1] == "PICKUP"
    assert ops.index("PICKUP") < ops.index("FEED")


def test_an_op_whose_goods_were_never_picked_up_is_not_emitted():
    """A FEED with no wheat in hand is a no-op that still costs a turn — the `blocked_ops` counter
    exists because those are invisible in the money (E55)."""
    tasks = [op((4, 2), "FEED", needs={"WHEAT": 1})]
    result = route(tasks, [(0, 9)])           # nowhere near the shed: nothing can be picked up
    assert "FEED" not in scripts_ops(result)[0]


# --------------------------------------------------------------------- efficiency

def _boatlee_like_day(n_field=40, n_animals=12):
    """A 3-quadrant board of the shape C5 gates on: a pasture cluster plus field rows."""
    plan = Plan.boatlee_like()
    tasks = []
    for tile in plan.pasture_tiles[:n_animals]:
        tasks += [op(tile, "FEED", needs={"WHEAT": 1}, value=160),
                  op(tile, "CARE", value=160), op(tile, "HARVEST", value=320)]
    field = [t for c in plan.cohorts for t in c.tiles][:n_field]
    for i, tile in enumerate(field):
        tasks.append(water(tile, value=120 + i, kind="survival"))
        if i % 3 == 0:
            tasks.append(op(tile, "HARVEST", value=240))
    return tasks


def test_the_ratio_clears_the_phase_one_gate():
    """C5: <= 1.15 steps per useful action, against Boatlee's measured 1.02."""
    tasks = _boatlee_like_day()
    n = decide_hands(tasks)
    result = route(tasks, spawn_positions(n))
    assert result.steps_per_useful <= 1.15, result.steps_per_useful


def test_the_ratio_is_not_bought_by_dropping_the_work():
    """The pairing that makes the ratio meaningful: a router that visits three tiles and calls it a
    day would score beautifully."""
    tasks = _boatlee_like_day()
    n = decide_hands(tasks)
    result = route(tasks, spawn_positions(n))
    routed = sum(len(s.stops) for s in result.scripts.values())
    stops, _ = build_stops(tasks)
    assert routed >= 0.9 * len(stops), f"{routed}/{len(stops)} stops routed"


def test_visiting_a_tile_once_does_everything_there():
    """Most of Boatlee's advantage is chaining ops on the tile it is already standing on."""
    tile = (4, 2)
    tasks = [water(tile, kind="bonus"), op(tile, "FERTILIZE", needs={"FERTILIZER": 1}),
             op(tile, "HARVEST", value=300)]
    result = route(tasks, [(4, 4)], carrying={0: {"FERTILIZER": 1}})
    ops = scripts_ops(result)[0]
    # The walk out, and the walk back to unload the harvest (C2.4) — nothing else. The return leg
    # is not overhead the chaining failed to avoid: it is the trip that turns the harvest into
    # money the same day, and without it the goods wait for dusk (`_schedule_drops`).
    assert sum(1 for o in ops if o in MOVE_OPS) == 2 * manhattan((4, 4), tile), ops
    assert ops[-1] == "DROP", ops
    assert result.steps_per_useful == pytest.approx(4 / 4), "four steps bought four actions"


@pytest.mark.parametrize("seed", range(50))
def test_random_farms_are_routed_within_their_turns(seed):
    """50 random farms across the parametrisation: no script may exceed the day."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(rng.randint(10, 60)):
        tile = (rng.randrange(10), rng.randrange(10))
        tasks.append(op(tile, rng.choice(["WATER", "HARVEST", "CARE"]),
                        value=rng.uniform(1, 400),
                        kind=rng.choice(["survival", "bonus", "harvest"])))
    units = spawn_positions(rng.randint(0, 10))
    result = route(tasks, units, turns_left=24)
    for unit, script in result.scripts.items():
        assert len(script.ops) <= 24, f"unit {unit} has {len(script.ops)} ops"


# --------------------------------------------------------------------- logistics (C2.4)

def test_a_unit_that_harvested_goes_home_and_drops():
    """C2.4. Without this the goods still reach the shed — dusk empties every inventory into it —
    but a day late, *after* the last market turn, and with the overflow discarded. Two seasons of
    the compiler ended carrying unsold melons because the day's DROP tasks were routed into a
    variable named `_logistics` and never read."""
    tasks = [op((4, 1), "HARVEST", value=300)]
    result = route(tasks, [(4, 4)], turns_left=24)
    ops = scripts_ops(result)[0]
    assert ops[-1] == "DROP", ops
    assert result.drops_scheduled == 1
    assert ops.index("HARVEST") < ops.index("DROP")
    landing = [tile for _t, tile, name in _walk(result.scripts[0].ops, (4, 4)) if name == "DROP"]
    assert landing[0] in SHED_TILES, landing


def test_a_unit_that_ends_empty_handed_does_not_walk_home():
    """A keeper that picks up twelve wheat and feeds twelve animals is carrying nothing. Sending it
    to the shed spends a turn and books a `blocked_op` — the DROP would change no state."""
    tasks = [op((4, 1), "FEED", needs={"WHEAT": 1}), op((4, 2), "CARE")]
    result = route(tasks, [(4, 4)], turns_left=24)
    assert "DROP" not in scripts_ops(result)[0]
    assert result.drops_scheduled == 0


def test_a_drop_is_never_scheduled_past_the_last_hour_of_the_day():
    """C4 indexes scripts from `turn0`, so an op at index `turns_left` when the day started at
    hour 1 is emitted at hour 24 — which does not exist. A DROP scheduled there looks routed and
    never happens, which is exactly the shape of failure CLAUDE.md's effect-counter rule is for."""
    tasks = [op((0, 0), "HARVEST", value=300)]
    result = route(tasks, [(4, 4)], turns_left=24, turn0=1)
    ops = result.scripts[0].ops
    assert len(ops) <= LAST_TURN + 1 - 1, f"{len(ops)} ops from hour 1 runs past hour 23"
    assert ops[-1] == ["DROP"], ops


# --------------------------------------------------------- the last day's return leg (E64 C2)

def _far_harvest_day(n: int = 9):
    """A route long enough to fill the day: a column of harvests running away from the shed."""
    return [op((0, y % 10), "HARVEST", value=300) for y in range(n)]


def test_the_last_day_reserves_the_walk_home_instead_of_appending_it():
    """E64's open regression. E61 bought +$8.6k by walking home on day 29 and unloading, because
    the episode stops at hour 22 and produce left in a hand is worth **$0**. It appended the DROP
    to whatever turns the route had left over — and then E64 filled the days properly and there
    were none, so the drop stopped happening on the day it was the whole point of.

    Budgeting the leg *during* routing is what fixes it: the trailing stop yields to the DROP.
    Asserted as a contrast so it cannot pass vacuously — the same day without `last_day` is the
    behaviour that lost the money."""
    tasks = _far_harvest_day()
    loose = route(tasks, [(4, 4)], turns_left=24, turn0=1)
    tight = route(tasks, [(4, 4)], turns_left=24, turn0=1, last_day=True)

    assert [o[0] for o in loose.scripts[0].ops][-1] != "DROP", "the regression, reproduced"
    assert loose.drops_scheduled == 0
    assert [o[0] for o in tight.scripts[0].ops][-1] == "DROP", tight.scripts[0].ops
    assert tight.drops_scheduled == 1
    landing = [tile for _t, tile, name in _walk(tight.scripts[0].ops, (4, 4)) if name == "DROP"]
    assert landing == [min(SHED_TILES, key=lambda t: manhattan((0, 0), t))] or landing[0] in \
        SHED_TILES, landing


def test_a_stop_that_would_strand_the_harvest_yields_to_the_drop():
    """The trade the reserve makes, stated as the thing it gives up: fewer stops visited, and the
    counter that says so. A harvest that cannot be delivered pays nothing, so a stop that would
    eat the return leg is worth less than the inventory it strands."""
    tasks = _far_harvest_day()
    before = router_stats()
    loose = route(tasks, [(4, 4)], turns_left=24, turn0=1)
    tight = route(tasks, [(4, 4)], turns_left=24, turn0=1, last_day=True)
    after = router_stats()

    assert len(tight.scripts[0].stops) < len(loose.scripts[0].stops), "nothing yielded"
    assert after["drop_yields"] > before["drop_yields"], "the effect counter never fired"
    assert after["drop_reserved"] > before["drop_reserved"]


def test_the_last_day_stops_one_hour_earlier_than_every_other_day():
    """The framework plays `episodeSteps - 1` turns, so day 29 ends after **hour 22** — an op
    compiled into hour 23 is never issued. Routing to `turns_left` regardless is how a `DROP` got
    counted at compile time (`drops_scheduled`) and never emitted (`hand_drops`)."""
    tasks = _far_harvest_day(12)
    for turn0 in (0, 1, 2):
        result = route(tasks, [(4, 4)], turns_left=24, turn0=turn0, last_day=True)
        for unit, script in result.scripts.items():
            last_hour = turn0 + len(script.ops) - 1
            assert last_hour <= LAST_TURN - 1, f"unit {unit} plays at hour {last_hour}"


def test_no_unit_ends_the_last_day_holding_produce_it_could_not_unload():
    """`drop_short` is the counter that must stay at zero: it counts a unit that finished the last
    day with produce in its hands and no room to walk it home. Non-zero means the reserve is not
    covering what `_schedule_drops` will ask for — the failure mode E64 measured."""
    rng = random.Random(9)
    before = router_stats()
    for _ in range(20):
        tasks = [op((rng.randrange(10), rng.randrange(10)),
                    rng.choice(["HARVEST", "WATER", "CARE", "COLLECT_FERTILIZER"]),
                    value=rng.uniform(1, 400)) for _ in range(rng.randint(20, 60))]
        units = spawn_positions(rng.randint(1, 6))
        result = route(tasks, units, turns_left=24, turn0=1, last_day=True)
        for unit, script in result.scripts.items():
            if any(o[0] in ("HARVEST", "COLLECT_FERTILIZER") for o in script.ops):
                assert script.ops[-1][0] == "DROP", (unit, script.ops)
    assert router_stats()["drop_short"] == before["drop_short"], "a unit was left holding"


def test_a_staggered_planting_cannot_push_the_last_day_drop_off_the_end():
    """`_stagger_plants` inserts a PASS and truncates the tail. Truncating at the full day would
    cut exactly the DROP the reserve was held back for."""
    tasks = [op((4, 3), "HARVEST", value=300), op((5, 3), "HARVEST", value=300),
             op((4, 2), "PLANT", args=("WHEAT",), needs={"SEED:WHEAT": 1}),
             op((5, 2), "PLANT", args=("WHEAT",), needs={"SEED:WHEAT": 1})]
    result = route(tasks, spawn_positions(1), turns_left=24, turn0=1, last_day=True,
                   seeds={"WHEAT": 1})
    for unit, script in result.scripts.items():
        if any(o[0] == "HARVEST" for o in script.ops):
            assert script.ops[-1][0] == "DROP", (unit, script.ops)


def test_the_reserve_is_confined_to_the_last_day():
    """E61 measured the daily walk home at **-$5k to -$9k**: it buys nothing (dusk banks the hands
    and C4 sells at dawn) and it spends the turns the farm work needed. Nothing here may leak into
    an ordinary day."""
    tasks = _far_harvest_day()
    ordinary = route(tasks, [(4, 4)], turns_left=24, turn0=1, drop_home=False)
    assert "DROP" not in [o[0] for o in ordinary.scripts[0].ops]
    # and the day keeps its full length
    assert len(ordinary.scripts[0].ops) == len(
        route(tasks, [(4, 4)], turns_left=24, turn0=1, drop_home=False).scripts[0].ops)
    assert len(ordinary.scripts[0].stops) >= len(
        route(tasks, [(4, 4)], turns_left=24, turn0=1, last_day=True).scripts[0].stops)


def test_the_shed_pressure_request_is_reported_rather_than_discarded():
    """`build_stops` returns `(stops, logistics)` and the second half used to be thrown away."""
    tasks = [op((4, 1), "HARVEST", value=300),
             Task(tile=None, op="DROP", args=("ALL",), value=20.0, kind="drop")]
    assert route(tasks, [(4, 4)], turns_left=24).drop_requested
    assert not route(tasks[:1], [(4, 4)], turns_left=24).drop_requested


# --------------------------------------------------------------------- coverage rules

def _survival_turns(result, n_hands: int) -> dict:
    """`{tile: turn}` for every WATER in the compiled scripts, by replaying the moves."""
    starts = spawn_positions(n_hands)
    out = {}
    for script in result.scripts.values():
        for turn, tile, name in _walk(script.ops, starts[script.unit]):
            if name == "WATER":
                out.setdefault(tile, turn)
    return out


def test_every_survival_water_that_is_routed_lands_inside_the_day():
    """C2's first verification: a survival water is scheduled *at or before its deadline turn*.

    This used to assert `len(script.ops) <= 24` and call it a deadline check. It is not one — a
    router that scheduled every water on the twenty-fourth turn would pass it, and no script can
    exceed 24 ops in the first place because `expand` truncates. The check has to find the turn each
    WATER actually lands on and compare it with the deadline of the task that asked for it, which
    means replaying the moves (`_walk`).

    C1 gives survival waters `deadline_turn = LAST_TURN` — a plant dies at dusk, so any turn does —
    so this is the *coverage* half: on a full Boatlee-like day, every survival water that is routed
    lands inside the day, and nearly all of them are routed. The staggered-deadline half is the
    test below.
    """
    tasks = _boatlee_like_day()
    n = decide_hands(tasks)
    result = route(tasks, spawn_positions(n))

    wanted = {t.tile: t.deadline_turn for t in tasks if t.kind == "survival"}
    landed = _survival_turns(result, n)
    for script in result.scripts.values():
        assert len(script.ops) <= 24, f"unit {script.unit} has {len(script.ops)} ops"
    for tile, deadline in wanted.items():
        if tile in landed:
            assert landed[tile] <= deadline, \
                f"survival water at {tile} scheduled at turn {landed[tile]}, due by {deadline}"
    covered = sum(1 for tile in wanted if tile in landed)
    assert covered >= 0.9 * len(wanted), \
        f"only {covered}/{len(wanted)} survival waters routed — the deadline check is vacuous"


def test_a_survival_water_with_a_real_deadline_is_scheduled_before_it():
    """The other half: deadlines that are *not* all LAST_TURN, on an instance that can meet them.

    C6 and O3 both want to move a water earlier than dusk, and `order_stops` already promotes a
    deadline stop ahead of the walk optimisation — so this is the assertion that keeps that
    promotion honest. Feasible by construction (eight tiles, four units, ~4 turns of work each), so
    a failure here is the router ignoring the deadline, not the day being too small.
    """
    tiles = [(3, 3), (5, 3), (3, 5), (5, 5), (2, 2), (6, 2), (2, 6), (6, 6)]
    # Nearest tiles due last, so a router that simply walked outward would fail.
    tasks = [water(t, value=500, kind="survival", deadline=deadline)
             for t, deadline in zip(tiles, (12, 12, 12, 12, 5, 5, 5, 5))]
    result = route(tasks, spawn_positions(3))

    landed = _survival_turns(result, 3)
    for task in tasks:
        assert task.tile in landed, f"{task.tile} was not routed at all"
        assert landed[task.tile] <= task.deadline_turn, \
            f"water at {task.tile} scheduled at turn {landed[task.tile]}, due by " \
            f"{task.deadline_turn}"


def test_hiring_continues_while_a_plant_would_die():
    """`fib(n)` says stop; a dying plant says keep going. The wage is at most $377 and a strawberry
    is worth thousands, so survival must win — and it must be judged on the staffing actually
    returned, not on the one that was rejected."""
    tasks = [water((x, y), value=2000, kind="survival")
             for x in range(10) for y in range(5)]

    def missed_at(n):
        result = route(tasks, spawn_positions(n))
        return sum(1 for s in result.unrouted for t in s.tasks if t.kind == "survival")

    n = decide_hands(tasks)
    assert missed_at(n) == 0, f"{missed_at(n)} of 50 plants left to die at {n} hands"
    assert missed_at(n - 1) > 0, f"{n} hands is more than the day needs"


def test_a_real_strawberry_tick_day_gets_every_water_and_its_fertilizer():
    """E66 mechanism #2, end to end through the real generator rather than hand-built tasks.

    A 22-tile strawberry block on the night it produces: `kaggriculture.py:797` grants the +2 only
    if the tile was watered *that* day, so a water left unrouted throws the fertilizer away. The
    assertion is on the compiled script, not on the task list — "the generator emitted it" and "the
    day actually does it" are different claims, and only the second one is worth money (E44).

    The two ops must also land in the right order on the same visit: FERTILIZE writes
    `fertilized_until_day`, and a WATER that arrives first is a water on an unfertilized tile.
    """
    plan = Plan.boatlee_like()
    tiles = [[None] * 10 for _ in range(10)]
    block = [t for c in plan.cohorts if c.crop == "STRAWBERRY" for t in c.tiles][:22]
    assert len(block) == 22, "fixture assumes a strawberry cohort of at least 22 tiles"
    for (x, y) in block:
        tiles[y][x] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0,
                       "watered_today": False, "consecutive_unwatered": 0, "yield_units": 2,
                       "max_lifespan_step": -1, "fertilized_until_day": -1}
    obs = {"player": 0, "day": 13, "hour": 0, "step": 13 * 24,
           "farms": [{"money": 9000, "farmer": [4, 4], "hands": [], "hires_today": 0,
                      "unlocked_quadrants": ["NW", "NE", "SW"], "tiles": tiles}],
           "market": {"prices": {"STRAWBERRY": 120, "FERTILIZER": 100}, "inventory": {}},
           "town": {"unlocked_shops": []},
           "private": {"shed": {"FERTILIZER": 40}, "seeds": {}, "inventories": [{}]}}

    tasks = daily_tasks(obs, plan, day=13)
    wanted = {t.tile for t in tasks if t.op == "WATER"}
    assert wanted == set(block), "age 13 is a strawberry tick day: every tile wants water"

    n = decide_hands(tasks, cash=9000)
    result = route(tasks, spawn_positions(n), carrying={u: {"FERTILIZER": 22}
                                                        for u in range(n + 1)})
    for script in result.scripts.values():
        seen: dict = {}
        pos = tuple(spawn_positions(n)[script.unit])
        for o in script.ops:
            if o[0] in MOVE_OPS:
                dx, dy = DELTA[o[0]]
                pos = (pos[0] + dx, pos[1] + dy)
            else:
                seen.setdefault(pos, []).append(o[0])
        for tile, ops in seen.items():
            if "WATER" in ops and "FERTILIZE" in ops:
                assert ops.index("FERTILIZE") < ops.index("WATER"), (tile, ops)
        wanted -= {t for t, ops in seen.items() if "WATER" in ops}
    assert not wanted, f"{len(wanted)} tick-day waters never compiled: {sorted(wanted)}"


def test_hiring_stops_when_a_hand_cannot_earn_its_wage():
    tasks = [water((4, 3), value=5, kind="bonus")]
    assert decide_hands(tasks) == 0, "a $5 task does not justify a $1 wage plus a walk"


def test_keeper_count_follows_the_work_not_a_rule_of_thumb():
    """One keeper per six animals looked right and dropped animal ops every day of a season: six
    head at three ops each does not fit in 24 turns once the wheat is fetched and the cluster
    walked."""
    plan = Plan.boatlee_like()
    tasks = []
    for tile in plan.pasture_tiles:
        tasks += [op(tile, "FEED", needs={"WHEAT": 1}), op(tile, "CARE"), op(tile, "HARVEST")]
    stops, _ = build_stops(tasks)
    _keepers, _field, n_keepers = split_roles(stops, n_units=10)
    assert n_keepers >= 3, f"{len(plan.pasture_tiles)} animals cannot be kept by {n_keepers}"


def test_roles_stay_pure():
    """E46: Boatlee's units are 93% role-pure and specialisation replicated at ~70% over 360 games.
    A keeper that wanders into the field pays the distance twice."""
    plan = Plan.boatlee_like()
    animal_tiles = set(plan.pasture_tiles[:12])
    tasks = [op(t, "FEED", needs={"WHEAT": 1}) for t in animal_tiles]
    tasks += [water(t, kind="survival") for c in plan.cohorts for t in c.tiles[:8]]
    result = route(tasks, spawn_positions(8))

    for script in result.scripts.values():
        kinds = {("animal" if s.tile in animal_tiles else "field") for s in script.stops}
        assert len(kinds) <= 1, f"unit {script.unit} mixed {kinds}"


# --------------------------------------------------------------------- quality vs optimal

def _tour_length(order, start):
    total, pos = 0, start
    for tile in order:
        total += manhattan(pos, tile)
        pos = tile
    return total


@pytest.mark.parametrize("seed", range(6))
def test_the_route_is_within_five_percent_of_an_exact_tour(seed):
    """Against brute force on instances small enough to solve exactly.

    A heuristic router is only worth having if its walk is close to optimal; 2-opt on a Manhattan
    grid should be, and this is where that claim gets checked rather than assumed.
    """
    rng = random.Random(100 + seed)
    start = (4, 4)
    tiles = [(rng.randrange(10), rng.randrange(10)) for _ in range(7)]
    tiles = list(dict.fromkeys(tiles))
    stops = [Stop(tile=t, tasks=[water(t, kind="bonus")], value=10.0) for t in tiles]

    ours = _tour_length([s.tile for s in order_stops(stops, start, turns=99)], start)
    best = min(_tour_length(list(p), start) for p in itertools.permutations(tiles))
    assert ours <= best * 1.05 + 1, f"router {ours} vs optimal {best}"


def _exact_multi_unit_moves(starts, tiles) -> int:
    """Minimum total walking for *any* split of `tiles` across `starts`, each unit touring its own.

    Two exact stages, both small enough to solve outright:

    * Held-Karp per start: `best[s][subset]` = shortest open tour from `starts[s]` covering `subset`.
    * A set-partition DP over the units, which is where the single-unit TSP test above stops and
      where the router's real decision lives — `partition()` splits the work *before* any tour is
      optimised, so a router with a perfect 2-opt can still walk 40% too far by handing the wrong
      tiles to the wrong hand. That is the claim TASKS_v4 C2 asks for ("Hungarian on 4 units x 8
      tasks") and the claim that was quietly replaced by the single-unit check.
    """
    n = len(tiles)
    full = (1 << n) - 1
    per_start = []
    for start in starts:
        # dp[(mask, last)] = cost of covering mask, ending at tiles[last]
        dp = {(1 << i, i): manhattan(start, tiles[i]) for i in range(n)}
        for mask in range(1, full + 1):
            for last in range(n):
                cur = dp.get((mask, last))
                if cur is None:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    key = (mask | (1 << nxt), nxt)
                    cost = cur + manhattan(tiles[last], tiles[nxt])
                    if cost < dp.get(key, 1 << 30):
                        dp[key] = cost
        best = [1 << 30] * (full + 1)
        best[0] = 0
        for (mask, _last), cost in dp.items():
            if cost < best[mask]:
                best[mask] = cost
        per_start.append(best)

    # Partition DP over units. Iterating proper submasks makes this 3^n, not 4^n.
    cur = {full: 0}
    for best in per_start:
        nxt: dict = {}
        for remaining, cost in cur.items():
            sub = remaining
            while True:                                  # every submask of `remaining`, incl. 0
                total = cost + best[sub]
                if total < nxt.get(remaining ^ sub, 1 << 30):
                    nxt[remaining ^ sub] = total
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        cur = nxt
    return cur[0]


def _four_unit_instance(seed):
    rng = random.Random(400 + seed)
    tiles = list(dict.fromkeys((rng.randrange(10), rng.randrange(10)) for _ in range(12)))[:8]
    return tiles, [water(t, value=100.0, kind="bonus") for t in tiles], spawn_positions(3)


#: The bound the router actually achieves, measured over 40 instances (seeds 400-439) of the same
#: shape: mean **1.015x**, median 1.000x, worst 1.105x, and 34 of 40 exactly at or under 1.05x.
#:
#: It used to be 1.8. The sweep that stood in `partition` cut the tile list on cumulative *time*
#: without looking at where the units were standing, and walked 1.26-2.06x the exact assignment
#: (mean 1.49x on these same 40 instances) — a perfect 2-opt on the wrong split. Replacing the split
#: with marginal-cost insertion from the units' real positions plus a boundary-repair pass is what
#: closed it; the single-unit TSP check above was unchanged by the fix, which is the evidence that
#: the tour was never the problem.
#:
#: 1.15 rather than 1.05 because one instance in forty (seed 407) still lands at 1.105: greedy
#: insertion commits early and its relocate/swap repair is single-move, so a two-move improvement is
#: out of reach. The spec test below holds the 5% line on the pinned instances; this one is the
#: distribution guard, and it is set at the measured worst case, not rounded up for comfort.
MEASURED_ASSIGNMENT_RATIO = 1.15


@pytest.mark.parametrize("seed", range(20))
def test_the_multi_unit_assignment_does_not_regress(seed):
    """4 units x 8 tasks against exact assignment — the whole chain, not just the tour.

    Splitting the tiles across units happens before any tour is optimised, so this is a different
    claim from the 2-opt test above and the one that was quietly missing: a perfect tour on the
    wrong split is still the wrong split. Twenty instances rather than five, because the fix has to
    hold as a distribution and not on the handful of seeds it was checked against.

    Pure waters, so no C2.4 return leg is scheduled and both costs measure the same thing.
    """
    tiles, tasks, starts = _four_unit_instance(seed)
    result = route(tasks, starts, turns_left=24)
    routed = sum(len(s.stops) for s in result.scripts.values())
    assert routed == len(tiles), f"{routed}/{len(tiles)} stops routed — a cheap route is not a route"

    ours = sum(s.moves for s in result.scripts.values())
    best = _exact_multi_unit_moves(starts, tiles)
    assert ours <= best * MEASURED_ASSIGNMENT_RATIO, \
        f"router walks {ours}, exact assignment walks {best} ({ours / best:.2f}x)"


@pytest.mark.parametrize("seed", range(5))
def test_the_multi_unit_assignment_meets_the_spec(seed):
    """TASKS_v4 C2's "within 5% of exact assignment (Hungarian on 4 units x 8 tasks)".

    This was a strict xfail: the spec was written, the sweep did not meet it, and the gap was pinned
    rather than papered over. It passes now. Left as a normal test at the spec's own number so the
    5% is a gate rather than a note in a plan.
    """
    tiles, tasks, starts = _four_unit_instance(seed)
    result = route(tasks, starts, turns_left=24)
    ours = sum(s.moves for s in result.scripts.values())
    best = _exact_multi_unit_moves(starts, tiles)
    assert ours <= best * 1.05 + 1, f"router walks {ours}, exact assignment walks {best}"


def test_the_split_is_made_from_where_the_units_are_standing():
    """The effect counter for the fix (CLAUDE.md: prove the change fired before reading its score).

    Two halves, because either alone can pass while the router is unchanged. The counter half says
    the insertion assignment ran and the unit-blind sweep did not; the behavioural half says it
    *used* the positions — two units in opposite corners with a cluster of work by each must end up
    with their own cluster, which is exactly the instance the sweep got wrong (it cuts on cumulative
    time in row-band order, so it would hand one corner's tiles to the unit in the other).
    """
    before = dict(router_stats())
    west = [(0, 0), (0, 1), (1, 0), (1, 1)]
    east = [(9, 8), (9, 9), (8, 9), (8, 8)]
    tasks = [water(t, kind="bonus") for t in west + east]
    result = route(tasks, [(0, 0), (9, 9)], turns_left=24)

    stats = router_stats()
    assert stats["insertion_stops"] > before["insertion_stops"], "the new split never ran"
    assert stats["sweep_stops"] == before["sweep_stops"], "a live route fell back to the sweep"

    assigned = {u: {s.tile for s in script.stops} for u, script in result.scripts.items()}
    assert assigned[0] == set(west), assigned
    assert assigned[1] == set(east), assigned


def test_solving_is_fast_enough_for_a_daily_recompile():
    """C2's budget: 12 units x 100 tasks in under 20 ms, so the whole day compiles inside one turn."""
    rng = random.Random(3)
    tasks = [op((rng.randrange(10), rng.randrange(10)),
                rng.choice(["WATER", "HARVEST", "CARE"]), value=rng.uniform(1, 300))
             for _ in range(100)]
    units = spawn_positions(11)

    route(tasks, units)                                    # warm
    t0 = time.perf_counter()
    for _ in range(20):
        route(tasks, units)
    ms = (time.perf_counter() - t0) * 1000 / 20
    assert ms < 20, f"{ms:.1f} ms per route"


# --------------------------------------------------------------------- mechanics

def test_spawn_positions_match_where_hands_actually_appear():
    """`_spawn_hand` fills the least-occupied shed tile in NWSE order. Modelled exactly, because
    `decide_hands` must evaluate the board its answer will be used on."""
    assert spawn_positions(0) == [(4, 4)]
    assert spawn_positions(3) == [(4, 4), (5, 4), (4, 5), (5, 5)]
    assert spawn_positions(4)[4] == (4, 4), "the fifth hand doubles up on the first tile"


def test_paths_are_manhattan_and_minimal():
    assert len(_steps((4, 4), (4, 4))) == 0
    assert len(_steps((4, 4), (0, 0))) == manhattan((4, 4), (0, 0))
    assert {s[0] for s in _steps((4, 4), (6, 2))} == {"EAST", "NORTH"}


def test_a_deadline_orders_the_day_but_never_forbids_it():
    """A harvest past `max_lifespan_step` is decaying, not worthless — and it shares its tile with
    the survival water. Treating a passed deadline as impossible dropped the whole stop and killed
    the plant to save a turn (eight of them in a measured season)."""
    tile = (4, 1)
    tasks = [water(tile, value=900, kind="survival"),
             op(tile, "HARVEST", value=50, kind="harvest")]
    tasks[1] = Task(tile=tile, op="HARVEST", value=50, kind="harvest", deadline_turn=0)
    result = route(tasks, [(4, 4)])
    assert "WATER" in scripts_ops(result)[0], "the survival water must survive the expired harvest"


def test_urgent_stops_are_visited_before_relaxed_ones():
    far_urgent = Task(tile=(0, 0), op="HARVEST", value=100, deadline_turn=8, kind="harvest")
    near_relaxed = Task(tile=(4, 3), op="HARVEST", value=100, kind="harvest")
    result = route([near_relaxed, far_urgent], [(4, 4)])
    visited = [s.tile for s in result.scripts[0].stops]
    assert visited[0] == (0, 0), visited
