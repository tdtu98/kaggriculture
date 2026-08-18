"""C1: the day's task list — nothing missing, nothing impossible, priced in one currency.

These are hand-built boards with answers that can be derived from `kaggriculture.py` by hand, which
is the point: the generator's job is to encode the rules exactly, and a test that just re-ran the
generator's own arithmetic would confirm nothing (E39).

The values matter as much as the presence of a task. The router has no priority ladder — it trades
tasks off by dollars — so if a survival water is priced below a harvest, a plant dies and no test
that only checked "a WATER task exists" would notice.
"""

from __future__ import annotations

import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS

from agent.plan import Plan
from agent.tasks import (
    LAST_TURN,
    Task,
    daily_tasks,
    market_needs,
    remaining_units,
    replant_cycle,
    tick_days,
    water_window,
    wave_size,
    _fertilize_gain,
)

PRICES = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
          "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}


def plant(crop, planted_day, *, watered=False, unwatered=0, units=None, fert_until=-1,
          lifespan=None):
    cd = CROPS[crop]
    return {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "watered_today": watered, "consecutive_unwatered": unwatered,
        "yield_units": (0 if cd["ongoing"] else 1) if units is None else units,
        "max_lifespan_step": (-1 if cd["ongoing"] else (planted_day + cd["max_yield_day"] + 1) * 24)
        if lifespan is None else lifespan,
        "fertilized_until_day": fert_until,
    }


def animal(species, placed_day, *, fed=False, cared=False, units=0, unfed=0, bank=0, fert=False):
    return {
        "kind": ANIMALS[species]["structure"], "animal": species, "placed_day": placed_day,
        "yield_units": units, "consecutive_unfed": unfed, "fed_today": fed,
        "cared_today": cared, "fertilizer_available": fert, "pending_care_bonus": bank,
    }


def board(at=None):
    """A 10x10 board, `at={(x, y): tile}`."""
    tiles = [[None] * 10 for _ in range(10)]
    for (x, y), tile in (at or {}).items():
        tiles[y][x] = tile
    return tiles


def obs_for(tiles, *, day=0, shed=None, seeds=None, inventories=None, hands=()):
    return {
        "player": 0, "day": day, "hour": 0, "step": day * 24,
        "farms": [{"money": 3000, "farmer": [4, 4], "hands": [list(h) for h in hands],
                   "unlocked_quadrants": ["NW", "NE", "SW"], "hires_today": 0, "tiles": tiles}],
        "market": {"prices": dict(PRICES), "inventory": {k: 10000 for k in PRICES}},
        "town": {"unlocked_shops": []},
        "private": {"shed": dict(shed or {}), "seeds": dict(seeds or {}),
                    "inventories": [dict(i) for i in (inventories or [{}])]},
    }


EMPTY_PLAN = Plan(pasture_tiles=(), land_days={"NE": 99, "SW": 99, "SE": 99}, herd=(),
                  cohorts=(), hands="auto", consts={})


def find(tasks, op, tile=None):
    return [t for t in tasks if t.op == op and (tile is None or t.tile == tile)]


def one(tasks, op, tile=None) -> Task:
    got = find(tasks, op, tile)
    assert len(got) == 1, f"expected exactly one {op}, got {got}"
    return got[0]


# --------------------------------------------------------------------- survival水

def test_a_plant_one_day_from_death_always_yields_a_survival_water():
    """`consecutive_unwatered == 1` and dry means it becomes a WEED at this dusk."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=1)})
    tasks = daily_tasks(obs_for(tiles, day=5), EMPTY_PLAN)

    water = one(tasks, "WATER", (2, 2))
    assert water.kind == "survival"
    assert water.value > 0


def test_a_newly_planted_tile_is_a_survival_water_the_same_day():
    """New plants start at `consecutive_unwatered = 1` — planting day counts as unwatered — so a
    cohort planted and not watered dies that evening. This is the failure the counters caught the
    harness missing entirely (E55)."""
    tiles = board({(1, 1): plant("MELON", 3, unwatered=1)})
    tasks = daily_tasks(obs_for(tiles, day=3), EMPTY_PLAN)
    assert one(tasks, "WATER", (1, 1)).kind == "survival"


def test_a_watered_plant_asks_for_nothing():
    tiles = board({(2, 2): plant("STRAWBERRY", 0, watered=True, unwatered=0)})
    assert find(daily_tasks(obs_for(tiles, day=5), EMPTY_PLAN), "WATER") == []


def test_survival_value_falls_as_a_plant_nears_the_end_of_its_life():
    """A ladder cannot express this, and it is the whole reason value is a number: a strawberry
    with three ticks left must outbid one with none."""
    young = board({(2, 2): plant("STRAWBERRY", 0, unwatered=1)})
    spent = board({(2, 2): plant("STRAWBERRY", 0, unwatered=1)})
    v_young = one(daily_tasks(obs_for(young, day=9), EMPTY_PLAN), "WATER").value
    v_spent = one(daily_tasks(obs_for(spent, day=15), EMPTY_PLAN), "WATER").value
    assert v_young > v_spent, (v_young, v_spent)


# --------------------------------------------------------------------- bonus water

def test_a_one_time_crop_inside_its_window_is_worth_watering():
    """WHEAT accrues on the WATER op itself over ages 2-4, +1 a day (+2 fertilized)."""
    assert water_window("WHEAT") == (2, 4)
    tiles = board({(0, 0): plant("WHEAT", 0, unwatered=0)})
    task = one(daily_tasks(obs_for(tiles, day=3), EMPTY_PLAN), "WATER", (0, 0))
    assert task.kind == "bonus"
    assert task.value == PRICES["WHEAT"]


def test_fertilizer_doubles_what_a_bonus_water_is_worth():
    tiles = board({(0, 0): plant("WHEAT", 0, unwatered=0, fert_until=5)})
    task = one(daily_tasks(obs_for(tiles, day=3), EMPTY_PLAN), "WATER", (0, 0))
    assert task.value == 2 * PRICES["WHEAT"]


def test_a_one_time_crop_outside_its_window_is_not_worth_a_turn():
    """Age 1 is before wheat's window opens: the water keeps it alive but adds nothing, so it is
    only offered when survival needs it."""
    tiles = board({(0, 0): plant("WHEAT", 0, unwatered=0)})
    assert find(daily_tasks(obs_for(tiles, day=1), EMPTY_PLAN), "WATER") == []


def test_an_ongoing_crop_is_only_worth_watering_on_a_fertilized_tick_day():
    """Ongoing crops accrue at dusk, and the fertilizer bonus requires the tile to have been
    watered that day — so on a tick day with fertilizer live, the water is worth a unit.

    Age 11 for the negative arm, not 9: age 9 is a fertilize age, and a tile being fertilized today
    is fertilized by the time dusk reads it (see below)."""
    assert tick_days("STRAWBERRY") == (9, 11, 13, 15)
    fertilized = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0, fert_until=11)})
    task = one(daily_tasks(obs_for(fertilized, day=11), EMPTY_PLAN), "WATER", (2, 2))
    assert task.value == PRICES["STRAWBERRY"]

    plain = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0)})
    assert find(daily_tasks(obs_for(plain, day=11), EMPTY_PLAN), "WATER") == []


def test_an_ongoing_crop_off_its_tick_is_not_worth_watering():
    """Age 10 is between strawberry ticks: nothing accrues that dusk however wet the tile is."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0, fert_until=11)})
    assert find(daily_tasks(obs_for(tiles, day=10), EMPTY_PLAN), "WATER") == []


# ------------------------------------------------------- E66: the tick-day water

def test_a_tick_day_water_is_survival_class():
    """`kaggriculture.py:797` — `fertilized = was_watered and fertilized_until_day >= current_day`.
    Miss the water on the production night and the whole fertilizer application is void, and the
    turn does not come back. That is the same unrepeatable loss a thirsty plant faces, so it takes
    the same class: the router rescues dropped survival stops and `decide_hands` will not sign off
    a staffing level that leaves one unrouted (E66 measured 67.0% tick-day watered vs 98.6%)."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0, fert_until=11)})
    assert one(daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN), "WATER", (2, 2)).survival


def test_a_strawberry_fertilized_today_still_wants_its_tick_day_water():
    """The defect itself. Strawberry fertilizes at ages 9 and 13, and its ticks are at 9/11/13/15 —
    so on half of all tick days the fertilizer has *not* landed yet when the day is planned. Reading
    `fertilized_until_day` alone priced those waters at zero and emitted no task at all, which is
    why the router watered 67% of tick days while watering more than boatlee overall."""
    for age in (9, 13):
        tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0,
                                     units=0 if age == 9 else 2)})
        tasks = daily_tasks(obs_for(tiles, day=age, shed={"FERTILIZER": 4}), EMPTY_PLAN)
        water = one(tasks, "WATER", (2, 2))
        assert water.survival, age
        assert water.value == PRICES["STRAWBERRY"], age
        # And the fertilize it depends on is on the same tile, so the router does both in one visit.
        assert one(tasks, "FERTILIZE", (2, 2)).tile == (2, 2)


def test_tomato_wants_a_water_on_every_one_of_its_tick_days():
    """TOMATO first yields at 8 with interval 1, so it ticks on ages 7-10 and fertilizes at 7 and
    10 — the same trap, on a crop that produces every night."""
    assert tick_days("TOMATO") == (7, 8, 9, 10)
    for age, fert_until in ((7, -1), (8, 9), (9, 9), (10, -1)):
        tiles = board({(3, 3): plant("TOMATO", 0, unwatered=0, units=0, fert_until=fert_until)})
        water = one(daily_tasks(obs_for(tiles, day=age), EMPTY_PLAN), "WATER", (3, 3))
        assert water.survival, age
        assert water.value == PRICES["TOMATO"], age


def test_a_ripe_tick_day_tile_is_still_worth_watering():
    """A tile sitting at the cap of 4 looks like it has no room for tonight's +2 — but it is being
    harvested today, which empties it. Reading `yield_units` as-is priced the water at zero on
    exactly the tiles carrying the most fruit."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0, units=4, fert_until=11)})
    tasks = daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN)
    assert find(tasks, "HARVEST", (2, 2))
    assert one(tasks, "WATER", (2, 2)).value == PRICES["STRAWBERRY"]


def test_a_tick_day_water_is_not_doubled_up():
    """Exactly one WATER per tile per day: the survival lane and the tick lane are the same task,
    not two. A second one is a wasted turn, and turns are what E64 showed convert to money."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=1, fert_until=11)})
    tasks = daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN)
    assert len(find(tasks, "WATER", (2, 2))) == 1
    # Thirst still prices it: the whole rest of the plant, not one unit.
    assert one(tasks, "WATER", (2, 2)).value > PRICES["STRAWBERRY"]


def test_an_already_watered_tick_day_tile_asks_for_nothing():
    tiles = board({(2, 2): plant("STRAWBERRY", 0, watered=True, unwatered=0, fert_until=11)})
    assert find(daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN), "WATER") == []


def test_tick_day_waters_are_counted():
    """CLAUDE.md: prove the change fired before reading its score. A zero counter is an unfinished
    implementation, not a negative result."""
    from agent.tasks import STATS

    before = dict(STATS)
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0),          # fertilized today
                   (3, 3): plant("STRAWBERRY", 0, unwatered=0, fert_until=11)})
    daily_tasks(obs_for(tiles, day=9), EMPTY_PLAN)
    daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN)
    # day 9: both tiles tick, (2, 2) is the fertilize-today case. day 11: only (3, 3) is covered.
    assert STATS["tick_waters"] - before["tick_waters"] == 3
    assert STATS["tick_waters_fert_today"] - before["tick_waters_fert_today"] == 1


# --------------------------------------------------------------------- fertilize

def test_strawberry_at_age_nine_wants_water_then_fertilizer():
    """The canonical case from TASKS_v4 C1: a dry strawberry at age 9 yields WATER and FERTILIZE,
    and the fertilize carries its requirement so the router knows to pick fertilizer up first."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=1)})
    tasks = daily_tasks(obs_for(tiles, day=9, shed={"FERTILIZER": 2}), EMPTY_PLAN)

    assert one(tasks, "WATER", (2, 2)).kind == "survival"
    fert = one(tasks, "FERTILIZE", (2, 2))
    assert fert.needs == {"FERTILIZER": 1}
    # covers ages 9-11, which catches the ticks at 9 and 11: two extra units
    assert fert.value == 2 * PRICES["STRAWBERRY"]


def test_fertilizer_is_only_offered_on_the_ages_that_pay():
    """Ages 9 and 13 double all four strawberry ticks; every other age is wasted fertilizer."""
    for age, wanted in ((8, False), (9, True), (10, False), (13, True), (14, False)):
        tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0)})
        tasks = daily_tasks(obs_for(tiles, day=age), EMPTY_PLAN)
        assert bool(find(tasks, "FERTILIZE")) is wanted, f"age {age}"


def test_fertilizer_is_not_offered_twice_while_it_is_still_active():
    tiles = board({(2, 2): plant("STRAWBERRY", 0, unwatered=0, fert_until=11)})
    assert find(daily_tasks(obs_for(tiles, day=9), EMPTY_PLAN), "FERTILIZE") == []


def test_tomato_and_melon_fertilize_on_their_own_schedules():
    """The audit's correction: tomato at 7 and 10, melon at 6 — not strawberry's ages."""
    assert _fertilize_gain(plant("TOMATO", 0), 7) == 3       # covers ticks 7, 8, 9
    assert _fertilize_gain(plant("TOMATO", 0), 10) == 1      # covers the last tick
    assert _fertilize_gain(plant("MELON", 0), 6) == 3        # three watered days doubled


# --------------------------------------------------------------------- harvest

def test_an_ongoing_crop_is_harvested_whenever_it_holds_units():
    tiles = board({(2, 2): plant("STRAWBERRY", 0, watered=True, units=2)})
    task = one(daily_tasks(obs_for(tiles, day=10), EMPTY_PLAN), "HARVEST", (2, 2))
    assert task.value == 2 * PRICES["STRAWBERRY"]


def test_a_one_time_crop_is_not_offered_for_harvest_mid_window():
    """HARVEST clears a one-time tile, so harvesting at age 2 does not bank early income — it
    throws away the two waterings still to come."""
    tiles = board({(0, 0): plant("WHEAT", 0, units=2)})
    assert find(daily_tasks(obs_for(tiles, day=2), EMPTY_PLAN), "HARVEST") == []


def test_a_one_time_crop_is_harvested_once_it_stops_accruing():
    tiles = board({(0, 0): plant("WHEAT", 0, units=4)})
    assert find(daily_tasks(obs_for(tiles, day=4), EMPTY_PLAN), "HARVEST", (0, 0))


def test_wheat_past_its_window_carries_a_same_day_deadline():
    """`max_lifespan_step = (planted + max_yield_day + 1) * 24` — dawn of age 5 for wheat — and
    decay costs a unit every two steps from there."""
    tiles = board({(0, 0): plant("WHEAT", 0, units=4)})
    task = one(daily_tasks(obs_for(tiles, day=5), EMPTY_PLAN), "HARVEST", (0, 0))
    assert task.deadline_turn < LAST_TURN, "a decaying crop cannot wait for the evening"


def test_a_full_ongoing_tile_facing_a_tick_tonight_must_be_emptied_today():
    """Cap 4 and a fertilized tick of +2: leave it and tonight's production is thrown away."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, watered=True, units=3, fert_until=11)})
    task = one(daily_tasks(obs_for(tiles, day=11), EMPTY_PLAN), "HARVEST", (2, 2))
    assert task.value > 3 * PRICES["STRAWBERRY"], "the overflow is part of what the harvest saves"


# --------------------------------------------------------------------- animals

def test_an_unfed_cow_yields_a_feed_that_needs_wheat():
    """The canonical animal case: FEED carries its wheat requirement, which is what makes the
    router schedule a PICKUP before walking to the pasture."""
    tiles = board({(4, 3): animal("COW", 0)})
    tasks = daily_tasks(obs_for(tiles, day=3, shed={"WHEAT": 10}), EMPTY_PLAN)

    feed = one(tasks, "FEED", (4, 3))
    assert feed.needs == {"WHEAT": 1}
    assert find(tasks, "PICKUP"), "the day needs wheat carried out to the pasture"


def test_an_animal_facing_its_second_unfed_dusk_is_a_survival_task():
    """It escapes at `consecutive_unfed >= 2`, leaving an empty structure and a wasted $400."""
    tiles = board({(4, 3): animal("COW", 0, unfed=1)})
    feed = one(daily_tasks(obs_for(tiles, day=3), EMPTY_PLAN), "FEED", (4, 3))
    assert feed.kind == "survival"
    assert feed.value > ANIMALS["COW"]["cost"]


def test_feeding_on_a_production_day_is_worth_the_whole_care_bank():
    """The bank pays out only if the animal is fed that day, and is cleared otherwise."""
    tiles = board({(4, 3): animal("COW", 0, bank=2)})
    feed = one(daily_tasks(obs_for(tiles, day=7), EMPTY_PLAN), "FEED", (4, 3))
    assert feed.value == 3 * PRICES["MILK"], "1 base + 2 banked, all at risk"


def test_care_is_offered_every_day_and_harvest_when_the_bucket_has_milk():
    tiles = board({(4, 3): animal("COW", 0, fed=True, units=3, fert=True)})
    tasks = daily_tasks(obs_for(tiles, day=8), EMPTY_PLAN)
    assert one(tasks, "CARE", (4, 3)).value == PRICES["MILK"]
    assert one(tasks, "HARVEST", (4, 3)).value >= 3 * PRICES["MILK"]
    assert one(tasks, "COLLECT_FERTILIZER", (4, 3)).value == PRICES["FERTILIZER"]


# --------------------------------------------------------------------- planting and weeds

def test_cohorts_due_today_become_plant_tasks_that_need_seed():
    plan = Plan.boatlee_like()
    tiles = board()
    tasks = daily_tasks(obs_for(tiles, day=0), plan)
    plants = find(tasks, "PLANT")

    due = [c for c in plan.cohorts if c.plant_day == 0]
    # Every cohort's *first* fill is whole, cycling or not: only the re-sowing is rationed.
    assert len(plants) == sum(c.n_tiles for c in due)
    assert all(t.needs for t in plants)
    assert all(t.value > 0 for t in plants), "a cohort worth planting is worth more than its seed"


def test_an_occupied_tile_is_not_replanted():
    plan = Plan.boatlee_like()
    day0 = [c for c in plan.cohorts if c.plant_day == 0][0]
    tiles = board({day0.tiles[0]: plant(day0.crop, 0)})
    tasks = daily_tasks(obs_for(tiles, day=0), plan)
    assert not find(tasks, "PLANT", day0.tiles[0])


def test_a_cycling_cohort_replants_a_tile_that_has_been_harvested():
    """`replant` is the gene that turns 141 wheat plantings into one cohort (E56)."""
    plan = Plan.boatlee_like()
    wheat = [c for c in plan.cohorts if c.replant][0]
    tasks = daily_tasks(obs_for(board(), day=wheat.plant_day + 4), plan)
    assert find(tasks, "PLANT", wheat.tiles[0]), "a cleared tile in a cycling cohort is replanted"


def test_a_cycling_cohort_is_resown_a_wave_at_a_time_not_as_a_batch():
    """E66 mechanism 3: a block that comes back all at once goes out all at once, and the base
    collapses to nothing for a dawn every cycle."""
    plan = Plan.boatlee_like()
    wheat = [c for c in plan.cohorts if c.replant][0]
    cycle = replant_cycle(wheat.crop)
    later = wheat.plant_day + cycle + 3
    wave = wave_size(wheat, later, wheat.plant_day)
    assert 0 < wave < len(wheat.tiles), "the fixture must have more tiles than one wave"

    # The first fill is whole — capping it starves the day-7 strawberry block of cash (-$45.8k).
    first = find(daily_tasks(obs_for(board(), day=wheat.plant_day), plan), "PLANT")
    assert len([t for t in first if t.tile in wheat.tiles]) == len(wheat.tiles)

    # After a cycle the same empty block is rationed, and stays rationed: the cap is a rate, which
    # is what stops the cohort re-batching after any run of bad days.
    for day in (later, later + 4):
        resown = find(daily_tasks(obs_for(board(), day=day), plan), "PLANT")
        assert len([t for t in resown if t.tile in wheat.tiles]) == wave


def test_a_sown_once_cohort_is_not_staggered():
    plan = Plan.boatlee_like()
    once = [c for c in plan.cohorts if not c.replant and c.plant_day == 0][0]
    plants = find(daily_tasks(obs_for(board(), day=0), plan), "PLANT")
    assert len([t for t in plants if t.tile in once.tiles]) == len(once.tiles)


def test_wave_size_spreads_a_cycling_block_over_its_own_cycle():
    plan = Plan.boatlee_like()
    for cohort in plan.cohorts:
        if not cohort.replant:
            continue
        cycle = replant_cycle(cohort.crop)
        assert wave_size(cohort) * cycle >= len(cohort.tiles), \
            "a wave a day must be able to keep the whole block in the ground"
        assert wave_size(cohort) >= 1


def test_wheat_replant_cycle_matches_the_env_lifespan_rule():
    """`max_lifespan_step = (planted_day + max_yield_day + 1) * 24` — kaggriculture.py:224."""
    assert replant_cycle("WHEAT") == 5
    assert replant_cycle("MELON") == 13
    assert replant_cycle("STRAWBERRY") == 1, "ongoing crops never free their tile"


def test_weeds_are_dug_only_where_the_plan_wants_the_ground():
    plan = Plan.boatlee_like()
    wanted = plan.cohorts[0].tiles[0]
    spare = (9, 9)                      # SE: boatlee_like never buys it and no cohort claims it
    assert spare not in plan.occupied()
    tiles = board({wanted: {"kind": "WEED"}, spare: {"kind": "WEED"}})
    tasks = daily_tasks(obs_for(tiles, day=plan.cohorts[0].plant_day), plan)

    assert find(tasks, "DIG", wanted)
    assert not find(tasks, "DIG", spare), "digging ground the plan does not want is walking"


# --------------------------------------------------------------------- logistics

def test_a_full_shed_produces_a_drop_before_dusk():
    """Dusk drops inventories into the shed and *discards* the overflow, so a day's harvest can
    evaporate at 90+; the DROP has to happen today, not tomorrow."""
    tiles = board({(2, 2): plant("STRAWBERRY", 0, watered=True)})
    tasks = daily_tasks(obs_for(tiles, day=5, shed={"WHEAT": 95}), EMPTY_PLAN)
    assert find(tasks, "DROP")


def test_market_needs_reports_only_the_shortfall():
    tiles = board({(4, 3): animal("COW", 0), (2, 2): plant("STRAWBERRY", 0, unwatered=0)})
    o = obs_for(tiles, day=9, shed={"WHEAT": 0, "FERTILIZER": 0})
    tasks = daily_tasks(o, EMPTY_PLAN)
    needs = market_needs(tasks, o["private"])
    assert needs.get("WHEAT") == 1 and needs.get("FERTILIZER") == 1

    stocked = obs_for(tiles, day=9, shed={"WHEAT": 5, "FERTILIZER": 5})
    assert market_needs(daily_tasks(stocked, EMPTY_PLAN), stocked["private"]) == {}


def test_seed_shortfalls_are_reported_separately_from_carried_goods():
    """Seeds never pass through a unit's hands — they are consumed from `private['seeds']` by the
    PLANT itself — so they are bought but never picked up."""
    plan = Plan.boatlee_like()
    o = obs_for(board(), day=0)
    tasks = daily_tasks(o, plan)
    needs = market_needs(tasks, o["private"])
    assert any(k.startswith("SEED:") for k in needs)
    assert not find(tasks, "PICKUP", None) or all(
        not t.args[0].startswith("SEED") for t in find(tasks, "PICKUP"))


# --------------------------------------------------------------------- budget

def test_the_generator_is_fast_enough_to_run_every_dawn():
    """C4 compiles inside the turn budget, so the task list cannot be the expensive part."""
    import time

    plan = Plan.boatlee_like()
    tiles = board()
    for i, (x, y) in enumerate(plan.cohorts[0].tiles + plan.cohorts[1].tiles):
        tiles[y][x] = plant("STRAWBERRY", 0, unwatered=i % 2)
    for (x, y) in plan.pasture_tiles:
        tiles[y][x] = animal("COW", 0)
    o = obs_for(tiles, day=9, shed={"WHEAT": 20, "FERTILIZER": 20})

    daily_tasks(o, plan)                                   # warm
    t0 = time.perf_counter()
    for _ in range(20):
        tasks = daily_tasks(o, plan)
    ms = (time.perf_counter() - t0) * 1000 / 20

    assert len(tasks) > 40, "this board should be busy"
    assert ms < 2.0, f"{ms:.2f} ms per call on a 100-tile farm"


# --------------------------------------------------------------------- C6 price seam

def test_price_fn_is_the_live_quote_until_the_gene_turns_it_on():
    """C1′'s seam. `projected_pricing` is 0 by default — measured: pricing every task at the
    projected board lost $2,675 vs `starter` over 240 paired games (CI [-4,602, -749]) — so the
    default path must be the old one, exactly."""
    from dataclasses import replace

    from agent import projection
    from agent.tasks import make_price_fn

    projection.reset()
    o = obs_for(board(), day=6)
    o["town"]["unlocked_shops"] = ["PIZZA_SHOP", "PIZZA_SHOP"]
    plan = Plan.boatlee_like()

    spot = make_price_fn(o, replace(plan, consts={**plan.consts, "projected_pricing": 0}))
    assert spot("TOMATO") == PRICES["TOMATO"]
    assert spot("TOMATO", 8) == PRICES["TOMATO"], "a lead time must not move the live quote"

    projection.reset()
    ahead = make_price_fn(o, replace(plan, consts={**plan.consts, "projected_pricing": 1}))
    assert ahead("TOMATO", 0) == pytest.approx(PRICES["TOMATO"], abs=2)
    assert ahead("TOMATO", 8) > ahead("TOMATO", 0), "eight days of shop drain must show up"


def test_a_cohort_is_priced_at_the_board_it_will_meet():
    """The whole point of the seam: a strawberry sown today meets its first unit ten days out.
    Pricing it on today's board is what made melon look worthless (D17/E48)."""
    from agent.tasks import _cohort_value

    seen = {}

    def price(product, days_ahead=0):
        seen[product] = days_ahead
        return PRICES[product]

    _cohort_value("STRAWBERRY", price)
    assert seen["STRAWBERRY"] == CROPS["STRAWBERRY"]["first_yield_day"] == 10
    _cohort_value("WHEAT", price)
    assert seen["WHEAT"] == CROPS["WHEAT"]["first_yield_day"] == 2


def test_a_one_argument_price_fn_still_works():
    """Older callers and tests hand in `lambda p: ...`; the seam must not require a signature
    change from every call site."""
    from agent.tasks import _cohort_value

    assert _cohort_value("WHEAT", lambda p: PRICES[p]) == 4 * 25 - 10


def test_the_generator_is_still_fast_with_the_projection_on():
    """The projection is consulted per task valuation, so it has to be built once per turn and not
    once per tile — the difference between 0.3 ms and 30 ms."""
    import time
    from dataclasses import replace

    from agent import projection

    projection.reset()
    plan = Plan.boatlee_like()
    plan = replace(plan, consts={**plan.consts, "projected_pricing": 1})
    tiles = board()
    for i, (x, y) in enumerate(plan.cohorts[0].tiles + plan.cohorts[1].tiles):
        tiles[y][x] = plant("STRAWBERRY", 0, unwatered=i % 2)
    for (x, y) in plan.pasture_tiles:
        tiles[y][x] = animal("COW", 0)
    o = obs_for(tiles, day=9, shed={"WHEAT": 20, "FERTILIZER": 20})

    daily_tasks(o, plan)
    t0 = time.perf_counter()
    for i in range(20):
        o["step"] = 9 * 24 + i          # a new turn each time: the cache must not hide the cost
        tasks = daily_tasks(o, plan)
    ms = (time.perf_counter() - t0) * 1000 / 20

    assert len(tasks) > 40
    assert ms < 2.0, f"{ms:.2f} ms per call with the projection on"
