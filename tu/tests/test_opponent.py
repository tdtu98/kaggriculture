"""O2: naming the opponent from the shared board, and pricing what they are about to sell.

The rule this file is written against is CLAUDE.md's "prove the change fired": a fingerprint that
returns the right string when handed a hand-built dict is not evidence that it locks in a game. So
the load-bearing tests here **play**: they run boatlee, starter, executor_v7 and our own compiler
through kagsim and read the answer off the ledger `main_v4` keeps during the episode.

The tables in `agent/opponent.py` are measurements, and a measurement goes stale when the thing it
measured changes (CLAUDE.md, four occurrences). `test_sell_schedule_still_matches_a_replay` and
`test_profiles_still_match_a_replay` re-derive them from a live game, so a boatlee rebuild or a
plan change fails here rather than silently degrading O3.
"""

from __future__ import annotations

import time

import pytest

from agent import main_v4, opponent
from agent.opponent import (
    CHECKPOINTS,
    CONDITIONAL_SELLS,
    FEATURES,
    PROFILES,
    SELL_SCHEDULE,
    THRESHOLD,
    census,
    distance,
    fingerprint,
    forecast_supply,
    new_memory,
    next_sell,
    sell_schedule,
    sell_units,
)


def _farm(tiles=(), quads=("NW",), hands=()):
    board = [[None] * 10 for _ in range(10)]
    for x, y, tile in tiles:
        board[y][x] = tile
    return {"tiles": board, "unlocked_quadrants": list(quads), "hands": list(hands),
            "farmer": [4, 4], "money": 3000.0}


def _plant(crop, planted_day=0, yield_units=0):
    return {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
            "yield_units": yield_units, "watered_today": True, "consecutive_unwatered": 0,
            "fertilized_until_day": -1, "max_lifespan_step": -1}


def _play(a_name, b_name, seed, steps, snap_at=()):
    """Run a real game; hand back seat 1's view of seat 0's farm at `snap_at`, and the actions.

    Snapshots have to be taken *at* the step, not read off the final board — a census is a moving
    object and comparing day 3 against a day-1 profile is how the first version of this test
    "refuted" a working fingerprint.
    """
    import kagsim

    from harness import registry

    agents = [registry.get(a_name).build(), registry.get(b_name).build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    trace, snaps = [], {}
    for step in range(steps):
        if step in snap_at:
            # `farms` is shared observation (`kaggriculture.json`, `kaggriculture.py:271`), which
            # is the whole premise of O2: seat 1 can read seat 0's tiles.
            snaps[step] = sim.observation(1)["farms"][0]
        acts = [agents[p](sim.observation(p)) for p in (0, 1)]
        trace.append(acts)
        sim.step(acts)
    return snaps, trace


# ------------------------------------------------------------------ the census

def test_census_counts_crops_animals_pasture_quads_and_hands():
    farm = _farm(
        tiles=[(0, 0, _plant("WHEAT")), (1, 0, _plant("WHEAT")), (2, 0, _plant("MELON")),
               (3, 0, {"kind": "PASTURE", "animal": "COW", "placed_day": 1}),
               (4, 0, {"kind": "PASTURE", "animal": None})],
        quads=("NW", "NE"), hands=([1, 1], [2, 2], [3, 3]))
    c = census(farm)
    assert c["WHEAT"] == 2 and c["MELON"] == 1 and c["CARROT"] == 0
    assert c["COW"] == 1 and c["SHEEP"] == 0
    assert c["PASTURE"] == 2 and c["QUADS"] == 2 and c["HANDS"] == 3
    assert set(c) == set(FEATURES)


def test_census_ignores_weeds_and_empty_tiles():
    """The one part of the day-1..3 board the RNG moves — and the only thing that varied across
    the 70 boatlee observations the profiles were measured from."""
    clean = _farm(tiles=[(0, 0, _plant("WHEAT"))])
    weedy = _farm(tiles=[(0, 0, _plant("WHEAT")),
                         (5, 5, {"kind": "WEED"}), (6, 5, {"kind": "WEED"})])
    assert census(clean) == census(weedy)


def test_boatlee_is_separable_from_every_other_profile_by_more_than_the_threshold():
    """The threshold is only meaningful next to the gap it has to sit inside.

    Measured: boatlee's census spread over 70 observations is 0, the nearest other profile is 6
    away at every checkpoint. A threshold of 2 therefore cannot make two profiles alive at once
    where one of them is boatlee — which is the false positive the O2 bar is about.
    """
    for step in CHECKPOINTS:
        for name, prof in PROFILES.items():
            if name == "boatlee":
                continue
            d = distance(PROFILES["boatlee"][step], prof[step])
            assert d > 2 * THRESHOLD, f"{name} is only {d} from boatlee at step {step}"


def test_a_census_off_by_more_than_the_threshold_kills_the_profile():
    """Mutation check on the distance gate itself: nudge the board past `THRESHOLD` and the lock
    must not happen. Without this the gate could be `<= 999` and every test above still passes."""
    mem = new_memory()
    tiles = [(x, 0, _plant("WHEAT")) for x in range(5)] + \
            [(x, 1, _plant("MELON")) for x in range(5)] + \
            [(x, 2, {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0}) for x in range(4)] + \
            [(4, 2, {"kind": "PASTURE", "animal": "COW", "placed_day": 0})]
    exact = _farm(tiles=tiles)
    assert distance(census(exact), PROFILES["boatlee"][24]) == 0
    off = _farm(tiles=tiles + [(x, 3, _plant("WHEAT")) for x in range(THRESHOLD + 1)])
    fingerprint(off, 24, mem)
    fingerprint(off, 48, mem)
    assert mem["known"] != "boatlee"
    assert "boatlee" not in mem["alive"]


def test_fingerprint_only_works_at_the_checkpoints():
    mem = new_memory()
    assert fingerprint(_farm(), 23, mem) is None
    assert mem["checks"] == 0


# ------------------------------------------------------------------ in play

@pytest.mark.parametrize("seed", [66300, 66301])
def test_fingerprint_locks_boatlee_in_a_real_game(seed):
    snaps, _ = _play("boatlee", "pass", seed, max(CHECKPOINTS) + 1, snap_at=CHECKPOINTS)
    mem = new_memory()
    known = None
    for step in CHECKPOINTS:
        known = fingerprint(snaps[step], step, mem)
    assert known == "boatlee"
    assert mem["lock_step"] == 48


@pytest.mark.parametrize("name", ["starter", "executor_v7", "compiler"])
def test_fingerprint_never_names_boatlee_for_anyone_else(name):
    snaps, _ = _play(name, "pass", 66300, max(CHECKPOINTS) + 1, snap_at=CHECKPOINTS)
    mem = new_memory()
    for step in CHECKPOINTS:
        fingerprint(snaps[step], step, mem)
    assert mem["known"] != "boatlee"
    assert "boatlee" not in mem["alive"]


def test_the_counters_fire_in_play_and_only_against_boatlee():
    """E44: the counter, read from the agent's own season ledger during a game."""
    import kagsim

    from harness import registry

    seen = {}
    for opp in ("boatlee", "starter"):
        agents = [registry.get("compiler").build(), registry.get(opp).build()]
        sim = kagsim.Sim({"episodeSteps": 720, "seed": 66302})
        for _ in range(max(CHECKPOINTS) + 1):
            sim.step([agents[p](sim.observation(p)) for p in (0, 1)])
        seen[opp] = main_v4.counters(0)

    assert seen["boatlee"]["opponent_is_boatlee"] == 1
    assert seen["boatlee"]["fingerprint_lock_step"] == 48
    assert seen["boatlee"].get("fingerprint_errors", 0) == 0
    assert seen["starter"].get("opponent_is_boatlee", 0) == 0
    assert seen["starter"]["opponent_is_starter"] == 1


def test_profiles_still_match_a_replay():
    """The library is a measurement; this is the tripwire for it going stale."""
    for name in ("boatlee", "compiler", "starter", "executor_v7"):
        snaps, _ = _play(name, "pass", 66303, max(CHECKPOINTS) + 1, snap_at=CHECKPOINTS)
        for step in CHECKPOINTS:
            seen = census(snaps[step])
            d = distance(seen, PROFILES[name][step])
            assert d <= THRESHOLD, f"{name} at step {step} is {d} from its profile: {seen}"


def test_sell_schedule_still_matches_a_replay():
    """Every step in the table is a step boatlee really orders at, and nothing is missing.

    Orders, not sales — the table is explicitly the order book (`SETTLE_RATE` carries the rest),
    so this compares against what the agent asked for, which is exactly what it records.
    """
    _, trace = _play("boatlee", "pass", 66304, 719)
    seen: dict[str, dict[int, int]] = {}
    for step, acts in enumerate(trace):
        for order in (acts[0].get("market") or [])[:10]:
            if order and order[0] == "SELL":
                seen.setdefault(order[1], {})[step] = \
                    seen.setdefault(order[1], {}).get(step, 0) + int(order[2])

    assert set(seen) == set(SELL_SCHEDULE)
    for item, rows in SELL_SCHEDULE.items():
        conditional = {s for s, _, _ in CONDITIONAL_SELLS.get(item, ())}
        assert set(seen[item]) - conditional == {s for s, _ in rows}, item
        for step, units in rows:
            assert seen[item][step] == units, (item, step)


# ------------------------------------------------------------------ forecast

def test_forecast_reads_their_standing_crops_off_the_shared_board():
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    today = 5
    farm = _farm(tiles=[(0, 0, _plant("MELON", planted_day=today))])
    out = forecast_supply(farm, today * 24)
    assert "MELON" in out
    arrival = (today + int(CROPS["MELON"]["max_yield_day"])) * 24
    assert [s for s, _ in out["MELON"]] == [arrival]
    assert out["MELON"][0][1] > 0


def test_forecast_upgrades_to_the_measured_schedule_once_boatlee_is_known():
    farm = _farm(tiles=[(0, 0, _plant("MELON", planted_day=9))])
    step = 240
    unknown = forecast_supply(farm, step, known=None)
    known = forecast_supply(farm, step, known="boatlee")
    assert known["MELON"] != unknown["MELON"]
    expected = [s for s, _ in SELL_SCHEDULE["MELON"] if step < s <= step + 14 * 24]
    assert [s for s, _ in known["MELON"]] == expected


def test_forecast_never_reports_a_step_in_the_past():
    farm = _farm(tiles=[(0, 0, _plant("STRAWBERRY", planted_day=0))])
    for step in (0, 240, 480, 700):
        for rows in forecast_supply(farm, step, known="boatlee").values():
            assert all(s > step for s in [r[0] for r in rows])


def test_sell_accessors_answer_only_for_boatlee():
    assert sell_schedule("MELON")[:2] == [252, 255]
    assert sell_schedule("MELON", known="starter") == []
    assert sell_units("MELON", 252) == 6
    assert sell_units("MELON", 253) == 0
    assert next_sell("MELON", 253) == (255, 6)
    assert next_sell("MELON", 999) is None


def test_fingerprint_and_forecast_fit_the_turn_budget():
    """5 ms per call is the O2 bar; the p99 turn budget it has to sit inside is 100 ms."""
    snaps, _ = _play("boatlee", "pass", 66305, max(CHECKPOINTS) + 1, snap_at=(24,))
    farm = snaps[24]
    mem = new_memory()
    worst = 0.0
    for _ in range(200):
        t0 = time.perf_counter()
        known = fingerprint(farm, 24, dict(mem))
        forecast_supply(farm, 24, known)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 5e-3, f"{worst * 1e3:.2f} ms"
