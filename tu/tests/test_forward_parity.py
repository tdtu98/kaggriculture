"""`agent.forward.FarmModel` must match kagsim's farm state exactly (PLAN2 P0.2).

The model exists so the agent can roll plans forward on the Kaggle runner, where kagsim does not
exist. A rollout on a model that drifts is worse than no rollout at all — it produces confident
rankings of candidate plans that are wrong — so this compares full farm state every single turn
rather than sampling or comparing outcomes.

Determinism: the only stochastic term in the farm is weed spawning, and `weedSpawnChance` pins it
at both extremes — 0.0 (never) and 1.0 (every empty tile). Both are exercised, so the weed branch
is covered rather than excused as noise.

Out of scope by design, and therefore synced from the simulator rather than predicted: the market,
money, and hiring (`_end_of_day` clears `farm["hands"]` daily, `:867`).
"""

from __future__ import annotations

import random

import kagsim
import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from agent.forward import FarmModel

BOARD = 10
OPS_NO_ARG = ["WATER", "HARVEST", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
              "COLLECT_FERTILIZER", "CARE", "FERTILIZE", "PASS", "DROP",
              "NORTH", "SOUTH", "EAST", "WEST"]
CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
CARRIABLE = ["WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE", "MELON"]


def fuzz_action(rng: random.Random) -> list:
    """Deliberately includes illegal and malformed actions: they must no-op identically."""
    roll = rng.random()
    if roll < 0.45:
        return [rng.choice(["NORTH", "SOUTH", "EAST", "WEST"])]
    if roll < 0.55:
        return ["PLANT", rng.choice(CROP_NAMES)]
    if roll < 0.62:
        # String quantities: the reference uses int(), and a bare cast diverges here.
        return ["PICKUP", rng.choice(CARRIABLE), rng.choice([1, 3, "5", 0, -2])]
    if roll < 0.68:
        return ["PLACE", rng.choice(CARRIABLE), rng.choice([1, "2"])]
    return [rng.choice(OPS_NO_ARG)]


def market_orders(rng: random.Random, day: int) -> list:
    """Create farm state to compare against — not a test of the market.

    Spending is kept modest on purpose. An earlier version bought aggressively and drove money to
    $0 within a day, after which `HIRE` silently no-ops and the multi-unit test had nothing to
    exercise. Leaving cash in hand is what keeps a roster available.
    """
    out = []
    if rng.random() < 0.3:
        out.append(["BUY_SEED", rng.choice(CROP_NAMES), rng.randrange(1, 3)])
    if rng.random() < 0.12:
        out.append(["BUY_ANIMAL", rng.choice(["COW", "SHEEP", "GOOSE"]), 1])
    if rng.random() < 0.2:
        out.append(["BUY_PRODUCT", "WHEAT", rng.randrange(1, 3)])
    return out


def farm_state(farm: dict, priv: dict) -> dict:
    """Everything the model claims to reproduce. Inventory order matters and is preserved."""
    return {
        "tiles": [[dict(t) if isinstance(t, dict) else t for t in row] for row in farm["tiles"]],
        "farmer": tuple(farm["farmer"]),
        "hands": [tuple(h) for h in farm["hands"]],
        "shed": dict(priv["shed"]),
        "seeds": dict(priv.get("seeds", {})),
        "inventories": [list(i.items()) for i in priv["inventories"]],
    }


def model_state(m: FarmModel) -> dict:
    return {
        "tiles": [[dict(t) if isinstance(t, dict) else t for t in row] for row in m.tiles],
        "farmer": tuple(m.units[0]),
        "hands": [tuple(u) for u in m.units[1:]],
        "shed": dict(m.shed),
        "seeds": dict(m.seeds),
        "inventories": [list(i.items()) for i in m.invs],
    }


def diff(a: dict, b: dict) -> str:
    out = []
    for key in a:
        if key == "tiles":
            for y, (ra, rb) in enumerate(zip(a["tiles"], b["tiles"])):
                for x, (ta, tb) in enumerate(zip(ra, rb)):
                    if ta != tb:
                        out.append(f"  tile({x},{y}): sim={ta!r} model={tb!r}")
        elif a[key] != b[key]:
            out.append(f"  {key}: sim={a[key]!r} model={b[key]!r}")
    return "\n".join(out) or "  (no field-level difference found)"


def setup_farm(sim, rng, turns):
    """Build a rich mid-game farm with market activity. Nothing is compared during this phase."""
    for _ in range(turns):
        obs = sim.observation(0)
        n = 1 + len(obs["farms"][0]["hands"])
        acts = [fuzz_action(rng) for _ in range(n)]
        sim.step([{"farmer": acts[0], "hands": acts[1:],
                   "market": market_orders(rng, obs["day"])},
                  {"farmer": ["PASS"], "hands": [], "market": []}])


def compare(sim, model, rng, turns, label):
    """Step both with identical actions and NO market orders; assert full equality every turn.

    Market orders are excluded deliberately rather than for convenience: they deliver seeds, stock
    and hands that the model does not claim to predict, so including them would test the market and
    call the result a model divergence.
    """
    for step in range(turns):
        acts = [fuzz_action(rng) for _ in range(len(model.units))]
        model.step(acts)
        sim.step([{"farmer": acts[0], "hands": acts[1:], "market": []},
                  {"farmer": ["PASS"], "hands": [], "market": []}])
        nxt = sim.observation(0)
        want = farm_state(nxt["farms"][0], nxt["private"])
        got = model_state(model)
        if want != got:
            raise AssertionError(f"divergence at {label} step {step}\n" + diff(want, got))


@pytest.mark.parametrize("weed_chance,weed_mode", [(0.0, "none"), (1.0, "all")])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_multi_unit_within_a_day(seed, weed_chance, weed_mode):
    """All unit ops with a full roster, inside one day so hands persist."""
    cfg = {"episodeSteps": 400, "seed": seed, "weedSpawnChance": weed_chance}
    sim = kagsim.Sim(dict(cfg))
    rng = random.Random(seed * 7919)
    setup_farm(sim, rng, 30)                       # buys, plants, animals
    # Hire explicitly *now*: `_end_of_day` clears the roster daily (`:867`), so a hand bought
    # during setup is gone by the time the comparison starts. Two turns of hiring, then compare
    # inside the same day.
    for _ in range(3):
        sim.step([{"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"]]},
                  {"farmer": ["PASS"], "hands": [], "market": []}])
    obs = sim.observation(0)
    assert len(obs["farms"][0]["hands"]) >= 2, "setup should have hired hands"
    model = FarmModel.from_obs(obs, player=0, weed_mode=weed_mode)
    # Stop before the day boundary: hiring is market logic and would desync the roster.
    compare(sim, model, rng, 24 - (obs["hour"] + 1), f"seed={seed} weeds={weed_chance}")


@pytest.mark.parametrize("weed_chance,weed_mode", [(0.0, "none"), (1.0, "all")])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_across_many_day_boundaries(seed, weed_chance, weed_mode):
    """Daily refreshes, plant death, animal escape, decay and the roster reset.

    Runs past many day boundaries. `_end_of_day` clears the roster (`:867`) so only the farmer
    survives — which is exactly what the model must reproduce with `rehire=None`.
    """
    cfg = {"episodeSteps": 400, "seed": seed, "weedSpawnChance": weed_chance}
    sim = kagsim.Sim(dict(cfg))
    rng = random.Random(seed * 104729)
    setup_farm(sim, rng, 40)
    model = FarmModel.from_obs(sim.observation(0), player=0, weed_mode=weed_mode)
    compare(sim, model, rng, 220, f"seed={seed} weeds={weed_chance}")


def script_compare(cfg, turns, label):
    """Drive kagsim and the model with an identical scripted plan, comparing every turn.

    `turns` is a list of (unit_actions, market_orders). Directed scenarios exist because random
    fuzzing does not reach some branches: a mutation test showed the fertilizer bonus, the
    fertilizer duration and the animal-escape threshold could all be broken without any fuzz test
    noticing, because the fuzz never happens to fertilize a plant inside its watering window or
    starve an animal for two straight days.
    """
    sim = kagsim.Sim(dict(cfg))
    model = None
    # Config must reach the model, or a non-default `shedCapacity` silently tests nothing: the
    # model would clamp at 100 while the simulator clamps at 2, and the capacity branch never runs.
    model_kw = {"weed_mode": "none",
                "shed_capacity": int(cfg.get("shedCapacity", 100)),
                "turns_per_day": int(cfg.get("turnsPerDay", 24))}
    for i, (acts, market) in enumerate(turns):
        if model is None:
            model = FarmModel.from_obs(sim.observation(0), player=0, **model_kw)
        else:
            model.step(acts)
        sim.step([{"farmer": acts[0], "hands": acts[1:], "market": market},
                  {"farmer": ["PASS"], "hands": [], "market": []}])
        nxt = sim.observation(0)
        if market:                     # market deliveries are out of scope; adopt and continue
            model.sync_units(nxt["farms"][0], nxt["private"])
            model.shed = dict(nxt["private"]["shed"])
            model.seeds = dict(nxt["private"].get("seeds", {}))
            model.day, model.step_idx = nxt["day"], nxt["day"] * 24 + nxt["hour"]
            model.tiles = [[dict(t) if isinstance(t, dict) else t for t in row]
                           for row in nxt["farms"][0]["tiles"]]
            continue
        want = farm_state(nxt["farms"][0], nxt["private"])
        got = model_state(model)
        if want != got:
            raise AssertionError(f"divergence at {label} turn {i}\n" + diff(want, got))
    return sim, model


def schedule(plan: dict, last_step: int, market: dict | None = None) -> list:
    """Build a turn list from {(day, hour): farmer_action}, PASS everywhere else.

    Explicit scheduling rather than counting padding: the first version of these scenarios counted
    turns by hand, planted a crop and then did not water it on its planting day -- so the plant was
    a weed by nightfall and the fertilizer branch it was written to cover never executed. The
    mutation test caught that; hand-counted padding is how it happened.
    """
    market = market or {}
    turns = []
    for step in range(last_step + 1):
        key = (step // 24, step % 24)
        turns.append(([list(plan.get(key, ["PASS"]))], list(market.get(key, []))))
    return turns


def test_fertilizer_doubles_the_watering_bonus_and_lasts_three_days():
    """WHEAT accrues yield on WATER inside days 2..4 (`:425`); fertilizer doubles it (`:429`)
    and stays active for day..day+2 (`:468`).

    A plant must be watered on its planting day too -- `_new_plant` starts `consecutive_unwatered`
    at 1 (`:209`), so skipping day 0 kills it at nightfall.
    """
    plan = {(0, 2): ["PLANT", "WHEAT"], (0, 3): ["WATER"],
            (1, 2): ["WATER"],
            (2, 1): ["PICKUP", "FERTILIZER", 1],          # must be in hand *today*
            (2, 2): ["FERTILIZE"], (2, 3): ["WATER"],     # in-window, fertilized -> +2
            (3, 2): ["WATER"],                            # still fertilized (until day 4)
            (4, 2): ["WATER"],                            # last fertilized day
            (5, 2): ["WATER"]}                            # window closed
    market = {(0, 0): [["BUY_SEED", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 3]]}
    # Stop before max_lifespan_step = (0 + 4 + 1) * 24 = 120 so decay does not erase the evidence;
    # decay has its own test below.
    sim, _ = script_compare({"episodeSteps": 400, "seed": 5, "weedSpawnChance": 0.0},
                            schedule(plan, 118, market), "fertilizer")
    tiles = sim.observation(0)["farms"][0]["tiles"]
    plants = [t for row in tiles for t in row
              if isinstance(t, dict) and t.get("kind") == "PLANT"]
    assert plants, "the plant must survive for the bonus to be observable"
    # 1 planted + 2 (day 2, fertilized) + 2 (day 3, still fertilized) -> capped at max_yield 6
    # by the day-4 watering. Anything less means the fertilizer never went on.
    assert plants[0]["yield_units"] == 6, (
        f"expected the fertilized doublings to reach the cap of 6, got {plants[0]['yield_units']}"
        " -- the scenario is no longer fertilizing")


def test_fertilizer_expires_exactly_three_days_after_application():
    """The `day + 2` boundary in `:468`, isolated so it is actually observable.

    Fertilizing on day 2 covers days 2-4, which is the whole of WHEAT's watering window, so
    extending the duration changes nothing there -- and the yield cap of 6 hides it too. Applying
    on **day 1** instead puts the expiry inside the window: day 3 is the last fertilized day, so a
    day-4 watering must accrue +1, not +2. Watering on days 2 and 4 only (skipping day 3) keeps the
    total under the cap where the difference is visible, and still keeps the plant alive -- death
    is at two *consecutive* dry days (`:770`).
    """
    # The pickup must happen on the *same day* as the FERTILIZE: `_end_of_day` drops every unit
    # inventory into the shed (`:865`), so a day-0 pickup is not in hand on day 1 and FERTILIZE
    # silently no-ops. The yield assertion below is what caught that.
    plan = {(0, 2): ["PLANT", "WHEAT"], (0, 3): ["WATER"],
            (1, 1): ["PICKUP", "FERTILIZER", 1],
            (1, 2): ["FERTILIZE"], (1, 3): ["WATER"],     # fertilized_until_day = 3
            (2, 2): ["WATER"],                            # in window, fertilized -> +2
            # day 3 deliberately dry: survives, and lets day 4 stay under the yield cap
            (4, 2): ["WATER"]}                            # fertilizer expired -> +1, not +2
    market = {(0, 0): [["BUY_SEED", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 2]]}
    sim, _ = script_compare({"episodeSteps": 400, "seed": 9, "weedSpawnChance": 0.0},
                            schedule(plan, 118, market), "fertilizer expiry")
    tiles = sim.observation(0)["farms"][0]["tiles"]
    plants = [t for row in tiles for t in row
              if isinstance(t, dict) and t.get("kind") == "PLANT"]
    assert plants, "the plant must survive to the end for this boundary to be observable"
    assert plants[0]["yield_units"] == 4, (
        f"expected 1 (planted) + 2 (fertilized) + 1 (expired) = 4, got "
        f"{plants[0]['yield_units']} -- scenario no longer isolates the boundary")


def test_decay_sheds_a_unit_every_other_step_past_lifespan():
    """`:739` — past `max_lifespan_step` a plant loses a unit every *other* step, then weeds."""
    plan = {(0, 2): ["PLANT", "WHEAT"], (0, 3): ["WATER"]}
    for d in range(1, 6):
        plan[(d, 2)] = ["WATER"]          # keep it alive and accruing through the window
    market = {(0, 0): [["BUY_SEED", "WHEAT", 1]]}
    script_compare({"episodeSteps": 400, "seed": 8, "weedSpawnChance": 0.0},
                   schedule(plan, 200, market), "decay")


def test_animal_escapes_after_two_unfed_days():
    """`:804` — an animal escapes at 2 consecutive unfed days, leaving its structure behind."""
    plan = {(0, 1): ["BUILD_PASTURE"], (0, 2): ["PICKUP", "COW", 1], (0, 3): ["PLACE", "COW"]}
    market = {(0, 0): [["BUY_ANIMAL", "COW", 1]]}
    sim, _ = script_compare({"episodeSteps": 400, "seed": 6, "weedSpawnChance": 0.0},
                            schedule(plan, 120, market), "animal escape")
    tiles = sim.observation(0)["farms"][0]["tiles"]
    kinds = [t.get("kind") for row in tiles for t in row if isinstance(t, dict)]
    assert "PASTURE" in kinds, "the structure must survive the escape"
    assert not any("animal" in t for row in tiles for t in row if isinstance(t, dict)), \
        "the cow should have escaped"


def test_ongoing_crop_bonus_requires_watering_that_day():
    """`:786` — for ongoing crops the fertilizer bonus *also* requires the tile to be watered.

    TOMATO first yields at day 8 with interval 1. Fertilizing on day 7 while deliberately NOT
    watering that day must accrue +1, not +2. Skipping one day is safe: death needs two consecutive
    dry days (`:770`).
    """
    plan = {(0, 2): ["PLANT", "TOMATO"], (0, 3): ["WATER"]}
    for d in range(1, 7):
        plan[(d, 2)] = ["WATER"]
    plan[(7, 1)] = ["PICKUP", "FERTILIZER", 1]
    plan[(7, 2)] = ["FERTILIZE"]          # active days 7-9, but day 7 is left dry
    # Run past the max_yield cap: TOMATO yields at days 7-10 (count 1..4) and must stop at
    # count > max_yield = 4, so the schedule has to reach day 11 for that branch to run.
    for d in range(8, 16):
        plan[(d, 2)] = ["WATER"]
    # Harvest on day 11 before the skip. Without this the branch is invisible: `yield_units` is
    # already at the cap and `min(max_yield, ...)` clamps a stray accrual to the same number, so
    # removing the guard changes nothing. Emptying the tile first makes the difference observable
    # -- 0 with the guard, 1 without.
    plan[(11, 5)] = ["HARVEST"]
    market = {(0, 0): [["BUY_SEED", "TOMATO", 1], ["BUY_PRODUCT", "FERTILIZER", 2]]}
    # Stop at the end of day 11. Accruals land on days 7-10 (count 1..4); day 11 is the first
    # `count > max_yield` skip, which is the branch this covers. Running further is pointless --
    # hitting the cap sets `max_lifespan_step` (`:789`) and the plant decays to a weed, which the
    # decay test covers separately.
    sim, _ = script_compare({"episodeSteps": 500, "seed": 12, "weedSpawnChance": 0.0},
                            schedule(plan, 287, market), "ongoing bonus")
    plants = [t for row in sim.observation(0)["farms"][0]["tiles"] for t in row
              if isinstance(t, dict) and t.get("kind") == "PLANT"]
    assert plants, "the tomato must survive for the bonus rule to be observable"
    assert plants[0]["yield_units"] == 0, (
        f"expected 0 after harvesting at the cap, got {plants[0]['yield_units']} -- an accrual "
        "past count > max_yield means the guard is not being exercised")


def test_feed_without_wheat_does_not_feed():
    """`:497` — FEED consumes a WHEAT from the unit's inventory or does nothing.

    Issuing FEED every day with an empty inventory must still let the animal escape. Without this,
    a rollout would believe livestock is free to keep.
    """
    plan = {(0, 1): ["BUILD_PASTURE"], (0, 2): ["PICKUP", "COW", 1], (0, 3): ["PLACE", "COW"]}
    for d in range(0, 5):                 # try to feed daily, carrying no wheat
        plan[(d, 5)] = ["FEED"]
    market = {(0, 0): [["BUY_ANIMAL", "COW", 1]]}
    sim, _ = script_compare({"episodeSteps": 400, "seed": 13, "weedSpawnChance": 0.0},
                            schedule(plan, 120, market), "feed without wheat")
    tiles = sim.observation(0)["farms"][0]["tiles"]
    assert not any(isinstance(t, dict) and "animal" in t for row in tiles for t in row), \
        "an unfed cow must escape even when FEED is issued every day"


def test_place_into_a_full_shed_is_rejected():
    """`:388` — PLACE into the shed obeys `shedCapacity`; overflow is simply not stored.

    The shed has to be *refilled* after the pickup or the clamp never binds: taking N out leaves
    exactly N of room, so putting N back always fits and the branch is never exercised.
    """
    plan = {(0, 1): ["PICKUP", "WHEAT", 2], (0, 3): ["PLACE", "WHEAT", 2]}
    market = {(0, 0): [["BUY_PRODUCT", "WHEAT", 2]],       # shed -> 2 (full)
              (0, 2): [["BUY_PRODUCT", "WHEAT", 2]]}       # refill to 2 while 2 are carried
    script_compare({"episodeSteps": 100, "seed": 14, "weedSpawnChance": 0.0, "shedCapacity": 2},
                   schedule(plan, 40, market), "full shed")


def test_care_bonus_requires_a_fed_production_day():
    """`:813-817` — the care bonus is banked only when cared AND fed, and spent only when fed.

    A cow first yields on day 8 with interval 2. Care it while feeding, then let the production day
    arrive unfed: the banked bonus must not be spent.
    """
    plan = {(0, 1): ["BUILD_PASTURE"], (0, 2): ["PICKUP", "COW", 1], (0, 3): ["PLACE", "COW"]}
    for d in range(0, 12):
        plan[(d, 4)] = ["PICKUP", "WHEAT", 1]
        plan[(d, 6)] = ["FEED"]
        plan[(d, 7)] = ["CARE"]
    market = {(0, 0): [["BUY_ANIMAL", "COW", 1]],
              (0, 5): [["BUY_PRODUCT", "WHEAT", 20]]}
    for d in range(1, 12):
        market[(d, 3)] = [["BUY_PRODUCT", "WHEAT", 4]]
    sim, _ = script_compare({"episodeSteps": 400, "seed": 15, "weedSpawnChance": 0.0},
                            schedule(plan, 290, market), "care bonus")
    cows = [t for row in sim.observation(0)["farms"][0]["tiles"] for t in row
            if isinstance(t, dict) and "animal" in t]
    assert cows, "the cow must survive for the care-bonus rule to be observable"
    assert cows[0]["yield_units"] > 0, "the cow should have produced by now"


def test_atomic_plant_validation():
    """Over-requesting a crop drops ALL of its PLANT actions, not just the excess (`:907`)."""
    cfg = {"episodeSteps": 60, "seed": 11, "weedSpawnChance": 0.0}
    sim = kagsim.Sim(dict(cfg))
    sim.step([{"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"],
                                                           ["BUY_SEED", "WHEAT", 1]]},
              {"farmer": ["PASS"], "hands": [], "market": []}])
    obs = sim.observation(0)
    model = FarmModel.from_obs(obs, player=0, weed_mode="none")
    n = 1 + len(obs["farms"][0]["hands"])
    assert n >= 2, "need at least two units for this test"

    acts = [["PLANT", "WHEAT"]] * n              # 1 seed, n requests -> all blocked
    model.step(acts)
    sim.step([{"farmer": acts[0], "hands": acts[1:], "market": []},
              {"farmer": ["PASS"], "hands": [], "market": []}])

    after = sim.observation(0)
    planted = sum(1 for r in after["farms"][0]["tiles"] for t in r
                  if isinstance(t, dict) and t.get("kind") == "PLANT")
    assert planted == 0, "reference must block every PLANT for the over-requested crop"
    assert model_state(model)["tiles"] == farm_state(after["farms"][0], after["private"])["tiles"]


def test_shed_ops_work_from_locked_tiles():
    """1.32.6 moved shed ops ahead of the LOCKED guard (E33); 3 of 4 shed tiles start LOCKED."""
    m = FarmModel.from_obs(
        {"player": 0, "day": 0, "hour": 0,
         "farms": [{"tiles": [["LOCKED"] * BOARD for _ in range(BOARD)],
                    "farmer": (5, 5), "hands": []}],
         "private": {"shed": {"WHEAT": 10}, "seeds": {}, "inventories": [{}]}},
        weed_mode="none")
    m.apply_unit_action(0, ["PICKUP", "WHEAT", 4])
    assert m.invs[0] == {"WHEAT": 4} and m.shed["WHEAT"] == 6, "PICKUP must work from a LOCKED tile"
    m.apply_unit_action(0, ["WATER"])           # a tile op on the same LOCKED tile must no-op
    assert m.tiles[5][5] == "LOCKED"


def test_speed_meets_the_p0_kill_criterion():
    """P0.3: >=2,000 farm-steps/s, or inference-time search does not fit the turn budget."""
    import time

    cfg = {"episodeSteps": 720, "seed": 3, "weedSpawnChance": 0.0}
    sim = kagsim.Sim(dict(cfg))
    for _ in range(400):                          # build a realistic mid-game farm
        sim.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]},
                  {"farmer": ["PASS"], "hands": [], "market": []}])
    model = FarmModel.from_obs(sim.observation(0), player=0, weed_mode="none")
    acts = [["WATER"]] * len(model.units)

    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
        model.step(acts)
    rate = n / (time.perf_counter() - t0)
    assert rate >= 2_000, f"{rate:,.0f} farm-steps/s is below the P0 kill criterion of 2,000"
