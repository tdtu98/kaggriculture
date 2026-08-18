"""S1: what a genome is worth, and the two exploiters that stop it being worth it against one agent.

The fitness function is the objective S2 will spend tens of thousands of games climbing, so the
tests that matter here are the ones that catch a *silently wrong* objective:

* it must be **deterministic** — S2 caches evaluations and keeps elites across generations, and a
  fitness that wobbles turns elitism into a random walk;
* it must **rank a dead plan below a live one**, which is the only external check on the formula;
* it must **carry the counters back**, because S2's job is to reject plans that win by breaking
  them (E44: a change that never fired and a strategy that does not work read identically);
* and it must **never spend an acceptance seed**, which is the one failure the money would never
  reveal.

The exploiters are tested *in play* rather than as plans. `flooder`'s first draft validated
perfectly, decoded to 75 strawberry tiles, and lost 8/8 to the plain compiler because it could not
afford to stand its herd up — a plan-shaped assertion would have passed on every one of those
games.
"""

from __future__ import annotations

import pytest

import kagsim
from agent.plan import COHORT_SLOTS, GENE_INDEX, Plan, decode, encode
from harness.registry import EXPLOITER_NOTES, SEARCH_POOL, VEC_PREFIX, get, vec_name
from search.exploiters import flooder_plan, tomato_rusher_plan
from search.fitness import (ACCEPTANCE_BLOCKS, MARGIN_WEIGHT, WIN_WEIGHT, evaluate,
                            normalised_margin, search_seeds)

CONFIG = {"episodeSteps": 720}


def _season(agent, seed: int, steps: int = 719):
    """One full season against `starter`, returning (money, tiles-by-day snapshots)."""
    other = get("starter").build()
    sim = kagsim.Sim({**CONFIG, "seed": seed})
    sim.collect_stats = True
    by_day: dict[int, list] = {}
    for _ in range(steps):
        obs = sim.observation(0)
        by_day.setdefault(int(obs.get("day", 0) or 0), obs["farms"][0]["tiles"])
        sim.step([agent(obs), other(sim.observation(1))])
    return sim.money(0), by_day


def _crop_tiles(tiles, crop: str) -> int:
    return sum(1 for row in tiles for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop)


# --------------------------------------------------------------- the exploiters exist and play

@pytest.mark.parametrize("name", sorted(EXPLOITER_NOTES))
def test_an_exploiter_plays_a_whole_season_and_earns(name):
    """The submission-shaped smoke test: 719 turns, no exception, money above the starting bank.

    $3,000 is not an arbitrary bar — it is exactly what an agent that raised on turn 1 scores, and
    it is what a miswired pool member would quietly contribute to every search evaluation (E21).
    """
    money, _ = _season(get(name).build(), seed=60101)
    assert money > 3_000, f"{name} scored the starting bank — it is not playing"


@pytest.mark.parametrize("name", sorted(EXPLOITER_NOTES))
def test_an_exploiters_counters_are_live(name):
    """A pool member whose effect counters read zero is an unfinished implementation, not a result."""
    agent = get(name).build()
    _season(agent, seed=60102, steps=24 * 12)
    effects = agent.ctx.effects
    assert effects.get("compiled_days", 0) >= 10, f"{name} stopped compiling: {effects}"
    assert effects.get("projection_calls", 0) > 0, f"{name} never projected: {effects}"


def test_the_flooder_actually_over_plants_strawberry_and_cows():
    """Against the baseline it is derived from, not against a remembered number.

    1.86x rather than 2.0x is measured, not aspirational: 20 pastures have to stand somewhere, and
    what is left of the board after them is 67 tiles. The bar is set below that so the test fails on
    a *regression* (a doubling that `decode` trimmed back) and not on the documented shortfall.
    """
    base, flood = Plan.boatlee_like(), flooder_plan()
    straw = lambda p: sum(c.n_tiles for c in p.cohorts if c.crop == "STRAWBERRY")  # noqa: E731
    cows = lambda p: sum(1 for s, _ in p.herd if s == "COW")                        # noqa: E731

    assert flood.is_valid(), flood.validate()
    assert straw(flood) >= 1.7 * straw(base), f"{straw(flood)} vs {straw(base)} strawberry tiles"
    assert cows(flood) >= 1.5 * cows(base), f"{cows(flood)} vs {cows(base)} cows"
    assert not flood.notes, f"decode trimmed the flooder: {flood.notes}"


def test_the_flooder_really_gets_that_strawberry_into_the_ground():
    """In play, not in the plan. The rejected draft had 75 tiles on paper and grew almost none."""
    _, by_day = _season(get("flooder").build(), seed=60103, steps=24 * 20)
    peak = max(_crop_tiles(t, "STRAWBERRY") for t in by_day.values())
    assert peak >= 40, f"only {peak} strawberry tiles ever planted — the flooder is not flooding"


def test_the_tomato_rusher_really_gets_its_tomato_in_the_ground():
    """The plan says day 6; the board says day 13, and the board is the claim being tested.

    NE is bought on day 6 in the plan and the compiler cannot afford it until ~day 12 — the wallet
    sits at $4-$1,000 through days 4-9 (E70), so **every** NE/SW `plant_day` in this genome is
    advisory, `Plan.boatlee_like()`'s included. That is a property of the compiler, not of this
    exploiter, and the exploiter's job (contest tomato) is done either way. Asserting day 6 here
    would be asserting something no plan on this compiler can currently deliver.
    """
    plan = tomato_rusher_plan()
    tomato = [c for c in plan.cohorts if c.crop == "TOMATO"]
    assert len(tomato) == 1 and tomato[0].plant_day == 6 and tomato[0].n_tiles == 15

    _, by_day = _season(get("tomato_rusher").build(), seed=60104, steps=24 * 16)
    assert max(_crop_tiles(by_day[d], "TOMATO") for d in by_day if d <= 15) >= 10, (
        "the rush never happened: fewer than 10 tomato tiles by day 15")


# --------------------------------------------------------------- the genome reaches a worker

def test_a_genome_is_addressable_as_a_pool_name():
    """S2 hands the runner a vector, and the runner only speaks names (`registry.VEC_PREFIX`)."""
    vec = encode(Plan.boatlee_like())
    name = vec_name(vec)

    assert name.startswith(VEC_PREFIX)
    assert get(name).fingerprint == get(name).fingerprint
    assert get(name).fingerprint != get(vec_name(_no_pastures())).fingerprint, (
        "two different genomes must not share a fingerprint")
    assert decode([float(x) for x in name[len(VEC_PREFIX):].split(",")]) == decode(vec)


def _no_pastures():
    v = encode(Plan.boatlee_like())
    v[GENE_INDEX["n_pastures"]] = 0.0
    return v


def _dead_plan():
    """No pastures and no cohorts: a farm that grows nothing and keeps nothing."""
    v = _no_pastures()
    for i in range(COHORT_SLOTS):
        v[GENE_INDEX[f"c{i}_tiles"]] = 0.0
    return v


# --------------------------------------------------------------- the objective itself

def test_normalised_margin_is_bounded_and_centred_on_a_dead_heat():
    assert normalised_margin(0.0) == pytest.approx(0.5)
    assert 0.0 <= normalised_margin(-10 ** 9) < 1e-6
    assert 1 - 1e-6 < normalised_margin(10 ** 9) <= 1.0
    assert normalised_margin(-1) < normalised_margin(0) < normalised_margin(1)
    assert WIN_WEIGHT + MARGIN_WEIGHT == pytest.approx(1.0)


def test_search_seeds_rotate_and_never_touch_an_acceptance_block():
    a, b = search_seeds(0), search_seeds(1)

    assert len(a) == len(b) == 6
    assert not set(a) & set(b), "two generations shared seeds — the rotation is not rotating"
    for gen in range(0, 400, 37):
        for seed in search_seeds(gen):
            assert not any(lo <= seed < hi for lo, hi in ACCEPTANCE_BLOCKS), seed


def test_evaluate_refuses_to_spend_an_acceptance_seed():
    """The one mistake the result would look completely healthy after."""
    with pytest.raises(ValueError, match="acceptance block"):
        evaluate(encode(Plan.boatlee_like()), [54000, 54001], [("starter", 1.0)])


def test_the_same_genome_on_the_same_seeds_gives_the_identical_result():
    """S2 keeps elites across generations and caches evaluations; both are wrong if this drifts."""
    vec, seeds, pool = encode(Plan.boatlee_like()), [60110], [("starter", 1.0)]
    a = evaluate(vec, seeds, pool, workers=2)
    b = evaluate(vec, seeds, pool, workers=2)

    for key in ("fitness", "win_rate", "mean_margin", "counters", "per_opponent", "fingerprint"):
        assert a[key] == b[key], f"{key} is not reproducible: {a[key]} vs {b[key]}"


def test_a_dead_plan_scores_near_zero_and_a_live_one_does_not():
    """S1's acceptance check. Both arms play the same seeds and the same opponents."""
    seeds, pool = [60111, 60112], [("starter", 1.0), ("tomato_rusher", 1.0)]
    dead = evaluate(_dead_plan(), seeds, pool, workers=4)
    live = evaluate(encode(Plan.boatlee_like()), seeds, pool, workers=4)

    assert dead["fitness"] < 0.05, f"a farm that grows nothing scored {dead['fitness']:.3f}"
    assert dead["win_rate"] == 0.0
    assert live["fitness"] > 0.6 > dead["fitness"], (live["fitness"], dead["fitness"])


def test_the_counters_come_back_with_the_money():
    """S2 rejects a plan that "wins" by killing its plants, which needs the counters, not a scalar."""
    r = evaluate(encode(Plan.boatlee_like()), [60113], [("starter", 1.0)], workers=1,
                 with_solo=True)
    c = r["counters"]

    assert r["solo"] is not None and r["solo"] > 100_000, f"solo ${r['solo']:,.0f}"
    for key in ("steps_per_useful", "plants_lost_thirst", "blocked_ops", "strawberry_per_plant",
                "milk_per_cow_day", "shed_overflow_discarded"):
        assert key in c, f"{key} did not survive the fitness function"
    assert c["effect_count"]["compiled_days"] > 0, "the agent's own effect counters are missing"


def test_the_search_pool_is_the_one_s1_specifies():
    """Boatlee x2 and the two exploiters — the E10 lesson, written down where the search reads it."""
    weights = dict(SEARCH_POOL)

    assert weights["boatlee"] == 2.0
    assert set(weights) == {"boatlee", "flooder", "tomato_rusher", "executor_v7"}


def test_a_weighted_pool_is_not_the_same_as_an_unweighted_one():
    """The x2 has to reach the arithmetic; a weight that is stored and ignored is no weight."""
    seeds = [60114]
    even = evaluate(_dead_plan(), seeds, [("starter", 1.0), ("tomato_rusher", 1.0)], workers=2)
    tilted = evaluate(_dead_plan(), seeds, [("starter", 9.0), ("tomato_rusher", 1.0)], workers=2)

    # The dead plan loses to both, but by very different margins, so re-weighting must move the
    # mean margin. If it does not, the weights are decorative.
    assert even["mean_margin"] != tilted["mean_margin"]
