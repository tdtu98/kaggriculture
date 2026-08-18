"""S2: the parts of the search that fail *silently*.

A genome search is unusually good at looking healthy while producing nothing, and this project has
paid for four different versions of that already. So these tests deliberately do not check that the
search "runs" — the smoke run does that with real games. They check the four things whose failure
would be invisible in a log full of plausible fitness numbers:

* a **mutation that decodes to a different plan than the genome says**, which would have the search
  reporting scores for candidates it silently rewrote (E44's shape);
* a **cache that conflates two seed blocks**, which would hand elitism a stale number — seeds
  rotate every generation, so the same vector is a different measurement each time;
* an **acceptance ledger that reuses a spent sub-block**, which turns the holdout into part of the
  training set and makes the gate agree with whatever the search found;
* a **counter guard that does not outrank the money**, which is how a plan that wins by killing
  plants gets promoted.

No test here plays a game against a real block: `dry_run` and a temp ledger everywhere, because
54000:54080 is 80 seeds in total and a test suite that spent one per run would exhaust it in a day.
"""

from __future__ import annotations

import json
import random

import pytest

from agent.plan import (COHORT_SLOTS, FLOOR_PRODUCTS, GENES, GENE_INDEX, HALF, NEVER, Plan,
                        QUADRANTS, decode, encode, floor_gene, snap, random_vector)
from search import accept, ga
from search.fitness import ACCEPTANCE_BLOCKS, SEARCH_BLOCK, search_seeds


def _base():
    return ga.canonical(encode(Plan.boatlee_like()))


# --------------------------------------------------------------------- mutation validity

def test_500_mutations_all_decode_valid_and_are_canonical():
    """The load-bearing invariant: everything the GA puts in the population is a plan the game
    could play, and is its own fixed point under decode/encode.

    Not a fixed point means two vectors name one plan; the cache keys on the vector, so the search
    would pay twice for one measurement and record it as two data points.
    """
    rng = random.Random(0)
    vec = _base()
    for i in range(500):
        child, ops, _repaired = ga.mutate(vec, rng)
        plan = decode(child)
        assert plan.validate() == [], f"mutation {i} ({ops}) produced an invalid plan"
        assert snap(encode(plan)) == snap(child), f"mutation {i} ({ops}) is not canonical"
        vec = child if rng.random() < 0.5 else vec        # random-walk as well as star-shaped


def test_mutations_of_random_genomes_stay_valid():
    """Mutating a *random* point is the harsher case: crossover produces genomes far from the seed
    plan and every operator has to survive them (empty cohort lists, land never bought, 0 pastures).
    """
    rng = random.Random(1)
    for i in range(200):
        child, ops, _r = ga.mutate(ga.canonical(random_vector(rng)), rng)
        assert decode(child).validate() == [], f"{i}: {ops}"


def test_mutation_almost_never_returns_its_parent_unchanged():
    """A child identical to its parent is a wasted population slot. `MUTATE_RETRIES` exists for
    this; without it 18% of single mutations on `boatlee_like` are absorbed by canonicalisation."""
    rng = random.Random(2)
    base = _base()
    same = sum(ga.mutate(base, rng)[0] == base for _ in range(200))
    assert same <= 4, f"{same}/200 mutants were copies of their parent"


def test_ga_never_writes_a_cma_gene():
    """The two optimisers partition the space. If a GA operator nudged a CMA coordinate, the inner
    loop would spend its whole budget re-measuring the outer loop's last mutation."""
    rng = random.Random(3)
    base = _base()
    idx = [GENE_INDEX[n] for n in ga.CMA_GENES]
    for _ in range(300):
        child, _ops, _r = ga.mutate(base, rng)
        assert [child[i] for i in idx] == [base[i] for i in idx]


def test_gene_sets_partition_the_genome():
    assert set(ga.GA_GENES).isdisjoint(ga.CMA_GENES)
    assert set(ga.GA_GENES) | set(ga.CMA_GENES) == {g.name for g in GENES}
    assert len(ga.GA_GENES) + len(ga.CMA_GENES) == len(GENES)


@pytest.mark.parametrize("op", [o for o, _ in ga.OPERATORS])
def test_every_operator_is_reachable_and_safe(op):
    """Each operator applied alone, on both a hand plan and a random one. An operator that only
    ever returns None (its precondition never holds) is dead weight in the weighted draw, and one
    that raises would take the whole generation down."""
    rng = random.Random(4)
    fired = 0
    for base in (_base(), ga.canonical(random_vector(random.Random(5)))):
        for _ in range(60):
            v = list(base)
            tag = op(v, rng)
            fired += tag is not None
            assert decode(snap(v)).validate() == []
    assert fired > 0, f"{op.__name__} never fired"


def test_crossover_recombines_blocks_and_keeps_the_consts_of_the_first_parent():
    rng = random.Random(6)
    a = _base()
    b = ga.canonical(random_vector(random.Random(7)))
    seen_from_b = 0
    for _ in range(100):
        c = ga.crossover(a, b, rng)
        assert decode(c).validate() == []
        for name in ga.CMA_GENES:
            assert c[GENE_INDEX[name]] == a[GENE_INDEX[name]]
        seen_from_b += any(c[GENE_INDEX[n]] == b[GENE_INDEX[n]] != a[GENE_INDEX[n]]
                           for block in ga.BLOCKS for n in block)
    assert seen_from_b > 50, "crossover is not actually mixing the two parents"


def test_diversity_is_zero_for_a_collapsed_population_and_positive_otherwise():
    base = _base()
    assert ga.diversity([base] * 48) == pytest.approx(0.0, abs=1e-12)
    assert ga.unique_fraction([base] * 48) == pytest.approx(1 / 48)
    rng = random.Random(8)
    pop = [ga.mutate(base, rng)[0] for _ in range(48)]
    assert ga.diversity(pop) > 0.01
    assert ga.unique_fraction(pop) > 0.7


# --------------------------------------------------------------------- cache

def test_cache_keys_on_the_seed_block_not_just_the_genome():
    """Seeds rotate every generation (`search_seeds`), so a vector scored on gen 4's seeds says
    nothing about gen 5's. A cache that ignored the seeds would freeze the elite set."""
    c = ga.Cache()
    v = _base()
    c.put(v, [1, 2, 3], {"fitness": 0.5})
    assert c.get(v, [1, 2, 3])["fitness"] == 0.5
    assert c.get(v, [1, 2, 4]) is None
    assert c.get(v, [3, 2, 1]) is None            # order is part of the block, not a set


def test_cache_normalises_the_genome_but_not_across_plans():
    """Two vectors that `snap` to the same point are one measurement; two that do not are two."""
    c = ga.Cache()
    v = _base()
    c.put(v, [1], {"fitness": 0.5})
    jittered = list(v)
    jittered[GENE_INDEX["n_cows"]] += 0.2          # integral gene: rounds back
    assert c.get(jittered, [1]) is not None
    other = list(v)
    other[GENE_INDEX["n_cows"]] += 1
    assert c.get(other, [1]) is None


def test_cache_hit_rate_counts_both_ways():
    c = ga.Cache()
    v = _base()
    assert c.get(v, [1]) is None
    c.put(v, [1], {"fitness": 0.1})
    c.get(v, [1])
    c.get(v, [2])
    assert (c.hits, c.misses) == (1, 2)
    assert c.hit_rate == pytest.approx(1 / 3)


def test_cache_reloads_from_the_log_and_drops_the_interrupted_tail(tmp_path):
    """`--resume` replays the log into the cache. Rows above the resumed generation are the tail of
    a generation that was killed halfway through; keeping them would let a partially-scored
    generation leak into the one being replayed."""
    path = tmp_path / "log.jsonl"
    v = _base()
    result = {"fitness": 0.4, "win_rate": 0.5, "mean_margin": 100.0, "counters": {},
              "per_opponent": {}, "seeds": [60000], "games": 48, "seconds": 6.0}
    ga.log_row(str(path), 0, "ga", v, result, [], False)
    ga.log_row(str(path), 1, "ga", v, {**result, "seeds": [60006]}, [], False)

    c = ga.Cache()
    assert c.load_jsonl(str(path)) == 2
    assert c.get(v, [60006])["fitness"] == 0.4

    c2 = ga.Cache()
    assert c2.load_jsonl(str(path), max_gen=0) == 1
    assert c2.get(v, [60006]) is None
    assert c2.get(v, [60000])["fitness"] == 0.4


def test_a_log_row_carries_everything_the_run_can_be_reconstructed_from(tmp_path):
    path = tmp_path / "log.jsonl"
    v = _base()
    result = {"fitness": 0.62, "win_rate": 0.7, "mean_margin": 4200.0, "solo": 120000.0,
              "counters": {"steps_per_useful": 0.8, "plants_lost_thirst": 1.2,
                           "effect_count": {"fallbacks": 0}},
              "per_opponent": {"boatlee": {"win_rate": 0.5, "mean_margin": -100.0}},
              "seeds": [60000, 60001], "games": 48, "seconds": 6.5}
    result["fingerprint"] = "abc123"
    row = ga.log_row(str(path), 3, "cma", v, result, [], False)
    parsed = json.loads(path.read_text().strip())
    assert parsed == row
    for key in ("gen", "kind", "vec", "fitness", "score", "guard", "win_rate", "mean_margin",
                "solo", "counters", "seeds", "games", "seconds", "cached", "per_opponent",
                "fingerprint"):
        assert key in parsed
    # The fingerprint covers the compiler source, so a tree that changes mid-run is visible in
    # the log rather than showing up as an unexplained step in the fitness curve.
    assert parsed["fingerprint"] == "abc123"
    assert len(parsed["vec"]) == len(GENES)
    assert parsed["counters"]["steps_per_useful"] == 0.8
    assert parsed["fallbacks"] == 0


# --------------------------------------------------------------------- counter guard

def test_the_guard_fires_on_each_c5_bar_independently():
    ok = {"plants_lost_thirst": 2.5, "steps_per_useful": 0.79, "effect_count": {"fallbacks": 0}}
    assert ga.guard_violations(ok) == []
    assert ga.guard_violations({**ok, "plants_lost_thirst": 10.5})
    assert ga.guard_violations({**ok, "steps_per_useful": 1.2})
    assert ga.guard_violations({**ok, "effect_count": {"fallbacks": 1}})
    # exactly at the bar is legal; the bars are "> x" rejections
    assert ga.guard_violations({**ok, "plants_lost_thirst": 10.0, "steps_per_useful": 1.15}) == []


def test_a_missing_counter_block_is_not_a_violation():
    """An evaluation with no counters at all is a harness problem. Failing those candidates would
    look exactly like a search that cannot find anything — the most expensive kind of silence."""
    assert ga.guard_violations({}) == []


def test_a_guarded_candidate_scores_below_every_legal_one():
    legal = ga.score_of(0.0, [])
    illegal = ga.score_of(1.0, ["plants_lost_thirst=40>10"])
    assert illegal < legal
    # ...but violators keep their relative order, so an all-illegal population is still selectable
    assert ga.score_of(0.9, ["x"]) > ga.score_of(0.4, ["x"])


def test_counter_value_reads_effect_counters_too():
    assert ga.counter_value({"effect_count": {"fallbacks": 3}}, "fallbacks") == 3.0
    assert ga.counter_value({"steps_per_useful": 0.9}, "steps_per_useful") == 0.9
    assert ga.counter_value({}, "fallbacks") == 0.0


# --------------------------------------------------------------------- acceptance ledger

def test_the_ledger_never_hands_out_a_spent_block(tmp_path):
    path = str(tmp_path / "ledger.json")
    first = accept.claim_block("one", path=path)
    second = accept.claim_block("two", path=path)
    assert first != second
    assert first == (54000, 54040) and second == (54040, 54080)
    assert accept.load_ledger(path) == [[54000, 54040], [54040, 54080]]


def test_the_ledger_raises_when_the_reserve_is_exhausted(tmp_path):
    path = str(tmp_path / "ledger.json")
    while accept.remaining(path):
        accept.claim_block(path=path)
    with pytest.raises(RuntimeError, match="exhausted"):
        accept.claim_block(path=path)


def test_a_dry_run_claims_nothing(tmp_path):
    path = str(tmp_path / "ledger.json")
    a = accept.claim_block(path=path, dry_run=True)
    b = accept.claim_block(path=path, dry_run=True)
    assert a == b, "a dry run must not advance the ledger"
    assert accept.load_ledger(path) == []


def test_a_claim_survives_a_crash_because_it_is_written_before_the_games(tmp_path):
    """The ledger is the reason a re-run after a crash gets fresh seeds. It is written by
    `claim_block`, before a single game is played, so an interrupted acceptance run burns its
    block — which is the safe direction."""
    path = str(tmp_path / "ledger.json")
    accept.claim_block("crashed run", path=path)
    assert accept.remaining(path) == [(54040, 54080)]


def test_every_acceptance_block_is_reserved_and_unsearchable(tmp_path):
    path = str(tmp_path / "ledger.json")
    blocks = []
    while accept.remaining(path):
        blocks.append(accept.claim_block(path=path))
    lo, hi = accept.ACCEPT_BLOCK
    assert [b[0] for b in blocks] == list(range(lo, hi, accept.SUB_BLOCK))
    for b in blocks:
        assert any(a_lo <= b[0] and b[1] <= a_hi for a_lo, a_hi in ACCEPTANCE_BLOCKS)
        assert not (SEARCH_BLOCK[0] <= b[0] < SEARCH_BLOCK[1])
    for gen in range(0, 200):
        assert not any(lo <= s < hi for s in search_seeds(gen))


def test_wilson_brackets_the_point_estimate_and_needs_a_real_sample():
    lo, hi = accept.wilson(48, 80)
    assert lo < 0.6 < hi
    assert lo < 0.5, "48/80 = 60% must NOT clear 50%; that is the D19 lesson"
    lo, hi = accept.wilson(56, 80)
    assert lo > 0.5, "70% of 80 games should clear"
    assert accept.wilson(4, 8)[0] < 0.5, "an 8-game sample can never clear"


def test_accept_rejects_on_the_counter_guard_regardless_of_the_win_rate(monkeypatch, tmp_path):
    """The guard has to outrank the money in the *gate* as well as in the objective, or the search
    rejects bug-exploits and the promotion accepts them."""
    monkeypatch.setattr(accept, "head_to_head", lambda *a, **k: {
        "games": 80, "successes": 80.0, "win_rate": 1.0, "mean_margin": 50_000.0,
        "mean_money": 200_000.0,
        "counters": {"plants_lost_thirst": 44.0, "steps_per_useful": 0.8,
                     "effect_count": {"fallbacks": 0}}})
    res = accept.accept(_base(), _mutant(), ledger=str(tmp_path / "l.json"), dry_run=True)
    assert res["accept"] is False
    assert res["guard"] and "counter guard" in res["reason"]


def test_accept_needs_the_ci_clear_of_fifty(monkeypatch, tmp_path):
    clean = {"plants_lost_thirst": 1.0, "steps_per_useful": 0.8, "effect_count": {"fallbacks": 0}}

    def h2h(successes):
        return lambda *a, **k: {"games": 80, "successes": float(successes),
                                "win_rate": successes / 80, "mean_margin": 1.0,
                                "mean_money": 1.0, "counters": clean}

    ledger = str(tmp_path / "l.json")
    monkeypatch.setattr(accept, "head_to_head", h2h(48))
    assert accept.accept(_base(), _mutant(), ledger=ledger, dry_run=True)["accept"] is False
    monkeypatch.setattr(accept, "head_to_head", h2h(56))
    res = accept.accept(_base(), _mutant(), ledger=ledger, dry_run=True)
    assert res["accept"] is True and res["ci"][0] > 0.5
    assert res["block_claimed"] is False


def test_accept_refuses_to_compare_a_genome_with_itself(tmp_path):
    with pytest.raises(ValueError, match="same genome"):
        accept.accept(_base(), _base(), block=(54000, 54040), ledger=str(tmp_path / "l.json"))


def _mutant():
    return ga.mutate(_base(), random.Random(11))[0]


# --------------------------------------------------------------------- state / resume

def test_state_round_trips_the_population_and_the_rng(tmp_path):
    """Resume reproduces a run rather than continuing one, and that hangs entirely on the RNG
    state travelling with the population."""
    path = str(tmp_path / "state.json")
    rng = random.Random(12)
    pop = [ga.mutate(_base(), rng)[0] for _ in range(48)]
    best = {"gen": 1, "score": 0.5, "fitness": 0.5, "win_rate": 0.5, "vec": _base()}
    ga.save_state(path, 7, pop, rng, best)
    gen, pop2, rng2, best2 = ga.load_state(path)
    assert gen == 7 and pop2 == [snap(p) for p in pop] and best2 == best
    assert [rng2.random() for _ in range(20)] == [rng.random() for _ in range(20)]


def test_state_is_written_atomically(tmp_path):
    path = str(tmp_path / "state.json")
    rng = random.Random(13)
    ga.save_state(path, 1, [_base()], rng, None)
    ga.save_state(path, 2, [_base()], rng, None)
    assert ga.load_state(path)[0] == 2
    assert not (tmp_path / "state.json.tmp").exists()


# --------------------------------------------------------------------- the driver, without games

def test_the_seed_population_is_boatlee_plus_47_distinct_mutants():
    from search.run import Search, build_parser

    s = Search(build_parser().parse_args([]))
    s.seed_population()
    assert len(s.population) == 48
    assert s.population[0] == ga.canonical(encode(Plan.boatlee_like()))
    assert ga.unique_fraction(s.population) > 0.75
    assert all(decode(v).validate() == [] for v in s.population)
    assert 0.0 <= s.repair_rate <= 1.0


def test_breeding_keeps_the_elites_and_fills_the_rest():
    from search.run import Search, build_parser

    s = Search(build_parser().parse_args(["--pop", "12", "--elites", "3"]))
    s.seed_population()
    ranked = [(v, {"fitness": 0.5}, 1.0 - i) for i, v in enumerate(s.population)]
    pop = s.breed(ranked)
    assert len(pop) == 12
    assert pop[:3] == [v for v, _r, _s in ranked[:3]], "elitism must copy the top genomes verbatim"
    assert all(decode(v).validate() == [] for v in pop)


def _stub_evaluate(monkeypatch):
    """A deterministic fitness that never plays a game.

    The resume test has to isolate the *search* from the *tree*. Scored through real games it
    would also be testing that `agent/` has not changed between the two runs — and when it has
    (a concurrent edit, a new env pin) the test fails for a reason that has nothing to do with
    resumability. A pure function of (genome, seeds) fails only when resume is actually broken.
    """
    import search.run as run_mod

    def fake(vec, seeds, **kw):
        h = hash((tuple(snap(vec)), tuple(seeds))) % 10_000
        return {"fitness": h / 10_000, "win_rate": h / 20_000, "mean_margin": float(h),
                "solo": 100_000.0, "counters": {"steps_per_useful": 0.8,
                                                "plants_lost_thirst": 1.0,
                                                "effect_count": {"fallbacks": 0}},
                "per_opponent": {}, "seeds": list(seeds), "games": 48, "seconds": 0.0,
                "fingerprint": "stub"}

    monkeypatch.setattr(run_mod, "evaluate", fake)


def _rows(path, gen):
    return [json.loads(x) for x in open(path) if json.loads(x)["gen"] == gen]


def test_a_resumed_run_reproduces_the_generation_an_uninterrupted_one_would_have_played(
        monkeypatch, tmp_path):
    """Kill after generation 2, resume, and the generations that follow must be identical — same
    genomes, same order, same scores.

    This is the property that makes a multi-hour search runnable in bounded chunks at all. It
    holds because the state file carries the RNG *and* the population, and because the inner CMA
    is seeded from (run seed, generation, rank) rather than from a stream that a restart would
    reset.

    **Two** generations are replayed, not one, and that is load-bearing. A generation's log rows
    are written before `breed` runs, so the *first* resumed generation does not touch `self.rng`
    at all: a resume that dropped the RNG state entirely would reproduce it perfectly and diverge
    only on the generation after. Checking one generation passed against exactly that bug.
    """
    from search.run import Search, build_parser

    _stub_evaluate(monkeypatch)
    common = ["--pop", "12", "--elites", "2", "--seeds", "2", "--cma-top", "2",
              "--cma-budget", "4", "--cma-every", "1", "--workers", "1"]

    a_log, a_state = str(tmp_path / "a.jsonl"), str(tmp_path / "a.json")
    args = build_parser().parse_args(common + ["--log", a_log, "--state", a_state])
    sa = Search(args)
    sa.seed_population()
    sa.run(4)

    b_log, b_state = str(tmp_path / "b.jsonl"), str(tmp_path / "b.json")
    args_b = build_parser().parse_args(common + ["--log", b_log, "--state", b_state])
    sb = Search(args_b)
    sb.seed_population()
    sb.run(2)

    args_r = build_parser().parse_args(common + ["--log", b_log, "--state", b_state, "--resume"])
    sr = Search(args_r)
    sr.resume()
    assert sr.gen == 2
    sr.run(2)

    for gen in (2, 3):
        ra, rb = _rows(a_log, gen), _rows(b_log, gen)
        assert ra and len(ra) == len(rb), f"generation {gen}: {len(ra)} rows vs {len(rb)}"
        for x, y in zip(ra, rb):
            for key in ("kind", "vec", "fitness", "score", "seeds", "guard"):
                assert x[key] == y[key], f"resume diverged on {key} in generation {gen}"
    assert sr.best == sa.best


def test_resume_rebuilds_the_cache_from_the_log(monkeypatch, tmp_path):
    """The other half of resumability: a resumed run must not re-play generations it already has.
    Without the log replay the elites of the resumed generation would be re-evaluated from
    scratch — correct, but it throws away exactly the work a chunked run is trying to keep."""
    from search.run import Search, build_parser

    _stub_evaluate(monkeypatch)
    log, state = str(tmp_path / "c.jsonl"), str(tmp_path / "c.json")
    common = ["--pop", "8", "--elites", "2", "--seeds", "2", "--cma-every", "0",
              "--log", log, "--state", state]
    s = Search(build_parser().parse_args(common))
    s.seed_population()
    s.run(2)

    r = Search(build_parser().parse_args(common + ["--resume"]))
    r.resume()
    assert len(r.cache) > 0
    # the replay must stop below the resumed generation, or a half-scored generation leaks in
    assert all(int(json.loads(x)["gen"]) <= 1 for x in open(log))


@pytest.mark.parametrize("budget,rounds", [(4, 1), (8, 2), (12, 3)])
def test_the_inner_loop_runs_one_cma_iteration_per_popsize_of_budget(monkeypatch, tmp_path,
                                                                    budget, rounds):
    """`--cma-budget` buys ask/tell *rounds*, and below two of them CMA-ES cannot adapt at all:
    it asks once, tells once, and stops without ever sampling the distribution it just updated.
    That configuration looks like a working inner loop in the log and is really just random
    perturbation, which is why the default is 8 and anything smaller warns.
    """
    from search.run import CMA_POPSIZE, Search, build_parser

    # `cma` is a `make setup` dependency, but `cma_polish` degrades to a no-op without it, so an
    # environment that lacks it should skip this rather than fail: the search still runs, it just
    # runs without an inner loop.
    pytest.importorskip("cma")
    _stub_evaluate(monkeypatch)
    args = build_parser().parse_args(
        ["--pop", "6", "--elites", "1", "--seeds", "2", "--cma-top", "1",
         "--cma-budget", str(budget), "--cma-every", "1",
         "--log", str(tmp_path / "l.jsonl"), "--state", str(tmp_path / "s.json")])
    s = Search(args)
    s.seed_population()
    seeds = [60000, 60001]
    ranked = s.ranked([(v, {"fitness": 0.5}, 0.5) for v in s.population])
    s.cma_polish(ranked, seeds)

    cma_rows = [json.loads(x) for x in open(tmp_path / "l.jsonl")
                if json.loads(x)["kind"] == "cma"]
    assert len(cma_rows) == rounds * CMA_POPSIZE


def test_ranking_puts_a_guarded_genome_last_however_good_its_fitness():
    from search.run import Search

    rows = [("a", {}, ga.score_of(0.9, ["thirst"])), ("b", {}, ga.score_of(0.2, []))]
    assert [v for v, _r, _s in Search.ranked(rows)] == ["b", "a"]


# --------------------------------------------------------------------- S3: the widened operators

def test_the_blocks_cover_every_ga_gene_exactly_once():
    """Crossover copies *blocks*; a GA gene in no block can only ever come from parent `a`, so the
    four slots S3 added would be inherited from one side and never recombined."""
    covered = [n for block in ga.BLOCKS for n in block]
    assert len(covered) == len(set(covered)), "a gene in two blocks would be copied twice"
    assert set(covered) == set(ga.GA_GENES)
    slot_blocks = [b for b in ga.BLOCKS if b[0].startswith("c") and b[0].endswith("_crop")]
    assert len(slot_blocks) == COHORT_SLOTS, "one block per cohort slot"


def test_every_floor_is_a_cma_gene_and_none_is_a_ga_gene():
    """S3's floors are continuous and correlated, so CMA owns them — and only CMA."""
    for product in FLOOR_PRODUCTS:
        assert floor_gene(product) in ga.CMA_GENES
        assert floor_gene(product) not in ga.GA_GENES


def test_op_cohort_rows_lands_on_whole_rows():
    """The row claim S3 asked for, expressed in the cohort genes: `decode` fills a quadrant
    row-major, so `N * HALF` tiles *is* "N rows of crop C in quadrant Q"."""
    rng = random.Random(11)
    base = _base()
    seen = set()
    for _ in range(200):
        v = list(base)
        tag = ga.op_cohort_rows(v, rng)
        assert tag is not None
        i = int(tag[len("cohort"):].split("_")[0])
        n = int(v[GENE_INDEX[f"c{i}_tiles"]])
        assert n in ga.ROW_CLAIMS, n
        assert n % HALF == 0
        seen.add(n)
    assert len(seen) >= 4, "the operator should reach most of the row band"


def test_op_claim_quadrant_buys_the_land_and_stands_crops_on_it():
    """The compound move that makes the extra slots reachable at all: `boatlee_like` has no free
    tiles on the land it buys, so a bare `op_cohort_add` is dropped by `decode` for want of a tile.
    Land and the cohorts that use it are one decision."""
    rng = random.Random(12)
    base = _base()
    assert int(ga.gene(base, "land_SE")) >= NEVER, "the fixture must have SE unbought"
    fired = 0
    for _ in range(60):
        v = list(base)
        tag = ga.op_claim_quadrant(v, rng)
        if tag is None:
            continue
        fired += 1
        assert tag == "claim_SE", tag
        plan = decode(snap(v))
        assert plan.validate() == []
        assert int(ga.gene(snap(v), "land_SE")) < NEVER
        se = [c for c in plan.cohorts if c.quadrant == "SE"]
        assert se, plan.to_table()
        assert all(c.plant_day >= plan.land_days["SE"] for c in se)
        assert len(plan.cohorts) > 6, plan.to_table()
    assert fired == 60


def test_op_claim_quadrant_declines_when_there_is_nothing_left_to_buy():
    rng = random.Random(13)
    v = list(_base())
    for q in ("NE", "SW", "SE"):
        v[GENE_INDEX[f"land_{q}"]] = 6.0
    assert ga.op_claim_quadrant(v, rng) is None


def test_placement_operators_never_propose_land_the_plan_does_not_buy():
    """A cohort on unbought land is not a mutation, it is a drop: `decode` discards it and `encode`
    writes the slot back empty. Measured, blind placement pulled the live-cohort mode from 6 to 3 —
    the population losing the width S3 just added, one mutation at a time."""
    rng = random.Random(14)
    base = _base()                                          # SE never bought
    se = QUADRANTS.index("SE")
    for op in (ga.op_cohort_add, ga.op_cohort_retarget):
        for _ in range(120):
            v = list(base)
            tag = op(v, rng)
            if tag is None:
                continue
            i = int(tag[len("cohort"):].split("_")[0])
            assert int(v[GENE_INDEX[f"c{i}_quad"]]) != se, op.__name__
            quad = QUADRANTS[int(v[GENE_INDEX[f"c{i}_quad"]])]
            opens = 0 if quad == "NW" else int(ga.gene(base, f"land_{quad}"))
            assert int(v[GENE_INDEX[f"c{i}_day"]]) >= opens, (op.__name__, quad)


def test_mutation_can_reach_the_slots_the_widening_added():
    """The E44 check on the widening itself: a wider genome the operators never fill is not a wider
    search. A 20-mutation walk from the 6-cohort seed must reach 8+ live cohorts sometimes."""
    rng = random.Random(15)
    base = _base()
    assert len(decode(base).cohorts) == 6
    reached = 0
    for _ in range(60):
        cur = base
        for _ in range(20):
            cur, _ops, _r = ga.mutate(cur, rng)
        reached += len(decode(cur).cohorts) >= 8
    assert reached >= 5, f"only {reached}/60 walks reached 8 cohorts"


def test_a_second_seed_genome_gets_half_the_mutants(tmp_path):
    """Round 2 seeds from the incumbent *and* the migrated candidate (S3).

    The two are far apart — the candidate beats the incumbent 80/80 and then loses 0/80 to Boatlee —
    so mutating only one of them throws away half of what round 1 bought. Round-robin, not
    "one seed plus 46 mutants of the other".
    """
    from search.run import Search, build_parser

    cand = list(encode(Plan.boatlee_like()))
    cand[GENE_INDEX["n_cows"]] = 6.0
    cand[GENE_INDEX[floor_gene("WOOL")]] = 0.131
    path = tmp_path / "cand.json"
    path.write_text(json.dumps([float(x) for x in cand]))

    s = Search(build_parser().parse_args(["--pop", "12", "--seed-genome", str(path)]))
    seeds = s.seed_genomes()
    assert len(seeds) == 2
    assert seeds[0] == ga.canonical(encode(Plan.boatlee_like()))
    assert decode(seeds[1]).consts["sell_floor"]["WOOL"] == 0.131
    s.seed_population()
    assert s.population[:2] == seeds, "both seeds must survive verbatim"
    assert len(s.population) == 12
    assert all(decode(v).validate() == [] for v in s.population)


def test_a_seed_genome_from_the_old_layout_is_migrated_not_rejected(tmp_path):
    """The harvested candidate is 54 floats on disk. `--seed-genome` takes it as it is."""
    from agent.plan import LAYOUT_V54
    from search.run import Search, build_parser

    old = encode(Plan.boatlee_like())
    old[GENE_INDEX[floor_gene("MELON")]] = 0.695
    v54 = [old[GENE_INDEX[n]] for n in LAYOUT_V54]
    path = tmp_path / "v54.json"
    path.write_text(json.dumps(v54))

    s = Search(build_parser().parse_args(["--pop", "4", "--seed-genome", str(path)]))
    seeds = s.seed_genomes()
    assert len(seeds[-1]) == len(GENES)
    assert decode(seeds[-1]).consts["sell_floor"]["MELON"] == 0.695


def test_a_duplicate_seed_genome_does_not_halve_its_own_mutant_budget(tmp_path):
    from search.run import Search, build_parser

    path = tmp_path / "same.json"
    path.write_text(json.dumps([float(x) for x in encode(Plan.boatlee_like())]))
    s = Search(build_parser().parse_args(["--pop", "6", "--seed-genome", str(path)]))
    assert len(s.seed_genomes()) == 1
