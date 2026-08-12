"""T2.1 — search-space encoding and the CEM loop.

The encoding is the part that fails silently: a knob that never decodes back into `Params`, or a
bound that lets an invalid config through, produces a search that appears to run and optimizes
nothing. These tests pin the round trip and the invariants CEM relies on.
"""

from __future__ import annotations

import random

import pytest

from dataclasses import asdict

from agent.params import Params
from search.cem import evaluate
from search.space import DIM, KNOBS, bounds, clip_vector, decode, describe, encode


# --------------------------------------------------------------------- encoding

def test_round_trip_is_exact_for_defaults():
    p = Params()
    assert decode(encode(p)) == p


@pytest.mark.parametrize("seed", range(8))
def test_round_trip_is_stable_for_random_points(seed):
    """decode is idempotent: decoding a clipped vector twice must not drift."""
    rng = random.Random(seed)
    lo, hi = bounds()
    vec = [rng.uniform(a, b) for a, b in zip(lo, hi)]
    p1 = decode(vec)
    p2 = decode(encode(p1))
    assert p1 == p2


def test_every_knob_reaches_params():
    """A knob whose path does not resolve would be searched over and silently discarded."""
    lo, hi = bounds()
    base = Params()
    for i, k in enumerate(KNOBS):
        vec = encode(base)
        # Move this one knob to the far end of its range.
        vec[i] = hi[i] if vec[i] - lo[i] < hi[i] - vec[i] else lo[i]
        changed = decode(vec)
        assert changed != base or k.kind == "bool", f"{k.path} had no effect on Params"


def test_bounds_are_enforced_in_both_directions():
    lo, hi = bounds()
    for vec, expect in [([-1e6] * DIM, lo), ([1e6] * DIM, hi)]:
        got = clip_vector(vec)
        for k, g, e in zip(KNOBS, got, expect):
            if k.kind == "int":
                assert g == round(e), k.path
            else:
                assert g == pytest.approx(e), k.path


def test_int_knobs_decode_to_ints_and_bools_to_bools():
    d = decode([0.6] * DIM)
    assert isinstance(d.hire_max, int) and isinstance(d.forecast_horizon, int)
    assert isinstance(d.buy_land, bool) and isinstance(d.care, bool)


def test_degenerate_crop_mix_falls_back_rather_than_planting_nothing():
    """An all-zero mix would leave the engine with nothing to plant for the whole season."""
    d = decode([0.0] * DIM)
    assert sum(d.crop_mix.values()) > 0
    assert d.crop_mix["MELON"] == 1.0


def test_describe_reports_only_differences():
    assert describe(encode(Params())) == "(defaults)"
    v = encode(Params(hire_max=3))
    assert "hire_max=3" in describe(v)
    assert "cash_floor" not in describe(v)


# -------------------------------------------------------------------- CEM loop

def test_evaluate_uses_common_random_numbers_and_both_seatings():
    """Identical candidates must score identically — otherwise CEM is chasing seed luck."""
    from search.cem import _init

    _init([asdict(Params())], {"episodeSteps": 120})
    v = encode(Params())
    scores = evaluate(None, [v, v], seeds=[1, 2])
    assert scores[0] == scores[1]


def test_a_clearly_worse_candidate_scores_lower():
    """Sanity that the fitness has the right sign: near-inaction must lose.

    Uses a **full-length** episode deliberately. The champion is a melon strategy, which is
    cash-negative until its first harvest around day 10-12: on a 300-step (~12-day) episode a
    do-nothing config leads it by ~$1,500 and this assertion inverts. Truncating episodes to save
    search time would silently invert the fitness for exactly the strategies that win.
    """
    from search.cem import _init

    _init([asdict(Params())], {"episodeSteps": 720})
    good = encode(Params())
    crippled = encode(Params(hire_max=0, seed_budget_frac=0.05, cash_floor=1500.0,
                             crop_mix={"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0,
                                       "STRAWBERRY": 1.0, "MELON": 0.0}))
    scores = evaluate(None, [good, crippled], seeds=[5])
    assert scores[0]["margin"] > scores[1]["margin"]


def test_fitness_is_a_margin_so_the_champion_scores_zero_against_itself():
    from search.cem import _init

    _init([asdict(Params())], {"episodeSteps": 200})
    scores = evaluate(None, [encode(Params())], seeds=[7])
    assert scores[0]["margin"] == pytest.approx(0.0, abs=1e-9), \
        "self-play with mirrored seatings must net exactly zero"


def test_default_params_are_not_the_champion():
    """`Params()` is wheat-based; the champion is melon.

    An earlier CEM run took its opponent from `Params()` and so optimized against a weak agent,
    reporting a +$34k margin that was worth ~$200 in the arena — the E5 mistake reintroduced
    through a dataclass default. `--champion` is now explicit; this pins the distinction so the
    defaults cannot quietly become the opponent again.
    """
    from arena.registry import REGISTRY

    champ = Params(**REGISTRY["melon"].params)
    assert Params().crop_mix["WHEAT"] == 1.0 and Params().crop_mix["MELON"] == 0.0
    assert champ.crop_mix["MELON"] == 1.0
    assert champ != Params()


def test_cem_champion_argument_is_required_to_exist():
    """A typo in `--champion` must fail loudly, not fall back to a weak default."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "search/cem.py", "--champion", "nope", "--gens", "1"],
        capture_output=True, text=True, cwd=".", env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode != 0
    assert "unknown champion" in (r.stdout + r.stderr)


def test_pool_fitness_is_equal_weighted_across_opponents():
    """A pool member that is easy to beat must not outvote one that is hard.

    This is the whole point of the pool: E10's champion beat every opponent in its search field and
    lost 0/80 to one outside it, so an average letting a weak member dominate reproduces exactly
    that failure. Asserted as the arithmetic property, not via a specific matchup — the champion
    changes, and after E11 it is *identical* to `x-dumper`, so any "champion loses to X" assumption
    goes stale.
    """
    from arena.registry import REGISTRY
    from search.cem import _init

    hard = REGISTRY["x-dumper"].params
    easy = REGISTRY["melon-wheat"].params
    _init([hard, easy], {"episodeSteps": 720})

    scores = evaluate(None, [encode(Params(**REGISTRY["champion"].params))],
                      seeds=[11], n_opponents=2)[0]
    per = scores["per_opponent"]
    assert len(per) == 2
    assert per[1][0] > per[0][0], "melon-wheat must be the easier of the two"
    # Equal weighting: the headline margin is the mean of the per-opponent means, so a blowout
    # against the weak member cannot drown out the result against the strong one.
    assert scores["margin"] == pytest.approx((per[0][0] + per[1][0]) / 2)
    assert scores["margin"] < per[1][0], "the blowout alone must not set the score"


def test_self_play_with_mirrored_seats_nets_exactly_zero():
    """Also documents E11: the champion and `x-dumper` are now the same agent, so this pairing is
    self-play and must score exactly 0 rather than approximately 0."""
    from arena.registry import REGISTRY
    from search.cem import _init

    # Not asserting champion == x-dumper any more: that was true of the E11-era champion, which
    # sold everything unconditionally. The current one carries a wool reserve (E19), so the two
    # have diverged again — a fact about one past agent is not an invariant.
    _init([REGISTRY["champion"].params], {"episodeSteps": 720})
    s = evaluate(None, [encode(Params(**REGISTRY["champion"].params))], [11], n_opponents=1)[0]
    assert s["margin"] == 0.0 and s["winrate"] == 0.5


def test_per_opponent_winrates_are_reported():
    """Without these, a pool-trained agent can hide a 0% against one member behind a good mean."""
    from arena.registry import REGISTRY
    from search.cem import _init

    _init([REGISTRY["champion"].params, REGISTRY["x-dumper"].params], {"episodeSteps": 720})
    s = evaluate(None, [encode(Params(**REGISTRY["champion"].params))], [3], n_opponents=2)[0]
    assert "per_opponent" in s and len(s["per_opponent"]) == 2
    assert all(0.0 <= r <= 1.0 for _, r in s["per_opponent"])


# ------------------------------------------------------------------- pinning

def test_pins_remove_a_knob_from_the_search():
    """Pinning must *fix* the value, not merely bias it — an ablation has to be an ablation."""
    from search.space import apply_pins, parse_pins

    pins = parse_pins("forecast_weight=0,reserve_frac.*=0")
    assert len(pins) == 10, "forecast_weight + all nine reserve_frac entries"

    rng = random.Random(0)
    lo, hi = bounds()
    for _ in range(20):
        vec = apply_pins([rng.uniform(a, b) for a, b in zip(lo, hi)], pins)
        d = decode(vec)
        assert d.forecast_weight == 0.0
        assert set(d.reserve_frac.values()) == {0.0}


def test_wildcard_pins_only_touch_their_own_group():
    from search.space import apply_pins, parse_pins

    pins = parse_pins("reserve_frac.*=0")
    d = decode(apply_pins(encode(Params(hire_max=11)), pins))
    assert set(d.reserve_frac.values()) == {0.0}
    assert d.hire_max == 11, "an unrelated knob must survive"
    assert d.crop_mix != {c: 0.0 for c in d.crop_mix}, "crop_mix is a different group"


def test_unknown_pin_fails_loudly():
    from search.space import parse_pins

    with pytest.raises(SystemExit):
        parse_pins("no_such_knob=1")


def test_every_default_lies_inside_its_search_bound():
    """A `Params` default outside its `Knob` range breaks the encode/decode round trip.

    Caught in practice: tightening `goose_min_cash`'s bound to 800 (E12) left the dataclass
    default at 900, so `decode(encode(p)) != p` for the defaults themselves. Cheap to assert, and
    it fails the moment a bound and a default drift apart again.
    """
    p = Params()
    for k in KNOBS:
        if "." in k.path:
            group, key = k.path.split(".")
            v = k.to_vector(getattr(p, group)[key])
        else:
            v = k.to_vector(getattr(p, k.path))
        assert k.lo <= v <= k.hi, f"default {k.path}={v} outside bound [{k.lo}, {k.hi}]"
