"""T0.8 — arena correctness.

The arena is what every later decision is accepted or rejected by (docs/decisions.md D10), so its
statistics and its seat handling have to be right. A biased schedule or an optimistic interval
would silently license acting on noise.
"""

from __future__ import annotations

import math

import pytest

from arena.registry import REGISTRY, AgentSpec, resolve
from arena.run import build_schedule, openskill_ratings, run, tabulate
from arena.stats import Winrate, games_needed


# ------------------------------------------------------------------ statistics

def test_wilson_interval_matches_known_values():
    # Textbook check: 50/100 -> roughly [40.4, 59.6].
    lo, hi = Winrate(50, 100).wilson()
    assert lo == pytest.approx(0.4038, abs=5e-4)
    assert hi == pytest.approx(0.5962, abs=5e-4)


def test_wilson_stays_inside_bounds_at_the_extremes():
    """The normal approximation gives negative lower bounds at 0%; Wilson does not."""
    lo, hi = Winrate(0, 30).wilson()
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = Winrate(30, 30).wilson()
    # The clamp keeps it from exceeding 1; the last bit is floating-point noise, not a bound error.
    assert hi <= 1.0 and hi == pytest.approx(1.0) and 0.8 < lo < 1.0


def test_interval_narrows_with_sample_size():
    assert Winrate(16, 32).half_width > Winrate(160, 320).half_width > Winrate(1600, 3200).half_width


def test_beats_even_requires_the_whole_interval_above_half():
    assert not Winrate(17, 32).beats_even(), "53% on 32 games is noise"
    assert Winrate(1700, 3200).beats_even(), "53% on 3200 games is real"


def test_games_needed_anchors_the_do_not_act_on_52_percent_rule():
    assert games_needed(0.02) == pytest.approx(2401, rel=0.01)
    assert games_needed(0.10) < 150


def test_draws_score_a_half():
    wr = Winrate(1.0, 2)      # one win, one draw
    assert wr.rate == 0.5


# ------------------------------------------------------------------- schedule

def test_schedule_is_seat_balanced():
    """Every pairing is played once in each seat on every seed, so seat advantage cancels."""
    sched = build_schedule(["a", "b", "c"], [0, 1])
    assert len(sched) == 3 * 2 * 2          # 3 pairings x 2 seeds x 2 seatings
    for m in sched:
        mirror = [x for x in sched
                  if x.a == m.a and x.b == m.b and x.seed == m.seed and x.a_first != m.a_first]
        assert len(mirror) == 1, f"no mirrored seating for {m}"


def test_schedule_has_no_self_play_or_duplicate_pairings():
    sched = build_schedule(["a", "b", "c"], [0])
    assert all(m.a != m.b for m in sched)
    pairs = {tuple(sorted((m.a, m.b))) for m in sched}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


# ------------------------------------------------------------------ tabulate

def _fake(a, b, ma, mb, seed=0, a_first=True):
    from arena.run import Match

    return (Match(a, b, seed, a_first), ma, mb, 0.0)


def test_pairwise_winrates_are_complementary():
    results = [_fake("x", "y", 10, 5), _fake("x", "y", 1, 9), _fake("x", "y", 3, 3)]
    pairwise, money, overall = tabulate(results)
    assert pairwise[("x", "y")].wins + pairwise[("y", "x")].wins == 3
    assert pairwise[("x", "y")].rate == pytest.approx(0.5)   # 1 win, 1 loss, 1 draw
    assert money["x"] == pytest.approx((10 + 1 + 3) / 3)


def test_overall_winrate_aggregates_every_opponent():
    results = [_fake("x", "y", 10, 1), _fake("x", "z", 0, 1)]
    _, _, overall = tabulate(results)
    assert overall["x"].games == 2 and overall["x"].wins == 1


def test_openskill_orders_a_transitive_field():
    results = ([_fake("strong", "weak", 10, 1)] * 20
               + [_fake("strong", "mid", 10, 1)] * 20
               + [_fake("mid", "weak", 10, 1)] * 20)
    r = openskill_ratings(["strong", "mid", "weak"], results)
    assert r["strong"] > r["mid"] > r["weak"]


# ---------------------------------------------------------------- end-to-end

def test_registry_specs_all_build():
    for name, spec in REGISTRY.items():
        assert callable(spec.build()), name


def test_fingerprint_tracks_config_not_name():
    a = AgentSpec("engine", {"hire_max": 8})
    b = AgentSpec("engine", {"hire_max": 8}, note="different note")
    c = AgentSpec("engine", {"hire_max": 9})
    assert a.fingerprint == b.fingerprint, "notes are documentation, not identity"
    assert a.fingerprint != c.fingerprint, "a changed knob must be a new fingerprint"


def test_resolve_rejects_unknown_agents():
    with pytest.raises(SystemExit):
        resolve(["starter", "does-not-exist"])


def test_arena_is_deterministic():
    """Same agents, same seeds -> identical results. Without this, nothing is comparable."""
    cfg = {"episodeSteps": 60}
    names = ["pass", "starter"]
    _, r1 = run(names, [0, 1], cfg, workers=1)
    _, r2 = run(names, [0, 1], cfg, workers=1)
    assert [(m.a, m.b, m.seed, ma, mb) for m, ma, mb, _ in r1] == \
           [(m.a, m.b, m.seed, ma, mb) for m, ma, mb, _ in r2]


def test_starter_beats_pass_and_pass_beats_random():
    """Sanity anchor with a known ordering: starter $3,495 > pass $3,000 > random $0."""
    cfg = {"episodeSteps": 200}
    _, results = run(["pass", "starter", "random"], [0, 1], cfg, workers=1)
    pairwise, money, _ = tabulate(results)
    assert pairwise[("starter", "pass")].rate == 1.0
    assert pairwise[("pass", "random")].rate == 1.0
    assert money["random"] < money["pass"] < money["starter"]


def test_seat_swap_actually_swaps_money_attribution():
    """Guards the easiest bug in the runner: reporting seat money instead of agent money."""
    cfg = {"episodeSteps": 200}
    _, results = run(["pass", "starter"], [0], cfg, workers=1)
    for m, ma, mb, _ in results:
        starter_money = ma if m.a == "starter" else mb
        pass_money = mb if m.a == "starter" else ma
        assert starter_money > pass_money, f"seating {m.a_first} mis-attributed"
        assert math.isclose(pass_money, 3000.0)
