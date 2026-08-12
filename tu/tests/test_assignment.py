"""Optimal unit->task assignment (P1.2, E39).

Greedy nearest-task assignment measures 9.6% worse than optimal in total walk distance, and
walking is 56% of our unit-turns. At 11 units the problem is exactly solvable in microseconds, so
the engine solves it rather than searching for it.

The solver is the kind of code that is easy to get subtly wrong and still look plausible, so it is
checked against an independent implementation (scipy, if present) and against brute force.
"""

from __future__ import annotations

import itertools
import json
import random

import kagsim
import pytest

from agent import Params, make_agent
from agent.engine import _optimal_assignment

CHAMPION = json.load(open("search/champion.json"))["params"]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def cost_of(cost, match):
    return sum(cost[i][j] for i, j in match.items())


def brute_force(cost, n, m):
    best = None
    for combo in itertools.permutations(range(m), n):
        c = sum(cost[i][combo[i]] for i in range(n))
        best = c if best is None else min(best, c)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_on_small_matrices(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 5)
    m = rng.randrange(n, n + 4)
    cost = [[float(rng.randrange(0, 20)) for _ in range(m)] for _ in range(n)]
    assert cost_of(cost, _optimal_assignment(cost, n, m)) == pytest.approx(brute_force(cost, n, m))


def test_matches_scipy_on_realistic_sizes():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = random.Random(7)
    for _ in range(100):
        n = rng.randrange(1, 13)
        m = rng.randrange(n, n + 25)
        cost = [[float(rng.randrange(0, 40)) for _ in range(m)] for _ in range(n)]
        r, c = scipy_opt.linear_sum_assignment(cost)
        want = sum(cost[i][j] for i, j in zip(r, c))
        assert cost_of(cost, _optimal_assignment(cost, n, m)) == pytest.approx(want)


def test_every_row_is_assigned_exactly_once():
    rng = random.Random(3)
    n, m = 8, 20
    cost = [[float(rng.randrange(0, 30)) for _ in range(m)] for _ in range(n)]
    match = _optimal_assignment(cost, n, m)
    assert sorted(match) == list(range(n))
    assert len(set(match.values())) == n, "two units cannot be sent to the same task"


def test_beats_greedy_on_the_thing_it_was_built_for():
    """A direct check of the claim in E39, rather than trusting the end-to-end money number."""
    rng = random.Random(11)
    greedy_tot = opt_tot = 0.0
    for _ in range(200):
        n, m = 8, 14
        cost = [[float(rng.randrange(0, 30)) for _ in range(m)] for _ in range(n)]
        taken, g = set(), 0.0
        for i in range(n):                       # greedy in row order, as the old engine did
            j = min((j for j in range(m) if j not in taken), key=lambda j: cost[i][j])
            taken.add(j)
            g += cost[i][j]
        greedy_tot += g
        opt_tot += cost_of(cost, _optimal_assignment(cost, n, m))
    assert opt_tot < greedy_tot


def test_needs_constraints_are_never_violated_in_play():
    """The solver sees a large finite cost for illegal pairs; the engine must filter them out."""
    agent = make_agent(Params(**{**CHAMPION, "assign_mode": "optimal"}))
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 5})
    for _ in range(400):
        obs = sim.observation(0)
        farm, priv = obs["farms"][0], obs["private"]
        acts = agent(obs)
        units = [acts.get("farmer")] + list(acts.get("hands") or [])
        for i, a in enumerate(units):
            if a and a[0] == "FEED":
                inv = priv["inventories"][i] if i < len(priv["inventories"]) else {}
                assert inv.get("WHEAT", 0) > 0, "FEED emitted without wheat in hand"
        sim.step([acts, PASS])


def test_optimal_mode_is_off_by_default():
    """Same discipline as every other behaviour added this session: the champion must not move."""
    assert Params().assign_mode == "sequential"
