"""The session-line agents load, and the executor actually plays.

`session_line/executor.py` ends in a bare `except Exception:` that returns all-PASS. That turns any
wiring mistake into a complete, well-formed season of doing nothing which scores the $3,000 starting
bank — a number indistinguishable from a measurement. It has already happened once: the executor
calls `planner.plan(obs, n, pool=...)`, only `planner-2.py` has `pool`, and against `planner.py`
every turn raised TypeError behind the guard.

So this file asserts the executor *acts*, in play, rather than that it imports. Track F's numbers
are worthless without it.
"""

from __future__ import annotations

import pytest

import kagsim
from session_line import load
from sim.baselines import AGENTS

AGENT_MODULES = ["executor", "warfare", "metered_LOSES", "denial_LOSES"]
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


@pytest.mark.parametrize("name", AGENT_MODULES)
def test_every_session_agent_loads_and_exposes_agent(name):
    """They hard-code sandbox paths; `_load` repoints them. If that breaks, it breaks here."""
    mod = load(name)
    assert callable(getattr(mod, "agent", None)), f"{name} has no agent(obs)"


def test_boatlee_substitution_points_at_the_identical_copy():
    """The whole line is built on Boatlee's helpers, so the file behind the substitution has to be
    the one they were written against — checked by content, not by filename."""
    import hashlib

    from session_line import _load

    digest = hashlib.sha256(open(_load.BOATLEE, "rb").read()).hexdigest()
    assert digest == "3c9b6e75d1bb9cc1f23b6bf5d8821c84193d1306d5bcb74ada1628359e3fb025"


def test_executor_is_not_silently_passing():
    """The guard test. Counts real ops in play; all-PASS is the failure this exists to catch."""
    executor = load("executor").agent
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 5})
    real_ops = 0
    market_orders = 0
    for _ in range(240):                      # ten days is plenty to see a farm being worked
        action = executor(sim.observation(0))
        for unit in [action.get("farmer")] + list(action.get("hands") or []):
            if unit and unit[0] != "PASS":
                real_ops += 1
        market_orders += len(action.get("market") or [])
        sim.step([action, PASS_ACTION])

    assert real_ops > 100, f"executor issued {real_ops} non-PASS ops in 10 days — it is inert"
    assert market_orders > 0, "executor never hired, bought or sold"


def test_executor_beats_the_starting_bank_over_a_season():
    """End to end: a full season against `starter` must clear $3,000 by a wide margin.

    Deliberately a floor, not a target — 4-seed smoke put this at 67-82k, but this file is a wiring
    guard and the real measurement is the F-gate (80 games, both seats, fresh block).
    """
    executor = load("executor").agent
    starter = AGENTS["starter"]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 5})
    for _ in range(719):
        sim.step([executor(sim.observation(0)), starter(sim.observation(1))])

    assert sim.money(0) > 20_000, (
        f"executor ended on ${sim.money(0):,.0f}. At exactly $3,000 it played a whole season of "
        f"PASS — check session_line/_load.py's substitutions before reading this as a result."
    )
