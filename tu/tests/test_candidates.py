"""Plan construction and sticky state (P1.2).

Originally written for a rollout search over candidate assignments. That search was superseded:
greedy was 9.6% off optimal, optimal is exactly solvable at 11 units, so the engine solves it
outright (E39). What survives is the part still used — building a plan without disturbing the
agent, and the guarantee that the default mode is unchanged behaviour.
"""

from __future__ import annotations

import json

import kagsim

from agent import Params, make_agent
from agent.engine import Engine

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
CHAMPION = json.load(open("search/champion.json"))["params"]


def mid_game(min_tasks=4, seed=5, limit=500):
    """Advance to a state that actually has work in it.

    Seeking rather than stepping a fixed number of turns, because task supply is spiky: at seed 5
    the engine has **zero** tasks at step 380 and 5 at step 260, while from day 12 it often has 20+.
    A hard-coded step count silently tests an empty task list -- which is how the first version of
    this file "passed" while asserting nothing (E39).
    """
    agent = make_agent(Params(**CHAMPION))
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    best = None
    for _ in range(limit):
        obs = sim.observation(0)
        farm = obs["farms"][0]
        n = len(agent.build_tasks(farm, obs["private"], obs["day"]))
        if n >= min_tasks and len(farm["hands"]) >= 1:
            best = obs
            break
        sim.step([agent(obs), PASS])
    assert best is not None, f"no state with >={min_tasks} tasks found in {limit} steps"
    return agent, best


def inputs(engine: Engine, obs: dict):
    farm, priv = obs["farms"][0], obs["private"]
    tasks = engine.build_tasks(farm, priv, obs["day"])
    units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    return units, tasks, priv["inventories"]


def test_uncommitted_plan_matches_committed_plan():
    agent, obs = mid_game()
    units, tasks, invs = inputs(agent, obs)
    before = dict(agent._assigned)
    a = agent.assign(units, tasks, invs, commit=False)
    agent._assigned = dict(before)
    b = agent.assign(units, tasks, invs)
    assert a == b, "an uncommitted plan must match the committed one"


def test_uncommitted_candidates_leave_sticky_state_alone():
    agent, obs = mid_game()
    units, tasks, invs = inputs(agent, obs)
    before = dict(agent._assigned)
    for _ in range(6):
        agent.assign(units, tasks, invs, commit=False)
    assert agent._assigned == before, "evaluating a candidate must not change the agent"


def test_committing_does_update_sticky_state():
    """The mirror of the test above: otherwise it could pass on an engine that never commits."""
    agent, obs = mid_game()
    units, tasks, invs = inputs(agent, obs)
    agent._assigned = {}
    agent.assign(units, tasks, invs, commit=True)
    assert agent._assigned, "a committed assignment must be remembered for stickiness"


def test_a_full_episode_is_unaffected_by_evaluating_candidates():
    """End to end: inspecting plans must not change the money earned."""
    def play(evaluate: bool):
        agent = make_agent(Params(**CHAMPION))
        sim = kagsim.Sim({"episodeSteps": 720, "seed": 11})
        for _ in range(400):
            obs = sim.observation(0)
            if evaluate:
                units, tasks, invs = inputs(agent, obs)
                for _ in range(4):
                    agent.assign(units, tasks, invs, commit=False)
            sim.step([agent(obs), PASS])
        return float(sim.observation(0)["farms"][0]["money"])

    assert play(evaluate=True) == play(evaluate=False)
