"""T0.6 — baselines must produce identical outcomes on kagsim and the reference env.

This closes the loop through the observation API: the agent reads kagsim's own observation and
its actions feed back in. A parity bug in `Sim.observation` that `test_parity.py` cannot see
(because that harness drives both sides with the *same* externally-generated actions) shows up
here as a different final bank balance.
"""

from __future__ import annotations

import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import pass_agent, starter_agent

import kagsim

STEPS = 718
AGENTS = {"pass": pass_agent, "starter": starter_agent}


def reference_money(name: str, seed: int) -> list[float]:
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.reset(num_agents=2)
    agent = AGENTS[name]
    for _ in range(STEPS):
        env.step([agent(env.state[p].observation) for p in range(2)])
    return [float(f["money"]) for f in env.state[0].observation["farms"]]


def kagsim_money(name: str, seed: int) -> list[float]:
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    agent = AGENTS[name]
    for _ in range(STEPS):
        sim.step([agent(sim.observation(p)) for p in range(2)])
    return [sim.money(0), sim.money(1)]


@pytest.mark.parametrize("name", ["pass", "starter"])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_closed_loop_matches_reference(name, seed):
    assert kagsim_money(name, seed) == reference_money(name, seed)


def test_starter_hits_the_documented_baseline():
    # 3495 -> 3488 at kaggle-environments 1.32.6, which cut town-centre demand ~4.7x and made shop
    # unlocks draw with replacement (E33). The value is pinned to catch drift, not because the
    # number means anything; `test_closed_loop_matches_reference` above is the real check, and it
    # compares kagsim against whatever the installed reference does.
    assert kagsim_money("starter", 7) == [3488.0, 3488.0]
    assert kagsim_money("pass", 7) == [3000.0, 3000.0]


def test_stats_expose_wasted_actions():
    """`starter` moves one tile and idles: the diagnostics must make that obvious."""
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 7})
    sim.collect_stats = True
    for _ in range(STEPS):
        sim.step([starter_agent(sim.observation(p)) for p in range(2)])
    s = sim.stats(0)
    assert s["actions_total"] == STEPS
    assert s["actions_noop"] / s["actions_total"] > 0.9, "starter should be >90% no-ops"
    assert s["sold_units"] == {"CARROT": 18}


def test_stats_are_off_by_default():
    sim = kagsim.Sim({"episodeSteps": 10, "seed": 1})
    assert sim.collect_stats is False
    for _ in range(8):
        sim.step([pass_agent(sim.observation(p)) for p in range(2)])
    assert sim.stats(0)["actions_total"] == 0
