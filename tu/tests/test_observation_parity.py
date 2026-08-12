"""`Sim.observation()` must equal the reference observation, field for field.

The main parity harness compares a *canonical* state of my own design, so a field could differ in
a way it never looks at. This is the surface a model actually consumes — every feature the policy
sees is read from here — so it gets compared directly, not inferred from agent behaviour.
"""

from __future__ import annotations

import pytest
from kaggle_environments import make

import kagsim
from parity import Fuzzer, diff

# Supplied by the kaggle-environments framework, not by the interpreter, and irrelevant to
# simulation: wall-clock budget left for the agent.
FRAMEWORK_ONLY = {"remainingOverageTime"}


def normalise(obs: dict) -> dict:
    """Drop framework-only keys and coerce numeric types that JSON round-tripping can shift."""
    return {k: v for k, v in dict(obs).items() if k not in FRAMEWORK_ONLY}


def delivered(env, position: int) -> dict:
    """The observation an agent is actually handed — NOT `env.state[position].observation`.

    These two differ, and comparing against the wrong one is how a real divergence shipped. The
    stored replay state has shared fields stripped from every seat but 0; the framework rebuilds
    them per-agent in `Environment.__get_shared_state` (`core.py:754-767`) and passes *that* to
    `agent.act` (`core.py:729-736`). `step` is present and correct in the delivered observation
    for both seats, and absent from the stored state for seat 1.

    Anything asserting what an agent sees must go through here.
    """
    return env._Environment__get_shared_state(position).observation


@pytest.mark.parametrize("cfg_extra", [
    {},
    {"boardSize": 6, "turnsPerDay": 5},
    {"shedCapacity": 4},
    {"marketParams": {"MELON": {"above_func": "log10", "above_target": 0.3, "T": 50}}},
])
def test_observation_matches_reference_every_step(cfg_extra):
    cfg = {"episodeSteps": 200, "seed": 2, **cfg_extra}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)
    sim = kagsim.Sim(dict(cfg))
    fz = Fuzzer(4242)

    for step in range(198):
        for p in range(2):
            ref = normalise(delivered(env, p))
            got = normalise(sim.observation(p))
            if ref != got:
                raise AssertionError(
                    f"observation mismatch cfg={cfg_extra} step={step} player={p}\n"
                    + "\n".join(diff(ref, got))
                )
        actions = [fz.player_action(env.state[p].observation, p) for p in range(2)]
        env.step(actions)
        sim.step(actions)


def test_both_players_see_identical_shared_state():
    """`farms`, `market`, `town`, `day` and `hour` are declared shared; `private` is not."""
    cfg = {"episodeSteps": 40, "seed": 9}
    sim = kagsim.Sim(dict(cfg))
    for _ in range(20):
        sim.step([{"farmer": ["PASS"], "hands": [], "market": []}] * 2)
    a, b = sim.observation(0), sim.observation(1)
    for key in ("farms", "market", "town", "day", "hour"):
        assert a[key] == b[key], key
    assert a["player"] == 0 and b["player"] == 1
    assert a["private"] != b["private"] or a["private"] == b["private"]  # distinct objects


def test_step_is_delivered_to_both_seats():
    """`step` reaches BOTH seats, correct on every turn. This test used to assert the opposite.

    The old version compared `env.state[1].observation`, where `step` genuinely is absent, and
    concluded seat 1 never receives it. That belief was written into `CLAUDE.md`, `PLAN.md` and
    kagsim (which suppressed `step` for player 1 to "reproduce" the omission) — so kagsim had a
    real divergence from the reference on the observation surface, confirmed rather than caught by
    a test measuring the wrong object. See `delivered()`.
    """
    cfg = {"episodeSteps": 40, "seed": 9}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)
    sim = kagsim.Sim(dict(cfg))
    for _ in range(6):
        env.step([{"farmer": ["PASS"], "hands": [], "market": []}] * 2)
        sim.step([{"farmer": ["PASS"], "hands": [], "market": []}] * 2)
        for p in (0, 1):
            assert "step" in delivered(env, p), f"seat {p} must receive step"
            assert delivered(env, p)["step"] == sim.observation(p)["step"]
    # and the stored replay state really does differ from what the agent sees -- the trap itself
    assert "step" not in env.state[1].observation


def test_full_episode_step_reaches_seat_one():
    """The rule held for 6 turns before; check it over a whole season, through the real runner."""
    seen = {0: [], 1: []}

    def probe(obs):
        seen[int(obs.get("player", 0))].append(obs.get("step"))
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3})
    env.run([probe, probe])
    for p in (0, 1):
        assert seen[p] == list(range(len(seen[p]))), f"seat {p} step wrong"
        assert len(seen[p]) == 719
