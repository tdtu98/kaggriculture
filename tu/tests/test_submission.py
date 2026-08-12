"""T1.5 — the submission entry point.

Everything here guards a failure that is invisible locally and forfeits real games: an import that
does not exist on the runner, state leaking between episodes, a seat-1-only bug, or an exception
that turns a won game into a loss.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import time

import pytest
from kaggle_environments import make

import main
from tools.build_submission import CONTENTS, check_no_local_only_imports

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


# ------------------------------------------------------------------ bundle shape

def test_bundle_has_no_local_only_imports():
    """`kagsim`, `numpy`, `torch`, `arena` and friends are not on the Kaggle runner.

    A missing import forfeits every game, and the only symptom is an empty scoreboard.
    """
    assert check_no_local_only_imports(CONTENTS) == []


def test_bundle_files_all_exist():
    import os

    assert [p for p in CONTENTS if not os.path.exists(p)] == []


def test_engine_does_not_import_kagsim_even_indirectly():
    """Stronger than the AST check: actually deny the import and re-import the agent."""
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "kagsim" or name.startswith("kagsim."):
            raise ImportError("kagsim is not available on the Kaggle runner")
        return real_import(name, *a, **kw)

    builtins.__import__ = blocked
    try:
        for mod in ("agent.params", "agent.engine", "agent", "main"):
            importlib.reload(importlib.import_module(mod))
        obs = _first_observation()
        assert isinstance(main.agent(obs), dict)
    finally:
        builtins.__import__ = real_import
        for mod in ("agent.params", "agent.engine", "agent", "main"):
            importlib.reload(importlib.import_module(mod))


def _first_observation(seed: int = 1):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.reset(num_agents=2)
    return env.state[0].observation


# ---------------------------------------------------------------- plays a game

@pytest.mark.parametrize("seat", [0, 1])
def test_plays_a_full_game_in_either_seat(seat):
    """Seat 1 is the one that matters: `obs["step"]` is absent there.

    An agent that reads it passes every seat-0 test and misbehaves in half of all real games.
    """
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 11})
    env.reset(num_agents=2)
    importlib.reload(main)
    for _ in range(718):
        acts = [PASS, PASS]
        acts[seat] = main.agent(env.state[seat].observation)
        env.step(acts)
    money = env.state[0].observation["farms"][seat]["money"]
    assert money > 20_000, f"seat {seat} only reached ${money:,.0f}"


def test_never_reads_the_step_field():
    """Belt and braces: hand it an observation with `step` removed entirely."""
    obs = dict(_first_observation())
    obs.pop("step", None)
    importlib.reload(main)
    assert isinstance(main.agent(obs), dict)


def test_turn_latency_is_far_inside_the_timeout():
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 5})
    env.reset(num_agents=2)
    importlib.reload(main)
    worst = 0.0
    for _ in range(300):
        t0 = time.perf_counter()
        a = main.agent(env.state[0].observation)
        worst = max(worst, time.perf_counter() - t0)
        env.step([a, PASS])
    assert worst < 0.25, f"worst turn {worst:.3f}s against a 1s actTimeout"


# ------------------------------------------------------------- episode handling

def test_state_resets_between_episodes():
    """The runner reuses the process; sticky task assignments must not leak across games."""
    importlib.reload(main)

    def play(seed, turns):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.reset(num_agents=2)
        for _ in range(turns):
            env.step([main.agent(env.state[0].observation), PASS])
        return env.state[0].observation["farms"][0]["money"]

    play(3, 200)                       # leave an episode half-finished
    fresh_after = play(9, 400)

    importlib.reload(main)             # a pristine process
    clean = play(9, 400)
    assert fresh_after == clean, "state leaked from the previous episode"


def test_a_failing_turn_passes_instead_of_forfeiting():
    """A raised exception loses the whole game; passing costs one turn."""
    importlib.reload(main)
    main.agent(_first_observation())            # build the engine

    def boom(_obs):
        raise RuntimeError("synthetic failure")

    main._ENGINE.__class__.__call__, original = boom, main._ENGINE.__class__.__call__
    try:
        assert main.agent(_first_observation()) == PASS
    finally:
        main._ENGINE.__class__.__call__ = original


def test_missing_champion_file_falls_back_to_defaults(tmp_path, monkeypatch):
    """A packaging slip must degrade to a working agent, not crash at import."""
    monkeypatch.setattr(main, "_HERE", str(tmp_path))
    params = main._load_params()
    from agent import Params

    assert params == Params()
