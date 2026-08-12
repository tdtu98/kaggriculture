"""Baseline agents, re-exported from the reference environment.

They are pure functions of the observation dict, so they run unmodified on kagsim's
`Sim.observation(player)`. Benchmarks: `starter` vs `starter` = $3,495; `random` ends at $0
because it buys seeds until broke.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    pass_agent,
    random_agent,
    starter_agent,
)

AGENTS = {
    "pass": pass_agent,
    "random": random_agent,
    "starter": starter_agent,
}
