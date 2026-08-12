"""Episode runner over kagsim, with per-day diagnostics (T0.6).

Parallelism is by process, not by thread: baseline and scripted agents are Python callables, so
they cannot run inside Rust's rayon threads. The array-based batched VecEnv is only worth building
once the policy is a neural net that can be evaluated on a whole batch at once (T3.1).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import kagsim

Agent = Callable[[dict], dict]

DEFAULT_CONFIG = {"episodeSteps": 720, "turnsPerDay": 24}


@dataclass
class EpisodeResult:
    seed: int
    money: list[float]
    winner: int | None            # None on a tie
    steps: int
    stats: list[dict] = field(default_factory=list)
    daily: list[dict] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return self.money[0] - self.money[1]


def _daily_snapshot(sim: kagsim.Sim, player: int) -> dict:
    """The metrics that actually drive improvements: idle land, weeds, and shed pressure."""
    obs = sim.observation(player)
    farm = obs["farms"][player]
    tiles = [t for row in farm["tiles"] for t in row]
    priv = obs["private"]
    return {
        "day": obs["day"],
        "money": farm["money"],
        "tiles_empty": sum(1 for t in tiles if t is None),
        "tiles_locked": sum(1 for t in tiles if t == "LOCKED"),
        "tiles_weed": sum(1 for t in tiles if isinstance(t, dict) and t.get("kind") == "WEED"),
        "tiles_plant": sum(1 for t in tiles if isinstance(t, dict) and t.get("kind") == "PLANT"),
        "tiles_animal": sum(1 for t in tiles if isinstance(t, dict) and "animal" in t),
        "shed_used": sum(priv["shed"].values()),
        "hands": len(farm["hands"]),
    }


def play_episode(
    agents: Sequence[Agent],
    seed: int = 0,
    config: dict | None = None,
    collect_stats: bool = False,
) -> EpisodeResult:
    cfg = {**DEFAULT_CONFIG, **(config or {}), "seed": seed}
    sim = kagsim.Sim(dict(cfg))
    if collect_stats:
        sim.collect_stats = True

    tpd = cfg.get("turnsPerDay", 24)
    # 719 agent turns for episodeSteps=720, not 718. The reference framework calls each agent
    # `episodeSteps - 1` times (measured: `env.run` yields exactly 719 observations per seat), and
    # kagsim is bit-identical to it at that count against a third-party agent -- off by one it is
    # not. The dropped turn is the terminal one, where end-of-season liquidation happens, so the
    # whole arena and every CEM score were optimising a season one turn shorter than the real one.
    steps = cfg["episodeSteps"] - 1
    daily: list[dict] = []

    for i in range(steps):
        sim.step([agents[p](sim.observation(p)) for p in range(2)])
        if collect_stats and (i + 1) % tpd == 0:
            daily.append({p: _daily_snapshot(sim, p) for p in range(2)})

    money = [sim.money(0), sim.money(1)]
    winner = None if money[0] == money[1] else int(money[1] > money[0])
    return EpisodeResult(
        seed=seed,
        money=money,
        winner=winner,
        steps=steps,
        stats=[sim.stats(p) for p in range(2)] if collect_stats else [],
        daily=daily,
    )


_POOL_STATE: dict = {}


def _init_worker(agent_names, config, collect_stats):
    from .baselines import AGENTS

    _POOL_STATE["agents"] = [AGENTS[n] for n in agent_names]
    _POOL_STATE["config"] = config
    _POOL_STATE["collect_stats"] = collect_stats


def _run_one(seed: int) -> EpisodeResult:
    return play_episode(
        _POOL_STATE["agents"], seed, _POOL_STATE["config"], _POOL_STATE["collect_stats"]
    )


def play_many(
    agent_names: Sequence[str],
    seeds: Sequence[int],
    config: dict | None = None,
    collect_stats: bool = False,
    workers: int | None = None,
) -> list[EpisodeResult]:
    """Run many episodes across processes. `agent_names` index into `sim.baselines.AGENTS`."""
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        _init_worker(agent_names, config, collect_stats)
        return [_run_one(s) for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(list(agent_names), config, collect_stats)
    ) as pool:
        return pool.map(_run_one, list(seeds))


def summarize(results: Sequence[EpisodeResult]) -> dict:
    n = len(results)
    wins = sum(1 for r in results if r.winner == 0)
    losses = sum(1 for r in results if r.winner == 1)
    return {
        "episodes": n,
        "p0_mean_money": sum(r.money[0] for r in results) / n,
        "p1_mean_money": sum(r.money[1] for r in results) / n,
        "p0_winrate": wins / n,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
    }
