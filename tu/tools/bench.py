"""T0.5 — throughput benchmark: reference env vs kagsim.

Also re-checks that replaying a recorded `starter` game through kagsim reproduces the reference's
final money exactly, so the benchmark can never be "fast but wrong".
"""

from __future__ import annotations

import time

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

import kagsim

STEPS = 718
CFG = {"episodeSteps": 720, "seed": 7}


def record_starter_game() -> tuple[list, float]:
    """Run the reference env with `starter` vs `starter`, capturing every action."""
    env = make("kaggriculture", configuration=CFG)
    env.reset(num_agents=2)
    actions = []
    for _ in range(STEPS):
        obs = [env.state[p].observation for p in range(2)]
        a = [starter_agent(o) for o in obs]
        actions.append(a)
        env.step(a)
    return actions, float(env.state[0].observation["farms"][0]["money"])


def time_reference() -> float:
    t0 = time.perf_counter()
    env = make("kaggriculture", configuration=CFG)
    env.run([starter_agent, starter_agent])
    return time.perf_counter() - t0


def time_kagsim(actions: list, repeats: int = 20) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        sim = kagsim.Sim(dict(CFG))
        for a in actions:
            sim.step(a)
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    actions, ref_money = record_starter_game()

    sim = kagsim.Sim(dict(CFG))
    for a in actions:
        sim.step(a)
    sim_money = sim.money(0)
    ok = sim_money == ref_money
    print(f"starter final money: reference=${ref_money:,.0f}  kagsim=${sim_money:,.0f}  "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("kagsim does not reproduce the reference outcome")

    ref_t = time_reference()
    sim_t = time_kagsim(actions)
    print(f"\n{'implementation':<24}{'sec/episode':>14}{'steps/s':>14}{'speedup':>12}")
    print(f"{'reference (python)':<24}{ref_t:>14.3f}{STEPS / ref_t:>14,.0f}{'1x':>12}")
    print(f"{'kagsim (rust)':<24}{sim_t:>14.5f}{STEPS / sim_t:>14,.0f}{ref_t / sim_t:>11.0f}x")
    print("\nNote: this path still marshals Python action dicts per step. T0.5's array-based "
          "batch API removes that overhead.")


if __name__ == "__main__":
    main()
