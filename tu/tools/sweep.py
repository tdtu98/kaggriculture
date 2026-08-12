"""Quick parameter sweep through the arena. A poor man's T2.1, for confirming a lever is real.

NOTE the `__main__` guard: `arena.run` uses the `spawn` start method, so every worker re-imports
this module. Without the guard the workers re-execute the sweep at import time and each spawns its
own pool — a recursive process explosion that never terminates.
"""

import sys

from arena.registry import REGISTRY, AgentSpec, mix
from arena.run import openskill_ratings, run, tabulate


def add(name, **params):
    REGISTRY[name] = AgentSpec("engine", {"hire_max": 8, "crop_mix": mix(MELON=1), **params}, name)
    return name


def main():
    # Joint grid. Coordinate-wise tuning misled twice: at horizon 4 the best weight was 0.6 and
    # w=1.0 was the worst config; at horizon 10 w=1.0 is the best. The knobs must be searched
    # together.
    names = ["melon-static"]
    for w in [0.7, 0.85, 1.0]:
        for h in [10, 14, 20]:
            names.append(add(f"w{w}h{h}", forecast_weight=w, forecast_horizon=h))

    games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    _, results = run(names, list(range(games)), {"episodeSteps": 720})
    pairwise, money, overall = tabulate(results)
    ratings = openskill_ratings(names, results)
    print(f"\n{'config':<10}{'skill':>8}{'winrate (95% Wilson)':>28}{'mean $':>11}")
    for n in sorted(names, key=lambda n: -ratings[n]):
        print(f"{n:<10}{ratings[n]:>8.2f}{str(overall[n]):>28}{money[n]:>11,.0f}")


if __name__ == "__main__":
    main()
