"""Shadow one agent with another: same state, every turn — what would WE do here?

Five hypotheses about why our execution trails boatlee's were each plausible, each supported by a
real symptom, and each refuted (E47). The common flaw was comparing *outcomes* and inferring a
cause. This compares *decisions*: boatlee drives the simulation, and at every turn our engine is
handed the identical observation and asked what it would do. Disagreements are located exactly,
turn by turn, with no inference in between.

    PYTHONPATH=. python tools/shadow.py --driver boatlee --shadow champion --seed 5
"""

from __future__ import annotations

import argparse
import collections

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def kind(op: str) -> str:
    if op in MOVES:
        return "move"
    if op == "PASS":
        return "idle"
    if op in ("PICKUP", "DROP"):
        return "haul"
    return "work"


def run(driver: str, shadow: str, seed: int, steps: int = 719) -> dict:
    import kagsim
    from arena.registry import REGISTRY

    drv = REGISTRY[driver].build()
    shd = REGISTRY[shadow].build()
    sim = kagsim.Sim({"episodeSteps": steps + 1, "seed": seed})

    agree = collections.Counter()
    pairs = collections.Counter()
    kinds = collections.Counter()
    n = 0
    for _ in range(steps):
        obs = sim.observation(0)
        a = drv(obs)
        b = shd(obs)                      # same observation, same seat, same everything
        av = [a.get("farmer")] + list(a.get("hands") or [])
        bv = [b.get("farmer")] + list(b.get("hands") or [])
        for x, y in zip(av, bv):
            if not x or not y:
                continue
            n += 1
            agree["same" if x[0] == y[0] else "diff"] += 1
            pairs[(x[0], y[0])] += 1
            kinds[(kind(x[0]), kind(y[0]))] += 1
        sim.step([a, {"farmer": ["PASS"], "hands": [], "market": []}])
    return {"n": n, "agree": agree, "pairs": pairs, "kinds": kinds,
            "driver": driver, "shadow": shadow}


def show(r: dict) -> None:
    n = r["n"]
    print(f"\n  {r['driver']} drives; what would {r['shadow']} do in the same state?")
    print(f"  identical op on {r['agree']['same']:,} of {n:,} unit-turns "
          f"({100 * r['agree']['same'] / n:.0f}%)\n")

    print(f"  {'boatlee does':<12}{'we would':<12}{'count':>8}{'share':>8}")
    print(f"  {'-' * 40}")
    for (x, y), c in r["kinds"].most_common(9):
        flag = "" if x == y else "   <--"
        print(f"  {x:<12}{y:<12}{c:>8}{100 * c / n:>7.0f}%{flag}")

    print(f"\n  the specific disagreements that matter most:")
    diffs = [(p, c) for p, c in r["pairs"].items() if p[0] != p[1]]
    for (x, y), c in sorted(diffs, key=lambda t: -t[1])[:10]:
        print(f"    boatlee {x:<20} we would {y:<20}{c:>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="boatlee")
    ap.add_argument("--shadow", default="champion")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    show(run(args.driver, args.shadow, args.seed))


if __name__ == "__main__":
    main()
