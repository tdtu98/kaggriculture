"""Measure routing efficiency, the quantity E28 identified as the root cause of the loss.

    steps walked per productive action:  ours 1.84   boatlee 1.01

That single number gates everything else. It sets how many productive actions the same 11 units
deliver (2,011 vs 2,801), which sets how many crop tiles can be serviced (10 vs 63), which is what
makes BUY_LAND pay -- and BUY_LAND is necessary and sufficient for beating us (E26).

Runs entirely on kagsim, which is bit-exact against the reference environment on money, action
counts and these very metrics (E27/E28) at ~14x the speed for a single episode. Both seatings are
played for every seed, because the seats are not symmetric.

Usage:
    PYTHONPATH=. python tools/routing_bench.py                       # champion vs boatlee
    PYTHONPATH=. python tools/routing_bench.py --games 24 --opponent champion
    PYTHONPATH=. python tools/routing_bench.py --set water_mode=both --set buy_land=True
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import time

import kagsim

from agent import Params, make_agent
from arena.registry import REGISTRY

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
CHAMPION = "search/champion.json"


def _coerce(text: str):
    """`--set` values arrive as strings; keep JSON literals but leave bare words alone."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _metrics(traces: list[dict]) -> dict:
    """Productive share and mean walk between consecutive productive actions, per unit."""
    walk: collections.Counter = collections.Counter()
    gaps: list[int] = []
    prod = moves = total = 0
    for act in traces:
        units = [act.get("farmer")] + list(act.get("hands") or [])
        for i, u in enumerate(units):
            if not u:
                continue
            total += 1
            if u[0] in MOVES:
                walk[i] += 1
                moves += 1
            elif u[0] == "PASS":
                pass
            else:
                prod += 1
                gaps.append(walk[i])
                walk[i] = 0
    if not total:
        return {"productive": 0, "productive_pct": 0.0, "steps_per_action": 0.0, "move_pct": 0.0}
    return {
        "productive": prod,
        "productive_pct": 100 * prod / total,
        "steps_per_action": sum(gaps) / max(1, len(gaps)),
        "move_pct": 100 * moves / total,
    }


def play(ours: dict, opponent: str, seeds: list[int], steps: int = 719) -> dict:
    """Play both seatings of every seed; return our metrics and the head-to-head record."""
    wins = games = 0
    our_money: list[float] = []
    their_money: list[float] = []
    traces: list[dict] = []

    for seed in seeds:
        for our_seat in (0, 1):
            agents = [None, None]
            agents[our_seat] = make_agent(Params(**ours))
            agents[1 - our_seat] = REGISTRY[opponent].build()
            sim = kagsim.Sim({"episodeSteps": steps + 1, "seed": seed})
            for _ in range(steps):
                acts = [agents[0](sim.observation(0)), agents[1](sim.observation(1))]
                traces.append(acts[our_seat])
                sim.step(acts)
            farms = sim.observation(0)["farms"]
            ours_m = float(farms[our_seat]["money"])
            theirs_m = float(farms[1 - our_seat]["money"])
            our_money.append(ours_m)
            their_money.append(theirs_m)
            wins += ours_m > theirs_m
            games += 1

    out = _metrics(traces)
    out.update(
        winrate=100 * wins / games,
        wins=wins,
        games=games,
        our_money=st.mean(our_money),
        their_money=st.mean(their_money),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="boatlee", help="registered agent to play against")
    ap.add_argument("--games", type=int, default=12, help="seeds; each is played in both seats")
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--params", default=CHAMPION)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Params field, e.g. --set water_mode=both")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    params = dict(json.load(open(args.params))["params"])
    for item in args.set:
        key, _, value = item.partition("=")
        params[key] = _coerce(value)

    seeds = list(range(args.seed0, args.seed0 + args.games))
    t0 = time.perf_counter()
    r = play(params, args.opponent, seeds)
    elapsed = time.perf_counter() - t0

    label = args.label or (", ".join(args.set) if args.set else "champion")
    print(f"\n{label}  vs {args.opponent}   ({r['games']} games, {elapsed:.1f}s)")
    print(f"  steps per productive action  {r['steps_per_action']:>8.2f}   <- the target (boatlee 1.01)")
    print(f"  productive unit-turns        {r['productive_pct']:>7.1f}%   ({r['productive']:,})")
    print(f"  movement                     {r['move_pct']:>7.1f}%")
    print(f"  winrate                      {r['winrate']:>7.1f}%   ({r['wins']}/{r['games']})")
    print(f"  money                        ${r['our_money']:>9,.0f} vs ${r['their_money']:,.0f}")


if __name__ == "__main__":
    main()
