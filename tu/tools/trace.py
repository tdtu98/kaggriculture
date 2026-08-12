"""Record a full episode — every action and the field state — for both players, and diff them.

Written after a session in which every diagnostic was a throwaway script that printed to a terminal
and was then lost. The cost of that was real: the same quantities were re-derived several times,
one comparison measured a re-implementation of our own engine instead of the engine (E39/E41), and
several findings survived only because they happened to be typed into a document.

A trace is a file. It can be re-queried without re-running anything, diffed against another agent
turn by turn, and kept as evidence for a claim rather than a number someone remembers.

    PYTHONPATH=. python tools/trace.py --seeds 5,6,7          # -> traces/boatlee_vs_champion_s5.json
    PYTHONPATH=. python tools/trace.py --load traces/... --crop STRAWBERRY
    PYTHONPATH=. python tools/trace.py --load traces/... --actions

Traces live in `traces/` in the repo, not a temp directory: they are evidence for claims in
`docs/experiments.md`, and this project has repeatedly lost findings that existed only in a
terminal. Written as plain indented JSON so they can simply be opened and read; pass an `--out` ending in
`.gz` if space ever matters. They are exactly reproducible from (agent a, agent b, seed) since
kagsim is deterministic, so an old trace can always be regenerated rather than kept forever.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os

TRACE_DIR = "traces"


def trace_path(a: str, b: str, seed: int) -> str:
    return os.path.join(TRACE_DIR, f"{a}_vs_{b}_s{seed}.json")


def save(tr: dict, path: str) -> int:
    """Plain JSON, indented. The point of a trace is that a human can open it and look."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        json.dump(tr, f, indent=1)
    return os.path.getsize(path)


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)

CROP_FIELDS = ("crop", "planted_day", "yield_units", "watered_today",
               "consecutive_unwatered", "fertilized_until_day")
ANIMAL_FIELDS = ("animal", "placed_day", "yield_units", "fed_today", "cared_today",
                 "consecutive_unfed", "fertilizer_available")


def _tile(v):
    """Compact, lossless-enough snapshot of one tile."""
    if v is None:
        return None
    if not isinstance(v, dict):
        return "L"                                   # LOCKED
    if v.get("kind") == "PLANT":
        return {k: v.get(k) for k in CROP_FIELDS}
    if "animal" in v:
        return {k: v.get(k) for k in ANIMAL_FIELDS}
    return {"kind": v.get("kind")}                   # WEED / empty structure


def record(a_name: str, b_name: str, seed: int, steps: int = 719) -> dict:
    import kagsim
    from arena.registry import REGISTRY

    for n in (a_name, b_name):
        if n not in REGISTRY:
            raise SystemExit(f"unknown agent {n!r}; have {len(REGISTRY)} in the registry")
    agents = [REGISTRY[a_name].build(), REGISTRY[b_name].build()]
    sim = kagsim.Sim({"episodeSteps": steps + 1, "seed": seed})
    turns, days = [], []

    for t in range(steps):
        acts = [agents[p](sim.observation(p)) for p in (0, 1)]
        turns.append([{"f": a.get("farmer"), "h": a.get("hands") or [], "m": a.get("market") or []}
                      for a in acts])
        sim.step(acts)
        if (t + 1) % 24 == 0:                        # one field snapshot per day
            obs = sim.observation(0)
            snap = {"day": obs["day"], "players": []}
            for p in (0, 1):
                farm = sim.observation(p)["farms"][p]
                priv = sim.observation(p)["private"]
                snap["players"].append({
                    "money": float(farm["money"]),
                    "tiles": [[_tile(v) for v in row] for row in farm["tiles"]],
                    "shed": dict(priv["shed"]),
                    "seeds": dict(priv.get("seeds", {})),
                    "hands": len(farm["hands"]),
                })
            days.append(snap)

    return {"a": a_name, "b": b_name, "seed": seed, "turns": turns, "days": days}


# ------------------------------------------------------------------ queries

def summarise(tr: dict) -> None:
    names = (tr["a"], tr["b"])
    print(f"\n{tr['a']} vs {tr['b']}, seed {tr['seed']}\n")
    print(f"{'day':>4} | {names[0][:14]:>14}{'crops':>7}{'weeds':>6} | "
          f"{names[1][:14]:>14}{'crops':>7}{'weeds':>6}")
    for snap in tr["days"]:
        cells = []
        for p in (0, 1):
            tiles = snap["players"][p]["tiles"]
            crops = sum(1 for r in tiles for v in r if isinstance(v, dict) and "crop" in v)
            weeds = sum(1 for r in tiles for v in r
                        if isinstance(v, dict) and v.get("kind") == "WEED")
            cells.append((snap["players"][p]["money"], crops, weeds))
        if snap["day"] % 3 == 0:
            print(f"{snap['day']:>4} | {cells[0][0]:>14,.0f}{cells[0][1]:>7}{cells[0][2]:>6} | "
                  f"{cells[1][0]:>14,.0f}{cells[1][1]:>7}{cells[1][2]:>6}")


def crop_lifecycle(tr: dict, crop: str) -> None:
    """Follow one crop type: how many exist, how old they get, how much yield they carry.

    The question this exists to answer: a plant that is alive is not the same as a plant that is
    producing. A crop that dies and is replanted every few days shows a healthy tile count and
    yields nothing.
    """
    for p in (0, 1):
        name = (tr["a"], tr["b"])[p]
        print(f"\n{name} — {crop}")
        print(f"  {'day':>4}{'plants':>8}{'max age':>9}{'yield on tiles':>16}{'harvest-ready':>15}")
        for snap in tr["days"]:
            tiles = snap["players"][p]["tiles"]
            live = [v for r in tiles for v in r
                    if isinstance(v, dict) and v.get("crop") == crop]
            if not live and snap["day"] % 3:
                continue
            ages = [snap["day"] - v["planted_day"] for v in live]
            y = sum(v["yield_units"] for v in live)
            ready = sum(1 for v in live if v["yield_units"] > 0)
            if snap["day"] % 3 == 0:
                print(f"  {snap['day']:>4}{len(live):>8}{(max(ages) if ages else 0):>9}"
                      f"{y:>16}{ready:>15}")


def action_mix(tr: dict) -> None:
    print(f"\n{'op':<20}{tr['a'][:14]:>14}{tr['b'][:14]:>14}")
    counts = [collections.Counter(), collections.Counter()]
    for turn in tr["turns"]:
        for p in (0, 1):
            for u in [turn[p]["f"]] + list(turn[p]["h"]):
                if u:
                    counts[p][u[0]] += 1
    for op in sorted(set(counts[0]) | set(counts[1]), key=lambda o: -counts[1][o]):
        print(f"{op:<20}{counts[0][op]:>14}{counts[1][op]:>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="boatlee")
    ap.add_argument("--b", default="champion")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--seeds", help="comma-separated, e.g. 5,6,7 — records one trace each")
    ap.add_argument("--out")
    ap.add_argument("--load")
    ap.add_argument("--crop", help="follow one crop type through the season")
    ap.add_argument("--actions", action="store_true")
    args = ap.parse_args()

    if args.load:
        tr = load(args.load)
    elif args.seeds:
        total = 0
        for sd in [int(x) for x in args.seeds.split(",")]:
            t = record(args.a, args.b, sd)
            path = args.out or trace_path(args.a, args.b, sd)
            n = save(t, path)
            total += n
            print(f"  {path}  ({n/1048576:.1f} MB)")
        print(f"\n  {len(args.seeds.split(','))} traces, {total/1048576:.1f} MB total")
        return
    else:
        tr = record(args.a, args.b, args.seed)
        path = args.out or trace_path(args.a, args.b, args.seed)
        print(f"wrote {path}  ({save(tr, path)/1048576:.1f} MB)")

    summarise(tr)
    if args.crop:
        crop_lifecycle(tr, args.crop)
    if args.actions:
        action_mix(tr)


if __name__ == "__main__":
    main()
