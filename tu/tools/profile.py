"""Full side-by-side profile of two agents, across every dimension we can measure.

Written after a session of one-factor-at-a-time investigation that kept finding nothing. E43 showed
why: fertiliser and eager watering are each worth ~nothing alone and +24% together, because the
environment only grants the fertiliser bonus on a day the tile was also watered. If the opponent's
advantage is a *combination*, testing knobs singly cannot find it -- and neither can a CEM that
samples each dimension independently around a mean (the same failure already recorded for
`goose_min_cash` in E12).

So: measure everything at once, and read the profile as a whole.

    PYTHONPATH=. python tools/profile.py --a boatlee --b champion --seeds 5,6,7,8
"""

from __future__ import annotations

import argparse
import collections
import statistics as st

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
GOODS = CROPS + ("MILK", "WOOL", "EGG", "FERTILIZER")


def profile(name: str, seeds: list[int], opponent: str) -> dict:
    import kagsim
    from arena.registry import REGISTRY

    acc = collections.defaultdict(float)
    counts = collections.Counter()
    sold = collections.Counter()
    bought = collections.Counter()
    planted = collections.Counter()
    peak_crop = collections.Counter()
    day_money = collections.defaultdict(list)

    for sd in seeds:
        me = REGISTRY[name].build()
        them = REGISTRY[opponent].build()
        sim = kagsim.Sim({"episodeSteps": 720, "seed": sd})
        for t in range(719):
            a = me(sim.observation(0))
            b = them(sim.observation(1))
            units = [a.get("farmer")] + list(a.get("hands") or [])
            for u in units:
                if not u:
                    continue
                counts["unit_turns"] += 1
                if u[0] in MOVES:
                    counts["move"] += 1
                elif u[0] == "PASS":
                    counts["pass"] += 1
                else:
                    counts[u[0]] += 1
                    if u[0] == "PLANT" and len(u) > 1:
                        planted[u[1]] += 1
            for o in (a.get("market") or []):
                if not o:
                    continue
                if o[0] == "SELL" and len(o) > 2:
                    sold[o[1]] += int(o[2])
                elif o[0] == "BUY_PRODUCT" and len(o) > 2:
                    bought[o[1]] += int(o[2])
                elif o[0] in ("HIRE", "BUY_LAND"):
                    counts[o[0]] += 1
                elif o[0] == "BUY_SEED" and len(o) > 2:
                    counts["seeds_bought"] += int(o[2])
                elif o[0] == "BUY_ANIMAL" and len(o) > 2:
                    counts["animals_bought"] += int(o[2])
            sim.step([a, b])
            # Sample at midday, not hour 0: `_end_of_day` clears the roster every night
            # (`kaggriculture.py:867`), so an hour-0 sample reports a farm that has not hired yet
            # and makes the workforce look three times smaller than it is.
            if t % 24 == 12:
                farm = sim.observation(0)["farms"][0]
                c = collections.Counter()
                for row in farm["tiles"]:
                    for v in row:
                        if isinstance(v, dict):
                            if v.get("kind") == "PLANT":
                                c[v["crop"]] += 1
                            elif "animal" in v:
                                c[v["animal"]] += 1
                            elif v.get("kind") == "WEED":
                                c["WEED"] += 1
                for k, n in c.items():
                    peak_crop[k] = max(peak_crop[k], n)
                acc["quadrants"] = max(acc["quadrants"], len(farm.get("unlocked_quadrants", [])))
                acc["hands"] = max(acc["hands"], len(farm["hands"]))
                day_money[t // 24].append(float(farm["money"]))
                acc["hand_days"] += len(farm["hands"])
        acc["money"] += float(sim.observation(0)["farms"][0]["money"])

    n = len(seeds)
    return {"money": acc["money"] / n, "quadrants": acc["quadrants"], "hands": acc["hands"],
            "mean_hands": acc["hand_days"] / (n * 30),
            "counts": {k: v / n for k, v in counts.items()}, "peak": dict(peak_crop),
            "sold": {k: v / n for k, v in sold.items()},
            "bought": {k: v / n for k, v in bought.items()},
            "planted": {k: v / n for k, v in planted.items()},
            "day_money": {d: st.mean(v) for d, v in day_money.items()}}


def show(a: dict, b: dict, na: str, nb: str) -> None:
    def row(label, x, y, fmt="{:,.0f}"):
        sx = fmt.format(x) if isinstance(x, (int, float)) else str(x)
        sy = fmt.format(y) if isinstance(y, (int, float)) else str(y)
        ratio = ""
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and y:
            ratio = f"{x / y:.1f}x" if y else ""
        print(f"  {label:<28}{sx:>12}{sy:>12}{ratio:>8}")

    print(f"\n  {'':<28}{na[:12]:>12}{nb[:12]:>12}{'ratio':>8}")
    print(f"  {'-'*60}")
    print("  LAND & LABOUR")
    row("final money", a["money"], b["money"])
    row("quadrants owned", a["quadrants"], b["quadrants"])
    row("peak hands (midday)", a["hands"], b["hands"])
    row("mean hands (midday)", a["mean_hands"], b["mean_hands"], "{:.1f}")
    row("HIRE orders", a["counts"].get("HIRE", 0), b["counts"].get("HIRE", 0))
    row("BUY_LAND orders", a["counts"].get("BUY_LAND", 0), b["counts"].get("BUY_LAND", 0))

    print("\n  FARM COMPOSITION (peak tiles)")
    for k in CROPS + ("COW", "SHEEP", "GOOSE", "WEED"):
        if a["peak"].get(k) or b["peak"].get(k):
            row(k.lower(), a["peak"].get(k, 0), b["peak"].get(k, 0))

    print("\n  WORK DONE (ops per season)")
    for k in ("PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
              "FEED", "CARE", "COLLECT_FERTILIZER", "PICKUP", "DROP"):
        row(k.lower(), a["counts"].get(k, 0), b["counts"].get(k, 0))

    print("\n  EFFORT SPLIT (% of unit-turns)")
    for lbl, keys in (("moving", ("move",)), ("idle", ("pass",)),
                      ("crop work", ("WATER", "HARVEST", "PLANT", "FERTILIZE")),
                      ("animal work", ("FEED", "CARE", "COLLECT_FERTILIZER", "PLACE")),
                      ("hauling", ("PICKUP", "DROP"))):
        fa = 100 * sum(a["counts"].get(k, 0) for k in keys) / max(1, a["counts"]["unit_turns"])
        fb = 100 * sum(b["counts"].get(k, 0) for k in keys) / max(1, b["counts"]["unit_turns"])
        row(lbl, fa, fb, "{:.1f}%")

    print("\n  TRADE (units per season)")
    row("seeds bought", a["counts"].get("seeds_bought", 0), b["counts"].get("seeds_bought", 0))
    row("animals bought", a["counts"].get("animals_bought", 0), b["counts"].get("animals_bought", 0))
    for k in GOODS:
        s_a, s_b = a["sold"].get(k, 0), b["sold"].get(k, 0)
        if s_a or s_b:
            row(f"{k.lower()} sold", s_a, s_b)
    for k in GOODS:
        if a["bought"].get(k) or b["bought"].get(k):
            row(f"{k.lower()} BOUGHT", a["bought"].get(k, 0), b["bought"].get(k, 0))

    print("\n  MONEY BY DAY")
    for d in range(0, 30, 6):
        if d in a["day_money"] and d in b["day_money"]:
            row(f"day {d}", a["day_money"][d], b["day_money"][d])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="boatlee")
    ap.add_argument("--b", default="champion")
    ap.add_argument("--seeds", default="5,6,7,8")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    a = profile(args.a, seeds, args.b)
    b = profile(args.b, seeds, args.a)
    show(a, b, args.a, args.b)


if __name__ == "__main__":
    main()
