"""Champion promotion gate.

Five promotions in a row installed a champion that a later, larger measurement showed was not the
best available. The cause was always the same: promoting on a difference smaller than the noise of
the sample that produced it.

| games | ± at 50% | smallest edge resolvable |
|---|---|---|
| 24 | 18.6pp | 68.6% |
| 64 | 11.9pp | 61.9% |
| 300 | 5.6pp | 55.6% |
| 1000 | 3.1pp | 53.1% |

Champions were promoted on 24–64 games — able to resolve only 12pp+ edges — from differences of
3–8pp. CEM's own held-out score has the same problem *and* a second one: it only ever sees the pool
it trained against.

This script is the gate. It refuses to promote unless every check passes, and it escalates sample
size only where a result is close, so the cost stays bounded.

    STAGE 1  beat the incumbent, interval excluding 50%
    STAGE 2  no *new* losses — do not lose to anything the incumbent beats
    STAGE 3  survive a neighbourhood sweep — no small perturbation beats it

Stage 2 used to demand a clean sweep of the registry. `make audit-champion` showed the **sitting
champion fails that** — it loses to `x-dumper` (22.2%), `r-melon` (41.0%), `herd5c8s` (28.0%) and
`boatlee` (0%) over 500 games each. A bar the incumbent cannot clear is not a bar; it just blocks
every candidate regardless of merit. The question a promotion actually has to answer is whether the
candidate is *worse than what we have*, so each loss is now re-checked against the incumbent and
only counts if the incumbent beats that opponent. External agents (`boatlee`) are reported as a
reference and never gate: they are the target we are chasing (D21), and progress toward a target
has to be measurable before you reach it.

Usage:
    PYTHONPATH=. python tools/promote.py search/best_params.json
    PYTHONPATH=. python tools/promote.py --candidate herd6c8s --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from arena.registry import REGISTRY, AgentSpec
from arena.run import pair_money, run, tabulate
from arena.stats import Winrate, games_needed

CHAMPION_PATH = "search/champion.json"
SCREEN_GAMES = 12          # 24 episodes: resolves >=69% quickly and cheaply
CONFIRM_GAMES = 250        # 500 episodes: resolves >=54.4%
NOISE_BAND = (0.35, 0.65)  # screen results in here are unresolved and get escalated

# Discrete knobs whose neighbours are worth probing. The herd mix is here because a hand sweep
# found 6 cows + 8 sheep beating CEM's 8 + 4 by 64.4% — CEM's optimum was simply not local.
# Deltas are sized to the knob, not copied between knobs. E52 measured `release_pressure`'s cliff
# **10 units** below its setting (83.8% at 65, 12.5% at 60), so the +-1/+-2 spacing that suits a herd
# count would have probed 68-72, found nothing, and passed cleanly -- a more convincing vacuum than
# the one E53 already caught. A neighbourhood sweep only means anything if its radius can reach the
# edge it is looking for.
NEIGHBOURHOOD = {
    "release_pressure": (-12, -6, 5, 10),
    "release_batch": (-4, -2, 4, 12),
    "cow_target": (-2, -1, 1, 2),
    "sheep_target": (-2, -1, 1, 2),
    "goose_target": (2, 4),
    "hire_max": (-1, 1),
}


def _neighbour_spec(base: str, params: dict, note: str) -> AgentSpec:
    """A perturbed copy of the candidate, preserving its `kind` so relay agents stay relay agents."""
    return AgentSpec(kind=REGISTRY[base].kind, params=params, note=note)


def _spec(params: dict, note: str) -> AgentSpec:
    return AgentSpec("engine", params, note)


def _duel(a: str, b: str, games: int, seed0: int, config: dict) -> tuple[Winrate, float, float]:
    _, results = run([a, b], list(range(seed0, seed0 + games)), config)
    pairwise, _, _ = tabulate(results)
    ours, theirs = pair_money(results, a, b)
    return pairwise[(a, b)], ours, theirs


def _resolved(wr: Winrate) -> str:
    lo, hi = wr.wilson()
    if lo > 0.5:
        return "win"
    if hi < 0.5:
        return "loss"
    return "unresolved"


def stage1(cand: str, champ: str, seed0: int, config: dict) -> bool:
    print(f"\nSTAGE 1 — {cand} vs incumbent {champ}")
    wr, ours, theirs = _duel(cand, champ, CONFIRM_GAMES, seed0, config)
    verdict = _resolved(wr)
    print(f"  {wr}   ${ours:,.0f} vs ${theirs:,.0f}   -> {verdict}")
    if verdict != "win":
        print(f"  REFUSED: a candidate must beat the incumbent with the interval clear of 50%.")
        if verdict == "unresolved":
            edge = abs(wr.rate - 0.5)
            need = games_needed(edge) if edge > 0.005 else 999_999
            print(f"  (edge {100 * wr.rate:.1f}% would need ~{need:,} games to resolve)")
        return False
    return True


def stage2(cand: str, seed0: int, config: dict) -> bool:
    """Beat the field. External agents are a *reference*, not a rung.

    `boatlee` beats every agent in this repo 100%, so requiring a candidate to beat it makes the
    gate unsatisfiable -- the sitting champion cannot pass it either. It is the target we are
    chasing (D21), and progress toward it has to be measurable before we arrive. Its result is
    reported on every run and never causes a refusal; the criterion is that a candidate must not
    lose to anything *we* built.
    """
    others = [n for n in REGISTRY if n != cand]
    reference = {n for n in others if REGISTRY[n].kind == "external"}
    print(f"\nSTAGE 2 — gauntlet vs {len(others)} registered agents "
          f"(screen {2 * SCREEN_GAMES} games, escalate close calls to {2 * CONFIRM_GAMES})")
    _, results = run([cand] + others, list(range(seed0, seed0 + SCREEN_GAMES)), config,
                     gauntlet=cand)
    pairwise, _, _ = tabulate(results)

    close = [n for n in others
             if NOISE_BAND[0] <= pairwise[(cand, n)].rate <= NOISE_BAND[1]
             or pairwise[(cand, n)].rate < NOISE_BAND[0]]
    clear_losses = [n for n in others if _resolved(pairwise[(cand, n)]) == "loss"]
    print(f"  screen: {len(others) - len(close)} clear wins, {len(close)} needing confirmation")

    failures = [n for n in clear_losses if n not in reference]
    for n in close:
        wr, ours, theirs = _duel(cand, n, CONFIRM_GAMES, seed0 + 5000, config)
        verdict = _resolved(wr)
        if verdict == "loss" and n not in reference:
            failures.append(n)
        mark = {"win": "", "loss": "  <-- LOSS", "unresolved": "  (tie)"}[verdict]
        if n in reference:
            mark = "  <-- reference (target, not a gate)"
        print(f"    {n:<20}{str(wr):>30}  ${ours:>9,.0f} vs ${theirs:>9,.0f}{mark}")

    if not failures:
        return True

    # A loss only counts if the incumbent does *better* against that same opponent. Otherwise it is
    # the status quo, not a regression introduced by this candidate.
    print(f"\n  checking {len(failures)} loss(es) against the incumbent — a loss it also suffers "
          f"is not a regression")
    regressions = []
    for n in sorted(set(failures)):
        wr, ours, theirs = _duel("champion", n, CONFIRM_GAMES, seed0 + 7000, config)
        inc = _resolved(wr)
        if inc == "loss":
            print(f"    {n:<20} incumbent also loses ({wr})  -> not a regression")
        else:
            print(f"    {n:<20} incumbent {inc}s ({wr})  -> REGRESSION")
            regressions.append(n)
    if regressions:
        print(f"  REFUSED: loses to {regressions}, which the incumbent does not")
        return False
    return True


def stage3(cand_params: dict, cand: str, seed0: int, config: dict) -> bool:
    print("\nSTAGE 3 — neighbourhood sweep (is this optimum even local?)")
    probes: list[str] = []
    for knob, deltas in NEIGHBOURHOOD.items():
        base = cand_params.get(knob)
        if base is None:
            continue
        for d in deltas:
            v = int(base) + d
            if v < 0:
                continue
            name = f"_nb_{knob}{v}"
            REGISTRY[name] = _neighbour_spec(cand, {**cand_params, knob: v},
                                             f"neighbour {knob}={v}")
            probes.append(name)
    if not probes:
        # E53: this used to `return True`, so the gate printed ALL STAGES PASSED after performing
        # no check at all. A stage that cannot run must not report success -- that is the same
        # class of defect as a test that passes while proving nothing (E36).
        print("  NOT APPLICABLE — this candidate exposes no knob in NEIGHBOURHOOD.")
        print("  Stage 3 cannot vouch for it. Sweep the knobs by hand and record the range,")
        print("  or add them to the agent's spec so they can be probed.")
        return None

    _, results = run([cand] + probes, list(range(seed0, seed0 + SCREEN_GAMES)), config,
                     gauntlet=cand)
    pairwise, _, _ = tabulate(results)
    suspects = [n for n in probes if pairwise[(cand, n)].rate <= NOISE_BAND[1]]
    print(f"  screened {len(probes)} neighbours, {len(suspects)} need confirmation")

    beaten_by = []
    for n in suspects:
        wr, ours, theirs = _duel(cand, n, CONFIRM_GAMES, seed0 + 9000, config)
        if _resolved(wr) == "loss":
            beaten_by.append((n, wr, theirs))
        print(f"    {n:<24}{str(wr):>30}  ${ours:>9,.0f} vs ${theirs:>9,.0f}"
              f"{'  <-- BETTER' if _resolved(wr) == 'loss' else ''}")

    if beaten_by:
        best = min(beaten_by, key=lambda t: t[1].rate)
        print(f"  REFUSED: {best[0]} is better. Promote that instead, or re-search around it.")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="?", help="path to a params json (e.g. search/best_params.json)")
    ap.add_argument("--candidate", help="name of a registered agent instead of a file")
    ap.add_argument("--apply", action="store_true", help="write champion.json if every stage passes")
    ap.add_argument("--audit", action="store_true",
                    help="run stages 2-3 on the *incumbent* — does the sitting champion still hold?")
    ap.add_argument("--seed0", type=int, default=int.from_bytes(os.urandom(3), "big") * 1000)
    ap.add_argument("--episode-steps", type=int, default=720)
    args = ap.parse_args()

    config = {"episodeSteps": args.episode_steps}

    if args.audit:
        cand = "champion"
        cand_params = dict(REGISTRY["champion"].params)
        print("AUDIT — does the sitting champion still survive the gauntlet and its neighbourhood?")
        print(f"seeds start at {args.seed0:,}")
        ok = stage2(cand, args.seed0 + 100_000, config)
        ok = stage3(cand_params, cand, args.seed0 + 200_000, config) and ok
        print("\n" + ("CHAMPION HOLDS." if ok else "CHAMPION IS NOT THE BEST AVAILABLE."))
        sys.exit(0 if ok else 1)

    if args.candidate:
        cand, cand_params = args.candidate, dict(REGISTRY[args.candidate].params)
    elif args.params:
        cand_params = json.load(open(args.params))["params"]
        cand = "_candidate"
        REGISTRY[cand] = _spec(cand_params, "promotion candidate")
    else:
        raise SystemExit("give a params file or --candidate NAME")

    champ = "champion"
    print(f"PROMOTION GATE — candidate {cand!r} vs incumbent {champ!r}")
    print(f"seeds start at {args.seed0:,} (fresh; no search has used these)")

    ok = stage1(cand, champ, args.seed0, config)
    ok = ok and stage2(cand, args.seed0 + 100_000, config)
    s3 = stage3(cand_params, cand, args.seed0 + 200_000, config) if ok else False

    print()
    if not ok or s3 is False:
        print("NOT PROMOTED — champion.json unchanged.")
        sys.exit(1)
    if s3 is None:
        print("STAGES 1-2 PASSED; STAGE 3 DID NOT RUN.")
        print("Not a pass. Promote only with a hand-run neighbourhood sweep recorded in "
              "docs/experiments.md.")
        sys.exit(2)
    print("ALL STAGES PASSED.")
    if args.apply and REGISTRY[cand].kind != "engine":
        raise SystemExit(
            f"REFUSING --apply: {cand!r} is a {REGISTRY[cand].kind!r} agent, and {CHAMPION_PATH} is "
            f"read back as engine `Params(**params)` (arena/registry.py `_load`). Writing "
            f"{sorted(cand_params)} there would raise TypeError on the next import and break the "
            f"whole registry. A non-engine champion needs its own pointer file first."
        )
    if args.apply:
        with open(CHAMPION_PATH, "w") as f:
            json.dump({"params": cand_params}, f, indent=2)
        print(f"promoted -> {CHAMPION_PATH}")
    else:
        print("(dry run — pass --apply to write champion.json)")


if __name__ == "__main__":
    main()
