"""Cross-entropy method over the engine's `Params`, evaluated against the reigning champion.

Two design points that follow directly from measurement:

* **Fitness is the money margin against the champion, not money.** Scoring is relative, and E6
  showed the arena winner earning *less* money than configs it beats 100% of the time. Margin is
  the continuous surrogate for the actual win condition; winrate is reported alongside but is too
  coarse to steer on once candidates start winning every game.
* **The opponent is a *pool*, not one agent.** A champion tuned against a single opponent is a
  single-opponent optimum: T2.2 found the CEM champion losing **0/80** to a naive "sell everything
  immediately" dumper, because it had only ever faced opponents that also held inventory. Fitness
  is the *equal-weighted* mean margin across the pool, so crushing one member cannot hide losing
  to another.
* **Never evaluate against `starter`.** E5 measured the ranking inverting under real competition.
  The champion is named explicitly (`--champion`, default the arena's current best) rather than
  taken from `Params()`: the dataclass defaults are wheat-based, so an earlier run silently tuned
  against a weak opponent and reported a +$34k improvement that was worth ~$200 in the arena.

Common random numbers: every candidate in a generation plays the *same* seed list in both seat
assignments, so differences between candidates are not confounded with seed luck.

**Selection is validated on held-out seeds.** The first run of this search reported +$34,013 and a
100% winrate, then placed *third* in the arena on fresh seeds — behind the champion it had
supposedly beaten. With 16 games per candidate the Wilson half-width is ~22pp, so CEM was
selecting seed luck. Each generation now re-scores the updated mean on a fresh seed set and keeps
the best *validated* mean, not the last one.

**Do not shorten `--episode-steps` to save time.** The strong strategies are melon-based and run
cash-negative until their first harvest around day 10-12; on a truncated episode a do-nothing
config outscores them by ~$1,500, so the fitness inverts for precisely the configs worth finding.

**Do not blanket-pin `reserve_frac` to zero any more.** That ablation was correct when it was run
(E11) and is now wrong in scope: wool became floodable once the champion kept sheep, and a reserve
on it beats no reserve (E19). Pin `forecast_weight` if you want the market-timing ablation, but
leave the per-product reserves free.

Usage:
    PYTHONPATH=. python search/cem.py --gens 8 --pop 16 --seeds 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict

from agent.params import Params
from search.space import (DIM, apply_pins, bounds, clip_vector, decode, describe,
                          encode, parse_pins)

_W: dict = {}


def _init(pool: list[dict], config: dict):
    # Entries are either a Params dict or {"external": "<registry name>"}. The latter lets the
    # search train against an agent nobody here wrote -- until now every opponent CEM ever faced
    # was one of my own, which is D16 applied to the search itself rather than just the arena.
    _W["pool"] = pool
    _W["config"] = config


def _build_opponent(spec: dict):
    if "external" in spec:
        from arena.registry import REGISTRY

        return REGISTRY[spec["external"]].build()
    return None


def _play(job):
    """One episode. `job` = (candidate params dict, seed, candidate_first, opponent index)."""
    import kagsim
    from agent import make_agent

    cand_params, seed, cand_first, opp_ix = job
    cand = make_agent(Params(**cand_params))
    spec = _W["pool"][opp_ix]
    opp = _build_opponent(spec) or make_agent(Params(**spec))
    agents = [cand, opp] if cand_first else [opp, cand]

    cfg = {**_W["config"], "seed": seed}
    sim = kagsim.Sim(cfg)
    # 719 agent turns for episodeSteps=720, not 718. Same off-by-one that was fixed in
    # arena/run.py and sim/runner.py (E27); the dropped turn is the terminal one, where
    # end-of-season liquidation happens, so every CEM score before this was fitted to a season one
    # turn shorter than the one we are scored on.
    for _ in range(cfg["episodeSteps"] - 1):
        sim.step([agents[p](sim.observation(p)) for p in range(2)])
    ci = 0 if cand_first else 1
    return sim.money(ci) - sim.money(1 - ci)


def evaluate(proc_pool, candidates: list[list[float]], seeds: list[int],
             n_opponents: int = 1) -> list[dict]:
    """-> per-candidate {margin, winrate, per_opponent}.

    Common random numbers: every candidate meets every opponent on the same seeds in both seats.
    The headline margin is the mean *of the per-opponent means*, so a pool member that is easy to
    beat cannot outvote one that is hard.
    """
    jobs, spans = [], []
    for vec in candidates:
        params = asdict(decode(vec))
        start = len(jobs)
        for o in range(n_opponents):
            for s in seeds:
                jobs.append((params, s, True, o))
                jobs.append((params, s, False, o))
        spans.append((start, len(jobs)))

    margins = (proc_pool.map(_play, jobs, chunksize=1) if proc_pool
               else [_play(j) for j in jobs])

    per_opp_games = 2 * len(seeds)
    out = []
    for a, b in spans:
        chunk = margins[a:b]
        by_opp = [chunk[i * per_opp_games:(i + 1) * per_opp_games] for i in range(n_opponents)]
        means = [sum(c) / len(c) for c in by_opp]
        rates = [(sum(1 for m in c if m > 0) + 0.5 * sum(1 for m in c if m == 0)) / len(c)
                 for c in by_opp]
        out.append({"margin": sum(means) / len(means),
                    "winrate": sum(rates) / len(rates),
                    "per_opponent": list(zip(means, rates))})
    return out


def run_cem(gens: int, pop: int, n_seeds: int, elite_frac: float, episode_steps: int,
            seed: int, workers: int | None, out_path: str,
            opponents: list[Params | dict] | None = None, start: Params | None = None,
            pins: dict[str, float] | None = None, sigma_frac: float = 0.25) -> Params:
    rng = random.Random(seed)
    lo, hi = bounds()

    opponents = opponents or [Params()]
    pins = pins or {}
    engine_opps = [o for o in opponents if isinstance(o, Params)]
    mu = apply_pins(encode(start or (engine_opps[0] if engine_opps else Params())), pins)
    # Initial spread, as a fraction of each knob's range.
    #
    # 0.25 is a *global* search setting and it is actively wrong when `start` is a specific
    # combination worth keeping. CEM samples every dimension independently, so a spread of 25% of
    # range across 42 knobs scatters generation 0 far from the starting point in all directions at
    # once -- destroying exactly the conjunction it was seeded with. Observed directly: seeded from
    # `mimic`, held-out winrate went 20.8% -> 4.9% -> 0.0% over three generations, walking away
    # from the start rather than refining it.
    #
    # For refinement pass something small (~0.05). CEM cannot *discover* a conjunction, but with a
    # tight sigma it is a serviceable local hill-climber around one it is given.
    sigma = [(h - l) * sigma_frac for l, h in zip(lo, hi)]
    n_elite = max(2, int(pop * elite_frac))
    history = []
    # Held-out seeds, fixed for the whole run and disjoint from the training draws, so the
    # validation score of one generation is comparable with another's.
    val_rng = random.Random(seed + 99_991)
    val_seeds = [val_rng.randrange(500_000, 1_000_000) for _ in range(max(16, n_seeds * 2))]
    best_val, best_mu = float("-inf"), list(mu)

    ctx = mp.get_context("spawn")
    workers = workers or os.cpu_count() or 1
    proc = ctx.Pool(workers, initializer=_init,
                    initargs=([asdict(o) if isinstance(o, Params) else o for o in opponents],
                              {"episodeSteps": episode_steps}))
    n_opp = len(opponents)

    try:
        for g in range(gens):
            t0 = time.perf_counter()
            seeds = [rng.randrange(10_000) for _ in range(n_seeds)]   # shared by all candidates

            cands = [apply_pins(clip_vector([rng.gauss(m, s) for m, s in zip(mu, sigma)]), pins)
                     for _ in range(pop - 1)]
            cands.append(list(mu))          # always re-test the current mean

            scores = evaluate(proc, cands, seeds, n_opp)
            ranked = sorted(zip(scores, cands), key=lambda t: -t[0]["margin"])
            elite = [v for _, v in ranked[:n_elite]]

            mu = apply_pins([sum(v[i] for v in elite) / n_elite for i in range(DIM)], pins)
            sigma = [
                max((sum((v[i] - mu[i]) ** 2 for v in elite) / n_elite) ** 0.5,
                    (hi[i] - lo[i]) * 0.02)                 # floor keeps it from collapsing early
                for i in range(DIM)
            ]

            # Validate the *updated* mean on held-out seeds. Training score is not trustworthy:
            # elites are partly selected for doing well on this generation's draw.
            val = evaluate(proc, [clip_vector(mu)], val_seeds, n_opp)[0]
            if val["margin"] > best_val:
                best_val, best_mu = val["margin"], list(mu)

            best = ranked[0][0]
            history.append({"gen": g, "train_best": best["margin"],
                            "train_winrate": best["winrate"],
                            "val_margin": val["margin"], "val_winrate": val["winrate"]})
            flag = "  <- best" if val["margin"] == best_val else ""
            per = " ".join(f"{100 * r:.0f}%" for _, r in val["per_opponent"])
            print(f"gen {g:>2}  train ${best['margin']:>+9,.0f}  "
                  f"| held-out ${val['margin']:>+9,.0f} (avg {100 * val['winrate']:>5.1f}%, "
                  f"per-opp {per})  ({time.perf_counter() - t0:.0f}s){flag}", flush=True)

        mu = best_mu
        # The base supplies fields the search does not cover, so it must be a Params -- and with an
        # external agent in the pool, `opponents[0]` may not be one.
        final = decode(clip_vector(mu), engine_opps[0] if engine_opps else Params())
        print(f"\nkeeping the best *validated* mean: held-out margin ${best_val:+,.0f}")
    finally:
        proc.close()
        proc.join()

    with open(out_path, "w") as f:
        json.dump({"params": asdict(final), "vector": mu, "history": history}, f, indent=2)
    print(f"\nbest params -> {out_path}")
    print("differs from defaults by:", describe(mu))
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=16, help="episodes per candidate = 2x this")
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--episode-steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="initial spread as a fraction of each knob's range. 0.25 searches "
                         "globally; use ~0.05 to refine around --start instead of scattering off "
                         "it (E46).")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="search/best_params.json")
    ap.add_argument("--pin", default="",
                    help='freeze knobs out of the search, e.g. "forecast_weight=0,reserve_frac.*=0"')
    ap.add_argument("--champions", default="champion",
                    help="comma-separated registry agents to beat; a pool, never `starter`")
    ap.add_argument("--start", help="registry agent to start the search FROM (default: the first "
                    "engine opponent). Use this to search around a different shape of farm rather "
                    "than perturbing a champion that is already a tuned local optimum -- every "
                    "single-knob change to it has lost money (E39-E45).")
    args = ap.parse_args()

    from arena.registry import REGISTRY

    names = args.champions.split(",")
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown champion(s) {unknown}; have {sorted(REGISTRY)}")
    # External agents cannot be expressed as Params; pass them through as a registry reference so
    # the worker builds them from the registry instead. Training against `boatlee` is the point of
    # this run: every CEM search before it optimised against opponents written by the same author
    # as the candidate, which is D16 applied to the search itself.
    pool = [{"external": n} if REGISTRY[n].kind == "external" else Params(**REGISTRY[n].params)
            for n in names]
    engine_pool = [o for o in pool if isinstance(o, Params)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"CEM over {DIM} knobs: pop={args.pop}, elite={int(args.pop * args.elite_frac)}, "
          f"{2 * args.seeds} games/candidate, {args.gens} generations")
    pins = parse_pins(args.pin)
    if pins:
        print(f"pinned out of the search: {sorted(pins)}")
    print(f"opponent pool = {names}  ({2 * args.seeds * len(pool)} games/candidate)")
    print("fitness = equal-weighted mean margin across the pool\n")
    if args.start:
        start = Params(**REGISTRY[args.start].params)
        print(f"starting the search from {args.start!r}, not from the incumbent")
    else:
        start = engine_pool[0] if engine_pool else Params(**REGISTRY["champion"].params)
    run_cem(args.gens, args.pop, args.seeds, args.elite_frac, args.episode_steps,
            args.seed, args.workers, args.out, opponents=pool, start=start, pins=pins,
            sigma_frac=args.sigma)


if __name__ == "__main__":
    main()
