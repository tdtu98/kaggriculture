"""S2: the genome search. Outer GA over the layout, inner CMA-ES over the consts.

    .venv/bin/python -u -m search.run --generations 50 --workers 8
    .venv/bin/python -u -m search.run --generations 50 --resume        # after a kill

**Why two levels.** The genome is two different search problems wearing one vector. The layout
of them are combinatorial and discontinuous — a cohort is in NE or it is not, SE is bought or it is
not, the tenth cow exists or it does not — and the only sane operator on those is a structured
mutation (`search/ga.py`). The other nine are smooth scalars where the fitness surface is
genuinely correlated (a higher `sell_floor_melon` changes what `release_pressure` should be), which
is precisely what a full-covariance CMA-ES is for and precisely what a GA is bad at. Running one
optimiser over both would use the wrong operator on half the space.

**Where the budget goes.** The outer loop is the workhorse: 48 genomes/generation, each 48 games
(4 pool opponents x 6 rotated seeds x 2 seats). Measured 5.2 s/evaluation at 9-10 games/s, so a
bare GA generation is ~4.2 min and 50 of them are ~3.5 h. The inner loop is a polish pass on the
top 4 only, every *other* generation: +16 evaluations/generation, ~5.6 min/generation, ~4.7 h for
50. `--cma-every 1 --cma-top 8` is the thorough setting (~10 h); `--cma-every 0` is the cheapest.
The outer loop gets the budget because at this stage the layout is where the money is (E69: the
residual Boatlee gap is plan-bound).

**Run it detached.** Measured on this machine: a search started as a child of an agent/CI shell is
suspended whenever that shell is idle, and its `t` timestamps show a **9% duty cycle** — 5 s of
evaluation spread over 90 s of wall, which reads as "the search got slow" and is really "the
search was not running". Launch it from a real terminal, or double-fork it into its own session
(`os.setsid`). The `t` field in every log row exists to make that visible: compare it against
`seconds` and the two agreeing means the process is actually getting the CPU.

**Resumability is not a convenience.** A production run is hours long and this project kills
long runs routinely. `--resume` restores the population, the generation index and the RNG state
from `search/state.json`, and replays the log into the evaluation cache, so a resumed run produces
*the same generations* an uninterrupted one would. The smoke test asserts exactly that (kill after
gen 2, resume, compare gen 3).

Nothing here writes to the acceptance block: seeds come from `fitness.search_seeds`, which raises
rather than spend one. `search/accept.py` is the only thing allowed to touch 54000:54080.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from agent.plan import GENE_INDEX, GENES, Plan, encode, migrate, snap
from search import ga
from search.fitness import evaluate, search_seeds, summary
from harness import registry

POP = 48
ELITES = 4
TOURNAMENT = 3
CROSSOVER_RATE = 0.6

#: The inner loop's shape. `popsize` 4 with a budget of 8 is two CMA iterations — not enough to
#: converge, which is the point: this is a polish pass repeated every generation on whatever the
#: outer loop currently likes, not a standalone optimisation.
#:
#: Two iterations is also the *floor*. At `budget == popsize` the loop asks once, tells once and
#: stops, so the updated distribution is never sampled: that is not CMA-ES, it is four random
#: perturbations wearing its name, and it would have shown up in the log as a working inner loop
#: with a disappointing effect size. `--cma-budget` below `2 * CMA_POPSIZE` warns for that reason.
CMA_POPSIZE = 4
CMA_SIGMA = 0.25            # in normalised [0, 1] gene units


def _sub(vec, names):
    return [float(vec[GENE_INDEX[n]]) for n in names]


def _norm(values, names):
    out = []
    for name, v in zip(names, values):
        g = GENES[GENE_INDEX[name]]
        out.append((v - g.lo) / (g.hi - g.lo) if g.hi > g.lo else 0.0)
    return out


def _denorm(values, names):
    out = []
    for name, v in zip(names, values):
        g = GENES[GENE_INDEX[name]]
        out.append(g.lo + v * (g.hi - g.lo))
    return out


def _with(vec, names, values) -> list[float]:
    out = list(vec)
    for name, v in zip(names, values):
        out[GENE_INDEX[name]] = float(v)
    return ga.canonical(out)


class Search:
    def __init__(self, args):
        self.args = args
        self.cache = ga.Cache()
        self.log = args.log
        self.state_path = args.state
        self.rng = random.Random(args.seed)
        self.gen = 0
        self.best: dict | None = None
        self.population: list[list[float]] = []
        self.repairs = 0
        self.mutants = 0
        self.evals = 0

    # ------------------------------------------------------------------ population

    def seed_genomes(self) -> list[list[float]]:
        """The plans the population starts from: `Plan.boatlee_like()`, plus any `--seed-genome`.

        Seeding from hand plans rather than at random is deliberate: `boatlee_like` is the incumbent
        S3 has to beat, a random population would spend ten generations rediscovering "buy
        pastures", and a search that cannot improve on its own seed is a result worth having early.

        Round 2 (S3, after the representation widening) seeds from **two**: the incumbent and the
        harvested candidate, migrated into the current layout. They are far apart in the space — the
        candidate wins 80/80 against the incumbent and then loses 0/80 to Boatlee — so one basin of
        mutants around either alone throws away half of what round 1 paid for. `migrate` is applied
        here so an old 54-gene vector can be pasted straight off the log.
        """
        out = [ga.canonical(encode(Plan.boatlee_like()))]
        for path_or_json in (self.args.seed_genome or []):
            text = (open(path_or_json).read() if os.path.exists(path_or_json) else path_or_json)
            out.append(ga.canonical(migrate(json.loads(text))))
        # A duplicate seed would silently halve the mutant budget of the one it duplicates.
        seen: dict = {}
        for v in out:
            seen.setdefault(tuple(v), v)
        return list(seen.values())

    def seed_population(self) -> None:
        """Every seed genome verbatim, then mutants round-robin so each basin gets equal share."""
        seeds = self.seed_genomes()
        pop = list(seeds)
        while len(pop) < self.args.pop:
            base = seeds[(len(pop) - len(seeds)) % len(seeds)]
            child, _ops, repaired = ga.mutate(base, self.rng)
            self.mutants += 1
            self.repairs += bool(repaired)
            pop.append(child)
        self.population = pop

    # ------------------------------------------------------------------ evaluation

    def score(self, vec, seeds, kind: str) -> dict:
        cached = self.cache.get(vec, seeds)
        if cached is not None:
            ga.log_row(self.log, self.gen, kind, vec, cached,
                       ga.guard_violations(cached.get("counters") or {}), True)
            return cached
        result = evaluate(vec, seeds, workers=self.args.workers, with_solo=self.args.solo,
                          jsonl=None)
        self.evals += 1
        self.cache.put(vec, seeds, result)
        ga.log_row(self.log, self.gen, kind, vec, result,
                   ga.guard_violations(result.get("counters") or {}), False)
        return result

    @staticmethod
    def ranked(scored):
        """(vec, result, score) sorted best first. The *score* is guarded fitness, never fitness:
        C5's bars outrank the money and this is the one place that is enforced."""
        return sorted(scored, key=lambda t: -t[2])

    def evaluate_population(self, seeds) -> list:
        out = []
        for i, vec in enumerate(self.population):
            r = self.score(vec, seeds, "ga")
            viol = ga.guard_violations(r.get("counters") or {})
            out.append((vec, r, ga.score_of(r["fitness"], viol)))
            if self.args.progress:
                flag = " GUARD:" + ",".join(viol) if viol else ""
                print(f"    [{self.gen}] {i + 1}/{len(self.population)} "
                      f"fit {r['fitness']:.4f} win {r['win_rate']:.2f}{flag}", flush=True)
        return out

    # ------------------------------------------------------------------ inner CMA

    def cma_polish(self, ranked, seeds) -> list:
        """CMA-ES over `ga.CMA_GENES` for the top `--cma-top` genomes, in place.

        Each genome gets its own strategy started at its own consts, seeded from (gen, rank) so a
        resumed run reproduces the same samples. A genome only moves if the polished point *beats*
        it on the same seeds — the comparison is paired, which is the only reason 8 evaluations
        can say anything at all.
        """
        try:
            import cma
        except ImportError:                                   # pragma: no cover - env has it
            if self.args.progress:
                print("    cma not installed; skipping inner loop", flush=True)
            return ranked

        names = ga.CMA_GENES
        lo, hi = [0.0] * len(names), [1.0] * len(names)
        out = list(ranked)
        for rank in range(min(self.args.cma_top, len(out))):
            vec, base_result, base_score = out[rank]
            x0 = _norm(_sub(vec, names), names)
            es = cma.CMAEvolutionStrategy(
                x0, CMA_SIGMA,
                {"bounds": [lo, hi], "popsize": CMA_POPSIZE, "verbose": -9,
                 "seed": (self.args.seed * 7919 + self.gen * 97 + rank) % (2 ** 31 - 1)})
            spent = 0
            best = (base_score, vec, base_result)
            while spent < self.args.cma_budget and not es.stop():
                xs = es.ask()
                ys = []
                for x in xs:
                    cand = _with(vec, names, _denorm(x, names))
                    r = self.score(cand, seeds, "cma")
                    viol = ga.guard_violations(r.get("counters") or {})
                    s = ga.score_of(r["fitness"], viol)
                    ys.append(-s)
                    spent += 1
                    if s > best[0]:
                        best = (s, cand, r)
                es.tell(xs, ys)
            if best[1] is not vec:
                out[rank] = (best[1], best[2], best[0])
                if self.args.progress:
                    print(f"    [{self.gen}] cma rank{rank}: {base_score:.4f} -> {best[0]:.4f}",
                          flush=True)
        return self.ranked(out)

    # ------------------------------------------------------------------ breeding

    def tournament(self, ranked) -> list[float]:
        # `ranked` is sorted best-first, so the *smallest* index wins the tournament.
        pick = min(self.rng.sample(range(len(ranked)), min(TOURNAMENT, len(ranked))))
        return ranked[pick][0]

    def breed(self, ranked) -> list[list[float]]:
        elites = [v for v, _r, _s in ranked[:self.args.elites]]
        pop = list(elites)
        while len(pop) < self.args.pop:
            a = self.tournament(ranked)
            if self.rng.random() < CROSSOVER_RATE:
                child = ga.crossover(a, self.tournament(ranked), self.rng)
            else:
                child = a
            child, _ops, repaired = ga.mutate(child, self.rng)
            self.mutants += 1
            self.repairs += bool(repaired)
            pop.append(child)
        return pop

    # ------------------------------------------------------------------ the loop

    def run(self, generations: int) -> None:
        stop = self.gen + generations
        while self.gen < stop:
            t0 = time.perf_counter()
            seeds = search_seeds(self.gen, n=self.args.seeds)
            print(f"[gen {self.gen}] pop {len(self.population)} seeds {seeds[0]}..{seeds[-1]} "
                  f"div {ga.diversity(self.population):.4f} "
                  f"uniq {ga.unique_fraction(self.population):.2f}", flush=True)

            ranked = self.ranked(self.evaluate_population(seeds))
            if self.args.cma_every and self.gen % self.args.cma_every == 0:
                ranked = self.cma_polish(ranked, seeds)

            top_vec, top_res, top_score = ranked[0]
            if self.best is None or top_score > self.best["score"]:
                self.best = {"gen": self.gen, "score": top_score, "fitness": top_res["fitness"],
                             "win_rate": top_res["win_rate"], "vec": list(snap(top_vec))}
            elapsed = time.perf_counter() - t0
            print(f"[gen {self.gen}] best {top_score:.4f} (fit {top_res['fitness']:.4f} "
                  f"win {top_res['win_rate']:.1%} margin {top_res['mean_margin']:+,.0f})  "
                  f"mean {sum(s for _v, _r, s in ranked) / len(ranked):.4f}  "
                  f"cache {self.cache.hit_rate:.2f} ({len(self.cache)})  "
                  f"repair {self.repair_rate:.2f}  {elapsed:.1f}s", flush=True)

            self.population = self.breed(ranked)
            self.gen += 1
            ga.save_state(self.state_path, self.gen, self.population, self.rng, self.best)

        if self.best:
            print("\nbest genome:", json.dumps(self.best["vec"]), flush=True)
            print(registry.get(registry.vec_name(self.best["vec"])).fingerprint, flush=True)

    @property
    def repair_rate(self) -> float:
        return self.repairs / self.mutants if self.mutants else 0.0

    # ------------------------------------------------------------------ resume

    def resume(self) -> None:
        self.gen, self.population, self.rng, self.best = ga.load_state(self.state_path)
        n = self.cache.load_jsonl(self.log, max_gen=self.gen - 1)
        print(f"resumed at generation {self.gen}, {len(self.population)} genomes, "
              f"{n} log rows -> {len(self.cache)} cached evaluations", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="S2 genome search (GA + inner CMA-ES)")
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pop", type=int, default=POP)
    p.add_argument("--elites", type=int, default=ELITES)
    p.add_argument("--seeds", type=int, default=6, help="seeds per generation (x2 seats x pool)")
    # Defaults sized against S1's measured 7.4 games/s and 48 games per evaluation: the GA alone
    # is 48 evals = 5.2 min/generation (4.3 h for 50 generations, the spec's budget), and this
    # inner loop adds 16 evals/generation on average -> ~6.9 min/generation, ~5.8 h for 50.
    # `--cma-every 0` buys back the spec's number exactly if the wall clock is binding.
    p.add_argument("--cma-top", type=int, default=4)
    p.add_argument("--cma-budget", type=int, default=8, help="evaluations per polished genome")
    p.add_argument("--cma-every", type=int, default=2, help="0 disables the inner loop")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=20250816, help="RNG seed for the search itself")
    p.add_argument("--solo", action="store_true", help="also play starter (calibration, not fitness)")
    p.add_argument("--progress", action="store_true", help="one line per evaluation")
    p.add_argument("--log", default=ga.LOG)
    p.add_argument("--state", default=ga.STATE)
    p.add_argument("--seed-genome", action="append", default=None,
                   help="extra seed genome: a path to a JSON list, or the JSON itself. Repeatable. "
                        "An older gene layout is migrated. Round 2 wants the harvested candidate "
                        "here alongside the default boatlee_like seed.")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cma_every and args.cma_budget < 2 * CMA_POPSIZE:
        print(f"warning: --cma-budget {args.cma_budget} is below {2 * CMA_POPSIZE} "
              f"(2 x popsize), so CMA-ES never samples its updated distribution — the inner loop "
              f"degenerates to {args.cma_budget} random perturbations per genome.", flush=True)
    s = Search(args)
    if args.resume:
        s.resume()
    else:
        s.seed_population()
        ga.save_state(args.state, 0, s.population, s.rng, None)
    s.run(args.generations)


if __name__ == "__main__":
    main()
