"""Harness v2 — the measurement rules from CLAUDE.md, made the default rather than the option.

Every result is: >= 80 games, both seats, a seed block never used for tuning, and the counters
printed next to the money. Those are not flags to remember; they are what the tool does.

    PYTHONPATH=. python harness/run.py --agents boatlee,executor_v7 --seeds 21000:21040 --games 80
    PYTHONPATH=. python harness/run.py --pool --games 80 --seeds 21000:21040

Three things it refuses to let you do quietly:

* **Report a number without its resolution.** Winrates carry a Wilson interval and the table marks
  which ones clear 50%. Five promotions in a row were wrong because a 3-8pp edge was read off a
  sample that could only resolve 12pp (E18/D19).
* **Reuse a tuning seed block.** `--seeds` is explicit and recorded in every result row, so a
  held-out block stays held out by being visibly different, not by being remembered.
* **Read money without effect counters.** A change that never fired and a strategy that does not
  work produce identical money (E44), so `blocked_ops`, `plants_lost`, `fertilize_hits` and the
  rest print with every result.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from arena.stats import Winrate, games_needed
from harness.counters import Observer, mean_counters

RESULTS = "results/games.jsonl"
DEFAULT_CONFIG = {"episodeSteps": 720}


@dataclass(frozen=True)
class Match:
    a: str
    b: str
    seed: int
    a_first: bool


@dataclass
class GameResult:
    match: Match
    money_a: float
    money_b: float
    counters_a: dict
    counters_b: dict
    seconds: float


_WORKER: dict = {}


def _init_worker(names: list[str], config: dict) -> None:
    from harness import registry

    _WORKER["pool"] = {n: registry.get(n) for n in names}
    _WORKER["config"] = config


def _agent(name: str):
    """The pool entry for `name`, resolved lazily and memoised in the worker.

    Lazy because S2 reuses one worker pool across generations and every candidate is a genome the
    pool had never heard of when it was forked (`registry.VEC_PREFIX`). Memoised because resolving
    is where the compiler's source is hashed, which is not free at 2,300 games a generation — but
    the *agent* is still rebuilt per game, which is the part that must not be shared (state leaks).
    """
    pool = _WORKER["pool"]
    if name not in pool:
        from harness import registry

        pool[name] = registry.get(name)
    return pool[name]


def _play(m: Match) -> GameResult:
    """One game. Agents are rebuilt here, inside the worker, so no state crosses games."""
    import kagsim

    cfg = _WORKER["config"]
    first, second = (m.a, m.b) if m.a_first else (m.b, m.a)
    agents = [_agent(first).build(), _agent(second).build()]
    observers = [Observer(player=0), Observer(player=1)]

    t0 = time.perf_counter()
    sim = kagsim.Sim({**cfg, "seed": m.seed})
    sim.collect_stats = True
    # 719 turns for episodeSteps=720: the framework calls each agent `episodeSteps - 1` times, and
    # the dropped turn is the terminal one where liquidation happens (arena/run.py has the full
    # note — getting this wrong optimised a season one turn short).
    for _ in range(cfg["episodeSteps"] - 1):
        actions = []
        for p in range(2):
            obs = sim.observation(p)
            action = agents[p](obs)
            observers[p].observe(obs, action)
            actions.append(action)
        sim.step(actions)
    seconds = time.perf_counter() - t0

    counters = [observers[p].finish(sim, agents[p]) for p in range(2)]
    money = [sim.money(0), sim.money(1)]
    if m.a_first:
        return GameResult(m, money[0], money[1], counters[0], counters[1], seconds)
    return GameResult(m, money[1], money[0], counters[1], counters[0], seconds)


def build_schedule(names: list[str], seeds: list[int], both_seats: bool,
                   mirror: bool = False) -> list[Match]:
    """All pairings. `mirror` adds each agent against a fresh copy of itself.

    A mirror is not a filler match: against a deterministic script it is the sharpest health check
    the harness has. Boatlee vs Boatlee ties exactly on most seeds, and the seeds where it does not
    are seat-order effects in the shared market — so a mirror that stops tying means something in
    the pipeline has become non-deterministic.
    """
    pairs = list(itertools.combinations(names, 2))
    if mirror:
        pairs += [(n, n) for n in names]
    out: list[Match] = []
    for a, b in pairs:
        for s in seeds:
            out.append(Match(a, b, s, True))
            if both_seats and a != b:
                out.append(Match(a, b, s, False))   # mirrored seating cancels seat advantage
    return out


def parse_seeds(spec: str) -> list[int]:
    """`21000:21040` -> the block [21000, 21040). A block, not a count: it is quoted in results."""
    if ":" not in spec:
        raise SystemExit("--seeds takes a block like 21000:21040")
    lo, hi = spec.split(":", 1)
    seeds = list(range(int(lo), int(hi)))
    if not seeds:
        raise SystemExit(f"empty seed block {spec}")
    return seeds


def run(names: list[str], seeds: list[int], games: int, both_seats: bool, config: dict,
        workers: int | None = None, jsonl: str | None = RESULTS,
        mirror: bool = False, schedule: list[Match] | None = None,
        quiet: bool = False) -> list[GameResult]:
    """`schedule` overrides the all-pairs default — S1 scores one candidate against a pool and has
    no use for the opponents' games against each other. `quiet` suppresses the timing line, which
    is information at the command line and noise at 100 evaluations a generation."""
    if schedule is None:
        per_seed = 2 if both_seats else 1
        need = -(-games // per_seed)                       # ceil
        if need > len(seeds):
            raise SystemExit(
                f"--games {games} over {per_seed} seat(s)/seed needs {need} seeds, block has "
                f"{len(seeds)}. Widen --seeds rather than reusing one."
            )
        seeds = seeds[:need]
        schedule = build_schedule(names, seeds, both_seats, mirror=mirror)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    # Resolve every agent in the parent first. An exception inside a Pool initializer does not
    # propagate — the worker dies, the pool respawns it, and `map` waits forever. A typo in an
    # agent name is then indistinguishable from a slow run, which has already cost one debugging
    # detour. Failing here turns that into a KeyError with the pool listed.
    fingerprints = _resolve(names)

    t0 = time.perf_counter()
    if workers == 1:
        _init_worker(names, config)
        results = [_play(m) for m in schedule]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(names, config)) as pool:
            results = pool.map(_play, schedule)
    elapsed = time.perf_counter() - t0

    if jsonl:
        _append(jsonl, fingerprints, seeds, config, results)
    if not quiet:
        print(f"{len(schedule)} games, {len(names)} agents, {workers} workers, {elapsed:.1f}s "
              f"({elapsed / max(1, len(schedule)):.2f}s/game)\n")
    return results


def _resolve(names: list[str]) -> dict[str, str]:
    """Build every agent once in the parent: validates the names and yields the fingerprints."""
    from harness import registry

    out = {}
    for n in names:
        agent = registry.get(n)
        agent.build()               # constructing is where a broken agent actually fails
        out[n] = agent.fingerprint
    return out


def _append(path: str, fps: dict[str, str], seeds: list[int], config: dict,
            results: list[GameResult]) -> None:
    """Append every game. Rows carry the agent fingerprints and the seed block, so two runs can
    never be pooled by name alone — the mistake `arena/registry.py` guards for the arena."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    block = f"{seeds[0]}:{seeds[-1] + 1}"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a") as fh:
        for r in results:
            fh.write(json.dumps({
                "ts": stamp,
                "seed_block": block,
                "config": config,
                "a": r.match.a, "b": r.match.b,
                "a_fingerprint": fps[r.match.a], "b_fingerprint": fps[r.match.b],
                "seed": r.match.seed, "a_first": r.match.a_first,
                "money_a": r.money_a, "money_b": r.money_b,
                "counters_a": r.counters_a, "counters_b": r.counters_b,
            }) + "\n")


# ----------------------------------------------------------------------- reporting

COUNTER_COLUMNS = [
    ("steps_per_useful", "{:.2f}", "steps/useful"),
    ("idle_pct", "{:.0%}", "idle"),
    ("blocked_ops", "{:.0f}", "blocked"),
    ("plants_lost_thirst", "{:.1f}", "lost:thirst"),
    ("plants_lost_age", "{:.1f}", "lost:age"),
    ("strawberry_per_plant", "{:.1f}", "straw/plant"),
    ("milk_per_cow_day", "{:.2f}", "milk/cow-d"),
    ("unharvested_ripe_at_end", "{:.0f}", "ripe@end"),
    ("shed_overflow_discarded", "{:.0f}", "overflow"),
    ("fertilize_hits", "{:.0f}", "fert hits"),
]


def report(names: list[str], results: list[GameResult], per_seed: bool = False) -> None:
    wins: dict = defaultdict(float)
    played: dict = defaultdict(int)
    margins: dict = defaultdict(list)
    money: dict = defaultdict(list)
    counters: dict = defaultdict(list)
    pair: dict = defaultdict(lambda: [0.0, 0])

    for r in results:
        a, b = r.match.a, r.match.b
        outcome = 1.0 if r.money_a > r.money_b else 0.0 if r.money_a < r.money_b else 0.5
        wins[a] += outcome
        wins[b] += 1 - outcome
        played[a] += 1
        played[b] += 1
        margins[a].append(r.money_a - r.money_b)
        margins[b].append(r.money_b - r.money_a)
        money[a].append(r.money_a)
        money[b].append(r.money_b)
        counters[a].append(r.counters_a)
        counters[b].append(r.counters_b)
        pair[(a, b)][0] += outcome
        pair[(a, b)][1] += 1

    print(f"{'agent':<16}{'winrate (95% Wilson)':>26}{'mean $':>10}{'margin':>11}")
    print("-" * 63)
    order = sorted(names, key=lambda n: -(wins[n] / played[n] if played[n] else 0))
    for n in order:
        wr = Winrate(wins[n], played[n])
        mean_money = sum(money[n]) / len(money[n]) if money[n] else 0.0
        mean_margin = sum(margins[n]) / len(margins[n]) if margins[n] else 0.0
        print(f"{n:<16}{str(wr):>26}{mean_money:>10,.0f}{mean_margin:>+11,.0f}")

    print(f"\n{'counters (per game, mean)':<16}" +
          "".join(f"{label:>13}" for _, _, label in COUNTER_COLUMNS))
    print("-" * (16 + 13 * len(COUNTER_COLUMNS)))
    for n in order:
        agg = mean_counters(counters[n])
        cells = "".join(f"{fmt.format(agg.get(key, 0)):>13}" for key, fmt, _ in COUNTER_COLUMNS)
        print(f"{n:<16}{cells}")
        effects = agg.get("effect_count") or {}
        if effects:
            print(f"{'  effects':<16}" +
                  "  ".join(f"{k}={v:.1f}" for k, v in sorted(effects.items())))

    if len(names) > 1:
        print("\npairwise (row vs column), * = interval excludes 50%")
        width = max(len(n) for n in names) + 2
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for a in names:
            row = f"{a:<{width}}"
            for b in names:
                if a == b:
                    row += f"{'—':>{width}}"
                    continue
                w, g = pair[(a, b)] if (a, b) in pair else (
                    pair[(b, a)][1] - pair[(b, a)][0], pair[(b, a)][1])
                cell = "—" if not g else (
                    f"{100 * Winrate(w, g).rate:.1f}"
                    + ("*" if Winrate(w, g).wilson()[0] > 0.5 or Winrate(w, g).wilson()[1] < 0.5
                       else "")
                )
                row += f"{cell:>{width}}"
            print(row)

    n_games = min(played.values()) if played else 0
    hw = Winrate(n_games / 2, n_games).half_width if n_games else float("nan")
    print(f"\nresolution: {n_games} games per agent gives +/-{100 * hw:.1f}pp at 50%. "
          f"Separating 52% from 50% needs ~{games_needed(0.02):,} games — do not act on a 52%.")

    if per_seed:
        print("\nper-seed rows")
        for r in sorted(results, key=lambda r: (r.match.a, r.match.b, r.match.seed)):
            seat = "a-first" if r.match.a_first else "b-first"
            print(f"  {r.match.a} vs {r.match.b}  seed {r.match.seed:<6} {seat:<8} "
                  f"{r.money_a:>10,.0f} vs {r.money_b:>10,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=None, help="comma-separated pool names")
    ap.add_argument("--pool", action="store_true", help="use harness.registry.DEFAULT_POOL")
    ap.add_argument("--seeds", default="21000:21040", help="held-out block, e.g. 21000:21040")
    ap.add_argument("--games", type=int, default=80, help="games per pairing (>=80 to act on)")
    ap.add_argument("--one-seat", action="store_true", help="disable mirrored seating (not for results)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--episode-steps", type=int, default=720)
    ap.add_argument("--mirror", action="store_true", help="also play each agent against itself")
    ap.add_argument("--per-seed", action="store_true", help="print every game")
    ap.add_argument("--no-jsonl", action="store_true")
    ap.add_argument("--list", action="store_true", help="print resolvable agent names and exit")
    args = ap.parse_args()

    from harness import registry

    if args.list:
        print("\n".join(registry.names()))
        print("\npending (need the compiler):")
        for name, why in registry.PENDING.items():
            print(f"  {name:<16} {why}")
        return

    if args.pool or not args.agents:
        names = list(registry.DEFAULT_POOL)
    else:
        names = [n.strip() for n in args.agents.split(",") if n.strip()]
    if len(names) < 2 and not args.mirror:
        raise SystemExit("need at least two agents (or one with --mirror)")

    both = not args.one_seat
    if args.games < 80:
        print(f"warning: {args.games} games is below the 80-game bar; treat this as a smoke run\n")
    if not both:
        print("warning: single-seat run — seat advantage is not cancelled\n")

    config = {"episodeSteps": args.episode_steps}
    results = run(names, parse_seeds(args.seeds), args.games, both, config,
                  workers=args.workers, jsonl=None if args.no_jsonl else RESULTS,
                  mirror=args.mirror)
    report(names, results, per_seed=args.per_seed)


if __name__ == "__main__":
    main()
