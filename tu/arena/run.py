"""Round-robin arena — the only evaluation that exists in a simulation competition.

Every pairing is played on the same seed list in **both seat assignments**, so any first-player
advantage cancels exactly instead of being averaged over. Results are keyed by an agent's spec
fingerprint, so a table can never silently mix two different configs under one name.

Usage:
    PYTHONPATH=. python arena/run.py --agents starter,melon,melon-wheat --games 64
    PYTHONPATH=. python arena/run.py --all --games 128 --html arena/report.html
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass

from arena.registry import REGISTRY, AgentSpec, resolve
from arena.stats import Winrate, games_needed

DB_PATH = "arena/results.sqlite"


@dataclass(frozen=True)
class Match:
    a: str
    b: str
    seed: int
    a_first: bool


_WORKER: dict = {}


def _init_worker(specs: dict[str, dict], config: dict):
    from arena.registry import AgentSpec as Spec

    _WORKER["specs"] = {k: Spec(**v) for k, v in specs.items()}
    _WORKER["config"] = config
    _WORKER["cache"] = {}


def _play(m: Match) -> tuple[Match, float, float, float]:
    """Return (match, money_a, money_b, seconds). Agents are rebuilt per match: the engine is
    stateful within an episode, so reusing an instance across games would leak state."""
    import kagsim

    specs, cfg = _WORKER["specs"], _WORKER["config"]
    first, second = (m.a, m.b) if m.a_first else (m.b, m.a)
    agents = [specs[first].build(), specs[second].build()]

    t0 = time.perf_counter()
    sim = kagsim.Sim({**cfg, "seed": m.seed})
    # 719 agent turns for episodeSteps=720, not 718. The reference framework calls each agent
    # `episodeSteps - 1` times (measured: `env.run` yields exactly 719 observations per seat), and
    # kagsim is bit-identical to it at that count against a third-party agent -- off by one it is
    # not. The dropped turn is the terminal one, where end-of-season liquidation happens, so the
    # whole arena and every CEM score were optimising a season one turn shorter than the real one.
    steps = cfg["episodeSteps"] - 1
    for _ in range(steps):
        sim.step([agents[p](sim.observation(p)) for p in range(2)])
    dt = time.perf_counter() - t0

    money = [sim.money(0), sim.money(1)]
    money_a, money_b = (money[0], money[1]) if m.a_first else (money[1], money[0])
    return m, money_a, money_b, dt


def build_gauntlet(hero: str, names: list[str], seeds: list[int]) -> list[Match]:
    """One agent against every other. N-1 pairings instead of N(N-1)/2.

    The right shape for "has the champion been tested against everything we have?" — a full
    round-robin over 52 agents is 1,326 pairings and mostly answers questions about how the
    *losers* rank against each other.
    """
    out: list[Match] = []
    for other in names:
        if other == hero:
            continue
        for s in seeds:
            out.append(Match(hero, other, s, True))
            out.append(Match(hero, other, s, False))
    return out


def build_schedule(names: list[str], seeds: list[int]) -> list[Match]:
    out: list[Match] = []
    for a, b in itertools.combinations(names, 2):
        for s in seeds:
            out.append(Match(a, b, s, True))
            out.append(Match(a, b, s, False))   # mirrored seating cancels seat advantage
    return out


def run(names: list[str], seeds: list[int], config: dict, workers: int | None = None,
        gauntlet: str | None = None):
    specs = resolve(names)
    schedule = (build_gauntlet(gauntlet, names, seeds) if gauntlet
                else build_schedule(names, seeds))
    workers = workers or os.cpu_count() or 1

    payload = {k: {"kind": v.kind, "params": v.params, "note": v.note} for k, v in specs.items()}
    if workers == 1:
        _init_worker(payload, config)
        results = [_play(m) for m in schedule]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(payload, config)) as pool:
            results = pool.map(_play, schedule, chunksize=1)
    return specs, results


def pair_money(results, a: str, b: str) -> tuple[float, float]:
    """Mean money *within one pairing*. The gauntlet's headline money column must not use the
    agent's overall mean — that averages a blowout against `random` into a knife-edge matchup."""
    xs, ys = [], []
    for m, ma, mb, _ in results:
        if {m.a, m.b} != {a, b}:
            continue
        x, y = (ma, mb) if m.a == a else (mb, ma)
        xs.append(x)
        ys.append(y)
    n = max(len(xs), 1)
    return sum(xs) / n, sum(ys) / n


def tabulate(results) -> tuple[dict, dict, dict]:
    """-> (pairwise winrates, mean money, per-agent aggregate winrate)."""
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    money: dict[str, list[float]] = defaultdict(list)

    for m, ma, mb, _ in results:
        money[m.a].append(ma)
        money[m.b].append(mb)
        score = 1.0 if ma > mb else 0.0 if ma < mb else 0.5
        wins[(m.a, m.b)] += score
        games[(m.a, m.b)] += 1
        wins[(m.b, m.a)] += 1.0 - score
        games[(m.b, m.a)] += 1

    pairwise = {k: Winrate(wins[k], games[k]) for k in games}
    mean_money = {k: sum(v) / len(v) for k, v in money.items()}

    overall: dict[str, Winrate] = {}
    for name in mean_money:
        w = sum(wins[(name, o)] for o in mean_money if o != name)
        g = sum(games[(name, o)] for o in mean_money if o != name)
        overall[name] = Winrate(w, g)
    return pairwise, mean_money, overall


def openskill_ratings(names: list[str], results) -> dict[str, float]:
    """OpenSkill ordinals, which is roughly how Kaggle's own leaderboard behaves."""
    from openskill.models import PlackettLuce

    model = PlackettLuce()
    r = {n: model.rating(name=n) for n in names}
    for m, ma, mb, _ in results:
        if ma == mb:
            teams, ranks = [[r[m.a]], [r[m.b]]], [1, 1]
        else:
            hi, lo = (m.a, m.b) if ma > mb else (m.b, m.a)
            teams, ranks = [[r[hi]], [r[lo]]], [1, 2]
        rated = model.rate(teams, ranks=ranks)
        for team in rated:
            for player in team:
                r[player.name] = player
    return {n: r[n].ordinal() for n in names}


def persist(db_path: str, specs: dict[str, AgentSpec], results, config: dict) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS matches(
        ts REAL, a TEXT, b TEXT, a_fp TEXT, b_fp TEXT, seed INTEGER, a_first INTEGER,
        money_a REAL, money_b REAL, seconds REAL, config TEXT)""")
    now = time.time()
    con.executemany(
        "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(now, m.a, m.b, specs[m.a].fingerprint, specs[m.b].fingerprint, m.seed,
          int(m.a_first), ma, mb, dt, repr(config)) for m, ma, mb, dt in results],
    )
    con.commit()
    con.close()


def report(names, pairwise, mean_money, overall, ratings, seeds, elapsed) -> str:
    order = sorted(names, key=lambda n: -ratings[n])
    w = max(len(n) for n in names) + 1
    lines = [
        f"\n{len(seeds)} seeds x 2 seatings = {2 * len(seeds)} games per pairing "
        f"({len(names)} agents, {elapsed:.1f}s)\n",
        f"{'agent':<{w}}{'skill':>7}{'winrate (95% Wilson)':>28}{'mean $':>12}",
        "-" * (w + 47),
    ]
    for n in order:
        lines.append(f"{n:<{w}}{ratings[n]:>7.2f}{str(overall[n]):>28}{mean_money[n]:>12,.0f}")

    lines += ["", "pairwise winrate (row vs column), * = interval excludes 50%", ""]
    lines.append(" " * w + "".join(f"{n[:9]:>11}" for n in order))
    for a in order:
        row = f"{a:<{w}}"
        for b in order:
            if a == b:
                row += f"{'—':>11}"
            else:
                wr = pairwise[(a, b)]
                lo, hi = wr.wilson()
                star = "*" if lo > 0.5 or hi < 0.5 else " "   # interval excludes 50%
                row += f"{100 * wr.rate:>10.1f}{star}"
        lines.append(row)

    lines += [
        "",
        f"resolution: {2 * len(seeds)} games gives +/-{100 * Winrate(0.5 * 2 * len(seeds), 2 * len(seeds)).half_width:.1f}pp "
        f"at 50%. Separating 52% from 50% needs ~{games_needed(0.02):,} games — do not act on a 52%.",
    ]
    return "\n".join(lines)


def html_report(path, names, pairwise, mean_money, overall, ratings, seeds, elapsed) -> None:
    order = sorted(names, key=lambda n: -ratings[n])

    def cell(a, b):
        if a == b:
            return '<td class="d">—</td>'
        wr = pairwise[(a, b)]
        lo, hi = wr.wilson()
        sig = lo > 0.5 or hi < 0.5
        shade = int(255 - min(abs(wr.rate - 0.5) * 2, 1.0) * 90)
        colour = f"rgb({shade},255,{shade})" if wr.rate > 0.5 else f"rgb(255,{shade},{shade})"
        style = f'style="background:{colour}"' if sig else ""
        return (f'<td {style} title="{wr}">{100 * wr.rate:.1f}'
                f'<span class="ci">±{100 * wr.half_width:.0f}</span></td>')

    rows = "".join(
        f"<tr><th>{a}</th>" + "".join(cell(a, b) for b in order) + "</tr>" for a in order
    )
    standings = "".join(
        f"<tr><td>{i + 1}</td><th>{n}</th><td>{ratings[n]:.2f}</td>"
        f"<td>{100 * overall[n].rate:.1f}% ±{100 * overall[n].half_width:.1f}</td>"
        f"<td>${mean_money[n]:,.0f}</td><td class='n'>{REGISTRY[n].note}</td></tr>"
        for i, n in enumerate(order)
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>Kaggriculture arena</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:1100px}}
 table{{border-collapse:collapse;margin:1rem 0}} td,th{{border:1px solid #ddd;padding:.3rem .6rem;text-align:right}}
 th{{background:#f6f6f6;text-align:left}} .d{{color:#bbb}} .ci{{color:#666;font-size:.8em;margin-left:.3em}}
 .n{{text-align:left;color:#666;font-size:.9em}} caption{{text-align:left;font-weight:600;padding:.4rem 0}}
</style>
<h1>Kaggriculture arena</h1>
<p>{len(seeds)} seeds x 2 seatings = <b>{2 * len(seeds)} games</b> per pairing,
{len(names)} agents, {elapsed:.1f}s. Shaded cells are pairings whose 95% Wilson interval
excludes 50%; everything else is noise.</p>
<table><caption>Standings</caption>
<tr><th>#</th><th>agent</th><th>skill</th><th>winrate</th><th>mean $</th><th>note</th></tr>
{standings}</table>
<table><caption>Pairwise winrate — row vs column</caption>
<tr><th></th>{''.join(f'<th>{n}</th>' for n in order)}</tr>{rows}</table>
<p style="color:#666">Separating 52% from 50% needs ~{games_needed(0.02):,} games.</p>"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="pass,random,starter")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--games", type=int, default=32, help="seeds per pairing (x2 seatings)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--episode-steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--html", default=None)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--gauntlet", default=None,
                    help="play this agent against every other, instead of a full round-robin")
    args = ap.parse_args()

    names = sorted(REGISTRY) if args.all else args.agents.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.games))
    config = {"episodeSteps": args.episode_steps}

    t0 = time.perf_counter()
    if args.gauntlet and args.gauntlet not in names:
        names = [args.gauntlet] + names
    specs, results = run(names, seeds, config, args.workers, gauntlet=args.gauntlet)
    elapsed = time.perf_counter() - t0

    pairwise, mean_money, overall = tabulate(results)
    if args.gauntlet:
        hero = args.gauntlet
        rows = sorted((n for n in names if n != hero),
                      key=lambda n: pairwise[(hero, n)].rate)
        print(f"\nGAUNTLET: {hero} vs {len(rows)} agents, "
              f"{2 * len(seeds)} games each ({elapsed:.0f}s)\n")
        print(f"{'opponent':<20}{'champion winrate (95% Wilson)':>32}{'our $':>11}{'their $':>11}")
        print("-" * 74)
        losses = 0
        for n in rows:
            wr = pairwise[(hero, n)]
            lo, _ = wr.wilson()
            flag = "  <-- LOSS" if wr.rate < 0.5 else ("  (tie)" if lo <= 0.5 else "")
            losses += wr.rate < 0.5
            ours, theirs = pair_money(results, hero, n)
            print(f"{n:<20}{str(wr):>32}{ours:>11,.0f}{theirs:>11,.0f}{flag}")
        print(f"\nlost to {losses} of {len(rows)} agents; "
              f"overall {overall[hero]}")
    else:
        ratings = openskill_ratings(names, results)
        print(report(names, pairwise, mean_money, overall, ratings, seeds, elapsed))

    if not args.no_db:
        persist(args.db, specs, results, config)
        print(f"\n{len(results)} matches -> {args.db}")
    if args.html:
        html_report(args.html, names, pairwise, mean_money, overall, ratings, seeds, elapsed)
        print(f"report -> {args.html}")


if __name__ == "__main__":
    main()
