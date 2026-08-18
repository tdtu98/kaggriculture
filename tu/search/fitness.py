"""What a genome is worth (TASKS_v4 S1): one vector, one number, and the counters behind it.

`evaluate(vec, seeds, opponents)` decodes the vector to a `Plan`, hands it to the compiler, plays it
both seats against a pool, and returns

    {fitness, win_rate, mean_margin, solo, counters, per_opponent, ...}

with `fitness = 0.7 * win_rate + 0.3 * normalised_margin`.

Four decisions in here are load-bearing, and each of them is a mistake this project has already
paid for:

* **Win rate first, money second.** Mean money is only comparable inside a fixed opponent field
  (E5: `melon-wheat` earns 7% more than `melon` against `starter` and then loses to it 32/32), so
  the fitness is 70% pairwise winrate. The margin term is the other 30% because pure winrate is
  flat — every genome that loses to Boatlee scores exactly 0, and a search cannot climb a flat
  surface. The margin says *how badly*, which is the gradient.

* **The margin is squashed, not averaged.** `tanh(margin / MARGIN_SCALE)` — a raw mean would let one
  seed where the opponent collapsed outvote five seeds of steady losing, and would put the two terms
  on incomparable scales.

* **The pool is weighted and includes our own exploiters.** `registry.SEARCH_POOL`. E10: a champion
  that was correctly seed-validated lost **0/80** to a dumper it had never faced. Varying the seeds
  does not detect a single-opponent optimum; varying the opponent does.

* **Counters come back with the money.** S2 rejects plans that "win" by breaking C5's counters, and
  it cannot do that if the fitness is a scalar. `counters` is the full `mean_counters` block,
  `effect_count` included, so a candidate whose overlays never fired is visible as such (E44).

Seeds are the caller's business and are never defaulted: `search_seeds(generation)` hands out a
rotating slice of the search block, and `ACCEPTANCE_BLOCKS` is asserted disjoint from it so a
tuning run cannot quietly consume the block its own result will be judged on.
"""

from __future__ import annotations

import math
import time

from harness import registry
from harness.counters import mean_counters
from harness.run import Match, run

#: 0.7 winrate + 0.3 margin (TASKS_v4 S1).
WIN_WEIGHT = 0.7
MARGIN_WEIGHT = 0.3

#: The margin at which `normalised_margin` reaches tanh(1) = 0.76. Set to the scale of a decided
#: game rather than a tuned constant: the compiler solos ~$120k and loses to Boatlee by ~$84k
#: (E70), so $50k is "clearly beaten / clearly beating" and the interesting region is inside it.
MARGIN_SCALE = 50_000.0

DEFAULT_CONFIG = {"episodeSteps": 720}

#: Blocks the search must not touch. 54000:54080 is S2's acceptance gate; 50000/51000 were spent on
#: the C5 gates (E69/E70) and are quoted in the experiment record.
ACCEPTANCE_BLOCKS = ((54000, 54080), (50000, 50040), (51000, 51200))

#: Where search-side seeds come from. Disjoint from every acceptance block, by assertion below.
SEARCH_BLOCK = (60000, 64000)


def _is_reserved(seed: int) -> bool:
    return any(lo <= seed < hi for lo, hi in ACCEPTANCE_BLOCKS)


def search_seeds(generation: int, n: int = 6, block: tuple[int, int] = SEARCH_BLOCK) -> list[int]:
    """`n` seeds for one generation, rotated so no two generations share a slice.

    Rotation is the point: six seeds is far below the 80-game bar, and a fixed six would be an
    invitation to overfit them. Across a 50-generation run the population is scored on 300 distinct
    seasons; each *comparison within* a generation is still paired, because every candidate in a
    generation gets the same six.
    """
    lo, hi = block
    span = hi - lo
    if n > span:
        raise ValueError(f"{n} seeds requested from a block of {span}")
    seeds = [lo + (generation * n + i) % span for i in range(n)]
    bad = [s for s in seeds if _is_reserved(s)]
    if bad:
        raise ValueError(f"generation {generation} would spend acceptance seeds {bad}")
    return seeds


def normalised_margin(margin: float) -> float:
    """Money margin -> [0, 1], 0.5 at a dead heat. Squashed; see the module note."""
    return 0.5 * (1.0 + math.tanh(margin / MARGIN_SCALE))


def _pool(opponents) -> list[tuple[str, float]]:
    """Accept `["boatlee", ...]` or `[("boatlee", 2.0), ...]`; return the weighted form."""
    out = []
    for item in opponents:
        if isinstance(item, str):
            out.append((item, 1.0))
        else:
            name, weight = item
            out.append((str(name), float(weight)))
    if not out:
        raise ValueError("an empty opponent pool cannot score anything")
    return out


def evaluate(vec, seeds, opponents=registry.SEARCH_POOL, *, config: dict | None = None,
             workers: int | None = None, with_solo: bool = False,
             solo_opponent: str = "starter", jsonl: str | None = None) -> dict:
    """Score one genome. Deterministic: same `vec` and `seeds` give a bit-identical result.

    Every pairing is played in **both seats** on every seed, because seat order moves real money in
    a shared market (the harness' own mirror test measures $771 on seed 1 from seating alone).

    `with_solo` adds a `starter` pairing whose money is reported as `solo` and which is **excluded
    from the fitness**. It is a calibration number — "is this plan still worth $120k on its own?" —
    and E5 is the standing reason it must never be a ranking signal.
    """
    seeds = [int(s) for s in seeds]
    spent = [s for s in seeds if _is_reserved(s)]
    if spent:
        raise ValueError(f"seeds {spent} are in an acceptance block; use search_seeds()")

    pool = _pool(opponents)
    cand = registry.vec_name(vec)
    if cand in {n for n, _ in pool}:
        raise ValueError("the candidate cannot be its own opponent")

    scored = [n for n, _ in pool]
    names = [cand] + scored + ([solo_opponent] if with_solo else [])
    schedule: list[Match] = []
    for opp in scored + ([solo_opponent] if with_solo else []):
        for s in seeds:
            schedule.append(Match(cand, opp, s, True))
            schedule.append(Match(cand, opp, s, False))

    t0 = time.perf_counter()
    results = run(names, seeds, games=len(schedule), both_seats=True,
                  config=config or DEFAULT_CONFIG, workers=workers, jsonl=jsonl,
                  schedule=schedule, quiet=True)
    seconds = time.perf_counter() - t0

    by_opp: dict[str, list] = {}
    for r in results:
        by_opp.setdefault(r.match.b, []).append(r)

    per_opponent = {}
    weights = dict(pool)
    counters: list[dict] = []
    for opp, rows in by_opp.items():
        wins = sum(1.0 if r.money_a > r.money_b else 0.0 if r.money_a < r.money_b else 0.5
                   for r in rows)
        per_opponent[opp] = {
            "win_rate": wins / len(rows),
            "mean_margin": sum(r.money_a - r.money_b for r in rows) / len(rows),
            "mean_money": sum(r.money_a for r in rows) / len(rows),
            "games": len(rows),
        }
        if opp in weights:
            counters.extend(r.counters_a for r in rows)

    total_w = sum(weights.values())
    win_rate = sum(per_opponent[o]["win_rate"] * w for o, w in weights.items()) / total_w
    mean_margin = sum(per_opponent[o]["mean_margin"] * w for o, w in weights.items()) / total_w
    fitness = WIN_WEIGHT * win_rate + MARGIN_WEIGHT * normalised_margin(mean_margin)

    return {
        "fitness": fitness,
        "win_rate": win_rate,
        "mean_margin": mean_margin,
        "solo": per_opponent[solo_opponent]["mean_money"] if with_solo else None,
        "counters": mean_counters(counters),
        "per_opponent": per_opponent,
        "candidate": cand,
        "fingerprint": registry.get(cand).fingerprint,
        "seeds": seeds,
        "opponents": pool,
        "games": len(results),
        "seconds": seconds,
        "games_per_sec": len(results) / seconds if seconds else 0.0,
    }


def summary(result: dict) -> str:
    """One block a human can read, with the counters next to the money (harness rule)."""
    c = result["counters"]
    rows = [f"fitness {result['fitness']:.4f}   winrate {result['win_rate']:.1%}   "
            f"margin {result['mean_margin']:+,.0f}"
            + (f"   solo ${result['solo']:,.0f}" if result.get("solo") is not None else "")]
    for opp, d in sorted(result["per_opponent"].items()):
        rows.append(f"  vs {opp:<16}{d['win_rate']:>7.1%}  {d['mean_margin']:>+11,.0f}  "
                    f"${d['mean_money']:>10,.0f}  ({d['games']} games)")
    rows.append(f"  counters  steps/useful {c.get('steps_per_useful', 0):.2f}  "
                f"thirst {c.get('plants_lost_thirst', 0):.1f}  "
                f"blocked {c.get('blocked_ops', 0):.1f}  "
                f"straw/plant {c.get('strawberry_per_plant', 0):.1f}  "
                f"milk/cow-d {c.get('milk_per_cow_day', 0):.2f}")
    rows.append(f"  {result['games']} games in {result['seconds']:.1f}s "
                f"({result['games_per_sec']:.2f} games/s)")
    return "\n".join(rows)


assert not any(_is_reserved(s) for s in range(*SEARCH_BLOCK)), \
    "the search block overlaps an acceptance block"
