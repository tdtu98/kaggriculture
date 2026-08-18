"""S3's gate: does the candidate actually beat the incumbent, on seeds nobody has ever tuned on?

    accept(candidate_vec, incumbent_vec) -> {"accept": bool, "win_rate", "ci", "guard", "block"}

Three rules, and each one exists because breaking it has already cost this project a promotion.

* **The block is reserved and it is spent, not borrowed.** 54000:54080 is the only seed range this
  module will touch, `fitness.evaluate` refuses those seeds outright, and every acceptance run
  claims a 40-seed sub-block through a ledger on disk *before* it plays a game. Re-running after a
  crash therefore gets fresh seeds rather than the ones the crashed run already saw. When the
  reserve is gone the function raises: a gate that quietly recycles its holdout is not a gate.

* **The CI has to clear 50%, not the point estimate.** Five straight promotions on this project
  were wrong because a 3-8pp difference was read off a sample that can only resolve 12pp (D19).
  80 games (40 seeds, both seats) is the bar; the Wilson interval is what decides.

* **Counters outrank the money.** A candidate that breaks C5's bars is rejected however it scored.
  `search/ga.py:GUARD_BARS` is the same table the search itself uses, so the gate cannot disagree
  with the objective about what a legal plan is.

Ties count as half a win on both sides, which is what the harness' mirror wobble makes them: on
most seeds two identical scripts tie exactly, and calling those a loss for the challenger would
build a bias into the gate itself.
"""

from __future__ import annotations

import json
import math
import os

from agent.plan import migrate
from harness import registry
from harness.counters import mean_counters
from harness.run import Match, run
from search import ga

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(_ROOT, "search", "accept_ledger.json")

#: The reserved acceptance range (TASKS_v4 S2/S3). Nothing else in `search/` may read it.
ACCEPT_BLOCK = (54000, 54080)

#: 40 seeds x 2 seats = the 80-game block the spec asks for. Two sub-blocks exist in total; that
#: scarcity is deliberate and is the reason the ledger is on disk rather than in memory.
SUB_BLOCK = 40

DEFAULT_CONFIG = {"episodeSteps": 720}


def load_ledger(path: str = LEDGER) -> list[list[int]]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [list(map(int, b)) for b in json.load(fh).get("spent", [])]


def _save_ledger(path: str, spent, note: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"spent": [list(b) for b in spent], "last": note}, fh, indent=1)
    os.replace(tmp, path)


def remaining(path: str = LEDGER) -> list[tuple[int, int]]:
    """Sub-blocks still unspent, in order."""
    spent = {tuple(b) for b in load_ledger(path)}
    lo, hi = ACCEPT_BLOCK
    return [(s, s + SUB_BLOCK) for s in range(lo, hi, SUB_BLOCK)
            if (s, s + SUB_BLOCK) not in spent]


def claim_block(note: str = "", path: str = LEDGER, dry_run: bool = False) -> tuple[int, int]:
    """Take the next unspent sub-block and mark it spent. Raises when the reserve is exhausted.

    `dry_run` returns the block it *would* claim without writing the ledger — the only way to
    exercise this code (tests, smoke runs) without burning a holdout that cannot be regenerated.
    """
    free = remaining(path)
    if not free:
        raise RuntimeError(
            f"the acceptance reserve {ACCEPT_BLOCK[0]}:{ACCEPT_BLOCK[1]} is exhausted "
            f"({len(load_ledger(path))} sub-blocks spent). Reserve a new range in "
            "search/fitness.ACCEPTANCE_BLOCKS and record it in the experiment log — do not "
            "re-run a spent block.")
    block = free[0]
    if not dry_run:
        _save_ledger(path, load_ledger(path) + [list(block)], note)
    return block


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion, half-wins allowed."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def head_to_head(candidate: str, incumbent: str, seeds, workers=None, config=None) -> dict:
    """Both seats on every seed. Seat order moves real money in a shared market, so a one-seat
    gate measures the seating as much as the plan."""
    schedule = []
    for s in seeds:
        schedule.append(Match(candidate, incumbent, s, True))
        schedule.append(Match(candidate, incumbent, s, False))
    results = run([candidate, incumbent], list(seeds), games=len(schedule), both_seats=True,
                  config=config or DEFAULT_CONFIG, workers=workers, jsonl=None,
                  schedule=schedule, quiet=True)
    wins = sum(1.0 if r.money_a > r.money_b else 0.0 if r.money_a < r.money_b else 0.5
               for r in results)
    return {
        "games": len(results),
        "successes": wins,
        "win_rate": wins / len(results),
        "mean_margin": sum(r.money_a - r.money_b for r in results) / len(results),
        "mean_money": sum(r.money_a for r in results) / len(results),
        "counters": mean_counters([r.counters_a for r in results]),
    }


def accept(candidate_vec, incumbent_vec, block: tuple[int, int] | None = None, *,
           workers=None, config=None, dry_run: bool = False, ledger: str = LEDGER,
           note: str = "") -> dict:
    """The S3 decision for one candidate. `block=None` claims the next reserved sub-block.

    An explicit `block` is honoured (a re-analysis of a run that already spent one) but is *not*
    checked out of the ledger, so passing one is a deliberate act rather than a way around it.
    """
    # A genome from an older layout is embedded rather than rejected: the whole point of the gate is
    # to compare a harvested candidate against the incumbent, and the harvest predates the widening.
    candidate_vec = migrate(candidate_vec)
    incumbent_vec = migrate(incumbent_vec)
    claimed = block is None
    if claimed:
        block = claim_block(note=note or "accept()", path=ledger, dry_run=dry_run)
    seeds = list(range(block[0], block[1]))

    cand = registry.vec_name(candidate_vec)
    inc = registry.vec_name(incumbent_vec)
    if cand == inc:
        raise ValueError("candidate and incumbent are the same genome")

    h = head_to_head(cand, inc, seeds, workers=workers, config=config)
    lo, hi = wilson(h["successes"], h["games"])
    violations = ga.guard_violations(h["counters"])
    verdict = bool(lo > 0.5 and not violations)
    return {
        "accept": verdict,
        "reason": ("counter guard: " + ", ".join(violations)) if violations
                  else ("CI includes 50%" if lo <= 0.5 else "beats the incumbent"),
        "win_rate": h["win_rate"],
        "ci": (lo, hi),
        "mean_margin": h["mean_margin"],
        "mean_money": h["mean_money"],
        "games": h["games"],
        "guard": violations,
        "counters": {k: round(ga.counter_value(h["counters"], k), 4) for k in ga.LOG_COUNTERS},
        "fallbacks": ga.counter_value(h["counters"], "fallbacks"),
        "block": list(block),
        "block_claimed": claimed and not dry_run,
        "candidate": cand,
        "incumbent": inc,
    }


def summary(res: dict) -> str:
    lo, hi = res["ci"]
    return (f"{'ACCEPT' if res['accept'] else 'REJECT'}  {res['reason']}\n"
            f"  win {res['win_rate']:.1%}  CI [{lo:.1%}, {hi:.1%}]  "
            f"margin {res['mean_margin']:+,.0f}  ${res['mean_money']:,.0f}  "
            f"{res['games']} games on {res['block'][0]}:{res['block'][1]}\n"
            f"  counters {res['counters']}  fallbacks {res['fallbacks']}")


def main(argv=None) -> None:                                   # pragma: no cover - CLI
    import argparse

    from agent.plan import Plan, encode

    p = argparse.ArgumentParser(description="S3 acceptance gate on the reserved block")
    p.add_argument("--candidate", required=True,
                   help="JSON list of gene values; an older layout is migrated")
    p.add_argument("--incumbent", default=None, help="JSON list; default Plan.boatlee_like()")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="do not spend a sub-block")
    a = p.parse_args(argv)
    cand = json.loads(a.candidate)
    inc = json.loads(a.incumbent) if a.incumbent else encode(Plan.boatlee_like())
    print(summary(accept(cand, inc, workers=a.workers, dry_run=a.dry_run)))


if __name__ == "__main__":                                     # pragma: no cover
    main()
