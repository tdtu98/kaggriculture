"""S2's outer loop, in parts: what a mutation is, what a genome costs, and what a run remembers.

`search/run.py` is the driver; this module is everything it needs to be *resumable* and *honest*.
Four things live here and each one is a lesson the project has already paid for.

* **Mutations are structural, and the repair is visible.** The operators below say "move a cohort
  two days", "±3 tiles", "one more cow", "toggle SE" — not "jitter gene 27". A gene-wise jitter
  shreds a cohort (crop/quadrant/tiles/day only mean anything together) and spends the budget on
  candidates `decode` will trim. `decode` repairs whatever is left, and `repair_rate` reports how
  often it had to, because a search whose candidates are quietly being rewritten is exploring a
  different space than the one it reports (`agent/plan.py`'s own note, E44's shape).

* **The GA never touches the CMA genes and the CMA never touches the GA genes.** `GA_GENES` and
  `CMA_GENES` partition the genome, asserted at import. Two optimisers writing the same coordinate is
  not a hybrid, it is noise: the CMA would spend its budget measuring the GA's last mutation.

* **The counter guard outranks the money.** `guard_violations` applies C5's bars — thirst,
  steps/useful, fallbacks — and a candidate that breaks one is scored *below every legal genome*
  rather than merely penalised. A plan that "wins" by killing plants is a bug exploit (TASKS_v4 S2),
  and the money number cannot tell the two apart.

* **The cache keys on (genome, seed block), never on the genome alone.** `search_seeds` rotates, so
  the same vector on generation 4 and generation 5 is two different measurements; a cache that
  conflated them would hand elitism a stale number and turn the elite set into a random walk.
  `fitness.evaluate` is deterministic given (vec, seeds), which is what makes the cache sound —
  and what makes `--resume` reproduce a run rather than merely continue one.
"""

from __future__ import annotations

import json
import os
import random
import time

from agent.plan import (CROPS, COHORT_SLOTS, DAYS, FLOOR_PRODUCTS, GENE_INDEX, GENES, HALF,
                        LAND_ORDER, NEVER, QUADRANTS, decode, encode, floor_gene, migrate, snap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(_ROOT, "search", "log.jsonl")
STATE = os.path.join(_ROOT, "search", "state.json")

#: The continuous consts the inner CMA-ES owns (TASKS_v4 S2 + C6's five). `release_pressure`,
#: `frontrun_lead`, `max_scarce_tiles` and `projected_pricing` are integral in the genome but
#: continuous in *meaning* — CMA proposes floats and `snap` rounds them, which is a step function
#: the strategy handles by construction (it is the same treatment the CEM line used).
#:
#: The eight per-product floors (S3, E76) are CMA genes, all of them: a floor is a continuous price
#: threshold and the floors are *correlated* — holding melon back fills the shed that wool then has
#: to defend — which is exactly the coupling a full covariance is for. They are deliberately **not**
#: given a GA jitter operator: two optimisers writing one coordinate is the partition violation this
#: module's docstring exists to prevent, and the assert below would fail.
CMA_GENES: tuple[str, ...] = tuple(
    floor_gene(p) for p in FLOOR_PRODUCTS
) + (
    "release_pressure", "frontrun_lead",
    "theta_hinge", "theta_sqrt", "max_scarce_tiles", "scarce_floor", "projected_pricing",
)

#: Everything else: the discrete layout the GA searches.
GA_GENES: tuple[str, ...] = tuple(g.name for g in GENES if g.name not in CMA_GENES)

assert set(GA_GENES).isdisjoint(CMA_GENES)
assert set(GA_GENES) | set(CMA_GENES) == {g.name for g in GENES}

#: C5's bars (TASKS_v4 C5/S2). `fallbacks` is an *effect* counter: it is the compiler admitting it
#: could not compile a turn, so any non-zero value is a broken candidate rather than a bad one.
GUARD_BARS: tuple[tuple[str, float], ...] = (
    ("plants_lost_thirst", 10.0),
    ("steps_per_useful", 1.15),
    ("fallbacks", 0.0),
)

#: Everything a log row keeps from the counter block. Narrow on purpose: the full block carries
#: per-product dicts that would make `log.jsonl` unreadable at 10k rows.
LOG_COUNTERS = ("steps_per_useful", "plants_lost_thirst", "plants_lost", "blocked_ops",
                "strawberry_per_plant", "milk_per_cow_day", "unharvested_ripe_at_end",
                "shed_overflow_discarded", "fertilize_hits")


def gene(vec, name: str) -> float:
    return vec[GENE_INDEX[name]]


def counter_value(counters: dict, name: str) -> float:
    """Scalar counters live at the top level; overlay effects live under `effect_count`."""
    if name in counters:
        return float(counters[name])
    return float((counters.get("effect_count") or {}).get(name, 0.0))


def guard_violations(counters: dict) -> list[str]:
    """C5's bars, as a list of what broke. Empty means the candidate is legal.

    Missing counters are *not* violations: an evaluation that produced no counter block at all
    (an agent the observer could not read) is a harness problem, and silently failing every such
    candidate would look exactly like a search that cannot find anything.
    """
    out = []
    for name, bar in GUARD_BARS:
        if not counters:
            return []
        v = counter_value(counters, name)
        if v > bar:
            out.append(f"{name}={v:.3g}>{bar:g}")
    return out


#: How far below every legal genome a guard-breaking candidate sits. Fitness lives in [0, 1], so
#: the penalty must be *strictly* greater than 1: at exactly 1.0 a perfect illegal genome (1.0)
#: ties the worst legal one (0.0) and can win a tournament. Violators keep their relative order,
#: so a population that briefly goes all-illegal is still selectable — a stall, not a rejection.
GUARD_PENALTY = 2.0


def score_of(fitness: float, violations) -> float:
    return float(fitness) - (GUARD_PENALTY if violations else 0.0)


# ------------------------------------------------------------------------ mutation

def _live_slots(vec) -> list[int]:
    return [i for i in range(COHORT_SLOTS) if int(gene(vec, f"c{i}_tiles")) > 0]


def _empty_slots(vec) -> list[int]:
    return [i for i in range(COHORT_SLOTS) if int(gene(vec, f"c{i}_tiles")) <= 0]


def _open_quadrants(vec) -> list[int]:
    """Quadrant indices the plan actually buys, NW always included.

    Placing a cohort on land the plan never buys is not a mutation, it is a *drop*: `decode` throws
    that cohort away and `encode` writes the slot back empty. Measured over a 20-mutation walk from
    `boatlee_like`, letting the placement ops pick blindly pulled the live-cohort mode down to 3 —
    the population losing the width S3 just added, one blind retarget at a time.
    """
    out = [QUADRANTS.index("NW")]
    for q in QUADRANTS:
        if q != "NW" and int(gene(vec, f"land_{q}")) < DAYS:
            out.append(QUADRANTS.index(q))
    return out


def _free_quadrants(vec) -> list[int]:
    """Bought quadrants that still have room, by tile count. Falls back to all bought ones.

    Same argument as `_open_quadrants`, one level finer: a cohort added to a quadrant whose 25 tiles
    are already claimed is dropped by `decode` for want of a free tile. The pasture cluster is not
    counted (it straddles quadrant borders), so this over-estimates NW by a few tiles — a heuristic
    for *where to propose*, not a capacity rule; `decode` remains the authority.
    """
    claimed: dict = {}
    for i in range(COHORT_SLOTS):
        n = int(gene(vec, f"c{i}_tiles"))
        if n > 0:
            qi = int(gene(vec, f"c{i}_quad"))
            claimed[qi] = claimed.get(qi, 0) + n
    opens = _open_quadrants(vec)
    room = [qi for qi in opens if claimed.get(qi, 0) < HALF * HALF]
    return room or opens


def _bump(vec, name: str, delta: float) -> None:
    vec[GENE_INDEX[name]] = vec[GENE_INDEX[name]] + delta


def op_cohort_day(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    _bump(vec, f"c{i}_day", rng.choice((-2, -1, 1, 2)))
    return f"cohort{i}_day"


def op_cohort_tiles(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    _bump(vec, f"c{i}_tiles", rng.choice((-3, 3)))
    return f"cohort{i}_tiles"


def op_cohort_crop(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    cur = int(gene(vec, f"c{i}_crop"))
    vec[GENE_INDEX[f"c{i}_crop"]] = float(rng.choice([c for c in range(len(CROPS)) if c != cur]))
    return f"cohort{i}_crop"


def op_cohort_quadrant(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    cur = int(gene(vec, f"c{i}_quad"))
    vec[GENE_INDEX[f"c{i}_quad"]] = float(rng.choice([q for q in range(len(QUADRANTS)) if q != cur]))
    return f"cohort{i}_quad"


def op_cohort_replant(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    vec[GENE_INDEX[f"c{i}_replant"]] = 1.0 - int(gene(vec, f"c{i}_replant"))
    return f"cohort{i}_replant"


def op_cohort_add(vec, rng) -> str | None:
    """Fill an empty slot. Without this the GA can only ever shrink the plan: `op_cohort_drop`
    zeroes a slot and nothing brings it back, so the population monotonically loses cohorts."""
    slots = _empty_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    q = rng.choice(_free_quadrants(vec))
    name = QUADRANTS[q]
    opens = 0 if name == "NW" else int(gene(vec, f"land_{name}"))
    vec[GENE_INDEX[f"c{i}_crop"]] = float(rng.randrange(len(CROPS)))
    vec[GENE_INDEX[f"c{i}_quad"]] = float(q)
    vec[GENE_INDEX[f"c{i}_tiles"]] = float(rng.randint(3, 18))
    # On land the plan buys, and not before it arrives: `decode` would slide the day itself and
    # report the slide as a repair, which is the search reading its own repair rate as exploration.
    vec[GENE_INDEX[f"c{i}_day"]] = float(min(DAYS - 1, opens + rng.randrange(0, DAYS - opens)))
    vec[GENE_INDEX[f"c{i}_replant"]] = float(rng.random() < 0.4)
    return f"cohort{i}_add"


#: Tiles in one row of a quadrant. A quadrant is HALF x HALF and `decode` fills it row-major, so
#: "N rows of crop C in quadrant Q" *is* a cohort of `N * HALF` tiles — the row representation S3
#: suggested, expressed inside the cohort genes rather than beside them (see the module note in
#: `agent/plan.py` and the S3 report).
ROW = HALF
ROW_CLAIMS = tuple(ROW * n for n in range(1, HALF + 1))          # 5, 10, 15, 20, 25


def op_cohort_rows(vec, rng) -> str | None:
    """Snap a slot's tile count to a whole number of rows — the "claim a row band" move.

    `op_cohort_tiles` walks +/-3, which takes six mutations to get a 3-tile cohort to a full 25-tile
    quadrant and passes through nothing but half-claims on the way. Production is the axis S3 says
    the representation was short on, so it gets a move that crosses it in one step.
    """
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    cur = int(gene(vec, f"c{i}_tiles"))
    choices = [n for n in ROW_CLAIMS if n != cur]
    vec[GENE_INDEX[f"c{i}_tiles"]] = float(rng.choice(choices))
    return f"cohort{i}_rows"


def op_cohort_retarget(vec, rng) -> str | None:
    """Move a whole cohort to another quadrant *and* re-crop it, day slid to when that land opens.

    Crop, quadrant and day are one decision (the crossover blocks say so); changing only the
    quadrant leaves a day the new land does not reach, and `decode` then slides the day itself and
    reports a repair. Doing all three here is the difference between proposing a plan and proposing
    something the repair will rewrite.
    """
    slots = _live_slots(vec)
    if not slots:
        return None
    i = rng.choice(slots)
    q = rng.choice(_open_quadrants(vec))
    vec[GENE_INDEX[f"c{i}_quad"]] = float(q)
    vec[GENE_INDEX[f"c{i}_crop"]] = float(rng.randrange(len(CROPS)))
    name = QUADRANTS[q]
    opens = 0 if name == "NW" else int(gene(vec, f"land_{name}"))
    vec[GENE_INDEX[f"c{i}_day"]] = float(min(DAYS - 1, opens + rng.randrange(0, 4)))
    return f"cohort{i}_retarget"


def op_claim_quadrant(vec, rng) -> str | None:
    """Buy the next unbought quadrant *and* stand crops on it, in one move.

    This is the operator S3's "per-quadrant crop rows" was asking for, and the reason the extra
    slots are reachable at all. `boatlee_like` already covers 59 of the 62 tiles its bought land
    has free, so `op_cohort_add` on its own has nowhere to put a cohort and `decode` drops it: the
    seed population reached 7+ cohorts twice in 48 tries. New production needs *new land*, and land
    plus the cohorts that use it is one decision — split across two mutations, the intermediate
    (land bought, nothing planted on it) is strictly worse than either end and selection kills it
    before the second half arrives.
    """
    unbought = [q for q in LAND_ORDER if int(gene(vec, f"land_{q}")) >= NEVER]
    empty = _empty_slots(vec)
    if not unbought or not empty:
        return None
    q = unbought[0]                                  # land is sold in order; only the next is buyable
    prev = max([0] + [int(gene(vec, f"land_{p}")) for p in LAND_ORDER
                      if p != q and int(gene(vec, f"land_{p}")) < NEVER])
    day = rng.randint(prev, min(DAYS - 8, max(prev, DAYS - 8)))
    vec[GENE_INDEX[f"land_{q}"]] = float(day)
    qi = float(QUADRANTS.index(q))
    for i in empty[:rng.randint(1, 2)]:
        vec[GENE_INDEX[f"c{i}_crop"]] = float(rng.randrange(len(CROPS)))
        vec[GENE_INDEX[f"c{i}_quad"]] = qi
        vec[GENE_INDEX[f"c{i}_tiles"]] = float(rng.choice(ROW_CLAIMS))
        vec[GENE_INDEX[f"c{i}_day"]] = float(min(DAYS - 1, day + rng.randrange(0, 3)))
        vec[GENE_INDEX[f"c{i}_replant"]] = float(rng.random() < 0.4)
    return f"claim_{q}"


def op_cohort_drop(vec, rng) -> str | None:
    slots = _live_slots(vec)
    if len(slots) <= 1:
        return None
    i = rng.choice(slots)
    vec[GENE_INDEX[f"c{i}_tiles"]] = 0.0
    return f"cohort{i}_drop"


def op_pasture(vec, rng) -> str | None:
    """"Swap a pasture tile" in genome terms: the cluster is canonical (`pasture_order`), so the
    only thing a plan can say about it is *how many* — and the tile that joins or leaves is the
    one at the edge of the cluster, which is exactly the swap the spec asks for."""
    _bump(vec, "n_pastures", rng.choice((-1, 1)))
    return "pastures"


def op_animal(vec, rng) -> str | None:
    name = rng.choice(("n_sheep", "n_cows", "n_geese"))
    _bump(vec, name, rng.choice((-1, 1)))
    return name


def op_animal_start(vec, rng) -> str | None:
    name = rng.choice(("sheep_start", "cow_start", "geese_start"))
    _bump(vec, name, rng.choice((-2, -1, 1, 2)))
    return name


def op_animals_per_day(vec, rng) -> str | None:
    _bump(vec, "animals_per_day", rng.choice((-1, 1)))
    return "animals_per_day"


def op_hands(vec, rng) -> str | None:
    """±1 hand on a day. The curve is (start, cap, ramp), so "a day" is one of the three knobs
    that decide which day it lands on — the genome has no per-day hand gene to nudge."""
    name = rng.choice(("hands_start", "hands_cap", "hands_ramp"))
    _bump(vec, name, rng.choice((-1, 1)))
    return name


def op_toggle_se(vec, rng) -> str | None:
    """SE on/off. Land is the mechanic that decides the matchup (E26) and it is a *discrete*
    decision — a ±1 walk from `NEVER`=99 would take twenty generations to discover day 20."""
    if int(gene(vec, "land_SE")) >= NEVER:
        vec[GENE_INDEX["land_SE"]] = float(rng.randint(8, DAYS - 6))
    else:
        vec[GENE_INDEX["land_SE"]] = float(NEVER)
    return "land_SE_toggle"


def op_land_day(vec, rng) -> str | None:
    name = rng.choice(("land_NE", "land_SW", "land_SE"))
    if int(gene(vec, name)) >= NEVER:
        return None
    _bump(vec, name, rng.choice((-2, -1, 1, 2)))
    return name


#: (operator, weight). Weighted because the operators are not equally likely to be useful: cohort
#: geometry is most of the plan's money, `toggle_se` is one bit and does not need a fifth of the
#: budget.
OPERATORS: tuple[tuple, ...] = (
    (op_cohort_day, 3.0),
    (op_cohort_tiles, 3.0),
    (op_cohort_crop, 2.0),
    (op_cohort_quadrant, 1.5),
    (op_cohort_replant, 1.0),
    # 2.5, not the 1.0 it carried at six slots. The population is seeded from `boatlee_like`, which
    # uses six of the ten slots, so `op_cohort_add` is the *only* route into the width S3 just paid
    # for: at weight 1.0 it was 4% of the operator budget and the extra slots would have sat empty
    # for the whole run, which is the E44 shape — a widening that never fired.
    (op_cohort_add, 2.5),
    (op_cohort_drop, 0.7),
    (op_cohort_rows, 1.5),
    (op_cohort_retarget, 1.2),
    (op_claim_quadrant, 1.5),
    (op_pasture, 2.0),
    (op_animal, 3.0),
    (op_animal_start, 1.5),
    (op_animals_per_day, 0.8),
    (op_hands, 2.0),
    (op_toggle_se, 0.6),
    (op_land_day, 1.5),
)

_OPS = [o for o, _ in OPERATORS]
_WEIGHTS = [w for _, w in OPERATORS]


def canonical(vec) -> list[float]:
    """The genome as the game would actually play it: decode (which repairs) then re-encode.

    Every mutant goes through this, so the population only ever holds vectors that are their own
    fixed point. Without it two different vectors can name the same plan and the cache keys on the
    vector — the search would pay twice for one measurement and call them two data points.
    """
    return snap(encode(decode(vec)))


#: How many times `mutate` will re-roll a mutant that canonicalises straight back to its parent.
#: Measured 18% of single mutations on `boatlee_like` are absorbed this way — a gene the plan
#: does not use (geese count, geese start day) or a clamp at a bound — and a population where a
#: fifth of the children are copies of their parent is a fifth less search for the same games.
MUTATE_RETRIES = 4


def mutate(vec, rng, n_ops: int = 0) -> tuple[list[float], list[str], bool]:
    """One mutant: apply 1-3 structural ops, repair through `decode`, return it canonical.

    Returns `(vector, applied_ops, repaired)`. `repaired` is true when `decode` had to change the
    genome to make it fit the board — the number `run.py` reports as the decode-repair rate.
    """
    parent = snap(vec)
    result = (parent, [], False)
    for _ in range(MUTATE_RETRIES):
        out = list(parent)
        applied: list[str] = []
        n = n_ops or (1 if rng.random() < 0.6 else (2 if rng.random() < 0.75 else 3))
        for _ in range(n):
            op = rng.choices(_OPS, weights=_WEIGHTS, k=1)[0]
            tag = op(out, rng)
            if tag:
                applied.append(tag)
        plan = decode(snap(out))
        result = (snap(encode(plan)), applied, bool(plan.notes))
        if result[0] != parent:
            break
    return result


# ------------------------------------------------------------------------ crossover

#: Genes that must travel together. A cohort's crop, quadrant, tile count and day are one decision;
#: uniform *per-gene* crossover would routinely produce "22 tiles of melon in NE on day 7" out of a
#: strawberry parent and a wheat parent — a child neither parent's fitness says anything about.
#: Land and herd are blocks for the same reason (land is order-constrained, the herd is paced).
def _blocks() -> list[tuple[str, ...]]:
    out = [("n_pastures", "land_NE", "land_SW", "land_SE"),
           ("n_sheep", "sheep_start", "n_cows", "cow_start", "n_geese", "geese_start",
            "animals_per_day"),
           ("hands_auto", "hands_start", "hands_cap", "hands_ramp")]
    for i in range(COHORT_SLOTS):
        out.append(tuple(f"c{i}_{k}" for k in ("crop", "quad", "tiles", "day", "replant")))
    return out


BLOCKS = _blocks()


def crossover(a, b, rng) -> list[float]:
    """Uniform per *block*, with the CMA genes taken whole from `a` (the fitter parent).

    Crossover is worth having here — the plan is genuinely modular, and "this parent's herd with
    that parent's strawberry cohort" is a meaningful recombination rather than a random point
    between them. It is *not* worth applying to the continuous consts: those are the inner CMA's
    coordinates, tuned as a correlated block against a specific layout, and splicing half of one
    covariance-fitted vector into another destroys the thing CMA just paid for.
    """
    out = list(snap(a))
    src_b = snap(b)
    for block in BLOCKS:
        if rng.random() < 0.5:
            for name in block:
                out[GENE_INDEX[name]] = src_b[GENE_INDEX[name]]
    return canonical(out)


# ------------------------------------------------------------------------ diversity

def diversity(population) -> float:
    """Mean normalised per-gene spread across the population, in [0, 1].

    The number that says whether the search is still searching. A population that has collapsed
    onto the seed plan scores ~0 and every subsequent generation is measuring noise; the smoke test
    checks this rather than eyeballing the best-fitness curve, which keeps rising in a collapsed
    population too.
    """
    if len(population) < 2:
        return 0.0
    total = 0.0
    for i, g in enumerate(GENES):
        span = g.hi - g.lo
        col = [v[i] for v in population]
        mean = sum(col) / len(col)
        var = sum((c - mean) ** 2 for c in col) / len(col)
        total += (var ** 0.5) / span if span else 0.0
    return total / len(GENES)


def unique_fraction(population) -> float:
    return len({tuple(snap(v)) for v in population}) / max(1, len(population))


# ------------------------------------------------------------------------ cache + log

class Cache:
    """(genome, seed block) -> evaluation. Reloadable from `log.jsonl`, which is what makes
    `--resume` cheap as well as correct."""

    def __init__(self):
        self._d: dict = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(vec, seeds) -> str:
        return ",".join(repr(float(v)) for v in snap(vec)) + "|" + ",".join(map(str, seeds))

    def get(self, vec, seeds):
        v = self._d.get(self.key(vec, seeds))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, vec, seeds, value) -> None:
        self._d[self.key(vec, seeds)] = value

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def __len__(self) -> int:
        return len(self._d)

    def load_jsonl(self, path: str = LOG, max_gen: int | None = None) -> int:
        """Rebuild from the log. Rows above `max_gen` are dropped: on `--resume` they are the
        tail of an interrupted generation, and keeping them would let a partially-scored
        generation leak into the one being replayed."""
        if not os.path.exists(path):
            return 0
        n = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if max_gen is not None and int(row.get("gen", -1)) > max_gen:
                    continue
                vec, seeds = row.get("vec"), row.get("seeds")
                if not vec or not seeds:
                    continue
                # `migrate`: a log written under an older gene layout still keys correctly, and the
                # cached number is still that genome's — the migration is an embedding, so the
                # migrated vector decodes to the same plan the row was scored on.
                self._d[self.key(migrate(vec), seeds)] = \
                    row["result"] if "result" in row else _row_to_result(row)
                n += 1
        return n


def _row_to_result(row: dict) -> dict:
    """A log row is a *projection* of an evaluation; this is the inverse the cache needs.

    Only the fields selection reads are restored. `per_opponent` is kept when it is there because
    it costs nothing and is the first thing a human asks about a resumed elite.
    """
    return {
        "fitness": row.get("fitness"),
        "win_rate": row.get("win_rate"),
        "mean_margin": row.get("mean_margin"),
        "solo": row.get("solo"),
        "counters": row.get("counters") or {},
        "per_opponent": row.get("per_opponent") or {},
        "seeds": row.get("seeds"),
        "seconds": row.get("seconds", 0.0),
        "games": row.get("games", 0),
        "from_log": True,
    }


def log_row(path: str, gen: int, kind: str, vec, result: dict, violations,
            cached: bool, extra: dict | None = None) -> dict:
    """Append one evaluation. Every evaluation, including cached ones and CMA's inner samples —
    the log is the run's ground truth and a filtered log cannot answer "what did it try?"."""
    counters = result.get("counters") or {}
    row = {
        "gen": gen,
        "kind": kind,
        "vec": [round(float(v), 6) for v in snap(vec)],
        "fitness": result.get("fitness"),
        "score": score_of(result.get("fitness") or 0.0, violations),
        "guard": list(violations),
        "win_rate": result.get("win_rate"),
        "mean_margin": result.get("mean_margin"),
        "solo": result.get("solo"),
        "counters": {k: round(float(counter_value(counters, k)), 4) for k in LOG_COUNTERS
                     if counters},
        "fallbacks": counter_value(counters, "fallbacks") if counters else None,
        "per_opponent": {k: {"win_rate": v["win_rate"], "mean_margin": round(v["mean_margin"], 1)}
                         for k, v in (result.get("per_opponent") or {}).items()},
        "seeds": list(result.get("seeds") or []),
        "games": result.get("games", 0),
        "seconds": round(float(result.get("seconds") or 0.0), 3),
        # Wall clock, next to the evaluation's own duration. The two disagreeing is the signal
        # that the run is not getting the CPU it thinks it is: a suspended or throttled process
        # shows 5 s of evaluation spread across 90 s of wall, and without both numbers that is
        # indistinguishable from a search that has become slow.
        "t": round(time.time(), 3),
        "cached": bool(cached),
        # The candidate's fingerprint covers the *compiler source* as well as the plan
        # (`harness/registry._plan_agent`). Logging it is how a run notices that the tree changed
        # underneath it: two generations with different fingerprints were not scored by the same
        # agent, and comparing their fitness is comparing two experiments. Cheap, and the only
        # thing that makes a long search auditable after the fact.
        "fingerprint": result.get("fingerprint"),
    }
    if extra:
        row.update(extra)
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return row


# ------------------------------------------------------------------------ state

def save_state(path: str, gen: int, population, rng: random.Random, best: dict | None) -> None:
    """The whole resumable state: the next generation index, the population that enters it, and
    the RNG. Written atomically — a half-written state file after a kill is indistinguishable from
    a corrupt one, and `--resume` would silently start a *different* search."""
    st = rng.getstate()
    payload = {
        "gen": gen,
        "population": [[float(x) for x in snap(v)] for v in population],
        "rng": [st[0], list(st[1]), st[2]],
        "best": best,
        "version": 1,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def load_state(path: str) -> tuple[int, list, random.Random, dict | None]:
    with open(path) as fh:
        payload = json.load(fh)
    rng = random.Random()
    v, keys, gauss = payload["rng"]
    rng.setstate((v, tuple(int(k) for k in keys), gauss))
    return (int(payload["gen"]), [migrate(p) for p in payload["population"]], rng,
            payload.get("best"))
