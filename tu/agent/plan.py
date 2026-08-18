"""The plan genome: a whole season as 80 numbers (PLAN_v4 §2.1, TASKS_v4 C0/S3).

PLAN_v4's bet is that Boatlee found the right *structure* — decide the season up front, then execute
— and paid for it with zero adaptivity, while our reactive engines have adaptivity and no structure.
This module is the structure half: small enough to search (S2), explicit enough to compile (C1/C2),
and re-read every dawn against the true state (O1) rather than replayed.

**A plan says what to grow and where to put it, never how to walk.** Routing is the compiler's job
(C2) and is re-solved daily; nothing here encodes an action sequence. That is the whole difference
from Boatlee's 719-step table, and it is what lets the same plan survive a weed, a missed water or
an opponent flooding our market.

Two representations, one canonical:

* the **dataclasses** — what the compiler reads, with concrete tile coordinates;
* the **vector** — fixed-length, bounded, what the search mutates.

`decode` materialises tiles from counts through a deterministic layout, so `decode(encode(p)) == p`
for any plan the genome can express. Plans with hand-picked tiles are allowed (they are useful for
diagnostics) but `encode` refuses them rather than silently dropping the layout — a search that
quietly re-tiled the plan it was handed would be optimising something other than what it reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

BOARD = 10
HALF = BOARD // 2
SHED = (HALF - 1, HALF - 1)                       # (4, 4): the farmer's dawn spawn
DAYS = 30                                         # 720 steps / 24 turns

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
SPECIES = ("SHEEP", "COW", "GOOSE")
QUADRANTS = ("NW", "NE", "SW", "SE")
LAND_ORDER = ("NE", "SW", "SE")                   # `kaggriculture.py` LAND_ORDER, $1k/$2k/$4k
NEVER = 99                                        # a land_day meaning "do not buy"

#: Fertilizer timing is arithmetic, not a knob (PLAN_v4 §2.1): strawberry ticks at the end of ages
#: 9/11/13/15 and one application covers three days, so 9 and 13 double all four. Verified in
#: tests/test_env_facts.py; the search must not be allowed to "tune" it.
FERT_AGES = {"STRAWBERRY": (9, 13), "TOMATO": (7, 10), "MELON": (6,), "WHEAT": (2,)}

#: How many plantings a plan may name (S3's representation widening, E76).
#:
#: Six was the shipped width and it is not enough to *say* what Boatlee does. A cohort is
#: "N tiles of crop C in quadrant Q from day D", so a plan that carries all four quadrants with a
#: mix needs two to three slots per quadrant; at six the fourth quadrant is silently dropped by
#: `encode` (measured: a 10-cohort mixed plan truncated to 6 lost SE entirely). Ten is what the
#: measured max-production shapes need and no more: the widest mix that reaches the plant ceiling
#: uses 10, and every extra slot is five more GA dimensions.
#:
#: Extra slots are *empty* by default — `c{i}_tiles == 0` is skipped by `decode` and written as 0 by
#: `encode` — so every plan that predates the widening embeds unchanged (see `migrate`).
COHORT_SLOTS = 10

#: Products a plan may set a sell floor on, and the floor each behaves as when unset.
#:
#: One gene per sellable product rather than the two (wool, melon) the genome shipped with. The
#: harvest showed the search pushing hard on exactly the two dials it had — wool to 0.131, melon to
#: 0.695 — which is the signature of a constrained control set, not of a solved market.
#:
#: A floor of 0 means "sell on sight", which is `_sell_orders`' measured default (E11), so the new
#: products default to **0.0** and a plan written before the widening behaves identically. Wool and
#: melon keep their historical 0.35 so `boatlee_like` and the exploiters are untouched.
#:
#: FERTILIZER is deliberately absent. It is an *input* the compiler consumes (fertilize tasks, feed
#: reserve); a "floor" on it would be a hoarding rule wearing a market gene's name, and the reserve
#: logic already owns that decision.
FLOOR_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")

FLOOR_DEFAULTS = {p: (0.35 if p in ("WOOL", "MELON") else 0.0) for p in FLOOR_PRODUCTS}


def floor_gene(product: str) -> str:
    """The gene name carrying `product`'s sell floor. `WOOL` -> `sell_floor_wool`."""
    return f"sell_floor_{product.lower()}"

#: (species, count gene, start-day gene). One table so the three places that walk the herd genes
#: agree by construction rather than by three copies of the same string surgery.
SPECIES_GENES = (
    ("SHEEP", "n_sheep", "sheep_start"),
    ("COW", "n_cows", "cow_start"),
    ("GOOSE", "n_geese", "geese_start"),
)


def quadrant_of(x: int, y: int) -> str:
    return ("NW" if x < HALF else "NE") if y < HALF else ("SW" if x < HALF else "SE")


def quadrant_tiles(q: str) -> list[tuple[int, int]]:
    """Row-major tiles of one quadrant."""
    xs = range(0, HALF) if q in ("NW", "SW") else range(HALF, BOARD)
    ys = range(0, HALF) if q in ("NW", "NE") else range(HALF, BOARD)
    return [(x, y) for y in ys for x in xs]


def pasture_order(land_days: dict) -> list[tuple[int, int]]:
    """Candidate pasture tiles, nearest the shed first, **restricted to land the plan buys**.

    Keepers pay for distance twice a day — feed and care — so the herd belongs against the shed,
    and E46 showed unit *role purity* is worth real money, which needs the animals in one cluster
    rather than scattered.

    The land filter is not a detail: the ring one step from the shed reaches into all four
    quadrants, so an unfiltered cluster puts pastures on SE for a plan that never buys SE. Those
    animals can never be placed, and the plan would look fine until the compiler tried to build it.

    Ordered by **unlock day first**, distance second. Distance-first looks better on a map and is
    worse in practice: it puts the fourth pasture on a tile whose land is not bought until day 10,
    which silently pushes the fourth animal to day 10 — an animal deferred a third of the season to
    save one step of walking. Cheap land now beats close land later.
    """
    tiles = []
    for y in range(BOARD):
        for x in range(BOARD):
            q = quadrant_of(x, y)
            day = 0 if q == "NW" else land_days.get(q, NEVER)
            if day >= NEVER:
                continue
            tiles.append(((x, y), day))
    return [t for t, _ in sorted(
        tiles, key=lambda td: (td[1],
                               abs(td[0][0] - SHED[0]) + abs(td[0][1] - SHED[1]),
                               td[0][1], td[0][0]))]


def hands_curve(start: int, cap: int, ramp: int) -> tuple[int, ...]:
    """Hands per day as a straight ramp from `start` to `cap` over `ramp` days, then flat.

    Three numbers rather than thirty because the shape that matters is "staff up as the farm grows,
    then hold": Boatlee's measured curve is 4 hands on day 0 rising to 10-14 by day 10 and holding.
    The 14th hire costs $377/day, so the cap is a real economic choice and worth searching.
    """
    ramp = max(1, ramp)
    return tuple(int(round(start + (cap - start) * min(1.0, d / ramp))) for d in range(DAYS))


def pasture_available_day(tile, land_days: dict) -> int:
    """The first day a pasture can stand on this tile — i.e. when its quadrant is bought."""
    q = quadrant_of(*tile)
    return 0 if q == "NW" else int(land_days.get(q, NEVER))


@dataclass(frozen=True)
class Cohort:
    """One planting: a crop, a count of tiles in a quadrant, and the day it goes in.

    `replant` is what turns 141 wheat plantings into one gene: the compiler re-sows the tile when
    it is harvested, which is how a short crop cycles all season without the plan listing every
    round.
    """

    crop: str
    quadrant: str
    n_tiles: int
    plant_day: int
    replant: bool = False
    tiles: tuple[tuple[int, int], ...] = ()

    def with_tiles(self, tiles) -> "Cohort":
        return replace(self, tiles=tuple(tiles))


@dataclass(frozen=True)
class Branch:
    """A forward-only plan patch, applied when a condition first holds (O1).

    Carried here so the representation is stable before O1 lands; `Plan.branches` is empty until
    then. Patches must never dig a live plant or sell an animal — adaptivity changes what we do
    *next*, not what we already committed.
    """

    day_from: int
    condition: str
    patch: dict


@dataclass(frozen=True)
class Plan:
    pasture_tiles: tuple[tuple[int, int], ...]
    land_days: dict                                  # {"NE": 6, "SW": 10, "SE": 99}
    herd: tuple[tuple[str, int], ...]                # ((species, buy day), ...)
    cohorts: tuple[Cohort, ...]
    hands: tuple[int, ...] | str                     # per-day counts, or "auto" (C2 decides)
    consts: dict
    branches: tuple[Branch, ...] = ()

    #: What `decode` had to change to make the genome fit the board — a cohort that asked for more
    #: tiles than its quadrant had free, a herd bigger than the bought land. Excluded from equality
    #: (they describe the decoding, not the plan) but printed, because a search whose candidates
    #: are quietly being trimmed is exploring a different space than it reports.
    notes: tuple[str, ...] = field(default=(), compare=False)

    # ------------------------------------------------------------------ layout

    def occupied(self) -> dict:
        """`{(x, y): what}` for every tile the plan claims, pastures and cohorts alike."""
        out: dict = {}
        for t in self.pasture_tiles:
            out[tuple(t)] = "PASTURE"
        for i, c in enumerate(self.cohorts):
            for t in c.tiles:
                out[tuple(t)] = f"cohort{i}:{c.crop}"
        return out

    def unlocked_by(self, day: int) -> set:
        return {"NW"} | {q for q, d in self.land_days.items() if d <= day}

    # ------------------------------------------------------------------ checks

    def validate(self) -> list[str]:
        """Every way a plan can be impossible, as a list of reasons. Empty means it is buildable.

        Deliberately returns reasons rather than raising: the search generates plans by mutation and
        needs to know *why* one is invalid to repair it, and a silent auto-repair would let a
        genome drift away from what it reports.
        """
        errors: list[str] = []

        seen: dict = {}
        for owner, tiles in [("pasture", self.pasture_tiles)] + \
                            [(f"cohort{i}", c.tiles) for i, c in enumerate(self.cohorts)]:
            for t in tiles:
                t = tuple(t)
                if not (0 <= t[0] < BOARD and 0 <= t[1] < BOARD):
                    errors.append(f"{owner}: tile {t} is off the board")
                elif t in seen:
                    errors.append(f"{owner}: tile {t} already used by {seen[t]}")
                else:
                    seen[t] = owner

        for q, d in self.land_days.items():
            if q not in LAND_ORDER:
                errors.append(f"land_days: {q} is not purchasable")
            elif not (0 <= d <= NEVER):
                errors.append(f"land_days[{q}] = {d} out of range")
        # Land is bought in a fixed order at fixed prices, so buying SW before NE is not a plan the
        # environment can execute, whatever the days say.
        bought = [(self.land_days.get(q, NEVER), q) for q in LAND_ORDER]
        for (d1, q1), (d2, q2) in zip(bought, bought[1:]):
            if d2 < NEVER and d1 > d2:
                errors.append(f"land order: {q2} (day {d2}) cannot precede {q1} (day {d1})")

        for i, c in enumerate(self.cohorts):
            if c.crop not in CROPS:
                errors.append(f"cohort{i}: unknown crop {c.crop}")
            if not (0 <= c.plant_day < DAYS):
                errors.append(f"cohort{i}: plant_day {c.plant_day} outside the season")
            if len(c.tiles) != c.n_tiles:
                errors.append(f"cohort{i}: {len(c.tiles)} tiles resolved for n_tiles={c.n_tiles}")
            unlocked = self.unlocked_by(c.plant_day)
            for t in c.tiles:
                q = quadrant_of(*t)
                if q not in unlocked:
                    errors.append(
                        f"cohort{i}: tile {t} is in {q}, locked on day {c.plant_day}"
                        f" (bought day {self.land_days.get(q, NEVER)})")
                    break

        for species, day in self.herd:
            if species not in SPECIES:
                errors.append(f"herd: unknown species {species}")
            if not (0 <= day < DAYS):
                errors.append(f"herd: buy day {day} outside the season")
        # One structure per tile, one animal per structure. Geese live in a COOP and the others in
        # a PASTURE, but both are built on a cluster tile, so capacity is shared.
        pastured = sorted(d for _s, d in self.herd)
        if len(pastured) > len(self.pasture_tiles):
            errors.append(
                f"herd: {len(pastured)} animals for {len(self.pasture_tiles)} structure tiles")

        for t in self.pasture_tiles:
            q = quadrant_of(*t)
            if q != "NW" and self.land_days.get(q, NEVER) >= NEVER:
                errors.append(f"pasture: tile {tuple(t)} is in {q}, which the plan never buys")
                break
        # An animal bought before its pasture's land is a purchase with nowhere to stand: the
        # buy order fails against a full shed, and the animal sits there unfed.
        available = sorted(pasture_available_day(t, self.land_days) for t in self.pasture_tiles)
        for i, buy_day in enumerate(pastured):
            if i < len(available) and buy_day < available[i]:
                errors.append(
                    f"herd: animal {i + 1} bought on day {buy_day} but pasture {i + 1} is not "
                    f"available until day {available[i]}")
                break

        # Quadrant over-claim. Tile overlap is caught above, but only for tiles that were actually
        # resolved; a plan can also claim *more area than the quadrant has* — two 25-tile cohorts in
        # NE, or 20 tiles of melon in a quadrant already carrying 13 pastures. At six slots that was
        # barely reachable; at ten it is one mutation away, and `decode` silently trims it to what
        # fits. Named here so the trim is a reported repair rather than a plan quietly shrinking.
        claimed: dict = {}
        for t in self.pasture_tiles:
            claimed[quadrant_of(*t)] = claimed.get(quadrant_of(*t), 0) + 1
        for c in self.cohorts:
            claimed[c.quadrant] = claimed.get(c.quadrant, 0) + int(c.n_tiles)
        for q, n in sorted(claimed.items()):
            if n > HALF * HALF:
                errors.append(f"quadrant {q}: {n} tiles claimed, {HALF * HALF} exist")

        floors = (self.consts or {}).get("sell_floor") or {}
        for product, value in sorted(floors.items()):
            if product not in FLOOR_PRODUCTS:
                errors.append(f"sell_floor: {product} has no floor gene")
            elif not (0.0 <= float(value) <= 1.0):
                errors.append(f"sell_floor[{product}] = {value} outside 0..1")

        if self.hands != "auto":
            if len(self.hands) != DAYS:
                errors.append(f"hands: {len(self.hands)} entries for {DAYS} days")
            elif any(h < 0 or h > 14 for h in self.hands):
                errors.append("hands: outside 0..14 (the 14th hire costs $377)")
        return errors

    def is_valid(self) -> bool:
        return not self.validate()

    # ------------------------------------------------------------------ display

    def to_table(self) -> str:
        rows = [f"{'PLAN':<12}{len(self.pasture_tiles)} pastures, "
                f"{len(self.herd)} animals, {len(self.cohorts)} cohorts"]
        land = ", ".join(f"{q}@{self.land_days.get(q, NEVER)}" if self.land_days.get(q, NEVER) < NEVER
                         else f"{q}=never" for q in LAND_ORDER)
        rows.append(f"{'land':<12}{land}")

        herd_by = {}
        for s, d in self.herd:
            herd_by.setdefault(s, []).append(d)
        rows.append(f"{'herd':<12}" + ", ".join(
            f"{len(days)}x{s} days {min(days)}-{max(days)}" for s, days in sorted(herd_by.items())))

        rows.append(f"{'cohorts':<12}" + ("" if self.cohorts else "(none)"))
        for i, c in enumerate(self.cohorts):
            flag = " replant" if c.replant else ""
            fert = FERT_AGES.get(c.crop)
            rows.append(f"  [{i}] {c.crop:<11}{c.n_tiles:>3} tiles in {c.quadrant}"
                        f"  day {c.plant_day:<3}{flag:<9}"
                        f"fert@{','.join(map(str, fert)) if fert else '-'}")

        hands = ("auto" if self.hands == "auto"
                 else f"{min(self.hands)}-{max(self.hands)} (day 0: {self.hands[0]})")
        rows.append(f"{'hands':<12}{hands}")
        rows.append(f"{'consts':<12}" + ", ".join(f"{k}={v}" for k, v in sorted(self.consts.items())))
        if self.branches:
            rows.append(f"{'branches':<12}{len(self.branches)}")
        for note in self.notes:
            rows.append(f"{'note':<12}{note}")
        errors = self.validate()
        rows.append(f"{'valid':<12}" + ("yes" if not errors else f"NO — {errors[0]}"))
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.to_table()

    # ------------------------------------------------------------------ canonical plans

    @staticmethod
    def boatlee_like() -> "Plan":
        """A plan with Boatlee's measured shape — the Phase-1 calibration case (PLAN_v4 §2.3).

        Every number here was read off a played season rather than taken from prose (seed 3,
        `probe_boatlee_plan`): land NE day 6 / SW day 10 / never SE; 13 pastures grown outward from
        the shed; 4 sheep and 9 cows; and crops 141 WHEAT / 36 STRAWBERRY / 19 MELON — the wheat
        being ~20 tiles cycled all season rather than 141 separate decisions.

        It is the *shape*, not a replay. If our compiler cannot get near Boatlee's score from
        Boatlee's own shape, the compiler is the bug and not the plan — which is exactly the
        question Phase 1 exists to answer.
        """
        return decode(encode_fields(
            n_pastures=13,
            land_days={"NE": 6, "SW": 10, "SE": NEVER},
            n_sheep=4, sheep_start=0,
            # `cow_start` is read off the trace's *first* cow, not its bulk: Boatlee's opening
            # market turn is `HIRE x4, BUY_ANIMAL COW 1, BUY_ANIMAL SHEEP 4`
            # (`traces/boatlee_vs_champion_s5.json`, day 0 hour 0) and the remaining cows follow on
            # days 5-8. The old 5 encoded only the bulk, and the genome's `animals_per_day` pacing
            # then held every cow behind four sheep: measured first COW on day 11 against a first
            # animal on day 2 (E66 re-gate). Day 1 is the earliest this genome can say "cows as soon
            # as the wallet allows" — and the wallet, not the calendar, is what actually paces them.
            # Measured +$3.5k +/- 2.9k over `cow_start=5` on 400 paired games; day 0 is *worse*
            # (three cows plus four sheep on the opening turn starves the seed budget: -$72k), and
            # day 2 is indistinguishable (-$0.2k +/- 3.6k).
            n_cows=9, cow_start=1,
            n_geese=0, geese_start=0,
            animals_per_day=3,
            # Sized against what each quadrant actually has free once the pastures are placed:
            # 13 pastures take NW down to 12 tiles, NE opens on day 6 with 25, SW on day 10.
            cohorts=[
                # strawberry: the money crop at 36 tiles, in two waves as each quadrant opens
                ("STRAWBERRY", "NE", 22, 7, False),
                ("STRAWBERRY", "SW", 14, 10, False),
                # melon: 13 tiles (Boatlee runs 19; the rest of its melon rides on tiles freed
                # mid-season, which is the compiler's business to reuse, not the plan's to claim)
                ("MELON", "NW", 5, 0, False),
                ("MELON", "SW", 8, 11, False),
                # wheat: cycled, which is what makes 141 plantings ~10 tiles
                ("WHEAT", "NW", 7, 0, True),
                ("WHEAT", "NE", 3, 6, True),
            ],
            hands_mode="curve", hands_start=4, hands_cap=12, hands_ramp=10,
            sell_floor_wool=0.35, sell_floor_melon=0.35,
            release_pressure=70, frontrun_lead=10,
        ))


# --------------------------------------------------------------------------- genome

@dataclass(frozen=True)
class Gene:
    name: str
    lo: float
    hi: float
    integral: bool = True


#: The search space. Order is the vector order and must stay stable — a saved genome is only
#: meaningful against the layout that produced it.
GENES: tuple[Gene, ...] = (
    Gene("n_pastures", 0, 24),
    Gene("land_NE", 0, NEVER),
    Gene("land_SW", 0, NEVER),
    Gene("land_SE", 0, NEVER),
    Gene("n_sheep", 0, 12),
    Gene("sheep_start", 0, DAYS - 1),
    Gene("n_cows", 0, 14),
    Gene("cow_start", 0, DAYS - 1),
    Gene("n_geese", 0, 12),
    Gene("geese_start", 0, DAYS - 1),
    Gene("animals_per_day", 1, 4),
) + tuple(
    g for i in range(COHORT_SLOTS) for g in (
        Gene(f"c{i}_crop", 0, len(CROPS) - 1),
        Gene(f"c{i}_quad", 0, len(QUADRANTS) - 1),
        Gene(f"c{i}_tiles", 0, 40),
        Gene(f"c{i}_day", 0, DAYS - 1),
        Gene(f"c{i}_replant", 0, 1),
    )
) + (
    Gene("hands_auto", 0, 1),
    Gene("hands_start", 0, 14),
    Gene("hands_cap", 0, 14),
    Gene("hands_ramp", 1, DAYS - 1),
) + tuple(
    Gene(floor_gene(p), 0.0, 1.0, integral=False) for p in FLOOR_PRODUCTS
) + (
    Gene("release_pressure", 0, 100),
    Gene("frontrun_lead", 0, 24),
    # --- C6: the scarcity hunter. Thresholds are in `T` units, the same units the price curve is
    # written in, so 1.0 means "the town has eaten a whole field's output more than the board has
    # produced". `hinge` products (carrot/tomato/egg) are calm below the knee and run away above it,
    # so they get their own gene; `sqrt`/`linear` products pay smoothly and are worth chasing sooner.
    Gene("theta_hinge", 0.0, 3.0, integral=False),
    Gene("theta_sqrt", 0.0, 3.0, integral=False),
    Gene("max_scarce_tiles", 0, 24),
    Gene("scarce_floor", 0.0, 1.0, integral=False),
    Gene("projected_pricing", 0, 1),
)

GENE_INDEX = {g.name: i for i, g in enumerate(GENES)}
BOUNDS = tuple((g.lo, g.hi) for g in GENES)


def snap(vector) -> list[float]:
    """Clip to bounds and round the integral genes: the canonical form of a vector.

    Search operators produce arbitrary floats; this is where they become a plan the game could
    actually execute, once, in one place.
    """
    # `zip` would silently truncate. A 54-gene vector from the S2 harvest read through the widened
    # layout is not a shorter plan, it is a *different* one — and a search that keyed its cache on
    # the truncation would look like it was working. Say so instead, and name the way out.
    vector = list(vector)
    if len(vector) != len(GENES):
        raise ValueError(f"{len(vector)} genes for a {len(GENES)}-gene layout; "
                         f"use plan.migrate() on a genome from an older layout")
    out = []
    for g, v in zip(GENES, vector):
        v = min(g.hi, max(g.lo, float(v)))
        out.append(float(round(v)) if g.integral else v)
    return out


def encode_fields(**kw) -> list[float]:
    """Build a vector from named values — the readable way to write a plan by hand."""
    v = [0.0] * len(GENES)

    def put(name, value):
        v[GENE_INDEX[name]] = float(value)

    put("n_pastures", kw["n_pastures"])
    for q in LAND_ORDER:
        put(f"land_{q}", kw["land_days"].get(q, NEVER))
    for _species, count_gene, start_gene in SPECIES_GENES:
        put(count_gene, kw[count_gene])
        put(start_gene, kw[start_gene])
    put("animals_per_day", kw["animals_per_day"])

    for i, c in enumerate(kw["cohorts"][:COHORT_SLOTS]):
        crop, quad, n, day, replant = c
        put(f"c{i}_crop", CROPS.index(crop))
        put(f"c{i}_quad", QUADRANTS.index(quad))
        put(f"c{i}_tiles", n)
        put(f"c{i}_day", day)
        put(f"c{i}_replant", 1 if replant else 0)

    put("hands_auto", 1 if kw.get("hands_mode") == "auto" else 0)
    put("hands_start", kw.get("hands_start", 4))
    put("hands_cap", kw.get("hands_cap", 12))
    put("hands_ramp", kw.get("hands_ramp", 10))
    for product, default in FLOOR_DEFAULTS.items():
        put(floor_gene(product), kw.get(floor_gene(product), default))
    put("release_pressure", kw.get("release_pressure", 70))
    put("frontrun_lead", kw.get("frontrun_lead", 10))
    for name, default in C6_DEFAULTS.items():
        put(name, kw.get(name, default))
    return snap(v)


#: C6's genes and the values a plan that predates them behaves as. Kept next to `encode_fields` so
#: the default a hand-written plan gets and the default the genome writes are one number, not two.
C6_DEFAULTS = {
    "theta_hinge": 1.0,
    "theta_sqrt": 0.5,
    "max_scarce_tiles": 8,
    "scarce_floor": 0.0,
    "projected_pricing": 0,
}


def decode(vector) -> Plan:
    """Vector -> Plan, materialising tiles through the canonical layout.

    Tile assignment is deterministic and greedy: pastures take the cluster nearest the shed, then
    cohorts fill their quadrant in row order, skipping anything already claimed. A cohort that
    cannot fit shrinks to what is available rather than overlapping — an overlapping plan is not a
    plan, and `validate` would only report it after the fact.
    """
    v = snap(vector)

    def g(name):
        return v[GENE_INDEX[name]]

    land_days = {q: int(g(f"land_{q}")) for q in LAND_ORDER}
    land_days = _monotone_land(land_days)

    notes: list[str] = []
    n_pastures = int(g("n_pastures"))
    taken: set = set()
    pastures = []
    for t in pasture_order(land_days):
        if len(pastures) >= n_pastures:
            break
        pastures.append(t)
        taken.add(t)
    if len(pastures) < n_pastures:
        notes.append(f"pastures: {n_pastures} requested, {len(pastures)} fit on the bought land")

    # Animals need somewhere to stand, and the cluster is shared: one structure per tile, built as
    # a PASTURE or a COOP depending on which species claims it (the compiler's choice, not the
    # plan's). A genome asking for 22 animals on 6 tiles is repaired by keeping what fits, species
    # by species, rather than by shifting days — dropping a suffix leaves each species' (count,
    # start) readable straight back out, so the repair survives re-encoding unchanged.
    available = sorted(pasture_available_day(t, land_days) for t in pastures)
    herd: list[tuple[str, int]] = []
    per_day = max(1, int(g("animals_per_day")))
    for species, count_gene, start_gene in SPECIES_GENES:
        count = int(g(count_gene))
        start = int(g(start_gene))
        kept = 0
        for k in range(count):
            day = min(DAYS - 1, start + k // per_day)
            slot = len(herd)
            if slot >= len(available) or day < available[slot]:
                break                      # no structure standing by then: stop this species here
            herd.append((species, day))
            kept += 1
        if kept < count:
            notes.append(f"herd: {count} {species} requested, {kept} had a structure in time")
    herd.sort(key=lambda sd: (sd[1], sd[0]))

    cohorts = []
    for i in range(COHORT_SLOTS):
        n = int(g(f"c{i}_tiles"))
        if n <= 0:
            continue
        crop = CROPS[int(g(f"c{i}_crop"))]
        quad = QUADRANTS[int(g(f"c{i}_quad"))]
        day = int(g(f"c{i}_day"))

        # A cohort cannot be planted on land the plan has not bought yet. Mutation produces this
        # constantly — "20 melon in SE on day 4" when SE is bought on day 22, or never — so it is
        # repaired rather than rejected: the day slides to the purchase, and a cohort on land the
        # plan never buys is dropped. Left alone it produced plans that `validate` rejected, which
        # would have had the search spending its budget generating candidates it must throw away.
        opens = 0 if quad == "NW" else land_days.get(quad, NEVER)
        if opens >= DAYS:
            notes.append(f"cohort{i}: {crop} in {quad} dropped, that land is never bought")
            continue
        if day < opens:
            notes.append(f"cohort{i}: {crop} in {quad} planted day {day} -> {opens}, when {quad} opens")
            day = opens

        tiles = []
        for t in quadrant_tiles(quad):
            if len(tiles) >= n:
                break
            if t not in taken:
                tiles.append(t)
                taken.add(t)
        if not tiles:
            # No room at all: the cohort does not exist, rather than existing with zero tiles.
            # A zero-tile cohort would survive encode/decode as a phantom the compiler must skip.
            notes.append(f"cohort{i}: {crop} in {quad} dropped, no free tiles")
            continue
        if len(tiles) < n:
            notes.append(f"cohort{i}: {crop} in {quad} requested {n} tiles, {len(tiles)} were free")
        cohorts.append(Cohort(crop=crop, quadrant=quad, n_tiles=len(tiles), plant_day=day,
                              replant=bool(int(g(f"c{i}_replant"))), tiles=tuple(tiles)))

    hands: tuple[int, ...] | str
    curve = None
    if int(g("hands_auto")):
        hands = "auto"
    else:
        start, cap, ramp = int(g("hands_start")), int(g("hands_cap")), max(1, int(g("hands_ramp")))
        curve = (start, cap, ramp)
        hands = hands_curve(*curve)

    consts = {
        "fert_ages": FERT_AGES,
        "care": True,
        # The three genes the curve came from, kept because the curve cannot be inverted: rounding
        # makes a shallow ramp reach its cap early, so (12, 13, 20) and (12, 13, 10) both start
        # reading 13 on the same day. Storing beats guessing.
        **({"hands_curve": curve} if curve else {}),
        # Same reason: pacing is applied *per species*, so a schedule with sheep and cows arriving
        # on the same days reads back as twice the pacing, and a schedule clamped at the last day
        # reads back as a crowd. Both produced a different herd on the way home.
        "animals_per_day": per_day,
        # Every product gets a key, including the ones sitting at 0.0: `_sell_orders` reads
        # `floor <= 0` as "sell on sight", so a zero key and a missing key are the same order list.
        # Writing them all is what makes the round-trip exact.
        "sell_floor": {p: round(g(floor_gene(p)), 3) for p in FLOOR_PRODUCTS},
        "release_pressure": int(g("release_pressure")),
        "frontrun_lead": int(g("frontrun_lead")),
        # C6. Rounded to the same precision `encode` writes back, so the round-trip is exact.
        "theta_hinge": round(g("theta_hinge"), 3),
        "theta_sqrt": round(g("theta_sqrt"), 3),
        "max_scarce_tiles": int(g("max_scarce_tiles")),
        "scarce_floor": round(g("scarce_floor"), 3),
        "projected_pricing": int(g("projected_pricing")),
    }
    return Plan(pasture_tiles=tuple(pastures), land_days=land_days, herd=tuple(herd),
                cohorts=tuple(cohorts), hands=hands, consts=consts, notes=tuple(notes))


def _monotone_land(land_days: dict) -> dict:
    """Land is sold in a fixed order, so later quadrants cannot be bought earlier.

    Repaired here rather than rejected: mutation will constantly produce `SW` before `NE`, and a
    search that threw away those candidates would be searching a smaller space than it thinks.
    """
    out = {}
    prev = 0
    skipped = False
    for q in LAND_ORDER:
        d = int(land_days.get(q, NEVER))
        # `_do_buy_land` indexes LAND_ORDER by how much extra land you already own, so a quadrant
        # can only be bought after the one before it. Skipping NE does not make SW cheaper — it
        # makes SW unreachable.
        if skipped or d >= NEVER:
            skipped = True
            out[q] = NEVER
            continue
        d = max(d, prev)
        prev = d
        out[q] = d
    return out


def encode(plan: Plan) -> list[float]:
    """Plan -> vector. Raises if the plan is not something the genome can express.

    Refusing is the point: a plan with hand-picked tiles or a bespoke hand curve would come back
    from `decode` re-tiled, and a search reporting scores for plans it silently rewrote is the
    E44 failure mode with extra steps.
    """
    v = [0.0] * len(GENES)

    def put(name, value):
        v[GENE_INDEX[name]] = float(value)

    canonical_pastures = pasture_order(plan.land_days)[:len(plan.pasture_tiles)]
    if list(map(tuple, plan.pasture_tiles)) != canonical_pastures:
        raise ValueError("pasture_tiles are not the canonical cluster; this plan is not encodable")

    put("n_pastures", len(plan.pasture_tiles))
    for q in LAND_ORDER:
        put(f"land_{q}", plan.land_days.get(q, NEVER))

    for species, count_gene, start_gene in SPECIES_GENES:
        days = sorted(d for s, d in plan.herd if s == species)
        put(count_gene, len(days))
        put(start_gene, days[0] if days else 0)
    put("animals_per_day", plan.consts.get("animals_per_day") or _infer_per_day(plan.herd))

    if len(plan.cohorts) > COHORT_SLOTS:
        raise ValueError(f"{len(plan.cohorts)} cohorts exceeds {COHORT_SLOTS} slots")
    for i, c in enumerate(plan.cohorts):
        put(f"c{i}_crop", CROPS.index(c.crop))
        put(f"c{i}_quad", QUADRANTS.index(c.quadrant))
        put(f"c{i}_tiles", c.n_tiles)
        put(f"c{i}_day", c.plant_day)
        put(f"c{i}_replant", 1 if c.replant else 0)

    if plan.hands == "auto":
        put("hands_auto", 1)
        put("hands_start", 4)
        put("hands_cap", 12)
        put("hands_ramp", 10)
    else:
        curve = plan.consts.get("hands_curve")
        if curve is None:
            # A hand-written curve. `cap` is where the curve *ends*, not its maximum — a ramp-down
            # plan (start 12, cap 3) peaks on day 0 — and the ramp is recoverable only when the
            # slope is steep enough that no earlier day rounds to the cap.
            start, cap = plan.hands[0], plan.hands[-1]
            curve = (start, cap, max(1, next((d for d, h in enumerate(plan.hands) if h == cap), 1)))
        if hands_curve(*curve) != tuple(plan.hands):
            raise ValueError("hands curve is not expressible as (start, cap, ramp)")
        put("hands_auto", 0)
        put("hands_start", curve[0])
        put("hands_cap", curve[1])
        put("hands_ramp", curve[2])

    floors = plan.consts.get("sell_floor", {})
    unknown = sorted(set(floors) - set(FLOOR_PRODUCTS))
    if unknown:
        # A floor on a product the genome has no gene for would be dropped silently, and the plan
        # would come back from `decode` selling something it meant to hold — `encode`'s whole
        # contract is that it refuses rather than rewrites.
        raise ValueError(f"sell_floor on unrepresentable product(s) {unknown}")
    for product, default in FLOOR_DEFAULTS.items():
        put(floor_gene(product), floors.get(product, default))
    put("release_pressure", plan.consts.get("release_pressure", 70))
    put("frontrun_lead", plan.consts.get("frontrun_lead", 10))
    for name, default in C6_DEFAULTS.items():
        put(name, plan.consts.get(name, default))
    return snap(v)


def _infer_per_day(herd) -> int:
    """The pacing gene: the most animals the plan buys on any one day."""
    if not herd:
        return 1
    counts: dict = {}
    for _s, d in herd:
        counts[d] = counts.get(d, 0) + 1
    return max(1, min(4, max(counts.values())))


# --------------------------------------------------------------------------- migration

#: The 54-gene layout the S2 harvest ran on (6 cohort slots, wool/melon floors only). Kept verbatim
#: because a saved genome is only meaningful against the layout that produced it: `search/log.jsonl`,
#: `search/state.json` and the harvested candidate are all lists of 54 bare floats with no header,
#: so the only way to read one back is to know what its coordinates were called.
LAYOUT_V54: tuple[str, ...] = (
    "n_pastures", "land_NE", "land_SW", "land_SE",
    "n_sheep", "sheep_start", "n_cows", "cow_start", "n_geese", "geese_start",
    "animals_per_day",
) + tuple(
    f"c{i}_{k}" for i in range(6) for k in ("crop", "quad", "tiles", "day", "replant")
) + (
    "hands_auto", "hands_start", "hands_cap", "hands_ramp",
    "sell_floor_wool", "sell_floor_melon", "release_pressure", "frontrun_lead",
    "theta_hinge", "theta_sqrt", "max_scarce_tiles", "scarce_floor", "projected_pricing",
)

#: Every historical layout, by length. `len(GENES)` is handled by identity.
LAYOUTS: dict = {len(LAYOUT_V54): LAYOUT_V54}


def migrate(vector) -> list[float]:
    """An old genome in the current layout, behaviour unchanged.

    Embedding, not translation: every gene of the old layout exists in the new one under the same
    name, and every gene the old layout lacked has a **neutral** value — an empty cohort slot
    (`tiles = 0`, skipped by `decode`) or an absent sell floor (`0.0`, "sell on sight", which is what
    a plan with no floor for that product already did). So the migrated vector decodes to the same
    `Plan` behaviour as the original did under the layout that produced it.

    Zeros are that neutral value by construction, which is why this is a scatter into a zero vector
    rather than a merge with a default plan — there is no default to disagree with.
    """
    vector = [float(x) for x in vector]
    if len(vector) == len(GENES):
        return snap(vector)
    names = LAYOUTS.get(len(vector))
    if names is None:
        raise ValueError(f"no known gene layout has {len(vector)} genes "
                         f"(current is {len(GENES)}, known: {sorted(LAYOUTS)})")
    out = [0.0] * len(GENES)
    for name, value in zip(names, vector):
        out[GENE_INDEX[name]] = value
    return snap(out)


def random_vector(rng) -> list[float]:
    """A random point in the space — the search's starting population (S2)."""
    return snap([rng.uniform(g.lo, g.hi) for g in GENES])
