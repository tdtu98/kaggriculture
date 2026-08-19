"""O1 — shop-draw branch points: forward-only plan patches, applied when a condition first holds.

C4 re-compiles every dawn, so adaptivity does not need a policy — it needs a way to say "if the
town turns out to want wool, the next two animals are sheep". That is a `Branch`: a window of days,
a condition on the town's unlocked shop *instances*, and a patch on the working plan. The patch is
applied **once**, the first day the condition holds inside the window, and the patched plan is what
every later dawn compiles.

**Why the town is worth branching on.** The shop table is the only public, per-seed thing that moves
sustained demand for a product (`kaggriculture.py:103-112`):

    BAKERY         EGG, WHEAT                          PIZZA_SHOP     MILK, TOMATO, WHEAT
    BRUNCH_SPOT    EGG, WHEAT, STRAWBERRY              YARN_STORE     WOOL            (single)
    ICE_CREAM_SHOP STRAWBERRY, MILK, WHEAT             PET_CAFE       CARROT          (single)
    SMOOTHIE_SHOP  STRAWBERRY, MILK                    FARMERS_MARKET WHEAT, CARROT, TOMATO, STRAWBERRY

One instance eats 6 units/day of each of its products, doubled to 12 for a single-product shop
(`:733,736,741`); the town centre eats 1/day of everything but fertilizer (`:734,745-747`). An
instance is appended at the end of day `d` when `(d+1) % 3 == 0`, drawn **with replacement** from
the eight names and capped at 8, so the count during day `D` is `min(8, D // 3)`
(`:867,886-891`) and duplicates are real: three Pet Cafes is a normal draw, not a freak one.

That makes the draw a *market structure* fact, not a flavour one. WOOL has `T = 105` and no demand
at all unless YARN_STORE is drawn, at which point it gains 12/day. MILK has `T = 122` and three
possible demanders; a town with none of them regenerates milk at 1/day, and nine cows then sell
into a market that cannot recover. STRAWBERRY has `T = 100` and four demanders. Those three are the
branches this module ships.

**Forward-only, enforced rather than intended.** A patch may never dig a live plant or sell an
animal. Two rules make that checkable instead of aspirational:

* a cohort whose `plant_day <= today` is in the ground and is immutable — patches may only append
  cohorts, or edit ones not yet sown;
* the herd is only editable **beyond the purchase frontier**, `owned + animals_per_day`. `_paced_plan`
  releases the herd as a prefix capped at `owned + lead`, so every entry at or past that index is
  provably still unbought whatever the wallet did. Nothing before it is touched, and pasture tiles
  are never removed (a structure that stands, stays).

`forward_only_errors` is the gate, and `apply` refuses any patch that trips it or that fails
`Plan.validate()` — an impossible patch is counted (`branch_rejected`) and dropped, never repaired.

**Everything here is off by default.** `decode` never writes `branch_set`, so every stored genome
and `Plan.boatlee_like()` get `active(plan) == ()` and `apply` hands the plan straight back — the
same object, not a copy. Proven rather than argued: 40 games (10 seeds x 2 seats x starter/boatlee)
with the hook in and with the hook physically deleted, **40/40 byte-identical money**.

## What the gate measured (fresh block 91000:91300 vs `starter`, 91000:91200 vs `boatlee`)

Paired on the identical (seed, seat) schedule, split by whether the branch **actually fired**, which
is the exact condition-true set rather than a predicted one. Dormant games were **byte-identical in
every arm of every run** (2,321 of them) — the branch is free when it does not fire.

| branch | fires | compiler vs starter | compiler vs boatlee | R1 vs starter |
|---|---|---|---|---|
| `yarn_sheep`   | 46% | +$1,518 [−78, +3,114] | **+$4,224 [+2,342, +6,106]** | never fires |
| `straw_cohort` | 30% | **+$3,039 [+1,364, +4,715]** | −$434 [−2,617, +1,748] | **−$2,860 [−4,178, −1,542]** |
| `milk_cap`     | 25% | +$312 [−1,945, +2,569] | **+$3,286 [+1,504, +5,067]** | never fires |
| all three      | 75% | **+$2,845 [+1,723, +3,968]** | **+$2,056 [+629, +3,484]** | −$2,860 (straw only) |

(The "all three" row is a *fresh* block again, 93000:93300 / 93000:93200 — E43's rule that a
conjunction is measured on ground the parts did not use. It reproduced both single-branch signs.)

**The headline is that the sign is a property of the plan, not of the game.** `straw_cohort` is the
compiler's best branch and R1's worst, by CI, in the same direction the market arithmetic predicts:
`boatlee_like` has three free tiles left and 36 strawberry planted, R1 has six free tiles and the
same 36, and `T = 100`. Three more tiles is a top-up; six more is a flood, and D17's rule is about
*sustained* capacity while the price curve still decides what one batch fetches (E48). By the same
mechanism `yarn_sheep` and `milk_cap` **never fire at all** on R1: its `animals_per_day` is 4 and its
twelve head are due by day 3, so the purchase frontier has swallowed the whole herd before either
window opens. A herd branch needs a plan that paces its animals.

So there is no branch set to switch on globally. `branch_set=all` is the measured recommendation
**for the incumbent compiler** and for nothing else; a new genome has to be re-measured, which is
what `branch_set` being a `consts` flag rather than a gene is for.
"""

from __future__ import annotations

from dataclasses import replace

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, SHOPS

from agent.plan import (
    NEVER,
    Branch,
    Cohort,
    quadrant_of,
    quadrant_tiles,
)
from agent.projection import SHOP_DRAIN_PER_DAY

#: `{product: (shop names that demand it, ...)}` — derived from the env's own `SHOPS`, never typed
#: out. PET_CAFE sells CARROT and YARN_STORE sells WOOL; the names are not a reliable guide (the
#: task text had PET_CAFE down as eggs), which is why this is a comprehension over the source table.
DEMANDERS = {
    product: tuple(sorted(s for s, items in SHOPS.items() if product in items))
    for product in sorted({p for items in SHOPS.values() for p in items})
}

#: Products a single instance of each shop eats per day, reused from C6 rather than re-derived
#: (E39: a comparison against a re-implementation of a rule measures the re-implementation).
DRAIN = SHOP_DRAIN_PER_DAY


# --------------------------------------------------------------------------- the town, observed

def shop_instances(obs) -> list:
    """The town's unlocked shop instances, duplicates included (draws are with replacement)."""
    return list((obs.get("town", {}) or {}).get("unlocked_shops") or [])


def count_shop(obs, shop: str) -> int:
    return shop_instances(obs).count(shop)


def demand_instances(obs, product: str) -> int:
    """How many unlocked instances demand `product`. Instances, not distinct shops."""
    wanted = set(DEMANDERS.get(product, ()))
    return sum(1 for s in shop_instances(obs) if s in wanted)


def demand_per_day(obs, product: str) -> int:
    """Units of `product` the town's shops eat per day — the number the branches are sized on.

    A single-product shop counts double (`kaggriculture.py:741`), so one YARN_STORE is 12 wool/day
    where one SMOOTHIE_SHOP is 6 milk/day. The town centre's 1/day is deliberately *not* included:
    it is present in every town and so cannot distinguish one draw from another.
    """
    return sum(DRAIN.get(s, {}).get(product, 0) for s in shop_instances(obs))


# --------------------------------------------------------------------------- conditions

def _parse(condition: str) -> tuple:
    """`"demand>=2:STRAWBERRY"` -> `("demand", ">=", 2, "STRAWBERRY")`.

    Three forms, all reading the town and nothing else:
      * `has:<SHOP>`            — at least one instance of that shop name
      * `demand<op><n>:<PROD>`  — instance count demanding a product, `op` in >= <= ==
      * `drain<op><n>:<PROD>`   — units/day the shops eat, for thresholds quoted in units
    """
    head, _, arg = condition.partition(":")
    head = head.strip()
    arg = arg.strip().upper()
    if head == "has":
        return ("has", ">=", 1, arg)
    for kind in ("demand", "drain"):
        if head.startswith(kind):
            rest = head[len(kind):]
            for op in (">=", "<=", "==", "<", ">"):
                if rest.startswith(op):
                    return (kind, op, int(rest[len(op):]), arg)
    raise ValueError(f"unparseable branch condition {condition!r}")


def _compare(value: int, op: str, threshold: int) -> bool:
    return {">=": value >= threshold, "<=": value <= threshold, "==": value == threshold,
            "<": value < threshold, ">": value > threshold}[op]


def holds(obs, condition: str) -> bool:
    """Does `condition` hold against the town as observed right now?"""
    kind, op, threshold, arg = _parse(condition)
    if kind == "has":
        if arg not in SHOPS:
            raise ValueError(f"unknown shop {arg!r}")
        return _compare(count_shop(obs, arg), op, threshold)
    if kind == "demand":
        return _compare(demand_instances(obs, arg), op, threshold)
    return _compare(demand_per_day(obs, arg), op, threshold)


# --------------------------------------------------------------------------- the purchase frontier

def owned_animals(obs, seat: int) -> int:
    """Animals the farm actually has — on the board or sitting in the shed.

    The same count `_paced_plan` uses, deliberately: the frontier below is only sound because it is
    derived from the identical quantity that gates the release of the herd.
    """
    farm = obs["farms"][seat]
    shed = (obs.get("private", {}) or {}).get("shed", {}) or {}
    n = sum(1 for row in farm["tiles"] for t in row
            if isinstance(t, dict) and t.get("animal"))
    return n + sum(int(shed.get(s, 0) or 0) for s in ANIMALS)


def frontier(obs, plan, seat: int) -> int:
    """First herd index that is **provably unbought**.

    `_paced_plan` hands the compiler `herd[:owned + lead]` and never more, so an entry at index
    `owned + lead` or beyond has never been offered to `_animal_orders` on any turn so far — whatever
    the wallet, whatever the re-sort `_purchase_order` applies inside the released prefix. Editing
    from here on is forward-only by construction rather than by inspection.
    """
    lead = max(1, int((plan.consts or {}).get("animals_per_day", 1) or 1))
    return owned_animals(obs, seat) + lead


# --------------------------------------------------------------------------- forward-only gate

def forward_only_errors(before, after, day: int, frontier_index: int) -> list:
    """Every way `after` is not a forward-only patch of `before`, as reasons.

    Returns a list for the same reason `Plan.validate` does: a caller that wants to count *why* a
    patch was refused should not have to parse an exception message.
    """
    errors: list = []

    live = {i: c for i, c in enumerate(before.cohorts) if c.plant_day <= day}
    for i, c in live.items():
        if i >= len(after.cohorts):
            errors.append(f"cohort{i}: {c.crop} planted on day {c.plant_day} was removed")
        elif after.cohorts[i] != c:
            errors.append(f"cohort{i}: {c.crop} planted on day {c.plant_day} was edited")
    # A pending cohort may be edited, but not by taking tiles a *sown* cohort is standing on.
    sown_tiles = {t for c in live.values() for t in c.tiles}
    for i, c in enumerate(after.cohorts):
        if i in live:
            continue
        stolen = sown_tiles.intersection(c.tiles)
        if stolen:
            errors.append(f"cohort{i}: {c.crop} would re-sow live ground at {sorted(stolen)[0]}")

    if tuple(after.herd[:frontier_index]) != tuple(before.herd[:frontier_index]):
        errors.append(f"herd: entries below the purchase frontier ({frontier_index}) were changed")

    kept = set(map(tuple, after.pasture_tiles))
    for t in before.pasture_tiles:
        if tuple(t) not in kept:
            errors.append(f"pasture: tile {tuple(t)} was removed")
            break

    if after.land_days != before.land_days:
        errors.append("land_days: a branch may not re-time land")
    return errors


# --------------------------------------------------------------------------- patches

def _free_tiles(plan, quad: str, n: int) -> list:
    taken = set(plan.occupied())
    out = []
    for t in quadrant_tiles(quad):
        if len(out) >= n:
            break
        if t not in taken:
            out.append(t)
    return out


def _patch_swap_species(obs, plan, seat: int, day: int, spec: dict, count_fn) -> object:
    """The next `count` unbought animals of `from` become `to`.

    Restricted to species that live in the same structure: `_structure_tasks` builds slot k's
    structure from the plan's own species, so a COOP bought against a PASTURE slot would leave the
    animal in the shed. COW and SHEEP share PASTURE, which is the swap this branch needs.
    """
    src, dst = spec.get("from", "COW"), spec.get("to", "SHEEP")
    if ANIMALS[src]["structure"] != ANIMALS[dst]["structure"]:
        return None
    start = frontier(obs, plan, seat)
    want = int(spec.get("count", 2))
    herd = list(plan.herd)
    changed = 0
    for i in range(start, len(herd)):
        if changed >= want:
            break
        if herd[i][0] == src:
            herd[i] = (dst, herd[i][1])
            changed += 1
    if not changed:
        return None
    count_fn("branch_swapped_animals", changed)
    return replace(plan, herd=tuple(herd))


def _patch_cap_species(obs, plan, seat: int, day: int, spec: dict, count_fn) -> object:
    """Cap the season's head-count of one species, dropping only unbought entries.

    Dropping from the tail keeps the surviving entries' days and order intact, so the plan still
    reads back as "N of this species from day D" — and because only entries past the frontier go,
    the cap can never fall below what the farm already owns. That asymmetry is the point: this is a
    *stop buying* patch, not a liquidation.
    """
    species = spec.get("species", "COW")
    cap = int(spec.get("max", 6))
    start = frontier(obs, plan, seat)
    herd = list(plan.herd)
    have = sum(1 for s, _d in herd if s == species)
    drop = have - cap
    if drop <= 0:
        return None
    keep = [True] * len(herd)
    for i in range(len(herd) - 1, start - 1, -1):
        if drop <= 0:
            break
        if herd[i][0] == species:
            keep[i] = False
            drop -= 1
    dropped = keep.count(False)
    if not dropped:
        return None
    count_fn("branch_capped_animals", dropped)
    return replace(plan, herd=tuple(h for h, k in zip(herd, keep) if k))


def _patch_add_cohort(obs, plan, seat: int, day: int, spec: dict, count_fn) -> object:
    """Append one cohort of `crop`, in the first listed quadrant with room by its plant day.

    "Room" is the plan's own free ground (`Plan.occupied`), the same authority `decode` tiles
    against, and the quadrant must be unlocked by the plant day or `validate` rejects the patch.
    A cohort that can only be part-filled is still taken if it clears `min_tiles`; below that it is
    not a cohort, it is a gesture, and the seed money is better left in the wallet.
    """
    crop = spec.get("crop", "STRAWBERRY")
    n = int(spec.get("n_tiles", 6))
    floor = int(spec.get("min_tiles", max(3, n // 2)))
    plant_day = day + int(spec.get("delay", 1))
    if plant_day >= 30:
        return None
    unlocked = plan.unlocked_by(plant_day)
    for quad in spec.get("quadrants", ("NE", "SW", "NW", "SE")):
        if quad not in unlocked:
            continue
        tiles = _free_tiles(plan, quad, n)
        if len(tiles) < floor:
            continue
        count_fn("branch_cohort_tiles", len(tiles))
        return replace(plan, cohorts=plan.cohorts + (
            Cohort(crop=crop, quadrant=quad, n_tiles=len(tiles), plant_day=plant_day,
                   replant=bool(spec.get("replant", False)), tiles=tuple(tiles)),))
    return None


PATCHES = {
    "swap_species": _patch_swap_species,
    "cap_species": _patch_cap_species,
    "add_cohort": _patch_add_cohort,
}


# --------------------------------------------------------------------------- the shipped branches

#: Every threshold a shipped branch has, and the value it ships at. They are overridable from
#: `consts` (`compiler#branch_set=milk,branch_milk_cap=4`) so a threshold can be *measured* rather
#: than argued: the S-track that was going to search these is closed (E76/E77), so each one is set
#: by a tuning block and gated on a fresh one. Defaults below are the tuned values.
DEFAULTS = {
    "branch_yarn_sheep": 2,        # unbought cows converted to sheep when the yarn store lands
    "branch_yarn_until": 15,       # last day the swap can still pay for itself
    "branch_straw_shops": 3,       # instances demanding STRAWBERRY that justify another cohort
    "branch_straw_day": 12,        # the day the count is taken (four draws are in)
    "branch_straw_tiles": 6,
    "branch_milk_cap": 7,          # head-count ceiling on COW when the town wants no milk
    "branch_milk_day": 9,          # three draws in; ~25% of towns still have no milk shop
}
#
# Where these came from (tuning block 90000:90200 vs `starter`, 400 games/arm, paired on the true
# subset; the gate is a fresh block):
#
#   yarn_sheep   count 1 / **2** / 4       $282 / **$2,311** / $2,336   — 2 and 4 are the same
#                                          patch in practice: the frontier only ever exposes ~2
#                                          unbought cows, so asking for 4 swaps 3.8 and buys
#                                          nothing extra. 1 is null.
#   straw_cohort (day, shops)   9,2: $2,272   12,2: $1,787   **12,3: $3,136**   12,4: $3,377 (n=33)
#                                          15,3: $1,345. Later and stricter is better up to the
#                                          point the sample thins: 12/3 fires on a third of towns
#                                          with the largest effect that still has n > 100.
#   milk_cap     (day, cap)  12,6: **-$355**  12,4: -$684  12,7: +$1,357  9,6: -$739
#                                          **9,7: +$1,496 [-1,408, +4,401]**  6,7: +$407
#                                          Every variant's CI straddles zero. 9/7 is carried to
#                                          the gate as the best of the sweep, not as a result.
#
# Two structural facts the sweep exposed, both worth more than the numbers:
#   * `branch_cohort_tiles` is **3.0 on every seed**, never 6. `boatlee_like` claims its quadrants
#     so completely that three tiles is all the free ground there is, and `branch_straw_tiles=10`
#     (min 5) therefore never fires at all — 0/400, all noop. The branch is "+3 tiles", and the
#     effect below is what three tiles of strawberry are worth.
#   * `cap_species` is frontier-limited in the other direction: at day 12 it can only ever drop 3
#     head and at day 15 it drops none, because by then the herd is bought and forward-only means
#     bought is bought. A milk cap has to fire early or not at all.


def _const(plan_consts: dict, key: str) -> int:
    return int(plan_consts.get(key, DEFAULTS[key]))


def yarn_sheep(c: dict) -> Branch:
    """WOOL is the sharpest draw in the table: `T = 105`, `above_func = "sq"`, and **zero** shop
    demand unless YARN_STORE is drawn. One instance is 12 units/day — over the ~15 days between the
    earliest draw and the last sale that is ~180 units of regeneration against a T of 105, the
    difference between wool being a market and wool being a one-shot. So: yarn store on the board,
    the next unbought pasture animals become sheep instead of cows."""
    return Branch(day_from=3, day_to=_const(c, "branch_yarn_until"),
                  condition="has:YARN_STORE", name="yarn_sheep",
                  patch={"swap_species": {"from": "COW", "to": "SHEEP",
                                          "count": _const(c, "branch_yarn_sheep")}})


def straw_cohort(c: dict) -> Branch:
    """STRAWBERRY has four demanders out of eight, so two instances by day 9 (three draws) is a
    coin flip. Two instances is 12 units/day of sustained drain into `T = 100` — the whole market's
    capacity every eight days."""
    day = _const(c, "branch_straw_day")
    n = _const(c, "branch_straw_tiles")
    return Branch(day_from=day, day_to=day, name="straw_cohort",
                  condition=f"demand>={_const(c, 'branch_straw_shops')}:STRAWBERRY",
                  patch={"add_cohort": {"crop": "STRAWBERRY", "n_tiles": n,
                                        "min_tiles": max(3, n // 2), "delay": 1,
                                        "quadrants": ("NE", "SW", "NW")}})


def milk_cap(c: dict) -> Branch:
    """MILK has three demanders; a town with none by day 12 (four draws, ~15% of seeds) drains milk
    at the town centre's 1/day alone. Nine cows produce ~200 units into `T = 122` with
    `above_target = 1.60` — the back half of the herd sells into a floor it dug itself. The cap
    stops the *purchases*; the cows already standing keep producing."""
    return Branch(day_from=_const(c, "branch_milk_day"), day_to=_const(c, "branch_milk_day"),
                  condition="demand==0:MILK", name="milk_cap",
                  patch={"cap_species": {"species": "COW", "max": _const(c, "branch_milk_cap")}})


#: Named sets, so a branch can be switched on from a pool name (`compiler#branch_set=yarn`) without
#: touching the genome. `Plan.branches` is the other input and the two are concatenated.
BRANCH_SETS = {
    "": (),
    "none": (),
    "yarn": (yarn_sheep,),
    "straw": (straw_cohort,),
    "milk": (milk_cap,),
    "all": (yarn_sheep, straw_cohort, milk_cap),
}


def active(plan) -> tuple:
    """The branches in force for `plan`: its own, plus any named set in `consts["branch_set"]`."""
    consts = plan.consts or {}
    named = str(consts.get("branch_set", "") or "").strip()
    factories = BRANCH_SETS.get(named)
    if factories is None:
        raise KeyError(f"unknown branch_set {named!r}; have {sorted(BRANCH_SETS)}")
    return tuple(plan.branches or ()) + tuple(f(consts) for f in factories)


# --------------------------------------------------------------------------- season state

#: `{seat: {"plan": Plan, "base": id, "fired": set}}`. Module-level for the same reason
#: `projection._REDIRECTS` is: the patched plan has to outlive the turn that made it, and the agent
#: shell is handed a fresh (immutable) base plan on every call.
_SEASON: dict = {}


def reset(seat: int | None = None) -> None:
    if seat is None:
        _SEASON.clear()
    else:
        _SEASON.pop(seat, None)


def fired(seat: int = 0) -> tuple:
    return tuple(sorted((_SEASON.get(seat) or {}).get("fired", ())))


def apply(obs, plan, seat: int, day: int, note=None):
    """The working plan for `day`: `plan` with every branch that has fired patched into it.

    Returns `plan` **itself** — the same object — when no branches are configured, so a plan without
    branches is not merely equivalent to the unpatched one, it is identical.
    """
    branches = active(plan)
    if not branches:
        return plan

    def count(key, n=1):
        if note is not None:
            note(key, n)

    season = _SEASON.get(seat)
    if season is None or season.get("base") is not plan:
        season = {"plan": plan, "base": plan, "fired": set()}
        _SEASON[seat] = season

    working = season["plan"]
    for index, branch in enumerate(branches):
        key = branch.name or f"branch{index}"
        if key in season["fired"]:
            continue
        if not (branch.day_from <= day <= branch.day_to):
            continue
        try:
            if not holds(obs, branch.condition):
                continue
        except (ValueError, KeyError):
            count("branch_bad_condition")
            season["fired"].add(key)
            continue

        # The patch's own counters are held back until the patch is *accepted*: a rejected patch
        # that had already bumped `branch_swapped_animals` would read, in the ledger, exactly like a
        # patch that fired — which is E44's failure with the sign flipped.
        pending: list = []
        patched = working
        applied = 0
        for op, spec in sorted(branch.patch.items()):
            fn = PATCHES.get(op)
            if fn is None:
                continue
            out = fn(obs, patched, seat, day, dict(spec),
                     lambda k, n=1: pending.append((k, n)))
            if out is not None:
                patched = out
                applied += 1
        if not applied:
            # The condition held but the patch had nothing to bite on (every cow already bought,
            # no quadrant with room). Marked fired so it is not retried every dawn, and counted
            # separately from a rejection so "never applicable" and "rejected as unsafe" stay
            # distinguishable in the ledger.
            count("branch_noop")
            count(f"branch_{key}_noop")
            season["fired"].add(key)
            continue

        reasons = forward_only_errors(working, patched, day, frontier(obs, working, seat))
        reasons += patched.validate()
        if reasons:
            count("branch_rejected")
            count(f"branch_{key}_rejected")
            season["fired"].add(key)
            continue

        working = patched
        season["fired"].add(key)
        for k, n in pending:
            count(k, n)
        count("branches_fired")
        count(f"branch_{key}_fired")

    season["plan"] = working
    return working
