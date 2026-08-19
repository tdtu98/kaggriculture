"""P2 — the dawn tree search over macro-decisions.

**What this is.** P1 (`agent/season.py`) can roll a season forward from any dawn to the buzzer in
~2 ms. This module is the search that uses it: at each dawn from `planner_start` on, it builds a
compact menu of macro-moves off the *current* board, scores each by rolling the rest of the season
out under it, and commits the best one. Nothing here models a day; nothing in `season.py` chooses
one.

**Why the candidates are plan patches and not `DayDecisions`.** `season.DayDecisions` can express a
day's overrides inside a rollout, but it cannot be *executed*: the shipped shell plants, buys and
hires off a `Plan`, so a decision that only exists inside the model is a decision that never
happens (E44 in its purest form — a search whose winner cannot be played). A patched `Plan`, by
contrast, is evaluated by `rollout(state, plan=patched)` **and** applied by handing it to
`_act`'s existing machinery, which is how O1's branches already work. One representation, two uses,
no translation layer to get wrong.

**The objective is relative, not absolute.** P1's fidelity harness measured Spearman +0.85 at day 8
and +0.94 at day 15 between the modelled terminal bank and the realised one, against a *median
absolute* error of 13.5% at day 8. So the model ranks well and predicts badly, and the search is
built to consume exactly the first: every candidate at one dawn is rolled from the same state
against the **same sampled shop draws** (common random numbers), and only the differences are read.
`_MIN_GAIN` is the one place an absolute number enters, and it is a deviation *threshold*, not a
prediction.

**The shop draw is the dominant error term and it is sampled, not guessed.** `_end_of_day` draws
the next shop uniformly from the eight names with replacement (`kaggriculture.py:874,891`), and six
of the eight instances are still undrawn at day 8. P1 measured that substituting the true draw takes
the day-8 error from 13.5% to 5.4%. Averaging over `draw_samples` sampled continuations through
`SeasonState.future_shops` is the honest version of that: a candidate that only wins under one draw
loses to one that wins under four.

**Not before day `planner_start`.** P1 measured day-3 rank fidelity at 19.7% — the board is three
days old, almost nothing is standing, and the model is ranking noise. Day 6 is the default and the
knob exists so it can be moved by measurement rather than by argument.

**Everything is forward-only, and the gate is O1's.** `branches.forward_only_errors` already encodes
"never dig a live plant, never sell an animal, never edit a cohort that is in the ground"; this
module reuses it rather than restating it, with one deliberate widening — a planner *may* re-time
land it has not bought yet, which a branch may not (see `_forward_only`).

**Nothing here raises.** `apply` is called from a turn that must not forfeit the episode (E21); any
failure returns the unpatched plan and counts `planner_errors`, which is the C4 fallback pattern.

--------------------------------------------------------------------------------------------------
**MEASURED: the flag ships OFF because the search loses money, and the reason is specific.**

Paired both-seat runs, 80 games per arm, R2 + `slot_align=1`, replicated on two fresh blocks:

|                          | 74000:74040 | 75000:75040 |
|--------------------------|---|---|
| solo (vs `starter`)      | **−$5,984** [−8,909, −3,058] | **−$4,706** [−7,803, −1,609] |
| margin (vs `boatlee`)    | **−$4,546** [−6,594, −2,499] | **−$4,287** [−6,809, −1,764] |

Per-kind arms (`planner_kinds`) put the whole loss on **`cohort_shift`** — every other decision type
is inert on this plan, and for a structural reason: the R2 genome claims every tile of every
quadrant it buys, so `cohort` and `wheat_ramp` find no free ground before day 16 and the wallet
gates land until day 12 (E80's "every branch verdict is a property of the PLAN", again).

**Why cohort timing cannot be ranked by this model.** One-move-per-season calibration, 20 seasons:
mean modelled gain **+$1,657**, mean realised **−$4,393**, correlation **−0.15**, sign agreement
**4/20**. The realised deltas are bimodal — exactly $0, or −$10k to −$27k — and the counter diff on
one −$16,200 case says what the cliff is: moving a **five-tile** strawberry cohort three days took
`plants_lost_thirst` **2 → 9**, `fertilize_hits` **52 → 43**, `prune_days` **7 → 10** and
`strawberry_per_plant` **6.33 → 4.73**. The shift re-phased the day and collided with work the
router already had; the season model prices labour as a scalar budget and assumes every plant is
watered (`season.py` assumptions 1 and 2), so the entire cost of the decision lives in the two
places the model declares out of scope. **The search is not mis-ranking noise; it is ranking a
quantity the model does not contain.** A cohort-timing search needs a labour-routing model, i.e.
`router`, not `season`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
)

from agent import branches as branch_points
from agent import projection, season
from agent.plan import NEVER, Cohort, Plan, quadrant_of, quadrant_tiles

#: Knob defaults. Every one is overridable from `plan.consts`, i.e. from a registry name
#: (`vec:...#planner=1,planner_beam=6`), so a threshold is measurable rather than arguable — the
#: same contract `branches.DEFAULTS` has.
DEFAULTS = {
    "planner": 0,                # master flag; 0 means this module is inert and returns `plan`
    "planner_start": 6,          # first dawn a search runs (P1: day-3 fidelity is 19.7%)
    "planner_beam": 6,           # candidates carried from the screen into the full evaluation
    "planner_draws": 4,          # sampled shop-draw continuations averaged per evaluation
    # Decision points searched, including today's. **Ships at 1, and that is a measurement, not a
    # simplification.** Depth costs a factor of the menu size in rollouts and buys a value that is a
    # *maximum over single-draw child estimates* — biased upward by however many children the node
    # had, and biased differently for nodes with different menus. Measured on 74000: with depth the
    # search reported a $10.6k gain for a move a 32-draw depth-1 reference ranked *below* follow, and
    # the arm is carried into the gate as `planner_lookahead=2` rather than assumed away. Receding
    # horizon supplies most of what depth would: the next decision point is re-searched tomorrow, on
    # a real board, at a fifth of the error.
    "planner_lookahead": 1,
    "planner_stride": 4,         # days between decision points
    "planner_ms": 70,            # wall-clock budget for one dawn's search, milliseconds
    "planner_min_gain": 750,     # dollars of modelled gain a deviation must show to be taken
    "planner_max_moves": 12,     # deviations one season may commit
    "planner_menu_cap": 30,      # menu size after pruning
    "planner_confirm": 1,        # re-score the winner on a held-out draw sample before committing
    # Decision kinds the menu may offer, comma-separated; empty means all of them. This is an
    # *attribution* knob, not a tuning one: "which decision type drove the delta" is otherwise
    # inferred from counters, and E39 is the standing reminder that a comparison against a
    # re-derivation of the rule measures the re-derivation. One arm per kind measures the kind.
    "planner_kinds": "",
    # P2b. `"fast"` is P1's day-granular season model (`season.rollout`, ~2 ms); `"true"` is the
    # compiler-in-the-loop rollout (`agent/true_rollout.py`, ~45-60 ms), which evaluates a candidate
    # by *compiling and playing* every remaining day through C1/C2/C3 instead of pricing labour as a
    # scalar. E83 is the reason the second one exists: the fast model cannot contain the quantity
    # `cohort_shift` moves, and a search steers into exactly that gap.
    "planner_value": "fast",
    # How many days of a `"true"` evaluation are actually compiled before the tail is handed to P1's
    # fast model. 0 means the whole season. This is the *latency* knob for P2b — `actTimeout` is 1 s
    # and one compiled day costs 0.7-2.7 ms against `season.rollout`'s 2 ms for a whole season — and
    # it is a defensible split rather than a shortcut: E83's mechanism is a re-phasing collision that
    # bites within days of the decision, while the tail is the steady state E82 already measured at
    # ~5%.
    "planner_true_days": 0,
}

#: How far a cohort's plant day may be moved in one decision.
#:
#: Wider than it looks like it needs to be, and the width is the fix for a measured pathology. With
#: `(-2, -1, 1, 2, 3)` the search walked a cohort two days earlier at every dawn — day 12 said
#: "cohort5 two days earlier, +$7.4k", day 15 said it again from the new plan, and so on. The model
#: has a *gradient* there, not a step, and a menu too narrow to reach the optimum turns a single
#: decision into a five-dawn crawl that spends the move budget and never arrives. One decision
#: should be able to land on the answer.
SHIFT_DELTAS = (-6, -4, -3, -2, -1, 1, 2, 3, 4)

#: Crops the menu may start a new cohort in. Not all five: CARROT is the cheapest per tile and the
#: only single-shop crop that regenerates (PET_CAFE, `kaggriculture.py:103-112` — E80 corrected this
#: from eggs), TOMATO is the late-ground crop the R2 genome already found, and WHEAT/STRAWBERRY/MELON
#: are the three the boatlee choreography is actually built from. Ordering is the menu's tie-break.
MENU_CROPS = ("WHEAT", "STRAWBERRY", "MELON", "TOMATO", "CARROT")

#: Row-sized. A quadrant is 5x5 and the compiler routes a row at a time, so a cohort that is not a
#: multiple of 5 costs the same walk for less crop.
COHORT_SIZES = (5, 10)

#: The wheat ramp, kept separate from the ordinary cohort menu because it is a *pair* of moves — buy
#: the land and fill it — and neither half is worth anything alone. This is the boatlee move E77
#: identified as the production the compiler line never buys: wheat is the widest market on the board
#: (5 demanders, D17) and the SE quadrant sits unbought on every genome we have searched.
WHEAT_RAMP_SIZES = (10, 15, 25)

#: Species the herd menu may add, best-first by the shell's own ranking (`main_v4._yield_per_dollar`,
#: restated in `season._yield_per_dollar`).
HERD_SPECIES = ("SHEEP", "COW", "GOOSE")

#: How many shop names a sampled continuation carries. The town tops out at 8 instances
#: (`townShopUnlockInterval` note), so eight names is a whole season and `shops_on` ignores the tail.
DRAW_LENGTH = 8


def _const(consts: dict, key: str):
    value = (consts or {}).get(key)
    return DEFAULTS[key] if value is None else value


def _int(consts: dict, key: str) -> int:
    return int(_const(consts, key))


# --------------------------------------------------------------------------- candidates

@dataclass(frozen=True)
class Candidate:
    """One macro-move, as the plan it produces.

    `kind` is what the counters attribute a deviation to, and it is deliberately coarser than
    `name`: the question the measurement has to answer is "which decision type paid", not "which of
    the four land offsets".
    """

    name: str
    kind: str
    plan: Plan
    #: What this move re-decides (`cohort3`, `land_SE`, `herd`), and which way it pushes it. The
    #: pair is the oscillation lock: see `menu`'s `locks` argument.
    target: str = ""
    direction: int = 0


@dataclass(frozen=True)
class Ctx:
    """What the menu needs to know about the board, from either an observation or a rollout state.

    The second source is the point: a depth-2 search builds its menu at a dawn that has not happened,
    where there is no observation to read. Keeping the menu's input to five scalars is what lets the
    same builder serve both.
    """

    day: int
    money: float
    unlocked: frozenset
    animals: int
    seat: int


def ctx_from_obs(obs, seat: int) -> Ctx:
    farm = obs["farms"][seat]
    return Ctx(
        day=int(obs.get("day", 0) or 0),
        money=float(farm.get("money", 0) or 0),
        unlocked=frozenset(farm.get("unlocked_quadrants") or ["NW"]),
        animals=sum(1 for row in farm["tiles"] for t in row
                    if isinstance(t, dict) and t.get("animal")),
        seat=seat,
    )


def ctx_from_state(state: season.SeasonState) -> Ctx:
    return Ctx(day=int(state.day), money=float(state.money),
               unlocked=frozenset(state.unlocked), animals=int(state.n_animals),
               seat=int(state.seat))


def _next_land(plan: Plan, ctx: Ctx) -> tuple:
    """`(quadrant, index, plan day)` for the next quadrant in `LAND_ORDER` that is not owned."""
    for i, quad in enumerate(LAND_ORDER):
        if quad in ctx.unlocked:
            continue
        return quad, i, int((plan.land_days or {}).get(quad, NEVER))
    return None, -1, NEVER


def _free_ground(plan: Plan, quad: str, want: int) -> list:
    """Tiles in `quad` the plan does not already claim, in the plan's own row order.

    `Plan.occupied` is the same authority `decode` tiles against and `branches._free_tiles` uses;
    reusing it is what keeps a menu cohort from being tiled onto a pasture (E39: a comparison run
    against a re-implementation of the rule measures the re-implementation).
    """
    taken = set(plan.occupied())
    out = []
    for t in quadrant_tiles(quad):
        if len(out) >= want:
            break
        if t not in taken:
            out.append(t)
    return out


def _with_cohort(plan: Plan, crop: str, quad: str, tiles: list, day: int,
                 replant: bool = False) -> Plan:
    return replace(plan, cohorts=plan.cohorts + (
        Cohort(crop=crop, quadrant=quad, n_tiles=len(tiles), plant_day=day,
               replant=replant, tiles=tuple(tiles)),))


def _seed_cost(crop: str, n: int) -> float:
    return float(CROPS[crop]["seed"]) * n


def _pending_cohorts(plan: Plan, day: int) -> list:
    """Indices of cohorts that have not opened yet, soonest first — the ones a shift may move."""
    return sorted((i for i, c in enumerate(plan.cohorts) if c.plant_day > day),
                  key=lambda i: plan.cohorts[i].plant_day)[:2]


def menu(plan: Plan, ctx: Ctx, consts: dict, locks: dict | None = None) -> list:
    """The candidate set at one decision point. `follow` is always element zero.

    The menu is built off the plan and the board and pruned by both: a cohort with no free ground is
    not offered, a land purchase the wallet cannot reach within a day's income is not offered, and
    the whole list is capped at `planner_menu_cap`. Follow-plan is first and is never pruned — it is
    the incumbent and therefore the performance floor, and a search that can drop its own floor is a
    search that can lose money by finding nothing.

    `locks` is `{target: direction}` for moves already committed this season, and it forbids the
    *reverse* of each one. Receding-horizon control re-decides everything every dawn, and without
    this it churns: measured on seed 74000, the search moved cohort 0 two days later on day 8, one
    more on day 9, and then **two days earlier on day 10** — three moves, three modelled gains,
    and a plan a day from where it started. Locking the direction (not the target) keeps the useful
    half of re-deciding — walking further the same way as the board confirms the call — and drops
    the half that is the model disagreeing with itself.
    """
    day = ctx.day
    locks = locks or {}
    out = [Candidate("follow", "follow", plan)]
    if day >= season.LAST_DAY:
        return out

    def allowed(target: str, direction: int) -> bool:
        prior = locks.get(target)
        return prior is None or prior == direction

    quad, index, due = _next_land(plan, ctx)

    # -- land timing ---------------------------------------------------------
    # Moving the *next* quadrant only. Later ones are re-decided at a later dawn, which is what
    # receding-horizon control buys: a decision that does not have to be made today is not made
    # today, so the menu stays small and every choice is taken on the freshest board.
    if quad is not None:
        price = float(LAND_PRICES[index])
        land_days = dict(plan.land_days or {})
        target = f"land_{quad}"
        if ctx.money >= price and due != day and allowed(target, -1):
            out.append(Candidate(f"land_{quad}_now", "land",
                                 replace(plan, land_days={**land_days, quad: day}), target, -1))
        if due < NEVER:
            for delta in (2, 4):
                if due + delta <= season.LAST_DAY and allowed(target, 1):
                    out.append(Candidate(f"land_{quad}_+{delta}", "land",
                                         replace(plan, land_days={**land_days, quad: due + delta}),
                                         target, 1))
        else:
            # The quadrant the plan never buys. This is the one land move that is a *strategy*
            # change rather than a timing change, and on every genome we have searched SE is
            # exactly this case.
            for delta in (2, 5):
                if day + delta <= season.LAST_DAY and ctx.money >= price * 0.5 \
                        and allowed(target, -1):
                    out.append(Candidate(f"land_{quad}_open+{delta}", "land",
                                         replace(plan, land_days={**land_days, quad: day + delta}),
                                         target, -1))

    # -- cohort timing -------------------------------------------------------
    for i in _pending_cohorts(plan, day):
        cohort = plan.cohorts[i]
        target = f"cohort{i}"
        for delta in SHIFT_DELTAS:
            when = cohort.plant_day + delta
            direction = 1 if delta > 0 else -1
            if when < day or when > season.LAST_DAY or not allowed(target, direction):
                continue
            if when + int(CROPS[cohort.crop]["first_yield_day"]) > season.LAST_DAY:
                continue
            cohorts = list(plan.cohorts)
            cohorts[i] = replace(cohort, plant_day=when)
            sign = "+" if delta > 0 else ""
            out.append(Candidate(f"cohort{i}_{sign}{delta}", "cohort_shift",
                                 replace(plan, cohorts=tuple(cohorts)), target, direction))

    # -- an extra cohort -----------------------------------------------------
    for crop in MENU_CROPS:
        if day + 1 + int(CROPS[crop]["first_yield_day"]) > season.LAST_DAY:
            continue
        for q in ("NE", "SW", "NW", "SE"):
            if q not in ctx.unlocked:
                continue
            offered = False
            for size in COHORT_SIZES:
                tiles = _free_ground(plan, q, size)
                if len(tiles) < size or ctx.money < _seed_cost(crop, len(tiles)):
                    continue
                offered = True
                out.append(Candidate(f"plant_{crop}_{q}_{size}", "cohort",
                                     _with_cohort(plan, crop, q, tiles, day + 1,
                                                  replant=not bool(CROPS[crop]["ongoing"])),
                                     f"plant_{crop}_{q}", -1))
            if offered:
                break                               # one quadrant per crop keeps the menu compact

    # -- the wheat ramp ------------------------------------------------------
    # Land plus wheat in one move. Offered from `planner_start` on rather than "late" for a measured
    # reason: wheat's first yield is early enough that a day-20 ramp cannot pay for a $4,000
    # quadrant, so the affordability prune below is what makes it a late move on a poor board and an
    # early one on a rich one, instead of a hard-coded day.
    if quad is not None:
        price = float(LAND_PRICES[index])
        for size in WHEAT_RAMP_SIZES:
            if day + 1 + int(CROPS["WHEAT"]["first_yield_day"]) > season.LAST_DAY:
                break
            if ctx.money < price + _seed_cost("WHEAT", size):
                continue
            ramped = replace(plan, land_days={**dict(plan.land_days or {}), quad: day})
            tiles = _free_ground(ramped, quad, size)
            if len(tiles) < size:
                continue
            out.append(Candidate(f"wheat_ramp_{quad}_{size}", "wheat_ramp",
                                 _with_cohort(ramped, "WHEAT", quad, tiles, day + 1, replant=True),
                                 f"land_{quad}", -1))

    # -- the herd ------------------------------------------------------------
    # Adding a head needs somewhere to put it, so a pasture tile comes with it. Removing one may
    # only touch entries above the purchase frontier, which `_forward_only` enforces — the menu
    # offers the move and the gate decides whether this board still allows it.
    for species in HERD_SPECIES:
        cost = float(ANIMALS[species]["cost"])
        for k in (1, 2):
            if ctx.money < cost * k or not allowed("herd", 1):
                continue
            extra = _pasture_room(plan, ctx, k)
            if len(extra) < k:
                continue
            out.append(Candidate(f"herd_+{k}{species}", "herd", replace(
                plan,
                pasture_tiles=tuple(plan.pasture_tiles) + tuple(extra),
                herd=tuple(plan.herd) + tuple((species, day) for _ in range(k))), "herd", 1))
    for k in (1, 2):
        if len(plan.herd) > k and allowed("herd", -1):
            out.append(Candidate(f"herd_-{k}", "herd", replace(plan, herd=tuple(plan.herd[:-k])),
                                 "herd", -1))

    allow = str(_const(consts, "planner_kinds") or "").strip()
    if allow:
        wanted = {k.strip() for k in allow.split(",") if k.strip()} | {"follow"}
        out = [c for c in out if c.kind in wanted]
    cap = _int(consts, "planner_menu_cap")
    return out[:max(1, cap)]


def _pasture_room(plan: Plan, ctx: Ctx, k: int) -> list:
    """`k` unclaimed tiles in an owned quadrant that a new pasture can stand on.

    Adjacency is not modelled — `plan.pasture_order` clusters pastures against the shed because
    keepers pay for distance twice a day, and a menu that appended a pasture in the far corner would
    be buying an animal and a walk. So the search only ever extends into ground next to the cluster
    it already has, by taking the plan's own quadrant order.
    """
    taken = set(plan.occupied())
    out = []
    for quad in ("NW", "NE", "SW", "SE"):
        if quad not in ctx.unlocked:
            continue
        for t in quadrant_tiles(quad):
            if len(out) >= k:
                return out
            if t not in taken:
                out.append(t)
    return out


# --------------------------------------------------------------------------- the forward-only gate

def _forward_only(before: Plan, after: Plan, day: int, frontier_index: int, owned: frozenset) -> list:
    """O1's gate, widened by exactly one rule: land the planner does not own yet may be re-timed.

    `branches.forward_only_errors` refuses *any* `land_days` edit, and that is right for a branch —
    a branch fires once on a condition and has no way to know whether the money it is re-timing was
    already spent. The planner re-decides every dawn against the live board, so the safe statement
    is the sharper one: a quadrant that is **already unlocked** is a purchase that has happened and
    may not be rewritten, and a purchase day may not be moved into the past. Everything else in the
    gate — live cohorts, sown ground, the herd below the frontier, pastures — is unchanged and
    imported, not restated.
    """
    probe = replace(after, land_days=dict(before.land_days or {}))
    errors = list(branch_points.forward_only_errors(before, probe, day, frontier_index))
    before_land = dict(before.land_days or {})
    for quad, when in (after.land_days or {}).items():
        if int(when) == int(before_land.get(quad, NEVER)):
            continue
        if quad in owned:
            errors.append(f"land_days: {quad} is already owned and may not be re-timed")
        elif int(when) < day:
            errors.append(f"land_days: {quad} moved to day {when}, before today ({day})")
    return errors


def _admissible(before: Plan, cand: Candidate, ctx: Ctx, frontier_index: int) -> bool:
    if cand.plan is before:
        return True
    if _forward_only(before, cand.plan, ctx.day, frontier_index, ctx.unlocked):
        return False
    return not cand.plan.validate()


# --------------------------------------------------------------------------- evaluation

#: Salts for the two draw samples. `SELECT` ranks the menu; `CONFIRM` is held out and re-scores only
#: the winner, which is what makes the reported gain an estimate rather than a maximum.
SELECT_SALT = 0x5EED
CONFIRM_SALT = 0xC0FFEE


def draws(day: int, seat: int, n: int, salt: int = SELECT_SALT) -> list:
    """`n` sampled shop-draw continuations, common to every candidate at this dawn.

    Common random numbers, and the reason the objective can be relative: the draw is a shock shared
    by every candidate at one dawn (P1: Spearman +0.85 at day 8 against a 13.5% absolute error), so
    scoring all of them against the *same* sample removes it from the comparison instead of adding
    it to the noise. Seeded from `(day, seat)` rather than from `random` so a dawn's search is
    reproducible: a planner whose choice depends on how many other draws the process has made is a
    planner whose trace cannot be replayed.
    """
    names = projection.SHOP_NAMES
    rng = random.Random((int(day) * 7919) ^ (int(seat) * 104729) ^ salt)
    return [[rng.choice(names) for _ in range(DRAW_LENGTH)] for _ in range(max(1, int(n)))]


def value(base: season.SeasonState, plan: Plan, sample: list) -> float:
    """Mean terminal bank over `sample` shop-draw continuations. Deterministic given both."""
    total = 0.0
    for continuation in sample:
        state = base.clone()
        state.future_shops = continuation
        total += season.rollout(state, plan=plan)
    return total / max(1, len(sample))


def _true_mode(consts: dict) -> bool:
    return str(_const(consts, "planner_value") or "fast").strip().lower() == "true"


def valuer(consts: dict):
    """The value function this search will rank with — P1's fast model, or P2b's true rollout.

    A bound name rather than a branch inside `value`, so the two paths cannot drift apart at some
    call site that forgot which mode it was in, and so a mutant that swaps them is caught by one
    test rather than needing one per call.
    """
    if _true_mode(consts):
        from agent import true_rollout

        days = _int(consts, "planner_true_days") or None
        return lambda base, plan, sample: true_rollout.value(base, plan, sample, days=days)
    return value


def base_state(obs, seat: int, plan: Plan, consts: dict):
    """The object `valuer(consts)` rolls forward. Same construction cost either way (~0.07 ms)."""
    if _true_mode(consts):
        from agent import true_rollout

        return true_rollout.World.from_obs(obs, seat=seat, plan=plan)
    return season.SeasonState.from_obs(obs, seat=seat, plan=plan)


def _advance(base: season.SeasonState, plan: Plan, continuation: list, days: int
             ) -> season.SeasonState:
    """Roll `base` forward `days` days under `plan`, following it — the default policy.

    This is how a depth-2 node reaches the dawn its own menu is built at. The tail beyond the last
    decision point is the same default policy, which is what makes the search's value a lower bound
    on the plan it commits: everything it does not decide, it decides by following.
    """
    state = base.clone()
    state.future_shops = continuation
    state.plan = plan
    stop = min(season.LAST_DAY, state.day + max(0, int(days)))
    while state.day < stop:
        season.step_day(state, None)
    return state


# --------------------------------------------------------------------------- the search

class _Budget:
    """Wall-clock guard. The search is anytime: every stage asks whether it may start.

    A fixed candidate count would be a latency promise the submission host cannot keep — it is
    slower than this laptop and the rollout cost falls through the season — so the budget is time
    and the depth is whatever fits. `planner_ms` is therefore a *latency* knob, and the counters
    record what it bought.
    """

    __slots__ = ("t0", "limit")

    def __init__(self, ms: float) -> None:
        self.t0 = time.perf_counter()
        self.limit = max(0.0, float(ms)) / 1000.0

    @property
    def spent(self) -> float:
        return time.perf_counter() - self.t0

    def room(self, fraction: float = 1.0) -> bool:
        return self.spent < self.limit * fraction


def search(base: season.SeasonState, plan: Plan, ctx: Ctx, consts: dict, count,
           locks: dict | None = None) -> tuple:
    """`(chosen candidate, modelled gain)` for one dawn.

    Three stages, each one gated on the clock:

    1. **Screen.** Every menu candidate against draw 0. One rollout each, and the point of it is to
       spend the expensive `planner_draws` averaging only on candidates that survived a first look.
    2. **Evaluate.** The top `planner_beam` (follow-plan always among them) against the whole
       sample. This is where the ranking is actually taken.
    3. **Deepen.** Roll each surviving node forward `planner_stride` days under itself, build a menu
       at *that* dawn, and take its best child. A deviation two decision points out is never
       committed — only today's move is — but a move whose value depends on a follow-up we would in
       fact make is scored as such.

    **Depth is all-or-nothing, and that is a correctness rule rather than a scheduling one.** A
    node's depth-2 value is a *maximum* over its children's single-draw estimates, so it is biased
    upward by however many children it had; a depth-1 value is an average and is not. Mixing them —
    the first version took `max(depth-1 average, depth-2 maximum)` per node, and left `follow` at
    depth 1 because its menu was smallest — reported gains of **$17.5k** where a 32-draw reference
    measured **$2.0k**, and every one of them was the optimism of the maximum. So either every node
    in the beam gets deepened against the same draw and the ranking is taken at depth 2, or none
    does and it is taken at depth 1.
    """
    budget = _Budget(_const(consts, "planner_ms"))
    beam_width = max(1, _int(consts, "planner_beam"))
    sample = draws(ctx.day, ctx.seat, _int(consts, "planner_draws"))
    lookahead = max(1, _int(consts, "planner_lookahead"))
    stride = max(1, _int(consts, "planner_stride"))
    evaluate = valuer(consts)
    if _true_mode(consts):
        # Depth costs a factor of the menu size in rollouts, and a true rollout is 25x a fast one.
        # `_advance` is also `season.step_day`, i.e. the model P2b exists to stop trusting — a
        # depth-2 node reached by the fast model and scored by the true one would be ranking a dawn
        # that the true model says never happens. All-or-nothing depth (see the docstring) makes
        # "none" the only coherent answer here.
        lookahead = 1

    frontier_index = ctx.animals + max(1, int((plan.consts or {}).get("animals_per_day", 1) or 1))
    follow = Candidate("follow", "follow", plan)
    candidates = [c for c in menu(plan, ctx, consts, locks)
                  if _admissible(plan, c, ctx, frontier_index)]
    count("planner_candidates", len(candidates))
    if len(candidates) <= 1:
        return follow, 0.0

    # 1. screen
    screen = []
    head = sample[:1]
    for cand in candidates:
        if not budget.room(0.45) and cand.name != "follow":
            count("planner_screen_truncated")
            break
        screen.append((evaluate(base, cand.plan, head), cand))
    screen.sort(key=lambda row: -row[0])

    # 2. evaluate. `follow` is re-inserted rather than merely kept: it is the floor, and a beam that
    #    happened to screen it out would be comparing the winners only against each other.
    kept = [cand for _v, cand in screen[:beam_width]]
    if not any(c.name == "follow" for c in kept):
        kept.append(candidates[0])
    scored = []
    tail = sample[1:]
    for cand in kept:
        screen_value = next(v for v, c in screen if c is cand)
        v = screen_value
        if tail and budget.room(0.75):
            v = (screen_value + len(tail) * evaluate(base, cand.plan, tail)) / (1 + len(tail))
        scored.append((v, cand))
    scored.sort(key=lambda row: -row[0])

    # 3. deepen — all of the beam or none of it, per the docstring.
    if lookahead > 1 and budget.room(0.60):
        deep = []
        for _v, cand in scored:
            if not budget.room(0.95):
                break
            deep.append((_deepen(base, cand, ctx, consts, sample, budget, lookahead - 1, stride,
                                 count, locks), cand))
        if len(deep) == len(scored) and all(v > float("-inf") for v, _c in deep):
            count("planner_deepened")
            deep.sort(key=lambda row: -row[0])
            scored = deep

    follow_value = next(v for v, c in scored if c.name == "follow")
    best_value, best = scored[0]
    gain = best_value - follow_value
    min_gain = float(_const(consts, "planner_min_gain"))
    if best.name == "follow" or gain < min_gain:
        return follow, 0.0

    # 4. confirm on a held-out sample. The winner of a maximum over ~25 noisy estimates is biased
    #    upward by the maximum itself — the winner's curse — and the bias is large here: on 74000 the
    #    selection sample priced the day-6 winner at **+$10,550** where a 32-draw reference put the
    #    same move *below* follow. Re-scoring the winner and the floor on draws the selection never
    #    saw costs two evaluations and turns `planner_modelled_gain` from a maximum into an estimate,
    #    which is what the calibration check needs it to be. Skipped only when the clock has run out,
    #    and then the selection gain is reported and the move is still taken — the confirmation is a
    #    debias, not a second gate on whether the search ran.
    if int(_const(consts, "planner_confirm")) and budget.room(0.9):
        held = draws(ctx.day, ctx.seat, _int(consts, "planner_draws"), CONFIRM_SALT)
        confirmed = evaluate(base, best.plan, held) - evaluate(base, plan, held)
        count("planner_confirmed")
        if confirmed < min_gain:
            count("planner_unconfirmed")
            return follow, 0.0
        gain = confirmed
    return best, gain


def _deepen(base: season.SeasonState, cand: Candidate, ctx: Ctx, consts: dict, sample: list,
            budget: _Budget, depth: int, stride: int, count, locks: dict | None) -> float:
    """The best value reachable from `cand` after one more decision point, `stride` days out.

    Returns `-inf` when the node could not be expanded — no days left, or the clock ran out
    mid-expansion — because a partly-expanded node's maximum is not comparable with a fully-expanded
    one's, and `search` uses that sentinel to abandon depth for the whole beam rather than rank a
    mixture.
    """
    ahead = _advance(base, cand.plan, sample[0], stride)
    if ahead.day >= season.LAST_DAY:
        return float("-inf")
    child_ctx = ctx_from_state(ahead)
    child_locks = dict(locks or {})
    if cand.target:
        child_locks[cand.target] = cand.direction
    frontier_index = child_ctx.animals + max(
        1, int((cand.plan.consts or {}).get("animals_per_day", 1) or 1))
    children = [c for c in menu(cand.plan, child_ctx, consts, child_locks)
                if _admissible(cand.plan, c, child_ctx, frontier_index)]
    count("planner_deep_nodes", len(children))
    best = float("-inf")
    head = sample[:1]
    for child in children:
        if not budget.room(0.95):
            count("planner_deep_truncated")
            return float("-inf")
        v = value(ahead, child.plan, head)
        if depth > 1:
            deeper = _deepen(ahead, child, child_ctx, consts, sample, budget, depth - 1, stride,
                             count, child_locks)
            if deeper == float("-inf"):
                return float("-inf")
            v = max(v, deeper)
        if v > best:
            best = v
    return best


# --------------------------------------------------------------------------- season state

#: `{seat: {"plan": Plan, "base": Plan, "moves": int, "log": [...]}}`. Module-level for the reason
#: `branches._SEASON` is: the patched plan has to outlive the turn that made it, and the shell is
#: handed a fresh immutable base plan on every call. `main_v4._state` clears it on step 0, so a
#: worker process playing many games cannot inherit the previous season's deviations.
_SEASON: dict = {}


def reset(seat: int | None = None) -> None:
    if seat is None:
        _SEASON.clear()
    else:
        _SEASON.pop(seat, None)


def log(seat: int = 0) -> tuple:
    """`(day, name, kind, modelled gain)` per committed deviation — the calibration trace's input."""
    return tuple((_SEASON.get(seat) or {}).get("log", ()))


def enabled(plan: Plan) -> bool:
    return bool(int(_const(plan.consts or {}, "planner")))


def apply(obs, plan: Plan, seat: int, day: int, hour: int, note=None) -> Plan:
    """The working plan for this turn.

    Returns `plan` **itself** — the same object — whenever the planner is off, has nothing to do, or
    fails. Identity rather than equality, so a flag-off season is byte-identical by construction and
    not merely by arithmetic (the invariance test reads money, this reads `is`).
    """
    def count(key, n=1):
        if note is not None:
            note(key, n)

    consts = plan.consts or {}
    if not int(_const(consts, "planner")):
        return plan

    season_state = _SEASON.get(seat)
    if season_state is None or season_state.get("base") is not plan:
        season_state = {"plan": plan, "base": plan, "moves": 0, "log": [], "day": -1, "locks": {}}
        _SEASON[seat] = season_state
    working = season_state["plan"]

    if hour != 0 or day < _int(consts, "planner_start") or day == season_state["day"]:
        return working
    season_state["day"] = day
    if season_state["moves"] >= _int(consts, "planner_max_moves"):
        return working

    t0 = time.perf_counter()
    try:
        ctx = ctx_from_obs(obs, seat)
        base = base_state(obs, seat, working, consts)
        chosen, gain = search(base, working, ctx, consts, count, season_state["locks"])
    except Exception:                               # pragma: no cover - defensive, see module note
        count("planner_errors")
        return working
    finally:
        count("planner_ms", int(round((time.perf_counter() - t0) * 1000)))
        count("planner_dawns")

    if chosen.plan is working:
        return working
    season_state["plan"] = chosen.plan
    season_state["moves"] += 1
    season_state["log"].append((day, chosen.name, chosen.kind, gain))
    if chosen.target:
        season_state["locks"][chosen.target] = chosen.direction
    count("planner_deviations")
    count(f"planner_kind_{chosen.kind}")
    count("planner_modelled_gain", int(round(gain)))
    return chosen.plan
