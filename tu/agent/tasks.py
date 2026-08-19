"""C1 — the day's work, as valued tasks with deadlines.

`daily_tasks(obs, plan, day)` turns a farm state and a plan into everything worth doing today, each
item priced in dollars and carrying the turn by which it must happen. It decides *what*; the router
(C2) decides who does it and in what order, and the verifier (C3) prunes what does not fit.

Two things make this more than a to-do list:

**Value is in one currency.** A survival water, a fertilize and a harvest are all quoted in dollars,
so the router can trade them off instead of following a hand-ordered priority ladder. The engine
line's ladder is exactly what E17 found to be harmful — priorities that looked sensible cost money —
and a ladder cannot express "skip this water, the plant is nearly spent".

**Deadlines come from the rules, not from intuition.** Wheat is worthless after the dawn of age 5;
a fertilized strawberry left two ticks throws away the second one against a cap of 4; a cow at
`max_held` produces into a full bucket. Each of those is a real number in `kaggriculture.py`, and
they are quoted below with the line that sets them.

The prices come through `price_fn` so that C6's projection can replace the current quote without
touching this module — pricing a harvest at today's board price is exactly the mistake that made
melon look worthless (D17/E48).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS

from agent.plan import FERT_AGES

TURNS_PER_DAY = 24
LAST_TURN = TURNS_PER_DAY - 1

#: Shed-access tiles: all four centre squares, LOCKED or not (`kaggriculture.py` shed ops resolve
#: ahead of the LOCKED guard — verified in tests/test_env_facts.py).
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))

#: Seed prices, for the net value of a planting.
SEED_COST = {c: CROPS[c]["seed"] for c in CROPS}

#: Dusk empties every unit inventory into the shed and discards whatever will not fit
#: (`shedCapacity`, default 100). C1 emits a DROP above this mark so the goods can be sold today,
#: while they still exist — deferring to tomorrow is how a day's harvest evaporates.
SHED_PRESSURE = 90

#: A season is 30 days. Nothing is planted that cannot reach its first yield inside it.
LAST_DAY = 29

#: Effect counters for the tick-day water rule (E66 mechanism #2). Module-level and cumulative,
#: because the question "did this ever fire in a real season" is not answerable from one call.
#: `tick_waters` counts WATER tasks emitted because tonight is a fertilized production tick;
#: `tick_waters_fert_today` the subset whose fertilizer has not landed yet at planning time — the
#: half of them the old `fertilized_until_day >= day` test could not see at all.
STATS = {"tick_waters": 0, "tick_waters_fert_today": 0}



@dataclass(frozen=True)
class Task:
    """One thing worth doing today.

    `needs` is what the unit must be *carrying* to do it (a FEED needs wheat in hand, a FERTILIZE
    needs fertilizer). The router turns those into PICKUP stops; C4 turns any shortfall into a
    market order at hour 0.
    """

    tile: tuple | None
    op: str
    args: tuple = ()
    value: float = 0.0
    deadline_turn: int = LAST_TURN
    needs: dict = field(default_factory=dict)
    kind: str = ""
    #: Turn after which this op is a guaranteed no-op because the tile will be gone. Only decaying
    #: plants have one: past `max_lifespan_step` a crop sheds a unit every two steps and becomes a
    #: WEED when it hits zero, so an op that arrives afterwards spends a turn on bare ground.
    expires_turn: int | None = None

    @property
    def survival(self) -> bool:
        return self.kind == "survival"

    def __repr__(self) -> str:                                   # readable test failures
        where = f"{self.tile}" if self.tile else "shed"
        args = f" {' '.join(map(str, self.args))}" if self.args else ""
        return (f"<{self.op}{args} @{where} ${self.value:,.0f}"
                f"{' !' + str(self.deadline_turn) if self.deadline_turn < LAST_TURN else ''}"
                f"{' ' + self.kind if self.kind else ''}>")


# --------------------------------------------------------------------------- pricing

def current_prices(obs) -> dict:
    return dict(obs.get("market", {}).get("prices", {}) or {})


def make_price_fn(obs, plan=None):
    """`price(product, days_ahead=0) -> $/unit`.

    The C6 seam, now filled. A harvest that lands in ten days should be priced at the inventory it
    will *meet*, not today's — pricing a plan on the board it was written against is what made melon
    look worthless (D17/E48), and ten days of shop drain is a whole price regime on a `T` of 100.

    `days_ahead` is 0 for everything that sells today (a ripe tile, tonight's milk) and the crop's
    lead time for everything that does not (a cohort's NPV, an animal's lifetime output). With the
    projection off — `plan.consts["projected_pricing"] == 0` — every lookup collapses to the live
    quote and this is the old function exactly.
    """
    prices = current_prices(obs)

    def spot(product, days_ahead: int = 0):
        return float(prices.get(product, 0) or 0)

    if plan is not None and not int(_const(plan, "projected_pricing")):
        return spot
    try:
        from agent import projection

        proj = projection.for_obs(obs, plan)
    except Exception:
        return spot

    def priced(product, days_ahead: int = 0):
        if product not in prices:
            return 0.0
        return proj.price(product, days_ahead)

    return priced


def _const(plan, name: str):
    """A plan const with C6's default when the plan predates the gene (hand-written plans do)."""
    from agent.projection import DEFAULTS

    v = (getattr(plan, "consts", None) or {}).get(name)
    return DEFAULTS[name] if v is None else v


def _quote(price, product: str, days_ahead: int = 0) -> float:
    """Call a price function that may or may not accept a lead time.

    Tests and older callers hand in a one-argument `lambda p: ...`; the projection's is two. Trying
    the richer signature first keeps both working without every call site knowing which it has.
    """
    if days_ahead:
        try:
            return price(product, days_ahead)
        except TypeError:
            pass
    return price(product)


# --------------------------------------------------------------------------- crop arithmetic

def water_window(crop: str) -> tuple[int, int]:
    """Ages over which a WATER adds yield to a one-time crop (`kaggriculture.py` WATER op)."""
    cd = CROPS[crop]
    return ((cd["max_yield_day"] + 1) // 2, cd["max_yield_day"])


def replant_cycle(crop: str) -> int:
    """Days between sowing a one-time crop and being able to sow that tile again.

    A one-time plant is stamped `max_lifespan_step = (planted_day + max_yield_day + 1) * 24`
    (`kaggriculture.py:224`), so it must be off the tile by the dawn of age `max_yield_day + 1`;
    the tile is cleared by the HARVEST and the replant task appears at the next dawn. For WHEAT
    (`max_yield_day` 4) that is a five-day round: sow d, water d+2..d+4 (`kaggriculture.py:440`),
    harvest d+4, sow again d+5.
    """
    cd = CROPS[crop]
    return 1 if cd["ongoing"] else int(cd["max_yield_day"]) + 1


def tick_days(crop: str) -> tuple[int, ...]:
    """Ages at whose dusk an ongoing crop produces.

    `days_since_first = next_day - planted - first_yield_day`, stepped by `interval` and capped at
    `max_yield` productions — so strawberry (first 10, interval 2, cap 4) pays at the end of ages
    9, 11, 13 and 15, which is why fertilizing at 9 and 13 covers all four.
    """
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return ()
    return tuple(cd["first_yield_day"] - 1 + k * cd["interval"] for k in range(cd["max_yield"]))


def is_tick_tonight(tile: dict, day: int) -> bool:
    return (day - tile["planted_day"]) in tick_days(tile["crop"])


def remaining_units(tile: dict, day: int) -> int:
    """Units this plant can still deliver if it is kept alive and worked.

    What a survival water is *worth*: a plant one day from its last tick is worth much less than a
    fresh one, and a ladder that treats every water as equally urgent cannot see that.
    """
    crop = tile["crop"]
    cd = CROPS[crop]
    age = day - tile["planted_day"]
    held = int(tile.get("yield_units", 0) or 0)
    if cd["ongoing"]:
        # Every remaining tick pays 2 under the plan's fertilizer schedule, and the cap binds the
        # tile rather than the season because the harvest cadence empties it between ticks.
        ticks_left = sum(1 for t in tick_days(crop) if t >= age)
        return held + 2 * ticks_left
    lo, hi = water_window(crop)
    waterings_left = max(0, hi - max(age, lo) + 1) if age <= hi else 0
    return min(cd["max_yield"], held + waterings_left)


def expiry_step(tile: dict) -> int | None:
    """The step at which a decaying plant becomes a WEED.

    `_decay_plants` runs every step and takes a unit every second one from `max_lifespan_step`, so
    a crop holding `n` units disappears at `mls + 2n`. Worth computing exactly: a wheat whose
    lifespan ends at this dawn is gone by turn 4, and both the water and the harvest routed to it
    are turns spent on a weed.
    """
    mls = int(tile.get("max_lifespan_step", -1))
    if mls < 0:
        return None
    return mls + 2 * max(0, int(tile.get("yield_units", 0) or 0))


def decay_step(tile: dict) -> int | None:
    """The step at which this plant starts losing units, if it is known.

    One-time crops know it from planting: `(planted + max_yield_day + 1) * turns_per_day`. Ongoing
    crops only learn it after their last production, when the field flips from -1.
    """
    mls = int(tile.get("max_lifespan_step", -1))
    return None if mls < 0 else mls


def _deadline(step_limit: int | None, day: int, turn: int) -> int:
    """Convert an absolute decay step into a turn-of-day deadline for today."""
    if step_limit is None:
        return LAST_TURN
    day_start = day * TURNS_PER_DAY
    if step_limit >= day_start + TURNS_PER_DAY:
        return LAST_TURN                       # not today's problem
    return max(turn, min(LAST_TURN, step_limit - day_start - 1))


# --------------------------------------------------------------------------- the generator

def daily_tasks(obs, plan, day: int | None = None, turn: int = 0, price_fn=None) -> list[Task]:
    """Everything worth doing on `day`, for the seat `obs` belongs to.

    Order is not priority: the router sorts by value against its own routing cost. What matters
    here is that nothing worth doing is missing and nothing impossible is present.
    """
    day = int(obs.get("day", 0) or 0) if day is None else day
    # The scarcity hunter runs *here* rather than in the shell, and that is deliberate: `daily_tasks`
    # is the single source of both the day's PLANT tasks and (through `verify._afford`) the dawn
    # BUY_SEED list, so a redirect applied here cannot leave the shopping list buying seed for a
    # crop the plan no longer sows. It is idempotent and remembered for the season — see
    # `projection.redirect`.
    try:
        from agent import projection

        plan = projection.redirect(obs, plan, day)
        # O3 counter-mix, on the redirect's output and for the same reason it lives here: this is
        # the one place both the PLANT tasks and the dawn BUY_SEED list are derived from, so a
        # cohort steered off a contested crop cannot leave dawn buying the seed it no longer sows.
        plan = projection.counter_mix(obs, plan, day)
    except Exception:
        pass
    price = price_fn or make_price_fn(obs, plan)
    farm = obs["farms"][_seat(obs)]
    private = obs.get("private", {}) or {}
    tiles = farm["tiles"]
    wanted = plan.occupied()

    tasks: list[Task] = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                tasks += _plant_tasks((x, y), tile, day, turn, price)
            elif isinstance(tile, dict) and tile.get("animal"):
                tasks += _animal_tasks((x, y), tile, day, price)
            elif isinstance(tile, dict) and tile.get("kind") == "WEED" and (x, y) in wanted:
                # A weed only matters on ground the plan wants back; digging elsewhere is walking.
                tasks.append(Task((x, y), "DIG", value=_dig_value(plan, (x, y), day, price),
                                  kind="dig"))
    tasks += _structure_tasks(plan, tiles, day, private, price)
    tasks += _planting_tasks(obs, plan, day, price, tiles, private)
    tasks += _logistics_tasks(tasks, private, day, price)
    return tasks


def _seat(obs) -> int:
    return 1 if int(obs.get("player", 0) or 0) == 1 else 0


# --------------------------------------------------------------------------- plants

def _plant_tasks(pos, tile, day: int, turn: int, price) -> list[Task]:
    crop = tile["crop"]
    cd = CROPS[crop]
    age = day - tile["planted_day"]
    unit_price = price(crop)
    out: list[Task] = []

    # Today's FERTILIZE lands *before* today's WATER (`router.OP_ORDER`: FERTILIZE 4, WATER 5, same
    # stop, back to back), so a tile that is dry now is fertilized by the time dusk reads it. Pricing
    # the water off `fertilized_until_day` alone therefore sees only half the tick days — for
    # strawberry, ages 11 and 15 are covered by an earlier application but 9 and 13 are fertilized
    # today — and emitted no water at all on the other half. That is E66 mechanism #2.
    fert_today = _fertilizes_today(tile, day)

    # --- water -------------------------------------------------------------
    if not tile["watered_today"]:
        doomed = _expires_today(tile, day)
        if doomed is not None and doomed <= LAST_TURN:
            pass          # decay takes this tile today whatever we do; watering it buys nothing
        elif int(tile.get("consecutive_unwatered", 0)) >= 1:
            # Dies at this dusk if it is not watered: `consecutive_unwatered >= 2` becomes a WEED.
            # Value is the whole rest of the plant, which is what makes it outbid everything else
            # on a route without needing a special case in the router.
            out.append(Task(pos, "WATER", value=remaining_units(tile, day) * unit_price,
                            kind="survival"))
        else:
            bonus = _water_bonus(tile, day, fert_today)
            if bonus:
                # `kaggriculture.py:797` — `fertilized = was_watered and fertilized_until_day >=
                # current_day`. On an ongoing crop's production night the fertilizer bonus is void
                # unless the tile was watered *that day*, so this water is not optional filler: miss
                # it and the whole application is wasted. Measured, the router watered 67.0% of
                # strawberry tick days against boatlee's 98.6% while watering *more* overall — the
                # water was landing on the wrong days (E66). Survival class is what fixes that: the
                # router rescues dropped survival stops onto idle units and `decide_hands` will not
                # sign off a staffing level that leaves one unrouted. It is the same guarantee a
                # thirsty plant gets, for the same reason — the turn is unrepeatable.
                one_shot = not cd["ongoing"]
                if not one_shot:
                    STATS["tick_waters"] += 1
                    if fert_today:
                        STATS["tick_waters_fert_today"] += 1
                out.append(Task(pos, "WATER", value=bonus * unit_price,
                                kind="bonus" if one_shot else "survival"))

    # --- fertilize ---------------------------------------------------------
    if fert_today:
        out.append(Task(pos, "FERTILIZE", value=_fertilize_gain(tile, day) * unit_price,
                        needs={"FERTILIZER": 1}, kind="fertilize"))

    # --- harvest -----------------------------------------------------------
    # For a one-time crop, HARVEST *clears the tile* — so harvesting mid-window is not "early
    # income", it is destroying the rest of the plant. Offer it only once the crop has stopped
    # accruing: at the cap, or past its last watering day.
    held = int(tile.get("yield_units", 0) or 0)
    ripe = held > 0 and age >= cd["first_yield_day"]
    if not cd["ongoing"]:
        ripe = ripe and (held >= cd["max_yield"] or age >= cd["max_yield_day"])
    if ripe:
        deadline = _deadline(decay_step(tile), day, turn)
        # A cap of 4 with a fertilized tick of +2 means an unharvested ongoing tile throws away its
        # next production; that is a deadline as real as decay, and invisible in the tile state.
        if cd["ongoing"] and is_tick_tonight(tile, day):
            tick = 2 if (fert_today or int(tile.get("fertilized_until_day", -1)) >= day) else 1
            if held + tick > cd["max_yield"]:
                deadline = min(deadline, LAST_TURN)
                value_bonus = (held + tick - cd["max_yield"]) * unit_price
            else:
                value_bonus = 0.0
        else:
            value_bonus = 0.0
        out.append(Task(pos, "HARVEST", value=held * unit_price + value_bonus,
                        deadline_turn=deadline, kind="harvest",
                        expires_turn=_expires_today(tile, day)))
    return out


def _expires_today(tile: dict, day: int) -> int | None:
    """Turn of *today* at which this tile becomes a weed, if that happens today."""
    step = expiry_step(tile)
    if step is None:
        return None
    turn = step - day * TURNS_PER_DAY
    return turn if 0 <= turn <= LAST_TURN else None


def _fertilizes_today(tile, day: int) -> bool:
    """Will a FERTILIZE task be emitted for this tile today?

    Kept as one predicate because two places need the same answer and E39's lesson is that a rule
    written twice is a rule that will disagree with itself: the FERTILIZE task is emitted from it,
    and the tick-day water is priced and classed from it.
    """
    crop = tile["crop"]
    age = day - tile["planted_day"]
    return (age in FERT_AGES.get(crop, ())
            and int(tile.get("fertilized_until_day", -1)) < day
            and _fertilize_gain(tile, day) > 0)


def _projected_units(tile, day: int) -> int:
    """Units the tile will still be holding at dusk, after today's harvest.

    An ongoing crop with produce on it and past its first yield day always gets a HARVEST task, so
    reading the *current* `yield_units` says a full tile has no headroom for tonight's tick and
    prices its water at zero — on exactly the ripe tiles where the bonus is largest.
    """
    cd = CROPS[tile["crop"]]
    held = int(tile.get("yield_units", 0) or 0)
    if cd["ongoing"] and held > 0 and (day - tile["planted_day"]) >= cd["first_yield_day"]:
        return 0
    return held


def _water_bonus(tile, day: int, fert_today: bool = False) -> int:
    """Units a WATER would add — directly for a one-time crop, at tonight's tick for an ongoing one.

    An ongoing crop's WATER op adds nothing by itself. It matters at dusk: `_daily_refresh_plants`
    grants `2 if fertilized else 1`, where `fertilized = was_watered and fertilized_until_day >=
    current_day` (`kaggriculture.py:797`). So on a production night with fertilizer live the water
    is worth exactly the difference — one unit, capped by `max_yield` — and on any other night it is
    worth nothing, because the tick pays 1 either way.
    """
    crop = tile["crop"]
    cd = CROPS[crop]
    cap = cd["max_yield"]
    fertilized = fert_today or int(tile.get("fertilized_until_day", -1)) >= day
    age = day - tile["planted_day"]
    if not cd["ongoing"]:
        held = int(tile.get("yield_units", 0) or 0)
        if held >= cap:
            return 0
        lo, hi = water_window(crop)
        if lo <= age <= hi:
            return min(2 if fertilized else 1, cap - held)
        return 0
    if not (is_tick_tonight(tile, day) and fertilized):
        return 0
    held = _projected_units(tile, day)
    return min(cap, held + 2) - min(cap, held + 1)


def _fertilize_gain(tile, day: int) -> int:
    """Extra units from fertilizing now — one application covers `day..day+2`."""
    crop = tile["crop"]
    cd = CROPS[crop]
    age = day - tile["planted_day"]
    if cd["ongoing"]:
        return sum(1 for t in tick_days(crop) if age <= t <= age + 2)
    lo, hi = water_window(crop)
    return sum(1 for a in range(max(age, lo), min(hi, age + 2) + 1))


# --------------------------------------------------------------------------- animals

def _animal_tasks(pos, tile, day: int, price) -> list[Task]:
    species = tile["animal"]
    spec = ANIMALS[species]
    product = spec["product"]
    unit_price = price(product)
    out: list[Task] = []

    produces_tonight = _animal_produces(tile, day)
    held_now = int(tile.get("yield_units", 0) or 0)
    if not tile.get("fed_today"):
        unfed = int(tile.get("consecutive_unfed", 0) or 0)
        if unfed >= 1:
            # Second unfed dusk and the animal escapes, leaving the structure empty.
            value = spec["cost"] + unit_price * spec["max_held"]
            kind = "survival"
        elif produces_tonight:
            # Feeding on a production day is what releases the care bank; skipping it pays 1 and
            # clears the bank (verified in tests/test_env_facts.py). The payout is capped by
            # `max_held`, so a bank of seven on a cow that holds six is worth six, not seven —
            # which is also why the plan harvests every cycle instead of hoarding.
            banked = 1 + int(tile.get("pending_care_bonus", 0) or 0)
            value = min(spec["max_held"] - held_now, banked) * unit_price
            kind = "feed"
        else:
            # Not producing tonight and in no danger: what is at stake is the care bank, which only
            # accrues on a day the animal is both fed and cared for, and pays a unit at the next
            # production.
            value = unit_price
            kind = "feed"
        out.append(Task(pos, "FEED", value=value, needs={"WHEAT": 1}, kind=kind))

    if not tile.get("cared_today"):
        # The care bank only accrues on a day the animal is both fed and cared for, and pays out at
        # the next production — so a care today is worth one unit then, not now.
        out.append(Task(pos, "CARE", value=unit_price, kind="care"))

    held = held_now
    if held > 0:
        deadline = LAST_TURN
        value = held * unit_price
        if produces_tonight and held + 1 > spec["max_held"]:
            value += unit_price          # tonight's production would overflow the bucket
        out.append(Task(pos, "HARVEST", value=value, deadline_turn=deadline, kind="harvest"))

    if tile.get("fertilizer_available"):
        out.append(Task(pos, "COLLECT_FERTILIZER", value=price("FERTILIZER"), kind="collect"))
    return out


def _animal_produces(tile, day: int) -> bool:
    spec = ANIMALS[tile["animal"]]
    since = (day + 1) - int(tile["placed_day"]) - spec["first_yield_day"]
    return since >= 0 and since % spec["interval"] == 0


# --------------------------------------------------------------------------- planting

def _structure_tasks(plan, tiles, day: int, private, price) -> list[Task]:
    """Build the animal cluster, and put the animals in it.

    The herd is the plan's second-largest asset and it does not appear on the board by itself: a
    structure has to be built on bare ground, the animal bought into the shed, carried out by a
    unit, and placed. Miss any link and the money is spent on livestock standing in a shed.

    Structures are matched to the herd in schedule order — the k-th tile of the cluster serves the
    k-th animal — so a plan that buys nine cows builds nine pastures and no more.
    """
    shed = dict(private.get("shed", {}) or {})
    out: list[Task] = []

    due = [(species, buy_day) for species, buy_day in plan.herd if buy_day <= day]
    for index, (species, _buy_day) in enumerate(due):
        if index >= len(plan.pasture_tiles):
            break
        pos = tuple(plan.pasture_tiles[index])
        x, y = pos
        tile = tiles[y][x]
        spec = ANIMALS[species]
        # Same lead-time argument as `_cohort_value`: an animal placed today meets the market at
        # `first_yield_day`, not this morning.
        product_value = _quote(price, spec["product"], int(spec["first_yield_day"])) \
            * spec["max_held"]

        if tile is None:
            out.append(Task(pos, f"BUILD_{spec['structure']}", value=product_value,
                            kind="build"))
        elif isinstance(tile, dict) and tile.get("kind") == spec["structure"] \
                and "animal" not in tile and shed.get(species, 0) > 0:
            # PLACE takes the animal from the *unit's* inventory, so it has to be carried out.
            shed[species] -= 1
            out.append(Task(pos, "PLACE", args=(species,), value=product_value,
                            needs={species: 1}, kind="place"))
    return out


def _planting_tasks(obs, plan, day: int, price, tiles, private) -> list[Task]:
    """Cohorts due today, plus replants on tiles a cycling cohort has had harvested.

    A PLANT is admitted here on its own merits; whether the day can actually *sustain* it — the
    water it will need tomorrow and the day after — is C3's call, because that depends on the route.
    """
    seeds = dict(private.get("seeds", {}) or {})
    out: list[Task] = []
    for cohort in plan.cohorts:
        if cohort.plant_day > day:
            continue
        # A cohort's plant day is the earliest day, not the only one, and the only thing that
        # closes it is the horizon: nothing goes in that cannot reach its first yield before the
        # season ends (PLAN_v4 §2.4).
        #
        # A fixed catch-up window was tried and is wrong, because it is anchored to the day the
        # *plan* hoped for the land. Cash delayed one measured NE purchase from day 6 to day 12, by
        # which time the window on a 22-tile strawberry block had closed and the largest cohort of
        # the season was never planted at all. The land arriving late is exactly when the catch-up
        # is needed.
        if day + CROPS[cohort.crop]["first_yield_day"] > LAST_DAY:
            continue
        free = [(x, y) for (x, y) in cohort.tiles if tiles[y][x] is None]
        # occupied, weeded or still LOCKED tiles are not plantable, and a cycling cohort that has
        # already been round its loop is re-sown a wave a day (see `wave_size`).
        for (x, y) in free[:wave_size(cohort, day, cohort_opens(plan, cohort))]:
            out.append(Task((x, y), "PLANT", args=(cohort.crop,),
                            value=_cohort_value(cohort.crop, price),
                            needs={f"SEED:{cohort.crop}": 1}, kind="plant"))
    # Seeds are bought by C4 at hour 0; a PLANT without one is dropped by the environment's atomic
    # rule, so the task carries the requirement rather than assuming stock.
    _ = seeds
    return out


#: Waves of catch-up a cycling block may carry. 1 is the break-even rate that just keeps a phased
#: block standing; see `wave_size` for why the extra slack that scores better on the tile counter is
#: not the setting that ships.
WAVE_SLACK = 1


def cohort_opens(plan, cohort) -> int:
    """The first day this cohort could go in: its plan day, or the day its quadrant is bought."""
    if cohort.quadrant == "NW":
        return int(cohort.plant_day)
    return max(int(cohort.plant_day), int((plan.land_days or {}).get(cohort.quadrant, 0)))


def wave_size(cohort, day: int | None = None, opens: int | None = None) -> int:
    """How many tiles of a cohort may be *re*-sown on one day.

    A `replant` cohort is meant to be a standing base — the plan says "ten tiles of wheat, always"
    and encodes it as one gene (`agent/plan.py::Cohort`). Sowing every tile on the same day does not
    produce that: the whole block matures together, is harvested together, and the tile count
    collapses to zero for a dawn before the next batch goes in. Measured on 49100:49104 the base
    oscillated 0-10 around a mean of 7.1, which starves both the wheat revenue and
    `_feedable_animals`' tile term (E66 mechanism 3).

    So once the block has been round the loop, it is re-sown at a *rate*. The first batch harvest is
    thereby spread over a cycle and the block comes back phased, and it stays that way — the cap is
    self-healing, so any later event that empties several tiles at once is re-spread rather than
    re-batched.

    The rate is `ceil(WAVE_SLACK x tiles / cycle)` and `WAVE_SLACK` is **1** — break-even, with no
    room to make up a day the verifier prunes. That is a deliberate choice *against* the tile
    counter: `WAVE_SLACK = 2` holds a visibly better base (mean 7.06 against 6.22 on 49000:49020 vs
    `starter`, same collapse-day count) and **loses $12.5k** on a held-out block (49040:49080, 80
    paired games, CI [-19.9k, -5.1k]) while gaining $3.9k against `boatlee`. Slack of 1 is the only
    setting positive against both opponents on both blocks (4/4 arms, pooled +$2.2k). A faster
    refill buys tiles with the crew-turns and cash the strawberry block and the herd are competing
    for, and wheat is not worth what they are — the farm eats almost all of it (26 units sold a
    season against ~220 produced), so a wheat tile is a feed subsidy, not revenue.

    **The first fill is deliberately uncapped**, and that is not a detail. Capping it costs
    **-$45.8k** (80 paired games, 49000:49040, against `starter`): the day-0 wheat block is the
    farm's entire cash engine, and a block that reaches full strength on day 4 instead of day 0 does
    not pay for the 22-tile strawberry cohort on day 7 — `plants_started` fell 102 -> 66 and
    `strawberry_per_plant` 5.5 -> 1.2. This is `_planting_tasks`' late-land lesson again: throttling
    the *arrival* of a block is never the same trade as throttling its refill.

    Two other mechanisms were rejected. A fixed per-tile phase (`(day - plant_day) % cycle`) is
    rigid: a tile that misses its slot to cash or to C3's pruning waits a whole cycle for another.
    Splitting the cohort into sub-cohorts at decode time changes the genome's meaning and the plan's
    tile map for a scheduling detail the compiler can own.

    Non-cycling cohorts are unaffected: they are sown once, in full, exactly as before.
    """
    if not getattr(cohort, "replant", False):
        return len(cohort.tiles)
    cycle = replant_cycle(cohort.crop)
    if day is not None and day < (cohort.plant_day if opens is None else opens) + cycle:
        return len(cohort.tiles)
    return max(1, -(-len(cohort.tiles) * WAVE_SLACK // cycle))


def _cohort_value(crop: str, price) -> float:
    """What one tile of this crop is worth over its life, net of the seed.

    Priced at the board the crop will *meet*, not the one it is sown on (C1′): a strawberry sown
    today meets its first unit ten days out, by which time the town has eaten another 60-120 units
    of it. This is the number the redirect competes on, so getting the lead time in is the whole
    point of the seam.
    """
    cd = CROPS[crop]
    unit_price = _quote(price, crop, int(cd["first_yield_day"]))
    if cd["ongoing"]:
        units = cd["max_yield"] * 2                     # fertilized, harvested on cadence
    else:
        lo, hi = water_window(crop)
        units = min(cd["max_yield"], 1 + (hi - lo + 1))
    return units * unit_price - SEED_COST[crop]


def _dig_value(plan, pos, day: int, price) -> float:
    """A weed is worth clearing only if the plan wants to plant there again this season."""
    for cohort in plan.cohorts:
        if pos in cohort.tiles and (cohort.replant or cohort.plant_day >= day):
            return _cohort_value(cohort.crop, price) * 0.5
    return 1.0


# --------------------------------------------------------------------------- logistics

def _logistics_tasks(tasks, private, day: int, price) -> list[Task]:
    """PICKUPs for what the day's work needs, and a DROP when the shed is near its cap.

    Quantities are totals: the router splits them across the units it actually assigns. Emitting
    one PICKUP per carrier here would be deciding the assignment, which is C2's job.
    """
    shed = dict(private.get("shed", {}) or {})
    inventories = [dict(inv) for inv in (private.get("inventories") or [])]
    carried = {}
    for inv in inventories:
        for item, n in inv.items():
            carried[item] = carried.get(item, 0) + int(n)

    need: dict = {}
    for t in tasks:
        for item, n in (t.needs or {}).items():
            if item.startswith("SEED:"):
                continue                       # seeds never pass through a unit's hands
            need[item] = need.get(item, 0) + n

    out: list[Task] = []
    for item, n in sorted(need.items()):
        short = n - carried.get(item, 0)
        if short > 0:
            # The full shortfall, not what the shed holds right now: C4 buys at hour 0 and the
            # pickup happens after, so sizing to the current shed would under-carry every morning.
            out.append(Task(None, "PICKUP", args=(item, short), value=0.0, kind="pickup"))

    # Dusk drops every unit inventory into the shed and *discards* the overflow, so a full shed
    # turns a day's harvest into nothing. Selling happens at the shed, so the DROP has to land
    # before dusk rather than being deferred to tomorrow.
    projected = sum(shed.values()) + sum(carried.values())
    if projected > SHED_PRESSURE:
        out.append(Task(None, "DROP", args=("ALL",), value=float(projected - SHED_PRESSURE),
                        kind="drop"))
    return out


# --------------------------------------------------------------------------- for C4

def market_needs(tasks, private) -> dict:
    """What must be bought at hour 0 for today's tasks to be doable.

    Buy-on-demand rather than stockpiling: `BUY_PRODUCT` fails outright against a full shed, and a
    shed full of speculative wheat is a shed that discards the evening's harvest.
    """
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    carried: dict = {}
    for inv in (private.get("inventories") or []):
        for item, n in dict(inv).items():
            carried[item] = carried.get(item, 0) + int(n)

    want: dict = {}
    for t in tasks:
        for item, n in (t.needs or {}).items():
            want[item] = want.get(item, 0) + n

    out: dict = {}
    for item, n in want.items():
        if item.startswith("SEED:"):
            crop = item.split(":", 1)[1]
            short = n - int(seeds.get(crop, 0))
            if short > 0:
                out[f"SEED:{crop}"] = short
        else:
            short = n - int(shed.get(item, 0)) - int(carried.get(item, 0))
            if short > 0:
                out[item] = short
    return out
