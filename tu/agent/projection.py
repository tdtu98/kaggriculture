"""C6 — what the market will look like when the harvest actually arrives.

Three things live here, and they are the same idea applied at three ranges:

**`project(product, days, obs)`** — the market inventory curve. Everything that moves inventory is
in the source and none of it is a guess: the town's shops eat a fixed number of units per day, the
town centre eats one, and both farms' standing crops and animals are visible on the board, so their
supply is a schedule rather than a forecast. Only *future* shop unlocks are random, and their
identity is drawn uniformly, so they enter as an expectation rather than a shrug.

**`scarcity_signal(product, obs)`** — `(I0 - min projected inventory) / T`, the same units the price
curve is written in. `T` is one field's seasonal output, so a signal of 1.0 means the town has eaten
a whole field's worth more than the board has produced; for the three `hinge` products that is
exactly where the price stops being calm and runs away (`kaggriculture.py:54-73`).

**The scarcity hunter** — `redirect()` points the next unplanted cohort at a product the town is
starving for, and `scarcity_plan()` puts a price floor under it so the cohort is sold into the spike
rather than through it.

Why this is not "price it at today's quote": pricing a harvest at the board it was *planned* on is
what made melon look worthless (D17/E48). Ten days of shop drain is 60-120 units on a `T` of 100-200
— a whole price regime — and the plan that reacts to it has to see it before it plants.

Every mechanic below cites the line in
`.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`
that sets it. The numbers were read there, not from the task text: the spec's "drain 6/day" is
right, but it is right *because* `townShopSellInterval` is 4 and a day is 24 turns, and it stops
being right the moment a configuration changes either — so it is computed, not written down.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

TURNS_PER_DAY = 24
LAST_DAY = 29

#: `_town_consume` runs on every step with `step % townShopSellInterval == 0`
#: (`kaggriculture.py:733,736`), so with the default interval of 4 and a 24-turn day the shops eat
#: at hours 0, 4, 8, 12, 16 and 20 — six times a day, every day, forever.
SHOP_SELL_INTERVAL = 4
SHOP_HOURS = tuple(range(0, TURNS_PER_DAY, SHOP_SELL_INTERVAL))
SHOP_EVENTS_PER_DAY = len(SHOP_HOURS)

#: The town centre eats one of every non-fertilizer product on `step % townCenterSellInterval == 0`
#: (`kaggriculture.py:734,745-747`) — once a day, at hour 0.
CENTER_INTERVAL = TURNS_PER_DAY
CENTER_HOURS = (0,)

#: A shop instance is appended at the *end* of day `d` when `(d + 1) % townShopUnlockInterval == 0`
#: and the town is under `MAX_SHOP_INSTANCES` (`kaggriculture.py:867,886-891`). So the count present
#: *during* day D is `min(8, D // 3)` — nothing on days 0-2, one from day 3, eight from day 24.
SHOP_UNLOCK_INTERVAL = 3

#: Which shop unlocks is `rng.choice(sorted(SHOPS))` off a seed we cannot see
#: (`kaggriculture.py:874,891`) — uniform over the eight names, drawn **with replacement**. So a
#: future instance contributes its expectation: the mean drain over the eight shops.
SHOP_NAMES = tuple(sorted(SHOPS))

#: Horizon in days. A strawberry cohort is ten days from sowing to its first unit, which is the
#: longest lead any decision here has to price.
HORIZON = 14

#: What one production is worth when the animal is cared for every day. `pending_care_bonus`
#: accrues +1 per fed+cared day and is spent on the production day (`kaggriculture.py:826-830`), so
#: an animal on a k-day interval banks k bonuses between productions: 1 + interval units, capped at
#: `max_held`.
def _animal_units_per_production(species: str) -> int:
    a = ANIMALS[species]
    return min(int(a["max_held"]), 1 + int(a["interval"]))


#: A fertilized ongoing crop pays 2 units per tick rather than 1 (`kaggriculture.py:797`). Both
#: sides of every measured game fertilize, so assuming the bonus is closer than assuming it away.
ONGOING_UNITS_PER_TICK = 2

#: Fraction of forecast production that actually reaches the market, per product.
#:
#: Everything above is arithmetic from the source. This is the one measured constant, and it is
#: measured rather than reasoned because the gap between "produced" and "sold" is the mistake
#: CLAUDE.md's "an order is not a sale" rule exists to catch, one level up: a farm eats, hoards,
#: overflows its shed and buys back. Fitted on the spent 45000 block (6 seeds x {boatlee, starter},
#: 960 five-day predictions, minimising median error on days where the deviation is at least 0.25T)
#: and then verified on the untouched 52000 block.
#:
#: **WHEAT is ~0.** The farm eats almost all of it: 26 units sold against ~220 produced in a
#: measured season (E68), and what is short gets *bought back*, which moves the inventory the other
#: way. Forecasting wheat production as market supply was a 27.5% error; dropping it is 7.4%.
#: STRAWBERRY 0.6 and MILK 0.7 are the same story more weakly — plants die, ticks are missed, the
#: shed overflows at dusk. The rest are sold as fast as they appear.
SELL_THROUGH = {
    "WHEAT": 0.15, "CARROT": 1.0, "TOMATO": 1.0, "STRAWBERRY": 0.6, "MELON": 1.0,
    "EGG": 1.0, "MILK": 0.7, "WOOL": 0.9, "FERTILIZER": 0.5,
}

#: Products the redirect can actually chase.
#:
#: Two exclusions, for different reasons. EGG, MILK and WOOL come from animals, so a starving market
#: in them is a signal the redirect can price but not answer. **STRAWBERRY is excluded because it is
#: always scarce** — three shops against a `T` of 100 puts its signal above any sensible `theta_sqrt`
#: from about day 6 of every game — so including it turned the hunter into a constant plan edit
#: rather than a response to a spike: measured, it converted the day-11 melon cohort to strawberry on
#: **110 of 120 seeds**, worth +$1,513 vs boatlee (CI [-260, +3,286]) and **-$4,215 vs starter**
#: (CI [-6,267, -2,163]) over 240 paired games. "Plant strawberry instead of melon" is a cohort gene
#: — S2's job, where it is searched against both opponents — not something the hunter should decide
#: every game on a signal that is never off. What is left is what C6 was for: the two `hinge` crops,
#: which are calm until they genuinely run out (`kaggriculture.py:54-73`).
REDIRECTABLE = ("TOMATO", "CARROT")

#: Which threshold gene governs which product. `hinge` products are calm below the knee and spike
#: above it, so their signal has to clear ~1.0 to be worth chasing; `sqrt`/`log` products pay
#: smoothly and are worth chasing much earlier (`kaggriculture.py:41-50`).
def threshold_gene(product: str) -> str:
    return "theta_hinge" if MARKET_PARAMS[product]["below_func"] == "hinge" else "theta_sqrt"


def _shop_multiplier(shop: str, item: str) -> int:
    """Units of `item` one instance of `shop` eats per consume event.

    `multiplier = 2 if len(products) == 1 else 1` (`kaggriculture.py:741`) — a single-product shop
    eats double. That is the whole reason WOOL and CARROT can starve while WHEAT never does.
    """
    products = SHOPS[shop]
    if item not in products:
        return 0
    return 2 if len(products) == 1 else 1


#: `{item: units per day per instance}` for each shop name.
SHOP_DRAIN_PER_DAY = {
    shop: {item: SHOP_EVENTS_PER_DAY * _shop_multiplier(shop, item)
           for item in TOWN_CENTER_PRODUCTS + ["FERTILIZER"]
           if _shop_multiplier(shop, item)}
    for shop in SHOP_NAMES
}

#: Expected daily drain of one *unknown* future instance.
EXPECTED_DRAIN_PER_INSTANCE = {
    item: sum(SHOP_DRAIN_PER_DAY[s].get(item, 0) for s in SHOP_NAMES) / len(SHOP_NAMES)
    for item in TOWN_CENTER_PRODUCTS + ["FERTILIZER"]
}


def instances_present(day: int) -> int:
    """Shop instances consuming during `day` (`kaggriculture.py:886-891`)."""
    return max(0, min(MAX_SHOP_INSTANCES, int(day) // SHOP_UNLOCK_INTERVAL))


# --------------------------------------------------------------------------- effect counters

#: Season totals, per seat, so "did the hunter ever fire" is answerable from the harness rather than
#: from the last call (E44). Handed to `main_v4._STATE[seat]["effects"]` when that module is
#: present, and kept here as well so the tests can read them without an agent.
STATS: dict = {}


def reset_stats(seat: int = 0) -> None:
    STATS[seat] = {}


def stats(seat: int = 0) -> dict:
    return dict(STATS.get(seat) or {})


def _count(obs, key: str, n: int = 1) -> None:
    seat = _seat(obs)
    STATS.setdefault(seat, {})
    STATS[seat][key] = STATS[seat].get(key, 0) + int(n)
    # Mirror into the shell's ledger, which is what `harness/counters.py:_effect_counts` reads.
    # Imported lazily: `main_v4` imports this module's consumers, so a top-level import is a cycle,
    # and a projection used from a test has no shell at all.
    try:
        from agent import main_v4

        main_v4._note(main_v4._state(obs, seat), key, n)
    except Exception:
        pass


def _seat(obs) -> int:
    return 1 if int(obs.get("player", 0) or 0) == 1 else 0


# --------------------------------------------------------------------------- the projection

class Projection:
    """Projected market inventory for every product, for `horizon` days from `obs`.

    Built once per turn and cached on the observation (`for_obs`): `daily_tasks` prices every task
    through it, and recomputing a 14-day two-farm forecast per tile is how a 0.3 ms generator
    becomes a 30 ms one.
    """

    __slots__ = ("day", "hour", "shops", "curve", "_price_cache", "horizon")

    def __init__(self, obs, plan=None, horizon: int = HORIZON):
        self.day = int(obs.get("day", 0) or 0)
        self.hour = int(obs.get("hour", 0) or 0)
        self.horizon = int(horizon)
        self.shops = list((obs.get("town", {}) or {}).get("unlocked_shops") or [])
        inventory = dict((obs.get("market", {}) or {}).get("inventory", {}) or {})
        supply = _supply_schedule(obs, self.day, self.horizon)
        self.curve = {}
        self._price_cache = {}
        for item in MARKET_PARAMS:
            self.curve[item] = self._integrate(item, float(inventory.get(item, 10_000)), supply)

    # -- construction -------------------------------------------------------

    def _integrate(self, item: str, now: float, supply: dict) -> list:
        """`[inv_0 .. inv_horizon]`; `inv_k` is the dawn of day `today + k`, `inv_0` is right now.

        The first step is a partial day — the board already shows this morning's consumption, and
        double-counting it is a 6-unit error on a `T` of 100 — so today contributes only the
        consume events still ahead of `hour`.
        """
        arrivals = supply.get(item, {})
        out = [now]
        inv = now
        for k in range(1, self.horizon + 1):
            inv -= self._drain_rest_of_today(item) if k == 1 else self._drain(item, self.day + k - 1)
            inv += arrivals.get(k, 0.0)
            out.append(inv)
        return out

    def _drain(self, item: str, day: int) -> float:
        """A full day of town demand on `day`: known instances, expected new ones, town centre."""
        known = sum(SHOP_DRAIN_PER_DAY[s].get(item, 0) for s in self.shops)
        extra = max(0, instances_present(day) - len(self.shops))
        centre = 1.0 if item in TOWN_CENTER_PRODUCTS else 0.0
        return known + extra * EXPECTED_DRAIN_PER_INSTANCE.get(item, 0.0) + centre

    def _drain_rest_of_today(self, item: str) -> float:
        """Only the consume events still ahead of us today — the board already shows the rest."""
        # `>=`, not `>`: the agent acts *before* `_town_consume` runs for this step
        # (`kaggriculture.py:728` is called from the interpreter after the actions resolve), so at
        # hour 0 all six of today's events are still ahead of us.
        events = sum(1 for h in SHOP_HOURS if h >= self.hour)
        known = sum(SHOP_DRAIN_PER_DAY[s].get(item, 0) / SHOP_EVENTS_PER_DAY for s in self.shops)
        extra = max(0, instances_present(self.day) - len(self.shops))
        per_event = known + extra * EXPECTED_DRAIN_PER_INSTANCE.get(item, 0.0) / SHOP_EVENTS_PER_DAY
        centre = 1.0 if (item in TOWN_CENTER_PRODUCTS and self.hour <= CENTER_HOURS[0]) else 0.0
        return events * per_event + centre

    # -- reading ------------------------------------------------------------

    def inventory(self, product: str, days_ahead: int = 0) -> float:
        curve = self.curve.get(product)
        if not curve:
            return 10_000.0
        return curve[max(0, min(self.horizon, int(days_ahead)))]

    def project(self, product: str, days: int | None = None) -> list:
        days = self.horizon if days is None else min(self.horizon, int(days))
        return list(self.curve.get(product, [10_000.0])[:days + 1])

    def price(self, product: str, days_ahead: int = 0) -> float:
        if product not in MARKET_PARAMS:
            return 0.0
        key = (product, max(0, min(self.horizon, int(days_ahead))))
        hit = self._price_cache.get(key)
        if hit is None:
            hit = float(market_price(product, int(round(self.inventory(*key)))))
            self._price_cache[key] = hit
        return hit

    def signal(self, product: str, days: int | None = None) -> float:
        """`(I0 - min projected inventory) / T` — scarcity in the price curve's own units."""
        p = MARKET_PARAMS.get(product)
        if p is None:
            return 0.0
        return (p["I0"] - min(self.project(product, days))) / float(p["T"])


_CACHE: dict = {}


def for_obs(obs, plan=None, horizon: int = HORIZON) -> Projection:
    """The projection for this turn, built at most once per (seat, step)."""
    seat = _seat(obs)
    step = int(obs.get("step", 0) or 0)
    if step == 0:
        # Season reset. These are module globals and the harness rebuilds the *agent* per game, not
        # the module — a hunter that remembered last season's redirect would plant tomato on a board
        # that never spiked, and the counters would report a fire that belonged to another game.
        _REDIRECTS.pop(seat, None)
        _REDIRECTS.pop((seat, "crop"), None)
        STATS.pop(seat, None)
    key = (seat, step, int(horizon))
    hit = _CACHE.get(seat)
    if hit is not None and hit[0] == key:
        return hit[1]
    proj = Projection(obs, plan=plan, horizon=horizon)
    _CACHE[seat] = (key, proj)
    _count(obs, "projection_calls")
    return proj


def project(product: str, days: int, obs) -> list:
    """Module-level form of the spec's signature: `project(product, days, state) -> [inv_t]`."""
    return for_obs(obs, horizon=max(1, int(days))).project(product, days)


def scarcity_signal(product: str, obs, days: int | None = None) -> float:
    return for_obs(obs).signal(product, days)


# --------------------------------------------------------------------------- supply forecast

def _supply_schedule(obs, today: int, horizon: int) -> dict:
    """`{item: {days_ahead: units}}` — everything both farms are about to put on the market.

    O2's opponent model does not exist yet, so this is the tiles-only version the task calls for:
    their standing crops and animals are on the shared board (`obs["farms"]`), and a plant's whole
    remaining schedule follows from its crop and planting day. It assumes both sides sell on sight,
    which is what every agent in the pool actually does (E11).
    """
    out: dict = {}
    farms = obs.get("farms") or []
    for farm in farms:
        _farm_supply(farm, today, horizon, out)
    # Our own shed is stock that exists and is already priced into nothing — it sells at the next
    # market turn. Theirs is invisible, which is the asymmetry O2 is for.
    shed = ((obs.get("private", {}) or {}).get("shed") or {})
    for item, n in shed.items():
        if item in MARKET_PARAMS and int(n or 0) > 0:
            _add(out, item, 1, float(n), horizon)
    return out


def _add(out: dict, item: str, days_ahead: int, units: float, horizon: int) -> None:
    if units <= 0 or days_ahead < 1 or days_ahead > horizon:
        return
    slot = out.setdefault(item, {})
    slot[days_ahead] = slot.get(days_ahead, 0.0) + float(units) * SELL_THROUGH.get(item, 1.0)


def _farm_supply(farm, today: int, horizon: int, out: dict) -> None:
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                _plant_supply(tile, today, horizon, out)
            elif tile.get("animal"):
                _animal_supply(tile, today, horizon, out)


def _plant_supply(tile, today: int, horizon: int, out: dict) -> None:
    crop = tile.get("crop")
    cd = CROPS.get(crop)
    if cd is None:
        return
    planted = int(tile.get("planted_day", today))
    held = int(tile.get("yield_units", 0) or 0)
    # Whatever is already ripe reaches the market at the next market turn.
    _add(out, crop, 1, held, horizon)

    if cd["ongoing"]:
        # Ticks at ages `first_yield_day - 1 + k*interval` for `max_yield` productions
        # (`kaggriculture.py:783-800`), each at dusk — so the units meet a market turn the next day.
        for k in range(int(cd["max_yield"])):
            age = int(cd["first_yield_day"]) - 1 + k * int(cd["interval"])
            _add(out, crop, planted + age + 1 - today, ONGOING_UNITS_PER_TICK, horizon)
    else:
        # A one-time crop is harvested at `max_yield_day` and sold the same day; it carries one
        # unit per watered day inside its bonus window, capped at `max_yield`.
        window = int(cd["max_yield_day"]) - (int(cd["max_yield_day"]) + 1) // 2 + 1
        units = min(int(cd["max_yield"]), 1 + window) - held
        _add(out, crop, planted + int(cd["max_yield_day"]) - today, units, horizon)


def _animal_supply(tile, today: int, horizon: int, out: dict) -> None:
    species = tile.get("animal")
    a = ANIMALS.get(species)
    if a is None:
        return
    placed = int(tile.get("placed_day", today))
    per = _animal_units_per_production(species)
    _add(out, a["product"], 1, int(tile.get("yield_units", 0) or 0), horizon)
    # Production on day d when `(d + 1) - placed_day - first_yield_day >= 0` and divisible by
    # `interval` (`kaggriculture.py:822-823`); the units meet a market turn the day after.
    d = placed + int(a["first_yield_day"]) - 1
    while d - today <= horizon:
        _add(out, a["product"], d + 1 - today, per, horizon)
        d += int(a["interval"])
        if d > today + horizon + int(a["interval"]):
            break


# --------------------------------------------------------------------------- the hunter

#: Per-seat, per-season memory of what the hunter committed to. A redirect that flickers on and off
#: with the signal plants half a cohort of one crop and half of another; once a cohort has been
#: pointed at a product it stays pointed at it.
_REDIRECTS: dict = {}


def reset(seat: int | None = None) -> None:
    """Season reset. `main_v4` rebuilds its per-seat state on step 0; this follows it."""
    if seat is None:
        _REDIRECTS.clear()
        _CACHE.clear()
        STATS.clear()
        return
    _REDIRECTS.pop(seat, None)
    _CACHE.pop(seat, None)
    STATS.pop(seat, None)


#: Default gene values, used when a plan predates the genes (every hand-written plan does).
#: Measured on 240 paired games over three blocks (51300/51400/52000), both seats, vs `starter` and
#: `boatlee`; see the C6 report. `projected_pricing` is **0** because pricing every task at the
#: projected board *lost* $2,675 vs starter (CI [-4,602, -749]) while doing nothing vs boatlee
#: (+$46) — the seam is built, proven to fire, and off until a search finds a use for it.
#: `scarce_floor` is 0 for the same reason in the other direction: at 0.6 the floor provably never
#: binds (identical money to the cent), at 0.95 it binds and is worth +$800 / -$56 — null. Both are
#: genes, so S2 decides; what ships is the setting that is measured not to hurt.
DEFAULTS = {
    "theta_hinge": 1.0,
    "theta_sqrt": 0.5,
    "max_scarce_tiles": 8,
    "scarce_floor": 0.0,
    "projected_pricing": 0,
}


def _consts(plan) -> dict:
    return (getattr(plan, "consts", None) or {})


def _gene(plan, name: str, default=None) -> float:
    v = _consts(plan).get(name)
    if v is None:
        v = DEFAULTS[name] if default is None else default
    return float(v)


def scarce_products(obs, plan, products=None) -> list:
    """Products whose projected scarcity clears their threshold gene, worst first."""
    proj = for_obs(obs, plan)
    out = []
    for product in (products or tuple(MARKET_PARAMS)):
        if product == "FERTILIZER":
            continue
        theta = _gene(plan, threshold_gene(product))
        if theta <= 0:
            continue
        s = proj.signal(product)
        if s >= theta:
            out.append((s, product))
    out.sort(reverse=True)
    return [p for _s, p in out]


def redirect(obs, plan, day: int):
    """Point the next unplanted cohort(s) at a starving crop. Returns a (possibly new) Plan.

    Only cohorts that have not gone in yet are eligible, and only up to `max_scarce_tiles` tiles:
    the hunter is a redirect, not a rewrite. A crop that cannot reach its first yield before the
    season ends is not a candidate at all — that is the same horizon test `_planting_tasks` applies,
    and it is the difference between chasing a spike and burning the seed money.

    **The threshold is a filter, not the decision.** The decision is `tile_value`: the redirect
    happens only when the scarce crop is worth more per tile, at the price it will *meet*, than the
    crop the plan already chose, by `SWITCH_MARGIN`. Without that the hunter fires every game, on
    every seed — STRAWBERRY sits in three shops against a `T` of 100 and is structurally scarce
    from day 6 onwards, so a threshold alone would turn "chase the spike" into "always plant
    strawberry", which is not a hypothesis, it is a different plan. With it, the counters are zero
    on a seed where nothing spiked, which is exactly what makes them evidence.
    """
    if not getattr(plan, "cohorts", ()):
        return plan
    for_obs(obs, plan)                 # builds the projection *and* runs the step-0 season reset,
    seat = _seat(obs)                  # which must happen before the memories below are read
    budget = int(_gene(plan, "max_scarce_tiles"))
    committed = _REDIRECTS.setdefault(seat, {})
    crops = _REDIRECTS.setdefault((seat, "crop"), {})

    if budget > 0 and not _closed(committed, budget):
        proj = for_obs(obs, plan)
        wanted = [p for p in scarce_products(obs, plan, REDIRECTABLE)
                  if day + int(CROPS[p]["first_yield_day"]) <= LAST_DAY]
        spent = sum(committed.values())
        for target in wanted:
            gain = tile_value(target, proj)
            for i, c in enumerate(plan.cohorts):
                if spent >= budget:
                    break
                if i in committed or c.crop == target or c.replant:
                    continue
                if c.plant_day < day or not c.tiles:
                    continue                    # already open; redirecting it re-sows mixed ground
                if gain < tile_value(c.crop, proj) * (1.0 + SWITCH_MARGIN):
                    continue                    # scarce, but not worth more than what is planned
                take = min(len(c.tiles), budget - spent)
                if take < len(c.tiles):
                    continue                    # a part-cohort is two half-plans; skip it
                committed[i] = take
                spent += take
                crops[i] = target
                _count(obs, "scarce_cohorts")
                _count(obs, "scarce_tiles", take)
                _count(obs, f"scarce_to_{target}", take)
                _count(obs, f"scarce_from_{c.crop}", take)

    if not crops:
        return plan

    from dataclasses import replace

    cohorts = tuple(replace(c, crop=crops[i]) if i in crops else c
                    for i, c in enumerate(plan.cohorts))
    return replace(plan, cohorts=cohorts)


def _closed(committed: dict, budget: int) -> bool:
    return sum(committed.values()) >= budget


#: How much better the scarce crop has to be before the plan is overridden. A redirect throws away
#: a cohort the search chose and pays a new seed bill, so a coin-flip improvement is not worth it.
SWITCH_MARGIN = 0.25


def tile_value(crop: str, proj: "Projection") -> float:
    """What one tile of `crop` is worth over its life, at the price it will actually meet.

    The same arithmetic as `tasks._cohort_value`, deliberately duplicated in miniature rather than
    imported: `tasks` imports this module, and a cycle here would be paid at every dawn. Kept honest
    by `tests/test_projection.py::test_tile_value_matches_the_task_generator`.
    """
    cd = CROPS[crop]
    if cd["ongoing"]:
        units = int(cd["max_yield"]) * ONGOING_UNITS_PER_TICK
    else:
        window = int(cd["max_yield_day"]) - (int(cd["max_yield_day"]) + 1) // 2 + 1
        units = min(int(cd["max_yield"]), 1 + window)
    return units * proj.price(crop, int(cd["first_yield_day"])) - int(cd["seed"])


def scarcity_plan(obs, plan, seat: int | None = None):
    """`plan` with a sell floor under every product the town is currently starving for.

    This is the metering half of the hunter. `main_v4._quantity_above_floor` already binary-searches
    the largest quantity whose *last* unit clears `floor x base` — which is exactly "sell in batches
    that stay above the floor", sized to the curve instead of to a guess. So the hunter does not
    need its own sell path; it needs to tell the existing one that this product is worth defending,
    and at what price.

    The floor is quoted against the *spike*, not the base: `scarce_floor x projected price / base`,
    so a tomato at $400 is defended at 0.75 x $400 = $300 rather than at 0.75 x $60.
    """
    floor = _gene(plan, "scarce_floor")
    if floor <= 0:
        return plan
    hot = scarce_products(obs, plan)
    if not hot:
        return plan
    proj = for_obs(obs, plan)
    floors = dict(_consts(plan).get("sell_floor") or {})
    fired = 0
    for product in hot:
        base = float(MARKET_PARAMS[product]["base"])
        spot = proj.price(product, 0)
        if spot <= base:
            # Scarce on the inventory counter but not yet paying for it: a floor here would only
            # withhold stock at the price it was going to fetch anyway.
            continue
        want = floor * spot / base
        if want > floors.get(product, 0.0):
            floors[product] = round(want, 3)
            fired += 1
    if not fired:
        return plan

    from dataclasses import replace

    _count(obs, "scarce_meter_days")
    return replace(plan, consts={**_consts(plan), "sell_floor": floors})
