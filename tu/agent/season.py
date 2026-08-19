"""P1 — the season model: one day at a time, from any dawn to the buzzer.

**What this is for.** The compiler line ceilings at $120–133k solo against boatlee's $150–159k, and
E77/E81 diagnosed the deficit as *staging* rather than market timing: the plan is decided once, up
front, by a search that never sees the board it will actually meet. The route out of that is a dawn
-time tree search over macro-decisions — start a cohort, buy the land, buy a cow, hire — evaluated by
rolling each candidate forward to day 29 and reading the bank. This module is the roll-forward. P2 is
the search on top of it; nothing here searches anything.

**Why day-granular.** A turn-level model of both farms is `kagsim`, which is not importable on the
submission host, and 719 turns x thousands of candidates is not a 1000 ms turn budget. A *day* is the
natural quantum of this game anyway: crops tick at dusk, animals produce at dusk, the shops eat on a
fixed six-times-a-day schedule, hands are re-hired every dawn, and the plan itself is written in days
(`agent/plan.py::Cohort.plant_day`). What a day-granular model gives up is intra-day ordering — which
market slot our sell lands in, whether a hand reaches the last tile before dusk — and both of those
are execution, which C2/C3 own and which this model prices with a labour budget instead.

**Nothing is re-derived.** Every mechanic below is either imported from the module that already owns
it or cited to
`.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`:

* plant and animal ageing, production, fertilizer and death: `agent.forward.FarmModel`'s
  `_daily_refresh_plants` / `_daily_refresh_animals`, which are the env's own `:756` / `:792` loops
  and are differentially tested against `kagsim` (`tests/test_forward_parity.py`). This module drives
  them with `watered_today` / `fed_today` / `cared_today` set — see the assumption list.
* town drain, expected drain of a not-yet-drawn shop, shop unlock schedule: `agent.projection`'s
  `drain_per_day`, `SHOP_DRAIN_PER_DAY`, `EXPECTED_DRAIN_PER_INSTANCE`, `instances_present`.
* the opponent's supply: `agent.opponent.forecast_supply` (tiles for an unknown opponent, the
  measured `SELL_SCHEDULE` x `SETTLE_RATE` for a fingerprinted boatlee, `CONDITIONAL_SELLS`
  excluded).
* price: the env's own `market_price`, called per unit, because a sale moves the quote one unit at a
  time (`kaggriculture.py:597,660`) and pricing a 34-unit strawberry dump at its first quote is a
  2x error on that dump.
* cohort opening, replant cadence, water windows, tick days: `agent.tasks`.

**The assumption list** (each one is a place this model is deliberately optimistic or coarse; the
fidelity harness is what says whether it matters):

1. **Watering is satisfied.** Every living plant is watered every day, and on its tick day, so the
   fertilizer bonus lands. C3 guarantees this in real play — the C5 gate measures thirst deaths at
   0.5–3.2 a season against 80+ plants and tick-day watering at 97.6% (E69/E81) — so the residual is
   small and it is *not* free: the waters are charged to the labour budget below, and a day that
   cannot pay for them loses production pro rata.
2. **Labour is a scalar budget, not a route.** A day's useful ops are counted (plant, water,
   fertilize, harvest, feed, care, collect, place) and compared against
   `hands x OPS_PER_HAND`; production scales by the shortfall. `OPS_PER_HAND` is derived from the
   measured `steps_per_useful`, not tuned by hand.
3. **Hiring is demand-driven.** `main_v4._dawn` decides hands by *routing* the day
   (`router.decide_hands`), which this model cannot afford, so hands are the crew the day's op count
   needs, capped at 13 and by what the wallet can pay. `plan.hands` is deliberately ignored, because
   the shipped shell ignores it too.
4. **We sell what we hold, every day, honouring the plan's floors.** `_sell_orders`' measured
   default is sell-on-sight (E11); the floors and `release_pressure` are modelled, the intra-day
   slot is not. Terminal liquidation happens on day 29.
5. **The opponent's future *plantings* are invisible** unless they are fingerprinted. An unknown
   opponent contributes only what is standing on their board today; a boatlee contributes its whole
   measured sell table. Not-yet-drawn shops contribute their expectation, per projection — and this
   is **the model's dominant error term**, not a footnote: six of the eight instances are still
   undrawn at day 8 and the thin markets are thin, so replaying the same rollouts with the true draw
   substituted takes the day-8 median error from **13.5% to 5.4%** (72000 block). `shops_on` /
   `future_shops` is the seam for averaging over draws instead.
6. **Our own execution losses are one measured scalar per product** (`YIELD_REALISED`), fitted the
   way `projection.SELL_THROUGH` was: on a spent seed block, against the settled money, never on the
   block the gate is read from.

Nothing here raises: `rollout` is called from a turn that must not forfeit the episode (E21).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    PRODUCTS,
    market_price,
)

from agent import opponent as opponent_module
from agent import projection
from agent.forward import FarmModel, _new_animal, _new_plant
from agent.plan import NEVER, quadrant_of
from agent.tasks import tick_days, water_window, wave_size

TURNS_PER_DAY = 24

#: The last day of the season. `episodeSteps = 720` and the framework plays `episodeSteps - 1` turns,
#: so the board runs 0..718 = day 29 hour 22 — day 29 is played, its dusk production is not.
LAST_DAY = 29

#: `shedCapacity`, default 100 (`kaggriculture.py` `_commit_unit`, `_drop_inventories_to_shed`).
SHED_CAPACITY = 100

#: `_hire_cost` is `mult * fib(n_already_today)` with `fib` indexed 1,1,2,3,5,...
#: (`kaggriculture.py:690-698`). The n-th hand of the day costs `HIRE_FIB[n-1]`; a crew of `h` costs
#: `sum(HIRE_FIB[:h])` every single dawn, because `_end_of_day` clears the roster (`:866-867`).
HIRE_FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)
MAX_HANDS = 13

#: Products the farm can put on the market. Seeds and livestock are not sellable.
SELLABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

BASE_PRICE = {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}

#: Useful ops one hand can complete in a day.
#:
#: Derived, not tuned. A day is 24 turns and hour 0 is the market turn, so a hand has 23. The
#: compiler's measured `steps_per_useful` is 0.71–0.78 (E81 guards), i.e. each useful op is preceded
#: by ~0.75 moves, so a hand completes `23 / 1.75` ~ 13 of them. This is the number that turns "the
#: plan wants 90 waters today" into "the day needs 7 hands", which is what makes a hire decision
#: cost something in a rollout.
OPS_PER_HAND = 13.0

#: `FEED` takes one WHEAT per animal per day and two missed dusks lose the animal
#: (`kaggriculture.py:504-511,792-800`). Modelled as a hard daily draw on the shed, bought in when
#: the shed is short — which is what `main_v4._feedable_animals` says the shell does.
FEED_PER_ANIMAL = 1

#: Fraction of a product's *modelled* production that reaches the market.
#:
#: The one fitted table in this module, and it exists for the reason `projection.SELL_THROUGH` does:
#: the model above computes what a perfectly-worked farm produces, and a real season loses some of it
#: to routing, to a shed that overflows at dusk, to a plant harvested a day late, and — for WHEAT —
#: to the farm eating almost all of it (E68: 26 units sold against ~220 produced).
#:
#: Fitted on the **72500 block** (5 seeds x {champion, compiler} x {starter, boatlee} x both seats,
#: 40 games) against **settled money** — a coordinate search over the five products with material
#: volume, objective = median absolute terminal-bank error pooled over the day-8/15/22 checkpoints.
#: Read afterwards on the untouched 72000 block, which is the only block the gate is quoted from.
#:
#: **Only STRAWBERRY binds**, at 0.90, and it is the one product whose per-tile arithmetic the model
#: can be checked against directly: four fertilized ticks is 8 units a tile, and the C5 gate measures
#: 6.77–7.9 (E69/E70), i.e. 0.85–0.99. Every other product searched to 1.0 or was indistinguishable
#: from it.
#:
#: **The unit-ratio fit was tried first and is wrong.** Setting each factor to
#: `units that moved / units predicted` (MELON 0.70, MILK 0.80, WOOL 0.70, FERTILIZER 0.71) *doubled*
#: the day-22 error and pushed the bank $24k low: the units and the money are not proportional,
#: because cutting supply lifts the quote the rest of it sells at, and because less revenue means
#: fewer animals and later land, which compounds. Units are the diagnostic; money is the objective.
#: Kept here as the reason this table is not simply read off the `sold_units` table.
#:
#: **Nothing is set above 1.0 on purpose.** This is a table of *execution losses*, so a product the
#: model under-predicts is a modelling gap, not a loss to be papered over with a multiplier: WHEAT
#: fits at 27x and TOMATO at 3.4x, and both are gaps — the model feeds every wheat unit to the herd
#: and never buys the surplus the shell buys, and the real compiler fills its day-21 tomato ground
#: from tiles this model has not freed. Between them they are ~$2.2k of a $75k bank; multiplying them
#: up would tell a P2 that *chose* more wheat or tomato that it earns a threefold return it does not.
YIELD_REALISED = {
    "WHEAT": 1.0, "CARROT": 1.0, "TOMATO": 1.0, "STRAWBERRY": 0.90, "MELON": 1.0,
    "EGG": 1.0, "MILK": 1.0, "WOOL": 1.0, "FERTILIZER": 1.0,
}

#: Global scale on modelled production, same fitting rules. Separate from `YIELD_REALISED` so a
#: uniform miss (labour, staging) is not smeared across nine per-product numbers.
PRODUCTION_SCALE = 1.0

#: From this day everything goes: stock at the buzzer is worth zero
#: (`main_v4.TERMINAL_STEP = 700`, which is day 29 hour 4).
TERMINAL_DAY = 29


# --------------------------------------------------------------------------- decisions

@dataclass(frozen=True)
class DayDecisions:
    """The macro-decision set for one day — P2's action space, and this module's only input knob.

    Every field is an *override* on the plan: the default instance is "follow the plan", so a
    decision-free rollout is `step_day(state, None)` repeated, and a search perturbs one day.
    """

    #: Extra cohorts to start today: `(crop, quadrant, n_tiles)`. Tiles are taken from the
    #: quadrant's free ground in the plan's own row order.
    plant: tuple = ()
    #: `True` forces today's land purchase, `False` suppresses it, `None` follows `plan.land_days`.
    buy_land: object = None
    #: Extra animals to buy today, on top of the plan's due herd.
    buy_animals: tuple = ()
    #: Hands on top of (or below) what the day's op count asks for.
    hire_delta: int = 0
    #: Per-product sell-floor overrides for today, in `plan.consts["sell_floor"]`'s units.
    sell_floor: dict = field(default_factory=dict)
    #: Plan cohort indices that must *not* be sown today.
    hold_cohorts: frozenset = frozenset()


NO_DECISIONS = DayDecisions()


# --------------------------------------------------------------------------- state

class SeasonState:
    """A day-granular snapshot of everything a rollout needs, and nothing it does not.

    Our farm is a `FarmModel` (`agent/forward.py`) because that is the env's own tile arithmetic
    already ported and parity-tested; the market, the money and the opponent are the fields this
    module adds, because `FarmModel` deliberately models none of them.
    """

    __slots__ = ("day", "money", "farm", "market_inv", "shops", "opp_supply", "plan",
                 "seat", "known", "unlocked", "hands", "counters", "sold", "revenue",
                 "cohort_state", "future_shops", "n_animals")

    def __init__(self, day, money, farm, market_inv, shops, opp_supply, plan, seat,
                 known=None, unlocked=None, hands=0):
        self.day = int(day)
        self.money = float(money)
        self.farm = farm
        self.market_inv = dict(market_inv)
        self.shops = list(shops)
        #: A sampled continuation of the shop draw, for scenario-averaged rollouts. See `shops_on`.
        self.future_shops: list = []
        self.opp_supply = opp_supply                 # {day: {product: units}}
        self.plan = plan
        self.seat = int(seat)
        self.known = known
        self.unlocked = set(unlocked or {"NW"})
        self.hands = int(hands)
        self.counters: dict = {}
        self.sold: dict = {}                         # product -> units that actually moved
        self.revenue: dict = {}                      # product -> proceeds of those units
        self.cohort_state: dict = {}                 # cohort index -> first day it was sown
        #: Head on the board, kept in step by `_scan`, so dawn can decline to walk a hundred tiles
        #: on the twenty days a season when the herd is already complete.
        self.n_animals = 0

    # -- construction -------------------------------------------------------

    @classmethod
    def from_obs(cls, obs, seat: int | None = None, plan=None, known: str | None = None):
        """Build a state from a **delivered** observation (`__get_shared_state`, not the replay).

        `known` is the opponent fingerprint verdict. In play `main_v4` already has it
        (`_STATE[seat]["opp_known"]`, and `opponent.locked(seat)` mirrors it for exactly this kind of
        cross-module read); passed as `None` the opponent is forecast from their standing tiles,
        which is all an unknown opponent ever offers.
        """
        seat = int(obs.get("player", 0) or 0) if seat is None else int(seat)
        farm_obs = obs["farms"][seat]
        priv = obs.get("private", {}) or {}
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        step = day * TURNS_PER_DAY + hour

        farm = FarmModel.from_obs(obs, player=seat, weed_mode="none", rehire="keep")
        # Hands are re-hired every dawn (`kaggriculture.py:866-867`), so the roster standing on the
        # board at hour 0 is empty and carries no information; the model hires its own below.
        farm.units = farm.units[:1]
        farm.invs = farm.invs[:1]

        if known is None:
            known = opponent_module.locked(seat)

        farms = obs.get("farms") or []
        opp_supply: dict = {}
        if len(farms) > 1 and not opponent_module.is_self(known):
            rows = opponent_module.forecast_supply(
                farms[1 - seat], step, known, horizon=LAST_DAY - day + 1)
            for product, entries in rows.items():
                for s, units in entries:
                    d = int(s) // TURNS_PER_DAY
                    if d < day or units <= 0:
                        continue
                    slot = opp_supply.setdefault(d, {})
                    slot[product] = slot.get(product, 0) + int(units)

        state = cls(
            day=day,
            money=float(farm_obs.get("money", 0) or 0),
            farm=farm,
            market_inv={p: int((obs.get("market", {}) or {}).get("inventory", {}).get(p, 10_000))
                        for p in PRODUCTS},
            shops=list((obs.get("town", {}) or {}).get("unlocked_shops") or []),
            opp_supply=opp_supply,
            plan=plan,
            seat=seat,
            known=known,
            unlocked=set(farm_obs.get("unlocked_quadrants") or ["NW"]),
            hands=len(farm_obs.get("hands") or []),
        )
        # Cohorts already in the ground are recorded so the rollout does not re-sow them: a tile that
        # is standing is not free, but a *replant* cohort has to know it is past its first fill.
        if plan is not None:
            for i, cohort in enumerate(plan.cohorts):
                for (x, y) in cohort.tiles:
                    tile = farm.tiles[y][x]
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        state.cohort_state[i] = min(state.cohort_state.get(i, 99),
                                                    int(tile.get("planted_day", day)))
        state.n_animals = len(state.animals())
        _ = priv
        return state

    def clone(self) -> "SeasonState":
        out = SeasonState.__new__(SeasonState)
        out.day = self.day
        out.money = self.money
        out.farm = self.farm.clone()
        out.market_inv = dict(self.market_inv)
        out.shops = self.shops
        out.future_shops = self.future_shops
        out.opp_supply = self.opp_supply
        out.plan = self.plan
        out.seat = self.seat
        out.known = self.known
        out.unlocked = set(self.unlocked)
        out.hands = self.hands
        out.counters = dict(self.counters)
        out.sold = dict(self.sold)
        out.revenue = dict(self.revenue)
        out.cohort_state = dict(self.cohort_state)
        out.n_animals = self.n_animals
        return out

    # -- reading ------------------------------------------------------------

    @property
    def shed(self) -> dict:
        return self.farm.shed

    def bank(self) -> float:
        return self.money

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(n)

    def shops_on(self, day: int) -> list:
        """The shop instances consuming on `day`, drawn instances first.

        With `future_shops` empty this is just the instances already on the board, and every
        not-yet-drawn one enters through `projection.drain_per_day`'s expectation term. That is the
        right default and it is also **the model's single largest error source**: the thin markets
        are thin (STRAWBERRY `T` 100, MILK 122, WOOL 105), six of the eight instances are still
        undrawn at day 8, and which ones arrive moves the price trajectory by more than anything the
        farm does. Measured on the 72000 block: replaying the same rollouts with the true draw
        substituted takes the day-8 median error from **13.5% to 5.4%**.

        So `future_shops` is the seam for a P2 that wants to average over draws rather than pretend
        it knows one: set it to a sampled continuation (uniform over `projection.SHOP_NAMES`, with
        replacement — `kaggriculture.py:874,891`), roll out, repeat, average. Names past the count
        the town can have on `day` are ignored, so one long list serves the whole season.
        """
        n = projection.instances_present(day) - len(self.shops)
        if n <= 0 or not self.future_shops:
            return self.shops
        return self.shops + list(self.future_shops[:n])

    def animals(self) -> list:
        return [t for row in self.farm.tiles for t in row
                if isinstance(t, dict) and t.get("animal")]

    def plants(self) -> list:
        return [t for row in self.farm.tiles for t in row
                if isinstance(t, dict) and t.get("kind") == "PLANT"]


# --------------------------------------------------------------------------- market arithmetic

#: `market_price` with the per-product amplitude hoisted out of the inner loop.
#:
#: A rollout makes ~3,200 price calls — one per unit sold, plus each floor's binary search — and the
#: reference implementation re-derives `amp = target * base / f(T)` on every one of them. That was
#: half the day-3 rollout. The arithmetic below is `kaggriculture.py:192-206` unchanged, including
#: the `int(round(...))` (banker's rounding, which the env's own quotes depend on) and the
#: `PRICE_FLOOR`; `tests/test_season.py` pins it equal to `market_price` across the whole range.
#:
#: Rebuilt whenever `MARKET_PARAMS` changes, checked once a day rather than once a unit: Kaggle can
#: ship `marketParams` overrides and the tests patch the table, and a cached curve that outlived its
#: parameters would misprice a whole season silently (CLAUDE.md: do not assume the defaults).
_PRICE_TABLE: dict = {}
_PRICE_SIGNATURE = None


def _refresh_price_table() -> None:
    global _PRICE_SIGNATURE
    signature = tuple((p["base"], p["I0"], p["T"], p["below_func"], p["below_target"],
                       p["above_func"], p["above_target"]) for p in MARKET_PARAMS.values())
    if signature == _PRICE_SIGNATURE:
        return
    from kaggle_environments.envs.kaggriculture.kaggriculture import _shape

    _PRICE_TABLE.clear()
    for item, p in MARKET_PARAMS.items():
        _PRICE_TABLE[item] = (
            float(p["base"]), int(p["I0"]), float(p["T"]),
            p["below_func"], p["below_target"] * p["base"] / _shape(p["below_func"], p["T"], p["T"]),
            p["above_func"], p["above_target"] * p["base"] / _shape(p["above_func"], p["T"], p["T"]),
        )
    _PRICE_SIGNATURE = signature


def _price(item: str, inventory: int) -> int:
    row = _PRICE_TABLE.get(item)
    if row is None:
        return market_price(item, inventory)
    base, I0, T, below, below_amp, above, above_amp = row
    if inventory < I0:
        func, amp, x = below, below_amp, I0 - inventory
        price = base + amp * _SHAPES[func](x, T)
    else:
        func, amp, x = above, above_amp, inventory - I0
        price = base - amp * _SHAPES[func](x, T)
    return 1 if price < 1 else int(round(price))


#: `kaggriculture.py:61-73`'s `_shape`, split per name so the dispatch is a dict lookup rather than a
#: chain of string comparisons. `hinge` keeps `HINGE_GAIN` by import, not by copy.
def _shapes() -> dict:
    from kaggle_environments.envs.kaggriculture.kaggriculture import HINGE_GAIN
    import math

    def hinge(x, T):
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2

    return {
        "linear": lambda x, T: x,
        "sq": lambda x, T: x * x,
        "sqrt": lambda x, T: x ** 0.5,
        "log": lambda x, T: math.log(1.0 + x),
        "log10": lambda x, T: math.log10(1.0 + x),
        "hinge": hinge,
    }


_SHAPES = _shapes()


def _quantity_above_floor(product: str, held: int, inventory: int, floor: float) -> int:
    """Largest quantity whose *last* unit still clears `floor x base`.

    The same rule and the same binary search as `main_v4._quantity_above_floor`, priced off the env's
    own curve. Duplicated in miniature rather than imported because that one reaches for `kagsim`
    when it can, and a rollout must behave identically on the submission host where it cannot.
    """
    if floor <= 0:
        return held
    limit = floor * BASE_PRICE.get(product, 50)
    lo, hi = 0, held
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _price(product, inventory + mid - 1) >= limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _sell_units(state: SeasonState, product: str, units: int) -> float:
    """Move `units` onto the market, one unit at a time, and return the proceeds.

    Per-unit because `_commit_unit` adds one to the inventory per unit and the *next* unit is
    re-quoted at the new inventory inside the same loop (`kaggriculture.py:597,660`). Batching at the
    first quote is not an approximation, it is a different game: a 34-unit strawberry dump takes its
    own quote from $120 to $55.

    `if price > 1` mirrors `:664` — a sale at the floor does not add supply.
    """
    if units <= 0:
        return 0.0
    inv = state.market_inv[product]
    shed = state.farm.shed
    have = int(shed.get(product, 0) or 0)
    units = min(units, have)
    proceeds = 0.0
    moved = 0
    for _ in range(units):
        price = _price(product, inv)
        proceeds += price
        if price > 1:
            inv += 1
        moved += 1
    state.market_inv[product] = inv
    if moved:
        shed[product] = have - moved
        if shed[product] <= 0:
            del shed[product]
        state.sold[product] = state.sold.get(product, 0) + moved
        state.revenue[product] = state.revenue.get(product, 0.0) + proceeds
    return proceeds


def _buy_product(state: SeasonState, product: str, units: int) -> int:
    """`BUY_PRODUCT`, per unit, quoted at the post-buy inventory (`kaggriculture.py:612`)."""
    bought = 0
    inv = state.market_inv[product]
    shed = state.farm.shed
    for _ in range(max(0, int(units))):
        if sum(shed.values()) >= SHED_CAPACITY:
            break
        price = _price(product, inv - 1)
        if state.money < price:
            break
        state.money -= price
        inv -= 1
        shed[product] = shed.get(product, 0) + 1
        bought += 1
    state.market_inv[product] = inv
    return bought


def _town_day(state: SeasonState) -> None:
    """A full day of town demand, then whatever the opponent is scheduled to land.

    The drain is `projection.drain_per_day`, i.e. the *same* arithmetic C6 prices tasks with: known
    shop instances at their exact multipliers, not-yet-drawn instances at the mean over the eight
    names, and the town centre's one unit of every non-fertilizer product.
    """
    day = state.day
    shops = state.shops_on(day)
    inv = state.market_inv
    for item in PRODUCTS:
        inv[item] -= projection.drain_per_day(item, day, shops)
    for product, units in (state.opp_supply.get(day) or {}).items():
        if product in inv:
            inv[product] += int(units)


# --------------------------------------------------------------------------- the day

#: `tasks.tick_days` per crop, resolved once. It is a pure function of `CROPS`, and it was 138k
#: calls per 200 rollouts before this.
TICK_DAYS = {crop: tick_days(crop) for crop in CROPS}
WATER_WINDOW = {crop: water_window(crop) for crop in CROPS}


def _fert_ages(plan) -> dict:
    """`{crop: (age, ...)}` — the plan's fertilizer calendar, resolved once per day rather than once
    per tile (`plan.consts["fert_ages"]`, which `agent/plan.py` fixes as arithmetic, not a gene)."""
    return ((plan.consts or {}).get("fert_ages") or {}) if plan is not None else {}


def _free_tiles(state: SeasonState, positions) -> list:
    tiles = state.farm.tiles
    return [(x, y) for (x, y) in positions
            if quadrant_of(x, y) in state.unlocked and tiles[y][x] is None]


def _dawn_land(state: SeasonState, decisions: DayDecisions) -> None:
    """`BUY_LAND`, one quadrant at a time in `LAND_ORDER`, at `LAND_PRICES` (`:712-723`)."""
    plan = state.plan
    forced = decisions.buy_land
    if forced is False:
        return
    for i, quad in enumerate(LAND_ORDER):
        if quad in state.unlocked:
            continue
        due = (plan.land_days or {}).get(quad, NEVER) if plan is not None else NEVER
        if forced is not True and (due >= NEVER or state.day < due):
            return
        price = LAND_PRICES[i]
        if state.money < price:
            return
        state.money -= price
        state.unlocked.add(quad)
        for y, row in enumerate(state.farm.tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED" and quadrant_of(x, y) == quad:
                    row[x] = None
        state.count("land_bought")
        return


def _dawn_animals(state: SeasonState, decisions: DayDecisions) -> None:
    """The plan's due herd plus any decision extras, placed on the plan's pasture cluster.

    Buying and placing are one act here: the shell buys at dawn and a unit carries the animal out the
    next day (`main_v4._structure_lead`), which at day granularity is a one-day offset the animal's
    own `first_yield_day` dwarfs. The structure is built implicitly — it is one op, charged to the
    labour budget, and a pasture with no animal in it is worth nothing to a rollout.
    """
    plan = state.plan
    if plan is None:
        return
    due_total = sum(1 for _s, d in plan.herd if d <= state.day) + len(decisions.buy_animals)
    if state.n_animals >= due_total:
        return
    placed: dict = {}
    for tile in state.animals():
        placed[tile["animal"]] = placed.get(tile["animal"], 0) + 1
    want: dict = {}
    for species, buy_day in plan.herd:
        if buy_day <= state.day:
            want[species] = want.get(species, 0) + 1
    for species in decisions.buy_animals:
        want[species] = want.get(species, 0) + 1

    # A pasture slot is free whether or not its structure is already standing. Testing only for
    # `None` was a real defect and not a subtle one: `_structure_tasks` builds the PASTURE a day
    # before the animal arrives, so on every board this model is ever handed the empty slots read
    # `{"kind": "PASTURE"}`, no slot was ever free, and the rollout placed almost no animals —
    # measured, WOOL came out at 0.37x actual before this line was fixed.
    free = []
    for t in plan.pasture_tiles:
        if quadrant_of(*t) not in state.unlocked:
            continue
        tile = state.farm.tiles[t[1]][t[0]]
        if tile is None or (isinstance(tile, dict) and "animal" not in tile
                            and tile.get("kind") in ("PASTURE", "COOP")):
            free.append(t)
    for species in sorted(want, key=lambda s: -_yield_per_dollar(s)):
        need = want[species] - placed.get(species, 0)
        while need > 0 and free:
            cost = ANIMALS[species]["cost"]
            if state.money < cost:
                return
            state.money -= cost
            x, y = free.pop(0)
            state.farm.tiles[y][x] = _new_animal(species, state.day)
            state.count("animals_placed")
            state.n_animals += 1
            need -= 1


def _yield_per_dollar(species: str) -> float:
    """`main_v4._yield_per_dollar`'s ranking, restated so the buy order matches the shell's."""
    spec = ANIMALS[species]
    interval = max(1, int(spec["interval"]))
    return ((1 + interval) / interval * BASE_PRICE.get(spec["product"], 50)
            / max(1.0, float(spec["cost"])))


def _plantings(state: SeasonState, decisions: DayDecisions) -> list:
    """`(crop, x, y)` for everything that goes into the ground today.

    The horizon test and the replant cadence are `agent.tasks`': a cohort is admitted from its
    opening day onward for as long as it can still reach a first yield (`_planting_tasks`), and a
    cycling cohort refills at `wave_size` a day.
    """
    plan = state.plan
    day = state.day
    out: list = []
    if plan is None:
        return out
    for i, cohort in enumerate(plan.cohorts):
        if i in decisions.hold_cohorts:
            continue
        opens = cohort.plant_day if cohort.quadrant == "NW" else max(
            int(cohort.plant_day), int((plan.land_days or {}).get(cohort.quadrant, NEVER)))
        if day < opens:
            continue
        if day + int(CROPS[cohort.crop]["first_yield_day"]) > LAST_DAY:
            continue
        free = _free_tiles(state, cohort.tiles)
        if not free:
            continue
        started = state.cohort_state.get(i)
        limit = (len(cohort.tiles) if started is None
                 else wave_size(cohort, day, opens))
        for (x, y) in free[:limit]:
            out.append((cohort.crop, x, y))
        state.cohort_state.setdefault(i, day)

    for (crop, quad, n) in decisions.plant:
        if day + int(CROPS[crop]["first_yield_day"]) > LAST_DAY or quad not in state.unlocked:
            continue
        from agent.plan import quadrant_tiles

        free = _free_tiles(state, quadrant_tiles(quad))
        for (x, y) in free[:int(n)]:
            out.append((crop, x, y))
    return out


def _sow(state: SeasonState, plantings: list) -> int:
    """Buy the seed and put it in the ground; returns tiles actually sown.

    Seed is bought at its fixed price (`BUY_SEED` quotes `CROPS[item]["seed"]`, `:602`) and a
    planting the wallet cannot cover simply does not happen — which is the shape of the real
    failure, since `_afford` sizes dawn's seed order against the same money.
    """
    sown = 0
    for (crop, x, y) in plantings:
        cost = int(CROPS[crop]["seed"])
        if state.money < cost:
            break
        if state.farm.tiles[y][x] is not None:
            continue
        state.money -= cost
        tile = _new_plant(crop, state.day, TURNS_PER_DAY)
        # `_new_plant` starts `consecutive_unwatered` at 1 — the planting day counts as unwatered
        # (`kaggriculture.py:202`) — so a seedling that is not watered *on the day it goes in* hits
        # two dry days at its first dusk and is a weed by morning. C3 routes the water with the
        # plant for exactly this reason; assumption 1 says it lands, and the op is charged below.
        tile["watered_today"] = True
        state.farm.tiles[y][x] = tile
        sown += 1
    state.count("planted", sown)
    return sown


def _scan(state: SeasonState) -> tuple:
    """One walk of the board per day: `(plants, animals)`, as `(row, x, tile)` triples.

    Four separate walks a day showed up as 20% of the rollout in the profile, which for a model whose
    whole purpose is to be called thousands of times at a dawn is not a micro-optimisation.
    """
    plants, animals = [], []
    for row in state.farm.tiles:
        for x, tile in enumerate(row):
            if type(tile) is not dict:
                continue
            if tile.get("kind") == "PLANT":
                plants.append((row, x, tile))
            elif tile.get("animal"):
                animals.append(tile)
    state.n_animals = len(animals)
    return plants, animals


def _work_day(state: SeasonState, throughput: float, plants, animals, fert) -> dict:
    """Water, fertilize, harvest and tend, and return `{product: units}` reaching the shed today.

    Order mirrors the env's day: the WATER op accrues one-shot yield during the day (`:425`), the
    HARVEST op takes what is standing, and only then does dusk's `_daily_refresh_*` add tonight's
    production (`:756`, `:792`). Getting that order wrong shifts every ongoing crop's revenue a day,
    which at day granularity is the difference between a strawberry cohort paying four times and
    three.
    """
    day = state.day
    harvest: dict = {}
    fert_have = int(state.farm.shed.get("FERTILIZER", 0) or 0)

    for row, x, tile in plants:
        crop = tile["crop"]
        cd = CROPS[crop]
        age = day - tile["planted_day"]
        if age in fert.get(crop, ()) and fert_have > 0:
            fert_have -= 1
            tile["fertilized_until_day"] = day + 2              # `:468` — day, day+1, day+2
        tile["watered_today"] = True
        if not cd["ongoing"]:
            lo, hi = WATER_WINDOW[crop]
            if lo <= age <= hi:
                bonus = 2 if tile["fertilized_until_day"] >= day else 1
                tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + bonus)
        # A one-time crop is *legal* to harvest from `first_yield_day` and is only *worth* harvesting
        # at `max_yield_day`, because every watered day in between adds a unit to the same plant
        # (`kaggriculture.py:425`) and pulling it early throws the rest away. `tasks._cohort_value`
        # prices the cohort on exactly that basis, so harvesting on sight here would have the model
        # take 3 units off a wheat tile that the compiler takes 6 from — measured, that was the whole
        # wheat and melon crop at a third of its size.
        ripe = age >= (cd["first_yield_day"] if cd["ongoing"] else cd["max_yield_day"])
        if tile["yield_units"] > 0 and ripe:
            units = tile["yield_units"]
            tile["yield_units"] = 0
            harvest[crop] = harvest.get(crop, 0) + units
            if not cd["ongoing"]:
                row[x] = None
        elif not cd["ongoing"] and age > cd["max_yield_day"]:
            row[x] = None                                       # decayed away (`:739`)
        # An **ongoing** crop is stamped with a lifespan the dusk it makes its last production
        # (`forward._daily_refresh_plants`, `kaggriculture.py:786-800`); from that step `_decay_plants`
        # sheds a unit every other step and the tile is a WEED as soon as it is empty (`:739`). Left
        # standing it blocks its own cohort's replant forever — measured, that alone was strawberry
        # 165 units against an actual 210 on the compiler-vs-starter arm, because the season's
        # biggest cohort re-sowed nothing after its first round.
        elif cd["ongoing"] and tile["max_lifespan_step"] >= 0 \
                and day * TURNS_PER_DAY >= tile["max_lifespan_step"]:
            row[x] = None

    for tile in animals:
        product = ANIMALS[tile["animal"]]["product"]
        if tile["yield_units"] > 0:
            harvest[product] = harvest.get(product, 0) + tile["yield_units"]
            tile["yield_units"] = 0
        if tile.get("fertilizer_available"):
            tile["fertilizer_available"] = False
            harvest["FERTILIZER"] = harvest.get("FERTILIZER", 0) + 1
        tile["fed_today"] = True
        tile["cared_today"] = True

    state.farm.shed["FERTILIZER"] = fert_have
    if fert_have <= 0:
        state.farm.shed.pop("FERTILIZER", None)

    scale = PRODUCTION_SCALE * max(0.0, min(1.0, throughput))
    into_shed: dict = {}
    for product, units in harvest.items():
        # `round`, not truncate: `int()` biases every product down by up to a unit a day,
        # which on a nine-product season is a systematic shortfall rather than a rounding.
        got = int(round(units * scale * YIELD_REALISED.get(product, 1.0)))
        if got > 0:
            into_shed[product] = got
    shed = state.farm.shed
    for item, n in into_shed.items():
        shed[item] = shed.get(item, 0) + n
    return into_shed


def _clamp_shed(state: SeasonState) -> None:
    """The 100-unit shed, applied to what is **still held at dusk**.

    Deliberately not applied to the day's flow. Goods are DROPped and sold on the same turn
    (`main_v4._act`'s drop hook, and `_carried` exists precisely because the shed in `obs` predates
    the drop), so a day that produces 130 units and sells them does not discard 30 — it discards only
    what a floor or a full market left standing when the last inventory drop lands (`:830-857`).
    Clamping the flow instead cost 136 spurious discarded units a game in the first version.
    """
    shed = state.farm.shed
    total = sum(shed.values())
    if total <= SHED_CAPACITY:
        return
    excess = total - SHED_CAPACITY
    state.count("shed_discarded", excess)
    # Cheapest first: the dusk drop order is not observable at this granularity and discarding the
    # melon before the wheat would be an assumption dressed as arithmetic.
    for item in sorted(shed, key=lambda p: BASE_PRICE.get(p, 50)):
        if excess <= 0:
            break
        take = min(shed[item], excess)
        shed[item] -= take
        excess -= take
        if shed[item] <= 0:
            del shed[item]


def _ops_needed(state: SeasonState, plantings: int, plants, animals, fert) -> float:
    """Useful ops the day asks for. The denominator of the hiring decision and of `throughput`."""
    # A planting costs two: the PLANT and the WATER that has to go with it (`_sow`).
    ops = 2.0 * plantings
    day = state.day
    for _row, _x, tile in plants:
        crop = tile["crop"]
        cd = CROPS[crop]
        age = day - tile["planted_day"]
        ops += 1.0                                       # water
        if age in fert.get(crop, ()):
            ops += 1.0
        if cd["ongoing"]:
            if age in TICK_DAYS[crop]:
                ops += 1.0                               # harvest the tick, a day later
        elif age >= cd["max_yield_day"]:
            ops += 1.0
    # feed + care + collect fertilizer, every day; the product harvest on production days is folded
    # into the same visit and is not charged twice.
    ops += 3.0 * len(animals)
    return ops


def _feed(state: SeasonState, n_animals: int) -> None:
    """One WHEAT per animal per day, out of the shed, bought in when the shed is short."""
    need = FEED_PER_ANIMAL * n_animals
    if need <= 0:
        return
    shed = state.farm.shed
    have = int(shed.get("WHEAT", 0) or 0)
    if have < need:
        _buy_product(state, "WHEAT", need - have)
        have = int(shed.get("WHEAT", 0) or 0)
    take = min(have, need)
    if take:
        shed["WHEAT"] = have - take
        if shed["WHEAT"] <= 0:
            del shed["WHEAT"]
    if take < need:
        state.count("feed_short", need - take)


def _sell_day(state: SeasonState, decisions: DayDecisions, n_animals: int, n_plants: int,
              products=None) -> float:
    """Sell the shed, honouring the plan's floors, releasing them under shed pressure.

    The precedence is `main_v4._sell_orders`': terminal liquidation beats everything, then the floor
    sizes the order, then `release_pressure` frees what the floor withheld once the shed is fuller
    than the plan is willing to carry.

    `products` restricts the sale to a subset, which is how P2b's dawn models
    `main_v4._dispatch`'s ten-slot queue: a day whose supply orders fill the queue defers the sells
    that did not fit rather than getting them for free. `None` — the P1 default — sells everything
    sellable and is the behaviour every existing caller has.
    """
    _refresh_price_table()
    plan = state.plan
    consts = (plan.consts or {}) if plan is not None else {}
    floors = dict(consts.get("sell_floor") or {})
    floors.update(decisions.sell_floor or {})
    reserve = _feed_reserve(n_animals, n_plants)
    terminal = state.day >= TERMINAL_DAY

    proceeds = 0.0
    withheld: list = []
    for product in (SELLABLE if products is None else products):
        held = int(state.farm.shed.get(product, 0) or 0) - int(reserve.get(product, 0) or 0)
        if held <= 0:
            continue
        if terminal:
            proceeds += _sell_units(state, product, held)
            continue
        floor = float(floors.get(product, 0) or 0)
        allowed = held if floor <= 0 else _quantity_above_floor(
            product, held, state.market_inv[product], floor)
        if allowed > 0:
            proceeds += _sell_units(state, product, allowed)
        if allowed < held:
            withheld.append((product, held - allowed))

    if not terminal and withheld:
        pressure = int(consts.get("release_pressure", 0) or 0)
        excess = sum(int(n or 0) for n in state.farm.shed.values()) - pressure
        if pressure > 0 and excess > 0:
            withheld.sort(key=lambda pn: -BASE_PRICE.get(pn[0], 50))
            for product, spare in withheld:
                if excess <= 0:
                    break
                take = min(int(spare), excess)
                proceeds += _sell_units(state, product, take)
                excess -= take
                state.count("release_valve_units", take)
    state.money += proceeds
    return proceeds


def _feed_reserve(n_animals: int, n_plants: int) -> dict:
    """`main_v4._feed_reserve`: two days of wheat per animal, and fertilizer for the standing crop."""
    return {"WHEAT": n_animals * 2, "FERTILIZER": min(6, n_plants)}


def _hire(state: SeasonState, ops: float, decisions: DayDecisions) -> float:
    """Hire the crew the day's work needs, and pay for it. Returns the capacity actually bought.

    `main_v4._dawn` decides this by routing the day (`router.decide_hands`), which costs more than a
    whole rollout; the crew that covers the op count is the same answer to within a hand, and it is
    the one that makes a `hire_delta` decision cost money in a search.
    """
    want = int(-(-ops // OPS_PER_HAND)) + int(decisions.hire_delta)
    want = max(0, min(MAX_HANDS, want))
    paid = 0
    for n in range(want):
        wage = HIRE_FIB[min(n, len(HIRE_FIB) - 1)]
        if state.money < wage:
            break
        state.money -= wage
        paid += 1
    state.hands = paid
    state.count("hand_days", paid)
    # The farmer works too (`:900-933` gives it the same op budget as a hand), minus the market turn.
    return (paid + 1) * OPS_PER_HAND


# --------------------------------------------------------------------------- the step

def step_day(state: SeasonState, decisions: DayDecisions | None = None) -> SeasonState:
    """Advance ONE DAY, deterministically, in the env's own order.

    `decisions=None` means "follow the plan", so a rollout with no overrides is a pure forecast of
    what the shipped shell would do. The state is mutated **and** returned, so a caller that wants a
    branch calls `clone()` first — a rollout that deep-copied every day would cost more than the
    arithmetic it protects.

    The order is dawn (land, animals, seed, hire), work (water, fertilize, harvest, tend), market
    (town drain, opponent supply, our sells), dusk (production, ageing, death). Dusk last is not a
    style choice: goods produced at dusk of day `d` are in the shed on day `d + 1`, and moving that
    boundary is a whole tick of every ongoing crop.
    """
    d = NO_DECISIONS if decisions is None else decisions

    _dawn_land(state, d)
    _dawn_animals(state, d)

    fert = _fert_ages(state.plan)
    plants, animals = _scan(state)
    plantings = _plantings(state, d)
    ops = _ops_needed(state, len(plantings), plants, animals, fert)
    capacity = _hire(state, ops, d)
    throughput = 1.0 if ops <= 0 else min(1.0, capacity / ops)
    if throughput < 1.0:
        state.count("labour_short_days")
        plantings = plantings[:max(0, int(len(plantings) * throughput))]

    _sow(state, plantings)
    _work_day(state, throughput, plants, animals, fert)
    _feed(state, len(animals))

    _town_day(state)
    _sell_day(state, d, len(animals), len(plants))
    _clamp_shed(state)

    # Dusk. `FarmModel`'s refreshes are the env's `:756` / `:792` loops verbatim.
    state.farm.day = state.day
    state.farm._daily_refresh_plants()
    state.farm._daily_refresh_animals()
    state.farm.day = state.day + 1
    state.day += 1
    return state


def rollout(state: SeasonState, plan=None, overrides: dict | None = None) -> float:
    """Step to the end of the season and return the terminal bank.

    `overrides` is `{day: DayDecisions}` — P2's hook, and the only reason `step_day` takes a
    decision argument at all. `plan` replaces the state's plan for the whole rollout, which is how a
    candidate *plan* (rather than a candidate day) is evaluated.

    Never raises: a rollout is called from a turn that must not forfeit the episode (E21). A model
    that blows up mid-season returns the money it had reached, which is a bad candidate rather than a
    dead agent.
    """
    if plan is not None:
        state.plan = plan
    overrides = overrides or {}
    try:
        while state.day <= LAST_DAY:
            step_day(state, overrides.get(state.day))
    except Exception:                                   # pragma: no cover - defensive, see docstring
        state.count("rollout_errors")
    return state.money


def rollout_from_obs(obs, seat: int | None = None, plan=None, known: str | None = None,
                     overrides: dict | None = None) -> float:
    """`from_obs` + `rollout`, which is what a caller almost always wants."""
    return rollout(SeasonState.from_obs(obs, seat=seat, plan=plan, known=known),
                   overrides=overrides)
