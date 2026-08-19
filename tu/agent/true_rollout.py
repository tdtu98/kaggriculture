"""P2b — the compiler-in-the-loop rollout: an imagined day is a *compiled* day.

**Why this exists.** P2 (`agent/planner.py`) searched over plan patches and lost $5-6k a season, and
E83 named the mechanism precisely: the value model (`agent/season.py`) prices labour as a scalar
budget and assumes every plant is watered, so a `cohort_shift` — the one decision type that ever
fired — was scored on a quantity the model does not contain. Predicted +$1,657 against a realised
−$4,393, correlation **−0.15**, sign agreement **4/20**. A search does not merely tolerate a model's
blind spots, it *steers into them*.

**What changes here, and what deliberately does not.** The market half of P1 was measured at ~5%
mechanical accuracy (E82) and is reused **verbatim** — `season._town_day`, `_sell_day`,
`_buy_product`, `_price`, `shops_on`/`future_shops`, `opponent.forecast_supply`. What is replaced is
the *farm day*: instead of `season.step_day`'s scalar-labour arithmetic, an imagined day runs the
**real daily stack** —

    C1 `tasks.daily_tasks`  ->  C2 `router.route` / `decide_hands`  ->  C3 `verify.compile_day`

— against a `forward.FarmModel` of our own farm, and then *plays the compiled script out turn by
turn* on that model. Thirst deaths, blocked ops, missed tick-day waters, pruned plantings and the
re-phasing that E83's worked example is made of are therefore not modelled: they **happen**, in the
same code the real day happens in. That is the entire point. A cohort moved three days collides with
the router's schedule inside the rollout exactly as it does in play, because it is the same router.

**Cost.** One compiled day is 0.7 ms (day 6) to 2.7 ms (day 22); a 24-day rollout from day 6 is
~45-60 ms, against `season.rollout`'s 2 ms. The dawn budget (`planner_ms`) is therefore the binding
constraint and the search is anytime, exactly as it already was.

--------------------------------------------------------------------------------------------------
**The honest assumption list.** Everything the imagined day still does not contain:

1. **Weeds do not spawn.** `weedSpawnChance` is stochastic (`kaggriculture.py:823`) and
   `FarmModel(weed_mode="none")` skips it, which is also what `verify.verify_day` already does in
   real play — so C3's own view of a day has always been weed-free and the rollout is consistent
   with the compiler it is modelling, not more optimistic than it.
2. **The opponent's board is frozen** at the dawn the rollout starts from. Their *supply* still
   arrives on `opponent.forecast_supply`'s schedule (P1's, unchanged); what is frozen is the tile
   grid `projection._supply_schedule` reads when pricing our tasks. They also do not react to us.
3. **Shop draws are sampled, not known** — P1's dominant error term (E82), unchanged, and a common
   shock across candidates at one dawn (`planner.draws`).
4. **The crew size is an op-count proxy, not `decide_hands`.** `decide_hands` routes the day up to
   fourteen times (13-23 ms at day 15+, i.e. ten times a whole compiled day) and is unaffordable
   inside a rollout. `OPS_PER_HAND` is fitted against it rather than assumed: at 10 useful ops a
   hand the proxy is exact on **58%** of dawns and within one hand on **92%** (3 seasons vs
   `boatlee`, 90 dawns). The *routing* is real; only the headcount is estimated.
5. **Intra-day market slots are not modelled.** Sells land at dawn, on the turns a unit DROPs, and
   at hour 23 — the same three hooks the shell sells on (`main_v4._act`) — but the ten-slot queue is
   modelled only at dawn (`main_v4._dispatch`, reused), and `slot_align` is a pure permutation
   (E79) so it is a no-op here by construction.
6. **The day's town drain is applied in one lump at dawn**, before the dawn sell, which is
   `season.step_day`'s own ordering. Keeping it identical is deliberate: the market half is the half
   that was measured, and changing its phase would invalidate E82's fidelity numbers.

Nothing here raises: it is called from a turn that must not forfeit the episode (E21).

--------------------------------------------------------------------------------------------------
**MEASURED: Gate 1 fails, and the residual is NOT the market.**

E83's calibration protocol, re-run against this model: one deviation per season
(`planner_max_moves=1`), paired ON/OFF, 20 seasons on the fresh 76000:76010 block.

|                  | mean predicted | mean realised | sign agreement | corr |
|------------------|---|---|---|---|
| E83 (fast model) | +$1,657 | −$4,393 | 4/20 (20%) | **−0.15** |
| this, vs starter | +$4,836 | −$2,752 | 9/20 (45%) | **+0.312** |
| this, vs boatlee | +$3,173 | −$559 | 8/20 (40%) | **+0.042** |

The bar was corr >= +0.4 **and** sign agreement >= 60%. Compiling the day moves the correlation from
*negative* to weakly positive and roughly doubles the sign agreement, and it is still not enough:
the realised deltas stay bimodal (−$20k to −$28k, or nothing) and this model still prices those
catastrophes at **+$1.7k to +$6.7k**.

**[The decisive ablation] The residual is on the farm, not in the market.** Re-pricing the same 20
moves at the same dawns with the shop draw the season actually produced — E82's ablation, which took
P1's day-8 error from 13.5% to 5.4% — buys **nothing**: corr +0.297 (K=1) -> +0.291 (K=8) ->
**+0.317 (true draw)**, sign 9/20 -> 9/20 -> 10/20. So the price term is exonerated and what is left
is the farm half, *even with the real router in the loop*.

**[The mechanism] The rollout contains the router's plan for each day, but not its failure to
execute it.** Worked example, seed 76003 seat 1 vs `starter`, `cohort0_+4` (predicted **+$5,814**,
realised **−$25,521**):

| | realised (counters) | this rollout |
|---|---|---|
| STRAWBERRY units | 174 -> **117** (−33%) | 167 -> 147 (−12%) |
| strawberry/plant | 5.8 -> **3.9** | — |
| `fertilize_hits` | 53 -> **39** | fertilizer *up*, 104 -> 119 units sold |
| collision signal | `prune_days` 9 -> 11 | `compile_deaths` 1 -> 5, `overcommit_days` 1 -> 4 |

The rollout sees the collision — it is the same router, so it must — but at about a third of its
realised size, and the model's other products more than pay for it. The reason is structural rather
than a tuning error: **an imagined day is compiled by C3 and then executed exactly as compiled**, so
every day C3 certifies as sustainable *is* sustainable here, by construction. Real days diverge from
their scripts (`blocked_ops` ~39 a game in play against 0-2 across a whole imagined season), and a
re-phased cohort is precisely the case where that divergence compounds. Rolling out *through* C3
inherits C3's blind spots along with its arithmetic.

Two smaller farm-side gaps ride along and are named rather than fixed: the crew-size proxy responds
to a collided day by hiring (`hand_days` 133 -> 132, i.e. barely at all) where the real dawn loses
hands to the ten-slot queue (`hires_dropped` 9 -> 5); and weeds are absent, which is measured to be
*not* the mechanism — `weedSpawnChance` is 0.005 a tile a day (`kaggriculture.py:865`), about 3.6
weeds a season over ~30 empty tiles, far too small to be a $25k swing.

**[The worked example, re-checked] E83's named counter-example is now priced negative — and that is
not enough either.** Re-pricing the *fast* planner's own 20 moves on E83's own 74000 block: the
−$16,200 five-tile strawberry shift (74000, seat 1, day 6, `cohort1_+3`) goes from the fast model's
**+$2,446** to this model's **−$5,534**. Pinned as a regression in
`tests/test_true_rollout.py::test_e83_worked_example_is_priced_negative`. Over the whole 20-move set,
though, the same re-pricing comes back **corr −0.610, sign 11/20** — the true rollout has simply
traded E83's false positives for false negatives of its own (74009 seat 0: predicted **−$14,526**,
realised **+$16,268**). Fixing the named case is not the same as calibrating the model, which is why
the gate is a correlation and not an anecdote.

**Cost, for the record, since it was never the constraint.** One rollout: **95 ms** at day 6, 64 ms
at day 16, 25 ms at day 25. One dawn's search over a 12-candidate menu at one draw: **625-996 ms**
uncapped, **180-390 ms** with `planner_true_days=10` — against a 1 s `actTimeout`. Affordable; just
not right.

**Verdict: Gate 1 fails, no money was run, and the flag ships OFF (`planner_value` defaults to
`"fast"`, and `planner` itself defaults to 0).**
"""

from __future__ import annotations

import copy
import math

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, LAND_PRICES

from agent import projection, season
from agent import tasks as tasks_module
from agent.plan import LAND_ORDER, NEVER, quadrant_of
from agent.router import HIRE_COST, spawn_positions
from agent.verify import _afford, compile_day

TURNS_PER_DAY = 24
LAST_TURN = TURNS_PER_DAY - 1
LAST_DAY = season.LAST_DAY
MAX_ORDERS = 10

#: Useful ops one hand completes in a day, for the crew-size proxy — see assumption 4. Fitted
#: against `router.decide_hands` over 90 real dawns, not derived: 10 is the value that minimises
#: |proxy − decide_hands| (MAE 0.51 hands, bias +0.22, exact 58%, within one 92%). `season.py`'s
#: `OPS_PER_HAND = 13` is a *different* quantity — it converts an op count into a throughput
#: fraction, where this converts a task count into a headcount.
OPS_PER_HAND = 10.0

#: Task ops that are logistics rather than work, and so do not size the crew.
_LOGISTICS_OPS = frozenset({"PICKUP", "DROP"})

#: `main_v4.LATE_COHORT_DAYS` — how overdue a quadrant's cohorts must be before the dawn that buys
#: it re-derives its shopping list against the unlocked board (E73). Imported by value rather than
#: restated as a rule; see `_dawn`.
LATE_COHORT_DAYS = 6


# --------------------------------------------------------------------------- module-state sandbox

class _sandbox:
    """Isolate the per-season module globals an imagined day would otherwise write into.

    `daily_tasks` runs `projection.redirect` and `projection.counter_mix`, both of which *commit*
    their decisions to module-level per-seat dicts so they are idempotent for the season; and
    `projection._count` mirrors every counter into `main_v4._STATE`. A rollout that ran them
    unguarded would have the search's imagined day 14 decide a redirect that the real day 7 then
    considers already made, and would inflate the shell's own counters with work that never
    happened — E44's failure mode with the sign flipped: a change that fired only inside a dream,
    wearing the counters of one that fired in play.

    The live commitments are *copied in* (so an imagined day sees the redirects the season has
    really made) and thrown away on exit. Deep copies, because the values are nested dicts and a
    shallow copy would leak the mutation it exists to contain.
    """

    __slots__ = ("_saved",)

    _TARGETS = ((projection, "_REDIRECTS", True), (projection, "_CONTESTED", True),
                (projection, "_CACHE", False), (projection, "STATS", False),
                # `tasks.STATS` is carried rather than emptied because its consumers index it
                # directly (`STATS["tick_waters"] += 1`); an empty dict is a KeyError, not a clean
                # slate.
                (tasks_module, "STATS", True))

    def __enter__(self):
        from agent import main_v4

        self._saved = []
        for module, name, carry in self._TARGETS:
            live = getattr(module, name)
            self._saved.append((module, name, live))
            setattr(module, name, copy.deepcopy(live) if carry else type(live)())
        self._saved.append((main_v4, "_STATE", main_v4._STATE))
        # Pre-seeded, **not** empty, and this is a measured defect rather than tidiness.
        # `main_v4._state` treats a missing seat as the start of a season and calls
        # `branches.reset(seat)` and `planner.reset(seat)` on it. With an empty dict here the first
        # `projection._count` inside the first imagined day wiped the *real* planner's season state,
        # so `planner_max_moves=1` committed **11** deviations and `planner.log()` came back empty —
        # a change firing eleven times while its own trace said it never fired at all (E44's exact
        # shape). Only `_note` reads this inside a rollout, so `{"effects": {}}` is the whole
        # contract.
        main_v4._STATE = {0: {"effects": {}}, 1: {"effects": {}}}
        return self

    def __exit__(self, *exc):
        for module, name, live in self._saved:
            setattr(module, name, live)
        return False


# --------------------------------------------------------------------------- the world

class World:
    """Our farm as a `FarmModel`, plus P1's market, money and opponent — one imagined dawn.

    Thin on purpose: everything except the farm day is `season.SeasonState`, so the market
    arithmetic that E82 measured is the *same object*, not a second copy of it.
    """

    __slots__ = ("state", "opp_farm", "counters")

    def __init__(self, state: season.SeasonState, opp_farm, counters=None):
        self.state = state
        self.opp_farm = opp_farm
        self.counters = counters if counters is not None else {}

    @classmethod
    def from_obs(cls, obs, seat: int, plan=None, known: str | None = None) -> "World":
        state = season.SeasonState.from_obs(obs, seat=seat, plan=plan, known=known)
        farms = obs.get("farms") or []
        opp = farms[1 - seat] if len(farms) > 1 else None
        return cls(state, _freeze_farm(opp))

    def clone(self) -> "World":
        return World(self.state.clone(), self.opp_farm, dict(self.counters))

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(n)

    # -- the observation shim ------------------------------------------------

    def view(self, hour: int, tiles=None) -> dict:
        """A **delivered**-shaped observation of this imagined turn.

        The shape `__get_shared_state` hands an agent (CLAUDE.md's standing rule: verify against the
        surface the runner uses), because everything downstream — `daily_tasks`, `_afford`,
        `compile_day`, `_animal_orders`, `_paced_plan` — reads that surface and nothing else. Built
        fresh per call rather than cached: it aliases the live farm model's tiles, shed and
        inventories, so a stale copy would be a day compiled against yesterday's board.

        `tiles` overrides the seat's grid, which is the one thing dawn needs to lie about: a
        quadrant bought this turn is unlocked in the model immediately (`BUY_LAND` settles in the
        market turn) but the shell only re-derives dawn's shopping list against it when the cohorts
        there are badly overdue (`main_v4._dawn`'s `_unlocked_view`, E73).
        """
        state = self.state
        farm = state.farm
        season._refresh_price_table()
        me = {
            "tiles": farm.tiles if tiles is None else tiles,
            "farmer": list(farm.units[0]),
            "hands": [list(u) for u in farm.units[1:]],
            "money": state.money,
            "unlocked_quadrants": sorted(state.unlocked),
        }
        farms = [None, None]
        farms[state.seat] = me
        farms[1 - state.seat] = self.opp_farm
        return {
            "player": state.seat,
            "day": state.day,
            "hour": int(hour),
            "step": state.day * TURNS_PER_DAY + int(hour),
            "farms": farms,
            "private": {"shed": farm.shed, "seeds": farm.seeds, "inventories": farm.invs},
            "market": {
                "inventory": state.market_inv,
                "prices": {p: season._price(p, inv) for p, inv in state.market_inv.items()},
            },
            "town": {"unlocked_shops": state.shops_on(state.day)},
        }


def _freeze_farm(farm):
    """A snapshot of the opponent's board — assumption 2, made explicit rather than aliased.

    Aliasing the live observation would be worse than freezing it: the rollout would then read
    whatever the *real* board did while the search was still running, which is neither a forecast
    nor the present.
    """
    if farm is None:
        return {"tiles": [[None] * 10 for _ in range(10)], "farmer": [4, 4], "hands": [],
                "money": 0.0, "unlocked_quadrants": ["NW"]}
    return {
        "tiles": [[dict(t) if isinstance(t, dict) else t for t in row] for row in farm["tiles"]],
        "farmer": list(farm.get("farmer") or [4, 4]),
        "hands": [list(h) for h in (farm.get("hands") or [])],
        "money": float(farm.get("money", 0) or 0),
        "unlocked_quadrants": list(farm.get("unlocked_quadrants") or ["NW"]),
    }


# --------------------------------------------------------------------------- dawn

def _hands_for(tasks, cash: float) -> int:
    """The crew-size proxy — assumption 4, and the only piece of scalar labour left in here."""
    ops = sum(1 for t in tasks if t.op not in _LOGISTICS_OPS)
    want = max(0, min(13, int(math.ceil(ops / OPS_PER_HAND)) - 1))
    while want > 0 and sum(HIRE_COST[:want]) > cash:
        want -= 1
    return want


def _relocked(tiles, quad: str):
    """`tiles` with `quad` put back to LOCKED — dawn's pre-purchase view of the board."""
    out = [list(row) for row in tiles]
    for y, row in enumerate(out):
        for x, tile in enumerate(row):
            if tile is None and quadrant_of(x, y) == quad:
                row[x] = "LOCKED"
    return out


def _buy_land(world: World, plan) -> str | None:
    """`main_v4._dawn`'s land block: the next due quadrant in `LAND_ORDER`, if the wallet reaches."""
    state = world.state
    for i, quad in enumerate(LAND_ORDER):
        due = (plan.land_days or {}).get(quad, NEVER)
        if due >= NEVER or state.day < due or quad in state.unlocked:
            continue
        price = float(LAND_PRICES[i])
        if state.money < price:
            return None
        state.money -= price
        state.unlocked.add(quad)
        for y, row in enumerate(state.farm.tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED" and quadrant_of(x, y) == quad:
                    row[x] = None
        world.count("rollout_land_bought")
        return quad
    return None


def _settle(world: World, orders: list, plan) -> int:
    """Execute dawn's surviving market orders, in queue order. Returns hands hired.

    Money is spent here and nowhere else at dawn, so a candidate that cannot pay for its own plan
    finds out the same way the shell does — by the order simply not settling.
    """
    state = world.state
    farm = state.farm
    hired = 0
    for order in orders:
        op = order[0]
        if op == "HIRE":
            wage = HIRE_COST[min(hired, len(HIRE_COST) - 1)]
            if state.money < wage:
                continue
            state.money -= wage
            hired += 1
        elif op == "BUY_SEED":
            crop, n = order[1], int(order[2])
            cost = float(CROPS[crop]["seed"]) * n
            if state.money < cost:
                continue
            state.money -= cost
            farm.seeds[crop] = farm.seeds.get(crop, 0) + n
        elif op == "BUY_PRODUCT":
            season._buy_product(state, order[1], int(order[2]))
        elif op == "BUY_ANIMAL":
            species = order[1]
            cost = float(ANIMALS[species]["cost"])
            if state.money < cost or sum(farm.shed.values()) >= farm.shed_capacity:
                continue
            state.money -= cost
            farm.shed[species] = farm.shed.get(species, 0) + 1
    state.hands = hired
    state.count("hand_days", hired)
    return hired


def _sell(world: World, slots: int | None = None, hook: str = "dawn") -> float:
    """P1's sell, unchanged, at one of the shell's three selling hooks.

    `slots` is dawn's remaining share of the ten-order queue; elsewhere the turn's market list is
    only ever sells, so the cap cannot bind.

    `hook` is which of `main_v4._act`'s three selling turns this is, counted rather than assumed:
    a rollout that quietly stopped selling on drop turns or at hour 23 would still return a
    plausible number, and the counter is the only thing that says which hooks actually fired (E44).
    """
    world.count(f"rollout_sell_hook_{hook}")
    state = world.state
    n_animals, n_plants = 0, 0
    for row in state.farm.tiles:
        for tile in row:
            if type(tile) is not dict:
                continue
            if tile.get("animal"):
                n_animals += 1
            elif tile.get("kind") == "PLANT":
                n_plants += 1
    products = None
    if slots is not None:
        shed = state.farm.shed
        ranked = sorted((p for p in season.SELLABLE if shed.get(p, 0)),
                        key=lambda p: -shed[p] * season.BASE_PRICE.get(p, 50))
        if len(ranked) > max(0, slots):
            world.count("rollout_sells_deferred", len(ranked) - max(0, slots))
        products = tuple(ranked[:max(0, slots)])
    return season._sell_day(state, season.NO_DECISIONS, n_animals, n_plants, products=products)


# --------------------------------------------------------------------------- the day

def step_day(world: World, plan) -> None:
    """One imagined day: dawn's market turn, then the compiled day played out on the farm model.

    The order mirrors `main_v4._act` rather than `season.step_day`: hour 0 is the market turn, hour 1
    compiles, hours 1-23 replay, dusk closes. The one place P1's ordering is kept instead is the
    town drain, which lands in a lump before the dawn sell (assumption 6).
    """
    from agent import main_v4

    state = world.state
    day = state.day
    farm = state.farm
    farm.day = day
    farm.step_idx = day * TURNS_PER_DAY

    # -- the market's own day (P1, unchanged) --------------------------------
    season._town_day(state)

    # -- hour 0: the market turn ---------------------------------------------
    paced = main_v4._paced_plan(world.view(0), plan, state.seat, day, None)
    bought = _buy_land(world, paced)
    tiles = None
    if bought is not None and main_v4._quadrant_backlog(paced, bought, day) < LATE_COHORT_DAYS:
        # Not overdue: the shell's dawn still reads the quadrant as LOCKED even though the purchase
        # has settled, and sizes seed, hands and tasks against the smaller board (E73).
        tiles = _relocked(farm.tiles, bought)
    view = world.view(0, tiles=tiles)

    money = state.money
    tasks = list(tasks_module.daily_tasks(view, paced, day=day, turn=1))
    wanted = _hands_for(tasks, money * 0.45)

    orders: list = []
    if bought is not None:
        orders.append(["BUY_LAND"])
    budget = money
    for n in range(wanted):
        wage = HIRE_COST[min(n, len(HIRE_COST) - 1)]
        if budget < wage:
            break
        orders.append(["HIRE"])
        budget -= wage
    _kept, needs, _dropped = _afford(tasks, view, money)
    prices = view["market"]["prices"]
    for item, count in sorted(needs.items()):
        if item.startswith("SEED:"):
            crop = item.split(":", 1)[1]
            cost = float(CROPS[crop]["seed"]) * count
            if budget >= cost:
                orders.append(["BUY_SEED", crop, count])
                budget -= cost
        else:
            price = float(prices.get(item, season.BASE_PRICE.get(item, 50)) or 50)
            afford = int(budget // max(1.0, price))
            if afford > 0:
                orders.append(["BUY_PRODUCT", item, min(count, afford)])
                budget -= min(count, afford) * price
    orders += main_v4._animal_orders(view, paced, state.seat, day, budget, None)

    # `_dispatch` is the ten-slot queue, reused rather than restated (E39). The land order was
    # already settled above — it ranks first and so can never be the one truncated — and the sells
    # are sized against whatever slots the supply half leaves.
    kept = main_v4._dispatch(view, orders, {})
    hired = _settle(world, kept, paced)
    _sell(world, slots=MAX_ORDERS - len(kept), hook="dawn")

    # -- hour 1: compile against the board dawn actually left behind ---------
    farm.units = [tuple(p) for p in spawn_positions(hired, farm.units[0])]
    farm.invs = [{} for _ in farm.units]
    compiled = compile_day(world.view(1), paced, hands=hired, turns=TURNS_PER_DAY,
                           start_turn=1, cash=0.0)
    world.count("rollout_blocked_ops", len(compiled.report.blocked_ops))
    world.count("rollout_pruned", len(compiled.pruned))
    if compiled.overcommit:
        world.count("rollout_overcommit_days")

    # -- hours 0..23: play it -----------------------------------------------
    n_units = len(farm.units)
    farm.step([["PASS"]] * n_units)                       # hour 0: the lone farmer's market turn
    for turn in range(1, TURNS_PER_DAY):
        actions = []
        for unit in range(n_units):
            script = compiled.scripts.get(unit)
            ops = script.ops if script is not None else []
            index = turn - compiled.start_turn
            actions.append(list(ops[index]) if 0 <= index < len(ops) else ["PASS"])
        if turn == LAST_TURN:
            # The shell's hour-23 sell. Taken *before* the turn resolves rather than after, so the
            # atomic-PLANT rule in `FarmModel.step` is never re-derived here (E39); the difference is
            # only the goods a unit DROPs at hour 23, which this rollout sells at the next dawn.
            _sell(world, hook="dusk")
        farm.step(actions)
        if turn < LAST_TURN and any(a and a[0] == "DROP" for a in actions):
            _sell(world, hook="drop")                     # `main_v4._act`'s drop hook
    world.count("rollout_compile_deaths", len(compiled.report.deaths))

    state.day = day + 1
    farm.day = state.day


def _sync_cohort_state(state: season.SeasonState, plan) -> None:
    """Tell P1 which cohorts are already standing, before it takes over the tail.

    `season._plantings` re-sows a cohort's *whole* tile list the first time it sees it and a
    `wave_size` slice afterwards, and it decides which by looking at `cohort_state`. Handing it a
    board full of standing crops with an empty `cohort_state` would have the tail re-fill every
    cohort at full width on its first day — a free harvest the search would learn to buy. Same
    derivation as `SeasonState.from_obs`, from the board rather than from a remembered decision.
    """
    if plan is None:
        return
    tiles = state.farm.tiles
    for i, cohort in enumerate(plan.cohorts):
        for (x, y) in cohort.tiles:
            tile = tiles[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                state.cohort_state[i] = min(state.cohort_state.get(i, 99),
                                            int(tile.get("planted_day", state.day)))


def rollout(world: World, plan, days: int | None = None) -> float:
    """Play the rest of the season and return the terminal bank.

    `days` truncates the *compiled* part of the rollout: the first `days` days are played through
    the real daily stack and the tail is handed to P1's fast model. That is the budget knob, and it
    is a defensible split rather than a shortcut — E83's mechanism is a **re-phasing collision**,
    which shows up within a few days of the decision, while the tail is the steady state P1 was
    already measured at ~5% on (E82). `None` compiles the whole season.

    Never raises, for `season.rollout`'s reason: this is called from a turn that must not forfeit
    the episode. A world that blows up mid-season returns the money it had reached, which is a bad
    candidate rather than a dead agent.
    """
    state = world.state
    try:
        with _sandbox():
            stop = LAST_DAY if days is None else min(LAST_DAY, state.day + max(0, int(days)) - 1)
            while state.day <= stop:
                step_day(world, plan)
    except Exception:                                     # pragma: no cover - defensive
        world.count("rollout_errors")
        return state.money
    if state.day <= LAST_DAY:
        world.count("rollout_fast_tail_days", LAST_DAY - state.day + 1)
        _sync_cohort_state(state, plan)
        season.rollout(state, plan=plan)
    return state.money


def value(base: World, plan, sample: list, days: int | None = None) -> float:
    """Mean terminal bank over `sample` shop-draw continuations — `planner.value`'s contract.

    Same signature and same meaning as the fast path's, so `planner.search` chooses between them by
    binding a name rather than by branching at every call site.
    """
    total = 0.0
    for continuation in sample:
        world = base.clone()
        world.state.future_shops = continuation
        total += rollout(world, plan, days=days)
    return total / max(1, len(sample))
