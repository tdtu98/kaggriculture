"""C4 — the agent shell: the compiler, wired to a game.

The day has a fixed shape, forced by when information exists:

* **hour 0** — the hands from yesterday are gone and today's have not been hired. So this turn is
  the market turn: hire, buy land, buy seed, buy the animals the plan is due, restock wheat and
  fertilizer, and sell what is in the shed. The farmer does one op.
* **hour 1** — the hands are standing on the centre tiles and their positions are known, so the day
  is compiled here (C1 -> C2 -> C3) and cached.
* **hours 2-23** — replay the cached script. No thinking, no drift.

Re-compiling every dawn from the real board is the whole adaptivity story: a weed, a missed water or
an opponent flooding our market changes tomorrow's plan without any repair logic, because tomorrow's
plan is computed from tomorrow's board.

**Nothing here may raise.** A single exception is an ERROR for the entire episode, which scores the
$3,000 starting bank — measured, 0-40 (E21). Every turn goes through a bare `except` that falls back
to PASS, and the fallbacks are counted so a silent failure cannot masquerade as a bad strategy.
"""

from __future__ import annotations

from agent import branches as branch_points
from agent import planner as season_planner
from agent import opponent
from agent.plan import LAND_ORDER, NEVER, Plan
from agent.tasks import SHED_TILES, current_prices, daily_tasks, market_needs
from agent.router import HIRE_COST, decide_hands
from agent.verify import MAX_PRUNE_ROUNDS, compile_day

TURNS_PER_DAY = 24
LAST_TURN = TURNS_PER_DAY - 1

#: `maxMarketOrdersPerTurn`, default 10 — the queue is *truncated* past this, silently
#: (`_process_market`), so an over-long order list loses its tail rather than erroring.
MAX_ORDERS = 10

#: Products we can sell. Seeds and livestock are not sellable.
SELLABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
            "FERTILIZER")

#: Base prices, for the sell floors (`price >= floor x base`).
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
              "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}

#: From this step, liquidate: unsold stock is worth nothing at the buzzer.
TERMINAL_STEP = 700

# --------------------------------------------------------------------------- O3 overlays
#
# Three overlays on the market and the plan, each behind its own `plan.consts` flag, each with its
# own effect counter, all three **off by default** so the incumbent genomes are byte-identical
# without them (E43: measured alone first, together in O4).
#
# What the env actually does, because both overlays are priced off it
# (`kaggle_environments/envs/kaggriculture/kaggriculture.py`, verified line by line, and
# `kagsim/src/market.rs:291-361`, which is the same loop):
#
# * **A sale moves the price, one unit at a time.** `_commit_unit` adds **1** to
#   `market["inventory"][item]` per unit sold (`:660`) and the *next* unit is re-quoted at the new
#   inventory inside the same while-loop (`:597`). `market_price` falls as inventory rises above
#   `I0` with the product's `above_func` (`:192-206`), and the thin markets fall off a cliff:
#   STRAWBERRY has `T = 100` and `above_target = 1.60`, so boatlee's 34-unit dump at step 528 takes
#   a $120 quote to **$55** (measured, `/private/tmp/o3/probe_hold.py`). MILK (`T = 122`) goes
#   $160 -> $116 on its 21-unit dump at 406. Selling *before* that is the entire front-run.
# * **Orders are processed slot-by-slot, in lockstep across the two players.** The outer loop of
#   `_process_market` runs over the **order index** `i` (`:563`), drains slot `i` for *both* players
#   to exhaustion in an inner loop, and only then calls `_refresh_prices` (`:628`) and moves to slot
#   `i+1`. Within one slot both players are quoted at the same pre-commit inventory (`:612-618`), so
#   seat order buys nothing there — but a sell in slot 0 is fully executed against a virgin
#   inventory before a slot-2 sell of the same product is quoted at all. Slot index, not seat, is
#   the priority. That is what `slot_align` exploits.

#: `frontrun_lead` is a gene with range 0-24; a lead of 0-5 cannot straddle a dump usefully at 24
#: turns a day and 24 is a whole day of shed. Clamped rather than re-ranged so old genomes decode.
FRONTRUN_LEAD_MIN = 6
FRONTRUN_LEAD_MAX = 24

#: Dumps smaller than this are not worth emptying a shed for. Boatlee's milk table has 3-unit
#: dribbles next to 21-unit drops and its wheat table has 1-unit ones; a 3-unit milk sale moves the
#: quote by under 4% while the shed we dumped to get ahead of it was worth the full spread.
FRONTRUN_MIN_UNITS = 8

#: How far a *forecast* dump (unknown opponent, supply read off their tiles) is smeared, in turns.
#: One day either side — the spec's "±1 day".
FRONTRUN_FORECAST_SPREAD = TURNS_PER_DAY


def _flag(plan: Plan, name: str, default=0):
    """A `consts` value, defaulting only when the key is **absent**.

    Not `get(name) or default`: `frontrun_lead=0` is a legitimate setting that the `or` idiom
    silently turned into 10, so the clamp below was never reached and a genome asking for no lead
    got the default one instead.
    """
    value = (plan.consts or {}).get(name)
    return default if value is None else value

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

_STATE: dict = {}


def _seat(obs) -> int:
    return 1 if int(obs.get("player", 0) or 0) == 1 else 0


def _state(obs, seat: int) -> dict:
    st = _STATE.get(seat)
    if st is None or int(obs.get("step", 0) or 0) == 0:
        st = {"compiled": None, "compiled_day": -1, "hands_wanted": 0,
              "fallbacks": 0, "blocked": 0, "bought_land": set(), "effects": {}}
        _STATE[seat] = st
        # O1's patched plan is season state living in its own module, so it has to be cleared on the
        # same edge this dict is: a worker process plays many games and the second one must not
        # inherit the first one's branches.
        branch_points.reset(seat)
        # P2's working plan is season state in its own module for the same reason O1's is, and it
        # has to be cleared on the same edge: a worker plays many games and the second one must not
        # start on the first one's deviations.
        season_planner.reset(seat)
    return st


def _note(st: dict, key: str, n: int = 1) -> None:
    """Record that something fired, as a *season* total.

    `counters()` used to read the last compiled day, so a season that pruned forty tasks on day 3
    and none on day 29 reported zero — a change that fired reading exactly like one that never did,
    which is the failure CLAUDE.md's "prove the change fired" rule exists to stop (E44). These are
    cumulative and are handed to the harness through `make_agent`'s `ctx`.
    """
    effects = st.setdefault("effects", {})
    effects[key] = effects.get(key, 0) + int(n)


class Ctx:
    """The handle `harness/counters.py:_effect_counts` looks for on an agent.

    It carries nothing of its own: `effects` is rebound each turn to the live per-seat dict in
    `_STATE`, so the harness reads the same totals the agent accumulated rather than a copy that
    can drift from them.
    """

    __slots__ = ("effects",)

    def __init__(self) -> None:
        self.effects: dict = {}


def agent(obs, plan: Plan | None = None):
    """Entry point. Never raises: an exception here forfeits the whole episode."""
    seat = _seat(obs)
    try:
        return _act(obs, seat, plan or _default_plan())
    except Exception:
        st = _STATE.get(seat) or {}
        st["fallbacks"] = st.get("fallbacks", 0) + 1
        _note(st, "fallbacks")
        n_hands = 0
        try:
            n_hands = len(obs["farms"][seat]["hands"])
        except Exception:
            pass
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}


_PLAN: Plan | None = None


def _default_plan() -> Plan:
    global _PLAN
    if _PLAN is None:
        _PLAN = Plan.boatlee_like()
    return _PLAN


def _act(obs, seat: int, plan: Plan) -> dict:
    st = _state(obs, seat)
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    farm = obs["farms"][seat]
    n_hands = len(farm.get("hands") or [])

    step = day * TURNS_PER_DAY + hour
    _watch_opponent(obs, st, seat, step)

    # O1. Ahead of everything that reads the plan — dawn's shopping list and hour 1's compile both
    # have to see the same working plan, and a patch that only reached the compiler would buy the
    # cows it had just decided not to buy. `apply` returns the identical object when no branches are
    # configured, so the default tree is untouched rather than merely equivalent.
    plan = branch_points.apply(obs, plan, seat, day, lambda k, n=1: _note(st, k, n))

    # P2. After O1, because the planner searches from the plan the day will actually be played
    # with, and ahead of everything that reads it — dawn's shopping list and hour 1's compile both
    # have to see the same working plan (a land purchase decided here has to reach `_dawn`'s
    # `BUY_LAND` on this very turn, or it is a decision that was never executed). Returns the
    # identical object when the flag is clear.
    plan = season_planner.apply(obs, plan, seat, day, hour, lambda k, n=1: _note(st, k, n))

    if hour == 0:
        return _dawn(obs, st, plan, seat, day)

    if st["compiled_day"] != day or st["compiled"] is None:
        plan = _paced_plan(obs, plan, seat, day, st)
        # Compile once the crew exists. `start_turn=hour` so the scripts are indexed from now.
        #
        # `cash=0`, deliberately: the shopping happened at hour 0 and nothing more is bought today,
        # so the day must be planned around what is actually in the shed. Passing the leftover cash
        # let the compiler schedule plantings against seeds it *could* have bought, which then hit
        # the atomic PLANT rule and took the whole crop's requests down with them.
        st["compiled"] = compile_day(obs, plan, hands=n_hands, turns=TURNS_PER_DAY,
                                     start_turn=hour, cash=0.0)
        st["compiled_day"] = day
        _tally_compile(st, st["compiled"])

    compiled = st["compiled"]
    index = hour - compiled.start_turn
    actions = []
    for unit in range(n_hands + 1):
        script = compiled.scripts.get(unit)
        ops = script.ops if script is not None else []
        actions.append(list(ops[index]) if 0 <= index < len(ops) else ["PASS"])

    # Sell on any turn a unit unloads, not only at dusk. Unit actions resolve *before* market orders
    # inside a step, so goods DROPped at hour h are sellable at hour h — and they have to be, twice
    # over. A DROP into a full shed **discards** the excess outright (`kaggriculture.py:344-354`),
    # so holding four hands' harvest in the shed until hour 23 is how the second hand's melons
    # evaporate; and the framework plays `episodeSteps - 1` turns, so day 29 ends at hour 22 and an
    # hour-23-only sell never runs at all on the last day of the season.
    dropping = [u for u, a in enumerate(actions) if a and a[0] == "DROP"]
    if dropping:
        _note(st, "hand_drops", len(dropping))
    # O3 front-run. A turn that would emit no market list at all still gets one when the opponent's
    # next dump is inside the lead and we are holding the product: the shed is only ever *sold* on
    # dawns, drop turns and hour 23, so without this the front-run could only fire when the routing
    # calendar happened to agree with the opponent's sell calendar.
    targets = _frontrun_now(obs, st, plan, seat, step)
    scheduled = bool(dropping) or hour == LAST_TURN
    market: list = []
    if scheduled or targets:
        if targets and not scheduled:
            _note(st, "frontrun_turns")
        market = _sell_orders(obs, plan, seat, st, extra=_carried(obs, dropping),
                              frontrun=targets, forced=bool(targets) and not scheduled)
    return {"farmer": actions[0], "hands": actions[1:],
            "market": _dispatch(obs, market, st, hoist=_slot_align_targets(st, plan, step))}


def _watch_opponent(obs, st: dict, seat: int, step: int) -> None:
    """O2, wired passively: name the opponent at steps 24/48/72 and stash their supply forecast.

    Nothing downstream reads `st["opp_known"]` or `st["opp_forecast"]` yet — O3 does. It is here
    now so the counters can be read *in play* rather than from a helper that returned the right
    value in a test (E44), and so the cost of running it shows up in the latency numbers.

    Its own `try` rather than the one in `agent()`: a passive observer that can forfeit a turn is a
    strictly worse trade than one that quietly records nothing.
    """
    if step == 0:
        # The season reset for the module-level verdict O3's counter-mix reads (`opponent.LOCKED`).
        # `_state` rebuilds `st` on step 0; this follows it, for the same reason.
        opponent.reset_locked(seat)
        st.pop("frontrun_map", None)
        st.pop("frontrun_key", None)
    if step not in opponent.CHECKPOINTS:
        return
    try:
        farms = obs.get("farms") or []
        if len(farms) < 2:
            return
        opp_farm = farms[1 - seat]
        memory = st.get("opp_memory")
        if memory is None:
            memory = st["opp_memory"] = opponent.new_memory()
        known = opponent.fingerprint(opp_farm, step, memory)
        st["opp_known"] = known
        st["opp_forecast"] = opponent.forecast_supply(opp_farm, step, known)
        opponent.set_locked(seat, known)
        st.pop("frontrun_map", None)            # the verdict moved; the window map is stale
        st.pop("frontrun_key", None)
        _note(st, "fingerprint_checks")
        if known is not None and st.get("opp_lock_step") is None:
            st["opp_lock_step"] = step
            _note(st, "opponent_known")
            _note(st, f"opponent_is_{known}")
            if not st.get("opp_lock_noted"):
                # A step, not a count: `_note` sums, so this is noted **once** per season and the
                # harness reads the mean back as the step the lock happened on. A re-lock after an
                # unlock would otherwise report step 48 as 96.
                st["opp_lock_noted"] = True
                _note(st, "fingerprint_lock_step", step)
        elif known is None and st.get("opp_lock_step") is not None:
            st["opp_lock_step"] = None
            _note(st, "fingerprint_unlocked")
        if memory.get("unknown") and not st.get("opp_unknown_noted"):
            st["opp_unknown_noted"] = True
            _note(st, "opponent_unknown")
    except Exception:
        _note(st, "fingerprint_errors")


def _frontrun_map(st: dict, plan: Plan) -> dict:
    """`{step: {product: units}}` for the whole season, built once per fingerprint verdict.

    Built eagerly and cached rather than scanned per turn: the table is ~160 rows and a lead of 24
    smears each over 25 steps, so the expanded map is a few thousand small ints — cheaper to build
    once at the lock than to re-scan six sorted lists on every one of 719 turns, and it keeps the
    hot path a single dict lookup (the p99 turn budget is 100 ms and the compile already owns most
    of it).
    """
    lead = max(FRONTRUN_LEAD_MIN,
               min(FRONTRUN_LEAD_MAX, int(_flag(plan, "frontrun_lead", 10))))
    min_units = max(1, int(_flag(plan, "frontrun_min_units", FRONTRUN_MIN_UNITS)))
    known = st.get("opp_known")
    key = (known, lead, min_units, id(st.get("opp_forecast")))
    if st.get("frontrun_key") == key:
        return st.get("frontrun_map") or {}

    if opponent.is_self(known):
        # A mirror match is a different problem (`opponent.SELF_PROFILES`): front-running a copy of
        # ourselves front-running us is not a hypothesis this task tested.
        built: dict = {}
    elif known == "boatlee":
        built = opponent.windows_from_rows(opponent.settled_schedule(known), lead, min_units)
    else:
        # Unknown: their *production* is still public, so the forecast stands in for the schedule,
        # smeared a day either way because a harvest arriving on day d may be sold on d or d+1.
        built = opponent.windows_from_rows(st.get("opp_forecast") or {}, lead, min_units,
                                           spread=FRONTRUN_FORECAST_SPREAD)
    st["frontrun_key"] = key
    st["frontrun_map"] = built
    return built


def _frontrun_now(obs, st: dict, plan: Plan, seat: int, step: int) -> dict:
    """The products to get ahead of on this turn, restricted to ones we are actually holding.

    Holding is checked here and not in `_sell_orders` because it also decides whether the turn gets
    a market list at all: a front-run that only fires on the turns a hand happens to unload is a
    front-run bound to the routing calendar rather than to the opponent's.
    """
    if not int(_flag(plan, "frontrun", 0)):
        return {}
    targets = _frontrun_map(st, plan).get(step)
    if not targets:
        return {}
    shed = ((obs.get("private", {}) or {}).get("shed") or {})
    reserve = _feed_reserve(obs, seat)
    return {p: u for p, u in targets.items()
            if int(shed.get(p, 0) or 0) - int(reserve.get(p, 0) or 0) > 0}


def _slot_align_targets(st: dict, plan: Plan, step: int) -> frozenset:
    """Products a fingerprinted opponent trades on **this exact step** (see `opponent.sell_steps`)."""
    if not int(_flag(plan, "slot_align", 0)):
        return frozenset()
    known = st.get("opp_known")
    if known is None or opponent.is_self(known):
        return frozenset()
    steps = st.get("slot_steps")
    if st.get("slot_steps_key") != known:
        steps = st["slot_steps"] = opponent.sell_steps(known)
        st["slot_steps_key"] = known
    return frozenset(steps.get(step) or ())


def _carried(obs, units) -> dict:
    """What the units unloading this turn are holding.

    The shed in `obs` is the shed *before* this turn's DROP, so an order sized from it alone asks
    for zero of everything the hands are about to bring in — the drop lands, and nothing sells.
    `SELL` moves `min(requested, shed)` (`kaggriculture.py:641-658`), so over-asking costs nothing
    and under-asking is the whole loss.
    """
    inventories = (obs.get("private", {}) or {}).get("inventories") or []
    out: dict = {}
    for unit in units:
        if unit < len(inventories):
            for item, n in dict(inventories[unit]).items():
                out[item] = out.get(item, 0) + int(n)
    return out


def _tally_compile(st: dict, compiled) -> None:
    """C3's repair record, accumulated over the season rather than overwritten every dawn."""
    _note(st, "compiled_days")
    drops = sum(1 for s in compiled.scripts.values() for o in s.ops if o and o[0] == "DROP")
    if drops:
        # C2.4's effect counter, read at compile time. `hand_drops` counts the same drops when they
        # are actually emitted; the two disagreeing means the script is being indexed past the end
        # of the day, which is precisely how the first version of this silently did nothing.
        _note(st, "drops_scheduled", drops)
    if compiled.pruned:
        _note(st, "pruned_tasks", len(compiled.pruned))
        _note(st, "prune_days")
    if compiled.overcommit:
        _note(st, "overcommit_days")
    if compiled.rounds > MAX_PRUNE_ROUNDS:
        # `compile_day` sets `rounds = max_rounds + 1` only on the hands-escalation branch.
        _note(st, "hand_escalations")
    if compiled.report is not None and compiled.report.blocked_ops:
        _note(st, "verifier_blocked_ops", len(compiled.report.blocked_ops))


# --------------------------------------------------------------------------- hour 0

#: How many days a cohort in the arriving quadrant must already be *late* before dawn re-sizes
#: itself against the land it is buying this same turn. See `_dawn`.
#:
#: **Not zero, and the number is measured rather than reasoned.** Re-sizing on *every* land dawn is
#: the coherent rule and it scores slightly better on the cliff, but it also moves the incumbent
#: plans — `Plan.boatlee_like()`, `flooder`, `tomato_rusher` — by **-$1.6k [-3.0k, -0.3k]** on
#: 60100:60140 and **-$2.8k [-4.3k, -1.4k]** on the held-out 61000:61040, 480 paired games each.
#: Those plans were tuned under this behaviour and their crew, herd and cohort sizes are the
#: measured answer to it, so the fix has to leave their land dawns alone (`docs/experiments.md`'s
#: scoping rule).
#:
#: A quadrant that arrives near its plan day carries a cohort due today or tomorrow, and shopping
#: for it a day late is a one-day tax the incumbents already pay for. The defect only compounds
#: when the land is *badly* late — the backlog at the buying dawn, measured over 12 seeds each:
#:
#: | plan | backlog at `BUY_LAND` |
#: |---|---|
#: | `boatlee` | 2x9, 3x2, 4x9, 6x3, 12x1 |
#: | `flooder` | 0x12, 2x13, 4x9, 5x2 |
#: | `tomato_rusher` | 2x9, 3x1, 4x9, 5x1, 6x4 |
#: | S1's cliff genome | **6x20, 8x4** |
#:
#: There is no threshold that separates them cleanly — the incumbents run late too — so this is a
#: judgement about where the tax stops being worth paying, not a clean partition. Six is the
#: highest value that still covers every dawn of the cliff genome, and it costs the incumbents
#: least. Three was also measured and is worse on both blocks.
LATE_COHORT_DAYS = 6


def _quadrant_backlog(plan: Plan, quad: str, day: int) -> int:
    """Days by which the oldest cohort standing in `quad` is already overdue."""
    from agent.tasks import cohort_opens

    return max((day - cohort_opens(plan, c) for c in plan.cohorts if c.quadrant == quad),
               default=0)


def _unlocked_view(obs, seat: int, quad: str):
    """`obs` as it will read once today's `BUY_LAND` settles: that quadrant's tiles free.

    A copy, never a mutation — `obs` is the live observation the opponent's seat and the rest of
    the turn also read from. Only the seat's own tile grid and `unlocked_quadrants` change; every
    other field is shared by reference, which is what `daily_tasks` needs (shed, prices, private).
    """
    from agent.plan import quadrant_of

    farm = obs["farms"][seat]
    tiles = [list(row) for row in farm["tiles"]]
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile == "LOCKED" and quadrant_of(x, y) == quad:
                row[x] = None
    new_farm = dict(farm)
    new_farm["tiles"] = tiles
    new_farm["unlocked_quadrants"] = list(farm.get("unlocked_quadrants") or []) + [quad]
    farms = list(obs["farms"])
    farms[seat] = new_farm
    view = dict(obs)
    view["farms"] = farms
    return view


def _dawn(obs, st: dict, plan: Plan, seat: int, day: int) -> dict:
    """The market turn. The farmer is alone on the board, so it does one useful op and shops."""
    farm = obs["farms"][seat]
    private = obs.get("private", {}) or {}
    money = float(farm.get("money", 0) or 0)

    orders: list = []
    budget = money

    # Land first: it is the gate on everything the plan does later, and it is cheap early.
    bought_quad = None
    for i, quad in enumerate(LAND_ORDER):
        due = plan.land_days.get(quad, NEVER)
        if due < NEVER and day >= due and quad not in set(farm.get("unlocked_quadrants") or []):
            price = (1000, 2000, 4000)[i]
            if budget >= price:
                orders.append(["BUY_LAND"])
                budget -= price
                bought_quad = quad
            break                               # one quadrant at a time, in order

    # When the land is badly late, the rest of dawn is sized against the board the day will *have*
    # rather than the one it opened on (E73). `BUY_LAND` settles in this same market turn and hour 1
    # compiles against the unlocked quadrant, but everything below used to be derived from tiles
    # that still read `"LOCKED"`: `_planting_tasks` saw no free tiles in the arriving quadrant,
    # `decide_hands` sized the crew for a board a quarter smaller than the one it would work, and
    # `_afford` asked for none of the arriving cohort's seed. Hour 1 then compiled a 25-tile cohort
    # against an empty shed and pruned every PLANT in it.
    #
    # Normally that is a one-day tax — the cohort goes in tomorrow — which is why it survived. It
    # stops being a one-day tax when the land arrives late on a harvest windfall, because then two
    # quadrants arrive back to back: the first morning's cash goes to the quadrant and the herd, the
    # second morning's to the *next* quadrant, and there is never again a dawn with both free tiles
    # and the $2.5k of strawberry seed. That is S1's cliff, and it is a compiler defect rather than a
    # strategy: one wheat tile either side of the edge is `plants_started` 66 -> 43 and
    # **$91.5k -> $46.5k** (repro `/private/tmp/s1/v3.py`, 8 seeds from 60050, vs `starter`). A
    # genome search reads the $45k as the shape's fault and steers away from a whole region.
    #
    # `LATE_COHORT_DAYS` is what keeps this off the incumbents' land dawns; see its note.
    view = obs
    if bought_quad is not None and _quadrant_backlog(plan, bought_quad, day) >= LATE_COHORT_DAYS:
        view = _unlocked_view(obs, seat, bought_quad)
        _note(st, "land_day_reseed")

    tasks = daily_tasks(view, _paced_plan(view, plan, seat, day, None), day=day, turn=1)
    wanted = decide_hands(tasks, tuple(farm["farmer"]), turns_left=TURNS_PER_DAY - 1,
                          cash=money * 0.45)
    st["hands_wanted"] = wanted
    st["compiled"] = None                       # the crew changes; compile at hour 1

    for n in range(wanted):
        wage = HIRE_COST[min(n, len(HIRE_COST) - 1)]
        if budget < wage:
            break
        orders.append(["HIRE"])
        budget -= wage

    # What the day is *funded* for, decided by the same routine that compiles it — so the shopping
    # list and the script cannot disagree about which plantings are happening.
    from agent.verify import _afford

    _kept, needs, _dropped = _afford(tasks, view, money)
    for item, count in sorted(needs.items()):
        if item.startswith("SEED:"):
            crop = item.split(":", 1)[1]
            cost = _seed_cost(crop) * count
            if budget >= cost:
                orders.append(["BUY_SEED", crop, count])
                budget -= cost
        else:
            price = float(current_prices(obs).get(item, BASE_PRICE.get(item, 50)) or 50)
            afford = int(budget // max(1.0, price))
            if afford > 0:
                orders.append(["BUY_PRODUCT", item, min(count, afford)])
                budget -= min(count, afford) * price

    step = day * TURNS_PER_DAY
    orders += _animal_orders(obs, plan, seat, day, budget, st)
    orders += _sell_orders(obs, plan, seat, st,
                           frontrun=_frontrun_now(obs, st, plan, seat, step))

    farmer_op = _dawn_farmer_op(obs, seat, private)
    return {"farmer": farmer_op, "hands": [],
            "market": _dispatch(obs, orders, st, hoist=_slot_align_targets(st, plan, step))}


#: Dawn's ten market slots, in the order they are claimed. Below `_SELL_RANK` is the *supply* half of
#: the queue; sells rank last because a dropped sell is deferred and re-derived (E62) while a dropped
#: buy is gone for the day.
#:
#: `_dawn` used to append HIREs **before** the buys, so a ten-hire day bought nothing at all — a
#: simultaneous seed, feed and fertilizer outage, after which hour 1 compiled at `cash=0` and pruned
#: ~50 tasks (E68, traced on seed 49100 days 11-13 and 24). Measured on 53000:53040 the old order
#: dropped **71 orders a game**: 8.2 essential buys, 12.5 animal purchases and the rest sells, with
#: **6.0 dawns a game** losing at least one essential buy.
#:
#: The ranking is by what breaks if the order does not go out today:
#:
#: 0. `BUY_LAND` — the quadrant must be unlocked before hour 1 compiles against it. At most one.
#: 1. `BUY_SEED` / `BUY_PRODUCT` — hour 1 compiles with `cash=0` against the shed *as it stands*, so
#:    seed, feed and fertilizer that do not arrive at dawn are invisible to the whole day's script
#:    and every task needing them is pruned. This is the class HIRE was starving.
#: 2. `BUY_ANIMAL` — the herd is the season's compounding asset and a purchase deferred a day is a
#:    day of production lost at the far end. Measured worth ~$2.5k over ranking it below HIRE.
#: 3. `HIRE` — last, and this is the whole change: the marginal hand on an oversubscribed dawn is
#:    the cheapest thing in the queue to lose, because the *rest* of the crew is idle for want of
#:    supplies. It is not that fewer hands are better — see the control in the docstring below.
_DAWN_RANK = {"BUY_LAND": 0, "BUY_SEED": 1, "BUY_PRODUCT": 1, "BUY_ANIMAL": 2, "HIRE": 3}
_SELL_RANK = 4

#: The classes that must land at dawn or be wasted for the day.
_ESSENTIAL = ("BUY_SEED", "BUY_PRODUCT")


def _dispatch(obs, orders: list, st: dict, hoist=()) -> list:
    """Dawn's ten slots go to the orders that cannot wait, and HIRE waits last.

    **Why not spread the queue over the day?** `maxMarketOrdersPerTurn` is a **per-turn** cap, not a
    per-day one: `_process_market` runs on every step of the interpreter, unconditionally
    (`kaggriculture.py:941`; `kagsim/src/rules.rs:585`) and truncates *that turn's* queue at ten
    (`:551,560`). A day has 240 order slots. That looks like the fix, and it was measured three ways
    on 53000:53040 against `starter`, 80 paired games each, all of them worse than simply letting
    the marginal hire go:

    | variant | vs pristine |
    |---|---:|
    | this one — rank, and drop the hires that do not fit | **+$28,875** [25.0k, 32.7k], 75/80 |
    | roll the spilled hires to hour 1, recompile for them at hour 2 | +$22,457, blocked_ops 39 |
    | roll them and idle the crew at hour 1 so one compile sees them | +$25,436 |
    | rank HIRE above BUY_ANIMAL, roll the spilled animals to hour 1 | +$26,363 |

    A hand hired at hour h stands on the board at h+1, so a rolled hire either costs a mid-day
    recompile — which the counters price at **39 blocked ops a game** — or costs the *whole crew* an
    idle hour waiting for it. Both are dearer than the hand. The later hours stay free for the sell
    hooks, which is what they were already being used for.

    **And it is not a headcount cut in disguise.** The obvious alternative reading — that the gain is
    just hiring 17 fewer hands a game — was measured as its own arm: pristine with `wanted - 2` and
    nothing else scores **-$36,629** [-42.2k, -31.1k], 7/80. Hands are worth a great deal; what they
    are not worth is the feed the tenth of them displaces.
    """
    naive_kept = sum(1 for o in orders[:MAX_ORDERS] if o[0] in _ESSENTIAL)
    naive_want = sum(1 for o in orders if o[0] in _ESSENTIAL)
    if naive_want > naive_kept:
        # The disease, counted in the old append order so it stays comparable across the fix.
        _note(st, "dawn_starved_naive")
        _note(st, "dawn_starved_naive_orders", naive_want - naive_kept)
    if len(orders) <= MAX_ORDERS:
        return _hoist(orders, hoist, st)

    _note(st, "orders_truncated")
    prices = current_prices(obs)

    def rank(item):
        i, o = item
        if o[0] == "SELL":
            # Within the sells, biggest proceeds first — E62's rule, kept.
            value = float(prices.get(o[1], BASE_PRICE.get(o[1], 50)) or 50) * int(o[2])
            return (_SELL_RANK, -value, i)
        return (_DAWN_RANK.get(o[0], _SELL_RANK), 0.0, i)

    ranked = [o for _i, o in sorted(enumerate(orders), key=rank)]
    keep, spill = ranked[:MAX_ORDERS], ranked[MAX_ORDERS:]

    starved = sum(1 for o in spill if o[0] in _ESSENTIAL)
    if starved:
        # The tripwire. Only reachable when dawn wants more than ten essentials on its own, which is
        # measured at zero — a nonzero value means the ranking has stopped being sufficient and the
        # deferral arms above are worth revisiting.
        _note(st, "dawn_starved")
        _note(st, "dawn_starved_orders", starved)
    hires = sum(1 for o in spill if o[0] == "HIRE")
    if hires:
        _note(st, "hires_dropped", hires)
    deferred = sum(1 for o in spill if o[0] == "SELL")
    if deferred:
        _note(st, "sells_deferred", deferred)
    _note(st, "orders_dropped", len(spill))
    return _hoist(keep, hoist, st)


def _hoist(orders: list, hoist, st: dict) -> list:
    """O3 slot alignment: put contested sells in the lowest market slots.

    `_process_market`'s outer loop is over the **order index**, and slot `i` is drained for both
    players and re-priced before slot `i+1` is quoted (`kaggriculture.py:563,628`; `_dispatch`'s
    docstring above has the per-turn cap half of the same loop). So when both farms sell STRAWBERRY
    on one step, the one whose order sits at a lower index sells its whole line into the untouched
    inventory and the other one is quoted after the price has already fallen. Within a *single* slot
    both players see the same pre-commit quote (`:612-618`), so being early is worth a great deal and
    being "first" inside a slot is worth nothing — there is no seat advantage to take, only a slot
    one, and it is taken by moving the order, not by winning a tie.

    Applied **after** truncation, so it can only reorder what was already going out: hoisting a sell
    must never be able to push an essential buy off the ten-slot queue (E68's failure mode). It is a
    pure permutation — nothing is added and nothing is dropped.

    Measured on `compiler` vs `boatlee` (seed 67000): 17 same-step same-product collisions a season,
    of which we sat in the later slot on 11. Small, and free.
    """
    if not hoist or not orders:
        return orders
    front = [o for o in orders if o and o[0] == "SELL" and o[1] in hoist]
    if not front or orders[:len(front)] == front:
        return orders
    _note(st, "sells_reslotted", len(front))
    rest = [o for o in orders if not (o and o[0] == "SELL" and o[1] in hoist)]
    return front + rest


def _seed_cost(crop: str) -> int:
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    return int(CROPS[crop]["seed"])


#: How far the structure schedule may run ahead of the animals actually acquired.
#:
#: Boatlee buys structure and animal 1:1 and never owns an empty pasture; our task generator cannot
#: match that exactly, because `_structure_tasks` only emits a PLACE for a structure that already
#: stood at dawn — so an animal bought on day *d* needs its pasture built on day *d*, and is placed
#: on *d+1*. One day of lead is therefore the strict-1:1 analogue, and the lead is sized by the
#: plan's own `animals_per_day` so a day that buys three animals has three homes waiting for them
#: rather than placing one a day for three days.
def _structure_lead(plan: Plan) -> int:
    return max(1, int((plan.consts or {}).get("animals_per_day", 1) or 1))


def _paced_plan(obs, plan: Plan, seat: int, day: int, st: dict | None) -> Plan:
    """The plan's herd, trimmed to the animals the farm has actually got hold of.

    `_structure_tasks` builds one structure per *scheduled* animal, so a plan whose herd is due by
    day 7 puts thirteen pastures on the board by day 8 while three animals stand in them (E66): the
    labour is spent, the homes are empty, and the schedule says nothing about whether the wallet
    ever reached $400. Trimming the herd to `owned + lead` makes the structures track the animals
    instead of the calendar, which is Boatlee's 1:1 pattern.

    This is a *pacing* device only. Species order and the plan's buy days are untouched — the prefix
    is taken in the plan's own schedule order, so slot k still serves the k-th animal and
    `pasture_tiles[k]` still belongs to it. Nothing is ever brought forward: `due` is the cap.
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

    herd = plan.herd
    if not herd:
        return plan
    due = sum(1 for _s, d in herd if d <= day)
    if due <= 0:
        return plan

    farm = obs["farms"][seat]
    shed = (obs.get("private", {}) or {}).get("shed", {}) or {}
    owned = sum(1 for row in farm["tiles"] for t in row
                if isinstance(t, dict) and t.get("animal"))
    owned += sum(int(shed.get(s, 0) or 0) for s in ANIMALS)

    allowed = min(due, owned + _structure_lead(plan))
    if allowed >= due:
        return plan
    if st is not None:
        _note(st, "herd_paced_days")
        _note(st, "structures_deferred", due - allowed)
    from dataclasses import replace

    return replace(plan, herd=herd[:allowed])


#: The last day of the season, and the last day a purchase can still pay for itself.
LAST_DAY = 29

_LAST_BUY_DAY: dict[str, int] = {}


def _last_buy_day(species: str) -> int:
    """The last dawn on which buying one head of `species` can still repay its own price.

    Arithmetic, not a tuned constant:

    * An animal bought at dawn of day `d` is carried out and PLACEd by a unit on `d + 1` —
      `_structure_tasks` only emits a PLACE into a structure that already stood at dawn (see
      `_structure_lead`), so `placed_day = d + 1` is the earliest the board ever shows.
    * Production happens at dusk: `days_since_first = next_day - placed_day - first_yield_day`, and
      a unit appears when that is >= 0 and divisible by `interval`
      (`kaggriculture.py:822-823`). So the k-th production closes the day
      `placed_day + first_yield_day + (k - 1) * interval`, and it must land on or before day 29 to
      be collectable and sellable within the season.
    * A fed-and-cared animal yields `1 + interval` units per production: base 1 plus one accumulated
      care bonus for each day since the last one (`kaggriculture.py:824-829`).

    So the head repays `cost` only if `k = ceil(cost / ((1 + interval) * base))` productions still
    fit. At base prices that is one production for a COW ($480 of milk against $400) and for a SHEEP
    ($800 of wool against $500), three for a GOOSE ($300 of eggs against $300) — cutoffs of day 20,
    22 and 22. It is deliberately generous: it uses the *base* price, ignores feed and the labour a
    late animal takes away from the rest of the herd, and every real sale this season has cleared
    base. What it removes is the purchase that cannot produce at all — E66's re-gate measured 7
    animals a season ordered on day 25+, where a cow's 8-day warm-up alone runs past the buzzer.
    """
    if species not in _LAST_BUY_DAY:
        from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

        spec = ANIMALS[species]
        per_production = (1 + int(spec["interval"])) * BASE_PRICE.get(spec["product"], 50)
        need = max(1, -(-int(spec["cost"]) // max(1, per_production)))
        last_production = LAST_DAY - (need - 1) * int(spec["interval"])
        _LAST_BUY_DAY[species] = last_production - int(spec["first_yield_day"]) - 1
    return _LAST_BUY_DAY[species]


def _yield_per_dollar(species: str) -> float:
    """Units a head returns per day of ownership, per dollar it costs.

    `(1 + interval)` units every `interval` days (`kaggriculture.py:822-829`) at the product's base
    price, divided by the purchase price. COW $0.60/day/$, SHEEP $0.53, GOOSE $0.33 — the cow is
    both the cheapest head on the board and the best earner per dollar, which is why it goes first
    when the wallet cannot afford everything the day is due.
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

    spec = ANIMALS[species]
    interval = max(1, int(spec["interval"]))
    rate = (1 + interval) / interval * BASE_PRICE.get(spec["product"], 50)
    return rate / max(1.0, float(spec["cost"]))


def _purchase_order(plan: Plan, day: int) -> list:
    """The plan's due herd, richest-per-dollar first.

    The schedule order is the *building* order, and it is also — wrongly — the buying order: the
    loop below stops at the first head the wallet cannot afford, so a $500 sheep standing ahead of a
    $400 cow in the list spends the early cash on the worse animal and leaves the cow waiting. E66's
    re-gate measured the consequence: first animal on day 2, first **cow** on day 11.

    Reordering is safe for the slot -> pasture mapping only while every due head wants the same kind
    of structure — `_structure_tasks` picks each slot's structure from the plan's own order, so a
    COOP bought against a PASTURE slot would sit in the shed. COW and SHEEP both want a PASTURE, so
    the common case reorders; a mixed herd keeps the schedule order untouched.
    """
    due = [(species, buy_day) for species, buy_day in plan.herd if buy_day <= day]
    from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

    if len({ANIMALS[s]["structure"] for s, _d in due}) > 1:
        return due
    return sorted(due, key=lambda sd: (-_yield_per_dollar(sd[0]), sd[1]))


def _animal_orders(obs, plan: Plan, seat: int, day: int, budget: float,
                   st: dict | None = None) -> list:
    """Buy the animals the plan is due, but only when there is somewhere to put them.

    `BUY_ANIMAL` fails outright against a full shed and an animal in the shed earns nothing, so the
    order waits until its structure is built or being built today.
    """
    from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

    farm = obs["farms"][seat]
    private = obs.get("private", {}) or {}
    shed = dict(private.get("shed", {}) or {})
    tiles = farm["tiles"]

    placed: dict = {}
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal"):
                placed[tile["animal"]] = placed.get(tile["animal"], 0) + 1

    total_animals = sum(placed.values()) + sum(int(shed.get(s, 0)) for s in ANIMALS)

    orders = []
    for species, buy_day in _purchase_order(plan, day):
        spec = ANIMALS[species]
        have = placed.get(species, 0) + int(shed.get(species, 0))
        want = sum(1 for s, d in plan.herd if s == species and d <= day)
        if have >= want:
            continue
        if day > _last_buy_day(species):
            # Too late to repay its own price. Skipping the species rather than breaking is safe:
            # `_structure_tasks` picks each slot's structure kind from the plan's own herd order, so
            # a slot whose animal is never bought simply never gets its PLACE, and the slots of the
            # other species are unaffected.
            if st is not None:
                _note(st, "animals_past_payback")
            continue
        if budget < spec["cost"]:
            # Strict schedule order: skipping an unaffordable animal to buy a cheaper one further
            # down the list breaks the slot->pasture mapping `_structure_tasks` relies on (slot k
            # holds a SHEEP, the shed holds a COW, and the PLACE never fires).
            if st is not None:
                _note(st, "animals_blocked_cash")
            break
        # An animal eats one wheat a day and escapes after two missed dusks, so the herd is capped
        # by what the farm can actually *feed*. Without any cap the agent re-bought a herd it could
        # not feed three times in one season, ~$450 a head, and the money curve looked like a
        # strategy failing rather than livestock starving. The cap is recomputed against the budget
        # this purchase would leave behind, so it is a solvency test and not a calendar.
        if total_animals + 1 > _feedable_animals(obs, seat, budget - spec["cost"]):
            if st is not None:
                _note(st, "animals_blocked_feed")
            break
        if st is not None and total_animals + 1 > _feedable_animals(obs, seat):
            # The old wheat-tile-only rule would have refused this one (E66's throttle).
            _note(st, "feed_throttle_relaxed")
        if sum(shed.values()) >= 95:
            break                                # a full shed rejects the purchase outright
        orders.append(["BUY_ANIMAL", species, 1])
        shed[species] = shed.get(species, 0) + 1
        total_animals += 1
        budget -= spec["cost"]
        if st is not None:
            _note(st, "animals_ordered")
            _note(st, f"animals_ordered_d{min(29, day) // 5 * 5:02d}")
    return orders


#: `FEED` takes exactly one WHEAT out of the keeper's inventory, once per animal per day
#: (`kaggriculture.py:504-511`), and two missed dusks lose the animal.
FEED_PER_ANIMAL_PER_DAY = 1

#: Days of feed the wallet must still cover after a purchase. Measured, not reasoned: on
#: 46000:46040 against `starter` the herd fix is worth +$17.0k at two days and +$22.9k at four,
#: with escapes falling 4.9 -> 3.3 a season, and the ordering repeats on the held-out 47000 block
#: (+$32.2k / +$38.6k). Five and above give the money back (+$4.5k at five), so this is a shoulder
#: rather than a spike — but it is a *tuned* number, and it expires with the plan and the router
#: that produced it.
FEED_BUFFER_DAYS = 4


def _feedable_animals(obs, seat: int, budget: float = 0.0) -> int:
    """How many head the farm can keep fed: what it grows, what is in the shed, what it can buy.

    A wheat tile yields 4 units over a ~5 day cycle, so it sustains roughly 0.8 animals, and the
    shed carries the rest for a day or two. The third term is the one E66 found missing: **wheat is
    purchasable** (`BUY_PRODUCT` accepts WHEAT and FERTILIZER), and it is how Boatlee feeds twelve
    head off thirteen tiles. Judging the herd by tile count alone refused animals on days when the
    wallet held $6,490 and a cow's whole season of feed costs about $30 at the base price — the herd
    landed on day 18.6 against a plan that said day 5.

    `budget` is what the wallet would still hold *after* the purchase being considered, so this
    stays a genuine solvency guard rather than an unconditional yes.
    """
    farm = obs["farms"][seat]
    private = obs.get("private", {}) or {}
    wheat_tiles = sum(1 for row in farm["tiles"] for t in row
                      if isinstance(t, dict) and t.get("crop") == "WHEAT")
    shed_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0) or 0)
    price = float(current_prices(obs).get("WHEAT", 0) or BASE_PRICE["WHEAT"])
    purchasable = int(max(0.0, budget) // max(1.0, price))
    days = FEED_BUFFER_DAYS * FEED_PER_ANIMAL_PER_DAY
    return int(0.8 * wheat_tiles) + (shed_wheat + purchasable) // days


def _dawn_farmer_op(obs, seat: int, private: dict) -> list:
    """One op for the lone farmer at hour 0: drop what it is carrying, else wait for the crew."""
    inventories = private.get("inventories") or []
    carrying = dict(inventories[0]) if inventories else {}
    if carrying and tuple(obs["farms"][seat]["farmer"]) in SHED_TILES:
        item = max(carrying, key=lambda k: carrying[k])
        return ["DROP", item, int(carrying[item])]
    return ["PASS"]


# --------------------------------------------------------------------------- selling

def _sell_orders(obs, plan: Plan, seat: int, st: dict | None = None,
                 extra: dict | None = None, frontrun: dict | None = None,
                 forced: bool = False) -> list:
    """Sell the shed, honouring the plan's floors — until the shed itself becomes the constraint.

    Sell-on-sight is the measured default (E11: the whole market-timing apparatus lost to dumping),
    with three exceptions the plan carries: goods we flood ourselves get a price floor, from
    step 700 everything goes because stock left at the buzzer is worth nothing, and the floors are
    released once the shed is under pressure.

    That last one is `release_pressure` (TASKS_v4 C4). A floor that holds stock back is holding it
    in a 100-unit shed, and at dusk every carried inventory drops into that shed with the **overflow
    discarded outright** (`kaggriculture.py:843-857`). So a floor defended past capacity does not
    protect a price, it throws the day's harvest away at par with the price it was refusing. Above
    the threshold the withheld stock is released, most valuable first, and only down to the
    threshold — the floor still governs everything below it.

    `extra` is stock that is not in the shed yet but will be by the time these orders are processed:
    the inventories of the units DROPping this same turn (see `_carried`).

    **Precedence, O3.** Four rules can want a different quantity of the same product; they are
    resolved in this order, highest first, and the order is a claim about which one is *right* when
    they disagree:

    1. `step >= TERMINAL_STEP` — sell everything. Unchanged and unconditional: stock at the buzzer
       is worth zero, so no floor and no schedule can beat liquidation.
    2. **front-run** (`frontrun`, this turn's products) — sell everything, **the floor suspended**.
       A floor is a bet that the price will recover before we need the space; a scheduled dump of
       34 strawberries is the event that says it will not, and the quote it lands on ($120 -> $55,
       measured) is below any floor worth setting. Defending a floor into a dump does not protect
       the price, it just sells the same stock later at the price the dump left behind. The units
       the floor *would* have withheld are counted separately (`frontrun_floor_override`) so this
       precedence can be re-litigated from the counters rather than from the argument.
    3. `sell_floor` — the binary search, unchanged.
    4. `release_pressure` — releases what 3 withheld, unchanged. Moot for a front-run product,
       whose stock has already gone out under 2.

    `forced` says this turn had no market list of its own (no DROP, not hour 23, not dawn) and
    exists only for the front-run, so the counterfactual quantity for the counters is zero rather
    than what the floor would have allowed.
    """
    fr = frontrun or {}
    from agent import projection
    plan = projection.scarcity_plan(obs, plan, seat)
    private = obs.get("private", {}) or {}
    shed = dict(private.get("shed", {}) or {})
    for item, n in (extra or {}).items():
        shed[item] = shed.get(item, 0) + int(n)
    prices = current_prices(obs)
    step = int(obs.get("step", 0) or 0)
    floors = (plan.consts or {}).get("sell_floor", {}) or {}
    reserve = _feed_reserve(obs, seat)

    orders = []
    withheld: list = []
    for item in SELLABLE:
        held = int(shed.get(item, 0) or 0) - int(reserve.get(item, 0) or 0)
        if held <= 0:
            continue
        if step >= TERMINAL_STEP:
            orders.append(["SELL", item, held])
            continue
        floor = float(floors.get(item, 0) or 0)
        if item in fr:
            # `frontrun_keep_floor` is the second half of the precedence question, made measurable
            # instead of argued. With it clear (the default) the front-run suspends the floor; with
            # it set the floor still sizes the order and the overlay contributes only the *turn* —
            # the sell happens now rather than at the next DROP. Both arms were run; see the
            # module note in `_frontrun_note`.
            allowed = held
            if floor > 0 and int(_flag(plan, "frontrun_keep_floor", 0)):
                allowed = _quantity_above_floor(item, held, prices, floor, obs)
            if allowed > 0:
                orders.append(["SELL", item, allowed])
            _frontrun_note(st, item, allowed, held, floor, forced, prices, obs)
            if allowed < held:
                withheld.append((item, held - allowed))
            continue
        if floor <= 0:
            orders.append(["SELL", item, held])
            continue
        quantity = _quantity_above_floor(item, held, prices, floor, obs)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
        if quantity < held:
            withheld.append((item, held - quantity))
            # A floor that never withholds a unit is a gene that never fired. S3 widened this from
            # two products to eight and the only way to tell a binding floor from an inert one is
            # per product, in play (E44) — so the counter is per product as well as in total.
            if st is not None:
                _note(st, "floor_withheld_units", held - quantity)
                _note(st, f"floor_withheld_{item}", held - quantity)
                _note(st, "floor_binds")

    if step < TERMINAL_STEP and withheld:
        _release_pressure(obs, plan, shed, prices, orders, withheld, st)
    return orders


def _frontrun_note(st, item: str, sold: int, held: int, floor: float, forced: bool,
                   prices: dict, obs) -> None:
    """Count what the front-run *moved*, never what it ordered.

    "An order is not a sale" (CLAUDE.md) cuts both ways here. `frontrun_units` is the counterfactual
    difference — units that would still be sitting in the shed on this turn without the overlay —
    and not the size of the order, most of which would have gone out anyway. `frontrun_value` prices
    them at the live quote, which is an upper bound on what they fetch (the quote falls as the line
    executes); the money question is settled by the paired money numbers, and this is the proof the
    change fired at all (E44).
    """
    if st is None:
        return
    allowed = held if floor <= 0 else _quantity_above_floor(item, held, prices, floor, obs)
    base = 0 if forced else allowed
    moved = sold - base
    if moved <= 0:
        return
    _note(st, "sells_frontrun")
    _note(st, "frontrun_units", moved)
    _note(st, f"frontrun_{item}", moved)
    _note(st, "frontrun_value",
          int(moved * float(prices.get(item, BASE_PRICE.get(item, 50)) or 50)))
    if sold > allowed:
        _note(st, "frontrun_floor_override", sold - allowed)


def _release_pressure(obs, plan: Plan, shed: dict, prices: dict, orders: list,
                      withheld: list, st: dict | None) -> None:
    """Free `shed_total - release_pressure` units of space out of the floor-protected stock."""
    pressure = int((plan.consts or {}).get("release_pressure", 0) or 0)
    if pressure <= 0:
        return
    excess = sum(int(n or 0) for n in shed.values()) - pressure
    if excess <= 0:
        return

    # Highest current quote first: the same unit of shed space is freed either way, so free it with
    # the sale that earns the most.
    withheld.sort(key=lambda p: -float(prices.get(p[0], BASE_PRICE.get(p[0], 50)) or 50))
    index = {o[1]: o for o in orders if o[0] == "SELL"}
    released = 0
    for item, spare in withheld:
        if excess <= 0:
            break
        take = min(int(spare), excess)
        if take <= 0:
            continue
        existing = index.get(item)
        if existing is not None:
            existing[2] = int(existing[2]) + take
        else:
            order = ["SELL", item, take]
            orders.append(order)
            index[item] = order
        excess -= take
        released += take

    if released and st is not None:
        _note(st, "release_valve_fires")
        _note(st, "release_valve_units", released)


def _feed_reserve(obs, seat: int) -> dict:
    """Wheat and fertilizer the farm needs for itself before any of it is for sale.

    Selling the shed flat is nearly right and was catastrophically wrong in one place: wheat is
    also feed. Dumping it at dawn left the keepers with nothing to carry, and an animal missing two
    dusks escapes — a herd bought for $5,000 walked off the board twice in a measured season, which
    read as "the plan cannot afford livestock" rather than "we sold their dinner".

    Two days of feed, because a purchase can fail on a full shed or an empty wallet.
    """
    farm = obs["farms"][seat]
    animals = sum(1 for row in farm["tiles"] for t in row
                  if isinstance(t, dict) and t.get("animal"))
    fertilizing = sum(1 for row in farm["tiles"] for t in row
                      if isinstance(t, dict) and t.get("kind") == "PLANT")
    return {"WHEAT": animals * 2, "FERTILIZER": min(6, fertilizing)}


def _quantity_above_floor(item: str, held: int, prices: dict, floor: float, obs) -> int:
    """Largest quantity whose *last* unit still clears `floor x base`.

    Priced against the market's own curve rather than a guess: selling into a floor is how melon at
    scale collapses to $1 (E41), and how a self-flooded good earns less than half of what a metered
    one does.
    """
    try:
        import kagsim

        inventory = int((obs.get("market", {}).get("inventory", {}) or {}).get(item, 10_000))
        index = SELLABLE.index(item)
        limit = floor * BASE_PRICE.get(item, 50)
        lo, hi = 0, held
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if kagsim.market_price(index, inventory + mid - 1) >= limit:
                lo = mid
            else:
                hi = mid - 1
        return lo
    except Exception:
        # No simulator on the submission host: fall back to the live quote, which is right for the
        # first unit and conservative for the rest.
        return held if float(prices.get(item, 0) or 0) >= floor * BASE_PRICE.get(item, 50) else 0


# --------------------------------------------------------------------------- diagnostics

def counters(seat: int = 0) -> dict:
    """Effect counters for the harness: a zero here means the shell never fired (E44).

    `pruned`/`overcommit` are **season** totals. They used to be read off `st["compiled"]`, which is
    only the most recently compiled day, so they reported the last dawn and nothing else — and
    nothing consumed them either (0 of 488 rows in `results/games.jsonl` carried one).
    """
    st = _STATE.get(seat) or {}
    compiled = st.get("compiled")
    effects = dict(st.get("effects") or {})
    out = {
        "fallbacks": st.get("fallbacks", 0),
        "hands_wanted": st.get("hands_wanted", 0),
        "compiled_day": st.get("compiled_day", -1),
        "pruned": effects.get("pruned_tasks", 0),
        "overcommit": effects.get("overcommit_days", 0),
        "pruned_today": len(compiled.pruned) if compiled else 0,
        "overcommit_today": bool(compiled.overcommit) if compiled else False,
    }
    out.update(effects)
    return out


def make_agent(plan: Plan | None = None):
    """A fresh agent bound to a plan — what the harness registry builds."""
    chosen = plan or Plan.boatlee_like()
    seats: dict = {}
    ctx = Ctx()

    def play(obs):
        seat = _seat(obs)
        if int(obs.get("step", 0) or 0) == 0:
            seats[seat] = True
        out = agent(obs, chosen)
        # Rebind rather than copy: `_state` builds a fresh dict on step 0, so a ctx that held the
        # old one would report the previous season's totals (the agents are rebuilt per game, but
        # `_STATE` is a module global and outlives them).
        ctx.effects = (_STATE.get(seat) or {}).get("effects", ctx.effects)
        return out

    play.plan = chosen
    play.ctx = ctx
    return play
