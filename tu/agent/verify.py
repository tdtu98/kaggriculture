"""C3 — run the compiled day before committing to it, and prune what does not fit.

The router hands over a plausible day. This runs it through `agent/forward.py` — the same farm model
that is parity-tested against kagsim step by step — and asks two questions the router cannot answer
about its own output:

**Did every op actually do something?** An op that arrives to find a weed, an empty tile or an empty
hand costs a full turn and changes nothing. That is `blocked_ops`, the compiler's health counter,
and it is measured here the way kagsim measures it: by fingerprinting the state around each op
rather than by re-deriving which ops "should" work (E39).

**Does the day sustain itself?** A plan can be perfectly routable today and still be a mistake,
because planting is a promise to water tomorrow and the day after. The engine line's whole failure
was starting plants it could not finish — 50 lost to thirst in E44 — and it is invisible on the day
you plant them. So the check is explicitly forward-looking, and when it fails the cheapest plants
are dropped until the day is honest.

The order of the repair is fixed and deliberate: **prune plantings, never survival work.** A plant
that dies costs its whole remaining life; a plant not started costs one seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.forward import FarmModel
from agent.router import route, spawn_positions
from agent.tasks import LAST_DAY, daily_tasks, market_needs

TURNS_PER_DAY = 24
MOVE_OPS = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})

#: How many prune-and-reroute rounds before accepting an over-committed day. Bounded because a
#: compile runs inside a turn, and because if three rounds have not fixed it the problem is the
#: plan, not the routing — which is S2's business, not the compiler's.
MAX_PRUNE_ROUNDS = 3

#: Turns a unit spends walking and loading rather than working, as a share of the day. Used only to
#: forecast *tomorrow's* capacity, where an exact figure is not available and not needed.
OVERHEAD_SHARE = 0.35


@dataclass
class DayReport:
    deaths: list = field(default_factory=list)            # plants lost to thirst — a routing failure
    expired: list = field(default_factory=list)           # plants that decayed after paying out
    blocked_ops: list = field(default_factory=list)       # (unit, turn, op) that changed nothing
    unwatered_tomorrow: int = 0
    capacity_tomorrow: float = 0.0
    end_state: FarmModel | None = None

    @property
    def sustainable(self) -> bool:
        """Can tomorrow water everything that will need it?"""
        return self.unwatered_tomorrow <= self.capacity_tomorrow

    @property
    def ok(self) -> bool:
        return not self.deaths and self.sustainable

    def __repr__(self) -> str:
        return (f"<DayReport thirst={len(self.deaths)} age={len(self.expired)} "
                f"blocked={len(self.blocked_ops)} "
                f"tomorrow={self.unwatered_tomorrow:.0f}/{self.capacity_tomorrow:.0f}>")


@dataclass
class CompiledDay:
    scripts: dict
    hands: int
    report: DayReport
    tasks: list
    start_turn: int = 0
    pruned: list = field(default_factory=list)
    overcommit: bool = False
    rounds: int = 0

    def actions_at(self, turn: int) -> dict:
        out = {}
        for unit, script in self.scripts.items():
            out[unit] = script.ops[turn] if turn < len(script.ops) else ["PASS"]
        return out


# --------------------------------------------------------------------------- verification

def _fingerprint(model: FarmModel, unit: int):
    """Position, the tile under the unit, and what it is carrying.

    The same three things kagsim compares around an op (`rules.rs:340`) to decide whether anything
    happened. Reusing the definition rather than inventing one keeps `blocked_ops` comparable
    between the compiler and the harness.
    """
    x, y = model.units[unit]
    tile = model.tiles[y][x]
    return ((x, y),
            tuple(sorted(tile.items())) if isinstance(tile, dict) else tile,
            tuple(sorted(model.invs[unit].items())))


def verify_day(obs, scripts: dict, turns: int = TURNS_PER_DAY, start_turn: int = 0,
               model: FarmModel | None = None, crew: int | None = None,
               supplies: dict | None = None) -> DayReport:
    """Play the compiled day on the forward model and report what really happens.

    Runs to the dusk refresh deliberately: a plant dies at `consecutive_unwatered >= 2`, which is an
    end-of-day event, so a verifier that stopped at the last turn would miss exactly the failure it
    exists to catch.

    `crew` is the number of units the scripts were routed for. At dawn the hands do not exist yet —
    they are hired in hour 0's market and stand up at hour 1 — so verifying an eight-unit day
    against the one farmer on the board has seven of the scripts execute against nobody. That read
    as 470 plant deaths a season: a verifier failing the day it was checking.
    """
    model = (model or FarmModel.from_obs(obs, obs.get("player", 0))).clone()
    if supplies:
        # What C4 buys at hour 0. Without it every PLANT verifies as blocked for want of a seed
        # that will, in the real day, have been bought two turns earlier — a verifier reporting a
        # failure of a step it declined to model.
        for item, n in supplies.items():
            if item.startswith("SEED:"):
                crop = item.split(":", 1)[1]
                model.seeds[crop] = model.seeds.get(crop, 0) + int(n)
            else:
                model.shed[item] = model.shed.get(item, 0) + int(n)
    if crew is not None:
        farmer = model.units[0]
        for pos in spawn_positions(max(0, crew - len(model.units)), farmer)[1:]:
            model.units.append(tuple(pos))
            model.invs.append({})
    n_units = len(model.units)
    before = _plant_map(model)

    blocked: list = []
    for turn in range(start_turn, turns):
        actions = []
        for unit in range(n_units):
            script = scripts.get(unit)
            ops = script.ops if script is not None else []
            idx = turn - start_turn
            actions.append(list(ops[idx]) if idx < len(ops) else ["PASS"])

        marks = [_fingerprint(model, u) for u in range(n_units)]
        model.step(actions)
        for unit, action in enumerate(actions):
            if action[0] in MOVE_OPS or action[0] == "PASS":
                continue
            if unit < len(model.units) and _fingerprint(model, unit) == marks[unit]:
                blocked.append((unit, turn, action[0]))

    # A plant leaves the board two ways and they mean opposite things: harvested (a one-time crop
    # clears its tile — that is the *point*) or dead of thirst, which turns the tile into a WEED.
    # Counting disappearances made every successful melon harvest read as a death.
    # Split the two ways a plant leaves as a weed, exactly as the harness counter does (E55): old
    # age is a crop that paid out and expired, thirst is one the routing failed. Only thirst is a
    # compiler failure, and treating decay as one had the verifier prune healthy plantings to
    # atone for wheat that had simply finished.
    after = _plant_map(model)
    deaths, expired = [], []
    for key, tile in before.items():
        if key in after or not _is_weed(model, key[0], key[1]):
            continue
        mls = int(tile.get("max_lifespan_step", -1))
        if 0 <= mls <= model.step_idx:
            expired.append(key)
        else:
            deaths.append(key)

    unwatered = sum(1 for t in after.values() if int(t.get("consecutive_unwatered", 0)) >= 1)
    capacity = _capacity(n_units, model, turns)   # n_units already includes the injected crew
    return DayReport(deaths=deaths, expired=expired, blocked_ops=blocked,
                     unwatered_tomorrow=unwatered, capacity_tomorrow=capacity, end_state=model)


def _is_weed(model: FarmModel, x: int, y: int) -> bool:
    tile = model.tiles[y][x]
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _plant_map(model: FarmModel) -> dict:
    out = {}
    for y, row in enumerate(model.tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                out[(x, y, int(tile.get("planted_day", -1)))] = tile
    return out


def _capacity(n_units: int, model: FarmModel, turns: int) -> float:
    """Roughly how many waters tomorrow's crew can deliver after the animals are served.

    Deliberately crude and deliberately *forward-looking*: the exact figure needs tomorrow's route,
    which needs tomorrow's board. What matters is catching the day where we plant thirty tiles that
    nobody can reach, which no exact model of today would ever notice.
    """
    animals = sum(1 for row in model.tiles for t in row
                  if isinstance(t, dict) and t.get("animal"))
    animal_turns = animals * 3
    return max(0.0, n_units * turns * (1 - OVERHEAD_SHARE) - animal_turns)


def _seed_stock(obs, supplies: dict) -> dict:
    """Seeds in hand at hour 1: what the shed had plus what hour 0 bought."""
    seeds = dict((obs.get("private", {}) or {}).get("seeds", {}) or {})
    for item, n in (supplies or {}).items():
        if item.startswith("SEED:"):
            crop = item.split(":", 1)[1]
            seeds[crop] = seeds.get(crop, 0) + int(n)
    return seeds


def _afford(tasks, obs, cash: float | None):
    """Keep only the work the wallet can actually supply, and say what to buy for it.

    The day is planned against the plan, but executed against the bank. A PLANT with no seed is not
    a planting that fails politely — it is a turn spent standing on bare ground, and the WATER the
    router chains behind it is a second one. Measured on a real season that was **538 blocked ops**,
    nearly all of them plantings the farm could not fund and their orphaned waters.

    Rationing is by value, so a cash-poor day buys the strawberry seed and skips the wheat rather
    than the other way round.
    """
    private = obs.get("private", {}) or {}
    if cash is None:
        return list(tasks), market_needs(tasks, private), []

    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    prices = dict((obs.get("market", {}) or {}).get("prices", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    carried: dict = {}
    for inv in (private.get("inventories") or []):
        for item, n in dict(inv).items():
            carried[item] = carried.get(item, 0) + int(n)

    budget = float(cash)
    kept, dropped = [], []
    supplies: dict = {}
    for task in sorted(tasks, key=lambda t: -t.value):
        need = dict(task.needs or {})
        if not need:
            kept.append(task)
            continue
        bill = 0.0
        buy: dict = {}
        for item, count in need.items():
            if item.startswith("SEED:"):
                crop = item.split(":", 1)[1]
                # Stock only. A seed already in `supplies` was bought to cover the exact shortfall
                # of the task that ordered it and is spoken for, so it is neither added here (it is
                # not spare) nor subtracted (the `- supplies * 0` this replaced was a term that
                # could only ever have double-counted, disabled by a `* 0` and so dead either way).
                have = seeds.get(crop, 0)
                short = count - have
                if short > 0:
                    bill += CROPS[crop]["seed"] * short
                    buy[item] = short
                else:
                    seeds[crop] = have - count
            else:
                # Carried stock is drained before shed stock, and drained at all: it used to be
                # counted towards `have` and never decremented, so three fertilizer tasks all saw
                # the same one unit in a hand and none of them ordered a replacement.
                have = shed.get(item, 0) + carried.get(item, 0)
                short = count - have
                if short > 0:
                    bill += float(prices.get(item, 50) or 50) * short
                    buy[item] = short
                else:
                    from_hand = min(count, carried.get(item, 0))
                    carried[item] = carried.get(item, 0) - from_hand
                    shed[item] = max(0, shed.get(item, 0) - (count - from_hand))
        if bill > budget:
            dropped.append(task)
            continue
        budget -= bill
        for item, n in buy.items():
            supplies[item] = supplies.get(item, 0) + n
            # A purchase covers the *shortfall*, so whatever stock there was is consumed by this
            # task too. Leaving it in place (`setdefault(crop, 0)` / `shed[item] = shed[item]`, both
            # no-ops) let the next task see the same units again and order too little.
            if item.startswith("SEED:"):
                seeds[item.split(":", 1)[1]] = 0
            else:
                shed[item] = 0
                carried[item] = 0
        kept.append(task)
    return kept, supplies, dropped


# --------------------------------------------------------------------------- compilation

def compile_day(obs, plan, hands: int | None = None, turns: int = TURNS_PER_DAY,
                start_turn: int = 0, price_fn=None, max_rounds: int = MAX_PRUNE_ROUNDS,
                cash: float | None = None) -> CompiledDay:
    """Tasks -> hands -> route -> verify -> prune, until the day stands up.

    Returns the day the agent should actually play, plus the report that justifies it. When even a
    fully pruned day cannot keep everything alive, `overcommit` is set rather than silently
    accepted — an over-committed day is a fact about the *plan*, and the search needs to see it.
    """
    from agent.router import decide_hands

    day = int(obs.get("day", 0) or 0)
    # C2.4's walk-home DROP, on the one day it pays for itself. Every other day the goods reach the
    # shed for free at dusk and are sold at the next dawn, so the walk is pure cost — measured at
    # -$5k to -$9k a season when it ran daily. On the last day there is no next dawn: the framework
    # plays `episodeSteps - 1` turns, so day 29 stops at hour 22 and whatever the hands are holding
    # is never sold at all. `agent/router.py::_schedule_drops` carries the numbers.
    last_day = day >= LAST_DAY
    tasks = list(daily_tasks(obs, plan, day=day, turn=start_turn, price_fn=price_fn))
    farmer = tuple(obs["farms"][obs.get("player", 0)]["farmer"])

    live_hands = len(obs["farms"][obs.get("player", 0)]["hands"])
    n_hands = hands if hands is not None else max(
        live_hands, decide_hands(tasks, farmer, turns_left=turns, cash=cash))

    pruned: list = []
    working, supplies, unfunded = _afford(tasks, obs, cash)
    pruned.extend(unfunded)
    report = None
    crew = n_hands + 1
    # Route from where the units *are*, not from where a model says they spawn. The two agree at
    # dawn and diverge the moment anything is different — and a route computed from the wrong start
    # is wrong at every step after it, which shows up as units walking into the board edge: 184
    # blocked moves a game, none of which the verifier could see, because it was fed the same wrong
    # positions. Modelled spawns are only for hands that have been ordered but have not stood up.
    live = [tuple(obs["farms"][obs.get("player", 0)]["farmer"])] + \
           [tuple(h) for h in (obs["farms"][obs.get("player", 0)].get("hands") or [])]
    for round_no in range(max_rounds + 1):
        units = live + spawn_positions(n_hands, farmer)[len(live):] if len(live) < crew else live
        result = route(working, units, turns_left=turns, turn0=start_turn,
                       seeds=_seed_stock(obs, supplies), drop_home=last_day,
                       last_day=last_day)
        report = verify_day(obs, result.scripts, turns=turns, start_turn=start_turn, crew=crew,
                            supplies=supplies)
        if report.ok:
            return CompiledDay(scripts=result.scripts, hands=n_hands, report=report,
                               tasks=working, start_turn=start_turn, pruned=pruned,
                               rounds=round_no)

        droppable = [t for t in working if t.op == "PLANT"]
        if not droppable:
            break
        # Cheapest promises first: a planting not made costs a seed, while a plant that dies of
        # thirst costs everything it would have produced (E44's 50-plant seasons).
        droppable.sort(key=lambda t: t.value)
        cut = max(1, len(droppable) // 3)
        losing = set(id(t) for t in droppable[:cut])
        working = [t for t in working if id(t) not in losing]
        pruned.extend(droppable[:cut])

    # One escalation before giving up: more hands can rescue a day that pruning cannot, because
    # some of the load is animals and survival water rather than plantings.
    if report is not None and not report.ok and n_hands < 13:
        rescued = n_hands + 2
        units = live + spawn_positions(rescued, farmer)[len(live):] \
            if len(live) < rescued + 1 else live
        result = route(working, units, turns_left=turns, turn0=start_turn,
                       drop_home=last_day, last_day=last_day)
        rescue_report = verify_day(obs, result.scripts, turns=turns, start_turn=start_turn,
                                   crew=rescued + 1, supplies=supplies)
        if rescue_report.ok or len(rescue_report.deaths) < len(report.deaths):
            return CompiledDay(scripts=result.scripts, hands=rescued, report=rescue_report,
                               tasks=working, start_turn=start_turn, pruned=pruned,
                               rounds=max_rounds + 1, overcommit=not rescue_report.ok)

    units = live + spawn_positions(n_hands, farmer)[len(live):] if len(live) < crew else live
    result = route(working, units, turns_left=turns, turn0=start_turn,
                   drop_home=last_day, last_day=last_day)
    report = verify_day(obs, result.scripts, turns=turns, start_turn=start_turn, crew=crew,
                        supplies=supplies)
    return CompiledDay(scripts=result.scripts, hands=n_hands, report=report, tasks=working,
                       start_turn=start_turn, pruned=pruned, overcommit=True, rounds=max_rounds)
