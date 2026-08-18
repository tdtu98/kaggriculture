"""C2 — the router: whose day is it, and in what order.

`route(tasks, units, turns_left)` turns C1's valued task list into a per-turn script for every unit.
This is where PLAN_v4's central bet is won or lost: Boatlee walks **1.01 steps per useful action**
and our reactive engines walk 1.70–2.24 (E55), which is the same farm doing a third less work.

Three ideas carry that ratio:

**Visit a tile once and do everything there.** Tasks are grouped into *stops*, so a strawberry that
needs fertilizer, water and harvesting is one walk and three turns rather than three walks. This
alone is most of the gap: Boatlee issues 46% of its actions without having moved.

**Keep roles pure.** Keepers work the animal cluster and field hands work the crops (E46: Boatlee's
units are 93% role-pure, and specialisation replicated at ~70% winrate over 360 games). A unit that
alternates between pasture and field pays the distance twice.

**Order by rules, not by taste.** Within a stop the op order is forced by the environment:
FERTILIZE before WATER (a one-time crop reads `fertilized_until_day` *during* the WATER op), WATER
before HARVEST (harvesting a one-time crop clears the tile), and a PLANT drags a WATER after it
because a new plant starts at `consecutive_unwatered = 1` and dies at dusk otherwise.

What this module does *not* do is decide whether the day is over-committed — that is C3's job, which
runs the compiled day through the forward model and prunes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.tasks import LAST_TURN, SHED_TILES, Task

#: Op order inside one tile visit. Every constraint here is a rule, not a preference:
#: DIG clears the ground before a PLANT can use it; FERTILIZE must land before the WATER that
#: reads it; HARVEST goes last because on a one-time crop it removes the plant.
OP_ORDER = {"DIG": 0, "BUILD_PASTURE": 1, "BUILD_COOP": 1, "PLACE": 2, "PLANT": 3,
            "FERTILIZE": 4, "WATER": 5, "FEED": 6, "CARE": 7, "HARVEST": 8,
            "COLLECT_FERTILIZER": 9}

ANIMAL_OPS = frozenset({"FEED", "CARE", "COLLECT_FERTILIZER"})

#: Ops that put sellable goods in a unit's hands. These are the ones worth walking to the shed for
#: (C2.4): everything else a unit carries is feed or fertilizer, which banks itself at dusk.
PRODUCE_OPS = frozenset({"HARVEST", "COLLECT_FERTILIZER"})

#: Turns a keeper loses before it does anything useful: the PICKUP of the day's wheat, plus the
#: walk out to the cluster.
KEEPER_OVERHEAD = 3

#: `fib(n)` is what the n-th hire costs (1,1,2,3,5,8,13,21,34,55,89,144,233,377).
HIRE_COST = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)

MOVES = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}


@dataclass
class Stop:
    """One tile, and everything a unit would do while standing on it."""

    tile: tuple
    tasks: list
    value: float = 0.0
    deadline: int = LAST_TURN
    needs: dict = field(default_factory=dict)

    @property
    def n_ops(self) -> int:
        # A PLANT is two turns: the planting, then the water that keeps it alive to dusk.
        return len(self.tasks) + sum(1 for t in self.tasks if t.op == "PLANT")

    @property
    def is_animal(self) -> bool:
        return any(t.op in ANIMAL_OPS for t in self.tasks)


@dataclass
class UnitScript:
    unit: int
    ops: list
    stops: list
    value: float = 0.0
    moves: int = 0
    useful: int = 0


@dataclass
class RouteResult:
    scripts: dict
    unrouted: list
    steps_per_useful: float = 0.0
    value: float = 0.0
    #: Effect counter for C2.4 — how many units were sent to the shed to unload. Zero on a day
    #: where nothing was harvested is fine; zero on a harvest day is an unfinished implementation.
    drops_scheduled: int = 0
    #: C1 asked for a shed-pressure DROP this day (`agent/tasks.py::_logistics_tasks`).
    drop_requested: bool = False

    def actions_at(self, turn: int) -> dict:
        """`{unit: action}` for one turn — what C4 emits."""
        out = {}
        for unit, script in self.scripts.items():
            out[unit] = script.ops[turn] if turn < len(script.ops) else ["PASS"]
        return out


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def spawn_positions(n_hands: int, farmer=(4, 4), shed_tiles=SHED_TILES) -> list:
    """Where the units will actually be standing at hour 1.

    `_spawn_hand` puts each new hand on the least-occupied shed-access tile, in NWSE order — so
    hands spread across the four centre squares rather than stacking on the farmer. Modelled here
    because the router's distances start from these tiles, and because `decide_hands` has to
    evaluate the same board its answer will be used on: guessing a different spawn order made it
    approve a staffing level whose real route missed a survival water.
    """
    positions = [tuple(farmer)]
    for _ in range(n_hands):
        occupancy = {t: 0 for t in shed_tiles}
        for p in positions:
            if p in occupancy:
                occupancy[p] += 1
        positions.append(min(shed_tiles, key=lambda t: (occupancy[t], shed_tiles.index(t))))
    return positions


# --------------------------------------------------------------------------- stops

def build_stops(tasks) -> tuple[list, list]:
    """Group tile tasks into stops; return `(stops, logistics)`."""
    by_tile: dict = {}
    logistics = []
    for t in tasks:
        if t.tile is None:
            logistics.append(t)
            continue
        by_tile.setdefault(tuple(t.tile), []).append(t)

    stops = []
    for tile, group in by_tile.items():
        group.sort(key=lambda t: OP_ORDER.get(t.op, 99))
        needs: dict = {}
        for t in group:
            for item, n in (t.needs or {}).items():
                if not item.startswith("SEED:"):
                    needs[item] = needs.get(item, 0) + n
        stops.append(Stop(tile=tile, tasks=group,
                          value=sum(t.value for t in group),
                          deadline=min(t.deadline_turn for t in group),
                          needs=needs))
    return stops, logistics


# --------------------------------------------------------------------------- assignment

def split_roles(stops, n_units: int, turns: int = LAST_TURN + 1) -> tuple[list, list, int]:
    """Animal stops to keepers, the rest to field hands (E46: role purity is worth money).

    Keeper count comes from the actual work, not a rule of thumb. "One keeper per six animals"
    looked right and was not: six head at three ops each is 18 turns, and by the time the wheat is
    picked up and the cluster walked there is no room for the eighteenth — so two keepers silently
    dropped two to four animal ops every day of a measured season, which is a cow going unfed.
    """
    keeper_stops = [s for s in stops if s.is_animal]
    field_stops = [s for s in stops if not s.is_animal]
    if not keeper_stops:
        return [], field_stops, 0
    work = sum(s.n_ops + 1 for s in keeper_stops) + KEEPER_OVERHEAD
    n_keepers = min(n_units, max(1, -(-work // turns)))
    if not field_stops:
        n_keepers = n_units
    return keeper_stops, field_stops, n_keepers


#: Effect counters for the assignment (CLAUDE.md: prove the change fired before reading its score).
#: Module-level and cumulative, because `route()` is called many times per day and the interesting
#: question is "did the insertion assignment ever run in a real season", which no single call can
#: answer. `insertion_stops` is the number of stops placed by marginal-cost insertion,
#: `boundary_moves` the number the local-search pass then moved to a different unit, and
#: `sweep_stops` the number that still went through the unit-blind sweep (only reachable when the
#: caller does not know where its units are standing).
#:
#: The last-day drop budget adds three more (C2.4, E64's open regression). `drop_reserved` counts
#: routes compiled with the return leg reserved up front, `drop_yields` the stops that lost their
#: place to it, and `drop_short` the units that still ended holding produce with no room to unload —
#: the one that must stay near zero for the mechanism to be doing its job.
STATS = {"insertion_calls": 0, "insertion_stops": 0, "boundary_moves": 0,
         "sweep_calls": 0, "sweep_stops": 0,
         "drop_reserved": 0, "drop_yields": 0, "drop_short": 0}

#: Load balance, as a penalty that only bites near the end of the day. Pure cheapest-insertion
#: minimises *total* walking — which is exactly what the exactness oracle measures, and on a small
#: instance its optimum is genuinely lopsided, one unit touring a cluster while another stays put.
#: A real day is 24 turns long, though, and a unit filled to the brim drops the tail of its route
#: while a neighbour stands idle: measured, one survival water a day fell out that way and was
#: rescued by a *keeper*, breaking role purity (E46). So the penalty is zero below `SOFT_LOAD` of
#: the day and steep above it: the oracle instances never reach the knee, and a real day cannot pile
#: up behind one hand.
BALANCE_WEIGHT = 1.0
SOFT_LOAD = 0.6


def partition(stops, n_groups: int, starts=None, turns: int = LAST_TURN + 1) -> list:
    """Split stops into `n_groups` compact, time-balanced groups, one per unit.

    **The split is the routing decision, not the tour.** `order_stops`' 2-opt is within 5% of an
    exact tour, and the router as a whole was still walking 1.26-2.06x an exact multi-unit
    assignment (E61, measured on 40 4-unit x 8-tile instances against Held-Karp + a set-partition
    DP, mean 1.49x). The cause was here: the old sweep sorted by quadrant and row band and cut on
    cumulative *time*, **without ever looking at where the units were standing** — so a perfect tour
    ran on a split that had already handed a unit tiles across the board from it.

    `starts` closes that. Stops are placed one at a time into the unit whose route grows least by
    taking them (cheapest insertion on the open Manhattan tour from that unit's actual position),
    hardest first; each group is then re-toured with the same `order_stops` the day will really be
    compiled from, and a relocate pass moves boundary stops between units while that shortens the
    total walk. On the same 40 instances: **mean 1.015x, worst 1.105x**, and 1.71 ms for the 12
    units x 100 stops that C2 budgets 20 ms for.

    `starts=None` keeps the old sweep, for callers that genuinely do not know unit positions. It is
    kept rather than deleted because it is the deterministic fallback the tests for `decide_hands`
    lean on, but nothing in the live path uses it — `STATS["sweep_stops"]` says so.
    """
    if n_groups <= 0:
        return []
    if n_groups == 1:
        return [list(stops)]
    if starts is not None:
        return _insertion_assign(list(stops), list(starts)[:n_groups], turns)

    STATS["sweep_calls"] += 1
    STATS["sweep_stops"] += len(stops)
    ordered = sorted(stops, key=lambda s: (s.tile[1] >= 5, s.tile[0] >= 5, s.tile[1], s.tile[0]))
    total = sum(s.n_ops + 1 for s in ordered)
    target = total / n_groups
    groups: list = [[] for _ in range(n_groups)]
    load = 0.0
    g = 0
    for s in ordered:
        if g < n_groups - 1 and load >= target * (g + 1):
            g += 1
        groups[g].append(s)
        load += s.n_ops + 1
    return groups


def _insert_cost(tour, start, tile) -> tuple:
    """Cheapest place for `tile` in an *open* tour that begins at `start`: `(delta, position)`.

    Open, not closed: a unit does not walk back to where it spawned, so appending to the end costs
    only the last leg. That asymmetry is why the classic closed-tour insertion formula would be
    wrong here — it would price the tail of every route as if it had to come home.
    """
    x, y = tile
    if not tour:
        return abs(start[0] - x) + abs(start[1] - y), 0
    # The distance is inlined rather than calling `manhattan`: this is the router's inner loop by
    # two orders of magnitude (800k calls to compile one day), and the call overhead alone was
    # most of the compile budget.
    ax, ay = start
    bx, by = tour[0]
    best_delta = (abs(ax - x) + abs(ay - y) + abs(x - bx) + abs(y - by)
                  - abs(ax - bx) - abs(ay - by))
    best_pos = 0
    ax, ay = bx, by
    for i in range(1, len(tour)):
        bx, by = tour[i]
        delta = (abs(ax - x) + abs(ay - y) + abs(x - bx) + abs(y - by)
                 - abs(ax - bx) - abs(ay - by))
        if delta < best_delta:
            best_delta, best_pos = delta, i
        ax, ay = bx, by
    tail = abs(ax - x) + abs(ay - y)                  # `ax, ay` is now the last tile
    if tail < best_delta:
        best_delta, best_pos = tail, len(tour)
    return best_delta, best_pos


def _insertion_assign(stops, starts, turns: int) -> list:
    """Marginal-cost insertion from the units' real positions, then a boundary-repair pass.

    Two seed orders are tried and the shorter total walk kept. Greedy insertion commits early, so
    the order the stops arrive in is a real degree of freedom — farthest-first is the better default
    (a remote stop has few good homes and placing it late drags a unit across the board for a
    leftover) but not always, and a second pass costs a fraction of a millisecond.
    """
    n = len(starts)
    if n <= 1:
        return [list(stops)] + [[] for _ in range(n - 1)]
    STATS["insertion_calls"] += 1
    STATS["insertion_stops"] += len(stops)

    far = sorted(stops, key=lambda s: (-min(manhattan(p, s.tile) for p in starts), s.tile))
    best_groups, best_walk = None, None
    for ordered in (far, far[::-1]):
        groups, total = _insertion_pass(ordered, starts, turns)
        if best_walk is None or total < best_walk:
            best_groups, best_walk = groups, total
    return best_groups


def _insertion_pass(ordered, starts, turns: int) -> tuple:
    """One insertion + repair pass over `ordered`; returns `(groups, total walk)`."""
    n = len(starts)
    tours: list = [[] for _ in range(n)]          # tiles, in insertion order
    groups: list = [[] for _ in range(n)]         # the stops themselves, same order
    walk = [0] * n                                # tour length so far
    load = [0] * n                                # walk + ops: what the day actually costs

    for stop in ordered:
        work = stop.n_ops
        best_key = None
        best = None
        knee = SOFT_LOAD * turns
        for u in range(n):
            delta, pos = _insert_cost(tours[u], starts[u], stop.tile)
            projected = load[u] + delta + work
            key = (projected > turns,
                   delta + BALANCE_WEIGHT * max(0.0, projected - knee), load[u], u)
            if best_key is None or key < best_key:
                best_key, best = key, (u, pos, delta)
        u, pos, delta = best
        tours[u].insert(pos, stop.tile)
        groups[u].insert(pos, stop)
        walk[u] += delta
        load[u] += delta + work

    _repair_boundaries(tours, groups, starts, walk, load, turns)
    # Insertion order is not the order the unit will actually walk — `order_stops` re-tours every
    # group with the same NN + 2-opt the day is compiled from. Re-tour here too and repair again, so
    # the second pass prices a relocation against the route that will really be walked rather than
    # against the artefact of the insertion sequence.
    for u in range(n):
        groups[u] = order_stops(groups[u], starts[u], turns)
        tours[u] = [s.tile for s in groups[u]]
        retoured = _tour_cost(tours[u], starts[u])
        load[u] += retoured - walk[u]
        walk[u] = retoured
    _repair_boundaries(tours, groups, starts, walk, load, turns)
    return groups, sum(_tour_cost(t, starts[u]) for u, t in enumerate(tours))


def _tour_cost(tiles, start) -> int:
    total = 0
    ax, ay = start
    for bx, by in tiles:
        total += abs(ax - bx) + abs(ay - by)
        ax, ay = bx, by
    return total


def _repair_boundaries(tours, groups, starts, walk, load, turns: int, rounds: int = 1) -> None:
    """Move a stop to another unit while that shortens the total walk.

    Insertion is greedy and commits early, so the stops it gets wrong are the ones on the seam
    between two units — cheap to find (only relocations that *reduce* the sum are considered) and
    the difference between "close to optimal" and "close to the sweep". One round by default: on 40
    measured instances a second and a fourth round changed no assignment at all, and this runs
    fourteen times a day inside `decide_hands`.

    Far units are skipped by a bounding-box bound rather than costed. Inserting a tile between two
    points costs, per axis, twice the distance by which it lies outside their interval — so the L1
    distance from the tile to a unit's bounding box (its start included) is a genuine lower bound on
    what that unit would pay, and a candidate that cannot beat `gain_out` never has its tour walked.
    """
    n = len(starts)
    for _ in range(rounds):
        improved = False
        boxes = [_bounding_box(tours[u], starts[u]) for u in range(n)]
        for src in range(n):
            i = 0
            while i < len(groups[src]):
                stop = groups[src][i]
                tile = stop.tile
                without = _tour_cost(tours[src][:i] + tours[src][i + 1:], starts[src])
                gain_out = walk[src] - without
                work = stop.n_ops
                best = None
                for dst in range(n):
                    if dst == src:
                        continue
                    if _box_distance(boxes[dst], tile) >= gain_out:
                        continue                      # cannot beat the gain, whatever the tour
                    delta, pos = _insert_cost(tours[dst], starts[dst], tile)
                    if delta >= gain_out:
                        continue                      # not shorter overall; nothing to repair
                    if load[dst] + delta + work > turns:
                        continue                      # would not fit in the receiving unit's day
                    if best is None or delta < best[0]:
                        best = (delta, dst, pos)
                if best is None:
                    i += 1
                    continue
                delta, dst, pos = best
                del tours[src][i]
                del groups[src][i]
                walk[src] = without
                load[src] -= gain_out + work
                tours[dst].insert(pos, tile)
                groups[dst].insert(pos, stop)
                walk[dst] += delta
                load[dst] += delta + work
                # Widen the receiver's box; the donor's is left as it was, which can only cost a
                # prune that would have been safe — never allow a move that is not an improvement.
                x0, y0, x1, y1 = boxes[dst]
                boxes[dst] = (min(x0, tile[0]), min(y0, tile[1]),
                              max(x1, tile[0]), max(y1, tile[1]))
                STATS["boundary_moves"] += 1
                improved = True
        if not improved:
            break


def _bounding_box(tiles, start) -> tuple:
    xs = [start[0]] + [t[0] for t in tiles]
    ys = [start[1]] + [t[1] for t in tiles]
    return min(xs), min(ys), max(xs), max(ys)


def _box_distance(box, tile) -> int:
    x0, y0, x1, y1 = box
    x, y = tile
    return max(0, x0 - x, x - x1) + max(0, y0 - y, y - y1)


# --------------------------------------------------------------------------- ordering

def order_stops(stops, start, turns: int) -> list:
    """Nearest-neighbour from `start`, then 2-opt on the visit order.

    Deadlines go first-class: a stop that must happen by turn 9 is visited before the walk is
    optimised, because a shorter route that misses a decay window is not shorter, it is cheaper and
    wrong.
    """
    if not stops:
        return []
    urgent = [s for s in stops if s.deadline < LAST_TURN]
    normal = [s for s in stops if s.deadline >= LAST_TURN]

    route = []
    pos = start
    for pool in (sorted(urgent, key=lambda s: s.deadline), normal):
        remaining = list(pool)
        while remaining:
            nxt = min(remaining, key=lambda s: (manhattan(pos, s.tile), -s.value))
            remaining.remove(nxt)
            route.append(nxt)
            pos = nxt.tile
    return _two_opt(route, start) if len(route) > 3 else route


def _two_opt(route, start, rounds: int = 3) -> list:
    """Classic 2-opt on the visit order, bounded so it stays inside the turn budget.

    Only reorders stops that share a deadline class, so tightening the walk can never push a
    deadline stop behind a relaxed one.
    """
    best = list(route)
    n = len(best)
    for _ in range(rounds):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a = best[i - 1].tile if i else start
                if best[i].deadline != best[j].deadline:
                    continue
                before = manhattan(a, best[i].tile) + manhattan(best[j].tile,
                                                                best[j + 1].tile) if j + 1 < n \
                    else manhattan(a, best[i].tile)
                after = manhattan(a, best[j].tile) + manhattan(best[i].tile,
                                                              best[j + 1].tile) if j + 1 < n \
                    else manhattan(a, best[j].tile)
                if after < before:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
        if not improved:
            break
    return best


# --------------------------------------------------------------------------- expansion

def _home_cost(tile, shed_tiles=SHED_TILES) -> int:
    """Turns to walk from `tile` to the nearest centre tile and DROP there."""
    return min(manhattan(tile, t) for t in shed_tiles) + 1


def expand(stops, start, turns: int, turn0: int, carrying: dict, shed_tiles=SHED_TILES,
           reserve_drop: bool = False):
    """Walk the route turn by turn, emitting the actual unit actions.

    Returns `(ops, visited, dropped)`. A stop that cannot be reached in time — or whose deadline has
    passed by the time the unit arrives — is dropped rather than half-done, so the caller can hand
    it to somebody else or let C3 prune the plan that created it.

    `reserve_drop` is the last day, and it changes what "in time" means. Every other day a unit may
    fill its route to the last turn: dusk banks whatever is in its hands and C4 sells it at the next
    dawn. On day 29 there is no next dawn, so produce still in a hand at the buzzer is worth **$0**
    (E61) — and once E64 filled the days properly there was no leftover turn at the end for the
    walk-home `DROP` to be appended to, which is exactly how E61's +$8.6k eroded. So the return leg
    is budgeted *here*, before the route is committed: from the moment a stop would put produce in
    the unit's hands, every stop must leave room to reach a centre tile and unload. A stop that does
    not fit with the leg reserved yields to it, which is the right trade by construction — a harvest
    that cannot be delivered pays nothing, and any other task on the last day is worth less than the
    inventory it would strand.
    """
    ops: list = []
    pos = start
    visited: list = []
    dropped: list = []
    inventory = dict(carrying)

    # Load first, while standing on the shed at dawn: every unit spawns on a centre tile, so a
    # PICKUP here costs one turn and no walking.
    need: dict = {}
    for s in stops:
        for item, n in s.needs.items():
            need[item] = need.get(item, 0) + n
    if pos in shed_tiles:
        for item in sorted(need):
            short = need[item] - inventory.get(item, 0)
            if short > 0 and len(ops) < turns:
                ops.append(["PICKUP", item, short])
                inventory[item] = inventory.get(item, 0) + short

    # Deliberately starts False even when the unit already carries produce: `_schedule_drops` only
    # sends a unit home if its route contains a PRODUCE op, so reserving for a hand that will never
    # trigger the drop would spend the turns and buy nothing.
    holding = False
    for stop in stops:
        path = _steps(pos, stop.tile)
        # On the last day, keep back the turns the unit needs to get its produce home. The reserve
        # switches on at the first stop that would put produce in a hand, and stays on.
        reserve = 0
        if reserve_drop and (holding or any(t.op in PRODUCE_OPS for t in stop.tasks)):
            reserve = _home_cost(stop.tile, shed_tiles)
        # Deadlines order the day; they do not veto it. A harvest past `max_lifespan_step` is
        # decaying at a unit per two steps, so arriving late is worth less — not worth nothing —
        # and treating it as impossible let one expired harvest drag the survival water sharing
        # its tile into the bin, killing the plant to save a turn.
        if len(ops) + len(path) + stop.n_ops + reserve > turns:
            dropped.append(stop)
            if reserve and len(ops) + len(path) + stop.n_ops <= turns:
                STATS["drop_yields"] += 1
            continue
        if reserve:
            holding = True
        ops.extend(path)
        pos = stop.tile
        for task in stop.tasks:
            if task.expires_turn is not None and turn0 + len(ops) > task.expires_turn:
                # The tile will be a weed by the time we get here; spending the turn is worse than
                # doing nothing, because the turn is what the next tile needed.
                continue
            if task.op in ("FEED", "FERTILIZE", "PLACE"):
                item = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}.get(
                    task.op, task.args[0] if task.args else "")
                if inventory.get(item, 0) <= 0:
                    continue                       # nothing carried: the op would be a no-op
                inventory[item] -= 1
            ops.append([task.op, *task.args] if task.args else [task.op])
            if task.op == "PLANT" and len(ops) < turns:
                # A new plant is created at `consecutive_unwatered = 1`; unwatered today it is a
                # weed by dusk. The water is part of planting, not a separate decision.
                ops.append(["WATER"])
        visited.append(stop)

    return ops[:turns], visited, dropped


def _steps(a, b) -> list:
    """Manhattan path as unit moves; vertical first, which keeps columns tidy on this board."""
    out = []
    x, y = a
    while y != b[1]:
        step = 1 if b[1] > y else -1
        out.append([MOVES[(0, step)]])
        y += step
    while x != b[0]:
        step = 1 if b[0] > x else -1
        out.append([MOVES[(step, 0)]])
        x += step
    return out


# --------------------------------------------------------------------------- the router

def route(tasks, units, turns_left: int = LAST_TURN + 1, turn0: int = 0,
          carrying=None, shed_tiles=SHED_TILES, seeds=None,
          drop_home: bool = True, last_day: bool = False) -> RouteResult:
    """Compile a day: `{unit index: [action per turn]}`.

    `units` is a list of positions, unit 0 being the farmer. Everything else follows from the tasks.
    """
    carrying = carrying or {}
    stops, logistics = build_stops(tasks)
    if not units:
        return RouteResult(scripts={}, unrouted=stops)

    # The last day is one turn shorter than every other, and the router had not been told. The
    # framework plays `episodeSteps - 1` turns, so day 29 stops after **hour 22**: an op compiled
    # into hour 23 is never issued. Routes were being filled to `turns_left` regardless, so on the
    # day it matters most the tail op — and any `DROP` appended to it — was written into a turn that
    # does not exist. That is most of the gap between `drops_scheduled` (counted at compile) and
    # `hand_drops` (counted when the action is actually emitted).
    horizon = max(0, min(turns_left, LAST_TURN - turn0)) if last_day else turns_left

    keeper_stops, field_stops, n_keepers = split_roles(stops, len(units))
    keeper_units = list(range(n_keepers))
    field_units = list(range(n_keepers, len(units)))
    if not field_units and field_stops:
        # Nobody but keepers: they take the field work too rather than leaving the day empty.
        field_units = keeper_units

    assignment: dict = {u: [] for u in range(len(units))}
    for pool, group_units in ((keeper_stops, keeper_units), (field_stops, field_units)):
        if not group_units or not pool:
            continue
        # The unit positions go in with the work: the split is what decides the walk (E61), and a
        # split made without them hands a unit tiles across the board from where it stands.
        starts = [tuple(units[u]) for u in group_units]
        for chunk, unit in zip(partition(pool, len(group_units), starts, horizon), group_units):
            assignment[unit].extend(chunk)

    scripts: dict = {}
    unrouted: list = []
    for unit, unit_stops in assignment.items():
        start = tuple(units[unit])
        ordered = order_stops(unit_stops, start, horizon)
        ops, visited, dropped = expand(ordered, start, horizon, turn0,
                                       dict(carrying.get(unit, {})), shed_tiles,
                                       reserve_drop=last_day)
        unrouted.extend(dropped)
        moves = sum(1 for op in ops if op[0] in MOVES.values())
        useful = sum(1 for op in ops if op[0] not in MOVES.values() and op[0] != "PASS")
        scripts[unit] = UnitScript(unit=unit, ops=ops, stops=visited,
                                   value=sum(s.value for s in visited),
                                   moves=moves, useful=useful)

    # The rescue and the stagger both lengthen routes, and on the last day they would lengthen them
    # into the turns `expand` just held back for the walk home. `reserve` is what each unit still
    # owes the shed; both passes are given the budget net of it.
    reserve = _drop_reserve(scripts, units, shed_tiles) if last_day else {}
    if reserve and any(reserve.values()):
        STATS["drop_reserved"] += sum(1 for v in reserve.values() if v)
    unrouted = _rescue_survival(scripts, unrouted, units, horizon, reserve)
    if seeds is not None:
        # Only when the caller actually knows the seed stock. Absent means unknown, which must mean
        # unconstrained — reading it as "no seeds" staggers every planting out of the day.
        _stagger_plants(scripts, dict(seeds), horizon, reserve)

    # Last, because it must survive `_stagger_plants`' truncation and because it appends to the
    # tail of a route that both of the passes above may still lengthen.
    drops = _schedule_drops(scripts, units, horizon, carrying, logistics, shed_tiles,
                            turn0, budgeted=bool(last_day)) if drop_home else 0

    total_moves = sum(s.moves for s in scripts.values())
    total_useful = sum(s.useful for s in scripts.values())
    return RouteResult(
        scripts=scripts,
        unrouted=unrouted,
        steps_per_useful=(total_moves / total_useful) if total_useful else 0.0,
        value=sum(s.value for s in scripts.values()),
        drops_scheduled=drops,
        drop_requested=any(t.op == "DROP" for t in logistics),
    )


def _projected_inventory(ops, started: dict) -> dict:
    """What the unit is still holding when its route ends.

    Cheaper and more honest than "did it harvest anything": a keeper that picks up twelve wheat and
    feeds twelve animals ends the day empty, and sending it to the shed to DROP nothing spends a
    turn and books a `blocked_op`.
    """
    inv = {k: int(v) for k, v in (started or {}).items() if int(v) > 0}
    for o in ops:
        name = o[0]
        if name == "PICKUP" and len(o) >= 3:
            inv[o[1]] = inv.get(o[1], 0) + int(o[2])
        elif name == "DROP":
            inv = {}
        elif name == "HARVEST":
            inv["HARVESTED"] = inv.get("HARVESTED", 0) + 1
        elif name == "COLLECT_FERTILIZER":
            inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
        elif name == "FEED":
            inv["WHEAT"] = inv.get("WHEAT", 0) - 1
        elif name == "FERTILIZE":
            inv["FERTILIZER"] = inv.get("FERTILIZER", 0) - 1
        elif name == "PLACE" and len(o) >= 2:
            inv[o[1]] = inv.get(o[1], 0) - 1
    return {k: v for k, v in inv.items() if v > 0}


def _drop_reserve(scripts: dict, units, shed_tiles=SHED_TILES) -> dict:
    """`{unit: turns owed to the walk home}` for the routes that will end holding produce.

    Read off the compiled route rather than re-derived from the tasks, for E39's reason: a rule
    re-implemented in a second place is a rule that will disagree with itself. The predicate is
    exactly `_schedule_drops`', so the turns held back are the turns that will be spent.
    """
    out: dict = {}
    for unit, script in scripts.items():
        if not any(o[0] in PRODUCE_OPS for o in script.ops):
            out[unit] = 0
            continue
        pos = script.stops[-1].tile if script.stops else tuple(units[unit])
        out[unit] = _home_cost(pos, shed_tiles)
    return out


def _schedule_drops(scripts: dict, units, turns_left: int, carrying: dict, logistics,
                    shed_tiles=SHED_TILES, turn0: int = 0, budgeted: bool = False) -> int:
    """C2.4 — walk to a centre tile and DROP whatever the day put in the unit's hands.

    **Only worth doing on the last day, and that is a measurement, not a design taste.** Every other
    day the goods reach the shed for free: dusk empties every inventory into it
    (`_drop_inventories_to_shed`) and C4 sells the shed at the next dawn. The walk home buys nothing
    there and it is not free — it is turns, and the turns come out of the same day's farm work.
    Measured vs `starter`, 80 games each on two blocks (40000:40040 / 41000:41040):

        every day   $41,730 / $38,397   milk/cow-day 0.23   thirst 3.4   steps/useful 1.15
        last day    $54,207 / $51,685   milk/cow-day 0.48   thirst 1.7   steps/useful 1.05
        never       $46,648 / $43,876   milk/cow-day 0.48   thirst 1.4   steps/useful 1.04

    So the caller decides, through `route(drop_home=...)`, and `compile_day` sets it on the last day
    of the season and nowhere else. What makes that day different is that there is no next dawn: the
    framework plays `episodeSteps - 1` turns, so day 29 ends at **hour 22**, the dusk drop is never
    followed by a market turn, and everything in the hands at that moment is worth nothing.

    `logistics` is C1's shed-pressure DROP (the list that used to be bound to `_logistics` and
    thrown away). It does not change *whether* a carrying unit unloads, but it is reported through
    `RouteResult.drop_requested` so a C1 request the day had no room for is visible instead of
    silent. Measured caveat so nobody builds on it: at hour 1 the shed has just been sold and every
    inventory was emptied at dusk, so C1's `projected > SHED_PRESSURE` test is almost never true —
    C4's `release_pressure` valve is what actually keeps shed overflow near zero.
    """
    # An op past the day's last hour is never emitted: C4 indexes scripts from `turn0` and the day
    # ends at `LAST_TURN`. Appending a DROP at `turns_left` when `turn0 > 0` looks scheduled and
    # never happens.
    usable = min(turns_left, LAST_TURN + 1 - turn0)
    scheduled = 0
    for unit, script in scripts.items():
        if not any(o[0] in PRODUCE_OPS for o in script.ops):
            # Only produce is worth the walk. Leftover wheat or fertilizer banks at dusk for free
            # and is not what overflows the shed, so fetching it home buys nothing but steps.
            continue
        if not _projected_inventory(script.ops, carrying.get(unit, {})):
            continue
        pos = script.stops[-1].tile if script.stops else tuple(units[unit])
        target = min(shed_tiles, key=lambda t: (manhattan(pos, t), t))
        path = _steps(pos, target)
        if len(script.ops) + len(path) + 1 > usable:
            # No room. Only counted when the leg was budgeted during routing (`budgeted`, i.e. the
            # last day): everywhere else — `decide_hands` probes above all — a full route with no
            # room to unload is the expected case and the counter would be noise. On the last day a
            # non-zero `drop_short` is the mechanism failing, and it is the counter to watch (E64).
            if budgeted:
                STATS["drop_short"] += 1
            continue
        script.ops.extend(path)
        script.ops.append(["DROP"])
        script.moves += len(path)
        script.useful += 1
        scheduled += 1
    return scheduled


def _stagger_plants(scripts: dict, seeds: dict, turns_left: int, reserve: dict | None = None) -> None:
    """Never let two units plant the same crop in one turn with one seed in the bag.

    `_apply_unit_action`'s PLANT validation is **atomic per crop per turn**: if the requests for a
    crop exceed the seeds held, *every* request for that crop is dropped, not just the excess. Two
    hands reaching for the same wheat seed therefore plant nothing at all — and each wastes the
    WATER the router chained behind it. Measured on a real season: 39 dead plantings and 37 orphaned
    waters, three quarters of all blocked ops.

    The excess is delayed rather than cancelled: a PASS shifts that unit's remaining script by a
    turn, so the planting still happens, one turn later, with a seed to itself.
    """
    if not scripts:
        return
    stock = {crop: int(n) for crop, n in seeds.items()}
    for turn in range(turns_left):
        wanting: dict = {}
        for unit, script in scripts.items():
            if turn < len(script.ops) and script.ops[turn][0] == "PLANT":
                crop = script.ops[turn][1] if len(script.ops[turn]) > 1 else ""
                wanting.setdefault(crop, []).append(unit)
        for crop, units_wanting in wanting.items():
            allowed = stock.get(crop, 0)
            for unit in units_wanting[allowed:]:
                script = scripts[unit]
                script.ops.insert(turn, ["PASS"])
                # The tail is cut at the unit's own budget: on the last day that is the day minus
                # the turns it owes the walk home, so a stagger cannot push the DROP off the end.
                del script.ops[turns_left - (reserve or {}).get(unit, 0):]
            stock[crop] = max(0, allowed - min(allowed, len(units_wanting)))


def _rescue_survival(scripts: dict, unrouted: list, units, turns_left: int,
                     reserve: dict | None = None) -> list:
    """Offer dropped survival work to any unit that still has turns.

    Assignment is done per unit and in one pass, so a stop can be dropped by a busy unit while its
    neighbour finishes the day idle. That is not a rounding error: it was worth exactly one plant a
    season, on day 16, on all twenty seeds tested — a single strawberry left to die within reach of
    four units with hours to spare. Cheap to fix here and impossible to see from inside one unit's
    route.
    """
    survival = [s for s in unrouted if any(t.kind == "survival" for t in s.tasks)]
    if not survival:
        return unrouted

    rescued = set()
    for stop in sorted(survival, key=lambda s: -s.value):
        best, best_cost, best_path = None, None, None
        for unit, script in scripts.items():
            pos = script.stops[-1].tile if script.stops else tuple(units[unit])
            path = _steps(pos, stop.tile)
            cost = len(path) + stop.n_ops
            # A rescue that lands the unit somewhere new also moves the walk home, so the budget is
            # the day minus the *new* return leg rather than the one priced during expansion.
            owed = _home_cost(stop.tile) if (reserve or {}).get(unit) else 0
            if len(script.ops) + cost + owed > turns_left:
                continue
            if best_cost is None or cost < best_cost:
                best, best_cost, best_path = unit, cost, path
        if best is None:
            continue
        script = scripts[best]
        script.ops.extend(best_path)
        for task in stop.tasks:
            script.ops.append([task.op, *task.args] if task.args else [task.op])
        script.stops.append(stop)
        script.moves += len(best_path)
        script.useful += len(stop.tasks)
        script.value += stop.value
        rescued.add(id(stop))

    return [s for s in unrouted if id(s) not in rescued]


# --------------------------------------------------------------------------- hiring

def decide_hands(tasks, farmer=(4, 4), turns_left: int = LAST_TURN + 1,
                 max_hands: int = 13, cash: float | None = None) -> int:
    """How many hands to hire, by marginal value against `fib(n)`.

    Hire while the work the next hand would actually route is worth more than its wage — and always
    hire enough to cover the survival waters, because a plant lost to thirst costs its whole
    remaining life, not one turn.
    """
    base = route(tasks, spawn_positions(0, farmer), turns_left=turns_left)
    best, best_value = 0, base.value
    missing = _misses_survival(base)

    for n in range(1, max_hands + 1):
        if cash is not None and sum(HIRE_COST[:n]) > cash:
            break
        result = route(tasks, spawn_positions(n, farmer), turns_left=turns_left)
        gain = result.value - best_value
        # Stop when the next hand does not earn its wage — but only once the staffing we are
        # actually returning covers every survival water. Judging that on the *rejected* level
        # instead of the accepted one is not a rounding error: it returns a day that lets plants
        # die, while reporting that survival was checked.
        if gain <= HIRE_COST[n - 1] and not missing:
            break
        best, best_value = n, result.value
        missing = _misses_survival(result)
    return best


def _misses_survival(result: RouteResult) -> bool:
    return any(t.kind == "survival" for s in result.unrouted for t in s.tasks)
