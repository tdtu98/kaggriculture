"""Scripted economic engine (T1.1-T1.4, seed version).

Structure: each turn, generate the task list from farm state, assign tasks to units by distance
with deadline priority, and emit one op per unit. Market orders are decided separately.

Deliberately *not* a per-turn reflex agent — the task/assignment split is what T3 will keep when
the model replaces the task *selection* while the executor keeps doing the routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The reference environment's own pricing function, not a copy: it is exact by construction
# (including Python's banker's rounding), it honours `marketParams` overrides, and it exists on
# the Kaggle runner — where `kagsim` does not.
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

from .params import Params

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
              "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
PRODUCT_IDX = {p: i for i, p in enumerate(BASE_PRICE)}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
# Which structure each species needs. Every animal yields 1 fertilizer/day regardless of species
# and regardless of feeding (`kaggriculture.py:809`), so the species choice is really a choice of
# secondary product on top of a common fertilizer stream.
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
BUILD_OP = {"COOP": "BUILD_COOP", "PASTURE": "BUILD_PASTURE"}
# Livestock work happens on pastures and coops; crop work happens on the fields. They are in
# different places, so a unit that alternates between them pays the walk every time it switches.
ANIMAL_OPS = frozenset({"FEED", "CARE", "COLLECT_FERTILIZER", "PLACE"})
# Shed logistics rather than farm work. Always available at the shed, so any "stay put" rule that
# accepts them turns into camping.
HAUL_OPS = frozenset({"PICKUP", "DROP"})


_INF = float("inf")


def _optimal_assignment(cost: list[list[float]], n: int, m: int) -> dict[int, int]:
    """Min-cost assignment, Jonker-Volgenant style. Pure Python: no numpy on the Kaggle runner.

    Greedy nearest-task assignment measures **9.6% worse than optimal** in total walk distance over
    five seeds, and walking is 56% of our unit-turns (E39). This is exactly solvable at our
    size, so there is no reason to search for it: 12 units x 30 tasks solves in 66 us, against a
    1000 ms turn budget. Verified against `scipy.optimize.linear_sum_assignment` on 300 random
    matrices.
    """
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return {p[j] - 1: j - 1 for j in range(1, m + 1) if p[j]}


@dataclass
class Task:
    x: int
    y: int
    op: str
    arg: str | None
    prio: int          # lower runs first
    needs: str | None = None   # item the unit must already be carrying
    n: int | None = None       # quantity, for PICKUP


def _harvest_day(crop: str, params: Params) -> int:
    c = CROPS[crop]
    if params.harvest_early:
        return c["first_yield_day"]
    if crop == "MELON":
        return 10   # yield caps at 6 by age 10; ages 11-12 add nothing
    return c["max_yield_day"]


def _needs_water(tile: dict, day: int, eager_ongoing: bool = False) -> bool:
    """Water to survive (death is at 2 consecutive misses) or to earn the in-window bonus.

    `eager_ongoing` waters ongoing crops every day rather than only when they are already one miss
    from death. The default (False) is the original behaviour and keeps every ongoing plant on a
    knife edge all season: it is watered only at `consecutive_unwatered >= 1`, so a single turn
    where the WATER task loses out to the ~42 daily animal chores kills it. Measured: a
    strawberry-only farm plants 41, loses 23 to weeds, and ends the season with **zero** strawberry
    -- the plants die around days 5 and 11, and strawberry first yields on day 10 (E43).
    """
    if tile["watered_today"]:
        return False
    if tile["consecutive_unwatered"] >= 1:
        return True
    c = CROPS[tile["crop"]]
    if c["ongoing"]:
        return eager_ongoing
    age = day - tile["planted_day"]
    return (c["max_yield_day"] + 1) // 2 <= age <= c["max_yield_day"]


def _wants_fertilizer(tile: dict, day: int) -> bool:
    """Whether applying FERTILIZER to this tile today would actually buy anything.

    Fertilizer doubles yield accrual for 3 days (`day`..`day+2`) per unit applied, but by two
    different routes, and outside their windows it is worth exactly nothing:

    * one-shot crops (WHEAT, CARROT, MELON) accrue on the WATER op itself, and only inside the
      window `[(max_yield_day+1)//2, max_yield_day]` -- `bonus = 2 if fertilized else 1`
      (`kaggriculture.py:378-388`).
    * ongoing crops (TOMATO, STRAWBERRY) accrue at the daily refresh, `+2 if fertilized else +1`,
      and the bonus additionally requires the tile to have been watered that day
      (`kaggriculture.py:776-778`).

    Either way it is pointless once `yield_units` has reached `max_yield`.
    """
    c = CROPS[tile["crop"]]
    if tile["yield_units"] >= c["max_yield"]:
        return False
    age = day - tile["planted_day"]
    if c["ongoing"]:
        return age >= c["first_yield_day"] - 1
    return (c["max_yield_day"] + 1) // 2 <= age <= c["max_yield_day"]


class Engine:
    """Stateful agent. One instance per player per episode."""

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self._day = -1
        # Sticky per-unit assignments. Without these the greedy nearest-task choice is recomputed
        # every turn, so a unit walking toward A flips to B as soon as B becomes marginally
        # closer, and can oscillate forever without arriving anywhere.
        self._assigned: dict[int, tuple[int, int, str, str | None, int | None]] = {}
        self._demand: dict[str, int] | None = None
        self._last_plan: dict[int, int] = {}
        self._n_animals = 0
        self._n_plants = 0
        self._deferred: dict[int, tuple] = {}
        self._role_cache: list[str] = []
        self._role_key: tuple | None = None

    def _in_flight(self, item: str, p: Params) -> int:
        """Units of `item` already claimed by a PICKUP a unit is walking to.

        Fetch need was recomputed every turn from what units are *carrying*, which is zero for the
        whole walk to the shed. So while one unit was en route the others were told the item still
        had not been fetched, and each was sent for its own load. Measured cost: 1,359 wheat
        collected to service 290 FEED ops, 484 PICKUP turns against boatlee's 135, and 214 DROPs to
        shed the surplus -- 35% of our productive turns spent hauling, against their 7% (E29).

        The waste is real and measured. This particular correction is **not** an improvement:
        enabling it scored $68,206 against the champion's $78,032 and pushed steps-per-action from
        1.89 to 2.15. Kept behind a flag, off by default, because the diagnosis stands even though
        the fix does not.
        """
        if not p.fetch_in_flight:
            return 0
        return sum(
            (a[4] or 0) for a in self._assigned.values() if a[2] == "PICKUP" and a[3] == item
        )

    # ------------------------------------------------------------------ tasks

    def build_tasks(self, farm: dict, priv: dict, day: int) -> list[Task]:
        p = self.p
        tiles = farm["tiles"]
        tasks: list[Task] = []
        n_animals = 0
        n_plants = 0
        empty: list[tuple[int, int]] = []
        empty_coops_by_kind: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}

        feed_today = True
        if p.feed_alternate:
            # A newly placed animal survives day 0 unfed but must eat on day 1, so feed on odd days.
            feed_today = day % 2 == 1

        for y, row in enumerate(tiles):
            for x, t in enumerate(row):
                if t == "LOCKED":
                    continue
                if t is None:
                    empty.append((x, y))
                    continue
                kind = t.get("kind")
                if kind == "WEED":
                    tasks.append(Task(x, y, "DIG", None, 6))
                elif kind == "PLANT":
                    n_plants += 1
                    age = day - t["planted_day"]
                    c = CROPS[t["crop"]]
                    ready = t["yield_units"] > 0 and age >= c["first_yield_day"]
                    # Harvest and water are NOT alternatives. This was an `elif`, so a tile with
                    # yield pending stopped emitting WATER until someone harvested it -- and two
                    # consecutive unwatered days turn the tile into a WEED, destroying the plant.
                    #
                    # It only bites `ongoing` crops, because they are the ones that sit with
                    # `yield_units > 0` for days at a time: TOMATO and STRAWBERRY. The one-shot
                    # crops gate harvest on `age >= _harvest_day(...)`, so they keep being watered
                    # right up to the end and never noticed. That is why the champion is
                    # melon-based and sold 8 strawberries in a season while a real opponent sold
                    # 286 -- and why CEM learned a melon mix despite MELON having zero shop demand
                    # and STRAWBERRY four (D17).
                    harvestable = ready and (c["ongoing"] or age >= _harvest_day(t["crop"], p))
                    if harvestable:
                        tasks.append(Task(x, y, "HARVEST", None, 1))
                    if _needs_water(t, day, p.water_ongoing_eager) and not (
                        p.water_mode == "elif" and harvestable
                    ) and not (
                        p.water_mode == "survival" and harvestable
                        and t["consecutive_unwatered"] < 1
                    ):
                        # Emitting both unconditionally costs more than it saves: WATER outranks
                        # HARVEST, so every ready tile would pull a unit away from picking it.
                        # Water alongside a pending harvest only when the plant would otherwise
                        # die (death is at 2 consecutive misses), which is the case the old `elif`
                        # got wrong for ongoing crops.
                        tasks.append(Task(x, y, "WATER", None, 0))
                    # Re-apply once the previous dose has lapsed; a dose covers day..day+2.
                    if (p.fertilize and t.get("fertilized_until_day", -1) < day
                            and _wants_fertilizer(t, day)):
                        tasks.append(Task(x, y, "FERTILIZE", None, 2, needs="FERTILIZER"))
                elif "animal" in t:
                    n_animals += 1
                    if t["yield_units"] > 0:
                        tasks.append(Task(x, y, "HARVEST", None, 1))
                    if t["fertilizer_available"]:
                        tasks.append(Task(x, y, "COLLECT_FERTILIZER", None, 3))
                    if feed_today and not t["fed_today"]:
                        tasks.append(Task(x, y, "FEED", None, 0, needs="WHEAT"))
                    if p.care and not p.feed_alternate and not t["cared_today"]:
                        tasks.append(Task(x, y, "CARE", None, 4))
                else:  # an empty structure — kind decides which species can go on it
                    empty_coops_by_kind[t.get("kind", "COOP")].append((x, y))

        self._n_animals, self._n_plants = n_animals, n_plants

        # Wheat fetching. Without this, FEED is permanently unassignable (`needs="WHEAT"` and no
        # unit ever carries any), so every animal placed starves within two days.
        unfed = sum(1 for t in tasks if t.op == "FEED")
        if unfed:
            carried = sum(i.get("WHEAT", 0) for i in priv["inventories"])
            shed_wheat = priv["shed"].get("WHEAT", 0)
            short = min(unfed - carried - self._in_flight("WHEAT", p), shed_wheat)
            per_trip = max(1, min(p.feed_batch, short))
            trips = math.ceil(short / per_trip) if short > 0 else 0
            for k in range(min(trips, len(SHED_TILES))):
                x, y = SHED_TILES[k]
                tasks.append(Task(x, y, "PICKUP", "WHEAT", 0, n=per_trip))

        # Fertilizer fetching, exactly as for wheat above: FERTILIZE is gated on `needs`, so
        # without a trip to the shed the task is permanently unassignable. Units that ran
        # COLLECT_FERTILIZER are already carrying some, so only the shortfall needs collecting.
        n_fert = sum(1 for t in tasks if t.op == "FERTILIZE")
        if n_fert:
            carried_f = sum(i.get("FERTILIZER", 0) for i in priv["inventories"])
            short_f = min(n_fert - carried_f - self._in_flight("FERTILIZER", p),
                          priv["shed"].get("FERTILIZER", 0))
            if short_f > 0:
                per_trip = max(1, min(p.fertilize_batch, short_f))
                trips = math.ceil(short_f / per_trip)
                for k in range(min(trips, len(SHED_TILES))):
                    x, y = SHED_TILES[k]
                    tasks.append(Task(x, y, "PICKUP", "FERTILIZER", 2, n=per_trip))

        # --- animals: build structures, fetch stock, place it -------------------------------
        targets = {"GOOSE": p.goose_target, "COW": p.cow_target, "SHEEP": p.sheep_target}
        if self._demand is not None:
            # An animal is only worth its structure, its daily wheat and its daily trip if
            # something buys what it makes. Wool has no shop buyer in 36% of games (E33) and the
            # champion breeds 8 sheep regardless. Fertilizer is unconditional, so never zero these
            # out entirely -- floor at 1 -- but scale the herd by the product's actual demand.
            for animal, target in list(targets.items()):
                if target <= 0:
                    continue
                # Every animal yields 1 fertilizer/day regardless of species and regardless of
                # feeding (`kaggriculture.py:809`), and fertilizer was ~$10k of our revenue -- so a
                # herd is never worthless even when nothing buys its secondary product. Cut it only
                # when the product has *no* shop buyer at all, and never below the floor.
                d = self._demand.get(ANIMAL_PRODUCT[animal], 0)
                if d <= p.animal_demand_floor:
                    targets[animal] = max(p.animal_min_target, target // 2)
        placed = {a: 0 for a in targets}
        for row in farm["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("animal"):
                    placed[t["animal"]] = placed.get(t["animal"], 0) + 1

        free_struct = {"COOP": list(empty_coops_by_kind["COOP"]),
                       "PASTURE": list(empty_coops_by_kind["PASTURE"])}
        # Nearest-first, and animals get first pick. Measured: structures were being placed from
        # the raw row-major list while *crops* took the nearest tiles — exactly inverted. An animal
        # needs a wheat fetch from the shed almost every day, so its distance is paid daily; a crop
        # needs watering, which requires no shed trip at all. Animals sat at mean distance 7.1 and
        # movement was 65% of every unit's turns.
        remaining_empty = self._nearest_first(list(empty))

        for animal, target in targets.items():
            if target <= 0:
                continue
            struct = ANIMAL_STRUCTURE[animal]
            in_shed = priv["shed"].get(animal, 0)
            carried = sum(i.get(animal, 0) for i in priv["inventories"])

            # Place stock we already hold onto a free structure of the right kind.
            n_place = min(len(free_struct[struct]), in_shed + carried)
            for _ in range(n_place):
                x, y = free_struct[struct].pop(0)
                tasks.append(Task(x, y, "PLACE", animal, 2, needs=animal))

            # Fetch from the shed. PLACE is gated on already carrying the animal, and without this
            # nothing ever collects one — the defect that left coops empty for nine days.
            short = min(in_shed,
                        len(free_struct[struct]) - carried - self._in_flight(animal, p))
            if in_shed > 0 and short > 0:
                x, y = SHED_TILES[0]
                tasks.append(Task(x, y, "PICKUP", animal, 1, n=short))

            # Build what is still missing, cheapest structure first come first served.
            want = max(0, target - placed[animal] - len(free_struct[struct]))
            for _ in range(min(want, len(remaining_empty))):
                x, y = remaining_empty.pop(0)
                tasks.append(Task(x, y, BUILD_OP[struct], None, 5))

        # Cap standing crops at what the labour force can actually keep watered. Planting past
        # this point is worse than leaving the tile idle: an unwatered plant becomes a weed, which
        # then costs a DIG to clear.
        n_units = 1 + p.hire_max
        budget = max(0, int(n_units * p.tiles_per_unit) - n_plants)

        # That is a cap on the *stock* of crops. It says nothing about the *rate*, and the rate is
        # what kills us: planting is ordered whenever seeds and empty tiles happen to coincide,
        # which in practice means whenever cash arrives. Measured (E44/E45), our farm plants 14
        # tiles on day 0, nothing until day 15, then 46 tiles across three days. Every plant in a
        # burst is created with `consecutive_unwatered = 1` and needs water the same day, so the
        # burst exceeds what the units can service and the whole cohort dies together at age 2 --
        # 46 of our 56 crop deaths. boatlee plants a handful every single day and loses 6 all
        # season, mostly to old age.
        if p.plant_rate_per_day:
            planted_today = sum(
                1 for row in tiles for t in row
                if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("planted_day") == day
            )
            budget = min(budget, max(0, p.plant_rate_per_day - planted_today))

        remaining_empty = self._nearest_first(remaining_empty)[:budget]

        queue = self._plant_queue(priv, len(remaining_empty))
        for (x, y), crop in zip(remaining_empty, queue):
            # A crop that cannot reach its first yield before the season ends is pure loss: the
            # seed, the planting turn, and every watering it will be given. We planted 42 wheat on
            # day 27 (E45).
            if p.plant_stop_late and day + CROPS[crop]["first_yield_day"] > p.season_days - 1:
                continue
            tasks.append(Task(x, y, "PLANT", crop, 5))

        return tasks

    @staticmethod
    def _nearest_first(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Prefer tiles near the shed — units respawn there daily, so travel dominates cost."""
        return sorted(cells, key=lambda c: abs(c[0] - 4.5) + abs(c[1] - 4.5))

    def _effective_mix(self) -> dict[str, float]:
        """Crop weights, optionally re-weighted by the demand this game actually has.

        Under 1.32.4 every game unlocked all 8 shops, so the best mix was a constant and could be
        baked into the parameter vector. Under 1.32.6 a product can have four shops buying it or
        none, and the draw is visible in `obs["town"]["unlocked_shops"]` (E33). Melon is the extreme
        case: no shop has ever bought it, so its entire seasonal demand is the town centre's 1/day.
        """
        mix = {k: v for k, v in self.p.crop_mix.items() if v > 0}
        if not self._demand:
            return mix
        e = self.p.demand_exponent
        # Weight by *revenue capacity* (units absorbed per day x base price), not by units alone.
        # Weighting by units was measured and it lost: it gives wheat ~31x melon's weight because
        # many shops buy wheat, while ignoring that wheat sells at $25 and melon at $250. Revenue
        # capacity ranks strawberry (~23/day x $120) above wheat (~31/day x $25) above melon
        # (1/day x $250), which is also the ordering the leaderboard agent's mix implies.
        return {k: v * ((max(1, self._demand.get(k, 0)) * BASE_PRICE[k]) ** e) for k, v in mix.items()}

    def _plant_queue(self, priv: dict, n: int) -> list[str]:
        """Crops to plant, limited by seeds actually on hand."""
        mix = {k: v for k, v in self._effective_mix().items() if v > 0}
        if not mix:
            return []
        total = sum(mix.values())
        out: list[str] = []
        for crop, w in sorted(mix.items(), key=lambda kv: -kv[1]):
            want = min(int(round(n * w / total)) + 1, priv["seeds"].get(crop, 0))
            out.extend([crop] * want)
        return out[:n]

    # ------------------------------------------------------------- assignment

    def _roles(self, units: list, tasks: list[Task]) -> list[str]:
        """Split the workforce into livestock hands and field hands, in proportion to the work.

        boatlee keeps its units **93% pure** -- a waterer waters all season, and two hands do
        nothing but feed, care and collect, one of them confined to `y` 0-4. Our champion switches
        task type on **33% of consecutive actions** (E46). Pastures and fields are in different
        places, so each switch costs a crossing, and that is most of the gap between our 1.84 steps
        per productive action and their 1.01.

        The split is derived from the current task mix rather than fixed: their ~2 animal hands
        against 12 animals and 90 crops is the right ratio for *their* farm, not ours.
        """
        # Roles are **sticky**, and that is the whole point. Recomputing them every turn from the
        # current task mix -- the first implementation -- lets a unit be a livestock hand on one
        # turn and a field hand on the next, which is the very churn the penalty exists to stop. It
        # moved purity only 0.64 -> 0.69; boatlee sits at 0.93. Roles are therefore fixed for the
        # day and re-derived only when the day turns or the roster size changes.
        key = (self._day, len(units))
        if self._role_key != key:
            # Split by what the FARM is made of, not by the momentary task list. The task list is
            # violently uneven within a day -- at dawn it is almost entirely "feed everything", so
            # sizing the livestock crew from it locks most of the roster onto animal work that is
            # finished by mid-morning, after which they are barred from the fields and idle. Tile
            # counts are stable and are what the work will average out to.
            n_animal = self._n_animals
            total = n_animal + self._n_plants
            share = n_animal / total if total else 0.0
            k = min(len(units), max(0, round(share * len(units))))
            # Low indices take livestock: the farmer and earliest hands are the longest-lived.
            self._role_cache = ["A"] * k + ["C"] * (len(units) - k)
            self._role_key = key
        return self._role_cache

    @staticmethod
    def _here_bonus(unit, task: Task, bonus: float) -> float:
        """Discount for work on the tile a unit already stands on.

        Without it, priority arithmetic drives units off tiles that still have work. With
        `priority_weight = 0.49`, a CARE (priority 4) **on the current tile** costs 0 + 0.49*4 =
        1.96, while a WATER (priority 0) **one step away** costs 1.00 -- so the unit walks. Measured
        consequence: 27% of all moves leave a tile with a pending task, dominated by CARE (569),
        COLLECT_FERTILIZER (348) and FEED (311).

        boatlee issues 46% of its actions without having moved, running FEED -> CARE ->
        COLLECT_FERTILIZER -> HARVEST in a single visit; we manage 28% (E47).
        """
        if not bonus:
            return 0.0
        return -bonus if (unit[0] == task.x and unit[1] == task.y) else 0.0

    @staticmethod
    def _role_cost(role: str, task: Task, penalty: float) -> float:
        if not penalty:
            return 0.0
        is_animal = task.op in ANIMAL_OPS
        return penalty if (role == "A") != is_animal else 0.0

    def assign(self, units: list[tuple[int, int]], tasks: list[Task], invs: list[dict],
               commit: bool = True) -> list[list]:
        """Match units to tasks. `assign_mode` picks how; see `Params.assign_mode`.

        `commit=False` leaves `self._assigned` untouched, so a plan can be built and inspected
        without the sticky state drifting.

        A `perturb` argument used to sit here, generating candidate permutations for a rollout
        search. It was removed once the search was superseded: greedy assignment was 18% off
        optimal and optimal is exactly solvable at this size, so the engine solves it (E39). The
        perturbations also had almost nothing to permute -- 53% of turns carry fewer doable tasks
        than units.
        """
        saved = None if commit else dict(self._assigned)
        actions: list[list] = []
        p_weight = self.p.priority_weight
        rp = self.p.role_penalty
        hb = self.p.here_bonus
        roles = self._roles(units, tasks) if rp else ["C"] * len(units)
        taken: set[int] = set()
        by_key = {(t.x, t.y, t.op, t.arg, t.n): ti for ti, t in enumerate(tasks)}

        keep: dict[int, int] = {}

        # FIRST: a unit standing on work it can do, does it.
        #
        # This has to run before everything else, because everything else pre-empts it. Measured
        # (E47): units walk away from eligible work on their own tile 829 times a season -- 419
        # because a sticky assignment made on an earlier turn is honoured before the cost function
        # is consulted at all, and 381 because another unit (often further away) claimed the task
        # first. A distance-zero discount in the cost function cannot fix either: it was consulted
        # in 4% of those cases.
        #
        # boatlee issues 46% of its actions without moving, chaining FEED -> CARE ->
        # COLLECT_FERTILIZER on one visit; we manage 28% while having *more* co-located work
        # available (41% of our actions have another task pending on the same tile, against their
        # 30%).
        if self.p.finish_tile:
            for idx, (ux, uy) in enumerate(units):
                inv = invs[idx] if idx < len(invs) else {}
                for j, t in enumerate(tasks):
                    if j in taken or (t.x, t.y) != (ux, uy):
                        continue
                    # Farm work only. The first version accepted any task, and the tile a unit has
                    # most often "not finished" is the **shed**, which always has another PICKUP
                    # waiting -- so units camped there. Measured: PICKUP +52%, hauling 7.3% -> 9.7%,
                    # movement *up* 52.8% -> 55.8%, and of the 142 extra same-tile actions it
                    # produced, 117 were shed shuffling and farm work actually fell (E47).
                    if t.op in HAUL_OPS:
                        continue
                    if t.needs and inv.get(t.needs, 0) <= 0:
                        continue
                    keep[idx] = j
                    taken.add(j)
                    break

        # Honour existing assignments so units actually finish the walk they started.
        for idx in range(len(units)):
            if idx in keep:
                # Local work is a *detour*, not a change of plan. The first version popped the
                # sticky target here, so a unit three tiles into a five-tile walk that paused to do
                # something underfoot forgot where it was going and re-decided from scratch next
                # turn. That restarted journeys instead of shortening them -- movement rose
                # 52.7% -> 55.0% and hauling 7.7% -> 9.4%, the opposite of the intent. The target
                # is parked and resumed once the detour is done.
                if idx in self._assigned:
                    self._deferred[idx] = self._assigned.pop(idx)
                continue
            prev = self._assigned.get(idx) or self._deferred.pop(idx, None)
            if prev is None:
                continue
            ti = by_key.get(prev)
            inv = invs[idx] if idx < len(invs) else {}
            if ti is None or ti in taken:
                self._assigned.pop(idx, None)
                continue
            if tasks[ti].needs and inv.get(tasks[ti].needs, 0) <= 0:
                self._assigned.pop(idx, None)
                continue
            keep[idx] = ti
            taken.add(ti)

        # Global assignment over all free (unit, task) pairs, cheapest first.
        #
        # This loop used to run `for idx in enumerate(units)`, each unit greedily taking its own
        # best remaining task. Serving units in *index order* gives unit 0 first pick of the whole
        # farm, so it can take a task another unit is already standing on and send that unit
        # walking instead. Sorting every pair by cost and assigning greedily costs nothing at this
        # size (~11 units x ~50 tasks) and removes the index bias entirely.
        #
        # Still an approximation, not optimal assignment -- but E28 makes travel the thing to beat:
        # 1.84 steps walked per productive action against boatlee's 1.01.
        if self.p.assign_mode == "optimal":
            free = [i for i in range(len(units)) if i not in keep]
            avail = [j for j in range(len(tasks)) if j not in taken]
            if free and avail:
                # A `needs` violation is forbidden, not merely expensive: a large finite cost keeps
                # the matrix solvable when there are more units than tasks, and the result is
                # filtered afterwards so no illegal pair is ever committed.
                big = 1e6
                cost = []
                for i in free:
                    ux, uy = units[i]
                    inv = invs[i] if i < len(invs) else {}
                    cost.append([
                        big if (tasks[j].needs and inv.get(tasks[j].needs, 0) <= 0)
                        else (abs(tasks[j].x - ux) + abs(tasks[j].y - uy)
                              + p_weight * tasks[j].prio
                              + self._role_cost(roles[i], tasks[j], rp)
                              + self._here_bonus((ux, uy), tasks[j], hb))
                        for j in avail
                    ])
                if len(free) <= len(avail):
                    match = _optimal_assignment(cost, len(free), len(avail))
                else:
                    # More units than tasks: solve the transpose so rows <= cols, then invert.
                    t_cost = [[cost[i][j] for i in range(len(free))] for j in range(len(avail))]
                    match = {v: k for k, v in
                             _optimal_assignment(t_cost, len(avail), len(free)).items()}
                for i, j in match.items():
                    if cost[i][j] >= big:
                        continue
                    keep[free[i]] = avail[j]
                    taken.add(avail[j])

        if self.p.assign_mode == "global":
            free = [i for i in range(len(units)) if i not in keep]
            pairs = []
            for idx in free:
                ux, uy = units[idx]
                inv = invs[idx] if idx < len(invs) else {}
                for cand, t in enumerate(tasks):
                    if cand in taken or (t.needs and inv.get(t.needs, 0) <= 0):
                        continue
                    pairs.append(((abs(t.x - ux) + abs(t.y - uy)) + p_weight * t.prio
                                  + self._role_cost(roles[idx], t, rp)
                                  + self._here_bonus((ux, uy), t, hb), idx, cand))
            pairs.sort()
            claimed: set[int] = set()
            for _, idx, cand in pairs:
                if idx in claimed or cand in taken:
                    continue
                keep[idx] = cand
                taken.add(cand)
                claimed.add(idx)

        # Sequential fill, in unit order, exactly as before -- but resolved into a complete
        # mapping *before* anything is emitted, so a candidate can permute it.
        for idx, (ux, uy) in enumerate(units):
            if idx in keep:
                continue
            inv = invs[idx] if idx < len(invs) else {}
            best, best_key = None, None
            for cand, t in enumerate(tasks):
                if cand in taken:
                    continue
                if t.needs and inv.get(t.needs, 0) <= 0:
                    continue
                # Distance and priority on one scale: a task `priority_weight` tiles further
                # away must be a whole priority level more urgent to be worth the walk.
                key = ((abs(t.x - ux) + abs(t.y - uy)) + p_weight * t.prio
                       + self._role_cost(roles[idx], t, rp)
                       + self._here_bonus((ux, uy), t, hb))
                if best_key is None or key < best_key:
                    best, best_key = cand, key
            if best is not None:
                keep[idx] = best
                taken.add(best)

        # Diagnostics hook: the finished plan, before it is turned into one step per unit.
        # Measuring routing quality from the emitted actions is impossible (they are direction
        # steps), and re-deriving the plan in an analysis script measures the script, not the
        # engine -- which is how the first version of the E39 comparison was done.
        self._last_plan = dict(keep)

        for idx, (ux, uy) in enumerate(units):
            inv = invs[idx] if idx < len(invs) else {}
            ti = keep.get(idx)
            if ti is None:
                self._assigned.pop(idx, None)
                actions.append(self._fallback(ux, uy, inv, tasks))
                continue
            t = tasks[ti]
            self._assigned[idx] = (t.x, t.y, t.op, t.arg, t.n)
            if (ux, uy) != (t.x, t.y):
                actions.append(self._step_toward(ux, uy, t.x, t.y))
            else:
                self._assigned.pop(idx, None)   # arriving completes it
                if t.n is not None:
                    actions.append([t.op, t.arg, t.n])
                elif t.arg:
                    actions.append([t.op, t.arg])
                else:
                    actions.append([t.op])
        if saved is not None:
            self._assigned = saved
        return actions

    def _fallback(self, ux: int, uy: int, inv: dict, tasks: list[Task]) -> list:
        """Nothing assignable: fetch an item a blocked task needs, or dump what we're carrying."""
        wanted = next((t.needs for t in tasks if t.needs and inv.get(t.needs, 0) <= 0), None)
        at_shed = (ux, uy) in SHED_TILES
        if wanted:
            if at_shed:
                return ["PICKUP", wanted, 4]
            return self._step_toward(ux, uy, 4, 4)
        if inv and at_shed:
            return ["DROP"]
        if inv:
            return self._step_toward(ux, uy, 4, 4)
        return ["PASS"]

    @staticmethod
    def _step_toward(ux: int, uy: int, tx: int, ty: int) -> list:
        if ux < tx:
            return ["EAST"]
        if ux > tx:
            return ["WEST"]
        if uy < ty:
            return ["SOUTH"]
        if uy > ty:
            return ["NORTH"]
        return ["PASS"]

    # --------------------------------------------------------------- forecast

    def opponent_supply(self, obs: dict, me: int) -> dict[str, int]:
        """Units of each product the opponent will harvest within `forecast_horizon` days.

        Their farm is public (`farms` is shared) and crop maturity is deterministic, so their
        entire production schedule is computable. Their *shed* is private, so this is incoming
        supply only — not stock they may already be sitting on.
        """
        opp = 1 - me
        farm = obs["farms"][opp]
        day = obs["day"]
        horizon = self.p.forecast_horizon
        out: dict[str, int] = {}

        for row in farm["tiles"]:
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    c = CROPS[t["crop"]]
                    age = day - t["planted_day"]
                    ready_at = c["first_yield_day"] if c["ongoing"] else _harvest_day(t["crop"], self.p)
                    eta = ready_at - age
                    if eta <= horizon:
                        # Already-accumulated units can be sold immediately; otherwise assume the
                        # crop is taken to its cap, which is what any competent agent does.
                        units = t["yield_units"] if eta <= 0 else c["max_yield"]
                        out[t["crop"]] = out.get(t["crop"], 0) + units
                elif "animal" in t:
                    p = ANIMAL_PRODUCT[t["animal"]]
                    a = ANIMALS[t["animal"]]
                    produced = t["yield_units"] + max(0, horizon // max(1, a["interval"]))
                    out[p] = out.get(p, 0) + produced
        return out

    @staticmethod
    def town_drain_per_day(obs: dict) -> dict[str, int]:
        """How much the town removes per day — the only thing that lets a crashed price recover.

        Since kaggle-environments 1.32.6 this is also the **demand model**, not just a price input.
        Shops are drawn WITH replacement, so `unlocked_shops` can list the same shop several times
        and each instance consumes independently; a product may end a game with four shops buying
        it or none at all. Wool has no shop buyer in 36% of games, melon in 100% (E33).

        Assumes the default intervals (24 turns/day, shops every 4 turns, centre every 24). The
        environment configuration is genuinely not exposed in the observation, so this cannot be
        read from the game — it is the one place the engine has to assume.
        """
        drain: dict[str, int] = {}
        for shop in obs["town"]["unlocked_shops"]:
            products = SHOPS[shop]
            mult = 2 if len(products) == 1 else 1
            for item in products:
                drain[item] = drain.get(item, 0) + mult * 6      # 24/4 = 6 ticks/day
        # 1.32.6 deleted TOWN_CENTER_DEMAND_SCHEDULE and moved the interval 12 -> 24, so the centre
        # buys exactly one of each non-fertilizer product per day, flat, all season. This used to
        # read `(4 if day>=20 else 2 if day>=10 else 1) * 2 ticks`, which now overstates the centre
        # by 2-8x and, through `forecast_price`, made every product look far more absorbable than
        # it is.
        for item in TOWN_CENTER_PRODUCTS:
            drain[item] = drain.get(item, 0) + 1                 # 24/24 = 1 tick/day
        return drain

    def forecast_price(self, item: str, inv: dict, supply: dict, drain: dict,
                       params: dict | None = None) -> float:
        """Price we expect once the opponent's wave lands and the town has drained a little."""
        h = self.p.forecast_horizon
        future = inv[item] + supply.get(item, 0) - drain.get(item, 0) * h
        return float(market_price(item, max(0, future), params))

    # ----------------------------------------------------------------- market

    def market(self, obs: dict, farm: dict, priv: dict, day: int, hour: int) -> list[list]:
        """Spend against an explicit budget, in ROI order, within the 10-slot cap.

        Two things drive the ordering:

        * **Sells go first.** `_process_market` commits order slots sequentially, so proceeds from
          a SELL in slot 0 are available to fund a BUY in a later slot *within the same turn*.
        * **Cheap, fast-payback purchases outrank expensive, slow ones.** Seeds are working capital
          that recycles in 4-5 days; land is a $1k-$4k outlay that only pays off if the labour
          exists to work it. Buying land first is exactly what starved the engine of seed money and
          invalidated the goose experiment (docs/experiments.md E3).
        """
        p = self.p
        inv = obs["market"]["inventory"]
        tiles = [t for row in farm["tiles"] for t in row]
        n_animals = sum(1 for t in tiles if isinstance(t, dict) and "animal" in t)
        n_structs = sum(
            1 for t in tiles
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and "animal" not in t
        )
        empty = sum(1 for t in tiles if t is None)

        orders: list[list] = []

        # 1. Sell first — the proceeds fund everything below, this same turn.
        feed_reserve = (n_animals + n_structs) * p.wheat_reserve_per_animal
        # `marketParams` overrides land in the shared observation; without threading them through,
        # every price the engine reasons about is wrong whenever the episode overrides a curve.
        mparams = obs["market"].get("params")
        sells = self._sell_orders(priv, inv, day, feed_reserve, obs)[:3]
        orders += sells
        money = farm["money"] + sum(
            self._estimate_proceeds(o[1], o[2], inv, mparams) for o in sells)

        def afford(cost: float, extra_reserve: float = 0.0) -> bool:
            nonlocal money
            if money - cost < p.cash_floor + extra_reserve:
                return False
            money -= cost
            return True

        def buy_animals():
            """One purchase pass per species, cheapest-first.

            Every animal yields 1 fertilizer/day regardless of species and regardless of feeding,
            so a bird and a cow share that stream; the species differs only in the secondary
            product and in price ($300 / $400 / $500). Whether the dearer species ever pays is a
            measurement, not an assumption — cows and sheep were simply unbuildable before V3.
            """
            for animal in ("GOOSE", "COW", "SHEEP"):
                target = {"GOOSE": p.goose_target, "COW": p.cow_target,
                          "SHEEP": p.sheep_target}[animal]
                if target <= 0:
                    continue
                carried = sum(i.get(animal, 0) for i in priv["inventories"])
                have = sum(1 for row in farm["tiles"] for t in row
                           if isinstance(t, dict) and t.get("animal") == animal)
                owned = have + priv["shed"].get(animal, 0) + carried
                room = n_structs + empty
                if owned >= target or room <= 0:
                    continue
                cost = ANIMALS[animal]["cost"]
                want = min(target - owned, room, 2)
                n = 0
                while n < want and afford(cost, p.goose_min_cash):
                    n += 1
                if n:
                    orders.append(["BUY_ANIMAL", animal, n])

        # 2. Seeds — working capital, fastest payback of anything on this list.
        mix = {k: v for k, v in p.crop_mix.items() if v > 0}
        if mix and empty > 0:
            total = sum(mix.values())
            seed_budget = max(0.0, (money - p.cash_floor) * p.seed_budget_frac)
            for crop, w in sorted(mix.items(), key=lambda kv: -kv[1]):
                want = int(empty * w / total) + 1
                have = priv["seeds"].get(crop, 0)
                cost = CROPS[crop]["seed"]
                n = min(want - have, int(seed_budget // cost))
                if n > 0 and afford(n * cost):
                    orders.append(["BUY_SEED", crop, n])
                    seed_budget -= n * cost

        # 3. Feed for animals already owned — starving one writes off $300+.
        need_wheat = (n_animals + n_structs) * p.wheat_reserve_per_animal
        have_wheat = priv["shed"].get("WHEAT", 0)
        if n_animals and need_wheat > have_wheat:
            n = min(need_wheat - have_wheat, 10)
            if afford(n * obs["market"]["prices"]["WHEAT"]):
                orders.append(["BUY_PRODUCT", "WHEAT", n])

        # 4. Hands — spread across turns so HIRE cannot eat every order slot at hour 0.
        if p.hire_max > 0 and hour < 6 and len(orders) < 8:
            n = min(3, p.hire_max - farm["hires_today"], 8 - len(orders))
            for _ in range(max(0, n)):
                if afford(4.0):
                    orders.append(["HIRE"])

        if not p.animals_before_seeds:
            buy_animals()

        # 6. Land — largest and slowest outlay, and it needs *labour* headroom, not just cash.
        #    Measured: buying it with too few hands halves final money (experiments.md E1).
        # Buy land when the land we already own is nearly full *and* the labour force could work
        # more than it holds. The previous form compared `worked` against total labour capacity,
        # which at hire_max=8 is 54 tiles — unreachable when only 25 are unlocked, so the branch
        # was dead and land was never bought at all. Measured symptom: units idle 38-58% of turns.
        if p.buy_land and len(farm["unlocked_quadrants"]) < 4:
            cost = [1000, 2000, 4000][len(farm["unlocked_quadrants"]) - 1]
            unlocked = sum(1 for t in tiles if t != "LOCKED")
            occupied = sum(1 for t in tiles
                           if isinstance(t, dict) and t.get("kind") != "WEED")
            capacity = (1 + p.hire_max) * p.tiles_per_unit
            if (occupied >= p.land_fill_frac * unlocked
                    and capacity > unlocked
                    and afford(cost, p.land_min_cash)):
                orders.append(["BUY_LAND"])

        return orders[:10]

    @staticmethod
    def _estimate_proceeds(item: str, n: int, inv: dict, params: dict | None = None) -> float:
        """Revenue a pending sell will raise, so this turn's buys can budget against it."""
        base = inv[item]
        return float(sum(market_price(item, base + k, params) for k in range(min(n, 200))))

    def _sell_orders(self, priv: dict, inv: dict, day: int, feed_reserve: int,
                     obs: dict | None = None) -> list[list]:
        """Marginal reservation pricing: sell while the *next* unit still clears the reserve.

        With `forecast_weight > 0` the reserve is blended toward the price we expect after the
        opponent's visible supply lands. Holding out for a price their harvest is about to destroy
        is how a static rule loses the race into the good part of the curve (experiments E6).
        """
        p = self.p
        out: list[list] = []
        dump = day >= p.sell_all_after_day
        supply = drain = None
        mparams = obs["market"].get("params") if obs is not None else None
        if obs is not None and p.forecast_weight > 0:
            supply = self.opponent_supply(obs, obs["player"])
            drain = self.town_drain_per_day(obs)
        for item, held in priv["shed"].items():
            if held <= 0 or item not in BASE_PRICE:
                continue
            if item == "WHEAT" and not dump:
                # Must match the buy-side reserve exactly, or the engine sells back the feed it
                # just bought — a churn loop that burns market slots and inflates revenue stats
                # while netting zero.
                held -= feed_reserve
                if held <= 0:
                    continue
            if dump:
                out.append(["SELL", item, held])
                continue
            reserve = p.reserve_frac.get(item, 0.0) * BASE_PRICE[item]
            if supply is not None:
                w = p.forecast_weight
                reserve = ((1 - w) * reserve
                           + w * self.forecast_price(item, inv, supply, drain, mparams))
            n = 0
            cur = inv[item]
            while n < held and market_price(item, cur + n, mparams) >= reserve:
                n += 1
            if n > 0:
                out.append(["SELL", item, n])
        return out

    # ------------------------------------------------------------------- main

    def __call__(self, obs: dict, configuration: dict | None = None) -> dict:
        # `configuration` is accepted and ignored purely so this instance can be handed straight to
        # `kaggle_environments`. That framework passes (observation, configuration) to a *callable
        # object* while passing only (observation) to a plain function, so a one-argument
        # `__call__` raises TypeError on the first turn -- the agent then scores the $3,000
        # starting bank and the run still reports DONE. It reads as a simulator divergence.
        # `main.py` was never affected: it wraps this in `def agent(obs)`.
        player = obs["player"]
        farm = obs["farms"][player]
        priv = obs["private"]
        day, hour = obs["day"], obs["hour"]

        # `_day` was initialised to -1 in `__init__` and never assigned anywhere -- dead state.
        # It is the natural key for anything that should be stable within a day (roles), so it is
        # now actually maintained.
        self._day = day

        # Observed per-product demand. Since 1.32.6 shops draw WITH replacement, so this is a
        # per-game random variable rather than the constant it used to be, and it is the only
        # signal available for deciding what to grow (E33, E34).
        self._demand = self.town_drain_per_day(obs) if self.p.adaptive_mix else None

        units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        invs = priv["inventories"]
        tasks = self.build_tasks(farm, priv, day)
        unit_actions = self.assign(units, tasks, invs)

        return {
            "farmer": unit_actions[0],
            "hands": unit_actions[1:],
            "market": self.market(obs, farm, priv, day, hour),
        }


def make_agent(params: Params | None = None):
    """kagsim agents are called once per turn, so the engine keeps its own per-episode state."""
    return Engine(params)
