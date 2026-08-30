"""Legality, as data.

Two things live here.

**`TurnState`** -- the running simulation of one turn, from the delivered observation forward.  A
turn is decoded autoregressively: the farmer decides, then hand 1 is told what the farmer chose,
and so on, and *then* the market orders resolve (`_process_market` runs at `kag.py:941`, after the
unit loop at `:935-941`).  That ordering is not an aesthetic choice.  It is forced twice over:

* the planting cliff -- if the turn's total `PLANT` requests for one crop exceed the seeds held,
  **every one of them is dropped** (`kag.py:920-933`), so a per-unit-independent decoder falls off
  a cliff the expert sits right on top of;
* effective shed -- a worker can drop wheat into the shed and the same turn's SELL can sell it,
  which is PLAN_BC Assertion 4.

`TurnState` mirrors `_apply_unit_action` (`kag.py:311-530`) for exactly the state legality depends
on: tiles, unit positions, unit inventories, shed, seeds, money, hires.  It is a re-implementation,
which E39 warns about -- so it is never used to *judge* the environment, only to answer "is this
legal", and it is checked by Assertion 3: if the simulation drifted, expert actions would start
being rejected and the counter would stop reading zero.

**`compute_masks`** -- fixed-shape boolean arrays, one call per decision point.  Data, not control
flow: the same function runs in `decode.py` (Assertion 3) and later at inference, so a mask that is
wrong is wrong in both places and shows up as a counter rather than as a mystery.

**`PASS` is never masked** (`kag.py:334`).  A fully-masked row produces NaN and poisons training;
the reference implementation had to filter `|log_prob| > 1e5` for exactly this reason.
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as kag

from . import vocab as V

SHED_ACCESS = frozenset(tuple(t) for t in kag._shed_access_tiles(V.GRID))


def flat_tiles(farm):
    """`farms[p]["tiles"]` is a NESTED 10x10 list, not 100 flat tiles (PLAN_BC Ch3).

    Indexing it as if it were flat returns nothing useful and raises no error -- it silently broke
    a verifier's first probe.  Flatten explicitly, assert the length.
    """
    rows = farm["tiles"]
    out = [t for row in rows for t in row]
    assert len(out) == V.N_TILES, f"tiles flattened to {len(out)}, expected {V.N_TILES}"
    return out


def tile_index(x, y):
    return int(y) * V.GRID + int(x)


def tile_xy(idx):
    return int(idx) % V.GRID, int(idx) // V.GRID


def manhattan(a, b):
    """True distance: `kag.py:326-331` bounds-checks only and its comment states LOCKED tiles are
    passable, so walking is never obstructed and Manhattan distance is exact everywhere."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


# --------------------------------------------------------------------------------------
# The running turn state
# --------------------------------------------------------------------------------------

class TurnState:
    """Mutable "decisions so far" for one turn of one seat.

    Built from a *delivered* observation (see `decode.delivered_obs`), then advanced one unit
    action and one market order at a time.
    """

    __slots__ = ("player", "tiles", "positions", "inventories", "shed", "seeds", "money",
                 "hires_today", "unlocked", "market_inv", "market_params", "day", "hour", "step",
                 "n_orders", "shed_cap", "blocked_crops", "plant_demand", "deposits",
                 "withdrawals", "n_animal_placed")

    def __init__(self, obs, player, shed_cap=V.SHED_CAPACITY):
        farm = obs["farms"][player]
        priv = obs["private"]
        self.player = player
        self.tiles = [dict(t) if isinstance(t, dict) else t for t in flat_tiles(farm)]
        self.positions = [list(farm["farmer"])] + [list(h) for h in farm["hands"]]
        invs = priv.get("inventories") or [{}]
        self.inventories = [dict(invs[i]) if i < len(invs) else {} for i in range(len(self.positions))]
        self.shed = {k: int(v) for k, v in priv["shed"].items()}
        self.seeds = {k: int(v) for k, v in priv["seeds"].items()}
        self.money = float(farm["money"])
        self.hires_today = int(farm.get("hires_today", 0))
        self.unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
        self.market_inv = {k: int(v) for k, v in obs["market"]["inventory"].items()}
        self.market_params = obs["market"].get("params")
        self.day = int(obs["day"])
        self.hour = int(obs["hour"])
        self.step = int(obs["step"])
        self.n_orders = 0
        self.shed_cap = int(shed_cap)
        self.blocked_crops = frozenset()
        # This turn's PLANT requests per crop, as issued.  Bookkeeping only -- the live budget an
        # autoregressive decode spends against is `self.seeds`, which `apply_unit` decrements.
        self.plant_demand = {}
        # Assertion-4 bookkeeping: what units moved in and out of the shed during THIS turn.
        self.deposits = {}
        self.withdrawals = {}
        # Counts animal PLACEs that actually took the tile branch.  A zero here would mean the
        # carve-out never fired and Assertion 4 was measuring nothing (CLAUDE.md, E44).
        self.n_animal_placed = 0

    # -- queries -------------------------------------------------------------------------

    @property
    def n_units(self):
        return len(self.positions)

    def shed_used(self):
        return sum(self.shed.values())

    def shed_free(self):
        return max(0, self.shed_cap - self.shed_used())

    def price(self, item, offset=0):
        return kag.market_price(item, self.market_inv[item] + offset, self.market_params)

    def effective_shed(self):
        """The shed as `_process_market` will find it: what the observation showed, plus what units
        put in and minus what they took out during this turn (`kag.py:935-941`)."""
        return dict(self.shed)

    def shed_plus_deposits(self):
        """PLAN_BC Assertion 4 as literally written: observation shed + this turn's deposits, with
        withdrawals NOT netted out.  Kept alongside `effective_shed` so the two can be reported
        separately rather than one quietly standing in for the other."""
        out = {k: v for k, v in self.shed.items()}
        for item, n in self.withdrawals.items():
            out[item] = out.get(item, 0) + n
        return out

    # -- the unit interpreter, mirroring kag.py:311-530 ---------------------------------

    def set_blocked_crops(self, unit_actions):
        """Atomic PLANT validation (`kag.py:920-933`), computed over the whole turn's unit actions
        before any of them are applied: if demand for a crop exceeds seeds, ALL its PLANTs drop."""
        demand = {}
        for a in unit_actions:
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                demand[a[1]] = demand.get(a[1], 0) + 1
        self.blocked_crops = frozenset(c for c, n in demand.items() if n > self.seeds.get(c, 0))
        self.plant_demand = demand
        return self.blocked_crops

    def plant_demand_at_the_edge(self):
        """Crops whose demand this turn exactly equals the seed stock -- one more request and the
        whole burst would have been dropped.  PLAN_BC Ch5 measured 66 such turns in the sample and
        uses them to argue the decoder must be autoregressive; this reports the same thing over the
        whole corpus, and proves the cliff check is examining live data rather than nothing."""
        return [c for c, n in self.plant_demand.items() if n > 0 and n == self.seeds.get(c, 0)]

    def apply_unit(self, idx, action):
        """Apply one unit action exactly as the environment would.  Silent no-ops stay no-ops."""
        if not isinstance(action, list) or not action or idx >= len(self.positions):
            return
        op = action[0]
        if op == "PLANT" and len(action) >= 2 and action[1] in self.blocked_crops:
            return                                          # rewritten to PASS by the interpreter
        pos = self.positions[idx]
        fx, fy = int(pos[0]), int(pos[1])
        inv = self.inventories[idx]

        if op in kag.FARMER_MOVES:
            dx, dy = kag.FARMER_MOVES[op]
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < V.GRID and 0 <= ny < V.GRID:
                self.positions[idx] = [nx, ny]
            return
        if op == "PASS":
            return

        ti = tile_index(fx, fy)
        tile = self.tiles[ti]
        adjacent = (fx, fy) in SHED_ACCESS

        # Shed ops resolve before the LOCKED guard (:344 comment): three of the four shed-access
        # tiles start LOCKED, and guarding first would make the shed unreachable from them.
        if op == "DROP":
            if not adjacent:
                return
            for item, n in list(inv.items()):
                if n <= 0:
                    del inv[item]
                    continue
                room = self.shed_free()
                take = min(n, room)
                if take > 0:
                    self.shed[item] = self.shed.get(item, 0) + take
                    self.deposits[item] = self.deposits.get(item, 0) + take
                del inv[item]
            return

        if op == "PICKUP":
            if not adjacent or len(action) < 2:
                return
            item = action[1]
            n = int(action[2]) if len(action) >= 3 else 1
            if n <= 0:
                return
            n = min(n, self.shed.get(item, 0))
            if n <= 0:
                return
            self.shed[item] -= n
            inv[item] = inv.get(item, 0) + n
            self.withdrawals[item] = self.withdrawals.get(item, 0) + n
            return

        if op == "PLACE":
            if len(action) < 2:
                return
            item = action[1]
            if (item in kag.ANIMALS and isinstance(tile, dict)
                    and tile.get("kind") == kag.ANIMALS[item]["structure"] and "animal" not in tile):
                # Animal placement puts the animal on the TILE, never in the shed (:381-392).
                # This carve-out is what makes PLAN_BC's 216/216 effective-shed result hold.
                if inv.get(item, 0) >= 1:
                    inv[item] -= 1
                    if inv[item] == 0:
                        del inv[item]
                    self.tiles[ti] = kag._new_animal(item, self.day)
                    self.n_animal_placed += 1
                return
            if adjacent:
                n = int(action[2]) if len(action) >= 3 else 1
                n = min(n, inv.get(item, 0), self.shed_free())
                if n <= 0:
                    return
                inv[item] -= n
                if inv[item] == 0:
                    del inv[item]
                self.shed[item] = self.shed.get(item, 0) + n
                self.deposits[item] = self.deposits.get(item, 0) + n
            return

        if tile == "LOCKED":            # every op below mutates the tile, so it must be owned (:414)
            return

        if op == "PLANT":
            if len(action) < 2 or action[1] not in kag.CROPS or tile is not None:
                return
            crop = action[1]
            if self.seeds.get(crop, 0) <= 0:
                return
            self.seeds[crop] -= 1
            self.tiles[ti] = kag._new_plant(crop, self.day, V.TURNS_PER_DAY)
            return

        if op == "WATER":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT") or tile["watered_today"]:
                return
            tile["watered_today"] = True
            cd = kag.CROPS[tile["crop"]]
            if not cd["ongoing"]:
                age = self.day - tile["planted_day"]
                if (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]:
                    bonus = 2 if tile["fertilized_until_day"] >= self.day else 1
                    tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + bonus)
            return

        if op == "HARVEST":
            if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
                return
            if tile.get("kind") == "PLANT":
                cd = kag.CROPS[tile["crop"]]
                if self.day - tile["planted_day"] < cd["first_yield_day"]:
                    return
                units = tile["yield_units"]
                tile["yield_units"] = 0
                inv[tile["crop"]] = inv.get(tile["crop"], 0) + units
                if not cd["ongoing"]:
                    self.tiles[ti] = None
            elif "animal" in tile:
                units = tile["yield_units"]
                tile["yield_units"] = 0
                product = kag.ANIMALS[tile["animal"]]["product"]
                inv[product] = inv.get(product, 0) + units
            return

        if op == "FERTILIZE":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                return
            if inv.get("FERTILIZER", 0) < 1:
                return
            inv["FERTILIZER"] -= 1
            if inv["FERTILIZER"] == 0:
                del inv["FERTILIZER"]
            tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), self.day + 2)
            return

        if op == "DIG":
            if tile is None or (isinstance(tile, dict) and "animal" in tile):
                return
            self.tiles[ti] = None
            return

        if op in ("BUILD_COOP", "BUILD_PASTURE"):
            if tile is not None:
                return
            self.tiles[ti] = {"kind": "COOP" if op == "BUILD_COOP" else "PASTURE"}
            return

        if op == "FEED":
            if not (isinstance(tile, dict) and "animal" in tile) or tile["fed_today"]:
                return
            if inv.get("WHEAT", 0) < 1:
                return
            inv["WHEAT"] -= 1
            if inv["WHEAT"] == 0:
                del inv["WHEAT"]
            tile["fed_today"] = True
            return

        if op == "COLLECT_FERTILIZER":
            if not (isinstance(tile, dict) and "animal" in tile) or not tile["fertilizer_available"]:
                return
            tile["fertilizer_available"] = False
            inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
            return

        if op == "CARE":
            if not (isinstance(tile, dict) and "animal" in tile) or tile["cared_today"]:
                return
            tile["cared_today"] = True
            return

    # -- the market simulation, mirroring kag.py:562-686 --------------------------------

    def apply_market(self, order):
        """Advance money/shed/seeds/hires by one order, *our side only*.

        The real settlement interleaves both players unit by unit (`kag.py:615-625`), so this is
        our own budget rather than a prediction of what the market does.  That is precisely what a
        slot mask needs: PLAN_BC Ch5's "running simulation of money, shed and price".
        """
        if not isinstance(order, list) or not order:
            return
        self.n_orders += 1
        op = order[0]
        if op == "HIRE":
            cost = kag._hire_cost(self.hires_today)
            if self.money >= cost:
                self.money -= cost
                self.hires_today += 1
                self.positions.append(list(kag._spawn_hand(
                    {"farmer": self.positions[0], "hands": self.positions[1:]}, V.GRID)))
                self.inventories.append({})
            return
        if op == "BUY_LAND":
            extra = len(self.unlocked) - 1
            if extra >= len(kag.LAND_ORDER):
                return
            cost = kag.LAND_PRICES[extra]
            if self.money < cost:
                return
            self.money -= cost
            quad = kag.LAND_ORDER[extra]
            self.unlocked.append(quad)
            for i in range(V.N_TILES):
                x, y = tile_xy(i)
                if kag._quadrant_of(x, y, V.GRID) == quad and self.tiles[i] == "LOCKED":
                    self.tiles[i] = None
            return
        if len(order) < 3:
            return
        item = order[1]
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            return
        for _ in range(max(0, n)):
            if op == "SELL":
                if item not in kag.PRODUCTS or self.shed.get(item, 0) <= 0:
                    return
                price = self.price(item)
                self.shed[item] -= 1
                self.money += price
                if price > 1:                       # $1 sales do not add supply (:648-651)
                    self.market_inv[item] += 1
            elif op == "BUY_PRODUCT":
                if item not in V.BUYABLE_PRODUCTS:
                    return
                price = self.price(item, offset=-1)  # quoted at post-buy inventory (:600)
                if self.money < price or self.shed_used() >= self.shed_cap:
                    return
                self.money -= price
                self.shed[item] = self.shed.get(item, 0) + 1
                self.market_inv[item] -= 1
            elif op == "BUY_SEED":
                if item not in kag.CROPS:
                    return
                cost = kag.CROPS[item]["seed"]
                if self.money < cost:
                    return
                self.money -= cost
                self.seeds[item] = self.seeds.get(item, 0) + 1
            elif op == "BUY_ANIMAL":
                if item not in kag.ANIMALS:
                    return
                cost = kag.ANIMALS[item]["cost"]
                if self.money < cost or self.shed_used() >= self.shed_cap:
                    return
                self.money -= cost
                self.shed[item] = self.shed.get(item, 0) + 1
            else:
                return


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------

class Masks:
    """Fixed-shape boolean arrays for one decision point.  Every field is `np.bool_`."""

    __slots__ = ("unit_valid", "unit_verb", "unit_item", "market_op", "market_item", "market_qty")

    def __init__(self, unit_valid, unit_verb, unit_item, market_op, market_item, market_qty):
        self.unit_valid = unit_valid      # (MAX_UNITS,)                  which slots hold a worker
        self.unit_verb = unit_verb        # (MAX_UNITS, N_VERBS)          raw verb, at its own tile
        self.unit_item = unit_item        # (MAX_UNITS, N_VERBS, N_ITEM_SLOTS)
        self.market_op = market_op        # (N_MARKET_OPS,)
        self.market_item = market_item    # (N_MARKET_OPS, N_ITEM_SLOTS)
        self.market_qty = market_qty      # (N_MARKET_OPS, N_ITEM_SLOTS, N_QTY)


def _tile_at(ts, x, y):
    return ts.tiles[tile_index(x, y)]


def verb_mask_at(ts, idx, xy=None, macro=False):
    """Legality of every verb for unit `idx` standing on `xy` (its own tile when `xy is None`).

    The macro variant asks the same question about a *target* tile the walker will reach, and drops
    the four moves in favour of `MOVE`.  `MOVE` is legal whenever the target is not where the unit
    already stands.  The tile pointer itself is never masked -- every tile is reachable, and
    legality is enforced here instead (PLAN_BC Ch5).
    """
    n = V.N_MACRO_VERBS if macro else V.N_VERBS
    m = np.zeros(n, dtype=bool)
    if idx >= len(ts.positions):
        return m
    here = ts.positions[idx]
    x, y = (int(here[0]), int(here[1])) if xy is None else (int(xy[0]), int(xy[1]))
    inv = ts.inventories[idx]
    tile = _tile_at(ts, x, y)
    adjacent = (x, y) in SHED_ACCESS
    is_dict = isinstance(tile, dict)
    is_plant = is_dict and tile.get("kind") == "PLANT"
    has_animal = is_dict and "animal" in tile
    locked = tile == "LOCKED"

    def put(name, ok):
        if macro:
            m[V.MACRO_VERB_INDEX[name]] = ok
        else:
            m[V.VERB_INDEX[name]] = ok

    if macro:
        m[V.MV_MOVE] = manhattan((x, y), here) > 0
    else:
        for v, (dx, dy) in kag.FARMER_MOVES.items():
            m[V.VERB_INDEX[v]] = (0 <= x + dx < V.GRID) and (0 <= y + dy < V.GRID)

    put("PASS", True)                                                                     # :334
    put("DROP", adjacent and any(v > 0 for v in inv.values()))                            # :344
    put("PICKUP", adjacent and any(v > 0 for v in ts.shed.values()))                      # :359
    can_place_animal = any(
        inv.get(a, 0) >= 1 and is_dict and tile.get("kind") == kag.ANIMALS[a]["structure"]
        and not has_animal for a in kag.ANIMALS)
    can_place_shed = adjacent and ts.shed_free() > 0 and any(v > 0 for v in inv.values())
    put("PLACE", can_place_animal or can_place_shed)                                      # :376-409
    # The running seed budget IS `ts.seeds`: `apply_unit` spends a seed the moment a unit commits
    # to a PLANT, so an autoregressive decode can never push the turn's demand past the stock and
    # fall off the cliff at kag.py:920-933.  (Subtracting `plant_demand` here as well double-counts
    # -- that bug rejected 693 of the expert's own PLANTs before it was caught by Assertion 3.)
    seed_ok = any(ts.seeds.get(c, 0) > 0 for c in kag.CROPS)
    put("PLANT", (tile is None) and seed_ok)                                        # :417-429, :920
    put("WATER", is_plant and not tile["watered_today"])                                  # :431
    harvestable = is_dict and tile.get("yield_units", 0) > 0 and (
        (not is_plant) or (ts.day - tile["planted_day"] >= kag.CROPS[tile["crop"]]["first_yield_day"]))
    put("HARVEST", harvestable)                                                           # :446
    put("FERTILIZE", is_plant and inv.get("FERTILIZER", 0) >= 1)                           # :475
    put("DIG", (tile is not None) and not locked and not has_animal)                       # :484
    put("BUILD_COOP", tile is None)                                                        # :493
    put("BUILD_PASTURE", tile is None)                                                     # :498
    put("FEED", has_animal and not tile["fed_today"] and inv.get("WHEAT", 0) >= 1)          # :505
    put("COLLECT_FERTILIZER", has_animal and tile["fertilizer_available"])                  # :515
    put("CARE", has_animal and not tile["cared_today"])                                     # :524

    if locked:
        # Only the position-only ops survive on a LOCKED tile (:414).  PASS is never masked.
        keep = ("PASS", "DROP", "PICKUP", "PLACE")
        idxs = [(V.MACRO_VERB_INDEX if macro else V.VERB_INDEX)[k] for k in keep]
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[idxs] = True
        if macro:
            keep_mask[V.MV_MOVE] = True
        else:
            keep_mask[list(V.MOVE_IDS)] = True
        m &= keep_mask
        # ... except animal PLACE, which cannot match a LOCKED tile anyway (it is a string).
        put("PLACE", can_place_shed)
    return m


def item_mask_at(ts, idx, verb, xy=None, macro=False):
    """Legal items for `verb` at `xy`, as a mask over `N_ITEM_SLOTS`."""
    m = np.zeros(V.N_ITEM_SLOTS, dtype=bool)
    name = V.MACRO_VERBS[verb] if macro else V.VERBS[verb]
    if name == "MOVE" or not V.VERB_ITEMS.get(name):
        m[V.ITEM_NONE] = True
        return m
    if idx >= len(ts.positions):
        return m
    here = ts.positions[idx]
    x, y = (int(here[0]), int(here[1])) if xy is None else (int(xy[0]), int(xy[1]))
    inv = ts.inventories[idx]
    tile = _tile_at(ts, x, y)
    adjacent = (x, y) in SHED_ACCESS
    if name == "PICKUP":
        for it in V.ITEMS:
            m[V.ITEM_INDEX[it]] = adjacent and ts.shed.get(it, 0) > 0
    elif name == "PLACE":
        room = ts.shed_free() > 0
        for it in V.ITEMS:
            held = inv.get(it, 0) >= 1
            to_tile = (it in kag.ANIMALS and isinstance(tile, dict)
                       and tile.get("kind") == kag.ANIMALS[it]["structure"] and "animal" not in tile)
            m[V.ITEM_INDEX[it]] = held and (to_tile or (adjacent and room))
    elif name == "PLANT":
        for c in kag.CROPS:
            m[V.ITEM_INDEX[c]] = ts.seeds.get(c, 0) > 0
    return m


def qty_mask_at(ts, idx, verb, item, xy=None, macro=False):
    """Legal quantity buckets for `(verb, item)`."""
    name = V.MACRO_VERBS[verb] if macro else V.VERBS[verb]
    if name not in ("PICKUP", "PLACE") or item == V.ITEM_NONE:
        m = np.zeros(V.N_QTY, dtype=bool)
        m[0] = True
        return m
    it = V.ITEMS[item]
    if name == "PICKUP":
        return V.qty_mask(ts.shed.get(it, 0))
    return V.qty_mask(min(ts.inventories[idx].get(it, 0), ts.shed_free())
                      if idx < len(ts.inventories) else 0)


def market_masks(ts):
    """Op / item / quantity legality for the next order slot, under the running slot simulation."""
    op = np.zeros(V.N_MARKET_OPS, dtype=bool)
    item = np.zeros((V.N_MARKET_OPS, V.N_ITEM_SLOTS), dtype=bool)
    qty = np.zeros((V.N_MARKET_OPS, V.N_ITEM_SLOTS, V.N_QTY), dtype=bool)
    if ts.n_orders >= V.MAX_MARKET_ORDERS:            # extras are silently dropped (:551, :560)
        return op, item, qty

    room = ts.shed_free() > 0
    for it in kag.PRODUCTS:                                                        # SELL  :653-654
        have = ts.shed.get(it, 0)
        if have > 0:
            item[V.M_SELL, V.ITEM_INDEX[it]] = True
            qty[V.M_SELL, V.ITEM_INDEX[it]] = V.qty_mask(have)
    for c in kag.CROPS:                                                            # BUY_SEED :673
        if ts.money >= kag.CROPS[c]["seed"]:
            item[V.M_BUY_SEED, V.ITEM_INDEX[c]] = True
            qty[V.M_BUY_SEED, V.ITEM_INDEX[c]] = V.qty_mask(
                int(ts.money // kag.CROPS[c]["seed"]))
    for it in V.BUYABLE_PRODUCTS:                                            # BUY_PRODUCT :598,:662
        price = max(1, ts.price(it, offset=-1))
        if room and ts.money >= price:
            item[V.M_BUY_PRODUCT, V.ITEM_INDEX[it]] = True
            qty[V.M_BUY_PRODUCT, V.ITEM_INDEX[it]] = V.qty_mask(
                min(int(ts.money // price), ts.shed_free()))
    for a in kag.ANIMALS:                                                          # BUY_ANIMAL :679
        cost = kag.ANIMALS[a]["cost"]
        if room and ts.money >= cost:
            item[V.M_BUY_ANIMAL, V.ITEM_INDEX[a]] = True
            qty[V.M_BUY_ANIMAL, V.ITEM_INDEX[a]] = V.qty_mask(
                min(int(ts.money // cost), ts.shed_free()))
    if ts.money >= kag._hire_cost(ts.hires_today):                                 # HIRE :690-706
        op[V.M_HIRE] = True
        item[V.M_HIRE, V.ITEM_NONE] = True
        qty[V.M_HIRE, V.ITEM_NONE, 0] = True
    extra = len(ts.unlocked) - 1
    if extra < len(kag.LAND_ORDER) and ts.money >= kag.LAND_PRICES[extra]:         # BUY_LAND :712
        op[V.M_BUY_LAND] = True
        item[V.M_BUY_LAND, V.ITEM_NONE] = True
        qty[V.M_BUY_LAND, V.ITEM_NONE, 0] = True
    for o in (V.M_SELL, V.M_BUY_SEED, V.M_BUY_PRODUCT, V.M_BUY_ANIMAL):
        op[o] = bool(item[o].any())
    return op, item, qty


def compute_masks(obs, decisions_so_far=None, player=None):
    """The one entry point: `(observation, decisions so far) -> boolean arrays`.

    `decisions_so_far` is a `TurnState`; pass `None` for "nothing decided yet this turn", in which
    case one is built from `obs` (and `player` must be given, or `obs["player"]` is used).
    """
    ts = decisions_so_far
    if ts is None:
        p = player if player is not None else int(obs["player"])
        ts = TurnState(obs, p)

    unit_valid = np.zeros(V.MAX_UNITS, dtype=bool)
    unit_verb = np.zeros((V.MAX_UNITS, V.N_VERBS), dtype=bool)
    unit_item = np.zeros((V.MAX_UNITS, V.N_VERBS, V.N_ITEM_SLOTS), dtype=bool)
    for i in range(min(V.MAX_UNITS, ts.n_units)):
        unit_valid[i] = True
        unit_verb[i] = verb_mask_at(ts, i)
        for v in range(V.N_VERBS):
            if unit_verb[i, v]:
                unit_item[i, v] = item_mask_at(ts, i, v)
    mop, mitem, mqty = market_masks(ts)
    return Masks(unit_valid, unit_verb, unit_item, mop, mitem, mqty)


# --------------------------------------------------------------------------------------
# Assertion 3: the mask must never reject the expert
# --------------------------------------------------------------------------------------

def unit_action_legality(ts, idx, action):
    """`(verb_ok, item_ok, qty_ok)` for one expert unit action against the running state.

    Returns three separate flags so a failure says *which* rule is wrong rather than only that
    something is.  A mask that rejects the expert means our mask is wrong, not the expert
    (PLAN_BC Assertion 3) -- and a model trained under a leaky mask produces loss numbers that mean
    nothing at all.
    """
    try:
        vi, ii, _qb, qr, _extra = V.encode_unit_action(action)
    except V.VocabError:
        return False, False, False
    if idx >= len(ts.positions):
        return False, False, False
    vm = verb_mask_at(ts, idx)
    if not vm[vi]:
        return False, False, False
    im = item_mask_at(ts, idx, vi)
    if not im[ii]:
        return True, False, False
    if not V.VERB_TAKES_QTY[vi]:
        return True, True, True
    qm = qty_mask_at(ts, idx, vi, ii)
    return True, True, bool(qm[V.encode_qty(qr)]) or qr <= 0


def market_order_legality(ts, order):
    """`(op_ok, item_ok)` for one expert market order against the running slot simulation.

    Quantity is deliberately not checked: `SELL` settles `min(requested, shed)` (`kag.py:641-658`),
    so an oversized request is legal, not illegal.  Assertion 4 is where quantity is examined.
    """
    try:
        oi, ii, _qb, _qr, _extra = V.encode_market_order(order)
    except V.VocabError:
        return False, False
    mop, mitem, _mqty = market_masks(ts)
    if not mop[oi]:
        return False, False
    return True, bool(mitem[oi, ii])
