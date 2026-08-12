"""Pure-Python forward model of *our own farm* (PLAN2 P0).

Exists so the agent can roll candidate plans forward inside Kaggle's 1000 ms turn budget, where
`kagsim` is not importable. Models tiles, crops, animals, unit positions, unit inventories and the
shed. **Not** the market, the opponent, or money — those need the other player and are out of
scope by design.

Faithfulness is the whole point, so every branch cites the reference line it mirrors
(`kaggle_environments/envs/kaggriculture/kaggriculture.py`, pinned at 1.32.6 by
`tests/test_env_version.py`). Verified by differential test against kagsim in
`tests/test_forward_parity.py`.

Two things it deliberately does not predict, both documented rather than hidden:

* **Weed spawning** is stochastic (`weedSpawnChance`, `:823`). `weed_mode="none"` skips it and
  `weed_mode="all"` weeds every empty tile — the two deterministic extremes, which is how the
  parity test covers the branch without fighting the RNG.
* **Hiring** is a market action. `_end_of_day` really does clear `farm["hands"]` every day
  (`:867`), so hands are re-hired daily. `rehire` re-spawns a fixed number for rollouts; the parity
  test instead calls `sync_units()` to adopt the simulator's roster.

Speed measured at 572k–836k farm-steps/s for the skeleton of this loop (E32), against a 2,000/s
requirement.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, FARMER_MOVES

# Base prices, duplicated here rather than imported from `agent.engine`: P1 makes the engine import
# this module, and the reverse dependency would be a cycle.
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
              "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}


def _shed_access_tiles(board_size: int) -> list[tuple[int, int]]:
    """`:119` — four inner-corner tiles around the shed, NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _quadrant_of(x: int, y: int, board_size: int) -> str:
    """`:114`."""
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _default_spawn(board_size: int) -> tuple[int, int]:
    """`:148` — first shed-access tile inside the always-owned NW quadrant."""
    for tile in _shed_access_tiles(board_size):
        if _quadrant_of(tile[0], tile[1], board_size) == "NW":
            return tile
    return (0, 0)


def _new_plant(crop: str, day: int, turns_per_day: int) -> dict:
    """`:202`. Note `consecutive_unwatered` starts at 1: planting day counts as unwatered."""
    cd = CROPS[crop]
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 0 if cd["ongoing"] else 1,
        "max_lifespan_step": (
            -1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day
        ),
        "fertilized_until_day": -1,
    }


def _new_animal(animal: str, day: int) -> dict:
    """`:216`."""
    return {
        "kind": ANIMALS[animal]["structure"],
        "animal": animal,
        "placed_day": day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


class FarmModel:
    """One player's farm, steppable without the environment."""

    __slots__ = ("tiles", "units", "invs", "shed", "seeds", "day", "step_idx",
                 "board_size", "turns_per_day", "shed_capacity", "weed_mode", "rehire")

    def __init__(self, tiles, units, invs, shed, seeds, day, step_idx,
                 board_size=10, turns_per_day=24, shed_capacity=100,
                 weed_mode="none", rehire=None):
        self.tiles = tiles              # tiles[y][x]: None | "LOCKED" | dict
        self.units = units              # [ (x, y), ... ] index 0 is the farmer
        self.invs = invs                # per-unit dicts, insertion order is observable
        self.shed = shed
        self.seeds = seeds
        self.day = day
        self.step_idx = step_idx
        self.board_size = board_size
        self.turns_per_day = turns_per_day
        self.shed_capacity = shed_capacity
        self.weed_mode = weed_mode
        self.rehire = rehire

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_obs(cls, obs: dict, player: int | None = None, **kw) -> "FarmModel":
        player = obs["player"] if player is None else player
        farm = obs["farms"][player]
        priv = obs["private"]
        tiles = [
            [dict(t) if isinstance(t, dict) else t for t in row]
            for row in farm["tiles"]
        ]
        units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        invs = [dict(i) for i in priv["inventories"]]
        # `private["inventories"]` can be shorter than the unit list on the turn a hand spawns.
        while len(invs) < len(units):
            invs.append({})
        board = len(farm["tiles"])
        return cls(tiles, units, invs, dict(priv["shed"]), dict(priv.get("seeds", {})),
                   obs["day"], obs["day"] * kw.get("turns_per_day", 24) + obs["hour"],
                   board_size=board, **kw)

    def clone(self) -> "FarmModel":
        """Rollouts branch constantly, so this avoids `copy.deepcopy` deliberately."""
        return FarmModel(
            [[dict(t) if type(t) is dict else t for t in row] for row in self.tiles],
            list(self.units), [dict(i) for i in self.invs], dict(self.shed), dict(self.seeds),
            self.day, self.step_idx, self.board_size, self.turns_per_day,
            self.shed_capacity, self.weed_mode, self.rehire,
        )

    def sync_units(self, farm: dict, priv: dict) -> None:
        """Adopt the simulator's roster and inventories — hiring is market logic, not ours."""
        self.units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        self.invs = [dict(i) for i in priv["inventories"]]
        while len(self.invs) < len(self.units):
            self.invs.append({})

    # ------------------------------------------------------------- inventory

    @staticmethod
    def _inv_add(inv: dict, item: str, n: int = 1) -> None:
        """`:286`."""
        if n <= 0:
            return
        inv[item] = inv.get(item, 0) + n

    @staticmethod
    def _inv_take(inv: dict, item: str, n: int = 1) -> bool:
        """`:290`."""
        have = inv.get(item, 0)
        if have < n:
            return False
        if have == n:
            del inv[item]
        else:
            inv[item] = have - n
        return True

    def _is_shed_adjacent(self, pos) -> bool:
        return tuple(pos) in set(_shed_access_tiles(self.board_size))

    # ----------------------------------------------------------- unit action

    def apply_unit_action(self, idx: int, action) -> None:
        """`:299` — one unit's op. Invalid or illegal actions are silent no-ops."""
        if not isinstance(action, list) or not action:
            return
        op = action[0]
        if idx >= len(self.units):
            return
        fx, fy = self.units[idx]
        inv = self.invs[idx]
        bs = self.board_size

        if op in FARMER_MOVES:
            dx, dy = FARMER_MOVES[op]
            nx, ny = fx + dx, fy + dy
            if not (0 <= nx < bs and 0 <= ny < bs):
                return
            # `:315` — movement onto LOCKED is allowed; tile ops still no-op there.
            self.units[idx] = (nx, ny)
            return

        if op == "PASS":
            return

        tile = self.tiles[fy][fx]

        # `:326` — shed ops resolve BEFORE the LOCKED guard (1.32.6). Three of the four
        # shed-access tiles start LOCKED, so guarding first would make the shed unreachable.
        if op == "DROP":
            if not self._is_shed_adjacent((fx, fy)):
                return
            for item, n in list(inv.items()):
                if n <= 0:
                    del inv[item]
                    continue
                room = max(0, self.shed_capacity - sum(self.shed.values()))
                take = min(n, room)
                if take > 0:
                    self.shed[item] = self.shed.get(item, 0) + take
                del inv[item]
            return

        if op == "PICKUP":
            if not self._is_shed_adjacent((fx, fy)) or len(action) < 2:
                return
            item = action[1]
            n = int(action[2]) if len(action) >= 3 else 1     # `int()`, not a bare cast
            if n <= 0:
                return
            n = min(n, self.shed.get(item, 0))
            if n <= 0:
                return
            self.shed[item] -= n
            self._inv_add(inv, item, n)
            return

        if op == "PLACE":
            if len(action) < 2:
                return
            item = action[1]
            if (item in ANIMALS and isinstance(tile, dict)
                    and tile.get("kind") == ANIMALS[item]["structure"] and "animal" not in tile):
                if self._inv_take(inv, item, 1):
                    self.tiles[fy][fx] = _new_animal(item, self.day)
                return
            if self._is_shed_adjacent((fx, fy)):
                n = int(action[2]) if len(action) >= 3 else 1
                if n <= 0:
                    return
                n = min(n, inv.get(item, 0))
                if n <= 0:
                    return
                n = min(n, max(0, self.shed_capacity - sum(self.shed.values())))
                if n <= 0:
                    return
                inv[item] -= n
                if inv[item] == 0:
                    del inv[item]
                self.shed[item] = self.shed.get(item, 0) + n
            return

        # `:401` — everything below mutates the tile, so it must be owned.
        if tile == "LOCKED":
            return

        if op == "PLANT":
            if len(action) < 2:
                return
            crop = action[1]
            if crop not in CROPS or tile is not None or self.seeds.get(crop, 0) <= 0:
                return
            self.seeds[crop] -= 1
            self.tiles[fy][fx] = _new_plant(crop, self.day, self.turns_per_day)
            return

        if op == "WATER":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                return
            if tile["watered_today"]:
                return
            tile["watered_today"] = True
            cd = CROPS[tile["crop"]]
            if not cd["ongoing"]:
                # `:425` — one-shot crops accrue on the WATER op, inside a window, and the
                # fertilizer bonus doubles it.
                age = self.day - tile["planted_day"]
                if (cd["max_yield_day"] + 1) // 2 <= age <= cd["max_yield_day"]:
                    bonus = 2 if tile["fertilized_until_day"] >= self.day else 1
                    tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + bonus)
            return

        if op == "HARVEST":
            if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
                return
            if tile.get("kind") == "PLANT":
                cd = CROPS[tile["crop"]]
                if self.day - tile["planted_day"] < cd["first_yield_day"]:
                    return
                units = tile["yield_units"]
                tile["yield_units"] = 0
                self._inv_add(inv, tile["crop"], units)
                if not cd["ongoing"]:
                    self.tiles[fy][fx] = None
            elif "animal" in tile:
                units = tile["yield_units"]
                tile["yield_units"] = 0
                self._inv_add(inv, ANIMALS[tile["animal"]]["product"], units)
            return

        if op == "FERTILIZE":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                return
            if not self._inv_take(inv, "FERTILIZER", 1):
                return
            # `:468` — active for day, day+1, day+2 inclusive.
            tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), self.day + 2)
            return

        if op == "DIG":
            if tile is None:
                return
            if isinstance(tile, dict) and "animal" in tile:
                return          # `:475` — a placed animal is not removable by DIG
            self.tiles[fy][fx] = None
            return

        if op == "BUILD_COOP":
            if tile is None:
                self.tiles[fy][fx] = {"kind": "COOP"}
            return

        if op == "BUILD_PASTURE":
            if tile is None:
                self.tiles[fy][fx] = {"kind": "PASTURE"}
            return

        if op == "FEED":
            if not (isinstance(tile, dict) and "animal" in tile) or tile["fed_today"]:
                return
            if not self._inv_take(inv, "WHEAT", 1):
                return
            tile["fed_today"] = True
            return

        if op == "COLLECT_FERTILIZER":
            if not (isinstance(tile, dict) and "animal" in tile):
                return
            if not tile["fertilizer_available"]:
                return
            tile["fertilizer_available"] = False
            self._inv_add(inv, "FERTILIZER", 1)
            return

        if op == "CARE":
            if not (isinstance(tile, dict) and "animal" in tile) or tile["cared_today"]:
                return
            tile["cared_today"] = True
            return

    # ------------------------------------------------------------ daily work

    def _decay_plants(self) -> None:
        """`:739` — past its lifespan a plant sheds a unit every other step, then weeds."""
        step = self.step_idx
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                mls = tile["max_lifespan_step"]
                if mls < 0 or step < mls or (step - mls) % 2 != 0:
                    continue
                tile["yield_units"] -= 1
                if tile["yield_units"] <= 0:
                    row[x] = {"kind": "WEED"}

    def _daily_refresh_plants(self) -> None:
        """`:756` — death at 2 dry days, then ongoing-crop accrual."""
        next_day = self.day + 1
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                was_watered = tile["watered_today"]
                tile["consecutive_unwatered"] = 0 if was_watered else tile["consecutive_unwatered"] + 1
                tile["watered_today"] = False
                if tile["consecutive_unwatered"] >= 2:
                    row[x] = {"kind": "WEED"}
                    continue
                cd = CROPS[tile["crop"]]
                if not cd["ongoing"]:
                    continue
                dsf = next_day - tile["planted_day"] - cd["first_yield_day"]
                if dsf < 0 or dsf % cd["interval"] != 0:
                    continue
                count = dsf // cd["interval"] + 1
                if count > cd["max_yield"]:
                    continue
                # `:786` — the fertilizer bonus additionally requires the tile to have been watered.
                fertilized = was_watered and tile.get("fertilized_until_day", -1) >= self.day
                tile["yield_units"] = min(cd["max_yield"],
                                          tile["yield_units"] + (2 if fertilized else 1))
                if count == cd["max_yield"]:
                    tile["max_lifespan_step"] = (next_day + 1) * self.turns_per_day

    def _daily_refresh_animals(self) -> None:
        """`:792` — escape at 2 unfed days; care bonus only lands on a fed production day."""
        next_day = self.day + 1
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if not (isinstance(tile, dict) and "animal" in tile):
                    continue
                tile["consecutive_unfed"] = 0 if tile["fed_today"] else tile["consecutive_unfed"] + 1
                if tile["consecutive_unfed"] >= 2:
                    row[x] = {"kind": ANIMALS[tile["animal"]]["structure"]}
                    continue
                a = ANIMALS[tile["animal"]]
                dsf = next_day - tile["placed_day"] - a["first_yield_day"]
                if dsf >= 0 and dsf % a["interval"] == 0:
                    bonus = tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0
                    tile["yield_units"] = min(a["max_held"], tile["yield_units"] + 1 + bonus)
                    tile["pending_care_bonus"] = 0
                if tile["cared_today"] and tile["fed_today"]:
                    tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
                # `:818` — unconditional: an animal makes fertilizer even unfed and uncared for.
                tile["fertilizer_available"] = True
                tile["fed_today"] = False
                tile["cared_today"] = False

    def _drop_inventories_to_shed(self) -> None:
        """`:830` — overflow is discarded, and the entry is removed either way."""
        for inv in self.invs:
            for item, n in list(inv.items()):
                if n <= 0:
                    del inv[item]
                    continue
                room = max(0, self.shed_capacity - sum(self.shed.values()))
                take = min(n, room)
                if take > 0:
                    self.shed[item] = self.shed.get(item, 0) + take
                del inv[item]

    def _end_of_day(self) -> None:
        """`:847`, minus the market and the RNG."""
        self._daily_refresh_plants()
        self._daily_refresh_animals()
        if self.weed_mode == "all":                    # `weedSpawnChance = 1.0`
            for row in self.tiles:
                for x, tile in enumerate(row):
                    if tile is None:
                        row[x] = {"kind": "WEED"}
        self._drop_inventories_to_shed()
        # `:866-867` — the roster really is cleared every day and re-hired through the market.
        # `rehire=None` is faithful to that. Rollouts pass `rehire="keep"` instead, because an
        # agent that re-hires every morning effectively carries its workforce across the boundary,
        # and a rollout that watched its hands evaporate would mis-rank every candidate.
        if self.rehire is None:
            n_hands = 0
        elif self.rehire == "keep":
            n_hands = len(self.units) - 1
        else:
            n_hands = int(self.rehire)
        spawn = _default_spawn(self.board_size)
        self.units = [spawn] + [spawn] * max(0, n_hands)
        self.invs = [{} for _ in self.units]
        self.day += 1

    # ---------------------------------------------------------------- value

    def score(self, cash: float = 0.0, prices: dict | None = None,
              unripe: float = 0.5, season_end_day: int = 30) -> float:
        """Standing value of the farm: what it is worth if the season stopped being kind to us.

        Deliberately simple and explainable — the rollout is what supplies the intelligence, and a
        clever value function is a second thing that can be wrong. Three parts:

        * goods already in hand (shed, unit inventories) at base price;
        * yield already accrued on tiles, which only needs collecting;
        * yield a tile can still produce before the season ends, discounted by `unripe` because it
          has to be watered, fed and harvested first.

        `unripe = 0` scores only what exists, `1` treats future yield as banked. The default of
        0.5 is a starting point to be tuned, not a claim.

        **Known limitation: goods are valued at BASE_PRICE, which is not what they will sell for.**
        Price falls as market inventory rises, and since 1.32.6 the town absorbs far less: the
        centre takes ~30 units of a product per season and shops are drawn with replacement, so
        MELON has no shop buyer in any game and WOOL none in 36% of them (E33). This function will
        happily price a field of melons at $250/unit into a market that cannot take them.

        That is tolerable for what P1 uses it for -- comparing candidate *worker assignments*,
        where both options involve the same crops and the mispricing cancels. It is **not** safe
        for deciding what to plant or which animals to keep; P1.5-A tried that and lost (E35).
        Pass `prices=` a demand-aware table (see `Engine.town_drain_per_day`) if that changes, and
        measure it -- three plausible-looking improvements have cost money already.
        """
        px = prices or BASE_PRICE
        v = float(cash)
        for item, n in self.shed.items():
            v += n * px.get(item, 0)
        for inv in self.invs:
            for item, n in inv.items():
                v += n * px.get(item, 0)

        # Seeds are purchased capacity: they cannot be sold, but a farm holding seeds is not the
        # same as one that has spent everything. Valued at cost, which is conservative -- a seed in
        # the ground is worth its crop, a seed in the bag is worth what it took to buy.
        for crop, n in self.seeds.items():
            if n > 0:
                v += n * CROPS[crop]["seed"]

        days_left = max(0, season_end_day - self.day)
        for row in self.tiles:
            for t in row:
                if type(t) is not dict:
                    continue
                if t.get("kind") == "PLANT":
                    cd = CROPS[t["crop"]]
                    price = px.get(t["crop"], 0)
                    v += t["yield_units"] * price
                    age = self.day - t["planted_day"]
                    if cd["ongoing"]:
                        produced = max(0, (age - cd["first_yield_day"]) // cd["interval"] + 1) \
                            if age >= cd["first_yield_day"] else 0
                        left = max(0, cd["max_yield"] - produced)
                        # Also bounded by how many production ticks the season still allows.
                        left = min(left, days_left // max(1, cd["interval"]))
                    else:
                        # One-shot crops stop accruing after max_yield_day (`:428`).
                        left = (max(0, cd["max_yield"] - t["yield_units"])
                                if age <= cd["max_yield_day"] else 0)
                    v += unripe * left * price
                elif "animal" in t:
                    a = ANIMALS[t["animal"]]
                    price = px.get(a["product"], 0)
                    v += t["yield_units"] * price
                    if t["fertilizer_available"]:
                        v += px.get("FERTILIZER", 0)
                    # Fertilizer accrues daily and unconditionally (`:818`); the product does not.
                    v += unripe * days_left * px.get("FERTILIZER", 0)
                    ticks = days_left // max(1, a["interval"])
                    v += unripe * min(ticks, a["max_held"]) * price
        return v

    # ----------------------------------------------------------------- step

    def step(self, unit_actions: list) -> None:
        """One turn: `:900-933`, farm only."""
        # `:907` — atomic PLANT validation: if requests for a crop exceed seeds held, ALL of that
        # crop's PLANT requests are dropped, not just the excess.
        demand: dict[str, int] = {}
        for a in unit_actions:
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                demand[a[1]] = demand.get(a[1], 0) + 1
        blocked = {c for c, n in demand.items() if n > self.seeds.get(c, 0)}

        for idx, a in enumerate(unit_actions):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                a = ["PASS"]
            self.apply_unit_action(idx, a)

        self._decay_plants()
        if (self.step_idx + 1) % self.turns_per_day == 0:
            self._end_of_day()
        self.step_idx += 1
