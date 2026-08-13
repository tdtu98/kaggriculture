# Competitive Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and empirically tune a self-contained 00_baseline Kaggriculture agent that maximizes final money with a compact Goose-and-Wheat farm and substantially outperforms demo_agent.py and starter.

**Architecture:** Keep all competition behavior in 00_baseline/main.py, with observation-derived market and field planners that share no hidden persistent state. Test decisions through the public agent(obs) contract, verify source-file loading against the installed Kaggle loader, and measure full-season results in a separate benchmark runner before documenting the selected strategy.

**Tech Stack:** Python 3.11+, standard-library unittest, argparse, importlib, statistics, and the installed kaggle-environments package.

## Global Constraints

- 00_baseline/main.py must be self-contained and expose agent(obs) as its final module-level callable.
- Every action must contain farmer, hands, and market; emit exactly one action per observed hand and at most ten market orders.
- Optimize mean final money over default 720-turn games while requiring a 100% win rate against demo_agent.py and starter on seeds 0 through 9 in both seats.
- The selected baseline must reach at least twice demo_agent.py's average final money over the same benchmark.
- Use only public farms, the current player's private observation, current market data, and current town data.
- Do not print from submission code.
- Keep benchmark and diagnostics outside main.py.
- Create and maintain 00_baseline/STRATEGY_SUMMARY.md with implemented behavior and measured results.
- This directory is not a Git repository; record completed checklist items instead of committing.

---

### Task 1: Submission contract and observation helpers

**Files:**
- Create: 00_baseline/main.py
- Create: 00_baseline/test_main.py

**Interfaces:**
- Consumes: Kaggriculture observation dictionaries.
- Produces: agent(obs: dict) -> dict with farmer, hands, and market.
- Provides internally: _farm(obs), _inventory(obs, unit_index), _iter_tiles(farm), _shed_total(private), and _pass_action(obs).

- [ ] **Step 1: Write the failing contract tests**

Create 00_baseline/test_main.py with a path-based loader because a directory beginning with digits is not a normal Python package name:

    import importlib.util
    import unittest
    from pathlib import Path

    BASELINE_DIR = Path(__file__).resolve().parent
    SPEC = importlib.util.spec_from_file_location(
        "baseline_main", BASELINE_DIR / "main.py"
    )
    baseline = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(baseline)


    def observation(*, hands=None, hour=0, day=0, step=0, money=3000):
        hands = [] if hands is None else hands
        tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)]
                 for y in range(10)]
        return {
            "player": 0,
            "step": step,
            "day": day,
            "hour": hour,
            "farms": [{
                "money": money,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": hands,
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
            }, {
                "money": 3000,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
            }],
            "private": {
                "shed": {
                    "WHEAT": 0, "EGG": 0, "FERTILIZER": 0,
                    "GOOSE": 0,
                },
                "seeds": {"WHEAT": 0},
                "inventories": [{} for _ in range(1 + len(hands))],
            },
            "market": {
                "prices": {"WHEAT": 25, "EGG": 50, "FERTILIZER": 100},
                "inventory": {
                    "WHEAT": 10000, "EGG": 10000, "FERTILIZER": 10000,
                },
            },
            "town": {"unlocked_shops": []},
        }


    class SubmissionContractTests(unittest.TestCase):
        def test_returns_complete_action_for_every_unit(self):
            obs = observation(hands=[[3, 4], [4, 3]])
            action = baseline.agent(obs)

            self.assertEqual(set(action), {"farmer", "hands", "market"})
            self.assertEqual(len(action["hands"]), 2)
            self.assertTrue(all(isinstance(a, list) and a for a in action["hands"]))
            self.assertLessEqual(len(action["market"]), 10)

        def test_does_not_mutate_observation(self):
            import copy

            obs = observation()
            before = copy.deepcopy(obs)
            baseline.agent(obs)
            self.assertEqual(obs, before)


    if __name__ == "__main__":
        unittest.main()

- [ ] **Step 2: Run the contract tests and verify RED**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: import failure because 00_baseline/main.py does not exist.

- [ ] **Step 3: Implement the minimal submission skeleton**

Create 00_baseline/main.py with constants first, helpers next, and agent last:

    """Competitive Kaggriculture Goose-and-Wheat baseline."""

    MAX_MARKET_ORDERS = 10
    SHED_CAPACITY = 100
    TURNS_PER_DAY = 24


    def _farm(obs):
        return obs["farms"][obs["player"]]


    def _inventory(obs, unit_index):
        inventories = obs.get("private", {}).get("inventories", [])
        return inventories[unit_index] if unit_index < len(inventories) else {}


    def _iter_tiles(farm):
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                yield x, y, tile


    def _shed_total(private):
        return sum(max(0, int(n)) for n in private.get("shed", {}).values())


    def _pass_action(obs):
        hand_count = len(_farm(obs).get("hands", []))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in range(hand_count)],
            "market": [],
        }


    def agent(obs):
        """Return a valid action derived only from the current observation."""
        return _pass_action(obs)

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: two tests pass.

- [ ] **Step 5: Record Task 1**

Mark Task 1 complete in this plan. Do not create a commit because the workspace is not a Git repository.

---

### Task 2: Bounded market and opening policy

**Files:**
- Modify: 00_baseline/main.py
- Modify: 00_baseline/test_main.py

**Interfaces:**
- Consumes: _farm(obs), _shed_total(private), current day/hour, current prices, money, animals on public tiles, and current private stocks.
- Produces: _market_orders(obs) -> list[list] capped at MAX_MARKET_ORDERS.
- Uses tunable constants: OPENING_GEESE, TARGET_GEESE, DAILY_HANDS, OPENING_FEED, OPENING_SEEDS, LAST_GOOSE_BUY_DAY, FEED_DAYS_RESERVE.

- [ ] **Step 1: Write failing market-policy tests**

Append these tests, using observation from Task 1:

    class MarketPolicyTests(unittest.TestCase):
        def test_opening_fits_hires_and_three_purchases_in_ten_orders(self):
            action = baseline.agent(observation(day=0, hour=0, money=3000))
            market = action["market"]

            self.assertLessEqual(len(market), 10)
            self.assertEqual(sum(order == ["HIRE"] for order in market), 7)
            self.assertIn(["BUY_ANIMAL", "GOOSE", 6], market)
            self.assertIn(["BUY_PRODUCT", "WHEAT", 24], market)
            self.assertIn(["BUY_SEED", "WHEAT", 12], market)

        def test_sells_eggs_and_profitable_fertilizer_from_shed(self):
            obs = observation(day=5, hour=2, money=2000)
            obs["private"]["shed"]["EGG"] = 5
            obs["private"]["shed"]["FERTILIZER"] = 3

            market = baseline.agent(obs)["market"]

            self.assertIn(["SELL", "EGG", 5], market)
            self.assertIn(["SELL", "FERTILIZER", 3], market)

        def test_does_not_buy_goose_when_shed_is_full_or_season_is_late(self):
            full = observation(day=5, hour=2, money=10000)
            full["private"]["shed"]["WHEAT"] = 100
            late = observation(day=25, hour=2, money=10000)

            for obs in (full, late):
                orders = baseline.agent(obs)["market"]
                self.assertFalse(any(o[:2] == ["BUY_ANIMAL", "GOOSE"] for o in orders))

        def test_market_orders_are_always_capped(self):
            obs = observation(day=4, hour=0, money=10000)
            obs["private"]["shed"].update({"EGG": 20, "FERTILIZER": 20})
            self.assertLessEqual(len(baseline.agent(obs)["market"]), 10)

- [ ] **Step 2: Run market tests and verify RED**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: failures because agent currently returns no market orders.

- [ ] **Step 3: Implement market inventory counts and ordered policy**

Insert these constants and helpers before agent:

    OPENING_GEESE = 6
    TARGET_GEESE = 8
    DAILY_HANDS = 7
    OPENING_FEED = 24
    OPENING_SEEDS = 12
    LAST_GOOSE_BUY_DAY = 20
    FEED_DAYS_RESERVE = 4
    GOOSE_COST = 300
    WHEAT_SEED_COST = 10


    def _animal_count(farm, animal):
        return sum(
            1 for _, _, tile in _iter_tiles(farm)
            if isinstance(tile, dict) and tile.get("animal") == animal
        )


    def _all_carried(private, item):
        return sum(inv.get(item, 0) for inv in private.get("inventories", []))


    def _append_order(orders, order):
        if len(orders) < MAX_MARKET_ORDERS:
            orders.append(order)


    def _market_orders(obs):
        farm = _farm(obs)
        private = obs.get("private", {})
        shed = private.get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        orders = []

        if hour == 0:
            missing_hands = max(0, DAILY_HANDS - int(farm.get("hires_today", 0)))
            for _ in range(missing_hands):
                _append_order(orders, ["HIRE"])

        if day == 0 and hour == 0:
            _append_order(orders, ["BUY_ANIMAL", "GOOSE", OPENING_GEESE])
            _append_order(orders, ["BUY_PRODUCT", "WHEAT", OPENING_FEED])
            _append_order(orders, ["BUY_SEED", "WHEAT", OPENING_SEEDS])
            return orders[:MAX_MARKET_ORDERS]

        eggs = max(0, int(shed.get("EGG", 0)))
        if eggs:
            _append_order(orders, ["SELL", "EGG", eggs])

        fertilizer = max(0, int(shed.get("FERTILIZER", 0)))
        wheat_price = max(1, int(prices.get("WHEAT", 25)))
        fertilizer_price = max(1, int(prices.get("FERTILIZER", 100)))
        if fertilizer and fertilizer_price > 2 * wheat_price:
            _append_order(orders, ["SELL", "FERTILIZER", fertilizer])

        geese = _animal_count(farm, "GOOSE") + int(shed.get("GOOSE", 0))
        feed = int(shed.get("WHEAT", 0)) + _all_carried(private, "WHEAT")
        needed_feed = geese * FEED_DAYS_RESERVE
        room = max(0, SHED_CAPACITY - _shed_total(private))
        if geese and feed < needed_feed and room:
            _append_order(
                orders,
                ["BUY_PRODUCT", "WHEAT", min(room, needed_feed - feed)],
            )

        seeds = int(private.get("seeds", {}).get("WHEAT", 0))
        if seeds < OPENING_SEEDS // 2 and farm.get("money", 0) >= WHEAT_SEED_COST * 6:
            _append_order(orders, ["BUY_SEED", "WHEAT", 6])

        desired = OPENING_GEESE if day < 5 else TARGET_GEESE
        if (
            day <= LAST_GOOSE_BUY_DAY
            and geese < desired
            and room > 0
            and farm.get("money", 0) >= GOOSE_COST + needed_feed * wheat_price
        ):
            affordable = int(
                (farm["money"] - needed_feed * wheat_price) // GOOSE_COST
            )
            quantity = min(desired - geese, room, max(0, affordable))
            if quantity:
                _append_order(orders, ["BUY_ANIMAL", "GOOSE", quantity])

        return orders[:MAX_MARKET_ORDERS]

Change agent so its pass-shaped result uses _market_orders(obs):

    def agent(obs):
        action = _pass_action(obs)
        action["market"] = _market_orders(obs)
        return action

- [ ] **Step 4: Run market and contract tests**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: all six tests pass.

- [ ] **Step 5: Record Task 2**

Mark Task 2 complete in this plan.

---

### Task 3: Spatial field-task planner

**Files:**
- Modify: 00_baseline/main.py
- Modify: 00_baseline/test_main.py

**Interfaces:**
- Produces: _field_actions(obs) -> tuple[list, list[list]] for farmer and hands.
- Internal task shape: {"priority": int, "target": (x, y), "action": list, "requires": str | None, "key": tuple}.
- Provides: _coop_targets(desired), _wheat_targets(desired), _build_tasks(obs), _compatible(task, inventory), _move_toward(position, target), and _assign_tasks(obs, tasks).

- [ ] **Step 1: Write failing field-priority tests**

Append:

    def plant(crop="WHEAT", *, planted_day=0, watered=False, yield_units=1):
        return {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": planted_day,
            "watered_today": watered,
            "consecutive_unwatered": 0,
            "yield_units": yield_units,
            "max_lifespan_step": 999,
            "fertilized_until_day": -1,
        }


    def goose(*, fed=False, cared=False, fertilizer=False, yield_units=0):
        return {
            "kind": "COOP",
            "animal": "GOOSE",
            "placed_day": 0,
            "yield_units": yield_units,
            "fed_today": fed,
            "consecutive_unfed": 0,
            "cared_today": cared,
            "fertilizer_available": fertilizer,
            "pending_care_bonus": 0,
        }


    class FieldPlannerTests(unittest.TestCase):
        def test_waters_new_plant_before_other_work(self):
            obs = observation(day=2, hour=4)
            obs["farms"][0]["farmer"] = [0, 0]
            obs["farms"][0]["tiles"][0][0] = plant(planted_day=2)
            self.assertEqual(baseline.agent(obs)["farmer"], ["WATER"])

        def test_feeds_goose_when_worker_carries_wheat(self):
            obs = observation(day=2, hour=4)
            obs["farms"][0]["tiles"][4][4] = goose()
            obs["private"]["inventories"][0]["WHEAT"] = 1
            self.assertEqual(baseline.agent(obs)["farmer"], ["FEED"])

        def test_services_two_distinct_tiles_without_duplicate_action(self):
            obs = observation(day=2, hour=4, hands=[[1, 0]])
            obs["farms"][0]["farmer"] = [0, 0]
            obs["farms"][0]["tiles"][0][0] = plant(planted_day=1)
            obs["farms"][0]["tiles"][0][1] = plant(planted_day=1)

            action = baseline.agent(obs)

            self.assertEqual(action["farmer"], ["WATER"])
            self.assertEqual(action["hands"], [["WATER"]])

        def test_does_not_overrequest_shared_seed(self):
            obs = observation(day=3, hour=5, hands=[[1, 0]])
            obs["farms"][0]["farmer"] = [0, 0]
            obs["private"]["seeds"]["WHEAT"] = 1

            action = baseline.agent(obs)
            actions = [action["farmer"], *action["hands"]]
            plants = [a for a in actions if a[:2] == ["PLANT", "WHEAT"]]
            self.assertEqual(len(plants), 1)

        def test_does_not_plant_at_last_hour(self):
            safe = observation(day=3, hour=22)
            late = observation(day=3, hour=23)
            for obs in (safe, late):
                obs["farms"][0]["farmer"] = [0, 0]
                obs["private"]["seeds"]["WHEAT"] = 10
            self.assertEqual(baseline.agent(safe)["farmer"], ["PLANT", "WHEAT"])
            self.assertNotEqual(baseline.agent(late)["farmer"], ["PLANT", "WHEAT"])

        def test_harvests_full_goose_after_daily_needs(self):
            obs = observation(day=8, hour=5)
            obs["farms"][0]["tiles"][4][4] = goose(
                fed=True, cared=True, fertilizer=False, yield_units=4
            )
            self.assertEqual(baseline.agent(obs)["farmer"], ["HARVEST"])

- [ ] **Step 2: Run field tests and verify RED**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: field actions remain PASS, so watering, feeding, and harvesting tests fail.

- [ ] **Step 3: Implement deterministic targets and task construction**

Insert before agent:

    COOP_ORDER = (
        (4, 4), (3, 4), (4, 3), (3, 3),
        (2, 4), (4, 2), (2, 3), (3, 2),
        (1, 4), (4, 1),
    )
    NW_TILES = tuple(
        sorted(
            ((x, y) for y in range(5) for x in range(5)),
            key=lambda p: (abs(4 - p[0]) + abs(4 - p[1]), -p[1], -p[0]),
        )
    )


    def _desired_geese(obs):
        return OPENING_GEESE if obs.get("day", 0) < 5 else TARGET_GEESE


    def _coop_targets(desired):
        return COOP_ORDER[:desired]


    def _wheat_targets(desired):
        coops = set(_coop_targets(desired))
        return tuple(p for p in NW_TILES if p not in coops)


    def _task(priority, target, action, key, requires=None):
        return {
            "priority": priority,
            "target": target,
            "action": action,
            "requires": requires,
            "key": key,
        }


    def _shed_access(board_size=10):
        half = board_size // 2
        return (
            (half - 1, half - 1), (half, half - 1),
            (half - 1, half), (half, half),
        )


    def _build_tasks(obs):
        farm = _farm(obs)
        private = obs.get("private", {})
        tiles = farm["tiles"]
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        desired = _desired_geese(obs)
        tasks = []
        seed_slots = int(private.get("seeds", {}).get("WHEAT", 0))

        for x, y, tile in _iter_tiles(farm):
            if not isinstance(tile, dict) or "animal" not in tile:
                continue
            target = (x, y)
            if not tile.get("fed_today", False):
                tasks.append(_task(0, target, ["FEED"], ("feed", target), "WHEAT"))
            elif not tile.get("cared_today", False):
                tasks.append(_task(2, target, ["CARE"], ("care", target)))
            elif tile.get("yield_units", 0) >= 4:
                tasks.append(_task(3, target, ["HARVEST"], ("animal_harvest", target)))
            elif tile.get("fertilizer_available", False):
                tasks.append(
                    _task(4, target, ["COLLECT_FERTILIZER"], ("fertilizer", target))
                )
            elif tile.get("yield_units", 0) >= 2:
                tasks.append(_task(5, target, ["HARVEST"], ("animal_harvest", target)))

        unfed = sum(
            1 for _, _, tile in _iter_tiles(farm)
            if isinstance(tile, dict)
            and tile.get("animal") == "GOOSE"
            and not tile.get("fed_today", False)
        )
        carried_feed = _all_carried(private, "WHEAT")
        pickup_feed = min(
            max(0, unfed - carried_feed),
            int(private.get("shed", {}).get("WHEAT", 0)),
        )
        access = _shed_access(len(tiles))
        for index in range(pickup_feed):
            target = access[index % len(access)]
            tasks.append(
                _task(1, target, ["PICKUP", "WHEAT", 1], ("feed_pickup", index))
            )

        coop_targets = _coop_targets(desired)
        empty_coops = []
        for target in coop_targets:
            x, y = target
            tile = tiles[y][x]
            if tile is None:
                tasks.append(_task(6, target, ["BUILD_COOP"], ("build", target)))
            elif isinstance(tile, dict) and tile.get("kind") == "COOP" and "animal" not in tile:
                empty_coops.append(target)
                tasks.append(
                    _task(5, target, ["PLACE", "GOOSE"], ("place", target), "GOOSE")
                )

        carried_geese = _all_carried(private, "GOOSE")
        pickup_geese = min(
            max(0, len(empty_coops) - carried_geese),
            int(private.get("shed", {}).get("GOOSE", 0)),
        )
        for index in range(pickup_geese):
            target = access[index % len(access)]
            tasks.append(
                _task(5, target, ["PICKUP", "GOOSE", 1], ("goose_pickup", index))
            )

        for target in _wheat_targets(desired):
            x, y = target
            tile = tiles[y][x]
            if tile == "LOCKED":
                continue
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append(_task(7, target, ["DIG"], ("dig", target)))
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if tile.get("crop") != "WHEAT":
                    continue
                age = day - int(tile.get("planted_day", day))
                if not tile.get("watered_today", False):
                    priority = 0 if tile.get("planted_day") == day or hour >= 20 else 1
                    tasks.append(_task(priority, target, ["WATER"], ("water", target)))
                elif age >= 4 and tile.get("yield_units", 0) > 0:
                    tasks.append(_task(4, target, ["HARVEST"], ("crop_harvest", target)))
            elif tile is None and seed_slots > 0 and hour < TURNS_PER_DAY - 1:
                tasks.append(_task(8, target, ["PLANT", "WHEAT"], ("plant", target)))
                seed_slots -= 1

        return tasks

- [ ] **Step 4: Implement compatibility, global greedy assignment, movement, and local deposit**

Insert:

    def _distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def _move_toward(position, target):
        x, y = position
        tx, ty = target
        if x < tx:
            return ["EAST"]
        if x > tx:
            return ["WEST"]
        if y < ty:
            return ["SOUTH"]
        if y > ty:
            return ["NORTH"]
        return ["PASS"]


    def _compatible(task, inventory):
        required = task.get("requires")
        return required is None or inventory.get(required, 0) > 0


    def _assign_tasks(obs, tasks):
        farm = _farm(obs)
        private = obs.get("private", {})
        positions = [farm["farmer"], *farm.get("hands", [])]
        inventories = private.get("inventories", [])
        actions = [["PASS"] for _ in positions]
        assigned_units = set()
        assigned_tasks = set()
        edges = []

        for unit_index, position in enumerate(positions):
            inventory = inventories[unit_index] if unit_index < len(inventories) else {}
            for task_index, task in enumerate(tasks):
                if _compatible(task, inventory):
                    edges.append(
                        (
                            task["priority"],
                            _distance(position, task["target"]),
                            unit_index,
                            task_index,
                        )
                    )

        for _, distance, unit_index, task_index in sorted(edges):
            if unit_index in assigned_units or task_index in assigned_tasks:
                continue
            task = tasks[task_index]
            position = positions[unit_index]
            actions[unit_index] = (
                task["action"] if distance == 0
                else _move_toward(position, task["target"])
            )
            assigned_units.add(unit_index)
            assigned_tasks.add(task_index)

        access = set(_shed_access(len(farm["tiles"])))
        for unit_index, position in enumerate(positions):
            if unit_index in assigned_units or tuple(position) not in access:
                continue
            inventory = inventories[unit_index] if unit_index < len(inventories) else {}
            sale_items = sum(
                n for item, n in inventory.items()
                if item in {"EGG", "FERTILIZER"}
            )
            if sale_items:
                actions[unit_index] = ["DROP"]

        return actions[0], actions[1:]


    def _field_actions(obs):
        return _assign_tasks(obs, _build_tasks(obs))

Update agent:

    def agent(obs):
        farmer, hands = _field_actions(obs)
        return {
            "farmer": farmer,
            "hands": hands,
            "market": _market_orders(obs),
        }

- [ ] **Step 5: Run all unit tests**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: all twelve tests pass.

- [ ] **Step 6: Record Task 3**

Mark Task 3 complete in this plan.

---

### Task 4: Live loader and episode integration

**Files:**
- Modify: 00_baseline/test_main.py
- Modify: 00_baseline/main.py only if a failing integration test identifies a submission defect.

**Interfaces:**
- Consumes: installed kaggle_environments.make.
- Produces: successful loading from 00_baseline/main.py and valid terminal states.

- [ ] **Step 1: Add loader and short-episode integration tests**

Append:

    class KaggleIntegrationTests(unittest.TestCase):
        def test_source_loader_selects_agent(self):
            from kaggle_environments.agent import get_last_callable

            raw = (BASELINE_DIR / "main.py").read_text(encoding="utf-8")
            loaded = get_last_callable(raw, path=str(BASELINE_DIR / "main.py"))
            self.assertEqual(loaded.__name__, "agent")

        def test_short_match_reaches_done_with_file_agent(self):
            from kaggle_environments import make

            env = make(
                "kaggriculture",
                configuration={"episodeSteps": 48, "seed": 7},
                debug=True,
            )
            env.run([str(BASELINE_DIR / "main.py"), "pass"])

            self.assertTrue(all(state.status == "DONE" for state in env.steps[-1]))
            self.assertGreater(env.steps[-1][0].reward, 0)

- [ ] **Step 2: Run integration tests**

Run: .venv/bin/python -m unittest -v 00_baseline/test_main.py

Expected: both tests pass. If either fails, preserve the failing test, use systematic debugging to identify the root cause, make the smallest submission-only correction, and rerun this exact command.

- [ ] **Step 3: Run the complete suite**

Run: .venv/bin/python -m unittest -v test_demo_agent.py test_observer_agent.py 00_baseline/test_main.py

Expected: all existing demo, observer, baseline unit, and integration tests pass.

- [ ] **Step 4: Record Task 4**

Mark Task 4 complete in this plan.

---

### Task 5: Reproducible full-season benchmark

**Files:**
- Create: 00_baseline/benchmark.py
- Modify: 00_baseline/test_main.py

**Interfaces:**
- Produces: load_agent(path), run_match(left, right, seed), summarize(scores), and CLI main(argv=None).
- CLI defaults: seeds 0 through 9, opponents demo and starter, both seats, episodeSteps 720.
- Output: one concise table with games, wins, average, median, minimum, maximum, opponent average, and average margin.

- [ ] **Step 1: Write failing benchmark-helper tests**

Append:

    class BenchmarkTests(unittest.TestCase):
        def test_summary_reports_profit_and_margin(self):
            benchmark_spec = importlib.util.spec_from_file_location(
                "baseline_benchmark", BASELINE_DIR / "benchmark.py"
            )
            benchmark = importlib.util.module_from_spec(benchmark_spec)
            benchmark_spec.loader.exec_module(benchmark)

            summary = benchmark.summarize([(8000, 3400), (9000, 3500)])

            self.assertEqual(summary["games"], 2)
            self.assertEqual(summary["wins"], 2)
            self.assertEqual(summary["average"], 8500)
            self.assertEqual(summary["median"], 8500)
            self.assertEqual(summary["minimum"], 8000)
            self.assertEqual(summary["maximum"], 9000)
            self.assertEqual(summary["opponent_average"], 3450)
            self.assertEqual(summary["average_margin"], 5050)

- [ ] **Step 2: Run benchmark test and verify RED**

Run: python3.12 -m unittest -v 00_baseline/test_main.py

Expected: import failure because 00_baseline/benchmark.py does not exist.

- [ ] **Step 3: Implement benchmark.py**

Create:

    """Benchmark 00_baseline over deterministic full Kaggriculture seasons."""

    import argparse
    import importlib.util
    from pathlib import Path
    from statistics import mean, median

    from kaggle_environments import make

    ROOT = Path(__file__).resolve().parents[1]
    BASELINE_PATH = Path(__file__).resolve().parent / "main.py"
    DEMO_PATH = ROOT / "demo_agent.py"


    def load_agent(path):
        path = Path(path)
        spec = importlib.util.spec_from_file_location(path.stem + "_loaded", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.agent


    def run_match(left, right, seed):
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": seed},
            debug=False,
        )
        env.run([left, right])
        final = env.steps[-1]
        if any(state.status != "DONE" for state in final):
            raise RuntimeError(
                f"non-terminal match for seed {seed}: "
                f"{[state.status for state in final]}"
            )
        return float(final[0].reward), float(final[1].reward)


    def summarize(scores):
        ours = [score[0] for score in scores]
        theirs = [score[1] for score in scores]
        return {
            "games": len(scores),
            "wins": sum(a > b for a, b in scores),
            "average": mean(ours),
            "median": median(ours),
            "minimum": min(ours),
            "maximum": max(ours),
            "opponent_average": mean(theirs),
            "average_margin": mean(a - b for a, b in scores),
        }


    def _both_seats(baseline, opponent, seeds):
        scores = []
        for seed in seeds:
            ours, theirs = run_match(baseline, opponent, seed)
            scores.append((ours, theirs))
            theirs, ours = run_match(opponent, baseline, seed)
            scores.append((ours, theirs))
        return scores


    def main(argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seeds", type=int, default=10)
        parser.add_argument(
            "--opponents", nargs="+", choices=("demo", "starter"),
            default=("demo", "starter"),
        )
        args = parser.parse_args(argv)

        baseline = load_agent(BASELINE_PATH)
        opponents = {"demo": load_agent(DEMO_PATH), "starter": "starter"}
        print(
            "opponent games wins average median minimum maximum "
            "opponent_avg avg_margin"
        )
        for name in args.opponents:
            result = summarize(
                _both_seats(baseline, opponents[name], range(args.seeds))
            )
            print(
                name,
                result["games"],
                result["wins"],
                round(result["average"], 2),
                round(result["median"], 2),
                round(result["minimum"], 2),
                round(result["maximum"], 2),
                round(result["opponent_average"], 2),
                round(result["average_margin"], 2),
            )
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())

- [ ] **Step 4: Run helper and short benchmark verification**

Run: .venv/bin/python -m unittest -v 00_baseline/test_main.py

Expected: benchmark helper test passes.

Run: .venv/bin/python 00_baseline/benchmark.py --seeds 1

Expected: four terminal matches complete and one result row appears for each opponent.

- [ ] **Step 5: Record Task 5**

Mark Task 5 complete in this plan.

---

### Task 6: Tune profit and write the strategy summary

**Files:**
- Modify: 00_baseline/main.py
- Create: 00_baseline/STRATEGY_SUMMARY.md
- Modify: 00_baseline/test_main.py if a selected constant intentionally changes an asserted opening.

**Interfaces:**
- Consumes: benchmark.py output for seeds 0 through 2 during screening and 0 through 9 for final validation.
- Produces: selected opening/target/hiring constants and a factual STRATEGY_SUMMARY.md.

- [ ] **Step 1: Establish the untuned three-seed result**

Run:

    .venv/bin/python 00_baseline/benchmark.py --seeds 3

Record the exact two output rows in working notes. Do not alter multiple constants before obtaining this baseline.

- [ ] **Step 2: Screen maximum Goose count**

Keep OPENING_GEESE=6 and DAILY_HANDS=7. Test TARGET_GEESE values 6, 8, and 10 one at a time, running the three-seed benchmark after each change. Retain the value with the highest mean final money among variants that win all twelve screening matches; break a tie using the higher minimum.

- [ ] **Step 3: Screen daily hand count**

With the selected TARGET_GEESE fixed, test DAILY_HANDS values 6, 7, and 8 one at a time. Update the opening-market test to assert the selected hand count while preserving the ten-order cap. Retain the value by the same mean-then-minimum rule.

- [ ] **Step 4: Screen opening Goose count**

With TARGET_GEESE and DAILY_HANDS fixed, test OPENING_GEESE values 5, 6, and 7. For each value, set OPENING_FEED to four times OPENING_GEESE and keep OPENING_SEEDS=12. Update the exact opening test with the selected quantities. Retain the best all-win variant by mean, then minimum.

- [ ] **Step 5: Run final required benchmark**

Run:

    .venv/bin/python 00_baseline/benchmark.py --seeds 10

Verify from the output:

- Each opponent row reports 20 games and 20 wins.
- Baseline average exceeds twice the freshly measured demo average.
- Every match completed with DONE status; benchmark.py would raise otherwise.

If any check fails, keep the exact failed output, identify the weakest economic or routing behavior from replay observations, add a failing behavioral test, change one policy at a time, and rerun the ten-seed benchmark.

- [ ] **Step 6: Write the factual strategy summary**

Create 00_baseline/STRATEGY_SUMMARY.md with these exact sections and fill them using the selected constants and Step 5's measured numeric output:

    # Baseline 00 Strategy Summary

    ## Objective and Environment

    State the default 720-turn objective, installed Kaggriculture environment,
    seeds 0 through 9, both seats, and required opponents.

    ## Strategy Thesis

    Explain why the selected Goose-and-Wheat engine maximizes observed money.

    ## Opening and Layout

    Record the selected opening Geese, feed, seeds, hands, Coop coordinates,
    and Wheat-tile policy.

    ## Daily Priorities

    Record the implemented survival, harvest, fertilizer, care, setup, crop,
    movement, and shed ordering.

    ## Market and Reinvestment

    Record Egg sales, Fertilizer threshold, feed reserve, Goose payback cutoff,
    and order-cap behavior.

    ## Selected Constants

    Record every tuned constant as an explicit name/value pair and explain the
    winning comparison.

    ## Benchmark Command

        .venv/bin/python 00_baseline/benchmark.py --seeds 10

    ## Benchmark Results

    Copy the final demo and starter rows and explain the win rate, average
    profit multiple, minimum result, and average margin.

    ## Known Weaknesses and Next Experiments

    Describe observed routing waste, market-version sensitivity, late shop
    opportunities, and the next single-variable improvements supported by
    benchmark evidence.

- [ ] **Step 7: Run fresh completion verification**

Run:

    .venv/bin/python -m unittest -v test_demo_agent.py test_observer_agent.py 00_baseline/test_main.py
    .venv/bin/python 00_baseline/benchmark.py --seeds 10
    .venv/bin/python -m py_compile 00_baseline/main.py 00_baseline/benchmark.py

Expected: all tests pass, both benchmark rows satisfy Task 6 Step 5, and compilation exits zero.

- [ ] **Step 8: Record Task 6**

Mark every completed checkbox in this plan. Do not commit because the directory is not a Git repository.
