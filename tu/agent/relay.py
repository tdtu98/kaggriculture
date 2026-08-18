"""Relay agent: a fixed 719-step action table plus an ordered stack of overlays (PLAN3 R0).

The agent that beats us is an offline-optimised action table replayed by `obs["step"]`, with a few
closed-loop patches bolted on. Measured over 6 games, **100% of its unit actions were the raw table
and not one market order was added, removed or resized** (E48) -- it wins without reacting.

This module restructures that shape into something we can extend. `BASE_OVERLAYS` reproduces the
reference agent's five layers exactly, so `make_relay()` with no arguments is bit-identical to it
(`tests/test_relay_parity.py`, PLAN3 R0.2). New behaviour is added by appending overlays, never by
editing `reference/kaggriculture/1/submission.py` -- that file is the arena's only external opponent
and the only non-self-referential measurement in the project (D16).

**The constraint that governs every overlay** (PLAN3 §2): the table is choreography. Every step
assumes the farm looks a certain way -- *"at step 300, water the tile you are standing on"*. Change
the farm early and that step does nothing, and so does every step after it. An overlay must
therefore be **market-only**, **structure-preserving** (COW<->SHEEP: both live on a PASTURE), or it
must **resync** the way `weed_repair` does. `Ctx.blocked_ops` (R0.5) is what makes that rule
enforceable rather than aspirational -- it counts scripted ops that arrived to find the wrong thing
on the tile, and must not rise above `relay-base` for any overlay claimed to be safe.

stdlib only: this ships in the submission bundle, where nothing but `kaggle_environments` imports.
"""

from __future__ import annotations

import math

from .relay_table import load_table

# The reference agent hardcodes these rather than reading the episode configuration, and the table
# is tuned for `townCenterSellInterval = 24`. Kept identical for bit-parity; an overlay that wants
# the real configuration should take it from the runner rather than changing this.
OFFICIAL_CONFIGURATION = {
    "turnsPerDay": 24,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
}

PRICE_FLOOR = 1
DEMAND_ALPHA = 0.25

# base, I0, T, below_func, below_target, above_func, above_target
#
# 1.32.7 moved CARROT / TOMATO / EGG to the `hinge` below-I0 shape (carrot's below_target with it,
# 0.2 -> 1.0). The old table under-priced scarcity in those three by up to 5x, so anything that
# sized a sell off it -- absorption caps, release floors, the relay's own margin test -- was wrong
# exactly when the market was most worth selling into. tests/test_market_params.py pins this
# against the reference.
MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "hinge", 1.0, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "hinge", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "hinge", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}

SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
# Both live on a PASTURE, which is the entire reason this pair is safe to substitute: the tile kind,
# the BUILD_PASTURE op and every downstream FEED / CARE / COLLECT_FERTILIZER are unchanged.
PASTURE_ANIMALS = ("COW", "SHEEP")
#: `kaggriculture.py` ANIMALS costs. A sheep is $100 dearer than a cow, which is why the swap
#: needs an affordability guard rather than being a free relabelling.
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

WEED_REPLAY_STEPS = 8

# --- ops classified for the desync counter (R0.5) --------------------------------------------
_NEEDS_PLANT = frozenset({"WATER", "HARVEST", "FERTILIZE"})
_NEEDS_EMPTY = frozenset({"PLANT", "BUILD_COOP", "BUILD_PASTURE"})
_NEEDS_ANIMAL = frozenset({"FEED", "CARE", "COLLECT_FERTILIZER"})
_MOVES = frozenset({"NORTH", "SOUTH", "EAST", "WEST", "PASS"})


def _shed_access_tiles(board_size: int) -> list[tuple[int, int]]:
    """`kaggriculture.py:119` — the four inner-corner tiles; the shed itself is not a tile."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


# ------------------------------------------------------------------ observation access

def _get(value, key, default=None):
    """`obs` may be a dict or the framework's attribute-style Struct; both appear in play."""
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _seat(obs) -> int:
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat) -> dict:
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _copy_action(action) -> dict:
    """Deep-ish copy of one turn. Lists are rebuilt so an overlay can never mutate the table."""
    action = action or {}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _align_hands(action: dict, obs) -> dict:
    """Pad or trim `hands` to the roster the runner actually hired this turn.

    The table was recorded against a particular hiring schedule. If a HIRE failed for want of cash
    the roster is shorter than the script expects, and an over-long `hands` list is silently
    truncated by the environment -- so aligning here keeps unit *indices* stable, which is what the
    rest of the script's routing depends on.
    """
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


# ------------------------------------------------------------------------ market model

#: `HINGE_GAIN`, `kaggriculture.py:60`.
HINGE_GAIN = 8.0


def _shape(name: str, value: float, scale: float = 0.0) -> float:
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    if name == "hinge":
        # Flat-ish below T, quadratic above it; scaled so f(T) == 1. Degenerates to linear when
        # T is missing or non-positive, as the reference does.
        if not scale or scale <= 0:
            return value
        u = value / scale
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(name)


def market_price(item: str, inventory: float) -> int:
    """`price(inv) = base ± amp · f(|inv − I0|)`, floored at $1 and rounded (docs/README.md)."""
    base, equilibrium, scale, below_f, below_t, above_f, above_t = MARKET_PARAMS[item]
    if inventory < equilibrium:
        amplitude = below_t * base / _shape(below_f, scale, scale)
        price = base + amplitude * _shape(below_f, equilibrium - inventory, scale)
    else:
        amplitude = above_t * base / _shape(above_f, scale, scale)
        price = base - amplitude * _shape(above_f, inventory - equilibrium, scale)
    return max(PRICE_FLOOR, int(round(price)))


def _is_sell(order) -> bool:
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in MARKET_PARAMS
    )


def demand_per_day(obs, item: str, configuration: dict | None = None) -> float:
    """Units of `item` the town removes per day, from the shops actually unlocked this game.

    Since 1.32.6 shops draw **with replacement**, so `unlocked_shops` can list the same shop several
    times and each instance consumes independently: a product may end a game with four shops buying
    it or none. WOOL has no buyer in 36% of games, MELON in 100% (E33). This is the signal R1 uses.
    """
    configuration = configuration or OFFICIAL_CONFIGURATION
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (2 if len(products) == 1 else 1)
    if item != "FERTILIZER":
        centre = max(1, int(_get(configuration, "townCenterSellInterval", 24) or 24))
        demand += turns_per_day / centre
    return demand


# --------------------------------------------------------------------------- context

class Ctx:
    """Per-seat, per-episode state, plus the instrumentation PLAN3 §6 requires.

    The reference agent keeps this in module-level dicts keyed by seat and resets them when the step
    counter restarts. Same discipline here, because an agent object can outlive one episode in the
    arena -- carrying `last_sale` or a `locked` route flag across a game boundary would silently
    change behaviour in the *second* game of a pair and in no other.
    """

    __slots__ = ("seat", "step", "weed", "route", "wool", "rc2", "blocked_ops", "effects",
                 "schedule")

    def __init__(self, seat: int):
        self.seat = seat
        # Derived from the table, not the episode, so it survives `reset` and is built at most once.
        self.schedule = None
        self.reset(0)

    def reset(self, step: int) -> None:
        self.step = step
        self.weed = {"active": {}}
        self.route = {"yarn": None}
        self.wool = {"last_sale": -1000}
        self.rc2 = {"checks": {}, "locked": False, "due_step": -1, "due": 0}
        # R0.5 / PLAN3 §6. `blocked_ops` is the desync counter -- scripted ops that arrived to find
        # the wrong thing on the tile. `effects` is where each overlay records that it fired; an
        # overlay whose count is zero is an unfinished implementation, not a negative result.
        self.blocked_ops = {}
        self.effects = {}

    def note(self, key: str, n: int = 1) -> None:
        self.effects[key] = self.effects.get(key, 0) + n

    def block(self, cause: str, n: int = 1) -> None:
        self.blocked_ops[cause] = self.blocked_ops.get(cause, 0) + n

    @property
    def blocked_total(self) -> int:
        return sum(self.blocked_ops.values())


# -------------------------------------------------------------------------- overlays
# Each is `fn(obs, action, ctx, table, step) -> action`. Order is fixed in `BASE_OVERLAYS` and
# matters: farm-affecting overlays run before market overlays, so a market overlay prices the final
# farm decisions rather than the ones they replaced.


def _trace_actor_action(table, step, actor):
    trace = table[min(max(int(step), 0), len(table) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def weed_repair(obs, action, ctx, table, step):
    """Dig a weed blocking a scripted PLANT/BUILD, then replay the missed steps to resynchronise.

    This is the only *farm-affecting* overlay in the base stack, and it is safe because it explicitly
    resyncs: the intended action runs one step late, then the following `WEED_REPLAY_STEPS` steps are
    re-issued shifted by one, which walks the unit back onto the script.
    """
    action = _align_hands(action, obs)
    seat = _seat(obs)
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = ctx.weed["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(table, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]
        ctx.note("weed_repair")

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


# The reference agent's own livestock rule, kept verbatim so `relay-base` is bit-identical. It is a
# single binary condition on one product, and E48 measured it firing in **0 of 6** games -- which is
# exactly the hypothesis R1 replaces, and exactly the trap PLAN3 §6 exists to catch: judged on money
# alone this looks like a strategy that does not pay, when it is a rule that never ran.
SWITCH_STEP = 161
PURCHASE_STEPS = frozenset([192])
ACTOR_WINDOWS = [(4, 193, 212)]


def _yarn_route(obs, ctx, step) -> bool:
    if step < SWITCH_STEP:
        return False
    if ctx.route.get("yarn") is None:
        shops = tuple(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
        blocked = {"SMOOTHIE_SHOP", "PIZZA_SHOP"}
        ctx.route["yarn"] = "YARN_STORE" in shops and not any(s in blocked for s in shops)
    return bool(ctx.route["yarn"])


def _swap_cow(order):
    order = list(order or ["PASS"])
    if len(order) >= 2 and order[1] == "COW":
        order[1] = "SHEEP"
    return order


def convert_livestock(obs, action, ctx, table, step):
    """COW -> SHEEP when a yarn store exists and nothing else is buying milk."""
    action = _copy_action(action)
    if not _yarn_route(obs, ctx, step):
        return action
    if step in PURCHASE_STEPS:
        before = [list(o) for o in (action.get("market") or [])]
        action["market"] = [_swap_cow(o) for o in before]
        if action["market"] != before:
            ctx.note("livestock_swap_market")
    hands = [list(o or ["PASS"]) for o in (action.get("hands") or [])]
    for actor, start, end in ACTOR_WINDOWS:
        if int(start) <= step <= int(end) and int(actor) < len(hands):
            before_h = list(hands[int(actor)])
            hands[int(actor)] = _swap_cow(hands[int(actor)])
            if hands[int(actor)] != before_h:
                ctx.note("livestock_swap_unit")
    action["hands"] = hands
    return action


WOOL_GATES = ((480, 170), (600, 120), (672, 80), (719, 1))
WOOL_PRESSURE = 78
WOOL_BATCH = 16
WOOL_GAP = 6


def _wool_gate(step: int) -> int:
    for end_step, gate in WOOL_GATES:
        if step < int(end_step):
            return int(gate)
    return 1


def wool_controller(obs, action, ctx, table, step):
    """Release wool on a decaying price gate, or force it out when the shed fills.

    `WOOL_PRESSURE = 78` is a **shed-capacity valve**: at 78 of the shed's 100 slots it dumps wool
    regardless of price. Worth noting for R2a -- our own champion has no such valve, pegs at 100 for
    9 of 29 days holding 52 unsold wool, and that correlates with its worst seeds (E48). The
    reference author hit the same wall and patched it for one product; R2a generalises it.
    """
    action = _copy_action(action)
    if not _yarn_route(obs, ctx, step):
        return action
    market_orders = [list(o) for o in (action.get("market") or [])]
    if any(len(o) >= 2 and o[0] == "SELL" and o[1] == "WOOL" for o in market_orders):
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    wool = max(0, int(_get(shed, "WOOL", 0) or 0))
    if wool <= 0 or len(market_orders) >= 10:
        return action
    total = sum(max(0, int(v or 0)) for v in shed.values())
    price = max(0, int(_get(_get(obs, "market", {}) or {}, "prices", {}).get("WOOL", 0) or 0))
    terminal = step >= 713
    pressure = total >= WOOL_PRESSURE
    price_ok = price >= _wool_gate(step)
    gap_ok = step - int(ctx.wool.get("last_sale", -1000)) >= WOOL_GAP
    if not terminal and not pressure and (not price_ok or not gap_ok):
        return action
    quantity = wool if terminal else min(wool, WOOL_BATCH)
    if pressure:
        quantity = min(wool, max(quantity, total - (WOOL_PRESSURE - 12)))
    market_orders.append(["SELL", "WOOL", max(1, int(quantity))])
    action["market"] = market_orders[:10]
    ctx.wool["last_sale"] = step
    ctx.note("wool_release")
    return action


def _impact_score(obs, order) -> float:
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = int(_get(_get(market, "inventory", {}) or {}, item, 10000) or 0)
    quote = float(_get(_get(market, "prices", {}) or {}, item, market_price(item, inventory)) or 0)
    later = float(market_price(item, inventory + quantity))
    return float(quantity) * max(0.0, quote - later)


def _order_score(obs, order, configuration) -> float:
    score = _impact_score(obs, order)
    if score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    inventory = int(_get(_get(_get(obs, "market", {}) or {}, "inventory", {}) or {},
                         item, 10000) or 0)
    demand = max(0.25, demand_per_day(obs, item, configuration))
    excess = max(0.0, inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + DEMAND_ALPHA * urgency)


def rank_sell_slots(obs, action, ctx, table, step, configuration=None):
    """Reorder SELL orders by price impact. Market-only, so it cannot desynchronise the table.

    E48: this is the **only** layer of the five that ever fired against our champion -- 69 turns of
    6 games, all pure permutations.
    """
    configuration = configuration or OFFICIAL_CONFIGURATION
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_order_score(obs, order, configuration), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    reordered = [next(ranked) if _is_sell(o) else o for o in market]
    if reordered != market:
        ctx.note("sell_reorder")
    action["market"] = reordered
    return action


RC2_CHECKPOINTS = (216, 240, 264)
RC2_DISTANCE_MAX = 8
RC2_LEAD = 3
RC2_FIRST = 278
RC2_LAST = 662


def _public_signature(farm):
    counts = {k: 0 for k in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED")}
    for row in (_get(farm, "tiles", []) or []):
        for tile in row if isinstance(row, list) else [row]:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value in counts:
                    counts[value] += 1
                    break
    return (len(_get(farm, "hands", []) or []),
            len(_get(farm, "unlocked_quadrants", []) or []),
            tuple(counts[k] for k in sorted(counts)))


def _route_distance(obs) -> int:
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10 ** 9
    left, right = _public_signature(farms[0]), _public_signature(farms[1])
    return (abs(left[0] - right[0]) + 3 * abs(left[1] - right[1])
            + sum(abs(a - b) for a, b in zip(left[2], right[2])))


def _future_fertilizer_quantity(table, step: int) -> int:
    future = step + RC2_LEAD
    if not (RC2_FIRST <= step <= RC2_LAST and 281 <= future <= 665
            and future % 24 == 17 and future < len(table)):
        return 0
    return sum(max(0, int(o[2])) for o in (table[future].get("market") or [])
               if len(o) >= 3 and o[0] == "SELL" and o[1] == "FERTILIZER")


def _rc2_repay(action, ctx, step):
    due_step = int(ctx.rc2.get("due_step", -1))
    if due_step != step:
        if 0 <= due_step < step:
            ctx.rc2["due_step"], ctx.rc2["due"] = -1, 0
        return action
    remaining = max(0, int(ctx.rc2.get("due", 0)))
    market = []
    for raw in (action.get("market") or []):
        order = list(raw)
        if remaining > 0 and len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER":
            requested = max(0, int(order[2]))
            reduction = min(requested, remaining)
            requested -= reduction
            remaining -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market
    ctx.rc2["due_step"], ctx.rc2["due"] = -1, 0
    return action


def market_relay(obs, action, ctx, table, step, configuration=None):
    """Sell fertilizer three turns early, but only in a near-mirror match.

    An anti-mirror device: it front-runs a *future* scripted sale and repays it later, and arms only
    when the two farms' public signatures stay within `RC2_DISTANCE_MAX` at three checkpoints. E48
    measured it never arming against our champion, which is correct behaviour -- our farms look
    nothing alike.
    """
    configuration = configuration or OFFICIAL_CONFIGURATION
    action = _copy_action(action)
    action = _rc2_repay(action, ctx, step)
    if step in RC2_CHECKPOINTS and step not in ctx.rc2["checks"]:
        ctx.rc2["checks"][step] = _route_distance(obs) <= RC2_DISTANCE_MAX
        if all(c in ctx.rc2["checks"] for c in RC2_CHECKPOINTS):
            ctx.rc2["locked"] = all(ctx.rc2["checks"].values())
    if not ctx.rc2.get("locked") or ctx.rc2.get("due", 0):
        return action
    target = _future_fertilizer_quantity(table, step)
    if target <= 0:
        return action
    market = [list(o) for o in (action.get("market") or [])]
    if len(market) >= 10:
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    available = max(0, int(_get(shed, "FERTILIZER", 0) or 0))
    for order in market:
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER":
            available = max(0, available - max(0, int(order[2])))
    quantity = min(target, available)
    if quantity <= 0:
        return action
    market.append(["SELL", "FERTILIZER", quantity])
    action["market"] = market[:10]
    ctx.rc2["due_step"] = step + RC2_LEAD
    ctx.rc2["due"] = quantity
    ctx.note("fertilizer_relay")
    return rank_sell_slots(obs, action, ctx, table, step, configuration)


# ------------------------------------------------------------- R1: adaptive livestock

def _shop_demand(obs, item: str) -> float:
    """Units/day drawn by **shops only**, excluding the flat town-centre term.

    The centre buys one of every non-fertilizer product per day regardless, so that term is equal
    across products and is pure noise for a comparison between two of them -- while still being
    large enough for a price multiplier to break the tie on it. Separated out so a herd decision
    can require real evidence and otherwise defer to the table.
    """
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    per_day = OFFICIAL_CONFIGURATION["turnsPerDay"] / OFFICIAL_CONFIGURATION["townShopSellInterval"]
    total = 0.0
    for shop in shops:
        products = SHOP_PRODUCTS.get(shop, ())
        if item in products:
            total += per_day * (2 if len(products) == 1 else 1)
    return total


def _pasture_animal_in(store: dict, prefer: str) -> str | None:
    """Whichever pasture animal `store` actually holds, preferring the scripted species."""
    if int(_get(store, prefer, 0) or 0) > 0:
        return prefer
    for animal in PASTURE_ANIMALS:
        if int(_get(store, animal, 0) or 0) > 0:
            return animal
    return None


def adaptive_livestock(obs, action, ctx, table, step):
    """Choose COW vs SHEEP from the shops this game actually drew (PLAN3 R1).

    **Why this is safe.** Both live on a `PASTURE`, so the tile kind, the `BUILD_PASTURE` op and
    every downstream FEED / CARE / COLLECT_FERTILIZER are untouched. The table does not desynchronise.

    **Why it needs care.** The purchase, the shed pickup and the placement are three ops several
    turns apart. Rewrite the purchase alone and the scripted `PICKUP COW` finds no cow, the animal
    never leaves the shed, and the pasture sits empty for the season -- measured at **1,667 blocked
    ops and $689** against `relay-base`'s $128,485 (E50). So the logistics are not rewritten from a
    remembered decision, which can drift: they are resolved **against what the shed and the unit are
    actually holding**, which cannot disagree with the purchase whatever the purchase was.

    **Why it should pay.** WOOL has no shop buyer in 36% of games, MILK in 2% (E33), and the
    reference agent commits its herd before any shop unlocks. Its own conditional fired in 0 of 6
    traced games (E48) -- the failure PLAN3 SS6 exists to catch.
    """
    action = _copy_action(action)

    # 1. Purchases: pick the species whose product this game can actually sell.
    market = [list(o) for o in (action.get("market") or [])]
    money = float(_get(_farm(obs, _seat(obs)), "money", 0) or 0)
    swapped = False
    for order in market:
        if len(order) >= 2 and order[0] == "BUY_ANIMAL" and order[1] in PASTURE_ANIMALS:
            scripted = order[1]
            # **Shop demand only.** The town centre buys one of every non-fertilizer product per
            # day, so its term is identical for milk and wool and carries no information -- but it
            # is not zero, so including it lets the *price* multiplier break the tie. With no shops
            # unlocked that ranked WOOL above MILK on every day-0 purchase, turning nine $400 cows
            # into $500 sheep on a $3,000 bank: measured **$29,105 and 1,534 blocked ops** against
            # `relay-base`'s $126,166 (E50). Only genuine shop evidence may move the herd.
            #
            # Given evidence, rank by revenue capacity rather than units: one yarn store absorbs
            # 12 wool/day at $200 against a milk shop's 6/day at $160, so units alone rank them
            # wrongly -- the error E35 made and measured losing.
            milk = _shop_demand(obs, "MILK") * MARKET_PARAMS["MILK"][0]
            wool = _shop_demand(obs, "WOOL") * MARKET_PARAMS["WOOL"][0]
            # No shop buys either: keep the script. It was tuned offline against the average draw,
            # which is strictly better than a coin flip made on no evidence.
            want = scripted
            if wool > milk:
                want = "SHEEP"
            elif milk > wool:
                want = "COW"
            # **Affordability guard.** A sheep costs $500 against a cow's $400, and the table's cash
            # is budgeted to the edge -- it is an offline-optimised plan that spends nearly
            # everything. Swapping upward made the purchase fail outright, leaving the pasture empty
            # for the season: measured **1.2 stranded structures per game** and blocked ops 1.68 ->
            # 19.18 before this guard existed (E50). Never trade a certain animal for a better one
            # we cannot pay for.
            qty = int(order[2]) if len(order) >= 3 else 1
            if want != scripted and money < ANIMAL_COST[want] * max(1, qty):
                ctx.note("herd_swap_unaffordable")
                want = scripted
            if want != scripted:
                order[1] = want
                money -= ANIMAL_COST[want] * max(1, qty)
                swapped = True
                ctx.note("herd_purchase_swapped")
    if swapped:
        action["market"] = market

    # 2. Logistics: follow the stock, not the script. A PICKUP for a species we did not buy is a
    #    silent no-op, and that single mismatch is what destroys the farm.
    priv = _get(obs, "private", {}) or {}
    shed = _get(priv, "shed", {}) or {}
    invs = list(_get(priv, "inventories", []) or [])
    units = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    for idx, act in enumerate(units):
        if not isinstance(act, list) or len(act) < 2 or act[1] not in PASTURE_ANIMALS:
            continue
        if act[0] == "PICKUP":
            have = _pasture_animal_in(shed, act[1])
            if have and have != act[1]:
                act[1] = have
                ctx.note("herd_pickup_retargeted")
        elif act[0] == "PLACE":
            inv = invs[idx] if idx < len(invs) else {}
            have = _pasture_animal_in(inv, act[1])
            if have and have != act[1]:
                act[1] = have
                ctx.note("herd_place_retargeted")
    action["farmer"] = units[0] if units else ["PASS"]
    action["hands"] = units[1:]
    return action


def adaptive_livestock_downgrade(obs, action, ctx, table, step):
    """R1, restricted to the swap direction that cannot fail for cash.

    Exists to separate two explanations of R1's loss that the full version confounds: *the idea is
    wrong*, or *the implementation strands animals*. A sheep costs $500 and a cow $400, so
    SHEEP -> COW always settles if the scripted purchase would have. If the herd choice still loses
    with `blocked_ops` at the `relay-base` baseline, that is a clean refutation of the hypothesis
    rather than of our code.

    It is also the direction E33 actually motivates: WOOL has no shop buyer in 36% of games, so the
    value is in *not* breeding sheep into a game with no yarn store.
    """
    action = _copy_action(action)
    market = [list(o) for o in (action.get("market") or [])]
    swapped = False
    for order in market:
        if len(order) >= 2 and order[0] == "BUY_ANIMAL" and order[1] == "SHEEP":
            milk = _shop_demand(obs, "MILK") * MARKET_PARAMS["MILK"][0]
            wool = _shop_demand(obs, "WOOL") * MARKET_PARAMS["WOOL"][0]
            if milk > wool:
                order[1] = "COW"
                swapped = True
                ctx.note("herd_purchase_swapped")
    if swapped:
        action["market"] = market

    priv = _get(obs, "private", {}) or {}
    shed = _get(priv, "shed", {}) or {}
    invs = list(_get(priv, "inventories", []) or [])
    units = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    for idx, act in enumerate(units):
        if not isinstance(act, list) or len(act) < 2 or act[1] not in PASTURE_ANIMALS:
            continue
        store = shed if act[0] == "PICKUP" else (invs[idx] if idx < len(invs) else {})
        if act[0] not in ("PICKUP", "PLACE"):
            continue
        have = _pasture_animal_in(store, act[1])
        if have and have != act[1]:
            act[1] = have
            ctx.note("herd_%s_retargeted" % act[0].lower())
    action["farmer"] = units[0] if units else ["PASS"]
    action["hands"] = units[1:]
    return action


# --------------------------------------------------- R2e: release what the script never sells

#: Feed, not produce. The table buys wheat to feed 13 animals daily and that consumption is not a
#: SELL order, so wheat held for feed looks exactly like surplus. Selling it starves the herd.
NEVER_RELEASE = frozenset({"WHEAT"})

#: Sell a surplus unit only while its marginal price clears this share of base. Milk's curve is
#: `linear x 1.6` off `T = 122`, so a careless dump reaches the $1 floor fast -- releasing stock is
#: only worth doing at a price that beats leaving it in the shed, which scores zero either way.
RELEASE_RESERVE_FRAC = 0.5
RELEASE_BATCH = 8
#: Shed slots used before the valve opens. The shed caps at 100 and end-of-day overflow is
#: **discarded**, so waiting for 100 is waiting until the loss has already happened.
#:
#: Swept, not assumed (E52). 78 was the reference agent's *wool* threshold, borrowed as a starting
#: point; it leaves money on the table. Winrate is flat at 83.8% across 65-78 while the money delta
#: runs +$175 (78) -> +$339 (70) -> +$362 (65) -- and then **falls off a cliff below 65**:
#: -$806 at 60, -$1,400 at 55, with `blocked_ops` rising above the `relay-base` baseline at exactly
#: the same point. Selling that hard shifts cash, which changes which purchases settle, which
#: desynchronises the farm.
#:
#: 70 takes 93% of the available gain with **twice the margin to the cliff** that 65 has. E46's
#: `role_penalty` wall is the precedent: sitting one step from a cliff is not worth the last 7%.
#: **Safe range is ~65-78. A search sampling this uniformly will mostly sample the bad side.**
RELEASE_PRESSURE = 70


def _scheduled_sales_after(table) -> list[dict]:
    """Suffix sums: for each step, how many units of each product the script still intends to sell.

    This is the measurement R2e turns on. Stock beyond what the remaining script will move is stock
    that gets harvested into a full shed and **discarded** -- `relay-base` strands only 0.1 items a
    season, but that is because its production and its sell list were tuned together. Change either
    and the difference becomes waste (E50).
    """
    suffix = [{} for _ in range(len(table) + 1)]
    running: dict[str, int] = {}
    for step in range(len(table) - 1, -1, -1):
        for order in table[step].get("market") or []:
            if len(order) >= 3 and order[0] == "SELL" and order[1] in MARKET_PARAMS:
                running[order[1]] = running.get(order[1], 0) + max(0, int(order[2]))
        suffix[step] = dict(running)
    return suffix


def make_surplus_release(pressure: int = RELEASE_PRESSURE, batch: int = RELEASE_BATCH,
                         reserve_frac: float = RELEASE_RESERVE_FRAC, once: bool = False):
    """Build a `surplus_release` with explicit thresholds, so they can be swept rather than assumed.

    The defaults are the reference agent's own **wool** numbers (`WOOL_PRESSURE = 78`,
    `WOOL_BATCH = 16` -> 8 here). They were tuned for a different product and a different purpose;
    borrowing them was a reasonable start, not a measurement. `D19` stage 3 exists because a search
    optimum need not be a *local* optimum, and the same applies to a number copied from elsewhere.
    """
    def surplus_release(obs, action, ctx, table, step):
        return _surplus_release(obs, action, ctx, table, step, pressure, batch, reserve_frac, once)
    surplus_release.__doc__ = _surplus_release.__doc__
    return surplus_release


def _surplus_release(obs, action, ctx, table, step,
                     pressure=RELEASE_PRESSURE, batch=RELEASE_BATCH,
                     reserve_frac=RELEASE_RESERVE_FRAC, once=False):
    """Sell product the remaining script will never get to. Market-only.

    Pairs with a production-side change: `PLAN3` §2's second constraint is that the table's sell
    orders are a fixed list, so extra output cannot become money on its own. On an unmodified
    `relay-base` this overlay is close to a no-op by construction -- there is no surplus -- which
    doubles as its null test.
    """
    action = _copy_action(action)
    market = [list(o) for o in (action.get("market") or [])]
    if len(market) >= 10:
        return action

    schedule = ctx.schedule
    if schedule is None:
        schedule = ctx.schedule = _scheduled_sales_after(table)
    planned = schedule[min(step + 1, len(schedule) - 1)]

    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    inventory = _get(_get(obs, "market", {}) or {}, "inventory", {}) or {}
    already = {}
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            already[order[1]] = already.get(order[1], 0) + max(0, int(order[2]))

    total = sum(max(0, int(v or 0)) for v in shed.values())
    # Two independent triggers. The first version used only `held > remaining planned sales` and
    # **never fired once in 160 games** -- `held` is one shed's worth (tens) and `planned` is the
    # whole remaining season's orders (hundreds), so the difference is negative essentially always
    # (E51). The pressure term is what actually catches mid-season waste, and it is the mechanism
    # the reference agent already uses for wool (`WOOL_PRESSURE = 78`).
    if once and ctx.effects.get('surplus_released', 0):
        return action
    under_pressure = total >= pressure

    def marginal(item: str) -> int:
        return market_price(item, int(_get(inventory, item, 10000) or 0))

    for item in sorted((k for k in shed if k in MARKET_PARAMS and k not in NEVER_RELEASE),
                       key=lambda k: -marginal(k)):
        if len(market) >= 10:
            break
        held = int(_get(shed, item, 0) or 0) - already.get(item, 0)
        if held <= 0:
            continue
        # Stock past the script's *last* scheduled sale of this item is genuinely dead -- nothing
        # will ever move it, and unsold stock scores zero (docs/README.md, Reward).
        dead = held - int(planned.get(item, 0))
        want = held if dead > 0 and step > 700 else (max(dead, 0) or (held if under_pressure else 0))
        if want <= 0:
            continue
        base = MARKET_PARAMS[item][0]
        floor = reserve_frac * base
        inv = int(_get(inventory, item, 10000) or 0)
        # Marginal pricing: sell while the *next* unit still clears the reserve, so a thin market
        # takes a few units and a deep one takes the batch. Freeing a shed slot is only worth doing
        # at a price that beats the zero it scores sitting there -- but not at any price.
        qty = 0
        while qty < min(want, batch) and market_price(item, inv + qty) >= floor:
            qty += 1
        if qty > 0:
            market.append(["SELL", item, qty])
            ctx.note("surplus_released")
            ctx.note("surplus_units_%s" % item, qty)

    action["market"] = market[:10]
    return action


#: Default instance -- the thresholds R2e was measured with (E51).
surplus_release = make_surplus_release()


#: The reference agent's pipeline, in its exact order. `make_relay()` with this stack is
#: bit-identical to `reference/kaggriculture/1/submission.py` -- proven, not assumed, by
#: `tests/test_relay_parity.py` (PLAN3 R0.2).
BASE_OVERLAYS = (weed_repair, convert_livestock, wool_controller, rank_sell_slots, market_relay)

#: R1. `adaptive_livestock` replaces the reference agent's own single-condition `convert_livestock`
#: -- running both would let two rules fight over the same orders. It sits before the market
#: overlays so they price the herd we actually bought.
R1_OVERLAYS = (weed_repair, adaptive_livestock, wool_controller, rank_sell_slots, market_relay)


# ------------------------------------------------------------------- desync counter (R0.5)

def count_blocked_ops(obs, action, ctx) -> None:
    """Count scripted ops that arrived to find the wrong thing on the tile.

    This is the instrument that makes PLAN3 §2's safety rule enforceable: an overlay claimed to be
    market-only or structure-preserving **must not raise this above `relay-base`**. It is pure
    instrumentation -- it never changes the action.
    """
    seat = _seat(obs)
    farm = _farm(obs, seat)
    priv = _get(obs, "private", {}) or {}
    shed = _get(priv, "shed", {}) or {}
    invs = list(_get(priv, "inventories", []) or [])
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    board = len(_get(farm, "tiles", []) or []) or 10
    access = set(_shed_access_tiles(board))

    for idx, (position, act) in enumerate(zip(positions, unit_actions)):
        if not isinstance(act, list) or not act:
            continue
        op = act[0]
        if op in _MOVES or op == "DIG":
            continue
        inv = invs[idx] if idx < len(invs) else {}
        here = (int(position[0]), int(position[1])) if position else (-1, -1)

        # Logistics ops. These are where an animal substitution goes wrong: swap a purchase to
        # SHEEP and leave the scripted `PICKUP COW` behind, and the pickup no-ops, the animal never
        # reaches the pasture, and the tile sits empty for the rest of the season. That is R1's
        # exact failure mode, so the counter has to be able to see it.
        if op == "PICKUP":
            item = act[1] if len(act) > 1 else None
            if here not in access:
                ctx.block("PICKUP_off_shed")
            elif int(_get(shed, item, 0) or 0) <= 0:
                ctx.block("PICKUP_%s_absent" % item)
            continue
        if op == "DROP":
            if here not in access:
                ctx.block("DROP_off_shed")
            continue
        if op == "PLACE":
            item = act[1] if len(act) > 1 else None
            tile = _tile_at(farm, position)
            on_structure = (isinstance(tile, dict)
                            and tile.get("kind") in ("COOP", "PASTURE")
                            and not tile.get("animal"))
            if on_structure:
                # Standing on an empty structure: this is an animal placement, and it silently
                # does nothing unless the unit is carrying that exact species.
                if int(_get(inv, item, 0) or 0) <= 0:
                    ctx.block("PLACE_%s_not_carried" % item)
            elif here in access:
                if int(_get(inv, item, 0) or 0) <= 0:
                    ctx.block("PLACE_%s_not_carried" % item)
            else:
                ctx.block("PLACE_nowhere")
            continue

        tile = _tile_at(farm, position)
        if op == "HARVEST":
            # HARVEST is legal on a plant **or** on an occupied coop/pasture -- collecting milk,
            # wool and eggs is the same op. Counting it as plant-only reported 94 false desyncs per
            # episode on a farm that was perfectly in sync, which is precisely the instrument bug
            # PLAN3 SS6 exists to catch: a counter is as capable of being wrong as the thing it
            # counts (E39).
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                ctx.block("HARVEST_on_weed")
            elif not (isinstance(tile, dict)
                      and (tile.get("kind") == "PLANT" or tile.get("animal"))):
                ctx.block("HARVEST_on_empty")
        elif op in _NEEDS_PLANT:
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                ctx.block("%s_on_weed" % op)
            elif not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                ctx.block("%s_on_empty" % op)
        elif op in _NEEDS_EMPTY:
            if tile is not None and tile != "LOCKED":
                ctx.block("%s_on_occupied" % op)
        elif op in _NEEDS_ANIMAL:
            if not (isinstance(tile, dict) and tile.get("animal")):
                ctx.block("%s_no_animal" % op)


# ------------------------------------------------------------------------------- agent

def make_relay(overlays=None, configuration: dict | None = None, table=None):
    """Build a relay agent. `overlays=None` means `BASE_OVERLAYS` -- bit-identical to the reference.

    The returned callable carries `.ctx`, so a bench can read `blocked_ops` and `effects` after an
    episode. PLAN3 §6: **check those before reading any money number.** An overlay whose effect
    count is zero did not run, and reporting that as a refutation is the error E44/E36/E39/E46 all
    made in different disguises.
    """
    stack = BASE_OVERLAYS if overlays is None else tuple(overlays)
    configuration = configuration or OFFICIAL_CONFIGURATION
    table = load_table() if table is None else table
    state = {}

    def agent(obs, _configuration=None):
        try:
            seat = _seat(obs)
            step = min(max(0, int(_get(obs, "step", 0) or 0)), len(table) - 1)
            ctx = state.get(seat)
            if ctx is None:
                ctx = state[seat] = Ctx(seat)
            # Same reset discipline as the reference: a new episode restarts the step counter, and
            # carrying `last_sale` or a locked route flag across that boundary would change
            # behaviour in the second game of a pair and nowhere else.
            if step == 0 or step < ctx.step:
                ctx.reset(step)
            ctx.step = step

            action = _copy_action(table[step])
            for overlay in stack:
                action = overlay(obs, action, ctx, table, step)
                if overlay is rank_sell_slots:
                    # The reference aligns hands between the sell ranking and the relay; kept so the
                    # relay sees the same roster the reference's does.
                    action = _align_hands(action, obs)
            action = _align_hands(action, obs)
            count_blocked_ops(obs, action, ctx)
            return action
        except Exception:
            farm = _farm(obs, _seat(obs))
            return {"farmer": ["PASS"],
                    "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
                    "market": []}

    agent.ctx_by_seat = state
    agent.overlays = stack
    return agent
