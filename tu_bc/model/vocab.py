"""Every enumerated constant of the action space, in one place.

PLAN_BC Chapter 3, "Settle the vocabulary as one constant, before anyone writes a head width":
two panel proposals disagreed *arithmetically* about how many verbs and items exist, and a
one-entry disagreement between a head width and the mask table makes Assertion 3 fire and look
like a data problem. So the widths live here, they are derived from the environment's own dispatch
tables rather than from our sample, and everything else reads them.

Derived, not copied: `CROPS`, `ANIMALS`, `PRODUCTS`, `FARMER_MOVES`, `LAND_PRICES`, `SHOPS` and
`MARKET_PARAMS` are imported from the installed environment module, so a Kaggle version bump that
changes them changes us too instead of drifting silently.

The two questions PLAN_BC left open are settled here by measurement against the corpus:

* **Is FERTILIZER inside the 9 products?**  Yes. `PRODUCTS` has nine entries and FERTILIZER is the
  ninth (`kag.py:26`).  The shed additionally holds the three live animals, so the *item* alphabet
  a unit or an order can name is 9 + 3 = **12**, plus a NONE slot for verbs that take no item.
* **How many verbs?**  4 moves + 14 non-move ops = **18**.  All 18 appear in the corpus (an
  earlier count of 17 was taken on one sample game, which missed `BUILD_COOP` -- 38 occurrences
  across all 100 games).

Observed arity, measured over all 200 seats of the corpus: `PICKUP` is always 3-long
(verb, item, n); `PLANT` always 2-long; `PLACE` is **2-long 2,036 times and 3-long 5,842 times**
-- genuinely variable, exactly as PLAN_BC warns.  `FEED` and `FERTILIZE` also appear with one
trailing argument (578 and 144 times) that `kag.py` never reads (`:505-512`, `:475-478`); those are
counted as ignored extras rather than treated as vocabulary.
"""

from __future__ import annotations

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as kag

# --------------------------------------------------------------------------------------
# Board / roster shapes
# --------------------------------------------------------------------------------------

GRID = 10                      # boardSize default (kag.py:264)
N_TILES = GRID * GRID          # tiles[y][x] is NESTED -- flatten explicitly, PLAN_BC Ch3
MAX_UNITS = 16                 # farmer + 15 hands; the mechanism's max, not the observed one
MAX_MARKET_ORDERS = 10         # maxMarketOrdersPerTurn (kag.py:551); extras silently dropped :560
TURNS_PER_DAY = 24
DAYS = 30
EPISODE_STEPS = 720
PAIRS_PER_SEAT = EPISODE_STEPS - 1   # steps[0].action is a placeholder nobody chose
SHED_CAPACITY = 100
OPENING_STEPS = 32             # Ryo's fixed opening (PLAN_BC Ch2); metrics reported with/without

# --------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------

CROPS = tuple(kag.CROPS)                    # WHEAT CARROT TOMATO STRAWBERRY MELON
ANIMALS = tuple(kag.ANIMALS)                # GOOSE COW SHEEP
PRODUCTS = tuple(kag.PRODUCTS)              # the 9 tradeable goods; FERTILIZER is the 9th
ITEMS = PRODUCTS + ANIMALS                  # 12: everything the shed can hold (kag.py:165)
ITEM_INDEX = {name: i for i, name in enumerate(ITEMS)}
ITEM_NONE = len(ITEMS)                      # verbs/ops that name no item
N_ITEM_SLOTS = len(ITEMS) + 1               # 13

BUYABLE_PRODUCTS = ("WHEAT", "FERTILIZER")  # BUY_PRODUCT is restricted to these two (kag.py:598)

assert len(PRODUCTS) == 9 and PRODUCTS[-1] == "FERTILIZER"
assert len(ITEMS) == 12
assert set(kag.PRODUCTS) >= set(kag.CROPS), "every crop must be a tradeable product"

# --------------------------------------------------------------------------------------
# Unit verbs
# --------------------------------------------------------------------------------------

MOVE_VERBS = ("NORTH", "SOUTH", "EAST", "WEST")
assert set(MOVE_VERBS) == set(kag.FARMER_MOVES)

NON_MOVE_VERBS = (
    "PASS",                 # :334
    "DROP",                 # :344
    "PICKUP",               # :359
    "PLACE",                # :376
    "PLANT",                # :417
    "WATER",                # :431
    "HARVEST",              # :446
    "FERTILIZE",            # :475
    "DIG",                  # :484
    "BUILD_COOP",           # :493
    "BUILD_PASTURE",        # :498
    "FEED",                 # :505
    "COLLECT_FERTILIZER",   # :515
    "CARE",                 # :524
)
VERBS = MOVE_VERBS + NON_MOVE_VERBS
VERB_INDEX = {v: i for i, v in enumerate(VERBS)}
N_VERBS = len(VERBS)
assert N_VERBS == 18

MOVE_IDS = tuple(VERB_INDEX[v] for v in MOVE_VERBS)
IS_MOVE = np.zeros(N_VERBS, dtype=bool)
IS_MOVE[list(MOVE_IDS)] = True
V_PASS = VERB_INDEX["PASS"]

# The macro action space (PLAN_BC Ch5): "walk to tile T, then do V there".  Movement stops being a
# decision, so the 14 non-move verbs plus one explicit MOVE make up the alphabet.  MOVE covers the
# fallback space PLAN_BC names when frac_segments_shortest_path < 0.9, and it is also what a
# segment truncated by the nightly roster wipe (kag.py:879-882) is labelled with.
MACRO_VERBS = NON_MOVE_VERBS + ("MOVE",)
MACRO_VERB_INDEX = {v: i for i, v in enumerate(MACRO_VERBS)}
N_MACRO_VERBS = len(MACRO_VERBS)
MV_MOVE = MACRO_VERB_INDEX["MOVE"]
MV_PASS = MACRO_VERB_INDEX["PASS"]

# Arity beyond the verb itself, read off kag.py's dispatch.  (required, meaningful_max).
# Anything past `meaningful_max` is never read by the environment; we count it, we do not honour it.
VERB_ARITY = {v: (0, 0) for v in VERBS}
VERB_ARITY["PICKUP"] = (1, 2)   # item, [n]  -- n defaults to 1 (:365-366); the corpus always sends n
VERB_ARITY["PLANT"] = (1, 1)    # crop                                                      (:418-421)
VERB_ARITY["PLACE"] = (1, 2)    # item, [n]  -- VARIABLE: animal-place takes 1, shed-place 1 or 2

# Which items each verb may name.  Empty tuple => the verb takes ITEM_NONE.
VERB_ITEMS = {v: () for v in VERBS}
VERB_ITEMS["PICKUP"] = ITEMS    # seeds are not pickupable, but every shed item is (:359-374)
VERB_ITEMS["PLACE"] = ITEMS     # animal -> tile (:381-392); anything else -> shed (:393-409)
VERB_ITEMS["PLANT"] = CROPS     # (:417-429)

VERB_TAKES_ITEM = np.zeros(N_VERBS, dtype=bool)
for _v, _its in VERB_ITEMS.items():
    VERB_TAKES_ITEM[VERB_INDEX[_v]] = bool(_its)

VERB_TAKES_QTY = np.zeros(N_VERBS, dtype=bool)
VERB_TAKES_QTY[VERB_INDEX["PICKUP"]] = True
VERB_TAKES_QTY[VERB_INDEX["PLACE"]] = True

# --------------------------------------------------------------------------------------
# Market ops
# --------------------------------------------------------------------------------------

MARKET_OPS = ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND")
MARKET_OP_INDEX = {o: i for i, o in enumerate(MARKET_OPS)}
N_MARKET_OPS = len(MARKET_OPS)
M_SELL, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_HIRE, M_BUY_LAND = range(N_MARKET_OPS)

# `_parse_order` (kag.py:626-645): HIRE and BUY_LAND are 1-long; the other four are exactly 3-long.
MARKET_OP_ARITY = {
    "SELL": (2, 2), "BUY_SEED": (2, 2), "BUY_PRODUCT": (2, 2), "BUY_ANIMAL": (2, 2),
    "HIRE": (0, 0), "BUY_LAND": (0, 0),
}
MARKET_OP_ITEMS = {
    "SELL": PRODUCTS,                 # `item in PRODUCTS` (:597)
    "BUY_SEED": CROPS,                # `item in CROPS`    (:601)
    "BUY_PRODUCT": BUYABLE_PRODUCTS,  # WHEAT/FERTILIZER   (:598)
    "BUY_ANIMAL": ANIMALS,            # `item in ANIMALS`  (:603)
    "HIRE": (), "BUY_LAND": (),
}
MARKET_OP_ATOMIC = ("HIRE", "BUY_LAND")   # resolved once, in player order (:572-580)

MARKET_OP_TAKES_ITEM = np.zeros(N_MARKET_OPS, dtype=bool)
for _o, _its in MARKET_OP_ITEMS.items():
    MARKET_OP_TAKES_ITEM[MARKET_OP_INDEX[_o]] = bool(_its)

# The item alphabet a market op is allowed to draw from, as a mask over N_ITEM_SLOTS.
MARKET_OP_ITEM_ALPHABET = np.zeros((N_MARKET_OPS, N_ITEM_SLOTS), dtype=bool)
for _o, _its in MARKET_OP_ITEMS.items():
    _oi = MARKET_OP_INDEX[_o]
    if _its:
        for _it in _its:
            MARKET_OP_ITEM_ALPHABET[_oi, ITEM_INDEX[_it]] = True
    else:
        MARKET_OP_ITEM_ALPHABET[_oi, ITEM_NONE] = True

VERB_ITEM_ALPHABET = np.zeros((N_VERBS, N_ITEM_SLOTS), dtype=bool)
for _v, _its in VERB_ITEMS.items():
    _vi = VERB_INDEX[_v]
    if _its:
        for _it in _its:
            VERB_ITEM_ALPHABET[_vi, ITEM_INDEX[_it]] = True
    else:
        VERB_ITEM_ALPHABET[_vi, ITEM_NONE] = True

MACRO_VERB_ITEM_ALPHABET = np.zeros((N_MACRO_VERBS, N_ITEM_SLOTS), dtype=bool)
for _v in MACRO_VERBS:
    _mi = MACRO_VERB_INDEX[_v]
    if _v == "MOVE":
        MACRO_VERB_ITEM_ALPHABET[_mi, ITEM_NONE] = True
    else:
        MACRO_VERB_ITEM_ALPHABET[_mi] = VERB_ITEM_ALPHABET[VERB_INDEX[_v]]

# --------------------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------------------

# PLAN_BC Ch5: buckets, never a regression head -- quantities are small integers with a very
# peaked distribution.  Measured over the corpus: unit quantities span 1..16, market quantities
# 1..999 with a long thin tail (the 999 is a single "sell everything" order).
QTY_BINS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64)   # the last bin means "64 or more"
QTY_ALL = len(QTY_BINS)                                     # index 13: "as many as are available"
N_QTY = len(QTY_BINS) + 1
assert N_QTY == 14

_QTY_EDGES = np.array(QTY_BINS)


def encode_qty(n: int, available: int | None = None) -> int:
    """Bucket a requested quantity.  Returns `QTY_ALL` when the request means "everything".

    `available` is the number of units the request could possibly move right now (shed contents
    for a SELL, carried units for a PLACE, and so on).  PLAN_BC Ch3 measured that 216 of 412 SELL
    orders in the sample game ask for more than the shed holds: those are "sell the lot" intents,
    and bucketing them numerically would throw the intent away.  Pass `available=None` to bucket
    numerically regardless.
    """
    n = int(n)
    if available is not None and available > 0 and n >= available:
        return QTY_ALL
    if n <= 0:
        return 0
    return int(np.searchsorted(_QTY_EDGES, n, side="right") - 1)


def decode_qty(bucket: int, available: int = 0) -> int:
    """Inverse of `encode_qty`, for turning a head's pick back into a request."""
    if bucket == QTY_ALL:
        return max(1, int(available))
    return QTY_BINS[int(bucket)]


def qty_mask(max_n: int) -> np.ndarray:
    """Legal quantity buckets when at most `max_n` units can move.  PASS-safe: never all-False."""
    m = np.zeros(N_QTY, dtype=bool)
    if max_n <= 0:
        return m
    m[: int(np.searchsorted(_QTY_EDGES, max_n, side="right"))] = True
    m[QTY_ALL] = True
    return m


# --------------------------------------------------------------------------------------
# Encoding / decoding whole actions
# --------------------------------------------------------------------------------------

class VocabError(ValueError):
    """An action names something outside the vocabulary.  Never swallowed."""


def encode_unit_action(action, available_fn=None):
    """`["PICKUP", "WHEAT", 6]` -> `(verb_id, item_id, qty_bin, qty_raw, n_ignored_extra)`.

    `available_fn(verb, item)` optionally returns the count that would make this request an "ALL".
    """
    if not isinstance(action, list) or not action:
        raise VocabError(f"unit action is not a non-empty list: {action!r}")
    verb = action[0]
    if verb not in VERB_INDEX:
        raise VocabError(f"unknown verb {verb!r}")
    vi = VERB_INDEX[verb]
    need, keep = VERB_ARITY[verb]
    args = action[1:]
    if len(args) < need:
        raise VocabError(f"{verb} needs {need} argument(s), got {args!r}")
    extra = max(0, len(args) - keep)

    item_id = ITEM_NONE
    if keep >= 1 and args:
        item = args[0]
        allowed = VERB_ITEMS[verb]
        if allowed:
            if item not in ITEM_INDEX:
                raise VocabError(f"{verb} names unknown item {item!r}")
            if item not in allowed:
                raise VocabError(f"{verb} may not name {item!r}")
            item_id = ITEM_INDEX[item]

    qty_raw = 1
    if VERB_TAKES_QTY[vi] and len(args) >= 2:
        qty_raw = int(args[1])
    avail = None
    if available_fn is not None and item_id != ITEM_NONE:
        avail = available_fn(verb, ITEMS[item_id])
    qty_bin = encode_qty(qty_raw, avail) if VERB_TAKES_QTY[vi] else 0
    return vi, item_id, qty_bin, qty_raw, extra


def encode_market_order(order, available_fn=None):
    """`["SELL", "WHEAT", 20]` -> `(op_id, item_id, qty_bin, qty_raw, n_ignored_extra)`."""
    if not isinstance(order, list) or not order:
        raise VocabError(f"market order is not a non-empty list: {order!r}")
    op = order[0]
    if op not in MARKET_OP_INDEX:
        raise VocabError(f"unknown market op {op!r}")
    oi = MARKET_OP_INDEX[op]
    need, keep = MARKET_OP_ARITY[op]
    args = order[1:]
    if len(args) < need:
        raise VocabError(f"{op} needs {need} argument(s), got {args!r}")
    extra = max(0, len(args) - keep)

    item_id, qty_raw = ITEM_NONE, 1
    if need == 2:
        item = args[0]
        if item not in ITEM_INDEX:
            raise VocabError(f"{op} names unknown item {item!r}")
        if item not in MARKET_OP_ITEMS[op]:
            raise VocabError(f"{op} may not name {item!r}")
        item_id = ITEM_INDEX[item]
        qty_raw = int(args[1])
    avail = None
    if available_fn is not None and item_id != ITEM_NONE:
        avail = available_fn(op, ITEMS[item_id])
    qty_bin = encode_qty(qty_raw, avail) if need == 2 else 0
    return oi, item_id, qty_bin, qty_raw, extra


def decode_unit_action(verb_id, item_id=ITEM_NONE, qty_bin=0, available=0):
    verb = VERBS[int(verb_id)]
    _, keep = VERB_ARITY[verb]
    out = [verb]
    if keep >= 1 and item_id != ITEM_NONE:
        out.append(ITEMS[int(item_id)])
        if VERB_TAKES_QTY[int(verb_id)]:
            out.append(decode_qty(qty_bin, available))
    return out


def decode_market_order(op_id, item_id=ITEM_NONE, qty_bin=0, available=0):
    op = MARKET_OPS[int(op_id)]
    if MARKET_OP_ARITY[op][0] == 0:
        return [op]
    return [op, ITEMS[int(item_id)], decode_qty(qty_bin, available)]


# --------------------------------------------------------------------------------------
# The corpus check
# --------------------------------------------------------------------------------------

def validate_against_corpus(actions_iter, strict_arity: bool = False) -> dict:
    """Assert every action string in the data is in this vocabulary.

    `actions_iter` yields the per-step action dicts (`{"farmer": [...], "hands": [...],
    "market": [...]}`) exactly as they appear in the replay.  Anything naming a verb, op or item
    this module does not know raises `VocabError` -- a vocabulary that lags the data is precisely
    the "one-entry disagreement" PLAN_BC Ch3 warns makes Assertion 3 look like a data problem.

    Trailing arguments the environment never reads are *counted*, not rejected, unless
    `strict_arity`.  Returns a counter dict; every key is printed by `build_shards.py`.
    """
    c = {
        "n_actions": 0, "n_unit_actions": 0, "n_market_orders": 0,
        "n_ignored_extra_args": 0, "n_orders_over_cap": 0, "n_units_over_cap": 0,
        "max_unit_qty": 0, "max_market_qty": 0, "max_units": 0, "max_orders": 0,
    }
    verbs_seen, ops_seen, items_seen = set(), set(), set()
    for action in actions_iter:
        c["n_actions"] += 1
        units = [action.get("farmer", ["PASS"])] + list(action.get("hands") or [])
        c["max_units"] = max(c["max_units"], len(units))
        if len(units) > MAX_UNITS:
            c["n_units_over_cap"] += 1
        for u in units:
            vi, ii, _qb, qr, extra = encode_unit_action(u)
            c["n_unit_actions"] += 1
            c["n_ignored_extra_args"] += extra
            c["max_unit_qty"] = max(c["max_unit_qty"], qr)
            verbs_seen.add(VERBS[vi])
            if ii != ITEM_NONE:
                items_seen.add(ITEMS[ii])
            if strict_arity and extra:
                raise VocabError(f"unexpected trailing arguments: {u!r}")
        orders = list(action.get("market") or [])
        c["max_orders"] = max(c["max_orders"], len(orders))
        if len(orders) > MAX_MARKET_ORDERS:
            c["n_orders_over_cap"] += 1
        for o in orders:
            oi, ii, _qb, qr, extra = encode_market_order(o)
            c["n_market_orders"] += 1
            c["n_ignored_extra_args"] += extra
            c["max_market_qty"] = max(c["max_market_qty"], qr)
            ops_seen.add(MARKET_OPS[oi])
            if ii != ITEM_NONE:
                items_seen.add(ITEMS[ii])
            if strict_arity and extra:
                raise VocabError(f"unexpected trailing arguments: {o!r}")
    c["verbs_seen"] = sorted(verbs_seen)
    c["market_ops_seen"] = sorted(ops_seen)
    c["items_seen"] = sorted(items_seen)
    return c
