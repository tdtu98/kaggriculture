"""Delivered observation -> fixed-shape token arrays (PLAN_BC Chapter 5).

A **token** is a short list of numbers describing one thing.  Orbit Wars made one per planet; we
make one per farm tile (100), one per worker slot (16, padded), one per market product (9), one
global, and one opponent summary.  Everything is deterministic from the observation the agent was
actually handed -- no kagsim state, no replay-only fields (that seam is a Chapter 8 entry
requirement, and dragging `make verify` into scope now would buy nothing).

Three design decisions carried over from the plan, each with its reason:

* **Seat is not a feature.**  Everything canonicalizes into "me" and "them".  Ryo plays both seats
  across the corpus, so canonicalizing makes both one distribution instead of two.
* **The opponent gets one summary token, not 100 tile tokens.**  Their tile detail reaches us
  through exactly one channel -- future market supply -- so that is what we compute.
* **`n_shops_buying_me` and `town_drain_per_day` are in the product token deliberately.**  They
  encode D17 (rank a product by how many shops want it, not by its price curve -- judging by the
  curve ranked the markets almost backwards and cost a 2.4x gain).  `T` and `above_func` encode the
  *correction* to D17 (E48/E41): melon at 114 units a season is excellent and melon at 360 units is
  worthless.  Both terms belong in the token; the model works out the trade-off.

`FEATURE_VERSION` is asserted on shard load and on checkpoint load.  Silent feature/weight skew
produces a model that scores like an untrained one and is indistinguishable from "BC didn't work".
"""

from __future__ import annotations

import math

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as kag

from . import masks as M
from . import vocab as V

FEATURE_VERSION = 1

# Storage dtype.  Every feature below is a small magnitude (money is divided by 10,000, counts by
# their caps), so float16's ~3 decimal digits are ample and it halves the shards.
STORE_DTYPE = np.float16

TILE_KINDS = ("LOCKED", "EMPTY", "PLANT", "WEED", "COOP", "PASTURE", "COOP_ANIMAL", "PASTURE_ANIMAL")
SHAPE_FUNCS = ("linear", "sq", "sqrt", "log", "hinge")
QUADRANTS = ("NW", "NE", "SW", "SE")

N_TILE_FEATS = 45
N_WORKER_FEATS = 26
N_PRODUCT_FEATS = 26
N_GLOBAL_FEATS = 28
N_OPPONENT_FEATS = 7

FEATURE_SHAPES = {
    "tiles": (V.N_TILES, N_TILE_FEATS),
    "workers": (V.MAX_UNITS, N_WORKER_FEATS),
    "worker_mask": (V.MAX_UNITS,),
    "products": (len(V.PRODUCTS), N_PRODUCT_FEATS),
    "global": (N_GLOBAL_FEATS,),
    "opponent": (N_OPPONENT_FEATS,),
}

# Which product each animal yields, and which tile-kinds produce which product.
_ANIMAL_PRODUCT = {a: kag.ANIMALS[a]["product"] for a in kag.ANIMALS}
_SHED_ACCESS_LIST = sorted(M.SHED_ACCESS)


def _dist_to_shed(x, y):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in _SHED_ACCESS_LIST)


_DIST_TO_SHED = np.array([[_dist_to_shed(x, y) for x in range(V.GRID)] for y in range(V.GRID)],
                         dtype=np.float32)
_QUAD_OF = [[kag._quadrant_of(x, y, V.GRID) for x in range(V.GRID)] for y in range(V.GRID)]


def _onehot(out, base, options, value):
    if value in options:
        out[base + options.index(value)] = 1.0


# --------------------------------------------------------------------------------------
# Shop demand and town drain -- D17's lesson, made into two numbers
# --------------------------------------------------------------------------------------

def shop_demand(town, cfg=None):
    """`(n_shops_buying[item], drain_per_day[item])`.

    `_town_consume` (`kag.py:723-742`): every unlocked shop *instance* consumes independently every
    `townShopSellInterval` steps, at 2x when the shop wants a single product; the town centre
    consumes one of every non-fertilizer product every `townCenterSellInterval` steps.  Shops are
    drawn with replacement, so `unlocked_shops` may list the same shop several times and each copy
    counts.
    """
    shop_interval = int((cfg or {}).get("townShopSellInterval", 4))
    center_interval = int((cfg or {}).get("townCenterSellInterval", 24))
    n_shops = {p: 0 for p in V.PRODUCTS}
    drain = {p: 0.0 for p in V.PRODUCTS}
    per_day_shop = V.TURNS_PER_DAY / max(1, shop_interval)
    per_day_center = V.TURNS_PER_DAY / max(1, center_interval)
    for name in (town or {}).get("unlocked_shops", []):
        products = kag.SHOPS[name]
        mult = 2 if len(products) == 1 else 1
        for item in products:
            n_shops[item] += 1
            drain[item] += mult * per_day_shop
    for item in kag.TOWN_CENTER_PRODUCTS:
        drain[item] += per_day_center
    return n_shops, drain


def forecast_supply(farm, day, horizons=(0, 3, 7)):
    """How many units of each product a farm's tiles will have produced by `day + h`.

    Deliberately simple and public-information only: current `yield_units` plus what the growth
    rules will add, from `farms[p].tiles`, which is fully visible for both seats.  This is the one
    channel through which the opponent's farm reaches us (PLAN_BC Ch5), so it is computed for both
    and it is why the opponent needs no tile tokens of its own.
    """
    out = {h: {p: 0.0 for p in V.PRODUCTS} for h in horizons}
    for tile in M.flat_tiles(farm):
        if not isinstance(tile, dict):
            continue
        if tile.get("kind") == "PLANT":
            crop = tile["crop"]
            cd = kag.CROPS[crop]
            age = day - tile["planted_day"]
            for h in horizons:
                units = tile.get("yield_units", 0)
                if cd["ongoing"]:
                    since = age + h - cd["first_yield_day"]
                    if since >= 0:
                        units = min(cd["max_yield"], units + since // max(1, cd["interval"]) + 1)
                else:
                    if age + h >= cd["first_yield_day"]:
                        units = min(cd["max_yield"], units + h)
                out[h][crop] += units
        elif "animal" in tile:
            a = kag.ANIMALS[tile["animal"]]
            product = a["product"]
            age = day - tile["placed_day"]
            for h in horizons:
                units = tile.get("yield_units", 0)
                since = age + h - a["first_yield_day"]
                if since >= 0:
                    units = min(a["max_held"], units + since // max(1, a["interval"]) + 1)
                out[h][product] += units
    return out


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------

def tile_tokens(obs, me, unit_positions):
    day = int(obs["day"])
    farm = obs["farms"][me]
    tiles = M.flat_tiles(farm)
    prices = obs["market"]["prices"]
    unlocked = set(farm.get("unlocked_quadrants") or ["NW"])
    occupancy = np.zeros(V.N_TILES, dtype=np.float32)
    for p in unit_positions:
        occupancy[M.tile_index(int(p[0]), int(p[1]))] += 1.0

    out = np.zeros((V.N_TILES, N_TILE_FEATS), dtype=np.float32)
    for i, tile in enumerate(tiles):
        x, y = M.tile_xy(i)
        f = out[i]
        is_dict = isinstance(tile, dict)
        has_animal = is_dict and "animal" in tile
        kind = tile.get("kind") if is_dict else ("LOCKED" if tile == "LOCKED" else "EMPTY")
        if has_animal:
            kind = f"{kind}_ANIMAL"
        _onehot(f, 0, TILE_KINDS, kind)                                              # [0:8]  kind
        product = None
        if is_dict and kind == "PLANT":
            _onehot(f, 8, V.CROPS, tile["crop"])                                     # [8:13] crop
            product = tile["crop"]
        if has_animal:
            _onehot(f, 13, V.ANIMALS, tile["animal"])                                # [13:16] animal
            product = _ANIMAL_PRODUCT[tile["animal"]]

        if is_dict and tile.get("kind") == "PLANT":                                  # [16:25] plant
            cd = kag.CROPS[tile["crop"]]
            age = day - tile["planted_day"]
            f[16] = 1.0 if tile["watered_today"] else 0.0
            f[17] = tile["consecutive_unwatered"] / 2.0     # 2 dry days -> WEED (kag.py:806-810)
            f[18] = age / 30.0
            f[19] = tile.get("yield_units", 0) / 6.0
            f[20] = 1.0 if tile.get("fertilized_until_day", -1) >= day else 0.0
            f[21] = max(0, tile.get("fertilized_until_day", -1) - day) / 2.0
            f[22] = 1.0 if (tile.get("yield_units", 0) > 0 and age >= cd["first_yield_day"]) else 0.0
            mls = tile.get("max_lifespan_step", -1)
            f[23] = 0.0 if mls < 0 else max(0.0, min(1.0, (mls - int(obs["step"])) / 48.0))
            f[24] = 1.0 if cd["ongoing"] else 0.0

        if has_animal:                                                               # [25:32] animal
            a = kag.ANIMALS[tile["animal"]]
            age = day - tile["placed_day"]
            f[25] = 1.0 if tile["fed_today"] else 0.0
            f[26] = 1.0 if tile["cared_today"] else 0.0
            f[27] = tile["consecutive_unfed"] / 2.0        # 2 unfed days -> escapes (kag.py:846-850)
            f[28] = tile.get("yield_units", 0) / 6.0
            f[29] = 1.0 if tile["fertilizer_available"] else 0.0
            f[30] = age / 30.0
            since = age - a["first_yield_day"]
            f[31] = 0.0 if since < 0 else 1.0 - (since % max(1, a["interval"])) / max(1, a["interval"])

        price = float(prices.get(product, 0)) if product else 0.0                    # [32:35] econ
        f[32] = price / 250.0
        f[33] = (tile.get("yield_units", 0) if is_dict else 0) / 6.0
        f[34] = price * (tile.get("yield_units", 0) if is_dict else 0) / 1000.0

        f[35] = x / (V.GRID - 1.0)                                                   # [35:45] geom
        f[36] = y / (V.GRID - 1.0)
        f[37] = _DIST_TO_SHED[y][x] / 18.0
        f[38] = 1.0 if (x, y) in M.SHED_ACCESS else 0.0
        _onehot(f, 39, QUADRANTS, _QUAD_OF[y][x])
        f[43] = 1.0 if _QUAD_OF[y][x] in unlocked else 0.0
        f[44] = min(occupancy[i], 4.0) / 4.0
    return out


def worker_tokens(obs, me):
    farm = obs["farms"][me]
    priv = obs["private"]
    hour = int(obs["hour"])
    positions = [farm["farmer"]] + list(farm["hands"])
    invs = priv.get("inventories") or [{}]

    out = np.zeros((V.MAX_UNITS, N_WORKER_FEATS), dtype=np.float32)
    mask = np.zeros(V.MAX_UNITS, dtype=np.float32)
    for u in range(min(V.MAX_UNITS, len(positions))):
        mask[u] = 1.0
        f = out[u]
        x, y = int(positions[u][0]), int(positions[u][1])
        inv = invs[u] if u < len(invs) else {}
        f[0] = 1.0 if u == 0 else 0.0
        f[1] = u / (V.MAX_UNITS - 1.0)
        f[2] = x / (V.GRID - 1.0)
        f[3] = y / (V.GRID - 1.0)
        f[4] = 1.0 if (x, y) in M.SHED_ACCESS else 0.0
        f[5] = _DIST_TO_SHED[y][x] / 18.0
        for j, item in enumerate(V.ITEMS):
            f[6 + j] = min(inv.get(item, 0), 12) / 12.0                              # [6:18]
        total = sum(inv.values())
        f[18] = min(total, 12) / 12.0
        f[19] = 1.0 if total == 0 else 0.0
        f[20] = (V.TURNS_PER_DAY - 1 - hour) / (V.TURNS_PER_DAY - 1.0)
        f[21] = hour / (V.TURNS_PER_DAY - 1.0)
        f[22] = 1.0
        f[23] = 1.0 if inv.get("WHEAT", 0) >= 1 else 0.0                # FEED needs wheat  :505-512
        f[24] = 1.0 if inv.get("FERTILIZER", 0) >= 1 else 0.0           # FERTILIZE         :475-478
        f[25] = 1.0 if any(inv.get(a, 0) >= 1 for a in V.ANIMALS) else 0.0   # PLACE-animal :381-392
    return out, mask


def product_tokens(obs, me, effective_shed=None):
    market = obs["market"]
    prices, inv = market["prices"], market["inventory"]
    params = market.get("params") or kag.MARKET_PARAMS
    town = obs.get("town") or {}
    day = int(obs["day"])
    shed = obs["private"]["shed"]
    eff = effective_shed if effective_shed is not None else shed
    n_shops, drain = shop_demand(town)
    mine = forecast_supply(obs["farms"][me], day)
    theirs = forecast_supply(obs["farms"][1 - me], day)

    out = np.zeros((len(V.PRODUCTS), N_PRODUCT_FEATS), dtype=np.float32)
    for j, item in enumerate(V.PRODUCTS):
        p = params[item]
        f = out[j]
        T = max(1.0, float(p["T"]))
        f[0] = (inv[item] - p["I0"]) / T
        f[1] = prices[item] / max(1.0, float(p["base"]))
        f[2] = prices[item] / 250.0
        f[3] = n_shops[item] / 8.0                       # D17: how many shops want it
        f[4] = drain[item] / 24.0                        # D17: and how fast they drain it
        f[5] = T / 450.0
        _onehot(f, 6, SHAPE_FUNCS, p["above_func"])                                  # [6:11]
        f[11] = float(p["above_target"]) / 4.0
        f[12] = float(p["below_target"]) / 2.0
        f[13] = min(shed.get(item, 0), 100) / 100.0
        f[14] = min(eff.get(item, 0), 100) / 100.0
        f[15] = mine[0][item] / T
        f[16] = mine[3][item] / T
        f[17] = mine[7][item] / T
        f[18] = theirs[0][item] / T
        f[19] = theirs[3][item] / T
        f[20] = theirs[7][item] / T
        f[21] = (prices[item] - kag.market_price(item, inv[item] + 1, market.get("params"))) / 10.0
        supply7 = mine[7][item] + theirs[7][item]
        f[22] = kag.market_price(item, inv[item] + int(supply7), market.get("params")) / 250.0
        f[23] = 1.0 if item in V.BUYABLE_PRODUCTS else 0.0                  # BUY_PRODUCT  :598
        f[24] = 1.0 if item in V.CROPS else 0.0
        f[25] = (kag.CROPS[item]["seed"] / 100.0) if item in V.CROPS else 0.0
    return out


def global_token(obs, me):
    farm = obs["farms"][me]
    opp = obs["farms"][1 - me]
    priv = obs["private"]
    step, day, hour = int(obs["step"]), int(obs["day"]), int(obs["hour"])
    shed_used = sum(priv["shed"].values())
    unlocked = list(farm.get("unlocked_quadrants") or ["NW"])
    extra = len(unlocked) - 1
    n_shops = len((obs.get("town") or {}).get("unlocked_shops", []))

    f = np.zeros(N_GLOBAL_FEATS, dtype=np.float32)
    f[0] = step / (V.EPISODE_STEPS - 1.0)
    f[1] = day / (V.DAYS - 1.0)
    f[2] = hour / (V.TURNS_PER_DAY - 1.0)
    f[3] = math.sin(2 * math.pi * hour / V.TURNS_PER_DAY)
    f[4] = math.cos(2 * math.pi * hour / V.TURNS_PER_DAY)
    # Hiring only ever happens at dawn -- 601 HIRE orders across 105 distinct turns, all at the
    # day's first two decision hours (PLAN_BC Ch3).  So the hour must be first-class.
    f[5] = 1.0 if hour == 0 else 0.0
    f[6] = 1.0 if hour <= 1 else 0.0
    f[7] = float(farm["money"]) / 10000.0
    f[8] = float(opp["money"]) / 10000.0
    f[9] = (float(farm["money"]) - float(opp["money"])) / 10000.0
    f[10] = shed_used / float(V.SHED_CAPACITY)
    f[11] = max(0, V.SHED_CAPACITY - shed_used) / float(V.SHED_CAPACITY)
    for j, crop in enumerate(V.CROPS):
        f[12 + j] = min(priv["seeds"].get(crop, 0), 20) / 20.0                        # [12:17]
    f[17] = len(unlocked) / 4.0
    f[18] = (kag.LAND_PRICES[extra] / 4000.0) if extra < len(kag.LAND_ORDER) else 0.0
    f[19] = 1.0 if (extra < len(kag.LAND_ORDER)
                    and farm["money"] >= kag.LAND_PRICES[extra]) else 0.0
    f[20] = min(int(farm.get("hires_today", 0)), V.MAX_UNITS) / float(V.MAX_UNITS)
    f[21] = kag._hire_cost(int(farm.get("hires_today", 0))) / 100.0                   # Fibonacci :690
    f[22] = 1.0 if farm["money"] >= kag._hire_cost(int(farm.get("hires_today", 0))) else 0.0
    f[23] = (1 + len(farm["hands"])) / float(V.MAX_UNITS)
    f[24] = n_shops / float(kag.MAX_SHOP_INSTANCES)
    f[25] = (0.0 if n_shops >= kag.MAX_SHOP_INSTANCES
             else (3 - ((day + 1) % 3)) / 3.0)              # townShopUnlockInterval default 3
    f[26] = (V.EPISODE_STEPS - 1 - step) / (V.EPISODE_STEPS - 1.0)
    f[27] = 1.0 if day >= V.DAYS - 1 else 0.0
    return f


def opponent_token(obs, me, prev_obs=None):
    opp = obs["farms"][1 - me]
    tiles = M.flat_tiles(opp)
    planted = sum(1 for t in tiles if isinstance(t, dict) and t.get("kind") == "PLANT")
    animals = sum(1 for t in tiles if isinstance(t, dict) and "animal" in t)
    f = np.zeros(N_OPPONENT_FEATS, dtype=np.float32)
    f[0] = float(opp["money"]) / 10000.0
    f[1] = planted / 100.0
    f[2] = animals / 100.0
    f[3] = (1 + len(opp["hands"])) / float(V.MAX_UNITS)
    f[4] = len(opp.get("unlocked_quadrants") or ["NW"]) / 4.0
    f[5] = min(int(opp.get("hires_today", 0)), V.MAX_UNITS) / float(V.MAX_UNITS)
    if prev_obs is not None:
        f[6] = (float(opp["money"]) - float(prev_obs["farms"][1 - me]["money"])) / 1000.0
    return f


def extract(obs, me=None, prev_obs=None, effective_shed=None):
    """One delivered observation -> the five token arrays.  Deterministic, float32."""
    me = int(obs["player"]) if me is None else int(me)
    farm = obs["farms"][me]
    positions = [farm["farmer"]] + list(farm["hands"])
    workers, wmask = worker_tokens(obs, me)
    return {
        "tiles": tile_tokens(obs, me, positions),
        "workers": workers,
        "worker_mask": wmask,
        "products": product_tokens(obs, me, effective_shed),
        "global": global_token(obs, me),
        "opponent": opponent_token(obs, me, prev_obs),
    }


def check_shapes(feats):
    for k, shape in FEATURE_SHAPES.items():
        if feats[k].shape != shape:
            raise AssertionError(f"feature {k!r} has shape {feats[k].shape}, expected {shape}")
    for k, arr in feats.items():
        if not np.isfinite(arr).all():
            raise AssertionError(f"feature {k!r} contains non-finite values")
    return True
