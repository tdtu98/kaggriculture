"""Adaptive production planner for Kaggriculture.

The core thesis: the champion runs a FIXED menu and is hostage to shop luck
(same build earns 38k-101k depending on which shops unlock). This module
reads the realized demand each game and outputs a target tile allocation, so
production follows what THIS game actually rewards.

Interface:
    rank_products(obs)  -> ranked list of ProductScore
    plan(obs, n_tiles)  -> {product: target_tile_count}

Reuses the champion's calibrated market model and shop tables.
"""
import importlib.util
_spec = importlib.util.spec_from_file_location("stock", "/home/claude/main.py")
stock = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stock)

_market_price = stock._market_price
_SHOP_PRODUCTS = stock._SHOP_PRODUCTS
CONFIG = stock._V16_OFFICIAL_CONFIGURATION  # real env: townCenterSellInterval=24

# product -> (base_price, yield_per_tile_per_day, setup_cost, is_animal)
# yields/costs from README Object Types table.
ECON = {
    "WHEAT":      (25,  0.80, 10,  False),
    "CARROT":     (35,  0.75, 20,  False),
    "TOMATO":     (60,  0.33, 50,  False),
    "STRAWBERRY": (120, 0.24, 100, False),
    "MELON":      (250, 0.55, 80,  False),
    "EGG":        (50,  1.00, 301, True),   # goose 300 + coop 1
    "MILK":       (160, 0.50, 401, True),   # cow 400 + pasture 1
    "WOOL":       (200, 0.33, 501, True),   # sheep 500 + pasture 1
}
ALL_SHOPS = set(_SHOP_PRODUCTS)

def _current_demand(obs, item):
    """Units/day the town currently removes for `item` (keeps price near base)."""
    return stock._demand_per_day(obs, CONFIG, item)

def _future_demand_ev(obs, item):
    """Expected extra units/day from shops not yet unlocked (EV over the pool)."""
    town = stock._get(obs, "town", {}) or {}
    unlocked = set(stock._get(town, "unlocked_shops", []) or [])
    remaining = ALL_SHOPS - unlocked
    if not remaining:
        return 0.0
    tpd = int(CONFIG.get("turnsPerDay", 24))
    interval = int(CONFIG.get("townShopSellInterval", 4))
    # unlocks happen every 3 days; ~how many more will unlock this season
    day = int(stock._get(obs, "day", 0) or 0)
    days_left = max(0, 30 - day)
    slots_left = min(len(remaining), days_left // 3)
    if slots_left <= 0:
        return 0.0
    # per remaining shop, demand this item contributes if it's the one unlocked
    contrib = 0.0
    for shop in remaining:
        prods = _SHOP_PRODUCTS.get(shop, ())
        if item in prods:
            contrib += (tpd / interval) * (2 if len(prods) == 1 else 1)
    # probability-weighted: each of the next `slots_left` unlocks is uniform over `remaining`
    p_per_slot = 1.0 / len(remaining)
    return contrib * p_per_slot * slots_left * 0.5  # 0.5: discount (arrives mid-season)

class ProductScore:
    __slots__ = ("item", "value_density", "demand", "absorbable_tiles", "score")
    def __init__(self, item, value_density, demand, absorbable_tiles, score):
        self.item = item; self.value_density = value_density
        self.demand = demand; self.absorbable_tiles = absorbable_tiles
        self.score = score
    def __repr__(self):
        return (f"{self.item:<10} vd={self.value_density:6.1f} "
                f"demand/day={self.demand:5.1f} absorb_tiles={self.absorbable_tiles:4.1f}")

def rank_products(obs):
    rows = []
    for item, (base, ypd, cost, is_animal) in ECON.items():
        demand = _current_demand(obs, item) + _future_demand_ev(obs, item)
        value_density = base * ypd                    # revenue/tile/day at base price
        absorbable = demand / ypd if ypd > 0 else 0.0 # tiles the market clears near base
        # score = revenue you can capture near base = value_density * min(1, absorb/needed)
        # rank primarily by how much near-base revenue the product can yield this game
        rows.append(ProductScore(item, value_density, demand, absorbable,
                                 value_density * absorbable))
    rows.sort(key=lambda r: r.score, reverse=True)
    return rows

def plan(obs, n_tiles):
    """Greedy allocation driven by CURRENT demand (re-planned as shops unlock).
    Fill tiles with highest value-density products up to the tiles the market
    clears near base THIS game, so we never self-glut a thin lane."""
    caps = {}
    for item, (base, ypd, cost, is_animal) in ECON.items():
        cur = _current_demand(obs, item)                 # units/day now
        fut = 0.15 * _future_demand_ev(obs, item)        # small pre-investment nudge
        caps[item] = ((base * ypd), max(0.0, (cur + fut) / ypd) if ypd > 0 else 0.0)
    order = sorted(ECON, key=lambda it: caps[it][0], reverse=True)  # by value density
    assignment = {}; remaining = n_tiles
    for item in order:
        if remaining <= 0:
            break
        take = min(remaining, int(round(caps[item][1])))
        if take > 0:
            assignment[item] = take
            remaining -= take
    # leftover tiles: give to next-best value-density product past its cap
    # only if its decayed price still beats a fallow tile (skip if too glutted)
    if remaining > 0:
        for item in order:
            base, ypd, cost, is_animal = ECON[item]
            if remaining <= 0:
                break
            extra = remaining
            assignment[item] = assignment.get(item, 0) + extra
            remaining -= extra
            break
    return assignment
