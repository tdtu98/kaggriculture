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

def plan(obs, n_tiles, pool=None):
    """Allocate by value-density but capped by empirical MARKET ABSORPTION, not
    shop demand. Town+shops drain strawberry/milk/wheat almost unboundedly (they
    stay scarce and high-priced), while melon/wool floor after a few tiles. So we
    fill crop tiles with the best high-absorption product (strawberry) after
    taking a few of the glut-prone high-value ones (melon)."""
    # tiles a product can occupy before its market floors (empirical, vs town drain)
    ABSORB={"STRAWBERRY":60,"MILK":60,"WHEAT":60,"TOMATO":10,"CARROT":10,
            "EGG":10,"MELON":4,"WOOL":6}
    # small adaptive nudge toward products this game's shops actively demand
    town=stock._get(obs,"town",{}) or {}
    shops=set(stock._get(town,"unlocked_shops",[]) or [])
    demanded=set()
    for s in shops:
        for p in _SHOP_PRODUCTS.get(s,()): demanded.add(p)
    val={}
    for item,(base,ypd,cost,is_animal) in ECON.items():
        v=base*ypd
        if item in demanded: v*=1.3
        val[item]=v
    items=[it for it in ECON if (pool is None or it in pool)]
    order=sorted(items,key=lambda it:val[it],reverse=True)
    assignment={}; remaining=n_tiles
    for item in order:
        if remaining<=0: break
        take=min(remaining, ABSORB.get(item,6))
        if take>0:
            assignment[item]=take; remaining-=take
    # any leftover tiles -> the best high-absorption fillers
    fillers=[f for f in ("STRAWBERRY","MILK","WHEAT") if (pool is None or f in pool)] or order
    for fill in fillers:
        if remaining<=0: break
        assignment[fill]=assignment.get(fill,0)+remaining; remaining=0
    return assignment
