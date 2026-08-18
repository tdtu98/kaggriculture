"""Candidate: stock production core + market-headroom sell controller.

Meters premium goods so scheduled dumps don't crater the price to $1,
while guaranteeing terminal liquidation (unsold == worthless) and
relieving shed pressure (>shed cap == discarded).
"""
import importlib.util
_spec = importlib.util.spec_from_file_location("stock", "/home/claude/main.py")
stock = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stock)

_mp = stock._market_price
_PARAMS = stock._MARKET_PARAMS

# above_target: STRAWBERRY 1.6, MELON 3.6, WOOL 3.2, MILK 1.6, TOMATO 0.6
PREMIUM = {"STRAWBERRY", "MELON", "WOOL", "MILK", "TOMATO"}
BASE = {k: _PARAMS[k][0] for k in _PARAMS}

TERMINAL_STEP = 690      # last ~day: dump everything
SHED_PRESSURE = 85       # shed cap 100; relax metering above this
FLOOR_K = 0.45           # keep premium price >= 0.45 * base

def _floor_price(item, step, shed_total):
    if step >= TERMINAL_STEP or shed_total >= SHED_PRESSURE:
        return 1
    return max(1, int(BASE[item] * FLOOR_K))

def _safe_qty(item, inv, requested, floor):
    """Largest n<=requested with post-sell price still >= floor."""
    if _mp(item, inv + requested) >= floor:
        return requested
    if _mp(item, inv + 1) < floor:
        return 0
    lo, hi = 0, requested
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _mp(item, inv + mid) >= floor:
            lo = mid
        else:
            hi = mid - 1
    return lo

def agent(obs):
    a = stock.agent(obs)
    try:
        step = int(obs.get("step", 0) or 0)
        shed = (obs.get("private") or {}).get("shed") or {}
        shed_total = sum(max(0, int(v or 0)) for v in shed.values())
        inv = ((obs.get("market") or {}).get("inventory") or {})
        market = [list(o) for o in (a.get("market") or [])]
        out = []
        for o in market:
            if len(o) >= 3 and o[0] == "SELL" and o[1] in PREMIUM:
                item = o[1]
                cur_inv = int(inv.get(item, 10000) or 10000)
                req = max(0, int(o[2]))
                floor = _floor_price(item, step, shed_total)
                n = _safe_qty(item, cur_inv, req, floor)
                if n <= 0:
                    continue          # hold this premium good for a better price
                o = [o[0], o[1], n]
            out.append(o)
        a["market"] = out[:10]
        a = stock._align_hands(a, obs)
    except Exception:
        pass
    return a
