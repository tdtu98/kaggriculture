"""REAL denial version (gate dropped) -- demonstration.

Withholds our premium goods as ammunition, then DUMPS the whole reserve the
moment the opponent's visible farm shows ripe premium of that product, to floor
the price on their harvest. This is genuine denial (adds/removes orders), not
reordering. Expected to tie/lose vs a mirror because the maneuver is symmetric.
"""
import importlib.util
_s=importlib.util.spec_from_file_location("stock","/home/claude/main.py")
stock=importlib.util.module_from_spec(_s); _s.loader.exec_module(stock)
PREMIUM=["MILK","WOOL","MELON","STRAWBERRY","TOMATO"]
CROP2PROD={"TOMATO":"TOMATO","STRAWBERRY":"STRAWBERRY","MELON":"MELON"}
ANIM2PROD={"COW":"MILK","SHEEP":"WOOL"}
RIPE_TRIGGER=3     # opponent has >=3 ripe units of P -> dump our reserve to deny
SHED_RELEASE=55    # relieve shed pressure before the 100-cap discards our reserve

def _opp_ripe(opp):
    ripe={}
    for row in (stock._get(opp,"tiles",[]) or []):
        for t in row:
            if not isinstance(t,dict): continue
            yu=int(t.get("yield_units",0) or 0)
            if t.get("kind")=="PLANT": p=CROP2PROD.get(t.get("crop"))
            elif t.get("animal"): p=ANIM2PROD.get(t.get("animal"))
            else: p=None
            if p: ripe[p]=ripe.get(p,0)+yu
    return ripe

def agent(obs):
    a=stock.agent(obs)
    try:
        seat=stock._seat(obs); opp=stock._farm(obs,1-seat)
        step=int(stock._get(obs,"step",0) or 0)
        shed=stock._get(stock._get(obs,"private",{}) or {},"shed",{}) or {}
        opp_ripe=_opp_ripe(opp)
        terminal=step>=690
        market=[list(o) for o in (a.get("market") or [])]
        for P in PREMIUM:
            held=int(shed.get(P,0) or 0)
            dump = terminal or opp_ripe.get(P,0)>=RIPE_TRIGGER or held>=SHED_RELEASE
            # strip Boatlee's own sells of P (we control P's timing now)
            market=[o for o in market if not (len(o)>=2 and o[0]=="SELL" and o[1]==P)]
            if dump and held>0:
                market.append(["SELL",P,held])     # fire the whole reserve to floor their harvest
            # else: withhold -> P accumulates in shed as ammunition
        a["market"]=market[:10]
        a=stock._rank_sell_slots(obs,a,stock._V16_OFFICIAL_CONFIGURATION)
        a=stock._align_hands(a,obs)
    except Exception:
        pass
    return a
