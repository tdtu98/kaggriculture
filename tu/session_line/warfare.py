"""Opponent-aware market-warfare layer on top of Boatlee (champion).

Only the MARKET channel is touched; Boatlee's elite production is untouched.
In a true mirror it passes Boatlee through UNCHANGED (preserves the tie).
Against a DIFFERENT opponent it (a) front-runs the premium goods the opponent
is about to flood, and (b) diverts our sells toward lanes the opponent under-
serves -- capturing high prices before their glut and denying them theirs.
"""
import importlib.util
_s=importlib.util.spec_from_file_location("stock","/home/claude/main.py")
stock=importlib.util.module_from_spec(_s); _s.loader.exec_module(stock)
_mp=stock._market_price
CFG=stock._V16_OFFICIAL_CONFIGURATION
PREMIUM=["MILK","WOOL","MELON","STRAWBERRY","TOMATO"]
CROP2PROD={"WHEAT":"WHEAT","CARROT":"CARROT","TOMATO":"TOMATO","STRAWBERRY":"STRAWBERRY","MELON":"MELON"}
ANIM2PROD={"COW":"MILK","SHEEP":"WOOL","GOOSE":"EGG"}

def _profile(farm):
    """Per-product 'imminent supply' (ripe yield on the farm) + capacity."""
    ripe={}; cap={}
    for row in (stock._get(farm,"tiles",[]) or []):
        for t in row:
            if not isinstance(t,dict): continue
            k=t.get("kind"); yu=int(t.get("yield_units",0) or 0)
            if k=="PLANT":
                p=CROP2PROD.get(t.get("crop"))
            elif k in ("PASTURE","COOP") and t.get("animal"):
                p=ANIM2PROD.get(t.get("animal"))
            else:
                p=None
            if p:
                cap[p]=cap.get(p,0)+1
                ripe[p]=ripe.get(p,0)+yu
    return ripe,cap

def _is_mirror(me,opp):
    _,mc=_profile(me); _,oc=_profile(opp)
    keys=set(mc)|set(oc)
    diff=sum(abs(mc.get(k,0)-oc.get(k,0)) for k in keys)
    return diff<=2   # near-identical production => treat as mirror, stay passive

def agent(obs):
    a=stock.agent(obs)
    try:
        seat=stock._seat(obs); me=stock._farm(obs,seat); opp=stock._farm(obs,1-seat)
        if _is_mirror(me,opp):
            return a  # preserve the mirror tie: no warfare vs a clone
        opp_ripe,opp_cap=_profile(opp)
        my_ripe,my_cap=_profile(me)
        # asymmetry: >0 => opponent floods this product more than we do
        flood={p: opp_cap.get(p,0)-my_cap.get(p,0) for p in PREMIUM}
        market=[list(o) for o in (a.get("market") or [])]
        sells=[o for o in market if len(o)>=2 and o[0]=="SELL"]
        rest =[o for o in market if o not in sells]
        # front-run priority: products the opponent is about to flood / is ripe on,
        # and that we under-produce (so denial hurts them more than us) go FIRST.
        def prio(o):
            p=o[1]
            imminent = opp_ripe.get(p,0)          # they're about to sell it
            asym     = max(0,flood.get(p,0))      # they out-produce us here
            return (imminent + 40*asym)           # higher => sell earlier (grab price first)
        sells.sort(key=prio, reverse=True)
        a["market"]=(sells+rest)[:10]
        a=stock._align_hands(a,obs)
    except Exception:
        pass
    return a
