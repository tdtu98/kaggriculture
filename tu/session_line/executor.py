"""Closed-loop executor v2: crops fund the early game; animals (milk/wool) are
added incrementally out of surplus cash and paced at <=1/day so we never go
bankrupt before yields land (milk day8, wool day6).

Design fixes over v1:
  - CASH_RESERVE protects crop/seed economy; animals bought only from surplus.
  - >=1 animal purchase per day, ramped, capped to feedable count.
  - Farmer tends animals only once animal work exists; else it farms crops too.
"""
import importlib.util
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
stock   = _load("stock",   "/home/claude/main.py")
planner = _load("planner", "/home/claude/planner.py")
_mp = stock._market_price
BASE = {k: stock._MARKET_PARAMS[k][0] for k in stock._MARKET_PARAMS}

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
CROP = {"WHEAT":(2,4,False),"CARROT":(2,3,False),"MELON":(10,10,False),
        "TOMATO":(8,8,True),"STRAWBERRY":(10,10,True)}
SEED_COST = {"WHEAT":10,"CARROT":20,"TOMATO":50,"STRAWBERRY":100,"MELON":80}
ANIMAL = {"COW":("MILK",8,2), "SHEEP":("WOOL",6,3)}
ANIMAL_COST = {"COW":400,"SHEEP":500}
CASH_RESERVE = 900          # keep crops/seeds/wheat funded before buying animals

ANIMAL_TILES = [(4,4),(3,4),(4,3),(3,3),(2,4),(4,2),(2,3),(3,2),(1,4)]   # hug shed corner; cap 9
CROP_TILES   = [(x,y) for y in range(5) for x in range(5) if (x,y) not in ANIMAL_TILES]
QUAD={"NW":[(x,y) for y in range(5) for x in range(5)],
      "NE":[(x,y) for y in range(5) for x in range(5,10)],
      "SW":[(x,y) for y in range(5,10) for x in range(5)],
      "SE":[(x,y) for y in range(5,10) for x in range(5,10)]}
LAND_ORDER=["NE","SW","SE"]; LAND_COST={"NE":1000,"SW":2000,"SE":4000}
def _crop_tiles(farm):
    unlocked=set(stock._get(farm,"unlocked_quadrants",["NW"]) or ["NW"])
    tiles=list(CROP_TILES)
    for q in LAND_ORDER:
        if q in unlocked: tiles+=QUAD[q]
    return tiles
SHED_ADJ = {(4,4)}

_ST = {0:{"day":-1,"bought_today":False}, 1:{"day":-1,"bought_today":False}}

def _tile(farm,x,y):
    try: return (stock._get(farm,"tiles",[]) or [])[y][x]
    except Exception: return "LOCKED"

def _targets(obs):
    alloc = planner.plan(obs, len(ANIMAL_TILES), pool=("MILK","WOOL"))
    cows  = alloc.get("MILK",0); sheep = alloc.get("WOOL",0)
    n = len(ANIMAL_TILES)
    cows  = min(cows,  n)
    sheep = min(sheep, n - cows)
    return cows, sheep

def _plan_crop_counts(obs):
    ctiles=_crop_tiles(stock._farm(obs,stock._seat(obs))); alloc = planner.plan(obs, len(ctiles), pool=CROPS)
    counts = {c:0 for c in CROPS}
    for k,v in alloc.items():
        if k in counts: counts[k]+=v
    if sum(counts.values())==0:
        counts["WHEAT"]=6; counts["STRAWBERRY"]=6; counts["MELON"]=4
    tot=sum(counts.values()); cap=len(ctiles)
    if tot>cap:
        for c in CROPS: counts[c]=int(counts[c]*cap/tot)
    # always keep some wheat growing to help feed animals cheaply
    counts["WHEAT"]=max(counts["WHEAT"], 4)
    return counts

def _current_crops(farm):
    counts={c:0 for c in CROPS}
    for (x,y) in _crop_tiles(farm):
        t=_tile(farm,x,y)
        if isinstance(t,dict) and t.get("kind")=="PLANT" and t.get("crop") in counts:
            counts[t["crop"]]+=1
    return counts

def _next_crop(obs,farm):
    want=_plan_crop_counts(obs); have=_current_crops(farm)
    best,gap=None,0
    for c in CROPS:
        g=want.get(c,0)-have.get(c,0)
        if g>gap: best,gap=c,g
    return best

def _step_toward(px,py,tx,ty):
    if px<tx: return "EAST"
    if px>tx: return "WEST"
    if py<ty: return "SOUTH"
    if py>ty: return "NORTH"
    return None

def _crop_op(obs,farm,day,px,py):
    best=None; bd=1e9; wc=_next_crop(obs,farm)
    for (x,y) in _crop_tiles(farm):
        t=_tile(farm,x,y); pr=act=cr=None
        if isinstance(t,dict):
            if t.get("kind")=="PLANT":
                age=day-int(t.get("planted_day",day))
                fy,hv,ong=CROP.get(t.get("crop"),(2,4,False))
                if int(t.get("yield_units",0) or 0)>0 and age>=fy and (ong or age>=hv): pr,act=0,"HARVEST"
                elif not t.get("watered_today",False): pr,act=1,"WATER"
                else: continue
            elif t.get("kind")=="WEED": pr,act=2,"DIG"
            else: continue
        elif t is None and wc: pr,act,cr=3,"PLANT",wc
        else: continue
        d=abs(px-x)+abs(py-y)+pr*100
        if d<bd: bd=d; best=(x,y,act,cr)
    if not best: return ["PASS"]
    x,y,act,cr=best
    if (px,py)==(x,y): return [act,cr] if cr else [act]
    return [_step_toward(px,py,x,y) or "PASS"]


def _assign_zones(tiles, k, farm=None, day=0):
    """Resident sub-farms, but workers are allocated to quadrants by actual daily
    WORKLOAD (urgent waters, harvests, empties, weeds) rather than raw tile count,
    so a quiet far quadrant of healthy ongoing crops doesn't tie up a full-time
    resident commuting daily for skippable (non-urgent) watering."""
    if k<=0: return []
    qorder={(0,0):0,(1,0):1,(0,1):2,(1,1):3}
    groups={}
    for (x,y) in tiles:
        groups.setdefault(qorder.get((x//5,y//5),9),[]).append((x,y))
    quads=sorted(groups)
    # per-quadrant workload weight
    def demand(x,y):
        if farm is None: return 1.0
        t=farm and (stock._get(farm,"tiles",[]) or [])[y][x] if 0<=y<10 and 0<=x<10 else None
        try: t=(stock._get(farm,"tiles",[]) or [])[y][x]
        except Exception: t=None
        if t is None: return 1.0                       # needs planting
        if isinstance(t,dict):
            k_=t.get("kind")
            if k_=="WEED": return 1.0
            if k_=="PLANT":
                consec=int(t.get("consecutive_unwatered",0) or 0)
                if int(t.get("yield_units",0) or 0)>0: return 1.0   # ripe
                if not t.get("watered_today",False) and consec>=1: return 1.0  # urgent
                return 0.25                            # maintenance only (skippable today)
        return 0.1
    wq={q: sum(demand(x,y) for (x,y) in groups[q]) for q in quads}
    totw=max(1e-9, sum(wq.values()))
    alloc={}; assigned=0; rema=[]
    for q in quads:
        share=k*wq[q]/totw; a=int(share); alloc[q]=a; assigned+=a; rema.append((share-a,q))
    rema.sort(reverse=True)
    for _,q in rema[:k-assigned]: alloc[q]+=1
    for q in quads:                                    # guarantee >=1 per quadrant with work
        if alloc[q]==0 and wq[q]>0.5 and k>=len(quads):
            big=max(alloc,key=lambda z:alloc[z])
            if alloc[big]>1: alloc[big]-=1; alloc[q]=1
    zones=[]
    for q in quads:
        gt=groups[q]; kk=alloc.get(q,0)
        if kk<=0: continue
        rows={}
        for (x,y) in gt: rows.setdefault(y,[]).append(x)
        order=[]
        for i,y in enumerate(sorted(rows)):
            xs=sorted(rows[y]); order+=[(x,y) for x in (xs if i%2==0 else xs[::-1])]
        size=(len(order)+kk-1)//kk
        for j in range(kk): zones.append(order[j*size:(j+1)*size])
    while len(zones)<k: zones.append([])
    return zones[:k]

def _serpentine(tiles):
    """Order tiles grouped by quadrant (each serpentined internally), so contiguous
    zone-slices stay LOCAL to one quadrant instead of jumping across the board."""
    # quadrant order keeps travel near the shed corner first
    qorder={(0,0):0,(1,0):1,(0,1):2,(1,1):3}
    groups={}
    for (x,y) in tiles:
        groups.setdefault(qorder.get((x//5,y//5),9),[]).append((x,y))
    out=[]
    for q in sorted(groups):
        rows={}
        for (x,y) in groups[q]: rows.setdefault(y,[]).append(x)
        for i,y in enumerate(sorted(rows)):
            xs=sorted(rows[y])
            if i%2==1: xs=xs[::-1]
            out += [(x,y) for x in xs]
    return out

def _zones(tiles, k):
    """Split serpentine-ordered tiles into k near-equal contiguous zones."""
    if k<=0: return []
    order=_serpentine(tiles); n=len(order)
    if n==0: return [[] for _ in range(k)]
    size=(n+k-1)//k
    return [order[i*size:(i+1)*size] for i in range(k)]

ROUTER = "greedy"   # "greedy" | "anchored" | "route"

def _route(zone):
    """Nearest-neighbor sweep order through a zone, starting from the shed corner."""
    remaining=list(zone); order=[]; cur=(4,4)
    while remaining:
        nx=min(remaining, key=lambda t:abs(t[0]-cur[0])+abs(t[1]-cur[1]))
        order.append(nx); remaining.remove(nx); cur=nx
    return order

def _route_op(obs,farm,day,pos,route,wc):
    """Sweep the zone in fixed route order: act on the EARLIEST tile in the route
    that still needs work (forward progress, minimal backtracking)."""
    px,py=pos
    for (x,y) in route:
        t=_tile(farm,x,y); act=cr=None
        if isinstance(t,dict):
            if t.get("kind")=="PLANT":
                age=day-int(t.get("planted_day",day))
                fy,hv,ong=CROP.get(t.get("crop"),(2,4,False))
                consec=int(t.get("consecutive_unwatered",0) or 0)
                in_bonus=(not ong) and age>=max(1,(hv+1)//2) and age<=hv
                ripe=int(t.get("yield_units",0) or 0)>0 and age>=fy and (ong or age>=hv)
                if not t.get("watered_today",False) and (consec>=1 or in_bonus): act="WATER"
                elif ripe: act="HARVEST"
                elif not t.get("watered_today",False): act="WATER"
                else: continue
            elif t.get("kind")=="WEED": act="DIG"
            else: continue
        elif t is None and wc: act,cr="PLANT",wc
        else: continue
        if (px,py)==(x,y): return [act,cr] if cr else [act]
        return [_step_toward(px,py,x,y) or "PASS"]
    return ["PASS"]

def _zone_op(obs,farm,day,pos,zone,wc):
    """Best task within a worker's zone: WATER(survival) > HARVEST > PLANT > DIG."""
    px,py=pos; best=None; bd=1e9
    for (x,y) in zone:
        t=_tile(farm,x,y); pr=act=cr=None
        if isinstance(t,dict):
            if t.get("kind")=="PLANT":
                age=day-int(t.get("planted_day",day))
                fy,hv,ong=CROP.get(t.get("crop"),(2,4,False))
                consec=int(t.get("consecutive_unwatered",0) or 0)
                in_bonus=(not ong) and age>=max(1,(hv+1)//2) and age<=hv  # water daily for yield bonus
                ripe=int(t.get("yield_units",0) or 0)>0 and age>=fy and (ong or age>=hv)
                if not t.get("watered_today",False) and (consec>=1 or in_bonus): pr,act=0,"WATER"  # urgent/yield
                elif ripe: pr,act=1,"HARVEST"
                elif not t.get("watered_today",False): pr,act=4,"WATER"     # optional (every-other-day)
                else: continue
            elif t.get("kind")=="WEED": pr,act=3,"DIG"
            else: continue
        elif t is None and wc: pr,act,cr=2,"PLANT",wc
        else: continue
        d=abs(px-x)+abs(py-y)+pr*50
        if d<bd: bd=d; best=(x,y,act,cr)
    if not best: return ["PASS"]
    x,y,act,cr=best
    if (px,py)==(x,y): return [act,cr] if cr else [act]
    return [_step_toward(px,py,x,y) or "PASS"]

def agent(obs):
    try:
        seat=stock._seat(obs); farm=stock._farm(obs,seat)
        day=int(stock._get(obs,"day",0) or 0); step=int(stock._get(obs,"step",0) or 0)
        hour=int(stock._get(obs,"hour",0) or 0)
        money=stock._get(farm,"money",0) or 0
        priv=stock._get(obs,"private",{}) or {}
        shed=stock._get(priv,"shed",{}) or {}; seeds=stock._get(priv,"seeds",{}) or {}
        invs=stock._get(priv,"inventories",[]) or []
        hands=stock._get(farm,"hands",[]) or []
        positions=[list(stock._get(farm,"farmer",[4,4]) or [4,4])]+[list(h) for h in hands]
        cows_t, sheep_t = _targets(obs)
        st=_ST[seat]
        if step==0 or day!=st.get("day",-1):
            st={"day":day,"bought_today":False}; _ST[seat]=st

        # census
        animals=[]; empty_pastures=[]
        n_cows=n_sheep=0
        for (x,y) in ANIMAL_TILES:
            t=_tile(farm,x,y)
            if isinstance(t,dict) and t.get("kind")=="PASTURE":
                if t.get("animal")=="COW": animals.append((x,y,t)); n_cows+=1
                elif t.get("animal")=="SHEEP": animals.append((x,y,t)); n_sheep+=1
                else: empty_pastures.append((x,y))
        n_animals=len(animals)
        in_shed_animals=int(shed.get("COW",0) or 0)+int(shed.get("SHEEP",0) or 0)
        want_animals=cows_t+sheep_t

        # ---- market ----
        market=[]
        n_ct=len(_crop_tiles(farm))
        target_hands=min(11, 4 + n_ct//7)   # scale hands to staff up to 3 quadrants
        # cash guard: the k-th hire costs fib(k) (1,1,2,3,5,8,13,21..). Don't spend the
        # farm into bankruptcy on wages in a poor/contested game -- cap hires to what a
        # fraction of current cash covers, so production stays high only when affordable.
        if hour==0:
            budget=max(0.0, money*0.45)
            a=b=1; spent=0; hired=0
            for _ in range(target_hands):
                if spent+a>budget: break
                spent+=a; a,b=b,a+b; hired+=1
            for _ in range(hired): market.append(["HIRE"])
        # INCREMENTAL land: buy the next quadrant only once the CURRENT land is fully
        # planted and weed-free (labor already staged), never as a sudden slug.
        unlocked=set(stock._get(farm,"unlocked_quadrants",["NW"]) or ["NW"])
        planted=sum(1 for (x,y) in _crop_tiles(farm) if isinstance(_tile(farm,x,y),dict))
        weeds_now=sum(1 for (x,y) in _crop_tiles(farm) if isinstance(_tile(farm,x,y),dict) and _tile(farm,x,y).get("kind")=="WEED")
        util=planted/max(1,n_ct)
        BUYABLE=["NE","SW"]   # SE still excluded (too far from shed to tend)
        for q in BUYABLE:
            if q not in unlocked:
                staged = len(hands) >= target_hands-1
                if q=="NE":
                    # 2nd quadrant: cheap, near -> take it whenever the 1st is tended
                    if day<=20 and util>0.88 and weeds_now<=1 and staged and money>LAND_COST[q]+1800:
                        market.append(["BUY_LAND"])
                else:
                    # 3rd quadrant: occupy it once NW+NE are well-tended and cash allows,
                    # over a wider window so we reliably reach 3 quarters (not only rich games).
                    if day<=22 and util>0.85 and weeds_now<=1 and staged and money>LAND_COST[q]+2000:
                        market.append(["BUY_LAND"])
                break
        # wheat to feed animals (need 1/animal/day); keep a 2-day buffer in shed
        wheat_have=int(shed.get("WHEAT",0) or 0)
        wheat_need=max(0, n_animals*2 - wheat_have)
        if n_animals>0 and wheat_need>0 and money>CASH_RESERVE:
            market.append(["BUY_PRODUCT","WHEAT",min(wheat_need,12)])
        # buy at most ONE animal per day, only from surplus, paced with cashflow
        can_afford = money > (CASH_RESERVE + ANIMAL_COST["SHEEP"])
        placed_or_pending = n_animals + in_shed_animals + len(empty_pastures)
        if (not st["bought_today"] and can_afford and placed_or_pending < want_animals
                and n_animals < len(ANIMAL_TILES)):
            buy = "COW" if n_cows < cows_t else ("SHEEP" if n_sheep < sheep_t else None)
            if buy and money > ANIMAL_COST[buy] + CASH_RESERVE:
                market.append(["BUY_ANIMAL",buy,1]); st["bought_today"]=True
        # seeds
        cw=_plan_crop_counts(obs)
        for c in CROPS:
            if cw.get(c,0)>0 and int(seeds.get(c,0) or 0)<2 and money>SEED_COST[c]*2+50:
                market.append(["BUY_SEED",c,2]); break
        # sell (metered-lite, dump at terminal); keep a wheat feed reserve
        inv_mkt=stock._get(stock._get(obs,"market",{}) or {},"inventory",{}) or {}
        terminal=step>=700
        for item in ["MILK","WOOL","MELON","STRAWBERRY","TOMATO","CARROT","WHEAT","FERTILIZER"]:
            n=int(shed.get(item,0) or 0)
            if item=="WHEAT": n=max(0, n - n_animals*2)
            if n<=0: continue
            cur=int(inv_mkt.get(item,10000) or 10000)
            floor=1 if terminal else max(1,int(BASE.get(item,25)*0.34))
            q=n
            if _mp(item,cur+q)<floor:
                lo,hi=0,n
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if _mp(item,cur+mid)>=floor: lo=mid
                    else: hi=mid-1
                q=lo
            if q>0: market.append(["SELL",item,q])

        # ---- per-unit ops: up to 2 keepers tend animals, the rest farm crops ----
        animal_work = bool(animals or empty_pastures or in_shed_animals>0)
        n_keepers = 2 if (animal_work and (n_animals>=5 or want_animals>=5)) else (1 if animal_work else 0)
        # crop workers = units not assigned as keepers; give each a compact zone
        crop_worker_idx=[i for i in range(len(positions)) if i>=n_keepers]
        # per-day zone cache: compute once when the crew is stable so workers commit
        # to a daily anchor region (kills cross-quadrant thrash) instead of re-solving each turn
        cache_key=(day, len(crop_worker_idx))
        if ROUTER=="greedy" or st.get("zkey")!=cache_key:
            zs=_assign_zones(_crop_tiles(farm), len(crop_worker_idx), farm, day)
            zmap={}
            for slot,i in enumerate(crop_worker_idx):
                zmap[i]= zs[slot] if slot<len(zs) else []
            routes={i:_route(zmap[i]) for i in crop_worker_idx} if ROUTER=="route" else {}
            if ROUTER!="greedy":
                st["zkey"]=cache_key; st["zmap"]=zmap; st["routes"]=routes
        if ROUTER!="greedy":
            zmap=st.get("zmap",{}); routes=st.get("routes",{})
        def _crop_route(ui,pos):
            if ROUTER=="route": return _route_op(obs,farm,day,pos,routes.get(ui,[]),wc)
            return _zone_op(obs,farm,day,pos,zmap.get(ui,[]),wc)
        wc=_next_crop(obs,farm)
        ops=[]
        for ui,pos in enumerate(positions):
            px,py=pos; inv=invs[ui] if ui<len(invs) else {}
            if ui < n_keepers and animal_work:
                op=None
                my_animals = animals[ui::n_keepers]   # this keeper's share
                if ui==0:
                    # setup (keeper 0 only): place -> pickup -> build pasture
                    if empty_pastures and (int(inv.get("COW",0) or 0)+int(inv.get("SHEEP",0) or 0))>0:
                        tx,ty=empty_pastures[0]
                        op=["PLACE","COW" if inv.get("COW",0) else "SHEEP"] if (px,py)==(tx,ty) else [_step_toward(px,py,tx,ty) or "PASS"]
                    elif in_shed_animals>0 and empty_pastures:
                        op=["PICKUP","COW" if shed.get("COW",0) else "SHEEP",1] if (px,py) in SHED_ADJ else [_step_toward(px,py,4,4) or "PASS"]
                    elif (in_shed_animals>0 or (want_animals>n_animals and not st.get("bought_today"))) and len(empty_pastures)==0 and n_animals<len(ANIMAL_TILES):
                        for (x,y) in ANIMAL_TILES:
                            if _tile(farm,x,y) is None:
                                op=["BUILD_PASTURE"] if (px,py)==(x,y) else [_step_toward(px,py,x,y) or "PASS"]
                                break
                # daily husbandry over this keeper's animals: wheat -> feed -> harvest -> care
                if op is None:
                    need_feed=[(x,y,t) for (x,y,t) in my_animals if not t.get("fed_today",False)]
                    have_wheat=int(inv.get("WHEAT",0) or 0)
                    if need_feed and have_wheat<=0:
                        if (px,py) in SHED_ADJ and int(shed.get("WHEAT",0) or 0)>0:
                            op=["PICKUP","WHEAT",min(int(shed.get("WHEAT",0)), len(my_animals)+1)]
                        else:
                            op=[_step_toward(px,py,4,4) or "PASS"]
                    else:
                        ready=[(x,y,t) for (x,y,t) in my_animals if int(t.get("yield_units",0) or 0)>0]
                        need_care=[(x,y,t) for (x,y,t) in my_animals if not t.get("cared_today",False)]
                        cand=need_feed or ready or need_care
                        if cand:
                            tx,ty,tt=min(cand,key=lambda r:abs(px-r[0])+abs(py-r[1]))
                            if (px,py)==(tx,ty):
                                if not tt.get("fed_today",False) and have_wheat>0: op=["FEED"]
                                elif int(tt.get("yield_units",0) or 0)>0: op=["HARVEST"]
                                elif not tt.get("cared_today",False): op=["CARE"]
                                else: op=["PASS"]
                            else:
                                op=[_step_toward(px,py,tx,ty) or "PASS"]
                        else:
                            op=_crop_route(ui,pos)
                ops.append(op or ["PASS"])
            else:
                ops.append(_crop_route(ui,pos))
        return {"farmer":ops[0],"hands":ops[1:],"market":market[:10]}
    except Exception:
        farm=stock._farm(obs,stock._seat(obs))
        return {"farmer":["PASS"],"hands":[["PASS"] for _ in (stock._get(farm,"hands",[]) or [])],"market":[]}
