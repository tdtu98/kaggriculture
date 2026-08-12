"""T0.7 — settle the PLAN.md hypotheses by measurement, not arithmetic."""
import statistics as st
from sim.runner import play_episode
from sim.baselines import AGENTS
from agent import make_agent, Params

SEEDS = range(6)
MIX = lambda **kw: {**{c: 0.0 for c in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"]}, **kw}

G = dict(goose_min_cash=200)
CONFIGS = {
  "wheat":              Params(hire_max=8, crop_mix=MIX(WHEAT=1)),
  "carrot":             Params(hire_max=8, crop_mix=MIX(CARROT=1)),
  "melon":              Params(hire_max=8, crop_mix=MIX(MELON=1)),
  "melon+wheat":        Params(hire_max=8, crop_mix=MIX(MELON=1, WHEAT=1)),
  "melon+wheat+geese":  Params(hire_max=8, crop_mix=MIX(MELON=1, WHEAT=1), goose_target=8, **G),
  "wheat+geese/care":   Params(hire_max=8, crop_mix=MIX(WHEAT=1), goose_target=8, care=True, **G),
  "wheat+geese/alt":    Params(hire_max=8, crop_mix=MIX(WHEAT=1), goose_target=8,
                               care=False, feed_alternate=True, **G),
  "wheat, no land":     Params(hire_max=8, crop_mix=MIX(WHEAT=1), buy_land=False),
  "harvest early":      Params(hire_max=8, crop_mix=MIX(WHEAT=1), harvest_early=True),
}

print(f"{'config':<20}{'mean $':>10}{'sd':>8}{'wins':>6}{'move%':>7}{'weeds':>7}  top sales")
for name, p in CONFIGS.items():
    monies, wins, mv, wd, sales = [], 0, [], [], {}
    for s in SEEDS:
        r = play_episode([make_agent(p), AGENTS["starter"]], seed=s, collect_stats=True)
        monies.append(r.money[0]); wins += r.winner == 0
        t = r.stats[0]
        mv.append(100*t["actions_move"]/max(t["actions_total"],1))
        wd.append(r.daily[-1][0]["tiles_weed"])
        for k, v in t["sold_revenue"].items(): sales[k] = sales.get(k,0)+v
    top = ", ".join(f"{k}=${v//len(SEEDS):,}" for k,v in sorted(sales.items(), key=lambda kv:-kv[1])[:3])
    print(f"{name:<20}{st.mean(monies):>10,.0f}{st.pstdev(monies):>8,.0f}{wins:>4}/{len(SEEDS)}"
          f"{st.mean(mv):>7.0f}{st.mean(wd):>7.0f}  {top}")
