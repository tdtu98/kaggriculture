# Plan v4 audit against the environment source (kaggle-environments 1.32.7, `kaggriculture.py`)

I re-read the full interpreter (init, unit actions, market lockstep, town, decay, end-of-day, spawn, hire, land) and the config JSON, then checked every mechanic against PLAN_v4/TASKS_v4. Result: the plan's architecture stands, but the audit found **one large strategic lever the plan missed, one stale model we were about to inherit, and ~10 precise mechanics the compiler must encode**. Changes are listed in §3 and folded into PLAN_v4 §2.7 / TASKS_v4 (C1′, C6, I0 additions).

## 1. The big miss: scarcity "hinge" prices (new in 1.32.x)

`MARKET_PARAMS` in 1.32.7 gives CARROT, TOMATO and EGG a **`hinge` below-I0 shape**: price is calm until the market is drained past `T`, then a quadratic term with gain 8 takes over and the price runs away.

| product | base | T | price at −T | −1.5T | −2T | −2.5T |
|---|---:|---:|---:|---:|---:|---:|
| TOMATO | 60 | 200 | 84 | 144 | **300** | **552** |
| CARROT | 35 | 450 | 70 | 158 | **385** | 752 |
| EGG | 50 | 332 | 70 | 120 | 250 | 460 |

Measured in real games (Boatlee vs starter, 1.32.7): **tomato peaked at $247 / $399 / $445** in seeds with Pizza-Shop and Farmers-Market draws (min inventory −372…−462), and **carrot hit $202** with three Pet Cafés (−732). Nobody supplies these — Boatlee plants zero tomato and zero carrot; our executor plants ~none. A tomato tile with two fertilizes yields 8 units in ~12 days; at $150–400 that is $1,200–3,200 per tile-cycle, 3–6× a strawberry tile, and the market absorbs 100–200 units before the price falls back near $100. This is worth tens of thousands per game **in the seeds where the draw creates it**, and it is *only* reachable by an agent that reads `town.unlocked_shops` and `market.inventory` — exactly the adaptivity Boatlee cannot have.

Why we missed it: `main.py`'s vendored `_MARKET_PARAMS` (and therefore `planner.py`, `metered`, `denial`, and my absorption caps) still carry the **old** shapes (carrot `log`, tomato/egg `linear` below I0). Boatlee's own price model is stale for these three products. **Any code that prices with `stock._market_price` under-values tomato/carrot/egg scarcity by up to 5×.**

**Plan change (PLAN_v4 §2.7 "scarcity hunter"):** the daily re-plan computes, per product, projected inventory over the next ~10 days = current − Σ(shop instances × drain) − town-centre − expected opponent supply (from their tiles) + our planned supply; when the projection crosses −T for tomato/carrot/egg (or −0.5T for strawberry/milk under `sqrt`), the next cohort goes to that product (tomato in field rows; carrot short cycles; geese only if a coop cluster is affordable). Sell into the spike in metered batches (the hinge is steep both ways: 100 units sold at −444 → still ~$200). Genes: threshold multipliers, max tiles per scarce product, batch size. This also fixes the "shop luck" spread (38–101k) from the wrong side: instead of hoping the town wants what we grow, grow what the town has drained.

## 2. Mechanics checklist (source → plan coverage)

Legend: ✅ covered in v4 as written · ➕ added by this audit · ⚠ correction to a doc/plan assumption.

**Crops**
- ➕ Melon `max_yield_day = 12` in code (README table says 10). Window ages 6–12, cap 6 reached at age 10 unfertilized (age 8 fertilized); decay begins dawn of age 13; -1 unit / 2 steps → **harvest window ages 10–13**. Task generator must use the code table, not the README.
- ➕ Ongoing crops cap `yield_units` at `max_yield` (4). Fertilized strawberry adds 2/tick → **must be harvested at least every second tick (≤ 4 days)** or the third tick is wasted; tomato (interval 1) every second day. Add a harvest-cadence constraint to C1.
- ✅ Fertilize covers `day..day+2`; strawberry ticks at end of ages 9/11/13/15 → fertilize at 9 and 13; tomato ticks ages 7/8/9/10 → fertilize at 7 and 10 (8 units); wheat fertilized at age 2 → 6 units (marginal); melon fertilize at 6 → cap at age 8 (frees the tile 2 days early). Bonus needs the tile watered *that day* — order within the day doesn't matter.
- ➕ One-time crops start with `yield_units = 1` and get +1 (or +2 fertilized) per **watered day inside the window only**; watering outside the window is survival only. Wheat: window ages 2–4 → 4 units at age 4; decay from dawn of age 5, gone in 8 steps → **wheat harvest deadline = age 4, or ≤ hour 7 of age 5**.
- ✅ Death at `consecutive_unwatered ≥ 2`, planting day counts as 1 → a new plant must be watered the same day or the next; the compiler's "sustaining cost" check covers it. Deaths happen only at end-of-day, so a compiled day is deterministic mid-day.
- ✅ Decay after last production: `max_lifespan_step = (next_day+1)*24` after the 4th tick → harvest by that dawn.
- ➕ HARVEST on a one-time crop clears the tile (needs replant task the next day); on ongoing/animal it just zeroes `yield_units`.
- ✅ Atomic PLANT per crop per turn (buy seeds ahead; never let two units plant the same crop in one turn without seeds ≥ 2 — router must check).

**Animals**
- ✅ First production at `next_day − placed − first_yield ≡ 0 (mod interval)`; care bank +1 per fed+cared day, paid only if fed on the production day, else the bank is cleared; `max_held` 6 (goose 4) → harvest cows at least every 2nd production when cared.
- ➕ A newly placed animal survives its first day unfed (`consecutive_unfed` starts 0); an escaped animal leaves the empty structure — re-buy and PLACE, no rebuild.
- ✅ `fertilizer_available` set for every surviving animal at every dusk regardless of care → supply = animals/day, uncollected doesn't accumulate.
- ➕ FEED consumes 1 WHEAT from the *unit's* inventory; feed wheat is bought (`BUY_PRODUCT` quoted at inv−1) or grown; the shed must have room (`BUY_PRODUCT`/`BUY_ANIMAL` fail when `sum(shed) ≥ 100`).

**Units, shed, day cycle**
- ✅ Farmer respawns at (4,4) every dawn; hands are hired via market (after unit actions of that step) and spawn on the least-occupied of the four centre tiles in NWSE order → hand 1 usually at (5,4). Hands vanish at dusk; `hires_today` resets; cost `fib(n)`: 1,1,2,3,5,8,13,21,34,55,89,144,233,377 (14 hands = $986/day).
- ✅ All four centre tiles are shed-adjacent for PICKUP/DROP/PLACE-to-shed even while LOCKED (shed ops resolve before the LOCKED guard).
- ➕ Units may stand on the same tile; movement onto LOCKED is allowed; tile ops on LOCKED no-op.
- ✅ Dusk: all inventories drop to shed (overflow **discarded**), so DROP before dusk when the shed is near 100, and sell during the day; unit actions resolve before market orders in the same step, so a DROP at step s is sellable at step s.
- ➕ Weeds spawn only on `None` tiles (p = 0.005/tile/day). Empty pastures/coops are dicts → **no weeds on built structures**; digging a dead plant leaves `None` → replant or accept weed risk.

**Market**
- ✅ Sell price quoted at pre-sell inventory per unit; both players' i-th orders are processed in lockstep, one unit at a time at the same quote; sells at $1 don't add inventory; a failed commit aborts the rest of that order; malformed order aborts; ≤ 10 orders/turn; HIRE/BUY_LAND atomic and first.
- ➕ **Slot alignment vs a known opponent**: our slot-i order runs concurrently with their slot-i. If Boatlee sells STRAWBERRY in slot 2 at step s, our STRAWBERRY sell in slot 0 at step s takes the higher quotes first. With a fingerprinted script we know their slot order every step → put same-product sells in an earlier slot. Free micro-edge; add to O3.
- ⚠ Boatlee's `_market_price` copy is stale (§1). Vendor the 1.32.7 table including `hinge`; assert equality against `kaggle_environments.envs.kaggriculture.kaggriculture.market_price` in I0.
- ✅ Prices refresh after each order slot and after town consumption; town: each shop instance consumes its products every 4 steps (single-product shops ×2), town centre 1 of each every 24 steps (step % 24 == 0, so 30/season). Shop instances unlock at end of days 2,5,8,…,23 (when `next_day % 3 == 0`), 8 draws with replacement, uniform over the 8 shops → all draws known by day 24; branch points at those days.
- ✅ Only WHEAT and FERTILIZER buyable back. Land NE/SW/SE at $1k/$2k/$4k, fixed order.

**Episode / runtime**
- ✅ Reward = money; DONE fires at step ≥ 718 → last actions that count are at step 718; terminal sells before that.
- ➕ `actTimeout` 1 s per turn (framework overage default 60 s). Any exception or timeout = ERROR for the whole game → hard try/except and a per-turn time guard (compile at hour 1 must be < ~200 ms worst case; degrade to cached/greedy if over).
- ➕ `market.params` only appears in obs when overrides are configured; competition uses defaults — but read it if present.
- ✅ Opponent private (shed/inventories) hidden; their tiles, hands, money, quads public.

## 3. Changes applied to PLAN_v4 / TASKS_v4

1. **PLAN §2.7 Scarcity hunter** (new): projected-inventory model per product using the true 1.32.7 curves; cohort allocation to hinge products when the projection crosses −T; metered selling into spikes. Genes for thresholds/caps/batches. This is now the second headline edge after per-unit efficiency, ahead of front-running.
2. **I0** adds: assert vendored `market_price` == env's for all products at inventories {I0±T, I0±2T} (catches the stale table); assert melon `max_yield_day == 12`; assert ongoing `yield_units` cap 4.
3. **C1′ task values** use the true price curve *at the projected inventory when the harvest will be sold*, not base price; harvest deadlines from the exact decay rules (wheat age 4 / hour 7 of age 5; melon 10–13; strawberry every ≤ 2 ticks; cows every ≤ 2 productions).
4. **C6 (new) Product-projection module** shared by C1′, O1, O3: `project_inventory(product, days) → path`, inputs: current inventory, shop instances × drain, town centre, opponent forecast (O2), our own plan. Unit-tested against replayed games (predict day-t inventory from day t−5 within ±15%).
5. **O3** adds slot alignment against a fingerprinted opponent's known order list.
6. **Plan genome** adds: tomato/carrot cohort templates (rows, cycle length), goose/coop option, per-product `scarcity_threshold` and `max_scarce_tiles`.
7. **Pool** adds a "tomato-rusher" exploiter so the search doesn't assume the hinge products are always ours alone.
8. Executor fallback (F-track) gets a cheap version: when tomato/carrot projected below −T, redirect the next N empty tiles to that crop — measurable in a day.

## 4. What did *not* change
Architecture (plan → compile → execute, daily re-plan), the routing/labour model, fertilize/care timing, front-running, measurement rules, kill criteria. The audit strengthens the case for adaptivity: the largest single price events in the game (tomato $400+, carrot $200+) exist only in some draws and are invisible to a fixed script.