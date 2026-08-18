# Kaggriculture — Task Breakdown v3

Executable breakdown of `PLAN3.md`. Every task states **what to build**, **how to verify it**, and a
binary **done-when**. "It runs" is not done; done means the stated verification passed.

`TASKS2.md` (v2) stays valid for the engine line, which `PLAN3.md` §8 keeps as the named fallback.

## Status

| Task | State |
|---|---|
| R0 substrate, bit-identity gate, desync counter | **done (E49)** — 0 differing steps over 28,760; `blocked_ops` live |
| ~~R1 adaptive livestock~~ | **closed (E50, E51)** — a bad trade: −42 wool @ $200 for +46 milk @ $160. E50's "milk sales are capped" diagnosis was itself wrong (order sizes read as sales) and is corrected in place |
| **R2e sell-side overlay** | **WORKS (E51), tuned (E52)** — `relay-sell` beats `boatlee` 88–91% over 5 fresh blocks (~840 games). Thresholds swept: pressure 70 (safe range 65–78, cliff below), batch 8. Not promoted (D19 gate not run) |
| R2b opponent-aware sell timing | ready — survives the E50 review intact |
| R2c terminal liquidation | ready — plausible, price-impact only |
| ~~R2a shed valve~~ / ~~R2d blocked buys~~ | **premise gone (E50)** — justified on the champion's shed behaviour, not boatlee's |
| ~~R3 generalised trace repair~~ | **killed (E49)** — 2.35 blocked ops/season vs a <10 gate |
| R4 trace re-optimisation | last; deprioritised by E48 |
| V1 submit | user decision; see `PLAN3.md` preamble |

Standing gates, unchanged: `make verify` after any `kagsim/` change; `make promote` before any
champion install (D19); never rank on money vs `starter`; ≥80 games for any money claim (E42).

## Standing rule for every task here: prove it fired before you read its score

**No money number is interpreted until both counters below have been checked.** An overlay whose
effect counter is zero is not a refutation — it is an unfinished implementation, and this project
has already mistaken one for the other (`PLAN3.md` §6; E36, E39, E44, E46).

| check | question | built in |
|---|---|---|
| **effect counter** | did the thing actually happen? | per task, asserted in a test |
| **desync counter** | did we break the choreography? | `R0.5`, must not rise above `relay-base` |

Two habits that come with it, both earned:

* **Assert behaviour in play, not helper return values.** `tests/test_roles.py` caught three defects
  by asserting purity rose *during a game*; a unit test on the cost function would have passed on
  all three (E46).
* **Where it is cheap, mutate the overlay and confirm the test fails.** Mutation testing found four
  parity tests that passed while proving nothing (E36). A test that cannot fail is worse than none.

Every "Done when" below is additionally conditional on these. If a task's effect counter is zero,
the task is **not done and its result is not reportable** — fix the implementation and re-run.

---

## R0 — Substrate

### R0.1 — `agent/relay.py`

**Build.**

```python
load_table(path) -> list[dict]          # decompress the 719-step action table
make_relay(overlays=()) -> Callable     # agent(obs) -> action, overlays applied in order
```

An **overlay** is `fn(obs, action, ctx) -> action`. `ctx` carries per-seat, per-episode state and
resets when `obs["step"] == 0` or goes backwards — the same reset discipline boatlee uses, because
the module is re-used across episodes in the arena.

Order matters and is fixed: farm-affecting overlays run before market overlays, so a market overlay
sees the final farm decisions.

**Constraints.** stdlib only. Must survive `tools/build_submission.py`'s AST check. No import of
`kagsim`.

**Do not modify `reference/kaggriculture/1/submission.py`** — it is the arena's external opponent
and the only non-self-referential measurement we have (D16).

**Done when:** R0.2 passes.

### R0.2 — Bit-identity gate

**Build.** `tests/test_relay_parity.py`. Play `relay-base` (empty overlay stack) and `boatlee` in
the same seat, same seed, and assert the emitted action dicts are **equal at every one of the 719
steps**.

**Verify.** 20 seeds × 2 seats.

**Done when:** 0 differing steps across all 40 episodes.

**Kill criterion:** if bit-identity cannot be reached, stop. Every number downstream would be
uninterpretable. Do not proceed with "close enough".

### R0.3 — Registry entries

**Build.** Register in `arena/registry.py`: `relay-base`, and one entry per overlay variant as they
are built. Registered by name, never a throwaway script (E43).

**Done when:** `make gauntlet` runs `relay-base` and it scores within noise of `boatlee`.

### R0.4 — Submission path

**Verify.** `make submission` — AST import check plus the smoke test through Kaggle's real loader in
both seats.

**Done when:** bundle builds, both seats DONE, money above the starting bank. (E21: the submission
that scored the $3,000 starting bank is the cost of skipping this.)

### R0.5 — The desync counter *(built here because R1 onward depend on it)*

**Build.** Instrument the relay to count, per episode, **`blocked_ops`** — scripted actions that were
invalid at their target tile when they were issued. Break down by op and by cause:

| cause | signature |
|---|---|
| weed where a plant was expected | tile is `{"kind": "WEED"}`, op is `WATER`/`HARVEST`/`FERTILIZE` |
| tile empty where a plant was expected | tile is `None`, op is `WATER`/`HARVEST`/`FERTILIZE` |
| tile occupied where a plant was intended | tile is not `None`, op is `PLANT`/`BUILD_PASTURE` |
| animal missing where one was expected | no `animal` key, op is `FEED`/`CARE`/`COLLECT_FERTILIZER` |
| item missing for a `PICKUP`/`PLACE` | shed or inventory short |

**This one instrument does two jobs**, which is why it is built first rather than in R3:

1. It is the **safety check** for every farm-affecting overlay. `PLAN3.md` §2's rule — market-only,
   structure-preserving, or resync — is only enforceable if desync is *observable*. Any overlay
   claimed safe must not raise `blocked_ops` above `relay-base`.
2. It is the **gate measurement** for R3, which is otherwise a phase built on a guess.

**Verify.** Deliberately desynchronise: force one early `PLANT` to a different crop with a different
growth time, and assert `blocked_ops` rises sharply. A counter that stays flat under a known break
is not measuring anything (E36).

**Done when:** `blocked_ops` is reported by the bench for every run, `relay-base`'s baseline value is
recorded in `docs/experiments.md`, and the deliberate-desync test passes.

---

## R1 — Adaptive livestock

**Hypothesis.** Choosing COW vs SHEEP from the observed shop draw beats the fixed 9 cows + 4 sheep.

**Why it is safe.** Both animals live on a `PASTURE`. The tile type, the `BUILD_PASTURE` op, and
every downstream FEED / CARE / COLLECT_FERTILIZER are unchanged, so the trace does not desynchronise.
Boatlee already ships this swap (`_v16_convert_livestock`) — we are generalising a mechanism it has.

**Why it should pay.** WOOL has no shop buyer in **36%** of games, MILK in 2% (E33). Boatlee commits
its herd before any shop unlocks, and its one adaptation never fired in 6 traced seeds (E48).

### R1.1 — Decision points

**Build.** Identify every step where the species is still choosable:

* `BUY_ANIMAL` market orders (the trace buys COW on days 0, 5, 6, 7, 8, 15 and SHEEP on day 0);
* the `PICKUP <animal>` and `PLACE <animal>` unit ops that follow each purchase.

Both must be rewritten consistently — a `PICKUP COW` for an animal we bought as a sheep is a no-op
and loses the tile.

**Verify.** Unit test: for a forced swap of every COW to SHEEP, assert the placed herd on day 20
equals 13 sheep and 0 cows, with no empty pastures.

**Done when:** the forced-swap test passes at both extremes (all-cow, all-sheep).

### R1.2 — Demand-driven choice

**Build.** At each purchase step, count shop instances demanding MILK vs WOOL from
`obs["town"]["unlocked_shops"]` (shops draw **with replacement** since 1.32.6, so the same shop can
appear several times and each instance consumes independently). Choose the species with the higher
observed demand; keep the trace's default on a tie.

Reuse `Engine.town_drain_per_day` rather than reimplementing the count — it is already unit-tested
(`tests/test_demand_model.py`).

**Uncertainty is real.** The first purchase is on day 0, before any shop unlocks. Prefer deferring
the species choice to the latest step at which the trace still allows it, and only commit early for
animals whose product has already shown demand.

**Verify.** `--split-by-draw` on the bench: report the subset where observed demand contradicts a
9c/4s herd separately from the rest.

**Effect counter.** Report the realised herd composition per game, and the count of games where the
choice differed from 9c/4s. **If that count is zero the overlay did not fire and the result is not
reportable** — this is exactly the failure `PLAN3.md` §6 exists to prevent, and boatlee's own
`_v16_yarn_route` is a live example of a conditional that never triggers.

**Desync counter.** `blocked_ops` must not exceed `relay-base`. A `PICKUP COW` left behind after the
purchase was swapped to SHEEP shows up here as an empty pasture.

**Done when:** a sweep is recorded in `docs/experiments.md` with winrate and money on both subsets,
≥80 games, both seats, **and both counters check out**.

**Kill criterion — fixed now:** if it does not beat `relay-base` **on the disagreement subset**, the
effect is not there and R1 is dropped. That subset is where it must show up most clearly; no signal
there means no signal anywhere. (This is P1.5-A's kill criterion, which correctly killed E35.)

---

## R2 — Market overlays

The market queue cannot desynchronise the trace, so each of these is independently flaggable and
independently testable. **Build all four behind separate flags, measure each alone, then measure the
combination** — E43's fertiliser/eager-water pair was worth ~nothing separately and +24% together,
and one-factor-at-a-time cannot find that.

### R2a — Shed-pressure valve

**Build.** When total shed contents exceed a threshold, force-sell the lowest-value-per-slot product
until back under it. Boatlee has this for WOOL only (`_V16_WOOL_PRESSURE = 78`, dumping down to 66);
generalise it to every product.

**Why.** The shed caps at 100 items and end-of-day overflow is **discarded**; `BUY_PRODUCT` silently
fails when full (`kaggriculture.py:653`). Our champion pegs at 100 for 9 of 29 days holding 52 unsold
wool, and that correlates with its worst seeds (E48).

**Effect counter.** Valve activations per season, and mean days at shed cap. Zero activations means
the threshold is wrong, not that the idea failed.

**Done when:** ≥80 games recorded, activations non-zero, and days-at-cap reported alongside money.

### R2b — Opponent-aware sell timing

**Build.** Blend the sell decision toward the price expected once the opponent's *visible* incoming
supply lands. `Engine.opponent_supply()` and `Engine.forecast_price()` already exist, are tuned
(E7: joint optimum `w=1.0, horizon=10`), and depend only on the public `farms` field.

**Effect counter.** Number of sell orders whose turn or quantity differs from `relay-base`. At
`forecast_weight = 0` this must be **zero** — that is the built-in null test for the overlay.

**Done when:** ≥80 games recorded, swept over `forecast_weight ∈ {0, 0.5, 1.0}`, with the null test
at 0 passing.

### R2c — Terminal liquidation

**Build.** Spread the final-day dump across the remaining turns instead of one order. The trace sells
`WHEAT 165` in a single order on day 29.

**Effect counter.** Largest single sell order on the final day, before and after. If it is still
165 wheat, the overlay did not fire.

**Done when:** ≥80 games recorded, with realised final-day revenue reported.

### R2d — Suppress blocked buys

**Build.** Drop `BUY_PRODUCT` orders when the shed is at capacity — they cannot settle and they
consume slots against the 10-per-turn cap.

**Effect counter.** Orders suppressed per season. Cross-check against the settled-order count from
kagsim — a suppressed order that *would* have settled is a bug, not a saving.

**Done when:** ≥80 games recorded, with wasted order slots per season reported before and after.

### R2e — Combination

**Build.** All surviving R2 overlays enabled together.

**Effect counter.** Every constituent overlay's own counter must be non-zero *in the combined run*.
Overlays can silence each other — R2a dumping stock removes the pressure R2c was going to relieve —
and a combination whose parts stopped firing is not a conjunction test, it is a different agent.

**Done when:** ≥80 games vs `relay-base`, plus a full `make gauntlet`. Report whether the combination
beats the best single overlay — that is the conjunction test, and it is the point of the phase.

---

## R3 — Generalised trace repair

### R3.0 — Read the gate *(no build; the instrument is R0.5)*

**Verify.** Read `blocked_ops` for `relay-base` over ≥80 games, broken down by cause. The instrument
already exists — R0.5 built it as the safety check for R1 and R2, and it doubles as this gate.

**Kill criterion — fixed now:** **if fewer than 10 ops per season are blocked, skip R3 entirely.**
At `weedSpawnChance = 0.005` the expectation is ~6 weeds per season, so this phase may not exist.

### R3.1 — Extend the repair

Only if R3.0 clears. Extend `_weed_repair`-style handling from `PLANT`/`BUILD_PASTURE` to
`WATER`/`HARVEST`/`FERTILIZE`, keeping the replay-and-resync structure.

**Done when:** blocked ops per season fall by half, and ≥80 games show no money regression.

---

## R4 — Trace re-optimisation

Last, and **deprioritised on measurement**: swapping melon — the crop no shop buys in any game — for
wheat cost boatlee **$18,882** over 40 games (E48). The obvious slack is not there.

If attempted: hill-climb the table in kagsim. An open-loop plan needs no observation built, so
rollouts run at ~500/s/core, ~100× the throughput the CEM run used (E31).

**Kill criterion:** if a search with 100× the likely original budget cannot improve the trace by 5%
on held-out seeds, the route is closed and `PLAN3.md` §7 applies.

---

## Loose ends carried from v2

* **E47 is cited by `agent/engine.py` and was missing from `docs/experiments.md`.** Written up this
  session. `finish_tile` remains `False` by default.
* **`_fallback` hauls without bound** (`agent/engine.py:694`). Identified independently by E46 (at
  the `role_penalty` cliff) and E47 (the `n=4` signature). Unfixed. It sits under every engine
  configuration and must be fixed before the §8 fallback line is resumed.
* **H4 — re-baseline under 1.32.6** is still open, and E48's shed deadlock is a direct consequence:
  `reserve_frac["WOOL"] = 0.5` was set by E19/D18 when wool had a guaranteed buyer.
* **H1 — competition deadline** still unknown, raised five times.
