# Kaggriculture — Plan v3

**Supersedes `PLAN2.md` for sequencing.** PLAN2's diagnosis of our engine still stands and its
experiment record is intact; what changes is which agent we develop next.

**Decision (user, this session): build a new agent derived from boatlee's action trace**, rather
than continuing to improve `agent/`. This document explains what that means, why it is credible,
what could kill it, and how we will know.

> **Before V1.** Whether a derived submission is permissible under the competition rules is a
> question for the user to settle before submitting. Nothing in this plan depends on the answer —
> every phase is measured locally.

---

## 1. What boatlee is, in plain terms

Someone played one very good game, wrote down all 719 moves, and their agent replays that list
every time. It does not look at the board. It does not look at the opponent.

That is not a figure of speech. Measured over 6 games against our champion [MEASURED E48]:

| | |
|---|---|
| Unit actions identical to the stored script | **100%, 100%, 99.9%, 100%, 99.8%, 100%** |
| Market orders added, removed, or resized by its reactive layers | **zero** |
| Market turns that differed | 69 of ~4,300 — **all pure re-orderings** |

Four of its five closed-loop layers never fired at all. It beat us 6/6 without reacting once.

**It wins because the route was planned offline.** Its workers walk 1.01 steps per useful action;
ours walk about 2. It knows where every unit goes for all 719 steps, so it never oscillates.

## 2. The constraint that shapes the whole plan

The recording is choreography. Every move assumes the farm looks a certain way at that moment —
*"at step 300, water the tile you are standing on."*

Change something early and that tile may be empty by then. The step does nothing, **and every step
after it is now wrong too.**

This is why boatlee's own patches are so narrow, and reading them shows its author knew:

* `_v16_convert_livestock` swaps COW→SHEEP — safe **only because both animals live on a `PASTURE`**,
  so the tile type, the build op, and every downstream FEED/CARE/COLLECT are unchanged.
* `_weed_repair_action` injects a `DIG` and then **replays the next 8 steps shifted**, explicitly
  resynchronising.
* Everything else it added (`_rank_sell_slots`, `_v16_wool_controller`, `_rc2_market_relay`) touches
  **only the market queue**, which is independent of unit actions.

**Rule for all our work: an overlay must be market-only, structure-preserving, or it must resync.**

### The second constraint — the sell list is fixed, but it has slack

**This section originally claimed sales were hard-capped by the script and that no production change
could pay without a matching sales change. That was wrong and is corrected here** [E50, corrected].
The error: `241` was the quantity the table *orders*, not what *settles*. The environment sells
`min(requested, shed)`.

What is actually true:

* The sell orders are a fixed list at fixed steps, and only WOOL has a controller.
* But the list **over-orders**: 241 milk ordered against 174 produced, ~67 units of slack. A 26%
  production increase sold in full with no new overlay at all.
* `relay-base` strands 0.1 items per season and hits the shed cap 0.9 days of 30. It sells what it
  makes, with room to spare.

**So the working rule is weaker than "test in pairs":** a production change is not automatically
blocked by the sell list, because the list carries headroom. Check the headroom for the specific
product before assuming either way — and **measure units that arrive in the shed, never order
sizes.** Reading orders as sales has now cost three separate conclusions in this project (E48's
wheat, E50's milk, and nearly R2e).

### The compensating advantage

The base is **deterministic**. Every previous experiment in this project fought the fact that
changing one knob changed the entire trajectory, so nothing could be isolated. Here an overlay
changes only what it touches, and a clean A/B is possible for the first time.

## 3. Why this is not a new idea

P1.5-A — adapt the product mix to the observed shop draw — was killed in E35. Read *why*:

> "Selection does not matter while servicing is the constraint. We service 10–15 crop tiles; *which*
> crop sits on them is second-order when the first-order problem is that we cannot occupy more."

`TASKS2.md` records the conclusion in full:

> **Retest after P1, not before**: the hypothesis is untested for a 40-tile farm, only disproven for
> a 15-tile one.

The trace hands us a **63-tile farm with a working executor immediately**. That is precisely the
setting where E35 said the hypothesis remains open. This plan is the shelved retest, now with a
substrate.

## 4. The three places we can safely change

| zone | why safe | what it buys |
|---|---|---|
| **Market orders** | a separate channel; changing what we sell moves no unit | shed pressure, sell timing, liquidation |
| **COW ↔ SHEEP** | both live on `PASTURE`; boatlee already demonstrates the swap | adapt the herd to the shop draw |
| **Weed repair** | reality already broke the script; repair + resync | recover otherwise-lost steps |

Off limits: which crops to plant, and where anything goes. Those desynchronise the trace.

## 5. Roadmap

Every phase states **hypothesis → test → kill criterion**, fixed before the work starts (PLAN2 §1).
Executable detail is in `TASKS3.md`.

### Phase 0 — Substrate, no strategy change

`agent/relay.py`: load the table, apply an ordered overlay stack. Register `relay-base`.

**Done when** `relay-base` emits byte-identical actions to `boatlee` over 20 seeds × 2 seats × 719
steps. If our copy already differs, no later result is interpretable.

**Do not modify `reference/kaggriculture/1/submission.py`.** It is the arena's only external
opponent and the only non-self-referential measurement in the project (D16). Editing it destroys
the instrument we measure against.

**Kill:** if bit-identity cannot be reached, stop — everything downstream would be unmeasurable.

### Phase 1 — Adaptive livestock *(BUILT, LOST, CLOSED — E50, E51)*

**Hypothesis.** Choosing COW vs SHEEP from the observed shop draw beats the fixed 9 cows + 4 sheep.

**Result: loses 78–80% head-to-head on the games where it changes anything**, with `blocked_ops` at
the `relay-base` baseline and 140 of 240 games ending in *exact ties* — so the implementation was
clean and the loss is real.

**Why: it is a bad trade.** Swapping two sheep for two cows yields **+46 milk at $160 and −42 wool
at $200** — −$1,040 at base prices, −$3,219 realised. A cow produces every 2 days against a sheep's
3, but that 50% rate advantage does not cover wool's 25% price premium at these volumes.

*An intermediate diagnosis — that milk sales were capped by the script, so the herd change was
blocked on a matching sell overlay — was **wrong** and is recorded in E50. It came from reading
order sizes as sales. The pairing was then built and tested anyway (`relay-paired`) and lost.*

**Status: closed.** It failed its own kill criterion, and the pairing that was supposed to rescue it
was built and lost too.

**A structural ceiling found on the way, worth carrying forward.** The trace buys animals on days
0, 5, 6, 7, 8 and 15, and shops unlock every 3 days — so most of the herd is committed when 0–2 shop
instances are known out of ~8. The purchase cannot be deferred, either: the scripted `PICKUP` lands
one step later and would find an empty shed. **The decision point is early and immovable**, which
bounds how much any livestock adaptation can be worth here, paired or not.

### Phase 2 — Market overlays *(low farm risk, not zero)*

**Two corrections to this phase, both made after R1 [E50].**

*First, "zero farm risk" was too strong.* A market overlay cannot desynchronise the table directly,
but changing cash changes **which purchases settle**, and that changes the farm. `blocked_ops` will
catch it; the claim needed weakening rather than the phase.

*Second, 2a and 2d were justified with numbers measured on the wrong agent.* Their evidence read
"ours deadlocks at 100 for 9 of 29 days" — **that "ours" is the champion engine**, which this line no
longer develops. On the relay line we *are* boatlee, and boatlee sits at the shed cap **0.9 days of
30** and strands **0.1 items per season** [E50]. Both premises are gone. This is the same error that
lost R1: reasoning from a number measured on a different system.

| | overlay | status |
|---|---|---|
| 2b | opponent-aware sell timing | **survives** — changes *when*, not *how much*, so it needs no surplus to work on. `Engine.opponent_supply()` / `forecast_price()` are built and tuned [E7] |
| 2c | spread terminal liquidation | **plausible** — nothing is stranded, so this is purely a price-impact question about dumping 165 wheat in one order |
| 2a | shed-pressure valve | **premise gone** — boatlee hits the cap 0.9 days/season. Only revisit if a production-side change creates surplus |
| 2d | suppress buys at shed cap | **premise gone** — the shed is full 3% of the time |
| **2e** | **paired: herd change + matching sell schedule** | **new, and where §2's second constraint points.** R1's production half is built; this adds the sales half and gates the pair |

**Test each alone at ≥80 games, then in combination.** E43 is explicit that this project's wins are
conjunctions — fertiliser and eager watering were each worth ~nothing and **+24% together**.
One-factor-at-a-time cannot find those, and three have been missed so far.

**Kill:** per item, dropped if it does not clear `relay-base` with the Wilson interval off 50%.

### Phase 3 — Generalised trace repair *(measure before building)*

Their repair fires only when the scripted op is `PLANT`/`BUILD_PASTURE` onto a weed. Extend to
`WATER`/`HARVEST`/`FERTILIZE`.

**Measure first.** Instrument how often a scripted op is invalid at its target tile. At
`weedSpawnChance = 0.005` expect ~6 weeds/season. **If fewer than 10 ops/season are blocked, skip
this phase.**

### Phase 4 — Trace re-optimisation *(last)*

This is PLAN2's P4.0. **Deprioritised on measurement, not on taste:** the most obvious available
edit — swapping out melon, which no shop buys in any game — cost boatlee **$18,882** [MEASURED E48].
The trace has less slack than it looks.

## 6. Prove the change did what it says, before reading its score

**A money number from an overlay that never fired is measuring noise, and it reads exactly like a
refutation.** This has already cost this project real conclusions:

* **E44** — three measurements were wrong in one day in ways that looked plausible, including a
  profile tool that sampled the roster at hour 0, just after `_end_of_day` clears it, and reported
  the workforce three times too small. Its rule: *verify the configuration produced the intended
  farm before interpreting its money*, because "our engine cannot express this" and "this strategy
  does not work" are completely different findings that produce identical numbers.
* **E39** — a comparison ran against a *re-implementation of the greedy rule inside the analysis
  script* rather than the engine's own plan, and reported an 18% gap that was largely its own. The
  engine now records `_last_plan` precisely so routing quality is never re-derived.
* **E36** — mutation testing found four parity tests that passed while proving nothing: a fuzz that
  never fertilised, a scenario that killed its own plant before the branch ran, a harness that never
  passed `shedCapacity`, and a guard made invisible by a `min()` clamp.
* **E46** — writing `tests/test_roles.py` exposed three defects in the first implementation, *none
  of which the money numbers would have revealed*. The test that caught them asserts purity rises
  **in play**, not that the cost function returns the right number.

**The mechanism, not the intention: every overlay emits a counter proving it fired, and the counter
is checked before the money number is read.** An overlay whose counter is zero is not a negative
result — it is an unfinished implementation, and reporting it as a refutation is the error above.

Two checks are mandatory for every phase from R1 onward:

1. **Effect counter.** Did the thing happen? Herd composition actually changed; the shed valve
   actually dumped; the sell actually moved. Asserted in a test, and reported alongside the money.
2. **Desync counter.** Did we break the choreography? `blocked_ops` — scripted actions that were
   invalid at their target tile — is built in **R0.5** and must not rise above `relay-base` for any
   overlay claimed to be safe. This is the single instrument that makes §2's rule enforceable
   instead of aspirational.

Assert behaviour **in play**, never that a helper returned the right value (E46). And a test that
cannot fail is worse than no test: where it is cheap, mutate the overlay and confirm the test
catches it (E36).

## 7. Measurement rules

* **≥80 games, both seats, every claim.** Three results died between 16 and 80 games in a single
  session (E37→E41, E39→E40, E42). A promising number *is* the signal to re-run.
* **Gate on the full registry gauntlet, not the boatlee mirror.** E40: optimal assignment earned
  more money and *lost* winrate in a mirror, because both sides flood one market and the gain
  competes away. We will be tuning in a near-mirror the whole time.
* **Every variant registered by name** in `arena/registry.py`, never a throwaway script (E43).
* **`make submission` green** before anything is called done. E21's $3,000 submission is the cost of
  skipping it.

## 8. What would kill this direction

Stated now, before the work:

1. **Phase 0 cannot reach bit-identity** — the substrate is unreliable.
2. **Phase 1 and Phase 2 both hit their kill criteria** — the trace has no slack reachable without
   desynchronising it.

In either case the answer is the capacity-planning layer on our own engine (see §9), and this plan
is recorded as failed rather than patched.

## 9. The alternative this was chosen over

For the record, so it can be resumed without re-deriving it.

**Our engine's failure is that it has no model of what a planted tile costs to keep alive.** A tile
planted today is a commitment to water it daily for 10–30 days; nothing in `agent/` represents that.
So it plants 14 tiles on day 0, nothing for ten days, then 46 across three days [E45]. Every plant
in a burst starts at `consecutive_unwatered = 1`, the units cannot cover it, and the cohort dies at
age 2 — **46 of our 56 crop deaths.**

Everything else follows: movement is a symptom of bursts scattering demand; `buy_land=False` is
*correct* for an engine that cannot sustain the tiles; melon is *correct* for a 15-tile farm [E43].
CEM found the best small farm because a myopic executor can only sustain a small farm [E30].

The unbuilt fix is a **planting policy that computes its own sustaining cost** — arithmetic, not
search, as `CLAUDE.md` already prescribes — built on `mimic` rather than the champion (E45's
un-acted conclusion), with `role_penalty` on from the start.

Sustainable area is a product of two terms:

```
serviceable tiles  ≈  productive unit-turns/day  ÷  service cost per tile/day
```

Every experiment so far moved **one** term and got nothing. E46's `role_penalty` raised the first
(70% vs the champion, **0% vs boatlee**). E45's rate cap fixed the second (+72% on `mimic`, **0%
winrate on the champion**). Both are real; both are worthless alone.

## 10. Open risks

1. **Adaptivity has a ceiling here.** A fixed trace can adapt its market and its livestock; it
   cannot adapt its crops. Planting happens days 0–12 and shops unlock every 3 days, so the crop
   plan is committed on roughly half the draw.
2. **We tune in a near-mirror.** Every A/B is boatlee-derived vs boatlee. E40 showed mirrors can
   rank changes backwards. The gauntlet is the mitigation, not a cure.
3. **One external opponent, still** (D16). Unchanged from PLAN2 §6.
4. **Environment drift.** `tests/test_env_version.py` guards it. If it fires, re-verify parity
   before trusting any result here — and note the trace itself is tuned for
   `townCenterSellInterval = 24`.
5. **Competition deadline still unknown**, raised five times.
