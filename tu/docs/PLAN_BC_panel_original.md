# Kaggriculture — Plan BC: Behaviour Cloning → RL

**A parallel line, not a successor.** `PLAN3.md` (boatlee-derived overlay) and `PLAN_v4.md`
(compiler) keep the submission. This document owns the learned-model line: a behaviour-cloning
policy trained on Kaggle leaderboard replays, fine-tuned with PPO.

Tag convention, used on every number below:

* **[MEASURED, verified]** — measured by a panel proposer *and* independently re-derived by the
  fact-check verifier on this machine.
* **[MEASURED, single-source]** — measured once, by one panel member. Re-verify before it becomes
  load-bearing.
* Env citations are `kaggriculture.py:NNN` (hereafter `kag.py`), against
  `.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`.

---

## 0. Mission

### 0.1 How this plan was made

Produced by a five-agent Opus panel: three proposers (architecture / data+eval / roadmap+RL) and two
verifiers (a plan critic that red-teamed the merge, and a fact-checker that independently
re-derived the load-bearing measurements without reading the proposers' code). Where a proposal and
the fact-checker disagreed, **the fact-checker wins**; four proposer numbers were corrected or
dropped this way (§1.7). Single-source numbers survived only because nobody re-ran them — treat
them as hypotheses, per CLAUDE.md's "a search result is a hypothesis."

### 0.2 The dual goal

Two deliverables, weighted equally, and **either can succeed while the other fails**:

1. **Competitive.** A learned policy that beats the sitting champion through `make promote`.
2. **Pedagogical.** The user (solid ML, new to RL/IL) learns imitation learning and RL properly.

Goal 2 is not decoration. Every phase carries a **learning exit** alongside its competitive exit,
and the two are **independently killable**: a phase whose competitive exit fails but whose learning
exit passes is *not wasted work*, and the plan says so up front so that a bad arena number does not
retroactively rewrite what the evening was for. Learning exits are written entries in
`docs/learning.md`, in the user's own words — not a checkbox, not a link dump.

### 0.3 Relationship to the other lines, and what this costs (D22)

Three statements, then this is never relitigated:

**(i) What is forgone.** `PLAN_v4`'s compiler line is the competitively optimal use of the same
evenings. The BC line is expected to *trail it on the leaderboard for roughly the first six weeks*.
`PLAN_v4` §5 explicitly rejects both halves of this program — "deep RL from scratch (env speed,
action space…)" and "pure behaviour cloning (caps at a tie)" [verified verbatim]. Those objections
are answered, not ignored: nothing here is from scratch (BC is the warm start), env speed is refuted
by §1.6, and "caps at a tie" is correct *about boatlee as a teacher* and is why boatlee is an
opponent and never a teacher (§3.3).

**(ii) The interop contract.** BC checkpoints register as pool members: `arena/registry.py`
`AgentSpec` with `kind: "bc"`, `params: {"checkpoint": path}`. Specs must stay **plain picklable
data, never a loaded model or a closure** — workers run under the `spawn` start method
(`arena/registry.py:24`, docstring) [verified]. Consequence: every BC checkpoint strengthens the
compiler line's opponent pool at zero marginal cost, and the compiler is the BC line's oracle, its
data-source insurance, and its anchor. The lines feed each other.

**(iii) The stop-loss.** The BC line **never touches `main.py`.** The shipped submission stays owned
by the compiler/promote line for the whole program. A BC model reaches `main.py` only by passing
`make promote` (D19), like any other candidate.

> Record as **D22** in `docs/decisions.md`.

One correction to the panel's framing: under the user's stated goal the **human replays are the
primary teacher**; the compiler is the DAgger oracle and the insurance policy. If `PLAN_v4` Phase 1
dies, that degrades **P4**, not P3.

### 0.4 Standing rules (apply to every phase)

1. Every phase has a **competitive exit** and a **learning exit**, independently killable.
2. Every measurement is **pre-registered with an E-number before it runs** (§9.2). `docs/experiments.md`
   is at **E85**, `docs/decisions.md` at **D21** [verified] — this plan starts at **E86 / D22**.
3. Every online claim: **≥80 games, both seats, a fresh seed block**, via `harness/run.py`
   (its docstring already states this floor) [verified]. Nothing below ~80 games is believable
   (CLAUDE.md; E33/E37/E41/E42).
4. Every offline claim: **Wilson interval**, reported beside the majority-class floor
   (16.3% / 19.3%, §1.5). Validation set **≥20k labeled decisions** or the interval is not worth
   printing. A "70% vs 19%" claim on 800 examples is the 16-game result this repo has died on
   three times.
5. **Prove the change fired before reading its score.** Every change emits a counter; check the
   counter first. A zero counter is an unfinished implementation, not a negative result (E44).
   For a *policy*, the counter instrument is `harness/counters.py` `Observer` — `idle_pct`,
   `steps_per_useful`, `blocked_ops`, `plants_lost_thirst` (`:53`, emitted `:193–198`) [verified].
   A silently all-idle policy is one log line, not four weekends.
6. **Name the opponent explicitly.** Never rank against `starter` alone ($3,507 at 1.32.7) — it is
   a liveness floor, never a ranking signal.
7. Nothing ships except through `make promote`.

---

## 1. What the panel established

### 1.1 The corpus we actually hold

One replay: `data/kaggriculture/95029942.json`. `info.TeamNames == ["Ryo Hasegawa", "tetsuya"]`,
rewards **$104,996 / $90,833**, both `statuses == "DONE"`, `module_version 1.32.7`,
`info.seed 746105689` [MEASURED, verified].

**These are 26–30× the `starter` baseline and are plausibly real competitive submissions. They are
not verifiably top-K, and they are not boatlee** (boatlee exists only as internal arena traces).
Proposer C's "the two-human Kaggle replay" is downgraded to **"a 1.32.7 leaderboard replay of
unknown reactivity"** — a `TeamName` is a human; the *agent* behind it is code, and boatlee proves a
top agent in this competition can be a 719-step open-loop table. Reactivity **cannot be tested from
one episode** (slot-variance needs ≥2 from one agent), so the experiment that settles it (E86/E88)
runs before any pipeline is built.

Volume, all [MEASURED, verified]:

| Quantity | Value |
|---|---|
| Steps / usable (obs, action) pairs | 720 / **719 × 2 = 1,438** |
| Unit decisions | **15,057** (7,270 seat 0 + 7,787 seat 1) |
| — of which pure transit (moves) | **7,168 (47.6%)** |
| — of which macro decisions | **7,889** ≈ **3,945 per seat-game** |
| Market orders | 1,218 (660 + 558) |
| Distinct unit ops / market ops | **17** / **6** |
| Distinct full actions (seat 0 / 1) | 35 / 33 |
| Max hands observed | 12 → **13 units**, pad to 16 |
| Raw JSON / gzip -9 | 31,011,787 B / **394,523 B (78.6×)** |

### 1.2 The decoding contract — four assertions, all mandatory

Each of these silently produces a dataset that trains to a plausible-looking metric if botched.

**(a) The offset.** `steps[i].observation` is the state *after* step `i`'s transition;
`steps[i].action` is the action that *produced* it (`core.py:277`, `:293`, `:301`).

> **Contract: the pair is `(steps[i-1][p].observation, steps[i][p].action)` for `i` in `1..719`.**
> `steps[0].action` is the schema default and **must be dropped**. 719 pairs per seat, not 720.

Proposers A and B stated this in different index letters (`(steps[t].obs, steps[t+1].action)` vs
`(steps[i-1].obs, steps[i].action)`) — same contract, presented once here.

**Assertion 1 (hand-roster).** `len(steps[i][p].action["hands"]) == len(steps[i-1][p].observation["farms"][p]["hands"])`.
Measured **1,438/1,438 against the previous step, 1,275/1,438 against the same step — exactly 163
mismatches** [MEASURED, verified]. The roster is cleared every `_end_of_day` (`kag.py:879-882`,
`farm["hands"] = []`, `hires_today = 0`, `private["inventories"] = [{}]`) [verified], so its length
is a fingerprint of *which* step's observation the action came from. Free, and it re-proves (a).

**Why this matters more than it looks.** Under the naive alignment, the Manhattan-shortest probe
(§1.4) returned **0.0%**; under the correct alignment it returns **99.5% / 100%**. A whole
architectural conclusion inverted on this one index [MEASURED, verified].

**(b) The seat-1 `step` patch (E21).** Delivered observations come from
`__get_shared_state(position)` (`core.py:754-767`) and carry `step` to **both** seats. The *stored*
replay state for seat 1 does **not** — measured at indices 0/1/5/100/400/719, seat-0-only keys =
`['step']`, seat-1-only = `[]` [MEASURED, verified]. It is exactly reconstructible:
`day*24 + hour == i` for **all 720 steps** [verified].

> **Contract: `delivered_obs(i, p) = dict(steps[i][p].observation)` then `["step"] = i`.**
> Never read `obs["step"]` from stored replay state.

**Assertion 2 (shared-field md5).** For `p == 1`, `md5(obs[k]) == md5(steps[i][0].observation[k])`
for `k in {farms, market, town, day, hour}` — verified byte-identical at indices 1/2/10/100/400
[MEASURED, verified]. `private` is genuinely seat-1's own and needs no repair. If this assertion ever
fires, the replay format changed and every downstream number is void.

**(c) Effective shed, with the `PLACE·animal` carve-out.** 412 SELL orders; **216 request more than
`private["shed"]` holds at observation time; all 216 fit once same-turn `DROP`/`PLACE`-to-shed
inventory is added; zero remain oversized** [MEASURED, verified]. Cause: unit actions are applied
before `_process_market` (`kag.py:935-941`) [verified].

> **Contract: `effective_shed = shed + Σ(unit inv contributed by DROP / PLACE-to-shed this turn)`.**
> **Carve-out: `PLACE·animal` places to a *tile*, not the shed (`kag.py:376-392`) and must be
> excluded.** The 216/216 result holds only with this carve-out.

Consequence: the decode order is forced — **units first, market second** — and effective shed is a
*required* market-head input, not a feature nicety.

**Assertion 3 (mask legality).** `n_expert_actions_rejected_by_mask` must be **0**. A mask that
rejects the expert's own action is a mask bug and looks exactly like a hard example.

**Assertion 4 (macro segmentation).** `frac_segments_shortest_path ≥ 0.9` per shard (§1.4).

**(d) Orders are not sales.** `_commit_unit` settles one unit at a time, alternating between players
against a shared pre-commit inventory; `SELL` moves `min(requested, shed)`; `BUY_PRODUCT` fails
outright against a full shed (`kag.py:641-658`, `:615-625`). Therefore:

* **Labels use the request, unchanged** — BC clones the *policy*, and the policy emitted the request.
* **Features never derive anything from a request.** Any "units sold" quantity is computed by
  **differencing shed + per-unit inventories between `steps[i-1]` and `steps[i]`**. The next
  observation is settlement ground truth and is already in the file. Never re-derive settlement
  logic in the decoder — that error cost E39.
* Per-step money delta = `steps[i].obs.farms[p].money - steps[i-1].obs.farms[p].money`, settlement-true
  by construction. Diagnostics and value pretraining only, **never** the RL objective.

### 1.3 Traps that already cost a probe

* **`farms[p]["tiles"]` is a nested 10×10 list, not 100 flat tiles.** This silently produced a no-op
  in the fact-checker's own first probe. Flatten explicitly and assert `len == 100`.
* **Kaggle never imports `main.py`** — it `exec`s the source into an empty globals dict and calls
  the last module-level callable (`agent.py:47-63`). `__file__` does not exist. Load weights
  relative to `os.getcwd()` or inline them. E21's exact failure was `NameError` → $3,000 starting
  bank → 0-40 against a real opponent.
* **`import torch` is charged to turn 1.** `build_agent` (`agent.py:145-157`) defers the `exec` of
  the submission source to the **first invocation**, and `Agent.act` starts the clock immediately
  before it (`agent.py:191-192`). Proved: a module body doing `time.sleep(3.0)` produced per-step
  durations `[3.0018, 0.0, 0.0, …]` and drove `remainingOverageTime` 60 → **57.998**
  = 60 − (3.0018 − 1) [MEASURED, verified]. Overage burns only when a step exceeds `actTimeout`
  (`core.py:631-632`).

### 1.4 Movement is a solved sub-problem

Segmenting each `(day, unit)` trajectory into runs of `NORTH/SOUTH/EAST/WEST` terminated by a
non-move op, and comparing segment length to Manhattan distance from segment start to the tile where
the terminal op fires [MEASURED, verified — re-derived from an independently written segmenter]:

| seat | segments | exactly Manhattan-shortest | detours |
|---|---|---|---|
| 0 | 1,690 | **1,681 (99.5%)** | 9 |
| 1 | 1,407 | **1,407 (100.0%)** | 0 |

Movement is genuinely unobstructed: `kag.py:326-331` bounds-checks only and its comment states
**`LOCKED` tiles are passable** [verified]. So Manhattan distance *is* true distance, always.

**This is why the macro action space is credible — and it is measured on one episode.** It becomes a
**P1 exit criterion computed corpus-wide** (§2.1), not a settled fact.

Tie-break when `dx ≠ 0 and dy ≠ 0`: the trace goes x-first 566 / y-first 345
[MEASURED, single-source]. Behaviourally irrelevant — no collision rule exists (`_spawn_hand`
`kag.py:533-541` counts occupancy but nothing forbids co-location) — so any deterministic rule works.

### 1.5 Floors, vocabularies, and structure

**Majority-class floors** — every offline agreement number is meaningless without these
[MEASURED, verified]: seat 0 `WATER` **1,186/7,270 = 16.3%**; seat 1 `PASS` **1,503/7,787 = 19.3%**.

**Terminal verb+arg vocabulary: 35 tokens across both seats**, distribution reproduced unit-for-unit
[MEASURED, verified]: `WATER` 2010, `PASS` 1807, `HARVEST` 827, `FEED` 607, `COLLECT_FERTILIZER` 604,
`CARE` 571, `DROP` 281, `FERTILIZE` 246, `PLANT·WHEAT` 233, `PICKUP·WHEAT·1` 162,
`PLANT·STRAWBERRY` 82, `DIG` 79, … down to `PICKUP·FERTILIZER·6` at 1.

**The atomic-PLANT cliff.** If total `PLANT` requests for a crop this turn exceed available seeds,
**all** of them are dropped, not just the excess (`kag.py:920-933`) [verified]. Measured: PLANT
bursts of 2 (**50** turns), 3 (**14**), 4 (**2**), and **66 turns where demand equals the seed count
exactly** — sitting on the cliff [MEASURED, verified]. This is the mechanical reason cross-unit
autoregressive conditioning is not optional.

**HIRE is phase-locked to dawn.** Measured **601 HIRE orders across 105 distinct turns, exclusively
at the day's first two decision hours** [MEASURED, verified]. *(A's "289 hires at hours 1–2" is
unreproducible and is dropped; only the qualitative claim survives, and it is what the feature
design needs.)* Hour-of-day is therefore a first-class feature.

**`PLACE` has variable arity — 2 args ×27, 3 args ×16** [MEASURED, verified]. Max order quantity
**61**. The arg head must handle an optional second argument.

### 1.6 Machine facts that size the whole program

* **`kagsim` is Rust** (`kagsim/Cargo.toml`, pyo3, `crate-type=["cdylib","rlib"]`), installed as a
  `.so` [verified]. **2.20 ms/episode = 326k steps/s** with pre-built actions; **13.93 ms = 51.6k
  steps/s** with both-seat `observation()`; dict marshalling is **84.2%** of sim cost. Reference
  Python env: **1.95 s/episode** [MEASURED, verified].
* **Therefore the 6-of-6 "rewrite the env" consensus is already satisfied.** `docs/decisions.md:59`'s
  "860 steps/s… not viable" verdict was about *staying on the Python env* and is stale [verified].
  **Do not port kagsim to JAX; do not build the Rust `VecEnv` (`TASKS.md:435`).** The one env-side
  optimization that pays is a tensor path replacing `observation()` marshalling.
* **PyTorch on this M1 Pro** (nhead=4, B=512, T=40) [MEASURED, verified, re-run 10–35% faster than
  the proposer's numbers, same ordering]: params 0.40 / 0.79 / 1.78 / 4.84 / 7.10 / 12.61 M →
  infer/s 33,373 / 16,612 / 7,938 / 3,326 / 3,048 / 1,587.
* **Depth is the MPS tax.** d384/L4 (7.10M) runs at 3,048 infer/s and 837 train/s vs d224/L8 (4.84M)
  at 3,326 / 800 — **the same speed at +47% params** [MEASURED, verified]. Spend parameters on
  width, never depth.
* **PPO step cost** `1/infer + n_epochs/train`: **2.9 ms/step at 4.86M (~30M steps/day)** vs
  **0.70 ms/step at 0.81M (~124M steps/day)** at `n_epochs=2` → over ~150 h of overnight compute,
  ~190M steps at 5M params vs ~775M at 0.8M, against 7th place's 2.2B
  [MEASURED, single-source, derived from the verified throughput table]. **From-scratch PPO at 5M
  params is not reachable here; the warm start is the enabling condition, not a nicety.**
* **bf16 on MPS gives no gain** (slightly worse) — skip it; train fp32. MPS beats CPU 2.5–3×
  [MEASURED, single-source].
* **`torch` is installed at `/opt/miniconda3/bin/python` (2.13.0, MPS available) but not in `.venv`**
  [verified]. This is an install task, not a blocker. **Caution: miniconda is deliberately pinned to
  `kaggle-environments==1.32.6` (E54)** — fine for pure-torch benchmarking, **never** for anything
  touching the market curve.
* **`import torch` costs 0.93 s warm** [MEASURED, verified] — not the 1–3 s a proposer assumed.

### 1.7 Corrections applied (proposals lost these arguments)

| # | Claim as proposed | Correction |
|---|---|---|
| 1 | `agent/main_v4.agent(obs)` is a queryable s→a oracle for DAgger | **Refuted.** It compiles once per day and indexes a precomputed script by `hour - start_turn` (`main_v4._act:201-223`). Cold-queried at its own trajectory states it reproduces its own action **166/699 = 23.7%** — 29/29 at hour 0, 30/30 at hour 1, **0/30 at hour 2**, 1/30 at hour 3. The oracle is **`agent.verify.compile_day(obs, plan, hands, turns, start_turn=hour, cash=0.0) → ops[0]`** (§5 P4). |
| 2 | Ship a numpy inference replica; torch is too slow | **Not required.** CPU single-thread, B=1, T=230 (Kaggle-realistic — MPS does not exist there): **3.4 / 6.7 / 11.2 ms per turn** at 0.79M / 2.67M / 4.84M → 2.5 / 4.8 / 8.1 s per episode against `runTimeout 1200` and `actTimeout 1` [MEASURED, verified]. The proposer's supporting 2.7–5 ms figure was **MPS** and must not be the plan's justification. |
| 3 | 6,646 transit + 8,076 macro decisions | **Arithmetically wrong** (sums to 14,722 ≠ 15,057). Correct: **7,168 moves, 7,889 macro, 3,945 per seat, 47.6% transit.** Design conclusion survives; the dataset-sizing numbers do not. **Any derived ratio computed against 8,076 must be recomputed** — including the "macro PASS = 22%" class-imbalance figure. |
| 4 | 289 hires at hours 1–2 | Unreproducible. **601 orders / 105 turns**, first two decision hours only. Keep the qualitative claim, drop the counts. |
| 5 | gzip -9 = 565,986 B (54.8×) | **394,523 B (78.6×)**; storage is **~0.39 GB per 1,000 episodes**, not 0.55. |
| 6 | `torch` not installed → blocker | Available under miniconda; a `.venv` install task. |
| 7 | "The two-human Kaggle replay" is the teacher | Downgraded to "a 1.32.7 leaderboard replay of unknown reactivity." The conclusion was stated ahead of the experiment meant to establish it. |
| 8 | Regenerate per-step observations by replaying through kagsim | **Delete the workstream.** Per-step observations for **both** seats are already in the replay JSON (`obs.farms` = 85.9% of 31 MB). That proposal was describing `traces/` (per-day state), a different artifact. Keep kagsim regeneration only on the compiler-generated data path, where obs is generated anyway. |

### 1.8 Env citations confirmed (7/7)

Atomic PLANT validation `:920-933`; `maxMarketOrdersPerTurn` `:551` with silent truncation
`queues.append(q[:max_orders])` `:560`; `BUY_PRODUCT` restricted to `{WHEAT, FERTILIZER}` `:598`;
LOCKED tiles passable `:326-331`; hands roster cleared `:879-882`; market processed after unit
actions `:935-941`; HIRE cost `mult * _fib(hires_today)` `:690-706`. Also spot-checked:
`agent/engine.py:722 opponent_supply`, `:757 town_drain_per_day` (a proposer said 756, off by one),
`:784 forecast_price`; `docs/AGENTS.md:198-203` tarball submission [all verified].

---

## 2. Architecture

### 2.1 Action space: macro, **contingent on the corpus-wide probe**

**Order of operations is acquire → probe → choose.** The Manhattan probe cannot run "before any
model code" as proposed — it needs replays that do not exist yet. It is a **P1 exit criterion**,
computed over the whole acquired corpus and reported per shard as `frac_segments_shortest_path`.

**The decoder emits raw per-step labels and macro labels from the same pass** (raw labels are
literally the action; macro labels are the inverse-labeled segmentation). The fork is then a
**training flag, not a rewrite**, and the probe result decides it.

**Recommended (if probe ≥ 0.9): macro = `(target tile, verb, item, qty)`, movement by scripted mover.**
A unit is queried only when idle.

```
commit head   : {IDLE, ACT}                                  2-way, per unit
tile pointer  : softmax over 100 own-farm tile tokens        (logits = tiles @ q / sqrt(d))
verb head     : V-way, on [unit_emb, tile_emb, ar_state, global]   (V settled per §2.2)
item head     : I-way, masked to the verb's legal domain
qty head      : log-bins {1,2,3,4,5,6,8,12,16,24,32,48,64+} + ALL = 14 classes
```

A hand-written mover emits `NORTH/SOUTH/EAST/WEST` until arrival, then the verb.

**If the probe fails (<0.9): do *not* fall back to raw per-step.** Fall back to
**macro-with-explicit-`MOVE`** — keep the tile pointer and the semantic verb vocabulary, add `MOVE`
as an extra verb so deliberate loitering and path-shaping are expressible. Three independent
Orbit-Wars writeups report a raw action space killing learning outright ("a far cry from
competitive"; switching to four intents "increased learning speed by a lot"), and `TASKS.md` T3.1
already specifies per-unit `(target tile, intent)` and "**Never emit `NORTH`**" [verified].
Abandoning semantics because the *mover* is imperfect throws away the wrong thing.

**Market: a second AR decode, after units.** Length ≤10 ordered list; order is causally load-bearing
(a SELL in slot 0 funds a BUY in slot 3 within the same turn, `kag.py:562-628`), and later slots can
be starved by interleaved settlement (`:615-625`).

```
for k in 0..9:
    stop head : {EMIT, STOP}
    op head   : 6-way {SELL, BUY_SEED, BUY_PRODUCT, BUY_ANIMAL, HIRE, BUY_LAND}
    item head : masked per op
    qty head  : log-bins + ALL
```

Run a **running money/shed/price simulation** across the AR steps using `market_price`
(`kag.py:192-206`) so slot *k*'s mask is exact. Feed it the **effective shed** (§1.2c).

**Cross-unit AR conditioning: keep it**, for the atomic-PLANT cliff (§1.5). Decode order = unit index
(farmer = 0, then hands in list order) — matches the action-list layout, stable within a day, and
carries meaning (index = hire order). Do **not** sort by size; there is no analogous magnitude here.
AR state carries running budgets — `seeds_remaining[crop]`, `money`,
`shed_free = shedCapacity - sum(shed)` — plus a learned projection of `(tile_emb, verb_emb)` per
committed macro.

### 2.2 Settle the vocabulary as one enumerated constant — before any code

The proposals **contradict each other arithmetically**, and a one-entry disagreement between the
head width and the mask table makes Assertion 3 fire and look like a data problem:

* **Verbs.** One proposal specifies a **14**-way verb head (+4 moves = 18 total); the other measured
  **17 distinct unit ops** *observed in sample* — which is not the vocabulary.
* **Items.** One says 12 (9 PRODUCTS + 3 ANIMALS) while also referencing `PICKUP·FERTILIZER`; the
  other says the shed holds 12 items. Whether FERTILIZER is inside the 9 is **unresolved**.
* **Unit slots.** 16 (mechanism-max) vs 13 (observed-max). **Take 16** — padding is free and static
  shapes matter (§2.6).

> **Rule: `bc/features.py` defines `VERBS`, `ITEMS`, `MARKET_OPS`, `QTY_BINS`, `N_UNIT_SLOTS = 16`
> as single enumerated constants, derived from `kag.py`'s dispatch tables (not from the sample), and
> asserted at shard build, at train time, and at inference.** Head widths read from those constants.
> Nothing else may hardcode a width.

### 2.3 Tokenization (summary; ~230 tokens, five types)

| Token type | Count | Content |
|---|---|---|
| Own tiles | 100 | kind[8], crop[5], animal[3], plant dynamics[9], animal dynamics[7], economics[3], geometry[10] ≈ 44 feats. Everything needed for exact masking is in the **public** `farms[p].tiles`; nothing about tile state is hidden. |
| Own units | 16 (pad) | is_farmer, index, x/y, shed-adjacency, dist-to-shed, carried-inventory vector, is_idle, hours_left_today ≈ 26 feats. Masked by `n_units = 1 + len(hands)`. |
| Opponent | 1 aggregate | money, tiles planted, animals, units, unlocked quadrants, hires_today, money_delta_last_day. **Do not tokenize their 100 tiles.** Their tile detail matters through exactly one channel — future market supply — and that is computable via `agent/engine.py:722 opponent_supply()`. |
| Market products | 9 | inventory, price, `n_shops_buying_me`, `town_drain_per_day` (`agent/engine.py:757`), `T` and `above_func` (`kag.py:41-51`), my shed / effective shed, my + opponent forecast supply at h = 3/7/14, `d(price)/d(unit)`, `forecast_price(h=7)` ≈ 22 feats. |
| Global | 1 | `step/719`, `day/29`, `hour/23` **and** `sin/cos(2π·hour/24)`, `is_hour_0`, money, money margin, shed fill, seeds, unlocked quadrants, next `BUY_LAND` price, next HIRE cost, shops unlocked, days-to-next-shop-unlock ≈ 24 feats. |

Geometry as **plain normalized coordinates plus a learned 100-entry position embedding**, not
Fourier — Orbit Wars needed Fourier for a continuous 100×100 board; ours is a discrete 10×10 grid.
**Do not tokenize shops separately** — they are drawn with replacement, max 8 instances, and their
only effect is per-product drain; folding them into product tokens as a count is lossless.

`n_shops_buying_me` and `town_drain_per_day` are the encoding of CLAUDE.md's most expensive lesson
(D17); `T` and `above_func` are the encoding of its correction (melon at 114 units is fine, at 360
it is the floor — E48/E41). Both belong in the token; the model picks the tradeoff.

**Seat is not a feature.** Canonicalize everything to me/them in the extractor. Both seats then
become one distribution and the data honestly doubles.

**Extractor input surface: the reference-env obs dict only.** Making the extractor also callable
from a kagsim state doubles the surface and drags `make verify` (the coverage-gated parity suite)
into scope for zero pre-PPO benefit. **Deferred to P5 entry**, where fast rollouts make it a genuine
requirement.

### 2.4 Masking — complete and exact from the observation, with zero hidden state

Verb legality is fully determined by the target tile and the unit's inventory:
`PASS` always (`:334`); `DROP` needs shed-adjacency + non-empty inv (`:132-139`, `:344`);
`PICKUP i n` needs shed-adjacency + `shed[i] > 0`, **seeds are not pickupable** (`:359-374`);
`PLACE a` (animal) needs matching structure, no animal present, `inv[a] ≥ 1` (`:381-392`);
`PLACE i n` (shed) needs adjacency + `sum(shed) < cap` (`:393-409`);
`PLANT c` needs empty tile + `seeds[c] > 0` **and the running AR seed budget** (`:417-429`, `:920-933`);
`WATER` (`:431-435`); `HARVEST` (`:446-453`); `FERTILIZE` (`:475-478`); `DIG` (`:484-489`);
`BUILD_COOP`/`BUILD_PASTURE` (`:493-502`); `FEED` (`:505-512`); `COLLECT_FERTILIZER` (`:515-521`);
`CARE` (`:524-529`); all tile-mutating verbs additionally require `T != "LOCKED"` (`:414`)
[MEASURED, single-source except the seven citations verified in §1.8].
Market: `HIRE` (`:698-706`), `BUY_LAND` (`:712-719`), `BUY_PRODUCT` (`:598`, `:662-671`),
`BUY_ANIMAL` (`:679-686`), `SELL` (`:653-654`), `BUY_SEED` (`:673-677`).

**Tile pointer: never masked** (all 100 tiles are reachable). Mask only via the verb head.

**Two traps.** (i) A fully-masked row produces NaN — always leave `PASS` unmasked. The reference had
to filter `|log_prob| > 1e5` for exactly this. (ii) **Assertion 3 is the gate for the whole line**:
a loss number computed under a leaky mask is uninterpretable. Port the reference's loss-skip filter
only **after** Assertion 3 reads zero — in the reference it was papering over a masking bug, and
suppressing hard batches to hide a mask defect is precisely CLAUDE.md's "a zero counter is an
unfinished implementation."

### 2.5 Loss

```
L = CE(commit) + CE(tile | ACT) + CE(verb | tile) + CE(item | verb) + CE(qty | verb,item)
  + CE(mkt_stop) + CE(mkt_op) + CE(mkt_item) + CE(mkt_qty)
  + λ_v · MSE(value, terminal_margin)                      λ_v = 0.1
```

**Masked cross-entropy everywhere; no regression heads.** Quantities are small integers with a highly
peaked distribution (`PICKUP n ∈ 1..6`, max order 61) — **bin them** (14 classes incl. `ALL`). MSE on
a bimodal integer target is the wrong objective and is the loss the reference was least happy with.

**Class imbalance: do nothing in v1, then measure.** The genuinely rare class is `BUY_LAND`
(2× per seat-game [single-source]) against `HIRE` 601× [verified]. Report **per-class recall** for
`BUY_LAND`, `BUY_ANIMAL`, `BUILD_PASTURE`, `PLACE·animal`. Only if a recall is ≈0 intervene — and the
right intervention is probably a 3-feature rule (`BUY_LAND` is near-deterministic given
`(day, money, n_unlocked)`), not loss weighting, which distorts calibration and is a common way to
make a model look better per-class while playing worse.

**Value head is trained here, in P3 — not in a later phase.** Entering PPO with a random critic means
the first several million steps train the critic while the actor is driven by noise: the most common
way a warm start is destroyed. Target = **terminal margin only**. All six Orbit-Wars finishers found
shaping measurably hurt; 7th place ablated it to 34.6% against its own baseline.

### 2.6 Model ladder — one ladder, three rungs

| rung | size | shape | role |
|---|---|---|---|
| **v0** | 50–150k | one head, sklearn/numpy | plumbing; **thrown away** |
| **v1** | **≈430k** | pointer-MLP, no attention | **the BC-era model** |
| **v2** | **0.8–1.5M** | **wide-shallow** (d ≈ 256, L ≈ 2–4) | the PPO model |

v1 breakdown: tile/unit/market/global encoders (shared, 128-wide) ≈ 84k; pointer ≈ 65k; verb/item/qty
heads ≈ 204k; market heads ≈ 50k; AR update ≈ 35k; value ≈ 17k. Trains in minutes on MPS.

**Rationale.** **The dataset, not compute, binds v1** — 50 episodes is only ~400k macro examples, and
a 3M-param transformer on that memorizes while the arena winrate does not move, which is this repo's
single most-repeated failure mode. Orbit Wars' own first model was 460k params of exactly this shape
and beat its baseline 100%.

**Size is set by the PPO budget, and BC inherits it.** BC training is offline and cheap at any of
these sizes, so BC never justifies growth; PPO has a hard step budget (§1.6). Growing at BC time and
shrinking at PPO time would waste the warm start.

**A 3.1M `d192/L6` transformer is rejected from the ladder and moved to the backlog.** It is the
worst point on the measured MPS curve — depth is the tax. If a transformer ever happens, it is
wide-shallow, and it is promoted **solely on ≥80-game arena winrate, never on held-out NLL**.

### 2.7 Six free constraints, adopted now (they retire three risks at once)

1. Forward pass is a **pure function of `(params_pytree, batch) → logits`** — no `self`, no module
   state, no in-place ops.
2. **Masks as data, never control flow.** No `if` on a tensor value, anywhere.
3. **Static shapes everywhere** — units padded to 16, market slots 10, tiles 100.
4. Loss a pure function of `(params, batch)`; optimizer state explicit.
5. RNG seed threaded explicitly, never global.
6. AR decode is a **bounded static loop** (16 units, 10 slots), so it maps mechanically to `lax.scan`.

These make (a) the JAX port cheap (P6), (b) numpy-vs-torch inference a ~20-line adapter chosen at
submission time rather than an architecture fork, and (c) the parity test
(`max|Δlogit| < 1e-4` **and** identical argmax over 500 states) trivial to write.

**Framework:** PyTorch + MPS, fp32. No bf16 (measured no gain on MPS, and the reference has a
documented NaN history around it). JAX only at P6.

---

## 3. Data

### 3.1 Acquisition — one evening, throwaway quality, then branch on measured yield

Four unknowns gate everything and **none is verifiable offline**:

| ID | Unknown | Cheapest verification |
|---|---|---|
| T0-a | Does the Kaggle Episodes API expose this competition? | `POST .../competitions.EpisodeService/ListEpisodes` with `{"teamId": <int>}`; inspect for 200 + `episodes[]`. |
| T0-b | Replay payload endpoint + shape | `POST .../GetEpisodeReplay` with `{"episodeId": 95029942}`; diff the body against our sample. If it matches, §1.2 applies unchanged. |
| T0-c | Rate limit | Unknown. Measure: 20 sequential requests, record 429s and latency. **Do not guess a number into the plan.** |
| T0-d | Credential form | `~/.kaggle/` contains **only** `access_token` — no classic `kaggle.json`; `which kaggle` → not found; `import kaggle` → ModuleNotFoundError [verified]. Confirm which auth the endpoints accept. |

**Preferred acquisition:** Meta Kaggle's daily `Episodes.csv` / `EpisodeAgents.csv` for the *index*
(bulk, unrate-limited) to pick which episodes belong to which top-K agent, then spend the rate-limited
`GetEpisodeReplay` calls only on episodes already chosen. Falsified if Meta Kaggle's refresh lags the
competition — check `max(Episodes.CreateTime)` against today.

**Yield ladder — branch here, do not improvise later:**

| Measured yield | Plan |
|---|---|
| **≥200 eps, ≥10 agents** | As written. Agent-level `test-agent` split is honest. |
| **20–200 eps** | Train, but an agent-level held-out split is likely impossible. Drop to episode-level split and **state in the plan and in every result that generalization is unmeasured.** Supplement with compiler-generated data. |
| **<20 eps, or API closed** | **Pivot the teacher to the compiler** (§3.2). |

### 3.2 The compiler-as-teacher contingency — the plan's structural insurance

Missing from all three proposals. `agent.verify.compile_day(obs, …)` generates unlimited (s, a) pairs
at any state. The decoder, features, masks, model, loss and eval ladder are **~95% identical**; DAgger
becomes trivially available rather than contingent; and PLAN3's replay-provenance question disappears
entirely.

> **Build it as a seam from day one: `bc/sources/{replay,compiler}.py` behind one interface.**
> Retrofitting it later costs a rewrite; building it now costs one adapter.

On the compiler path, and **only** there, per-step observations are generated rather than read, so
kagsim regeneration lives on that path (§1.7 item 8).

### 3.3 Curation

**The dominant risk: BC on a deterministic script learns `step_index → action`.** `boatlee` — a top
agent — replays a fixed 719-move list, unit actions identical across six games at
**100/100/99.9/100/99.8/100%**, with **zero** market orders added, removed or resized by its reactive
layers [MEASURED, E48]. If a meaningful share of top-K episodes come from such agents, then N episodes
carry ≈ one episode of information, the observation varies while the label does not, and **gradient
descent's correct response is to ignore the observation and regress on step index** — posting a
near-perfect offline score while being a slower, lossier `agent/relay`, which this repo already has
bit-exact. **Offline agreement cannot detect this.**

| Axis | Policy |
|---|---|
| **Which agents** | Top-K by final rating, K ≈ 10–20, plus the slot-variance filter below. Never top-1 only. |
| **Slot-variance rule** | For each agent with ≥2 episodes, over the 719×13 (step, unit-slot) grid, compute the fraction of slots whose action differs across ≥2 of that agent's episodes. **Exact action-sequence hash collision → collapse to 1 episode. Slot-variance <5% → admit at most 3 episodes** (enough to expose the few reactive branches). This is a **per-agent curation rule, not a program kill** (§5 P0). |
| **Which seats** | **Both seats of every admitted episode.** The sample's *loser* banked $90,833 — 26× the starter baseline. Seat asymmetry is real (settlement interleaves in player order, `kag.py:615-625`), so training seat 0 only leaves seat-1 behaviour unmodelled. |
| **Reward floor** | Filter on **absolute** bank, not margin. Drop any seat with `statuses[p] != "DONE"` — TIMEOUT/ERROR seats have truncated or garbage action tails. |
| **Recency** | Weight by **env version**, never wall-clock date (§3.6). |

Dedup keys measured on the sample [MEASURED, single-source]: seat 0 `14dcc5a044f37bbe`
(113,740 B of canonical action JSON), seat 1 `cf841c2a80077642` (115,149 B).

**Minimum viable N.** ~3,945 macro decisions per seat-game → **≥50 episodes ≈ 400k macro decisions**
for the 430k v1. Below that, v1 shrinks or the compiler source fills the gap.

### 3.4 Storage — two tiers

**Tier 1, immutable archive: `data/replays/{episode_id}.json.gz`.** **394,523 B/episode** measured
→ **~0.39 GB per 1,000 episodes**, ~3.9 GB at 10,000. **Keep the raw bytes forever** — decoding bugs
are the likeliest defect in this pipeline and re-downloading under an unknown rate limit is the
expensive recovery path.

**Tier 2, training shards: `data/shards/{split}/{agent_slug}-{episode_id}.npz`** (gitignored). Flat
`.npz`, `glob`-discovered, shard-level *and* within-shard shuffling, ragged last batch dropped.

**State-major / label-minor layout.** There are ~10.1 (seat 0) / 10.8 (seat 1) unit decisions per
state [MEASURED, single-source], so one row per unit-decision would duplicate the state ~10×:
`states[1438, …]` int8/int16 + `labels[~15057, …]` rows carrying `state_idx` + `market_labels[~1218]`.
Estimated **~2.7 MB/episode uncompressed** [single-source]; **[VERIFY on the first 10 decoded
episodes — if compressed shards exceed ~1.5 MB/episode, revisit].**

Rejected: Parquet/Arrow (extra dependency, and the reference loader is `.npz`-shaped);
uncompressed JSON archive (31 MB/episode for zero benefit).

Size decomposition, for anyone optimizing later [MEASURED, single-source]: `obs.farms` 85.9%,
`obs.private` 1.7%, `obs.market` 1.2%, `action` 0.7%, `obs.town` 0.4%. `farms` is duplicated
verbatim across seats — storing shared fields once halves the archive before any encoding work.

### 3.5 Splits — agent first, then episode, **never step**

* **Step-level splitting is fatal.** Adjacent steps share ~99% of tile state; a random step split
  measures nearest-neighbour recall, not generalization.
* **Episode-level splitting is still leaky across a scripted agent's episodes** — two episodes of the
  same script share an identical 719-action label sequence.
* Therefore: **held-out AGENTS are the only honest generalization signal.**

| Split | Content |
|---|---|
| `train` | Episodes from agents in the training roster |
| `val` | Held-out **episodes** from *training* agents — measures fit, drives early stopping |
| `test-agent` | All episodes of 2–3 **entirely held-out agents** — the only number allowed to be called "generalization" |

Assign agents by `sha1(agent_name)` bucketing so splits are deterministic as new episodes arrive.
Agent identity from `info.TeamNames[p]`. If the yield ladder lands in 20–200, say so in every result.

### 3.6 Version skew

`module_version` is a top-level key of the replay JSON (sample: `"1.32.7"`), and `configuration` is
stored in full [MEASURED, verified]. **Train on 1.32.7 only; archive everything.** Justification is
the repo's own record: E33 measured 1.32.6 cutting demand **4.7×** and changing shop draws to
with-replacement; E54 measured 1.32.7 adding a scarcity spike. A market-timing policy cloned from
1.32.6 play is optimized against a demand curve that no longer exists.

Nuance: **unit heads are far less version-sensitive than market heads** — E33/E54 changed market
curves and shop draws, not tile mechanics. Ranked fallback if 1.32.7 is scarce: (1) 1.32.7 only;
(2) unit heads on 1.32.6+1.32.7, market heads on 1.32.7 only, with a version feature; (3) everything
with a version one-hot — **only** if an ablation shows it helps. Store `module_version` and a
`config_hash` as columns on every shard so all three are a filter, not a re-decode.

---

## 4. Evaluation ladder

**Rung 1 — offline, per head, every epoch.** Top-1 and top-3 agreement **per head separately**
(unit-op, unit-arg, market-op, market-item, market-qty) — never one blended number, which the unit
heads would dominate 15,057 : 1,218. **Every number reported against its majority-class floor
(16.3% / 19.3%) with a Wilson interval**, val ≥ 20k labeled decisions.

> **This rung can only falsify, never confirm.** High agreement is fully consistent with having
> learned `step_index → action`.

**Mandatory companion — the anti-clock ablation.** Re-score the val set with `step`/`day`/`hour`
features zeroed. **If accuracy drops <5pp, the model is a clock, not a policy — stop and fix the
data, not the model.**

**Rung 2 — online sanity (does it even run legally).** Wrap the checkpoint as `agent(obs)`; run one
full 720-step episode **in each seat**. Gate: no exception, `status == "DONE"`, final bank **above
the $3,000 starting bank**. This is exactly the trap `make submission` was built to catch. Record
`import` cost and p99 turn time here.

> **Overage caveat, and it is not optional.** `remainingOverageTime` is 60 s and the local `env.run`
> path **never raises TIMEOUT** — `DeadlineExceeded` comes only from Kaggle's production runner
> (`core.py:281`) [verified]. **Local testing cannot detect an overage blowout.** A cold-disk first
> import on Kaggle is unmeasurable offline and is the only real timing risk. Budget by the measured
> CPU numbers (§1.7 item 2) and keep the import small; do not conclude "it fits" from a green local
> run.

**Rung 3 — pool play, ≥80 games, both seats, fresh seeds.** `harness/run.py` against
`DEFAULT_POOL = ["boatlee", "executor_v7", "starter", "kagsim_champion"]`
(`harness/registry.py:212`) [verified], with `harness/counters.py` `Observer` run on every checkpoint.
**Mean bank is a diagnostic; pairwise winrate is the ranking.** `starter` is a liveness floor only.

**Rung 4 — head-to-head promotion gate. `make promote` only (D19).** `tools/promote.py` already
implements the protocol and is not to be rebuilt: `SCREEN_GAMES = 12` (24 episodes) escalating to
`CONFIRM_GAMES = 250` (**500 episodes**, resolves ≥54.4%) whenever the screen lands in
`NOISE_BAND = (0.35, 0.65)`; Stage 2 no-new-losses; Stage 3 neighbourhood sweep [all verified].
`boatlee` is reported as a **reference that never gates** (D21).

**A promotion gate is not a kill criterion.** Rung 4 is a later, optional event; it must never be
written into a phase kill, or a first learned model is required to beat a mature scripted champion as
a condition of existing.

---

## 5. Phases

### P0 — Acquisition and the two gates

**Goal.** Get replays; prove something state-conditioned exists to learn.
**Entry.** None.
**Exit.** T0-a..d settled; ≥N episodes archived to `data/replays/`; **G0-a** slot-variance computed
per agent (E88); **G0-b** clock-vs-state ablation run on boatlee as positive control, then on the
human replays (E86); teacher chosen per the yield ladder (§3.1); `torch` installed in `.venv` with
`make test` and `make verify` still green.
**G0-a (dataset property, no model).** Cross-episode slot-variance over the 719×13 grid. Needs ≥2
episodes per agent. Detects pure and near-replays.
**G0-b (model property, single episode).** Fit step-index-only against state-features-with-step-withheld;
compare held-out agreement. **Run boatlee first — the answer is known (E48), so it validates the
harness.** Survives when only one episode per agent is obtainable, which G0-a does not.
They answer different questions ("does this *agent* vary?" vs "is there state-conditioned structure in
the corpus at all?"). Neither substitutes for the other.
**Kill.** **G0-b shows <2pp state-over-clock on *every* candidate teacher, including the compiler.**
Then there is no imitable policy anywhere and the BC line dies for the cost of two evenings.
*A scripted leaderboard alone does not kill it — it re-points the teacher (§3.2).*
**Learning exit** (`docs/learning.md`). In your own words: why BC is supervised learning **with a
hostile test distribution**; why no-op dominance means a 97%-accurate always-idle model is the
*default* failure, not an edge case.
**Reading.** Pomerleau, *ALVINN* (NeurIPS 1988) — the original BC system, 5 pages; its "steer back
from the edge" augmentation hack is exactly the problem you are about to hit. Levine, CS285,
*Supervised Learning of Behaviors* (Lecture 2; numbering drifts by year) — the canonical why-naive-BC-fails
framing with the drift diagram. Lin et al., *Focal Loss for Dense Object Detection* (2017), §3 only —
for the no-op-dominance fix.
**Budget.** 2–3 evenings.

### P1 — Decode and extract: the data contract

**Goal.** Replay/compiler → verified shards, **both label sets from one pass**.
**Entry.** P0 exit.
**Exit.** All four decoder assertions at **zero failures**: hand-roster 1438/1438; seat-1 shared-field
md5; **`n_expert_actions_rejected_by_mask == 0`**; effective shed computed by differencing, never from
orders, with the `PLACE·animal` carve-out. `frac_segments_shortest_path` computed **corpus-wide** →
macro-vs-raw decided (E87, §2.1). Vocabulary settled as one asserted constant (§2.2). Splits
materialized agent-first. Majority-class floors recorded for the real corpus. Shard size measured
against the 1.5 MB/episode threshold.
**Kill.** The mask cannot be driven to zero expert rejections → **the action-space model is wrong;
redesign before any training.** A loss number computed under a leaky mask is uninterpretable — this
is the "prove the change fired" gate for the entire line.
**Learning exit.** What a *label* actually is here; why an off-by-one in the (obs, action) pair still
trains, still converges, and still produces a plausible accuracy curve while cloning "the action taken
one step ago"; why a mask that rejects the expert is indistinguishable from a hard example.
**Reading.** Re-read the reference `train_bc.py` loss-skip filter (lines 143–188, 251–268) and write
down *why* you are not enabling it yet.
**Budget.** 1 weekend + 2 evenings.

### P2 — Plumbing model (v0)

**Goal.** The smallest thing that runs end to end — then throw it away.
**Entry.** P1 exit.
**Exit.** (i) A one-head sklearn/numpy baseline beats its majority floor by a stated margin with a
Wilson interval excluding the floor. (ii) A raw-per-step v0 checkpoint wrapped as `agent(obs)`
completes a 720-step episode **in both seats**, `status == "DONE"`, no exception, bank above $3,000
(Rung 2).
**Kill.** Cannot beat the majority floor on the unit-op head → **encoding bug, not a model problem.
Fix P1; do not tune.**
**Learning exit.** Why "97% accuracy" under no-op dominance is worthless, computed on your own data;
the difference between offline agreement and online return, stated *before* you have any reason to
believe it.
**Budget.** 2–3 evenings.

### P3 — BC proper (v1, 430k pointer-MLP + value head)

**Goal.** The real model on the real corpus.
**Entry.** P2 exit.
**Exit.** (i) Per-head agreement with Wilson CIs against floors (E89); (ii) **the step/day/hour
ablation drops ≥5pp** (E90) — the anti-clock diagnostic; (iii) `BUY_LAND` per-class recall logged;
(iv) online money vs the pool at ≥80 games both seats via `harness/run.py` with `Observer` counters
(E92); (v) **the handover curve measured** (E91); (vi) inference timing recorded → numpy-vs-torch
decided (E93). **The value head is trained here**, `λ_v = 0.1`, terminal margin only.
**Kill.** The ablation shows <5pp drop → **the model is a clock; fix the *data*, not the model.**
Or: agreement stalls at the floor on any head *with the vocabulary verified* → the action encoding is
wrong. **Do not kill on "online money < expert"** — that is P4's problem, and tuning here burns weeks.
**Learning exit.** State the **O(T²ε)** bound and why T = 719 makes a 1% per-step error not a 1%
problem. Explain why per-timestep i.i.d. BC and teacher forcing are the same construction, and that
the drift is exposure bias by another name.
**Reading.** Ross & Bagnell, *Efficient Reductions for Imitation Learning* (AISTATS 2010) — the T²ε
bound; **the single most important paper in this plan.** Bengio et al., *Scheduled Sampling for
Sequence Prediction with RNNs* (2015) — teacher-forcing/exposure-bias made concrete. Vinyals et al.,
*Grandmaster level in StarCraft II* (Nature 2019), supervised-from-replays section — 971k replays, and
the supervised agent alone beat 84% of human players; it calibrates how far good BC gets.
**Budget.** 2 weekends.

### P4 — Close the distribution-shift gap

*(Renamed from "DAgger". DAgger is one of three instruments, and the phase's learning value does not
depend on any of them.)*

**Goal.** Beat compounding error.
**Oracle ladder, in order, with an explicit skip:**
1. **`agent.verify.compile_day(obs, plan, hands, turns, start_turn=hour, cash=0.0) → ops[0]`** — takes
   an obs, so it is queryable by signature, and it is state-responsive (removing all PLANT tiles
   changed the action 3/3) [verified]. **`cash=0.0` is hardcoded at `main_v4.py:213`, so money
   perturbations legitimately do nothing.** Season state (`_watch_opponent`, `branch_points`,
   `season_planner`, module-level `_STATE`) is **not reconstructible from a single obs and is declared
   out of scope for the oracle.**
2. Any repo **reactive** engine (`executor_v7`, `kagsim_champion`) — reactive engines are s→a
   functions by construction, queryable at arbitrary states.
3. **Skip DAgger; go straight to P5.** PPO fixes distribution shift too, just more expensively.

**Entry (a probe, not a discovery).** Before aggregating a single label, measure (E94): (i) the
fraction of oracle queries returning a no-op or an exception at BC-visited states, against a stated
bound; and (ii) whether **the oracle outscores the BC policy on the same seed block**. If (ii) fails,
the oracle is not an expert here and DAgger cannot help *by construction*. **Score the oracle
function, not `main_v4.agent`** — they are different policies (§1.7 item 1).
**Exit.** Online money improves over P3 by an interval excluding zero at ≥80 games both seats (E95);
the handover curve visibly flattens.
**Kill.** Two aggregation rounds move less than the 80-game CI, **or** no oracle passes entry →
**skip to P5 with the P3 checkpoint.** This is graceful degradation, not failure.
**Learning exit — survives either outcome.** Why DAgger is O(T) where BC is O(T²), and what
"no-regret online learning" buys. The handover curve (E91) measures compounding error **with no oracle
at all**, so the pedagogical value of this phase is unkillable even when its engineering value is.
**Reading.** Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
No-Regret Online Learning* (AISTATS 2011) — DAgger. Rajeswaran et al., *Learning Complex Dexterous
Manipulation with Deep RL and Demonstrations* (RSS 2018) — DAPG; the concrete recipe for the KL-to-BC
schedule used in P5. Ho & Ermon, *Generative Adversarial Imitation Learning* (2016) — read for the
framing of what a queryable expert lets you avoid.
**Budget.** 1 weekend + 2 evenings.

### P5 — PPO fine-tune (v2, 0.8–1.5M wide-shallow)

**Goal.** Turn the warm start into a policy that exceeds its teacher.
**Entry.** P3 or P4 checkpoint; a decided control surface; a model sized by §1.6 with a **stated step
budget**; the extractor made callable from a kagsim state (deferred here from P1, §2.3) with
`make verify` still green.
**Design decisions, each with its evidence.** Terminal reward only, as `sign(my_money − opp_money)` —
6 of 6 finishers, and 7th ablated shaping to 34.6% against its own baseline. *(Flag the open question:
if Kaggle ranks on absolute score rather than head-to-head, margin-sign is the wrong objective;
`PLAN_v4` §0 targets winrate, so follow that, but A/B normalized margin once and record it.)*
γ = **0.999**, not 1.0 — 1st place's single biggest regret was that γ=1.0 made his agent stall.
**KL-anchored to the BC checkpoint with a decaying coefficient.** **PFSP** opponent pool weighted
toward opponents beaten ~50% of the time, seeded with `boatlee`, the compiler, and
`search/exploiters.py`'s `flooder`/`tomato_rusher` [verified] — **never a live copy of self** (the
reference's "league" returns `league[-1]`, i.e. near-naive self-play; do better). Seat flipped per env
rather than augmenting data. **Not applicable, and say so:** early termination of decided games has no
clean analogue — economic accumulation has no settled state before the final day.
**Exit.** ≥60% vs the **frozen** BC checkpoint at ≥80 games both seats on fresh seeds (E97); no
regression vs `DEFAULT_POOL`; p99 turn time < 100 ms. Per-head entropy, `clip_frac` and `approx_kl`
logged every iteration — **a flat-lined head is a stop condition** (3rd place: entropy annealing was
"by far the most important knob"). E96 runs overnight.
**Kill.** 100M steps without reaching 55% vs frozen BC → drop to the market-only control surface.
**Do not add shaping. Do not grow the model.** If `approx_kl` spikes and entropy collapses in the
first 5M steps, the KL anchor is mis-scheduled — fix that before spending the budget.
**Learning exit.** The full chain in your own words: REINFORCE → baseline → actor-critic → GAE → **the
PPO clip**; and why 6 of 6 writeups found shaping hurt.
**Reading.** Schulman et al., *High-Dimensional Continuous Control Using GAE* (2015) then *Proximal
Policy Optimization Algorithms* (2017), in that order. Huang et al., *The 37 Implementation Details of
Proximal Policy Optimization* (ICLR Blog Track 2022) — a checklist, not a read; advantage
normalization, orthogonal init, value clipping and LR annealing are where the silent failures live.
Berner et al., *Dota 2 with Large Scale Deep RL* (2019), §self-play — 80% latest / 20% past — paired
with the AlphaStar league (main agents / main exploiters / league exploiters, PFSP) from the Nature
paper already read in P3.
**Budget.** 3–4 weekends hands-on plus overnight compute; **a fixed step budget per experiment, then a
tournament** (7th place's protocol).

### P6 — Scale-out: GPU + JAX *(contingent, unscheduled)*

**Goal.** Buy throughput only once it is the measured bottleneck.
**Entry.** P5 measured **training-throughput-bound** (plausible — the model is ~14× the sim's cost)
**and** a GPU actually rented.
**Exit.** JAX logits match PyTorch to 1e-4; ≥5× training speedup (E98).
**Kill.** <5× → stay in PyTorch.
**Learning exit.** Why the six constraints in §2.7 made this a port rather than a rewrite.

### Backlog *(not a phase — no entry/exit/kill)*

Aux future-prediction heads (7th place: extra heads predicting state 2/8/32/64 turns ahead, discarded
at inference, "helped quite a bit"; cheap during BC, forces a world model into the trunk).
A 3.1M transformer, **wide-shallow only**, promoted solely on ≥80-game arena winrate, never on NLL.
Offline-RL reading — Levine et al., *Offline RL: Tutorial, Review, and Perspectives* (2020) and
Fujimoto & Gu, *A Minimalist Approach to Offline RL* (TD3+BC, 2021, 4 pages) — read **to understand
why it is declined** (we can interact with the env at 51.6k steps/s, so the entire motivation is
absent), and because "BC plus a small RL term" is a direct sanity check on the P5 KL schedule.

---

## 6. Risk register

Ordered by probability × cost-to-discover-late.

| # | Risk | Where it bites | Mitigation |
|---|---|---|---|
| 1 | **Acquisition yield is too low** or the API is closed | P0 | Yield ladder (§3.1); compiler-as-teacher seam built from day one (§3.2) |
| 2 | **Top agents are scripts** — the model learns `step → action` | P0/P3 | G0-a slot-variance + G0-b clock ablation; the ≥5pp anti-clock ablation as a P3 kill |
| 3 | **(obs, action) off-by-one** — trains, converges, looks fine, clones the wrong action | P1 | Assertion 1, 1438/1438 hand-roster, hard assert in the decoder |
| 4 | **The mask rejects expert actions** — every loss number becomes uninterpretable | P1 | Assertion 3 == 0 as a hard P1 exit; vocabulary as one asserted constant |
| 5 | **The macro probe fails corpus-wide** | P1 | Both label sets emitted from one pass; fallback is macro-with-explicit-`MOVE`, never raw |
| 6 | **No qualifying oracle** | P4 | Oracle ladder with an explicit skip; the learning exit survives via the handover curve |
| 7 | **PPO compute is insufficient** | P5 | Size to the step budget (0.8–1.5M, wide-shallow); fixed step budget per experiment; drop to market-only surface |
| 8 | **numpy/torch parity break** — scores like an untrained model, indistinguishable from "BC didn't work" | P3/ship | Pure-function forward (§2.7); `tests/test_inference_parity.py`: `max|Δlogit| < 1e-4` **and** identical argmax over 500 states |
| 9 | **Feature/weight skew across a checkpoint reload** — the E21-class ML failure | all | `FEATURE_VERSION` asserted at load; code hash; dataset manifest hash (§7.3) |
| 10 | **Overage blowout on Kaggle**, undetectable locally | ship | §4 Rung 2 caveat; keep imports small; budget by the measured CPU numbers |

---

## 7. Repo layout, targets, reproducibility

### 7.1 Layout

```
bc/
  acquire.py            # throwaway-quality downloader; T0-a..d live here
  sources/replay.py     # Kaggle replay JSON  -> (obs, action) stream
  sources/compiler.py   # compile_day rollouts -> (obs, action) stream   [the insurance seam]
  decode.py             # the four assertions; raw AND macro labels, one pass
  features.py           # VERBS / ITEMS / MARKET_OPS / QTY_BINS / N_UNIT_SLOTS; the extractor
  dataset.py            # shard build, splits, manifest hashing
  model.py              # forward(params, batch) -> logits   (pure function, §2.7)
  train.py
  infer_numpy.py        # ~20-line adapter, only if §4 Rung 2 timing says so
  eval.py               # the four rungs
data/replays/{episode_id}.json.gz     # tier 1, archived forever
data/shards/{split}/*.npz             # tier 2, gitignored
tests/test_bc_*.py
docs/learning.md                      # the learning exits
```

### 7.2 Make targets

`make bc-shards`, `make bc-train`, `make bc-eval`. BC checkpoints wire into `harness/registry.py` and
`arena/registry.py` as `kind: "bc"`, `params: {"checkpoint": path}` (§0.3).

**`make verify` scope is unchanged for P0–P4.** The extractor is fed from the reference-env obs dict
only; kagsim-state support is a **P5 entry requirement**.

### 7.3 Checkpoint reproducibility

A checkpoint stores: **weights, `FEATURE_VERSION` (int, asserted at load), a code hash, and a dataset
manifest hash** (episode ids + `module_version` + `config_hash`). Silent feature/weight skew produces
a model that scores like an untrained one and is **indistinguishable from "BC didn't work."**
`tests/test_bc_checkpoint_roundtrip.py` is not optional.

---

## 8. First two weeks (evenings + one weekend)

1. **E1.** T0-a..d — hit `ListEpisodes` / `GetEpisodeReplay`; settle auth and rate limit. Install
   `torch` into `.venv`; re-run `make test` and `make verify` and confirm both stay green.
2. **E2.** Throwaway downloader; pull **10 episodes from one top agent** into `data/replays/`.
3. **E3.** **G0-a** slot-variance across those 10 (**E88**); **G0-b** on boatlee as the positive
   control.
4. **Weekend AM.** **G0-b** on the human replays. **Decide the teacher** per the yield ladder.
   Record as **E86**.
5. **Weekend PM.** `bc/decode.py` + the four assertions. Enumerate the verb/item vocabulary as one
   asserted constant (§2.2) **before either side writes head widths.**
6. **E4–E5.** `bc/features.py`; `frac_segments_shortest_path` corpus-wide → macro-vs-raw decision.
   Record as **E87**.
7. **E6.** Shard build; sizes measured against the 1.5 MB threshold; splits materialized. Write the
   **P0 learning-exit entry** in `docs/learning.md`.

---

## 9. Log entries

### 9.1 Decisions to add

**D22 — The BC line is a parallel program with a stop-loss.** Its cost is named (it trails the
compiler line on the leaderboard for ~6 weeks); its interop is defined (BC checkpoints register as
picklable `AgentSpec`s with `kind: "bc"`, strengthening the compiler line's opponent pool at zero
marginal cost); and it **never touches `main.py`** — a BC model ships only through `make promote`.
Every phase carries a competitive exit and a learning exit, independently killable.

### 9.2 Pre-registered experiments

E-numbers are a registry, not a schedule. Each is claimed before it runs.

| E | Phase | Measurement | Decides |
|---|---|---|---|
| **E86** | P0 | Clock-vs-state ablation (G0-b): step-index-only vs state-with-step-withheld, boatlee positive control then human replays | Whether an imitable policy exists at all; **program kill** |
| **E87** | P1 | `frac_segments_shortest_path`, corpus-wide, per shard | Macro vs macro-with-`MOVE` action space |
| **E88** | P0 | Slot-variance per agent (G0-a) over the 719×13 grid | Per-agent episode caps; teacher choice |
| **E89** | P3 | Per-head agreement + Wilson CIs vs the 16.3% / 19.3% floors | Whether v1 learned anything |
| **E90** | P3 | step/day/hour ablation | **Phase kill** at <5pp drop |
| **E91** | P3→P4 | Handover curve (expert plays 0..k, BC plays k..719) + agreement by state provenance | The size and season-phase of the compounding-error gap |
| **E92** | P3 | Online money vs `DEFAULT_POOL`, ≥80 games, both seats, with `Observer` counters | Whether the offline number carries online |
| **E93** | P3 | Inference timing: import cost + p99 turn time from one timed `env.run` | numpy vs torch at submission |
| **E94** | P4 | Oracle-quality probe: degenerate-response rate + oracle-vs-BC on the same seeds | **Phase entry**; skip-to-P5 decision |
| **E95** | P4 | Δ online money over 2 DAgger rounds vs the 80-game CI | Phase exit / kill |
| **E96** | P5 | PPO from scratch vs from BC vs from BC+KL, equal step budget | Whether the warm start is a floor or a cage; the KL coefficient |
| **E97** | P5 | ≥60% vs frozen BC, ≥80 games both seats, fresh seeds | Phase exit |
| **E98** | P6 | JAX↔PyTorch logit parity + training speedup | Stay or port |
| E99 | backlog | Cross-unit AR ablation (parallel decode) — NLL **and** ≥80-game winrate | Delete 16 sequential decode steps? |
| E100 | backlog | Market AR vs bag-of-orders head, same shards, one epoch | Delete 10 sequential decode steps? |
| E101 | backlog | Full opponent tile tokens vs the aggregate (>2% held-out NLL to promote) | +40% compute? |
| E102 | backlog | Binned qty vs sigmoid+MSE head | Loss form |

*(Proposer C's `E-L1` / `E-L2` / `E-L3` are renamed to **E86 / E91 / E96**; the "L" names survive as
prose aliases only, per the repo's numbering convention.)*
