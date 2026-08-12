# Kaggriculture — Task Breakdown

Execution plan for `PLAN.md`. Every task states **what to build**, **how to verify it**, and a
binary **done-when**. Do not mark a task done on "it runs" — done means the verification passed.

## Status

Last updated after E13 (rules audit). Gate: `make verify`. Ranking authority: `make arena` — but see the verification programme: that arena is currently
circular, and V1/V2 exist to break the circle.

| Task | State |
|---|---|
| T0.0–T0.6 simulator, parity, harness | **done** — see the verification summary below |
| T0.7 economics validation | **done** — `docs/experiments.md` E1–E3; 2 `PLAN.md` claims overturned |
| T0.8 local arena | **done** — `arena/`, OpenSkill + Wilson CIs, sqlite + HTML (E5) |
| T1.1–T1.4 scripted engine | **done** — task generator, routing, cash management, market policy |
| T1.3 market/selling policy | **done** — opponent supply forecasting (E7) |
| T2.1 CEM over `Params` | **done** — joint 29-knob search, held-out validated (E8, E9) |
| T1.2 routing rework | **reopened and partly done** — E6 reversed: units are now **movement-bound at 58-65%** (E17) |
| T1.5 submission wrapper | **done** — `main.py` + `make submission`; plays both seats on the reference env, **p99 0.1 ms** of a 1 s budget |
| T2.2 exploiters + opponent pool | **done** — champion lost **0/80** to a naive dumper; pool training fixed it (E10, E10b) |
| T2.6 livestock gate | **done** — animals were disabled by `goose_min_cash`, not rejected; +$19k of egg/fertilizer revenue recovered (E12) |
| T2.7 cow/sheep + other gated knobs | new — audit every conjunctive knob pair (E12) |
| T2.5 league / dump-only test | **done** — **the ablation won**: reserves + forecast pinned to 0 beats the full policy 84.4%. The dumper exploit collapsed into the champion (E11) |
| **V1 submit + leaderboard** | **THE critical path, blocked on you** — no longer just "the honest measurement"; it now decides whether Phase 3 gets built at all (`PLAN.md` §3.5) |
| V2 independently designed opponents | **done** — 4 briefs in `arena/opponents.py`; champion beats all **64/64** (E14) |
| V3 implement cows/sheep, then test | **done** — both beat the goose champion **48/48**; the capacity model was wrong (E15) |
| V4 re-test dumping vs non-melon opponents | **done** — E11 holds, but its *mechanism* was wrong; we never leave the scarcity regime (E16) |
| V5 headroom (measured) | **done, and it is bad news** — E20: farm is **25/25 full**, herd at target, land dead on a 4th test, 18% idle. The scripted agent is at a structural ceiling |
| V6 pin env version, detect rule changes | **done** — `tests/test_env_version.py` hashes the env source; kagsim mirrors it, so a silent Kaggle rule change would otherwise invalidate every measurement with nothing failing |
| V7 claim audit of `PLAN.md` | open — 2 claims already refuted |
| T3.x model, T4.x inference search | **not started, and deliberately gated on V1** — see `PLAN.md` §3.5 |

### Current champion

**First champion promoted through the gate** (`make promote`, D19) rather than by a search result.

`search/champion.json` — melon + carrot + strawberry, **6 cows + 8 sheep**, **10 hands**, a
0.35x-base reserve on shop-demanded products, near-nearest-task routing, no land.

| gate stage | result |
|---|---|
| 1. beat incumbent, 500 games | **77.2% [73.3, 80.7]**, $80,768 vs $75,832 |
| 2. gauntlet, 74 agents | **no losses** — 72 clear wins, 2 self-play ties |
| 3. neighbourhood sweep, 12 probes | **no neighbour better** (closest: `cow_target=5` at 51.6%) |

Progression: $27.6k -> $30.3k -> $38.0k -> $41.6k -> $67.0k -> $86.3k -> **$80.8k head-to-head**
(absolute money falls as opponents get stronger; the gate measures the margin, not the money).

### Verification status (T0.4)

| | |
|---|---|
| Tests | **171** — 9 cargo + 162 pytest |
| Reference simulation lines differentially covered | **100%** (682/754 executed, 72 excused with written reasons, 0 unexplained) |
| Divergences | **0** across fuzz, engine-driven play, 15 config variants, 8 `marketParams` overrides, full 718-step seasons, and the observation surface for both seats |
| Known behavioural differences | **none** |
| Gate | `make verify` — fails on any unexecuted simulation line or any divergence |

Covered surfaces: canonical state step-by-step, `Sim.observation()` field-by-field for **both**
seats, terminal reward, bit-exact RNG, Python `int()` coercion and raise-behaviour, every
documented config knob including `marketParams`.

### Bugs found by verification, in order of severity

Each was silent — none would have raised, and none were reachable by reasoning alone.

1. **`step` is delivered only to player 0.** A framework field, not declared `shared`, so seat 1
   gets nothing and `obs.get("step", 0)` reads 0 every turn there. kagsim was emitting it for both
   seats — a model would have learned a feature that silently reads 0 in half of all real games.
   *Found by diffing the observation surface directly, which no state-level check could see.*
2. **The terminal step was never executed.** The parity loop stopped one call short of
   `episodeSteps - 2`, leaving the DONE/reward path — the score we train on — untested.
3. **`int()` coercion divergences.** `["BUY_SEED","WHEAT","5"]` bought 5 seeds in the reference and
   0 in kagsim; `3.9` truncated vs. dropped; a bad value crashed the episode vs. silently becoming 1.
4. **`marketParams` silently ignored** — now fully implemented (sparse merge, truthiness-gated
   exposure, resolved table in the observation).
5. **`BUY_ANIMAL` indexed the shed by animal index (0–2) instead of item index (9–11)** — buying a
   cow added a carrot.
6. **Malformed market orders were dropped instead of keeping their slot**, shifting later orders
   forward and desyncing the two players' lockstep pricing.

### Engine bugs found by T0.7 (`docs/experiments.md` E2)

1. **No cash management** — spent to $0 on day 0; invalidated the whole goose experiment.
   Melon went $5,144 → $42,199 after the fix.
2. **Wheat churn loop** — bought feed and immediately sold it back, inflating reported revenue to
   $117,947, past what the wheat market can absorb.
3. **Animals could never be fed** — FEED required a unit to already carry wheat and nothing ever
   fetched any. 11 geese placed day 17, 9 dead by day 18.

**Sequencing change (T0.5).** The rayon-parallel array `VecEnv` is deferred until the policy is a
neural net (T3.1). Rust threads cannot call Python agents, so batched stepping buys nothing for
scripted agents or CEM — those parallelize by *process* instead (`sim.runner.play_many`, ~29
episodes/s across 8 cores). Building the array API now would fix the action encoding before the
action space is settled, which D7 says to derive from the model design.

## Conventions

- **ID** `T<phase>.<n>` — referenced by branch names (`t0.2-rust-rules`) and commit messages.
- **Size** S ≈ half a day, M ≈ 1–2 days, L ≈ 3–5 days.
- Tasks within a phase are ordered; `deps` lists hard prerequisites.
- **Every change that could affect agent strength is accepted or rejected by arena winrate
  (T0.8), never by whether it looks correct.** 7th place had a bundle of "small, sensible-looking
  network fixes" score 39% against its own baseline.

## Machine reality (probed 2026-08-09)

| | |
|---|---|
| Platform | macOS, Apple Silicon **arm64**, 8 cores, 16 GB unified memory |
| Python | `/opt/miniconda3/bin/python` — **3.13.13** (`/usr/bin/python3` has no `kaggle_environments`) |
| Installed | numpy 2.5.1, torch 2.13.0, pyyaml 6.0.3 |
| Missing | rust, maturin, pytest, numba |
| GPU | **MPS only — no CUDA.** `torch.cuda.is_available() == False` |

**Implication:** local machine is good for the simulator, scripted agent, arena, and CEM search —
all CPU-bound. It is *not* good for multi-billion-step RL. 3rd and 6th place both rented from
vast.ai. Budget for that at Phase 3 (T3.x); develop and debug locally on MPS with tiny models
first, then rent. See T3.0.

## Repo layout

```
main.py                 submission entry point — must expose agent(obs)   [not written yet]
Makefile                make sim | test | verify | audit | bench
CLAUDE.md PLAN.md TASKS.md
docs/
  README.md AGENTS.md   competition rules (read-only reference)
  decisions.md          pinned decisions, with reasoning and what would change them
  experiments.md        measured results; supersedes PLAN.md's arithmetic estimates
reference/orbit_war/    past-competition PPO template + OVERVIEW.md
kagsim/                 Rust crate, PyO3 bindings — the fast simulator
  src/{lib,state,rules,market,rng}.rs
agent/{engine,params}.py    scripted economic engine; Params is the CEM search vector
sim/{runner,baselines}.py   episode running, diagnostics, multiprocess fan-out
tests/
  parity.py                 canonical state, field-level diff, action fuzzer, script runner
  test_parity.py            fuzz parity layers A/C + rule scenarios
  test_rules_coverage.py    directed tests for rules play never reaches
  test_observation_parity.py  observation surface, both seats
  test_market_params.py     marketParams override parity
  test_rng.py test_baselines.py
tools/
  coverage_audit.py     THE gate: reference branch coverage under differential test
  audit.py              rule-level event coverage (fuzz + engine-driven)
  bench.py              steps/s, and starter-money reproduction
  experiments.py        T0.7 economics sweep
```

Not yet created: `arena/`, `search/`, `rl/`.

---

# Phase 0 — Fast simulator, arena, ground truth

The gate for everything else. All six published Orbit Wars top-10 solutions rewrote the
environment; ours runs at 860 steps/s and they needed 15k–40k.

## T0.0 — Decision record · S · deps: none

Pin the choices below in `docs/decisions.md` so they are not relitigated.

**Simulator language: Rust + PyO3, built with maturin.** Rationale:

- The game logic is heavily *branchy* (per-tile type dispatch, per-unit action dispatch, the
  market lockstep loop). This vectorizes badly in NumPy — the natural batched-array rewrite would
  be a near-total redesign for maybe 10×, whereas a direct port to a compiled language is close to
  a transcription for 100×+.
- Rust releases the GIL, so `rayon` gives true multicore rollout across the 8 cores. Numba/Cython
  need care here; pure-NumPy batching cannot use the branch-heavy path at all.
- Matches the field: 1st (Rust), 2nd (Rust→C), 5th (C++), 6th (C++). 3rd and 7th used JAX and both
  reported pain — 3rd: "In the future I would probably do this part in Rust or C++… compilation on
  the Kaggle servers just seems risky."
- Rejected: **Numba** (fragile on dict/object state, would need the same array rewrite anyway),
  **Cython** (similar effort, weaker tooling), **JAX** (static shapes fight a variable number of
  hired hands; see 3rd's compile-time problems).

**Fallback if Rust setup blocks for more than a day:** write the array-of-structs redesign in
Python first (T0.2 is language-agnostic in its state design), get correctness nailed with the
parity harness, then port. The parity harness is the durable asset, not the language.

**Scope:** the simulator models *both* farms, the shared market, and the town. It does **not** need
the kaggle-environments wrapper, replay JSON, or the renderer.

**Done when:** `docs/decisions.md` exists with these choices and their rationale.

---

## T0.1 — Toolchain and crate skeleton · S · deps: T0.0

```bash
# Rust (arm64 native)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env" && rustc --version   # expect aarch64-apple-darwin

/opt/miniconda3/bin/pip install maturin pytest pytest-xdist

cd /Users/tu/Desktop/kaggriculture
maturin new --bindings pyo3 kagsim
```

`kagsim/Cargo.toml`:

```toml
[lib]
name = "kagsim"
crate-type = ["cdylib", "rlib"]        # rlib so Rust unit tests can import the crate

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
rayon = "1"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

Build into the miniconda env with `maturin develop --release` (run from `kagsim/`, with that
interpreter active). Add a `Makefile` target `make sim` so this is one command.

**Verify:** `/opt/miniconda3/bin/python -c "import kagsim; print(kagsim.__file__)"` succeeds, and a
trivial exported `add(a,b)` returns the right answer.

**Done when:** `make sim && make test-sim` is green from a clean checkout.

---

## T0.2 — Port the game rules · L · deps: T0.1

Transcribe `kaggriculture.py` (1063 lines) into `kagsim/src/{state,rules,market}.rs`. **Read the
source, not `docs/README.md`** — the docs are accurate but lossy on ordering.

### State layout

Flat structs, no heap allocation per tile. One `Tile` with a `kind` tag covers every case:

```rust
#[derive(Clone, Copy, PartialEq)]
pub struct Tile {
    pub kind: TileKind,          // Empty | Locked | Weed | Plant | Coop | Pasture
    pub crop: Crop,              // valid iff kind == Plant
    pub animal: Animal,          // None until PLACEd; valid iff kind in {Coop, Pasture}
    pub planted_day: i16,        // doubles as placed_day for animals
    pub yield_units: i16,
    pub max_lifespan_step: i32,  // -1 for ongoing crops
    pub fertilized_until_day: i16,
    pub consecutive_unwatered: u8,  // doubles as consecutive_unfed
    pub pending_care_bonus: u8,
    pub flags: u8,               // bit0 watered/fed_today, bit1 cared_today, bit2 fertilizer_available
}

pub struct Farm { money: f64, tiles: [Tile; 100], farmer: (u8, u8),
                  hands: SmallVec<[(u8,u8); 32]>, unlocked: u8 /*bitmask NW,NE,SW,SE*/,
                  hires_today: u16 }
pub struct Private { shed: [i32; N_ITEMS], seeds: [i32; 5], inventories: Vec<[i32; N_ITEMS]> }
pub struct Market { inventory: [i64; 9], prices: [i32; 9] }
```

Keep `money` as `f64` — the reference uses Python floats and `_commit_unit` compares
`farm["money"] < price`; using integers would diverge on the comparison boundary.

### Porting checklist — the details that break parity

Work through these explicitly; each one is a real divergence risk found by reading the source:

1. **`int(round(x))` in `market_price` (`:192`) is Python banker's rounding** — round-half-to-even.
   Rust's `f64::round()` rounds half *away from zero*. `round(2.5)` is `2` in Python, `3` in Rust.
   Implement `round_half_even` and use it everywhere the reference calls `round`.
2. **Turn order in `interpreter` (`:871`)**: all unit actions for *both* players → `_process_market`
   → `_town_consume` → `_decay_plants` → `_end_of_day` if `(step+1) % turns_per_day == 0` → then
   increment step. Production for animals/plants happens inside the *daily* refresh, not per turn.
3. **Atomic PLANT (`:897`)**: `plant_demand` is tallied across farmer + all hands *before* any
   action is applied, and compared against the seed count at that moment. If demand exceeds
   supply, **every** PLANT of that crop that turn becomes PASS.
4. **`_fib` indexing (`:667`)**: `fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5`.
5. **Market lockstep (`:540`)**: iterate order slots `i`; handle HIRE / BUY_LAND atomically in
   player order first; then a per-unit `while` loop where **both players are quoted against the
   same pre-commit inventory** and then both commit. `_refresh_prices` after each slot, not each
   unit. Any failed commit aborts that player's whole order.
6. **SELL at the $1 floor does not add to market inventory (`:636`)**.
7. **BUY_PRODUCT is quoted at `inventory - 1` (`:578`)**, making a buy→sell round trip net zero.
8. **BUY_PRODUCT / BUY_ANIMAL fail if `sum(shed) >= shed_capacity` (`:644`, `:659`)**, aborting the
   order.
9. **`_daily_refresh_animals` (`:783`) ordering**: feed check → escape check → production (consuming
   `pending_care_bonus` only if `fed_today`) → *then* care banking → then reset flags. Care banked
   on day *d* pays out on day *d+1*.
10. **`_daily_refresh_plants` (`:747`)** captures `was_watered` before resetting, and computes
    `days_since_first` against `next_day`, not `day`.
11. **`_decay_plants` (`:730`)** runs every *step*, and only when `(step - max_lifespan_step) % 2 == 0`.
12. **New plants start `consecutive_unwatered = 1`** (`:208`) and one-time crops start
    `yield_units = 1`; ongoing crops start at 0.
13. **Water bonus window** `[(max_yield_day+1)//2, max_yield_day]` inclusive (`:384`), capped at
    `max_yield`.
14. **`_spawn_hand` (`:510`)** picks the shed-access tile with least occupancy, ties broken by NWSE
    index — and ignores whether the tile is locked.
15. **`_end_of_day` (`:838`)** resets farmer to `_default_spawn`, clears hands, zeroes
    `hires_today`, and resets `inventories` to a single empty dict.
16. **Movement onto LOCKED tiles is allowed (`:314`)**; tile ops no-op there.
17. **Episode terminates at `step >= episodeSteps - 2` (`:937`)**, reward = `money`.

### Rust unit tests (`cargo test`)

One test per checklist item above, asserting the specific behaviour in isolation — e.g.
`round_half_even(2.5) == 2`, planting with 1 seed and 2 requesting units leaves the tile empty and
the seed unconsumed, an animal fed on alternating days survives 10 days.

**Done when:** `cargo test` passes with ≥1 test per checklist item, and T0.4 layer A is green.

---

## T0.3 — CPython-compatible MT19937 · M · deps: T0.1

The only stochasticity is `_end_of_day` (`:848`):
`rng = random.Random((seed * 1_000_003) ^ day)`, then per player `_spawn_weeds` calls
`rng.random()` **once per empty unlocked tile** (scanning `y` then `x`), then possibly
`rng.choice(sorted(remaining))` for the shop unlock.

The number of `random()` draws depends on how many tiles are empty, which depends on gameplay — so
the stream **cannot be precomputed**. Bit-exact parity requires reimplementing CPython's RNG.

Implement in `kagsim/src/rng.rs`:

- **MT19937** core (`init_genrand`, `init_by_array`, `genrand_uint32`) — standard, well-specified.
- **Seeding**: CPython converts an int seed via `init_by_array` over its absolute value split into
  32-bit little-endian words. Note `(seed * 1_000_003) ^ day` can exceed 32 bits.
- **`random()`**: `a = genrand()>>5; b = genrand()>>6; (a*67108864.0 + b) * (1.0/9007199254740992.0)`.
- **`getrandbits(k)`** for `k <= 32`: `genrand() >> (32-k)`.
- **`_randbelow(n)`**: `k = n.bit_length(); loop { r = getrandbits(k); if r < n { return r } }`.
- **`choice(seq)`**: `seq[_randbelow(len(seq))]`.

**Verify** with a golden-file test:

```python
# tools/dump_rng_golden.py
import random, json
out = {}
for s in [0, 1, 42, 7, 123456789, (42*1000003)^17]:
    r = random.Random(s)
    out[str(s)] = {"random": [r.random() for _ in range(1000)]}
    r2 = random.Random(s)
    out[str(s)]["choice"] = [r2.choice(list(range(n))) for n in range(2, 200)]
json.dump(out, open("tests/golden_rng.json", "w"))
```

Rust test loads the golden file and asserts **exact** `f64` bit equality on `random()` and exact
integer equality on `choice`.

**Done when:** golden test passes for all seeds, 1000 draws each, bit-exact.

---

## T0.4 — Parity harness · L · deps: T0.2, T0.3

The most important verification asset in the project. 5th place: *"a faster environment is only
useful if it produces exactly the same game transitions and model inputs."*

### Canonical state

`canonical_state(obs_or_sim) -> dict` producing an identical, ordering-stable structure from either
implementation: both farms' money / tiles / farmer / hands / unlocked / hires_today, both privates'
shed / seeds / inventories, market inventory + prices, town shops, day, hour, step. Sort every dict
key; represent tiles as fixed-length tuples so a missing field can't hide.

### Action fuzzer

`tests/fuzz.py` generates action dicts biased to exercise edge paths, **not** only legal ones:

- ~40% legal-ish (move toward a real target, then a valid op for that tile)
- ~30% random op from the full vocabulary at the current position
- ~20% deliberately illegal (act on LOCKED, plant with no seeds, feed with no wheat, dig an
  occupied pasture, PICKUP away from the shed, sell what you don't have)
- ~10% market storms (over-length order lists, `n` larger than holdings, HIRE ×10, BUY_LAND repeatedly)

Seed the fuzzer separately from the env seed so failures are reproducible from two integers.

### Three layers, in order

- **Layer A — deterministic core.** `weedSpawnChance: 0.0`, `townShopUnlockInterval: 9999`. Removes
  all RNG. Run 500 episodes × 720 steps, comparing canonical state after **every** step. This
  validates T0.2 independently of T0.3.
- **Layer B — RNG in isolation.** T0.3's golden test.
- **Layer C — full config.** Default settings, weeds and shop unlocks live. 500 episodes.

### Divergence reporting

On mismatch, print the step number, both canonical states, and a **field-level diff** (first 20
differing paths). Persist the failing `(env_seed, fuzz_seed, step)` into
`tests/regressions/` and add it as a permanent test case. Chasing a divergence 300 steps in without
a field-level diff will waste a day.

### Hand-written scenario tests

Beyond fuzzing, script these explicitly — each targets a rule that fuzzing hits rarely:

| Scenario | Asserts |
|---|---|
| Shed at 100 items, end-of-day drop of 20 | overflow discarded, no bypass via unit inventory |
| 1 wheat seed, farmer + 1 hand both PLANT WHEAT | neither plants, seed not consumed |
| Animal fed days 0,2,4,6,8 | survives; `consecutive_unfed` oscillates 1→0 |
| Animal unfed 2 consecutive days | escapes, structure remains, tile diggable |
| CARE + FEED daily on a goose | `pending_care_bonus` pays out on next production, resets |
| CARE on an alternate-feed goose | bonus destroyed on unfed production day |
| Wheat planted d0, watered d0/2/3/4, harvested d4 | `yield_units == 4` |
| Same, fertilized | `yield_units == 6` (capped) |
| Wheat left to d6 | decay of 1 every 2 steps, then WEED |
| Both players `SELL MELON 200` same turn | identical per-unit prices, floor at $1, inventory unchanged at floor |
| `BUY_PRODUCT WHEAT 1` then `SELL WHEAT 1` | money delta exactly 0 |
| HIRE ×8 in one turn | costs 1+1+2+3+5+8+13+21 = 54 |
| BUY_LAND ×4 | third succeeds at $4000, fourth is a no-op |
| Money below order cost mid-order | order stops, prior units already committed |

**Done when:** `make verify` exits 0 — every simulation-logic line of the reference executed under
differential comparison, with zero divergences. **DONE.**

**Why coverage, not episode count.** A hand-written rule list can only report on behaviours someone
thought to name; 14 passing full episodes and a "34/34 rules covered" report still hid an untested
terminal-reward path. `coverage.py` over the reference is objective: a covered line was
differentially tested, an uncovered one is unverified. See `docs/decisions.md` D4b.

**Maintenance.** Re-run `make verify` after any change to `kagsim/` and after any
`kaggle-environments` upgrade. Excused lines each carry a written reason in
`tools/coverage_audit.py`; adding a new excuse is a decision, not a formality.

---

## T0.5 — Batched vectorized API + benchmark · M · deps: T0.4

Expose a rayon-parallel batch env from `kagsim/src/vecenv.rs`:

```python
env = kagsim.VecEnv(num_envs=1024, config={...}, seeds=np.arange(1024))
env.reset()                      # -> obs arrays
env.step(actions)                # -> (obs, reward, done, info), autoresets finished envs
env.state_snapshot(i)            # -> canonical dict, for debugging and parity
```

Actions cross the boundary as **fixed-shape NumPy arrays**, not Python dicts — dict marshalling per
unit per step would dominate the runtime. Suggested encoding: `unit_ops` `(N_env, MAX_UNITS, 3)`
int16 (op, arg0, arg1) and `market_ops` `(N_env, 10, 3)` int16.

Keep observation encoding in Python (NumPy) for now; move it into Rust only if profiling says so.
3rd place's warning applies — feature-engineering-in-the-fast-language is where the complexity
lands, so defer it until it's measurably the bottleneck.

**Benchmark** (`tools/bench.py`), reporting steps/s for: reference Python env (baseline **860**),
kagsim single-thread, kagsim 8 threads.

**Done when:** ≥**50k steps/s** single-thread (~60×) and ≥**250k steps/s** on 8 threads, *and*
`state_snapshot` still matches the reference on a fresh parity run after all optimizations.

---

## T0.6 — Baseline agents + metrics · S · deps: T0.5

Port `pass`, `random`, `starter` onto the kagsim interface (baselines: `starter` = **$3,496**,
`random` = **$0**). Add a diagnostics recorder:

- money per day; final money
- **actions wasted** — ops that were no-ops (biggest early lever)
- **turns spent moving** vs. acting, per unit
- tiles idle per day; weeds standing per day
- shed overflow **discarded** per day (the silent killer)
- units sold and realized price per product
- shed occupancy at end of day

**Done when:** running `starter` through kagsim reproduces $3,496 exactly on seed 7, and the
diagnostics render as a per-day table.

---

## T0.7 — Economics validation · M · deps: T0.6

**DONE** — results in `docs/experiments.md`. Settled by measurement, not arithmetic:

1. ✅ **Melon rush confirmed, decisively** — $42–45k vs. $12k for wheat, ~3.5×.
2. ✅ **CARE beats alternate-day feeding** — $17,644 vs. $10,013; eggs $9,022 vs. $3,702.
3. ✅ **Goose optimum ≈ 8**; at 16–25 the feed loop cannot keep up and the flock dies.
4. ✅ **Labour optimum ≈ 8 hands**; 12+ bankrupts (fib cost ≈ $376/day at 12, $2,583 at 16).
5. ✅ **Harvest at max yield** — harvesting early costs ~36% of final money.
6. ❌ **"Buy land ASAP" overturned** — not buying land *doubled* money and cut movement 70%→46%.

Still open, and re-scoped:

- **Front-runner exploitability** (`PLAN.md` §2.5) — now unblocked, since a $45k melon producer
  exists to steal from. *This number sizes the entire market-model effort*, so it should run right
  after T0.8 gives it a trustworthy measurement.
- Shed-cap binding at scale; staggered vs. single-wave melon; harvest-early-into-shed as private
  optionality. Reservation pricing already recovers much of the melon timing question.
- Whether solving T1.2 makes land purchase profitable again.

**Caveat carried forward:** the 6-seed sweep cannot rank close configs — `melon+wheat` once showed
a standard deviation larger than its mean. Re-run the ranking through T0.8 with CIs.

---

## T0.8 — Local arena · M · deps: T0.6 — **DONE** (E5)

The only evaluation that exists in a simulation competition, and the current blocker: T0.7's sweep
used 6 seeds, where `melon+wheat` posted a standard deviation *larger than its mean*. Nothing more
can be ranked without this. 2nd place: *"You can't do CV in a
simulation competition but you can build a local arena."*

- `arena/run.py --agents a,b,c --games 512 --seeds fixed` — round-robin, both seat assignments per
  seed pair to cancel first-player advantage, parallel across 8 cores.
- Ratings: **OpenSkill** (what 2nd used to mimic Kaggle matchmaking) plus raw pairwise winrate
  matrices — winrate is what you actually reason about.
- Report Wilson confidence intervals; 512 games gives roughly ±4pp, so **do not act on a 52%**.
- Persist results to `arena/results.sqlite` keyed by agent checkpoint hash.
- HTML report with the pairwise matrix (7th built theirs in ~20 minutes; it caught real bugs).
- **Snapshot the whole agent+feature pipeline per registered checkpoint** so old entrants stay
  evaluable after refactors (3rd's `zoo/` pattern).

**Done when:** `starter` vs `random` vs `pass` produces a sane matrix with CIs, reproducible from a
seed list, in under 2 minutes.

---

# Phase 1 — Scripted economic engine

**v1 is done**: best config reaches **$45,171** vs. `starter`'s $3,495 (target was $20k).
`agent/engine.py` generates tasks from farm state, assigns them to units with sticky greedy
routing, and spends against an explicit budget in ROI order. `agent/params.py` holds every knob —
that dataclass *is* the vector T2.1 optimizes.

Remaining work, in order of measured value:

## T1.2 — Routing rework · L · **DEPRIORITIZED** — see `docs/experiments.md` E6
Measuring the op mix before optimizing changed the task. Units are **idle 38–58% of turns**, not
travel-bound, so better routing cannot help: there is nothing more worth doing. Adding land to
fill the idle capacity *loses* money at every mix, because melon already captures most of the
~$26.5k its market absorbs — the constraint is **market absorption, not labour or travel**.

Also fixed here: the land-purchase gate was **unreachable by construction** (it required 48.6 of
25 unlocked tiles), so `BUY_LAND` never fired and E1's "land is harmful" verdict was measured on a
config that never bought land. With the gate corrected land still loses, now for a understood
reason, and `buy_land` defaults to `False`.

**Revisit only if** a future config makes labour binding again — e.g. a diversified mix with
enough market capacity to use idle turns. The diagnostic to watch is PASS share, not movement share.

## T1.3 — Market / selling policy · M · deps: T0.8 — **DONE** (E7)
The measured binding constraint. `melon` wins **every** arena pairing while earning *less* mean
money than `melon-wheat`: scoring is relative, so **denial beats revenue** — racing into the
high-price part of the melon curve and leaving the opponent the crashed remainder.

Current policy is a static reservation price per product (`reserve_frac`). Extend to use what is
observable: the opponent's tiles give their crop counts and planting days, hence their harvest
dates and incoming supply (`PLAN.md` §2.5). Sell ahead of a predicted wave; hold through it.
**Verify:** arena winrate vs. the current engine over >=128 games. **This is the same mechanism the
front-runner experiment probes, so run that alongside.**

## T1.3b — Engine must honour `marketParams` · S · deps: T1.3
`Engine._sell_orders` calls `kagsim.market_price(idx, inv)` with no params table, so its
reservation pricing is wrong whenever the episode overrides the price curves. kagsim now accepts
`kagsim.market_price(item, inv, params)`; thread `obs["market"].get("params")` through.
**Verify:** a parity-config sweep where an override materially moves a curve, asserting the engine
still prices at the margin correctly. Cheap, and it matters because T3 will train across varied
settings deliberately.

## T1.5 — Submission wrapper · M · deps: T1.4 — **DONE**

`main.py` at repo root exposing `agent(obs)`, loading `search/champion.json`, holding per-episode
engine state across calls.

Traps already measured, each of which would cost real games:

- **`obs["step"]` is safe to read** — it reaches both seats, correct on all 719 turns (E21). The
  earlier rule here said the opposite; it was measured against the stored replay state instead of
  the observation the runner delivers. `day` and `hour` are equally fine.
- **Never touch `__file__` in `main.py`** — Kaggle `exec`s the source with empty globals, so it
  raises `NameError` and the agent never loads (E21). Scored the $3,000 starting bank, 0-40.
- **The engine is stateful per episode** (sticky task assignments). The submission is called once
  per turn for one seat, so state must persist across calls and reset when a new episode starts —
  detect via `day == 0 and hour == 0`, not by object lifetime.
- **`kagsim` will not exist on the Kaggle runner.** `Engine._sell_orders` currently imports it for
  `market_price`. Either vendor a pure-Python `market_price` or bundle the wheel; the former is
  far simpler and the function is ~15 lines.
- **Latency**: `actTimeout` is 1 s/turn on an unpredictable CPU. Measure p99 per-turn, not mean.

**Verify:** `env.run(['main.py', 'starter'])` on the *reference* env reproduces the kagsim result
for the same seed; p99 turn latency well under 1 s; a full game as **seat 1** specifically, since
that is where the `step` trap bites.

**Done when:** a `submission.tar.gz` builds and plays a full 720-step game through the real
environment in both seats with no error and no timeout.

## T1.6 — Weed pressure · S · deps: T1.2
Weeds still accumulate on some configs. The `tiles_per_unit` cap limits *new* planting but does not
protect existing plants from going dry while units are walking. Likely subsumed by T1.2.

# Phase 2 — Black-box parameter search

## T2.1 — CEM / CMA-ES over agent config · M · deps: T1.4, T0.8
~25-dim continuous vector. Fixed seed set (≥200) per evaluation, opponent = current best. Log every
evaluation to sqlite so the search is resumable.
**Verify:** tuned config beats the hand-set default by a margin exceeding the arena CI.

## T2.2 — Exploiters and an opponent pool · M · deps: T2.1 — **DONE** (E10, E10b)

The champion was tuned against exactly one opponent, and the field has already shown that
rankings move under competition (E5) and that a config can be a counter rather than a
dominant strategy (E6). A single-opponent optimum may be brittle.

1. **Build deliberate exploiters** and see whether any beats the champion:
   - *front-runner*: reads the champion's public tiles, predicts its melon harvest day, dumps one
     day earlier. (Our own forecast policy is this — turn it against us.)
   - *denier*: ignores revenue, maximizes damage to the opponent's realized melon price.
   - *turtle*: never crashes the market, sells only into recovered prices.
   - *mirror*: the champion itself, as the control.
2. **If an exploiter wins**, re-run T2.1 with an *opponent pool* (sample the champion, exploiters,
   and past checkpoints) instead of a single opponent — the same league argument the Orbit Wars
   field converged on (`PLAN.md` §2.6: 7th measured training vs. a live copy at **20.7%**).
3. Persist accepted agents to `arena/zoo/` so old entrants stay evaluable after refactors.

**Verify:** arena, >=128 games, on seeds no search has used.

**Done when:** either no exploiter beats the champion (record the evidence), or the pool-trained
champion beats every exploiter.

## T2.5 — League to a fixed point, and the dump-only hypothesis · M · deps: T2.2 — **DONE** (E11)

T2.2 established that a champion tuned against one opponent loses to a strategy outside its search
field, and that pool training fixes *that* exploit — but the exploit regenerates from each new
champion (E10b). Two readings, not yet distinguished, and the work differs sharply between them.

**Reading A — an arms race with no fixed point.** Then the answer is a proper league:

1. Iterate: pool-train a challenger against `[champion, dumper(champion), turtle(champion), …]`.
2. Promote only on an arena win over the whole pool on unseen seeds.
3. Regenerate the exploiters from the new champion and repeat.
4. Stop when a promotion cycle fails to displace the champion, or when the margin stops moving.

Keep every promoted agent in `arena/zoo/` — a champion that only beats the *current* pool while
losing to an older member is cycling, not improving, and only the full history shows it.

**Reading B — dumping is simply near-optimal**, and the reserve/forecast apparatus is
over-engineering that paid off only against opponents sharing its assumptions.

Test it directly, and first, because it is cheap and would simplify the agent enormously: run CEM
with `forecast_weight` and every `reserve_frac` **pinned to 0** (search only the remaining knobs),
then arena it against the pool-trained champion.

**Why this matters beyond the scripted agent.** If B holds, the market model `PLAN.md` §2.5
predicts is valuable is valuable for a *different reason* than assumed — timing entry into the
race, not timing sales — and T3.1's action space should reflect that.

**Verify:** arena, >=128 games, seeds no search has used, exploiters regenerated from whichever
agent is being defended.

**Done when:** either a champion survives a full regenerate-and-challenge cycle, or the dump-only
policy is measured against the champion and the result recorded either way.

## T2.3 — Headroom estimate · S · deps: T2.2

**How much is actually left for a model to win?** `docs/decisions.md` D1 says the model's scope
should shrink if the game turns out to be close to a solved optimization, and the scripted engine
is now far past its Phase 1 target. This decides how much of Phase 3 to build.

- Compute the analytic ceiling: total revenue extractable from every price curve over 30 days,
  given town regeneration — the absolute cap on a single player's earnings.
- Compare with what the champion realizes, split by product.
- Measure the *contested* ceiling: two champions in the same game, which is the real setting.

**Done when:** a number for "champion realizes X% of the achievable" exists, and D1's trigger is
explicitly resolved either way in `docs/decisions.md`.

# Phase 3 — The model

**What Phase 1–2 changed about this phase.** Three findings should shape the model rather than be
rediscovered by it:

0a. **The constraint is production, not market absorption** (E16). The town drains faster than two
   players can supply, so prices *rise* all season and nothing saturates except melon. Every
   capacity table in `PLAN.md` describes a regime the game does not reach. A model should be
   optimizing throughput — actions, tiles, herd size — not sale timing or market depth.
0b. **Rank markets by shop demand, not by price.** Seasonal capacity is
   `one-shot cap + drain/day x days`, and the drain term dominates (E15, D17). Melon has **zero**
   shop demand and cannot regenerate; milk has three. Getting this backwards cost a 2.4x
   improvement and survived nine experiments before being caught. Any observation the model
   receives should carry per-product **drain**, not just price.
1. **The market decides the game — but through production timing, not sale timing.** Selling is
   trivially always-now: a policy that structurally cannot hold inventory beats the fully tuned
   reserve-and-forecast policy **84.4%** (E11). A reservation price is a bet that prices recover,
   and the opponent decides whether they do. **Do not give the model a "when to sell" action.**
   The contested resource is arriving at the market first with more units, which is decided days
   earlier by planting and harvest scheduling.
2. **Denial beats revenue, and speed beats both.** The champion suppresses its opponent to $3,190
   against a $3,000 start while earning less mean money than configs it beats 100% of the time
   (E6, E9). Sharper still: a strategy that simply *sells everything immediately* beat the tuned
   champion **0/80** (E10), and against the pool-trained successor the two finish $152 apart on
   $22,000 while the winrate is 3.8% — a race decided at the margin. A model trained to maximize
   its own money is optimizing the wrong thing; terminal **margin** is the objective, and the
   contested resource is *order in the queue*, not total production.
3. **Search overfits in two independent ways, and seed validation only catches one.** CEM produced
   a +$34,207 held-out result worth $207 in the arena (E8, wrong opponent inherited from a
   default). It then produced a *correctly* seed-validated champion that lost 0/80 to a strategy
   outside its search field (E10). RL will do both at greater expense. **Opponent diversity is a
   separate axis from seed diversity** — a league is not optional, and it is the same conclusion
   the Orbit Wars field reached (§2.6).

**Target ~2–10M parameters** — the sizes that took 2nd (4.3M), 3rd (6.2M), 6th (2.5M), 7th (9M).

## T3.0 — Compute plan · S · deps: T2.1

**First action on any new machine: `make verify`.** Float determinism (`sqrt`/`ln`/`log10` feeding
`market_price`) has only been verified on this Mac; a different libm would shift prices and skew
training silently. It is a two-minute check.

Local MPS is fine for shape-debugging and BC, not for billion-step PPO. Price vast.ai (3rd and 6th
both used it); a single 4090/5090 for ~1 week is the reference-class budget. **Implement cloud
checkpointing of weights + optimizer + opponent pool + win rates before the first long run** —
3rd's rented boxes crashed and this saved them.

## T3.1 — Observation and action encoding · L · deps: T0.5
Entity tokens: one per tile (both farms), 9 product tokens, player + global summary tokens.
Pairwise features as graphormer-style attention biases. **Relative features only** — no absolute
player ids (6th: "a massive gain").
**The action space must carry production timing, not sale timing** (E11):

- what to plant and when, so harvests land *before* the opponent's;
- whether to harvest early at a lower yield to beat a predicted wave;
- the opponent's public tiles as a **race position**, not a price forecast.

The T1.3 supply forecast keeps its value as an input to *planting* decisions — which is why
`forecast_horizon` stays in `Params` even though `forecast_weight` is pinned at 0.

**Start from the scripted engine's macro-actions, not from raw ops.** The engine already has a
working action vocabulary (`Task`: tile + intent) and a 29-knob strategy vector that CEM searched
successfully. The cheapest first model replaces the *daily plan vector* (crop mix, hire count,
reserves, forecast weight) with a learned policy while the executor keeps doing the routing — that
is a 29-dim continuous action space with a known-good baseline to clone from, and it can be
evaluated in the existing arena on day one.

**`obs["step"]` is a legitimate feature.** [CORRECTED E21] It is delivered to both seats with the
correct value on every turn of the episode. The previous text here claimed seat 1 never receives
it; that came from reading `env.state[1].observation` (the stored replay state, which does strip
it) rather than `__get_shared_state(1)` (what the agent is handed, which does not). kagsim was
emitting `step` correctly for both seats and was **changed to be wrong** on the strength of that
reading -- the correction restores the original behaviour.

Action space **semantic, per unit**: `(target tile, intent)` where intent ∈ {tend, harvest,
plant(crop), build, place, clear}; the executor pathfinds. **Never emit `NORTH`.** Separate market
head over per-product sell intents.
**Verify:** round-trip test — encode a state, decode a sampled action, apply it, and confirm the
executor produces exactly the intended op sequence with zero wasted actions.

## T3.2 — Behavior cloning · M · deps: T3.1, T1.4
Generate `(obs, action)` pairs from the tuned scripted engine. Masked cross-entropy over legal
actions only.
**Verify:** the BC policy scores within 15% of the scripted engine's money in the arena. If not, the
action encoding is lossy — fix T3.1, don't paper over it with RL.

## T3.3 — PPO training loop · L · deps: T3.2, T0.8
GAE, single epoch per rollout, KL-targeted adaptive LR, **entropy annealing schedule** (3rd: "by far
the most important knob"). **Terminal reward only — no shaping** (7th measured shaping at 34.6%).
**Opponent pool from day one**, never a live copy (7th measured that at 20.7%): the scripted engine
plus checkpoints sampled across the full run history. Early-terminate decided games.
**Verify:** monotone winrate against a frozen pool; every hyperparameter change judged by T0.8.

## T3.4 — Auxiliary heads · S · deps: T3.3
Predict market prices, own money, and tile states at +1/+3/+7 days; discard at inference. 7th
reported this "helped quite a bit."
**Verify:** arena winrate vs. the same run without aux heads, ≥512 games.

## T3.5 — Experiment discipline · ongoing · deps: T0.8
One change per isolated branch, fixed step budget per experiment, accept/reject by tournament only.
Keep a running table of every experiment and its winrate (7th ran ~200).

---

# Phase 4 — Inference-time search

## T4.1 — Greedy 2-step rollout · M · deps: T3.3, T0.5
6th place got **+30–40 leaderboard points** from this. Sample ~5 own actions, evaluate each against
the opponent's argmax, step once more, pick the best by value head. Drive the opponent's assumed
sell behaviour from their public supply forecast (§2.5).
**Budget:** 1 s/turn + 60 s overage — amortize onto the 30 day-boundary turns.
**Verify:** arena winrate vs. the no-search policy, and a **p99 per-turn latency** measurement that
must sit under the timeout with margin on a deliberately throttled CPU.

## T4.2 — Submission packaging · M · deps: T4.1
Bundle weights + `main.py` into `submission.tar.gz`. Confirm the file-size cap (Orbit Wars was
100 MiB) and measure real inference cost. At ~5M params quantization should be unnecessary — if it
isn't, 1st place's int8-for-speed / 4-bit-NF-for-size split is the reference.
**Verify:** end-to-end `kaggle competitions submit` dry run, and a full local game under a
CPU-throttled environment with no timeout.

---

## Next: a verification programme, not more optimization

The champion is strong against agents I wrote. That is the entire body of evidence, and it is
circular. Everything below is ordered by **how much unverified belief it removes**, not by expected
score. No task here is complete on a reasoned argument; each names the measurement that closes it.

### V0 — Promotion is gated · **DONE**, and now mandatory

`tools/promote.py` (`make promote`). Five consecutive promotions installed an agent a larger
measurement later showed was not the best available, always for the same reason: **promoting on a
3-8pp difference from a 24-64 game sample, which can only resolve 12pp+.**

Three stages on fresh seeds, escalating only where a result is close:

1. beat the incumbent over 500 games with the interval clear of 50%;
2. survive the full ~65-agent gauntlet with no resolved loss;
3. survive a neighbourhood sweep of the discrete knobs.

`make audit-champion` runs stages 2-3 against the sitting champion. Its first run **refused the
sitting champion** — it loses to `r-regen` (41.0% [36.8, 45.4]) and to its own `hire_max + 1`
neighbour (43.6% [39.3, 48.0]). Both were invisible to every earlier test. See D19.

**Rule: `search/champion.json` changes only through the gate.** A CEM result is a candidate, never
a promotion — its held-out score has the same sample-size problem *and* only ever sees the pool it
trained against.

### V1 — Submit and read the leaderboard · **blocked on you**
The only non-self-referential measurement available. Needs: joining the competition in the browser,
then `kaggle competitions submit`. `submission.tar.gz` builds and passes its smoke test today.
**Closes:** "is our strategy any good?" — currently pure assumption.

### V2 — Independently designed opponents · M — **DONE** (E14)
Every current exploiter is *the champion with two knobs changed*, so the arena samples one strategy
family. Write 3–4 agents from **different design premises**, not variants:
a goose-economy agent that treats crops as feed; a land-expansion agent with many hands; an
ongoing-crop agent (tomato/strawberry, no melon); a shop-reactive agent that plants to whatever the
town currently demands.
**Rule: written from the design brief, never by mutating champion params.**
**Closes:** "no known exploit" — currently means "no exploit I thought to build" (E10 showed a
two-line change beating a fully tuned champion 0/80).

### V3 — Implement cows and sheep, then test · M
`BUILD_PASTURE` appears nowhere in the engine, so the pasture line has never been *possible*. It was
never rejected on evidence. E1's prior says their markets floor at ~76 and ~59 units — that prior
has now been wrong about geese twice (E1, E12).
**Closes:** an untested third of the animal mechanics.

### V4 — Re-test the dumping conclusion · S — **more urgent after E15**
E11 concluded "sell immediately, never reserve" from a field where every agent produced melon —
a market with **zero shop demand**, where holding genuinely cannot pay because nothing replenishes
the price. The champion is now cow-led, selling into milk, which regenerates at ~19/day. **A
reservation price may be correct for a regenerating product and wrong for melon**, and the current
policy pins reserves to zero for everything.
**Measure:** re-run the dump-only ablation with reserves free for high-drain products only.
E11 ("selling immediately beats reserving") was measured **only against opponents who also produce
melon**. Against an opponent who ignores melon, a reservation price may be correct — and E11 is
currently a promoted default. Depends on V2.
**Closes:** a conditional result that is being treated as general.

### V5 — Headroom, measured not computed · M  *(was T2.3)*
**[MEASURED E13]** already: melon is at **82%** of its computed cap, eggs at **8%**. Extend to the
contested case (two champions, same game) and per-product.
**Closes:** D1's live trigger, and decides how much of Phase 3 to build.

### V6 — Pin the environment and detect rule changes · S
`kaggle-environments` is unpinned; an upgrade can change the rules under us. Record the version,
assert it in CI, and make `make verify` state which version it validated against.
**Closes:** silent divergence between the training simulator and the scoring environment.

### V7 — Claim audit · S
Walk `PLAN.md` and tag every claim **[MEASURED Ex] / [COMPUTED] / [ASSUMED] / [REFUTED Ex]**.
Two claims have already fallen this way (strawberry, and land). Anything left **[ASSUMED]** after
the pass is either scheduled for measurement or deleted.
**Closes:** the gap between what we believe and what we have shown.

Phase 3 (the model) starts after V1, V2 and V5 — its scope depends on all three.

## Standing rules

Every one of these was written after it cost something.

- **`make verify` after every `kagsim/` change and every `kaggle-environments` upgrade.** Two of
  the worst bugs were found by widening what gets compared, not by running more episodes.
- **`make arena` decides**, never inspection, and never mean money — E6 had the arena winner
  earning *less* than configs it beat 100% of the time. Do not act on a winrate whose interval
  includes 50%.
- **Name the opponent and the seeds; never inherit either.** Twice a dataclass default silently
  defined an experiment: `buy_land=True` made every config buy land (E6), and `Params()` being
  wheat-based made CEM optimize against a weak agent for a +$34k result worth $207 (E8).
- **A search or training result is a hypothesis** until it wins on seeds the search never saw.
- **Vary the opponent, not just the seeds.** Held-out seeds cannot detect a single-opponent
  optimum: the T2.1 champion was correctly seed-validated and lost **0/80** to a naive dumper it
  had never faced (E10). Search against a pool; keep the exploiters in the arena permanently.
- **Regenerate exploiters after every promotion.** They are defined relative to the champion, so a
  stale exploiter tests the previous agent's weakness, not the current one's.
- **Never shorten episodes to buy throughput.** Melon strategies are cash-negative until ~day 12,
  so the fitness inverts on truncated games (`tests/test_search.py`).
- **Re-run `make verify` on any new machine** before a long run — float determinism is verified on
  this Mac only.
- **Measure before optimizing.** T1.2 was scoped as a routing rework; the op-mix measurement showed
  units were idle 38–58% and the whole task was wrong.
