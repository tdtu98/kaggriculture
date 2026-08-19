# Top-1 Replay Inspection and Agent Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a robust top-1 production route in `02_inspect_top1`, qualify it against `01_baseline3k` with paired-seat held-out statistics, and preserve complete replay-analysis and benchmark evidence.

**Architecture:** A deterministic offline inspector validates the supplied replays, shifts replay actions to their originating observations, extracts strategy evidence, clusters supported branches, and selects a coherent medoid route. A standalone single-file agent embeds that route and layers isolated live guards, purchase recovery, inventory-aware selling, `01_baseline3k` weed repair, and premium-sale front-running. The benchmark runner gains paired-seed rows and a deterministic bootstrap interval so development and promotion use the seed pair—not individual games—as the statistical unit.

**Tech Stack:** Python 3.12 standard library (`argparse`, `base64`, `copy`, `csv`, `hashlib`, `importlib`, `json`, `random`, `statistics`, `unittest`, `zlib`) and `kaggle-environments==1.32.6`.

## Global Constraints

- Do not modify `duy/another_work/01_baseline3k/main.py`; it is the immutable champion reference.
- The final `duy/another_work/02_inspect_top1/main.py` must import and run without replay files or repository-local modules.
- Treat the public replay seeds only as discovery data.
- Use both seat orders for every benchmark seed.
- Use the two-seat average candidate margin as the primary observation for statistics.
- Development uses seeds `0..19` for screening and `0..49` for the complete finalist.
- Frozen confirmation uses seeds `1000..1099` against `01_baseline3k` exactly once per frozen candidate.
- Robustness uses seeds `2000..2049` against `00_baseline`.
- Promotion requires positive paired mean and median margins, more than 55% game wins, positive mean margins in both seats, no errors, and a 10,000-resample 95% bootstrap interval with lower bound greater than zero.
- Preserve the untracked `duy_explore/` replay directory; never stage or modify the downloaded replay JSON files.

---

## File Structure

- Modify `duy/benchmarks/benchmark.py`: paired-seed rows, deterministic bootstrap interval, artifacts, and formatted summary.
- Modify `duy/benchmarks/test_benchmark.py`: paired-statistics and artifact tests.
- Modify `duy/benchmarks/README.md`: paired interpretation and promotion examples.
- Create `duy/another_work/02_inspect_top1/inspect_replays.py`: validation, extraction, clustering, medoid selection, and CLI outputs.
- Create `duy/another_work/02_inspect_top1/replay_analysis.json`: deterministic replay evidence and selected route metadata.
- Create `duy/another_work/02_inspect_top1/canonical_route.json`: selected shifted action table plus replay-specific weed annotations.
- Create `duy/another_work/02_inspect_top1/main.py`: standalone candidate.
- Create `duy/another_work/02_inspect_top1/evaluate_variants.py`: development-only ablation runner with feature flags in metadata.
- Create `duy/another_work/02_inspect_top1/STRATEGY.md`: human-readable replay findings and retained lessons.
- Create `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`: ablations, frozen hashes, qualification metrics, and links.
- Create `duy/test_inspect_top1_replays.py`: inspector tests.
- Create `duy/test_inspect_top1_agent.py`: agent helper, standalone, state, and short integration tests.

---

### Task 1: Paired-seed benchmark statistics

**Files:**
- Modify: `duy/benchmarks/benchmark.py`
- Modify: `duy/benchmarks/test_benchmark.py`
- Modify: `duy/benchmarks/README.md`

**Interfaces:**
- Produces: `build_paired_rows(results: list[dict]) -> list[dict]`.
- Produces: `bootstrap_mean_ci(values: list[float], *, resamples: int = 10000, seed: int = 20260814) -> dict`.
- `summarize()` adds `paired_seeds` with `count`, `margin`, and `bootstrap_mean_95ci`.
- `write_artifacts()` adds `paired_seeds.csv` with `seed`, both seat margins, and paired margin.

- [ ] **Step 1: Write failing paired-row and bootstrap tests**

```python
class PairedSummaryTests(unittest.TestCase):
    def test_builds_one_row_per_complete_seed_pair(self):
        results = [
            {"seed": 7, "agent_a_seat": 0, "margin": 100.0},
            {"seed": 7, "agent_a_seat": 1, "margin": 300.0},
            {"seed": 8, "agent_a_seat": 0, "margin": -50.0},
            {"seed": 8, "agent_a_seat": 1, "margin": 150.0},
        ]
        self.assertEqual(
            benchmark.build_paired_rows(results),
            [
                {"seed": 7, "seat_0_margin": 100.0,
                 "seat_1_margin": 300.0, "paired_margin": 200.0},
                {"seed": 8, "seat_0_margin": -50.0,
                 "seat_1_margin": 150.0, "paired_margin": 50.0},
            ],
        )

    def test_rejects_missing_or_duplicate_seats(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "complete seat pair"):
            benchmark.build_paired_rows([
                {"seed": 7, "agent_a_seat": 0, "margin": 1.0}
            ])

    def test_constant_bootstrap_interval_is_exact(self):
        self.assertEqual(
            benchmark.bootstrap_mean_ci([25.0, 25.0], resamples=100, seed=9),
            {"confidence": 0.95, "lower": 25.0, "upper": 25.0,
             "resamples": 100, "seed": 9},
        )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd duy && .venv/bin/python -m unittest -v benchmarks.test_benchmark.PairedSummaryTests`

Expected: failures because `build_paired_rows` and `bootstrap_mean_ci` do not exist.

- [ ] **Step 3: Implement deterministic paired statistics**

```python
PAIRED_CSV_FIELDS = ("seed", "seat_0_margin", "seat_1_margin", "paired_margin")


def build_paired_rows(results):
    grouped = {}
    for result in results:
        seed = int(result["seed"])
        seat = int(result["agent_a_seat"])
        seats = grouped.setdefault(seed, {})
        if seat in seats:
            raise BenchmarkError(f"seed {seed} has duplicate agent-A seat {seat}")
        seats[seat] = float(result["margin"])
    rows = []
    for seed in sorted(grouped):
        if set(grouped[seed]) != {0, 1}:
            raise BenchmarkError(f"seed {seed} does not have a complete seat pair")
        seat_zero, seat_one = grouped[seed][0], grouped[seed][1]
        rows.append({
            "seed": seed,
            "seat_0_margin": seat_zero,
            "seat_1_margin": seat_one,
            "paired_margin": (seat_zero + seat_one) / 2,
        })
    return rows


def bootstrap_mean_ci(values, *, resamples=10000, seed=20260814):
    if not values:
        raise BenchmarkError("cannot bootstrap empty values")
    if resamples <= 0:
        raise BenchmarkError("bootstrap resamples must be positive")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.mean(rng.choices(values, k=count))
        for _ in range(resamples)
    )
    return {
        "confidence": 0.95,
        "lower": means[int(0.025 * (resamples - 1))],
        "upper": means[int(0.975 * (resamples - 1))],
        "resamples": resamples,
        "seed": seed,
    }
```

Update `summarize`, `format_summary`, `write_artifacts`, and metadata protocol version so every normal paired benchmark includes these fields.

Use this exact summary shape:

```python
paired_rows = build_paired_rows(results)
paired_values = [row["paired_margin"] for row in paired_rows]
summary["paired_seeds"] = {
    "count": len(paired_rows),
    "margin": _stats(paired_values),
    "bootstrap_mean_95ci": bootstrap_mean_ci(paired_values),
}
```

`write_artifacts` writes `paired_rows` through a `csv.DictWriter` configured
with `PAIRED_CSV_FIELDS`, and `format_summary` prints the paired mean, median,
and confidence bounds. Set `protocol_version` to `2` and add bootstrap seed,
resample count, and confidence level to metadata.

- [ ] **Step 4: Run benchmark tests and verify GREEN**

Run: `cd duy && .venv/bin/python -m unittest -v benchmarks.test_benchmark`

Expected: all benchmark tests pass and the artifact test finds `paired_seeds.csv`.

- [ ] **Step 5: Commit paired benchmark support**

```bash
git add duy/benchmarks/benchmark.py duy/benchmarks/test_benchmark.py duy/benchmarks/README.md
git commit -m "feat: add paired benchmark confidence metrics"
```

---

### Task 2: Replay validation and action alignment

**Files:**
- Create: `duy/another_work/02_inspect_top1/inspect_replays.py`
- Create: `duy/test_inspect_top1_replays.py`

**Interfaces:**
- Produces: `ReplayError`.
- Produces: `load_replay(path: Path) -> dict`.
- Produces: `find_seat(replay: dict, team_name: str, self_seat: int | None = None) -> int`.
- Produces: `shifted_actions(replay: dict, seat: int) -> list[dict]` with exactly 720 actions; the last is aligned `PASS` padding.

- [ ] **Step 1: Write synthetic failing validation and shift tests**

```python
def load_inspector():
    path = Path("another_work/02_inspect_top1/inspect_replays.py")
    spec = importlib.util.spec_from_file_location("inspect_top1_replays", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_replay(names=None):
    names = names or ["top", "opponent"]
    player = {
        "action": {"farmer": ["PASS"], "hands": [], "market": []},
        "observation": {"farms": [{"hands": []}, {"hands": []}]},
        "reward": 0,
        "status": "ACTIVE",
    }
    return {
        "configuration": {
            "startingMoney": 3000,
            "episodeSteps": 720,
            "turnsPerDay": 24,
            "townCenterSellInterval": 24,
        },
        "info": {"TeamNames": names, "seed": 7},
        "rewards": [0, 0],
        "steps": [copy.deepcopy([player, player]) for _ in range(720)],
    }


def test_shifted_actions_discards_initial_placeholder(self):
    inspector = load_inspector()
    replay = fake_replay()
    replay["steps"][1][0]["action"] = {
        "farmer": ["BUILD_PASTURE"], "hands": [], "market": [["HIRE"]]
    }
    actions = inspector.shifted_actions(replay, 0)
    self.assertEqual(actions[0]["farmer"], ["BUILD_PASTURE"])
    self.assertEqual(actions[0]["market"], [["HIRE"]])
    self.assertEqual(len(actions), 720)


def test_self_play_requires_explicit_seat(self):
    inspector = load_inspector()
    replay = fake_replay(names=["top", "top"])
    with self.assertRaisesRegex(inspector.ReplayError, "self-play"):
        inspector.find_seat(replay, "top")
    self.assertEqual(inspector.find_seat(replay, "top", self_seat=1), 1)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_replays.ReplayValidationTests`

Expected: import failure because the inspector does not exist.

- [ ] **Step 3: Implement strict validation and shifted actions**

Use `EXPECTED_CONFIGURATION = {"startingMoney": 3000, "episodeSteps": 720, "turnsPerDay": 24, "townCenterSellInterval": 24}`. Validate every state and action before returning. `shifted_actions` deep-copies states `1..719` and appends `{"farmer": ["PASS"], "hands": [], "market": []}`.

- [ ] **Step 4: Run focused tests and one real-replay check**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_replays.ReplayValidationTests`

Run: `cd duy && .venv/bin/python another_work/02_inspect_top1/inspect_replays.py --team-name カワシギ --self-seat 0 --analysis-output /tmp/top1-analysis.json --route-output /tmp/top1-route.json ../duy_explore/top1_14_Aug/*.json`

Expected: tests pass and the CLI validates 13 replays without modifying them.

- [ ] **Step 5: Commit replay loading**

```bash
git add duy/another_work/02_inspect_top1/inspect_replays.py duy/test_inspect_top1_replays.py
git commit -m "feat: validate and align top1 replays"
```

---

### Task 3: Strategy extraction, clustering, and medoid route

**Files:**
- Modify: `duy/another_work/02_inspect_top1/inspect_replays.py`
- Modify: `duy/test_inspect_top1_replays.py`
- Create: `duy/another_work/02_inspect_top1/replay_analysis.json`
- Create: `duy/another_work/02_inspect_top1/canonical_route.json`
- Create: `duy/another_work/02_inspect_top1/STRATEGY.md`

**Interfaces:**
- Produces: `inspect_replay(replay, seat, source_name) -> dict`.
- Produces: `branch_key(evidence) -> str` formatted as `c{cows}-s{sheep}-straw{strawberry}-melon{melon}`.
- Produces: `actor_disagreement(left, right) -> int`.
- Produces: `select_medoid(records) -> dict`, breaking ties by source filename.
- Produces: `build_outputs(paths, team_name, self_seat) -> tuple[dict, dict]`.

- [ ] **Step 1: Write failing extraction and deterministic medoid tests**

```python
def test_branch_key_uses_route_defining_purchases(self):
    inspector = load_inspector()
    evidence = {
        "purchases": {"COW": 10, "SHEEP": 4,
                      "STRAWBERRY_SEED": 34, "MELON_SEED": 20}
    }
    self.assertEqual(
        inspector.branch_key(evidence), "c10-s4-straw34-melon20"
    )


def test_medoid_minimizes_total_disagreement_and_breaks_ties_by_name(self):
    inspector = load_inspector()
    records = [
        {"source": "b.json", "comparison_actions": [["NORTH"], ["PASS"]]},
        {"source": "a.json", "comparison_actions": [["NORTH"], ["PASS"]]},
        {"source": "c.json", "comparison_actions": [["SOUTH"], ["PASS"]]},
    ]
    self.assertEqual(inspector.select_medoid(records)["source"], "a.json")
```

Add a real-set regression asserting 13 inputs, 12 public games, a 277-hire
default route, two land buys, a selected 10-cow/4-sheep branch, and a stable
selected-source filename. Break equal-size branch ties lexicographically by
`branch_key` before medoid selection.

- [ ] **Step 2: Run tests and verify RED**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_replays`

Expected: missing extraction functions fail.

- [ ] **Step 3: Implement evidence extraction and medoid selection**

Field comparison uses each scheduled actor's operation and arguments but ignores market quantities. Mark a `DIG` as `weed_only` only when the same actor's preceding observation locates it on a `WEED`. Replace weed-only operations with a comparison sentinel so random repair does not dominate medoid distance. Keep the untouched action in the canonical route and write annotations as `{step: ["farmer", "hand:0", ...]}`.

- [ ] **Step 4: Generate deterministic checked-in artifacts**

Run:

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/inspect_replays.py \
  --team-name カワシギ --self-seat 0 \
  --analysis-output another_work/02_inspect_top1/replay_analysis.json \
  --route-output another_work/02_inspect_top1/canonical_route.json \
  ../duy_explore/top1_14_Aug/*.json
```

Run it a second time to `/tmp` and use `cmp` against both checked-in outputs. Expected: byte-identical JSON with sorted keys and a trailing newline.

- [ ] **Step 5: Write `STRATEGY.md` from the generated evidence**

Document the opening, expansion timing, labor curve, livestock/crop branches, action-efficiency difference, market cadence, replay limitations, and the 9/12 matched-seed diagnostic. Clearly label submitted market quantities as requests rather than realized sales.

- [ ] **Step 6: Run tests and commit extraction artifacts**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_replays`

```bash
git add duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/another_work/02_inspect_top1/replay_analysis.json \
  duy/another_work/02_inspect_top1/canonical_route.json \
  duy/another_work/02_inspect_top1/STRATEGY.md \
  duy/test_inspect_top1_replays.py
git commit -m "feat: reconstruct top1 production route"
```

---

### Task 4: Standalone route agent and field safety

**Files:**
- Create: `duy/another_work/02_inspect_top1/main.py`
- Create: `duy/test_inspect_top1_agent.py`

**Interfaces:**
- Produces public `agent(obs) -> dict`.
- Produces internal `_copy_action`, `_align_hands`, `_tile_at`, `_actor_positions`, `_guard_field_actions`, `_weed_repair_action`, and `_reset_if_needed` helpers.
- Exposes `_ENABLE_FIELD_GUARDS`, `_ENABLE_PURCHASE_RECOVERY`, `_ENABLE_SALE_CAP`, and `_ENABLE_FRONT_RUN` booleans for development ablation.
- Embeds `_ROUTE` and `_WEED_ONLY` by base85-encoding one compressed JSON payload from `canonical_route.json`.

- [ ] **Step 1: Write failing helper and standalone tests**

```python
def load_agent_module():
    path = Path("another_work/02_inspect_top1/main.py")
    spec = importlib.util.spec_from_file_location("inspect_top1_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_observation(step=0, hands=None, farmer=None, tile=None,
                     shed=None, money=3000):
    hands = hands or []
    farmer = farmer or [4, 4]
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[farmer[1]][farmer[0]] = tile
    farms = [
        {"money": money, "farmer": farmer, "hands": hands, "tiles": tiles},
        {"money": 3000, "farmer": [4, 4], "hands": [], "tiles": tiles},
    ]
    return {
        "player": 0,
        "step": step,
        "farms": farms,
        "private": {"shed": shed or {}, "seeds": {},
                    "inventories": [{} for _ in range(1 + len(hands))]},
        "market": {"prices": {}},
        "town": {"unlocked_shops": []},
    }


module = load_agent_module()


class AgentShapeTests(unittest.TestCase):
    def test_aligns_hands_to_live_count(self):
        obs = make_observation(step=5, hands=[[4, 4], [3, 4]])
        action = {"farmer": ["PASS"], "hands": [["NORTH"]], "market": []}
        self.assertEqual(
            module._align_hands(action, obs)["hands"],
            [["NORTH"], ["PASS"]],
        )

    def test_suppresses_replay_specific_dig_when_tile_is_not_a_weed(self):
        obs = make_observation(step=10, farmer=[1, 1], tile=None)
        action = {"farmer": ["DIG"], "hands": [], "market": []}
        guarded = module._guard_field_actions(
            obs, action, step=10, weed_only={"farmer"}
        )
        self.assertEqual(guarded["farmer"], ["PASS"])

    def test_exception_fallback_passes_every_live_hand(self):
        original = module._route_action
        module._route_action = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            result = module.agent(make_observation(hands=[[1, 1], [2, 2]]))
        finally:
            module._route_action = original
        self.assertEqual(result["hands"], [["PASS"], ["PASS"]])
```

Add an AST/import test permitting only standard-library imports and prove that renaming `canonical_route.json` does not affect import or action generation.

- [ ] **Step 2: Run agent tests and verify RED**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_agent.AgentShapeTests`

Expected: import failure because `main.py` does not exist.

- [ ] **Step 3: Encode and embed the canonical payload**

Use a deterministic encoder equivalent to:

```python
payload = json.dumps(route_payload, sort_keys=True, separators=(",", ":")).encode()
encoded = base64.b85encode(zlib.compress(payload, level=9)).decode()
```

Insert the encoded literal into `main.py`; runtime decoding must produce 720 route actions and weed annotations without filesystem access.

- [ ] **Step 4: Implement route state, alignment, guards, and baseline weed repair**

Copy the narrow transaction semantics from `01_baseline3k` without changing that file. A guarded invalid maintenance action becomes `PASS`. A weed-blocked `PLANT` or `BUILD_PASTURE` becomes `DIG`, retries the intended action next turn, and replays at most eight delayed actor actions.

- [ ] **Step 5: Run focused and full-game smoke tests**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_agent`

Run a three-seed, both-seat smoke against `01_baseline3k`; require six `DONE` games and deterministic identical reruns.

- [ ] **Step 6: Commit the standalone route agent**

```bash
git add duy/another_work/02_inspect_top1/main.py duy/test_inspect_top1_agent.py
git commit -m "feat: add standalone top1 route agent"
```

---

### Task 5: Market recovery, sale capping, and premium front-running

**Files:**
- Modify: `duy/another_work/02_inspect_top1/main.py`
- Modify: `duy/test_inspect_top1_agent.py`

**Interfaces:**
- Produces: `_same_turn_deposits`, `_pickup_reserves`, `_cap_sales`, `_purchase_state`, `_recover_purchases`, `_front_run`, and `_repay`.
- Market priority is feed wheat, animals, hires, route-defining seeds, land, then sales.
- All functions return copied action dictionaries and preserve at most ten market orders.

- [ ] **Step 1: Write failing sale and recovery tests**

```python
def test_sale_cap_counts_shed_and_same_turn_place_but_reserves_pickups():
    obs = make_observation(shed={"MILK": 7})
    action = {
        "farmer": ["PICKUP", "MILK", 2],
        "hands": [["PLACE", "MILK", 3]],
        "market": [["SELL", "MILK", 20]],
    }
    self.assertEqual(module._cap_sales(action, obs)["market"],
                     [["SELL", "MILK", 8]])


def test_recovery_never_exceeds_cumulative_cow_target():
    state = {"purchased": {"COW": 9}, "pending": {"COW": 2}}
    targets = {"COW": 10}
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    recovered = module._recover_purchases(
        action, make_observation(money=100000), state, targets
    )
    self.assertEqual(recovered["market"], [["BUY_ANIMAL", "COW", 1]])
```

Also test ten-order capacity, feed priority, episode reset, front-run stock reservation, and exact next-turn debt subtraction.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_agent.MarketControllerTests`

Expected: missing helper failures.

- [ ] **Step 3: Implement minimal controllers**

Use cumulative targets precomputed from `_ROUTE`. Treat a scheduled purchase as fulfilled only when the next live observation shows the corresponding cumulative farm/private change. If evidence is unavailable, keep one bounded pending retry and clear it when the branch target is visible. Never add more than the remaining target or more orders than the current market capacity.

- [ ] **Step 4: Port and adapt premium front-running**

Retain the four premium items and exact debt repayment from `01_baseline3k`. Run sale capping before front-running, reserve scheduled pickups, and run final market-length validation after front-running.

- [ ] **Step 5: Run all agent and replay tests**

Run: `cd duy && .venv/bin/python -m unittest -v test_inspect_top1_agent test_inspect_top1_replays benchmarks.test_benchmark`

- [ ] **Step 6: Commit live controllers**

```bash
git add duy/another_work/02_inspect_top1/main.py duy/test_inspect_top1_agent.py
git commit -m "feat: add live recovery to top1 route"
```

---

### Task 6: Development ablation and candidate freeze

**Files:**
- Create: `duy/another_work/02_inspect_top1/evaluate_variants.py`
- Create: `duy/another_work/02_inspect_top1/ablations.json`
- Modify: `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`

**Interfaces:**
- `evaluate_variants.py` accepts `--candidate`, `--opponent`, `--seed-start`, `--seed-count`, `--output`, and repeated `--variant NAME:key=true,key=false` arguments.
- Each match freshly imports the candidate, applies the named feature booleans, exposes a one-argument `agent(obs)` wrapper, and records source hash plus exact flags.
- Output JSON contains per-game rows, paired rows, summary, and the immediately preceding variant comparison.

- [ ] **Step 1: Write a failing loader test inside `test_inspect_top1_agent.py`**

Assert that every variant callable has signature `(obs)`, a fresh module per game, exact recorded flags, and no state leakage.

- [ ] **Step 2: Implement the variant loader and JSON writer**

Reuse `benchmarks.benchmark.build_schedule`, result normalization, paired rows, and bootstrap helper. Do not use a defaulted second function argument; Kaggle interprets two declared parameters as observation plus configuration.

- [ ] **Step 3: Run the six-game deterministic smoke**

Use seeds `0..2`, both seats, for route-only and full variants. Rerun and compare rows byte-for-byte after excluding timestamps.

- [ ] **Step 4: Run screening ablations on seeds `0..19`**

Evaluate in order:

1. `route_only`: all live controller flags false.
2. `field_guards`: only field guards true.
3. `purchase_recovery`: field guards and recovery true.
4. `sale_cap`: previous flags plus sale capping.
5. `full`: previous flags plus premium front-running.

Advance a controller only with no errors and positive paired mean margin over the preceding variant. Remove rejected controller code or leave its final constant false only when tests require the isolated implementation.

- [ ] **Step 5: Run the surviving complete candidate on seeds `0..49`**

Require no errors and positive paired mean margin against `01_baseline3k`. Freeze `main.py` after this run and record its SHA-256 in `BENCHMARK_FINDINGS.md`.

- [ ] **Step 6: Commit the frozen development candidate**

```bash
git add duy/another_work/02_inspect_top1/evaluate_variants.py \
  duy/another_work/02_inspect_top1/ablations.json \
  duy/another_work/02_inspect_top1/main.py \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/test_inspect_top1_agent.py
git commit -m "perf: freeze top1 replay candidate"
```

---

### Task 7: Held-out confirmation and robustness panels

**Files:**
- Generate: `duy/benchmarks/results/<timestamp>_02_inspect_top1_vs_01_baseline3k/`
- Generate: `duy/benchmarks/results/<timestamp>_02_inspect_top1_vs_00_baseline/`
- Modify: `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`
- Modify: `duy/SUMMARY.md`

**Interfaces:**
- Consumes the frozen candidate hash from Task 6.
- Produces standard `games.csv`, `paired_seeds.csv`, `summary.json`, and `summary.txt` artifacts.

- [ ] **Step 1: Verify the frozen candidate hash and clean test state**

Run: `shasum -a 256 duy/another_work/02_inspect_top1/main.py`

Run: `cd duy && .venv/bin/python -m unittest -v benchmarks.test_benchmark test_inspect_top1_replays test_inspect_top1_agent test_demo_agent test_observer_agent`

- [ ] **Step 2: Run confirmation exactly once on seeds `1000..1099`**

```bash
cd duy
.venv/bin/python -m benchmarks.benchmark \
  another_work/02_inspect_top1/main.py \
  another_work/01_baseline3k/main.py \
  --seed-start 1000 --seed-count 100
```

Do not modify `main.py` after reading these results. Check every promotion threshold directly from `summary.json`.

- [ ] **Step 3: Run robustness on seeds `2000..2049`**

```bash
cd duy
.venv/bin/python -m benchmarks.benchmark \
  another_work/02_inspect_top1/main.py \
  another_work/00_baseline/main.py \
  --seed-start 2000 --seed-count 50
```

- [ ] **Step 4: Document the qualification decision**

Record hashes, exact result directories, paired mean/median, CI, win rate, seat means, minimum/maximum margins, status checks, and whether every threshold passed. Update `duy/SUMMARY.md` only if `02_inspect_top1` qualifies; otherwise keep `01_baseline3k` as champion and state why.

- [ ] **Step 5: Commit benchmark evidence**

Stage only the two new result directories, findings, and summary.

```bash
git add duy/benchmarks/results \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/SUMMARY.md
git commit -m "bench: qualify top1 replay candidate"
```

---

### Task 8: Final verification and handoff

**Files:**
- Verify all files changed in Tasks 1–7.

**Interfaces:**
- Produces no new behavior; proves the delivered tree matches the design and persisted evidence.

- [ ] **Step 1: Run the complete relevant test suite**

Run: `cd duy && .venv/bin/python -m unittest -v benchmarks.test_benchmark test_inspect_top1_replays test_inspect_top1_agent test_demo_agent test_observer_agent`

- [ ] **Step 2: Run static and artifact checks**

Run: `git diff --check HEAD~1..HEAD`

Run the inspector twice to temporary outputs and compare them with checked-in analysis and route files. Import `main.py` from a temporary directory containing no replay files and call it for both seats.

- [ ] **Step 3: Inspect repository state**

Run: `git status --short --branch`

Confirm `duy_explore/` remains untracked and unchanged, and no temporary variants or caches are staged.

- [ ] **Step 4: Prepare the handoff**

Report the qualification decision first, then the top-1 lessons, exact held-out evidence, key files, tests run, candidate hash, and any limitation that remains. Do not claim improvement if a promotion threshold failed.
