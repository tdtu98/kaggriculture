# Replay-Driven Baseline3k Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine general core-route and adaptive strategies from all 90 compatible top replays, test them as standalone Baseline3k-derived agents, and create `main.py` only when the exact candidate bytes pass every binding promotion gate.

**Architecture:** A deterministic inspector converts replays into compact evidence and a frozen 60/30 discovery/holdout split. A miner ranks coherent full-route families and present-observation adaptive hypotheses, a deterministic builder embeds selected evidence into one standalone agent, and an evaluator applies replay, development, and fresh-confirmation gates against the untouched Baseline3k.

**Tech Stack:** Python 3.12, standard library, `unittest`, Kaggle Environments/Kaggriculture 1.32.7, existing `duy/benchmarks/benchmark.py`, JSON, Markdown, and Git.

**Spec:** `docs/superpowers/specs/2026-08-18-boatlee-replay-design.md`

## Global Constraints

- Keep `duy/another_work/01_baseline3k/main.py` byte-for-byte at SHA-256 `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`.
- Keep `duy/another_work/02_boatleev3/main.py` untouched.
- Read exactly the 90 JSON files in `duy_explore/kaggriculture-episodes-2026-08-15/top-100`; do not modify or delete them.
- Require replay module version `1.32.7`, 720 states, and two seats; malformed input is a hard failure.
- Align each replay action to its producing observation by shifting stored actions one state earlier.
- Freeze an exact deterministic 60 discovery / 30 holdout split.
- Adaptive support requires at least 10 discovery replays, 5 holdout replays, 3 team/opponent signatures, and both seats when both are present; one replay contributes at most one unit.
- Generated agents are standalone standard-library `main.py` files with no runtime file, replay, notebook, repository, or network dependency.
- Replay gate covers both seats of all 90 replays with zero errors, invalid actions, nondeterminism, mean latency at least as fast as `<1 ms`, and p95 `<2 ms`.
- Development uses seeds `0..9` in both seats; fresh confirmation uses seeds `50..99` in both seats.
- Promotion requires positive paired mean and median, win rate `>55%`, deterministic bootstrap lower bound `>0`, positive mean in both seats, `DONE` games, and reward equal to final money.
- Create `duy/another_work/02_boatlee_replay/main.py` only from exact fully confirmed bytes; if nothing qualifies, leave it absent.
- Preserve all unrelated tracked deletions and untracked files already present in the workspace.

---

## File Structure

- `replay_inspector.py`: validation, action alignment, compact two-seat feature extraction, deterministic split, and stable catalog generation.
- `strategy_miner.py`: route distance/clustering/medoids, adaptive support aggregation, observable-trigger validation, and stable hypothesis generation.
- `candidate_builder.py`: Baseline3k hash binding, standalone source generation, deterministic manifest generation, and optional exact promotion.
- `evaluate.py`: replay action/speed/determinism gate, paired live-game gates, bootstrap statistics, and immutable candidate verification.
- `runtime_overlay.py`: development-only source template fragments that the builder embeds into generated candidates; generated agents never import it.
- `replay_catalog.json`: compact deterministic evidence for all 90 source files and both seats.
- `strategy_hypotheses.json`: ranked core families and evidence-qualified adaptive hypotheses.
- `EXPERIMENTS.md`: candidate hashes, gates, metrics, decisions, and final recommendation.
- `tests/`: focused unit tests colocated below `02_boatlee_replay` so this untracked project can be reviewed or removed as one unit.
- `results/`: generated benchmark artifacts; keep raw per-game outputs untracked unless needed for the final evidence record.

---

### Task 1: Validate and Align Replay Evidence

**Files:**
- Create: `duy/another_work/02_boatlee_replay/replay_inspector.py`
- Create: `duy/another_work/02_boatlee_replay/tests/test_replay_inspector.py`
- Create: `duy/another_work/02_boatlee_replay/tests/__init__.py`

**Interfaces:**
- Produces: `ReplayError`, `validate_replay(replay, source) -> None`, `load_replay(path) -> dict`, `shifted_actions(replay, seat) -> list[dict]`, `extract_seat_record(replay, source, seat) -> dict`, and `stable_json(payload) -> str`.
- Record keys: `source`, `source_sha256`, `seat`, `team`, `opponent`, `team_signature`, `winner_team`, `won`, `final_margin`, `shop_sequence`, `actions`, `features`, `canonical_states`, and `route_signature`.
- `canonical_states[str(step)]` contains only route-defining public/private state: ordered actor positions and carried inventories, unlocked land, canonical tile contents, seeds, shed, cumulative purchases, money obligations, and controller state. It never contains a future observation or final reward.

- [ ] **Step 1: Write fixture and failing validation/alignment tests**

```python
def test_shifted_actions_aligns_state_one_action_to_observation_zero():
    replay = fake_replay()
    replay["steps"][1][0]["action"] = action(["NORTH"], market=[["HIRE"]])
    aligned = inspector.shifted_actions(replay, 0)
    assert aligned[0]["farmer"] == ["NORTH"]
    assert aligned[0]["market"] == [["HIRE"]]
    assert aligned[-1] == {"farmer": ["PASS"], "hands": [], "market": []}

def test_validation_rejects_wrong_version_and_truncated_episode():
    with self.assertRaisesRegex(inspector.ReplayError, "module_version"):
        inspector.validate_replay(fake_replay(module_version="1.32.6"), "old")
    short = fake_replay()
    short["steps"].pop()
    with self.assertRaisesRegex(inspector.ReplayError, "720"):
        inspector.validate_replay(short, "short")

def test_extracts_both_seats_and_strategy_features():
    replay = fixture_with_purchase_hire_field_animal_inventory_and_sale()
    rows = [inspector.extract_seat_record(replay, "fixture.json", seat) for seat in (0, 1)]
    assert [row["seat"] for row in rows] == [0, 1]
    assert rows[0]["features"]["purchase_totals"]["COW"] == 1
    assert rows[0]["features"]["operation_counts"]["HARVEST"] == 1
    assert rows[0]["features"]["sale_totals"]["MILK"] == 3
```

- [ ] **Step 2: Run RED**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_replay_inspector.py' -v`

Expected: import/file failure because `replay_inspector.py` is absent.

- [ ] **Step 3: Implement strict validation and extraction**

Use `json.loads(Path(path).read_text())`, require exact root/configuration fields, validate each state/action, and deep-copy `steps[1:][seat].action + PASS`. Extract only compact counts/timings and canonical state snapshots; replace only observation-proven weed `DIG` with `WEED_ONLY`, while keeping market timing and purchase quantities exact and normalizing only `SELL` quantity for route comparison. The test module defines `fake_replay`, `action`, and `fixture_with_purchase_hire_field_animal_inventory_and_sale` as compact 720-state constructors above the test classes.

```python
EXPECTED_CONFIGURATION = {"episodeSteps": 720, "turnsPerDay": 24}
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}

def stable_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

def shifted_actions(replay, seat):
    return [copy.deepcopy(states[seat]["action"]) for states in replay["steps"][1:]] + [copy.deepcopy(PASS_ACTION)]
```

- [ ] **Step 4: Run GREEN and integrity check**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_replay_inspector.py' -v`

Run: `git diff --exit-code -- duy/another_work/01_baseline3k/main.py duy/another_work/02_boatleev3/main.py`

Expected: focused tests pass and both protected agents have empty diffs.

- [ ] **Step 5: Commit exact Task 1 files**

```bash
git add duy/another_work/02_boatlee_replay/replay_inspector.py duy/another_work/02_boatlee_replay/tests/__init__.py duy/another_work/02_boatlee_replay/tests/test_replay_inspector.py
git commit -m "feat: inspect replay strategy evidence"
```

---

### Task 2: Freeze the 60/30 Catalog

**Files:**
- Modify: `duy/another_work/02_boatlee_replay/replay_inspector.py`
- Modify: `duy/another_work/02_boatlee_replay/tests/test_replay_inspector.py`
- Create: `duy/another_work/02_boatlee_replay/replay_catalog.json`

**Interfaces:**
- Produces: `stratified_split(replay_summaries) -> dict[str, str]`, `build_catalog(replay_dir) -> dict`, and CLI `python replay_inspector.py --replay-dir DIR --output FILE`.
- Split values are exactly `discovery` or `holdout`; filename is the final deterministic tie-breaker.

- [ ] **Step 1: Write failing deterministic split tests**

```python
def test_split_is_exact_stable_and_source_order_independent():
    # synthetic_replay_summaries returns 90 uniquely named rows whose five
    # stratification fields cycle through fixed deterministic values.
    rows = synthetic_replay_summaries(90)
    left = inspector.stratified_split(rows)
    right = inspector.stratified_split(list(reversed(rows)))
    assert left == right
    assert Counter(left.values()) == {"discovery": 60, "holdout": 30}

def test_real_corpus_catalog_has_90_files_180_seat_records():
    catalog = inspector.build_catalog(TOP100_DIR)
    assert catalog["schema_version"] == 1
    assert len(catalog["replays"]) == 90
    assert len(catalog["seat_records"]) == 180
    assert Counter(row["split"] for row in catalog["replays"]) == {"discovery": 60, "holdout": 30}
    assert {row["module_version"] for row in catalog["replays"]} == {"1.32.7"}
```

- [ ] **Step 2: Run RED**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_replay_inspector.py' -v`

Expected: missing `stratified_split` and `build_catalog`.

- [ ] **Step 3: Implement deterministic balanced allocation**

Build each replay stratum from winner team, winner seat, opponent pairing, ordered shop unlock sequence, and coarse core family. Sort by `(stratum, sha256(source + stratum), source)` and choose holdout rows greedily to minimize squared per-label deviation from one third while keeping exactly 30; assign all remaining rows discovery. Assert uniqueness and exact counts before returning.

- [ ] **Step 4: Generate twice and compare bytes**

Run two CLI generations, one to the tracked path and one under `/tmp`, then run `cmp` on them. Expected: exit 0 and catalog reports 90/180/60/30.

- [ ] **Step 5: Run the focused and real-corpus tests**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_replay_inspector.py' -v`

Expected: all pass without modifying any replay.

- [ ] **Step 6: Commit exact Task 2 files**

```bash
git add duy/another_work/02_boatlee_replay/replay_inspector.py duy/another_work/02_boatlee_replay/tests/test_replay_inspector.py duy/another_work/02_boatlee_replay/replay_catalog.json
git commit -m "feat: freeze top replay evidence catalog"
```

---

### Task 3: Mine Core Families and General Adaptive Hypotheses

**Files:**
- Create: `duy/another_work/02_boatlee_replay/strategy_miner.py`
- Create: `duy/another_work/02_boatlee_replay/tests/test_strategy_miner.py`
- Create: `duy/another_work/02_boatlee_replay/strategy_hypotheses.json`

**Interfaces:**
- Consumes: `replay_catalog.json` schema 1.
- Produces: `route_distance(left, right) -> int`, `cluster_routes(records) -> list[dict]`, `select_medoid(records) -> dict`, `canonical_state(record, step) -> dict`, `handoff_compatible(left, right, step) -> dict`, `aggregate_support(events, split) -> dict`, `trigger_is_observable(trigger) -> bool`, and `mine_strategies(catalog) -> dict`.

- [ ] **Step 1: Write failing route and handoff tests**

```python
def test_medoid_minimizes_total_distance_and_breaks_ties_by_source_seat():
    records = route_records([("b.json", 0, "N"), ("a.json", 1, "N"), ("c.json", 0, "S")])
    assert miner.select_medoid(records)["source"] == "a.json"

def test_handoff_rejects_any_state_or_controller_mismatch():
    # compatible_records returns deep-independent records with every required
    # canonical state component equal at step 216.
    left, right = compatible_records(step=216)
    assert miner.handoff_compatible(left, right, 216)["compatible"]
    right["states"]["216"]["shed"]["WHEAT"] += 1
    report = miner.handoff_compatible(left, right, 216)
    assert not report["compatible"]
    assert "shed" in report["mismatches"]
```

- [ ] **Step 2: Write failing support and observability tests**

```python
def test_support_counts_each_replay_once_and_enforces_all_thresholds():
    # supported_events creates distinct filenames and exact split/signature/
    # seat values; the duplicate below must not increase support.
    events = supported_events(discovery=10, holdout=5, signatures=3, seats={0, 1})
    events.append(dict(events[0]))
    support = miner.aggregate_support(events)
    assert support["eligible"]
    assert support["discovery_replays"] == 10
    assert support["holdout_replays"] == 5

def test_future_or_final_fields_are_rejected():
    assert miner.trigger_is_observable({"all": [{"field": "town.unlocked_shops"}]})
    assert not miner.trigger_is_observable({"field": "future_prices.MILK"})
    assert not miner.trigger_is_observable({"field": "final_reward"})
```

- [ ] **Step 3: Run RED**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_strategy_miner.py' -v`

Expected: missing `strategy_miner.py`.

- [ ] **Step 4: Implement deterministic mining**

Cluster full normalized 720-step timelines by coarse purchase/layout signature, then split clusters when normalized disagreement exceeds the deterministic within-family threshold. Rank families by discovery replay count, holdout recurrence, distinct signatures, win rate, median margin, and medoid `(source, seat)`. On discovery only, instantiate a fixed vocabulary of current-observation relations: demand-matched crop/animal mix, purchase affordability recovery, bounded sales, shed protection, feed/weed/terminal safety, hire/land timing, and public opponent production pressure. Use holdout only to count recurrence of already-instantiated discovery hypotheses. An event contributes `replay source` once per hypothesis regardless of repeated decisions.

```python
MIN_DISCOVERY = 10
MIN_HOLDOUT = 5
MIN_SIGNATURES = 3
FORBIDDEN_TRIGGER_PREFIXES = ("future_", "final_", "rewards", "hidden_")

eligible = (
    len(discovery_sources) >= MIN_DISCOVERY
    and len(holdout_sources) >= MIN_HOLDOUT
    and len(signatures) >= MIN_SIGNATURES
    and (len(available_seats) < 2 or supported_seats == available_seats)
    and trigger_is_observable(trigger)
)
```

- [ ] **Step 5: Generate twice and verify stable hypotheses**

Run the miner against the tracked catalog twice, compare bytes, assert every `eligible: true` adaptive hypothesis satisfies all support fields, and assert every core candidate identifies its complete medoid route or a compatible handoff report.

- [ ] **Step 6: Commit exact Task 3 files**

```bash
git add duy/another_work/02_boatlee_replay/strategy_miner.py duy/another_work/02_boatlee_replay/tests/test_strategy_miner.py duy/another_work/02_boatlee_replay/strategy_hypotheses.json
git commit -m "feat: mine general replay strategies"
```

---

### Task 4: Build Deterministic Standalone Candidates

**Files:**
- Create: `duy/another_work/02_boatlee_replay/runtime_overlay.py`
- Create: `duy/another_work/02_boatlee_replay/candidate_builder.py`
- Create: `duy/another_work/02_boatlee_replay/tests/test_candidate_builder.py`
- Create: `duy/another_work/02_boatlee_replay/candidates/.gitkeep`

**Interfaces:**
- Produces: `CandidateSpec(core_id: str, adaptive_ids: tuple[str, ...])`, `build_candidate(spec, baseline_path, hypotheses_path) -> tuple[str, dict]`, `write_candidate(spec, output_path, manifest_path) -> dict`, and CLI flags `--core`, repeated `--adaptive`, `--output`, `--manifest`.
- Candidate manifest keys: `schema_version`, `baseline_path`, `baseline_sha256`, `core_id`, `adaptive_ids`, `source_sha256`.

- [ ] **Step 1: Write failing hash/determinism/rejection tests**

```python
def test_two_identical_builds_are_byte_identical():
    spec = builder.CandidateSpec("baseline3k", ("shed_pressure_guard",))
    first, manifest1 = builder.build_candidate(spec, BASELINE, HYPOTHESES)
    second, manifest2 = builder.build_candidate(spec, BASELINE, HYPOTHESES)
    assert first.encode() == second.encode()
    assert manifest1 == manifest2
    assert manifest1["source_sha256"] == sha256(first.encode()).hexdigest()

def test_rejects_changed_baseline_unknown_ids_duplicates_and_bad_handoff():
    with self.assertRaisesRegex(builder.BuildError, "baseline SHA-256"):
        builder.build_candidate(CandidateSpec("baseline3k", ()), changed_baseline, HYPOTHESES)
    with self.assertRaisesRegex(builder.BuildError, "unknown core"):
        builder.build_candidate(CandidateSpec("unknown", ()), BASELINE, HYPOTHESES)
    with self.assertRaisesRegex(builder.BuildError, "duplicate"):
        builder.build_candidate(CandidateSpec("baseline3k", ("x", "x")), BASELINE, HYPOTHESES)
    with self.assertRaisesRegex(builder.BuildError, "handoff"):
        builder.build_candidate(CandidateSpec("unsafe-splice", ()), BASELINE, HYPOTHESES)
```

- [ ] **Step 2: Write failing standalone/fallback tests**

```python
def test_generated_agent_imports_without_project_files_and_fallback_aligns_hands():
    source, _ = builder.build_candidate(builder.CandidateSpec("baseline3k", ()), BASELINE, HYPOTHESES)
    module = import_isolated(source)
    obs = {"player": 0, "step": "bad", "farms": [{"hands": [[0, 0], [0, 1], [0, 2]]}, {}]}
    assert module.agent(obs) == {
        "farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []
    }
```

- [ ] **Step 3: Run RED**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_candidate_builder.py' -v`

Expected: missing builder/runtime modules.

- [ ] **Step 4: Implement the minimal builder and embedded runtime**

For `baseline3k`, embed the exact frozen source and insert selected overlay functions/constants before `agent`; wrap the original agent as `_baseline_agent`; define a new `agent` that calls it and applies current-observation overlays. For a full core medoid, embed its 720 aligned action schedule and deterministic observation-proven weed repair. Never import `runtime_overlay.py` from the generated source. Reject any generated text containing `open(`, `Path(`, network modules, or imports outside the standard library allowlist. Exception fallback derives current hand count from `obs["farms"][player]["hands"]`.

- [ ] **Step 5: Run focused tests and isolated import**

Run candidate-builder tests twice and compare generated files with `cmp`. Import the candidate from a temporary directory after changing cwd away from the repository and patch `builtins.open` during the agent call.

- [ ] **Step 6: Commit exact Task 4 files**

```bash
git add duy/another_work/02_boatlee_replay/runtime_overlay.py duy/another_work/02_boatlee_replay/candidate_builder.py duy/another_work/02_boatlee_replay/tests/test_candidate_builder.py duy/another_work/02_boatlee_replay/candidates/.gitkeep
git commit -m "feat: build replay strategy candidates"
```

---

### Task 5: Enforce Replay and Live Promotion Gates

**Files:**
- Create: `duy/another_work/02_boatlee_replay/evaluate.py`
- Create: `duy/another_work/02_boatlee_replay/tests/test_evaluate.py`

**Interfaces:**
- Produces: `validate_action(action, observation) -> list[str]`, `replay_gate(candidate_path, replay_dir) -> dict`, `paired_schedule(seed_start, seed_count) -> list[tuple[int, int]]`, `bootstrap_mean_ci(margins, seed=20260818) -> dict`, `promotion_failures(summary, expected_games) -> list[str]`, and CLI subcommands `replay`, `development`, and `confirm`.

- [ ] **Step 1: Write failing replay gate tests**

```python
def test_action_validation_checks_keys_hands_orders_and_serialization():
    obs = observation_with_hands(2)
    assert evaluator.validate_action({"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []}, obs) == []
    failures = evaluator.validate_action({"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 11}, obs)
    assert failures == ["hand_count_mismatch", "too_many_market_orders"]

def test_replay_gate_calls_both_seats_all_90_and_detects_nondeterminism():
    summary = evaluator.replay_gate(DETERMINISTIC_CANDIDATE, TOP100_DIR)
    assert summary["replays"] == 90
    assert summary["calls"] == 90 * 2 * 720
    assert summary["failures"] == []
    assert "determinism_mismatch" in evaluator.replay_gate(NONDETERMINISTIC, FIXTURE_DIR)["failures"]
```

- [ ] **Step 2: Write failing paired-statistics tests**

```python
def test_promotion_gate_requires_every_binding_threshold():
    passing = passing_summary(games=100, mean=4, median=3, wins=56, lower=0.1, seat0=2, seat1=6)
    assert evaluator.promotion_failures(passing, 100) == []
    passing["paired_seeds"]["bootstrap_mean_95ci"]["lower"] = 0
    assert evaluator.promotion_failures(passing, 100) == ["bootstrap_lower_not_positive"]

def test_schedules_are_exact_and_disjoint():
    assert evaluator.paired_schedule(0, 10) == [(seed, seat) for seed in range(10) for seat in (0, 1)]
    assert set(evaluator.paired_schedule(0, 10)).isdisjoint(evaluator.paired_schedule(50, 50))
```

- [ ] **Step 3: Run RED**

Run: `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_evaluate.py' -v`

Expected: missing `evaluate.py`.

- [ ] **Step 4: Implement replay gate and live benchmark adapter**

Load a candidate once, call every observation twice with deep-copied input, validate action structure and equality, and record `perf_counter_ns` durations. Use nearest-rank p95. For live games use `duy/benchmarks/benchmark.py` to resolve agents, build paired schedules, run games, and write deterministic summaries. Strip wall-clock paths/timestamps from evidence identity. Check statuses and reward/money equality directly in per-game results.

- [ ] **Step 5: Implement immutable confirmation**

Before development store the candidate SHA. Before and after confirmation require the same bytes. Use deterministic bootstrap seed `20260818` and 10,000 resamples over paired-seed mean margins. Development failures require exactly 20 games, positive paired mean, and positive seat means. Confirmation uses the full binding `promotion_failures` order.

- [ ] **Step 6: Run focused tests and full project discovery**

Run the new suite with `duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -v` and the unaffected legacy suite with `duy/.venv/bin/python -m unittest -v duy.benchmarks.test_benchmark duy.test_demo_agent duy.test_observer_agent`. The orphaned `test_inspect_top1_*` and `test_baseline3k_market_overlay` modules remain excluded while their user-deleted `02_inspect_top1` implementation files are absent.

Expected: all existing and new tests pass; protected agent diffs remain empty.

- [ ] **Step 7: Commit exact Task 5 files**

```bash
git add duy/another_work/02_boatlee_replay/evaluate.py duy/another_work/02_boatlee_replay/tests/test_evaluate.py
git commit -m "feat: gate replay-derived candidates"
```

---

### Task 6: Screen Core and Adaptive Lanes

**Files:**
- Create/Modify: `duy/another_work/02_boatlee_replay/EXPERIMENTS.md`
- Create: `duy/another_work/02_boatlee_replay/results/*`
- Create: `duy/another_work/02_boatlee_replay/candidates/*.py`
- Create: `duy/another_work/02_boatlee_replay/candidates/*.json`

**Interfaces:**
- Consumes only core families and adaptive hypotheses published by Task 3.
- Produces immutable candidate source/manifest pairs and gate artifacts keyed by source SHA-256.

- [ ] **Step 1: Rank a bounded experiment set before running games**

Write the exact precommitted list to `EXPERIMENTS.md`: up to three highest-supported coherent core medoids with no overlays, plus each eligible adaptive rule independently on `baseline3k`. Break evidence ties by hypothesis/core identifier. Do not add a candidate after observing its game result without recording a new experiment round.

- [ ] **Step 2: Build every precommitted candidate twice**

Run the builder twice per spec and require `cmp` equality plus manifest hash equality. Reject any candidate that fails generation or isolated import and record the exact reason.

- [ ] **Step 3: Run the replay gate for every candidate**

Run all 90 replays, both seats, and reject any candidate with a structural, determinism, exception, mean-latency, or p95-latency failure. Record calls, mean, p95, maximum, and candidate SHA.

- [ ] **Step 4: Run the 20-game development screen**

For each replay-safe core candidate and each replay-safe independent adaptive candidate, run seeds 0 through 9 in both seats against Baseline3k. Reject non-positive paired mean or either non-positive seat mean; record all summary values and artifact path.

- [ ] **Step 5: Commit the evidence checkpoint**

Stage only stable candidate manifests, small summaries, and `EXPERIMENTS.md`; do not stage bulky raw observations or transient files.

```bash
git add duy/another_work/02_boatlee_replay/EXPERIMENTS.md duy/another_work/02_boatlee_replay/candidates
git commit -m "test: screen replay strategy candidates"
```

---

### Task 7: Confirm Winners, Test Minimal Combinations, and Promote Exact Bytes

**Files:**
- Modify: `duy/another_work/02_boatlee_replay/EXPERIMENTS.md`
- Modify: `duy/another_work/02_boatlee_replay/strategy_hypotheses.json`
- Conditionally create: `duy/another_work/02_boatlee_replay/main.py`
- Conditionally create: `duy/another_work/02_boatlee_replay/PROMOTION.json`

**Interfaces:**
- Confirmation consumes only frozen SHA-qualified candidates from Task 6.
- `PROMOTION.json` records `candidate_sha256`, `baseline_sha256`, replay/development/confirmation artifact paths, metrics, core id, adaptive ids, and `promoted_source_sha256`.

- [ ] **Step 1: Confirm independent development winners**

For each qualified component run the frozen bytes on seeds 50 through 99 in both seats. Require exactly 100 valid games and every confirmation threshold. Record rejected candidates without altering their bytes or gates.

- [ ] **Step 2: Build only minimal qualified combinations**

Combine independently confirmed components smallest-first. Run replay, development, and the same fresh 100-game panel. A combination can win only if it passes all gates and its paired mean is strictly greater than every included component on the identical confirmation panel.

- [ ] **Step 3: Select the winner deterministically**

Among legal finalists, sort by confirmation paired mean, bootstrap lower bound, paired median, win rate, worst paired margin, then source SHA. If no finalist exists, record `baseline3k remains winner` and do not create `main.py`.

- [ ] **Step 4: Promote by exact byte copy only when legal**

Verify the winning candidate SHA before copying its bytes to `main.py`; verify SHA equality afterward and write `PROMOTION.json`. Import the promoted file from an isolated temporary directory, run its full replay gate again, and require identical replay summary apart from timing samples.

- [ ] **Step 5: Run final verification**

Run focused tests, full `unittest discover`, stable regeneration of both JSON evidence artifacts, protected-agent diff checks, promoted SHA checks if applicable, and `git diff --check`.

- [ ] **Step 6: Commit only the final evidence and legal promotion**

If promoted:

```bash
git add duy/another_work/02_boatlee_replay/EXPERIMENTS.md duy/another_work/02_boatlee_replay/strategy_hypotheses.json duy/another_work/02_boatlee_replay/main.py duy/another_work/02_boatlee_replay/PROMOTION.json
git commit -m "feat: promote confirmed replay-driven agent"
```

If no candidate qualifies:

```bash
git add duy/another_work/02_boatlee_replay/EXPERIMENTS.md duy/another_work/02_boatlee_replay/strategy_hypotheses.json
git commit -m "docs: record replay candidate rejections"
```
