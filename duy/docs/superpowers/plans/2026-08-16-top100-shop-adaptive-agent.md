# Top-100 Shop-Adaptive Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `02_inspect_top1` candidate from the compatible top-100 replay cohort, select a safe livestock route from live shop unlocks, and promote it only if it decisively beats `01_baseline3k` on an unseen paired-seat confirmation panel.

**Architecture:** A version-aware offline inspector filters the 100 downloaded replays to Kaggriculture 1.32.7, finds the largest coherent 42-strawberry/12-melon opening family, selects one deterministic medoid for each supported livestock branch, and emits handoff-verified route data. A generated standalone candidate embeds those routes, freezes its branch from live shop observations at steps 144 and 216, and retains the already-staged safety controllers behind independent flags. A deterministic ablation runner benchmarks only against `01_baseline3k`; the current `main.py` is not replaced until the frozen 200-game promotion gate passes.

**Tech Stack:** Python 3.12 standard library (`argparse`, `base64`, `copy`, `hashlib`, `importlib`, `json`, `pathlib`, `random`, `statistics`, `tempfile`, `unittest`, `zlib`) and `kaggle-environments==1.32.7` in `duy/.venv`.

## Global Constraints

- Never modify `duy/another_work/01_baseline3k/main.py`.
- Never modify, remove, or stage any file under `duy_explore/`.
- Only replays whose root `module_version` is exactly `1.32.7` may influence route selection.
- Treat all replay seeds as discovery evidence, never promotion evidence.
- Preserve the already-staged field guards, purchase recovery, sale cap, premium front-run controllers, and their tests.
- Keep controller feature flags independent; enable a controller only after a positive isolated ablation.
- Keep the final `02_inspect_top1/main.py` standalone and standard-library only, with public interface `agent(obs)`.
- Keep runtime decisions lightweight: decode the embedded payload once at import, never inspect replay files at runtime, and require the frozen candidate to average below 1.0 ms per `agent(obs)` call with p95 below 2.0 ms on a replay-derived 720-step local profile before confirmation.
- Benchmark every seed in both seat assignments and use paired-seed mean margin as the primary development metric.
- Do not benchmark the candidate against the old `02`; benchmark only against `01_baseline3k`.
- Development uses seeds `0..49`. Frozen confirmation uses seeds `1000..1099` exactly once for the frozen finalist.
- Do not alter the current staged `main.py` during candidate development. Develop and benchmark `candidate_main.py`; promote exact bytes only after all gates pass.
- Use `git commit --only <paths>` for task commits so unrelated staged work and `duy_explore/` remain untouched.

---

## File Structure

- Modify `duy/another_work/02_inspect_top1/inspect_replays.py`: version filtering, team collection, opening fingerprints, route distance, family selection, medoids, handoff checks, and deterministic schema-v2 output.
- Modify `duy/test_inspect_top1_replays.py`: synthetic unit tests plus stable top-100 corpus regressions.
- Modify `duy/another_work/02_inspect_top1/replay_analysis.json`: compatible/rejected inventories and selected-family evidence.
- Modify `duy/another_work/02_inspect_top1/canonical_route.json`: four branch routes, weed annotations, selector metadata, and handoff reports.
- Create `duy/another_work/02_inspect_top1/build_agent.py`: deterministic route encoder and generated-payload updater.
- Create during development `duy/another_work/02_inspect_top1/candidate_main.py`: isolated candidate; delete after promotion or a documented failed qualification.
- Modify `duy/test_inspect_top1_agent.py`: path-selectable agent tests, shop classification, branch freeze/reset, route embedding, and integration smoke tests.
- Create `duy/another_work/02_inspect_top1/evaluate_variants.py`: deterministic controller ablations and promotion-gate evaluation.
- Modify `duy/another_work/02_inspect_top1/STRATEGY.md`: top-100 evidence, branch rules, handoffs, and limitations.
- Create `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`: development ablations, frozen hashes, confirmation results, and promotion decision.
- Write benchmark artifacts beneath `duy/benchmarks/results/top100-shop-adaptive/`.
- Modify `duy/another_work/02_inspect_top1/main.py` only after the frozen candidate passes every promotion gate.

---

### Task 1: Filter the heterogeneous replay corpus and collect target routes

**Files:**
- Modify: `duy/another_work/02_inspect_top1/inspect_replays.py`
- Modify: `duy/test_inspect_top1_replays.py`

**Interfaces:**
- Produces: `REQUIRED_MODULE_VERSION = "1.32.7"`.
- Produces: `replay_module_version(replay: dict, source: Path | str = "<memory>") -> str`.
- Produces: `load_compatible_replays(paths, required_module_version=REQUIRED_MODULE_VERSION) -> tuple[list[tuple[Path, dict]], list[dict]]`.
- Produces: `collect_team_records(replays, team_name, self_seat=None) -> list[dict]`.

- [ ] **Step 1: Update synthetic fixtures and write failing version-filter tests**

Change the existing fixture signature to
`fake_replay(names=None, module_version="1.32.7")` and add
`"module_version": module_version` to its returned root dictionary. Then add
tests that reject an older version without rejecting the entire mixed corpus:

```python
TOP100_DIR = (
    Path(__file__).parent.parent
    / "duy_explore"
    / "kaggriculture-episodes-2026-08-15"
    / "top-100"
)


def test_extracts_required_root_module_version(self):
    self.assertEqual(
        self.inspector.replay_module_version(fake_replay()), "1.32.7"
    )


def test_mixed_versions_are_partitioned_not_aborted(self):
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for name, version in (("new.json", "1.32.7"), ("old.json", "1.32.6")):
            path = Path(directory) / name
            path.write_text(json.dumps(fake_replay(module_version=version)))
            paths.append(path)
        accepted, rejected = self.inspector.load_compatible_replays(paths)
    self.assertEqual([path.name for path, _ in accepted], ["new.json"])
    self.assertEqual(
        rejected,
        [{"source": "old.json", "module_version": "1.32.6",
          "reason": "module_version_mismatch"}],
    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cd duy
.venv/bin/python -m unittest -v \
  test_inspect_top1_replays.ReplayValidationTests
```

Expected: new tests fail because the version-aware interfaces do not exist.

- [ ] **Step 3: Implement deterministic filtering and per-replay team lookup**

Use this behavior:

```python
REQUIRED_MODULE_VERSION = "1.32.7"


def replay_module_version(replay, source="<memory>"):
    version = replay.get("module_version") if isinstance(replay, dict) else None
    if not isinstance(version, str) or not version:
        raise ReplayError(f"{Path(source)}: missing root module_version")
    return version


def load_compatible_replays(paths, required_module_version=REQUIRED_MODULE_VERSION):
    accepted, rejected = [], []
    for path in sorted(Path(path) for path in paths):
        replay = load_replay(path)
        version = replay_module_version(replay, path)
        if version != required_module_version:
            rejected.append({
                "source": path.name,
                "module_version": version,
                "reason": "module_version_mismatch",
            })
            continue
        accepted.append((path, replay))
    return accepted, rejected


def collect_team_records(replays, team_name, self_seat=None):
    records = []
    for path, replay in replays:
        names = replay["info"]["TeamNames"]
        if team_name not in names:
            continue
        seat = find_seat(replay, team_name, self_seat)
        records.append(inspect_replay(replay, seat, path.name))
    return records
```

Keep malformed 1.32.7 replays fatal; only a valid, explicitly older replay is a normal rejection.

- [ ] **Step 4: Add and run the real-corpus regression**

```python
def test_top100_version_and_team_counts_are_stable(self):
    paths = sorted(TOP100_DIR.glob("*.json"))
    accepted, rejected = self.inspector.load_compatible_replays(paths)
    records = self.inspector.collect_team_records(accepted, "カワシギ")
    self.assertEqual(len(paths), 100)
    self.assertEqual(len(accepted), 90)
    self.assertEqual(len(rejected), 10)
    self.assertEqual({row["module_version"] for row in rejected}, {"1.32.6"})
    self.assertEqual(len(records), 69)
```

Run:

```bash
cd duy
.venv/bin/python -m unittest -v test_inspect_top1_replays
```

Expected: all replay tests pass.

- [ ] **Step 5: Commit version-aware corpus loading**

```bash
git add \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py
git commit --only \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py \
  -m "feat: filter top100 replay evidence by engine version"
```

---

### Task 2: Select the coherent opening family and deterministic branch medoids

**Files:**
- Modify: `duy/another_work/02_inspect_top1/inspect_replays.py`
- Modify: `duy/test_inspect_top1_replays.py`

**Interfaces:**
- Produces: `SUPPORTED_BRANCHES` containing the four 42-strawberry/12-melon branch keys.
- Produces: `normalize_market_order(order: list) -> list`.
- Produces: `comparison_timeline(record: dict) -> list[dict]`.
- Produces: `opening_fingerprint(record: dict, stop: int = 72) -> str`.
- Produces: `route_distance(left: dict, right: dict) -> int`.
- Produces: `select_opening_family(records: list[dict]) -> tuple[str, list[dict]]`.
- Produces: `select_branch_medoids(records: list[dict]) -> dict[str, dict]`.

- [ ] **Step 1: Write failing normalization, fingerprint, and family tests**

```python
SUPPORTED_BRANCHES = {
    "c10-s4-straw42-melon12",
    "c8-s6-straw42-melon12",
    "c6-s8-straw42-melon12",
    "c6-s12-straw42-melon12",
}


def test_market_comparison_normalizes_only_sell_quantity(self):
    normalize = self.inspector.normalize_market_order
    self.assertEqual(normalize(["SELL", "MILK", 99]), ["SELL", "MILK"])
    self.assertEqual(
        normalize(["BUY_ANIMAL", "COW", 2]),
        ["BUY_ANIMAL", "COW", 2],
    )


def test_opening_fingerprint_is_stable_and_prefix_bounded(self):
    record = {"comparison_timeline": [{"field": [["PASS"]], "market": []}] * 73}
    first = self.inspector.opening_fingerprint(record, stop=72)
    changed = copy.deepcopy(record)
    changed["comparison_timeline"][72]["field"] = [["NORTH"]]
    self.assertEqual(first, self.inspector.opening_fingerprint(changed, stop=72))
    self.assertEqual(len(first), 64)
```

Add one synthetic family-selection test in which the largest group lacks one supported branch; the selector must choose the largest group containing all four branches, then break equal-size ties by fingerprint.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cd duy
.venv/bin/python -m unittest -v \
  test_inspect_top1_replays.StrategyExtractionTests
```

- [ ] **Step 3: Implement comparison and family selection**

Use canonical JSON bytes for the opening hash and count both field and market-intent disagreement:

```python
SUPPORTED_BRANCHES = (
    "c10-s4-straw42-melon12",
    "c8-s6-straw42-melon12",
    "c6-s8-straw42-melon12",
    "c6-s12-straw42-melon12",
)


def normalize_market_order(order):
    normalized = list(order)
    if normalized and normalized[0] == "SELL":
        return normalized[:2]
    return normalized


def comparison_timeline(record):
    rows = []
    for field, action in zip(record["comparison_actions"], record["actions"]):
        rows.append({
            "field": field,
            "market": [normalize_market_order(order)
                       for order in action.get("market", [])],
        })
    return rows


def opening_fingerprint(record, stop=72):
    payload = json.dumps(
        record["comparison_timeline"][:stop],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def route_distance(left, right):
    distance = actor_disagreement(left, right)
    for step in range(max(len(left["comparison_timeline"]),
                          len(right["comparison_timeline"]))):
        left_market = (left["comparison_timeline"][step]["market"]
                       if step < len(left["comparison_timeline"]) else None)
        right_market = (right["comparison_timeline"][step]["market"]
                        if step < len(right["comparison_timeline"]) else None)
        distance += left_market != right_market
    return distance
```

`inspect_replay` stores `comparison_timeline`. `select_opening_family` first filters to the exact 42/12 crop plan, groups by the 72-step fingerprint, keeps only groups whose branch-key set contains all `SUPPORTED_BRANCHES`, and ranks by `(-group_size, fingerprint)`. `select_branch_medoids` computes total `route_distance` within each branch and breaks ties by source filename.

- [ ] **Step 4: Add stable real-corpus family and medoid assertions**

```python
def test_top100_family_and_medoids_are_stable(self):
    accepted, _ = self.inspector.load_compatible_replays(
        sorted(TOP100_DIR.glob("*.json"))
    )
    records = self.inspector.collect_team_records(accepted, "カワシギ")
    fingerprint, family = self.inspector.select_opening_family(records)
    medoids = self.inspector.select_branch_medoids(family)
    counts = Counter(record["branch"] for record in family)

    self.assertEqual(fingerprint[:12], "c860b6d9f00f")
    self.assertEqual(len(family), 35)
    self.assertEqual({record["branch"] for record in family}, SUPPORTED_BRANCHES)
    self.assertEqual(
        counts,
        Counter({
            "c10-s4-straw42-melon12": 23,
            "c8-s6-straw42-melon12": 4,
            "c6-s8-straw42-melon12": 4,
            "c6-s12-straw42-melon12": 4,
        }),
    )
    self.assertEqual(
        {branch: record["source"] for branch, record in medoids.items()},
        {
            "c10-s4-straw42-melon12": "93232089.json",
            "c8-s6-straw42-melon12": "93316226.json",
            "c6-s8-straw42-melon12": "93339617.json",
            "c6-s12-straw42-melon12": "93399364.json",
        },
    )
```

Run the full replay test module twice and confirm identical results.

- [ ] **Step 5: Commit coherent-family selection**

```bash
git add \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py
git commit --only \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py \
  -m "feat: select coherent top100 branch medoids"
```

---

### Task 3: Prove route handoffs and generate deterministic schema-v2 artifacts

**Files:**
- Modify: `duy/another_work/02_inspect_top1/inspect_replays.py`
- Modify: `duy/test_inspect_top1_replays.py`
- Modify: `duy/another_work/02_inspect_top1/replay_analysis.json`
- Modify: `duy/another_work/02_inspect_top1/canonical_route.json`
- Modify: `duy/another_work/02_inspect_top1/STRATEGY.md`

**Interfaces:**
- Produces: `cumulative_purchases(record: dict, stop: int) -> dict[str, int]`.
- Produces: `canonical_farm_state(replay: dict, seat: int, step: int) -> dict`.
- Produces: `build_handoff_report(base: dict, branch: dict, decision_step: int) -> dict`.
- Extends: `build_outputs(paths, team_name, self_seat=None, required_module_version="1.32.7") -> tuple[dict, dict]`.

- [ ] **Step 1: Write failing handoff acceptance and rejection tests**

The report compares route actions strictly before the decision step, purchase attempts strictly before that step, and live state at the decision observation:

```python
def pass_action():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def synthetic_record(actions, decision_step=5):
    state = {
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    return {
        "actions": copy.deepcopy(actions),
        "comparison_actions": [
            [copy.deepcopy(action["farmer"]),
             *copy.deepcopy(action.get("hands", []))]
            for action in actions
        ],
        "comparison_timeline": [
            {
                "field": [copy.deepcopy(action["farmer"]),
                          *copy.deepcopy(action.get("hands", []))],
                "market": [
                    list(order[:2]) if order[0] == "SELL" else list(order)
                    for order in action.get("market", [])
                ],
            }
            for action in actions
        ],
        "canonical_states": {str(decision_step): state},
    }


def test_handoff_rejects_a_changed_predecision_action(self):
    base = synthetic_record(actions=[pass_action() for _ in range(6)])
    branch = copy.deepcopy(base)
    branch["actions"][4]["farmer"] = ["NORTH"]
    report = self.inspector.build_handoff_report(base, branch, 5)
    self.assertFalse(report["safe"])
    self.assertEqual(report["first_field_difference"], 4)


def test_handoff_accepts_a_postdecision_difference(self):
    base = synthetic_record(actions=[pass_action() for _ in range(6)])
    branch = copy.deepcopy(base)
    branch["actions"][5]["farmer"] = ["NORTH"]
    report = self.inspector.build_handoff_report(base, branch, 5)
    self.assertTrue(report["safe"])
```

Synthetic records include canonical farm state at the decision step. A state mismatch must make `safe` false even when action prefixes match.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd duy
.venv/bin/python -m unittest -v \
  test_inspect_top1_replays.HandoffTests
```

- [ ] **Step 3: Implement strict, explainable handoff reports**

`canonical_farm_state` contains farmer position, ordered hand positions, unlocked quadrants, and a row-major tile description containing only route-defining fields:

```python
def _canonical_tile(tile):
    if tile is None or tile == "LOCKED":
        return tile
    if not isinstance(tile, dict):
        return str(tile)
    return {
        key: tile.get(key)
        for key in ("kind", "crop", "animal")
        if tile.get(key) is not None
    }


def canonical_farm_state(replay, seat, step):
    observation = replay["steps"][step][seat]["observation"]
    farm = observation["farms"][seat]
    return {
        "farmer": list(farm.get("farmer", [])),
        "hands": [list(position) for position in farm.get("hands", [])],
        "unlocked_quadrants": list(farm.get("unlocked_quadrants", [])),
        "tiles": [[_canonical_tile(tile) for tile in row]
                  for row in farm.get("tiles", [])],
    }
```

Store canonical states for steps 144 and 216 on each internal record during
inspection; omit them from the public record summaries. `build_handoff_report`
emits:

```python
{
    "decision_step": decision_step,
    "safe": all((field_prefix_equal, purchase_prefix_equal,
                 farm_state_equal)),
    "field_prefix_equal": field_prefix_equal,
    "purchase_prefix_equal": purchase_prefix_equal,
    "farm_state_equal": farm_state_equal,
    "first_field_difference": first_field_difference,
    "first_market_difference": first_market_difference,
}
```

The 6-cow/12-sheep branch is checked at step 144. The 6-cow/8-sheep and 8-cow/6-sheep branches are checked against the default at step 216. Generation raises `ReplayError` if any required handoff is unsafe; it must never silently splice an unsafe trace.

- [ ] **Step 4: Add real handoff regressions**

```python
def test_selected_branch_handoffs_are_safe(self):
    analysis, routes = self.inspector.build_outputs(
        sorted(TOP100_DIR.glob("*.json")), team_name="カワシギ",
        self_seat=None,
    )
    reports = routes["handoffs"]
    self.assertTrue(reports["c6-s12-straw42-melon12"]["safe"])
    self.assertEqual(
        reports["c6-s12-straw42-melon12"]["decision_step"], 144
    )
    for branch in ("c6-s8-straw42-melon12", "c8-s6-straw42-melon12"):
        self.assertTrue(reports[branch]["safe"])
        self.assertEqual(reports[branch]["decision_step"], 216)
```

- [ ] **Step 5: Emit deterministic schema-v2 artifacts**

Use schema version 2. `replay_analysis.json` includes required module version, 90 accepted filenames, 10 rejected records, 69 target records, selected fingerprint, 35 family members, branch counts, medoid sources, and public summaries. `canonical_route.json` has this shape:

```python
{
    "schema_version": 2,
    "required_module_version": "1.32.7",
    "team_name": "カワシギ",
    "opening_fingerprint": fingerprint,
    "selector": {
        "early_step": 144,
        "freeze_step": 216,
        "default_branch": "c10-s4-straw42-melon12",
    },
    "handoffs": handoff_reports,
    "branches": {
        branch: {
            "source": medoid["source"],
            "actions": medoid["actions"],
            "weed_only": medoid["weed_only"],
        }
        for branch, medoid in sorted(medoids.items())
    },
}
```

Generate twice and compare byte-for-byte:

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/inspect_replays.py \
  --team-name カワシギ \
  --analysis-output another_work/02_inspect_top1/replay_analysis.json \
  --route-output another_work/02_inspect_top1/canonical_route.json \
  ../duy_explore/kaggriculture-episodes-2026-08-15/top-100/*.json
.venv/bin/python another_work/02_inspect_top1/inspect_replays.py \
  --team-name カワシギ \
  --analysis-output /tmp/top100-replay-analysis.json \
  --route-output /tmp/top100-canonical-route.json \
  ../duy_explore/kaggriculture-episodes-2026-08-15/top-100/*.json
cmp another_work/02_inspect_top1/replay_analysis.json \
  /tmp/top100-replay-analysis.json
cmp another_work/02_inspect_top1/canonical_route.json \
  /tmp/top100-canonical-route.json
```

- [ ] **Step 6: Update strategy findings and commit artifacts**

Replace obsolete 1.32.6/top-13 findings with the compatible counts, family fingerprint, four medoids, step-144/216 handoff evidence, and exact shop classifier. State explicitly that the ten 1.32.6 replays and the other opening family are excluded.

```bash
git add \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py \
  duy/another_work/02_inspect_top1/replay_analysis.json \
  duy/another_work/02_inspect_top1/canonical_route.json \
  duy/another_work/02_inspect_top1/STRATEGY.md
git commit --only \
  duy/another_work/02_inspect_top1/inspect_replays.py \
  duy/test_inspect_top1_replays.py \
  duy/another_work/02_inspect_top1/replay_analysis.json \
  duy/another_work/02_inspect_top1/canonical_route.json \
  duy/another_work/02_inspect_top1/STRATEGY.md \
  -m "feat: generate handoff-safe shop routes"
```

---

### Task 4: Build the isolated standalone candidate and shop selector

**Files:**
- Create: `duy/another_work/02_inspect_top1/build_agent.py`
- Create: `duy/another_work/02_inspect_top1/candidate_main.py`
- Modify: `duy/test_inspect_top1_agent.py`

**Interfaces:**
- Candidate retains: `agent(obs)`.
- Candidate produces: `_select_branch(obs, state, step) -> str`.
- Candidate extends: `_route_action(step, branch=None) -> dict`.
- Candidate produces: `_weed_annotations(step, branch=None) -> set[str]`.
- Builder produces: `encode_payload(payload: dict) -> str` and `replace_payload(source: str, encoded: str) -> str`.

- [ ] **Step 1: Seed the candidate mechanically and add generated markers**

Copy the exact staged working-tree agent without changing `main.py`:

```bash
cp duy/another_work/02_inspect_top1/main.py \
  duy/another_work/02_inspect_top1/candidate_main.py
```

Use `apply_patch` for all semantic edits to `candidate_main.py`. Put `# BEGIN GENERATED ROUTES` immediately before `_PAYLOAD` and `# END GENERATED ROUTES` immediately after it. No other text is generated by the builder.

- [ ] **Step 2: Make agent tests path-selectable and write failing selector tests**

At test-module load time:

```python
import os

DEFAULT_AGENT_PATH = (
    Path(__file__).parent / "another_work" / "02_inspect_top1" / "main.py"
)
AGENT_PATH = Path(os.environ.get("INSPECT_TOP1_AGENT_PATH", DEFAULT_AGENT_PATH))
```

Add keyword parameters `shops=None` and `town_present=True` to the existing
`make_observation` fixture so tests can provide duplicate shops or
malformed/missing town data. Add:

```python
class ShopBranchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_agent_module(name=f"shop_branch_{id(self)}")

    def test_early_yarn_selects_six_cow_twelve_sheep_at_144(self):
        state = {"last_step": 143, "active": {}}
        obs = make_observation(step=144, shops=["YARN_STORE", "YARN_STORE"])
        self.assertEqual(
            self.module._select_branch(obs, state, 144),
            "c6-s12-straw42-melon12",
        )

    def test_late_yarn_selects_six_cow_eight_sheep(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}
        obs = make_observation(step=216, shops=["BAKERY", "YARN_STORE"])
        self.assertEqual(
            self.module._select_branch(obs, state, 216),
            "c6-s8-straw42-melon12",
        )

    def test_milk_or_strawberry_demand_selects_default(self):
        for shop in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"):
            state = {"last_step": 215, "active": {}, "early_yarn": False}
            self.assertEqual(
                self.module._select_branch(
                    make_observation(step=216, shops=[shop]), state, 216
                ),
                "c10-s4-straw42-melon12",
            )

    def test_low_demand_selects_eight_cow_six_sheep(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}
        self.assertEqual(
            self.module._select_branch(
                make_observation(step=216, shops=["BAKERY", "PET_CAFE"]),
                state, 216,
            ),
            "c8-s6-straw42-melon12",
        )

    def test_missing_town_falls_back_to_default_and_freezes(self):
        state = {"last_step": 215, "active": {}, "early_yarn": False}
        first = self.module._select_branch(
            make_observation(step=216, town_present=False), state, 216
        )
        second = self.module._select_branch(
            make_observation(step=300, shops=["YARN_STORE"]), state, 300
        )
        self.assertEqual(first, "c10-s4-straw42-melon12")
        self.assertEqual(second, first)
```

Add a backwards-step/reset test asserting branch state is cleared for that seat.

- [ ] **Step 3: Run candidate tests and verify RED**

```bash
cd duy
INSPECT_TOP1_AGENT_PATH=another_work/02_inspect_top1/candidate_main.py \
  .venv/bin/python -m unittest -v test_inspect_top1_agent.ShopBranchTests
```

- [ ] **Step 4: Implement the exact branch state machine**

Use presence checks while preserving the observation list unchanged:

```python
_DEFAULT_BRANCH = "c10-s4-straw42-melon12"
_EARLY_YARN_BRANCH = "c6-s12-straw42-melon12"
_LATE_YARN_BRANCH = "c6-s8-straw42-melon12"
_LOW_DEMAND_BRANCH = "c8-s6-straw42-melon12"
_MILK_DEMAND_SHOPS = {"PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"}


def _shop_list(obs):
    town = _get(obs, "town", None)
    shops = _get(town, "unlocked_shops", None) if town is not None else None
    if not isinstance(shops, (list, tuple)):
        return None
    return list(shops)


def _select_branch(obs, state, step):
    if state.get("branch_frozen"):
        return state.get("branch", _DEFAULT_BRANCH)
    shops = _shop_list(obs)
    if step >= 144 and "early_yarn" not in state:
        state["early_yarn"] = bool(
            shops is not None and "YARN_STORE" in shops[:2]
        )
    if state.get("early_yarn"):
        state["branch"] = _EARLY_YARN_BRANCH
    else:
        state.setdefault("branch", _DEFAULT_BRANCH)
    if step >= 216:
        if state.get("early_yarn"):
            branch = _EARLY_YARN_BRANCH
        elif shops is None:
            branch = _DEFAULT_BRANCH
        elif "YARN_STORE" in shops:
            branch = _LATE_YARN_BRANCH
        elif _MILK_DEMAND_SHOPS.intersection(shops):
            branch = _DEFAULT_BRANCH
        else:
            branch = _LOW_DEMAND_BRANCH
        state["branch"] = branch
        state["branch_frozen"] = True
    return state["branch"]
```

Initialize reset state with `branch=_DEFAULT_BRANCH`; retain current weed-repair and controller state. `_route_action` and `_weed_annotations` select data from `_BRANCHES[branch]`, falling back to `_DEFAULT_BRANCH` for malformed keys. In `agent`, call `_select_branch` after `_reset_if_needed` and before route lookup, then use the same branch for actions, weed annotations, and cumulative purchase targets.

- [ ] **Step 5: Implement deterministic payload generation**

`build_agent.py` reads `canonical_route.json`, retains only `schema_version`, `selector`, and each branch's `source`, `actions`, and `weed_only`, serializes with sorted compact JSON, zlib-compresses at level 9, base85-encodes, and replaces only the generated marker block:

```python
def encode_payload(payload):
    raw = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def replace_payload(source, encoded):
    start = source.index("# BEGIN GENERATED ROUTES")
    end = source.index("# END GENERATED ROUTES")
    block = (
        "# BEGIN GENERATED ROUTES\n"
        "_PAYLOAD = json.loads(\n"
        f"    zlib.decompress(base64.b85decode({encoded!r})).decode('utf-8')\n"
        ")\n"
        "# END GENERATED ROUTES"
    )
    return source[:start] + block + source[end + len("# END GENERATED ROUTES"):]
```

CLI arguments are `--routes`, `--template`, and `--output`. Refuse to write when markers are absent. Generate the candidate twice, asserting identical SHA-256:

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/build_agent.py \
  --routes another_work/02_inspect_top1/canonical_route.json \
  --template another_work/02_inspect_top1/candidate_main.py \
  --output /tmp/candidate-generated.py
cmp another_work/02_inspect_top1/candidate_main.py /tmp/candidate-generated.py
```

- [ ] **Step 6: Run all candidate and current-agent unit tests**

```bash
cd duy
INSPECT_TOP1_AGENT_PATH=another_work/02_inspect_top1/candidate_main.py \
  .venv/bin/python -m unittest -v test_inspect_top1_agent
.venv/bin/python -m unittest -v test_inspect_top1_agent
```

Expected: both the candidate and untouched current staged agent pass; candidate imports from an isolated temporary directory without route JSON.

- [ ] **Step 7: Commit candidate infrastructure without touching `main.py`**

```bash
git add \
  duy/another_work/02_inspect_top1/build_agent.py \
  duy/another_work/02_inspect_top1/candidate_main.py \
  duy/test_inspect_top1_agent.py
git commit --only \
  duy/another_work/02_inspect_top1/build_agent.py \
  duy/another_work/02_inspect_top1/candidate_main.py \
  duy/test_inspect_top1_agent.py \
  -m "feat: build isolated shop-adaptive candidate"
```

---

### Task 5: Add deterministic ablation and promotion-gate tooling

**Files:**
- Create: `duy/another_work/02_inspect_top1/evaluate_variants.py`
- Modify: `duy/test_inspect_top1_agent.py`

**Interfaces:**
- Produces: `render_variant(source: str, flags: dict[str, bool]) -> str`.
- Produces: `promotion_failures(summary: dict, expected_games: int = 200) -> list[str]`.
- Produces: `profile_candidate(candidate_path, replay_path, team_name) -> dict` with import, mean, p50, p95, maximum, and call-count timing fields.
- CLI benchmarks named variants in isolated temporary files and writes one deterministic result directory per variant.

- [ ] **Step 1: Write failing renderer and gate tests**

```python
def passing_summary():
    return {
        "games": 200,
        "wins": 112,
        "losses": 88,
        "ties": 0,
        "by_agent_a_seat": {
            "0": {"margin": {"mean": 100.0}},
            "1": {"margin": {"mean": 80.0}},
        },
        "paired_seeds": {
            "margin": {"mean": 90.0, "median": 75.0},
            "bootstrap_mean_95ci": {"lower": 10.0, "upper": 170.0},
        },
    }


class EvaluationToolTests(unittest.TestCase):
    def test_renderer_changes_only_named_boolean_constants(self):
        source = "_ENABLE_FIELD_GUARDS = True\n_ENABLE_SALE_CAP = False\n"
        rendered = self.evaluator.render_variant(
            source, {"_ENABLE_FIELD_GUARDS": False, "_ENABLE_SALE_CAP": True}
        )
        self.assertEqual(
            rendered,
            "_ENABLE_FIELD_GUARDS = False\n_ENABLE_SALE_CAP = True\n",
        )

    def test_promotion_gate_reports_every_failed_threshold(self):
        summary = passing_summary()
        summary["wins"] = 110
        summary["paired_seeds"]["bootstrap_mean_95ci"]["lower"] = 0.0
        self.assertEqual(
            self.evaluator.promotion_failures(summary),
            ["win_rate_not_above_55_percent", "bootstrap_lower_not_positive"],
        )
```

Also test a passing 200-game summary returns `[]`, and 199 games returns `unexpected_game_count`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd duy
.venv/bin/python -m unittest -v test_inspect_top1_agent.EvaluationToolTests
```

- [ ] **Step 3: Implement strict variant rendering and promotion checks**

Recognized flags are exactly:

```python
FLAG_NAMES = (
    "_ENABLE_FIELD_GUARDS",
    "_ENABLE_PURCHASE_RECOVERY",
    "_ENABLE_SALE_CAP",
    "_ENABLE_FRONT_RUN",
)
```

Each requested constant must occur exactly once as a full line or rendering fails. `promotion_failures` checks in order:

```python
def promotion_failures(summary, expected_games=200):
    failures = []
    if summary["games"] != expected_games:
        failures.append("unexpected_game_count")
    if summary["paired_seeds"]["margin"]["mean"] <= 0:
        failures.append("paired_mean_not_positive")
    if summary["paired_seeds"]["margin"]["median"] <= 0:
        failures.append("paired_median_not_positive")
    if summary["wins"] / summary["games"] <= 0.55:
        failures.append("win_rate_not_above_55_percent")
    if summary["by_agent_a_seat"]["0"]["margin"]["mean"] <= 0:
        failures.append("seat_zero_mean_not_positive")
    if summary["by_agent_a_seat"]["1"]["margin"]["mean"] <= 0:
        failures.append("seat_one_mean_not_positive")
    if summary["paired_seeds"]["bootstrap_mean_95ci"]["lower"] <= 0:
        failures.append("bootstrap_lower_not_positive")
    return failures
```

The existing benchmark runner already rejects non-`DONE` games and reward/money mismatches. Reuse `benchmarks.benchmark.resolve_agent`, `build_schedule`, `run_suite`, `summarize`, `build_metadata`, and `write_artifacts`; do not duplicate simulator logic.

Latency profiling imports the standalone candidate once, resolves the requested
team seat from one compatible replay, then calls `agent(obs)` in replay step
order for all 720 observations while timing with `time.perf_counter_ns`. The
profile is development evidence only; the candidate must not open the replay.
Write `latency.json` with `import_ms`, `calls`, `mean_ms`, `p50_ms`, `p95_ms`,
and `maximum_ms`. Fail the profile gate when `mean_ms >= 1.0` or
`p95_ms >= 2.0`; report `maximum_ms` without a hard threshold to avoid a
single scheduler interruption becoming a flaky gate.

Define base variants:

```python
VARIANTS = {
    "route_only": (False, False, False, False),
    "field_guards": (True, False, False, False),
    "purchase_recovery": (True, True, False, False),
    "sale_cap": (True, False, True, False),
    "front_run": (True, False, False, True),
}
```

The CLI accepts `--candidate`, `--baseline`, `--seed-start`, `--seed-count`, `--output-dir`, repeated `--variant`, and `--promotion-gate`. It fails if the output directory exists, records the rendered candidate SHA-256, and writes `ablations.json` containing flags, hashes, summary, and gate failures.

- [ ] **Step 4: Run tooling tests and a two-seat smoke benchmark**

```bash
cd duy
.venv/bin/python -m unittest -v test_inspect_top1_agent.EvaluationToolTests
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --candidate another_work/02_inspect_top1/candidate_main.py \
  --baseline another_work/01_baseline3k/main.py \
  --seed-start 0 --seed-count 1 \
  --variant route_only \
  --output-dir /tmp/top100-shop-smoke-a
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --candidate another_work/02_inspect_top1/candidate_main.py \
  --baseline another_work/01_baseline3k/main.py \
  --seed-start 0 --seed-count 1 \
  --variant route_only \
  --output-dir /tmp/top100-shop-smoke-b
cmp /tmp/top100-shop-smoke-a/route_only/games.csv \
  /tmp/top100-shop-smoke-b/route_only/games.csv
```

Expected: each run has two `DONE` games, reward equals money, and the repeated
seed/seat rows are byte-identical. No promotion gate is evaluated for smoke
runs.

- [ ] **Step 5: Commit evaluation tooling**

```bash
git add \
  duy/another_work/02_inspect_top1/evaluate_variants.py \
  duy/test_inspect_top1_agent.py
git commit --only \
  duy/another_work/02_inspect_top1/evaluate_variants.py \
  duy/test_inspect_top1_agent.py \
  -m "test: add shop-agent ablation gates"
```

---

### Task 6: Run development ablations against `01_baseline3k` only

**Files:**
- Modify: `duy/another_work/02_inspect_top1/candidate_main.py`
- Create/modify: `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`
- Create: `duy/benchmarks/results/top100-shop-adaptive/development/`

- [ ] **Step 1: Verify the environment and immutable baseline before spending the panel**

```bash
cd duy
.venv/bin/python -c \
  'import importlib.metadata; print(importlib.metadata.version("kaggle-environments"))'
.venv/bin/python -m pip check
git diff -- another_work/01_baseline3k/main.py
```

Expected: version `1.32.7`, no broken requirements, and no baseline diff.

- [ ] **Step 2: Run all five base candidates on seeds 0 through 49 in both seats**

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --candidate another_work/02_inspect_top1/candidate_main.py \
  --baseline another_work/01_baseline3k/main.py \
  --seed-start 0 --seed-count 50 \
  --variant route_only \
  --variant field_guards \
  --variant purchase_recovery \
  --variant sale_cap \
  --variant front_run \
  --output-dir benchmarks/results/top100-shop-adaptive/development/base
```

This is 500 games total. Do not run old `02` as either side.

- [ ] **Step 3: Select controller combinations without data leakage**

Rank complete candidates by paired mean margin. Reject a candidate if its paired mean is negative or both seat-specific mean margins are below the route-only candidate. Retain a controller for combination testing only if its isolated candidate improves paired mean over `field_guards` and does not make both seat means worse.

For every non-empty subset of retained optional controllers, add a named flags tuple to `VARIANTS`, then run that combination once over the same development panel under `benchmarks/results/top100-shop-adaptive/development/combinations`. If no optional controller qualifies, `field_guards` is the finalist and no combination run is needed.

- [ ] **Step 4: Freeze the development winner in `candidate_main.py`**

Set the four feature flags to the winning tuple with `apply_patch`. Re-run candidate tests and compute the frozen digest:

```bash
cd duy
INSPECT_TOP1_AGENT_PATH=another_work/02_inspect_top1/candidate_main.py \
  .venv/bin/python -m unittest -v test_inspect_top1_agent
shasum -a 256 another_work/02_inspect_top1/candidate_main.py \
  another_work/01_baseline3k/main.py
```

Do not change the candidate after recording this hash unless the confirmation is abandoned and a new, independently planned development cycle is started.

Profile the frozen candidate before spending confirmation seeds:

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --profile-candidate another_work/02_inspect_top1/candidate_main.py \
  --profile-replay \
    ../duy_explore/kaggriculture-episodes-2026-08-15/top-100/93232089.json \
  --profile-team-name カワシギ \
  --profile-output benchmarks/results/top100-shop-adaptive/development/latency.json
```

Expected: 720 calls, mean below 1.0 ms, p95 below 2.0 ms. A failed latency
gate blocks confirmation and requires optimization without changing route or
controller behavior.

- [ ] **Step 5: Document development evidence**

`BENCHMARK_FINDINGS.md` records every tested flag tuple, games/wins, paired mean/median/CI, both seat means, minimum paired margin, selected finalist, candidate hash, baseline hash, Python version, and Kaggriculture version. Link each result directory relative to the document.

- [ ] **Step 6: Commit the frozen candidate and development evidence**

```bash
git add \
  duy/another_work/02_inspect_top1/candidate_main.py \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/benchmarks/results/top100-shop-adaptive/development
git commit --only \
  duy/another_work/02_inspect_top1/candidate_main.py \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/benchmarks/results/top100-shop-adaptive/development \
  -m "perf: select top100 shop-agent finalist"
```

---

### Task 7: Run the frozen 200-game confirmation and conditionally promote

**Files:**
- Modify on pass only: `duy/another_work/02_inspect_top1/main.py`
- Modify: `duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md`
- Create: `duy/benchmarks/results/top100-shop-adaptive/confirmation/`
- Delete after decision: `duy/another_work/02_inspect_top1/candidate_main.py`

- [ ] **Step 1: Reconfirm frozen hashes and ensure confirmation output is absent**

```bash
cd duy
shasum -a 256 another_work/02_inspect_top1/candidate_main.py \
  another_work/01_baseline3k/main.py
test ! -e benchmarks/results/top100-shop-adaptive/confirmation
```

The digests must match the development finding exactly. If the output already exists, stop rather than silently rerunning the holdout.

- [ ] **Step 2: Spend the frozen confirmation panel once**

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --candidate another_work/02_inspect_top1/candidate_main.py \
  --baseline another_work/01_baseline3k/main.py \
  --seed-start 1000 --seed-count 100 \
  --variant frozen \
  --promotion-gate \
  --output-dir benchmarks/results/top100-shop-adaptive/confirmation
```

For the `frozen` variant, `evaluate_variants.py` uses the candidate's constants unchanged. Expected games: 200. It exits nonzero if any promotion failure is recorded, but still writes complete evidence.

- [ ] **Step 3A: If every gate passes, promote exact bytes**

Mechanically copy the already-tested candidate, then prove exact identity before removing the temporary candidate:

```bash
cp duy/another_work/02_inspect_top1/candidate_main.py \
  duy/another_work/02_inspect_top1/main.py
cmp duy/another_work/02_inspect_top1/candidate_main.py \
  duy/another_work/02_inspect_top1/main.py
```

Delete `candidate_main.py` with `apply_patch`, update `BENCHMARK_FINDINGS.md` to `PROMOTED`, and record every gate value.

- [ ] **Step 3B: If any gate fails, preserve the staged current agent**

Do not modify `main.py`. Delete `candidate_main.py` with `apply_patch`, update `BENCHMARK_FINDINGS.md` to `NOT PROMOTED`, and list the exact failures and confirmation metrics. This is a valid completion outcome; do not tune on seeds `1000..1099`.

- [ ] **Step 4: Commit the promotion decision**

On pass:

```bash
git add \
  duy/another_work/02_inspect_top1/main.py \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/benchmarks/results/top100-shop-adaptive/confirmation
git add -u duy/another_work/02_inspect_top1/candidate_main.py
git commit --only \
  duy/another_work/02_inspect_top1/main.py \
  duy/another_work/02_inspect_top1/candidate_main.py \
  duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md \
  duy/benchmarks/results/top100-shop-adaptive/confirmation \
  -m "perf: promote confirmed top100 shop agent"
```

On failure, stage `BENCHMARK_FINDINGS.md`, the confirmation directory, and the
candidate deletion with `git add`/`git add -u`; omit `main.py` from the same
`git commit --only` command and use message
`perf: record top100 candidate qualification`.

---

### Task 8: Final verification and handoff

**Files:**
- Verify all files above.
- Modify documentation only if verification reveals stale commands or metrics.

- [ ] **Step 1: Regenerate replay artifacts and compare byte-for-byte**

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/inspect_replays.py \
  --team-name カワシギ \
  --analysis-output /tmp/top100-final-analysis.json \
  --route-output /tmp/top100-final-route.json \
  ../duy_explore/kaggriculture-episodes-2026-08-15/top-100/*.json
cmp another_work/02_inspect_top1/replay_analysis.json \
  /tmp/top100-final-analysis.json
cmp another_work/02_inspect_top1/canonical_route.json \
  /tmp/top100-final-route.json
```

- [ ] **Step 2: Run all unit tests and standalone import checks**

```bash
cd duy
.venv/bin/python -m unittest -v \
  benchmarks.test_benchmark \
  test_inspect_top1_replays \
  test_inspect_top1_agent
.venv/bin/python -m unittest discover -v
```

If promoted, the default agent tests now exercise the exact confirmed bytes.

- [ ] **Step 3: Profile the final standalone agent**

```bash
cd duy
.venv/bin/python another_work/02_inspect_top1/evaluate_variants.py \
  --profile-candidate another_work/02_inspect_top1/main.py \
  --profile-replay \
    ../duy_explore/kaggriculture-episodes-2026-08-15/top-100/93232089.json \
  --profile-team-name カワシギ \
  --profile-output /tmp/top100-final-latency.json
```

Expected: 720 calls, mean below 1.0 ms, p95 below 2.0 ms. Record import,
mean, p50, p95, and maximum latency in the final report.

- [ ] **Step 4: Run a fresh two-seat smoke outside both benchmark panels**

Use seed 50000 only:

```bash
cd duy
.venv/bin/python benchmarks/benchmark.py \
  another_work/02_inspect_top1/main.py \
  another_work/01_baseline3k/main.py \
  --seed-start 50000 --seed-count 1 --steps 720 \
  --output-dir /tmp/top100-final-smoke
```

Expected: two `DONE` games and exact reward/money agreement. This is a validity smoke, not additional tuning evidence.

- [ ] **Step 5: Verify environment, scope, and repository hygiene**

```bash
cd duy
.venv/bin/python -c \
  'import importlib.metadata, platform; print(platform.python_version()); print(importlib.metadata.version("kaggle-environments"))'
.venv/bin/python -m pip check
git diff -- another_work/01_baseline3k/main.py
git status --short
```

Expected: Kaggriculture environment 1.32.7, no baseline diff, no modified replay JSON, and `duy_explore/` remains untracked. Inspect the final diff and confirm no generated candidate file remains.

- [ ] **Step 6: Final report**

Report the promotion decision first. If promoted, give the development winner flags and all seven confirmation gates with the result directory. If not promoted, state that the current staged `02` was preserved and list failed gates. In either case, include test counts, environment version, the four medoid sources, import and decision latency metrics, and confirm that `01_baseline3k` and `duy_explore/` were untouched.
