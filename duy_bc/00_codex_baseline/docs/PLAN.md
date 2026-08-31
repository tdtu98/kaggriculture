# Ryo Behavior-Cloning v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a leakage-safe PyTorch behavior-cloning baseline that predicts Ryo Hasegawa's worker operation from the current replay state and determines whether state features outperform a clock-only policy.

**Architecture:** A strict replay reader validates the fixed 70/15/15 corpus and converts each game into a step-indexed compressed NumPy shard. Lazy PyTorch datasets feed majority, clock-only, and state-aware models; training selects checkpoints on validation macro-F1, freezes every test input by hash, and evaluates once with game-level paired bootstrap intervals.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Python `unittest`, JSON/JSONL, compressed NumPy `.npz` shards

**Spec:** `duy_rl/DESIGN.md`

## Global Constraints

- All new code and artifacts live under `duy_rl/`; the replay corpus, simulator, arena, and scripted agents are read-only.
- The binding corpus is `duy_explore/ryo_hasegawa_100_stratified`: 70 train, 15 validation, and 15 test games from its existing manifest.
- Accept only environment module version `1.32.7`, 720 stored states, two seats, exactly one `Ryo Hasegawa`, a Ryo terminal win, and `DONE/DONE` terminal statuses.
- Align observation `steps[t]` to action `steps[t + 1]` for decision steps `0..718`; state 719 is never labeled.
- Predict only the fixed 17-operation vocabulary; preserve arguments as metadata but never use them in v0 loss or metrics.
- Every feature is derived from the current delivered observation. Reward, future observations, future market state, future shop state, and opponent-private state are forbidden.
- Fit normalization, label counts, class weights, majority rules, and every learned vocabulary from train only. Validation and test consume frozen train artifacts.
- Use Python 3.12 and PyTorch. Device preference is MPS, then CUDA, then CPU.
- Default seed is `20260824`; AdamW learning rate is `1e-3`; default batch size is 512; maximum epochs is 50; early-stopping patience is 5.
- Select by validation macro-F1 with validation loss as tie-breaker. Test evaluation requires a frozen selection manifest and verifies all hashes.
- Paired confidence intervals resample games 10,000 times with seed `20260824`.
- Generated `data/` and `runs/` artifacts must not silently overwrite incompatible content and remain ignored by Git.
- Follow red-green-refactor TDD. Run the focused test before implementation, then the complete `duy_rl` suite before each task commit.

## File and interface map

| File | Single responsibility |
|---|---|
| `duy_rl/pyproject.toml` | Package metadata, Python floor, and runtime dependencies |
| `duy_rl/configs/v0.json` | Frozen default dimensions, training values, seed, and schema versions |
| `duy_rl/src/duy_rl/constants.py` | Ordered categorical vocabularies, dimensions, and config loading |
| `duy_rl/src/duy_rl/scripts_support.py` | Atomic canonical JSON writes shared by command entry points |
| `duy_rl/src/duy_rl/replay.py` | Manifest loading, source hashing, replay validation, shifted decision extraction |
| `duy_rl/src/duy_rl/features.py` | Current-observation encoders, shard serialization, and logical shard identity |
| `duy_rl/src/duy_rl/prepare.py` | Full-corpus shard orchestration, support checks, and preparation audit |
| `duy_rl/src/duy_rl/dataset.py` | Train-only statistics, majority rules, lazy shard dataset, and batches |
| `duy_rl/src/duy_rl/models.py` | Clock-only/state-aware networks, device choice, and checkpoint round-trip |
| `duy_rl/src/duy_rl/metrics.py` | Classification metrics, slices, and paired game bootstrap |
| `duy_rl/src/duy_rl/train.py` | Seed control, epoch loop, early stopping, checkpointing, and selection freeze |
| `duy_rl/src/duy_rl/evaluate.py` | Frozen-artifact verification, model evaluation, success gate, JSON/Markdown reports |
| `duy_rl/scripts/prepare_data.py` | Preparation CLI |
| `duy_rl/scripts/train_v0.py` | Baseline and neural training CLI |
| `duy_rl/scripts/evaluate_v0.py` | Frozen test-evaluation CLI |

Shared public types are defined once and reused exactly:

```python
# replay.py
@dataclass(frozen=True)
class SourceReplay:
    split: str
    episode_id: str
    path: Path
    sha256: str
    source_date: str
    route_family: str

@dataclass(frozen=True)
class Decision:
    source: SourceReplay
    step: int
    seat: int
    observation: dict[str, Any]
    action: dict[str, Any]

# features.py
@dataclass(frozen=True)
class EncodedGame:
    grid: np.ndarray
    global_features: np.ndarray
    actor_features: np.ndarray
    step_index: np.ndarray
    label: np.ndarray
    argument_item: np.ndarray
    argument_quantity: np.ndarray
    metadata: dict[str, Any]

# dataset.py
@dataclass(frozen=True)
class NormalizationStats:
    global_mean: np.ndarray
    global_std: np.ndarray
    actor_mean: np.ndarray
    actor_std: np.ndarray

@dataclass(frozen=True)
class MajorityRules:
    global_label: int
    farmer_label: int
    hand_label: int
    global_ranking: tuple[int, ...]
    farmer_ranking: tuple[int, ...]
    hand_ranking: tuple[int, ...]

@dataclass(frozen=True)
class Batch:
    grid: torch.Tensor
    global_features: torch.Tensor
    actor_features: torch.Tensor
    clock_features: torch.Tensor
    label: torch.Tensor
    game_id: tuple[str, ...]
    slices: tuple[dict[str, str], ...]
```

---

### Task 1: Install PyTorch and lock the package contract

**Files:**
- Create: `duy_rl/pyproject.toml`
- Create: `duy_rl/configs/v0.json`
- Create: `duy_rl/src/duy_rl/__init__.py`
- Create: `duy_rl/src/duy_rl/constants.py`
- Create: `duy_rl/tests/test_constants.py`
- Modify: `duy_rl/.gitignore`
- Create: `duy_rl/README.md`

**Interfaces:**
- Consumes: `duy_rl/DESIGN.md` and the existing `duy/.venv` Python 3.12 environment.
- Produces: `load_config(path: Path) -> dict[str, Any]`; ordered constants `OPERATIONS`, `PRODUCTS`, `CROPS`, `ANIMALS`, `SHOPS`, `KNOWN_SHOPS`, `TILE_KINDS`, `ARGUMENT_ITEMS`; `EXPECTED_REPLAY_CONFIGURATION`; dimensions `GRID_CHANNELS=44`, `ACTOR_DIM=38`, `GLOBAL_DIM=62`, `CLOCK_DIM=8`.

- [ ] **Step 1: Confirm the missing dependency before changing the environment**

Run:

```bash
duy/.venv/bin/python -c "import sys; print(sys.version); import torch"
```

Expected: Python reports 3.12 and exits non-zero with `ModuleNotFoundError: No module named 'torch'`.

- [ ] **Step 2: Install PyTorch into the existing project environment**

Run:

```bash
duy/.venv/bin/python -m pip install torch
duy/.venv/bin/python -c "import torch; print(torch.__version__)"
```

Expected: both commands exit zero and print an installed PyTorch version.

- [ ] **Step 3: Run the existing replay-inspector regression tests after installation**

Run:

```bash
duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_*.py' -v
```

Expected: the pre-existing replay-inspector suite passes. If discovery reports no tests, list the directory with `rg --files duy/another_work/02_boatlee_replay/tests` and run each discovered `test_*.py` module directly before continuing.

- [ ] **Step 4: Write the failing constants/config test**

```python
# duy_rl/tests/test_constants.py
import json
import unittest
from pathlib import Path

from duy_rl.constants import (
    ACTOR_DIM, CLOCK_DIM, GLOBAL_DIM, GRID_CHANNELS, OPERATIONS, load_config,
)

ROOT = Path(__file__).resolve().parents[1]


class ConstantsTest(unittest.TestCase):
    def test_v0_contract_is_fixed(self) -> None:
        config = load_config(ROOT / "configs" / "v0.json")
        self.assertEqual(len(OPERATIONS), 17)
        self.assertEqual((GRID_CHANNELS, ACTOR_DIM, GLOBAL_DIM, CLOCK_DIM), (44, 38, 62, 8))
        self.assertEqual(config["schema_version"], "ryo-bc-v0")
        self.assertEqual(config["seed"], 20260824)
        self.assertEqual(config["training"]["batch_size"], 512)
        self.assertEqual(config["training"]["max_epochs"], 50)
        json.dumps(config, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python -m unittest duy_rl/tests/test_constants.py -v
```

Expected: FAIL because `duy_rl.constants` does not exist.

- [ ] **Step 6: Add package metadata, fixed constants, config, ignore rules, and the command skeleton**

Use this package configuration:

```toml
# duy_rl/pyproject.toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[project]
name = "duy-rl"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "torch>=2.5"]

[tool.setuptools.packages.find]
where = ["src"]
```

Use this exact configuration shape:

```json
{
  "schema_version": "ryo-bc-v0",
  "feature_schema_version": "ryo-features-v0",
  "seed": 20260824,
  "corpus_root": "duy_explore/ryo_hasegawa_100_stratified",
  "module_version": "1.32.7",
  "training": {
    "learning_rate": 0.001,
    "batch_size": 512,
    "max_epochs": 50,
    "patience": 5,
    "weight_cap": 4.0
  },
  "bootstrap_resamples": 10000
}
```

In `constants.py`, define tuples in the order fixed by `DESIGN.md`, create `OPERATION_TO_ID`, and define `KNOWN_SHOPS = SHOPS + ("FARMERS_MARKET",)` while keeping only `SHOPS` in the seven output columns. Define `ARGUMENT_ITEMS = PRODUCTS + ANIMALS`; all entries are unique because products exclude live animals. Define the exact replay configuration observed in the binding corpus:

```python
EXPECTED_REPLAY_CONFIGURATION = {
    "actTimeout": 1,
    "boardSize": 10,
    "episodeSteps": 720,
    "farmHandCostMult": 1,
    "marketParams": {},
    "maxMarketOrdersPerTurn": 10,
    "runTimeout": 1200,
    "seed": None,
    "shedCapacity": 100,
    "startingMoney": 3000,
    "townCenterSellInterval": 24,
    "townShopSellInterval": 4,
    "townShopUnlockInterval": 3,
    "turnsPerDay": 24,
    "weedSpawnChance": 0.005,
}
```

Validate required config keys and exact schema values in `load_config`. In `.gitignore`, ignore `/data/`, `/runs/`, `*.egg-info/`, and `__pycache__/`. In `README.md`, document these entry commands without claiming results:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/prepare_data.py --config duy_rl/configs/v0.json
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/train_v0.py --config duy_rl/configs/v0.json --run-id ryo-v0
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/evaluate_v0.py --run-id ryo-v0 --split test
```

- [ ] **Step 7: Run focused and package tests**

Run:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all discovered tests pass.

- [ ] **Step 8: Commit the package contract**

```bash
git add duy_rl/.gitignore duy_rl/README.md duy_rl/pyproject.toml duy_rl/configs/v0.json duy_rl/src/duy_rl/__init__.py duy_rl/src/duy_rl/constants.py duy_rl/tests/test_constants.py
git commit -m "build: scaffold Ryo behavior cloning package"
```

---

### Task 2: Validate the corpus and extract shifted decisions

**Files:**
- Create: `duy_rl/src/duy_rl/replay.py`
- Create: `duy_rl/tests/fixtures.py`
- Create: `duy_rl/tests/test_replay.py`

**Interfaces:**
- Consumes: `OPERATIONS`, `OPERATION_TO_ID`, config `module_version`, corpus `manifest.csv`, and replay JSON files.
- Produces: `ReplayError`; `SourceReplay`; `Decision`; `sha256_file(path: Path) -> str`; `load_split_manifest(corpus_root: Path) -> tuple[SourceReplay, ...]`; `load_validated_replay(source: SourceReplay, expected_module_version: str) -> dict[str, Any]`; `iter_decisions(source: SourceReplay, replay: dict[str, Any]) -> Iterator[Decision]`; `operation_and_arguments(unit_action: Any) -> tuple[int, int, int]`.

- [ ] **Step 1: Create a minimal valid 720-state replay fixture**

In `duy_rl/tests/fixtures.py`, implement:

```python
from duy_rl.constants import EXPECTED_REPLAY_CONFIGURATION


def make_replay(*, ryo_seat: int = 0, hands: int = 1) -> dict:
    names = ["Ryo Hasegawa", "Opponent"] if ryo_seat == 0 else ["Opponent", "Ryo Hasegawa"]
    steps = []
    for state in range(720):
        ryo_obs = {
            "step": state,
            "player": ryo_seat,
            "farms": [{"hands": [[4, 4]] * hands}, {"hands": [[4, 4]] * hands}],
        }
        opponent_obs = dict(ryo_obs, player=1 - ryo_seat)
        action = {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}
        agents = [
            {"observation": opponent_obs, "action": action, "reward": 0, "status": "ACTIVE"},
            {"observation": opponent_obs, "action": action, "reward": 0, "status": "ACTIVE"},
        ]
        agents[ryo_seat] = {"observation": ryo_obs, "action": action, "reward": 0, "status": "ACTIVE"}
        steps.append(agents)
    steps[-1][ryo_seat].update(reward=1, status="DONE")
    steps[-1][1 - ryo_seat].update(reward=0, status="DONE")
    return {
        "id": "fixture-uuid",
        "info": {
            "EpisodeId": "fixture-game",
            "Agents": [{"Name": name} for name in names],
            "TeamNames": names,
        },
        "module_version": "1.32.7",
        "configuration": dict(EXPECTED_REPLAY_CONFIGURATION),
        "rewards": [1, 0] if ryo_seat == 0 else [0, 1],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }
```

Keep `reward` only to prove validation can inspect terminal outcome; extraction must not copy it into a `Decision.observation` feature outside the observation dictionary.

- [ ] **Step 2: Write failing alignment, validation, and vocabulary tests**

```python
# duy_rl/tests/test_replay.py
import unittest
from pathlib import Path

from fixtures import make_replay
from duy_rl.replay import ReplayError, SourceReplay, iter_decisions, load_validated_replay, operation_and_arguments


class ReplayTest(unittest.TestCase):
    def source(self) -> SourceReplay:
        return SourceReplay("train", "fixture-game", Path("fixture.json"), "a" * 64, "21-08", "route-a")

    def test_shift_excludes_terminal_state(self) -> None:
        replay = make_replay(hands=2)
        replay["steps"][1][0]["action"] = {
            "farmer": ["NORTH"],
            "hands": [["PLANT", "WHEAT", 2], ["WATER"]],
            "market": [],
        }
        decisions = tuple(iter_decisions(self.source(), replay))
        self.assertEqual(len(decisions), 719)
        self.assertEqual(decisions[0].step, 0)
        self.assertEqual(decisions[-1].step, 718)
        self.assertEqual(decisions[0].action["farmer"], ["NORTH"])

    def test_unknown_operation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReplayError, "UNKNOWN"):
            operation_and_arguments(["UNKNOWN"])

    def test_hand_action_mismatch_names_step_and_seat(self) -> None:
        replay = make_replay(hands=2)
        replay["steps"][1][0]["action"]["hands"] = []
        with self.assertRaisesRegex(ReplayError, "step=0.*seat=0"):
            tuple(iter_decisions(self.source(), replay))
```

Add separate tests that write a temporary JSON file and assert rejection for wrong version, 719 states, missing/duplicate Ryo, non-winning Ryo, non-`DONE` terminal status, missing observation/action, duplicate episode/hash across manifest rows, cross-split reuse, and manifest hash mismatch.

- [ ] **Step 3: Run the replay tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_replay.py -v
```

Expected: FAIL because `duy_rl.replay` does not exist.

- [ ] **Step 4: Implement strict manifest and replay validation**

Implement `load_split_manifest` with `csv.DictReader` and require these existing columns: `episode_id`, `split`, `source_date`, `source_path`, `source_sha256`, and `route_family`. Accept only exact split strings `train`, `val`, and `test` and require totals `70/15/15`. Load `split_summary.json` and require schema version 1, selected/unique episode/unique hash counts of 100, matching `70/15/15` counts, and stratification fields exactly `source_date`, `opponent`, `ryo_seat`, `margin_quartile`, `shop_profile`, `route_family` in that order. Resolve each source as `corpus_root / split / f"{episode_id}.json"`, follow its symlink before hashing, and retain the manifest's `source_path` only as audit provenance. Maintain both `seen_episode: dict[str, str]` and `seen_hash: dict[str, str]`; raise `ReplayError` with both conflicting rows on reuse. Hash file bytes in 1 MiB chunks.

Implement replay checks before yielding decisions. Require `replay["module_version"] == expected_module_version`, `replay["configuration"] == EXPECTED_REPLAY_CONFIGURATION`, 720 steps, and two agent records in every step. Require `info.TeamNames` and the names in `info.Agents` to be identical two-element lists, determine `ryo_seat` from that list, require `str(replay["info"]["EpisodeId"]) == source.episode_id`, and verify top-level and terminal reward ordering and statuses before using:

```python
for step in range(719):
    observation = replay["steps"][step][ryo_seat]["observation"]
    action = replay["steps"][step + 1][ryo_seat]["action"]
    if observation.get("player") != ryo_seat:
        raise ReplayError(
            f"wrong delivered seat split={source.split} episode={source.episode_id} "
            f"step={step} seat={ryo_seat} observation_player={observation.get('player')}"
        )
    hand_count = len(observation["farms"][ryo_seat]["hands"])
    if set(action) != {"farmer", "hands", "market"} or len(action["hands"]) != hand_count:
        raise ReplayError(
            f"hand/action mismatch split={source.split} episode={source.episode_id} "
            f"step={step} seat={ryo_seat} actors={hand_count + 1} "
            f"actions={1 + len(action.get('hands', []))}"
        )
    yield Decision(source, step, ryo_seat, observation, action)
```

`operation_and_arguments` must accept a unit operation represented as a string or token list, map token zero through `OPERATION_TO_ID`, map an optional item through fixed `ARGUMENT_ITEMS`, and store an optional integer quantity, using `-1` for absence. Keep the original `farmer`, `hands`, and ignored `market` groups in `Decision.action`; actor expansion belongs to feature encoding.

- [ ] **Step 5: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_replay.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit replay extraction**

```bash
git add duy_rl/src/duy_rl/replay.py duy_rl/tests/fixtures.py duy_rl/tests/test_replay.py
git commit -m "feat: validate Ryo replays and align decisions"
```

---

### Task 3: Encode current-state features and deterministic shards

**Files:**
- Create: `duy_rl/src/duy_rl/features.py`
- Create: `duy_rl/tests/test_features.py`
- Create: `duy_rl/tests/test_shards.py`

**Interfaces:**
- Consumes: `SourceReplay`, `Decision`, `iter_decisions`, fixed vocabularies and dimensions from `constants.py`.
- Produces: `FeatureError`; `EncodedGame`; `encode_game(source: SourceReplay, replay: dict[str, Any]) -> EncodedGame`; `write_shard(game: EncodedGame, path: Path, *, allow_identical: bool = True) -> str`; `read_shard(path: Path) -> EncodedGame`; `logical_shard_identity(game: EncodedGame) -> str`.

- [ ] **Step 1: Extend the synthetic observation fixture with a complete feature state**

Add `make_observation(step: int, player: int, hands: int) -> dict` in `fixtures.py`. It must contain two 10x10 farms, farmer/hand positions, money, hires, quadrants, private inventories for the delivered player, private shed/seeds, nine market products with inventory/price, and town unlocks. Use distinct self/opponent values so seat swapping is observable. Include `FARMERS_MARKET` in one state to prove it is accepted but excluded from the seven shop-count columns; any other unknown shop must fail. Make `make_replay` call it separately for each delivered seat at every state.

- [ ] **Step 2: Write failing feature-contract tests**

```python
# duy_rl/tests/test_features.py
import unittest
from pathlib import Path
import numpy as np

from fixtures import make_replay
from duy_rl.features import FeatureError, encode_game
from duy_rl.replay import SourceReplay


class FeatureTest(unittest.TestCase):
    def source(self, seat: int) -> SourceReplay:
        return SourceReplay("train", "fixture-game", Path("fixture.json"), "a" * 64, "21-08", "route-a")

    def test_shapes_actor_count_and_current_state_only(self) -> None:
        replay = make_replay(ryo_seat=0, hands=2)
        replay["steps"][1][0]["observation"]["farms"][0]["money"] = 999999
        game = encode_game(self.source(0), replay)
        self.assertEqual(game.grid.shape, (719, 44, 10, 10))
        self.assertEqual(game.global_features.shape, (719, 62))
        self.assertEqual(game.actor_features.shape, (719 * 3, 38))
        self.assertEqual(game.step_index.shape, (719 * 3,))
        self.assertEqual(game.label.shape, (719 * 3,))
        self.assertTrue(np.all(game.step_index[:3] == 0))
        self.assertNotEqual(game.global_features[0, 6], game.global_features[1, 6])
        self.assertEqual(np.count_nonzero(game.grid[:, 21]), 0)
        self.assertEqual(np.count_nonzero(game.grid[:, 43]), 0)

    def test_self_farm_is_first_for_ryo_in_seat_one(self) -> None:
        seat_zero = encode_game(self.source(0), make_replay(ryo_seat=0))
        seat_one = encode_game(self.source(1), make_replay(ryo_seat=1))
        np.testing.assert_allclose(seat_zero.grid[:, :22], seat_one.grid[:, :22])

    def test_unexpected_category_fails(self) -> None:
        replay = make_replay()
        replay["steps"][0][0]["observation"]["farms"][0]["tiles"][0][0]["kind"] = "PORTAL"
        with self.assertRaisesRegex(FeatureError, "PORTAL.*step=0"):
            encode_game(self.source(0), replay)
```

The fixture's state-zero money must be ordinary and state-one money extreme. Assert row zero reflects only state zero, proving the encoder did not read the next observation. Also assert no encoded metadata key contains `reward`.

- [ ] **Step 3: Run the feature tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_features.py -v
```

Expected: FAIL because `duy_rl.features` does not exist.

- [ ] **Step 4: Implement the exact 44/38/62 encoders**

Implement focused helpers `_encode_farm_grid`, `_encode_actor`, and `_encode_global`. Allocate float32 arrays at their final sizes and advance a cursor after each field group. At the end of every helper, assert the cursor equals its constant dimension and raise `FeatureError` otherwise.

Use fixed pre-normalization transforms: x/y divide by 9; step divides by 719; day divides by 29; hour divides by 23; cyclic hour uses `sin(2*pi*hour/24)` and `cos(2*pi*hour/24)`; hand index divides by 8; money, carried inventory, shed, seeds, market inventory, and prices use `log1p(max(0, value))`; yield divides by 6 and clips to `[0, 1]`; consecutive-unwatered/unfed clips to `[0, 1]`; every flag is 0/1. The shed-adjacency flag is one exactly at `(4,4)`, `(5,4)`, `(4,5)`, or `(5,5)`. Unexpected negative counts, out-of-range coordinates, or non-finite numbers raise `FeatureError` with sample context.

Keep both stored actor-position grid channels (self channel 21 and opponent channel 43) at zero because `grid` has one row per step. The dataset will copy the step grid and set self channel 21 at the current sample's raw actor x/y; opponent channel 43 remains zero.

For each `Decision`, require `private.inventories` length to equal one farmer plus the current Ryo hand count. Encode its current observation once into `grid[step]` and `global_features[step]`, then expand `[decision.action["farmer"], *decision.action["hands"]]` into farmer actor zero and hands `1..N`. Append actor features, shifted step index, operation label, and int32 argument item/quantity metadata in that stable order; ignore `decision.action["market"]`. Encode farms as `[ryo_seat, 1 - ryo_seat]`. The clock feature source positions must stay stable: global columns `0:6` are normalized step/day/hour, sine hour, cosine hour, and seat; actor columns `0:2` are farmer flag and normalized hand index.

Populate canonical metadata with:

```python
metadata = {
    "schema_version": "ryo-features-v0",
    "split": source.split,
    "episode_id": source.episode_id,
    "ryo_seat": ryo_seat,
    "source_path": str(source.path),
    "source_sha256": source.sha256,
    "source_date": source.source_date,
    "route_family": source.route_family,
    "sample_count": int(label.shape[0]),
    "shapes": {
        "grid": list(grid.shape),
        "global_features": list(global_features.shape),
        "actor_features": list(actor_features.shape),
        "step_index": list(step_index.shape),
        "label": list(label.shape)
    }
}
```

- [ ] **Step 5: Write failing shard round-trip and identity tests**

```python
# duy_rl/tests/test_shards.py
import tempfile
import unittest
from pathlib import Path
import numpy as np

from fixtures import make_replay
from duy_rl.features import encode_game, logical_shard_identity, read_shard, write_shard


class ShardTest(unittest.TestCase):
    def test_round_trip_preserves_arrays_metadata_and_identity(self) -> None:
        game = encode_game(self.source(), make_replay())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.npz"
            first = write_shard(game, path)
            loaded = read_shard(path)
            second = logical_shard_identity(loaded)
        self.assertEqual(first, second)
        self.assertEqual(game.metadata, loaded.metadata)
        np.testing.assert_array_equal(game.grid, loaded.grid)
        np.testing.assert_array_equal(game.label, loaded.label)
```

Add tests that a second identical write is accepted, a non-identical existing file raises `FeatureError`, corrupt shape/dtype is rejected on read, and separately produced identical logical games have the same identity without comparing `.npz` container bytes.

- [ ] **Step 6: Implement canonical shard serialization**

Serialize arrays with `np.savez_compressed` and metadata as a scalar Unicode array containing `json.dumps(metadata, sort_keys=True, separators=(",", ":"))`. Compute identity by SHA-256 over each array name, dtype, shape, C-order bytes, and canonical metadata bytes in a fixed field order. Write to a sibling temporary file, read it back, verify the identity, then use `Path.replace` only when the target is absent or logically identical.

- [ ] **Step 7: Run feature, shard, and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_features.py duy_rl/tests/test_shards.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit feature and shard encoding**

```bash
git add duy_rl/src/duy_rl/features.py duy_rl/tests/fixtures.py duy_rl/tests/test_features.py duy_rl/tests/test_shards.py
git commit -m "feat: encode current replay state into deterministic shards"
```

---

### Task 4: Prepare the complete split and publish its audit

**Files:**
- Create: `duy_rl/src/duy_rl/scripts_support.py`
- Create: `duy_rl/src/duy_rl/prepare.py`
- Create: `duy_rl/scripts/prepare_data.py`
- Create: `duy_rl/tests/test_prepare_data.py`
- Modify: `duy_rl/README.md`

**Interfaces:**
- Consumes: `load_split_manifest`, `load_validated_replay`, `encode_game`, `write_shard`, config and corpus paths.
- Produces: `prepare(config_path: Path, output_root: Path, *, limit_per_split: int | None = None) -> dict[str, Any]`; generated `data/{train,val,test}/*.npz`; generated `data/audit.json`.

- [ ] **Step 1: Write a failing preparation audit test**

```python
# duy_rl/tests/test_prepare_data.py
import json
import tempfile
import unittest
from pathlib import Path

from duy_rl.prepare import prepare
from duy_rl.scripts_support import atomic_json_write


class PrepareAuditTest(unittest.TestCase):
    def test_atomic_json_write_refuses_mismatched_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            atomic_json_write(path, {"games": 1})
            atomic_json_write(path, {"games": 1})
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                atomic_json_write(path, {"games": 2})
            self.assertEqual(json.loads(path.read_text()), {"games": 1})
```

Add a temporary three-game mini-corpus test by mocking `load_split_manifest` to return one synthetic source per split and assert the audit has split game/sample totals, source hashes, shard identities, label counts, tensor shapes, and a `checks` object whose values are all `true`. Add a test that a validation/test operation absent from train raises a non-zero CLI error.

- [ ] **Step 2: Run the preparation test to verify it fails**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_prepare_data.py -v
```

Expected: FAIL because `duy_rl.scripts_support` and the preparation entry point do not exist.

- [ ] **Step 3: Implement atomic JSON output and the preparation orchestration**

Create the small shared helper `duy_rl/src/duy_rl/scripts_support.py` with:

```python
def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"refusing to overwrite non-matching artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    temp.replace(path)
```

In `prepare`, iterate sources in `(split, episode_id)` order. Record all successful validations in audit fields rather than inferring them later. Aggregate train label support, then require every validation/test label ID to be in train. Include SHA-256 values for both `manifest.csv` and `split_summary.json`, and derive the preparation identity from both hashes, sorted source hashes, shard identities, and canonical config bytes. On any exception, identify split, episode, step, seat, and actor when those values exist and let the CLI exit non-zero.

- [ ] **Step 4: Implement CLI argument parsing**

`prepare_data.py` accepts `--config`, optional `--output-root` defaulting to `duy_rl/data`, and optional `--limit-per-split` for smoke tests only. It resolves repository-relative paths from the repository root, calls `prepare`, and prints a one-line split/game/sample summary on success. A limited run still validates the complete 100-row manifest and every selected replay, but records `smoke_mode: true` and does not claim the full-corpus class-support gate; only an unlimited run may write `checks.operation_support_covered: true` and become a training input.

- [ ] **Step 5: Run tests and a real one-game-per-split smoke preparation**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/prepare_data.py --config duy_rl/configs/v0.json --output-root /tmp/duy_rl_smoke_data --limit-per-split 1
```

Expected: tests pass; the smoke command exits zero, writes three shards and an audit, reports one game in each split, and marks the audit as non-trainable smoke output. Remove only `/tmp/duy_rl_smoke_data` after confirming it is the explicit smoke directory.

- [ ] **Step 6: Update README preparation semantics and commit**

Document that preparation validates without mutating the source corpus, generated outputs are ignored, and rerunning accepts only logically identical artifacts.

```bash
git add duy_rl/src/duy_rl/scripts_support.py duy_rl/src/duy_rl/prepare.py duy_rl/scripts/prepare_data.py duy_rl/tests/test_prepare_data.py duy_rl/README.md
git commit -m "feat: prepare audited behavior cloning shards"
```

---

### Task 5: Build train-only statistics, baselines, and the lazy dataset

**Files:**
- Create: `duy_rl/src/duy_rl/dataset.py`
- Create: `duy_rl/tests/test_dataset.py`

**Interfaces:**
- Consumes: `.npz` shards from `read_shard`, `GLOBAL_DIM`, `ACTOR_DIM`, `CLOCK_DIM`, and 17 operation labels.
- Produces: `NormalizationStats`; `MajorityRules`; `fit_train_artifacts(train_shards: Sequence[Path], weight_cap: float = 4.0) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules]`; `save_train_artifacts(path: Path, stats: NormalizationStats, class_counts: np.ndarray, class_weights: np.ndarray, majority: MajorityRules, metadata: dict[str, Any]) -> str`; `load_train_artifacts(path: Path) -> tuple[NormalizationStats, np.ndarray, np.ndarray, MajorityRules, dict[str, Any]]`; `ShardDataset(shard_paths: Sequence[Path], stats: NormalizationStats, cache_size: int = 2)`; `collate_examples(rows: Sequence[dict[str, Any]]) -> Batch`.

- [ ] **Step 1: Write failing train-only-statistics tests**

```python
# duy_rl/tests/test_dataset.py
import unittest
import numpy as np

from duy_rl.dataset import fit_array_statistics, inverse_sqrt_class_weights, fit_majority_rules


class DatasetTest(unittest.TestCase):
    def test_statistics_use_only_supplied_training_rows(self) -> None:
        train = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        mean, std = fit_array_statistics([train])
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(std, [1.0, 2.0])

    def test_class_weights_are_mean_one_and_capped(self) -> None:
        weights = inverse_sqrt_class_weights(np.array([100, 25, 1]), cap=4.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 4.0)

    def test_majority_rules_have_global_farmer_and_hand_predictions(self) -> None:
        rules = fit_majority_rules(labels=np.array([1, 1, 2, 2, 2]), actor_is_farmer=np.array([1, 1, 0, 0, 0]))
        self.assertEqual((rules.global_label, rules.farmer_label, rules.hand_label), (2, 1, 2))
        self.assertEqual(rules.global_ranking[:3], (2, 1, 0))
```

Add tests for zero-variance columns becoming standard deviation one, all 17 class counts being required, validation/test arrays never being accepted by `fit_train_artifacts`, lazy cache eviction at size two, per-example grid lookup via `step_index`, and clock construction as `global_features[:6] + actor_features[:2]` with shape 8.

- [ ] **Step 2: Run the dataset tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_dataset.py -v
```

Expected: FAIL because `duy_rl.dataset` does not exist.

- [ ] **Step 3: Implement streaming statistics and class artifacts**

Use float64 running `count`, `sum`, and `sum_sq` across training shards, then emit float32 mean/std. Treat absolute std below `1e-8` as one. Count every label with `np.bincount(..., minlength=17)` and reject any zero train count before computing:

```python
raw = 1.0 / np.sqrt(counts.astype(np.float64))
weights = np.zeros_like(raw)
active = np.ones(raw.shape, dtype=bool)
remaining_total = float(raw.size)
while active.any():
    scale = remaining_total / raw[active].sum()
    over = active & (raw * scale > weight_cap)
    if not over.any():
        weights[active] = raw[active] * scale
        break
    weights[over] = weight_cap
    remaining_total -= weight_cap * int(over.sum())
    active[over] = False
if not np.isclose(weights.mean(), 1.0) or weights.max() > weight_cap:
    raise ValueError("class-weight normalization invariant failed")
```

`MajorityRules.predict(actor_is_farmer: np.ndarray) -> np.ndarray` chooses the farmer mode for `1` and hand mode for `0`; the global mode is evaluated separately. Store complete 17-class rankings for global, farmer, and hand counts so top-3 is defined. Sort by descending count with lower operation ID breaking ties; the first ranking entry must equal its corresponding label. For metric reuse, convert a ranking to logits by assigning score `17-rank_index` to each class. `fit_train_artifacts` rejects any shard whose metadata split is not exactly `train`.

Store train artifacts as one compressed `.npz` containing four normalization arrays, int64 class counts, float32 class weights, three majority labels, three 17-entry majority rankings, and canonical JSON metadata with schema versions, ordered operations, train shard identities, and preparation manifest hash. Compute a logical SHA-256 over names/dtypes/shapes/bytes plus canonical metadata. Accept an existing target only when its logical identity matches. `load_train_artifacts` verifies dimensions 62/38, 17 positive counts, finite mean-one weights no greater than the configured cap, valid ranking permutations, and vocabulary/schema equality.

- [ ] **Step 4: Implement the lazy shard dataset and batch contract**

Build a prefix-sum index of sample counts without retaining grid arrays. `__getitem__` locates the shard with `bisect_right`, loads at most `cache_size` `EncodedGame` objects in an `OrderedDict`, indexes `grid` and `global_features` by `step_index`, and copies the grid before mutation. Reconstruct integer actor coordinates by rounding raw actor columns 2 and 3 times 9, set `grid[21, y, x] = 1.0`, assert channel 43 stays zero, then normalize global and actor arrays. Build `clock_features` after normalization only from fixed source columns; do not expose grid or other state to `ClockOnlyModel`.

`collate_examples` stacks numeric tensors and returns `game_id` plus slices containing actor type, seat, source date, and route family. Derive day bands from the raw zero-based observation day: `0..6 -> days-1-7`, `7..13 -> days-8-14`, `14..20 -> days-15-21`, and `21..29 -> days-22-plus`.

- [ ] **Step 5: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_dataset.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit dataset and baseline fitting**

```bash
git add duy_rl/src/duy_rl/dataset.py duy_rl/tests/test_dataset.py
git commit -m "feat: add train-only artifacts and lazy shard dataset"
```

---

### Task 6: Implement models and checkpoint integrity

**Files:**
- Create: `duy_rl/src/duy_rl/models.py`
- Create: `duy_rl/tests/test_models.py`

**Interfaces:**
- Consumes: tensors in `Batch`, config architecture values, feature vocabularies, normalization stats, class weights, and source manifest hash.
- Produces: `ClockOnlyModel`; `StateAwareModel`; `choose_device() -> torch.device`; `save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, metadata: dict[str, Any], epoch: int) -> str`; `load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write failing model shape and isolation tests**

```python
# duy_rl/tests/test_models.py
import unittest
import torch

from duy_rl.models import ClockOnlyModel, StateAwareModel


class ModelTest(unittest.TestCase):
    def test_state_aware_output_shape_and_parameter_budget(self) -> None:
        model = StateAwareModel()
        logits = model(torch.zeros(4, 44, 10, 10), torch.zeros(4, 62), torch.zeros(4, 38))
        self.assertEqual(tuple(logits.shape), (4, 17))
        parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        self.assertGreater(parameters, 190_000)
        self.assertLess(parameters, 230_000)

    def test_clock_model_accepts_only_eight_features(self) -> None:
        model = ClockOnlyModel()
        self.assertEqual(tuple(model(torch.zeros(4, 8)).shape), (4, 17))
        with self.assertRaises(RuntimeError):
            model(torch.zeros(4, 9))
```

Add deterministic CPU checkpoint test: seed, construct model/AdamW, save, mutate weights, load into a fresh model, and assert identical evaluation logits. Assert metadata includes schema versions, vocabularies, stats, weights, manifest hash, architecture, optimizer state, epoch, and Python/NumPy/PyTorch random states. Add a corrupt checkpoint-hash rejection test.

- [ ] **Step 2: Run model tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_models.py -v
```

Expected: FAIL because `duy_rl.models` does not exist.

- [ ] **Step 3: Implement the two fixed architectures**

Implement `ClockOnlyModel` as `Linear(8, 64)`, ReLU, `Linear(64, 64)`, ReLU, `Linear(64, 17)`.

Implement `StateAwareModel` exactly as:

```python
self.tile = nn.Sequential(
    nn.Conv2d(44, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
)
self.actor = nn.Sequential(nn.Linear(38, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
self.global_encoder = nn.Sequential(nn.Linear(62, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
self.classifier = nn.Sequential(
    nn.Linear(448, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 17),
)
```

`choose_device` checks `torch.backends.mps.is_available()`, then `torch.cuda.is_available()`, else CPU.

- [ ] **Step 4: Implement atomic checkpoint payloads and verified loading**

Save a `.pt` payload to a temporary sibling, compute SHA-256 over the resulting file, replace the target only if absent, and write `<checkpoint>.sha256` atomically. Loading verifies the sidecar before `torch.load(..., map_location="cpu", weights_only=False)`. Capture `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, and CUDA states when available; restore them only on resume. Evaluation-only loading restores weights and validates metadata without mutating RNG.

- [ ] **Step 5: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_models.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit models and checkpointing**

```bash
git add duy_rl/src/duy_rl/models.py duy_rl/tests/test_models.py
git commit -m "feat: add clock and state behavior cloning models"
```

---

### Task 7: Implement metrics, slices, and paired game bootstrap

**Files:**
- Create: `duy_rl/src/duy_rl/metrics.py`
- Create: `duy_rl/tests/test_metrics.py`

**Interfaces:**
- Consumes: integer labels/predictions, logits, game IDs, and slice metadata from `Batch`.
- Produces: `classification_report(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]`; `slice_reports(logits: np.ndarray, labels: np.ndarray, slices: Sequence[dict[str, str]]) -> dict[str, dict[str, Any]]`; `paired_game_bootstrap(state_correct: np.ndarray, clock_correct: np.ndarray, game_ids: Sequence[str], resamples: int = 10000, seed: int = 20260824) -> dict[str, float]`.

- [ ] **Step 1: Write failing exact-metric tests**

```python
# duy_rl/tests/test_metrics.py
import unittest
import numpy as np

from duy_rl.metrics import classification_report, paired_game_bootstrap


class MetricsTest(unittest.TestCase):
    def test_classification_report_uses_all_seventeen_classes(self) -> None:
        logits = np.full((3, 17), -10.0)
        logits[0, 0], logits[1, 1], logits[2, 2] = 10.0, 10.0, 10.0
        report = classification_report(logits, np.array([0, 2, 2]))
        self.assertAlmostEqual(report["top1"], 2 / 3)
        self.assertEqual(len(report["per_class"]), 17)
        self.assertEqual(np.asarray(report["confusion_matrix"]).shape, (17, 17))

    def test_bootstrap_resamples_games_not_rows(self) -> None:
        games = np.array(["large"] * 100 + ["small"])
        state = np.array([1] * 100 + [0], dtype=bool)
        clock = np.array([0] * 100 + [1], dtype=bool)
        result = paired_game_bootstrap(state, clock, games, resamples=1000, seed=7)
        self.assertEqual(result["games"], 2)
        self.assertAlmostEqual(result["point_delta"], 0.0)
```

Add tests for top-3, zero-support precision/recall/F1 being zero, macro-F1 averaging all 17 classes, deterministic repeated bootstrap, paired length/game mismatch errors, and every required slice dimension.

- [ ] **Step 2: Run metric tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_metrics.py -v
```

Expected: FAIL because `duy_rl.metrics` does not exist.

- [ ] **Step 3: Implement metrics without adding a new dependency**

Compute the 17x17 confusion matrix with `np.add.at`. Derive per-class precision, recall, F1, and support with `np.divide(..., where=...)`, setting undefined results to zero. Compute top-3 from `np.argpartition(logits, -3, axis=1)[:, -3:]`. Return only JSON-serializable Python numbers and lists.

For each slice dimension (`actor_type`, `seat`, `day_band`, `source_date`, `route_family`), group exact values and run `classification_report` on the corresponding row mask. Retain zero-support operations inside every report.

- [ ] **Step 4: Implement deterministic paired game bootstrap**

First compute each model's accuracy within each game. The point delta is the unweighted mean of per-game state accuracy minus the unweighted mean of per-game clock accuracy. Draw `len(unique_games)` game indices with replacement for each replicate using `np.random.default_rng(seed)`, compute the paired delta, and return the 2.5/97.5 percentiles plus seed, resample count, and game count.

- [ ] **Step 5: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_metrics.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit metric evaluation primitives**

```bash
git add duy_rl/src/duy_rl/metrics.py duy_rl/tests/test_metrics.py
git commit -m "feat: add game-level behavior cloning metrics"
```

---

### Task 8: Train baselines and neural models with validation-only selection

**Files:**
- Create: `duy_rl/src/duy_rl/train.py`
- Create: `duy_rl/scripts/train_v0.py`
- Create: `duy_rl/tests/test_train.py`
- Modify: `duy_rl/README.md`

**Interfaces:**
- Consumes: prepared shard paths/audit, `ShardDataset`, train artifacts, `ClockOnlyModel`, `StateAwareModel`, `classification_report`, config.
- Produces: `seed_everything(seed: int) -> None`; `train_one_epoch(model: nn.Module, loader: Iterable[Batch], optimizer: Optimizer, criterion: nn.Module, device: torch.device, model_name: Literal["clock", "state"]) -> float`; `evaluate_loader(model: nn.Module, loader: Iterable[Batch], criterion: nn.Module, device: torch.device, model_name: Literal["clock", "state"]) -> dict[str, Any]`; `fit_model(model_name: Literal["clock", "state"], train_loader: DataLoader, val_loader: DataLoader, class_weights: np.ndarray, config: dict[str, Any], run_dir: Path, checkpoint_metadata: dict[str, Any], resume: bool = False) -> dict[str, Any]`; `freeze_selection(run_dir: Path, selected: dict[str, Path], config_path: Path, artifacts_path: Path, audit_path: Path) -> dict[str, Any]`; generated run histories/checkpoints/baseline validation reports/`selection.json`.

- [ ] **Step 1: Write failing one-step training and early-selection tests**

```python
# duy_rl/tests/test_train.py
import unittest
import torch

from duy_rl.models import ClockOnlyModel
from duy_rl.train import is_better_epoch, train_one_epoch


class TrainTest(unittest.TestCase):
    def test_macro_f1_wins_and_loss_breaks_ties(self) -> None:
        self.assertTrue(is_better_epoch(0.31, 1.2, 0.30, 0.8))
        self.assertTrue(is_better_epoch(0.30, 0.7, 0.30, 0.8))
        self.assertFalse(is_better_epoch(0.30, 0.9, 0.30, 0.8))

    def test_one_epoch_updates_clock_model(self) -> None:
        model = ClockOnlyModel()
        before = [value.detach().clone() for value in model.parameters()]
        loss = train_one_epoch(model, self.loader(), torch.optim.AdamW(model.parameters(), lr=1e-3), self.loss(), torch.device("cpu"), "clock")
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
```

Provide a tiny deterministic `Batch` loader in the test. Add tests that NaN loss raises, patience five stops after five non-improving epochs, JSONL has one canonical record per epoch, resume restores optimizer/epoch/RNG, run directory rejects a different config, and `freeze_selection` hashes the clock checkpoint, state checkpoint, config, train artifacts, and preparation audit.

- [ ] **Step 2: Run training tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_train.py -v
```

Expected: FAIL because `duy_rl.train` does not exist.

- [ ] **Step 3: Implement deterministic loaders and epoch functions**

`seed_everything` seeds Python, NumPy, PyTorch, and CUDA when available, and creates DataLoader generators from the same seed. Construct loaders with `num_workers=0`, `collate_fn=collate_examples`, training shuffle enabled from the seeded generator, and validation shuffle disabled. `train_one_epoch` routes only `batch.clock_features` to the clock model and routes grid/global/actor to the state model. It uses weighted cross-entropy, checks every loss with `torch.isfinite`, clips no gradients in v0, and returns sample-weighted mean loss.

`evaluate_loader` runs under `torch.inference_mode()`, accumulates CPU logits/labels/game IDs/slices, and returns loss plus `classification_report` and `slice_reports`. It never examines the test split during `fit_model`.

- [ ] **Step 4: Implement model fitting, resume, early stopping, and histories**

For each of `clock` and `state`, create AdamW at `1e-3`, run at most 50 epochs, and save immutable `epoch-<NNN>.pt` checkpoints. Record the best checkpoint path when `is_better_epoch` returns true; never overwrite an older checkpoint. Stop after exactly five completed non-improving epochs. Append canonical JSON objects to `<model>/epochs.jsonl`; on resume require prior records to be a prefix of the restored epoch. Record device, seeds, data identities, train loss, validation loss, validation metrics, best epoch, and elapsed seconds.

Fit and store both global and actor-stratified majority rules before neural training. Evaluate them on validation with the same metric functions and store `majority.validation.json`.

- [ ] **Step 5: Implement the selection freeze**

After both neural fits finish, `freeze_selection` creates `selection.json` containing run ID, schema versions, selected checkpoint paths and SHA-256s, config path/hash/canonical content, train artifact path/hash, audit path/hash, manifest hash, model architecture values, and creation timestamp. The timestamp is descriptive and excluded from the computed `selection_identity`. If a selection exists, accept only an identical identity.

- [ ] **Step 6: Implement the training CLI and update README**

`train_v0.py` accepts `--config`, `--run-id`, optional `--data-root`, optional `--runs-root`, and `--resume`. It requires `data/audit.json`, confirms the audit identity and operation support, fits train artifacts once, trains majority/clock/state, freezes selection, and exits non-zero on incompatible artifacts or an empty class.

README must explain that training reads train/validation only and that a completed command freezes the exact inputs required for test evaluation.

- [ ] **Step 7: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_train.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit training and selection**

```bash
git add duy_rl/src/duy_rl/train.py duy_rl/scripts/train_v0.py duy_rl/tests/test_train.py duy_rl/README.md
git commit -m "feat: train and freeze Ryo behavior cloning models"
```

---

### Task 9: Enforce frozen test evaluation and produce the go/no-go report

**Files:**
- Create: `duy_rl/src/duy_rl/evaluate.py`
- Create: `duy_rl/scripts/evaluate_v0.py`
- Create: `duy_rl/tests/test_evaluate.py`
- Modify: `duy_rl/README.md`

**Interfaces:**
- Consumes: `selection.json`, verified checkpoints, train artifacts, preparation audit, test shards, majority rules, metric and bootstrap functions.
- Produces: `verify_selection(run_dir: Path) -> dict[str, Any]`; `evaluate_frozen_run(run_dir: Path, data_root: Path, split: str = "test") -> dict[str, Any]`; `success_gate(report: dict[str, Any]) -> dict[str, Any]`; generated `evaluation.test.json` and `REPORT.md`.

- [ ] **Step 1: Write failing frozen-input and gate tests**

```python
# duy_rl/tests/test_evaluate.py
import unittest

from duy_rl.evaluate import EvaluationError, success_gate, verify_selection


class EvaluateTest(unittest.TestCase):
    def test_unfrozen_run_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "selection.json"):
            verify_selection(self.empty_run_dir())

    def test_gate_requires_every_condition(self) -> None:
        report = self.passing_report()
        self.assertTrue(success_gate(report)["pass"])
        report["bootstrap"]["ci95_low"] = 0.0
        result = success_gate(report)
        self.assertFalse(result["pass"])
        self.assertIn("bootstrap_lower_bound_positive", result["failed"])
```

Add tests that modified checkpoint/config/artifact/audit hashes are rejected; `split != "test"` uses a distinct output filename; identical frozen test reruns are accepted; changed test shard identity is rejected; all three systems have overall and required slice reports; and Markdown includes every gate condition plus explicit `PROCEED` or `STOP`.

- [ ] **Step 2: Run evaluation tests to verify they fail**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_evaluate.py -v
```

Expected: FAIL because `duy_rl.evaluate` does not exist.

- [ ] **Step 3: Implement selection and data verification**

`verify_selection` loads `selection.json`, recomputes every recorded SHA-256, validates schema/architecture/vocabulary equality, and rejects absent or changed inputs before loading any test labels. Compare current shard identities and manifest hash to the frozen audit. Reject non-finite logits, missing models, duplicate game IDs, and empty test data.

- [ ] **Step 4: Evaluate majority, clock, and state systems on identical rows**

Use one deterministic non-shuffled DataLoader. Accumulate labels/game IDs/slices once. Produce logits or predictions for global majority, actor-stratified majority, clock, and state; the three primary named systems in the gate are `majority_actor`, `clock`, and `state`. Report top-1, top-3, macro-F1, per-class values, confusion matrix, and all five slice families for each. Compute paired game bootstrap from state-versus-clock top-1 correctness with exactly the configured 10,000 resamples.

- [ ] **Step 5: Implement all six gate checks and both report formats**

Return explicit booleans for:

```python
checks = {
    "preparation_and_leakage_valid": report["audit"]["all_checks_passed"],
    "operation_support_covered": report["audit"]["operation_support_covered"],
    "validation_state_macro_f1_gt_clock": report["validation"]["state"]["macro_f1"] > report["validation"]["clock"]["macro_f1"],
    "test_state_macro_f1_gt_clock": report["test"]["state"]["macro_f1"] > report["test"]["clock"]["macro_f1"],
    "bootstrap_lower_bound_positive": report["bootstrap"]["ci95_low"] > 0.0,
    "complete_diagnostics": report["complete_diagnostics"],
}
```

Write canonical JSON with model metrics, slices, bootstrap, artifact identities, and gate. Render `REPORT.md` with corpus counts, selected epochs, a compact model comparison table, bootstrap interval, required slices, every failed/passed gate line, and final `PROCEED TO MULTI-HEAD CLONING` only when all checks are true; otherwise `STOP AND DIAGNOSE V0`.

- [ ] **Step 6: Implement the evaluation CLI and update README**

`evaluate_v0.py` accepts `--run-id`, optional `--data-root`, optional `--runs-root`, and `--split` restricted to `val` or `test`. For test it must call `verify_selection` first. It prints the report path, state/clock macro-F1, top-1 delta interval, and final decision. README states that changing any frozen input requires a new run ID.

- [ ] **Step 7: Run focused and complete tests**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_evaluate.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit frozen evaluation**

```bash
git add duy_rl/src/duy_rl/evaluate.py duy_rl/scripts/evaluate_v0.py duy_rl/tests/test_evaluate.py duy_rl/README.md
git commit -m "feat: evaluate frozen Ryo cloning run"
```

---

### Task 10: Prove the pipeline end to end and run the 100-game experiment

**Files:**
- Create: `duy_rl/tests/test_integration.py`
- Modify: `duy_rl/README.md`
- Generated, not committed: `duy_rl/data/**`
- Generated, not committed: `duy_rl/runs/ryo-v0/**`

**Interfaces:**
- Consumes: every public interface from Tasks 1-9 and the binding 100-game corpus.
- Produces: synthetic integration proof; complete preparation audit; majority, clock, and state checkpoints; frozen test metrics; `duy_rl/runs/ryo-v0/REPORT.md`; evidence-backed go/no-go result.

- [ ] **Step 1: Write the synthetic end-to-end integration test**

```python
# duy_rl/tests/test_integration.py
import tempfile
import unittest
from pathlib import Path

import torch

from fixtures import make_replay
from duy_rl.constants import OPERATIONS
from duy_rl.dataset import ShardDataset, collate_examples, fit_train_artifacts
from duy_rl.features import encode_game, write_shard
from duy_rl.models import StateAwareModel, load_checkpoint, save_checkpoint
from duy_rl.replay import SourceReplay
from duy_rl.train import train_one_epoch


class IntegrationTest(unittest.TestCase):
    def source(self) -> SourceReplay:
        return SourceReplay("train", "fixture-game", Path("fixture.json"), "a" * 64, "2026-08-21", "route-a")

    def metadata(self, stats, weights) -> dict:
        return {
            "schema_version": "ryo-bc-v0",
            "feature_schema_version": "ryo-features-v0",
            "vocabularies": {"operations": list(OPERATIONS)},
            "normalization": stats,
            "class_weights": weights,
            "manifest_sha256": "b" * 64,
            "architecture": {"grid_channels": 44, "actor_dim": 38, "global_dim": 62, "classes": 17},
        }

    def test_replay_to_checkpoint_reload_has_identical_logits(self) -> None:
        replay = make_replay(hands=1)
        for state in range(1, 720):
            operation = OPERATIONS[(state - 1) % len(OPERATIONS)]
            replay["steps"][state][0]["action"]["farmer"] = [operation]
            replay["steps"][state][0]["action"]["hands"] = [[operation]]
        encoded = encode_game(self.source(), replay)
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "train" / "fixture.npz"
            write_shard(encoded, shard)
            stats, counts, weights, majority = fit_train_artifacts([shard])
            dataset = ShardDataset([shard], stats)
            batch = collate_examples([dataset[0], dataset[1]])
            model = StateAwareModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
            train_one_epoch(model, [batch], optimizer, criterion, torch.device("cpu"), "state")
            model.eval()
            before = model(batch.grid, batch.global_features, batch.actor_features)
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint, model, optimizer, self.metadata(stats, weights), 1)
            restored = StateAwareModel()
            load_checkpoint(checkpoint, restored)
            restored.eval()
            after = restored(batch.grid, batch.global_features, batch.actor_features)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
```

Keep the fixture small in memory by using only the first two dataset rows after shard creation.

- [ ] **Step 2: Run the integration test and complete new suite**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest duy_rl/tests/test_integration.py -v
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
```

Expected: the integration test and every `duy_rl` test pass.

- [ ] **Step 3: Run the existing replay-inspector regression suite**

Run:

```bash
duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_*.py' -v
```

Expected: all discovered existing tests pass with no modifications to their source.

- [ ] **Step 4: Prepare all 100 games**

Run:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/prepare_data.py --config duy_rl/configs/v0.json --output-root duy_rl/data
```

Expected: exit zero; `audit.json` reports exactly 70/15/15 games, no duplicate ID/hash or cross-split leakage, 44/38/62 dimensions, 719 state rows per game, all 17 training classes, and full validation/test operation support.

- [ ] **Step 5: Train and freeze the v0 run**

Run:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/train_v0.py --config duy_rl/configs/v0.json --data-root duy_rl/data --runs-root duy_rl/runs --run-id ryo-v0
```

Expected: exit zero; majority validation metrics exist; clock and state training histories stop by epoch 50; selected checkpoints exist; `selection.json` contains verified hashes. If batch 512 causes an observed out-of-memory error, change only `training.batch_size` to the largest verified power of two that succeeds, use a new run ID, and preserve the failed run logs.

- [ ] **Step 6: Evaluate the frozen test exactly once for model choice**

Run:

```bash
PYTHONPATH=duy_rl/src duy/.venv/bin/python duy_rl/scripts/evaluate_v0.py --data-root duy_rl/data --runs-root duy_rl/runs --run-id ryo-v0 --split test
```

Expected: exit zero; `evaluation.test.json` and `REPORT.md` contain majority, clock, and state metrics, all required slices, the 10,000-resample paired game bootstrap, all six gate checks, and one explicit go/no-go decision.

- [ ] **Step 7: Record reproducibility details in README**

Add the actual Python, NumPy, PyTorch, and device versions; exact preparation identity; selection identity; run ID; chosen batch size; selected epochs; and final `PROCEED`/`STOP` result. Link to the generated report path without committing large shards or checkpoints.

- [ ] **Step 8: Run final verification and inspect repository scope**

Run:

```bash
PYTHONPATH=duy_rl/src:duy_rl/tests duy/.venv/bin/python -m unittest discover -s duy_rl/tests -p 'test_*.py' -v
duy/.venv/bin/python -m unittest discover -s duy/another_work/02_boatlee_replay/tests -p 'test_*.py' -v
git status --short
git diff --check
```

Expected: both suites pass; `git diff --check` reports no whitespace errors; generated `duy_rl/data/` and `duy_rl/runs/` do not appear in Git status; unrelated pre-existing changes remain untouched.

- [ ] **Step 9: Commit integration proof and documentation**

```bash
git add duy_rl/tests/test_integration.py duy_rl/README.md
git commit -m "test: verify Ryo cloning pipeline end to end"
```

Do not commit generated shards, checkpoints, JSONL histories, or evaluation artifacts. The final handoff cites their local paths and reports the measured gate outcome without changing the design after seeing test labels.
