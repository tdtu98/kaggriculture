# Ryo Behavior-Cloning v0 Design

Date: 2026-08-24

## Purpose

Build an offline behavior-cloning baseline from 100 unique Kaggriculture
replays won by Ryo Hasegawa. The first milestone is diagnostic: determine
whether a state-aware model predicts Ryo's worker operations better than a
model that only knows the game clock and worker identity.

This milestone does not produce a playable Kaggle agent. It establishes a
validated replay pipeline, a stable feature and label contract, honest
game-level evaluation, and a measured go/no-go decision for full multi-head
behavior cloning.

## Binding inputs

- Corpus root:
  `duy_explore/ryo_hasegawa_100_stratified`
- Train: 70 games under `train/`
- Validation: 15 games under `val/`
- Test: 15 games under `test/`
- Split manifest: `manifest.csv`
- Split audit: `split_summary.json`
- Expected environment module version: `1.32.7`
- Expected season length: 720 stored states and two seats
- Expected player: exactly one seat named `Ryo Hasegawa`
- Expected outcome: Ryo has the greater terminal reward and both players have
  `DONE` status

The source replay files and the existing split are read-only. No game may move
between splits. Symlinks are accepted locally and resolved before hashing.

## Non-goals

- Do not predict action arguments or market orders in v0.
- Do not build an executor, legal-action mask, online rollout loop, or
  submission artifact.
- Do not tune on the test games.
- Do not use terminal reward, future prices, future shop unlocks,
  opponent-private state, or later observations as features.
- Do not treat high raw accuracy as evidence that the policy uses state.
- Do not modify the existing scripted agents, replay corpus, simulator, or
  arena.

## Project layout

All new work lives under `duy_rl/`:

```text
duy_rl/
├── README.md
├── DESIGN.md
├── PLAN.md
├── pyproject.toml
├── configs/
│   └── v0.json
├── src/duy_rl/
│   ├── __init__.py
│   ├── constants.py
│   ├── replay.py
│   ├── features.py
│   ├── dataset.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_v0.py
│   └── evaluate_v0.py
├── tests/
├── data/                 # generated shards and audit; ignored by Git
└── runs/                 # checkpoints and reports; ignored by Git
```

`README.md` documents the three commands needed to prepare, train, and
evaluate. `PLAN.md` is the implementation sequence derived from this design.

## Replay and label contract

Kaggle replay actions are shifted. For each Ryo observation stored at state
`t`, the action produced from it is stored at state `t + 1`. Therefore each
game contributes decision steps `0..718`:

```text
input  = steps[t][ryo_seat].observation
label  = steps[t + 1][ryo_seat].action
```

State 719 has no producing action and is never a labeled example.

For every decision step, extraction verifies that the action has one farmer
operation and exactly one hand operation per hand visible in the observation.
It emits one example for the farmer and one for each current hand. The actor
index is `0` for the farmer and `1..N` for hands in observation order.

The v0 label is the first token of the unit operation. The fixed vocabulary is:

```text
NORTH SOUTH EAST WEST PASS PICKUP PLANT WATER HARVEST FERTILIZE
BUILD_COOP BUILD_PASTURE DIG PLACE FEED COLLECT_FERTILIZER CARE
```

An unknown operation is a hard error. Action arguments are preserved as
integer-coded metadata for later work but do not contribute to v0 loss or
metrics.

## Leakage controls and validation

Preparation fails on any of the following:

- malformed JSON;
- wrong module version or configuration;
- anything other than 720 states and two seats;
- missing observation or action fields;
- Ryo missing, duplicated, or not the winner;
- terminal status other than `DONE/DONE`;
- hand/action-count mismatch after the one-step action shift;
- unknown operation;
- duplicate episode ID or resolved-file SHA-256;
- an episode or hash appearing in more than one split;
- a source file whose hash differs from the split manifest.

Normalization statistics, class counts, class weights, and vocabularies are
fit from train only. Validation and test use the frozen train artifacts.

## Feature contract

Every feature must be derivable from the current delivered observation.
Categorical vocabularies are fixed in `constants.py`; unexpected categorical
values fail rather than silently mapping to a misleading class.

### Tile grid

The model receives both public 10x10 farms as channel-first float32 tensors.
Each farm encodes:

- tile kind: empty, locked, weed, plant, coop, pasture;
- crop: wheat, carrot, tomato, strawberry, melon;
- animal: goose, cow, sheep;
- normalized yield;
- watered, fed, cared, and fertilizer-available flags;
- normalized consecutive-unwatered and consecutive-unfed values;
- whether the tile is the current actor position.

The two farms are concatenated in stable `self, opponent` order regardless of
Ryo's seat. Each farm has 22 channels, so `C = 44`. The generated data audit
publishes this count and tests assert it.

### Actor features

- farmer-versus-hand flag;
- daily hand index;
- normalized x/y position;
- carried inventory for the fixed product, animal, and fertilizer vocabulary;
- whether the actor is adjacent to the shed;
- current-tile categorical and scalar features.

The carried-inventory vocabulary contains the nine products (five crops,
egg, milk, wool, and fertilizer) plus goose, cow, and sheep. The current-tile
encoding omits the actor-position channel already represented above, making
`A = 38` actor features.

### Global features

- normalized step, day, and hour plus cyclic hour encoding;
- Ryo seat;
- log-scaled self and opponent money;
- hand counts and hires-today for both farms;
- unlocked-quadrant flags for both farms;
- Ryo's private shed and seed counts;
- current market inventory and price for every product;
- counts of currently unlocked shops.

The shop vocabulary is `BAKERY`, `BRUNCH_SPOT`, `ICE_CREAM_SHOP`, `PET_CAFE`,
`PIZZA_SHOP`, `SMOOTHIE_SHOP`, and `YARN_STORE`. With nine market products,
five seed counts, twelve shed item counts, and the fields above, `G = 62`.

No sample includes reward or any field read from `steps[t + 1].observation`.

## Shard format

Preparation writes one compressed NumPy `.npz` shard per replay. A shard keeps
step-level state separate from unit-level examples to avoid duplicating the
10x10 grid for every worker:

- `grid`: `[719, C, 10, 10]`, float32;
- `global_features`: `[719, G]`, float32;
- `actor_features`: `[U, A]`, float32;
- `step_index`: `[U]`, int32, mapping each unit example to step state;
- `label`: `[U]`, int64 operation index;
- argument metadata arrays for later phases;
- stable JSON metadata containing schema version, split, episode ID, Ryo seat,
  source path, source SHA-256, sample count, and tensor shapes.

Preparation is deterministic. Running it twice from identical source bytes
must produce identical logical arrays and audit JSON. Zip container timestamps
are not used as an identity claim; identity is computed from array bytes and
canonical metadata.

## Baselines and model

All three systems consume exactly the same unit examples and splits.

### Majority baseline

Predict the most frequent train operation globally. Also report a stronger
stratified floor using the most frequent train operation for farmer versus
hand. Neither baseline reads validation or test labels during fitting.

### Clock-only baseline

A small MLP receives only step/day/hour, cyclic hour, seat, actor type, and hand
index. It is deliberately capable of learning a fixed schedule. Its purpose is
to expose a teacher whose actions are largely clock-driven.

### State-aware model

- Tile encoder: `Conv2d(44, 32, 3, padding=1)`, ReLU,
  `Conv2d(32, 64, 3, padding=1)`, ReLU, and adaptive 2x2 average pooling,
  producing 256 features;
- actor encoder: `38 -> 64 -> 64` with ReLU;
- global encoder: `62 -> 128 -> 128` with ReLU;
- classifier: concatenated 448 features -> 256 -> 128 -> 17, with ReLU and
  dropout 0.1 after the 256-unit layer;
- approximately 212,000 trainable parameters;
- 17 output logits.

The v0 model stays intentionally small so it trains on Apple Silicon and so a
failure cannot be blamed on an unnecessarily complex architecture.

## Training contract

- Framework: PyTorch on Python 3.12.
- Local device preference: MPS, then CUDA, then CPU.
- Current prerequisite: PyTorch is not installed in `duy/.venv`; installation
  is an explicit implementation step and must be followed by existing project
  regression tests.
- Default seed: `20260824`.
- Optimizer: AdamW.
- Initial learning rate: `1e-3`.
- Batch size: 512, lowered only for an observed memory failure.
- Maximum epochs: 50.
- Early-stopping patience: 5 epochs.
- Selection metric: validation macro-F1; validation loss breaks ties.
- Loss: cross-entropy with train-derived inverse-square-root class weights,
  normalized to mean one and capped at four.

The checkpoint contains model weights, model/config schema versions, feature
vocabularies, train normalization statistics, class weights, source manifest
hash, and code-visible architecture parameters. Resume checkpoints also carry
optimizer state, epoch, and random generator states.

The selected checkpoint and exact configuration are frozen in
`runs/<run_id>/selection.json` before test evaluation. The evaluator verifies
their hashes and refuses an unfrozen checkpoint.

## Metrics and statistical unit

Report all metrics separately for majority, clock-only, and state-aware:

- natural-distribution top-1 accuracy;
- top-3 accuracy;
- macro-F1;
- per-class precision, recall, F1, and support;
- confusion matrix;
- metrics by farmer/hand, seat, in-game day band, source date, and route
  family.

Unit decisions within a game are correlated, so confidence intervals and
model comparisons resample games, not individual worker rows. Use a
deterministic 10,000-resample paired game bootstrap with seed `20260824`.

## Test-set discipline and success gate

Validation games may be evaluated every epoch. Test games are evaluated only
after the state-aware checkpoint, clock-only checkpoint, configuration, and
normalization artifacts are frozen. Re-running the identical frozen test is
permitted for reproducibility; changing inputs creates a new run identity.

The v0 milestone passes only when all of these are true:

1. all preparation and leakage checks pass;
2. every operation present in validation and test is represented in train;
3. state-aware validation macro-F1 exceeds clock-only validation macro-F1;
4. on test, state-aware macro-F1 exceeds clock-only macro-F1;
5. the paired game-bootstrap 95% lower bound for state-aware minus clock-only
   top-1 accuracy is greater than zero;
6. the result report includes every baseline and diagnostic slice.

If the gate fails, do not add action arguments, market heads, attention, or
online play. Diagnose data alignment, representation, class imbalance, and
clock dependence first. A failure may correctly show that these Ryo wins are
not suitable state-responsive demonstrations.

## Error handling and observability

CLI commands return non-zero on invalid data, incompatible artifacts, NaN
loss, empty classes, hash mismatch, or leakage. Errors identify the split,
episode, step, seat, and actor where applicable.

Preparation writes `data/audit.json` with source hashes, shard identities,
sample counts, label counts, tensor shapes, split totals, and all validation
checks. Training writes JSONL epoch metrics and deterministic selection
metadata. Evaluation writes machine-readable JSON plus a concise Markdown
report. Generated artifacts never overwrite a non-matching run silently.

## Testing strategy

Implementation follows test-driven development.

Unit tests cover:

- stored-action shifting and terminal exclusion;
- Ryo-seat and winner validation;
- farmer/hand alignment;
- feature shapes and self/opponent seat normalization;
- absence of future and reward fields;
- fixed-vocabulary rejection;
- train-only normalization and class weights;
- shard round-trip and identity;
- split/hash leakage detection;
- model output shape and deterministic checkpoint reload;
- metric and paired-bootstrap calculations;
- frozen-test enforcement.

An integration fixture runs synthetic replay -> shard -> dataset -> one
training step -> checkpoint -> reload -> identical logits. A small real-corpus
smoke test prepares one game from each split. The final verification runs the
new suite plus the existing replay-inspector regression suite.

## Deliverables for v0

- documented and tested replay preparation pipeline;
- deterministic train/validation/test shards and audit;
- majority and clock-only baseline reports;
- trained state-aware checkpoint and training history;
- frozen test evaluation and statistical comparison;
- written decision to proceed to multi-head cloning or stop with evidence.
