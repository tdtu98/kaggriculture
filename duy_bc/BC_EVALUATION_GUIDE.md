# Evaluating a Behavior-Cloning Model

This guide explains how Tu can evaluate an existing or new Kaggriculture
behavior-cloning (BC) model using the same validation data and metrics as the
current Duy baseline.

## What we want to measure

Raw action accuracy is useful, but it treats every decision independently. In
Kaggriculture, decisions form a sequence: an incorrect early action can move an
actor into the wrong situation and make later actions less meaningful.

We therefore rank models with three metrics, in this order:

1. **Actor Step-prefix AUC@24** — can each farmer or hand reproduce consecutive
   expert decisions?
2. **Actor Daily-gated prefix AUC** — how much of each actor's day is correct
   before its first disagreement?
3. **Action macro-F1** — does the model handle every action type, rather than
   succeeding only on frequent actions?

Raw accuracy remains in the report as supporting information.

## A small example

Suppose one hand has six decisions:

```text
Expert: NORTH, PLANT, WATER, EAST, HARVEST, DROP
Model:  NORTH, PLANT, WEST,  EAST, HARVEST, DROP
Result: correct, correct, wrong, correct, correct, correct
```

Raw accuracy is `5/6 = 83.3%`. That looks strong, but it hides the break in the
middle of the behavior sequence.

### 1. Actor Step-prefix AUC@24

For every prefix length from 1 to 24, we slide a window along each actor's
observed decisions and ask whether the **whole window** is correct. We then
average the perfect-window rates:

$$
\operatorname{StepPrefixAUC@24}
= \frac{1}{H}\sum_{h=1}^{H}
\frac{\text{perfect actor windows of length }h}
     {\text{actor windows of length }h},
\qquad H=\min(24,\text{largest available horizon}).
$$

For the six-decision example, using a shorter horizon of 4:

| Window length | Perfect windows | All windows | Rate |
|---:|---:|---:|---:|
| 1 | 5 | 6 | 0.833 |
| 2 | 3 | 5 | 0.600 |
| 3 | 1 | 4 | 0.250 |
| 4 | 0 | 3 | 0.000 |

The Step-prefix AUC@4 is `(0.833 + 0.600 + 0.250 + 0.000) / 4 = 0.421`.

This metric is stricter than raw accuracy because longer correct behavior
chains receive more credit. It is not permanently unforgiving: once a rolling
window no longer contains the incorrect decision, the model can receive credit
again.

### 2. Actor Daily-gated prefix AUC

For each actor on each game day, count decisions only until that actor's first
disagreement. Divide this correct prefix by the actor's number of observed
decisions that day, then average across all actor-days:

$$
\operatorname{DailyGatedAUC}
= \frac{1}{|D|}\sum_{d\in D}
\frac{\text{correct decisions before the first disagreement in }d}
     {\text{observed actor decisions in }d}.
$$

If the six example decisions happen on one day, the correct prefix has length
2, so the actor-day score is `2/6 = 0.333`. The gate resets on the next day.

This answers a simple question: **how far can the model follow the expert before
it first departs from the day's plan?**

### 3. Action macro-F1

For every action represented in the evaluated split, calculate its F1 score and
give each supported action equal weight:

$$
F1_c = \frac{2P_cR_c}{P_c+R_c},
\qquad
\operatorname{MacroF1}
= \frac{1}{|C_{\mathrm{supported}}|}
\sum_{c\in C_{\mathrm{supported}}}F1_c.
$$

For example, a model could predict common movement actions correctly but miss
nearly every `PLANT`, `WATER`, or `HARVEST`. Raw accuracy could still look good;
macro-F1 exposes the missing behaviors.

## Actor-level versus strict joint-farm scores

The three primary metrics are actor-level. The farmer and every hand receive
separate credit.

Imagine one environment step with three active actors:

```text
Farmer: correct
Hand 1: correct
Hand 2: wrong
```

Actor-level evaluation preserves the two correct decisions. The strict
joint-farm result for that environment step is wrong because not every actor
matched.

The report retains these strict diagnostics:

- `joint_farm_step_prefix_auc_at_24`
- `joint_farm_daily_gated_prefix_auc`

They are helpful for answering “did the whole farm match exactly?”, but they
should not be the primary model-ranking scores because farms with more hands
have more ways for a joint step to fail.

## Run validation for the existing baseline

From the baseline directory:

```bash
cd duy_bc/00_codex_baseline
```

Create the local environment once:

```bash
make setup
```

After training has completed and the run contains `selection.json`, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_v0.py \
  --data-root data \
  --runs-root runs \
  --run-id ryo-v0-core-metrics \
  --split val
```

This writes:

```text
runs/ryo-v0-core-metrics/evaluation.val.json
runs/ryo-v0-core-metrics/REPORT.val.md
```

Open `REPORT.val.md` for the readable comparison table. Use
`evaluation.val.json` when another script needs the exact values.

Do not use the test split to choose an architecture or checkpoint. Reserve it
for the final frozen evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_v0.py \
  --data-root data \
  --runs-root runs \
  --run-id ryo-v0-core-metrics \
  --split test
```

## Evaluate a new model

The metrics do not depend on the model architecture. A transformer, recurrent
model, tree model, or another neural network can use the same calculation.

Tu does **not** need to add the model architecture to Duy's evaluator. Run the
new model on the same validation split and save four arrays:

| Input | Required form | Meaning |
|---|---|---|
| `predictions` | integer shape `(N,)` | Chosen operation ID for every row |
| `game_ids` | `N` strings | Replay identity for every row |
| `step_indices` | shape `(N,)` | Environment step for every row |
| `actor_ids` | shape `(N,)` | `0` for farmer, `1+` for hand slot |

The operation IDs must use this exact order:

```text
NORTH, SOUTH, EAST, WEST, PASS, PICKUP, DROP, PLANT, WATER,
HARVEST, FERTILIZE, BUILD_PASTURE, DIG, PLACE, FEED,
COLLECT_FERTILIZER, CARE
```

For a model that returns 17 logits, create the archive like this:

```python
import numpy as np

predictions = np.argmax(logits, axis=1).astype(np.int64)

np.savez_compressed(
    "tu-transformer-v1.val.npz",
    predictions=predictions,
    game_ids=np.asarray(game_ids, dtype=str),
    step_indices=np.asarray(step_indices, dtype=np.int64),
    actor_ids=np.asarray(actor_ids, dtype=np.int64),
)
```

Do not put expert labels in this file. The evaluator reads truth from the
authenticated validation shards, which prevents accidental label mismatch or
leakage. Save `game_ids` as NumPy Unicode strings, not `dtype=object`.

Each `(game_id, step_index, actor_id)` must identify exactly one decision. Rows
may be saved in a different order because the evaluator realigns them by this
identity. For the existing encoded games, the actor slot is:

```python
actor_ids = np.rint(game.actor_features[:, 1] * 8.0).astype(np.int64)
```

Collect that value before replacing the raw features with another model's
normalization.

### Run Tu's validation evaluation

The reference baseline run must be complete and contain `selection.json`. It
defines the exact authenticated validation split used for comparison.

From `duy_bc/00_codex_baseline`, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_predictions.py \
  --predictions tu-transformer-v1.val.npz \
  --model-name tu-transformer-v1 \
  --reference-run-id ryo-v0-core-metrics \
  --data-root data \
  --runs-root runs
```

The command creates:

```text
runs/ryo-v0-core-metrics/external-tu-transformer-v1.val.json
runs/ryo-v0-core-metrics/external-tu-transformer-v1.val.md
```

The JSON contains exact values for scripts and later comparison. The Markdown
contains the primary table, strict joint-farm diagnostics, interpretation, and
per-action precision/recall/F1 table for sharing with teammates.

The command authenticates the reference selection, source validation replays,
prepared validation shards, and prediction file before publishing results. A
missing, duplicate, or extra actor-step stops evaluation instead of silently
changing the comparison set.

## How to evaluate the result

Read the generated Markdown in this order:

1. **Actor Step-prefix AUC@24:** the primary measure. Higher means the model can
   sustain longer chains of correct behavior.
2. **Actor Daily-gated prefix AUC:** higher means each actor follows more of the
   day's expert plan before its first disagreement.
3. **Action macro-F1:** higher means performance is balanced across supported
   action types, including less frequent actions.
4. **Raw accuracy:** useful context, but do not rank models from this alone.
5. **Strict joint-farm diagnostics:** inspect for whole-farm deployment risk;
   do not use these brittle scores as the primary ranking.

Compare Tu's report against majority, clock, and state results from the **same
reference run and validation split**. A model is a convincing improvement when
the first metric improves without a major regression in the second or third.

Finally, inspect the per-action table. A high overall score can still hide a
weak operation such as `PLANT`, `WATER`, or `HARVEST`.

## Built-in versus external evaluation

`evaluate_v0.py` automatically loads the three registered baselines:

- train-fitted actor-majority baseline
- clock-only model
- state-aware model

`evaluate_predictions.py` handles any other architecture through its saved
actions. The metric implementation is shared, so a new architecture requires
no evaluator code changes.

## Comparison checklist

Before comparing two models, confirm that they use:

- the same validation replay split;
- an exact match of `(game_id, step_index, actor_id)` rows;
- the same 17-operation vocabulary;
- evaluator-owned labels rather than labels from the prediction file;
- actor-level metrics as the primary ranking;
- strict joint-farm scores only as diagnostics;
- no test-set feedback during model selection.
