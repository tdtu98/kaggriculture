# `model/` — the behavior-cloning pipeline and model

This package implements Chapters 3–6 of [`JOURNEY.md`](../JOURNEY.md): Kaggle replays → verified
training data → the v1 pointer-MLP that clones Ryo Hasegawa's play → the wrapper that lets it play
full games. The detailed spec lives in [`PLAN_BC.md`](../PLAN_BC.md) (same chapter numbers).

**Status:** data pipeline built and verified (all gates zero-fail), v1 model trained
(`checkpoints/bc_v1.pt`, 534,704 params, epoch 54 of 59 by early stopping, every head beats its
floor), Chapter 6 wrapper built and measured. It **beats `starter` 40/40** (mean $69,212 against
`starter`'s ~$3.5k). Registered as the arena agent `bc_v1` (D22 ii). Against `boatlee` it is
**still 0/80** (E101), but the money gap roughly halved vs the previous checkpoint (mean $46,526
vs boatlee's $118,745); Chapter 7 (DAgger) is the answer to the remaining gap — see "Where the
clone loses money" below.

`checkpoints/bc_v1_8ep.pt` is the superseded pre-fix checkpoint, kept so its published numbers stay
reproducible. It loads with `ar_gate=False` (see `checkpoint.LEGACY_NET_CONFIG`) and replays
exactly as it was measured.

## Setup (one time)

Everything runs inside the repo's own venv at `.venv/` — never the system python or miniconda
(miniconda deliberately pins the *older* `kaggle-environments` 1.32.6, which has a different
market curve; running against it silently measures the wrong game — see CLAUDE.md E54).

```bash
# 1. Build the venv + game env + Rust simulator toolchain (from the repo root).
#    Installs kaggle-environments==1.32.7, maturin, pytest, numpy, scipy into .venv/.
make setup

# 2. Build the Rust simulator (kagsim) into the venv — the fast backend play.py uses.
make sim

# 3. PyTorch is NOT in make setup (the rest of the repo doesn't need it). Add it:
.venv/bin/pip install torch

# 4. Verify the world is sane before trusting any number from it:
.venv/bin/python -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available())"
make test      # repo-wide suite, ~2 min
make verify    # kagsim-vs-reference parity gate: must report 0 divergences
```

Expected: torch ≥ 2.x with `mps: True` on Apple Silicon, all tests green, `simulator
divergences: 0`. The training data must sit at `data/sample_data_training_model/{train,val,test}/`
(100 replay JSONs + `manifest.csv`) — it ships with the repo copy, no download step.

## How the pieces fit

```
data/sample_data_training_model/*/*.json     (100 Kaggle replays, Ryo's games)
        │
        ▼
decode.py ──── uses masks.py (legality) + vocab.py (action alphabet)
        │      four hard assertions; stops on any suspicious data
        ▼
features.py    (observation → tile/worker/product/global/opponent tokens)
        │
        ▼
build_shards.py → data/shards/{train,val,test}/*.npz   + verification report
        │
        ▼
batching.py    (shards → torch batches, masks as data)
        │
        ▼
net.py + losses.py + train.py → checkpoints/bc_v1.pt
        │
        ▼
agent.py       free-running decode: checkpoint → agent(obs) → action
        │      (masks.py again, this time as the inference mask)
        ▼
play.py        N episodes vs a named opponent, both seats, Wilson interval, Observer counters
```

## File map

| File | What it does |
|---|---|
| `vocab.py` | Every constant in one place: 18 unit verbs, 12 items, 6 market ops, quantity bins, `MAX_UNITS=16`, `MAX_MARKET_ORDERS=10`. The single source of truth for head widths. |
| `decode.py` | Replay JSON → clean (state, action) pairs for Ryo's seat. Applies the off-by-one pairing `(obs[i-1], action[i])`, patches seat-1's missing `step`, computes the effective shed (with the animal carve-out), and enforces the four assertions from PLAN_BC Chapter 3. `--include-opponent-seat` exists but defaults off. |
| `masks.py` | The one shared legality function `compute_masks(obs, decisions_so_far)` → boolean arrays. Used by the decoder to prove the expert is never masked out; the same function will drive inference in Chapter 6. |
| `features.py` | Tokenizer: observation → tile `[100,·]`, worker `[16,·]`, product `[9,·]`, global, opponent-summary arrays. `FEATURE_VERSION` is stamped into every shard and checkpoint. |
| `dataset.py` | Shard writer/loader — one compressed `.npz` per episode, state-major layout. |
| `build_shards.py` | CLI that runs the whole pipeline over all 100 games and prints the verification report (counters, E87 probe, majority floors, shard sizes). Exit 1 if any gate fails. |
| `batching.py` | Shards → torch batches: dense label planes, masks as input tensors. See `MASK_NOTE` inside for what is and isn't masked at train time. |
| `baseline.py` | Chapter 4's tiny model: logistic regression, 20 hand-picked features, one head. Exists to prove the pipeline and to set the bar the real model must clear. |
| `net.py` | The v1 pointer-MLP (~535k params): shared-weight encoders → context vector → per-worker decision chain (commit → tile pointer → verb → item → qty) with the autoregressive seed-budget state, then market heads, plus the value head for the later RL stage. |
| `losses.py` | Masked cross-entropy per head (label −1 = ignore) + 0.1 × value MSE. |
| `train.py` | Training loop (MPS or CPU). Prints per-head **train and val** accuracy with Wilson intervals **next to the majority floor**, for all steps and steps ≥ 32 separately, plus a `gap` column. Early-stops on the mean per-head val top-1 and keeps the best epoch. |
| `checkpoint.py` | Save/load with `FEATURE_VERSION`, vocab hash, shard-manifest hash, and train config — a checkpoint refuses to load against mismatched features. |
| `agent.py` | **Chapter 6.** `make_agent(checkpoint) -> agent(obs)`. Inverts `decode.py`'s macro segmentation: an idle worker is asked for one "walk to tile T then do V", the macro is remembered, and the worker emits one move a turn until it arrives and fires. Greedy argmax, `masks.compute_masks` on every head, a post-hoc legality re-check that falls back to `PASS`, and 22 counters. |
| `play.py` | **Chapter 6's ladder.** `N` episodes vs a named opponent, both seats, per-game lines, a Wilson interval on the winrate, the wrapper's counters and `harness/counters.py`'s `Observer`. Two backends: `kagsim` (fast, Observer) and `env` (`kaggle_environments`, the only one that reports `status`). |

Tests live in `../tests/`: `test_bc_decode.py` (27 tests — the decoding contract, including
mutation tests proving the assertions can fail), `test_bc_model.py` (25 tests — shapes, the
masked-zero-probability check, overfit, checkpoint roundtrip, determinism, early stopping, and the
autoregressive gradient-explosion regression with its mutation control) and `test_bc_agent.py`
(16 tests — the free-running/teacher-forced parity gate and its mutation control, macro
bookkeeping over a real episode, the nightly-wipe clear, and a poisoned mask that must move
`n_fallback_pass`).

## Reproducing the whole flow, step by step

All commands run from the repo root with `PYTHONPATH=.` and the repo's venv. Running steps 1→5 in
order reproduces every number in this README from a fresh clone (after Setup above). Total time on
an Apple Silicon Mac: roughly 1 hour, most of it step 5's games.

### Step 1 — Build the training data (replays → shards)

What happens: `build_shards.py` streams each of the 100 replay JSONs, pairs each action with the
observation it was chosen from (the off-by-one contract), keeps Ryo's seat (from `manifest.csv`),
runs the four integrity assertions, tokenizes every state through `features.py`, and writes one
compressed `.npz` shard per episode.

```bash
PYTHONPATH=. .venv/bin/python -m model.build_shards
```

- Input: `data/sample_data_training_model/{train,val,test}/*.json` + `manifest.csv`
- Output: `data/shards/{train,val,test}/*.npz` (~28 MB total) + a verification report
- Takes: ~90 s. **Exit code 1 if any gate fails — never train on a failed build.**
- Expect in the report: train 50,330 states / val 10,785 / test 10,785; all four assertion
  counters at 0 fails; `n_expert_actions_rejected_by_mask: 0`; `frac_segments_shortest_path ≈
  0.992` (E87); majority floors ≈ 33.7% macro verb / 16.6% raw.

Quick smoke variant (5 episodes, writes nothing): append `--limit 5 --no-write`.

### Step 2 — The sanity baseline (before any real model)

What happens: a ~300-parameter logistic regression on 20 hand-picked features predicts the macro
verb. If this can't clearly beat the majority-class floor, the data pipeline is broken and no
bigger model is trustworthy.

```bash
PYTHONPATH=. .venv/bin/python -m model.baseline
```

- Takes: ~1 min. Expect: ~84% vs the ~34% floor.

### Step 3 — Train the v1 model

What happens: `train.py` loads the shards, batches decisions, and trains all nine heads + the
value head with masked cross-entropy, teacher-forced. Every epoch it prints train AND val accuracy
per head (Wilson intervals, beside the majority floor, all-steps and steps≥32) plus a `gap`
column, and a discarded-gradient-batch counter that must read 0. It early-stops when mean val
accuracy plateaus and saves the BEST epoch's weights, not the last.

```bash
PYTHONPATH=. .venv/bin/python -m model.train --max-epochs 60 --early-stop-patience 5 \
  --batch-size 256 --lr 3e-4 --device mps \
  --out-dir checkpoints --history-json checkpoints/history.json
```

- Input: `data/shards/train` (fit) + `data/shards/val` (early stopping)
- Output: `checkpoints/bc_v1.pt` (~2 MB) + `checkpoints/history.json` (full epoch curve)
- Takes: ~25 s/epoch; stopped at epoch 59 (best 54) ≈ 28 min on the published run
- Expect at the best epoch: tile ≈ 70% top-1 / 88% top-3, verb ≈ 99%, market op ≈ 94%
- Caveat from the design review: past ~epoch 25, offline accuracy and actual game money
  **anti-correlate** (sharper ≠ better play). Checkpoint choice should ultimately be settled by
  step 5's games, not this table.

Overfit smoke first if anything looks off (proves gradients reach every head; expect >90% train
verb accuracy in ~1 min):

```bash
PYTHONPATH=. .venv/bin/python -m model.train --limit-episodes 2 --val-split train \
  --epochs 50 --early-stop-patience 0 --batch-size 128 --lr 1e-3 --device mps --log-every 0
```

`--early-stop-patience 0` disables early stopping; `--epochs` is an alias for `--max-epochs`.

### Step 4 — Test the code (fast, offline)

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_bc_decode.py tests/test_bc_model.py tests/test_bc_agent.py -q
```

68 BC tests: the decoding contract (with mutation tests proving each assertion can actually
fail), model shapes, the masked-zero-probability check, checkpoint roundtrip, determinism, the
gradient-explosion regression, and the wrapper's free-running/teacher-forced parity gate. The full
repo suite is `PYTHONPATH=. .venv/bin/pytest tests/ -q` (~2 min).

### Step 5 — Test the model in real games (online evaluation)

What happens: `play.py` wraps the checkpoint as a live `agent(obs)` (free-running decode, live
legality masks, scripted mover), plays N full 720-step episodes against a named opponent,
alternating seats, and prints per-game money, the winrate with a Wilson interval, the wrapper's
22 counters, and the `Observer` diagnostics (idle %, plants lost, unharvested…).

Liveness check in the real Kaggle environment first (2 games, ~1 min):

```bash
PYTHONPATH=. .venv/bin/python -m model.play --opponent starter --games 2 \
  --seeds 40000:40001 --backend env --workers 1
```

Expect: both seats `DONE`, bank well above the $3,000 start. Then the measurements (kagsim
backend, ~2.4 s/game):

```bash
PYTHONPATH=. .venv/bin/python -m model.play --opponent starter --games 40 --seeds 42000:42020
```

```bash
PYTHONPATH=. .venv/bin/python -m model.play --opponent boatlee --games 80 \
  --seeds 45000:45040 --jsonl results/bc_v1_play.jsonl
```

Expect with `bc_v1.pt`: starter 40/40, mean ≈ $69k; boatlee 0/80, mean ≈ $46.5k (E101). Rules
that make these numbers meaningful: **≥80 games, both seats, a seed block never used for tuning**
— single games are noise.

The same match through the shared harness prints *both* agents' `Observer` counters side by side —
that table is the diagnosis, the money is only the symptom:

```bash
PYTHONPATH=. .venv/bin/python harness/run.py --agents bc_v1,boatlee --seeds 46000:46040 --games 80
```

### Optional — the recorded ablations

The claim-mode ablation (E100) that set the wrapper's coordination default:

```bash
for m in off verb tile; do
  PYTHONPATH=. .venv/bin/python -m model.play --opponent starter --games 40 \
    --seeds 42000:42020 --claims $m --label "claims=$m"
done
```

Every measurement above is recorded with an E-number in [`docs/experiments.md`](../docs/experiments.md)
(E87–E103) — check there before re-deriving a conclusion.

## Reading the numbers

Every epoch prints one table: **train and val side by side, per head, both step views, each with a
Wilson interval and its majority floor.** Read it in this order.

- **An accuracy number means nothing alone** — always read it against the majority-class floor
  printed beside it (Chapter 4's lesson). Current v1 val results, steps ≥ 32, at the selected
  epoch 54:

  | head | val top-1 | floor | head | val top-1 | floor |
  |---|---|---|---|---|---|
  | macro commit | 99.87% | 93.30% | market stop | 93.94% | 50.32% |
  | macro tile | 70.44% (t3 88.4%) | 7.27% | market op | 94.14% | 43.22% |
  | macro verb | 98.88% | 34.36% | market item | 91.79% | 43.53% |
  | macro item | 99.09% | 87.03% | market qty | 85.24% | 59.87% |
  | macro qty | 98.18% | 97.11% | **mean** | **92.40%** | |
- **All-steps vs steps ≥ 32**: Ryo's first ~30 steps are a fixed opening, so metrics are reported
  both ways. If the two ever diverge sharply, the model is memorizing the clock, not learning play.
- **The `gap` column is `train − val`**, and it is how you tell what to do next. The train column is
  a *fixed 2,013-state subsample* of the training split (~10,000 macro decisions, ±1% at 95%) — the
  same states every epoch, so a moving gap is the model moving, not the sample.

  | gap | reading | what to do |
  |---|---|---|
  | ≈ 0 and both still climbing | **underfit** — capacity or epochs, not data | train longer, or widen `d_model` |
  | ≈ 0 and both flat | **fit to the data's limit** | more data, or a better action space — not a bigger model |
  | growing, val flat or falling | **overfit** | early stopping already handles it; then weight decay / fewer params |

  With one teacher and 250,084 training decisions, a large positive gap is the expected failure —
  PLAN_BC Ch5 sizes v1 at ~430k parameters precisely because "the dataset binds us, not the
  compute". A *negative* gap of a few tenths of a percent is just sampling noise between two
  different sets of episodes.
- **Early stopping** watches the **mean of the nine per-head val top-1 accuracies, equally
  weighted, on steps ≥ 32** — printed each epoch as `mean per-head top-1`. Equal weighting is
  deliberate: by decision count the macro heads outnumber the market heads five to one, and the
  market half is the half PLAN_BC says a learned model actually belongs in. The best epoch's
  weights are saved to `--out-dir` and **restored into the returned model** — training past the
  optimum and then shipping the last epoch quietly hands the arena a worse model than the one that
  was measured. The full per-epoch train+val curve lands in `--history-json`.
- **`discarded-gradient batches N/196` must read 0.** `clip_grad_norm_` scales gradients by
  `max_norm / total_norm`, so a non-finite norm makes that coefficient **zero** and the batch
  teaches the model nothing — while the loss still falls on the batches that survive. That defect
  (an ungated LayerNorm over the all-zero autoregressive start state, amplifying gradients ~316×
  per idle worker slot) cost the first v1 run most of its batches and was invisible in every
  number it printed. It is now counted, and `tests/test_bc_model.py` pins it with a mutation
  control.
- **These are offline numbers.** High agreement does not mean the agent plays well — that is
  Chapter 6's lesson, and the online numbers below are what it cost.

## Chapter 6's online numbers

| opponent | checkpoint | winrate (95% Wilson) | my mean $ | their mean $ | n |
|---|---|---|---|---|---|
| `starter` | **`bc_v1.pt` (epoch 54)** | **100.0% [91.2, 100.0]** | **69,212** | 3,551 | 40 |
| `starter` | `bc_v1_8ep.pt` (superseded) | 95.0% [83.5, 98.6] | 38,323 | 3,546 | 40 |
| `boatlee` | **`bc_v1.pt` (epoch 54)** | 0.0% [0.0, 4.6] | **46,526** | 118,745 | 80 |
| `boatlee` | `bc_v1_8ep.pt` (superseded) | 0.0% [0.0, 4.6] | 17,362 | 139,191 | 80 |

Both seats; `starter` rows use seed block `42000:42020`, `boatlee` rows `45000:45040`, identical
settings for both checkpoints. The rematch (E101 in `docs/experiments.md`) is still a 0/80 loss,
but the money gap roughly halved: our mean 2.7×'d to $46,526 (max $89,363) while boatlee's fell
$139k → $119k — our expanded production pressures the shared market. Maintenance coverage is still
the binding constraint (239 plants started/ep but 33 lost and 41 left unharvested).

The two `starter` rows are the same 40 episodes, and the gap between them is what the training fix
below was worth: mean money +81%, the worst episode $347 → $32,841, and 38/40 → **40/40** above
the $3,000 starting bank. The wrapper counters move the same way — `idle_pct` 0.187 → **0.085**,
`steps_per_useful` 1.458 → **1.300**, `n_fallback_pass` 21.8 → **7.0** per episode,
`fertilize_ops` 13.9 → **48.2**, `unharvested_ripe_at_end` 20.7 → **13.2**. `blocked_ops` stays at
0: the masks were never the problem.

Six episodes re-run on `kaggle_environments` reproduce the kagsim money **to the dollar**, and
re-running three seeds reproduces both the money and every counter — greedy decode is
deterministic.

### Where the clone loses money

`harness/run.py`'s counter table, per game, is the diagnosis. Money is the symptom.

**These rows are the superseded `bc_v1_8ep.pt` against `boatlee`.** The epoch-54 rematch (E101)
confirms the direction: the current checkpoint's Observer reads idle_pct 0.050,
steps_per_useful 1.23 (boatlee: 1.02), fertilize_ops 29.9, plants_lost_thirst 11.2,
unharvested_ripe_at_end 40.5 — much better farming at much larger scale, but the care/harvest
loop still doesn't cover the board. The magnitudes below are the old checkpoint's.

| counter | boatlee | bc_v1 | reading |
|---|---|---|---|
| `blocked_ops` | 11 | **0** | the masks work; the clone never wastes an op on something illegal |
| `steps_per_useful` | 1.02 | **1.59** | 59% more walking per useful op — the 61% tile pointer |
| `idle_pct` | 15% | **26%** | a fifth of the roster does nothing on a given turn |
| `strawberry_per_plant` | 7.9 | **1.5** | the same plants yield a fifth as much |
| `fertilize_hits` | 68 | **1** | the strawberry fertilize window is essentially never hit |
| `milk_per_cow_day` | 1.11 | **0.49** | animals are fed and cared for about half the time |
| `plants_lost_thirst` | 0.0 | **3.8** | plants the routing failed |
| `unharvested_ripe_at_end` | 2 | **13** | no end-of-season liquidation |

**It is not the opening and it is not the market.** The clone hires like the expert (285 vs 289
`HIRE` orders a season) and its order book is the right shape. It plants roughly as much
(95 plants started a season). What it cannot do is *maintain* — the expert's care loop is watering
every plant every other day, fertilizing strawberry at ages 9 and 13, feeding and caring for every
animal, and it is a long-horizon habit rather than a single decision. At 61% top-1 pointer accuracy
about two in five walks end at the wrong tile, coverage of "everything gets watered today" decays,
plants dry out, the board leaves the distribution the expert ever demonstrated, and it never comes
back. That is the **T²·ε** bound with T = 719, and it is exactly what Chapter 7 (DAgger) exists to
fix: relabel *our own* rollout states with the expert.

## Known limitations (deliberate, documented)

1. **Train-time masks are the fixed alphabets only** (verb→item, op→item/qty from `vocab.py`).
   State-dependent legality (`masks.py`) is applied at *inference*, not during training — the
   shards do not store per-decision masks. If Chapter 6 shows wasted probability on illegal verbs,
   adding masks to the shards is the known fix. See `batching.MASK_NOTE`.
2. **`net.py`'s forward is the teacher-forced pass** (the expert's earlier decisions drive the AR
   state). Free-running decode — the model feeding itself — is Chapter 6's wrapper.
3. The value head trains against the terminal money margin but is unused at play time; it exists
   for the Chapter 8 RL stage.
