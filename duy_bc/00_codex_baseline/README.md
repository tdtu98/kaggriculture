# Duy BC v0 baseline

This directory is a standalone source handoff of Duy's validated
behavior-cloning baseline. It contains the majority, clock-only, and
state-aware models together with the authenticated preparation, training,
checkpoint-selection, and frozen-test evaluation pipeline.

No replay data or generated model artifacts are committed here.

## Quick start

Requirements:

- Python 3.12
- A local copy of the stratified 100-win Ryo Hasegawa replay corpus
- A shell with `make`

Edit only `corpus_root` in [`configs/v0.json`](configs/v0.json) to point to
your replay corpus. A relative path is resolved from the repository root that
contains `duy_bc/`; an absolute path can point anywhere on your machine.

Then run:

```bash
make setup
make reproduce
```

`make setup` creates `.venv/` and installs this project. `make reproduce`
prepares the configured corpus, fits the train-only majority rules, trains the
clock and state models, freezes validation-selected checkpoints, and performs
the authenticated test evaluation.

The default final outputs are:

```text
runs/ryo-v0/evaluation.test.json
runs/ryo-v0/REPORT.md
```

Prepared data lives under `data/`; checkpoints and reports live under
`runs/`. Both directories are ignored by Git.

## Model organization

- [`src/model/majority.py`](src/model/majority.py) owns the train-fitted global,
  farmer, and hand majority rankings and their prediction behavior.
- [`src/model/clock.py`](src/model/clock.py) owns `ClockOnlyModel`, which sees
  only the eight clock and acting-player identity features.
- [`src/model/state.py`](src/model/state.py) owns `StateAwareModel`, which
  combines grid, global-state, and actor-state encoders.
- [`src/bc_core/`](src/bc_core/) owns replay validation, feature encoding,
  datasets, training, checkpoint authentication, metrics, and frozen
  evaluation.

The baseline project does not import the separate root-level `duy_rl/`
package.

## Restart and compatibility behavior

`make reproduce` is safe to rerun with identical inputs:

- Preparation accepts existing shards and audits only when their authenticated
  logical content is identical.
- Training resumes when the run already contains the published
  `train_artifacts.npz` prefix.
- Resume validates the configuration, data audit, model architecture, complete
  checkpoint/history prefix, optimizer state, and random-number state before
  training continues.
- A completed compatible run returns its existing frozen selection and accepts
  an identical evaluation result without rewriting it.
- Missing manifests, changed schemas, incompatible checkpoints, unexpected run
  files, or authentication failures stop with an error instead of silently
  changing the experiment.

If you intentionally change a frozen input, use a new run ID:

```bash
make reproduce RUN_ID=tu-v0
```

## Focused commands

Use these when you do not want the complete pipeline:

```bash
make prepare
make train RUN_ID=tu-v0
make evaluate RUN_ID=tu-v0
make test
```

`make train` starts a fresh focused run. For automatic compatible resume and
the complete frozen test workflow, use `make reproduce`.

Paths and identifiers can be overridden without editing the Makefile:

```bash
make reproduce \
  CONFIG=configs/v0.json \
  DATA_ROOT=data \
  RUNS_ROOT=runs \
  RUN_ID=tu-v0
```

## Existing validated result

The following numbers are reference results from Duy's existing 100-win run;
they were not newly produced while packaging this handoff.

- Run ID: `ryo-v0-locality`
- Selected epochs: clock `32`, state `37`
- State top-1: `0.830836`
- State top-3: `0.986921`
- State macro-F1: `0.873457`
- State-minus-clock top-1 delta: `0.539658`
- Paired 95% confidence interval: `[0.531252, 0.547416]`
- Decision: `PROCEED TO MULTI-HEAD CLONING`

All six fixed evaluation gates passed in that existing run.

## Historical design material

[`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/PLAN.md`](docs/PLAN.md) preserve
the original baseline design and implementation record. Their older command
examples refer to the source project's former location; use the commands in
this README for the standalone handoff.
