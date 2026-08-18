# The env lives in this repo's own venv, not the shared miniconda: PLAN_v4 needs
# kaggle-environments 1.32.7 (hinge prices) while ../kaggriculture stays pinned to 1.32.6, and
# both import the same `kagsim` name. `make setup` builds it from scratch.
VENV := $(CURDIR)/.venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
CARGO_ENV := . "$$HOME/.cargo/env";

.PHONY: help setup sim test test-rust test-py bench clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv (env 1.32.7) and install the toolchain (rust, maturin, pytest)
	@command -v rustc >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
	@test -x $(PY) || /opt/miniconda3/bin/python -m venv $(VENV)
	$(PIP) install -q "kaggle-environments==1.32.7" maturin pytest pytest-xdist numpy scipy cma

sim:  ## Build the Rust simulator into .venv
	$(CARGO_ENV) cd kagsim && VIRTUAL_ENV=$(VENV) $(VENV)/bin/maturin develop --release

test-rust:  ## cargo unit tests (rule-level invariants)
	$(CARGO_ENV) cd kagsim && cargo test --lib

test-py:  ## pytest (rng parity, env parity, scenarios)
	$(PY) -m pytest tests/ -q

test: test-rust test-py  ## everything

verify:  ## THE parity gate: every simulation line of the reference differentially tested
	PYTHONPATH=. $(PY) tools/coverage_audit.py --fail

submission:  ## build submission.tar.gz (import check + full-game smoke test as seat 1)
	PYTHONPATH=. $(PY) tools/build_submission.py

search:  ## joint CEM over agent/params.py, validated on held-out seeds
	PYTHONPATH=. $(PY) search/cem.py --gens 10 --pop 24 --seeds 16

promote:  ## gate a candidate into champion.json (3 stages, fresh seeds) — see docs/decisions.md D19
	PYTHONPATH=. $(PY) tools/promote.py search/best_params.json

audit-champion:  ## does the sitting champion still beat everything we have?
	PYTHONPATH=. $(PY) tools/promote.py --audit

gauntlet:  ## champion vs every registered agent
	PYTHONPATH=. $(PY) arena/run.py --all --gauntlet champion --games 12

arena:  ## round-robin tournament over the registry (see arena/registry.py)
	PYTHONPATH=. $(PY) arena/run.py --all --games 16 --html arena/report.html

audit:  ## rule-level coverage report (fuzz + engine-driven play)
	PYTHONPATH=. $(PY) tools/audit.py 4 718

bench:  ## steps/s: reference env vs kagsim
	$(PY) tools/bench.py

clean:
	$(CARGO_ENV) cd kagsim && cargo clean
	rm -rf .pytest_cache tests/__pycache__
