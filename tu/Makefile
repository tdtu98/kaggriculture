PY  := /opt/miniconda3/bin/python
PIP := /opt/miniconda3/bin/pip
CARGO_ENV := . "$$HOME/.cargo/env";

.PHONY: help setup sim test test-rust test-py bench clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install toolchain (rust, maturin, pytest)
	@command -v rustc >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
	$(PIP) install -q maturin pytest pytest-xdist

sim:  ## Build the Rust simulator into the miniconda env
	$(CARGO_ENV) cd kagsim && CONDA_PREFIX=/opt/miniconda3 /opt/miniconda3/bin/maturin develop --release

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
