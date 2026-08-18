"""Detect a change to the reference environment (V6).

Everything in this project is measured on kagsim, which is a *reimplementation* of
`kaggle_environments/envs/kaggriculture/`. Its correctness is a claim about one specific version
of that file, not a permanent property. If Kaggle ships a rule change and we do not notice, kagsim
keeps running happily and every number it produces becomes confidently wrong -- the arena, CEM, the
champion, all of it -- with nothing failing to signal it.

So the source is fingerprinted. When these hashes stop matching, the parity work has to be redone
against the new source before any result is trusted; do not simply update the constants.

Deliberately hashing the *source*, not just the version string: a patch release can change
behaviour without changing what `importlib.metadata` reports, and the environment file is what
kagsim actually mirrors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os

import kaggle_environments
import pytest

# Verified state: 100% of simulation lines differentially covered, 0 divergences, and bit-exact
# against a third-party agent on money, action counts and routing metrics (E27, E28).
#
# History -- this guard has fired twice, in anger, and it worked both times:
#   1.32.4 -> 1.32.6 (E33). Kaggle cut town-centre demand ~4.7x, made shop unlocks draw WITH
#   replacement (so a product can have zero shops or four), and moved the shed operations ahead of
#   the LOCKED guard. kagsim was updated for all three and re-verified before this pin was moved.
#
#   1.32.6 -> 1.32.7 (PLAN_v4). The *only* change is the market curve, and it is a large one:
#   `_shape` gained a `hinge` function (`HINGE_GAIN = 8`, scaled so f(T) == 1) and CARROT, TOMATO
#   and EGG switched to it below I0 -- carrot's below_target moving 0.2 -> 1.0 with it. Below -T
#   those three now run away quadratically (tomato $300 at -2T against $84 at -T), which is the
#   whole basis of PLAN_v4 2.7. Every simulation rule outside `_shape`/`market_price` is
#   byte-identical: `diff` of the two sources touches nothing else. kagsim's `Shape::Hinge` and
#   `agent/relay.py`'s vendored copy were both ported, and `make verify` re-run to 0 divergences,
#   before this pin was moved.
PINNED_VERSION = "1.32.7"
PINNED_SHA256 = {
    "kaggriculture.py": "bc8a54879ef02c7e",
    "kaggriculture.json": "a82c89c1a2315b93",
}

_ENV_DIR = os.path.join(os.path.dirname(kaggle_environments.__file__), "envs", "kaggriculture")


def _digest(name: str) -> str:
    with open(os.path.join(_ENV_DIR, name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(PINNED_SHA256))
def test_environment_source_is_unchanged(name: str) -> None:
    assert _digest(name) == PINNED_SHA256[name], (
        f"{name} has changed. kagsim mirrors this file, so every measurement in the repo is "
        f"suspect until parity is re-verified against the new source. Re-run `make verify`, read "
        f"the diff for rule changes, and only then update PINNED_SHA256."
    )


def test_package_version_is_pinned() -> None:
    got = importlib.metadata.version("kaggle-environments")
    assert got == PINNED_VERSION, (
        f"kaggle-environments moved {PINNED_VERSION} -> {got}. Check the environment diff before "
        f"trusting any result; the source hashes above are the authoritative check."
    )
