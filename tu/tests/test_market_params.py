"""`marketParams` parity — per-product price-curve overrides.

Kaggle can change the environment configuration between now and scoring, so kagsim has to honour
every documented knob rather than assume defaults. `marketParams` is the most intricate one: it is
a *sparse* merge onto the defaults, and the resolved table lands in `obs["market"]["params"]`,
making it observable state that agents can read.
"""

from __future__ import annotations

import pytest

from parity import run_parity, run_scripted

ALL = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

CASES = {
    "empty-is-default": {},
    "readme-example": {"WOOL": {"above_target": 0.95}},
    "log10-shape": {"MELON": {"above_func": "log10", "above_target": 0.3, "T": 50}},
    "every-field": {"WHEAT": {"base": 5, "I0": 100, "T": 10, "below_func": "sq",
                              "above_func": "linear", "above_target": 9.0}},
    "unknown-product": {"BOGUS": {"base": 1}},
    "unknown-shape-name": {"EGG": {"below_func": "not_a_shape", "above_func": "weird"}},
    "non-dict-patch": {"CARROT": "not-a-dict"},
    "all-products": {p: {"I0": 500, "T": 20, "above_target": 2.0} for p in ALL},
}


@pytest.mark.parametrize("name", list(CASES))
def test_override_parity(name):
    run_parity(env_seed=0, fuzz_seed=3, steps=250, config={"marketParams": CASES[name]})


def test_empty_override_is_indistinguishable_from_absent():
    """`_new_market` only stores `params` when the table is not the module default (`:169`),
    and `_initialize` treats a falsy value as no override at all."""
    with_empty = run_scripted({"marketParams": {}}, [], steps=3)
    without = run_scripted({}, [], steps=3)
    assert "market_params" not in with_empty
    assert with_empty == without


def test_truthy_override_is_exposed_even_when_nothing_matched():
    """An override naming only unknown products is still truthy, so the reference builds a
    resolved table and exposes it — a subtle observable difference from having no override."""
    st = run_scripted({"marketParams": {"BOGUS": {"base": 1}}}, [], steps=3)
    assert "market_params" in st, "resolved table must be exposed"
    assert st["market_params"]["WHEAT"]["base"] == 25.0, "defaults untouched"


def test_overrides_actually_move_prices():
    """Sanity: the override has to change behaviour, or parity would be vacuous."""
    import kagsim

    crashy = {"MELON": {"above_target": 20.0}}
    assert kagsim.market_price(4, 10_000) == 250
    assert kagsim.market_price(4, 10_100) > kagsim.market_price(4, 10_100, crashy)


def test_initial_inventory_follows_overridden_I0():
    st = run_scripted({"marketParams": {"WHEAT": {"I0": 42}}}, [], steps=1)
    assert st["market_inv"]["WHEAT"] < 100, "inventory seeded from the product's own I0"
    assert st["market_inv"]["CARROT"] == 10_000 - 1, "others keep the default, less town drain"
