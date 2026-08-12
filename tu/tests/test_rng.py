"""T0.3 — verify kagsim's MT19937 is bit-exact against CPython's `random.Random`.

Required for layer-C parity (docs/decisions.md D5): `kaggriculture.py:848` seeds a fresh
`random.Random((seed * 1_000_003) ^ day)` each day and `_spawn_weeds` consumes a
gameplay-dependent number of `random()` draws from it.
"""

import random

import pytest

import kagsim

# Seeds chosen to cover: trivial, small, the reference default, a large multi-word value,
# and the actual expression the environment uses.
SEEDS = [
    0,
    1,
    42,
    7,
    123456789,
    2**31,
    2**32 + 1,
    (42 * 1_000_003) ^ 17,
    (123456789 * 1_000_003) ^ 29,
]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_is_bit_exact(seed):
    """`random()` must match to the last bit, not to a tolerance."""
    py = random.Random(seed)
    rs = kagsim.PyRandom(seed)
    for i in range(1000):
        a, b = py.random(), rs.random()
        assert a == b, f"seed={seed} draw={i}: python={a!r} rust={b!r}"


@pytest.mark.parametrize("seed", SEEDS)
def test_getrandbits_matches(seed):
    py = random.Random(seed)
    rs = kagsim.PyRandom(seed)
    for k in list(range(1, 33)) * 10:
        assert py.getrandbits(k) == rs.getrandbits(k), f"seed={seed} k={k}"


@pytest.mark.parametrize("seed", SEEDS)
def test_choice_matches(seed):
    """`_end_of_day` picks the next shop with `rng.choice(sorted(remaining))`.

    `choice` goes through `_randbelow`, which uses rejection sampling on `getrandbits`, so it
    consumes a variable number of draws — the stream position must stay in sync too.
    """
    py = random.Random(seed)
    rs = kagsim.PyRandom(seed)
    for n in range(2, 200):
        seq = list(range(n))
        assert py.choice(seq) == rs.choice_index(n), f"seed={seed} n={n}"


def test_interleaved_stream_stays_in_sync():
    """The real usage interleaves many `random()` draws with an occasional `choice`."""
    seed = (7 * 1_000_003) ^ 12
    py = random.Random(seed)
    rs = kagsim.PyRandom(seed)
    for day in range(30):
        for _ in range(100):  # one per empty tile, per player
            assert py.random() == rs.random()
        remaining = 8 - (day % 8)
        if remaining > 1:
            assert py.choice(list(range(remaining))) == rs.choice_index(remaining)


def test_banker_rounding_matches_python():
    """`market_price` does `int(round(price))`; Rust's f64::round() would diverge on .5."""
    for x in [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.4, 2.6, 24.5, 25.5, 249.5]:
        assert kagsim.round_half_even(x) == round(x), x
