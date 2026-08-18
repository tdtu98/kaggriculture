"""PLAN3 R0.2 -- `relay-base` must be bit-identical to the reference agent.

This is the gate the whole v3 direction rests on. If our restructured copy already differs from the
agent it was derived from, then every later A/B measures *both* the overlay under test and an
unrecorded behaviour change, and no result from R1 onward is interpretable. The kill criterion in
`PLAN3.md` §5 is explicit: if bit-identity cannot be reached, stop. Do not proceed with "close
enough".

Both seats are played because they are not symmetric, and the comparison is against the reference
loaded **the way Kaggle loads it** -- exec'd into empty globals, last module-level callable taken --
rather than imported, which is the surface E21 shipped two defects behind.
"""

from __future__ import annotations

import json

import pytest

import kagsim
from agent.relay import Ctx, count_blocked_ops, make_relay
from agent.relay_table import load_table

REFERENCE = "reference/kaggriculture/1/submission.py"
STEPS = 719


def _reference_agent():
    """Loaded exactly as the runner loads a submission (`kaggle_environments/agent.py:47-63`)."""
    from kaggle_environments.agent import get_last_callable

    with open(REFERENCE) as f:
        return get_last_callable(f.read(), path=REFERENCE)


def test_table_matches_reference():
    """Our embedded copy is the same 719 turns, not a drifted transcription."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ref_table", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert load_table() == module._REBALANCE_ACTIONS


@pytest.mark.parametrize("seed", range(4000, 4020))
@pytest.mark.parametrize("seat", (0, 1))
def test_relay_base_is_bit_identical(seed, seat):
    """20 seeds x 2 seats, every one of the 719 steps compared."""
    ours = make_relay()
    theirs = _reference_agent()
    filler = make_relay()

    sim = kagsim.Sim({"episodeSteps": STEPS + 1, "seed": seed})
    for step in range(STEPS):
        obs = sim.observation(seat)
        mine = ours(obs)
        reference = theirs(obs)
        assert mine == reference, (
            "step %d, seat %d, seed %d\n  relay:     %s\n  reference: %s"
            % (step, seat, seed, json.dumps(mine), json.dumps(reference))
        )
        acts = [None, None]
        acts[seat] = mine
        acts[1 - seat] = filler(sim.observation(1 - seat))
        sim.step(acts)


def test_effect_counters_are_wired():
    """PLAN3 SS6: an overlay that cannot report firing cannot be told apart from one that did not.

    `rank_sell_slots` is the layer E48 measured as the only one that ever fires, so it is the one
    that must show a non-zero count in a real game.
    """
    ours = make_relay()
    filler = make_relay()
    sim = kagsim.Sim({"episodeSteps": STEPS + 1, "seed": 4100})
    for _ in range(STEPS):
        sim.step([ours(sim.observation(0)), filler(sim.observation(1))])
    effects = ours.ctx_by_seat[0].effects
    assert effects.get("sell_reorder", 0) > 0, effects


def test_blocked_ops_detects_a_deliberate_desync():
    """R0.5: a counter that stays flat under a known break is not measuring anything (E36).

    Rather than trusting that `blocked_ops` works, break the choreography on purpose -- point every
    scripted WATER at a tile that cannot hold a plant -- and require the counter to notice.
    """
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    obs = {"player": 0, "farms": [farm, farm], "step": 0}
    ctx = Ctx(0)

    count_blocked_ops(obs, {"farmer": ["WATER"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_total == 1, ctx.blocked_ops

    farm["tiles"][0][0] = {"kind": "WEED"}
    count_blocked_ops(obs, {"farmer": ["HARVEST"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("HARVEST_on_weed") == 1, ctx.blocked_ops

    # ...and stays quiet when the tile is what the script expected.
    farm["tiles"][0][0] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 1}
    before = ctx.blocked_total
    count_blocked_ops(obs, {"farmer": ["WATER"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_total == before, ctx.blocked_ops


def test_blocked_ops_allows_harvesting_animals():
    """HARVEST is legal on an occupied pasture -- milk and wool use the same op as crops.

    The first version of the counter treated HARVEST as plant-only and reported **94 false desyncs
    per episode** on a farm that was perfectly in sync. Pinned here because a desync counter that
    cries wolf is worse than none: it would have vetoed every safe overlay in R1 and R2.
    """
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    obs = {"player": 0, "farms": [farm, farm], "step": 0}
    ctx = Ctx(0)

    farm["tiles"][0][0] = {"kind": "PASTURE", "animal": "COW", "yield_units": 3}
    count_blocked_ops(obs, {"farmer": ["HARVEST"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_total == 0, ctx.blocked_ops

    # An empty pasture is still a real desync -- the animal escaped or was never placed.
    farm["tiles"][0][0] = {"kind": "PASTURE"}
    count_blocked_ops(obs, {"farmer": ["HARVEST"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("HARVEST_on_empty") == 1, ctx.blocked_ops


def test_blocked_ops_sees_a_stranded_animal_placement():
    """R1's exact failure mode: the purchase was swapped but the scripted PICKUP/PLACE was not.

    A `PLACE COW` by a unit carrying a sheep silently does nothing, the pasture stays empty for the
    rest of the season, and the money number looks like "adaptive livestock does not pay". The
    counter has to catch it directly rather than waiting for it to surface as `HARVEST_on_empty`
    many turns later.
    """
    farm = {
        "farmer": [4, 4],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    obs = {
        "player": 0,
        "farms": [farm, farm],
        "step": 0,
        "private": {"shed": {"SHEEP": 2}, "inventories": [{"SHEEP": 1}]},
    }
    ctx = Ctx(0)

    # Standing on an empty pasture holding a SHEEP, but the script says place a COW.
    farm["tiles"][4][4] = {"kind": "PASTURE"}
    count_blocked_ops(obs, {"farmer": ["PLACE", "COW"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("PLACE_COW_not_carried") == 1, ctx.blocked_ops

    # The matching species is fine.
    count_blocked_ops(obs, {"farmer": ["PLACE", "SHEEP"], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("PLACE_SHEEP_not_carried") is None, ctx.blocked_ops

    # And a PICKUP for stock the shed does not hold is caught at the shed, not later.
    count_blocked_ops(obs, {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("PICKUP_COW_absent") == 1, ctx.blocked_ops

    count_blocked_ops(obs, {"farmer": ["PICKUP", "SHEEP", 1], "hands": [], "market": []}, ctx)
    assert ctx.blocked_ops.get("PICKUP_SHEEP_absent") is None, ctx.blocked_ops
