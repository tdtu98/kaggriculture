"""Finish the tile you are standing on before moving (E47).

Written the way this session learned to write them: assert the *mechanism engages*, not that money
went up. Two earlier attempts at this same idea were declared dead on outcome measurements while
the mechanism had never actually run —

* `here_bonus` (a distance-zero discount in the cost function) was consulted in **4%** of the cases
  it targeted, because sticky assignment and rival claims both resolve earlier. Measured 29% -> 30%
  same-tile share and looked refuted.
* role specialisation recomputed roles every turn, so purity moved 0.64 -> 0.69 and looked refuted,
  until the roles were made sticky.

The engine resolves assignments in layers — local work, then sticky targets, then the cost
function, then a fallback — and an intervention in a late layer is invisible if an earlier one
already decided. So these tests pin the layer.
"""

from __future__ import annotations

import collections
import json

import kagsim

from agent import Params, make_agent
from agent.engine import Engine, Task

CHAMPION = json.load(open("search/champion.json"))["params"]
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def test_local_work_beats_a_sticky_target():
    """The layer that used to win: a target chosen last turn is honoured before anything else."""
    e = Engine(Params(finish_tile=True))
    units = [(2, 2)]
    tasks = [Task(9, 9, "WATER", None, 0),      # the stale sticky target, far away
             Task(2, 2, "HARVEST", None, 1)]    # work underfoot
    e._assigned[0] = (9, 9, "WATER", None, None)
    acts = e.assign(units, tasks, [{}], commit=False)
    assert acts[0][0] == "HARVEST", f"expected local work, got {acts[0]}"


def test_local_work_is_claimed_before_a_distant_unit_can_take_it():
    """The other layer: whoever is served first claims the tile, even from far away."""
    e = Engine(Params(finish_tile=True))
    units = [(8, 8), (3, 3)]                    # unit 0 far, unit 1 standing on the work
    tasks = [Task(3, 3, "WATER", None, 0)]
    acts = e.assign(units, tasks, [{}, {}], commit=False)
    assert acts[1][0] == "WATER", f"the unit standing on it should get it, got {acts[1]}"


def test_eligibility_is_respected():
    """A task the unit cannot legally perform must not be grabbed just because it is underfoot."""
    e = Engine(Params(finish_tile=True))
    units = [(4, 4)]
    tasks = [Task(4, 4, "FEED", None, 0, needs="WHEAT"), Task(5, 4, "WATER", None, 0)]
    acts = e.assign(units, tasks, [{}], commit=False)          # carrying no wheat
    assert acts[0][0] != "FEED", "FEED was taken without wheat in hand"
    acts = e.assign(units, tasks, [{"WHEAT": 1}], commit=False)
    assert acts[0][0] == "FEED", "FEED should be taken when the wheat is there"


def test_two_units_on_one_task_do_not_both_take_it():
    e = Engine(Params(finish_tile=True))
    units = [(3, 3), (3, 3)]
    tasks = [Task(3, 3, "WATER", None, 0), Task(7, 7, "WATER", None, 0)]
    acts = e.assign(units, tasks, [{}, {}], commit=False)
    assert sum(a[0] == "WATER" for a in acts) >= 1
    assert acts[0] != acts[1], "both units took the same task"


def walkaways_and_purity(params, seed=5, steps=719):
    """Count moves away from work the unit was *eligible* to do, and the same-tile share."""
    agent = make_agent(Params(**params))
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    walk = stay = acted = 0
    last: dict[int, tuple | None] = {}
    for _ in range(steps):
        obs = sim.observation(0)
        farm, priv = obs["farms"][0], obs["private"]
        units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        agent._day = obs["day"]                 # match engine state before inspecting it
        tasks = agent.build_tasks(farm, priv, obs["day"])
        a = agent(obs)
        acts = [a.get("farmer")] + list(a.get("hands") or [])
        for i, (pos, ac) in enumerate(zip(units, acts)):
            if not ac:
                continue
            if ac[0] in MOVES:
                inv = priv["inventories"][i] if i < len(priv["inventories"]) else {}
                if any((t.x, t.y) == tuple(pos) and (not t.needs or inv.get(t.needs, 0) > 0)
                       for t in tasks):
                    walk += 1
                last[i] = None
            elif ac[0] != "PASS":
                acted += 1
                if last.get(i) == pos:
                    stay += 1
                last[i] = pos
        sim.step([a, PASS])
    return walk, stay / max(1, acted)


def test_mechanism_engages_in_a_real_episode():
    """The check both previous attempts skipped: does it change play, not just the cost function?"""
    off_walk, off_pure = walkaways_and_purity(CHAMPION)
    on_walk, on_pure = walkaways_and_purity({**CHAMPION, "finish_tile": True})
    assert on_walk < 0.6 * off_walk, (
        f"walk-aways from eligible local work {off_walk} -> {on_walk}: barely changed, so the "
        f"mechanism is not engaging (this is exactly how `here_bonus` looked)")
    assert on_pure > off_pure + 0.04, (
        f"same-tile share {off_pure:.0%} -> {on_pure:.0%}: not moving")


def test_default_is_off():
    assert Params().finish_tile is False
