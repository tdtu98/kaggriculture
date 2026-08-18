"""The promotion gate's stage 3 must never report success without probing (E53).

Stage 3 exists because a chosen point need not be a *local* optimum — measured at 64.4% for the herd
mix (E17), and again at a cliff for `release_pressure` (E52). It is the one stage whose whole value
is that it actually runs.

It has now failed to do so **twice**, in two different ways:

1. It `return True`d when the candidate exposed no knob in `NEIGHBOURHOOD`, so `relay-sell` printed
   ALL STAGES PASSED after zero probes (E53).
2. The fix for (1) left a stray `return True` above the sweep, so *every* candidate passed without
   probing — the same defect, reintroduced by its own repair, and invisible because nothing tested
   this file.

Both are the E36 pattern: a check that passes while proving nothing. These tests pin the contract
rather than the arithmetic, because the arithmetic was never what broke.
"""

from __future__ import annotations

import pytest

from arena.registry import REGISTRY
from tools import promote


def test_no_applicable_knobs_is_not_a_pass():
    """A candidate the sweep cannot probe must return None — 'did not run', never True."""
    result = promote.stage3({"unrelated_knob": 3}, "champion", 1, {"episodeSteps": 720})
    assert result is None, "stage 3 reported a verdict for a candidate it could not probe"


def test_probing_actually_happens(monkeypatch):
    """With applicable knobs, stage 3 must call the arena. This is the regression that shipped."""
    calls = []

    def fake_run(names, seeds, config, gauntlet=None):
        calls.append(names)
        return None, []

    def fake_tabulate(results):
        cand = calls[0][0]
        # every neighbour loses decisively, so nothing escalates and the stage passes
        return ({(cand, n): promote.Winrate(24, 24) for n in calls[0][1:]}, None, None)

    monkeypatch.setattr(promote, "run", fake_run)
    monkeypatch.setattr(promote, "tabulate", fake_tabulate)

    params = dict(REGISTRY["relay-sell"].params)
    result = promote.stage3(params, "relay-sell", 1, {"episodeSteps": 720})

    assert calls, "stage 3 returned without ever calling the arena"
    probed = calls[0][1:]
    assert probed, "stage 3 called the arena with no neighbours to probe"
    assert result is True


def test_the_sweep_can_reach_the_cliff():
    """A neighbourhood radius that cannot reach the known cliff is a more convincing vacuum.

    E52: `release_pressure` scores 83.8% at 65 and **12.5% at 60** — the edge is ~10 units from the
    setting of 70. The +-1/+-2 spacing that suits a herd count would probe 68-72, find nothing, and
    pass. Deltas must be sized per knob, and this asserts that they are.
    """
    base = REGISTRY["relay-sell"].params["release_pressure"]
    probes = [base + d for d in promote.NEIGHBOURHOOD["release_pressure"]]
    assert min(probes) <= 60, (
        f"sweep probes {sorted(probes)}, none of which reach the measured cliff at 60 (E52)"
    )

    batch = REGISTRY["relay-sell"].params["release_batch"]
    batch_probes = [batch + d for d in promote.NEIGHBOURHOOD["release_batch"]]
    assert max(batch_probes) >= 12, (
        f"sweep probes {sorted(batch_probes)}; batch 12 measured 51.2% (E52) and must be reachable"
    )


def test_neighbours_keep_the_candidate_kind():
    """A perturbed relay agent must stay a relay agent, not become an engine spec."""
    spec = promote._neighbour_spec("relay-sell", {**REGISTRY["relay-sell"].params,
                                                  "release_pressure": 60}, "probe")
    assert spec.kind == "relay"
    spec.build()   # must construct without raising

    engine_spec = promote._neighbour_spec("champion", {**REGISTRY["champion"].params,
                                                       "cow_target": 5}, "probe")
    assert engine_spec.kind == "engine"
    engine_spec.build()
