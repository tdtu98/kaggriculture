"""V2 — the independently designed opponents must stay *functional*.

A broken opponent tests nothing: it loses for the wrong reason and quietly inflates confidence in
the champion. These assert each one still plays a real game, so a future engine change cannot
silently degrade the arena into a walkover.
"""

from __future__ import annotations

import pytest

import kagsim
from arena.opponents import OPPONENTS

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _play(agent, seed=1, steps=718):
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    for _ in range(steps):
        sim.step([agent(sim.observation(0)), PASS])
    return sim


@pytest.mark.parametrize("name", sorted(OPPONENTS))
def test_opponent_is_functional_not_broken(name):
    """Functional means *plays the game*, not *plays it well*.

    Final money is the wrong criterion: `o-goose-baron` earns ~$35k of revenue and spends nearly
    all of it on birds and feed, ending below its starting bank. That is an over-investing
    strategy, which is a legitimate thing for an opponent to be — `random` ends at $0 and is also
    behaving correctly. What would make an opponent useless is failing to *act*: selling nothing,
    or idling away its turns.
    """
    sim = _play(OPPONENTS[name]())
    stats = sim.stats(0)
    revenue = sum(stats["sold_revenue"].values())
    assert revenue > 5000, f"{name} generated only ${revenue:,} of revenue — it is not playing"
    assert stats["sold_units"], f"{name} sold nothing all season"
    noop = stats["actions_noop"] / max(stats["actions_total"], 1)
    assert noop < 0.5, f"{name} wasted {100 * noop:.0f}% of its turns on no-ops"


@pytest.mark.parametrize("name", sorted(OPPONENTS))
def test_opponent_uses_its_action_budget(name):
    """A second angle on the same question, independent of money entirely."""
    sim = _play(OPPONENTS[name]())
    stats = sim.stats(0)
    effective = stats["actions_effective"] / max(stats["actions_total"], 1)
    assert effective > 0.05, f"{name} did something useful on only {100 * effective:.1f}% of turns"


@pytest.mark.parametrize("name", sorted(OPPONENTS))
def test_opponent_is_strategically_distinct(name):
    """Each brief must actually produce different behaviour from the champion.

    The point of V2 is to leave the champion's strategy family. An opponent whose revenue mix
    matches the champion's is a variant wearing a different name.
    """
    from agent import make_agent
    from agent.params import Params
    from arena.registry import REGISTRY

    champ = _play(make_agent(Params(**REGISTRY["champion"].params)))
    other = _play(OPPONENTS[name]())

    def mix(sim):
        rev = sim.stats(0)["sold_revenue"]
        total = sum(rev.values()) or 1
        return {k: v / total for k, v in rev.items()}

    a, b = mix(champ), mix(other)
    divergence = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))
    assert divergence > 0.25, (
        f"{name} earns its money the same way as the champion "
        f"(revenue-mix divergence {divergence:.2f}); it is a variant, not a new strategy"
    )


def test_shop_chaser_actually_reacts_to_the_town():
    """Its whole premise is reading `unlocked_shops`, which no other agent does."""
    agent = OPPONENTS["o-shop-chaser"]()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 2})
    for _ in range(24 * 2):
        sim.step([agent(sim.observation(0)), PASS])
    early = dict(agent.p.crop_mix)
    for _ in range(24 * 20):
        sim.step([agent(sim.observation(0)), PASS])
    late = dict(agent.p.crop_mix)
    assert early != late, "crop mix never responded to shop unlocks"
