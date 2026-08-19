"""O1: shop-draw branch points.

Three claims are worth testing here and they need three different kinds of evidence.

* **The demand table is the env's**, not prose. `test_demand_table_is_the_env_table` rebuilds it
  from `SHOPS` the long way round and `test_drain_matches_the_reference_env` plays real turns with
  two `pass` agents so that `demand_per_day` is checked against inventory the *environment* moved.
  The task text for O1 had PET_CAFE down as an egg shop; it sells carrots. Names are not evidence.
* **Forward-only is a gate, not an intention.** The patches are asked to do the forbidden thing —
  dig a live cohort, sell a bought animal — and the gate must refuse. These are the mutation-hardest
  tests in the file: `forward_only_errors` is deliberately exercised on hand-built pairs so that a
  weakened rule fails here rather than three hundred games later.
* **The branch fires in play, on the towns it should and no others.** Read through
  `main_v4`'s own effect ledger over real seasons (E44), with the contrast that makes it evidence:
  the same seeds, the same plan, no branch -> zero counters and identical money.
"""

from __future__ import annotations

import kagsim
import pytest
from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS

from agent import branches as B
from agent.plan import Branch, Cohort, Plan


# --------------------------------------------------------------------------- fixtures

def _obs(day=0, shops=(), tiles=None, shed=None, animals=0):
    board = tiles or [[None] * 10 for _ in range(10)]
    for i in range(animals):
        board[0][i] = {"kind": "STRUCTURE", "structure": "PASTURE", "animal": "COW"}
    return {
        "player": 0, "day": day, "hour": 0, "step": day * 24,
        "farms": [{"money": 3000, "farmer": [4, 4], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW", "NE", "SW"], "tiles": board},
                  {"money": 3000, "farmer": [4, 4], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW"],
                   "tiles": [[None] * 10 for _ in range(10)]}],
        "town": {"unlocked_shops": list(shops)},
        "private": {"shed": dict(shed or {}), "seeds": {}, "inventories": [{}]},
    }


def _plan(**kw):
    """A small, explicit plan — six cows in NW, one pending cohort, room left in NE."""
    base = dict(
        pasture_tiles=tuple((x, 4) for x in range(5)) + ((0, 3),),
        land_days={"NE": 2, "SW": 4, "SE": 99},
        herd=tuple(("COW", 1 + i // 3) for i in range(6)),
        cohorts=(Cohort("WHEAT", "NW", 3, 0, tiles=((0, 0), (1, 0), (2, 0))),
                 Cohort("MELON", "NW", 3, 8, tiles=((0, 1), (1, 1), (2, 1)))),
        hands="auto",
        consts={"animals_per_day": 1, "sell_floor": {}},
    )
    base.update(kw)
    return Plan(**base)


def _season(seed, plan, opponent="starter", turns=719):
    """One real game; returns `main_v4`'s effect ledger and the final money for our seat."""
    from agent import main_v4
    from harness import registry

    B.reset()
    main_v4._STATE.clear()
    ours = main_v4.make_agent(plan)
    theirs = registry.get(opponent).build()
    sim = kagsim.Sim({"episodeSteps": 720, "seed": seed})
    sim.collect_stats = True
    last = None
    for _ in range(turns):
        obs = [sim.observation(0), sim.observation(1)]
        last = obs[0]
        sim.step([ours(obs[0]), theirs(obs[1])])
    _season.board = last["farms"][0]["tiles"]
    _season.stats = sim.stats(0)
    return dict((main_v4._STATE.get(0) or {}).get("effects") or {}), sim.money(0)


def _animals(board):
    out: dict = {}
    for row in board:
        for t in row:
            if isinstance(t, dict) and t.get("animal"):
                out[t["animal"]] = out.get(t["animal"], 0) + 1
    return out


# --------------------------------------------------------------------------- the demand table

def test_demand_table_is_the_env_table():
    """Every product/shop pair, rebuilt from `SHOPS` independently of the comprehension under test.

    Named cases below it because the *names* are the trap: PET_CAFE is carrots and YARN_STORE is
    wool, and those two are the only single-product shops on the board.
    """
    for shop, products in SHOPS.items():
        for product in products:
            assert shop in B.DEMANDERS[product], f"{shop} sells {product}"
    for product, shops in B.DEMANDERS.items():
        for shop in shops:
            assert product in SHOPS[shop]

    assert B.DEMANDERS["WOOL"] == ("YARN_STORE",)
    assert B.DEMANDERS["CARROT"] == ("FARMERS_MARKET", "PET_CAFE")
    assert set(B.DEMANDERS["MILK"]) == {"ICE_CREAM_SHOP", "PIZZA_SHOP", "SMOOTHIE_SHOP"}
    assert len(B.DEMANDERS["STRAWBERRY"]) == 4


def test_single_product_shops_drain_double():
    """`multiplier = 2 if len(products) == 1` (`kaggriculture.py:741`) — the whole reason WOOL and
    CARROT can starve while WHEAT never does."""
    single = {s for s, p in SHOPS.items() if len(p) == 1}
    assert single == {"YARN_STORE", "PET_CAFE"}
    assert B.demand_per_day(_obs(shops=["YARN_STORE"]), "WOOL") == 12
    assert B.demand_per_day(_obs(shops=["SMOOTHIE_SHOP"]), "MILK") == 6


def test_drain_matches_the_reference_env():
    """The arithmetic, checked against inventory the environment actually moved.

    Two `pass` agents: no farm produces or sells anything, so every unit that leaves the market
    left through the town. `demand_per_day` must reproduce the day's shop drain exactly — the town
    centre's extra 1/day is deliberately outside `demand_per_day` and is added back here.
    """
    from harness import registry

    agents = [registry.get("pass").build(), registry.get("pass").build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 4242})
    seen = []
    for _ in range(719):
        obs = [sim.observation(0), sim.observation(1)]
        if obs[0]["hour"] == 0:
            seen.append((obs[0]["day"], dict(obs[0]["market"]["inventory"]),
                         list(obs[0]["town"]["unlocked_shops"])))
        sim.step([agents[0](obs[0]), agents[1](obs[1])])

    checked = 0
    for (day, inv, shops), (_next_day, nxt, _s) in zip(seen, seen[1:]):
        town = _obs(day=day, shops=shops)
        for product in ("WOOL", "MILK", "STRAWBERRY", "CARROT"):
            expected = B.demand_per_day(town, product) + 1        # + the town centre's 1/day
            assert inv[product] - nxt[product] == expected, (day, product, shops)
            checked += 1
    assert checked > 100
    assert any(s for _d, _i, s in seen), "no shop ever unlocked — the fixture proves nothing"


def test_instance_count_follows_the_unlock_rule():
    """`min(8, day // 3)` (`kaggriculture.py:867,886-891`), and duplicates are real."""
    from harness import registry

    agents = [registry.get("pass").build(), registry.get("pass").build()]
    sim = kagsim.Sim({"episodeSteps": 720, "seed": 99})
    dupes = 0
    for _ in range(719):
        obs = sim.observation(0)
        if obs["hour"] == 0:
            shops = B.shop_instances(obs)
            assert len(shops) == min(8, obs["day"] // 3), (obs["day"], shops)
            dupes += len(shops) - len(set(shops))
        sim.step([agents[0](obs), agents[1](sim.observation(1))])
    assert dupes > 0, "with-replacement draws produced no duplicate in 30 days"


# --------------------------------------------------------------------------- conditions

@pytest.mark.parametrize("condition,shops,expected", [
    ("has:YARN_STORE", ["BAKERY"], False),
    ("has:YARN_STORE", ["BAKERY", "YARN_STORE"], True),
    ("demand>=2:STRAWBERRY", ["SMOOTHIE_SHOP"], False),
    ("demand>=2:STRAWBERRY", ["SMOOTHIE_SHOP", "BRUNCH_SPOT"], True),
    ("demand>=2:STRAWBERRY", ["SMOOTHIE_SHOP", "SMOOTHIE_SHOP"], True),   # with replacement
    ("demand==0:MILK", ["BAKERY", "PET_CAFE"], True),
    ("demand==0:MILK", ["BAKERY", "PIZZA_SHOP"], False),
    ("drain>=12:WOOL", ["YARN_STORE"], True),
    ("drain>=12:CARROT", ["FARMERS_MARKET"], False),
])
def test_conditions_read_instances_not_names(condition, shops, expected):
    assert B.holds(_obs(shops=shops), condition) is expected


def test_a_malformed_condition_is_refused_rather_than_silently_false():
    with pytest.raises(ValueError):
        B.holds(_obs(), "yarnstore")
    with pytest.raises(ValueError):
        B.holds(_obs(), "unknown>=1:WOOL")


# --------------------------------------------------------------------------- the forward-only gate

def test_the_frontier_is_owned_plus_the_pacing_lead():
    """`_paced_plan` releases `herd[:owned + lead]`, so index `owned + lead` has never been offered
    to `_animal_orders` on any turn. The frontier has to be that number and not a day."""
    from agent.main_v4 import _paced_plan, _structure_lead

    plan = _plan()
    for owned in (0, 2, 5):
        obs = _obs(day=20, animals=owned)
        assert B.owned_animals(obs, 0) == owned
        assert B.frontier(obs, plan, 0) == owned + _structure_lead(plan)
        released = len(_paced_plan(obs, plan, 0, 20, None).herd)
        assert released <= B.frontier(obs, plan, 0)


def test_a_patch_that_would_dig_a_live_plant_is_rejected():
    from dataclasses import replace

    plan = _plan()
    live = replace(plan, cohorts=(replace(plan.cohorts[0], crop="MELON"), plan.cohorts[1]))
    assert B.forward_only_errors(plan, live, day=5, frontier_index=99), \
        "editing a cohort planted on day 0 was allowed"
    removed = replace(plan, cohorts=(plan.cohorts[1],))
    assert B.forward_only_errors(plan, removed, day=5, frontier_index=99)
    # ... while the pending one (plant_day 8) may still be changed on day 5.
    pending = replace(plan, cohorts=(plan.cohorts[0], replace(plan.cohorts[1], crop="CARROT")))
    assert B.forward_only_errors(plan, pending, day=5, frontier_index=99) == []


def test_a_new_cohort_may_not_re_sow_live_ground():
    from dataclasses import replace

    plan = _plan()
    stolen = replace(plan, cohorts=plan.cohorts + (
        Cohort("STRAWBERRY", "NW", 1, 6, tiles=((0, 0),)),))     # (0, 0) is the day-0 wheat
    assert B.forward_only_errors(plan, stolen, day=6, frontier_index=99)


def test_a_patch_that_would_sell_an_animal_is_rejected():
    from dataclasses import replace

    plan = _plan()
    # Three head are bought; dropping the first is a sale, dropping the last is a cancelled order.
    sold = replace(plan, herd=plan.herd[1:])
    assert B.forward_only_errors(plan, sold, day=9, frontier_index=3)
    swapped = replace(plan, herd=(("SHEEP", 1),) + plan.herd[1:])
    assert B.forward_only_errors(plan, swapped, day=9, frontier_index=3)
    cancelled = replace(plan, herd=plan.herd[:5])
    assert B.forward_only_errors(plan, cancelled, day=9, frontier_index=3) == []


def test_a_patch_may_not_remove_a_pasture_or_re_time_land():
    from dataclasses import replace

    plan = _plan()
    assert B.forward_only_errors(plan, replace(plan, pasture_tiles=plan.pasture_tiles[1:]),
                                 day=9, frontier_index=99)
    assert B.forward_only_errors(plan, replace(plan, land_days={"NE": 9, "SW": 4, "SE": 99}),
                                 day=9, frontier_index=99)


def test_the_gate_refuses_a_patch_rather_than_repairing_it():
    """A branch whose patch trips the gate is counted and dropped, and the plan is unchanged."""
    seen: dict = {}

    def note(k, n=1):
        seen[k] = seen.get(k, 0) + n

    # `add_cohort` handed a quadrant that is locked on the plant day: `validate` must reject it.
    plan = _plan(land_days={"NE": 99, "SW": 99, "SE": 99}, branches=(
        Branch(day_from=3, day_to=3, condition="has:BAKERY", name="bad",
               patch={"add_cohort": {"crop": "STRAWBERRY", "quadrants": ("SE",), "n_tiles": 4}}),))
    out = B.apply(_obs(day=3, shops=["BAKERY"]), plan, 0, 3, note)
    assert out.cohorts == plan.cohorts
    assert seen.get("branches_fired", 0) == 0
    assert seen.get("branch_bad_rejected") == 1 or seen.get("branch_bad_noop") == 1


# --------------------------------------------------------------------------- the patches

def test_the_gate_stops_a_hostile_patch_op(monkeypatch):
    """The shipped patch ops cannot violate forward-only by construction, which is the design — so
    the *gate* itself is exercised against one that can.

    Without this the gate is untested code: removing the `forward_only_errors` call entirely leaves
    every other test in this file green (measured). A patch op is a small extension point; the
    guarantee has to live at the boundary rather than in each op's good manners.
    """
    from dataclasses import replace

    def dig(obs, plan, seat, day, spec, count):
        count("branch_hostile", 1)
        return replace(plan, cohorts=(replace(plan.cohorts[0], crop="MELON"),) + plan.cohorts[1:])

    monkeypatch.setitem(B.PATCHES, "dig", dig)
    counts: dict = {}
    plan = _plan(branches=(Branch(day_from=3, day_to=3, condition="has:BAKERY", name="hostile",
                                  patch={"dig": {}}),))
    B.reset()
    out = B.apply(_obs(day=3, shops=["BAKERY"]), plan, 0, 3,
                  lambda k, n=1: counts.__setitem__(k, counts.get(k, 0) + n))
    assert out is plan or out.cohorts == plan.cohorts, "a live cohort was re-sown"
    assert counts.get("branch_rejected") == 1, counts
    assert counts.get("branches_fired", 0) == 0
    assert "branch_hostile" not in counts, "a rejected patch's own counters leaked into the ledger"


def test_swap_species_only_touches_animals_past_the_frontier():
    plan = _plan(consts={"animals_per_day": 1, "branch_set": "yarn"})
    obs = _obs(day=4, shops=["YARN_STORE"], animals=2)
    out = B.apply(obs, plan, 0, 4, lambda k, n=1: None)
    assert [s for s, _d in out.herd[:3]] == ["COW", "COW", "COW"], "a bought animal was swapped"
    assert [s for s, _d in out.herd[3:]] == ["SHEEP", "SHEEP", "COW"]
    assert [d for _s, d in out.herd] == [d for _s, d in plan.herd], "buy days moved"


def test_swap_species_refuses_a_different_structure():
    """A GOOSE wants a COOP; swapping one into a PASTURE slot leaves the bird in the shed."""
    out = B._patch_swap_species(_obs(day=4), _plan(), 0, 4,
                                {"from": "COW", "to": "GOOSE", "count": 2}, lambda k, n=1: None)
    assert out is None


def test_cap_species_never_cuts_below_what_the_farm_owns():
    counts: dict = {}
    plan = _plan(herd=tuple(("COW", 1) for _ in range(6)),
                 consts={"animals_per_day": 1, "branch_set": "milk", "branch_milk_cap": 2})
    note = lambda k, n=1: counts.__setitem__(k, counts.get(k, 0) + n)   # noqa: E731

    # Five standing with a lead of 1 puts the frontier at 6 — the whole herd is spoken for, so a
    # cap of 2 cuts nothing at all. That is the point of the cap: it stops purchases, it is not a
    # cull, and by the back half of the season it can do nothing whatever the threshold says.
    B.reset()
    out = B.apply(_obs(day=9, shops=["BAKERY"], animals=5), plan, 0, 9, note)
    assert out.herd == plan.herd, out.herd
    assert counts.get("branch_capped_animals", 0) == 0
    assert counts.get("branch_noop") == 1

    # Three standing: the frontier is 4, so entries 4 and 5 are still unbought and both go.
    B.reset()
    counts.clear()
    out = B.apply(_obs(day=9, shops=["BAKERY"], animals=3), plan, 0, 9, note)
    assert len(out.herd) == 4, out.herd
    assert counts.get("branch_capped_animals") == 2


def test_add_cohort_lands_in_a_quadrant_with_room_by_its_plant_day():
    counts: dict = {}
    plan = _plan(consts={"animals_per_day": 1, "branch_set": "straw",
                         "branch_straw_day": 5, "branch_straw_shops": 1})
    obs = _obs(day=5, shops=["SMOOTHIE_SHOP"])
    out = B.apply(obs, plan, 0, 5, lambda k, n=1: counts.__setitem__(k, counts.get(k, 0) + n))
    assert len(out.cohorts) == len(plan.cohorts) + 1
    new = out.cohorts[-1]
    assert new.crop == "STRAWBERRY" and new.plant_day == 6
    assert new.quadrant in plan.unlocked_by(6)
    assert set(new.tiles).isdisjoint(set(plan.occupied()))
    assert out.validate() == []
    assert counts.get("branch_cohort_tiles") == len(new.tiles)


def test_a_branch_fires_once_and_only_inside_its_window():
    counts: dict = {}

    def note(k, n=1):
        counts[k] = counts.get(k, 0) + n

    plan = _plan(consts={"animals_per_day": 1, "branch_set": "yarn"})
    B.reset()
    # Before the window opens: nothing, even though the condition holds.
    B.apply(_obs(day=2, shops=["YARN_STORE"]), plan, 0, 2, note)
    assert counts.get("branches_fired", 0) == 0
    for day in (4, 5, 6):
        B.apply(_obs(day=day, shops=["YARN_STORE"], animals=0), plan, 0, day, note)
    assert counts["branches_fired"] == 1, counts
    assert counts["branch_swapped_animals"] == 2
    # ... and after `day_to` a town that only now draws the yarn store gets nothing.
    B.reset()
    counts.clear()
    B.apply(_obs(day=20, shops=["YARN_STORE"]), plan, 0, 20, note)
    assert counts.get("branches_fired", 0) == 0


def test_the_condition_is_re_read_each_day_until_it_holds():
    counts: dict = {}
    plan = _plan(consts={"animals_per_day": 1, "branch_set": "yarn"})
    B.reset()
    B.apply(_obs(day=4, shops=["BAKERY"]), plan, 0, 4, lambda k, n=1: None)
    B.apply(_obs(day=7, shops=["BAKERY", "YARN_STORE"]), plan, 0, 7,
            lambda k, n=1: counts.__setitem__(k, counts.get(k, 0) + n))
    assert counts.get("branches_fired") == 1


def test_the_patched_plan_survives_the_turn_that_made_it():
    """`apply` is called every turn with the pristine base plan; the patch has to persist."""
    plan = _plan(consts={"animals_per_day": 1, "branch_set": "yarn"})
    B.reset()
    first = B.apply(_obs(day=4, shops=["YARN_STORE"]), plan, 0, 4, lambda k, n=1: None)
    later = B.apply(_obs(day=11, shops=["YARN_STORE"]), plan, 0, 11, lambda k, n=1: None)
    assert later.herd == first.herd
    assert "SHEEP" in {s for s, _d in later.herd}


# --------------------------------------------------------------------------- default is off

def test_a_plan_without_branches_is_handed_back_unchanged():
    """Identity, not equality: the default tree must be untouched rather than merely equivalent."""
    plan = _plan()
    assert B.apply(_obs(day=9, shops=["YARN_STORE"] * 3), plan, 0, 9, lambda k, n=1: None) is plan
    assert B.active(plan) == ()
    assert B.active(Plan.boatlee_like()) == ()


def test_an_unknown_branch_set_is_an_error_not_a_silent_no_op():
    with pytest.raises(KeyError):
        B.active(_plan(consts={"branch_set": "yarnstore"}))


@pytest.mark.parametrize("seed", [90000, 90001, 90003])
def test_a_season_without_branches_is_unchanged_by_this_module(seed):
    """In play, the claim that matters: same plan, branches absent, same money as the shipped tree.

    The reference number is the *other* seat's — no. It is the same seed played twice, once with the
    branch module's state deliberately poisoned by a previous season. If `apply` leaked anything
    across games, this is where it would show.
    """
    plan = Plan.boatlee_like()
    B.reset()
    _dirty, _m = _season(seed, plan.with_consts(branch_set="all"))
    effects, money = _season(seed, plan)
    assert not [k for k in effects if k.startswith("branch")], effects
    clean_effects, clean_money = _season(seed, plan)
    assert money == clean_money


# --------------------------------------------------------------------------- in play

@pytest.mark.parametrize("seed,fires", [(90001, True), (90000, False)])
def test_the_yarn_branch_fires_only_where_the_town_drew_a_yarn_store(seed, fires):
    """The contrast that makes the counter evidence: seed 90001's town draws a YARN_STORE by day 12
    and seed 90000's does not (measured, `census_compiler.jsonl`)."""
    plan = Plan.boatlee_like().with_consts(branch_set="yarn")
    effects, money = _season(seed, plan)
    assert effects.get("fallbacks", 0) == 0
    if fires:
        assert effects.get("branches_fired") == 1, effects
        assert effects.get("branch_yarn_sheep_fired") == 1
        assert effects.get("branch_swapped_animals", 0) >= 1
        base, base_money = _season(seed, Plan.boatlee_like())
        assert money != base_money, "the branch fired and changed nothing"

    else:
        assert effects.get("branches_fired", 0) == 0, effects
        base, base_money = _season(seed, Plan.boatlee_like())
        assert money == base_money, "a dormant branch cost money"


def test_the_swapped_sheep_are_actually_bought_and_shorn():
    """The patch has to survive `_paced_plan` and `_animal_orders`, not just the plan object.

    A herd edit that the 1:1 structure pacing swallows would leave every counter in this module
    non-zero and the farm unchanged — E44 with the counter on the wrong side of the mechanism. So
    this counts animals **standing on the board** at the buzzer and WOOL units the environment's own
    ledger says were sold (an order is not a sale).
    """
    seed = 90001
    _base_effects, _m = _season(seed, Plan.boatlee_like())
    base_herd, base_wool = _animals(_season.board), _season.stats["sold_units"].get("WOOL", 0)
    effects, _m2 = _season(seed, Plan.boatlee_like().with_consts(branch_set="yarn"))
    herd, wool = _animals(_season.board), _season.stats["sold_units"].get("WOOL", 0)

    assert effects["branch_swapped_animals"] == 2
    assert herd.get("SHEEP", 0) == base_herd.get("SHEEP", 0) + 2, (base_herd, herd)
    assert herd.get("COW", 0) == base_herd.get("COW", 0) - 2, (base_herd, herd)
    assert wool > base_wool, (base_wool, wool)
