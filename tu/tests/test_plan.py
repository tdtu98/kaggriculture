"""C0: the plan genome round-trips, and refuses plans the game could not execute.

Two properties carry the whole track. **Round-trip fidelity**, because S2 optimises the vector and
reports the plan — if those drift apart the search is scoring something other than what it hands
over. And **validation**, because the compiler (C1/C2) is entitled to assume its input is
buildable; a plan that puts a pasture on land it never buys should fail here, loudly, not three
modules downstream as a routing anomaly.

Both properties are checked against randomly generated genomes rather than hand-picked ones: the
search will produce shapes nobody thought to write down, and every round-trip bug found here was
found that way (a descending hand curve, a herd whose pacing read back doubled).
"""

from __future__ import annotations

import random

import pytest

from agent.plan import (
    BOUNDS,
    COHORT_SLOTS,
    CROPS,
    DAYS,
    FLOOR_PRODUCTS,
    GENE_INDEX,
    GENES,
    LAYOUT_V54,
    NEVER,
    Cohort,
    Plan,
    decode,
    encode,
    encode_fields,
    floor_gene,
    hands_curve,
    migrate,
    pasture_order,
    quadrant_of,
    random_vector,
    snap,
)


def _plans(n: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n):
        yield decode(random_vector(rng))


# --------------------------------------------------------------------- round-trip

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_plans_round_trip_exactly(seed):
    """`decode(encode(p)) == p` for everything the genome can express."""
    for plan in _plans(200, seed=seed):
        assert decode(encode(plan)) == plan


def test_decoding_reaches_a_fixed_point_in_one_pass():
    """The other direction, stated honestly.

    `encode(decode(v)) == snap(v)` is *not* true and should not be: decode repairs genomes that
    ask for the impossible (a cohort on land the plan never buys, more tiles than a quadrant has),
    so the vector that comes back is the repaired one. What has to hold is that repairing is
    idempotent — one pass reaches a plan that survives another unchanged. Otherwise a search would
    watch its candidates drift every time they were re-encoded.
    """
    rng = random.Random(7)
    for _ in range(200):
        v = random_vector(rng)
        once = decode(v)
        assert decode(encode(once)) == once
        assert encode(decode(encode(once))) == encode(once)


def test_boatlee_like_round_trips_and_is_buildable():
    plan = Plan.boatlee_like()
    assert plan.validate() == [], plan.validate()
    assert plan.notes == (), f"the calibration plan should fit the board exactly: {plan.notes}"
    assert decode(encode(plan)) == plan


def test_boatlee_like_matches_the_measured_shape():
    """It is the Phase-1 calibration case, so it has to look like what Boatlee actually does —
    measured on a played season, not taken from prose: 3 quadrants (NE day 6, SW day 10, never SE),
    13 pastures, 4 sheep + 9 cows, ~36 strawberry, melon, and wheat on a replant cycle."""
    plan = Plan.boatlee_like()
    assert plan.land_days == {"NE": 6, "SW": 10, "SE": NEVER}
    assert len(plan.pasture_tiles) == 13
    assert sorted(s for s, _ in plan.herd) == ["COW"] * 9 + ["SHEEP"] * 4

    by_crop: dict = {}
    for c in plan.cohorts:
        by_crop[c.crop] = by_crop.get(c.crop, 0) + c.n_tiles
    assert by_crop["STRAWBERRY"] == 36
    assert by_crop["MELON"] > 0
    assert any(c.replant for c in plan.cohorts if c.crop == "WHEAT"), "wheat cycles"
    assert "TOMATO" not in by_crop and "CARROT" not in by_crop, "boatlee plants neither"


def test_the_hand_curve_survives_a_descent():
    """A plan that sheds hands as the season ends is a legal plan, and it broke the first encoder:
    reading the cap as `max(hands)` rebuilt a ramp-down as a flat line."""
    down = hands_curve(12, 3, 10)
    assert down[0] == 12 and down[-1] == 3
    plan = decode(encode(Plan.boatlee_like()))
    custom = Plan(**{**plan.__dict__, "hands": down,
                     "consts": {**plan.consts, "hands_curve": (12, 3, 10)}})
    assert decode(encode(custom)).hands == down


# --------------------------------------------------------------------- validation

def test_every_random_plan_is_buildable():
    """`decode` must never produce an invalid plan: the search would spend its budget on repairs."""
    for plan in _plans(300, seed=11):
        assert plan.validate() == [], (plan.validate(), plan.to_table())


def test_tiles_are_never_double_booked():
    for plan in _plans(200, seed=3):
        claimed = [tuple(t) for t in plan.pasture_tiles]
        for c in plan.cohorts:
            claimed += [tuple(t) for t in c.tiles]
        assert len(claimed) == len(set(claimed))


def test_a_cohort_on_land_bought_too_late_is_rejected():
    plan = Plan.boatlee_like()
    early = Cohort(crop="STRAWBERRY", quadrant="SW", n_tiles=1, plant_day=2,
                   tiles=((0, 9),))          # SW is not bought until day 10
    broken = Plan(**{**plan.__dict__, "cohorts": plan.cohorts + (early,)})
    assert any("locked on day 2" in e for e in broken.validate()), broken.validate()


def test_a_pasture_on_land_never_bought_is_rejected():
    plan = Plan.boatlee_like()                # SE is never purchased
    broken = Plan(**{**plan.__dict__, "pasture_tiles": plan.pasture_tiles + ((9, 9),)})
    assert any("never buys" in e for e in broken.validate()), broken.validate()


def test_more_animals_than_pastures_is_rejected():
    plan = Plan.boatlee_like()
    broken = Plan(**{**plan.__dict__, "herd": plan.herd + (("COW", 20),) * 5})
    assert any("animals for" in e and "structure tiles" in e for e in broken.validate())


def test_an_animal_bought_before_its_pasture_exists_is_rejected():
    """The constraint that actually bit: with 13 pastures spilling into land bought on day 10, the
    eleventh animal cannot arrive on day 7 however the herd genes read."""
    plan = Plan.boatlee_like()
    late_land = dict(plan.land_days)
    broken = Plan(**{**plan.__dict__,
                     "pasture_tiles": tuple(plan.pasture_tiles[:10]) + ((0, 9), (1, 9), (2, 9)),
                     "land_days": late_land,
                     "herd": tuple(("COW", 5) for _ in range(13))})
    assert any("is not available until day 10" in e for e in broken.validate()), broken.validate()


def test_land_cannot_be_bought_out_of_order():
    plan = Plan.boatlee_like()
    broken = Plan(**{**plan.__dict__, "land_days": {"NE": 12, "SW": 4, "SE": NEVER}})
    assert any("cannot precede" in e for e in broken.validate())


def test_decode_repairs_land_order_rather_than_emitting_an_invalid_plan():
    """Mutation constantly produces SW-before-NE; rejecting those would shrink the search space."""
    rng = random.Random(5)
    for _ in range(100):
        plan = decode(random_vector(rng))
        days = [plan.land_days[q] for q in ("NE", "SW", "SE")]
        bought = [d for d in days if d < NEVER]
        assert bought == sorted(bought), plan.land_days


# --------------------------------------------------------------------- honesty about trimming

def test_a_cohort_that_does_not_fit_is_recorded_not_hidden():
    """`decode` shrinks a cohort to the tiles that exist — and must say so, or the search would be
    scoring plans it has quietly rewritten."""
    from agent.plan import encode_fields

    v = encode_fields(
        n_pastures=0, land_days={"NE": NEVER, "SW": NEVER, "SE": NEVER},
        n_sheep=0, sheep_start=0, n_cows=0, cow_start=0, n_geese=0, geese_start=0,
        animals_per_day=1,
        cohorts=[("WHEAT", "NW", 40, 0, False)],     # NW holds 25
        hands_mode="curve", hands_start=1, hands_cap=1, hands_ramp=1,
    )
    plan = decode(v)
    assert plan.cohorts[0].n_tiles == 25
    assert any("requested 40 tiles" in n for n in plan.notes), plan.notes


def test_a_cohort_on_land_that_is_never_bought_is_dropped_not_left_empty():
    """A zero-tile cohort would round-trip as a phantom the compiler has to skip every day."""
    from agent.plan import encode_fields

    v = encode_fields(
        n_pastures=0, land_days={"NE": NEVER, "SW": NEVER, "SE": NEVER},
        n_sheep=0, sheep_start=0, n_cows=0, cow_start=0, n_geese=0, geese_start=0,
        animals_per_day=1,
        cohorts=[("WHEAT", "SE", 5, 0, False)],      # SE is never purchased
        hands_mode="curve", hands_start=1, hands_cap=1, hands_ramp=1,
    )
    plan = decode(v)
    assert plan.cohorts == ()
    assert any("never bought" in n for n in plan.notes), plan.notes


def test_a_cohort_planted_before_its_land_opens_slides_to_the_purchase_day():
    from agent.plan import encode_fields

    v = encode_fields(
        n_pastures=0, land_days={"NE": 8, "SW": NEVER, "SE": NEVER},
        n_sheep=0, sheep_start=0, n_cows=0, cow_start=0, n_geese=0, geese_start=0,
        animals_per_day=1,
        cohorts=[("MELON", "NE", 4, 2, False)],      # NE does not open until day 8
        hands_mode="curve", hands_start=1, hands_cap=1, hands_ramp=1,
    )
    plan = decode(v)
    assert plan.cohorts[0].plant_day == 8
    assert any("-> 8" in n for n in plan.notes), plan.notes


def test_encode_refuses_a_plan_it_cannot_represent():
    """Hand-picked tiles are allowed for diagnostics but are not in the search space, and encode
    must say so rather than hand back a vector that decodes to a different farm."""
    plan = Plan.boatlee_like()
    moved = Plan(**{**plan.__dict__, "pasture_tiles": ((0, 0), (1, 1))})
    with pytest.raises(ValueError, match="not the canonical cluster"):
        encode(moved)


# --------------------------------------------------------------------- genome shape

def test_the_genome_is_small_and_fully_bounded():
    assert len(GENES) == len(BOUNDS)
    assert 30 <= len(GENES) <= 120, f"{len(GENES)} genes"   # PLAN_v4 §2.1: ~60-120 numbers
    assert all(lo < hi for lo, hi in BOUNDS)
    names = [g.name for g in GENES]
    assert len(names) == len(set(names))


def test_pastures_only_ever_sit_on_land_the_plan_buys():
    order = pasture_order({"NE": 6, "SW": NEVER, "SE": NEVER})
    assert {quadrant_of(*t) for t in order} == {"NW", "NE"}


def test_fertilizer_ages_are_not_in_the_search_space():
    """Fertilize timing is arithmetic (ages 9 and 13 for strawberry, verified in test_env_facts),
    so it is a constant. If it ever becomes a gene, this test should be the thing that objects."""
    assert not any("fert" in g.name for g in GENES)
    assert Plan.boatlee_like().consts["fert_ages"]["STRAWBERRY"] == (9, 13)


def test_snap_clips_and_rounds_into_the_legal_space():
    wild = [-999.0] * len(GENES)
    assert snap(wild) == [g.lo for g in GENES]
    wild = [999.0] * len(GENES)
    assert snap(wild) == [g.hi if not g.integral else float(round(g.hi)) for g in GENES]


def test_a_plan_prints_as_a_readable_table():
    text = Plan.boatlee_like().to_table()
    assert "PLAN" in text and "cohorts" in text and "valid" in text
    assert "STRAWBERRY" in text
    assert len(text.splitlines()) < 25, "a plan should fit on a screen"


# --------------------------------------------------------------------- S3: the widened genome

def test_the_widening_is_the_shape_S3_asked_for():
    """Ten cohort slots and one sell floor per sellable product (S3, E76).

    Pinned as numbers because the whole point of the widening is that the *previous* numbers (6 and
    2) were what the search hit its ceiling against: at six slots a mixed four-quadrant plan is
    truncated by `encode`, and at two floors the harvest pushed both of the dials it had to the rail
    (wool 0.131, melon 0.695) and had nowhere else to go.
    """
    assert COHORT_SLOTS == 10
    assert len(FLOOR_PRODUCTS) == 8
    assert "FERTILIZER" not in FLOOR_PRODUCTS, "fertilizer is an input, not a market position"
    assert [g.name for g in GENES if g.name.startswith("sell_floor_")] == \
           [floor_gene(p) for p in FLOOR_PRODUCTS]
    assert all(not GENES[GENE_INDEX[floor_gene(p)]].integral for p in FLOOR_PRODUCTS)


def test_a_ten_cohort_plan_survives_the_round_trip():
    """Six slots dropped the fourth quadrant on the floor; ten carries it."""
    cohorts = [("WHEAT", "NW", 7, 0, True), ("MELON", "NW", 5, 0, False),
               ("STRAWBERRY", "NE", 13, 6, False), ("WHEAT", "NE", 12, 6, True),
               ("STRAWBERRY", "SW", 13, 10, False), ("MELON", "SW", 6, 11, False),
               ("WHEAT", "SW", 6, 10, True), ("CARROT", "SE", 13, 18, True),
               ("WHEAT", "SE", 6, 18, True), ("TOMATO", "SE", 6, 18, False)]
    v = encode_fields(
        n_pastures=13, land_days={"NE": 6, "SW": 10, "SE": 18},
        n_sheep=4, sheep_start=0, n_cows=9, cow_start=1, n_geese=0, geese_start=0,
        animals_per_day=3, cohorts=cohorts,
        hands_mode="curve", hands_start=4, hands_cap=14, hands_ramp=8,
        release_pressure=70, frontrun_lead=10)
    plan = decode(v)
    assert plan.validate() == [], plan.validate()
    assert len(plan.cohorts) == 10, plan.to_table()
    assert {c.quadrant for c in plan.cohorts} == {"NW", "NE", "SW", "SE"}
    assert decode(encode(plan)) == plan


def test_the_extra_slots_are_empty_so_old_plans_embed_unchanged():
    """A six-cohort plan must not become a ten-cohort plan by being re-read."""
    plan = Plan.boatlee_like()
    assert len(plan.cohorts) == 6
    v = encode(plan)
    for i in range(6, COHORT_SLOTS):
        assert v[GENE_INDEX[f"c{i}_tiles"]] == 0.0
    assert decode(v) == plan


def test_the_new_floors_default_to_selling_on_sight():
    """Behaviour invariance for every plan written before the widening.

    `_sell_orders` reads `floor <= 0` as "sell everything", which is what a product with no floor
    already did — so the six new products must decode to 0.0 and wool/melon must keep their 0.35.
    A default of 0.35 across the board would have silently started metering seven products.
    """
    floors = Plan.boatlee_like().consts["sell_floor"]
    assert set(floors) == set(FLOOR_PRODUCTS)
    assert floors["WOOL"] == 0.35 and floors["MELON"] == 0.35
    assert [floors[p] for p in FLOOR_PRODUCTS if p not in ("WOOL", "MELON")] == [0.0] * 6


def test_each_floor_gene_reaches_its_own_product():
    """One gene, one product — a transposed table would be invisible in the money."""
    for i, product in enumerate(FLOOR_PRODUCTS):
        v = encode(Plan.boatlee_like())
        v[GENE_INDEX[floor_gene(product)]] = 0.5 + i / 100.0
        floors = decode(v).consts["sell_floor"]
        assert floors[product] == round(0.5 + i / 100.0, 3), product
        assert encode(decode(v))[GENE_INDEX[floor_gene(product)]] == round(0.5 + i / 100.0, 3)


def test_a_floor_on_a_product_with_no_gene_is_refused_not_dropped():
    plan = Plan.boatlee_like()
    bad = Plan(**{**plan.__dict__,
                  "consts": {**plan.consts, "sell_floor": {"FERTILIZER": 0.5}}})
    with pytest.raises(ValueError, match="unrepresentable product"):
        encode(bad)


def test_validate_catches_a_quadrant_claimed_twice_over():
    """Reachable at ten slots in one mutation: two full-quadrant cohorts on the same 25 tiles."""
    plan = Plan.boatlee_like()
    over = Plan(**{**plan.__dict__, "pasture_tiles": (), "cohorts": (
        Cohort(crop="WHEAT", quadrant="NE", n_tiles=20, plant_day=6),
        Cohort(crop="MELON", quadrant="NE", n_tiles=20, plant_day=6))})
    assert any("quadrant NE: 40 tiles claimed" in e for e in over.validate()), over.validate()


def test_validate_catches_an_out_of_range_floor():
    plan = Plan.boatlee_like()
    for value, needle in ((1.5, "outside 0..1"), (-0.2, "outside 0..1")):
        bad = Plan(**{**plan.__dict__,
                      "consts": {**plan.consts, "sell_floor": {"MELON": value}}})
        assert any(needle in e for e in bad.validate()), bad.validate()
    unknown = Plan(**{**plan.__dict__,
                      "consts": {**plan.consts, "sell_floor": {"HAY": 0.5}}})
    assert any("no floor gene" in e for e in unknown.validate())


def test_five_thousand_random_genomes_decode_valid_and_round_trip():
    """The property the widened space has to keep: 5,000 in-bounds vectors, no invalid plan and no
    drift. Scale matters here — the failure modes the widening adds (a quadrant claimed by four
    cohorts at once, a slot whose land is never bought) are rare per draw and certain over 5,000."""
    rng = random.Random(4242)
    for i in range(5000):
        v = random_vector(rng)
        plan = decode(v)
        assert plan.validate() == [], (i, plan.validate())
        assert decode(encode(plan)) == plan, i


# --------------------------------------------------------------------- S3: migration

def test_the_v54_layout_is_recorded_exactly_as_it_shipped():
    """`search/log.jsonl` and the harvested candidate are bare lists of 54 floats. This tuple is the
    only thing that says what those coordinates were called, so it is pinned, not derived."""
    assert len(LAYOUT_V54) == 54
    assert len(set(LAYOUT_V54)) == 54
    assert set(LAYOUT_V54) <= {g.name for g in GENES}, \
        "a v54 gene with no home in the current layout would be silently dropped by migrate()"
    assert LAYOUT_V54[:4] == ("n_pastures", "land_NE", "land_SW", "land_SE")
    assert LAYOUT_V54[-1] == "projected_pricing"


def test_migrate_embeds_an_old_genome_without_changing_what_it_does():
    """The invariance the harvested candidate rides on: same plan, same consts, wider vector."""
    old = encode(Plan.boatlee_like())
    v54 = [old[GENE_INDEX[n]] for n in LAYOUT_V54]
    assert len(v54) == 54
    assert migrate(v54) == snap(old)
    assert decode(migrate(v54)) == Plan.boatlee_like()


def test_migrate_preserves_the_two_floors_the_harvest_actually_tuned():
    old = encode(Plan.boatlee_like())
    old[GENE_INDEX[floor_gene("WOOL")]] = 0.131
    old[GENE_INDEX[floor_gene("MELON")]] = 0.695
    v54 = [old[GENE_INDEX[n]] for n in LAYOUT_V54]
    floors = decode(migrate(v54)).consts["sell_floor"]
    assert (floors["WOOL"], floors["MELON"]) == (0.131, 0.695)
    assert [floors[p] for p in FLOOR_PRODUCTS if p not in ("WOOL", "MELON")] == [0.0] * 6


def test_migrate_is_a_no_op_on_a_current_genome_and_refuses_an_unknown_width():
    v = encode(Plan.boatlee_like())
    assert migrate(v) == snap(v)
    with pytest.raises(ValueError, match="no known gene layout"):
        migrate([0.0] * 61)


def test_ten_slots_can_say_what_boatlee_plants():
    """The S3 kill criterion, as a property of the representation.

    Boatlee plants ~196 plants a season and our incumbent plants 92 (measured). The 10-slot
    max-production shape below reaches **182 in play** on the same compiler (six seeds, thirst 6.1
    against a bar of 10, `steps_per_useful` 0.74, zero fallbacks) — the plant ceiling is not what the
    genome was short of any more. This test does not replay that season (it costs a minute); it pins
    the *shape* the measurement was taken on, so a later change to `decode` that quietly trims it
    cannot leave the S3 claim standing on a plan the genome no longer expresses.
    """
    cohorts = [("WHEAT", "NW", 7, 0, True), ("WHEAT", "NW", 5, 0, True),
               ("WHEAT", "NE", 13, 6, True), ("WHEAT", "NE", 12, 6, True),
               ("WHEAT", "SW", 13, 10, True), ("WHEAT", "SW", 12, 10, True),
               ("WHEAT", "SE", 7, 20, True), ("WHEAT", "SE", 6, 20, True),
               ("WHEAT", "SE", 6, 20, True), ("WHEAT", "SE", 6, 20, True)]
    plan = decode(encode_fields(
        n_pastures=13, land_days={"NE": 6, "SW": 10, "SE": 20},
        n_sheep=4, sheep_start=0, n_cows=9, cow_start=1, n_geese=0, geese_start=0,
        animals_per_day=3, cohorts=cohorts,
        hands_mode="curve", hands_start=4, hands_cap=14, hands_ramp=8,
        release_pressure=70, frontrun_lead=10))
    assert plan.validate() == [], plan.validate()
    assert len(plan.cohorts) == 10, plan.to_table()
    # 100 tiles, 13 of them pastures: 87 is the whole plantable board.
    assert sum(c.n_tiles for c in plan.cohorts) == 87, plan.to_table()
    assert all(c.replant for c in plan.cohorts), "cycling is what turns 87 tiles into 180 plants"
