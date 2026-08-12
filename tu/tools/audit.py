"""Rule-coverage audit for the parity harness.

The fuzz parity in `tests/test_parity.py` proves kagsim agrees with the reference *on the paths it
reaches*. That is only worth as much as the coverage, so this script watches the reference env's
own state transitions and counts how many times each game rule actually fired.

Any rule with a count of 0 is untested by fuzzing, no matter how many episodes ran, and needs a
directed scenario test instead.

Usage:  PYTHONPATH=. python tools/audit.py [episodes] [steps]
"""

from __future__ import annotations

import sys
from collections import Counter

from kaggle_environments import make

import kagsim
from tests.parity import Fuzzer, diff, reference_canonical

EVENTS = [
    "plant_created", "plant_watered_bonus", "onetime_harvested", "ongoing_harvested",
    "plant_weeded_dry", "plant_decayed", "weed_dug", "weed_spawned_random",
    "coop_built", "pasture_built", "animal_placed", "animal_escaped", "animal_harvested",
    "animal_produced", "fertilizer_collected", "care_banked", "care_paid", "fertilize_applied",
    "hire", "hand_spawned_on_locked", "land_bought", "shed_overflow_discarded",
    "sold_at_floor", "sold_above_floor", "bought_product", "bought_animal", "bought_seed",
    "shop_unlocked", "market_price_rose", "market_price_fell", "atomic_plant_blocked",
    "pickup", "shed_drop", "unit_blocked_on_locked_tile",
]


def tile_kind(t: list) -> str:
    return t[0]


def count_events(prev: dict, cur: dict, ev: Counter) -> None:
    """Diff two canonical states and attribute the changes to rules."""
    for p in range(2):
        a, b = prev["farms"][p], cur["farms"][p]
        pa, pb = prev["privates"][p], cur["privates"][p]

        for t0, t1 in zip(a["tiles"], b["tiles"]):
            k0, k1 = tile_kind(t0), tile_kind(t1)
            if k0 == "EMPTY" and k1 == "PLANT":
                ev["plant_created"] += 1
            elif k0 == "PLANT" and k1 == "EMPTY":
                ev["onetime_harvested"] += 1
            elif k0 == "PLANT" and k1 == "PLANT":
                if t1[5] > t0[5] and t1[3] and not t0[3]:
                    ev["plant_watered_bonus"] += 1
                if t1[5] < t0[5]:
                    ev["ongoing_harvested" if t1[5] == 0 else "plant_decayed"] += 1
                if t1[7] > t0[7]:
                    ev["fertilize_applied"] += 1
            elif k0 == "PLANT" and k1 == "WEED":
                # A dry plant weeds at the day boundary; decay weeds it mid-day at 0 units.
                ev["plant_weeded_dry" if t0[4] >= 1 else "plant_decayed"] += 1
            elif k0 == "EMPTY" and k1 == "WEED":
                ev["weed_spawned_random"] += 1
            elif k0 == "WEED" and k1 == "EMPTY":
                ev["weed_dug"] += 1
            elif k0 == "EMPTY" and k1 == "COOP":
                ev["coop_built"] += 1
            elif k0 == "EMPTY" and k1 == "PASTURE":
                ev["pasture_built"] += 1
            elif k0 in ("COOP", "PASTURE") and k1 == f"{k0}_A":
                ev["animal_placed"] += 1
            elif k0.endswith("_A") and k1 == k0[:-2]:
                ev["animal_escaped"] += 1
            elif k0.endswith("_A") and k1 == k0:
                if t1[3] < t0[3]:
                    ev["animal_harvested"] += 1
                if t1[3] > t0[3]:
                    ev["animal_produced"] += 1
                if not t1[7] and t0[7]:
                    ev["fertilizer_collected"] += 1
                if t1[8] > t0[8]:
                    ev["care_banked"] += 1
                if t1[8] < t0[8]:
                    ev["care_paid"] += 1

        if b["hires_today"] > a["hires_today"]:
            ev["hire"] += b["hires_today"] - a["hires_today"]
            for h in b["hands"][len(a["hands"]):]:
                idx = h[1] * 10 + h[0]
                if tile_kind(b["tiles"][idx]) == "LOCKED":
                    ev["hand_spawned_on_locked"] += 1
        if len(b["unlocked"]) > len(a["unlocked"]):
            ev["land_bought"] += 1

        for k, v in pb["shed"].items():
            v0 = pa["shed"].get(k, 0)
            if v > v0 and k in ("GOOSE", "COW", "SHEEP"):
                ev["bought_animal"] += 1
        for k, v in pb["seeds"].items():
            if v > pa["seeds"].get(k, 0):
                ev["bought_seed"] += 1

        inv_before = sum(n for inv in pa["inventories"] for _, n in inv)
        inv_after = sum(n for inv in pb["inventories"] for _, n in inv)
        shed_before, shed_after = sum(pa["shed"].values()), sum(pb["shed"].values())
        if inv_after < inv_before and shed_after == shed_before and cur["hour"] == 0:
            ev["shed_overflow_discarded"] += 1
        if inv_after > inv_before and shed_after < shed_before:
            ev["pickup"] += 1
        if inv_after < inv_before and shed_after > shed_before and cur["hour"] != 0:
            ev["shed_drop"] += 1

    for k in cur["market_inv"]:
        d = cur["market_inv"][k] - prev["market_inv"][k]
        if d > 0:
            ev["sold_above_floor"] += d
        p0, p1 = prev["market_prices"][k], cur["market_prices"][k]
        if p1 > p0:
            ev["market_price_rose"] += 1
        elif p1 < p0:
            ev["market_price_fell"] += 1
        if p1 == 1 and p0 == 1:
            ev["sold_at_floor"] += 1
    if len(cur["shops"]) > len(prev["shops"]):
        ev["shop_unlocked"] += 1


def engine_configs():
    """Competent-play action sources.

    Fuzzing plays badly, so it never reaches the rules a real agent depends on — ongoing crops,
    the animal production loop, decay, fertilizer, the price floor. These configs do.
    """
    from agent import Params, make_agent

    def mix(**kw):
        base = {c: 0.0 for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]}
        return {**base, **kw}

    return {
        "melon": lambda: make_agent(Params(hire_max=8, crop_mix=mix(MELON=1))),
        "tomato": lambda: make_agent(Params(hire_max=8, crop_mix=mix(TOMATO=1))),
        "strawberry": lambda: make_agent(Params(hire_max=8, crop_mix=mix(STRAWBERRY=1))),
        "geese": lambda: make_agent(
            Params(hire_max=8, crop_mix=mix(WHEAT=1), goose_target=8,
                   care=True, goose_min_cash=200)
        ),
        "cheap-melon": lambda: make_agent(
            Params(hire_max=4, crop_mix=mix(MELON=1, WHEAT=1), reserve_frac={
                k: 0.0 for k in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                                 "EGG", "MILK", "WOOL", "FERTILIZER"]})
        ),  # zero reserve -> dumps into the floor, exercising sold_at_floor
    }


def run_one(name, make_actions, seed, steps, ev, verbose=True):
    cfg = {"episodeSteps": steps + 2, "seed": seed}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)
    sim = kagsim.Sim(dict(cfg))
    prev = reference_canonical(env)
    if prev != sim.canonical_state():
        print(f"INIT DIVERGENCE {name} seed={seed}")
        return 1
    act = make_actions()
    for i in range(steps):
        actions = act(env)
        env.step(actions)
        sim.step(actions)
        cur, got = reference_canonical(env), sim.canonical_state()
        if cur != got:
            print(f"DIVERGENCE {name} seed={seed} step={i}\n" + "\n".join(diff(cur, got)))
            return 1
        count_events(prev, cur, ev)
        prev = cur
    if verbose:
        print(f"  {name} seed={seed}: ok ({steps} steps)", flush=True)
    return 0


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 718
    ev: Counter = Counter()
    divergences = 0

    print("== fuzz-driven ==")
    for e in range(episodes):
        def mk(e=e):
            fz = Fuzzer(1000 + e)
            return lambda env: [fz.player_action(env.state[p].observation, p) for p in range(2)]
        divergences += run_one("fuzz", mk, e, steps, ev)

    print("== engine-driven (competent play) ==")
    for name, factory in engine_configs().items():
        for e in range(2):
            def mk(factory=factory):
                agents = [factory(), factory()]
                return lambda env: [agents[p](env.state[p].observation) for p in range(2)]
            divergences += run_one(name, mk, e, steps, ev)

    print(f"\n{'rule / event':<32}{'times fired':>12}")
    print("-" * 44)
    missed = []
    for name in EVENTS:
        n = ev.get(name, 0)
        print(f"{name:<32}{n:>12,}")
        if n == 0:
            missed.append(name)

    print(f"\nfuzz episodes={episodes}, engine episodes={2*len(engine_configs())}, "
          f"steps={steps}, DIVERGENCES={divergences}")
    # These cannot be attributed from a state diff (they are no-ops, or indistinguishable from
    # other transitions), so a 0 here means "no detector", not "untested".
    NO_DETECTOR = {"atomic_plant_blocked", "unit_blocked_on_locked_tile", "bought_product"}
    covered_elsewhere = {
        "fertilize_applied": "tests/test_rules_coverage.py::test_fertilizer_doubles_*",
        "sold_at_floor": "tests/test_parity.py::test_melon_price_crashes_and_floors",
        "atomic_plant_blocked": "tests/test_parity.py::test_atomic_plant_blocks_*",
        "unit_blocked_on_locked_tile": "tests/test_rules_coverage.py::test_unit_can_cross_*",
        "bought_product": "tests/test_parity.py::test_buy_then_sell_round_trip_nets_zero",
    }
    gaps = [m for m in missed if m not in covered_elsewhere]
    if missed:
        print("\nNot reached by fuzz or engine play:")
        for m in missed:
            tag = " (no state-diff detector)" if m in NO_DETECTOR else ""
            print(f"  {m}{tag} -> {covered_elsewhere.get(m, 'UNTESTED')}")
    if gaps:
        print(f"\n*** {len(gaps)} RULE(S) WITH NO TEST AT ALL: {', '.join(gaps)} ***")
    else:
        print("\nEvery tracked rule is exercised, by play or by a directed test.")


if __name__ == "__main__":
    main()
