"""Named agent specs for the arena.

Specs are plain picklable data, not closures, because workers run under the `spawn` start method
and have to rebuild the agent themselves. This doubles as the `zoo` pattern from 3rd place: an
entry pins everything needed to reconstruct a competitor, so old entrants stay evaluable after the
engine is refactored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]


def mix(**kw: float) -> dict[str, float]:
    return {**{c: 0.0 for c in CROPS}, **kw}


@dataclass(frozen=True)
class AgentSpec:
    """How to build one competitor. `kind` selects the constructor; `params` is its config."""

    kind: str                       # "baseline" | "engine"
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def build(self) -> Callable[[dict], dict]:
        if self.kind == "designed":
            from arena.opponents import OPPONENTS

            return OPPONENTS[self.params["name"]]()
        if self.kind == "baseline":
            from sim.baselines import AGENTS

            return AGENTS[self.params["name"]]
        if self.kind == "engine":
            from agent import Params, make_agent

            return make_agent(Params(**self.params))
        if self.kind == "relay":
            # PLAN3 R0. `overlays` names functions in `agent.relay`; empty means the reference
            # agent's own stack, which is bit-identical to `boatlee` (tests/test_relay_parity.py).
            from agent import relay

            names = self.params.get("overlays")
            if names is None:
                return relay.make_relay()
            # Thresholds live in the spec, not as module constants, so `tools/promote.py` stage 3
            # can sweep them. E53: the gate reported PASS on a neighbourhood check it never ran,
            # because a relay agent exposed no knobs it could see.
            built = []
            for n in names:
                if n == "surplus_release":
                    built.append(relay.make_surplus_release(
                        pressure=self.params.get("release_pressure", relay.RELEASE_PRESSURE),
                        batch=self.params.get("release_batch", relay.RELEASE_BATCH)))
                else:
                    built.append(getattr(relay, n))
            return relay.make_relay(overlays=tuple(built))
        if self.kind == "external":
            # An agent nobody here wrote. D16 says every other number in this project is
            # self-referential -- 77 agents from one author, reproducing one author's blind spots.
            # This is the only member of the field that can contradict them, and it already has:
            # it beats the champion 24/0 (E26).
            #
            # Loaded exactly the way Kaggle loads a submission -- exec into empty globals, take the
            # last module-level callable -- so what the arena scores is what the runner would run.
            from kaggle_environments.agent import get_last_callable

            path = self.params["path"]
            with open(path) as f:
                return get_last_callable(f.read(), path=path)
        raise ValueError(f"unknown agent kind: {self.kind}")

    @property
    def fingerprint(self) -> str:
        """Stable hash of the spec — the key results are stored under.

        Re-running an unchanged agent reuses its row; changing any knob makes a new one, so a
        results table can never silently mix two different agents under one name.
        """
        blob = json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _engine(note: str, **params: Any) -> AgentSpec:
    return AgentSpec(kind="engine", params=params, note=note)


# The default roster. `docs/experiments.md` E1 ranked these on 6 seeds with a standard deviation
# larger than the mean in one case — which is exactly why they need an arena.
REGISTRY: dict[str, AgentSpec] = {
    "pass": AgentSpec("baseline", {"name": "pass"}, "does nothing; floor at $3,000"),
    "random": AgentSpec("baseline", {"name": "random"}, "buys seeds until broke; ends at $0"),
    "starter": AgentSpec("baseline", {"name": "starter"}, "bundled carrot loop; $3,495"),
    "wheat": _engine("wheat monoculture", hire_max=8, crop_mix=mix(WHEAT=1)),
    "carrot": _engine("carrot monoculture", hire_max=8, crop_mix=mix(CARROT=1)),
    "melon": _engine("melon monoculture", hire_max=8, crop_mix=mix(MELON=1)),
    "melon-wheat": _engine("best config in E1", hire_max=8, crop_mix=mix(MELON=1, WHEAT=1)),
    "wheat-geese": _engine("geese with CARE", hire_max=8, crop_mix=mix(WHEAT=1),
                           goose_target=8, care=True, goose_min_cash=200),
    "melon-wheat-geese": _engine("all three lines", hire_max=8, crop_mix=mix(MELON=1, WHEAT=1),
                                 goose_target=8, care=True, goose_min_cash=200),
    # Land is now a real choice rather than a dead branch (the old gate was unreachable).
    "melon-wheat-land": _engine("diversified + buys quadrants", hire_max=8, buy_land=True,
                                crop_mix=mix(MELON=1, WHEAT=1)),
    "melon-geese": _engine("melon + a goose line", hire_max=8, crop_mix=mix(MELON=1),
                           goose_target=8, care=True, goose_min_cash=200),
    # T1.3: price against the opponent's visible incoming supply instead of a static reserve.
    # `melon` above now carries the tuned default (w=0.7, horizon=10); these pin the comparison.
    "melon-static": _engine("no forecast — the pre-T1.3 best", hire_max=8, crop_mix=mix(MELON=1),
                            forecast_weight=0.0),
    "melon-fc06h4": _engine("first working forecast setting", hire_max=8, crop_mix=mix(MELON=1),
                            forecast_weight=0.6, forecast_horizon=4),
    "melon-fc10": _engine("forecast only, no static reserve", hire_max=8, crop_mix=mix(MELON=1),
                          forecast_weight=1.0),
}


# Two files, deliberately separate:
#   search/champion.json    the ACCEPTED best — promoted only after winning in the arena on seeds
#                           the search never saw.
#   search/best_params.json the LATEST search output — a hypothesis until validated.
# Without the split, a search would silently overwrite the thing it is supposed to beat.
def _load(path: str, note: str) -> AgentSpec | None:
    import os

    if not os.path.exists(path):
        return None
    with open(path) as f:
        return AgentSpec("engine", json.load(f)["params"], note)


_champ = _load("search/champion.json", "accepted champion (CEM, T2.1)")
if _champ is not None:
    REGISTRY["champion"] = _champ
else:                       # bootstrap before the first promotion
    REGISTRY["champion"] = REGISTRY["melon"]

_ablation = _load("search/dumponly.json",
                  "T2.5 ablation: reserves + forecast pinned to 0, everything else tuned")
if _ablation is not None:
    REGISTRY["x-dumponly"] = _ablation

_latest = _load("search/best_params.json", "latest search output — UNVALIDATED")
if _latest is not None:
    REGISTRY["cem"] = _latest


def _all(v: float) -> dict[str, float]:
    return {q: v for q in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                           "EGG", "MILK", "WOOL", "FERTILIZER"]}


# T2.2 exploiters. The champion is a single-opponent optimum, so probe it from several directions:
# each of these is a *strategy* the champion never faced during its search.
def _variant(name: str, note: str, **over) -> None:
    base = dict(REGISTRY["champion"].params)
    base.update(over)
    REGISTRY[name] = AgentSpec("engine", base, note)


_variant("x-turtle", "never crashes the market; sells only into recovered prices",
         forecast_weight=0.0, reserve_frac=_all(0.9))
_variant("x-dumper", "sells everything instantly at any price — maximum denial",
         forecast_weight=0.0, reserve_frac=_all(0.0))
_variant("x-frontrun", "prices purely off the opponent's incoming supply, long horizon",
         forecast_weight=1.0, forecast_horizon=16)
_variant("x-melon-race", "pure melon, maximum aggression into the melon curve",
         crop_mix=mix(MELON=1.0), forecast_weight=1.0, forecast_horizon=10,
         reserve_frac=_all(0.0))
_variant("x-hoard", "holds everything until the final dump",
         forecast_weight=0.0, reserve_frac=_all(1.1), sell_all_after_day=29)
_variant("x-labour", "same strategy, far more hands", hire_max=12)


# Livestock probe: the champion with an animal line forced on. Eggs absorb ~$114k and fertilizer
# ~$25k — by far the largest markets — yet CEM chose goose_target = 0. Test rather than trust.
for _n in (4, 8, 12, 18):
    _variant(f"g{_n}", f"champion + {_n} geese", goose_target=_n, goose_min_cash=300, care=True)
_variant("g8-alt", "8 geese, alternate-day feeding", goose_target=8, goose_min_cash=300,
         care=False, feed_alternate=True)


_variant("a-first", "geese funded before seeds", animals_before_seeds=True)
_variant("a-first-g10", "geese first, target 10", animals_before_seeds=True, goose_target=10)
_variant("a-first-g14", "geese first, target 14", animals_before_seeds=True, goose_target=14)


# V3: species that were unbuildable until pastures were implemented.
_variant("v-cow8", "cows only, 8 head", goose_target=0, cow_target=8)
_variant("v-cow12", "cows only, 12 head", goose_target=0, cow_target=12)
_variant("v-cow6-goose3", "mixed herd", goose_target=3, cow_target=6)
_variant("v-sheep8", "sheep only, 8 head", goose_target=0, sheep_target=8)


# V4: E11 ("never reserve") was measured in a melon-only field. Melon has ZERO shop demand, so its
# price cannot recover and holding provably loses. Milk regenerates at ~19/day. Does a reservation
# price pay for regenerating products specifically?
def _res(**over):
    base = {q: 0.0 for q in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON",
                             "EGG","MILK","WOOL","FERTILIZER"]}
    base.update(over); return base

_HIGH = {q: 0.35 for q in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MILK","EGG","WOOL"]}
_variant("r-regen", "reserve on regenerating products only, melon still dumped",
         reserve_frac=_res(**_HIGH))
_variant("r-all", "reserve on everything including melon", reserve_frac=_res(
    **{q: 0.35 for q in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"]}))
_variant("r-melon", "reserve on melon only — should be worst", reserve_frac=_res(MELON=0.35))

# V5/E6 re-test: the champion runs 7 cows on 25 tiles. Pastures are tiles, so land now buys herd
# size — and land was rejected under a melon strategy that could not use the space.
_variant("l-cow16", "buy land, 16 cows, more hands", buy_land=True, cow_target=16,
         hire_max=11, land_min_cash=500.0, tiles_per_unit=9.0)
_variant("cow16", "16 cows, no land", cow_target=16)
_variant("cow-sheep", "two regenerating animal markets", cow_target=7, sheep_target=6)


# Prices sit ABOVE base all season (scarcity), so a reserve only binds above 1.0x base. The
# earlier 0.35x test never triggered — it compared two identical agents.
for _f in (0.9, 1.1, 1.3):
    _variant(f"h-res{_f}", f"reserve at {_f}x base on regenerating products",
             reserve_frac=_res(**{q: _f for q in
                                  ["WHEAT","CARROT","TOMATO","STRAWBERRY","MILK","EGG","WOOL"]}))
_variant("h-res-all1.1", "reserve 1.1x base on everything incl. melon",
         reserve_frac=_res(**{q: 1.1 for q in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON",
                                               "EGG","MILK","WOOL","FERTILIZER"]}))


# Routing: E6 concluded units were idle rather than travel-bound. With the cow-led champion that
# reversed — movement is 58-65% of every turn — so the assignment rule matters again.
for _w in (0.0, 1.0, 3.0):
    _variant(f"pw{_w}", f"priority_weight {_w}", priority_weight=_w)


# Herd-mix sweep: the gauntlet showed 7 cows + 6 sheep beating CEM's 8 cows + 4 sheep 64.4%.
# Both animals share one fertilizer stream; they differ in product (milk ~19/day drain vs wool
# ~14/day), price ($160 vs $200) and harvest cadence (every 2 days vs every 3).
for _c, _sh in [(6, 8), (7, 6), (5, 9), (8, 8), (4, 10), (10, 4), (6, 6), (0, 12),
                (5, 8), (5, 10), (4, 9), (6, 9), (3, 11), (5, 12)]:
    _variant(f"herd{_c}c{_sh}s", f"{_c} cows + {_sh} sheep", cow_target=_c, sheep_target=_sh)


# hire_max sweep. E1 measured 12+ hands as bankrupting — under the melon engine, before cows,
# routing, and cash management. The gate walked 8 -> 9 -> 10 one step at a time, so find the top.
for _h in (7, 8, 9, 10, 11, 12, 13, 14):
    _variant(f"hire{_h}", f"hire_max {_h}", hire_max=_h,
             reserve_frac={q: (0.35 if q in ("WHEAT","CARROT","TOMATO","STRAWBERRY",
                                             "EGG","MILK","WOOL") else 0.0)
                           for q in ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON",
                                     "EGG","MILK","WOOL","FERTILIZER"]})


# The first external opponent (E22, E25, E26). Beats the champion 24/0; loses 24/0 with its
# BUY_LAND orders stripped, which is how we learned land decides this matchup. The ablations are
# registered too -- they are the only opponents in the field with a known, causal reason for their
# strength, which makes them useful gauntlet rungs rather than just another agent.
REGISTRY["boatlee"] = AgentSpec(
    kind="external", params={"path": "reference/kaggriculture/1/submission.py"},
    note="external: hardcoded 719-step plan + closed-loop overlays; beats champion 24/0")


# E39: greedy assignment measured 18% off optimal; the engine can solve it exactly at this size.
_variant("assign-optimal", "optimal unit->task matching (E39)", assign_mode="optimal")


# Diagnostic variants for the strawberry investigation (E43). Registered rather than built in a
# throwaway script so the traces they produce are reproducible from a name.
_variant("straw-only", "strawberry-only crop mix, everything else champion",
         crop_mix={"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0, "STRAWBERRY": 1.0, "MELON": 0.0})
_variant("straw-only-eager", "strawberry-only, waters ongoing crops every day",
         crop_mix={"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0, "STRAWBERRY": 1.0, "MELON": 0.0},
         water_ongoing_eager=True)
_variant("straw-only-both", "strawberry-only, water alongside harvest",
         crop_mix={"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0, "STRAWBERRY": 1.0, "MELON": 0.0},
         water_mode="both")


# E44: match boatlee on ALL measured dimensions at once, rather than one knob at a time.
# Every link in land -> tiles -> seeds -> watering -> fertilising -> harvest is worthless without
# the others (E43), and CEM samples dimensions independently so it cannot find the conjunction
# (the failure already recorded for `goose_min_cash` in E12).
# Target profile: 3 quadrants, ~57 wheat + ~36 strawberry tiles, ~200 seeds, ~1000 waterings,
# ~72 fertilise ops, low hauling.
_variant("mimic", "matches boatlee's measured profile on every dimension",
         buy_land=True, land_min_cash=300.0, land_fill_frac=0.85,
         crop_mix={"WHEAT": 1.0, "CARROT": 0.0, "TOMATO": 0.0,
                   "STRAWBERRY": 0.63, "MELON": 0.25},
         seed_budget_frac=0.85, tiles_per_unit=12.0,
         water_mode="both", water_ongoing_eager=True,
         fertilize=True, fertilize_batch=6,
         reserve_frac={"WHEAT": 0.35, "CARROT": 0.35, "TOMATO": 0.35, "STRAWBERRY": 0.35,
                       "MELON": 0.0, "EGG": 0.35, "MILK": 0.35, "WOOL": 0.35,
                       "FERTILIZER": 0.6},
         assign_mode="optimal", cow_target=9, sheep_target=4)


# E45: refuse plantings that cannot reach first yield before the season ends. The champion plants
# 42 wheat on day 27, which can never yield -- the seed, the planting turn and every watering are
# pure loss.
_variant("stop-late", "no plantings that cannot mature before the season ends",
         plant_stop_late=True)


# E46: unit role specialisation. boatlee keeps its units 93% role-pure; ours switch between crop
# and animal work on 33% of consecutive actions, and the two happen in different parts of the farm.
_variant("role3", "champion + role specialisation (penalty 3, past the cliff)", role_penalty=3.0)
_variant("role2", "champion + role specialisation (penalty 2)", role_penalty=2.0)
_variant("role1.5", "champion + role specialisation (penalty 1.5)", role_penalty=1.5)
# Stage 3 of role1.5's gate, and independently the champion's own audit, both name cow_target=5.
_variant("role1.5-cow5", "role specialisation + the neighbour both gates pointed at",
         role_penalty=1.5, cow_target=5)


# E47: finish the tile you stand on before moving. Mechanism verified (walk-aways 829 -> 373,
# same-tile 29% -> 35%); kept registered so the trace comparison is reproducible.
_variant("finish-tile", "champion + finish the tile you are standing on", finish_tile=True)


# Land, re-tested. The farm is now 25/25 full with the herd at target, and every "land loses"
# measurement (E1, E6, E14) was taken under a melon strategy with 8 hands, bad routing and few
# animals. A conclusion is only valid for the strategy it was measured on — that has now bitten
# three times (E6->E17, E16->E19, hire_max 8->10).
_variant("land-a", "buy land, herd 10c+12s", buy_land=True, land_min_cash=400.0,
         cow_target=10, sheep_target=12, land_fill_frac=0.75)
_variant("land-b", "buy land, herd 14c+16s, 12 hands", buy_land=True, land_min_cash=400.0,
         cow_target=14, sheep_target=16, hire_max=12, land_fill_frac=0.75)
_variant("land-c", "buy land, keep herd, more crops", buy_land=True, land_min_cash=400.0,
         land_fill_frac=0.75, tiles_per_unit=6.0)
_variant("land-d", "buy land late (big reserve), big herd", buy_land=True, land_min_cash=4000.0,
         cow_target=12, sheep_target=14, land_fill_frac=0.9)


# V2: opponents written from a design brief, not by mutating champion parameters.
for _name in ("o-goose-baron", "o-shop-chaser", "o-land-baron", "o-sprinter"):
    REGISTRY[_name] = AgentSpec("designed", {"name": _name}, "independently designed (V2)")


# PLAN3 R0.3: the relay line. `relay-base` is the reference agent's 719-step table plus its own five
# overlays, restructured into `agent/relay.py` so behaviour can be added without touching
# `reference/kaggriculture/1/submission.py` -- the arena's only external opponent and the only
# non-self-referential measurement in the project (D16).
#
# It is **bit-identical** to `boatlee`, proven step-by-step over 20 seeds x 2 seats by
# `tests/test_relay_parity.py`, not asserted here. Every later variant is this entry plus one named
# overlay, so an A/B isolates the overlay and nothing else -- which is the whole reason the fixed
# table is a better experimental substrate than our own engine (E48).
REGISTRY["relay-base"] = AgentSpec(
    kind="relay", params={},
    note="PLAN3 R0: reference table + its own overlays; bit-identical to boatlee")

# PLAN3 R1. Same table, same logistics, but the herd is chosen from the shops this game drew
# instead of committed before any shop unlocks. Purchase, shed pickup and placement are resolved
# against the stock actually held, because rewriting the purchase alone strands the animal and
# costs the whole farm (E50: 1,667 blocked ops, $689).
REGISTRY["relay-herd"] = AgentSpec(
    kind="relay",
    params={"overlays": ["weed_repair", "adaptive_livestock", "wool_controller",
                         "rank_sell_slots", "market_relay"]},
    note="PLAN3 R1: COW/SHEEP chosen from observed shop demand")

# PLAN3 R1, restricted to the cheaper swap direction (SHEEP -> COW). Separates "the idea is wrong"
# from "the implementation strands animals": this direction always settles if the scripted purchase
# would have, so a loss here refutes the hypothesis rather than the code.
REGISTRY["relay-herd-down"] = AgentSpec(
    kind="relay",
    params={"overlays": ["weed_repair", "adaptive_livestock_downgrade", "wool_controller",
                         "rank_sell_slots", "market_relay"]},
    note="PLAN3 R1: drop sheep when no yarn store; cheaper direction only")

# PLAN3 R2e. The sell half alone -- on an unmodified table there is no surplus, so this is close to
# a no-op by construction. That is its null test (E50: relay-base strands 0.1 items/season).
REGISTRY["relay-sell"] = AgentSpec(
    kind="relay",
    params={"overlays": ["weed_repair", "convert_livestock", "wool_controller",
                         "surplus_release", "rank_sell_slots", "market_relay"],
            "release_pressure": 70, "release_batch": 8},
    note="PLAN3 R2e: release stock the remaining script will never sell")

# PLAN3 R2e proper: production and sales changed together. Neither half pays alone (PLAN3 SS2).
REGISTRY["relay-paired"] = AgentSpec(
    kind="relay",
    params={"overlays": ["weed_repair", "adaptive_livestock_downgrade", "wool_controller",
                         "surplus_release", "rank_sell_slots", "market_relay"]},
    note="PLAN3 R2e: adaptive herd PLUS a sell schedule that can follow it")


def resolve(names: list[str]) -> dict[str, AgentSpec]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown agent(s): {unknown}\navailable: {sorted(REGISTRY)}")
    return {n: REGISTRY[n] for n in names}


def spec_to_dict(spec: AgentSpec) -> dict:
    return asdict(spec)
