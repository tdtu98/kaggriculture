"""T0.4 — parity harness between the reference Python env and kagsim.

Layer A: deterministic core (weeds off, shop unlocks off) — isolates the rules from the RNG.
Layer B: RNG in isolation — see test_rng.py.
Layer C: full default config.
"""

from __future__ import annotations

import random
from typing import Any

from kaggle_environments import make

import kagsim

ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
         "FERTILIZER", "GOOSE", "COW", "SHEEP"]
PRODUCTS = ITEMS[:9]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

UNIT_OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "FERTILIZE",
            "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]


# --------------------------------------------------------------------------- canonical state

def tile_canonical(tile: Any) -> list:
    """Must match `Sim::canonical_state`'s tile encoding exactly.

    An *empty* structure is literally `{"kind": "COOP"}` with no other keys, so it gets a
    distinct shape from an occupied one rather than relying on `.get()` defaults.
    """
    if tile is None:
        return ["EMPTY"]
    if tile == "LOCKED":
        return ["LOCKED"]
    kind = tile.get("kind")
    if kind == "WEED":
        return ["WEED"]
    if kind == "PLANT":
        return ["PLANT", tile["crop"], tile["planted_day"], tile["watered_today"],
                tile["consecutive_unwatered"], tile["yield_units"],
                tile["max_lifespan_step"], tile["fertilized_until_day"]]
    if kind in ("COOP", "PASTURE"):
        if "animal" not in tile:
            return [kind]
        return [f"{kind}_A", tile["animal"], tile["placed_day"], tile["yield_units"],
                tile["consecutive_unfed"], tile["fed_today"], tile["cared_today"],
                tile["fertilizer_available"], tile["pending_care_bonus"]]
    raise AssertionError(f"unknown tile {tile!r}")


def reference_canonical(env) -> dict:
    obs0 = env.state[0].observation
    farms = []
    for f in obs0["farms"]:
        farms.append({
            "money": float(f["money"]),
            "farmer": list(f["farmer"]),
            "hands": [list(h) for h in f["hands"]],
            "unlocked": list(f["unlocked_quadrants"]),
            "hires_today": f["hires_today"],
            "tiles": [tile_canonical(t) for row in f["tiles"] for t in row],
        })
    privs = []
    for s in env.state:
        p = s.observation["private"]
        privs.append({
            "shed": {k: v for k, v in p["shed"].items() if v != 0},
            "seeds": {k: v for k, v in p["seeds"].items() if v != 0},
            # Insertion order is observable (see state.rs::Inv), so keep it as a list.
            "inventories": [[[k, v] for k, v in inv.items()] for inv in p["inventories"]],
        })
    m = obs0["market"]
    # `market["params"]` is only present when marketParams was supplied (`:169`), and it lands in
    # the shared observation, so it is observable state and must be compared.
    params = None
    if "params" in m:
        keys = ("base", "I0", "T", "below_func", "below_target", "above_func", "above_target")
        def norm(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        params = {k: {kk: norm(m["params"][k][kk]) for kk in keys} for k in PRODUCTS}
    return {
        "step": obs0["step"],
        "day": obs0["day"],
        "hour": obs0["hour"],
        "shops": list(obs0["town"]["unlocked_shops"]),
        "market_inv": {k: m["inventory"][k] for k in PRODUCTS},
        "market_prices": {k: m["prices"][k] for k in PRODUCTS},
        **({"market_params": params} if params is not None else {}),
        "farms": farms,
        "privates": privs,
    }


def diff(a: Any, b: Any, path: str = "", out: list | None = None, limit: int = 20) -> list[str]:
    """Field-level diff. Without this, a divergence 300 steps in tells you nothing."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: MISSING_IN_REF vs {b[k]!r}")
            elif k not in b:
                out.append(f"{path}.{k}: {a[k]!r} vs MISSING_IN_SIM")
            else:
                diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


# --------------------------------------------------------------------------- action fuzzer

class Fuzzer:
    """Biased toward exercising no-op and edge paths, not just legal play.

    Ports diverge on the paths nobody intends to take, so ~30% of generated ops are deliberately
    illegal and ~10% are market storms.
    """

    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def unit_action(self, obs: dict, player: int, pos: tuple[int, int]) -> list:
        r = self.r
        roll = r.random()
        priv = obs["private"]
        if roll < 0.40:
            # legal-ish: act on the tile we're standing on, or step toward something
            farm = obs["farms"][player]
            tile = farm["tiles"][pos[1]][pos[0]]
            if tile is None:
                seeds = [c for c, n in priv["seeds"].items() if n > 0]
                if seeds and r.random() < 0.6:
                    return ["PLANT", r.choice(seeds)]
                return [r.choice(["BUILD_COOP", "BUILD_PASTURE", "NORTH", "SOUTH", "EAST", "WEST"])]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                return [r.choice(["WATER", "HARVEST", "WATER", "FERTILIZE", "DIG"])]
            if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                if "animal" in tile:
                    return [r.choice(["FEED", "HARVEST", "COLLECT_FERTILIZER", "CARE"])]
                animals = [a for a in ANIMALS if priv["shed"].get(a, 0) > 0]
                if animals:
                    return ["PLACE", r.choice(animals)]
                return [r.choice(["DIG", "NORTH", "SOUTH"])]
            return [r.choice(UNIT_OPS)]
        if roll < 0.70:
            return [r.choice(UNIT_OPS)]
        if roll < 0.90:
            # deliberately illegal / edge
            return r.choice([
                ["PLANT", r.choice(CROPS)],
                ["PLANT", "BANANA"],
                ["PLANT"],                       # len < 2 -> _apply_unit_action guard
                ["PLACE"],                       # len < 2 -> PLACE guard
                ["PLACE", r.choice(ITEMS)],      # shed-drop path, possibly away from the shed
                ["PICKUP", r.choice(ITEMS)],
                ["PICKUP", r.choice(ITEMS), r.randint(-2, 5)],
                ["PLACE", r.choice(ITEMS), r.randint(-2, 5)],
                ["PLACE", r.choice(ANIMALS)],
                ["FEED"], ["CARE"], ["COLLECT_FERTILIZER"], ["DIG"], ["FERTILIZE"],
                ["DROP"], ["HARVEST"],
                ["NOT_AN_OP"], [], ["PICKUP"],
            ])
        return ["PASS"]

    def market(self, obs: dict, player: int) -> list:
        r = self.r
        priv = obs["private"]
        orders: list = []
        n = r.randint(0, 12)  # sometimes over the 10-order cap
        for _ in range(n):
            roll = r.random()
            if roll < 0.30:
                orders.append(["BUY_SEED", r.choice(CROPS), r.randint(1, 3)])
            elif roll < 0.50:
                held = [k for k, v in priv["shed"].items() if v > 0]
                item = r.choice(held) if held and r.random() < 0.7 else r.choice(ITEMS)
                orders.append(["SELL", item, r.randint(1, 40)])
            elif roll < 0.62:
                orders.append(["BUY_PRODUCT", r.choice(["WHEAT", "FERTILIZER", "MELON"]),
                               r.randint(1, 5)])
            elif roll < 0.72:
                orders.append(["BUY_ANIMAL", r.choice(ANIMALS), 1])
            elif roll < 0.88:
                orders.append(["HIRE"])
            elif roll < 0.94:
                orders.append(["BUY_LAND"])
            else:
                orders.append(r.choice([["SELL", "NOPE", 1], ["BUY_SEED", "BANANA", 1],
                                        ["SELL", "WHEAT", 0], ["SELL", "WHEAT", -3],
                                        ["SELL", "WHEAT"],      # len < 3 -> _parse_order guard
                                        ["BUY_SEED", "WHEAT"],  # len < 3
                                        ["BOGUS"], []]))
        return orders

    def player_action(self, obs: dict, player: int) -> dict:
        farm = obs["farms"][player]
        hands = farm["hands"]
        act = {
            "farmer": self.unit_action(obs, player, tuple(farm["farmer"])),
            "hands": [self.unit_action(obs, player, tuple(h)) for h in hands],
            "market": self.market(obs, player),
        }
        # Occasionally send more hand actions than there are hands.
        if self.r.random() < 0.05:
            act["hands"].append(["NORTH"])
        # ...and occasionally send a `hands` value that is not a list at all (`:894`).
        if self.r.random() < 0.03:
            act["hands"] = "not-a-list"
        return act


# --------------------------------------------------------------------------- runner

LAYER_A = {"weedSpawnChance": 0.0, "townShopUnlockInterval": 10**6}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def run_scripted(config: dict, script: list[dict], steps: int | None = None):
    """Drive both implementations with a fixed action script for player 0.

    Asserts parity every step and returns the final canonical state, so scenario tests get both
    a parity check and something concrete to assert on.
    """
    steps = steps if steps is not None else len(script)
    cfg = {"episodeSteps": steps + 2, "seed": 0,
           "weedSpawnChance": 0.0, "townShopUnlockInterval": 10**6, **config}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)
    sim = kagsim.Sim(dict(cfg))

    for i in range(steps):
        a0 = script[i] if i < len(script) else PASS
        actions = [a0, PASS]
        env.step(actions)
        sim.step(actions)
        ref, got = reference_canonical(env), sim.canonical_state()
        if ref != got:
            raise AssertionError(f"divergence at step {i}\n" + "\n".join(diff(ref, got)))
    return sim.canonical_state()


def run_parity(env_seed: int, fuzz_seed: int, steps: int, config: dict,
               to_termination: bool = False) -> None:
    """Raise AssertionError with a field-level diff at the first divergence.

    `to_termination=True` runs exactly the call that trips `step >= episodeSteps - 2`, so the
    DONE/reward path is exercised. It cannot go further: the framework raises FailedPrecondition
    on a step after done, which makes the interpreter's own `if env.done` early return
    unreachable through the public API.
    """
    cfg = {"episodeSteps": steps + 2, "seed": env_seed, **config}
    env = make("kaggriculture", configuration=cfg)
    env.reset(num_agents=2)

    sim_cfg = dict(cfg)
    sim = kagsim.Sim(sim_cfg)

    ref = reference_canonical(env)
    got = sim.canonical_state()
    if ref != got:
        raise AssertionError(
            f"divergence at INIT (env_seed={env_seed}, fuzz_seed={fuzz_seed})\n"
            + "\n".join(diff(ref, got))
        )

    fz = Fuzzer(fuzz_seed)
    total = steps + 1 if to_termination else steps
    for i in range(total):
        actions = [fz.player_action(env.state[p].observation, p) for p in range(2)]
        env.step(actions)
        sim.step(actions)
        ref = reference_canonical(env)
        got = sim.canonical_state()
        if ref != got:
            raise AssertionError(
                f"divergence at step {i} (env_seed={env_seed}, fuzz_seed={fuzz_seed}, "
                f"config={config})\n" + "\n".join(diff(ref, got))
            )
