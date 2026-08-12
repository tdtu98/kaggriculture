"""The search space: a bounded vector <-> `Params`.

Every strategic knob the engine has, with explicit bounds and types. CEM works in the unbounded
vector; `decode` clips, rounds, and thresholds it back into a valid `Params`.

Searched **jointly** on purpose. Coordinate-wise tuning landed on the wrong point twice during
T1.3: at `forecast_horizon=4` the best `forecast_weight` was 0.6 and 1.0 was the worst setting of
all, while at horizon 10 the ordering reverses completely (`docs/experiments.md` E7).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from agent.params import Params

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

Kind = Literal["float", "int", "bool", "choice"]


@dataclass(frozen=True)
class Knob:
    path: str            # "hire_max" or "crop_mix.MELON"
    lo: float
    hi: float
    kind: Kind = "float"
    choices: tuple[str, ...] = ()   # for kind="choice": the value at each integer index

    def clip_numeric(self, x: float) -> float:
        """Clamp within bounds but stay in vector space.

        `clip` returns the *typed* value a `Params` field wants, which for a categorical is a
        string -- so it cannot be used to normalise the search vector itself.
        """
        x = min(max(x, self.lo), self.hi)
        if self.kind in ("int", "choice"):
            return float(round(x))
        if self.kind == "bool":
            return 1.0 if x >= 0.5 else 0.0
        return float(x)

    def to_vector(self, value) -> float:
        """Typed `Params` value -> its position in the search vector."""
        if self.kind == "choice":
            return float(self.choices.index(value) if value in self.choices else 0)
        return float(value)

    def clip(self, x: float) -> float | int | bool | str:
        x = min(max(x, self.lo), self.hi)
        if self.kind == "int":
            return int(round(x))
        if self.kind == "bool":
            return x >= 0.5
        if self.kind == "choice":
            return self.choices[min(int(round(x)), len(self.choices) - 1)]
        return float(x)


KNOBS: list[Knob] = [
    # crop mix — relative weights, normalized by the engine, so the scale is arbitrary
    *[Knob(f"crop_mix.{c}", 0.0, 1.0) for c in CROPS],
    # land and labour
    Knob("buy_land", 0.0, 1.0, "bool"),
    Knob("land_fill_frac", 0.3, 1.0),
    Knob("land_min_cash", 0.0, 8000.0),
    Knob("hire_max", 0.0, 14.0, "int"),
    Knob("tiles_per_unit", 2.0, 14.0),
    # animals
    # V3: cows and sheep were unbuildable before pastures existed, so the search has never had
    # them. Both beat the goose-only champion 48/48 on first contact (E15).
    Knob("goose_target", 0.0, 24.0, "int"),
    Knob("cow_target", 0.0, 24.0, "int"),
    Knob("sheep_target", 0.0, 24.0, "int"),
    Knob("care", 0.0, 1.0, "bool"),
    Knob("feed_alternate", 0.0, 1.0, "bool"),
    Knob("wheat_reserve_per_animal", 1.0, 6.0, "int"),
    Knob("feed_batch", 1.0, 16.0, "int"),
    Knob("priority_weight", 0.0, 10.0),
    # Behaviours that did not exist when the champion was fitted, each measured *worse* on its own
    # against a champion tuned around its absence (E23, E24, E29). They are in the space because
    # the point of this search is to fit the parameters and the behaviours together rather than
    # judging each against a vector co-adapted to the old engine.
    Knob("water_mode", 0.0, 2.0, "choice", ("elif", "survival", "both")),
    # "optimal" was added to the engine (E39) and not to this tuple, so the search could not reach
    # it at all -- a knob the engine has and the search cannot express is worse than no knob.
    Knob("assign_mode", 0.0, 2.0, "choice", ("sequential", "global", "optimal")),
    Knob("fertilize", 0.0, 1.0, "bool"),
    Knob("fetch_in_flight", 0.0, 1.0, "bool"),
    Knob("fertilize_batch", 1.0, 10.0, "int"),
    # E43-E45. `water_ongoing_eager` and `fertilize` are worth ~nothing apart and +24% together,
    # because the environment grants the fertiliser bonus only on a day the tile was also watered.
    # `plant_rate_per_day` caps the *rate* of new work (the existing `tiles_per_unit` caps only the
    # stock), which is what stops a burst of plantings dying together at age 2.
    Knob("water_ongoing_eager", 0.0, 1.0, "bool"),
    Knob("plant_rate_per_day", 0.0, 16.0, "int"),
    Knob("plant_stop_late", 0.0, 1.0, "bool"),
    Knob("adaptive_mix", 0.0, 1.0, "bool"),
    # Upper bound cut 3000 -> 800 (E12). `goose_target` only has an effect when `goose_min_cash`
    # is small enough for a goose to ever be bought, so a high value silently *disables* the whole
    # livestock line. CEM samples each dimension independently around the mean, which makes a
    # conjunction of two specific values exponentially unlikely to be tried — so the gradient on
    # `goose_target` looked flat and the search drove it to 0, discarding a ~$19k revenue stream.
    Knob("goose_min_cash", 0.0, 800.0),
    # cash management
    Knob("cash_floor", 0.0, 1500.0),
    Knob("seed_budget_frac", 0.05, 1.0),
    # market timing
    Knob("forecast_weight", 0.0, 1.0),
    Knob("forecast_horizon", 1.0, 24.0, "int"),
    Knob("sell_all_after_day", 18.0, 30.0, "int"),
    *[Knob(f"reserve_frac.{p}", 0.0, 1.2) for p in PRODUCTS],
]

DIM = len(KNOBS)


def encode(p: Params) -> list[float]:
    out: list[float] = []
    for k in KNOBS:
        if "." in k.path:
            group, key = k.path.split(".")
            v = getattr(p, group)[key]
        else:
            v = getattr(p, k.path)
        # Categoricals live in the vector as their index, so CEM's Gaussian sampling and
        # elite-mean update work unchanged; `clip` maps back to the string on decode.
        out.append(k.to_vector(v))
    return out


def decode(vec: list[float], base: Params | None = None) -> Params:
    """Vector -> Params, clipped to bounds. `base` supplies anything not searched."""
    base = base or Params()
    flat: dict = {}
    groups: dict[str, dict] = {"crop_mix": {}, "reserve_frac": {}}
    for k, x in zip(KNOBS, vec):
        val = k.clip(x)
        if "." in k.path:
            group, key = k.path.split(".")
            groups[group][key] = float(val)
        else:
            flat[k.path] = val

    # A degenerate all-zero mix would leave the engine with nothing to plant.
    if sum(groups["crop_mix"].values()) <= 1e-6:
        groups["crop_mix"] = {c: (1.0 if c == "MELON" else 0.0) for c in CROPS}

    return replace(base, **flat, **groups)


def clip_vector(vec: list[float]) -> list[float]:
    return [k.clip_numeric(x) for k, x in zip(KNOBS, vec)]


def parse_pins(spec: str) -> dict[str, float]:
    """`"forecast_weight=0,reserve_frac.*=0"` -> {path: value}, `*` matching a whole group.

    Pinning removes a knob from the search rather than merely biasing it, which is what makes an
    ablation an ablation: T2.5 needs a policy that provably cannot hold inventory, not one that
    has merely learned not to.
    """
    pins: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        path, _, raw = part.partition("=")
        value = float(raw)
        path = path.strip()
        if path.endswith(".*"):
            group = path[:-2]
            for k in KNOBS:
                if k.path.startswith(group + "."):
                    pins[k.path] = value
        else:
            if not any(k.path == path for k in KNOBS):
                raise SystemExit(f"unknown knob {path!r}; have {[k.path for k in KNOBS]}")
            pins[path] = value
    return pins


def apply_pins(vec: list[float], pins: dict[str, float]) -> list[float]:
    out = list(vec)
    for i, k in enumerate(KNOBS):
        if k.path in pins:
            out[i] = pins[k.path]
    return out


def bounds() -> tuple[list[float], list[float]]:
    return [k.lo for k in KNOBS], [k.hi for k in KNOBS]


def describe(vec: list[float]) -> str:
    """Only the knobs that differ from the defaults, so a result is readable."""
    d = decode(vec)
    base = Params()
    parts = []
    for k in KNOBS:
        if "." in k.path:
            g, key = k.path.split(".")
            a, b = getattr(d, g)[key], getattr(base, g)[key]
        else:
            a, b = getattr(d, k.path), getattr(base, k.path)
        if isinstance(a, float) and isinstance(b, float):
            same = abs(a - b) < 1e-6
        else:
            same = a == b
        if not same:
            parts.append(f"{k.path}={a:.3g}" if isinstance(a, float) else f"{k.path}={a}")
    return ", ".join(parts) or "(defaults)"
