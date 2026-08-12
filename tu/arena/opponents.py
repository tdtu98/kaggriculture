"""Independently designed opponents (V2).

Every exploiter so far is *the champion with two knobs changed*, so the arena has only ever sampled
one strategy family. `docs/decisions.md` D16: "no known exploit" currently means "none I thought to
build", and E10 showed a two-line change beating a fully tuned champion 0/80.

These are written **from a design brief**, not by mutating champion parameters. Each overrides the
*strategic* methods — what to plant, what to buy, when to harvest — and inherits only the mechanical
executor (task assignment, pathing, op emission), which is not strategy.

Each one is a hypothesis about how the champion could be beaten:

* `GooseBaron`   — the champion realises 8% of the egg market's capacity (E13). Take it.
* `ShopChaser`   — demand is dynamic and observable; the champion ignores `unlocked_shops` entirely.
* `LandBaron`    — the champion refuses land on evidence gathered *before* routing improved (E6).
* `Sprinter`     — E11 says arriving first wins; harvest at the first legal day and race.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, SHOPS

from agent.engine import BASE_PRICE, SHED_TILES, Engine, Task
from agent.params import Params

CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]


class GooseBaron(Engine):
    """Brief: *the egg market absorbs ~$114k and nobody is touching it. Farm birds, not crops.*

    Crops exist only to feed the flock. Coops are built on every tile the labour force can service,
    and wheat is sized to feed rather than to sell.
    """

    def __init__(self, params: Params | None = None):
        super().__init__(params or Params(
            crop_mix={"WHEAT": 1.0, "CARROT": 0.0, "TOMATO": 0.0, "STRAWBERRY": 0.0, "MELON": 0.0},
            goose_target=20, goose_min_cash=0.0, care=True, hire_max=9,
            wheat_reserve_per_animal=3, cash_floor=50.0, seed_budget_frac=0.25,
            forecast_weight=0.0, sell_all_after_day=29,
        ))

    def build_tasks(self, farm, priv, day):
        tasks = super().build_tasks(farm, priv, day)
        # Coops outrank planting: a bird earns every day it exists, a crop only at harvest.
        for t in tasks:
            if t.op == "BUILD_COOP":
                t.prio = 1
            elif t.op == "PLANT":
                t.prio = 7
        return tasks

    def market(self, obs, farm, priv, day, hour):
        orders = super().market(obs, farm, priv, day, hour)
        # Birds first, always — the opposite of the champion's seeds-first ordering.
        birds = [o for o in orders if o[0] == "BUY_ANIMAL"]
        rest = [o for o in orders if o[0] != "BUY_ANIMAL"]
        sells = [o for o in rest if o[0] == "SELL"]
        other = [o for o in rest if o[0] != "SELL"]
        return (sells + birds + other)[:10]


class ShopChaser(Engine):
    """Brief: *demand is dynamic and public; plant whatever the town is currently hungriest for.*

    The champion's crop mix is fixed at t=0 and never reads `town.unlocked_shops`. This one
    recomputes the mix each day from actual shop demand times current price.
    """

    def __init__(self, params: Params | None = None):
        super().__init__(params or Params(
            hire_max=8, goose_target=4, goose_min_cash=200.0,
            forecast_weight=0.0, cash_floor=200.0,
        ))
        self._mix = dict(self.p.crop_mix)

    def _demand_weighted_mix(self, obs) -> dict[str, float]:
        """Units/day the town removes, weighted by what a unit currently fetches."""
        drain = self.town_drain_per_day(obs)
        prices = obs["market"]["prices"]
        raw = {c: drain.get(c, 0) * prices.get(c, BASE_PRICE[c]) for c in CROP_NAMES}
        total = sum(raw.values())
        if total <= 0:
            return dict(self.p.crop_mix)
        # Normalise, then damp toward uniform so one shop unlock cannot swing the whole farm.
        return {c: 0.25 + 0.75 * (v / total) for c, v in raw.items()}

    def __call__(self, obs):
        self.p.crop_mix = self._demand_weighted_mix(obs)
        return super().__call__(obs)


class LandBaron(Engine):
    """Brief: *75 tiles sit locked all game. Buy them and hire enough hands to work them.*

    E6 measured land as a loss — but with routing and cash management as they were then. This
    ignores the labour-headroom gate entirely and buys as soon as it can afford it.
    """

    def __init__(self, params: Params | None = None):
        super().__init__(params or Params(
            hire_max=11, tiles_per_unit=9.0, buy_land=True, land_fill_frac=0.55,
            land_min_cash=200.0, goose_target=6, goose_min_cash=300.0,
            forecast_weight=0.0, cash_floor=150.0, seed_budget_frac=0.6,
        ))

    def market(self, obs, farm, priv, day, hour):
        orders = super().market(obs, farm, priv, day, hour)
        # Buy land the moment it is affordable, regardless of how full the current land is.
        if len(farm["unlocked_quadrants"]) < 4 and not any(o[0] == "BUY_LAND" for o in orders):
            cost = [1000, 2000, 4000][len(farm["unlocked_quadrants"]) - 1]
            if farm["money"] >= cost + self.p.land_min_cash:
                orders = [o for o in orders if o[0] == "SELL"][:2] + [["BUY_LAND"]] + \
                         [o for o in orders if o[0] != "SELL"]
        return orders[:10]


class Sprinter(Engine):
    """Brief: *E11 says arriving first wins. So arrive first — harvest the moment it is legal.*

    Takes each crop at `first_yield_day` rather than at maximum yield, trading units per tile for
    cycles per season and for reaching the price curve before the opponent.
    """

    def __init__(self, params: Params | None = None):
        super().__init__(params or Params(
            hire_max=8, harvest_early=True, forecast_weight=0.0,
            goose_target=4, goose_min_cash=250.0, cash_floor=200.0,
            crop_mix={"WHEAT": 0.3, "CARROT": 0.8, "TOMATO": 0.0,
                      "STRAWBERRY": 0.0, "MELON": 0.9},
        ))

    def build_tasks(self, farm, priv, day):
        tasks = super().build_tasks(farm, priv, day)
        # Harvest ahead of everything: a unit sitting on ripe produce is a unit not in the race.
        for t in tasks:
            if t.op == "HARVEST":
                t.prio = 0
        return tasks


OPPONENTS = {
    "o-goose-baron": GooseBaron,
    "o-shop-chaser": ShopChaser,
    "o-land-baron": LandBaron,
    "o-sprinter": Sprinter,
}
