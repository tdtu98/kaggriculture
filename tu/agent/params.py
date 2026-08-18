"""Tunable knobs for the scripted engine.

This dataclass *is* the vector that T2.1's CEM/CMA-ES optimizes, so every strategic choice the
engine makes should be a field here rather than a constant in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class Params:
    # --- cash management -----------------------------------------------------
    # Measured: without a floor the engine spends to $0 on day 0 and never recovers, which
    # invalidated the entire goose experiment (docs/experiments.md E3). Seeds are working capital
    # — running out of cash to buy them costs more than anything else on this list.
    cash_floor: float = 150.0         # never spend below this
    land_min_cash: float = 2500.0     # surplus required *on top of* the quadrant price
    # Measured (E12): a high value silently disables the whole livestock line — `goose_target` has
    # no effect if a bird can never be afforded. The champion uses ~337. Kept inside the search
    # bound in search/space.py, which is also why that bound was cut to 800.
    goose_min_cash: float = 300.0     # surplus required on top of the bird's cost
    seed_budget_frac: float = 0.5     # max share of spendable cash committed to seeds per turn

    # --- land and labour -----------------------------------------------------
    # Measured: buying land *loses* money at every tested mix (docs/experiments.md E1, E6). Melon
    # saturates its own market at ~25 tiles, so extra tiles produce goods that cannot be sold
    # profitably while raising travel from 31% to 58% of every turn. Default off; opt in per config.
    buy_land: bool = False
    land_fill_frac: float = 0.8       # buy the next quadrant once this share of owned land is used
    hire_max: int = 8                 # measured optimum ~8; 12+ bankrupts (fib cost explodes)
    tiles_per_unit: float = 6.0       # standing crops per unit; over-planting just breeds weeds

    # --- crop mix ------------------------------------------------------------
    # Target share of plantable tiles per crop; normalized, so relative weights are what matter.
    crop_mix: dict[str, float] = field(
        default_factory=lambda: {"WHEAT": 1.0, "CARROT": 0.0, "MELON": 0.0,
                                 "TOMATO": 0.0, "STRAWBERRY": 0.0}
    )
    melon_stagger: int = 0            # plant melons in N waves rather than all at once

    # --- animals -------------------------------------------------------------
    # Per-species targets. Cows and sheep were unbuildable until V3 — `BUILD_PASTURE` appeared
    # nowhere in the engine — so they were never rejected on evidence, only never implemented.
    goose_target: int = 0
    cow_target: int = 0
    sheep_target: int = 0
    # Spend priority. A goose bought on day 1 returns ~11x over the season and starts paying on
    # day 4; bought on day 10 it returns ~7x. Seeds currently outrank animals in `Engine.market`,
    # so geese wait for the first melon revenue. This flips that ordering so the compounding asset
    # is funded first — searchable, because the tradeoff against seed throughput is not obvious.
    animals_before_seeds: bool = False
    care: bool = True                 # CARE + daily feed, vs alternate-day feeding
    feed_alternate: bool = False      # feed every other day (incompatible with a care bank)
    wheat_reserve_per_animal: int = 2
    # How many animals one shed trip should carry feed for. Was effectively 1: `per_trip` divided
    # the unfed count across *all* units, so eight units each walked to the shed for a single
    # wheat. A unit carrying N wheat can satisfy N FEED tasks in a row, so this converts N round
    # trips into one. Movement was 62-65% of every turn.
    feed_batch: int = 6
    # How much a priority step is worth in tiles walked. Assignment sorted by (priority, distance),
    # so any higher-priority task anywhere on the farm outranked a neighbouring one — units crossed
    # the map instead of clearing a cluster. 0 = pure nearest-task, large = strict priority.
    priority_weight: float = 0.0  # wheat kept in the shed per animal, for feeding

    # --- market --------------------------------------------------------------
    # Sell a unit only while its marginal price >= reserve_frac * base price.
    reserve_frac: dict[str, float] = field(
        default_factory=lambda: {"WHEAT": 0.0, "CARROT": 0.0, "TOMATO": 0.0,
                                 "STRAWBERRY": 0.0, "MELON": 0.55, "EGG": 0.0,
                                 "MILK": 0.5, "WOOL": 0.5, "FERTILIZER": 0.4}
    )
    sell_all_after_day: int = 28      # dump everything near the end; unsold stock scores nothing

    # --- market timing (T1.3) -------------------------------------------------
    # The opponent's tiles are public, so their crop counts and planting days give their harvest
    # dates and incoming supply (PLAN.md 2.5). Blend the static reserve with the price we expect
    # to see once that supply lands: 0 = ignore the opponent, 1 = price purely off the forecast.
    # Measured joint optimum (docs/experiments.md E7): w=1.0, horizon=10. Horizon 10 is melon's
    # plant-to-harvest cycle, so the forecast sees exactly one full wave.
    #
    # These two knobs interact strongly and coordinate-wise tuning misleads: at horizon 4 the best
    # weight was 0.6 and w=1.0 was the *worst* config of all; at horizon 10 w=1.0 is the best.
    # A mis-tuned forecast is also worse than none — several settings lose to forecast_weight=0.
    # T2.1 must search jointly.
    #
    # w=1.0 discards the static reserve entirely, leaving one interpretable rule: sell while the
    # marginal price is at least what the price is expected to be once the opponent's visible
    # supply lands.
    forecast_weight: float = 1.0
    forecast_horizon: int = 10        # days of opponent supply to look ahead

    # --- harvest timing ------------------------------------------------------
    # Maximum tiles planted per day; 0 disables the limit (the original behaviour). Caps the
    # *rate* of new work, which the `tiles_per_unit` stock cap does not (E45).
    plant_rate_per_day: int = 0
    # Refuse plantings that cannot reach first yield before the season ends. Off by default: it
    # changes the champion (it plants late), and every behaviour added this session is gated so the
    # baseline cannot drift underneath a comparison.
    plant_stop_late: bool = False
    season_days: int = 30
    harvest_early: bool = False       # harvest at first_yield_day instead of at max yield
    # Whether a tile with a pending harvest also emits a WATER task. Measured (E23): the right
    # answer depends on the crop mix, so it is searched, not fixed.
    #   "elif"     harvest suppresses water -- the original behaviour. Fine for one-shot crops,
    #              fatal for `ongoing` ones (TOMATO, STRAWBERRY), which sit with yield pending for
    #              days and then die at 2 unwatered days.
    #   "survival" water alongside a harvest only when the plant would otherwise die.
    #   "both"     always water when needed. Costs the melon champion ~$10k (WATER outranks
    #              HARVEST and steals turns) but is worth +23% to a strawberry mix and +39% to
    #              strawberry+land.
    # Default is "elif" so the sitting champion is bit-identical; the re-search picks the rest.
    water_mode: str = "elif"
    # Water ongoing crops (TOMATO, STRAWBERRY) every day instead of only when one miss from death.
    # Off by default so the champion is unchanged; see E43 for why it matters.
    water_ongoing_eager: bool = False
    # How units are matched to tasks. "sequential" serves units in index order, so unit 0 picks
    # first from the whole farm; "global" sorts every (unit, task) pair by cost and assigns
    # greedily, removing that index bias. Travel is the measured root cause of the loss (E28), so
    # this is searched rather than assumed. Default preserves the sitting champion exactly.
    assign_mode: str = "sequential"   # "sequential" | "global" | "optimal"
    # Extra cost, in tiles, for a unit taking work outside its role (livestock vs field). 0 is off.
    # boatlee's units are 93% role-pure; ours switch on 33% of consecutive actions, and pastures
    # and fields are in different places, so each switch is a crossing (E46).
    role_penalty: float = 0.0
    # Discount, in tiles, for work on the tile a unit already occupies. 0 is off. Finishing a tile
    # before moving is what lets boatlee issue 46% of its actions without moving, against our 28%.
    here_bonus: float = 0.0
    # Finish the tile you are standing on before moving. Runs ahead of sticky assignment and of
    # other units' claims, both of which pre-empt a mere cost discount (E47).
    finish_tile: bool = False
    # Subtract pickups already en route when sizing a fetch. Measured worse on its own (E29);
    # searchable because the hauling waste it targets is real (35% of our productive turns).
    fetch_in_flight: bool = False
    # --- shop-adaptive product selection (P1.5-A) -----------------------------
    # 1.32.6 made shop unlocks draw WITH replacement, so which products have a buyer is now a
    # per-game random variable that is fully observable at runtime (E33). Off by default so the
    # champion is bit-identical; the search decides whether it pays.
    adaptive_mix: bool = False
    demand_exponent: float = 1.0      # 0 = ignore demand, 1 = proportional, >1 = concentrate
    animal_demand_floor: int = 1      # herd is cut when its product drains <= this per day
    animal_min_target: int = 3        # ...but never below this: animals make fertilizer regardless
    # Apply collected FERTILIZER to crops instead of selling all of it. Fertilizer doubles yield
    # accrual for 3 days per unit (`kaggriculture.py:386`, `:777-778`); the engine could not emit
    # the verb at all until E24. Off by default so the sitting champion is unchanged -- turning it
    # on also needs `reserve_frac["FERTILIZER"] > 0`, or the market policy sells the stock first.
    fertilize: bool = False
    fertilize_batch: int = 4          # units carried per shed trip

    def to_vector(self) -> list[float]:
        """Flatten for black-box search (T2.1)."""
        out: list[float] = []
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                out.extend(float(v[k]) for k in sorted(v))
            else:
                out.append(float(v))
        return out
