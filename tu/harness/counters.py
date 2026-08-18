"""Per-step observer: the eleven counters TASKS_v4 I2 requires, for any agent, for free.

Why an observer rather than instrumentation inside the agents: PLAN_v4's whole method is "prove the
change fired before you read its score" (CLAUDE.md; E44 cost four measurements to this). A counter
that only exists inside our own agent cannot compare us to Boatlee, and a counter each agent
reports for itself is a counter each agent can be wrong about. This watches the *simulator*.

Two rules shape the implementation:

* **Never re-implement a rule to measure it.** E39 lost a whole comparison to an analysis script
  that re-derived the engine's own logic and got it subtly wrong. So `blocked_ops` is not "ops I
  think should have failed" — it is kagsim's own before/after state fingerprint (`rules.rs:340`,
  `actions_noop`), minus the ops that were explicitly PASS.
* **An order is not a sale; count units that arrive.** Production is measured as
  `held(shed + inventories) + sold`, differenced from the actual game state, never from what an
  agent asked for (CLAUDE.md; E48, E50).

Tile-derived counters are read from one snapshot per day rather than per step: deaths, plantings
and animal census all resolve at the daily refresh, so 30 snapshots a season carry the same
information as 719 at a twenty-fourth of the cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}

#: Products that cannot be bought back, so `held + sold` is exactly what the farm produced
#: (`kaggriculture.py` BUY_PRODUCT accepts WHEAT and FERTILIZER only).
#:
#: WHEAT and FERTILIZER are therefore *not* reported as production: both are bought constantly for
#: feeding and fertilizing, and kagsim does not tally purchased units, so held+sold would count a
#: shopping trip as a harvest. Known gap — C1's wheat cycles will want it, and closing it means a
#: `bought_units` stat in `kagsim/src/rules.rs`, not a guess in this file.
UNBUYABLE = [p for p in PRODUCTS if p not in ("WHEAT", "FERTILIZER")]

#: Strawberry ticks at the end of ages 9/11/13/15 and fertilizer covers day..day+2, so these are
#: the only two applications that double all four productions (verified in tests/test_env_facts.py).
STRAWBERRY_FERT_AGES = (9, 13)


def _tiles(obs: dict, player: int):
    return obs["farms"][player]["tiles"]


def _unit_positions(farm: dict) -> list[tuple[int, int]]:
    return [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]


@dataclass
class Observer:
    """Watches one seat for a whole episode. Feed it every (obs, action) pair, then `finish()`."""

    player: int
    pass_ops: int = 0
    fertilize_ops: int = 0
    fertilize_hits: int = 0
    plants_lost: int = 0
    plants_lost_thirst: int = 0
    plants_lost_age: int = 0
    plants_started: dict = field(default_factory=dict)
    animal_days: dict = field(default_factory=dict)
    _day: int = -1
    _day_start: dict | None = None     # plant identities at this day's first observation
    _last_tiles: list | None = None    # tiles as of the most recent observation
    _last_obs: dict | None = None

    # ---------------------------------------------------------------- per step

    def observe(self, obs: dict, action: dict) -> None:
        """Called with the observation the agent saw and the action it returned, before stepping.

        Positions are read from this observation, which is correct: a unit performs exactly one op
        per step, so where it stands when it decides is where the op lands.
        """
        self._last_obs = obs
        day = int(obs.get("day", 0) or 0)
        tiles = _tiles(obs, self.player)
        if self._last_tiles is not None:
            self._scan_deaths(self._last_tiles, tiles, int(obs.get("step", 0) or 0))
        if day != self._day:
            self._roll_day(tiles)
            self._day = day
        # kept every step, so `_roll_day` can see the board as it stood at yesterday's last hour
        self._last_tiles = tiles

        farm = obs["farms"][self.player]
        positions = _unit_positions(farm)
        units = [action.get("farmer")] + list(action.get("hands") or [])
        # Ops beyond the units that exist are discarded by the interpreter; do not count them.
        for pos, op in zip(positions, units[: len(positions)]):
            if not op:
                continue
            name = op[0]
            if name == "PASS":
                self.pass_ops += 1
            elif name == "FERTILIZE":
                self.fertilize_ops += 1
                if self._is_strawberry_fert_window(farm, pos, day):
                    self.fertilize_hits += 1

    def _is_strawberry_fert_window(self, farm: dict, pos, day: int) -> bool:
        x, y = pos
        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            return False
        if tile.get("crop") != "STRAWBERRY":
            return False
        return (day - int(tile.get("planted_day", 0))) in STRAWBERRY_FERT_AGES

    # ---------------------------------------------------------------- per day

    def _scan_deaths(self, before, after, step: int) -> None:
        """Count PLANT -> WEED transitions, per step, and say which way the plant died.

        Per step and not per day, because there are two death mechanisms on different clocks:
        thirst kills at the daily refresh, while a plant past `max_lifespan_step` decays and can
        turn at any step. Boatlee digs and replants the same day, so day-boundary sampling saw
        almost none of its losses — it read 0.1/season against 10 measured directly, which would
        have made C5's "plants lost <= 10" gate trivially passable against the wrong quantity.

        The split matters for Track F: old age is a plant that paid out and expired, thirst is a
        plant the routing failed. `max_lifespan_step` is the environment's own field, so this
        classifies from state rather than re-deriving the rule (E39).
        """
        for y, row_after in enumerate(after):
            row_before = before[y]
            for x, tile_after in enumerate(row_after):
                if not (isinstance(tile_after, dict) and tile_after.get("kind") == "WEED"):
                    continue
                tile_before = row_before[x]
                if not (isinstance(tile_before, dict) and tile_before.get("kind") == "PLANT"):
                    continue
                self.plants_lost += 1
                if step >= int(tile_before.get("max_lifespan_step", 1 << 30)):
                    self.plants_lost_age += 1
                else:
                    self.plants_lost_thirst += 1

    def _roll_day(self, new_tiles) -> None:
        """Close the previous day and open a new one.

        Three boards are involved and each answers a different question:

        * yesterday's **first** hour vs yesterday's **last** hour -> what was *planted* yesterday.
        * yesterday's **last** hour vs today's first hour -> what *died* at yesterday's dusk.

        Comparing only day-starts, which is the obvious implementation, silently loses the most
        important case: a plant is created with `consecutive_unwatered = 1`, so one that is not
        watered on its planting day is already at 2 by that evening and becomes a WEED before the
        next day begins. It would be born and lost between two snapshots and counted as neither —
        and "starts plants it cannot finish" is exactly the failure PLAN_v4 §1 attributes to our
        engine, so a counter blind to it would be blind to the thing it exists to measure.
        """
        day_end = _plant_map(self._last_tiles) if self._last_tiles is not None else None
        if day_end is not None and self._day_start is not None:
            for key, crop in day_end.items():
                if key not in self._day_start:
                    self.plants_started[crop] = self.plants_started.get(crop, 0) + 1
        self._day_start = _plant_map(new_tiles)
        for row in new_tiles:
            for tile in row:
                animal = isinstance(tile, dict) and tile.get("animal")
                if animal:
                    self.animal_days[animal] = self.animal_days.get(animal, 0) + 1

    # ---------------------------------------------------------------- summary

    def finish(self, sim, agent=None) -> dict:
        """Combine the observer's tallies with kagsim's own stats into the reported counters."""
        obs = sim.observation(self.player)
        # Fold in the final day: the last step's deaths, then the plants it started.
        final = _tiles(obs, self.player)
        if self._last_tiles is not None:
            self._scan_deaths(self._last_tiles, final, int(obs.get("step", 1 << 30) or 0))
        self._last_tiles = final
        self._roll_day(final)
        stats = sim.stats(self.player)
        produced = self._produced(obs, stats)

        total = int(stats["actions_total"])
        moves = int(stats["actions_move"])
        effective = int(stats["actions_effective"])
        noop = int(stats["actions_noop"])

        cow_days = self.animal_days.get("COW", 0)
        straw_plants = self.plants_started.get("STRAWBERRY", 0)

        out = {
            # movement economy — Boatlee's 1.01 is the target, ours has run 1.3-1.84
            "steps_per_useful": _ratio(moves, effective),
            "idle_pct": _ratio(self.pass_ops, total),
            # kagsim decides "did nothing" by state fingerprint, so PASS lands in actions_noop too
            "blocked_ops": max(0, noop - self.pass_ops),
            "plants_lost": self.plants_lost,
            "plants_lost_thirst": self.plants_lost_thirst,
            "plants_lost_age": self.plants_lost_age,
            "plants_started": sum(self.plants_started.values()),
            "strawberry_per_plant": _ratio(produced.get("STRAWBERRY", 0), straw_plants),
            "milk_per_cow_day": _ratio(produced.get("MILK", 0), cow_days),
            "unharvested_ripe_at_end": _unharvested(obs, self.player),
            "shed_overflow_discarded": int(stats["discarded_overflow"]),
            "fertilize_hits": self.fertilize_hits,
            "fertilize_ops": self.fertilize_ops,
            "produced": produced,
            "sold_units": dict(stats["sold_units"]),
            "sold_revenue": dict(stats["sold_revenue"]),
        }
        out["effect_count"] = _effect_counts(agent)
        return out

    def _produced(self, obs: dict, stats: dict) -> dict:
        """Units that actually appeared: what we still hold plus what we sold.

        Only for products that cannot be bought back. Overflow discarded at dusk is *not* counted
        (kagsim tallies it as one scalar, not per product), so on a farm that overflows this is a
        lower bound — `shed_overflow_discarded` is reported next to it so that is visible rather
        than silent.
        """
        priv = obs.get("private", {}) or {}
        held: dict[str, int] = {}
        for item, n in (priv.get("shed") or {}).items():
            held[item] = held.get(item, 0) + int(n)
        for inv in priv.get("inventories") or []:
            for item, n in dict(inv).items():
                held[item] = held.get(item, 0) + int(n)
        sold = dict(stats["sold_units"])
        return {p: held.get(p, 0) + int(sold.get(p, 0))
                for p in UNBUYABLE if held.get(p, 0) or sold.get(p, 0)}


def _effect_counts(agent) -> dict:
    """Overlay effect counters, for agents that carry a `ctx` (see `agent/relay.py`).

    A zero here means "the overlay never fired", which is an unfinished implementation and not a
    refutation — the distinction E44 was lost to.
    """
    ctx = getattr(agent, "ctx", None)
    if ctx is None:
        return {}
    out = {}
    effects = getattr(ctx, "effects", None)
    if isinstance(effects, dict):
        out.update({k: int(v) for k, v in effects.items()})
    blocked = getattr(ctx, "blocked_ops", None)
    if isinstance(blocked, dict) and blocked:
        out["agent_blocked_ops"] = sum(int(v) for v in blocked.values())
    return out


def _plant_map(tiles) -> dict:
    """`{(x, y, planted_day): crop}` — identity includes the planting day so that a replant on a
    tile is a new plant rather than a survivor."""
    out = {}
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                out[(x, y, int(tile.get("planted_day", -1)))] = tile.get("crop")
    return out


def _kind(tile) -> str:
    if isinstance(tile, dict):
        return str(tile.get("kind", ""))
    return "LOCKED" if tile == "LOCKED" else "EMPTY"


def _unharvested(obs: dict, player: int) -> int:
    return sum(int(t.get("yield_units", 0) or 0)
               for row in _tiles(obs, player) for t in row
               if isinstance(t, dict))


def _ratio(num, den) -> float:
    return float(num) / float(den) if den else 0.0


def mean_counters(rows: list[dict]) -> dict:
    """Average the scalar counters across games; sum the per-product dicts."""
    if not rows:
        return {}
    out: dict = {}
    for key in rows[0]:
        values = [r.get(key) for r in rows if key in r]
        if not values:
            continue
        if isinstance(values[0], dict):
            merged: dict = {}
            for v in values:
                for k, n in v.items():
                    merged[k] = merged.get(k, 0) + n
            out[key] = {k: merged[k] / len(values) for k in sorted(merged)}
        else:
            out[key] = sum(values) / len(values)
    return out
