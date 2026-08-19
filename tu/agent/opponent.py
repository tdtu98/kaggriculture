"""O2 — who is on the other side of the board, and what they are about to sell.

Both farms are public. `kaggriculture.json` marks `farms` as a **shared** observation field, and the
interpreter writes the same list into every seat's observation (`kaggriculture.py:261,271,952`), so
the opponent's tiles — crop, `planted_day`, `yield_units`, animals, `placed_day` — their farmer and
hand positions, their `unlocked_quadrants`, their `hires_today` and their money are all readable
every turn. What is *not* readable is their shed and their per-farmer inventories: `private` is
`"shared": false` and each seat gets its own (`kaggriculture.py:269`). Their market orders are not
in the observation at all.

That asymmetry is the whole reason this module exists. Their **production** is a schedule we can
read off the board; their **selling** is hidden. Against a fixed script the second half can be
recovered anyway — not by inference, but by having played it: `boatlee` is a 719-step choreography,
so its SELL steps and quantities were measured once and are quoted here as a table.

Three things live here:

* `census(farm)` / `fingerprint(...)` — a fifteen-number tile census at steps 24, 48 and 72,
  compared against a library of measured profiles. Cheap enough to run every turn; it runs three
  times a game.
* `forecast_supply(farm, step)` — units by the step they reach the market, from their standing
  crops and animals. This is C6's forecaster (`projection.farm_supply`) pointed at their farm and
  re-keyed from days-ahead to absolute steps, not a second implementation of it.
* `sell_schedule(product)` / `sell_units(...)` — for a fingerprinted `boatlee`, the measured steps
  at which it dumps each product. O3 front-runs off this; nothing here acts on it.

**How the tables were measured.** `boatlee` was replayed through the reference env (not kagsim) on
seeds 66100-66109, against `pass` and against our own `compiler`, with the env's own `_commit_unit`
(`kaggriculture.py:664`) patched to count *units that actually moved* — an order is not a sale
(CLAUDE.md; E48/E50), and boatlee orders 241 MILK a season while 209 settle. Its order steps and
quantities came out **identical on all ten seeds and identical against both opponents**, except six
WOOL steps that appear in 3/10 seeds (`CONDITIONAL_SELLS`) — a choreography, as PLAN3 says.

The census profiles came from the same kind of replay (kagsim, seeds 66000-66009, boatlee against
each pool member in both seats): 70 boatlee observations at each checkpoint, **zero** spread on the
features used here, against a nearest-neighbour distance of 6. Weeds and empty-tile counts are
deliberately *not* features: they are the one part of the early board that moves with the RNG
(`kaggriculture.py` weed spawn), and they were the only thing that varied across those 70.
"""

from __future__ import annotations

from agent import projection

TURNS_PER_DAY = 24

#: The three census points O2 is specified at — hour 0 of days 1, 2 and 3.
CHECKPOINTS = (24, 48, 72)

#: How many checkpoints a profile must survive before it can be locked. Two, not three: boatlee is
#: already the unique survivor at step 24 (nearest rival is 9 away), so waiting for 72 would give
#: away a day of front-running for nothing. A lock is still *re-checked* at every later checkpoint
#: and dropped if it stops matching, so an early lock costs nothing but an early answer.
MIN_CHECKS = 2

#: Max L1 distance on the census vector for a profile to stay alive at a checkpoint.
#:
#: Measured, not guessed. Over 70 boatlee observations per checkpoint the census vector never moved
#: at all (spread 0); our own plans move by up to 2 when a weed takes a tile out on day 1. The
#: nearest distance *between* two different profiles at any checkpoint is 6 (boatlee to compiler at
#: steps 48 and 72), so 2 sits with a margin of 2 above the noise and 4 below the signal.
THRESHOLD = 2

CROP_FEATURES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_FEATURES = ("COW", "SHEEP", "GOOSE")
#: Order is fixed so a census is comparable as a vector; the names are kept for readable diffs.
FEATURES = CROP_FEATURES + ANIMAL_FEATURES + ("PASTURE", "QUADS", "HANDS")

#: name -> {checkpoint step: census}. Measured; see the module docstring for how.
#:
#: `boatlee` is the opponent O3 exists to beat. `compiler`, `s2_r1` and `s2_r2` are *us* — the
#: incumbent and the two S2 search candidates (E76/E77) — and they are in the library for the
#: reason the task names: our plan was built from boatlee's measured season, so "is that boatlee or
#: is it a mirror of me?" is the question the library has to answer, not an afterthought. It
#: separates cleanly at every checkpoint: we hold no animals at step 24 where boatlee already has
#: four sheep and a cow, and our herd arrives days later (E70's `cow_start`).
PROFILES: dict[str, dict[int, dict[str, int]]] = {
    "boatlee": {
        24: {"WHEAT": 5, "MELON": 5, "COW": 1, "SHEEP": 4, "PASTURE": 5, "QUADS": 1},
        48: {"WHEAT": 5, "MELON": 5, "COW": 1, "SHEEP": 4, "PASTURE": 5, "QUADS": 1},
        72: {"WHEAT": 5, "MELON": 5, "COW": 1, "SHEEP": 4, "PASTURE": 5, "QUADS": 1},
    },
    "compiler": {
        24: {"WHEAT": 6, "MELON": 4, "PASTURE": 3, "QUADS": 1},
        48: {"WHEAT": 6, "MELON": 4, "SHEEP": 1, "PASTURE": 5, "QUADS": 1},
        72: {"WHEAT": 7, "MELON": 5, "COW": 1, "SHEEP": 3, "PASTURE": 8, "QUADS": 1},
    },
    "s2_r1": {
        24: {"WHEAT": 7, "MELON": 5, "QUADS": 1},
        48: {"WHEAT": 7, "MELON": 5, "PASTURE": 4, "QUADS": 1},
        72: {"WHEAT": 7, "MELON": 5, "COW": 4, "PASTURE": 8, "QUADS": 1},
    },
    "s2_r2": {
        24: {"WHEAT": 7, "MELON": 5, "QUADS": 1},
        48: {"WHEAT": 7, "MELON": 5, "PASTURE": 4, "QUADS": 1},
        72: {"WHEAT": 7, "MELON": 4, "COW": 4, "PASTURE": 8, "QUADS": 1},
    },
    "starter": {
        24: {"CARROT": 1, "QUADS": 1},
        48: {"CARROT": 1, "QUADS": 1},
        72: {"CARROT": 1, "QUADS": 1},
    },
    "executor_v7": {
        24: {"WHEAT": 1, "STRAWBERRY": 10, "COW": 1, "PASTURE": 1, "QUADS": 1},
        48: {"WHEAT": 3, "STRAWBERRY": 11, "MELON": 2, "COW": 1, "PASTURE": 2, "QUADS": 1},
        72: {"WHEAT": 3, "STRAWBERRY": 11, "MELON": 2, "COW": 1, "PASTURE": 2, "QUADS": 1},
    },
}

#: Which profiles are our own agent. O3 has no business front-running itself, and a mirror match is
#: a different problem from the boatlee matchup.
SELF_PROFILES = frozenset({"compiler", "s2_r1", "s2_r2"})

#: `boatlee`'s SELL orders: product -> [(step, units ordered)]. Present on all ten measured seeds
#: and identical against both opponents. **Units ordered, not units sold** — see `SETTLE_RATE`.
SELL_SCHEDULE: dict[str, list[tuple[int, int]]] = {
    "WHEAT": [
        (120, 17), (192, 2), (213, 6), (216, 7), (240, 2), (285, 2), (288, 10), (309, 1),
        (312, 24), (336, 3), (384, 10), (405, 6), (408, 19), (432, 3), (453, 1), (455, 1),
        (477, 8), (480, 3), (501, 12), (504, 7), (569, 3), (576, 20), (597, 4), (600, 31),
        (624, 30), (648, 2), (657, 3), (668, 6), (672, 45), (677, 2), (696, 53), (713, 16),
        (714, 4), (715, 26), (716, 23), (717, 36), (718, 7),
    ],
    "STRAWBERRY": [
        (336, 6), (384, 14), (408, 2), (432, 28), (456, 12), (479, 14), (480, 20), (504, 18),
        (522, 10), (528, 34), (552, 28), (575, 10), (576, 22), (597, 10), (600, 10), (624, 16),
        (648, 22), (670, 4), (672, 6),
    ],
    "MELON": [
        (252, 6), (255, 6), (257, 6), (260, 6), (262, 6), (486, 6), (488, 6), (490, 12),
        (492, 6), (493, 6), (495, 6), (496, 6), (502, 6), (503, 6), (504, 12), (528, 12),
    ],
    "MILK": [
        (216, 6), (264, 3), (311, 3), (336, 6), (360, 9), (377, 12), (381, 3), (406, 21),
        (408, 3), (432, 9), (455, 9), (456, 3), (480, 14), (502, 9), (504, 3), (518, 6),
        (524, 3), (527, 3), (552, 9), (571, 6), (597, 11), (600, 3), (620, 11), (624, 5),
        (648, 9), (666, 9), (672, 3), (685, 3), (690, 8), (691, 3), (715, 18), (716, 18),
    ],
    "WOOL": [
        (160, 5), (168, 15), (240, 16), (311, 4), (312, 12), (381, 8), (384, 8), (456, 16),
        (519, 8), (524, 8), (600, 16), (672, 16),
    ],
    "FERTILIZER": [
        (48, 5), (72, 5), (96, 5), (120, 5), (144, 5), (160, 3), (168, 3), (192, 7), (216, 10),
        (281, 12), (305, 9), (329, 12), (353, 8), (377, 7), (381, 2), (425, 9), (449, 8),
        (521, 4), (524, 1), (527, 2), (545, 11), (569, 8), (593, 9), (597, 1), (598, 1),
        (617, 11), (620, 2), (634, 1), (635, 1), (638, 1), (639, 2), (641, 9), (645, 2),
        (646, 1), (648, 5), (665, 8), (666, 2), (672, 3), (673, 5), (685, 3), (690, 3),
        (691, 4), (696, 3), (713, 3), (714, 3), (715, 5), (716, 5), (717, 1),
    ],
}

#: The six steps that are *not* on every seed: `(step, units, seeds_of_ten)`. All WOOL, all in the
#: back half — boatlee's relay overlay, which is the one part of its market play that branches.
#: Kept out of `SELL_SCHEDULE` so a front-run built from the table is never built on a coin flip.
CONDITIONAL_SELLS: dict[str, list[tuple[int, int, int]]] = {
    "WOOL": [(407, 12, 3), (457, 8, 3), (503, 8, 3), (598, 7, 3), (649, 4, 3), (691, 3, 3)],
}

#: Ordered units that actually settle, per season, mean over the ten measured seeds. MILK is the
#: one that matters: boatlee asks for 241 and moves 209, so sizing a counter-move off the order
#: book would over-state its milk supply by 15%.
SETTLE_RATE: dict[str, float] = {
    "WHEAT": 439.0 / 455.0, "STRAWBERRY": 285.2 / 286.0, "MELON": 1.0,
    "MILK": 209.2 / 241.0, "WOOL": 1.0, "FERTILIZER": 223.2 / 235.0,
}


# --------------------------------------------------------------------------- census

def census(farm) -> dict[str, int]:
    """The fifteen-number tile census. One pass over the board; no allocation per tile.

    Weeds and empty tiles are counted by nobody on purpose (module docstring): they are the only
    part of the day-1..3 board that the RNG moves.
    """
    out = dict.fromkeys(FEATURES, 0)
    for row in farm.get("tiles") or []:
        for tile in row:
            if type(tile) is not dict:
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop")
                if crop in out:
                    out[crop] += 1
            elif kind == "PASTURE":
                out["PASTURE"] += 1
                animal = tile.get("animal")
                if animal in out:
                    out[animal] += 1
    out["QUADS"] = len(farm.get("unlocked_quadrants") or ())
    out["HANDS"] = len(farm.get("hands") or ())
    return out


def distance(a: dict, b: dict) -> int:
    """L1 over `FEATURES`. Absent keys are zero, so profiles can be written sparsely."""
    return sum(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in FEATURES)


def new_memory() -> dict:
    """The per-season state `fingerprint` threads through its calls."""
    return {"alive": None, "checks": 0, "known": None, "lock_step": 0, "unknown": False,
            "last": None}


def fingerprint(opp_farm, step: int, memory: dict | None = None) -> str | None:
    """The opponent's name, or `None`.

    Call it at every step; it only does work at `CHECKPOINTS`. `memory` is a dict from
    `new_memory()` kept for the season — without it each call is independent and can only ever
    answer for one checkpoint, which is not what "matched on all checkpoints so far" means.

    A profile stays alive while its census distance is within `THRESHOLD` at **every** checkpoint
    reached. Once `MIN_CHECKS` checkpoints have passed and exactly one profile is still alive, that
    is the lock. Later checkpoints keep testing it: a lock that stops matching is dropped rather
    than defended, because a wrong name is worse to O3 than no name.
    """
    if memory is None:
        memory = new_memory()
    if step not in CHECKPOINTS:
        return memory["known"]

    seen = census(opp_farm)
    memory["last"] = seen
    alive = memory["alive"]
    if alive is None:
        alive = set(PROFILES)
    alive = {n for n in alive
             if step in PROFILES[n] and distance(seen, PROFILES[n][step]) <= THRESHOLD}
    memory["alive"] = alive
    memory["checks"] += 1

    if not alive:
        memory["unknown"] = True
        if memory["known"] is not None:
            memory["known"] = None
            memory["lock_step"] = 0
        return None

    if memory["known"] is not None and memory["known"] not in alive:
        memory["known"] = None
        memory["lock_step"] = 0
    if memory["known"] is None and memory["checks"] >= MIN_CHECKS and len(alive) == 1:
        memory["known"] = next(iter(alive))
        memory["lock_step"] = step
    return memory["known"]


def is_self(name: str | None) -> bool:
    return name in SELF_PROFILES


# --------------------------------------------------------------------------- supply

def forecast_supply(opp_farm, current_step: int, known: str | None = None,
                    horizon: int = projection.HORIZON) -> dict[str, list[tuple[int, int]]]:
    """`{product: [(step, units)]}` — what their board is about to put on the market.

    The tiles half is C6's forecaster (`projection.farm_supply`) pointed at their farm: a plant's
    whole remaining schedule follows from its crop and `planted_day`, an animal's from its species
    and `placed_day`, and both are public. It is re-keyed from days-ahead to absolute steps and
    reported at hour 0 of the arrival day, which is when a harvest first meets a market turn.

    For a fingerprinted `boatlee` the answer is better than a forecast: the measured sell table is
    what it will actually do, discounted by `SETTLE_RATE` because an order is not a sale. Steps
    already past are dropped; `CONDITIONAL_SELLS` are not included at all.
    """
    today = current_step // TURNS_PER_DAY
    raw: dict = {}
    projection.farm_supply(opp_farm, today, horizon, raw)
    out: dict[str, list[tuple[int, int]]] = {}
    for item, slots in raw.items():
        rows = [((today + d) * TURNS_PER_DAY, int(round(u)))
                for d, u in sorted(slots.items()) if u > 0]
        if rows:
            out[item] = rows

    if known == "boatlee":
        limit = current_step + horizon * TURNS_PER_DAY
        for item, rows in SELL_SCHEDULE.items():
            rate = SETTLE_RATE.get(item, 1.0)
            known_rows = [(s, int(round(n * rate))) for s, n in rows
                          if current_step < s <= limit]
            if known_rows:
                out[item] = known_rows
    return out


# --------------------------------------------------------------------------- O3 seams

#: Per-seat memory of the fingerprint verdict.
#:
#: `main_v4` owns the fingerprint and keeps it in its own per-seat state, but O3's counter-mix runs
#: inside `tasks.daily_tasks` -> `projection`, which is handed nothing but `obs`. Threading the name
#: down through four signatures that have no other use for it would be worse than one module-level
#: dict with an explicit season reset, which is the same shape `projection._REDIRECTS` already uses.
LOCKED: dict[int, str | None] = {}


def set_locked(seat: int, name: str | None) -> None:
    LOCKED[int(seat)] = name


def locked(seat: int) -> str | None:
    return LOCKED.get(int(seat))


def reset_locked(seat: int | None = None) -> None:
    if seat is None:
        LOCKED.clear()
    else:
        LOCKED.pop(int(seat), None)


def settled_schedule(known: str | None) -> dict[str, list[tuple[int, int]]]:
    """`SELL_SCHEDULE` in **units that land**, not units ordered (`SETTLE_RATE`; CLAUDE.md's
    "an order is not a sale"). Empty for anyone but a fingerprinted `boatlee`."""
    if known != "boatlee":
        return {}
    return {item: [(s, int(round(n * SETTLE_RATE.get(item, 1.0)))) for s, n in rows]
            for item, rows in SELL_SCHEDULE.items()}


def windows_from_rows(rows_by_product: dict, lead: int, min_units: int,
                      spread: int = 0) -> dict[int, dict[str, int]]:
    """`{step: {product: units}}` — every step that sits inside `lead` of a material dump.

    The window is `[s - lead - spread, s + spread]`, **inclusive of `s` itself**: a sale on the dump
    step still beats theirs when it occupies an earlier market slot, because `_process_market`
    drains slot `i` for both players before it touches slot `i+1` and re-prices in between
    (`kaggriculture.py:563,628`). `spread` is the forecast's uncertainty — one day for a supply
    estimate read off their tiles, zero for the measured table.

    `min_units` is the whole reason the quantities are in `SELL_SCHEDULE` and not just the steps:
    boatlee dribbles three units of milk on some turns and drops thirty-four strawberries on others,
    and dumping our shed to get ahead of the dribble is a pure loss.
    """
    out: dict[int, dict[str, int]] = {}
    lead = max(0, int(lead))
    spread = max(0, int(spread))
    for product, rows in (rows_by_product or {}).items():
        for s, n in rows:
            n = int(n)
            if n < min_units:
                continue
            for step in range(max(0, int(s) - lead - spread), int(s) + spread + 1):
                slot = out.setdefault(step, {})
                if n > slot.get(product, 0):
                    slot[product] = n
    return out


def sell_steps(known: str | None) -> dict[int, set]:
    """`{step: {product, ...}}` — the exact steps a fingerprinted script sells at.

    Slot alignment reads this, not `windows_from_rows`: hoisting our sell above theirs is only
    meaningful on the step they actually trade, and only a measured table names a step. A ±1-day
    forecast cannot.
    """
    out: dict[int, set] = {}
    for product, rows in settled_schedule(known).items():
        for s, n in rows:
            if n > 0:
                out.setdefault(int(s), set()).add(product)
    return out


def sell_schedule(product: str, known: str = "boatlee") -> list[int]:
    """The steps a fingerprinted script sells `product` at. Empty for anyone but boatlee."""
    if known != "boatlee":
        return []
    return [s for s, _ in SELL_SCHEDULE.get(product, ())]


def sell_units(product: str, step: int, known: str = "boatlee") -> int:
    """Units *ordered* at `step` (0 if none). Multiply by `SETTLE_RATE` for units that land."""
    if known != "boatlee":
        return 0
    for s, n in SELL_SCHEDULE.get(product, ()):
        if s == step:
            return n
    return 0


def next_sell(product: str, step: int, known: str = "boatlee") -> tuple[int, int] | None:
    """The first `(step, units)` at or after `step`. O3's front-run reads this."""
    if known != "boatlee":
        return None
    for s, n in SELL_SCHEDULE.get(product, ()):
        if s >= step:
            return (s, n)
    return None
