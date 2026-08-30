"""Replay JSON -> clean per-step training pairs, with the four assertions.

The least glamorous file and the most important one.  Every trap here has the same shape: get it
wrong and training still runs, the loss still goes down, and the accuracy curve still looks
plausible.  **Nothing tells you.**  So each fix ships with an assertion that stops the program
rather than letting it produce quiet garbage (PLAN_BC Chapter 3).

The verified contract, re-measured on this machine before a line of this file was written:

* **Pairing.** `steps[i].observation` is the state *after* step `i`; `steps[i].action` is what
  caused it.  The training pair is therefore `(steps[i-1][p].observation, steps[i][p].action)` for
  `i in 1..719`.  `steps[0].action` is a schema placeholder nobody chose -- dropped.  719 pairs per
  seat, not 720.
* **Assertion 1 (offset/roster).** The roster is wiped every night (`kag.py:879-882`), so its
  length fingerprints which step's observation an action came from.  Re-measured here on episode
  95029942: **1,438 / 1,438 match against the previous step, 1,275 / 1,438 against the same step --
  exactly 163 mismatches**, reproducing PLAN_BC's number.
* **Assertion 2 (seat-1 integrity).** The stored seat-1 observation has no `step` key -- the only
  key seat 0 has and seat 1 lacks.  It is exactly reconstructible (`day*24 + hour == i` for all 720
  steps, re-measured).  The shared fields are byte-identical between seats, spot-checked by md5.
* **Assertion 3 (expert legality).** Every expert action must be legal under `model.masks`.
* **Assertion 4 (effective shed).** Unit actions are applied before `_process_market`
  (`kag.py:935-941`), so a worker can drop wheat into the shed and the same turn's SELL can sell
  it.  `PLACE`-of-an-animal goes to a *tile*, not the shed (`kag.py:376-409`), and must be excluded.

Two label sets come out of one pass -- raw per-step actions and macro "walk to tile T and do V
there" -- so choosing between them stays a training flag rather than a rewrite.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

from . import masks as M
from . import vocab as V

CORPUS_DIR = os.path.join("data", "sample_data_training_model")
TEACHER = "Ryo Hasegawa"
SHARED_KEYS = ("farms", "market", "town", "day", "hour")
MD5_SPOT_INDICES = (1, 2, 10, 100, 400, 700)     # >= 5 per episode, as the brief requires


class DecodeError(AssertionError):
    """A hard contract violation.  Never downgraded, never caught inside this module."""


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------

def read_manifest(corpus_dir=CORPUS_DIR):
    """`manifest.csv` -> `{episode_id: row}`.  Read with the `csv` module; pandas is not a
    dependency of this pipeline."""
    path = os.path.join(corpus_dir, "manifest.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        r["ryo_seat"] = int(r["ryo_seat"])
        for k in ("ryo_reward", "opponent_reward", "margin"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        out[str(r["episode_id"])] = r
    if len(out) != len(rows):
        raise DecodeError(f"duplicate episode_id in manifest: {len(rows)} rows, {len(out)} unique")
    return out


def episode_paths(corpus_dir=CORPUS_DIR, splits=("train", "val", "test")):
    """`[(episode_id, split, path), ...]`, split taken from the DIRECTORY so our splits can never
    drift from the provided ones (PLAN_BC Ch2/Ch3)."""
    out = []
    for split in splits:
        d = os.path.join(corpus_dir, split)
        for name in sorted(os.listdir(d)):
            if name.endswith(".json"):
                out.append((name[:-5], split, os.path.join(d, name)))
    return out


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------

def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def delivered_obs(steps, i, p):
    """The observation as the *agent* was handed it, not as the replay stored it.

    `__get_shared_state(position)` (`core.py:754-767`) gives both seats `step`; the saved replay
    for seat 1 does not have it.  Never read `obs["step"]` straight out of a stored replay -- and
    never suppress it either, which is the mistake E21 records on the other side.
    """
    obs = dict(steps[i][p]["observation"])
    want = int(obs["day"]) * V.TURNS_PER_DAY + int(obs["hour"])
    if want != i:
        raise DecodeError(f"step {i} seat {p}: day*24+hour == {want}, expected {i}")
    if "step" in obs and int(obs["step"]) != i:
        raise DecodeError(f"step {i} seat {p}: stored step {obs['step']} != {i}")
    obs["step"] = i
    obs["player"] = p
    return obs


# --------------------------------------------------------------------------------------
# Macro segmentation
# --------------------------------------------------------------------------------------

class _Segment:
    __slots__ = ("state_idx", "start", "n_moves")

    def __init__(self, state_idx, start):
        self.state_idx = state_idx
        self.start = (int(start[0]), int(start[1]))
        self.n_moves = 0


class MacroSegmenter:
    """Re-describe a unit's day as "go to tile X and do Y there".

    A run of `NORTH/SOUTH/EAST/WEST` terminated by a non-move op becomes one macro decision,
    attributed to the state where the run *started*.  A `PASS` issued while standing still is not a
    macro at all -- it is the `IDLE` branch of the commit head.  A run cut off by the nightly
    roster wipe is emitted with the explicit `MOVE` verb and excluded from the shortest-path
    statistic, because it has no intended target to compare against.
    """

    def __init__(self):
        self.open = {}          # unit_idx -> _Segment
        self.rows = []          # (state_idx, unit, tile, macro_verb, item, qty_bin, qty_raw,
                                #  seg_len, manhattan, is_idle, truncated)
        self.n_walk_runs = 0
        self.n_shortest = 0
        self.n_truncated = 0
        self.n_idle = 0

    def step(self, state_idx, unit_idx, pos, action, item=V.ITEM_NONE, qty_bin=0, qty_raw=1):
        verb = action[0]
        if verb in V.MOVE_VERBS:
            seg = self.open.get(unit_idx)
            if seg is None:
                seg = self.open[unit_idx] = _Segment(state_idx, pos)
            seg.n_moves += 1
            return
        seg = self.open.pop(unit_idx, None)
        here = (int(pos[0]), int(pos[1]))
        if verb == "PASS" and seg is None:
            self.n_idle += 1
            self.rows.append((state_idx, unit_idx, M.tile_index(*here), V.MV_PASS,
                              V.ITEM_NONE, 0, 1, 0, 0, 1, 0))
            return
        start = seg.start if seg is not None else here
        n_moves = seg.n_moves if seg is not None else 0
        start_state = seg.state_idx if seg is not None else state_idx
        dist = M.manhattan(start, here)
        if n_moves > 0:
            self.n_walk_runs += 1
            if n_moves == dist:
                self.n_shortest += 1
        self.rows.append((start_state, unit_idx, M.tile_index(*here),
                          V.MACRO_VERB_INDEX[verb], item, qty_bin, qty_raw, n_moves, dist, 0, 0))

    def flush_day(self, positions_by_unit):
        """Close every open run at the roster wipe.  `positions_by_unit` maps unit -> final tile."""
        for unit_idx, seg in sorted(self.open.items()):
            here = positions_by_unit.get(unit_idx, seg.start)
            here = (int(here[0]), int(here[1]))
            self.n_truncated += 1
            self.rows.append((seg.state_idx, unit_idx, M.tile_index(*here), V.MV_MOVE,
                              V.ITEM_NONE, 0, 1, seg.n_moves, M.manhattan(seg.start, here), 0, 1))
        self.open.clear()


# --------------------------------------------------------------------------------------
# The decoder
# --------------------------------------------------------------------------------------

class DecodedEpisode:
    __slots__ = ("episode_id", "split", "seat", "opponent", "reward", "opponent_reward",
                 "module_version", "seed", "config_hash", "states", "raw", "market", "macro",
                 "counters")


def _counters():
    return {
        "n_steps": 0,
        "n_unit_actions": 0,
        "n_market_orders": 0,
        # Assertion 1
        "a1_roster_checks": 0,
        "a1_roster_fail": 0,
        "a1_roster_fail_same_step": 0,      # the mutation control: pairing (obs[i], action[i])
        # Assertion 2
        "a2_md5_checks": 0,
        "a2_md5_fail": 0,
        "a2_step_patched": 0,
        # Assertion 3
        "a3_expert_verb_rejected": 0,
        "a3_expert_item_rejected": 0,
        "a3_expert_qty_rejected": 0,
        "n_expert_actions_rejected_by_mask": 0,
        "a3_expert_order_op_rejected": 0,
        "a3_expert_order_item_rejected": 0,
        "n_expert_orders_rejected_by_mask": 0,
        # Assertion 4
        "a4_sells": 0,
        "a4_sell_over_effective_shed": 0,        # running: shed drained slot by slot
        "a4_sell_over_effective_shed_static": 0,  # each order vs the full effective shed
        "a4_sell_over_observed_shed": 0,          # the naive check the carve-out exists to fix
        "a4_animal_place_to_tile": 0,             # proves the carve-out actually fired
        "a4_deposits": 0,
        "a4_withdrawals": 0,
        # macro
        "macro_walk_runs": 0,
        "macro_shortest": 0,
        "macro_truncated": 0,
        "macro_idle": 0,
        "macro_rows": 0,
        # misc
        "n_plant_blocked_by_cliff": 0,
        "n_expert_plant_rewritten_to_pass": 0,
        "n_plant_bursts": 0,                       # turns where 2+ units planted the same crop
        "n_turns_plant_demand_equals_seeds": 0,    # ... and sat exactly on the cliff edge
        "n_orders_dropped_over_cap": 0,
        "n_units_over_cap": 0,
        "n_ignored_extra_args": 0,
    }


def decode_episode(path, seat, episode_id=None, split=None, manifest_row=None,
                   check_teacher=True, counters=None):
    """Decode one replay for one seat.  Returns a `DecodedEpisode`; raises on a contract violation.

    Streaming discipline: exactly one episode is held in memory at a time (the raw files are ~30 MB
    each and expand several-fold as Python objects).
    """
    with open(path) as f:
        d = json.load(f)
    steps = d["steps"]
    if len(steps) != V.EPISODE_STEPS:
        raise DecodeError(f"{path}: {len(steps)} steps, expected {V.EPISODE_STEPS}")
    info = d.get("info") or {}
    teams = info.get("TeamNames") or []
    if check_teacher and teams and teams[seat] != TEACHER:
        raise DecodeError(f"{path}: TeamNames[{seat}] == {teams[seat]!r}, expected {TEACHER!r} "
                          f"-- manifest and files have drifted apart")
    for p, st in enumerate(d.get("statuses") or []):
        if st != "DONE":
            raise DecodeError(f"{path}: statuses[{p}] == {st!r}; truncated action tails")

    c = counters if counters is not None else _counters()
    ep = DecodedEpisode()
    ep.episode_id = episode_id or str(info.get("EpisodeId") or os.path.basename(path)[:-5])
    ep.split = split
    ep.seat = seat
    ep.opponent = (manifest_row or {}).get("opponent") or (
        teams[1 - seat] if len(teams) > 1 else "")
    rewards = d.get("rewards") or [None, None]
    ep.reward = rewards[seat]
    ep.opponent_reward = rewards[1 - seat]
    ep.module_version = d.get("module_version")
    ep.seed = info.get("seed")
    ep.config_hash = _md5(d.get("configuration") or {})

    # --- Assertion 2: seat-1 integrity, spot-checked by md5 --------------------------------
    if seat == 1:
        for i in MD5_SPOT_INDICES:
            if i >= len(steps):
                continue
            o0, o1 = steps[i][0]["observation"], steps[i][1]["observation"]
            for k in SHARED_KEYS:
                c["a2_md5_checks"] += 1
                if _md5(o0[k]) != _md5(o1[k]):
                    c["a2_md5_fail"] += 1
                    raise DecodeError(
                        f"{path}: seat-1 shared field {k!r} differs from seat 0 at step {i} -- "
                        f"the replay format changed and every number downstream of it is void")

    states, raw, market, macro = [], [], [], MacroSegmenter()

    for i in range(1, V.EPISODE_STEPS):
        state_idx = i - 1
        obs = delivered_obs(steps, state_idx, seat)
        if "step" not in steps[state_idx][seat]["observation"]:
            c["a2_step_patched"] += 1
        action = steps[i][seat]["action"]
        if not isinstance(action, dict):
            raise DecodeError(f"{path} step {i} seat {seat}: action is {type(action)}, not a dict")

        farm = obs["farms"][seat]
        hands_actions = list(action.get("hands") or [])
        # --- Assertion 1: the off-by-one -----------------------------------------------
        c["a1_roster_checks"] += 1
        if len(hands_actions) != len(farm["hands"]):
            c["a1_roster_fail"] += 1
            raise DecodeError(
                f"{path} step {i} seat {seat}: action names {len(hands_actions)} hands but the "
                f"previous observation had {len(farm['hands'])} -- the (state, action) pairing is "
                f"off by one")
        if len(hands_actions) != len(steps[i][seat]["observation"]["farms"][seat]["hands"]):
            c["a1_roster_fail_same_step"] += 1   # the control: how badly the WRONG pairing fails

        unit_actions = [action.get("farmer", ["PASS"])] + hands_actions
        if len(unit_actions) > V.MAX_UNITS:
            c["n_units_over_cap"] += 1

        ts = M.TurnState(obs, seat)
        blocked = ts.set_blocked_crops(unit_actions)
        c["n_plant_blocked_by_cliff"] += sum(
            1 for a in unit_actions
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked)
        c["n_plant_bursts"] += sum(1 for n in ts.plant_demand.values() if n > 1)
        if ts.plant_demand_at_the_edge():
            c["n_turns_plant_demand_equals_seeds"] += 1

        for u, a in enumerate(unit_actions):
            pos = list(ts.positions[u]) if u < len(ts.positions) else [0, 0]
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                # The interpreter rewrites a cliff-blocked PLANT to ["PASS"] BEFORE applying it
                # (`_allowed`, kag.py:928-933), so PASS is what the expert's policy actually did
                # this turn.  Labelling the request instead would teach the model an action our
                # own mask forbids it from ever emitting -- and would leave Assertion 3 reading
                # non-zero for a reason that is not a mask defect.  See the report: 5 occurrences
                # in episode 95029942 seat 0, every one of them a crop with zero seeds.
                c["n_expert_plant_rewritten_to_pass"] += 1
                a = ["PASS"]
            vi, ii, qb, qr, extra = V.encode_unit_action(a)
            c["n_ignored_extra_args"] += extra
            c["n_unit_actions"] += 1

            # --- Assertion 3: the mask must never reject the expert --------------------
            verb_ok, item_ok, qty_ok = M.unit_action_legality(ts, u, a)
            if not verb_ok:
                c["a3_expert_verb_rejected"] += 1
            elif not item_ok:
                c["a3_expert_item_rejected"] += 1
            elif not qty_ok:
                c["a3_expert_qty_rejected"] += 1
            if not (verb_ok and item_ok and qty_ok):
                c["n_expert_actions_rejected_by_mask"] += 1

            if u < V.MAX_UNITS:
                raw.append((state_idx, u, vi, ii, qb, qr))
            macro.step(state_idx, u, pos, a, ii, qb, qr)
            ts.apply_unit(u, a)

        c["a4_animal_place_to_tile"] += ts.n_animal_placed

        # --- Assertion 4: effective shed, with the animal carve-out ---------------------
        eff = ts.effective_shed()
        eff_static = dict(eff)
        observed = {k: int(v) for k, v in obs["private"]["shed"].items()}
        c["a4_deposits"] += sum(ts.deposits.values())
        c["a4_withdrawals"] += sum(ts.withdrawals.values())

        orders = list(action.get("market") or [])
        if len(orders) > V.MAX_MARKET_ORDERS:
            c["n_orders_dropped_over_cap"] += len(orders) - V.MAX_MARKET_ORDERS
        for slot, o in enumerate(orders[: V.MAX_MARKET_ORDERS]):
            oi, ii, _qb, qr, extra = V.encode_market_order(o)
            c["n_ignored_extra_args"] += extra
            c["n_market_orders"] += 1
            op_ok, item_ok = M.market_order_legality(ts, o)
            if not op_ok:
                c["a3_expert_order_op_rejected"] += 1
            elif not item_ok:
                c["a3_expert_order_item_rejected"] += 1
            if not (op_ok and item_ok):
                c["n_expert_orders_rejected_by_mask"] += 1

            if oi == V.M_SELL:
                item = V.ITEMS[ii]
                c["a4_sells"] += 1
                if qr > eff.get(item, 0):
                    c["a4_sell_over_effective_shed"] += 1
                if qr > eff_static.get(item, 0):
                    c["a4_sell_over_effective_shed_static"] += 1
                if qr > observed.get(item, 0):
                    c["a4_sell_over_observed_shed"] += 1
                eff[item] = max(0, eff.get(item, 0) - qr)
                # Quantity is bucketed against what is actually there, so an oversized "sell the
                # lot" request encodes as ALL rather than as a number nobody meant (PLAN_BC Ch3).
                qb = V.encode_qty(qr, eff_static.get(item, 0))
            else:
                qb = V.encode_qty(qr) if V.MARKET_OP_ARITY[V.MARKET_OPS[oi]][0] == 2 else 0
            market.append((state_idx, slot, oi, ii, qb, qr))
            ts.apply_market(o)

        states.append(obs)
        c["n_steps"] += 1

        if (i % V.TURNS_PER_DAY) == 0:                       # roster wiped (kag.py:879-882)
            # Where each unit ended up AFTER this turn's moves -- the run's real terminal tile.
            macro.flush_day({u: p for u, p in enumerate(ts.positions)})

    macro.flush_day({})
    c["macro_walk_runs"] += macro.n_walk_runs
    c["macro_shortest"] += macro.n_shortest
    c["macro_truncated"] += macro.n_truncated
    c["macro_idle"] += macro.n_idle
    c["macro_rows"] += len(macro.rows)

    ep.states = states
    ep.raw = raw
    ep.market = market
    ep.macro = macro.rows
    ep.counters = c
    del d
    return ep
