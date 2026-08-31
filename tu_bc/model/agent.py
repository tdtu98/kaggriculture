"""Chapter 6: the trained checkpoint, wrapped as a real `agent(obs) -> action`.

`net.py`'s `forward` is the **teacher-forced** pass -- the expert's own earlier picks drive the
autoregressive state, because at training time we condition on states *the expert* reached.  This
file is the other half: the **free-running** decode, where the model feeds itself.  That difference
is exactly the exposure bias PLAN_BC Chapter 6 exists to measure, so the two paths are kept
provably identical apart from where the choices come from: `_SlotDecoder` below calls the same
submodules in the same order as `net._unit_slots` / `net._market_slots`, and
`tests/test_bc_agent.py::test_free_running_matches_teacher_forced` drives it with the expert's
picks and requires the logits to match `net.forward`'s to within float noise.  A hand-rolled second
implementation of the forward pass that silently drifted would look exactly like "BC didn't work"
(E39, PLAN_BC risk 9).

**The macro action space, inverted.**  `decode.py` turned the expert's per-step actions into
"walk to tile T, then do V there".  This file turns that back into per-step actions: an idle worker
is asked for one macro, the macro is remembered, and the worker emits one move a turn until it
arrives and fires the verb.  Nothing about the walk is a network decision.

**Four rules that are not obvious.**

1. *The daily reset wipes everything.*  `_end_of_day` (`kag.py:879-882`) drops every inventory into
   the shed, teleports the farmer to spawn and sets `hands = []`.  So at hour 0 there is no roster
   to carry a macro on, the farmer is somewhere else than it planned from, and whatever it was
   carrying is gone.  **Every macro is cleared at the day boundary, the farmer's included.**  That
   is also what the training labels say: `MacroSegmenter.flush_day` closes every open run at the
   wipe, so the expert is *always* asked for a fresh decision at hour 0 and never for a resumption.
   Keeping the farmer's macro would put the model on states its labels never described.
2. *Seed reservations.*  The mask's live seed budget (`masks.verb_mask_at`) is spent when a `PLANT`
   *fires*, but a macro commits to a plant that fires several turns later.  Three idle workers each
   committing to the last wheat seed would all arrive and all be dropped by the atomic-PLANT rule
   (`kag.py:920-933`).  So outstanding `PLANT` macros hold a reservation against `ts.seeds` while
   they walk.  Counted as `n_plant_reservations`, so a zero would say the mechanism never fired.
3. *fp16.*  The shards store features as float16 (`features.STORE_DTYPE`) and `batching.make_batch`
   widens them back.  The weights therefore learned fp16-rounded inputs, so inference rounds too.
4. *Never emit an illegal action.*  Masks are applied to the logits, and then the assembled action
   is re-checked against `masks.unit_action_legality` / `masks.market_order_legality` before it
   leaves.  Anything that fails becomes `PASS` and increments `n_fallback_pass`.  A nonzero count is
   a wrapper bug to report, not to hide.

Greedy argmax throughout for v1: with deterministic features and a fixed seed the whole episode
reproduces, which is what makes the Chapter 6 ladder measurable at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as kag

from . import checkpoint as C
from . import features as F
from . import masks as M
from . import net as N
from . import vocab as V

# Macro verb id -> raw verb id, for `vocab.decode_unit_action`.  `MOVE` has no raw counterpart.
MACRO_TO_RAW = tuple(
    V.VERB_INDEX[name] if name != "MOVE" else -1 for name in V.MACRO_VERBS
)


@dataclass
class Macro:
    """One outstanding "walk to `tile`, then `verb` there" commitment."""

    tile: int
    verb: int            # index into V.MACRO_VERBS
    item: int            # index into V.ITEMS, or V.ITEM_NONE
    qty_bin: int
    issued_step: int


def _counters():
    return {
        "n_turns": 0,
        "n_decisions": 0,           # commit-head calls == times a worker was idle and asked
        "n_commit_act": 0,
        "n_commit_idle": 0,
        "n_macros_issued": 0,       # ACT decisions whose target was not the worker's own tile
        "n_macros_completed": 0,    # ... that walked all the way there and fired the verb
        "n_macros_abandoned": 0,    # ... cleared at a day boundary or illegal on arrival
        "n_macros_rearrived_move": 0,   # verb was MOVE: arrive, then decide again
        "n_immediate_verbs": 0,     # ACT decisions fired in place (target == own tile)
        "n_moves_emitted": 0,
        "n_unit_pass": 0,
        "n_fallback_pass": 0,       # assembled unit action failed the post-hoc legality check
        "n_orders_emitted": 0,
        "n_order_fallback": 0,      # assembled order failed the post-hoc legality check
        "n_market_stop": 0,         # STOP head ended the order list
        "n_market_no_legal_op": 0,  # ... or nothing was legal
        "n_plant_reservations": 0,
        # Options removed because another walking worker already claimed them.  A VOLUME counter,
        # not a decision counter: in `claims="tile"` mode it counts hidden *tile slots* summed over
        # every decision (so ~10k a season at ~3 claims x 3,400 decisions), and in `claims="verb"`
        # mode it counts hidden verbs.  Read it as "the mechanism fired", never as "actions lost".
        "n_claim_blocked": 0,
        "n_animal_place_qty_carveout": 0,
        "n_day_resets": 0,
        "n_episode_resets": 0,
        "n_missing_step": 0,
    }


class _SlotDecoder:
    """The per-slot forward pass, in the same order as `net._unit_slots` / `net._market_slots`.

    Every method calls the model's own submodules; nothing here re-derives a weight.  Batch size is
    always 1, so the tensors are `[1, ...]` exactly as the trained path expects.
    """

    def __init__(self, model, torch, device="cpu"):
        self.m = model
        self.torch = torch
        self.dev = torch.device(device)
        self.scale = 1.0 / math.sqrt(model.cfg.d_model)

    # -- small tensor helpers ------------------------------------------------------------

    def b(self, arr):
        """A numpy boolean mask -> a `[1, ...]` tensor on the model's device."""
        return self.torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(self.dev)

    def ids(self, i):
        return self.torch.tensor([int(i)], dtype=self.torch.long, device=self.dev)

    def ones1(self):
        return self.torch.ones(1, 1, device=self.dev)

    # -- encoding ------------------------------------------------------------------------

    def encode(self, feats):
        torch = self.torch
        batch = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
                       .unsqueeze(0).to(self.dev)
                 for k, v in feats.items()}
        return self.m.encode(batch), batch["global"]

    # -- unit slots ----------------------------------------------------------------------

    def slot_h(self, workers, ctx, ar, scal, u):
        return self.m.slot_trunk(self.torch.cat([workers[:, u], ctx, ar, scal], dim=-1))

    def commit_logits(self, h):
        return self.m.commit_head(h)

    def tile_logits(self, tiles, h, mask=None):
        q = self.m.tile_query(h)
        logits = self.torch.einsum("btd,bd->bt", tiles, q) * self.scale
        return logits if mask is None else N.masked_logits(logits, mask)

    def verb_logits(self, h, tile_e, mask):
        return N.masked_logits(self.m.verb_head(self.torch.cat([h, tile_e], dim=-1)), mask)

    def item_logits(self, h, tile_e, verb_e, mask):
        return N.masked_logits(
            self.m.item_head(self.torch.cat([h, tile_e, verb_e], dim=-1)), mask)

    def qty_logits(self, h, tile_e, verb_e, item_e, mask):
        return N.masked_logits(
            self.m.qty_head(self.torch.cat([h, tile_e, verb_e, item_e], dim=-1)), mask)

    def ar_step(self, ar, tile_e, verb_e, scal, valid):
        # Delegated, not re-derived: the update rule is subtle (see `BCNet._ar_step` -- an
        # ungated LayerNorm over the all-zero start state silently zeroed whole training
        # batches), and it must be the same rule here or the parity gate is measuring two bugs
        # cancelling.  What this class still owns, and what the gate checks, is the *sequencing*.
        torch = self.torch
        return self.m._ar_step(
            self.m.ar_norm, ar,
            self.m.ar_update(torch.cat([ar, tile_e, verb_e, scal, valid], dim=-1)), valid)

    # -- market slots --------------------------------------------------------------------

    def market_init(self, ctx, ar):
        return self.torch.tanh(self.m.market_init(self.torch.cat([ctx, ar], dim=-1)))

    def market_h(self, ctx, mar, scal, k):
        slot = self.m.slot_onehot[k].unsqueeze(0)
        return self.m.market_trunk(self.torch.cat([ctx, mar, scal, slot], dim=-1))

    def stop_logits(self, h):
        return self.m.stop_head(h)

    def op_logits(self, h, mask):
        return N.masked_logits(self.m.op_head(h), mask)

    def mitem_logits(self, h, op_e, mask):
        return N.masked_logits(self.m.mitem_head(self.torch.cat([h, op_e], dim=-1)), mask)

    def mqty_logits(self, h, op_e, item_e, mask):
        return N.masked_logits(
            self.m.mqty_head(self.torch.cat([h, op_e, item_e], dim=-1)), mask)

    def market_ar_step(self, mar, op_e, item_e, valid):
        torch = self.torch
        return self.m._ar_step(
            self.m.market_ar_norm, mar,
            self.m.market_ar_update(torch.cat([mar, op_e, item_e, valid], dim=-1)), valid)


def drop_qty_all(mask, allow_all):
    """`QTY_ALL` is a legal *bucket* but it is only ever a **label** for `SELL`.

    `decode.py` encodes quantities with `available_fn=None` everywhere except the `SELL` branch
    (`decode.py:376-390`), so `QTY_ALL` never appears as a target for a unit `PICKUP`/`PLACE` or for
    `BUY_SEED`/`BUY_PRODUCT`/`BUY_ANIMAL`.  `masks.qty_mask` sets it unconditionally, which is
    correct as *legality* and wrong as an *alphabet*: offering the model a class it was never
    trained to emit is train/test skew, and the qty head's logit for it is unconstrained.

    Measured cost of not doing this, seed 43999 vs boatlee: `["BUY_SEED", "WHEAT", 1575]` --
    `QTY_ALL` resolves to `money // 10` -- fired three times in the last third of the season and
    took the bank from $15,753 to $3.  Final money: **$0**.

    Returns a copy; never returns an all-False row.
    """
    m = np.asarray(mask).copy()
    if allow_all or not m[V.QTY_ALL]:
        return m
    m[V.QTY_ALL] = False
    if not m.any():
        m[0] = True
    return m


def _next_move(x, y, tx, ty):
    """One step of the walk.  x first, then y -- a fixed tie-break so a replay reproduces."""
    if tx > x:
        return "EAST"
    if tx < x:
        return "WEST"
    if ty > y:
        return "SOUTH"
    if ty < y:
        return "NORTH"
    return None


def _unit_available(ts, idx, macro_verb, item_id):
    """How many units a `PICKUP` / `PLACE` could move right now -- what `QTY_ALL` resolves to."""
    if item_id == V.ITEM_NONE:
        return 0
    item = V.ITEMS[item_id]
    name = V.MACRO_VERBS[macro_verb]
    if name == "PICKUP":
        return int(ts.shed.get(item, 0))
    if name == "PLACE":
        inv = ts.inventories[idx] if idx < len(ts.inventories) else {}
        return int(min(inv.get(item, 0), ts.shed_free()))
    return 0


def _market_available(ts, op_id, item_id):
    if item_id == V.ITEM_NONE:
        return 0
    item = V.ITEMS[item_id]
    if op_id == V.M_SELL:
        return int(ts.shed.get(item, 0))
    if op_id == V.M_BUY_SEED:
        cost = kag.CROPS[item]["seed"]
        return int(ts.money // cost) if cost > 0 else 0
    if op_id == V.M_BUY_PRODUCT:
        price = max(1, ts.price(item, offset=-1))
        return int(min(ts.money // price, ts.shed_free()))
    if op_id == V.M_BUY_ANIMAL:
        cost = kag.ANIMALS[item]["cost"]
        return int(min(ts.money // cost, ts.shed_free()))
    return 0


class _SeedReservation:
    """Temporarily hide the seeds that outstanding PLANT macros have already spoken for."""

    __slots__ = ("ts", "reserved", "saved")

    def __init__(self, ts, reserved):
        self.ts = ts
        self.reserved = reserved
        self.saved = None

    def __enter__(self):
        if self.reserved:
            self.saved = dict(self.ts.seeds)
            for crop, n in self.reserved.items():
                self.ts.seeds[crop] = max(0, self.ts.seeds.get(crop, 0) - int(n))
        return self.ts

    def __exit__(self, *exc):
        if self.saved is not None:
            self.ts.seeds.clear()
            self.ts.seeds.update(self.saved)
            self.saved = None
        return False


class BCAgent:
    """A loaded checkpoint that plays.  One instance per episode -- it carries per-episode state.

    Reusing one across games leaks the previous season into the next; `harness/run.py` rebuilds its
    agents per game for exactly this reason, and `make_agent` returns a fresh one every call.
    """

    #: How an outstanding macro's target is protected from a second worker.  See `_verb_mask` and
    #: `_tile_mask`.  `"off"` is the literal Chapter 6 spec; the default is set by measurement, not
    #: by preference -- 40 episodes vs `starter`, both seats, seed block 42000:42020:
    #:
    #:     off    winrate 42.5% [28.5, 57.8]   mean money  $3,810   above $3,000: 18/40
    #:     verb   (8 seeds only) mean money $1,000 -- worse than `off`, and structurally so: the
    #:            pointer has already spent the decision on the tile, so blocking the verb there
    #:            leaves only PASS.  Kept as a named mode because that is the lesson.
    #:     tile   winrate 77.5% [62.5, 87.7]   mean money $16,629   above $3,000: 31/40
    #:
    #: The intervals do not overlap.  Record as **E100**.
    CLAIM_MODES = ("off", "verb", "tile")

    def __init__(self, model, torch, device="cpu", strict=True, claims="tile"):
        self.model = model
        self.torch = torch
        self.device = device
        self.strict = strict
        if claims not in self.CLAIM_MODES:
            raise ValueError(f"claims must be one of {self.CLAIM_MODES}, got {claims!r}")
        self.claim_mode = claims
        self.dec = _SlotDecoder(model, torch, device=device)
        self.counters = _counters()
        self.turn_times = []
        self.reset()

    # -- per-episode state ---------------------------------------------------------------

    def reset(self):
        self.macros: dict[int, Macro] = {}
        self.reserved: dict[str, int] = {}
        self.claims: dict[tuple, int] = {}
        self._prev_obs = None
        self._last_step = -1
        self._last_day = -1

    def _clear_macros(self, why):
        n = len(self.macros)
        self.counters["n_macros_abandoned"] += n
        self.macros.clear()
        self.reserved.clear()
        self.claims.clear()
        return n

    def _reserve(self, macro):
        if V.MACRO_VERBS[macro.verb] == "PLANT" and macro.item != V.ITEM_NONE:
            crop = V.ITEMS[macro.item]
            self.reserved[crop] = self.reserved.get(crop, 0) + 1
            self.counters["n_plant_reservations"] += 1
        key = (macro.tile, macro.verb)
        self.claims[key] = self.claims.get(key, 0) + 1

    def _release(self, macro):
        if V.MACRO_VERBS[macro.verb] == "PLANT" and macro.item != V.ITEM_NONE:
            crop = V.ITEMS[macro.item]
            if self.reserved.get(crop):
                self.reserved[crop] -= 1
        key = (macro.tile, macro.verb)
        if self.claims.get(key):
            self.claims[key] -= 1
            if not self.claims[key]:
                del self.claims[key]

    # -- masks ---------------------------------------------------------------------------

    def _reserved_seeds(self, ts):
        """`ts.seeds` with outstanding PLANT macros subtracted, as a context manager.

        `masks.verb_mask_at` spends a seed when a PLANT *fires*; a macro commits to one that fires
        several turns later.  Without the reservation, three idle workers all commit to the last
        wheat seed, all arrive together, and the atomic-PLANT rule (`kag.py:920-933`) drops every
        one of them.
        """
        return _SeedReservation(ts, self.reserved)

    def _verb_mask(self, ts, idx, tile):
        """Macro verb legality at `tile`, minus what other walking workers have already claimed.

        The claim is the same kind of correction as the seed reservation, and it exists for the
        same reason: **the wrapper throws away information the expert had.**  Ryo can see his own
        plan, so he never sends two workers to water one plant.  Our worker tokens carry no "this
        one is already walking to tile 47" feature and the AR state only remembers decisions taken
        *this turn*, so nothing stops the model reissuing a macro another worker is three steps
        from completing.  Measured on seed 7 before this existed: **38% of WATER macros targeted a
        tile another live macro already held, and 34% of them were dead on arrival** -- the second
        worker walks the whole way and finds the plant already watered.

        The claim is per `(tile, verb)`, not per tile, because an animal tile legitimately wants
        `FEED`, `CARE` and `COLLECT_FERTILIZER` from three different workers on the same day.

        `n_claim_blocked` proves it fires; a zero would mean this paragraph is fiction (E44).
        """
        xy = M.tile_xy(tile)
        with self._reserved_seeds(ts):
            mask = M.verb_mask_at(ts, idx, xy=xy, macro=True)
        if self.claims and self.claim_mode == "verb":
            mine = self.macros.get(idx)
            for verb in range(V.N_MACRO_VERBS):
                if verb in (V.MV_PASS, V.MV_MOVE) or not mask[verb]:
                    continue
                held = self.claims.get((tile, verb), 0)
                if mine is not None and mine.tile == tile and mine.verb == verb:
                    held -= 1                       # never let a worker block itself
                if held > 0:
                    mask[verb] = False
                    self.counters["n_claim_blocked"] += 1
        return mask, xy

    def _tile_mask(self, ts, idx):
        """`claims="tile"`: hide tiles another walking worker has already claimed from the pointer.

        PLAN_BC Ch5 says the tile pointer is never masked and training fed it all-ones, so this is
        a deviation and is measured as one.  Blocking the *verb* instead cannot work: by then the
        pointer has already spent the decision on that tile, and the only survivor is `PASS`.
        The worker's own square is always left open, so the row can never be all-False.
        """
        m = np.ones(V.N_TILES, dtype=bool)
        if self.claim_mode != "tile" or not self.claims:
            return m
        mine = self.macros.get(idx)
        for (tile, verb), n in self.claims.items():
            if mine is not None and mine.tile == tile and mine.verb == verb:
                n -= 1
            if n > 0:
                m[tile] = False
        pos = ts.positions[idx]
        here = M.tile_index(int(pos[0]), int(pos[1]))
        if not m.any():
            m[here] = True
        self.counters["n_claim_blocked"] += int((~m).sum())
        return m

    def _item_mask(self, ts, idx, verb, xy):
        with self._reserved_seeds(ts):
            return M.item_mask_at(ts, idx, verb, xy=xy, macro=True)

    # -- the turn ------------------------------------------------------------------------

    def __call__(self, obs, config=None):
        # `config` is accepted and ignored: `kaggle_environments/agent.py` inspects the callable's
        # arity and hands a two-argument agent `(observation, configuration)`.  A bound `__call__`
        # counts `self`, so a one-argument signature is called with two and raises TypeError --
        # which is the same class of failure as E21, where the submission never loaded at all.
        t0 = time.perf_counter()
        try:
            action = self._act(obs)
        finally:
            self.turn_times.append(time.perf_counter() - t0)
        return action

    def _act(self, obs):
        torch = self.torch
        c = self.counters

        if "step" not in obs:
            # The delivered observation carries `step` on BOTH seats (`core.py:754-767`, E21).  Its
            # absence means we are being handed a stored replay state, and every clock feature and
            # the restart detector below would be silently wrong.
            c["n_missing_step"] += 1
            if self.strict:
                raise KeyError("delivered observation has no 'step' -- see CLAUDE.md E21")
            obs = dict(obs)
            obs["step"] = int(obs["day"]) * V.TURNS_PER_DAY + int(obs["hour"])

        step = int(obs["step"])
        day = int(obs["day"])
        me = int(obs["player"])

        if step <= self._last_step or step != self._last_step + 1:
            if self._last_step >= 0:
                c["n_episode_resets"] += 1
            self.reset()
        elif self._last_step >= 0 and day != self._last_day:
            # `_end_of_day` wiped the roster, teleported the farmer and emptied every inventory.
            self._clear_macros("day boundary")
            c["n_day_resets"] += 1
        self._last_step, self._last_day = step, day

        ts = M.TurnState(obs, me)
        n_units = ts.n_units
        for idx in list(self.macros):
            if idx >= n_units:                       # roster shrank under us; never seen mid-day
                self._release(self.macros[idx])
                del self.macros[idx]
                c["n_macros_abandoned"] += 1

        feats = F.extract(obs, me, prev_obs=self._prev_obs)
        # Shards store fp16 and `batching.make_batch` widens them: the weights learned rounded
        # inputs, so inference rounds too.  Free, and it removes a whole class of "why is online
        # different from offline" question.
        feats = {k: v.astype(np.float16).astype(np.float32) for k, v in feats.items()}
        with torch.no_grad():
            (tiles_emb, workers_emb, ctx), g = self.dec.encode(feats)

            unit_actions, ar = self._decode_units(
                ts, obs, tiles_emb, workers_emb, ctx, g, n_units)
            orders = self._decode_market(ts, ctx, ar, g)

        self._prev_obs = obs
        c["n_turns"] += 1
        return {"farmer": unit_actions[0],
                "hands": unit_actions[1:n_units],
                "market": orders}

    # -- units ---------------------------------------------------------------------------

    def _decode_units(self, ts, obs, tiles_emb, workers_emb, ctx, g, n_units):
        torch, dec, c = self.torch, self.dec, self.counters
        ar = torch.zeros(1, self.model.cfg.d_ar, device=dec.dev)
        seeds = g[:, N.G_SEEDS] * 20.0
        money = g[:, N.G_MONEY:N.G_MONEY + 1]
        shed_free = g[:, N.G_SHED_FREE:N.G_SHED_FREE + 1]
        actions = [["PASS"] for _ in range(max(n_units, 1))]

        for u in range(min(V.MAX_UNITS, n_units)):
            pos = ts.positions[u]
            x, y = int(pos[0]), int(pos[1])
            macro = self.macros.get(u)

            # 1. Still walking -> emit the next move.  No decision, so no AR update: the training
            #    labels have no macro row at a mid-walk state either (commit == IGNORE).
            if macro is not None:
                tx, ty = M.tile_xy(macro.tile)
                mv = _next_move(x, y, tx, ty)
                if mv is not None:
                    actions[u] = self._emit(ts, u, [mv])
                    c["n_moves_emitted"] += 1
                    continue
                # 2. Arrived.
                if V.MACRO_VERBS[macro.verb] == "MOVE":
                    self._release(macro)
                    del self.macros[u]
                    c["n_macros_rearrived_move"] += 1
                    c["n_macros_completed"] += 1
                    macro = None                       # fall through and decide again, in place
                else:
                    actions[u] = self._fire(ts, u, macro)
                    self._release(macro)
                    del self.macros[u]
                    c["n_macros_completed"] += 1
                    continue

            # 3. Idle -> ask the network.
            c["n_decisions"] += 1
            scal = torch.cat([seeds / 20.0, money, shed_free], dim=-1)
            h = dec.slot_h(workers_emb, ctx, ar, scal, u)
            act = int(torch.argmax(dec.commit_logits(h), dim=-1).item()) == 1

            if act:
                c["n_commit_act"] += 1
                tmask = self._tile_mask(ts, u)
                tlog = dec.tile_logits(tiles_emb, h,
                                       None if tmask.all() else dec.b(tmask))
                tile = int(torch.argmax(tlog, dim=-1).item())
            else:
                c["n_commit_idle"] += 1
                tile = M.tile_index(x, y)

            tile_e = tiles_emb[:, tile]

            if act:
                vmask, xy = self._verb_mask(ts, u, tile)
                verb = int(torch.argmax(dec.verb_logits(h, tile_e, dec.b(vmask)),
                                        dim=-1).item())
            else:
                verb, xy = V.MV_PASS, (x, y)
            verb_e = self.model.verb_emb(dec.ids(verb))

            if act:
                imask = self._item_mask(ts, u, verb, xy)
                item = int(torch.argmax(dec.item_logits(h, tile_e, verb_e, dec.b(imask)),
                                        dim=-1).item())
            else:
                item = V.ITEM_NONE
            item_e = self.model.item_emb(dec.ids(item))

            if act:
                qmask = drop_qty_all(M.qty_mask_at(ts, u, verb, item, xy=xy, macro=True),
                                     allow_all=False)
                if not qmask.any():
                    qmask[0] = True     # bucket 0 == "1"; the post-hoc check is the real guard
                qty = int(torch.argmax(dec.qty_logits(h, tile_e, verb_e, item_e, dec.b(qmask)),
                                       dim=-1).item())
            else:
                qty = 0

            ar = dec.ar_step(ar, tile_e, verb_e, scal, dec.ones1())
            # The planting cliff, mirrored from `net._unit_slots` line for line: a slot that
            # commits to PLANT spends a seed of that crop in the running state the next slot is
            # told about.  The `clamp_min` is unconditional there, so it is unconditional here --
            # `tests/test_bc_agent.py::test_free_running_matches_teacher_forced` catches the
            # difference even though live seed counts are never negative.
            spent = np.zeros(len(V.CROPS), dtype=np.float32)
            if verb == V.MACRO_VERB_INDEX["PLANT"] and item != V.ITEM_NONE:
                for j, crop in enumerate(V.CROPS):
                    if V.ITEM_INDEX[crop] == item:
                        spent[j] = 1.0
            seeds = (seeds - torch.from_numpy(spent).unsqueeze(0).to(dec.dev)).clamp_min(0.0)

            if not act or verb == V.MV_PASS:
                actions[u] = ["PASS"]
                c["n_unit_pass"] += 1
                continue

            new = Macro(tile=tile, verb=verb, item=item, qty_bin=qty, issued_step=int(obs["step"]))
            if tile == M.tile_index(x, y) and V.MACRO_VERBS[verb] != "MOVE":
                c["n_immediate_verbs"] += 1
                actions[u] = self._fire(ts, u, new)
            else:
                mv = _next_move(x, y, *M.tile_xy(tile))
                if mv is None:                        # MOVE onto our own square: nothing to do
                    actions[u] = ["PASS"]
                    c["n_unit_pass"] += 1
                    continue
                self.macros[u] = new
                self._reserve(new)
                c["n_macros_issued"] += 1
                c["n_moves_emitted"] += 1
                actions[u] = self._emit(ts, u, [mv])

        return actions, ar

    def _fire(self, ts, u, macro):
        """Turn an arrived macro into a concrete unit action, then check it and apply it."""
        raw = MACRO_TO_RAW[macro.verb]
        if raw < 0:
            return self._emit(ts, u, ["PASS"])
        avail = _unit_available(ts, u, macro.verb, macro.item)
        action = V.decode_unit_action(raw, macro.item, macro.qty_bin, avail)
        return self._emit(ts, u, action)

    def _emit(self, ts, u, action):
        """Post-hoc legality check, then advance the running turn state.

        The mask that chose this action was evaluated at commit time, several turns and a whole
        opponent ago.  This is the check that makes "never emit an illegal action" true rather than
        intended -- and `n_fallback_pass` is how we find out when it is not.
        """
        if action[0] != "PASS":
            verb_ok, item_ok, qty_ok = M.unit_action_legality(ts, u, action)
            if verb_ok and item_ok and not qty_ok and self._is_animal_place(ts, u, action):
                # `PLACE <animal>` onto a matching structure puts the animal on the TILE and never
                # in the shed (`kag.py:381-392`), so the environment never reads the quantity.
                # `masks.qty_mask_at` measures `min(inv, shed_free)` regardless, so a full shed
                # makes it report "no legal quantity" for an action that is entirely legal.  The
                # corpus never hit it (Assertion 3 reads zero) because the expert's shed had room;
                # a free-running agent absolutely will.  Carve it out here rather than editing a
                # Phase-2 file, and count it so the claim is checkable.
                qty_ok = True
                self.counters["n_animal_place_qty_carveout"] += 1
            if not (verb_ok and item_ok and qty_ok):
                self.counters["n_fallback_pass"] += 1
                action = ["PASS"]
        if action[0] == "PASS":
            self.counters["n_unit_pass"] += 1
        ts.apply_unit(u, action)
        return action

    @staticmethod
    def _is_animal_place(ts, u, action):
        """Is this `PLACE` the tile branch of `kag.py:381-392` rather than a shed deposit?"""
        if action[0] != "PLACE" or len(action) < 2 or action[1] not in kag.ANIMALS:
            return False
        pos = ts.positions[u]
        tile = ts.tiles[M.tile_index(int(pos[0]), int(pos[1]))]
        return (isinstance(tile, dict) and "animal" not in tile
                and tile.get("kind") == kag.ANIMALS[action[1]]["structure"]
                and ts.inventories[u].get(action[1], 0) >= 1)

    # -- market --------------------------------------------------------------------------

    def _decode_market(self, ts, ctx, ar, g):
        torch, dec, c = self.torch, self.dec, self.counters
        scal = torch.cat([g[:, N.G_SEEDS], g[:, N.G_MONEY:N.G_MONEY + 1],
                          g[:, N.G_SHED_FREE:N.G_SHED_FREE + 1]], dim=-1)
        mar = dec.market_init(ctx, ar)
        orders = []

        for k in range(V.MAX_MARKET_ORDERS):
            h = dec.market_h(ctx, mar, scal, k)
            if int(torch.argmax(dec.stop_logits(h), dim=-1).item()) == 1:
                c["n_market_stop"] += 1
                break
            mop, mitem, mqty = M.market_masks(ts)
            if not mop.any():
                c["n_market_no_legal_op"] += 1
                break
            op = int(torch.argmax(dec.op_logits(h, dec.b(mop)), dim=-1).item())
            op_e = self.model.op_emb(dec.ids(op))

            im = mitem[op]
            if not im.any():
                c["n_market_no_legal_op"] += 1
                break
            item = int(torch.argmax(dec.mitem_logits(h, op_e, dec.b(im)), dim=-1).item())
            item_e = self.model.item_emb(dec.ids(item))

            qm = drop_qty_all(mqty[op, item], allow_all=(op == V.M_SELL))
            if not qm.any():
                qm[0] = True
            qty = int(torch.argmax(dec.mqty_logits(h, op_e, item_e, dec.b(qm)), dim=-1).item())

            avail = _market_available(ts, op, item)
            order = V.decode_market_order(op, item, qty, avail)
            op_ok, item_ok = M.market_order_legality(ts, order)
            if not (op_ok and item_ok):
                c["n_order_fallback"] += 1
                break
            orders.append(order)
            c["n_orders_emitted"] += 1
            ts.apply_market(order)
            mar = dec.market_ar_step(mar, op_e, item_e, dec.ones1())

        return orders

    # -- reporting -----------------------------------------------------------------------

    def timing(self):
        if not self.turn_times:
            return {"turns": 0}
        t = np.asarray(self.turn_times)
        return {
            "turns": int(t.size),
            "mean_ms": float(t.mean() * 1e3),
            "p99_ms": float(np.percentile(t, 99) * 1e3),
            "max_ms": float(t.max() * 1e3),
        }

    def report(self):
        return {**self.counters, "timing": self.timing()}


def load_model(checkpoint_path, device="cpu", threads=1):
    """Load once.  Returns `(model, payload, torch)`; the model is in eval mode and frozen."""
    import torch                                                          # noqa: PLC0415

    if threads:
        torch.set_num_threads(int(threads))
    model, payload = C.load(checkpoint_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload, torch


def make_agent(checkpoint_path, device="cpu", strict=True, threads=1, model=None, torch_mod=None,
               claims="tile"):
    """`checkpoint -> agent(obs) -> action`.  A fresh, stateful agent per call.

    Pass `model`/`torch_mod` to reuse an already-loaded checkpoint across games without paying the
    load twice; the returned agent still owns its own per-episode state.
    """
    if model is None:
        model, _payload, torch_mod = load_model(checkpoint_path, device=device, threads=threads)
    elif torch_mod is None:
        import torch as torch_mod                                         # noqa: PLC0415
    return BCAgent(model, torch_mod, device=device, strict=strict, claims=claims)
