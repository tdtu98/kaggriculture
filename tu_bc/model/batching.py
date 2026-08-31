"""Shards -> fixed-shape, state-major training batches.

The shards are state-major and label-minor (`dataset.py`): the 45-number tile tokens are stored
once per state, and every macro row carries a `state_idx` pointing back into them.  There are ~5
macro decisions per state, so a decision-major batch would re-encode each state five times.  This
module keeps the state-major layout all the way into the model instead: a batch is **B states**,
and every head is evaluated on all 16 worker slots and all 10 market slots of each one, with
`IGNORE` in the label wherever no decision was taken.

That layout is not only cheaper.  It is what the autoregressive decode needs anyway -- slot `u` is
told what slots `0..u-1` committed to (PLAN_BC Ch5, the planting cliff at `kag.py:920-933`), so the
16 slots of one state have to travel together.

**Masks are built here, as data.**  `net.py` never derives a mask; it consumes the tensors this
module produces, so the same input slots carry `masks.compute_masks` output at inference time.
What we can build exactly from a shard alone is the *vocabulary* legality -- which items a verb may
name and whether it takes a quantity -- and that is what ships.  See `MASK_NOTE` below for the one
place this falls short of PLAN_BC Ch5 and why the shortfall is deliberate.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from . import dataset as D
from . import vocab as V

IGNORE = -1

# Chapter 5's value target: the terminal margin, normalised and clipped.  Terminal outcome only --
# all six Orbit Wars top-10 finishers measured that shaped intermediate rewards hurt.
VALUE_SCALE = 50000.0

MASK_NOTE = """\
The macro VERB head and the market OP head are fed all-ones masks during training.

Their exact masks are state-dependent (`masks.verb_mask_at`, `masks.market_masks`) and the shards
do not carry them: `build_shards.py` stores the observation tokens and the labels, not the
per-decision mask arrays.  Reconstructing verb legality from the stored tile tokens would be a
second implementation of a rule that already exists once, which is the exact shape of E39.  And the
only version of it that is cheap -- reading the tile token at the step the macro action *fires*
(`state_idx + seg_len`) -- is a future the model does not have at decision time, so training under
it would be train/test skew.

The item and quantity masks below are exact, and both heads take them.  The verb and op mask
tensors exist as inputs so that Chapter 6's agent wrapper passes `masks.compute_masks` output into
the same slots with no change to the model."""


def _dense_macro(rows, n_states):
    """Macro label rows -> `[n_states, MAX_UNITS]` dense label planes."""
    shape = (n_states, V.MAX_UNITS)
    commit = np.full(shape, IGNORE, dtype=np.int8)
    tile = np.full(shape, IGNORE, dtype=np.int16)
    verb = np.full(shape, IGNORE, dtype=np.int8)
    item = np.full(shape, IGNORE, dtype=np.int8)
    qty = np.full(shape, IGNORE, dtype=np.int8)
    if rows.shape[0] == 0:
        return commit, tile, verb, item, qty
    keep = rows[:, 1] < V.MAX_UNITS
    rows = rows[keep]
    s, u = rows[:, 0], rows[:, 1]
    idle = rows[:, 9] == 1
    commit[s, u] = np.where(idle, 0, 1)
    verb[s, u] = rows[:, 3]
    item[s, u] = rows[:, 4]
    qty[s, u] = rows[:, 5]
    # PLAN_BC Ch5's loss reads `CE(tile | ACT)`: an IDLE slot pointed at nothing, and the tile the
    # segmenter recorded for it is just where the unit already stood.  Training the pointer on that
    # teaches "predict your own square" on 6% of rows and inflates the head.  The verb head keeps
    # its IDLE rows, where the label is a genuine `PASS` -- and keeping them is also what makes the
    # reported accuracy comparable to the recomputed 33.8% WATER floor, which counts every row.
    act = ~idle
    tile[s[act], u[act]] = rows[act, 2]
    return commit, tile, verb, item, qty


def _dense_market(rows, n_states):
    """Market order rows -> `[n_states, MAX_MARKET_ORDERS]` dense planes, plus the STOP plane.

    Slot `k` is EMIT for `k < n_orders`, STOP at `k == n_orders`, and ignored past that: a turn
    that issued three orders teaches "three, then stop", not "three, then seven ignored slots".
    """
    shape = (n_states, V.MAX_MARKET_ORDERS)
    stop = np.full(shape, IGNORE, dtype=np.int8)
    op = np.full(shape, IGNORE, dtype=np.int8)
    item = np.full(shape, IGNORE, dtype=np.int8)
    qty = np.full(shape, IGNORE, dtype=np.int8)
    n_orders = np.zeros(n_states, dtype=np.int64)
    if rows.shape[0]:
        s, k = rows[:, 0], rows[:, 1]
        op[s, k] = rows[:, 2]
        item[s, k] = rows[:, 3]
        qty[s, k] = rows[:, 4]
        stop[s, k] = 0
        np.add.at(n_orders, s, 1)
    has_stop = n_orders < V.MAX_MARKET_ORDERS
    stop[np.nonzero(has_stop)[0], n_orders[has_stop]] = 1
    return stop, op, item, qty


class Corpus:
    """Every shard of one split, held in memory as fp16 token planes and dense label planes.

    The whole training split is 19 MB on disk and ~0.5 GB expanded, so there is no streaming
    machinery here on purpose -- an epoch is a permutation of `n_states` indices.
    """

    STATE_KEYS = ("tiles", "workers", "worker_mask", "products", "global", "opponent")
    LABEL_KEYS = ("commit", "tile", "verb", "item", "qty",
                  "m_stop", "m_op", "m_item", "m_qty")

    def __init__(self, split, root=D.SHARD_ROOT, limit_episodes=None, paths=None):
        self.split = split
        paths = list(paths) if paths is not None else D.find_shards(split, root)
        if limit_episodes is not None:
            paths = paths[:limit_episodes]
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} under {root!r}")
        self.paths = paths

        states = {k: [] for k in self.STATE_KEYS}
        labels = {k: [] for k in self.LABEL_KEYS}
        step, value, episode_ids = [], [], []
        for path in paths:
            sh = D.Shard(path)
            n = sh.n_states
            for k in self.STATE_KEYS:
                states[k].append(sh[k])
            c, t, v, i, q = _dense_macro(sh.labels("macro"), n)
            ms, mo, mi, mq = _dense_market(sh.labels("market"), n)
            for k, arr in zip(self.LABEL_KEYS, (c, t, v, i, q, ms, mo, mi, mq)):
                labels[k].append(arr)
            step.append(np.arange(n, dtype=np.int16))
            margin = (float(sh["reward"]) - float(sh["opponent_reward"])) / VALUE_SCALE
            value.append(np.full(n, np.clip(margin, -1.0, 1.0), dtype=np.float32))
            episode_ids.append(sh.episode_id)

        self.states = {k: np.concatenate(v) for k, v in states.items()}
        self.labels = {k: np.concatenate(v) for k, v in labels.items()}
        self.step = np.concatenate(step)
        self.value = np.concatenate(value)
        self.episode_ids = episode_ids
        self.n_states = int(self.step.shape[0])
        self.t_tile = self._teacher_tile()

    def _teacher_tile(self):
        """The pointer target fed back into the autoregressive state, always in range.

        The tile *label* is `IGNORE` for an IDLE slot and for a slot that took no decision, but a
        gather cannot fault on a sentinel, and conditioning the verb head on tile 0 for every IDLE
        row would be a lie.  An IDLE unit's target is the square it is standing on, recovered
        exactly from the worker token's `x / (GRID - 1)`, `y / (GRID - 1)`.
        """
        tile = self.labels["tile"].astype(np.int64)
        w = self.states["workers"].astype(np.float32)
        here = (np.rint(w[..., 3] * (V.GRID - 1)) * V.GRID
                + np.rint(w[..., 2] * (V.GRID - 1))).astype(np.int64)
        return np.where(tile < 0, np.clip(here, 0, V.N_TILES - 1), tile).astype(np.int16)

    def __len__(self):
        return self.n_states

    def manifest_hash(self):
        """Episode ids plus feature version -- stamped into every checkpoint (PLAN_BC risk 9)."""
        h = hashlib.sha256()
        h.update(f"{self.split}|{D.MODULE_VERSION}|".encode())
        for eid in self.episode_ids:
            h.update(f"{eid},".encode())
        return h.hexdigest()[:16]

    def counts(self):
        lab = self.labels
        return {
            "states": self.n_states,
            "episodes": len(self.paths),
            "macro_decisions": int((lab["commit"] != IGNORE).sum()),
            "macro_act": int((lab["commit"] == 1).sum()),
            "market_orders": int((lab["m_op"] != IGNORE).sum()),
        }


# --------------------------------------------------------------------------------------
# Masks, as data
# --------------------------------------------------------------------------------------

def _clamped(labels):
    return np.maximum(labels, 0)


def macro_masks(verb_labels, item_labels):
    """`(item_mask, qty_mask)` for the macro heads, gathered from the vocabulary alphabets.

    Exact by construction: `MACRO_VERB_ITEM_ALPHABET` is built from `kag.py`'s own dispatch, so it
    can never reject an expert label that `vocab.encode_unit_action` accepted.  `assert_no_expert_
    rejected` proves that on the corpus rather than asserting it in prose.
    """
    item_mask = V.MACRO_VERB_ITEM_ALPHABET[_clamped(verb_labels)]
    takes_qty = np.zeros(V.N_MACRO_VERBS, dtype=bool)
    for name in ("PICKUP", "PLACE"):
        takes_qty[V.MACRO_VERB_INDEX[name]] = True
    qty_mask = np.zeros(item_labels.shape + (V.N_QTY,), dtype=bool)
    qty_mask[..., 0] = True                      # bucket 0 is the "no quantity" slot
    qty_mask = np.where(takes_qty[_clamped(verb_labels)][..., None],
                        np.ones_like(qty_mask), qty_mask)
    return item_mask, qty_mask


def market_masks(op_labels):
    """`(item_mask, qty_mask)` for the market heads, from `MARKET_OP_ITEM_ALPHABET`."""
    item_mask = V.MARKET_OP_ITEM_ALPHABET[_clamped(op_labels)]
    takes_qty = np.array([V.MARKET_OP_ARITY[o][0] == 2 for o in V.MARKET_OPS])
    qty_mask = np.zeros(op_labels.shape + (V.N_QTY,), dtype=bool)
    qty_mask[..., 0] = True
    qty_mask = np.where(takes_qty[_clamped(op_labels)][..., None],
                        np.ones_like(qty_mask), qty_mask)
    return item_mask, qty_mask


def build_masks(labels):
    """Every mask plane for one batch of dense labels.  Keys line up with `net.forward`'s batch."""
    m_item, m_qty = macro_masks(labels["verb"], labels["item"])
    mm_item, mm_qty = market_masks(labels["m_op"])
    shape = labels["commit"].shape
    return {
        # All-ones: see MASK_NOTE.  The slot exists so inference can pass the real thing.
        "m_verb": np.ones(shape + (V.N_MACRO_VERBS,), dtype=bool),
        "m_tile": np.ones(shape + (V.N_TILES,), dtype=bool),
        "m_item": m_item,
        "m_qty": m_qty,
        "m_mop": np.ones(labels["m_op"].shape + (V.N_MARKET_OPS,), dtype=bool),
        "m_mitem": mm_item,
        "m_mqty": mm_qty,
    }


def assert_no_expert_rejected(corpus):
    """PLAN_BC Assertion 3, restated for the masks this module actually feeds the model.

    A model trained under a leaky mask produces loss numbers that mean nothing at all, so this is
    the "prove the change fired" gate for training.  Returns the counter dict; every entry must be
    zero.
    """
    lab = corpus.labels
    masks = build_masks(lab)
    out = {}
    for label_key, mask_key in (("item", "m_item"), ("qty", "m_qty"),
                                ("m_item", "m_mitem"), ("m_qty", "m_mqty")):
        y = lab[label_key]
        take = y != IGNORE
        legal = np.take_along_axis(masks[mask_key], _clamped(y).astype(np.int64)[..., None],
                                   axis=-1)[..., 0]
        out[f"rejected_{label_key}"] = int((take & ~legal).sum())
    return out


# --------------------------------------------------------------------------------------
# Flat per-decision view, for the Chapter 4 baseline
# --------------------------------------------------------------------------------------

def flat_macro(corpus):
    """`(state, unit, tile, verb, step)` for every macro decision, IDLE rows included.

    An IDLE row has no pointer label, so its "target" is the square the unit is standing on --
    recovered exactly from the worker token, which stores `x / (GRID - 1)` and `y / (GRID - 1)`.
    Keeping those rows is what makes the baseline's score comparable to the recomputed majority
    floor, which counts every macro row.
    """
    lab = corpus.labels
    s, u = np.nonzero(lab["commit"] != IGNORE)
    tile = lab["tile"][s, u].astype(np.int64)
    w = corpus.states["workers"][s, u].astype(np.float32)
    here = (np.rint(w[:, 3] * (V.GRID - 1)) * V.GRID + np.rint(w[:, 2] * (V.GRID - 1)))
    tile = np.where(tile < 0, here.astype(np.int64), tile)
    return {
        "state": s.astype(np.int64),
        "unit": u.astype(np.int64),
        "tile": tile,
        "verb": lab["verb"][s, u].astype(np.int64),
        "step": corpus.step[s].astype(np.int64),
        "is_act": (lab["commit"][s, u] == 1),
    }


def default_root():
    return os.environ.get("BC_SHARD_ROOT", D.SHARD_ROOT)


# --------------------------------------------------------------------------------------
# The torch batch
# --------------------------------------------------------------------------------------

LABEL_TO_BATCH = {"commit": "y_commit", "tile": "y_tile", "verb": "y_verb", "item": "y_item",
                  "qty": "y_qty", "m_stop": "y_mstop", "m_op": "y_mop", "m_item": "y_mitem",
                  "m_qty": "y_mqty"}


def make_batch(corpus, idx, device="cpu", torch_mod=None):
    """`B` state indices -> the batch dict `net.BCNet.forward` consumes.

    Shapes are static by construction: every batch is `[B, 100, ...]`, `[B, 16, ...]`,
    `[B, 10, ...]` whatever the states in it happened to contain, with `IGNORE` marking the slots
    that took no decision.  Tokens are stored fp16 and widened here -- PLAN_BC Ch5 measured no gain
    from reduced precision on MPS and the reference code has a documented NaN history around it.
    """
    torch = torch_mod
    if torch is None:
        import torch as torch                                            # noqa: PLC0415
    dev = torch.device(device)
    idx = np.asarray(idx)

    def f32(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    def i64(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(dev)

    def boolean(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    lab = {k: v[idx] for k, v in corpus.labels.items()}
    masks = build_masks(lab)
    batch = {k: f32(corpus.states[k][idx]) for k in corpus.STATE_KEYS}
    batch.update({LABEL_TO_BATCH[k]: i64(v) for k, v in lab.items()})
    batch["y_value"] = f32(corpus.value[idx])
    batch["step"] = i64(corpus.step[idx])

    batch["t_tile"] = i64(corpus.t_tile[idx])
    batch["t_verb"] = i64(np.maximum(lab["verb"], 0))
    batch["t_item"] = i64(np.maximum(lab["item"], 0))
    batch["t_valid"] = f32(lab["commit"] != IGNORE)
    batch["t_mop"] = i64(np.maximum(lab["m_op"], 0))
    batch["t_mitem"] = i64(np.maximum(lab["m_item"], 0))
    batch["t_mvalid"] = f32(lab["m_op"] != IGNORE)

    for k, v in masks.items():
        batch[k] = boolean(v)
    return batch


def epoch_batches(corpus, batch_size, rng, device="cpu", drop_last=True, torch_mod=None):
    """A shuffled epoch.  The last ragged batch is dropped so shapes stay constant."""
    order = rng.permutation(corpus.n_states)
    n = corpus.n_states - (corpus.n_states % batch_size if drop_last else 0)
    for start in range(0, n, batch_size):
        yield make_batch(corpus, order[start:start + batch_size], device=device,
                         torch_mod=torch_mod)
