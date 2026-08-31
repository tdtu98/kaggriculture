"""Shard writer and a PyTorch-free loader.

One `.npz` per episode under `data/shards/{split}/{episode_id}.npz`, with `{split}` copied straight
from the corpus directory layout so our splits can never drift from the provided ones.

**State-major, label-minor** (PLAN_BC Ch3).  There are ~10.1-10.8 worker decisions per state, so a
decision-major layout would store every state about ten times.  Instead the states are stored once
and every label row carries a `state_idx` pointing back into them:

    states:  tiles[719, 100, 45]  workers[719, 16, 26]  worker_mask[719, 16]
             products[719, 9, 26]  global[719, 28]  opponent[719, 7]
    raw:     [n, 6]      state_idx, unit, verb, item, qty_bin, qty_raw
    market:  [n, 6]      state_idx, slot, op,   item, qty_bin, qty_raw
    macro:   [n, 11]     state_idx, unit, tile, verb, item, qty_bin, qty_raw,
                         seg_len, manhattan, is_idle, truncated

Every shard also stores `episode_id`, `split`, `seat`, `opponent`, rewards, `module_version`,
`config_hash`, `FEATURE_VERSION` and `MODULE_VERSION` -- so a future filter is a query rather than
a re-decode, and a feature/weight skew is caught on load rather than showing up as "BC didn't work"
(risk 9 in PLAN_BC's table).

Rejected, with reasons: Parquet/Arrow (an extra dependency, and the reference loader is
`.npz`-shaped so we would lose the reuse); an uncompressed JSON archive (31 MB per game, no
benefit).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from . import features as F
from . import vocab as V

MODULE_VERSION = 1          # bump when the shard LAYOUT changes; FEATURE_VERSION covers content
SHARD_ROOT = os.path.join("data", "shards")

RAW_COLS = ("state_idx", "unit", "verb", "item", "qty_bin", "qty_raw")
MARKET_COLS = ("state_idx", "slot", "op", "item", "qty_bin", "qty_raw")
MACRO_COLS = ("state_idx", "unit", "tile", "verb", "item", "qty_bin", "qty_raw",
              "seg_len", "manhattan", "is_idle", "truncated")


def shard_path(split, episode_id, root=SHARD_ROOT):
    return os.path.join(root, split, f"{episode_id}.npz")


def build_states(decoded):
    """Run the extractor over an episode's states.  Returns the five stacked arrays.

    `prev_obs` is threaded through so the opponent token can carry yesterday's money change, and
    `effective_shed` is deliberately NOT passed here: the stored shard holds the observation the
    agent was handed, and the market heads recompute effective shed from the running turn state at
    decode/inference time (PLAN_BC Ch3 -- never derive a feature from a request).
    """
    n = len(decoded.states)
    out = {k: np.zeros((n,) + shape, dtype=F.STORE_DTYPE)
           for k, shape in F.FEATURE_SHAPES.items()}
    prev = None
    for s, obs in enumerate(decoded.states):
        feats = F.extract(obs, decoded.seat, prev_obs=prev)
        if s == 0:
            F.check_shapes(feats)
        for k in out:
            out[k][s] = feats[k]
        prev = obs
    return out


def write_shard(decoded, states=None, root=SHARD_ROOT):
    """Write one episode's shard.  Returns the path and its size in bytes."""
    states = build_states(decoded) if states is None else states
    path = shard_path(decoded.split, decoded.episode_id, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def table(rows, width):
        if not rows:
            return np.zeros((0, width), dtype=np.int32)
        return np.asarray(rows, dtype=np.int32)

    np.savez_compressed(
        path,
        raw=table(decoded.raw, len(RAW_COLS)),
        market=table(decoded.market, len(MARKET_COLS)),
        macro=table(decoded.macro, len(MACRO_COLS)),
        episode_id=np.asarray(decoded.episode_id),
        split=np.asarray(decoded.split),
        seat=np.asarray(decoded.seat, dtype=np.int32),
        # `opponent_name`, not `opponent`: the latter is the opponent-summary FEATURE array.
        opponent_name=np.asarray(decoded.opponent or ""),
        reward=np.asarray(decoded.reward if decoded.reward is not None else np.nan,
                          dtype=np.float64),
        opponent_reward=np.asarray(
            decoded.opponent_reward if decoded.opponent_reward is not None else np.nan,
            dtype=np.float64),
        module_version=np.asarray(decoded.module_version or ""),
        config_hash=np.asarray(decoded.config_hash or ""),
        seed=np.asarray(decoded.seed if decoded.seed is not None else -1, dtype=np.int64),
        feature_version=np.asarray(F.FEATURE_VERSION, dtype=np.int32),
        shard_module_version=np.asarray(MODULE_VERSION, dtype=np.int32),
        **states,
    )
    return path, os.path.getsize(path)


class Shard:
    """A loaded shard.  numpy only -- nothing here imports torch."""

    def __init__(self, path):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        fv = int(self.data["feature_version"])
        mv = int(self.data["shard_module_version"])
        if fv != F.FEATURE_VERSION:
            raise ValueError(f"{path}: FEATURE_VERSION {fv} != {F.FEATURE_VERSION} -- rebuild the "
                             f"shards, or you are about to train weights against features they "
                             f"were not built for")
        if mv != MODULE_VERSION:
            raise ValueError(f"{path}: shard layout version {mv} != {MODULE_VERSION}")

    def __getitem__(self, key):
        return self.data[key]

    @property
    def episode_id(self):
        return str(self.data["episode_id"])

    @property
    def n_states(self):
        return int(self.data["global"].shape[0])

    def states(self, idx=None):
        keys = tuple(F.FEATURE_SHAPES)
        if idx is None:
            return {k: self.data[k] for k in keys}
        return {k: self.data[k][idx] for k in keys}

    def labels(self, kind="macro"):
        return self.data[kind]


def find_shards(split, root=SHARD_ROOT):
    return sorted(glob.glob(os.path.join(root, split, "*.npz")))


def iter_shards(split, root=SHARD_ROOT, shuffle=False, seed=0):
    paths = find_shards(split, root)
    if shuffle:
        np.random.default_rng(seed).shuffle(paths)
    for p in paths:
        yield Shard(p)


def split_counts(root=SHARD_ROOT, splits=("train", "val", "test")):
    out = {}
    for split in splits:
        n_states = n_raw = n_macro = n_market = 0
        for path in find_shards(split, root):
            with np.load(path, allow_pickle=False) as z:
                n_states += int(z["global"].shape[0])
                n_raw += int(z["raw"].shape[0])
                n_macro += int(z["macro"].shape[0])
                n_market += int(z["market"].shape[0])
        out[split] = {"episodes": len(find_shards(split, root)), "states": n_states,
                      "raw": n_raw, "macro": n_macro, "market": n_market}
    return out


def majority_floor(labels, col, keep=None):
    """The score a model that has learned nothing already gets.

    PLAN_BC Ch4: "a head reporting 70% accuracy means something only when printed next to the
    floor is 19.3%".  Those two numbers came from the two seats of one old sample game; this
    recomputes the floor on Ryo's seat over whichever rows are passed in.
    """
    if labels.shape[0] == 0:
        return {"n": 0, "floor": float("nan"), "argmax": -1}
    col_vals = labels[:, col] if keep is None else labels[keep, col]
    if col_vals.size == 0:
        return {"n": 0, "floor": float("nan"), "argmax": -1}
    counts = np.bincount(col_vals.astype(np.int64))
    best = int(counts.argmax())
    return {"n": int(col_vals.size), "floor": float(counts[best] / col_vals.size), "argmax": best}


def wilson(p, n, z=1.96):
    """The error bar for a percentage.  Reporting a rate without one is how you convince yourself
    of something that is not there (PLAN_BC Ch4)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


assert len(RAW_COLS) == 6 and len(MARKET_COLS) == 6 and len(MACRO_COLS) == 11
assert V.N_QTY == 14
