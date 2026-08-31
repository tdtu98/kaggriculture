"""v1: the pointer-MLP behaviour-cloning model (PLAN_BC Chapter 5).

Three ideas, and nothing else:

1. **The state is a set of tokens** -- one per farm tile (100), one per worker slot (16, padded),
   one per market product (9), one global, one opponent summary.  Orbit Wars made one per planet.
2. **The action is several small choices, called heads** -- commit, then which tile, then which
   verb, then which item, then how many -- instead of one giant "what is the whole action".
3. **Illegal choices are masked** before the pick, as an additive bias, so the model never spends
   capacity learning not to do the impossible.

The tile choice is a **pointer**: rather than 100 fixed output slots, one query vector is scored
against all 100 tile embeddings (`logits = tiles @ q / sqrt(d)`).

The 16 worker slots are decoded **autoregressively** -- slot `u` is told what slots `0..u-1`
committed to.  That is a mechanical requirement, not an aesthetic one: if the turn's total `PLANT`
requests for one crop exceed the seeds held, *every one of them is dropped* (`kag.py:920-933`), and
the expert sits right on that cliff (66 turns where demand equals the seed count exactly).  So the
running state carries a live seed budget, decremented the moment a slot commits to a `PLANT`, plus
a learned projection of the (tile, verb) each earlier slot chose.  The market's 10 slots decode the
same way, after the workers, because that is the order the environment settles them in
(`_process_market` at `kag.py:941`, after the unit loop at `:935-941`).

**Chapter 9's six portability rules are followed here, because they are free now and a rewrite
later.**  The forward pass reads nothing but `(self.parameters(), batch)`; masks are additive data
and never an `if`; every shape is static (100 tiles, 16 workers, 10 orders); there is no in-place
mutation; the RNG is an explicit `torch.Generator`; and both decode loops are bounded Python
`range`s over constants, which map mechanically onto `lax.scan`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import nn

from . import features as F
from . import vocab as V

# A finite floor, not `-inf`.  A padded slot has an all-False mask, and `softmax` of an all-`-inf`
# row is NaN -- which poisons every gradient in the batch.  The reference implementation had to
# filter `|log_prob| > 1e5` for exactly this reason; a finite floor removes the need.
MASK_BIAS = -1e9

# Item ids of the five crops, for the running seed budget.  Read from the vocabulary rather than
# assumed to be 0..4 -- `PRODUCTS` is the environment's own tuple and its order is not ours to fix.
CROP_ITEM_IDS = tuple(V.ITEM_INDEX[c] for c in V.CROPS)

# Indices into the global token (`features.global_token`).
G_MONEY = 7
G_SHED_FREE = 11
G_SEEDS = slice(12, 12 + len(V.CROPS))
N_SCALARS = len(V.CROPS) + 2


@dataclass(frozen=True)
class NetConfig:
    d_model: int = 128
    d_embed: int = 32          # verb / item / op embedding width
    d_ar: int = 128
    # Gate the autoregressive state on slot validity.  See `_ar_step`.  `False` reproduces the
    # first (defective) v1 forward and exists only so pre-fix checkpoints still replay exactly;
    # `checkpoint.load` supplies it for any checkpoint saved before the field existed.
    ar_gate: bool = True

    def as_dict(self):
        return asdict(self)


DEFAULT_CFG = NetConfig()


def masked_logits(logits, mask):
    """Additive `-1e9` on every illegal option.  Data, never control flow (Chapter 9, rule 2)."""
    return logits + (~mask) * MASK_BIAS


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class _Encoder(nn.Module):
    """Two layers and a norm.  Weights are shared across every token of a type."""

    def __init__(self, d_in, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.body(x))


class BCNet(nn.Module):
    """`forward(batch) -> logits`.  See the module docstring for the batch contract."""

    def __init__(self, cfg=DEFAULT_CFG, generator=None):
        super().__init__()
        self.cfg = cfg
        d, de, da = cfg.d_model, cfg.d_embed, cfg.d_ar

        self.tile_enc = _Encoder(F.N_TILE_FEATS, d)
        self.worker_enc = _Encoder(F.N_WORKER_FEATS, d)
        self.product_enc = _Encoder(F.N_PRODUCT_FEATS, d)
        self.global_enc = _Encoder(F.N_GLOBAL_FEATS, d)
        self.opponent_enc = _Encoder(F.N_OPPONENT_FEATS, d)
        # A learned 100-entry position embedding, not Fourier features: Orbit Wars needed Fourier
        # for a continuous 100x100 space, ours is a discrete 10x10 grid where a lookup is exact.
        self.tile_pos = nn.Embedding(V.N_TILES, d)
        self.context = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.LayerNorm(d))

        self.verb_emb = nn.Embedding(V.N_MACRO_VERBS, de)
        self.item_emb = nn.Embedding(V.N_ITEM_SLOTS, de)
        self.op_emb = nn.Embedding(V.N_MARKET_OPS, de)

        slot_in = d + d + da + N_SCALARS
        self.slot_trunk = nn.Sequential(nn.Linear(slot_in, d), nn.ReLU())
        self.commit_head = nn.Linear(d, 2)
        self.tile_query = nn.Linear(d, d)
        self.verb_head = _mlp(d + d, d, V.N_MACRO_VERBS)
        self.item_head = _mlp(d + d + de, d, V.N_ITEM_SLOTS)
        self.qty_head = _mlp(d + d + de + de, d, V.N_QTY)
        self.ar_update = nn.Linear(da + d + de + N_SCALARS + 1, da)
        self.ar_norm = nn.LayerNorm(da)

        self.market_init = nn.Linear(d + da, da)
        mkt_in = d + da + N_SCALARS + V.MAX_MARKET_ORDERS
        self.market_trunk = nn.Sequential(nn.Linear(mkt_in, d), nn.ReLU())
        self.stop_head = nn.Linear(d, 2)
        self.op_head = nn.Linear(d, V.N_MARKET_OPS)
        self.mitem_head = nn.Linear(d + de, V.N_ITEM_SLOTS)
        self.mqty_head = nn.Linear(d + de + de, V.N_QTY)
        self.market_ar_update = nn.Linear(da + de + de + 1, da)
        self.market_ar_norm = nn.LayerNorm(da)

        # Terminal outcome only, no shaping: six of six Orbit Wars top-10 finishers measured that
        # hand-crafted intermediate rewards hurt.
        self.value_head = _mlp(d, d, 1)

        self.register_buffer("crop_item_ids", torch.tensor(CROP_ITEM_IDS, dtype=torch.long),
                             persistent=False)
        self.register_buffer("slot_onehot", torch.eye(V.MAX_MARKET_ORDERS), persistent=False)
        self.reset_parameters(generator)

    def reset_parameters(self, generator=None):
        """Explicit RNG (Chapter 9, rule 5).  `generator=None` uses torch's global stream."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                bound = 1.0 / math.sqrt(module.weight.shape[-1])
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound, generator=generator)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.zero_()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _ar_step(self, norm, state, update, valid):
        """Advance one autoregressive state, skipping slots that took no decision.

        **The bug this exists to fix, because it is invisible in a loss curve.**  `state` starts at
        zeros, and most of the 16 worker slots hold no decision at any given turn.  The obvious
        update -- `norm(state + valid * tanh(update))` -- feeds LayerNorm an all-zero vector on
        every leading invalid slot, and LayerNorm at zero variance divides by `sqrt(eps)`, so it
        multiplies gradients by ~316.  Sixteen slots in a row compounds that to ~1e34, the squared
        sum overflows fp32, `clip_grad_norm_` returns `inf`, and the clip coefficient
        `max_norm / inf` is **zero** -- so the whole batch's gradients are silently thrown away.
        Training still runs and the loss still falls, on whichever batches happened to escape.

        Gating the *whole* update on `valid` cuts the amplifying path exactly where it is
        degenerate: with `valid == 0` no gradient reaches the normalisation at all, and the state
        is carried through unchanged.  Still data, not control flow -- `valid` is a tensor.
        """
        if not self.cfg.ar_gate:
            return norm(state + valid * torch.tanh(update))
        moved = norm(state + torch.tanh(update))
        return valid * moved + (1.0 - valid) * state

    # -- encoding ------------------------------------------------------------------------

    def encode(self, batch):
        """State tokens -> `(tile_embeddings [B,100,d], context [B,d])`."""
        tiles = self.tile_enc(batch["tiles"]) + self.tile_pos.weight.unsqueeze(0)
        workers = self.worker_enc(batch["workers"])
        products = self.product_enc(batch["products"])
        wmask = batch["worker_mask"].unsqueeze(-1)
        worker_mean = (workers * wmask).sum(1) / wmask.sum(1).clamp_min(1.0)
        ctx = self.context(torch.cat([
            self.global_enc(batch["global"]),
            tiles.mean(1),
            worker_mean,
            products.mean(1),
            self.opponent_enc(batch["opponent"]),
        ], dim=-1))
        return tiles, workers, ctx

    # -- the two bounded decode loops ----------------------------------------------------

    def _unit_slots(self, batch, tiles, workers, ctx):
        b = ctx.shape[0]
        d_ar = self.cfg.d_ar
        g = batch["global"]
        seeds = g[:, G_SEEDS] * 20.0                    # undo the feature normalisation
        money = g[:, G_MONEY:G_MONEY + 1]
        shed_free = g[:, G_SHED_FREE:G_SHED_FREE + 1]
        ar = torch.zeros(b, d_ar, device=ctx.device, dtype=ctx.dtype)
        scale = 1.0 / math.sqrt(self.cfg.d_model)
        rows = torch.arange(b, device=ctx.device)

        out = {k: [] for k in ("commit", "tile", "verb", "item", "qty")}
        for u in range(V.MAX_UNITS):
            scal = torch.cat([seeds / 20.0, money, shed_free], dim=-1)
            h = self.slot_trunk(torch.cat([workers[:, u], ctx, ar, scal], dim=-1))
            out["commit"].append(self.commit_head(h))

            q = self.tile_query(h)
            tile_logits = torch.einsum("btd,bd->bt", tiles, q) * scale
            out["tile"].append(masked_logits(tile_logits, batch["m_tile"][:, u]))

            # Teacher forcing: the expert's own earlier picks drive the running state, so training
            # conditions on states *the expert* reached (PLAN_BC Ch6 -- this is exactly the
            # exposure bias that Chapter 6 then measures).
            tile_e = tiles[rows, batch["t_tile"][:, u]]
            out["verb"].append(masked_logits(
                self.verb_head(torch.cat([h, tile_e], dim=-1)), batch["m_verb"][:, u]))

            t_verb = batch["t_verb"][:, u]
            verb_e = self.verb_emb(t_verb)
            out["item"].append(masked_logits(
                self.item_head(torch.cat([h, tile_e, verb_e], dim=-1)), batch["m_item"][:, u]))

            t_item = batch["t_item"][:, u]
            item_e = self.item_emb(t_item)
            out["qty"].append(masked_logits(
                self.qty_head(torch.cat([h, tile_e, verb_e, item_e], dim=-1)),
                batch["m_qty"][:, u]))

            valid = batch["t_valid"][:, u].unsqueeze(-1)
            ar = self._ar_step(self.ar_norm, ar,
                               self.ar_update(torch.cat([ar, tile_e, verb_e, scal, valid], -1)),
                               valid)

            # The planting cliff, as arithmetic rather than as a branch: a slot that commits to
            # PLANT spends one seed of that crop, so the next slot cannot push the turn's demand
            # past the stock.
            planting = ((t_verb == V.MACRO_VERB_INDEX["PLANT"])
                        .to(ctx.dtype).unsqueeze(-1) * valid)
            spent = (t_item.unsqueeze(-1) == self.crop_item_ids).to(ctx.dtype) * planting
            seeds = (seeds - spent).clamp_min(0.0)

        return {k: torch.stack(v, dim=1) for k, v in out.items()}, ar

    def _market_slots(self, batch, ctx, ar):
        b = ctx.shape[0]
        g = batch["global"]
        scal = torch.cat([g[:, G_SEEDS], g[:, G_MONEY:G_MONEY + 1],
                          g[:, G_SHED_FREE:G_SHED_FREE + 1]], dim=-1)
        mar = torch.tanh(self.market_init(torch.cat([ctx, ar], dim=-1)))

        out = {k: [] for k in ("m_stop", "m_op", "m_item", "m_qty")}
        for k in range(V.MAX_MARKET_ORDERS):
            slot = self.slot_onehot[k].unsqueeze(0).expand(b, -1)
            h = self.market_trunk(torch.cat([ctx, mar, scal, slot], dim=-1))
            out["m_stop"].append(self.stop_head(h))
            out["m_op"].append(masked_logits(self.op_head(h), batch["m_mop"][:, k]))

            t_op = batch["t_mop"][:, k]
            op_e = self.op_emb(t_op)
            out["m_item"].append(masked_logits(
                self.mitem_head(torch.cat([h, op_e], dim=-1)), batch["m_mitem"][:, k]))

            t_item = batch["t_mitem"][:, k]
            item_e = self.item_emb(t_item)
            out["m_qty"].append(masked_logits(
                self.mqty_head(torch.cat([h, op_e, item_e], dim=-1)), batch["m_mqty"][:, k]))

            valid = batch["t_mvalid"][:, k].unsqueeze(-1)
            mar = self._ar_step(self.market_ar_norm, mar,
                                self.market_ar_update(torch.cat([mar, op_e, item_e, valid], -1)),
                                valid)

        return {k: torch.stack(v, dim=1) for k, v in out.items()}

    def forward(self, batch):
        tiles, workers, ctx = self.encode(batch)
        unit_logits, ar = self._unit_slots(batch, tiles, workers, ctx)
        market_logits = self._market_slots(batch, ctx, ar)
        out = dict(unit_logits)
        out.update(market_logits)
        out["value"] = self.value_head(ctx).squeeze(-1)
        return out


HEADS = ("commit", "tile", "verb", "item", "qty", "m_stop", "m_op", "m_item", "m_qty")

# Which label plane and which mask plane each head reads, and how wide it is.
HEAD_SPEC = {
    "commit": ("y_commit", None, 2),
    "tile": ("y_tile", "m_tile", V.N_TILES),
    "verb": ("y_verb", "m_verb", V.N_MACRO_VERBS),
    "item": ("y_item", "m_item", V.N_ITEM_SLOTS),
    "qty": ("y_qty", "m_qty", V.N_QTY),
    "m_stop": ("y_mstop", None, 2),
    "m_op": ("y_mop", "m_mop", V.N_MARKET_OPS),
    "m_item": ("y_mitem", "m_mitem", V.N_ITEM_SLOTS),
    "m_qty": ("y_mqty", "m_mqty", V.N_QTY),
}

# Batch contract, in one place.  `y_*` are labels (`-1` = no decision, ignored by the loss);
# `t_*` are the teacher-forcing inputs, always in range so a gather can never fault; `m_*` are the
# masks.  `batching.make_batch` builds all three families.
BATCH_KEYS = {
    "state": ("tiles", "workers", "worker_mask", "products", "global", "opponent"),
    "labels": tuple(spec[0] for spec in HEAD_SPEC.values()) + ("y_value",),
    "teacher": ("t_tile", "t_verb", "t_item", "t_valid", "t_mop", "t_mitem", "t_mvalid"),
    "masks": ("m_tile", "m_verb", "m_item", "m_qty", "m_mop", "m_mitem", "m_mqty"),
}
