"""Masked cross-entropy per head, plus the value MSE (PLAN_BC Chapter 5).

    L = CE(commit) + CE(tile) + CE(verb) + CE(item) + CE(qty)
      + CE(mkt_stop) + CE(mkt_op) + CE(mkt_item) + CE(mkt_qty)
      + 0.1 * MSE(value, terminal_margin)

Every head is a classifier.  **No regression heads**: quantities here are small integers with a
very peaked distribution, so they go in buckets, and fitting a continuous value to a spiky integer
target is the loss the reference implementation's authors were least happy with.

Two rules that make the numbers mean anything:

* **`-1` labels are ignored.**  Most of the 16 worker slots and 10 market slots hold no decision at
  any given state, and averaging their zero loss into the mean would silently divide every head's
  loss by the padding factor.  Each head is normalised by *its own* count of real decisions.
* **Per-head losses are reported separately, never blended.**  The worker heads outnumber the
  market heads roughly five to one, so one summed number would be the worker heads wearing a hat.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn

from . import batching as B
from . import net as N

VALUE_COEF = 0.1


def masked_ce(logits, labels):
    """`(mean CE over real decisions, n_decisions)`.  Empty -> zero loss, not NaN."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    take = flat_labels != B.IGNORE
    n = take.sum()
    per = Fn.cross_entropy(flat_logits, flat_labels.clamp_min(0), reduction="none")
    return (per * take).sum() / n.clamp_min(1), n


def head_losses(logits, batch):
    """One scalar per head.  Keys are `net.HEADS`."""
    return {head: masked_ce(logits[head], batch[N.HEAD_SPEC[head][0]]) for head in N.HEADS}


def total_loss(logits, batch, value_coef=VALUE_COEF):
    """`(loss, parts)` -- `parts` carries every head separately, for the epoch log."""
    per_head = head_losses(logits, batch)
    loss = sum(v for v, _ in per_head.values())
    value_mse = Fn.mse_loss(logits["value"], batch["y_value"])
    loss = loss + value_coef * value_mse
    parts = {head: float(v.detach()) for head, (v, _) in per_head.items()}
    parts["value_mse"] = float(value_mse.detach())
    parts["total"] = float(loss.detach())
    parts["counts"] = {head: int(n) for head, (_, n) in per_head.items()}
    return loss, parts


@torch.no_grad()
def head_hits(logits, batch, topk=3, row_keep=None):
    """`{head: (top1_hits, topk_hits, n)}` -- the raw counts an accuracy table is built from.

    Counting hits rather than averaging rates is deliberate: an accuracy averaged per batch and
    then averaged again over batches is not the accuracy, and it cannot be handed to a Wilson
    interval.  `row_keep` is a `[B]` boolean selecting whole states -- that is how the "steps >= 32"
    view is taken, since Ryo's fixed ~30-step opening is nearly free to predict (E88).
    """
    out = {}
    for head in N.HEADS:
        labels = batch[N.HEAD_SPEC[head][0]]
        lg = logits[head]
        take = labels != B.IGNORE
        if row_keep is not None:
            take = take & row_keep.unsqueeze(-1)
        take = take.reshape(-1)
        lg = lg.reshape(-1, lg.shape[-1])
        k = min(topk, lg.shape[-1])
        top = lg.topk(k, dim=-1).indices
        safe = labels.reshape(-1).clamp_min(0).unsqueeze(-1)
        hit_k = ((top == safe).any(-1) & take).sum()
        hit_1 = ((top[:, :1] == safe).all(-1) & take).sum()
        out[head] = (int(hit_1), int(hit_k), int(take.sum()))
    return out


@torch.no_grad()
def class_recall(logits, batch, head, n_classes):
    """`(hits, totals)` per class for one head.

    PLAN_BC Ch5: print per-class recall for the rare market ops (`BUY_LAND` is 2 per player per
    game against `HIRE`'s 601) and intervene **only** if one is about zero -- and then with a
    three-feature rule, not loss weighting, which distorts calibration and is a common way to make
    a model look better per-class while playing worse.
    """
    labels = batch[N.HEAD_SPEC[head][0]].reshape(-1)
    pred = logits[head].reshape(-1, logits[head].shape[-1]).argmax(-1)
    take = labels != B.IGNORE
    hits = torch.zeros(n_classes, dtype=torch.long, device=labels.device)
    totals = torch.zeros(n_classes, dtype=torch.long, device=labels.device)
    safe = labels.clamp_min(0)
    totals = totals.index_add(0, safe, take.long())
    hits = hits.index_add(0, safe, (take & (pred == labels)).long())
    return hits.cpu().numpy(), totals.cpu().numpy()


@torch.no_grad()
def masked_probability_mass(logits, batch):
    """Total probability the model puts on options its mask forbade.  Must be exactly zero.

    This is the "prove the change fired" counter for masking (CLAUDE.md, E44).  A mask that is
    wired up but never applied looks identical in the loss curve to a mask that is applied.
    """
    worst = 0.0
    for head in N.HEADS:
        mask_key = N.HEAD_SPEC[head][1]
        if mask_key is None:
            continue
        mask = batch[mask_key]
        # A fully-masked row is a padded slot: softmax over all -1e9 is uniform, not illegal.
        live = mask.any(-1, keepdim=True)
        probs = logits[head].softmax(-1)
        worst = max(worst, float((probs * (~mask) * live).sum()))
    return worst
