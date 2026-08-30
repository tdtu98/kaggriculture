"""Behaviour-cloning training loop for the v1 pointer-MLP (PLAN_BC Chapter 5).

    PYTHONPATH=. .venv/bin/python -m model.train --epochs 3 --batch-size 128 --device mps

Two rules from Chapter 4 govern every number this prints, and they are the reason the table is as
wide as it is:

* **Every accuracy is printed next to its majority-class floor, with a Wilson interval.**  A head
  reporting 70% means something only when the floor is beside it.  "70% vs a 19% floor" computed on
  800 examples is the same mistake as the 16-game arena result this project has died on three
  times; validation here carries ~54,000 macro decisions and ~10,000 market orders.
* **Every number is reported twice** -- over all steps, and over steps >= 32 only.  Ryo plays a
  fixed ~30-step opening (E88) that is almost perfectly predictable from the step number.  A model
  that learned the opening and nothing else would still post a better-than-floor headline.

Teacher forcing: the autoregressive state is driven by the *expert's* earlier picks, which are in
the shards.  That is the construction Chapter 6 then has to pay for -- at training time we condition
on states the expert reached, at play time on states the model reached, and the gap between them is
exposure bias.  It is not a defect to fix here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import batching as B
from . import checkpoint as C
from . import dataset as D
from . import losses as L
from . import net as N
from . import vocab as V

# Which view drives model selection.  PLAN_BC Ch4: every metric is reported over all steps and
# over steps >= 32, and "the second number is the real one" -- Ryo's fixed ~30-step opening is
# nearly free to predict, so selecting on the all-steps number would let the opening cast a vote.
SELECT_VIEW = "ge32"

HEAD_LABEL = {
    "commit": "macro commit", "tile": "macro tile", "verb": "macro verb",
    "item": "macro item", "qty": "macro qty", "m_stop": "market stop",
    "m_op": "market op", "m_item": "market item", "m_qty": "market qty",
}


def pick_device(requested=None):
    """MPS when it is there, CPU when it is not.  Chapter 5: PyTorch on MPS, fp32, no bfloat16."""
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def majority_floors(corpus):
    """`{head: {"all": (floor, class), "ge32": (floor, class)}}`, recomputed on Ryo's seat.

    PLAN_BC Ch4 is emphatic that the old 16.3% / 19.3% sample figures do not transfer.  These are
    computed on the split being scored, over exactly the rows the loss is taken over.
    """
    out = {}
    ge32 = corpus.step >= V.OPENING_STEPS
    for head in N.HEADS:
        label_key = {v: k for k, v in B.LABEL_TO_BATCH.items()}[N.HEAD_SPEC[head][0]]
        y = corpus.labels[label_key]
        entry = {}
        for tag, rows in (("all", np.ones(corpus.n_states, bool)), ("ge32", ge32)):
            vals = y[rows]
            vals = vals[vals != B.IGNORE]
            if vals.size == 0:
                entry[tag] = (float("nan"), -1, 0)
                continue
            counts = np.bincount(vals.astype(np.int64))
            best = int(counts.argmax())
            entry[tag] = (float(counts[best] / vals.size), best, int(vals.size))
        out[head] = entry
    return out


@torch.no_grad()
def evaluate(model, corpus, batch_size, device, log=print, progress_every=0, indices=None):
    """Per-head top-1 / top-3 hit counts, both views, plus loss and the mask counter.

    `indices` scores a subset of the corpus's states -- that is how the train-side view is taken
    cheaply, on a fixed subsample (`train_eval_indices`) rather than all 50,330 training states.
    """
    model.eval()
    tally = {tag: {h: [0, 0, 0] for h in N.HEADS} for tag in ("all", "ge32")}
    loss_sum, n_batches = 0.0, 0
    parts_sum = {}
    recall_hits = np.zeros(V.N_MARKET_OPS, dtype=np.int64)
    recall_tot = np.zeros(V.N_MARKET_OPS, dtype=np.int64)
    worst_masked = 0.0
    t0 = time.time()
    order = np.arange(corpus.n_states) if indices is None else np.asarray(indices)
    for bi, start in enumerate(range(0, len(order), batch_size)):
        batch = B.make_batch(corpus, order[start:start + batch_size], device=device)
        logits = model(batch)
        loss, parts = L.total_loss(logits, batch)
        loss_sum += float(loss)
        n_batches += 1
        for k, v in parts.items():
            if isinstance(v, float):
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        worst_masked = max(worst_masked, L.masked_probability_mass(logits, batch))
        keep = batch["step"] >= V.OPENING_STEPS
        for tag, row_keep in (("all", None), ("ge32", keep)):
            for head, (h1, hk, n) in L.head_hits(logits, batch, row_keep=row_keep).items():
                t = tally[tag][head]
                t[0] += h1
                t[1] += hk
                t[2] += n
        h, t = L.class_recall(logits, batch, "m_op", V.N_MARKET_OPS)
        recall_hits += h
        recall_tot += t
        if progress_every and (bi + 1) % progress_every == 0:
            log(f"    eval batch {bi + 1}  ({time.time() - t0:.0f}s)")
    return {
        "tally": tally,
        "loss": loss_sum / max(1, n_batches),
        "parts": {k: v / max(1, n_batches) for k, v in parts_sum.items()},
        "masked_probability_mass": worst_masked,
        "op_recall": (recall_hits, recall_tot),
        "seconds": time.time() - t0,
    }


def accuracy(result, head, tag):
    h1, _hk, n = result["tally"][tag][head]
    return h1 / max(1, n), n


def mean_accuracy(result, tag=SELECT_VIEW):
    """The model-selection metric: the nine per-head top-1 accuracies, equally weighted.

    Equally weighted on purpose.  Weighting by decision count would let the macro heads -- 250,084
    decisions against the market's 47,090 -- decide when to stop on their own, and the market half
    is the half PLAN_BC says a learned model actually belongs in.
    """
    return float(np.mean([accuracy(result, h, tag)[0] for h in N.HEADS]))


def print_table(val, floors, train=None, log=print, train_n_states=None):
    """The per-head table.  `train` adds the train-side column so the gap is readable in place."""
    head_col = f"{'train top-1':>22}" if train else ""
    log(f"  {'head':<14} {'view':<9}{head_col} {'val top-1':>22} {'floor':>7} "
        f"{'val t3':>7} {'n(val)':>8}" + ("   gap" if train else ""))
    if train:
        log(f"  (train column is a fixed {train_n_states:,}-state subsample of the train split, "
            f"the same states every epoch)")
    for head in N.HEADS:
        for tag, view in (("all", "all steps"), ("ge32", "steps>=32")):
            acc, n = accuracy(val, head, tag)
            lo, hi = D.wilson(acc, n)
            floor, _cls, _ = floors[head][tag]
            verdict = "  BEATS" if lo > floor else ("  ties" if hi > floor else "  BELOW")
            cell, gap = "", ""
            if train:
                tacc, tn = accuracy(train, head, tag)
                tlo, thi = D.wilson(tacc, tn)
                cell = f"{tacc:7.2%} [{tlo:6.2%},{thi:6.2%}]"
                gap = f"  {tacc - acc:+.2%}"
            log(f"  {HEAD_LABEL[head]:<14} {view:<9}{cell} {acc:7.2%} [{lo:6.2%},{hi:6.2%}] "
                f"{floor:7.2%} {val['tally'][tag][head][1] / max(1, n):7.2%} {n:8,}"
                f"{gap}{verdict}")
    log(f"  masked-option probability mass: {val['masked_probability_mass']:.3e} "
        f"(must be 0.0)")
    hits, tot = val["op_recall"]
    rec = "  ".join(f"{V.MARKET_OPS[i]} {hits[i]}/{tot[i]}" for i in range(V.N_MARKET_OPS))
    log(f"  market-op per-class recall: {rec}")


class EarlyStopper:
    """Stop when the validation metric has not improved for `patience` epochs.

    Deliberately not "stop when val loss rises": the thing we care about is per-head agreement, and
    loss and accuracy do not have to turn at the same epoch.  `patience = 0` disables stopping
    entirely, which is what the fixed-epoch smoke runs want.

    The best epoch's weights are always kept.  Training past the optimum and then shipping the last
    epoch is a silent way to hand the arena a worse model than the one we measured.
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -float("inf")
        self.best_epoch = 0
        self.n_bad = 0
        self.should_stop = False

    def update(self, epoch, score):
        """Returns True when `score` is a new best (the caller then saves the checkpoint)."""
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.n_bad = score, epoch, 0
            return True
        self.n_bad += 1
        if self.patience and self.n_bad >= self.patience:
            self.should_stop = True
        return False


def train_eval_indices(corpus, target_decisions=10000, seed=12345):
    """A fixed subsample of train states, big enough for ~`target_decisions` macro decisions.

    Fixed so the train column is comparable epoch to epoch: a subsample that moved would make the
    train-vs-val gap a measurement of the subsample.  ~5 macro decisions per state, so 10,000
    decisions is about 2,000 states -- and a Wilson interval on 10,000 is +-1%, tight enough to
    read a gap that matters.
    """
    per_state = max(1.0, corpus.counts()["macro_decisions"] / max(1, corpus.n_states))
    n = int(min(corpus.n_states, math.ceil(target_decisions / per_state)))
    idx = np.random.default_rng(seed).choice(corpus.n_states, size=n, replace=False)
    return np.sort(idx)


def train(epochs=3, batch_size=128, lr=3e-4, weight_decay=1e-2, limit_episodes=None,
          device=None, seed=0, root=None, out_dir=C.CHECKPOINT_DIR, val_split="val",
          eval_batch_size=None, log_every=25, cosine=True, log=print, save_every_epoch=True,
          max_batches=None, early_stop_patience=5, checkpoint_name="bc_v1.pt",
          history_path=None, train_eval_decisions=10000):
    root = root or B.default_root()
    device = pick_device(device)
    eval_batch_size = eval_batch_size or batch_size
    log(f"device={device}  torch={torch.__version__}")

    t0 = time.time()
    train_c = B.Corpus("train", root=root, limit_episodes=limit_episodes)
    val_c = B.Corpus(val_split, root=root, limit_episodes=limit_episodes)
    log(f"train {train_c.counts()}")
    log(f"val   {val_c.counts()}   (loaded in {time.time() - t0:.1f}s)")

    # PLAN_BC Assertion 3, restated for the masks we actually feed: a loss computed under a leaky
    # mask is uninterpretable, so this reads zero before any number below is believed.
    rejected = B.assert_no_expert_rejected(train_c)
    log(f"expert-rejected-by-mask: {rejected}")
    if any(rejected.values()):
        raise AssertionError(f"mask rejects expert labels: {rejected} -- fix the mask, not the "
                             f"expert (PLAN_BC Assertion 3)")

    gen = torch.Generator().manual_seed(seed)
    model = N.BCNet(generator=gen).to(torch.device(device))
    log(f"model: {model.n_params():,} parameters  cfg={model.cfg.as_dict()}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    per_epoch = train_c.n_states // batch_size
    if max_batches:
        per_epoch = min(per_epoch, max_batches)
    total_steps = max(1, per_epoch * epochs)
    log(f"{per_epoch} batches/epoch x {epochs} epochs = {total_steps} steps\n")

    floors = majority_floors(val_c)
    train_idx = train_eval_indices(train_c, target_decisions=train_eval_decisions)
    log(f"train-side metrics on a fixed {len(train_idx):,}-state subsample "
        f"(~{int(train_c.counts()['macro_decisions'] / train_c.n_states * len(train_idx)):,} "
        f"macro decisions), the same states every epoch")
    stopper = EarlyStopper(patience=early_stop_patience)
    best_path = os.path.join(out_dir, checkpoint_name)
    log(f"early stopping: patience {early_stop_patience or 'off'} on the mean per-head val top-1 "
        f"(steps>={V.OPENING_STEPS}); best epoch is kept at {best_path}\n")

    rng = np.random.default_rng(seed)
    history, step = [], 0
    for epoch in range(epochs):
        model.train()
        te = time.time()
        run_loss, run_parts, n_b, n_bad_grad = 0.0, {}, 0, 0
        for bi, batch in enumerate(B.epoch_batches(train_c, batch_size, rng, device=device)):
            if max_batches and bi >= max_batches:
                break
            if cosine:
                for group in opt.param_groups:
                    group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            logits = model(batch)
            loss, parts = L.total_loss(logits, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # `clip_grad_norm_` scales by `max_norm / total_norm`.  A non-finite total norm makes
            # that coefficient ZERO, so the batch's gradients are silently discarded and training
            # merely looks slow.  That defect cost the first v1 run most of its batches, so the
            # clip is counted rather than trusted (CLAUDE.md: a zero counter is an unfinished
            # implementation, not a negative result).
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                n_bad_grad += 1
            opt.step()
            step += 1
            n_b += 1
            run_loss += parts["total"]
            for k, v in parts.items():
                if isinstance(v, float):
                    run_parts[k] = run_parts.get(k, 0.0) + v
            if log_every and (bi + 1) % log_every == 0:
                rate = (bi + 1) / (time.time() - te)
                log(f"  epoch {epoch + 1} batch {bi + 1}/{per_epoch}  loss {run_loss / n_b:7.4f}"
                    f"  {rate:5.2f} batch/s  eta {(per_epoch - bi - 1) / max(rate, 1e-6):5.0f}s")
        train_secs = time.time() - te
        heads = "  ".join(f"{HEAD_LABEL[h]} {run_parts[h] / max(1, n_b):.3f}" for h in N.HEADS)
        log(f"  epoch {epoch + 1} train loss {run_loss / max(1, n_b):.4f}  "
            f"value_mse {run_parts.get('value_mse', 0) / max(1, n_b):.4f}  "
            f"discarded-gradient batches {n_bad_grad}/{n_b}  ({train_secs:.0f}s)")
        log(f"    per-head CE: {heads}")
        if n_bad_grad:
            log(f"    WARNING: {n_bad_grad} batches produced a non-finite gradient norm and were "
                f"clipped to zero -- those batches taught the model nothing")

        res = evaluate(model, val_c, eval_batch_size, device, log=log)
        tres = evaluate(model, train_c, eval_batch_size, device, log=log, indices=train_idx)
        score = mean_accuracy(res)
        log(f"  epoch {epoch + 1} val loss {res['loss']:.4f}  train-subsample loss "
            f"{tres['loss']:.4f}  ({res['seconds'] + tres['seconds']:.0f}s)")
        print_table(res, floors, train=tres, log=log, train_n_states=len(train_idx))
        log(f"  mean per-head top-1 (steps>={V.OPENING_STEPS}): train "
            f"{mean_accuracy(tres):.4f}  val {score:.4f}  gap {mean_accuracy(tres) - score:+.4f}")

        is_best = stopper.update(epoch + 1, score)
        history.append({
            "epoch": epoch + 1, "train_loss": run_loss / max(1, n_b), "val_loss": res["loss"],
            "train_subsample_loss": tres["loss"], "train_seconds": train_secs,
            "eval_seconds": res["seconds"] + tres["seconds"],
            "select_score": score, "is_best": is_best,
            "mean_top1": {"train": mean_accuracy(tres), "val": score},
            "acc": {h: {t: {"train": accuracy(tres, h, t)[0], "val": accuracy(res, h, t)[0]}
                        for t in ("all", "ge32")} for h in N.HEADS},
        })
        if is_best and save_every_epoch:
            C.save(best_path, model, config={
                "max_epochs": epochs, "batch_size": batch_size, "lr": lr,
                "weight_decay": weight_decay, "seed": seed, "device": device,
                "limit_episodes": limit_episodes, "epoch": epoch + 1,
                "early_stop_patience": early_stop_patience, "select_view": SELECT_VIEW,
            }, manifest_hash=train_c.manifest_hash(),
                extra={"history": history, "floors": floors, "select_score": score})
            log(f"  new best (mean top-1 {score:.4f}) -- saved {best_path}")
        else:
            log(f"  no improvement ({score:.4f} vs best {stopper.best:.4f} at epoch "
                f"{stopper.best_epoch}); {stopper.n_bad}/{stopper.patience or '-'} bad epochs")
        log("")
        sys.stdout.flush()
        if stopper.should_stop:
            log(f"early stop: {stopper.patience} epochs without improvement on the mean per-head "
                f"val top-1.  Best was epoch {stopper.best_epoch} ({stopper.best:.4f}).")
            break

    # Restore the best weights.  Training past the optimum and then shipping the last epoch hands
    # the arena a worse model than the one that was measured.
    if save_every_epoch and os.path.exists(best_path) and stopper.best_epoch != len(history):
        model, _payload = C.load(best_path, device=device)
        log(f"restored the epoch-{stopper.best_epoch} weights from {best_path}")
    summary = {"best_epoch": stopper.best_epoch, "best_score": stopper.best,
               "stopped_early": stopper.should_stop, "epochs_run": len(history),
               "select_view": SELECT_VIEW, "history": history}
    if history_path:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"wrote {history_path}")
    return model, summary, floors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-epochs", type=int, default=60,
                    help="the cap; early stopping usually ends the run first")
    ap.add_argument("--epochs", type=int, default=None,
                    help="alias for --max-epochs, kept for the fixed-length smoke runs")
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="epochs without improvement on the mean per-head val top-1; 0 disables")
    ap.add_argument("--checkpoint-name", default="bc_v1.pt")
    ap.add_argument("--train-eval-decisions", type=int, default=10000,
                    help="size of the fixed train-side subsample used for the train column")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-root", default=None)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out-dir", default=C.CHECKPOINT_DIR)
    ap.add_argument("--no-cosine", action="store_true")
    ap.add_argument("--log-every", type=int, default=25, help="0 silences per-batch progress")
    ap.add_argument("--history-json", default=None)
    a = ap.parse_args(argv)
    train(epochs=a.epochs if a.epochs is not None else a.max_epochs,
          batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
          limit_episodes=a.limit_episodes, device=a.device, seed=a.seed, root=a.shard_root,
          out_dir=a.out_dir, val_split=a.val_split, cosine=not a.no_cosine,
          max_batches=a.max_batches, log_every=a.log_every,
          early_stop_patience=a.early_stop_patience, checkpoint_name=a.checkpoint_name,
          history_path=a.history_json, train_eval_decisions=a.train_eval_decisions)


if __name__ == "__main__":
    main()
