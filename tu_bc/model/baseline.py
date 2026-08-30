"""PLAN_BC Chapter 4 -- the dumbest thing that could work, on one head.

Multinomial logistic regression over **20 hand-picked scalars** predicting the macro verb.  315
parameters.  Its whole job is to answer one question before a weekend goes into Chapter 5: does the
Phase-1 pipeline carry any signal at all?

The chapter's rule, and the reason this file exists: *an accuracy number means nothing on its own.*
Every number below is printed next to its majority-class floor, with a Wilson interval, twice --
over all steps and over steps >= 32 only.  Ryo plays a fixed ~30-step opening (E88), those steps are
nearly free to predict, and a model that learned the opening and nothing else would still post a
better-than-floor headline.

    PYTHONPATH=. .venv/bin/python -m model.baseline

Chapter 4's stop rule: if this cannot beat the floor by an interval that excludes it, the defect is
in the encoding, not the model.  Go back to Chapter 3 -- do not tune.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import batching as B
from . import dataset as D
from . import vocab as V

FEATURE_NAMES = (
    "hour", "day",                                                        # global
    "is_farmer", "next_to_shed", "has_wheat", "has_fertilizer", "has_animal",   # worker
    *(f"kind:{k}" for k in ("LOCKED", "EMPTY", "PLANT", "WEED", "COOP", "PASTURE",
                            "COOP_ANIMAL", "PASTURE_ANIMAL")),            # target tile kind
    "watered", "harvest_ready", "fed_today", "cared_today", "fertilizer_available",
)
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 20


def features(corpus, flat):
    """`[n_decisions, 20]` float32.  Every column is read straight out of a stored token."""
    s, u, t = flat["state"], flat["unit"], flat["tile"]
    g = corpus.states["global"][s].astype(np.float32)
    w = corpus.states["workers"][s, u].astype(np.float32)
    tile = corpus.states["tiles"][s, t].astype(np.float32)
    return np.concatenate([
        g[:, 2:3], g[:, 1:2],                       # hour / (24-1), day / (30-1)
        w[:, 0:1], w[:, 4:5], w[:, 23:24], w[:, 24:25], w[:, 25:26],
        tile[:, 0:8],                               # tile kind one-hot
        tile[:, 16:17],                             # watered_today
        tile[:, 22:23],                             # harvestable
        tile[:, 25:26], tile[:, 26:27],             # fed_today / cared_today
        tile[:, 29:30],                             # fertilizer_available
    ], axis=1)


def fit(x, y, n_classes, epochs=60, lr=0.5, weight_decay=1e-4, hidden=0, seed=0, device="cpu",
        log=print):
    """Full-batch LBFGS-free gradient descent.  No sklearn in this venv, and none is needed."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y).to(dev)
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp_min(1e-6)
    xt = (xt - mu) / sd
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, n_classes))
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05, generator=gen)
            else:
                torch.nn.init.zeros_(p)
    else:
        net = torch.nn.Linear(x.shape[1], n_classes)
        torch.nn.init.zeros_(net.weight)
        torch.nn.init.zeros_(net.bias)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(xt), yt)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            log(f"    epoch {epoch + 1:3d}/{epochs}  train CE {loss.item():.4f}")
    n_params = sum(p.numel() for p in net.parameters())
    return net, (mu, sd), n_params


@torch.no_grad()
def predict(net, norm, x, device="cpu"):
    mu, sd = norm
    xt = (torch.from_numpy(x).to(torch.device(device)) - mu) / sd
    return net(xt).argmax(-1).cpu().numpy()


def report(name, correct, n, floor, floor_class, log=print):
    acc = correct / max(1, n)
    lo, hi = D.wilson(acc, n)
    beat = "BEATS" if lo > floor else ("ties" if hi > floor else "BELOW")
    log(f"  {name:<16} {acc:6.2%} [{lo:6.2%}, {hi:6.2%}]  floor {floor:6.2%} "
        f"({floor_class})  n={n:,}  -> {beat} the floor")
    return acc, lo, hi


def run(limit_episodes=None, epochs=60, lr=0.5, hidden=0, device="cpu", root=None, log=print):
    root = root or B.default_root()
    t0 = time.time()
    log("loading shards ...")
    train = B.Corpus("train", root=root, limit_episodes=limit_episodes)
    val = B.Corpus("val", root=root, limit_episodes=limit_episodes)
    log(f"  train {train.counts()}\n  val   {val.counts()}   ({time.time() - t0:.1f}s)")

    ftr, fva = B.flat_macro(train), B.flat_macro(val)
    xtr, ytr = features(train, ftr), ftr["verb"]
    xva, yva = features(val, fva), fva["verb"]
    log(f"features: {N_FEATURES} -> {V.N_MACRO_VERBS} macro verbs; "
        f"{len(ytr):,} train decisions, {len(yva):,} val decisions")

    counts = np.bincount(ytr, minlength=V.N_MACRO_VERBS)
    floor_class = int(counts.argmax())
    floors = {}
    for tag, keep in (("all steps", np.ones(len(yva), bool)),
                      ("steps>=32", fva["step"] >= V.OPENING_STEPS)):
        floors[tag] = (float((yva[keep] == floor_class).mean()), int(keep.sum()))

    net, norm, n_params = fit(xtr, ytr.astype(np.int64), V.N_MACRO_VERBS,
                              epochs=epochs, lr=lr, hidden=hidden, device=device, log=log)
    pred = predict(net, norm, xva, device=device)

    log(f"\nmacro-verb head, {n_params} parameters, majority class = "
        f"{V.MACRO_VERBS[floor_class]}")
    out = {"n_params": n_params, "majority_class": V.MACRO_VERBS[floor_class]}
    for tag, keep in (("all steps", np.ones(len(yva), bool)),
                      ("steps>=32", fva["step"] >= V.OPENING_STEPS)):
        floor, n = floors[tag]
        acc, lo, hi = report(tag, int((pred[keep] == yva[keep]).sum()), n, floor,
                             V.MACRO_VERBS[floor_class], log=log)
        out[tag] = {"acc": acc, "lo": lo, "hi": hi, "floor": floor, "n": n}
    log(f"\ntotal {time.time() - t0:.1f}s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=0, help="0 = logistic regression")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard-root", default=None)
    a = ap.parse_args(argv)
    run(limit_episodes=a.limit_episodes, epochs=a.epochs, lr=a.lr, hidden=a.hidden,
        device=a.device, root=a.shard_root)


if __name__ == "__main__":
    main()
