"""Save and load, with the four stamps that catch feature/weight skew.

PLAN_BC's risk 9: silent feature/weight skew produces a model that scores like an untrained one,
and **it is indistinguishable from "BC didn't work"**.  That is the same class of failure as E21,
where a submission raised `NameError`, never loaded, and scored the $3,000 starting bank.

So every checkpoint carries the weights plus `FEATURE_VERSION`, a vocabulary hash, the shard
manifest hash, and the training config -- and `load` asserts the first two rather than warning.
"""

from __future__ import annotations

import hashlib
import os

import torch

from . import features as F
from . import net as N
from . import vocab as V

CHECKPOINT_DIR = "checkpoints"
FORMAT_VERSION = 1


def vocab_hash():
    """A fingerprint of every head width and alphabet.

    A one-entry disagreement between a head width and the mask table is the failure PLAN_BC Ch3
    settled the vocabulary as one constant to prevent; this makes it a load-time error instead of
    a mystery in the loss.
    """
    h = hashlib.sha256()
    for name in (V.VERBS, V.MACRO_VERBS, V.ITEMS, V.MARKET_OPS, V.QTY_BINS):
        h.update("|".join(str(x) for x in name).encode())
        h.update(b";")
    h.update(f"{V.MAX_UNITS},{V.MAX_MARKET_ORDERS},{V.N_TILES},{V.N_ITEM_SLOTS}".encode())
    return h.hexdigest()[:16]


def save(path, model, config=None, manifest_hash=None, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "feature_version": F.FEATURE_VERSION,
        "vocab_hash": vocab_hash(),
        "manifest_hash": manifest_hash,
        "net_config": model.cfg.as_dict(),
        "train_config": dict(config or {}),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_params": model.n_params(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)
    return path


def load(path, device="cpu", strict_manifest=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("feature_version") != F.FEATURE_VERSION:
        raise ValueError(
            f"{path}: FEATURE_VERSION {payload.get('feature_version')} != {F.FEATURE_VERSION} -- "
            f"these weights were trained against different features and would score like an "
            f"untrained model")
    if payload.get("vocab_hash") != vocab_hash():
        raise ValueError(f"{path}: vocabulary hash {payload.get('vocab_hash')} != {vocab_hash()} "
                         f"-- a head width and the mask table have drifted apart")
    if strict_manifest is not None and payload.get("manifest_hash") != strict_manifest:
        raise ValueError(f"{path}: shard manifest {payload.get('manifest_hash')} != "
                         f"{strict_manifest}")
    model = N.BCNet(N.NetConfig(**legacy_net_config(payload["net_config"])))
    model.load_state_dict(payload["state_dict"])
    return model.to(torch.device(device)), payload


# Fields added to `NetConfig` after checkpoints already existed, and the value that reproduces the
# behaviour those checkpoints were trained under.  A new default must never silently rewrite the
# forward pass of a model someone already measured.
LEGACY_NET_CONFIG = {"ar_gate": False}


def legacy_net_config(stored):
    cfg = dict(stored)
    for key, legacy in LEGACY_NET_CONFIG.items():
        cfg.setdefault(key, legacy)
    return cfg
