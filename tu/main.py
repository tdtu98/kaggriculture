"""Kaggle submission entry point.

Bundle layout (tar.gz built by `tools/build_submission.py`):

    main.py
    agent/{__init__,engine,params}.py
    search/champion.json

Deliberately depends on nothing outside the bundle except `kaggle_environments` itself, which the
runner provides. In particular **not `kagsim`** — the Rust simulator exists only for local
training and would not import there.
"""

from __future__ import annotations

import json
import os
import sys

def _bundle_dir() -> str:
    """Locate the bundle root without relying on `__file__`.

    Kaggle does not import this file. It reads the source and runs
    `exec(compile(raw, path, "exec"), {})` (`kaggle_environments/agent.py:47-58`), and that empty
    globals dict has **no `__file__`** — touching it raises `NameError` at import time, the agent
    never loads, and the episode scores the starting bank. Measured: $3,000 flat, 0-40.

    What the loader *does* provide is `sys.path.append(os.path.dirname(path))` before the exec, so
    the bundle directory is on the path; find it by looking for the `agent` package inside it.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    for candidate in reversed(sys.path):
        if candidate and os.path.isdir(os.path.join(candidate, "agent")):
            return os.path.abspath(candidate)
    return os.getcwd()


_HERE = _bundle_dir()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent import Engine, Params  # noqa: E402

_SAFE_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _load_params() -> Params:
    """Champion knobs, falling back to defaults rather than failing the episode."""
    path = os.path.join(_HERE, "search", "champion.json")
    try:
        with open(path) as f:
            return Params(**json.load(f)["params"])
    except Exception as exc:                      # noqa: BLE001 - never fail at import time
        print(f"[kaggriculture] champion.json unavailable ({exc}); using defaults", flush=True)
        return Params()


_PARAMS = _load_params()
_ENGINE: Engine | None = None


def agent(obs):
    """One turn.

    Two things this has to get right that a local test would not reveal:

    * **Per-episode state.** The engine keeps sticky per-unit task assignments across turns, and
      the runner may reuse the process for several games, so state must be rebuilt at the start of
      each episode. Detected from `day == 0 and hour == 0` — the only turn where both hold —
      rather than from object lifetime.
    * **`obs["step"]` does not exist for seat 1.** It is a framework field, not declared `shared`,
      so `obs.get("step", 0)` silently reads 0 there on every turn. Nothing here reads it; `day`
      and `hour` are shared and are used instead.
    """
    global _ENGINE
    try:
        day, hour = obs["day"], obs["hour"]
        if _ENGINE is None or (day == 0 and hour == 0):
            _ENGINE = Engine(_PARAMS)
        return _ENGINE(obs)
    except Exception as exc:                      # noqa: BLE001
        # A raised exception forfeits the game. Passing costs one turn.
        print(f"[kaggriculture] turn failed: {exc!r}", flush=True)
        return dict(_SAFE_ACTION)
