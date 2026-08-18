"""Load the session-line agents without editing them.

Every file in this directory was written inside a Claude session sandbox and hard-codes
`/home/claude/main.py` (Boatlee) and `/home/claude/planner.py`. Those paths do not exist here, so
each module raises `FileNotFoundError` at *import* time.

The fix is deliberately a loader rather than an edit: the sources stay byte-identical to what
produced the session-line measurements, so the numbers in PLAN_v4 §1 remain attributable, and the
one substitution being made is visible in one place instead of being spread through seven diffs.

    from session_line import load
    executor = load("executor")          # -> module with .agent(obs)

`/home/claude/main.py` resolves to `reference/kaggriculture/1/submission.py`, which is sha256
identical to the `main.py` these files were written against.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_RESOLVED = os.path.join(_HERE, ".resolved")

#: The Boatlee copy these files were written against, verified identical by sha256.
BOATLEE = os.path.join(_ROOT, "reference", "kaggriculture", "1", "submission.py")

#: sandbox path -> what to put in its place. A value that names a file in this directory is
#: rewritten in turn: `executor.py` loads `planner.py` *by path*, and `planner.py` carries the
#: same sandbox reference, so substituting only the top-level source leaves the second hop broken.
SUBSTITUTIONS = {
    "/home/claude/main.py": BOATLEE,
    # -> planner-2, NOT planner. `executor.py` calls `planner.plan(obs, n, pool=...)` and only
    # planner-2 has the `pool` parameter; against planner.py every turn raises TypeError inside
    # the executor's blanket `except Exception`, which returns all-PASS. The agent then plays a
    # full season doing nothing and scores the $3,000 starting bank — a number that looks like a
    # measurement and is really an unwired import. The two files are the same module at two dates;
    # planner-2 is the one the executor was written against.
    "/home/claude/planner.py": "planner-2",     # resolved recursively, see _materialize
}

_cache: dict[str, types.ModuleType] = {}


def _substitute(src: str, who: str) -> str:
    for sandbox, target in SUBSTITUTIONS.items():
        if sandbox not in src:
            continue
        local = _materialize(target) if not os.path.isabs(target) else target
        if not os.path.exists(local):
            raise FileNotFoundError(f"{who} needs {sandbox} -> {local}, which is missing")
        src = src.replace(sandbox, local)
    return src


def _materialize(name: str) -> str:
    """Write a path-resolved copy of `session_line/<name>.py` and return its path.

    Only for modules loaded *by path* from inside another session file. The originals are never
    touched; `.resolved/` is a build artifact.
    """
    os.makedirs(_RESOLVED, exist_ok=True)
    out = os.path.join(_RESOLVED, f"{name}.py")
    with open(os.path.join(_HERE, f"{name}.py")) as fh:
        src = _substitute(fh.read(), f"{name}.py")
    with open(out, "w") as fh:
        fh.write(src)
    return out


def load(name: str, fresh: bool = False) -> types.ModuleType:
    """Exec `session_line/<name>.py` with the sandbox paths repointed at this repo.

    `fresh=True` returns a new module object instead of the cached one. These agents keep
    per-season state in module globals (`executor.py`'s `_ST`), so anything replaying seasons
    back to back needs its own copy — the arena's "rebuild the agent per match" rule.
    """
    if not fresh and name in _cache:
        return _cache[name]

    path = os.path.join(_HERE, f"{name}.py")
    with open(path) as fh:
        src = _substitute(fh.read(), f"{name}.py")

    mod = types.ModuleType(f"session_line.{name}")
    mod.__file__ = path
    if not fresh:
        # a fresh copy stays out of sys.modules so it cannot be picked up as "the" module
        sys.modules[mod.__name__] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    if not fresh:
        _cache[name] = mod
    return mod


def agent(name: str):
    """The `agent(obs)` callable from a session-line agent module."""
    return load(name).agent
