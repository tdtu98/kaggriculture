"""The v4 pool: names -> agent builders, with a fingerprint for every one.

`arena/registry.py` already holds 95 agents from the engine line and is the source of truth for
them; this does not duplicate it, it wraps it. What is added here is the handful of pool members
`TASKS_v4` I2 names that the arena has never had: the session-line executor, and the two in-family
exploiters that S1 will build once the compiler exists.

Every builder returns a **fresh** agent. These agents keep per-season state in closures or module
globals, so reusing one across games leaks the previous season into the next — the arena hit this
and rebuilds per match for the same reason.

Fingerprints are content hashes, not names: a table that silently mixed two different `executor_v7`
builds under one label would be worse than no table at all (`arena/registry.py` makes the same
argument, D-series).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Pool members that need a task they do not have yet. Named here so the runner fails with the
#: reason and the task ID rather than a KeyError. `flooder`/`tomato_rusher` left at S1.
PENDING: dict[str, str] = {}

#: A search candidate is not a name, it is a genome, and it still has to reach a worker process.
#: Encoding the vector *into* the name is what makes that work without any shared state: the parent
#: hands `Match` a string, the worker resolves it through `get()`, and a persistent pool created
#: before the candidate existed can still build it. The alternative (a module-global registered in
#: the parent and inherited through `fork`) ties the search to one start method and to a fresh pool
#: per evaluation.
VEC_PREFIX = "vec:"

#: The two in-family exploiters S1 builds (`search/exploiters.py`). They are the compiler bound to
#: a hostile plan, so they are resolved here and not in the arena registry, which predates plans.
EXPLOITER_NOTES = {
    "flooder": "S1: the compiler over-planting strawberry (67 tiles) and cows (14)",
    "tomato_rusher": "S1: the compiler planting 15 tomato tiles the day NE unlocks",
}

#: v4 name -> arena registry name, for pool members the arena already carries.
ALIASES = {
    "kagsim_champion": "champion",
}


@dataclass(frozen=True)
class PoolAgent:
    name: str
    build: Callable[[], Callable[[dict], dict]]
    fingerprint: str
    note: str


def _file_digest(*paths: str) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def _session_executor() -> PoolAgent:
    """`executor_v7` — the Claude-session closed-loop executor, Track F's subject.

    Fingerprinted over the executor *and* the planner it loads: they are one agent in two files,
    and swapping the planner silently changes the policy (it also breaks it — see
    `session_line/README.md`).
    """
    from session_line import _load

    files = (os.path.join(_ROOT, "session_line", "executor.py"),
             os.path.join(_ROOT, "session_line", "planner-2.py"))
    return PoolAgent(
        name="executor_v7",
        build=lambda: _load.load("executor", fresh=True).agent,
        fingerprint=_file_digest(*files),
        note="session-line closed-loop executor + planner-2 (TASKS_v4 Track F)",
    )


def _session_agent(name: str, note: str) -> PoolAgent:
    from session_line import _load

    path = os.path.join(_ROOT, "session_line", f"{name}.py")
    return PoolAgent(
        name=name,
        build=lambda: _load.load(name, fresh=True).agent,
        fingerprint=_file_digest(path, os.path.join(_ROOT, "reference", "kaggriculture", "1",
                                                    "submission.py")),
        note=note,
    )


def _arena_agent(name: str) -> PoolAgent:
    from arena.registry import REGISTRY

    spec = REGISTRY[name]
    return PoolAgent(name=name, build=spec.build, fingerprint=spec.fingerprint, note=spec.note)


def _compiler_files() -> tuple[str, ...]:
    return tuple(os.path.join(_ROOT, "agent", f) for f in
                 ("plan.py", "tasks.py", "router.py", "verify.py", "main_v4.py"))


def vec_name(vector) -> str:
    """The pool name for a genome. Round-trips exactly through `repr`, so the fingerprint of two
    identical vectors is identical and S2's cache can key on the name."""
    from agent.plan import snap

    return VEC_PREFIX + ",".join(repr(float(v)) for v in snap(vector))


def _plan_agent(name: str, plan, note: str) -> PoolAgent:
    """The compiler bound to a specific plan.

    The fingerprint covers the compiler's source *and* the plan: two agents that differ only by
    genome are two different agents, and a table that gave them one label would be the mistake the
    module docstring exists to prevent.
    """
    from agent.main_v4 import make_agent

    digest = hashlib.sha256(
        (_file_digest(*_compiler_files()) + "|" + repr(plan)).encode()).hexdigest()[:12]
    return PoolAgent(name=name, build=lambda: make_agent(plan), fingerprint=digest, note=note)


#: Separator for O3's overlay flags: `compiler#frontrun=1`, `vec:...#counter_mix=1,slot_align=1`.
#:
#: The name has to determine the agent *completely*, because `run()` ships names to worker processes
#: and each one resolves its own (that is why `VEC_PREFIX` encodes a whole genome into a string). An
#: overlay is a `consts` edit rather than a gene (`Plan.with_consts`), so it needs its own suffix
#: rather than a longer vector.
CONST_SEP = "#"


def const_name(base: str, **flags) -> str:
    """`base` plus overlay flags, in a form `get()` round-trips."""
    if not flags:
        return base
    return base + CONST_SEP + ",".join(f"{k}={v}" for k, v in sorted(flags.items()))


def _parse_consts(spec: str) -> dict:
    out: dict = {}
    for piece in spec.split(","):
        if not piece.strip():
            continue
        key, _, value = piece.partition("=")
        value = value.strip()
        try:
            out[key.strip()] = int(value) if value and "." not in value else float(value)
        except ValueError:
            out[key.strip()] = value
    return out


def get(name: str) -> PoolAgent:
    """Resolve a pool name. Raises with the task ID for members that are not built yet."""
    if CONST_SEP in name:
        base, _, spec = name.partition(CONST_SEP)
        inner = get(base)
        plan = getattr(inner.build(), "plan", None)
        if plan is None:
            raise KeyError(f"'{base}' is not a plan-driven agent; '{CONST_SEP}' flags need one")
        return _plan_agent(name, plan.with_consts(**_parse_consts(spec)),
                           f"{inner.note} + consts {spec}")
    if name in PENDING:
        raise NotImplementedError(f"'{name}' does not exist yet — {PENDING[name]}")
    if name.startswith(VEC_PREFIX):
        from agent.plan import decode

        vector = [float(x) for x in name[len(VEC_PREFIX):].split(",") if x]
        return _plan_agent(name, decode(vector), "S1/S2 search candidate (genome)")
    if name in EXPLOITER_NOTES:
        from search.exploiters import EXPLOITERS

        return _plan_agent(name, EXPLOITERS[name](), EXPLOITER_NOTES[name])
    if name == "compiler":
        from agent.main_v4 import make_agent
        from agent.plan import Plan

        return PoolAgent(name="compiler", build=lambda: make_agent(Plan.boatlee_like()),
                         fingerprint=_file_digest(*_compiler_files()),
                         note="C0-C4: plan -> tasks -> route -> verify, recompiled every dawn")
    if name == "executor_v7":
        return _session_executor()
    if name in ("warfare", "metered_LOSES", "denial_LOSES"):
        return _session_agent(name, "session-line overlay on boatlee (measurement input only)")
    name = ALIASES.get(name, name)
    from arena.registry import REGISTRY

    if name not in REGISTRY:
        raise KeyError(f"unknown agent '{name}'. Pool: {', '.join(sorted(names()))}")
    return _arena_agent(name)


def names() -> list[str]:
    """Everything resolvable right now — the arena's registry plus the session line."""
    from arena.registry import REGISTRY

    return sorted({*REGISTRY, *ALIASES, *EXPLOITER_NOTES, "compiler", "executor_v7", "warfare",
                   "metered_LOSES", "denial_LOSES"})


#: The default pool for a v4 measurement (TASKS_v4 I2).
DEFAULT_POOL = ["boatlee", "executor_v7", "starter", "kagsim_champion"]

#: The *search* pool (TASKS_v4 S1): who a candidate is scored against, and how much each one
#: counts. Boatlee is the opponent we have to beat and is weighted x2; the two exploiters are there
#: so a genome cannot assume strawberry/milk/tomato are free (E10); the executor is a genuinely
#: different policy shape rather than another compiled plan.
SEARCH_POOL: tuple[tuple[str, float], ...] = (
    ("boatlee", 2.0),
    ("flooder", 1.0),
    ("tomato_rusher", 1.0),
    ("executor_v7", 1.0),
)
