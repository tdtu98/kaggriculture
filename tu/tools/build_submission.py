"""Build submission.tar.gz for `kaggle competitions submit`.

Kaggle requires `main.py` at the archive root. Everything the agent needs travels with it; nothing
outside the bundle is importable there except `kaggle_environments` itself.

Usage:
    PYTHONPATH=. python tools/build_submission.py [--out submission.tar.gz]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time

CONTENTS = [
    "main.py",
    "agent/__init__.py",
    "agent/engine.py",
    "agent/forward.py",
    "agent/relay.py",
    "agent/relay_table.py",
    "agent/params.py",
    "search/champion.json",
]

# Modules that exist only for local training and would fail to import on the runner.
FORBIDDEN_IMPORTS = ["kagsim", "arena", "sim.", "search.cem", "numpy", "torch"]


def check_no_local_only_imports(paths: list[str]) -> list[str]:
    """A missing import on the runner forfeits every game, silently until the logs are read."""
    import ast

    problems = []
    for path in paths:
        if not path.endswith(".py"):
            continue
        tree = ast.parse(open(path).read(), path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                for bad in FORBIDDEN_IMPORTS:
                    if n == bad.rstrip(".") or n.startswith(bad):
                        problems.append(f"{path}:{node.lineno} imports {n!r}")
    return problems


def smoke_test() -> tuple[float, float]:
    """Play full games through the *reference* environment the way Kaggle actually loads an agent.

    This must go through `env.run(["main.py", ...])`, never `import main`. Kaggle does not import
    the submission: it `exec`s the source into an empty globals dict and then takes **the last
    callable defined** (`kaggle_environments/agent.py:47-63`). Two consequences that a plain import
    cannot see, both of which shipped undetected:

      * `__file__` does not exist there. Touching it raised `NameError`, the agent never loaded,
        and the episode scored the starting bank -- $3,000 flat, measured 0-40 against a real
        opponent while every local test reported the agent healthy.
      * if `agent` is not the last module-level callable, Kaggle silently calls something else.

    Playing a live opponent rather than PASS also matters: a crashed agent scores $3,000 and a
    passing one scores about the same, so PASS-vs-PASS cannot tell the two apart.
    """
    from kaggle_environments import make
    from kaggle_environments.agent import get_last_callable

    with open("main.py") as f:
        picked = get_last_callable(f.read(), path="main.py")
    if picked.__name__ != "agent":
        raise SystemExit(
            f"Kaggle would call {picked.__name__!r}, not 'agent' -- it takes the *last* "
            f"module-level callable. Move `agent` to the end of main.py."
        )

    worst, monies = 0.0, []
    for seat in (0, 1):                       # both seats: they are not symmetric
        pair = ["main.py", "starter"] if seat == 0 else ["starter", "main.py"]
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 4242})
        t0 = time.perf_counter()
        env.run(pair)
        worst = max(worst, (time.perf_counter() - t0) / 718)
        if env.steps[-1][seat]["status"] != "DONE":
            raise SystemExit(f"seat {seat} finished {env.steps[-1][seat]['status']}, not DONE")
        monies.append(env.steps[-1][seat]["reward"])

    money = min(monies)
    if money <= 3_000:                        # the starting bank: the agent never acted
        raise SystemExit(f"seat money ${money:,.0f} <= starting bank -- the agent did nothing")
    return money, worst


def main_() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.tar.gz")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    missing = [p for p in CONTENTS if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing from bundle: {missing}")

    problems = check_no_local_only_imports(CONTENTS)
    if problems:
        raise SystemExit("local-only imports would break on the Kaggle runner:\n  "
                         + "\n  ".join(problems))
    print(f"import check: clean ({len(CONTENTS)} files)")

    if not args.skip_smoke:
        money, worst = smoke_test()
        print(f"smoke test (both seats, reference env, Kaggle loader): ${money:,.0f}, "
              f"worst turn {1000 * worst:.1f}ms of a 1000ms budget")
        if money <= 3000:
            raise SystemExit("agent did not out-earn its starting bank; refusing to build")
        if worst > 0.5:
            raise SystemExit(f"worst turn {worst:.2f}s is too close to actTimeout")

    with tarfile.open(args.out, "w:gz") as tar:
        for path in CONTENTS:
            tar.add(path, arcname=path)
        # A short manifest, so a downloaded submission can be identified later.
        info = tarfile.TarInfo("BUILD.txt")
        body = (f"built {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"python {sys.version.split()[0]}\n"
                f"files: {', '.join(CONTENTS)}\n").encode()
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))

    size = os.path.getsize(args.out)
    print(f"\n{args.out}  ({size / 1024:.1f} KiB)")
    print("submit with:  kaggle competitions submit kaggriculture "
          f"-f {args.out} -m 'scripted champion (CEM T2.1)'")


if __name__ == "__main__":
    main_()
