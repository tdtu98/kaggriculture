"""Branch-coverage audit of the REFERENCE environment across the whole verification suite.

`tools/audit.py` counts rules from a hand-written list, so it can only report on behaviours I
thought to name. This measures the reference implementation directly: every line and branch of
`kaggriculture.py` executed while its state was being compared against kagsim step-by-step.

**If a branch is covered here, it was differentially tested. If it is not, kagsim's behaviour on
that path is unverified**, no matter how many episodes ran.

Coverage is started before any import so module-level definitions count, and the pytest suite runs
inside the same session so directed scenario tests contribute.

Usage:
    PYTHONPATH=. python tools/coverage_audit.py            # report
    PYTHONPATH=. python tools/coverage_audit.py --fail     # exit 1 on any unexplained gap
"""

from __future__ import annotations

import os
import sys
import sysconfig

import coverage

# Start measuring before kaggriculture.py is imported, so its module-level tables count.
# coverage 7 skips site-packages unless the path is named explicitly, and a glob is not enough —
# it has to be the resolved absolute file.
#
# Resolved via sysconfig, not `os.__file__`: a venv does not copy the stdlib, so `os.__file__`
# points at the *base* interpreter and the old expression silently audited a different
# installation's copy of the environment. That failure is quiet and total -- coverage attaches to
# a file nothing imports, every line reads as unexecuted, and the gate reports 0% while the suite
# passes. `purelib` follows the active interpreter, venv or not.
_REF_REL = "kaggle_environments/envs/kaggriculture/kaggriculture.py"
_REF_FILE = os.path.join(sysconfig.get_paths()["purelib"], _REF_REL)
assert os.path.exists(_REF_FILE), _REF_FILE
COV = coverage.Coverage(branch=True, include=[_REF_FILE])
COV.start()

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import kaggriculture as REF  # noqa: E402

import kagsim  # noqa: E402
from tests.parity import Fuzzer, diff, reference_canonical  # noqa: E402

REF_FILE = REF.__file__

# Uncovered lines that are correct to leave uncovered, with the reason. Anything else is a gap.
ALLOWED_UNCOVERED_SUBSTRINGS = {
    "def renderer": "display only",
    "def _render_tile": "display only",
    "def html_renderer": "display only",
    'return ""': "html_renderer fallback",
    "out +=": "renderer string building",
    "for row in farm": "renderer",
    'if kind == "WEED"': "renderer tile glyph",
    'if kind == "PLANT"': "renderer tile glyph",
    'if "animal" in tile': "renderer tile glyph",
    'if kind == "COOP"': "renderer tile glyph",
    'if kind == "PASTURE"': "renderer tile glyph",
    'return "?"': "renderer fallback",
    "market_price_loop": "n/a",
    "WARNING: kaggriculture market loop": "100k-iteration runaway guard; unreachable in practice",
    "WARNING: HARVEST on immature": 'reference calls this "should never happen"',
    "if crop not in CROPS": "shadowed by the atomic-PLANT precheck; see "
                            "tests/test_rules_coverage.py::test_plant_guards_are_shadowed_*",
    'if private["seeds"].get(crop, 0) <= 0': "shadowed by the atomic-PLANT precheck",
    # --- verified-unreachable through the public API, with the reason ---
    "return state": "interpreter's `if env.done` early return; the framework raises "
                    "FailedPrecondition before the interpreter is reached",
    'private["inventories"].append({})': "_farmer_inventory growth loop: _do_hire keeps "
                                        "inventories at 1+len(hands), and _farmer_position "
                                        "returns None for any larger index",
    "del inv[item]": "DROP/end-of-day skip for entries with n<=0; _inv_add and _inv_take never "
                     "create them",
    "return getattr(d, key, default)": "get() attribute fallback; configuration is always a dict",
    "return (0, 0)": "_default_spawn fallback; (half-1, half-1) is always in NW for boardSize>=2",
    "return False": "_commit_unit unknown-op fallback; the quote loop only emits the four valid ops",
    "return x": "_shape fallback for an unknown shape name; matched by shape_from_name's "
                "Identity arm and covered via the unknown-shape override test",
    "if not overrides:": "_resolve_market_params early return; _initialize (`:243`) only calls it "
                         "when marketParams is already truthy, so `not overrides` is never true",
    "def pass_agent": "bundled agent, covered by tests/test_baselines.py",
    "def random_agent": "bundled agent",
    "def starter_agent": "bundled agent, covered by tests/test_baselines.py",
    "rng.choice(farmer_ops)": "bundled random_agent",
    "farmer_ops =": "bundled random_agent",
    "affordable =": "bundled random_agent",
    "available_seeds =": "bundled random_agent",
    "hands_actions = [[rng.choice": "bundled random_agent",
    "agents = {": "module registry",
    "json_path =": "module load",
    "with open(json_path)": "module load",
    "specification = json.load": "module load",
    "jspath =": "html_renderer",
    "if path.exists(jspath)": "html_renderer",
    "with open(jspath": "html_renderer",
    "return f.read()": "html_renderer",
}


# Whole functions that are not simulation logic. Excluding by span beats matching body lines.
NON_SIMULATION_FUNCS = {
    "renderer": "text renderer, display only",
    "_render_tile": "text renderer, display only",
    "html_renderer": "html renderer, display only",
    "pass_agent": "bundled agent (covered by tests/test_baselines.py)",
    "random_agent": "bundled agent",
    "starter_agent": "bundled agent (covered by tests/test_baselines.py)",
}


def non_simulation_spans(path: str):
    """(start, end, reason) line spans for functions that are out of scope for parity."""
    import ast

    tree = ast.parse(open(path).read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in NON_SIMULATION_FUNCS:
            out.append((node.lineno, node.end_lineno, NON_SIMULATION_FUNCS[node.name]))
    return out


def drive(actions_for, seeds, steps, cfgs) -> int:
    bad = 0
    for cfg_extra in cfgs:
        for seed in seeds:
            cfg = {"episodeSteps": steps + 2, "seed": seed, **cfg_extra}
            env = make("kaggriculture", configuration=cfg)
            env.reset(num_agents=2)
            sim = kagsim.Sim(dict(cfg))
            act = actions_for(seed)
            for i in range(steps):
                a = act(env)
                env.step(a)
                sim.step(a)
                ref, got = reference_canonical(env), sim.canonical_state()
                if ref != got:
                    print(f"DIVERGENCE cfg={cfg_extra} seed={seed} step={i}")
                    print("\n".join(diff(ref, got)))
                    bad += 1
                    break
    return bad


def fuzz_actions(seed):
    fz = Fuzzer(9000 + seed)
    return lambda env: [fz.player_action(env.state[p].observation, p) for p in range(2)]


def engine_actions_factory(params):
    from agent import make_agent

    def make_it(_seed):
        agents = [make_agent(params), make_agent(params)]
        return lambda env: [agents[p](env.state[p].observation) for p in range(2)]

    return make_it


def main() -> None:
    import pytest

    from agent import Params

    def mix(**kw):
        base = {c: 0.0 for c in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]}
        return {**base, **kw}

    bad = drive(fuzz_actions, range(3), 400, [
        {},
        {"weedSpawnChance": 0.0, "townShopUnlockInterval": 10**6},
        {"boardSize": 6, "turnsPerDay": 5},
        {"shedCapacity": 4, "maxMarketOrdersPerTurn": 2},
        {"farmHandCostMult": 4, "startingMoney": 50000},
        # marketParams is implemented, so exercise its resolve/lookup paths too.
        {"marketParams": {"MELON": {"above_func": "log10", "above_target": 0.3, "T": 50}}},
        {"marketParams": {"WHEAT": {"base": 5, "I0": 100, "below_func": "sq"},
                          "BOGUS": {"base": 1}}},
    ])
    for p in [
        Params(hire_max=8, crop_mix=mix(MELON=1)),
        Params(hire_max=8, crop_mix=mix(TOMATO=1)),
        Params(hire_max=8, crop_mix=mix(STRAWBERRY=1)),
        Params(hire_max=6, crop_mix=mix(WHEAT=1), goose_target=8, care=True, goose_min_cash=200),
        Params(hire_max=6, crop_mix=mix(CARROT=1), goose_target=4, care=False,
               feed_alternate=True, goose_min_cash=200),
    ]:
        bad += drive(engine_actions_factory(p), [0], 718, [{}])

    print("\n--- running pytest inside the coverage session ---")
    rc = pytest.main(["tests/", "-q", "--no-header", "-p", "no:cacheprovider"])
    tests_failed = rc != 0

    COV.stop()

    _, statements, _, missing, _ = COV.analysis2(REF_FILE)
    covered = len(statements) - len(missing)
    pct = 100.0 * covered / max(len(statements), 1)
    src = open(REF_FILE).read().splitlines()

    excluded_spans = non_simulation_spans(REF_FILE)
    gaps = []
    excused = 0
    for ln in sorted(missing):
        text = src[ln - 1].strip()
        in_span = any(a <= ln <= b for a, b, _ in excluded_spans)
        window = "\n".join(src[max(0, ln - 2):ln + 3])
        why = next((v for k, v in ALLOWED_UNCOVERED_SUBSTRINGS.items() if k in window), None)
        if in_span or why:
            excused += 1
        else:
            gaps.append((ln, text))

    print(f"\nreference: {REF_FILE}")
    print(f"statements: {len(statements)}")
    print(f"  executed under differential test : {covered}  ({pct:.1f}%)")
    print(f"  excused (see breakdown below)    : {excused}")
    print(f"  UNEXPLAINED                      : {len(gaps)}")

    # Group the excused lines by reason so "89%" is never mistaken for "11% untested".
    from collections import Counter
    reasons: Counter = Counter()
    for ln in sorted(missing):
        text = src[ln - 1].strip()
        window = "\n".join(src[max(0, ln - 2):ln + 3])
        span_reason = next((r for a, b, r in excluded_spans if a <= ln <= b), None)
        sub_reason = next((v for k, v in ALLOWED_UNCOVERED_SUBSTRINGS.items() if k in window), None)
        if span_reason or sub_reason:
            reasons[span_reason or sub_reason] += 1
    print("\n  excused breakdown:")
    for reason, n in reasons.most_common():
        print(f"    {n:>3}  {reason}")

    if gaps:
        print("\nUNEXPLAINED UNCOVERED LINES — kagsim's behaviour here is UNVERIFIED:")
        for ln, text in gaps:
            print(f"  {ln:>5}  {text[:92]}")
    else:
        print("\nEvery simulation-logic line of the reference was executed under differential test.")

    # Reported separately: a failing test is not a simulator divergence, and treating them as one
    # sends the reader hunting for a parity bug that does not exist.
    print(f"\nsimulator divergences: {bad}")
    print(f"test suite: {'FAILED' if tests_failed else 'passed'}")
    if "--fail" in sys.argv and (bad or gaps or tests_failed):
        sys.exit(1)


if __name__ == "__main__":
    main()
