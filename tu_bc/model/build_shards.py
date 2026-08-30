"""Build the training shards, and print the verification report.

    PYTHONPATH=. .venv/bin/python -m model.build_shards
    PYTHONPATH=. .venv/bin/python -m model.build_shards --limit 10 --out /tmp/shards
    PYTHONPATH=. .venv/bin/python -m model.build_shards --include-opponent-seat

Every number in the report is a counter that had to be *incremented by real work* to be non-zero.
A zero counter is an unfinished implementation, not a negative result (CLAUDE.md, E44) -- so the
report prints the controls next to the assertions: `a1_roster_fail_same_step` shows how badly the
WRONG pairing fails, `a2_step_patched` shows the seat-1 repair actually firing,
`a4_sell_over_observed_shed` shows how many SELLs the naive shed check would have flagged, and
`a4_animal_place_to_tile` shows the carve-out taking the tile branch.

Hard gates (non-zero means stop, do not train):
    a1_roster_fail, a2_md5_fail, n_expert_actions_rejected_by_mask,
    n_expert_orders_rejected_by_mask, a4_sell_over_effective_shed
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from . import dataset as DS
from . import decode as D
from . import features as F
from . import vocab as V

HARD_GATES = ("a1_roster_fail", "a2_md5_fail", "n_expert_actions_rejected_by_mask",
              "n_expert_orders_rejected_by_mask", "a4_sell_over_effective_shed")
SHARD_SIZE_THRESHOLD = 1.5 * 1024 * 1024      # PLAN_BC Ch3: revisit the layout above this


def _fmt(n):
    return f"{n:,}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=D.CORPUS_DIR)
    ap.add_argument("--out", default=DS.SHARD_ROOT)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=0, help="stop after N episodes (a smoke run)")
    ap.add_argument("--include-opponent-seat", action="store_true",
                    help="also decode the opponent's seat. OFF by default: the other seat is 36 "
                         "assorted opponents of unknown quality, and cloning them would blur the "
                         "one policy we are trying to learn (PLAN_BC Ch2).")
    ap.add_argument("--no-write", action="store_true", help="verify only; write nothing")
    ap.add_argument("--report", default="", help="also dump the report as JSON here")
    args = ap.parse_args(argv)

    splits = tuple(s for s in args.splits.split(",") if s)
    manifest = D.read_manifest(args.corpus)
    episodes = D.episode_paths(args.corpus, splits)
    if args.limit:
        episodes = episodes[: args.limit]

    # Two tallies, never one.  The hard gates are claims about *the teacher*: Ryo's seat is clean
    # on all four assertions, and the 36 assorted opponents are not (measured: 73 illegal unit
    # actions, 481 illegal orders and 554 oversized SELLs in two episodes alone).  Mixing them
    # would turn a real gate into a permanently-red light and teach us to ignore it.
    counters = D._counters()
    opp_counters = D._counters()
    per_split = {s: {"episodes": 0, "states": 0, "raw": 0, "macro": 0, "market": 0}
                 for s in splits}
    sizes, vocab_counts = [], None
    raw_all, macro_all, market_all, step_of_state = [], [], [], []
    state_base = 0
    t0 = time.time()

    for n, (eid, split, path) in enumerate(episodes):
        row = manifest.get(eid)
        if row is None:
            raise D.DecodeError(f"{eid} is on disk but not in manifest.csv")
        if row["split"] != split:
            raise D.DecodeError(f"{eid}: manifest says split {row['split']!r}, found in {split!r}")
        seats = [row["ryo_seat"]] + ([1 - row["ryo_seat"]] if args.include_opponent_seat else [])
        for seat in seats:
            is_teacher = seat == row["ryo_seat"]
            ep = D.decode_episode(path, seat, episode_id=eid, split=split, manifest_row=row,
                                  check_teacher=is_teacher,
                                  counters=counters if is_teacher else opp_counters)
            if not is_teacher:
                ep.episode_id = f"{eid}_opp"
            if not args.no_write:
                _p, size = DS.write_shard(ep, root=args.out)
                sizes.append(size)

            ps = per_split[split]
            ps["episodes"] += 1
            ps["states"] += len(ep.states)
            ps["raw"] += len(ep.raw)
            ps["macro"] += len(ep.macro)
            ps["market"] += len(ep.market)

            if is_teacher:
                steps = np.arange(len(ep.states), dtype=np.int32)
                step_of_state.append(steps)
                for rows, sink in ((ep.raw, raw_all), (ep.macro, macro_all),
                                   (ep.market, market_all)):
                    if rows:
                        a = np.asarray(rows, dtype=np.int32)
                        a[:, 0] += state_base
                        sink.append(a)
                state_base += len(ep.states)
        if n % 10 == 0:
            print(f"  [{n + 1:3d}/{len(episodes)}] {eid} {split}  {time.time() - t0:5.1f}s",
                  file=sys.stderr, flush=True)

    # ---- the vocabulary check, over BOTH seats of every episode --------------------------
    def _all_actions():
        for eid, split, path in episodes:
            with open(path) as f:
                d = json.load(f)
            for st in d["steps"][1:]:
                for p in (0, 1):
                    yield st[p]["action"]
            del d

    vocab_counts = V.validate_against_corpus(_all_actions())

    step_of_state = np.concatenate(step_of_state) if step_of_state else np.zeros(0, np.int32)
    raw_all = np.concatenate(raw_all) if raw_all else np.zeros((0, 6), np.int32)
    macro_all = np.concatenate(macro_all) if macro_all else np.zeros((0, 11), np.int32)
    market_all = np.concatenate(market_all) if market_all else np.zeros((0, 6), np.int32)

    report = {
        "corpus": args.corpus,
        "out": args.out,
        "episodes": len(episodes),
        "seats": "ryo + opponent" if args.include_opponent_seat else "ryo only",
        "feature_version": F.FEATURE_VERSION,
        "shard_module_version": DS.MODULE_VERSION,
        "elapsed_s": round(time.time() - t0, 1),
        "per_split": per_split,
        "counters": counters,
        "opponent_counters": opp_counters if args.include_opponent_seat else None,
        "vocab": vocab_counts,
    }

    # ---- E87 ---------------------------------------------------------------------------
    walk_runs = counters["macro_walk_runs"]
    e87 = counters["macro_shortest"] / walk_runs if walk_runs else float("nan")
    lo, hi = DS.wilson(e87, walk_runs)
    report["E87"] = {"frac_segments_shortest_path": e87, "walking_runs": walk_runs,
                     "shortest": counters["macro_shortest"],
                     "detours": walk_runs - counters["macro_shortest"],
                     "truncated_excluded": counters["macro_truncated"],
                     "wilson": [lo, hi],
                     "decision": "macro action space" if e87 >= 0.9 else "macro-with-MOVE"}

    # ---- majority-class floors, recomputed on Ryo's seat --------------------------------
    macro_act = macro_all[macro_all[:, 9] == 0] if macro_all.shape[0] else macro_all
    report["floors"] = {
        "raw_verb_all": DS.majority_floor(raw_all, 2),
        "raw_verb_ge32": DS.majority_floor(
            raw_all, 2, step_of_state[raw_all[:, 0]] >= V.OPENING_STEPS if raw_all.shape[0] else None),
        "macro_verb_all": DS.majority_floor(macro_all, 3),
        "macro_verb_ge32": DS.majority_floor(
            macro_all, 3,
            step_of_state[macro_all[:, 0]] >= V.OPENING_STEPS if macro_all.shape[0] else None),
        "macro_commit_all": DS.majority_floor(macro_all, 9),
        "macro_verb_acting_only_all": DS.majority_floor(macro_act, 3),
        "market_op_all": DS.majority_floor(market_all, 2),
        "market_item_all": DS.majority_floor(market_all, 3),
        "market_qty_all": DS.majority_floor(market_all, 4),
    }
    for k, v in report["floors"].items():
        v["wilson"] = list(DS.wilson(v["floor"], v["n"])) if v["n"] else [float("nan")] * 2

    if sizes:
        report["shard_size"] = {
            "n": len(sizes), "mean_mb": statistics.mean(sizes) / 1e6,
            "median_mb": statistics.median(sizes) / 1e6,
            "max_mb": max(sizes) / 1e6, "total_mb": sum(sizes) / 1e6,
            "over_threshold": sum(1 for s in sizes if s > SHARD_SIZE_THRESHOLD),
            "threshold_mb": SHARD_SIZE_THRESHOLD / 1e6,
        }

    report["gates"] = {k: counters[k] for k in HARD_GATES}
    report["gates_pass"] = all(counters[k] == 0 for k in HARD_GATES)
    print_report(report)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return 0 if report["gates_pass"] else 1


def print_report(r):
    C = r["counters"]
    p = print
    p("")
    p("=" * 78)
    p(f"  MODEL SHARD BUILD -- {r['episodes']} episodes, {r['seats']}, "
      f"FEATURE_VERSION={r['feature_version']}, {r['elapsed_s']}s")
    p("=" * 78)

    p("\n-- states per split (PLAN_BC Ch2 expects 50,330 / 10,785 / 10,785) --")
    for split, v in r["per_split"].items():
        p(f"  {split:6s} episodes {v['episodes']:4d}   states {_fmt(v['states']):>9s}   "
          f"raw {_fmt(v['raw']):>9s}   macro {_fmt(v['macro']):>9s}   "
          f"market {_fmt(v['market']):>8s}")

    p("\n-- assertions (all five gates must read 0) --")
    p(f"  A1 roster checks                        {_fmt(C['a1_roster_checks'])}")
    p(f"  A1 roster FAIL (gate)                   {C['a1_roster_fail']}")
    p(f"     ... control: same-step pairing fails  {_fmt(C['a1_roster_fail_same_step'])}  "
      f"<- proves the assertion has teeth")
    p(f"  A2 md5 checks (seat 1 shared fields)    {_fmt(C['a2_md5_checks'])}")
    p(f"  A2 md5 FAIL (gate)                      {C['a2_md5_fail']}")
    p(f"     ... control: obs missing `step`       {_fmt(C['a2_step_patched'])}  "
      f"<- proves the seat-1 patch fired")
    p(f"  A3 unit actions checked                 {_fmt(C['n_unit_actions'])}")
    p(f"  A3 n_expert_actions_rejected_by_mask    {C['n_expert_actions_rejected_by_mask']}  (gate)")
    p(f"       verb / item / qty                  {C['a3_expert_verb_rejected']} / "
      f"{C['a3_expert_item_rejected']} / {C['a3_expert_qty_rejected']}")
    p(f"  A3 market orders checked                {_fmt(C['n_market_orders'])}")
    p(f"  A3 n_expert_orders_rejected_by_mask     {C['n_expert_orders_rejected_by_mask']}  (gate)")
    p(f"  A4 SELL orders                          {_fmt(C['a4_sells'])}")
    p(f"  A4 over EFFECTIVE shed (gate)           {C['a4_sell_over_effective_shed']}")
    p(f"     ... static variant                    {C['a4_sell_over_effective_shed_static']}")
    p(f"     ... control: over OBSERVED shed       {_fmt(C['a4_sell_over_observed_shed'])}  "
      f"<- what the naive check would flag")
    p(f"     ... deposits / withdrawals            {_fmt(C['a4_deposits'])} / "
      f"{_fmt(C['a4_withdrawals'])}")
    p(f"     ... animal PLACE took the tile branch {_fmt(C['a4_animal_place_to_tile'])}  "
      f"<- proves the carve-out fired")
    p(f"  PLANT cliff: expert requests rewritten   {C['n_expert_plant_rewritten_to_pass']} "
      f"(of {C['n_plant_blocked_by_cliff']} blocked)")
    p(f"     ... planting bursts (2+ on one crop)   {_fmt(C['n_plant_bursts'])}")
    p(f"     ... turns sitting EXACTLY on the edge  {_fmt(C['n_turns_plant_demand_equals_seeds'])}"
      f"  <- why the decoder must be autoregressive")

    if r.get("opponent_counters"):
        O = r["opponent_counters"]
        p("\n-- the SAME assertions on the opponent seats (diagnostic, NOT a gate) --")
        p("   These 36 assorted agents are not the teacher, and they do issue illegal actions.")
        p(f"  actions rejected by mask   {_fmt(O['n_expert_actions_rejected_by_mask'])}"
          f" / {_fmt(O['n_unit_actions'])}")
        p(f"  orders rejected by mask    {_fmt(O['n_expert_orders_rejected_by_mask'])}"
          f" / {_fmt(O['n_market_orders'])}")
        p(f"  SELL over effective shed   {_fmt(O['a4_sell_over_effective_shed'])}"
          f" / {_fmt(O['a4_sells'])}")
        p(f"  roster fail / md5 fail     {O['a1_roster_fail']} / {O['a2_md5_fail']}")

    p("\n-- vocabulary (both seats, every episode) --")
    v = r["vocab"]
    p(f"  unit actions {_fmt(v['n_unit_actions'])}   market orders {_fmt(v['n_market_orders'])}")
    p(f"  verbs seen {len(v['verbs_seen'])}/{V.N_VERBS}   ops {len(v['market_ops_seen'])}/"
      f"{V.N_MARKET_OPS}   items {len(v['items_seen'])}/{len(V.ITEMS)}")
    p(f"  max units {v['max_units']} (cap {V.MAX_UNITS})   max orders {v['max_orders']} "
      f"(cap {V.MAX_MARKET_ORDERS})   over-cap {v['n_units_over_cap']}/{v['n_orders_over_cap']}")
    p(f"  max qty unit {v['max_unit_qty']}  market {v['max_market_qty']}   "
      f"ignored trailing args {_fmt(v['n_ignored_extra_args'])}")
    missing = sorted(set(V.VERBS) - set(v["verbs_seen"]))
    if missing:
        p(f"  verbs in the vocabulary but never used: {missing}")

    e = r["E87"]
    p("\n-- E87: do workers walk the shortest path? --")
    p(f"  frac_segments_shortest_path = {e['frac_segments_shortest_path']:.4f}  "
      f"[{e['wilson'][0]:.4f}, {e['wilson'][1]:.4f}]  over {_fmt(e['walking_runs'])} walking runs")
    p(f"  detours {_fmt(e['detours'])}   truncated-at-nightfall (excluded) "
      f"{_fmt(e['truncated_excluded'])}")
    p(f"  threshold 0.90  ->  {e['decision']}")

    p("\n-- majority-class floors, RECOMPUTED on Ryo's seat over this corpus --")
    p("   (PLAN_BC's 16.3% / 19.3% came from the two seats of one old sample game)")
    for k, f in r["floors"].items():
        if not f["n"]:
            continue
        p(f"  {k:26s} {f['floor'] * 100:5.2f}%  [{f['wilson'][0] * 100:5.2f}, "
          f"{f['wilson'][1] * 100:5.2f}]   n={_fmt(f['n']):>9s}  argmax={f['argmax']}")

    if "shard_size" in r:
        s = r["shard_size"]
        p("\n-- shard sizes --")
        p(f"  mean {s['mean_mb']:.2f} MB   median {s['median_mb']:.2f} MB   "
          f"max {s['max_mb']:.2f} MB   total {s['total_mb']:.0f} MB")
        p(f"  over the {s['threshold_mb']:.1f} MB/episode threshold: {s['over_threshold']}"
          f"/{s['n']}")

    p("\n" + ("  GATES PASS" if r["gates_pass"] else "  *** GATES FAIL: "
                                                     f"{ {k: n for k, n in r['gates'].items() if n} } ***"))
    p("=" * 78 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
