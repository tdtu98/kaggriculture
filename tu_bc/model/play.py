"""Chapter 6's evaluation ladder: run the wrapped checkpoint against a named opponent.

    PYTHONPATH=. .venv/bin/python -m model.play --opponent starter --games 2  --seeds 40000:40001
    PYTHONPATH=. .venv/bin/python -m model.play --opponent boatlee --games 80 --seeds 41000:41040

Three rules from CLAUDE.md are the tool's behaviour rather than flags to remember:

* **Both seats, always.** Every seed is played twice, once in each seat, so seat order cancels.
  `--games 80` therefore means 40 seeds.
* **A winrate is printed with its Wilson interval.** Five promotions in a row were wrong because a
  3-8pp edge was read off a sample that could only resolve 12pp (E18/D19).
* **Counters print next to the money.** A change that never fired and a strategy that does not work
  produce identical money (E44), so the wrapper's own counters and `harness/counters.py`'s
  `Observer` are reported with every run.

Two backends, and the difference matters:

* `--backend kagsim` (default) is the Rust re-implementation the whole harness runs on
  (`make verify` is its parity gate).  It is ~10x faster and it is the only one `Observer` can
  read, because `Observer.finish` needs `sim.stats`.
* `--backend env` is `kaggle_environments` itself -- the ground truth, and the only one that can
  report a `status` of `DONE` or `ERROR`.  Rung 2's liveness gate is run here.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

CONFIG = {"episodeSteps": 720}
_WORKER: dict = {}


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------

def _build(checkpoint, claims, device):
    """Build the BC agent, reusing a per-process loaded checkpoint."""
    from model import agent as A

    key = ("model", checkpoint, device)
    if key not in _WORKER:
        _WORKER[key] = A.load_model(checkpoint, device=device)
    model, _payload, torch_mod = _WORKER[key]
    return A.make_agent(None, model=model, torch_mod=torch_mod, claims=claims, device=device)


def _opponent(name):
    from harness import registry

    key = ("opp", name)
    if key not in _WORKER:
        _WORKER[key] = registry.get(name)
    return _WORKER[key].build()


def play_kagsim(checkpoint, opponent, seed, seat, claims="tile", device="cpu", observe=True):
    """One episode on kagsim.  `seat` is which side the BC agent plays."""
    import kagsim
    from harness.counters import Observer

    me = _build(checkpoint, claims, device)
    them = _opponent(opponent)
    agents = [None, None]
    agents[seat] = me
    agents[1 - seat] = them
    obs_watch = Observer(player=seat) if observe else None

    t0 = time.perf_counter()
    sim = kagsim.Sim({**CONFIG, "seed": seed})
    sim.collect_stats = True
    for _ in range(CONFIG["episodeSteps"] - 1):
        actions = []
        for p in range(2):
            o = sim.observation(p)
            a = agents[p](o)
            if p == seat and obs_watch is not None:
                obs_watch.observe(o, a)
            actions.append(a)
        sim.step(actions)
    seconds = time.perf_counter() - t0

    return {
        "seed": seed, "seat": seat, "backend": "kagsim",
        "money": float(sim.money(seat)), "opp_money": float(sim.money(1 - seat)),
        "status": "DONE", "seconds": seconds,
        "counters": dict(me.counters), "timing": me.timing(),
        "observer": obs_watch.finish(sim, me) if obs_watch is not None else {},
    }


def play_env(checkpoint, opponent, seed, seat, claims="tile", device="cpu"):
    """One episode on `kaggle_environments` -- the ground truth, and the only `status` we trust."""
    from kaggle_environments import make

    me = _build(checkpoint, claims, device)
    them = _opponent(opponent)
    agents = [None, None]
    agents[seat] = me
    agents[1 - seat] = them

    t0 = time.perf_counter()
    env = make("kaggriculture", configuration={**CONFIG, "seed": seed}, debug=True)
    env.run(agents)
    seconds = time.perf_counter() - t0
    final = env.steps[-1]
    return {
        "seed": seed, "seat": seat, "backend": "env",
        "money": float(final[seat]["reward"]), "opp_money": float(final[1 - seat]["reward"]),
        "status": str(final[seat]["status"]),
        "opp_status": str(final[1 - seat]["status"]),
        "seconds": seconds,
        "counters": dict(me.counters), "timing": me.timing(), "observer": {},
    }


def _job(spec):
    checkpoint, opponent, seed, seat, claims, device, backend = spec
    fn = play_env if backend == "env" else play_kagsim
    try:
        return fn(checkpoint, opponent, seed, seat, claims=claims, device=device)
    except Exception as exc:                                   # noqa: BLE001 -- reported, not hidden
        import traceback

        return {"seed": seed, "seat": seat, "backend": backend, "money": float("nan"),
                "opp_money": float("nan"), "status": f"EXCEPTION: {exc}",
                "traceback": traceback.format_exc(), "seconds": 0.0,
                "counters": {}, "timing": {}, "observer": {}}


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def summarise(rows, label=""):
    from arena.stats import Winrate

    ok = [r for r in rows if r["money"] == r["money"]]        # drop NaN (crashed) episodes
    wins = sum(1.0 if r["money"] > r["opp_money"]
               else 0.5 if r["money"] == r["opp_money"] else 0.0 for r in ok)
    wr = Winrate(wins, len(ok))
    mine = [r["money"] for r in ok]
    theirs = [r["opp_money"] for r in ok]

    out = [f"\n{'=' * 78}", f"{label}  ({len(ok)}/{len(rows)} episodes completed)", "=" * 78]
    out.append(f"winrate      {wr}")
    for seat in (0, 1):
        sub = [r for r in ok if r["seat"] == seat]
        if sub:
            w = sum(1.0 if r["money"] > r["opp_money"]
                    else 0.5 if r["money"] == r["opp_money"] else 0.0 for r in sub)
            out.append(f"  seat {seat}      {Winrate(w, len(sub))}   "
                       f"mean money {statistics.mean(r['money'] for r in sub):>10,.0f}")
    if mine:
        out.append(f"my money     mean {statistics.mean(mine):>10,.0f}  "
                   f"median {statistics.median(mine):>10,.0f}  "
                   f"min {min(mine):>10,.0f}  max {max(mine):>10,.0f}")
        out.append(f"opp money    mean {statistics.mean(theirs):>10,.0f}  "
                   f"median {statistics.median(theirs):>10,.0f}  "
                   f"min {min(theirs):>10,.0f}  max {max(theirs):>10,.0f}")
        out.append(f"above $3,000 starting bank: {sum(1 for m in mine if m > 3000)}/{len(mine)}")

    bad = [r for r in rows if r["status"] != "DONE"]
    out.append(f"non-DONE episodes: {len(bad)}" + (f"  {bad[0]['status'][:120]}" if bad else ""))

    keys = sorted({k for r in ok for k in r["counters"]})
    if keys:
        out.append("\nwrapper counters (total over all episodes, and per episode):")
        for k in keys:
            tot = sum(r["counters"].get(k, 0) for r in ok)
            out.append(f"  {k:<32} {tot:>10,}   {tot / max(1, len(ok)):>10,.1f}/ep")
    times = [r["timing"] for r in ok if r.get("timing")]
    if times:
        out.append(f"\nturn time    mean {statistics.mean(t['mean_ms'] for t in times):.2f} ms   "
                   f"p99 {max(t['p99_ms'] for t in times):.2f} ms (worst episode)   "
                   f"max {max(t['max_ms'] for t in times):.2f} ms")
        out.append(f"episode wall mean {statistics.mean(r['seconds'] for r in ok):.1f} s")

    obs_keys = ("idle_pct", "steps_per_useful", "blocked_ops", "plants_started", "plants_lost",
                "plants_lost_thirst", "unharvested_ripe_at_end", "shed_overflow_discarded",
                "fertilize_ops")
    have = [r["observer"] for r in ok if r.get("observer")]
    if have:
        out.append("\nharness Observer (mean over episodes):")
        for k in obs_keys:
            vals = [h[k] for h in have if isinstance(h.get(k), (int, float))]
            if vals:
                out.append(f"  {k:<32} {statistics.mean(vals):>12,.3f}")
        prod = {}
        for h in have:
            for item, n in (h.get("produced") or {}).items():
                prod[item] = prod.get(item, 0) + n
        sold = {}
        for h in have:
            for item, n in (h.get("sold_revenue") or {}).items():
                sold[item] = sold.get(item, 0) + n
        if prod:
            out.append("  produced/ep   " + "  ".join(
                f"{k}={v / len(have):.0f}" for k, v in sorted(prod.items()) if v))
        if sold:
            out.append("  revenue/ep    " + "  ".join(
                f"{k}=${v / len(have):,.0f}" for k, v in sorted(sold.items()) if v))
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_seeds(spec):
    if ":" not in spec:
        raise SystemExit("--seeds takes a block like 41000:41040")
    lo, hi = spec.split(":", 1)
    seeds = list(range(int(lo), int(hi)))
    if not seeds:
        raise SystemExit(f"empty seed block {spec}")
    return seeds


def build_schedule(seeds, games, both_seats=True):
    per_seed = 2 if both_seats else 1
    need = -(-games // per_seed)
    if need > len(seeds):
        raise SystemExit(f"--games {games} over {per_seed} seat(s)/seed needs {need} seeds, the "
                         f"block has {len(seeds)}. Widen --seeds rather than reusing one.")
    out = []
    for s in seeds[:need]:
        out.append((s, 0))
        if both_seats:
            out.append((s, 1))
    return out[:games]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/bc_v1.pt")
    ap.add_argument("--opponent", default="starter",
                    help="a name from harness/registry.py, e.g. boatlee / starter / executor_v7")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--seeds", default="41000:41040")
    ap.add_argument("--backend", choices=("kagsim", "env"), default="kagsim")
    ap.add_argument("--claims", choices=("off", "verb", "tile"), default="tile",
                    help="how an outstanding macro's target is protected; see model/agent.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1, 1 = in-process")
    ap.add_argument("--one-seat", action="store_true", help="seat 0 only (debugging; never report)")
    ap.add_argument("--no-observer", action="store_true")
    ap.add_argument("--jsonl", default="", help="append every episode row here")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    schedule = build_schedule(seeds, args.games, both_seats=not args.one_seat)
    specs = [(args.checkpoint, args.opponent, s, seat, args.claims, args.device, args.backend)
             for s, seat in schedule]

    label = args.label or (f"bc_v1[{os.path.basename(args.checkpoint)}, claims={args.claims}] "
                           f"vs {args.opponent}  ({args.backend}, seeds {args.seeds})")
    print(f"{len(specs)} episodes  |  {label}", flush=True)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    rows = []
    t0 = time.perf_counter()
    if workers == 1:
        it = (_job(s) for s in specs)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers)
        it = pool.imap_unordered(_job, specs)
    for i, r in enumerate(it, 1):
        rows.append(r)
        flag = "WIN " if r["money"] > r["opp_money"] else ("draw" if r["money"] == r["opp_money"]
                                                           else "loss")
        note = "" if r["status"] == "DONE" else f"  !! {r['status'][:80]}"
        print(f"  [{i:>3}/{len(specs)}] seed {r['seed']} seat {r['seat']}  "
              f"me {r['money']:>10,.0f}   opp {r['opp_money']:>10,.0f}   {flag}"
              f"   {r['seconds']:.1f}s{note}", flush=True)
    if workers != 1:
        pool.close()
        pool.join()
    print(f"\n{len(rows)} episodes in {time.perf_counter() - t0:.1f}s")

    print(summarise(rows, label))

    if args.jsonl:
        os.makedirs(os.path.dirname(args.jsonl) or ".", exist_ok=True)
        with open(args.jsonl, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "opponent": args.opponent,
                                     "checkpoint": args.checkpoint, "claims": args.claims,
                                     "seed_block": args.seeds}) + "\n")
        print(f"\nrows appended to {args.jsonl}")

    bad = [r for r in rows if r["status"] != "DONE"]
    if bad:
        print("\nfirst failure traceback:\n" + bad[0].get("traceback", "(none)"), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
