# The Claude-session executor line

`PLAN_v4.md`'s first line says it synthesises "the Claude-session executor line, the kagsim/CEM/relay
line (E1–E53), the decoded Boatlee table, and the env source". This directory is that first input.
It was written in a Claude session sandbox and existed nowhere in this repo, which is why
`TASKS_v4.md` refers to files (`executor.py`, `planner.py`) that a fresh clone did not contain.

The sources are **verbatim**. Nothing here has been edited — the sandbox paths are repointed at
load time by `_load.py` instead, so the code that produced the session-line measurements stays
byte-identical and the one substitution being made is visible in a single place.

```python
from session_line import load
executor = load("executor").agent      # -> agent(obs)
```

## What each file is

| file | lines | what it is | plan role |
|---|---:|---|---|
| `executor.py` | 426 | Closed-loop executor v2: crops fund the early game, animals bought from surplus at ≤1/day, hands scaled to tile count with a 45%-of-cash wage guard | **The whole of Track F.** `TASKS_v4` F1–F5 are precision fixes to *this* file, and the F-gate ships it as the fallback submission. Pool name `executor_v7` |
| `planner-2.py` | 121 | Adaptive allocator: value-density capped by empirical market **absorption** (`ABSORB`), with a 1.3x nudge toward products this game's shops demand | The ancestor of C6 (`agent/projection.py`). Its `ABSORB` table is the pre-hinge guess that §2.7 replaces with a real projection |
| `planner.py` | 117 | The earlier allocator: greedy on *current shop demand* rather than absorption | Superseded by `planner-2`. Kept because the audit cites "planner.py" as carrying the stale market table |
| `warfare.py` | 67 | Opponent-aware market layer: front-runs the premium goods the opponent is about to flood, diverts sells to lanes they under-supply, passes a true mirror through unchanged | The prototype of O3 (front-run + counter-mix) and of `agent/opponent.py`'s fingerprint |
| `metered_LOSES.py` | 66 | Meters premium sells against market headroom, with guaranteed terminal liquidation | Measured to lose; the `_LOSES` suffix is the verdict. §2.6 keeps market rules simple because of this |
| `denial_LOSES.py` | 51 | Real denial: withhold premium goods, dump the reserve when the opponent's farm shows ripe premium | Measured to lose (−40k, E11). `PLAN_v4` §2.5 turns denial **off** on this evidence |
| `harness.py` | 31 | The session harness: candidate vs champion across seeds, both seats | Superseded by I2, which adds ≥80 games, fresh blocks, CIs and the counter suite. Kept so the session-line numbers stay attributable |

## What they depend on

All five agent/planner files load Boatlee at import time from `/home/claude/main.py`:

```
executor.py ──> main.py (helpers _get/_seat/_farm, _market_price, _MARKET_PARAMS)
           └──> planner.py  (allocation)  ── which itself loads main.py

warfare.py ─┐
metered  ───┼──> main.py — and these three call stock.agent(obs), so they ARE Boatlee
denial   ───┘         plus a market-channel overlay, same construction as `relay-sell`
```

`_load.py` maps `/home/claude/main.py` to `reference/kaggriculture/1/submission.py`, which is
sha256-identical to the `main.py` these were written against. Two consequences worth stating
plainly:

* **Only `executor.py` has a policy of its own.** It borrows Boatlee's accessor helpers and price
  model, but its production loop is its own code. `warfare`, `metered` and `denial` are Boatlee
  with an overlay, so `PLAN_v4` §5's provenance question applies to them exactly as it does to
  `relay-sell` — they are measurement inputs here, not shippable agents.
* **They price with Boatlee's `_MARKET_PARAMS`, which is stale.** Carrot, tomato and egg went
  `hinge` in 1.32.7 (E54); this table still has them as `log`/`linear`, so everything in this
  directory under-values scarcity in those three by up to 5x. That is the audit's §1 warning, and
  it is *not* fixed here — fixing it would mean editing the sources. Track F does it as part of
  F5.

## Two traps, both already sprung once

**1. `executor.py` needs `planner-2.py`, not `planner.py`.** It calls `planner.plan(obs, n, pool=…)`
and only planner-2 has `pool`. Against the older file every turn raises `TypeError`.

**2. `executor.py` swallows every exception and returns all-PASS.** Its `agent()` ends with a bare
`except Exception:` that returns PASS for the farmer, PASS for each hand and an empty market. So
trap 1 does not crash — it produces a complete, well-formed season of doing nothing, ending on the
$3,000 starting bank. Wired up naively, that reads as "the executor scores $3k" rather than "the
import is wrong". `tests/test_session_line.py` asserts the executor actually acts, so this cannot
recur silently.

Smoke, 4 seeds vs `starter` on kagsim, **not evidence** (the bar is 80 games, both seats, fresh
block): $71,592 / $81,131 / $67,384 / $82,379 — consistent with the 30–75k solo band `PLAN_v4` §1
attributes to this line.

## Not imported

`~/Downloads/main.py` is byte-identical to `reference/kaggriculture/1/submission.py` (already here),
and `~/Downloads/README.md` / `AGENTS.md` are identical to `docs/`. `~/Downloads/91192902.json`
(29 MB) is unexamined — the name and size suggest a Kaggle episode replay, which would belong to L2.
