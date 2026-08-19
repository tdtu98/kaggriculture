# Running the champion agent

The current champion is **R2 + slot_align**: the round-2 evolved plan (80-gene vector, E77)
with the slot-alignment overlay enabled (E79/E81). It lives as data, not as a named file:
the vector sits in `results/s2_r2/state.json` and the harness registry builds the agent
from it on demand.

Everything below uses **this repo's venv** — never the system python or miniconda
(the venv pins `kaggle-environments==1.32.7`; miniconda is deliberately on 1.32.6 and
silently measures the old market curve, see E54).

## The champion's registry name

Agents are addressed by registry name. Plan-vector agents use the `vec:` prefix with the
80 floats comma-joined, plus `#const=value` overrides. Build the champion's name from the
stored search state:

```python
import json
from harness import registry

vec = json.load(open("results/s2_r2/state.json"))["best"]["vec"]   # 80 floats, gen-38 winner
CHAMPION = registry.const_name(registry.vec_name(vec), slot_align=1)
```

> The name contains commas, so it **cannot** be passed through `harness.run`'s `--agents`
> CLI flag (which splits on commas). Drive `harness.run.run()` from Python instead — that is
> the supported path and what every measurement in E77–E84 used.

Other useful registry names (see `python -m harness.run --list` for all):

| name | what it is |
|---|---|
| `boatlee` | the strongest known opponent (their downloaded submission, read-only) |
| `starter` | the competition's built-in baseline (~$3.5k solo) |
| `executor_v7` | our session-line fallback agent (~$74k solo) |
| `compiler` | our agent with the hand-written `Plan.boatlee_like()` (the pre-search incumbent) |
| `flooder`, `tomato_rusher` | in-family exploiter plans from S1 |

## Run a matchup (the standard way)

Runs through kagsim (bit-exact vs the reference env, ~300× faster). Both seats by default;
results append to `results/games.jsonl` with counters.

```python
# save as /tmp/run_match.py and run with .venv/bin/python
import json
from harness import registry, run

vec = json.load(open("results/s2_r2/state.json"))["best"]["vec"]
CHAMPION = registry.const_name(registry.vec_name(vec), slot_align=1)

OPPONENT = "boatlee"          # or "starter", "executor_v7", "compiler", ...
SEEDS    = list(range(80000, 80040))   # pick a block nobody has used (see note below)

results = run.run(
    names=[CHAMPION, OPPONENT],
    seeds=SEEDS,
    games=80,                 # >= 80 or the number is not believable (E33)
    both_seats=True,          # never measure one seat only
    config={"episodeSteps": 720},
)
```

The run prints the pairwise table (winrate, Wilson CI, mean money) and the counter block
(steps_per_useful, thirst, blocked_ops, fallbacks, ...). Read the counters before the money.

**Seed hygiene:** every measured conclusion in this repo is scoped to fresh seeds. Blocks
already spent: 21000–21040, 30100s, 40000–52040, 53000–53105, 54000–54080 (acceptance,
exhausted), 60000–68040 (search + O-track), 70000–77040 (O4/P-track), 90000–93300 (O1).
Pick something new (80000+ is clean at the time of writing) and check `results/games.jsonl`
if unsure.

## Run a single game through the real Kaggle engine

For a spot-check through the actual `kaggle_environments` runner (what Kaggle executes),
build the agent and adapt the observation — the env hands a `Struct`, the agent expects a
plain dict (the E21 family of traps):

```python
import json
from harness import registry
from kaggle_environments import make

vec = json.load(open("results/s2_r2/state.json"))["best"]["vec"]
agent = registry.get(registry.const_name(registry.vec_name(vec), slot_align=1)).build()

def to_plain(o):
    if hasattr(o, "items"):
        return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(v) for v in o]
    return o

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 76543}, debug=True)
env.run([lambda obs, cfg=None: agent(to_plain(obs)), "boatlee_path_or_builtin"])
print("money:", env.steps[-1][0]["reward"], env.steps[-1][1]["reward"])
```

For the second seat: `"starter"` works as a built-in; boatlee's source is
`reference/kaggriculture/1/submission.py` (pass that path). A finished game shows
`status: DONE` on both seats — a $3,000 final bank means the agent never acted
(usually the Struct/dict trap above, or a wrapper exception).

## Expectations (so you can tell a broken run from a real one)

| matchup | expected result |
|---|---|
| champion vs `starter` | ~$115–125k vs ~$3.5k, wins ~100% |
| champion vs `boatlee` | **loses 0/80**, ~$60–66k vs ~$120–130k (the known −$60k gap, E81/E84) |
| champion vs `executor_v7` | wins 100%, ~+$65k margin |
| champion vs `flooder` / `tomato_rusher` | wins ~97–100% |
| counters, any matchup | `fallbacks` 0 · `steps_per_useful` ≤ 0.8 · thirst ≤ ~3 · `blocked_ops` ≤ ~3 |

If `fallbacks` is non-zero or money sits at $3,000, the run is broken, not the agent —
check the venv, the observation adapter, and that no repo file was half-edited.

## Flags worth knowing

All plan constants can be overridden in the registry name, e.g.
`#slot_align=1#planner=0`. The measured guidance:

- `slot_align=1` — **on** (champion default): +$500–800/game vs boatlee, provably inert vs others (E79/E81).
- `branch_set=all` — off on R2 (measured inert there); worth +$2.8k only on the `compiler` plan (E80).
- `frontrun`, `counter_mix` — **keep off**: refuted with dose-response evidence (E79).
- `planner`, `planner_value` — **keep off**: both planner variants killed (E83/E84).

## Provenance of the champion

Genome: `results/s2_r2/state.json → ["best"]["vec"]` (generation 38, score 0.6866).
Gate record: acceptance 79/80 vs the incumbent (E77), O4 operational pass + counters (E81).
Full history: `docs/experiments.md` E76–E84; task map: `TASKS_v4.md`.
