# PLAN_BC — the detailed companion to `JOURNEY.md`

`JOURNEY.md` is the plan we follow. It has nine chapters. **This file has the same nine chapters,
in the same order, with the fine print**: exact numbers, exact file and line citations, the traps
that have already cost this project measurements, the checks that catch them, and the rule for
when to stop.

**How to read it.** Every chapter is laid out the same way:

| Part | What it is |
|---|---|
| **The idea** | Plain words. What we are doing and why. No jargon without a gloss. |
| **What we build** | The files and the concrete work. |
| **The details** | The technical fine print — numbers, citations, traps, checks. |
| **How we know we're done** | The exit test. Passing it means move to the next chapter. |
| **When we stop** | The kill rule. Hitting it means this chapter's approach is wrong — do not tune, go back or go around. |
| **What you should be able to explain** | The learning exit. Write it in your own words in `docs/learning.md`. |
| **Reading** | Two or three canonical sources, if you want the theory. |

**Two goals, weighted equally, and either can succeed while the other fails.**

1. **Competitive** — a learned model that beats the sitting champion through `make promote`.
2. **Learning** — you (solid ML, new to RL and imitation learning) end up genuinely understanding
   imitation learning and reinforcement learning.

That is why every chapter has both an exit test *and* a "what you should be able to explain".
**They are killed independently.** A chapter whose competitive exit fails but whose learning exit
passes is **not** wasted work. Saying this up front matters, because otherwise a bad arena number
quietly rewrites what the evening was for.

### Two tags on every number

- **[MEASURED, verified]** — someone measured it, and a second agent independently re-derived it on
  this machine without looking at the first one's code. Trust these.
- **[MEASURED, single-source]** — measured once, by one person, not re-checked. Treat as a
  hypothesis. Re-run before building anything on it.

Env citations are `kaggriculture.py:NNN` (shortened to `kag.py`), against
`.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`.
That file is the ground truth for every game rule in this document.

### D22 — the standing decision about this whole line

Three statements, then we never argue about them again.

**(i) What it costs.** `PLAN_v4`'s compiler line is the competitively optimal use of the same
evenings. This BC line is expected to **trail it on the leaderboard for roughly the first six
weeks**. `PLAN_v4` §5 explicitly rejects both halves of this program — "deep RL from scratch (env
speed, action space…)" and "pure behaviour cloning (caps at a tie)" [MEASURED, verified: quoted
verbatim from the file]. Both objections are answered rather than ignored, and Chapters 2 and 8 say
how.

**(ii) How the two lines help each other.** Every model we save registers as an opponent in the
existing arena: `arena/registry.py` `AgentSpec` with `kind: "bc"` and
`params: {"checkpoint": path}`. Specs must be **plain picklable data — never a loaded model, never
a closure** — because arena workers start with the `spawn` method (`arena/registry.py:24` and its
docstring) [MEASURED, verified]. So every BC checkpoint strengthens the compiler line's opponent
pool for free, and the compiler line gives us a teacher (Chapter 7) and a fallback data source
(Chapter 2).

**(iii) The stop-loss.** **This line never touches `main.py`.** The live submission stays owned by
the compiler line for the whole program. A BC model reaches `main.py` only by winning
`make promote` (D19), like any other candidate.

> Record as **D22** in `docs/decisions.md`.

### Ground rules (they apply in every chapter)

1. **Never judge a change on fewer than 80 games**, both seats, on a seed block never used for
   tuning. Use `harness/run.py` — its docstring already states this floor [MEASURED, verified].
   Three separate results in this project looked real at 16–48 games and vanished at 80+
   (E37→E41, E39→E40, E42).
2. **Every accuracy number is printed next to its dumb baseline**, with an error bar (Chapter 4).
3. **Every step carries a counter that proves it actually ran.** We never trust, we check. A zero
   counter is an unfinished implementation, not a negative result (E44). For a *playing* model the
   instrument is `harness/counters.py` `Observer` (`:53`, keys emitted at `:193–198`): `idle_pct`,
   `steps_per_useful`, `blocked_ops`, `plants_lost_thirst` [MEASURED, verified]. A model that
   silently does nothing shows up as one log line instead of four wasted weekends.
4. **Name the opponent.** Never rank against `starter` alone ($3,507 at 1.32.7). It is a
   liveness floor, never a ranking.
5. Nothing ships except through `make promote`.

*(Where this came from: a five-agent panel — three proposers, two verifiers. The original panel
document is preserved verbatim at `docs/PLAN_BC_panel_original.md`. Its phases map to chapters
like this: P0 → Chapters 1–2, P1 → Chapter 3, P2 → Chapter 4, P3 → Chapters 5–6, P4 → Chapter 7,
P5 → Chapter 8, P6 → Chapter 9.)*

---

# Chapter 1 — Meet the game and the data

## The idea

Before writing code we learn three words, because everything else is built from them.

- **State** — everything the player can see this turn: the farm tiles, the market, the money, the
  day and hour. In the data file this is called the **observation**.
- **Action** — what the player does this turn: one instruction per worker, plus up to ten market
  orders.
- **Policy** — the rule that turns a state into an action. Code, or a neural network. "Training a
  model" means "learning a policy".

**Behaviour cloning (BC)** is one sentence: collect (state, action) pairs from an expert, then use
ordinary supervised learning to predict the action from the state. No rewards. No exploring. Copy
the master.

**Supervised learning** means: we have inputs (states) and correct answers (the expert's actions,
called **labels**), and we adjust the network so its guesses match the labels.

## What we build

Nothing yet. We read, poke at the sample file, and set up the toolchain.

1. Open `data/kaggriculture/95029942.json`. Find one step. Point at the observation. Point at the
   action. See that the action has a `farmer` part, a `hands` list, and a `market` list.
2. Skim `Orbit-Wars/README.md`. Their recipe is replays → BC → PPO self-play. That is our map.
3. Install `torch` into `.venv`, then re-run `make test` and `make verify` and confirm they are
   still green.

## The details

### The sample game

`info.TeamNames == ["Ryo Hasegawa", "tetsuya"]`, rewards **$104,996 / $90,833**, both
`statuses == "DONE"`, `module_version 1.32.7`, `info.seed 746105689` [MEASURED, verified].
The `starter` bot makes **$3,507**, so both of these players are 26–30× the baseline.

**But we do not know if they are top-K, and we do not know if they react to the game at all.** A
`TeamName` is a human; the *agent* behind it is code. Chapter 2 explains why that distinction is
the single most dangerous thing in this plan.

What is in one file [MEASURED, verified]:

| Quantity | Value |
|---|---|
| Steps / usable (state, action) pairs | 720 / **719 × 2 seats = 1,438** |
| Unit decisions (one per worker per turn) | **15,057** (7,270 seat 0 + 7,787 seat 1) |
| — of those, pure walking | **7,168 (47.6%)** |
| — of those, real decisions | **7,889** ≈ **3,945 per player per game** |
| Market orders | 1,218 (660 + 558) |
| Distinct worker ops / market ops | **17** / **6** |
| Distinct complete actions (seat 0 / 1) | 35 / 33 |
| Most workers at once | 12 hands + farmer = **13** |
| Raw JSON / gzip -9 | 31,011,787 bytes / **394,523 bytes (78.6× smaller)** |

### Game rules that shape the design (all confirmed against the source)

Each of these changes something later, so learn them now [all MEASURED, verified]:

- **Market happens after workers move.** `_process_market` runs at `kag.py:941`, after the loop
  that applies unit actions (`kag.py:935-941`). So a worker can drop wheat into the shed and the
  same turn's sell order can sell it. This forces our decode order in Chapter 3.
- **The worker roster is wiped every night.** `_end_of_day` sets `farm["hands"] = []`,
  `hires_today = 0`, and clears carried inventories (`kag.py:879-882`). Hiring restarts each day.
- **Hiring is all-or-nothing on planting.** If the total `PLANT` requests for one crop this turn
  exceed the seeds you own, **every one of them is dropped**, not just the excess
  (`kag.py:920-933`). This is a cliff, and Chapter 5 explains why the model must be built to avoid
  falling off it.
- **Hiring costs a Fibonacci curve.** `_hire_cost(n) = mult * _fib(n)` (`kag.py:690-706`), so the
  16th hire in one day is still cheap in total.
- **You can walk over LOCKED tiles.** `kag.py:326-331` bounds-checks only, and the source comment
  says so. Walking is never blocked. Chapter 5 depends on this.
- **You can only issue 10 market orders a turn, and extras are silently dropped.**
  `max_orders` at `kag.py:551`, `queues.append(q[:max_orders])` at `:560`. Silently — no error.
- **You can only *buy* WHEAT and FERTILIZER.** `kag.py:598` restricts `BUY_PRODUCT` to exactly
  those two.

### The machine

- **`kagsim` (our fast re-implementation of the game) is Rust**, not Python
  (`kagsim/Cargo.toml`, pyo3, built as a `.so`) [MEASURED, verified]. Speed: **2.20 ms per whole
  game = 326,000 steps/second** with pre-built actions, and **13.93 ms = 51,600 steps/second** when
  it also builds both players' observation dictionaries. Dictionary building is **84.2%** of the
  cost. The official Python environment takes **1.95 seconds per game** — 140× slower.
- **Therefore we do not need to rewrite the simulator.** All six Orbit Wars top-10 finishers
  rewrote their environment; ours is already rewritten and fast. `docs/decisions.md:59`'s "860
  steps/s… not viable" verdict was about *staying on the Python environment* and is now stale
  [MEASURED, verified]. **Do not port kagsim to JAX. Do not build the Rust `VecEnv`
  (`TASKS.md:435`).** The only env-side work that would pay is replacing the observation-dictionary
  path with a direct tensor path.
- **`torch` is already installed at `/opt/miniconda3/bin/python` (2.13.0, with MPS working) but not
  in `.venv`** [MEASURED, verified]. So this is an install task, not a blocker. **Warning:
  miniconda is deliberately pinned to `kaggle-environments==1.32.6` (E54).** That is fine for pure
  PyTorch benchmarking and **wrong for anything that touches the market curve** — the market
  changed between versions.

## How we know we're done

You can point at one step of the JSON and say: this is the state, this is the action, and the
policy is the invisible thing that connected them. `torch` imports inside `.venv`, and `make test`
and `make verify` are both green.

## When we stop

Nothing to kill here. This chapter cannot fail; it can only be skipped too fast.

## What you should be able to explain

What a state, an action, and a policy are, in your own words, using this game as the example. And:
why behaviour cloning is *just* supervised learning — but with a test set that the model itself
creates, which is the problem Chapter 6 is entirely about.

## Reading

- Pomerleau, **ALVINN: An Autonomous Land Vehicle in a Neural Network** (NeurIPS 1988). Five pages.
  The original behaviour-cloning system — a van that learned to steer by watching a human. Its
  "record what steering back from the edge looks like" hack is exactly the problem we hit in
  Chapter 6.
- Levine, **CS285, Lecture 2, *Supervised Learning of Behaviors*** (lecture numbering drifts by
  year). The standard picture of why naive BC drifts off the road.

---

# Chapter 2 — The data (already provided)

## The idea

A network needs many examples. **We have them.** A real training corpus was handed to us: 100
Kaggle games, pre-split, with an index file. This chapter is therefore no longer about acquiring
data — it is about **knowing exactly what we were given**, checking it, and understanding the one
way this particular dataset could still mislead us.

## What we build

Nothing downloads. We verify the corpus, read `manifest.csv`, and record the two clock-trap
measurements. `bc/sources/replay.py` reads from the provided directory.

## The details

### What is in `data/sample_data_training_model/`

All facts in this section are [MEASURED, single-source] — measured once, this session, on the
provided files.

| Property | Value |
|---|---|
| Size / files | **2.9 GB, 100 episode JSONs** |
| Splits (pre-made) | `train/` **70**, `val/` **15**, `test/` **15** |
| Index files | `manifest.csv` (one row per episode), `split_summary.json` |
| Filenames | the `EpisodeId`, matching `info.EpisodeId` |
| Schema | all 100 match the replay schema in Chapter 3 **exactly** |
| Version | all **`module_version 1.32.7`** — no version skew to handle |
| Status | all `statuses == "DONE"` on both seats |
| Shape | 720 steps × 2 seats in every file |
| Seeds | `info.seed` unique per file — no duplicate games |
| Overlap | `train/95029942.json` is **byte-identical** to `data/kaggriculture/95029942.json`, the sample file used throughout Chapter 1 |

`manifest.csv` columns: `episode_id, split, source_date, source_path, source_sha256, opponent,
ryo_seat, ryo_reward, opponent_reward, margin, margin_quartile, shop_profile, route_family`.

### This is a single-teacher dataset

**"Ryo Hasegawa" appears in all 100 episodes and won all 100** [MEASURED, single-source].

| | |
|---|---|
| Ryo's reward | min **$48,467** / median **$91,697** / max **$165,959** |
| `starter` baseline | $3,507 |
| Ryo's seat | varies per game — the `ryo_seat` column says which |
| Distinct opponents | **36**, long-tailed: `tetsuya` 25, `カワシギ` 10, `Arman Tuganbaev` 9, … and **22 opponents appear exactly once** |
| `boatlee` | appears **once**, as an opponent Ryo beat ($131,000 vs $124,469, episode 94436914, train split) |

So the task is not "learn how strong players play in general". It is **"clone Ryo"**. That is a
narrower and easier problem, and it is the right one to start with. Everything downstream follows
from it — the split design, the label emission in Chapter 3, and what the evaluation in Chapter 6
is entitled to claim.

### The curation, and the one bias it introduces

The 100 games were **selected from 126 candidate wins**, stratified across `opponent`, `ryo_seat`,
`source_date`, `margin_quartile`, `shop_profile` and `route_family` (those derived columns are in
`manifest.csv`) [MEASURED, single-source]. The stratification is good: it means we are not
accidentally training on 100 games against one opponent, or 100 games with the same shop draw.

**But the corpus is win-filtered, and that has a consequence worth stating plainly.**

> Ryo only ever appears winning. So the model will rarely, if ever, see what a *losing* position
> looks like, or what Ryo does to climb out of one. States where the game is going badly are
> underrepresented in training — and those are exactly the states our own model will reach first
> when it makes a mistake. That is Chapter 6's distribution-shift problem, made slightly worse by
> the data selection.

We do not fix this now. We note it, and we expect it to show up as a steeper handover curve
(Chapter 6) and as the thing DAgger (Chapter 7) or PPO (Chapter 8) has to repair.

### The splits are by GAME, not by player — and that is correct here

The provided split holds out **whole games**, not players. Chapter 2 of an earlier draft argued for
holding out *agents* instead. Both are right, for different datasets:

- **Held-out agents** answers "does this generalize to players we have never seen?" That question
  only exists if we train on multiple teachers.
- **Held-out games** answers **"can we reproduce Ryo's play in games they have never played?"** With a
  single teacher, that is the honest and the only meaningful question — and it is exactly what the
  provided split measures.

So: **use the provided split as-is.** The held-out-agent design stays in this plan only as the rule
to apply *if* we ever add a second teacher.

What still holds without change: **never split by step.** Two neighbouring steps in one game share
~99% of the farm, so a random step split measures "can you recall a nearly identical row", not "can
you play". The provided split already avoids this by construction, since it splits whole files.

### ⚠ The clock trap — the risk that could have killed this line, and the check that retired it

Some top agents in this competition **do not look at the board**. `boatlee`, a real leaderboard
agent, replays a fixed 719-move list. Measured over six games against our champion: unit actions
identical to the stored script at **100%, 100%, 99.9%, 100%, 99.8%, 100%**, and **zero** market
orders added, removed, or resized by its supposedly reactive layers [MEASURED, E48].

Now think about what happens if we train on an agent like that. The state changes between its
games (different seed, different opponent). The label does **not**. The mathematically correct
thing for gradient descent to do is **ignore the state entirely and predict from the step number**.
We would get a beautiful offline accuracy score, and a model that is a slower, lossier copy of a
lookup table this repo already stores exactly and for free.

**Offline accuracy cannot detect this.** It has to be measured on the data, before training.

### Test A — does the teacher vary between their own games? **PASSED** (E88)

Two measurements, both [MEASURED, single-source], both on the provided corpus:

**(a) Hash every player's complete action sequence, across all of their episodes.**
Result: **zero duplicate hashes anywhere in the 100 games**, for any player. Nobody in this corpus
submits the same tape twice.

**(b) Compare Ryo's games against each other, step by step.** 24 randomly sampled Ryo games,
compared as **12 disjoint pairs**, testing whole-action equality at each step.
Result: **74%–96% of steps differ, median 96%.** Even a pair played against the *same* opponent
(`tetsuya`) differed on **75%** of steps.

> **Verdict: Ryo is a genuine reactive teacher. The tape-recorder risk is retired for this
> dataset.**

**And one structural detail fell out of it, which we use later.** The steps that *are* identical
across a pair concentrate almost entirely at **indices 0–31** — verified on a pair that shared
exactly steps 0–31 plus the final step 719. So:

> **Ryo plays a fixed ~30-step opening routine, then plays reactively for the remaining ~690
> steps.**

Those first ~30 steps are nearly free to predict and carry close to zero learning signal. Chapter 4
and Chapter 5 both report metrics **with and without them**, so that a healthy-looking headline
number is not just the opening being memorized.

*(Method note, kept because the earlier draft got this wrong: this is a proper **pairwise** test —
game A against game B, step by step. Comparing every game against a single reference, or comparing
aggregate action counts, would not distinguish "reactive" from "same script, different order".)*

### Test B — is there state-driven structure at all? **Still to run** (E86)

Test A proves Ryo's actions *change* between games. It does not prove those changes are driven by
the **state** rather than by, say, the random seed leaking through the opponent. So the deeper
check still runs exactly as designed:

Fit two tiny models on the same corpus. Model 1 sees **only the step number**. Model 2 sees
**state features with the step number removed**. Compare held-out agreement.
**Run it on `boatlee` first**, where we already know the answer from E48 — that validates the
harness before we trust it on Ryo.

Test A asks "does this *agent* vary?" Test B asks "is the variation *state-driven*?" Passing A is
necessary and not sufficient; Chapter 6's anti-clock ablation (E90) is the same question asked of
the trained model.

### Where we land on the yield ladder: the top row

The earlier draft planned to branch on how many games we could get. We are in the best case and can
stop thinking about it:

| Measured yield | What we do | Us? |
|---|---|---|
| **≥100 games, one verified reactive teacher** | Train on the provided split as-is. | **← we are here (100)** |
| 20–100 games | Train anyway, and write "generalization is weakly measured" on every result. | — |
| <20 games, or no usable teacher | Switch teacher to the compiler. | — |

*(The original ladder's top row demanded "≥200 games, ≥10 agents" because it assumed a
multi-teacher corpus and a held-out-agent split. With a single teacher that requirement does not
apply; 100 games of one verified reactive teacher is the stronger position for the question we are
actually asking.)*

### Only if we need more data: the Kaggle API

**Skip this section unless the corpus turns out to be too small.** It is preserved because it is
the recovery path, not because it is on the critical path.

Four unknowns, none checkable offline. One evening, with a script you are happy to delete:

| ID | Unknown | Cheapest way to settle it |
|---|---|---|
| T0-a | Does the Kaggle Episodes API cover this competition? | `POST .../competitions.EpisodeService/ListEpisodes` with `{"teamId": <int>}`. Look for HTTP 200 and an `episodes[]` array. |
| T0-b | What does a downloaded replay look like? | `POST .../GetEpisodeReplay` with `{"episodeId": 95029942}`. Compare the body against our provided files — episode 95029942 is in `train/`, so this is a byte-for-byte check. |
| T0-c | What is the rate limit? | Unknown. **Do not guess a number into this plan.** Measure: 20 requests in a row, record HTTP 429s and latency. |
| T0-d | Which credentials work? | `~/.kaggle/` holds **only** `access_token` — no classic `kaggle.json`. `which kaggle` → not found. `import kaggle` → ModuleNotFoundError [MEASURED, verified]. Find out which auth these endpoints accept. |

Preferred approach if it comes to it: Meta Kaggle publishes `Episodes.csv` and `EpisodeAgents.csv`
daily. Use that as the *index* — bulk, no rate limit — to pick which episodes belong to which
agent, then spend the rate-limited `GetEpisodeReplay` calls only on episodes already chosen. If we
do add games from a second teacher, **the held-out-agent split rule comes back into force.**

### The compiler fallback — insurance, not a likely path

`agent.verify.compile_day(obs, …)` produces (state, action) pairs at **any** state we ask about, in
unlimited quantity. The decoder, the features, the masks, the model, the loss and the whole
evaluation ladder are **~95% identical** whether the data came from a replay or from the compiler.
DAgger (Chapter 7) becomes trivially available instead of contingent. And the awkward question
PLAN3 raises about deriving from someone else's replay disappears entirely.

> **Build this as a seam from day one: `bc/sources/replay.py` and `bc/sources/compiler.py` behind
> one interface.** Retrofitting it later is a rewrite; building it now is one adapter.

On the compiler path — and only there — observations are *generated* rather than read, so kagsim
regeneration lives on that path. **We do not regenerate observations for Kaggle replays**: they
already contain full per-step observations for both seats. (`obs.farms` alone is 85.9% of the 31 MB
file.) An earlier draft of this plan proposed regenerating them; that proposal was describing
`traces/`, a different artifact, and the workstream is deleted.

### How much data we actually have

We clone **Ryo's seat only** (below), so one game yields 719 pairs, not 1,438:

| Split | Games | State rows (719 each) | Worker decisions (~10.5/step) | Macro decisions (3,945/game) |
|---|---|---|---|---|
| `train` | 70 | **≈ 50,330** | **≈ 530,000** | **≈ 276,000** |
| `val` | 15 | ≈ 10,785 | ≈ 113,000 | ≈ 59,000 |
| `test` | 15 | ≈ 10,785 | ≈ 113,000 | ≈ 59,000 |

Two thresholds this comfortably clears:

- **The ≥20,000-labelled-decisions floor for offline metrics** (Chapter 4) — validation alone has
  roughly 113,000 worker decisions, five times over.
- **The ≥50-game / ~400,000-decision minimum for the 430k model** (Chapter 5) — 70 training games
  and ~530,000 worker decisions clear it.

So the data does not bind the model size. Chapter 5's "start small" argument stands on its other
leg: 276,000 macro decisions from **one** teacher is still a narrow distribution, and a big model
will memorize it.

### Which seat we clone

**Ryo's seat, in every game, read from the `ryo_seat` column of `manifest.csv`.**

This overturns an earlier recommendation, and the reason is the dataset, not a change of mind. The
earlier draft said "train on both seats of every game" — correct when both seats are strong players
and you want to double your data. Here the other seat is **36 different opponents of unknown and
varying quality**, 22 of whom appear exactly once. Cloning them would blur the single clean policy
we are trying to learn.

**But the decoder still emits opponent-seat labels behind a flag** (Chapter 3). They are free to
produce, and there are two later uses: a "both seats" ablation, and opponent-modelling if we ever
want it.

*Still true and unchanged:* seat asymmetry is real — market orders settle player-by-player,
interleaved (`kag.py:615-625`) — which is exactly why the features canonicalize into "me" and
"them" (Chapter 5) rather than feeding a seat index. Ryo plays both seats across the corpus
(`ryo_seat` varies), so canonicalizing means both are one distribution and nothing is lost.

*Also unchanged as a rule for any future data:* drop any seat whose `statuses[p] != "DONE"` —
TIMEOUT and ERROR seats have truncated or garbage action tails. All 200 seats in this corpus are
`DONE`, so nothing is dropped today.

### Version skew — nothing to do here, but keep the rule

**All 100 games are `module_version 1.32.7`** [MEASURED, single-source], which is the version this
repo is pinned to. There is no skew to handle.

Keep the rule for the day there is: **E33** measured that 1.32.6 cut market demand **4.7×** and
changed shop draws to with-replacement; **E54** measured 1.32.7 adding a scarcity spike. A
market-timing policy copied from 1.32.6 play is optimized against a demand curve that no longer
exists. Note also that walking/watering/harvesting behaviour is far less version-sensitive than
market behaviour — E33 and E54 changed market curves, not tile mechanics. Store `module_version`
and a `config_hash` as columns on every shard anyway, so that a future filter is a query rather than
a re-decode.

### Storage — the corpus is the archive

**Tier 1 is already on disk: `data/sample_data_training_model/{split}/{episode_id}.json` —
2.9 GB uncompressed, 100 games.** Do not modify these files, and do not delete them: decoding bugs
are the most likely defect in this whole pipeline, and the raw bytes are the only way to recover
from one. `manifest.csv` carries a `source_sha256` per episode, so corruption is detectable.

If we ever add downloaded games, store them gzipped as `data/replays/{episode_id}.json.gz` —
measured **394,523 bytes per game**, so **~0.39 GB per 1,000 games** [MEASURED, verified].

**Tier 2, training shards: `data/shards/{split}/{episode_id}.npz`** (gitignored), with `{split}`
taken straight from the provided directory layout so our splits can never drift from theirs. A
**shard** is just a file holding a chunk of pre-processed training rows. Flat `.npz`, found by
`glob`, shuffled at both the shard level and inside each shard, with the last ragged batch dropped
so shapes stay constant.

Where the bytes are, if anyone optimizes later [MEASURED, single-source]: `obs.farms` 85.9%,
`obs.private` 1.7%, `obs.market` 1.2%, `action` 0.7%, `obs.town` 0.4%. `farms` is stored twice —
once per seat, byte-identical — so storing shared fields once halves the archive before any clever
encoding.

## How we know we're done

**Mostly already done.** The corpus is verified against the schema, the splits and `manifest.csv`
are understood, the teacher is chosen (Ryo, by construction), and Test A is measured and passed
(**E88**). What remains is Test B — the clock-vs-state ablation on Ryo, with `boatlee` as the
positive control (**E86**).

## When we stop

**If Test B shows less than 2 percentage points of state-over-clock advantage on Ryo *and* on the
compiler, then there is no policy in reach to imitate, and the BC line dies here.**

That outcome is now unlikely — Test A already showed 74–96% of steps differ between Ryo's own games
— but "unlikely" is not "measured", and Test A answers a weaker question than Test B does. Run it.

Note what does *not* kill it: a scripted leaderboard elsewhere. That would only re-point the
teacher at the compiler.

## What you should be able to explain

Why a model trained on a scripted agent would score brilliantly offline and collapse online, and
why offline accuracy cannot detect that. Why holding out whole *games* is the right test for a
single-teacher dataset while holding out *players* would be the right test for a multi-teacher one.
And why a corpus of 100 wins under-represents exactly the situations our own model will hit first.

## Reading

- Lin et al., **Focal Loss for Dense Object Detection** (2017) — read §3 only. It is about the
  same problem in a different costume: one class dominates, so the loss is dominated by easy
  examples. Chapter 4 is where this bites us.

---

# Chapter 3 — Turn replays into training examples

## The idea

The least glamorous chapter and the most important one. Behaviour cloning is supervised learning,
and supervised learning is only as good as its labels. Our job: for every step, produce a clean
pair — **the state the player actually saw** and **the action they chose while looking at it**.

Every trap here has the same shape: get it wrong, and training still runs, the loss still goes
down, and the accuracy curve still looks plausible. **Nothing tells you.** So each fix ships with
an assertion — a check that stops the program rather than letting it produce quiet garbage.

## What we build

`bc/decode.py` and `bc/features.py`. Output: verified shards, plus a small set of counters printed
on every build.

## The details

### What the decoder reads, and which side it clones

**Input:** `data/sample_data_training_model/{train,val,test}/{episode_id}.json`, with the split
taken from the directory rather than recomputed, so our splits can never drift from the provided
ones.

**Seat:** look up `ryo_seat` for that `episode_id` in `manifest.csv` and emit training pairs for
**that seat only**. Chapter 2 explains why: the other seat is 36 assorted opponents of unknown
quality, and cloning them would blur the one policy we are trying to learn.

**But emit the opponent seat behind a flag** (`--include-opponent-seat`, off by default). It costs
nothing to produce, and it keeps two later options open: a "both seats" ablation, and opponent
modelling. What we must *not* do is quietly mix it into v1 training and then wonder why the model
plays like a committee.

Assert, per episode: `info.TeamNames[ryo_seat] == "Ryo Hasegawa"`. If that ever fails, the manifest
and the files have drifted apart, and every label after it is suspect.

### Assertion 1 — the off-by-one

In the replay file, `steps[i].observation` is the state **after** step `i` happened, and
`steps[i].action` is the action that **caused** it (`core.py:277`, `:293`, `:301`).

> **The contract: the training pair is `(steps[i-1][p].observation, steps[i][p].action)` for
> `i` in `1..719`.**
> `steps[0].action` is a schema placeholder that no agent ever chose. **Drop it.** That gives
> **719 pairs per seat, not 720.**

**The check that catches it for free.** The worker roster is wiped every night (`kag.py:879-882`),
so its length is a fingerprint of which step's observation an action came from:

```
len(steps[i][p].action["hands"]) == len(steps[i-1][p].observation["farms"][p]["hands"])
```

Measured: **1,438 out of 1,438 match against the previous step. Only 1,275 of 1,438 match against
the same step — exactly 163 mismatches** [MEASURED, verified].

**How much this matters.** With the naive pairing, the "do workers walk the shortest path?" probe
below returned **0.0%**. With the correct pairing it returns **99.5% / 100%**. An entire
architectural conclusion inverted on this one index [MEASURED, verified].

### Assertion 2 — player 2's missing `step`

When the game hands an agent its observation, it uses `__get_shared_state(position)`
(`core.py:754-767`), which gives **both** players the `step` field. But the *saved replay* for
player 2 does not have it — checked at indices 0/1/5/100/400/719, the only key seat 0 has and seat
1 lacks is `step` [MEASURED, verified].

It is exactly reconstructible: `day*24 + hour == i` for **all 720 steps** [MEASURED, verified].

> **The contract: `delivered_obs(i, p) = dict(steps[i][p].observation)`, then set `["step"] = i`.**
> Never read `obs["step"]` straight out of a stored replay.

**The check.** For `p == 1`, assert `md5(obs[k]) == md5(steps[i][0].observation[k])` for
`k in {farms, market, town, day, hour}` — measured byte-identical at indices 1/2/10/100/400
[MEASURED, verified]. `private` is genuinely player 2's own data and needs no repair. **If this
assertion ever fires, the replay format changed and every number downstream of it is void.**

### Assertion 3 — the mask must never reject the expert

A **mask** is a list of which actions are currently legal; we zero out the illegal ones before the
model chooses (Chapter 5 has the full rule table). The check is:

```
n_expert_actions_rejected_by_mask == 0
```

If our mask says the expert's real move was illegal, **our mask is wrong**, not the expert. And a
model trained under a leaky mask produces loss numbers that mean nothing at all. This is the
"prove the change fired" gate for the entire program.

*(Ordering note: the mask rules live in Chapter 5. Either write the mask table first, or run this
assertion the moment the table exists — but do not train before it reads zero.)*

### Assertion 4 — effective shed, with the animal carve-out

216 of 412 SELL orders in the sample ask to sell more than `private["shed"]` holds at the moment
the observation was taken. **All 216 fit once you add what workers dropped into the shed during the
same turn. Zero remain oversized** [MEASURED, verified]. The reason is the ordering rule from
Chapter 1: unit actions are applied before `_process_market` (`kag.py:935-941`).

> **The contract:
> `effective_shed = shed + Σ(worker inventory contributed by DROP and PLACE-to-shed this turn)`.**
> **Carve-out: `PLACE·animal` places an animal on a *tile*, not into the shed
> (`kag.py:376-392`), and must be excluded from the sum.** The 216/216 result only holds with this
> carve-out.

Two consequences: the decode order is forced — **workers first, market second** — and effective
shed is a *required* input to the market heads, not an optional nicety.

### Orders are not sales — the rule that has cost this project three conclusions

`_commit_unit` settles one player at a time, alternating, against a shared pre-commit inventory.
`SELL` moves `min(requested, shed)`. `BUY_PRODUCT` simply fails against a full shed
(`kag.py:641-658`, `:615-625`).

- **Labels: use the request exactly as written.** BC copies the *policy*, and the policy issued the
  request. What the game did with it is not the policy's decision.
- **Features: never derive anything from a request.** Any quantity like "units sold this turn" must
  be computed by **differencing shed + per-worker inventories between `steps[i-1]` and
  `steps[i]`**. The next observation is settlement ground truth and is already in the file. **Never
  re-implement the settlement rules inside the decoder** — that exact mistake cost E39.
- Per-step money change =
  `steps[i].obs.farms[p].money - steps[i-1].obs.farms[p].money`. Settlement-true by construction.
  Diagnostics and value-function pretraining only — **never** the RL objective (Chapter 8).

### A trap that already produced a silent no-op

**`farms[p]["tiles"]` is a nested 10×10 list, not 100 flat tiles.** Indexing it as if it were flat
returns nothing useful and raises no error. This silently broke a verifier's first probe. Flatten
explicitly and assert `len == 100`.

### Emit both label sets from one pass

The decoder writes **two kinds of label for every step**:

- **Raw labels** — literally the action, one per worker per turn. 15,057 of them per game.
- **Macro labels** — the same trajectory re-described as "go to tile X and do Y there", produced by
  segmenting each worker's day into runs of `NORTH/SOUTH/EAST/WEST` ending in a non-move op.
  7,889 of them per game.

Producing both from one pass makes the choice between them a **training flag, not a rewrite**.
Chapter 5 makes the choice, using the number below.

### The measurement that makes macro labels credible

For every worker, every day, compare the length of each walking run against the straight-line
(Manhattan) distance from where the run started to the tile where the final action fired
[MEASURED, verified — re-derived by an independently written segmenter]:

| seat | walking runs | exactly shortest | detours |
|---|---|---|---|
| 0 | 1,690 | **1,681 (99.5%)** | 9 |
| 1 | 1,407 | **1,407 (100.0%)** | 0 |

And walking is genuinely unobstructed: `kag.py:326-331` bounds-checks only, and its comment states
LOCKED tiles are passable. So Manhattan distance **is** true distance, always, everywhere.

**Emit this as a per-shard counter: `frac_segments_shortest_path`.** It must be ≥ 0.9. This number
is measured on **one game**; Chapter 5's decision needs it **corpus-wide**. Record that measurement
as **E87**.

Tie-break when the target is diagonal: the sample goes x-first 566 times, y-first 345
[MEASURED, single-source]. It does not matter behaviourally — no collision rule exists
(`_spawn_hand` at `kag.py:533-541` counts occupancy but nothing forbids two workers sharing a
tile) — so any fixed rule works.

### Settle the vocabulary as one constant, before anyone writes a head width

Two panel proposals disagreed **arithmetically** about how many verbs and items exist, and a
one-entry disagreement between the model's output width and the mask table makes Assertion 3 fire
and look like a data problem:

- **Verbs.** One said a 14-way verb head (+4 moves = 18); the other measured **17 distinct worker
  ops** — but that is *observed in one sample*, which is not the same as the vocabulary.
- **Items.** One said 12 (9 products + 3 animals) while also referring to `PICKUP·FERTILIZER`; the
  other said the shed holds 12 items. **Whether FERTILIZER is inside the 9 is unresolved.**
- **Worker slots.** 16 (the mechanism's max) vs 13 (the observed max). **Take 16** — padding is
  free and fixed shapes matter for Chapter 9.

> **The rule: `bc/features.py` defines `VERBS`, `ITEMS`, `MARKET_OPS`, `QTY_BINS`, and
> `N_UNIT_SLOTS = 16` as single enumerated constants, derived from `kag.py`'s own dispatch tables
> (not from our sample), asserted at shard build, at training time, and at inference. Every head
> width reads from those constants. Nothing else may hardcode a width.**

For reference, the full observed vocabulary is **35 verb+argument tokens across both seats**,
reproduced token-for-token by the fact-checker [MEASURED, verified]: `WATER` 2010, `PASS` 1807,
`HARVEST` 827, `FEED` 607, `COLLECT_FERTILIZER` 604, `CARE` 571, `DROP` 281, `FERTILIZE` 246,
`PLANT·WHEAT` 233, `PICKUP·WHEAT·1` 162, `PLANT·STRAWBERRY` 82, `DIG` 79, down to
`PICKUP·FERTILIZER·6` at 1.

Two more shape facts: **`PLACE` takes 2 arguments 27 times and 3 arguments 16 times** — variable
arity, the argument head must cope [MEASURED, verified]. Max order quantity is **61**.

### One structural regularity worth turning into a feature

**Hiring only ever happens at dawn.** Measured: **601 HIRE orders across 105 distinct turns, all at
the day's first two decision hours** [MEASURED, verified]. *(An earlier draft claimed "289 hires at
hours 1–2". That count is unreproducible and is dropped. The pattern survives; the number does
not.)* So hour-of-day must be a first-class feature in Chapter 5.

### Shard layout — state-major, label-minor

There are ~10.1 (seat 0) / 10.8 (seat 1) worker decisions per state [MEASURED, single-source]. If
we stored one row per decision, we would store each state about 10 times. Instead:

- `states[1438, …]` as int8/int16,
- `labels[~15057, …]` where each row carries a `state_idx` pointing back into `states`,
- `market_labels[~1218, …]` the same way.

Estimated **~2.7 MB per game uncompressed** [MEASURED, single-source]. **Verify on the first 10
decoded games: if compressed shards exceed ~1.5 MB per game, revisit the layout.**

Rejected: Parquet/Arrow (an extra dependency, and the Orbit-Wars loader is `.npz`-shaped, so we
lose the reuse); an uncompressed JSON archive (31 MB per game for no benefit).

### One thing we deliberately do not do yet

The feature extractor reads **only the official environment's observation dictionary.** Making it
also read a kagsim state doubles the surface area and drags `make verify` — the coverage-gated
parity suite — into scope for no benefit before Chapter 8. **Deferred to Chapter 8's entry
requirement**, where fast rollouts make it genuinely necessary.

## How we know we're done

The decoder runs over all **100** provided games with **zero** assertion failures on all four
assertions, and prints the row counts (expect ≈50,330 training states — Chapter 2's table).
`frac_segments_shortest_path` is computed corpus-wide over Ryo's seat (**E87**). The vocabulary is
one asserted constant. Shards are written under the provided split names. Shard sizes are measured
against the 1.5 MB threshold. The majority-class floors (Chapter 4) are recomputed **on Ryo's seat
over this corpus** — the 16.3% / 19.3% figures came from the old two-seat sample and do not
transfer.

## When we stop

**If we cannot drive `n_expert_actions_rejected_by_mask` to zero, our model of the action space is
wrong. Redesign the action space before training anything.** Do not tune. A loss computed under a
leaky mask is uninterpretable, so every number after it is noise.

## What you should be able to explain

What a *label* actually is here. Why an off-by-one in the pairing still trains, still converges,
and still shows a nice accuracy curve while cloning "the move from one step ago". And why a mask
that rejects the expert looks exactly like a hard training example.

## Reading

- Read the loss-skip filter in `Orbit-Wars/src/train_bc.py` (lines 143–188 and 251–268) and write
  down *why we are not enabling it yet*. In the reference it was covering up a masking bug — its
  own comments say so. Suppressing hard batches to hide a mask defect is exactly the failure
  CLAUDE.md calls "a zero counter is an unfinished implementation, not a negative result".

---

# Chapter 4 — The smallest possible model

## The idea

Before building anything clever, build the dumbest thing that could work, on **one** decision. For
example: "will this worker act this turn, or idle?" Logistic regression, or a two-layer network,
on a handful of hand-picked features.

This chapter exists for one lesson: **an accuracy number means nothing on its own.**

## What we build

A one-head baseline (sklearn or 30 lines of numpy). Then a throwaway end-to-end model — call it
**v0**, 50,000–150,000 parameters — wrapped as a real `agent(obs)` and run for one full game. Then
we throw v0 away.

## The details

### The "always do nothing" trap, with the real numbers

If you always predict the single most common action, you already score [MEASURED, verified]:

| Seat | Most common action | Score you get for free |
|---|---|---|
| 0 | `WATER` | **1,186 / 7,270 = 16.3%** |
| 1 | `PASS` | **1,503 / 7,787 = 19.3%** |

That is the **majority-class baseline** — the score of a model that has learned nothing. A head
reporting "70% accuracy" means something only when printed next to "the floor is 19.3%".

> **⚠ Those two numbers came from the two seats of the single old sample game. They are the right
> order of magnitude and the wrong numbers for us.** Recompute the floor **on Ryo's seat, over the
> 70 training games**, before quoting any accuracy against it. That was always the plan; it now
> points at `data/sample_data_training_model/train/`.

> **Rule for the rest of the journey: every accuracy number is reported against its majority-class
> floor, always, with an error bar.**

### ⚠ The opening steps inflate everything — report metrics twice

Chapter 2 measured that **Ryo plays a fixed ~30-step opening routine** and only then starts
reacting: across pairs of their games, the identical steps concentrate at indices 0–31
[MEASURED, single-source].

Those ~30 steps are almost perfectly predictable from the step number alone, and they are about 4%
of every game. A model that has learned nothing except the opening will still post a
better-than-floor headline number.

> **So every offline metric is reported twice: over all steps, and over steps ≥ 32 only.** The
> second number is the real one. If they differ a lot, the model has learned the opening and not
> much else.

### The error bar — what a Wilson interval is and why we use it

A **Wilson interval** is the error bar for a percentage. If you measure 70% accuracy on 800
examples, the true value could plausibly be anywhere from about 67% to 73%; on 80 examples the
range is roughly 59% to 79%. Reporting "70%" without that range is how you convince yourself of
something that is not there.

Concretely: **a "70% vs a 19% floor" claim computed on 800 examples is the same mistake as the
16-game arena result this project has died on three times.** So:

> **Validation must contain at least 20,000 labelled decisions, and every offline number is printed
> as `value [low, high]` next to its floor.**

Our `val` split clears this by a wide margin: 15 games × 719 steps × ~10.5 worker decisions ≈
**113,000** decisions (Chapter 2). Even after dropping the opening steps it is ~108,000.

### Which heads to report

Report **top-1 and top-3 agreement per head separately** — worker-op, worker-argument, market-op,
market-item, market-quantity. Never one blended number: the worker heads would swamp it 15,057 to
1,218.

### The v0 plumbing run

The point of v0 is not the score. It is to prove the whole pipe works end to end, in the real
runner, before we spend a weekend on the real model. Gate on:

- no exception,
- `status == "DONE"`,
- final bank **above the $3,000 starting bank**,
- **run it in both seats.**

This is exactly the trap `make submission` exists to catch. Kaggle never imports `main.py` — it
`exec`s the source into an empty globals dictionary and calls the **last module-level callable**
(`agent.py:47-63`). So **`__file__` does not exist**. Load weights via a path relative to
`os.getcwd()`, or inline them. The E21 version of this mistake raised `NameError`, so the agent
never loaded, scored the $3,000 starting bank, and measured 0-40 against a real opponent.

## How we know we're done

The one-head baseline beats its majority floor by a margin whose Wilson interval excludes the
floor. The v0 checkpoint completes a 720-step game in **both** seats, DONE, no exception, above
$3,000.

## When we stop

**If the model cannot beat the majority floor on the worker-op head, this is an encoding bug, not a
model problem. Go back to Chapter 3. Do not tune.**

## What you should be able to explain

Why "97% accurate" can describe a completely useless model, using this game's own numbers. And what
the difference is between offline agreement and online return — state it now, before you have any
reason to believe it, so that Chapter 6 lands.

---

# Chapter 5 — The real model

## The idea

Now we build the actual BC model, borrowing the *shape* of Orbit-Wars' network
(`src/orbit_net.py`) and adapting it to farming. Three ideas, in plain words:

1. **Describe the state as a set of tokens.** A **token** is just a short list of numbers
   describing one thing. They made one per planet. We make one per farm tile (100), one per worker,
   and one per market product.
2. **Break the action into small choices, called heads.** A **head** is a small output layer that
   picks one option from a list. Instead of one giant "what is the whole action" choice, we ask
   several small ones: act or idle → which tile → what verb → with which item → how many.
3. **Mask illegal choices.** Zero out anything the game would reject before the model picks.

## What we build

`bc/model.py` and `bc/train.py`. The model is **v1: about 430,000 parameters**.

## The details

### First: the action space, decided by measurement

Chapter 3 produced both label sets and the corpus-wide number
`frac_segments_shortest_path` (E87). Now use it:

**If it is ≥ 0.9 — use the macro action space.** Workers walk the shortest path in the data, so
walking is not a decision worth learning. A worker is asked for a choice **only when it is idle**:

```
commit head   : {IDLE, ACT}                                  2 options, per worker
tile pointer  : pick 1 of the 100 tile tokens
verb head     : pick 1 of the verbs (width from the Chapter 3 constant)
item head     : pick 1 of the items, masked to what the verb allows
qty head      : 14 buckets {1,2,3,4,5,6,8,12,16,24,32,48,64+, ALL}
```

A **pointer** means: instead of a fixed list of output slots, we score every tile token against a
query vector and pick the best. `logits = tiles @ q / sqrt(d)`. (A **logit** is the raw score
before it is turned into a probability.) Then ten lines of hand-written code walk the worker there
and fire the verb. **Almost half the decisions in the data (47.6%) disappear from the learning
problem this way.**

**If it is below 0.9 — do NOT fall back to raw per-step actions.** Fall back to
**macro-with-an-explicit-MOVE verb**: keep the tile pointer and the semantic verbs, and add `MOVE`
as one more verb so deliberate loitering and path-shaping become expressible. Three independent
Orbit-Wars writeups report that a raw action space killed learning outright — one called their raw
version "a far cry from competitive", another said switching to four semantic intents "increased
learning speed by a lot". `TASKS.md` T3.1 already specifies per-worker `(target tile, intent)` and
"**Never emit `NORTH`**" [MEASURED, verified]. Giving up semantics because the *walker* is
imperfect throws away the wrong half.

### The market: a second small sequence, decided after the workers

Market orders are an **ordered list of up to 10**, and the order matters causally — money from a
sell in slot 0 pays for a buy in slot 3 within the same turn (`kag.py:562-628`) — and a late slot
can be starved by the interleaved settlement (`kag.py:615-625`).

```
for k in 0..9:
    stop head : {EMIT, STOP}
    op head   : 6 options {SELL, BUY_SEED, BUY_PRODUCT, BUY_ANIMAL, HIRE, BUY_LAND}
    item head : masked per op
    qty head  : buckets + ALL
```

While decoding these ten slots we run a small **running simulation** of money, shed and price using
`market_price` (`kag.py:192-206`), so slot *k*'s mask is exact rather than approximate. Feed it the
**effective shed** from Chapter 3.

### Autoregressive decoding — one worker at a time, each told what the last one chose

**Autoregressive** sounds worse than it is: the model decides for worker 1 first, then worker 2 is
told what worker 1 chose, then worker 3 is told about both, and so on.

**We need this for a hard mechanical reason, not for elegance.** Remember the planting cliff from
Chapter 1: if the total `PLANT` requests for one crop exceed the seeds you own, **all of them are
dropped** (`kag.py:920-933`). And the experts sit right on that cliff — measured planting bursts of
2 workers (**50 turns**), 3 (**14**), 4 (**2**), and **66 turns where the demand equals the seed
count exactly** [MEASURED, verified]. A model that decides for each worker independently will fall
off it.

Decode order = worker index (farmer = 0, then hands in list order). That matches the action list's
own layout, is stable within a day, and carries meaning (index = hire order). **Do not sort workers
by size** — the Orbit-Wars code does, but there is no analogous magnitude here.

What the running state carries: `seeds_remaining[crop]`, `money`,
`shed_free = shedCapacity - sum(shed)`, plus a learned projection of the (tile, verb) each worker
has already committed to.

### The tokens

| Token type | How many | What is in it |
|---|---|---|
| Own tiles | 100 | kind [8], crop [5], animal [3], plant dynamics [9], animal dynamics [7], economics [3], geometry [10] ≈ 44 numbers. **Everything we need for exact masking is in the public `farms[p].tiles` — nothing about tile state is hidden from us.** |
| Own workers | 16 (padded) | is_farmer, index, x/y, next-to-shed, distance to shed, what it is carrying, is_idle, hours left today ≈ 26 numbers. Masked by `n_units = 1 + len(hands)`. |
| Opponent | 1 summary | money, tiles planted, animals, workers, unlocked quadrants, hires today, money change yesterday. **Do not make 100 tokens for their tiles.** Their tile detail reaches us through exactly one channel — future market supply — and that is directly computable via `agent/engine.py:722 opponent_supply()` [MEASURED, verified]. |
| Market products | 9 | inventory, price, `n_shops_buying_me`, `town_drain_per_day` (`agent/engine.py:757`), `T` and `above_func` (`kag.py:41-51`), my shed, my effective shed, my and the opponent's forecast supply at 3/7/14 days, the price slope, `forecast_price(h=7)` ≈ 22 numbers. |
| Global | 1 | `step/719`, `day/29`, `hour/23` **and** `sin/cos(2π·hour/24)`, `is_hour_0`, money, money margin, shed fill, seeds, unlocked quadrants, next land price, next hire cost, shops unlocked, days to next shop unlock ≈ 24 numbers. |

**Two design notes with reasons.**

*Geometry as plain coordinates plus a learned 100-entry position embedding, not Fourier features.*
Orbit Wars needed Fourier because their board was a continuous 100×100 space. Ours is a discrete
10×10 grid, where a learned lookup is both exact and cheaper.

*`n_shops_buying_me` and `town_drain_per_day` are in there deliberately.* They encode this
project's most expensive lesson (D17): **rank a product by how many shops want it, not by its price
curve** — judging by the price curve ranked the markets almost backwards and cost a 2.4× gain.
`T` and `above_func` encode the correction to that lesson (E48/E41): melon at 114 units a season is
excellent and melon at 360 units is worthless, because the price curve still decides what one batch
fetches. **Both terms belong in the token; the model works out the tradeoff.**

*Do not give shops their own tokens.* They are drawn with replacement, capped at 8 instances, and
their only effect is per-product drain. Folding them into the product tokens as a count loses
nothing and saves 8 tokens.

*Seat is not a feature.* Canonicalize everything into "me" and "them" in the extractor. Both seats
then become one distribution and the dataset honestly doubles.

### The masks — complete and exact, with nothing hidden

Every rule below is computable from the observation with zero hidden state. This is a stronger
position than Orbit Wars had — their mask checked the sun but not intervening planets.

Verb legality, given target tile `T` and the worker's inventory `inv`:

| verb | legal when | cite |
|---|---|---|
| `PASS` | always | `:334` |
| `DROP` | next to shed, `inv` non-empty | `:132-139`, `:344` |
| `PICKUP i n` | next to shed, `shed[i] > 0`; **seeds are not pickupable** | `:359-374` |
| `PLACE a` (animal) | `T.kind` matches the animal's structure, no animal there, `inv[a] ≥ 1` | `:381-392` |
| `PLACE i n` (to shed) | next to shed, `inv[i] ≥ 1`, `sum(shed) < capacity` | `:393-409` |
| `PLANT c` | tile empty, `seeds[c] > 0`, **and the running seed budget is still positive** | `:417-429`, `:920-933` |
| `WATER` | tile is a PLANT and not watered today | `:431-435` |
| `HARVEST` | `yield_units > 0`, and for plants also old enough | `:446-453` |
| `FERTILIZE` | tile is a PLANT, `inv["FERTILIZER"] ≥ 1` | `:475-478` |
| `DIG` | tile is not empty and has no animal | `:484-489` |
| `BUILD_COOP` / `BUILD_PASTURE` | tile empty | `:493-502` |
| `FEED` | animal present, not fed today, `inv["WHEAT"] ≥ 1` | `:505-512` |
| `COLLECT_FERTILIZER` | animal present, fertilizer available | `:515-521` |
| `CARE` | animal present, not cared for today | `:524-529` |

All tile-changing verbs additionally need `T != "LOCKED"` (`:414`). *(These citations are
[MEASURED, single-source] except the seven that the fact-checker confirmed independently, listed in
Chapter 1.)*

Market masks: `HIRE` needs the Fibonacci cost (`:698-706`); `BUY_LAND` needs an unlocked slot and
the money (`:712-719`); `BUY_PRODUCT` is WHEAT/FERTILIZER only plus money and shed room
(`:598`, `:662-671`); `BUY_ANIMAL` (`:679-686`); `SELL` needs `shed[item] > 0` (`:653-654`);
`BUY_SEED` (`:673-677`).

**The tile pointer is never masked** — all 100 tiles are reachable. Legality is enforced by the
verb head instead.

**Two traps.** (a) A fully-masked row produces NaN and poisons training — **always leave `PASS`
unmasked.** The Orbit-Wars code had to filter `|log_prob| > 1e5` for exactly this reason. (b) The
expert-legality assertion from Chapter 3 must read **zero** before any loss number is believed.

**Two things masks buy us that are worth naming.** The model can never emit an impossible action,
so BC never burns capacity learning "don't do the impossible". And the same mask code runs in
training and in play, so Chapter 8's RL stage inherits a correct action distribution for free.

### The loss

**Cross-entropy** is the standard "how wrong was this pick" loss for choosing one option from a
list.

```
L = CE(commit) + CE(tile | ACT) + CE(verb | tile) + CE(item | verb) + CE(qty | verb,item)
  + CE(mkt_stop) + CE(mkt_op) + CE(mkt_item) + CE(mkt_qty)
  + λ_v · MSE(value, terminal_margin)                     λ_v = 0.1
```

**Masked cross-entropy everywhere; no regression heads.** Quantities here are small integers with a
very peaked distribution (`PICKUP n` is 1..6; the biggest order in the sample is 61), so **we put
them in buckets** (14 classes including `ALL`). Fitting a continuous value to a spiky integer
target is the wrong objective, and it is the loss the Orbit-Wars authors were least happy with.

**Class imbalance: do nothing in v1, then measure.** The genuinely rare class is `BUY_LAND` —
2 per player per game [MEASURED, single-source] — against `HIRE`'s 601 [MEASURED, verified]. So:
print **per-class recall** for `BUY_LAND`, `BUY_ANIMAL`, `BUILD_PASTURE`, `PLACE·animal`. Only if a
recall is about zero do we intervene, and the right intervention is probably **not** loss
weighting: `BUY_LAND` is nearly deterministic given (day, money, quadrants unlocked), so a
three-feature rule beats any reweighted head. Loss weighting distorts calibration and is a common
way to make a model look better per-class while playing worse.

*(One number to recompute rather than copy: an earlier draft reported "macro PASS = 22%" using a
decision count of 8,076, which was arithmetically wrong. The correct total is 7,889. Recompute the
ratio; do not quote the old one.)*

### The value head — build it now, not later

A **value head** predicts "how well is this game going to end for me". We train it here, during BC,
with `λ_v = 0.1`, against **the final outcome only**.

Why now: entering PPO with a random value head means the first several million practice steps are
spent teaching the critic while the actor is driven by noise. That is the most common way a warm
start gets destroyed. It costs 17k parameters and one extra output.

**Terminal outcome only. No shaping.** All six Orbit Wars top-10 finishers found hand-crafted
intermediate rewards measurably hurt; 7th place ablated shaping and it scored 34.6% against its own
baseline. Do not add per-step money as a training signal, however tempting the dense feedback looks.

### Model sizes — one ladder, three rungs

| rung | size | shape | role |
|---|---|---|---|
| **v0** | 50–150k | one head, sklearn/numpy | plumbing, thrown away (Chapter 4) |
| **v1** | **≈430k** | pointer + MLP, no attention | **the BC model** |
| **v2** | **0.8–1.5M** | **wide and shallow** (d ≈ 256, layers 2–4) | the PPO model (Chapter 8) |

v1 breakdown: the four shared encoders (tiles, workers, products, global; 128 wide) ≈ 84k; pointer
≈ 65k; verb/item/quantity heads ≈ 204k; market heads ≈ 50k; sequence-state update ≈ 35k; value head
≈ 17k. It trains in minutes on this Mac.

**Why small: the dataset binds us, not the compute.** Our 70 training games give ~530,000 worker
decisions and ~276,000 macro decisions — enough for this model and no more, and all of it from a
**single** teacher, which is a much narrower distribution than the raw count suggests. A
3-million-parameter transformer on that memorizes: the loss curve looks wonderful and the arena
winrate does not move — this repo's single most repeated failure. Orbit Wars' own first model was
460k parameters of exactly this shape and beat its baseline 100% of the time.

**And the size is really set by Chapter 8, not by BC.** BC training is cheap at any of these sizes,
so BC never justifies growth. PPO has a hard step budget (Chapter 8). Growing now and shrinking
later would waste the warm start.

**A 3.1M `d192/L6` transformer is rejected from the ladder and moved to the backlog.** It is the
worst point on the measured speed curve for this machine: **depth is the tax.** d384 with 4 layers
(7.10M params) runs at 3,048 inferences/second and d224 with 8 layers (4.84M) runs at 3,326 — the
same speed at 47% more parameters [MEASURED, verified]. If a transformer ever happens here, it is
wide and shallow, and it is promoted **only on ≥80-game arena winrate, never on held-out loss.**

Full measured curve on this M1 Pro (batch 512, 40 tokens) [MEASURED, verified — re-run 10–35%
faster than the original, same ordering]: params 0.40 / 0.79 / 1.78 / 4.84 / 7.10 / 12.61 M →
inferences/second 33,373 / 16,612 / 7,938 / 3,326 / 3,048 / 1,587.

Framework: **PyTorch on MPS, fp32.** No bfloat16 — measured no gain on MPS (slightly worse), and
the reference code has a documented history of NaN problems around it. JAX only in Chapter 9.

## How we know we're done

Every head beats its Chapter 4 floor on games the model never saw, with Wilson intervals excluding
the floor (**E89**). The expert-rejected-by-mask counter reads zero. Per-class recall for
`BUY_LAND` is logged. The value head is trained.

## When we stop

**If agreement stalls at the majority floor on any head while the vocabulary is verified, the
action encoding is wrong** — go back to Chapter 3.

**Do NOT stop because "online money is below the expert's".** That is Chapter 6's and Chapter 7's
problem, and tuning the model here to fix it burns weeks.

## What you should be able to explain

What a head, a pointer, and a mask are. Why breaking one big choice into several small ones is
easier to learn. Why decoding workers one at a time is required by the planting rule and not just
an aesthetic preference. Why we start small.

---

# Chapter 6 — Make it play, and meet distribution shift

## The idea

We wrap the model as a real agent — `agent(obs)` → action — and let it play full 720-step games.

**And here is the central idea of imitation learning.** Our model only ever saw situations a *good
player* got into. The first time it makes a small mistake, it lands somewhere the expert never
demonstrated. So its next choice is worse. So it lands somewhere stranger still. **Errors
compound.** This is called **distribution shift**: the states at test time are drawn from a
different distribution than the states at training time — and the model itself is what changed the
distribution.

The theory says a per-step error rate ε can grow like **T²·ε** over T steps. Our T is **719**. A 1%
per-step error is not a 1% problem.

**Our dataset makes this slightly worse, and we should expect it.** All 100 training games are
games Ryo **won** (Chapter 2). So the model has barely seen a losing position, and "losing" is
exactly where its own first mistakes will put it. Expect a steeper handover curve than a
win/loss-balanced corpus would give.

## What we build

The agent wrapper, the four-rung evaluation ladder, and the handover experiment.

## The details

### The evaluation ladder — four rungs, each gating the next

**Rung 1 — offline, per head, every epoch.** Top-1 and top-3 agreement per head, each next to its
majority floor — **recomputed on Ryo's seat over the training split**, not the old 16.3% / 19.3%
sample figures — with a Wilson interval, on ≥20,000 validation decisions (we have ~113,000). Report
every number twice: all steps, and steps ≥ 32 only (Chapter 4's opening caveat).

> **Rung 1 can only falsify, never confirm.** A high score is entirely consistent with having
> learned "read the step number, output the memorized move".

**The mandatory companion — the anti-clock ablation (E90).** Re-score the validation set with the
`step`, `day` and `hour` features **zeroed out**. A real policy loses a lot of accuracy; a clock
loses almost none.

**Run this on steps ≥ 32 only.** Ryo's fixed ~30-step opening (Chapter 2) genuinely *is* a function
of the clock, so including it would mask a real problem: a model that learned the opening and
nothing else would show a small, reassuring drop.

> **If accuracy drops less than 5 percentage points, the model is a clock, not a policy. Stop and
> fix the DATA, not the model.**

**Rung 2 — does it even run legally.** One full 720-step game **in each seat**. Gate on: no
exception, `status == "DONE"`, final bank above the $3,000 starting bank. Also record import cost
and the 99th-percentile turn time here (**E93**).

**Rung 3 — pool play, ≥80 games, both seats, fresh seeds.** Via `harness/run.py`, against
`DEFAULT_POOL = ["boatlee", "executor_v7", "starter", "kagsim_champion"]`
(`harness/registry.py:212`) [MEASURED, verified]. Run `harness/counters.py` `Observer` on the
checkpoint every time. **Mean bank is a diagnostic; pairwise winrate is the ranking.** Record as
**E92**.

**Rung 4 — the promotion gate. `make promote` only (D19).** `tools/promote.py` already implements
the right protocol and must not be rebuilt: `SCREEN_GAMES = 12` (24 episodes), escalating to
`CONFIRM_GAMES = 250` (**500 episodes**, which can resolve a ≥54.4% edge) whenever the screen lands
in `NOISE_BAND = (0.35, 0.65)`; then a no-new-losses stage; then a neighbourhood sweep [all
MEASURED, verified]. `boatlee` is reported as a **reference that never gates** (D21).

> **A promotion gate is not a stop rule.** Rung 4 is a later, optional event. It must never be
> written into a chapter's stop rule, or we would be demanding that a first learned model beat a
> mature scripted champion as a condition of *existing*.

### Timing — torch is fine at play time, but its import is charged to turn 1

The budget, read from the replay's own configuration: `actTimeout: 1` second per turn,
`runTimeout: 1200` seconds per game, `remainingOverageTime: 60` seconds of slack for the whole
game.

**Measured, CPU, single-threaded, batch 1, 230 tokens — which is what Kaggle actually runs, because
there is no MPS there** [MEASURED, verified]:

| model | params | ms per turn | seconds per game |
|---|---|---|---|
| d128 L4 | 0.79M | **3.4** | 2.5 |
| d192 L6 | 2.67M | **6.7** | 4.8 |
| d224 L8 | 4.84M | **11.2** | 8.1 |

That is 1–3% of one turn's budget. **A numpy re-implementation of the model is NOT required.**
*(An earlier draft argued it was; a different draft argued it was not but supported that with MPS
numbers, which do not transfer to a CPU-only submission. The CPU table above is the reason.)*

**But the import is a real cost, and it lands in turn 1.** `build_agent` (`agent.py:145-157`)
defers the `exec` of the submission source to the **first call**, and `Agent.act` starts the clock
immediately before it (`agent.py:191-192`). Proved: an agent whose module body did
`time.sleep(3.0)` produced per-step durations `[3.0018, 0.0, 0.0, …]` and drove
`remainingOverageTime` from 60 down to **57.998** = 60 − (3.0018 − 1) [MEASURED, verified].
Overage is only consumed when a step exceeds `actTimeout` (`core.py:631-632`).
`import torch` costs **0.93 seconds warm** [MEASURED, verified].

> **⚠ Local testing cannot catch a timing blowout.** The local `env.run` path **never raises
> TIMEOUT** — `DeadlineExceeded` only comes from Kaggle's production runner (`core.py:281`)
> [MEASURED, verified]. A cold-disk first import on Kaggle is unmeasurable offline and is the only
> real timing risk here. Budget from the CPU table, keep the import small, and never conclude "it
> fits" from a green local run.

### The handover experiment — watch compounding error with your own eyes (E91)

For k in {0, 100, 200, …, 719}: let the **expert replay** play steps 0..k, then let **our model**
play k..719. Plot final money against k.

- A flat curve means BC is fine and the problem is somewhere else.
- A curve that collapses as k gets smaller **is the T²·ε bound, drawn on our own data** — and its
  shape tells us *which part of the season* the model loses.

A second view, same phenomenon: plot the model's action agreement on **expert-visited** states
against its agreement on **its own rollout** states, as a function of t. The curves start together
and diverge. **The area between them is distribution shift, in units you can put on a slide.**

A 20-minute bonus: score the same checkpoint with argmax versus temperature-1 sampling. AlphaGo's
supervised policy network was *weaker* head-to-head than its RL network yet *better* as a search
prior, because it was more spread out. A standing reminder that "the best policy" depends on what
consumes it.

## How we know we're done

The agent completes games in both seats without crashing and earns clearly more than $3,000. The
anti-clock ablation drops ≥5 percentage points. Online money vs the pool is measured at ≥80 games,
both seats, with `Observer` counters attached. The handover plot exists. The numpy-vs-torch decision
is recorded as a measurement, not an opinion.

## When we stop

**If the anti-clock ablation drops less than 5 points, the model is a clock. Fix the data (go back
to Chapter 2's teacher choice), not the model.**

Do not stop because online money is below the expert's — that is exactly what Chapters 7 and 8 are
for.

## What you should be able to explain

State the **T²·ε** bound and why T = 719 makes a 1% per-step error catastrophic. Explain why
per-timestep BC and teacher forcing are the same construction: at training time you condition on
states *the expert* reached; at test time you condition on states *you* reached. The drift is
exposure bias by another name.

## Reading

- Ross & Bagnell, **Efficient Reductions for Imitation Learning** (AISTATS 2010) — the T²ε bound.
  **The single most important paper in this plan.**
- Bengio et al., **Scheduled Sampling for Sequence Prediction with RNNs** (2015) — the
  teacher-forcing / exposure-bias analogy made concrete.
- Vinyals et al., **Grandmaster level in StarCraft II** (Nature 2019) — read the
  supervised-from-replays section: 971,000 replays, and the supervised agent *alone* beat 84% of
  human players. It calibrates how far good BC gets.

---

# Chapter 7 — Fight the drift (DAgger)

## The idea

**DAgger** in plain words: *let the student play; every time the student wanders somewhere new, ask
the teacher "what would you have done here?"; add those answers to the training data; retrain.*

Now the model has labels exactly where it gets lost. This turns the T²·ε problem into a T·ε
problem.

**The catch: the teacher must be askable at any situation.** A replay cannot answer questions — it
is a recording. So DAgger is structurally impossible against a replay and possible against a
function.

## What we build

The rollout-and-relabel loop, plus — before any of that — an entry probe that decides whether we
should build it at all.

## The details

### The oracle finding — the obvious teacher is not a teacher

An **oracle** here just means "a teacher we can call as a function at any state".

The obvious candidate was `agent/main_v4.py`'s `agent(obs)`. **It does not work, and this is
load-bearing.** `main_v4._act` (lines 201–223) compiles a plan **once per day**
(`if st["compiled_day"] != day`) and then, for the rest of that day, indexes a precomputed
per-worker script by `index = hour - compiled.start_turn`. It is open-loop within the day — the
same objection we levelled at `boatlee`, just at a 23-hour scale instead of 719.

Measured: run `main_v4` in both seats (seed 11), then cold-query `main_v4.agent(obs)` at each of its
own trajectory states with module state cleared. **It reproduces its own action 166 / 699 = 23.7% of
the time** — 29/29 at hour 0, 30/30 at hour 1, and **0/30 at hour 2**, 1/30 at hour 3, 3–27%
thereafter [MEASURED, verified]. The cause: a cold query recompiles with `start_turn = hour` and
returns step 0 of a **freshly re-planned day**, not step *h* of the original plan.

### The oracle ladder — in order, with an explicit skip

1. **`agent.verify.compile_day(obs, plan, hands, turns, start_turn=hour, cash=0.0) → ops[0]`.**
   This one takes an observation, so it is askable by signature, and it *is* state-responsive —
   removing all PLANT tiles changed its answer 3 times out of 3 [MEASURED, verified].
   **Two limits, both declared out of scope up front.** `cash=0.0` is hardcoded at
   `main_v4.py:213`, so money perturbations legitimately do nothing. And season state
   (`_watch_opponent`, `branch_points`, `season_planner`, the module-level `_STATE`) is **not
   reconstructible from a single observation** and is not part of the oracle.
   > **Important:** this is a *different policy* from the one that scores in the arena. So any
   > "score the teacher on the same seeds" check must score **this function**, not
   > `main_v4.agent`, or it compares two different things.
2. **Any reactive engine in the repo** (`executor_v7`, `kagsim_champion`). A reactive engine is a
   state → action function by construction, so it is askable at any state even if the compiler is
   not.
3. **Skip DAgger entirely and go to Chapter 8.** PPO fixes drift too, just more expensively.

### The entry probe — make teacher quality an entry test, not a discovery (E94)

Before aggregating a single label, measure two things:

1. **The fraction of oracle queries that return a no-op or throw an exception at the states our BC
   model actually visits**, against a stated bound. This is the "prove the change fired" check: if
   the teacher is being asked questions it was never designed for, we would be aggregating garbage
   labels and would never see it in the money numbers.
2. **Does the oracle actually outscore our BC model on the same seed block?** If it does not, the
   oracle is not an expert *here*, and DAgger cannot help **by construction**.

## How we know we're done

After two rounds of aggregation, online money improves over Chapter 6's number by an interval that
excludes zero at ≥80 games, both seats (**E95**), and the handover curve visibly flattens.

## When we stop

**If two rounds move money less than the 80-game confidence interval, or if no oracle passes the
entry probe, we skip to Chapter 8 with the Chapter 6 checkpoint.** This is graceful degradation,
not failure. Do not do a third round hoping.

## What you should be able to explain

Why DAgger is O(T) where plain BC is O(T²), and what "no-regret online learning" actually buys.
When you *cannot* query an expert, what replaces it (GAIL, offline RL) and why we do not need those
here.

**Note: this learning exit survives even if the chapter is skipped.** The handover curve from
Chapter 6 measures compounding error with no oracle at all, so the teaching value of this chapter is
unkillable even when its engineering value is.

## Reading

- Ross, Gordon & Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret
  Online Learning** (AISTATS 2011) — DAgger itself.
- Rajeswaran et al., **Learning Complex Dexterous Manipulation with Deep RL and Demonstrations**
  (RSS 2018) — "DAPG": BC pretraining plus a policy-gradient term with a decaying demonstration
  loss. This is the concrete recipe for Chapter 8's KL schedule.
- Ho & Ermon, **Generative Adversarial Imitation Learning** (2016) — read it for the framing of
  what having a queryable expert lets you avoid.

---

# Chapter 8 — Reinforcement learning: the model teaches itself

## The idea

BC can at best copy the teacher. To go *beyond* the teacher, the model must learn from **results**
instead of examples. Four words:

- **Reward** — a score for how the game went. Ours is natural: who ended with more money.
- **Policy gradient** — the core trick. Play games, then adjust the network to make the actions
  from *winning* games more likely and the actions from *losing* games less likely. That is the
  whole idea. Everything else is stabilization.
- **PPO (Proximal Policy Optimization)** — the standard stabilized version. "Proximal" means: do
  not change the policy too much in one update, or it collapses. `Orbit-Wars/src/orbit_ppo.py` is a
  complete working PPO we can read line by line and port.
- **Self-play** — the opponent is previous frozen versions of our own model, plus the scripted bots
  in this repo.

**Why we did BC first.** A network starting from random weights flails for millions of games before
discovering that planting is good. Starting from the BC model it plays sensibly from game one and
spends its practice improving rather than discovering farming. On this hardware that is not a
nicety — the numbers below say it is the difference between feasible and not.

## What we build

`v2`: **0.8–1.5M parameters, wide and shallow** (d ≈ 256, 2–4 layers). PPO on top of the BC
checkpoint.

## The details

### The compute argument that sizes the model

PPO's cost per collected step is `1/inference_rate + n_epochs/train_rate`. Using the measured
throughput table from Chapter 5, at `n_epochs = 2`:

| model size | cost per step | steps per day |
|---|---|---|
| 4.86M | **2.9 ms** | ~30M |
| 0.81M | **0.70 ms** | ~124M |

Over roughly 150 hours of overnight compute across six weeks that is **~190M steps at 5M parameters
versus ~775M at 0.8M** [MEASURED, single-source, derived from the verified throughput table].
Orbit Wars' 7th place used **2.2 billion** steps. Only the smaller model is within an order of
magnitude.

> **From-scratch PPO at 5M parameters is not reachable on this machine. The warm start is the
> enabling condition, not a nicety. Size the model to the step budget, not to the writeups.**

### Entry requirements

The Chapter 6 (or Chapter 7) checkpoint; a stated step budget; and **now** the feature extractor
must also be callable from a kagsim state, with `make verify` still green. This was deliberately
deferred from Chapter 3 — here fast rollouts make it a genuine requirement rather than extra
surface area.

### The settings, each with its evidence

- **Reward: the final result only**, as `sign(my_money − opp_money)`. All six top-10 finishers agreed
  shaping hurt; 7th place ablated shaping and scored **34.6%** against its own baseline.
  *Open question worth one A/B: if Kaggle ranks on absolute score rather than head-to-head, then
  margin-sign is the wrong objective. `PLAN_v4` §0 targets winrate, so follow that — but test
  normalized margin once and record the answer.*
- **Discount γ = 0.999, not 1.0.** A **discount** decides how much a future dollar counts now.
  1st place's single biggest regret was that γ = 1.0 made their agent stall.
- **KL-anchored to the BC checkpoint, with a decaying coefficient.** **KL** measures how far the new
  policy has drifted from the old one; anchoring keeps early RL from destroying the prior we paid
  for. The coefficient comes from measurement (E96), not folklore.
- **Opponent pool: PFSP**, weighted toward opponents we beat about half the time, seeded with
  `boatlee`, the compiler, and `search/exploiters.py`'s `flooder` and `tomato_rusher` [MEASURED,
  verified]. **Never a live copy of ourselves.** Note that Orbit-Wars' "league" is weaker than its
  name: `_get_last_opponent` (line 237) returns `league[-1]` — always the newest checkpoint — so it
  is near-naive self-play. We do better. The local proof this matters: E10, where a correctly
  seed-validated champion lost **0 of 80** to a naive dumper it had never faced.
- **Seat symmetry: flip the seat per environment** rather than duplicating the data.
- **Log per-head entropy, `clip_frac` and `approx_kl` every iteration.** **Entropy** measures how
  spread out the model's choices are; when it collapses, the model has locked into one behaviour.
  **A flat-lined head is a stop condition.** 3rd place said entropy annealing was "by far the most
  important knob."
- **Not applicable, and we say so:** ending games early once they are decided gave 6th and 7th place
  a big boost, but there is no clean analogue here — economic accumulation has no settled state
  before the last day. Do not force it.

### The overnight experiment worth running (E96)

Three PPO runs at an identical fixed step budget: **from scratch**, **from BC with no KL anchor**,
and **from BC with a decaying KL anchor**. This is a direct local replication of the field's one
real disagreement: 5th place built everything on BC init, while **2nd and 7th both found their
from-scratch runs beat their carefully initialized ones**.

If the un-anchored run diverges immediately and the anchored run never moves, we have bracketed the
coefficient in one night. And if from-scratch wins, the compute argument above is wrong and we want
to know that now.

**But note what this result is and is not.** If from-scratch matches from-BC, that is a *research
finding*, not a plan change — 2nd and 7th place found exactly this and still shipped. Record it,
keep the BC init as long as it is not actively worse (it is free), and move the KL-tuning budget
elsewhere.

## How we know we're done

**≥60% against the frozen BC checkpoint**, at ≥80 games, both seats, on fresh seeds (**E97**). No
regression against `DEFAULT_POOL`. 99th-percentile turn time under 100 ms.

## When we stop

**If 100 million steps go by without reaching 55% against the frozen BC checkpoint, stop and drop
to the market-only control surface** (let scripted code run the farm and let the model handle only
buy/sell/hold timing — which is where CLAUDE.md says a model belongs anyway: "the farm half is
analytically solvable; hand-code it").

**Do not add reward shaping. Do not grow the model.** Those are the two tempting moves and both are
refuted.

Separately: if `approx_kl` spikes and entropy collapses in the first 5M steps, the KL anchor is
mis-scheduled. Fix that before spending the rest of the budget.

## What you should be able to explain

The whole chain in your own words: REINFORCE → why you need a baseline → actor-critic → GAE → the
**PPO clip**. And why 6 out of 6 published writeups found reward shaping made things worse.

## Reading

- Schulman et al., **High-Dimensional Continuous Control Using GAE** (2015), then **Proximal Policy
  Optimization Algorithms** (2017), in that order.
- Huang et al., **The 37 Implementation Details of Proximal Policy Optimization** (ICLR Blog Track
  2022). Treat it as a checklist, not a read — advantage normalization, orthogonal init, value
  clipping and learning-rate annealing are where the silent failures live.
- Berner et al., **Dota 2 with Large Scale Deep RL** (OpenAI Five, 2019), the self-play section —
  80% latest opponent / 20% past. Pair it with the AlphaStar league (main agents, main exploiters,
  league exploiters, PFSP) from the Nature paper you already read in Chapter 6.

---

# Chapter 9 (optional) — Scale up

## The idea

If and only if Chapter 8 is working but too slow on the Mac: rent a GPU and port training to JAX.
That is a conversion **toward** our reference code — Orbit-Wars is written in JAX — with a working
PyTorch version to check numbers against.

## The details

### When we are allowed to start

Two conditions, both required:

1. Chapter 8 is measured **training-throughput-bound** — plausible, since the model is roughly 14×
   the simulator's cost — **and**
2. a GPU actually exists (rented, paid for).

Buying throughput before it is the measured bottleneck is how weeks disappear.

### The six rules we adopt from day one, because they are free

Follow these while writing the **PyTorch** code in Chapters 5–8, and this chapter becomes a port
instead of a rewrite:

1. **The forward pass is a pure function of `(params, batch) → logits`.** No `self`, no module
   state, no in-place operations. ("Pure" = same inputs always give the same outputs, and it
   changes nothing outside itself.)
2. **Masks are data, never control flow.** No `if` on a tensor value, anywhere.
3. **Static shapes everywhere** — workers padded to 16, market slots to 10, tiles fixed at 100.
   JAX compiles per shape, so a changing shape means recompiling.
4. **The loss is a pure function of `(params, batch)`**, with optimizer state passed explicitly.
5. **The random seed is threaded through explicitly**, never a global.
6. **The per-worker decode loop is bounded and fixed** (16 workers, 10 market slots) so it maps
   mechanically onto `lax.scan`.

**These same rules retire two other risks for free.** Rules 1–3 are exactly what would make a
numpy-only inference path a ~20-line adapter if Chapter 6's timing ever demanded one. And they make
the parity test trivial to write.

### The parity test

`tests/test_inference_parity.py`: same observation → PyTorch forward and JAX (or numpy) forward →
assert **max absolute logit difference < 1e-4** *and* **identical argmax over 500 random states**.

Why it must be a test and not a hope: a silent parity break produces a model that scores like an
untrained one, and **it is indistinguishable from "BC didn't work"**.

## How we know we're done

JAX logits match PyTorch to 1e-4, and training is **at least 5× faster** (**E98**).

## When we stop

**Under 5× — stay in PyTorch.** The port is not worth carrying two implementations for less.

## What you should be able to explain

Why the six rules above made this a port rather than a rewrite. What "static shapes" buys a
compiler.

---

# Closing section — the cross-chapter material

## Risk list, worst first

Ordered by probability × how expensive it is to discover late.

| # | Risk | Chapter | What protects us |
|---|---|---|---|
| 1 | ~~Not enough replays, or the API is closed~~ **RETIRED** | 2 | 100 verified games are on disk. The compiler seam stays as insurance. |
| 1b | **Single-teacher, win-filtered corpus** — narrow distribution, losing positions barely represented | 2, 6 | Named openly; expect a steeper handover curve; Chapters 7–8 are the repair |
| 2 | **The teacher is a script** — the model learns a clock — **Test A PASSED** | 2, 6 | Test A measured (74–96% of steps differ); Test B (E86) still to run; the ≥5pp anti-clock ablation on steps ≥32 as a hard stop |
| 3 | **The (state, action) off-by-one** — trains fine, converges fine, clones the wrong move | 3 | The 1,438/1,438 hand-roster assertion, hard-asserted in the decoder |
| 4 | **The mask rejects expert actions** — every loss number becomes meaningless | 3, 5 | `n_expert_actions_rejected_by_mask == 0` as a hard exit; the vocabulary as one asserted constant |
| 5 | **The shortest-path measurement does not hold corpus-wide** | 3, 5 | Both label sets emitted from one pass; the fallback is macro-with-`MOVE`, never raw actions |
| 6 | **No askable teacher exists** | 7 | The oracle ladder with an explicit skip; the learning exit survives via the handover curve |
| 7 | **Not enough compute for PPO** | 8 | Size to the step budget (0.8–1.5M, wide and shallow); fixed step budget per experiment; drop to the market-only surface |
| 8 | **A numpy/torch parity break** — scores like an untrained model | 6, 9 | Pure-function forward; the parity test with 500 states |
| 9 | **Feature/weight skew when reloading a checkpoint** | all | `FEATURE_VERSION` asserted at load, code hash, dataset manifest hash (below) |
| 10 | **A timing blowout on Kaggle, invisible locally** | 6 | The overage caveat; keep the import small; budget from the measured CPU table |

## Where the code lives

```
bc/
  sources/replay.py     # provided replay JSON -> (state, action) stream     (Chapter 2)
  sources/compiler.py   # compile_day rollouts -> (state, action) stream     (Chapter 2, the insurance seam)
  acquire.py            # downloader; T0-a..d — ONLY if we ever need more games
  decode.py             # the four assertions; raw AND macro labels          (Chapter 3)
  features.py           # VERBS / ITEMS / MARKET_OPS / QTY_BINS / N_UNIT_SLOTS; the extractor
  dataset.py            # shard build, manifest hashing                      (Chapters 2-3)
  model.py              # forward(params, batch) -> logits, a pure function  (Chapter 5)
  train.py                                                                   # (Chapter 5)
  infer_numpy.py        # ~20 lines, only if Chapter 6's timing demands it
  eval.py               # the four rungs                                     (Chapter 6)
data/sample_data_training_model/      # THE CORPUS — read-only, never modified
  {train,val,test}/{episode_id}.json  #   70 / 15 / 15 games
  manifest.csv                        #   ryo_seat, opponent, rewards, strata
  split_summary.json
data/replays/{episode_id}.json.gz     # only if we ever download more
data/shards/{split}/*.npz             # gitignored; {split} copied from the corpus layout
tests/test_bc_*.py
docs/learning.md                      # one page per chapter, your own words
```

**Make targets to add:** `make bc-shards`, `make bc-train`, `make bc-eval`. BC checkpoints wire into
`harness/registry.py` and `arena/registry.py` as `kind: "bc"` with `params: {"checkpoint": path}`.

**`make verify` scope is unchanged for Chapters 1–7.** The extractor reads the official
environment's observation dictionary only. kagsim-state support is a **Chapter 8 entry
requirement**.

## Checkpoint reproducibility

Every saved checkpoint stores four things: **the weights, a `FEATURE_VERSION` integer that is
asserted on load, a code hash, and a dataset manifest hash** (the episode ids plus `module_version`
plus `config_hash`).

Why this is not optional: silent feature/weight skew produces a model that scores like an untrained
one, and **it is indistinguishable from "BC didn't work"**. That is the same class of failure as
E21. `tests/test_bc_checkpoint_roundtrip.py` covers it.

## The first two weeks (evenings plus one weekend)

The old version of this list spent three evenings downloading and vetting data. **That work is
done** — the corpus arrived verified, and the acquisition items are parked in Chapter 2's
"only if we need more data" section. The list now starts at the decoder.

0. **Already done.** Corpus verified: 100 games, schema-matched, all 1.32.7, all DONE, unique
   seeds. Teacher chosen (Ryo, all 100 games, all wins). Test A measured and passed (**E88**).
1. **Evening 1.** Install `torch` into `.venv`; re-run `make test` and `make verify` and confirm
   both are green. Read `manifest.csv` end to end — opponents, `ryo_seat`, margins, strata.
2. **Evening 2.** Clock-vs-state test on `boatlee` as the positive control — we already know the
   answer there, so it validates the harness before we point it at Ryo.
3. **Weekend, morning.** Clock-vs-state test on Ryo's corpus. Record as **E86**. This is the last
   program-level stop rule; after it, we build.
4. **Weekend, afternoon.** `bc/decode.py` plus the four assertions, reading
   `data/sample_data_training_model/` and cloning `ryo_seat`. Write the verb/item vocabulary as one
   asserted constant — **before anyone writes a head width.**
5. **Evenings 3–4.** `bc/features.py`. Compute `frac_segments_shortest_path` corpus-wide on Ryo's
   seat and make the macro-vs-raw decision. Record as **E87**.
6. **Evening 5.** Build the shards under the provided split names. Measure sizes against the 1.5 MB
   threshold. Recompute the majority-class floors on Ryo's seat, both over all steps and over
   steps ≥ 32.
7. **Evening 6.** Write the Chapter 1–2 entry in `docs/learning.md`.

## The experiment registry

`docs/experiments.md` is at **E85** and `docs/decisions.md` at **D21** [MEASURED, verified], so this
plan starts at E86 and D22. **Every measurement claims its number before it runs.** The numbers are
a registry, not a schedule — E88 ran before E86.

| E | Chapter | What we measure | What it decides |
|---|---|---|---|
| **E86** | 2 | Clock-vs-state: step-number-only model vs state-features-minus-step. `boatlee` first as positive control, then Ryo's corpus | Whether an imitable policy exists. **The program-level stop rule. Still to run.** |
| **E87** | 3 | `frac_segments_shortest_path`, corpus-wide on Ryo's seat, per shard | Macro action space, or macro-with-`MOVE` |
| **E88** | 2 | **MEASURED — PASSED.** Action-sequence hashes across all 100 games: **zero duplicates**. Pairwise step-by-step comparison of Ryo's games (24 games, 12 disjoint pairs): **74–96% of steps differ, median 96%**; identical steps concentrate at indices **0–31** | Teacher is genuinely reactive; the tape-recorder risk is retired. Also gives us the fixed ~30-step opening, which every offline metric now reports around |
| **E89** | 5 | Per-head agreement with Wilson intervals against floors **recomputed on Ryo's seat** (not the old 16.3% / 19.3% sample figures), all steps and steps ≥ 32 | Whether v1 learned anything at all |
| **E90** | 6 | The step/day/hour ablation, **on steps ≥ 32** | **Stop rule** at under 5 points of drop |
| **E91** | 6 | The handover curve, plus agreement split by where the state came from | The size and the season-phase of the compounding-error gap |
| **E92** | 6 | Online money vs `DEFAULT_POOL`, ≥80 games, both seats, with `Observer` counters | Whether the offline number carries online |
| **E93** | 6 | Import cost plus 99th-percentile turn time from one timed `env.run` | numpy vs torch at submission time |
| **E94** | 7 | Oracle probe: degenerate-response rate, and does the oracle outscore our model on the same seeds | **Chapter entry**; the skip-to-Chapter-8 decision |
| **E95** | 7 | Change in online money over 2 DAgger rounds vs the 80-game interval | Chapter exit / stop |
| **E96** | 8 | PPO from scratch vs from BC vs from BC+KL, at an equal step budget | Whether the warm start is a floor or a cage; the KL coefficient |
| **E97** | 8 | ≥60% vs the frozen BC checkpoint, ≥80 games, both seats, fresh seeds | Chapter exit |
| **E98** | 9 | JAX↔PyTorch logit parity plus training speedup | Port, or stay |
| E99 | backlog | Ablate the one-worker-at-a-time decode (decide all workers in parallel) — loss **and** ≥80-game winrate | Can we delete 16 sequential steps? |
| E100 | backlog | Market as an ordered sequence vs an unordered bag of orders, same shards, one epoch | Can we delete 10 sequential steps? |
| E101 | backlog | Full opponent tile tokens vs the summary token (needs >2% held-out loss improvement to win) | Worth +40% compute? |
| E102 | backlog | Bucketed quantities vs a continuous quantity head | Loss form |

## Backlog (not chapters — no entry, exit, or stop rule)

- **Auxiliary prediction heads.** 7th place added heads predicting the state 2/8/32/64 turns ahead,
  discarded at play time; it "helped quite a bit". Cheap to add during BC, and it forces a world
  model into the shared trunk.
- **A 3.1M transformer, wide and shallow only**, promoted solely on ≥80-game arena winrate, never
  on held-out loss.
- **Offline RL reading — to understand why we are declining it**, not to implement it: Levine et
  al., **Offline RL: Tutorial, Review, and Perspectives** (2020) and Fujimoto & Gu, **A Minimalist
  Approach to Offline RL** (TD3+BC, 2021, four pages). Offline RL exists to extract a policy better
  than the data when you *cannot* interact with the environment. We can — kagsim runs at 51,600
  steps/second — so the entire motivation is absent. Read TD3+BC anyway: "BC plus a small RL term"
  being competitive with elaborate machinery is a direct sanity check on Chapter 8's KL schedule.

## Provenance

This document was produced by a five-agent panel: three proposers (architecture; data and
evaluation; roadmap and RL) and two verifiers — a plan critic that red-teamed the merge, and a
fact-checker that independently re-derived the load-bearing measurements on this machine without
reading the proposers' code. **Where a proposal and the fact-checker disagreed, the fact-checker
won.** Eight proposer claims were corrected or dropped that way; each correction is stated inline in
the chapter where it lands, so that nobody re-derives a refuted number from an old draft.

The original panel document is preserved verbatim at `docs/PLAN_BC_panel_original.md`.
