# Kaggriculture — Project Plan

Working plan for building a competitive Kaggriculture agent, informed by the Orbit Wars PPO
reference in `reference/orbit_war/` (see `reference/orbit_war/OVERVIEW.md`).

Everything in "Verified facts" and "The economics" was read off or computed from the installed
environment source, not the docs:
`/opt/miniconda3/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`

> ## Evidence standard
>
> Every claim in this document carries its provenance. Nothing is asserted from reasoning alone.
>
> | tag | meaning |
> |---|---|
> | **[MEASURED Ex]** | observed in play, with an experiment ID in `docs/experiments.md` |
> | **[COMPUTED]** | derived from the environment source — arithmetic, *not* evidence about play |
> | **[ASSUMED]** | plausible, untested. **Treat as a hypothesis, not a finding.** |
> | **[REFUTED Ex]** | measured and found false; kept visible rather than deleted |
>
> **[COMPUTED] is not [MEASURED].** A price-curve integral says what a market *could* absorb, not
> what an agent achieves or what an opponent leaves available. Three claims in §2 were computed,
> believed, and then contradicted by play (see the realised column).
>
> **Status.** This document is the *plan*; `docs/experiments.md` is the *record*, and where they
> disagree the record wins. Several predictions here have been measured and revised — most
> importantly §2.5's mechanism (E11) and §2's land and labour claims (E1, E6). Revisions are marked
> inline rather than deleted, because being wrong in a specific, checkable way is what made the
> measurement worth running. Live state: `TASKS.md`.

---

## 0. Thesis

**The deliverable is a learned model.** The competitive field will be models (in Orbit Wars the top
of the leaderboard was), and a hand-tuned script is a ceiling, not a plan.

But the game splits cleanly into two halves that want completely different treatment:

- **The farm half is single-agent, deterministic, and analytically solvable.** Watering schedules,
  harvest timing, routing, feed logistics — all computable. Learning these wastes samples on
  problems arithmetic already answers.
- **The market half is a genuine adversarial race** against an adaptive opponent (see §2.5).
  Static rules get exploited here. **This is where the model earns its keep.**
  *Revised by measurement (E11): the race is over **production timing**, not sale timing. Selling
  is trivially always-now — a policy that cannot hold inventory beats the tuned reserve-and-forecast
  policy 84.4%. The contested resource is arriving on the price curve first, decided days earlier
  by planting and harvest scheduling.*

So the scripted engine is **not an alternative to the model — it is on the critical path to it**:

1. It is the **opponent to train against**. Self-play from scratch over 720 steps with 30-day payoff
   horizons will not bootstrap from random.
2. It is the **expert for behavior cloning**, which warm-starts the policy out of the dead zone.
3. It **defines the macro-action space**. The raw action space (per-unit ops × a 10-slot market
   list × up to a dozen units) is far too large to learn end-to-end; the executor collapses it to a
   ~20-dim daily plan vector.
4. It is the **fallback submission** and the **evaluation baseline**.

Three facts set the strategy underneath all of it:

1. **The reward is raw final money** (`kaggriculture.py:940`), not ±1. Dense, observable every step.
2. **Each product's market has a seasonal capacity set by how many shops demand it**
   — *not* by where its price curve floors. **[REFUTED E15]** in its original form: this thesis
   originally said capacity was the one-shot integral, which ranked products almost inversely to
   the truth and cost a 2.4x improvement. The town *removes* inventory continuously, so
   `capacity ~= one-shot cap + drain/day x days`, and the drain term dominates. Melon is demanded
   by **zero** shops; milk by three. See §2.
3. **Labor is nearly free** ($20/day for six hands) while **land and market absorption are the real
   constraints.** The binding resource is *actions × market capacity*, not money.

The bar is low. The built-in `starter` agent finishes a 720-step season with **$3,496** from a
$3,000 start — a $496 profit. `random` ends at **$0** (it buys seeds until broke).

---

## 1. Verified environment facts

Read from source. Where I checked, **`docs/README.md` is accurate** — the yield windows, care
banking, decay rules, and price formula all match the code. Trust the docs, but these are the
details that matter and are easy to misread:

| Fact | Source | Why it matters |
|---|---|---|
| `reward = float(farm["money"])` | `:940` | Absolute money, not win/loss. Maximize your own bank. |
| Episode ends at `step >= episodeSteps - 2` | `:937` | 720 steps = 30 days × 24 turns. |
| `actTimeout: 1`s, `remainingOverageTime: 60` | `.json` | ~1s/turn budget. Env itself costs 1.2ms/step, so it's all yours. |
| Water bonus window = `[(max_yield_day+1)//2, max_yield_day]` | `:384` | Watering *outside* the window adds nothing but survival. |
| One-time crops start at `yield_units = 1` | `:209` | You get 1 unit even with zero watering. |
| `consecutive_unwatered` starts at **1** on planting | `:208` | Must water on the planting day or it weeds that night. |
| Death at `consecutive_unwatered >= 2` | `:761` | **Watering every *other* day is enough for survival.** |
| Same for animals: `consecutive_unfed >= 2` | `:795` | **Feeding every other day keeps animals alive.** |
| `fertilizer_available = True` for every surviving animal, unconditionally | `:809` | Fertilizer needs no feeding and no care. Free money. |
| Care bonus is destroyed on an unfed production day | `:804` | Alternate-feeding and `CARE` are **mutually exclusive** — pick one. |
| Ongoing crops cap at `max_yield` *productions*, not units | `:774` | Tomato = 4 productions × (1 or 2) = 8 units max, ever. |
| Decay: −1 `yield_units` every 2 steps past `max_lifespan_step` | `:740` | A ripe crop is wiped in ~8 turns. Harvest windows are tight. |
| Atomic PLANT: over-requesting a crop drops **all** plants of it that turn | `:905` | Multi-unit planting needs a seed-budget allocator. |
| Movement onto `LOCKED` tiles is legal; tile ops are not | `:314` | Units can path across unbought quadrants. |
| Shed cap 100, overflow **discarded**, no inventory bypass | `:821` | You cannot warehouse. Production must match daily sell-through. |
| `BUY_PRODUCT` quotes at `inventory - 1` | `:578` | Buy→sell round-trip nets exactly zero. No arbitrage. |
| Sales at the $1 floor do **not** add to market inventory | `:636` | The floor stays responsive; it never gets "more crashed". |
| Market orders execute *fully* within their slot | `:562` | `SELL EGG 100` is one of your 10 orders. Order count is not a constraint. |
| Hire cost `fib(n)`, resets daily | `:675` | 1,1,2,3,5,8,13,21… — six hands = **$20/day**. |
| Weeds spawn only on **empty** unlocked tiles, p=0.005 | `:817` | Keep tiles full → near-zero weed pressure. |
| Only `WHEAT` and `FERTILIZER` are buyable | `:575` | You cannot buy back premium goods to prop up their price. |

**Correction to my earlier read:** I assumed self-play would matter little. Half true — the *farm*
half is single-agent, but both players sell into one price curve, and for the capacity-limited
products (melon, wool, milk, strawberry) it is close to a race. That race is worth modeling.

---

## 2. The economics (this is the edge)

Cumulative revenue from selling into each market from the starting inventory `I0` until the price
hits the $1 floor, computed by walking the actual `market_price()` function:

**[COMPUTED]** — one-shot integrals of `market_price()`, ignoring town regeneration and ignoring
what an opponent takes first. See the realised column below before trusting any of it.

| Product | Units to $1 | Cumulative revenue | Price @ +50 | Price @ +100 | Verdict |
|---|---:|---:|---:|---:|---|
| **EGG** | 3000+ | **$113,763** | $43 | $42 | **Infinite sink.** The scale play. |
| **WHEAT** | 3000+ | **$57,464** | $22 | $21 | Infinite sink + animal feed. |
| **MELON** | 158 | **$26,485** | $225 | $150 | **Best revenue density.** Highest $/action in the game. |
| **FERTILIZER** | 493 | **$25,045** | $90 | $80 | Free byproduct of every animal. |
| **TOMATO** | 529 | $11,128 | $42 | $35 | Mediocre. |
| **CARROT** | 842 | $10,680 | $27 | $23 | Mediocre filler. |
| **WOOL** | 59 | $7,928 | $55 | $1 | **Trap.** Sheep costs $500. |
| **MILK** | 76 | $6,181 | $55 | $1 | **Trap.** Cow costs $400. |
| **STRAWBERRY** | 62 | $3,809 | $24 | $1 | **Worst in game.** $100 seed, 16 days, $3.8k total. |

### ⚠ The one-shot model above is the wrong model — **[REFUTED E15]**

A price-curve integral asks "how much can I dump before the price floors?". The season asks a
different question, because the town **removes** inventory continuously and the removal is driven
by how many *shops* demand the product. Correct capacity is roughly
`one-shot cap + daily drain x days`, and the drain term dominates:

| product | shops demanding it | drain/day | one-shot cap | **seasonal capacity** |
|---|---:|---:|---:|---:|
| WHEAT | 5 | 29 | $75,464 | ~$86k |
| **STRAWBERRY** | 4 | 24 | **$3,809** | **~$47k** |
| **MILK** | 3 | 19 | **$6,181** | **~$52k** |
| CARROT | 2 | 19 | $10,680 | ~$20k |
| EGG | 2 | 14 | $149,763 | ~$160k |
| TOMATO | 2 | 14 | $11,128 | ~$24k |
| **WOOL** | 1 (x2) | 14 | **$7,928** | **~$51k** |
| **MELON** | **0** | 5 | $26,485 | ~$44k, and *unrenewable* |
| FERTILIZER | 0 | 0 | $25,045 | $25,045 |

**Melon is demanded by zero shops.** Only the town centre touches it, which is why it saturates at
82% and never recovers — while the three products this plan called traps (milk, wool, strawberry)
are among the largest markets in the game once regeneration is counted.

Every "trap" verdict in the table above came from the one-shot number and is wrong. Measured
consequences: strawberry exceeds its "cap" in play (E13); cow and sheep herds beat the goose-only
champion **48/48 each** (E15); `o-sprinter` extracted 99% of melon and still lost (E14).

**Rule:** for any product, check how many shops demand it before judging its market.

> ### ⚠ And then: we never reach either regime — **[MEASURED E16]**
>
> Both capacity tables above describe *how much a market absorbs before flooring*. Measured, the
> town drains faster than two players can supply: inventory sits **below** `I0` all season for
> every shop-demanded product (wheat ends at −756, strawberry −494), so prices rise and nothing
> saturates. Melon is the only exception — no shop demands it, so it is the one product two players
> can flood, which is why every melon-era finding was about racing.
>
> **The binding constraint is production, not market absorption.** Capacity still decides the
> *ranking* of products (shop count ~ how much the town will keep taking), but "how much can this
> market absorb" is not the question to optimize against.

### What the champion actually realises — **[MEASURED E13]**

| Product | computed cap | realised in play | share | verdict on the computed claim |
|---|---:|---:|---:|---|
| **EGG** | $113,763 | **$8,651** | **8%** | "the scale play" — **unrealised**; ~$105k untapped |
| WHEAT | $57,464 | $12,402 | 22% | large headroom |
| MELON | $26,485 | $21,624 | **82%** | near saturation — little left |
| FERTILIZER | $25,045 | $8,258 | 33% | headroom |
| CARROT | $10,680 | $2,128 | 20% | headroom |
| **STRAWBERRY** | $3,809 | **$4,612** | **121%** | **[REFUTED E13]** — "worst in game, skip entirely" is wrong |

Two corrections fall out:

- ~~**Skip strawberry entirely.**~~ It *exceeds* its computed one-shot cap, because town drain
  regenerates it faster than any product except wheat (32/day post-day-20). The CEM-tuned champion
  independently chose a 0.30 strawberry weight. The one-shot integral was the wrong model.
- **Melon is near its ceiling (82%); eggs are at 8%.** Melon optimization is close to exhausted and
  the untapped capacity is in livestock. This is the sharpest available pointer for T2.3.

Town drain partially regenerates these — at full unlock post-day-20, roughly 38 wheat, 32
strawberry, 26 carrot, 26 milk, 20 tomato/egg/wool, and only **8 melon** per day. So capacity-limited
products recover slowly, and **melon barely recovers at all** — its entire season budget is close to
the one-shot 158 units.

### Value per action (the real currency)

Every action is one unit-turn. Rough lifecycle math at realistic prices:

| Line | Actions | Output | ≈ $/action | Capacity |
|---|---:|---|---:|---|
| **Melon tile** (11 days) | ~10 | 6 melons | ~$90 | **Unrenewable** — zero shop demand; saturates at ~82% |
| **Goose** (steady state) | 3–4/day | 1–2 eggs + 1 fertilizer | **~$25–33** | Uncapped |
| **Wheat tile** (5 days) | ~6 | 4 wheat | ~$12 | Uncapped |
| **Tomato tile** (12 days) | ~11 | 4 tomatoes | ~$12 | Mediocre |
| **Carrot tile** (4 days) | ~5 | 3 carrots | ~$11 | Mediocre |
| ~~Strawberry / cow / sheep~~ | — | — | **[REFUTED E15]** | **Cows beat the goose champion 48/48; the tuned champion runs 7 of them** |

### Falls straight out of this

- **Sell fertilizer, don't use it.** Fertilizing wheat converts a ~$50–90 fertilizer into +2 wheat
  (~$42) *and* costs an action. It's value-destroying at almost any fertilizer price.
- ~~**Geese are the only animal worth owning.**~~ **[REFUTED E15]** Cows are strictly better: milk
  regenerates at ~19/day against eggs' ~14, a cow needs a harvest only every second day where a
  goose needs one daily, and every animal yields the same 1 fertilizer/day regardless of species.
  Cow herds beat the goose-only champion **48/48**; the tuned champion runs **7 cows and no geese**.
  Cows and sheep were never rejected on evidence — `BUILD_PASTURE` was simply absent from the
  engine until V3.
- ~~**Skip strawberry entirely.**~~ **[REFUTED E13/E15]** Four shops demand it — the second-highest
  drain in the game — so its seasonal capacity is ~$47k, not the $3,809 the one-shot integral gave.

> **⚠️ Two claims below were measured and overturned — see `docs/experiments.md` E1.**
>
> - ~~**Buy all three quadrants ASAP.**~~ **Wrong at current labour levels.** Not buying land
>   *doubles* final money ($12,006 vs $5,960) and cuts movement from 70% to 46%. 25 tiles near the
>   shed beat 100 tiles you cannot reach. Land is a consequence of solving routing (T1.2), not a
>   precondition for it.
> - ~~**Hire aggressively.**~~ **Only to ~8 hands.** The Fibonacci cost is negligible at 6
>   (~$20/day) but reaches ~$376/day at 12 and ~$2,583/day at 16. Twelve hands **bankrupts** the
>   agent. Measured optimum ≈ 8.
>
> The generalizable lesson: **travel, not money, is the scarce resource**, and 46–73% of all unit
> actions are movement. Anything that adds tiles without adding reachable throughput loses.

### Opening hypothesis: the melon rush

Plant ~25 melons on day 0 ($2,000 of the $3,000 start), water for survival, water daily through
ages 6–10, harvest ~150 melons on day 10, sell into a fresh market at ~$168 average ≈ **$25k**.
That funds all land and a large goose operation by day 10.

Risks: 10 days of zero cash flow; if the opponent also melon-rushes you split the 158-unit budget.
**Flagged as a hypothesis — validate in Phase 0 before committing.**

---

## 2.5 The adversarial core: public supply, private inventory

This is the part of the game that actually requires a model.

**`farms` is shared** (`kaggriculture.json:95`) — the opponent sees every one of your tiles,
including `crop` and `planted_day`. Crop maturity is deterministic. **Therefore your entire future
supply schedule is public information.** An opponent can compute the exact day your 25 melons ripen
and dump theirs on day 9 to front-run you. With only 158 units of melon capacity for the whole
season and ~8/day of town regeneration, front-running is worth most of the melon money.

**`private` is not shared** (`:99`) — your shed contents and per-unit inventories are hidden. So once
harvested, *when* you sell is private information.

That makes the market a game of **public production commitment + private inventory + simultaneous
lockstep orders** (`:562`, where both players are quoted the same price for each unit before either
commits). Concretely:

- Planting melon is a **public, irreversible commitment** 10 days ahead of the payoff.
- Holding harvested melon in the shed (cap 100) is a **private option** on sell timing.
- Both players dumping the same capacity-limited product on the same day is the mutual-destruction
  outcome; both get pushed toward the $1 floor.

Counter-play that a static reservation rule will not find: staggering plantings to smear your public
supply curve, harvesting early into the shed to convert public commitment into private optionality,
baiting a front-run and selling into the recovery, and reading the opponent's tile counts to
forecast their supply and pre-empt it.

**Implication for the model:** the opponent's farm tiles are a *forecast of their future supply* and
belong in the market policy's observation as a first-class feature (per-product incoming units,
bucketed by days-to-maturity). This is the single most informative feature in the game and it is
free to compute.

> ### ✅ Confirmed, ❌ mechanism wrong — measured in `docs/experiments.md` E7, E10, E11
>
> **Confirmed:** the market is where the game is decided, and the opponent's public tiles are the
> most informative free feature. Both hold.
>
> **Wrong:** this section assumes the edge is in *sale timing* — holding inventory as a "private
> option", front-running a predicted wave, staggering to smear a supply curve. Measured, all of
> that loses. A policy that **structurally cannot hold inventory** (every reservation price pinned
> to zero) beats the fully tuned forecast-and-reserve policy **84.4%** [73.6, 91.3].
>
> A reservation price is a bet that prices recover before the season ends, and in a two-player
> market the opponent decides whether they do. Melon absorbs only ~$26.5k before flooring, so the
> units sold *first* take the good prices; waiting trades a certain gain for a bet against an
> adversary who profits from your patience.
>
> **The real mechanism is production timing.** The race is won days earlier, by planting and
> harvest scheduling that lands your units on the curve before theirs. The supply forecast built
> here is valuable — as an input to *planting*, not to selling.

---

## 2.6 Lessons from the Orbit Wars top-10 writeups

Read in full: [1st](https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin),
[2nd](https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars),
[3rd](https://www.kaggle.com/competitions/orbit-wars/writeups/3rd-place-ab-in-den-orbit),
[5th](https://www.kaggle.com/competitions/orbit-wars/writeups/orbit-wars-5th-place-solution),
[6th](https://www.kaggle.com/competitions/orbit-wars/writeups/6th-rl-league-search-with-a-custom-edge-atte),
[7th](https://www.kaggle.com/competitions/orbit-wars/writeups/7th-place-solution-how-structured-experiments-sa).
Same organizer, same submission constraints, same competition series — this is the closest available
reference class.

| | 1st | 2nd | 3rd | 5th | 6th | 7th |
|---|---|---|---|---|---|---|
| Params | 200M | **4.3M** | **6.2M** | ~small | **2.5M** | **9M** |
| Algo | PPO | PPO (PufferLib) | PPO + PFSP | IMPALA/V-trace | async PPO | PPO |
| Env rewrite | Rust | Rust→C | JAX | C++ | C++ | JAX |
| Init | scratch | scratch (IL earlier) | scratch | **behavior cloning** | small-model transfer | scratch |
| Reward | terminal ±1 | terminal (+0.5 slow win) | terminal ±1 | terminal, no shaping | terminal | terminal, no shaping |
| Steps | 15B | 10B | 8.4B | — | — | 2.2B |
| Hardware | 4×8×B200 | 8×H100 | **2 GPUs (3090→5090)** | **1× RTX 5090, 1 week** | **3090 + 1×A100** | **2 GPUs** |

### The universal consensus (6 of 6 did this)

1. **Everyone rewrote the environment.** Rust, C, C++, or JAX — no exceptions. The Python env was
   "far too slow for RL." They ran at 15k–40k steps/s; ours runs at **860**. Three of them
   explicitly parity-tested against the reference implementation. **This is the single hardest
   prerequisite and it is now Phase 0.**
2. **Transformer over entity tokens**, one token per game object plus summary/value tokens. No
   convolutions over a rasterized board, no flat feature vectors.
3. **Terminal reward only. Reward shaping actively hurt.** 7th ablated it: "Dominance reward
   shaping" scored **34.6%** against its own baseline. **This contradicts my earlier
   dense-`Δmoney`-shaping recommendation — revised below.**
4. **Semantic action spaces beat raw ones, decisively.** 1st started with raw launch angles, got "a
   far cry from competitive," and switched to target selection. 3rd switched from fixed ship
   fractions to four *intents* (send-all / sortie / hold / kill-at-arrival) and it "increased
   learning speed by a lot." 2nd went furthest: the two actions that mattered were **no-op and
   all-in at a short-ETA target**, and that was the whole action space of a 2nd-place model.
5. **Never train against only a live copy of yourself.** 7th measured it: training vs. a live copy
   = **20.7%** winrate vs. baseline. Historical opponent pools rescued 7th's 4p model and 6th's;
   3rd used PFSP; 5th used a frozen pool plus a *delayed moving teacher* with KL regularization;
   1st lists omitting league play as his main regret.
6. **A local arena is the only evaluation that exists.** "You can't do CV in a simulation
   competition, but you can build a local arena" (2nd, who used OpenSkill ratings to mimic Kaggle
   matchmaking). 7th ran ~200 experiments, each a fixed 100M-step budget judged by round-robin
   (512 games/matchup).

### The most important number for us

**Model size.** 2.5M / 4.3M / 6.2M / 9M parameters took 6th, 2nd, 3rd, and 7th place. Only 1st place
was large, and he had a free B200 cluster. 3rd, 5th, 6th, and 7th all trained on **one or two
consumer GPUs**. The 1st-place author, asked directly whether someone with spare time and a 4090
could compete, said: "For sure! A number of top-10 competitors did so."

**This competition is winnable on a single GPU with a ~5M-parameter model.** Design for that.

### Techniques with measured evidence behind them

- **Aux future-prediction losses** (7th): extra heads predicting game state 2/8/32/64 turns ahead,
  discarded at inference. "Helped quite a bit" — forces the trunk to build an internal world model.
- **Early termination of decided games** (6th, 7th): end the episode once the outcome is settled
  instead of training on dead frames. 6th: "big boost." 1st's single biggest regret was that
  `gamma=1.0` made his agent stall, wasting compute on decided games.
- **Relative-only features** (6th): no absolute coordinates, no absolute player ids — "a massive
  gain," removes per-seat behavior and needs no augmentation.
- **Entropy annealing** (3rd): "by far the most important knob during training."
- **Pairwise/edge features as attention biases**, graphormer-style (3rd, 6th, 7th) — plain attention
  sees two tokens but not the *relation* between them.
- **Let the model prune its own action space** (7th): only the top-4 targets by the model's own
  logits plus 4 random ones are sampleable. Cut 48 candidate targets to 8 with no measurable loss.
- **Gaussian histogram value loss**, 51 bins (6th): "a lot more stable than MSE."
- **Inference-time search** (6th): greedy 2-step rollout in 2p, worth +30–40 leaderboard points.
- **Bootstrap small, then transfer** (6th): train a fast 2-layer model with dense capture reward,
  then distill into the real model via teacher-KL. "Big speedup." This is the sane way to use
  reward shaping given point 3 above.

### Where the field disagrees — and what we should pick

**Behavior cloning.** 5th place built their entire solution on it ("avoid learning the game from
scratch… pure self-play spends a large amount of compute discovering basic behavior") and reached
5th on one RTX 5090 in a week. 2nd used IL to reach top-10, then found from-scratch RL beat it in
the final days. 1st refused it on principle. **Our situation resolves this: Kaggriculture is new, so
there are no strong-player replays to clone from — but our scripted engine is an expert we can
generate unlimited data from.** That makes BC cheap for us and it stays in the plan, as the
compute-poor side of the trade.

**A caution from 2nd and 7th:** both found their from-scratch or pool-diversified runs beat their
carefully-initialized ones. Treat BC as a warm start to be *outgrown*, not a destination.

### Process lessons (these mattered as much as the modeling)

- 7th ran every change in an isolated git worktree, trained it for a fixed 100M steps, then judged
  it by tournament. A bundle of "small, sensible-looking network fixes" scored **39%**. Their
  conclusion: *always trust the tournament, never your intuition about soundness.*
- 3rd kept a `zoo/` of snapshotted pipeline versions so old checkpoints could be evaluated with
  their original (buggy) feature code, and cloud-checkpointed everything because rented GPUs crash.
- 3rd: "Self-play RL is brutally noisy… you can realistically only test whether something speeds up
  learning, not whether it changes the final ceiling."
- Coding agents were used heavily by everyone (7th: "every line of code written by Claude"), with a
  consistent warning: it becomes trivial to add complexity faster than you can understand it. 3rd
  found a crippling bug 36 hours before the deadline.

### Submission constraints

1 s/turn + 60 s overage on a slow, unpredictable CPU, and a **100 MiB file cap** in Orbit Wars.
1st needed int8 quantization for speed and 4-bit NormalFloat to fit, plus a fallback to a 5M model
when overage ran out. At ~5M params none of this binds. **Confirm Kaggriculture's file cap and
measure real per-turn inference cost before sizing anything.**

---

## 3. Build plan

Phases 1–2 are scaffolding for Phase 3, not a detour around it. Phase 3 is the deliverable.

### Phase 0 — Fast simulator, arena, ground truth

**The fast simulator is the hardest prerequisite and the thing most likely to sink the project if
deferred.** All six top-10 writeups rewrote the environment; none succeeded without it. Ours runs
at 860 steps/s and they needed 15k–40k.

- **Rewrite the env.** Kaggriculture is much simpler to simulate than Orbit Wars — a 10×10 grid, no
  continuous-space collision detection, no orbital physics, no angle solving. Options by
  effort/payoff: batched NumPy over many farms → Numba → Rust+PyO3 / C++ +pybind11. Only our own
  farm, the opponent's public tiles, and the market need simulating.
- **Parity-test it** against the reference env over thousands of seeds with random and scripted
  actions, asserting full state equality each step. 1st, 5th, and 7th all did this; 5th put it
  best — "a faster environment is only useful if it produces exactly the same game transitions."
- **Local arena** (`arena/`): round-robin between agent checkpoints, N games per matchup with fixed
  seed lists, OpenSkill or Elo ratings. This is the *only* evaluation signal that exists in a
  simulation competition. Build it before the first model.
- **Experiment discipline from day one** (this is what produced 7th place): one change per isolated
  branch, a fixed step budget per experiment, accept/reject strictly by tournament winrate. Snapshot
  the whole feature pipeline per checkpoint so old models stay evaluable.
- `sim/` runner over the reference env: N episodes in parallel, fixed seeds, money + diagnostics.
  8 workers ≈ 10 episodes/s — enough for CEM and parity testing, not for PPO.
- **Confirm the submission file-size cap** and measure per-turn inference cost against the 1 s +
  60 s overage budget. At ~5M params this should not bind, but find the numbers first.
- **A analytic tile-planner sanity checker**: given a crop and a watering schedule, predict final
  `yield_units`, then assert against the real env. Catches every off-by-one in the yield windows.
- Validate the melon rush, alternate-feeding, and CARE-vs-alternate-feed questions empirically.
- Instrumentation: per-day money, actions used vs. wasted (no-ops!), tiles idle, shed overflow
  discarded, units sold per product, realized price per product. **Wasted actions and discarded
  overflow are the two metrics that will drive every improvement.**

### Phase 1 — Scripted economic engine (most of the score)

Structure it as a day-planner + executor, not a per-turn reflex agent:

1. **Task generator** — at each day boundary, emit the day's task list: water each plant that needs
   it *today* (respecting the every-other-day survival rule and the bonus window), feed/harvest/
   collect-fertilizer per animal, plant empty tiles per the crop-mix policy, dig weeds.
2. **Assignment & routing** — assign tasks to (farmer + hands) minimizing travel. Units respawn at
   the shed center daily, so this is a capacitated multi-vehicle routing problem on a 10×10 grid
   with a 24-action budget per unit. Greedy nearest-task with cluster pre-partitioning is fine to
   start; the tiles never move, so per-day plans can be cached.
3. **Hire policy** — hire until marginal tasks run out (cost is negligible; the real limit is that
   extra hands spend turns walking).
4. **Market module** — removed, then partially restored. Reservation pricing as *market timing*
   loses: a policy that structurally cannot hold inventory beat the fully tuned one **84.4%**
   (E11), and the town drains fast enough that prices rise rather than crash (E16). But a reserve
   as a **floor against our own glut** does pay on the one or two products our herd can flood —
   currently wool, since 8 sheep per side outpace its ~14/day drain (E19). The champion carries
   `0.35 x base` there and sells everything else on sight.
   *Which* products qualify depends on our own mix, so it is re-measured, never carried forward.
5. **Crop mix & land policy** — **[REVISED E15]** rank products by *seasonal* capacity (shop count),
   not by price or by one-shot depth. The tuned champion is melon + carrot + **7 cows**, no geese,
   no land. Melon still earns, but it is a finite pot to be raced for, not a line to build on.

Target: five figures. Anything under $20k means something in the above is broken.

### Phase 2 — Black-box parameter search

The engine will have ~20–30 knobs (melon quota, goose count target, wheat-per-goose ratio, per-product
land-buy day thresholds, hire cap, harvest-timing offsets; reserve multipliers were searched and
then **pinned to zero**, E11). Optimize with
CEM or CMA-ES against a fixed seed set, opponent = the current best script.

**This is the highest ROI per hour on the list.** A 25-dim search is orders of magnitude more
sample-efficient than a 500k-parameter policy gradient, and the env is fast enough to run thousands
of evaluations.

### Phase 3 — The model (the deliverable)

> **⚠ Re-assessment, current.** The design below is sound and stays. What is *not* settled is
> whether to build it, and that question is now the single most important one in this project.
> Read §3.5 before starting any of it.

**Target: a ~2–10M parameter transformer over entity tokens, trained on one GPU.** That is the size
that took 2nd, 3rd, 6th, and 7th place (§2.6). Do not design for anything larger.

**Architecture:** one token per farm tile (100 max, both farms), plus product tokens (9), plus
player and global summary tokens. Pairwise features (tile↔unit distance, unit↔task feasibility)
injected as graphormer-style attention biases. Relative features only — no absolute player ids.
Value head over the player token.

**Action space — semantic, not raw.** The clearest lesson of §2.6: 1st failed with raw angles and
succeeded with target selection; 3rd's four intents beat fixed fractions; 2nd reached 2nd place with
*only* no-op and all-in. The analog here is **per unit, pick (target tile, intent)** — where intent
is a small semantic set (`tend` = water/feed/care as appropriate, `harvest`, `plant <crop>`,
`build`, `clear`) and the executor computes the path and the exact op sequence. **Never make the
model emit `NORTH`.** Market actions are a separate head over per-product sell intents.

Find the Kaggriculture equivalent of "no-op and all-in" early — the two or three decisions that
carry most of the value — and consider making that the entire action space first.

**Decision cadence.** Farm intents can be re-selected per turn (units are idle otherwise), but the
*strategic* decisions — crop mix, goose count, land purchase — are daily. A melon planted today
pays out 240 turns later, so use a high gamma (3rd used 0.993–1.0; 1st used 1.0) and let the
day-level structure come from the observation rather than from a coarsened action space.

**Recipe, in order:**

1. **Behavior cloning** from the Phase-2 script over thousands of seeds. Warm-starts past the
   exploration dead zone; the compute-poor side of the trade (§2.6). Treat it as a start to be
   outgrown — both 2nd and 7th found from-scratch runs eventually beat their initialized ones.
2. **PPO fine-tune** with GAE, KL-targeted adaptive LR, and a real **entropy annealing schedule**
   (3rd: "by far the most important knob"). Single PPO epoch per rollout — 3rd found more epochs
   "mostly bought instability."
3. **Terminal reward, not shaping.** *Revised from my earlier recommendation:* 7th measured
   dominance shaping at **34.6%** against its own baseline. Use the game outcome (money margin vs.
   opponent). If a dense signal is needed to bootstrap, follow 6th's pattern — train a small fast
   model with dense reward, then transfer via teacher-KL — rather than shaping the real run.
4. **Opponent pool from day one**, never a live copy (7th measured that at **20.7%**). Mix: the
   tuned script, checkpoints sampled across the whole run history, and optionally 5th's delayed
   moving teacher with KL regularization for stability.
5. **Early-terminate decided games** — end the episode when the outcome is settled rather than
   simulating dead days (6th, 7th: "big boost"). In our game a large enough bank lead plus a
   production lead late in the season is effectively decided.
6. **Aux future-prediction heads** (7th): predict market prices, own money, and tile states at
   +1/+3/+7 days. Targets are free from the rollout, heads are discarded at inference.
7. **Split heads by timescale.** The farm plan is daily; the **market policy wants per-turn
   resolution**, since sell timing inside a day matters when the opponent is dumping.

**Observation — the important part.** Market inventory and prices, day/hour, own shed, and **the
opponent's incoming supply forecast derived from their public tiles** (per product, units bucketed
by days-to-maturity). See §2.5.

*Revised by E11:* read that forecast as a **race position**, not a price prediction. It answers
"will their melons hit the curve before mine?", which is a planting and harvest-scheduling
question. There is no sale-timing policy left to learn.

**Ported from the Orbit Wars template** (`reference/orbit_war/OVERVIEW.md`): variable-row batching,
the shared-weight permutation-equivariant candidate encoder, `-inf` action masking with the
all-masked-row rescue, side alternation. Crop/tile allocation is a direct reuse of the
candidate-scoring pattern — candidates = `(empty tile, crop)` pairs, mask illegal, categorical
over slots.

**Never learn:** movement, watering, feeding, harvest timing, routing. All computable; hand-code
them in the executor and spend the samples on decisions that are actually strategic.

## 3.5 Where the project actually stands — and the one thing blocking it

*Written after E20. This supersedes the sequencing in §3, not its contents.*

### The scripted agent has hit the ceiling of its own parameter space

Not "returns are diminishing" — a structural ceiling, measured:

- **The farm is physically full.** 25 of 25 unlocked tiles occupied: 14 animals, 10 crops,
  1 structure, **zero empty**. The herd sits exactly at its target.
- **More land does not help.** Four framings, all crushed, all intervals clear (E20). Land has now
  been rejected under four different strategies across four independent tests. It is settled.
- **18% of unit-turns are already idle.** The agent has labour it cannot spend.

So the three obvious growth levers — more tiles, more animals, more hands — are each either
exhausted or measured negative. What remains inside the parameter space is re-weighting 33 knobs
that CEM has already searched hard, gated by a promotion process that now refuses most candidates.
That is the definition of a local optimum, and the gate exists precisely to stop us mistaking noise
for progress there.

### But we do not know whether the ceiling is high or low

Per **D16**, every number in this repo was produced against agents I wrote. 74 of them. The
champion beating all 74 tells us it is the best of *my* ideas — it tells us nothing about whether
my ideas are any good. The failure mode is not subtle: I ranked markets by a one-shot price
integral, was wrong by 2.4×, and **nine consecutive experiments confirmed the error** because they
all shared the assumption. A self-referential arena reproduces its author's blind spots at every
sample size.

### That makes V1 the gating input, not a checkbox

`docs/decisions.md` **D1** records the trigger for shrinking or dropping Phase 3: *if the game is
close to a solved optimisation, the model's remaining headroom is only the market game.* Evidence
since D1 has pushed hard in that direction — E11/E16/E19 collapsed sale timing to near-trivial, and
E17 showed a plain nearest-task rule captures most of the routing gain. On local evidence the
honest read is that **there may be very little left for a model to learn.**

But that read is exactly the kind of conclusion this project has gotten wrong before, and it cannot
be checked from the inside. One leaderboard placement resolves it:

| what V1 shows | what it means | what to do |
|---|---|---:|
| **near the top** | my model of the game is roughly right; the ceiling is high | Phase 3 as designed, or shrink it to the heads with headroom |
| **mid-field** | the script is fine but something structural is missing | find *what* the field does differently before training anything |
| **far down** | a core assumption is wrong, as the melon error was | **do not train.** A model trained on a wrong game learns the wrong game |

In the third case, building the model would be the single most expensive mistake available to us:
weeks of GPU time spent optimising against a misunderstanding, with the same self-referential arena
reporting success the whole way. That risk is why V1 comes first.

### So: the plan does not need new *ideas*. It needs an external measurement.

**V1 is the next step, and it is blocked on you** — the competition has to be joined from a browser
under your own account, and I will not submit on your behalf unattended. Everything else worth
doing is small by comparison, and is listed in `TASKS.md`.

### Phase 4 — Inference-time search

Validated by 6th place, who got **+30–40 leaderboard points** from a greedy 2-step rollout search in
the 2-player mode: sample ~5 of your own actions, evaluate each against the opponent's argmax, step
once more, pick the best. Their note on *why* it worked is the useful part — it mostly avoided
short-sighted mistakes (committing to a target the opponent can trivially take back), which is
exactly the failure mode our market game has (dumping into a price the opponent is about to crash).

The Phase-0 fast simulator is what makes this possible. Budget: 1 s/turn plus 60 s overage, so
amortize heavy search onto the 30 day-boundary turns and execute cheaply the rest of each day. Score
leaves with the Phase-3 value head, and drive the opponent's assumed sell behavior from their public
supply forecast (§2.5).

6th also tried CFR over a one-step lookahead: theoretically better, too noisy in practice, and too
sample-hungry for the Kaggle VMs. Start with the greedy version.

---

## 4. What ports from the Orbit Wars reference

**Keep:**
- The module split: `game_types → features → policy → ppo → env → train`.
- Variable-row batching (owned planets → owned tiles/units); concatenate rows across parallel envs
  into one forward pass.
- The shared-weight, permutation-equivariant candidate encoder — a pointer-net-style scoring head.
- `-inf` masking of illegal candidates, with the all-masked-row rescue (`ppo.py:51`).
- Side alternation across episodes.

**Rewrite:**
- Terminal-only reward → dense `Δmoney` shaping plus terminal margin.
- The shared per-turn return → per-decision credit; worse here than in Orbit Wars because there are
  many more units per turn.
- Missing GAE → add it, plus value clipping and LR annealing.
- Self-play as primary driver → scripted opponent + an opponent *pool*, not a single lagging copy.

**Problems Orbit Wars didn't have:**
- **Simultaneous unit conflicts.** Two units requesting `PLANT WHEAT` with one seed → *neither*
  plants (`:905`). Needs a post-sampling seed-budget resolver or an autoregressive head.
- **The market head** is a variable-length ordered list and does not fit per-unit decomposition at
  all. Separate module.
- **Multi-day payoff horizons**, addressed by the day-granularity decision above.

---

## 5. Open questions to settle in Phase 0

1. Melon rush vs. wheat/carrot bootstrap — which funds land faster, and what happens when both
   players rush?
2. `CARE` (+1 egg/day for 1 action/day, requires daily feeding) vs. alternate-day feeding (halves
   feed actions and wheat cost, destroys the care bank). Which wins per action?
3. Optimal geese count given the action budget and the wheat needed to feed them. Is it worth
   growing feed wheat, or buying it (price rises on the `sqrt`/0.8 below-curve when you drain it)?
4. Does the 100-item shed cap actually bind at scale (~50 geese = 100 items/day), and does
   continuous intraday selling fully relieve it?
5. Do melon prices recover enough between harvest waves (8/day drain post-day-20) to justify
   staggered melon plantings over one big wave?
6. How much of a unit's 24-turn day is lost to travel, and does quadrant-clustered task assignment
   meaningfully beat greedy nearest?
7. **How exploitable is a static sell rule?** Build a deliberate front-runner (reads opponent tiles,
   dumps melon one day before their harvest) and measure how much it steals from the scripted
   engine. That number is the budget available to the market model, and it decides how much of
   Phase 3 to spend there.
8. Does harvesting early into the private shed (converting public commitment into private
   optionality) beat harvesting at max yield when the opponent is watching?

---

## 6. Repo layout and commands

```
main.py               # submission entry point (must define `agent`)
docs/                 # competition's own rules + getting-started (accurate; trust them)
reference/orbit_war/  # past-competition PPO template + OVERVIEW.md
PLAN.md               # this file
```

Proposed additions: `agent/` (scripted engine), `sim/` (parallel runner + metrics),
`search/` (CEM/CMA-ES), `rl/` (only if Phase 3 happens).

**Use the miniconda interpreter — `/usr/bin/python3` does not have `kaggle_environments`:**

```bash
/opt/miniconda3/bin/python -c "
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 7}, debug=True)
env.run(['main.py', 'starter'])
print([(i, s['reward'], s['status']) for i, s in enumerate(env.steps[-1])])
"
```

Built-in opponents by name: `"pass"`, `"random"`, `"starter"`. Benchmarks to beat:
`starter` = **$3,496**, `random` = **$0**.
