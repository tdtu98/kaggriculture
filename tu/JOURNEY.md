# Our Journey: Building a Behavior-Cloning Model for Kaggriculture

**This is the plan we follow.** It is written for a learner — simple words, one new idea per
chapter, and something that runs at the end of every chapter. The heavy technical details live in
`PLAN_BC.md`; we open that only when a chapter tells us to.

**The one-sentence goal:** teach a neural network to play Kaggriculture by showing it thousands of
moves made by top players — then, later, make it even better by letting it practice against itself.

**Our guide:** the `Orbit-Wars/` folder. Someone already walked this exact road for a different
Kaggle game (spaceships instead of farms). They trained a model on top-player replays
(`src/train_bc.py`), then improved it with self-play (`src/orbit_ppo.py`). We will do the same
thing, step by step, understanding each piece before we build our version.

---

## The map (read this first)

| Chapter | We build | We learn |
|---|---|---|
| 1 | Nothing — we read and poke | What a policy, state, and action are |
| 2 | Nothing — we check the data we were given | Where training data comes from, and the clock trap |
| 3 | A data decoder | Behavior cloning = supervised learning |
| 4 | A tiny model (one decision) | Baselines, and the "always do nothing" trap |
| 5 | The real model | Action heads, masks, the Orbit-Wars design |
| 6 | An agent that plays full games | Why test accuracy lies (distribution shift) |
| 7 | A better agent | DAgger — asking a teacher when we get lost |
| 8 | A self-improving agent | Reinforcement learning: rewards, PPO, self-play |
| 9 | (optional) A faster training setup | Scaling up: GPU and JAX |

Rule of the journey: **we never move to the next chapter until the current one's small test
passes.** Every chapter ends with "How we know we're done."

---

## Chapter 1 — Meet the game and the data (no code yet)

**The idea.** In RL language, everything is described with three words:

- **State** — what the player can see right now: the farm tiles, the market prices, the money,
  the day and hour. In our data this is called the *observation*.
- **Action** — what the player does this turn: each worker moves or plants or waters; maybe some
  market orders like "sell 5 wheat".
- **Policy** — the rule that turns a state into an action. A policy can be code (like the scripted
  agents in this repo) or a neural network (what we're building). "Training a model" = "learning a
  policy".

**Behavior cloning (BC) in one sentence:** watch an expert, record (state, action) pairs, and train
a network with ordinary supervised learning to predict the expert's action from the state. That's
it. No rewards, no exploring — just "copy the master".

**What we do.**
1. Open `data/kaggriculture/95029942.json` (a real game between two strong players, ~$105k vs
   ~$91k — the baseline bot only makes $3.5k; the same game is also game 1 of our training set, at
   `data/sample_data_training_model/train/95029942.json`). Look at one step: find the observation, find the
   action. See that the action has a part for the farmer, a part for each hired hand, and a part
   for market orders.
2. Skim `Orbit-Wars/README.md`. Notice their recipe: **replays → BC → PPO self-play**. That
   README is our map in miniature.

**How we know we're done.** You can point at one step of the JSON and say: "this is the state,
this is the action, and the policy is the hidden thing that connected them."

---

## Chapter 2 — The data (we already have it)

**Good news: the training data is already here.** It sits in
`data/sample_data_training_model/` — 2.9 GB, **100 real Kaggle games**, already split into
`train/` (70), `val/` (15) and `test/` (15), plus a `manifest.csv` describing every game.

**What it is, in one line: 100 games that one strong player — "Ryo Hasegawa" — won.**

- Ryo is in **all 100 games and won all 100**. Earnings ranged from $48,467 to $165,959, median
  $91,697. (The baseline bot makes $3,507.)
- 36 different opponents, so the games are varied. `boatlee`, the famous scripted bot in this repo,
  shows up once — as an opponent Ryo beat.
- Ryo sometimes plays as player 0 and sometimes as player 1. `manifest.csv` has a `ryo_seat` column
  telling us which, per game. **We always clone Ryo's side.**

So this is a **single-teacher** dataset: we are not learning "how good players play in general", we
are learning "how *this* player plays". That is a cleaner and easier problem, and it is the right
one to start with.

If we ever want more games, we write a downloader — see `PLAN_BC.md` Chapter 2 for how.

**⚠ The clock trap (important, and interesting — and we already tested for it).** Some top
"players" here are not reacting to the game at all: they are scripts that replay the same fixed 719
moves every time (`boatlee` does exactly this — proven). If we trained on a script, the smartest
thing our model could learn is "look at the step number, output the memorized move" — a *clock*,
not a player. It would score beautifully on paper and fall apart the moment a game went
differently.

So we checked, two ways:

1. **Hash every player's whole action sequence.** Zero duplicates anywhere in the 100 games. Nobody
   is submitting the same tape twice.
2. **Compare Ryo's games step by step, in pairs.** Result: **74%–96% of steps differ between two
   Ryo games, median 96%.** They really are reacting.

One lovely detail fell out of that: **the steps that *are* identical all sit at indices 0–31.** Ryo
plays a **fixed ~30-step opening routine** — same buy, same first plantings — and then plays
reactively for the remaining ~690 steps. That is worth knowing: those first 30 steps are almost
free to predict and carry almost no learning signal.

**Verdict: Ryo is a genuine reactive teacher. The tape-recorder risk is retired for this dataset.**

**How we know we're done.** You have opened `manifest.csv`, you can say which split a game is in,
which seat Ryo played, and who they beat — and you can explain in your own words why the clock trap
would have been invisible in an accuracy score.

---

## Chapter 3 — Turn replays into training examples

**The idea.** This is the least glamorous and most important chapter. BC is supervised learning,
and supervised learning is only as good as its labels. Our job: for every step, produce a clean
pair — *the state the player saw* and *the action they chose in it*.

**Two traps we already know about** (found by measuring, so trust them):

1. **The off-by-one.** In the JSON, the action stored at step *i* was chosen while looking at the
   observation from step *i−1*. If we pair them naively, we'd train a model to copy "the move from
   one step ago" — and the scary part is training would still work and accuracy would still look
   fine. We add an automatic check (the number of hands in the action must match the *previous*
   step's roster — it matches 1438/1438 times when paired correctly).
2. **Player 2's saved observation is missing its `step` field.** Easy fix: `step = day*24 + hour`.
   But we must do it, or player-2 data is subtly broken.

**What we do.** Write `bc/decode.py`: read a replay from `data/sample_data_training_model/`, look up
in `manifest.csv` which seat Ryo played, keep **his** side, apply the two fixes, and output clean
pairs with the checks built in as assertions (if a check fails, the program stops — we never train
on suspicious data).

**How we know we're done.** The decoder runs over all 100 provided games with **zero** assertion
failures, and prints how many training pairs we have (it should be about 50,000 states from the 70
training games).

---

## Chapter 4 — The smallest possible model

**The idea.** Before building anything fancy, we train the dumbest model that could work, on one
decision only — for example: "will this worker act this turn, or do nothing?" Logistic regression
or a 2-layer network, a few hand-picked features (hour of day, is a plant ready, etc.).

**The lesson this chapter exists for: the "always do nothing" trap.** In this game, the most
common single action is very frequent (in the old sample game, ~16–19% of decisions were just WATER
or PASS — we recompute the real number on Ryo's own games). A model can get a nice-looking accuracy
by *always predicting the most common thing*. So the rule for the rest of the journey: **an
accuracy number means nothing by itself — it only means something compared to the "always guess the
most common action" baseline.** We compute that baseline first, always.

**One extra catch from Chapter 2:** Ryo's first ~30 steps are a fixed opening they play every game.
They are trivially predictable and teach the model nothing. So we always report two numbers: over
all steps, and over steps 32 onwards. The second one is the real one.

**What we do.** Build features → train the tiny model → compare against the majority baseline.

**How we know we're done.** Our tiny model beats the majority baseline by a clear margin, and you
can explain to a friend why "97% accurate" can describe a completely useless model.

---

## Chapter 5 — The real model (our version of Orbit-Wars' network)

**The idea.** Now we build the actual BC model, copying the *shape* of Orbit-Wars' design
(`src/orbit_net.py`) but adapted to farming. Their three good ideas, in simple words:

1. **Encode the state as a set of "tokens".** They made one token per planet and per fleet. We
   make one token per farm tile (100 of them), per worker, and per market product. Each token is a
   short list of numbers ("this tile has wheat, watered today, worth $30...").
2. **Break the action into small choices ("heads").** They chose per planet: act or skip → where
   to send → how many ships. We choose per worker: act or idle → **which tile to go to** (a
   "pointer" that selects one of the 100 tile tokens) → what to do there (plant/water/harvest...)
   → with what (which crop). Then a few more heads for market orders. Small choices are much
   easier to learn than one giant choice.
3. **Mask illegal actions.** Before the model picks, we zero out everything the game would reject
   (can't water a tile with no plant, can't sell what you don't have — the game's own rules tell
   us). The model never wastes effort learning "don't do impossible things". One golden check
   comes free: **the expert's real action must never be masked out.** If it is, our mask is wrong,
   not the expert.

**One simplification we get for free:** in the replays, workers essentially always walk the
shortest path (measured: 99.5–100%). So the model doesn't learn walking — it picks the target
tile, and 10 lines of code walk there. Almost half of all decisions in the data are just walking,
and they all disappear from the learning problem.

**Model size: start small (~0.5M parameters).** Orbit-Wars' first model was this size and worked.
Big models memorize small datasets — they look great in training and play badly. We grow only when
the small model clearly runs out of capacity. Training runs on this Mac in PyTorch.

**What we do.** `bc/model.py` + `bc/train.py` (port the ideas from `Orbit-Wars/src/train_bc.py` to
PyTorch), train, and watch per-head accuracy vs. the baselines from Chapter 4.

**How we know we're done.** Every head beats its baseline on games the model never saw during
training, and the "expert action was masked out" counter reads zero.

---

## Chapter 6 — Make it play, and meet the most important idea in imitation learning

**The idea.** We wrap the model as a real agent (`agent(obs)` → action) and let it play full
720-step games against the bots in this repo.

**And here comes the big lesson: distribution shift.** Our model was trained only on situations a
*top player* got into — and, since all 100 of our games are games Ryo *won*, it has barely seen
what losing looks like. The first time it makes a small mistake, it lands in a situation the expert
never showed it — so it makes a bigger mistake, and lands somewhere stranger still. Errors
*compound*. This is THE gap between "high test accuracy" and "actually plays well", and it's why
BC alone almost never fully matches its teacher. (For the curious: the theory says a small
per-step error ε can grow like **T²·ε** over T steps — and our T is 719.)

**We won't just read about it — we'll measure it on our own model.** The *handover experiment*:
let the expert replay play the first k steps, then our model takes over. Plot final money against
k. If money collapses as k gets smaller, we are watching compounding error with our own eyes.

**How we know we're done.** The agent completes games in both seats without crashing, earns
clearly more than the $3,000 starting bank, and we have the handover plot. (Judging any agent
fairly needs **80+ games** — single games are noise in this game.)

---

## Chapter 7 — Fight the drift (DAgger)

**The idea.** DAgger, in plain words: *let the student play, and every time the student wanders
somewhere new, ask the teacher "what would you have done here?" — then add those answers to the
training data and retrain.* Now the model has labels exactly where it gets lost. This turns the
T²·ε problem into a T·ε problem.

**The catch:** the teacher must be *askable* at any situation. A replay can't answer questions —
it's a recording. Luckily this repo has a scripted expert we can call as a function (details in
`PLAN_BC.md` Chapter 7 — one specific function is a real teacher; the obvious one is not). If it turns
out no good teacher is askable, we simply skip this chapter — Chapter 8 fixes drift too, just at
a higher price.

**How we know we're done.** After 2 rounds of DAgger, average money over 80+ games goes up by a
clear margin — or we've decided to skip, knowing exactly why.

---

## Chapter 8 — Reinforcement learning: the model starts teaching itself

**The idea.** BC can at best copy the teacher. To go *beyond* the teacher, the model must learn
from results instead of examples. That's reinforcement learning:

- **Reward** — a score for how the game went. Ours is natural: the final bank balance (who won).
- **Policy gradient** — the core trick: play games, then adjust the network to make the actions
  from *winning* games more likely and the actions from *losing* games less likely. That's the
  entire idea; everything else is stabilization.
- **PPO** — the industry-standard stabilized version ("proximal" = don't change the policy too
  much in one step, or everything collapses). Orbit-Wars' `src/orbit_ppo.py` is a full working
  PPO we can read line by line and port.
- **Self-play** — the opponent is... previous versions of our own model, plus the scripted bots in
  this repo. One hard-won rule from the winners of the Orbit Wars competition: **never train
  against a live copy of yourself** (it leads to chasing your own tail); train against a *pool* of
  frozen past versions and other bots.

**Why we did BC first:** an RL model starting from random weights flails for millions of games
before discovering that planting is good. Starting from the BC model, it already plays sensibly
from game one and spends its practice improving, not discovering farming. (On a Mac, this isn't a
nicety — it's the difference between feasible and not.)

**Two more winner's lessons we adopt without debate** (all six published Orbit Wars winners
agree): use **only the final result** as reward — hand-crafted bonus rewards ("+1 for each
harvest") measurably made agents worse; and keep the fast simulator (`kagsim`, already in this
repo, ~50,000 steps/second) as the practice field.

**How we know we're done.** The PPO-trained model beats its own frozen BC starting point in at
least 60% of 80+ games — the student has surpassed the teacher.

---

## Chapter 9 (optional) — Scale up

If and only if Chapter 8 is working but too slow on the Mac: rent a GPU and port training to JAX —
which is exactly what Orbit-Wars is written in, so we'd be converting *toward* our reference code,
with a working PyTorch version to check numbers against. We write our PyTorch code in a plain
functional style from day one (the six rules are in `PLAN_BC.md` Chapter 9) precisely so this port
stays easy.

---

## The concepts you'll own by the end

By chapter: **1** state / action / policy · **2** the clock trap, and why a dataset must be checked
before it is trusted · **3** why data quality beats model cleverness ·
**4** baselines and class imbalance · **5** action heads, pointers, masking · **6** distribution
shift and compounding error — the heart of imitation learning · **7** DAgger ·
**8** reward, policy gradient, PPO, self-play · **9** what makes ML code portable.

After each chapter, write a few sentences in your own words in `docs/learning.md` (it has a page
per chapter with the question to answer and 1–3 classic papers if you want to go deeper). If you
can't write the entry, we're not done with the chapter — and that's fine, we stay.

## Ground rules (the short version of PLAN_BC.md's rulebook)

1. Never judge a change on fewer than **80 games**.
2. Every accuracy number is reported **next to its dumb baseline**.
3. Every pipeline step carries an **assertion that proves it worked** — we never trust, we check.
4. This line never touches `main.py` (the live submission) until a model wins the official
   promotion gate (`make promote`).

*Where this plan came from: distilled from `PLAN_BC.md`, which was produced and fact-checked by a
five-agent panel; all measured numbers quoted here were independently verified. When we need the
fine print — exact file layouts, kill criteria, the full evaluation ladder — we open `PLAN_BC.md`.*
