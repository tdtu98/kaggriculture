# Learning log — the BC → RL line

Companion to `PLAN_BC.md`. One entry per phase, **written in your own words** — not a checkbox, not
a link dump (PLAN_BC.md §0.2). A phase's learning exit passes when its entry here would let you
re-explain the idea to someone else without the paper open. Learning exits are independently
killable from competitive exits: a phase whose arena number failed but whose entry here is real was
not wasted work.

Convention: date each entry. If a later phase changes your understanding of an earlier entry, append
a correction rather than editing history — watching your own model of RL change is part of the point.

---

## P0 — Acquisition and the two gates

**Prompt.** Why is BC "supervised learning with a hostile test distribution"? Why does no-op
dominance make a 97%-accurate always-idle model the *default* failure, not an edge case?

**Reading.** Pomerleau, *ALVINN* (NeurIPS 1988) — 5 pages; its "steer back from the edge"
augmentation hack is exactly the problem you are about to hit. Levine, CS285, *Supervised Learning
of Behaviors* (Lecture 2; numbering drifts by year) — the drift diagram. Lin et al., *Focal Loss*
(2017), §3 only — the no-op-dominance fix.

**Entry.** *(unwritten)*

---

## P1 — Decode and extract: the data contract

**Prompt.** What is a *label* here, exactly? Why does an off-by-one in the (obs, action) pair still
train, still converge, and still produce a plausible accuracy curve — while cloning "the action
taken one step ago"? Why is a mask that rejects the expert's action indistinguishable from a hard
example?

**Reading.** The reference `Orbit-Wars/src/train_bc.py` loss-skip filter (lines 143–188, 251–268) —
write down *why* you are not enabling it yet.

**Entry.** *(unwritten)*

---

## P2 — Plumbing model (v0)

**Prompt.** Why is "97% accuracy" under no-op dominance worthless — computed on your own data, not
someone else's example? State the difference between offline agreement and online return *before*
you have any reason to believe it.

**Entry.** *(unwritten)*

---

## P3 — BC proper (v1)

**Prompt.** State the **O(T²ε)** compounding-error bound and why T = 719 makes a 1% per-step error
not a 1% problem. Explain why per-timestep i.i.d. BC and teacher forcing are the same construction,
and why the drift is exposure bias by another name.

**Reading.** Ross & Bagnell, *Efficient Reductions for Imitation Learning* (AISTATS 2010) — the T²ε
bound; the single most important paper in this plan. Bengio et al., *Scheduled Sampling* (2015).
Vinyals et al., *Grandmaster level in StarCraft II* (Nature 2019), supervised-from-replays section.

**Entry.** *(unwritten)*

---

## P4 — Close the distribution-shift gap

**Prompt.** Why is DAgger O(T) where BC is O(T²), and what does "no-regret online learning" buy?
Note: the handover curve (E91) measures compounding error with no oracle at all — this entry is
writable even if the DAgger phase itself is skipped.

**Reading.** Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction
to No-Regret Online Learning* (AISTATS 2011) — DAgger. Rajeswaran et al., *DAPG* (RSS 2018) — the
concrete recipe for the KL-to-BC schedule used in P5. Ho & Ermon, *GAIL* (2016) — for what a
queryable expert lets you avoid.

**Entry.** *(unwritten)*

---

## P5 — PPO fine-tune (v2)

**Prompt.** The full chain in your own words: REINFORCE → baseline → actor-critic → GAE → **the PPO
clip**. And: why did 6 of 6 Orbit Wars top-10 writeups find reward shaping hurt?

**Reading.** Schulman et al., *GAE* (2015) then *PPO* (2017), in that order. Huang et al., *The 37
Implementation Details of PPO* (ICLR Blog Track 2022) — a checklist, not a read. Berner et al.,
*Dota 2 with Large Scale Deep RL* (2019), §self-play, paired with the AlphaStar league (Nature
2019).

**Entry.** *(unwritten)*

---

## P6 — Scale-out: GPU + JAX *(contingent)*

**Prompt.** Why did the six constraints in PLAN_BC.md §2.7 make this a port rather than a rewrite?

**Entry.** *(unwritten)*

---

## Corrections and revisions

*(Append here when a later phase changes your understanding of an earlier entry.)*
