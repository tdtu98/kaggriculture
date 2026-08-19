# Two-Week Behavior Cloning Cooperation Plan

## Purpose

Two people will spend two weeks exploring behavior cloning (BC) for Kaggriculture. Both are new to
reinforcement learning but comfortable with Python, PyTorch, and the repository tooling.

The goal is not competition-level performance. At the end of the two weeks, both people should be
able to explain and reproduce a small end-to-end BC workflow and make an evidence-based decision
about what to study next.

Expected effort is about 1–2 hours per person on an available day. The plan does not assume that
every day will be occupied.

## Working principles

1. **Shared core, independent attempts.** Both people perform every core step rather than dividing
   permanently into data and model specialists.
2. **Compare and teach.** After independent attempts, each person explains their approach,
   confusion, and result to the other.
3. **One shared baseline.** The pair integrates the best-understood result, not automatically the
   result with the highest score.
4. **Optional self-exploration.** Personal experiments are encouraged after the shared core work.
   They never block shared progress.
5. **Checkpoints, not rigid deadlines.** The dates are suggested windows. A phase moves when both
   people understand and can reproduce its output.
6. **Reduce scope before increasing pressure.** If a task is too large, simplify the task while
   preserving the learning objective.

## Session pattern

A shared session may use this 60–90 minute shape:

| Time | Activity |
|---:|---|
| 10 minutes | Agree on one question and the smallest useful outcome. |
| 30 minutes | Attempt the same core task independently. |
| 20 minutes | Compare approaches and teach the differences. |
| 15 minutes | Integrate one shared result and record findings. |
| Optional | Continue with separate personal exploration. |

When working asynchronously, each person leaves a short note:

- What I tried
- What happened
- What confused me
- What I would test next

The **Integrator** combines the agreed result. The **Challenger** reproduces it, questions its
assumptions, and checks whether both people understand it. These roles rotate by phase; they do not
replace the shared core work.

## Two-week flow

| Suggested window | Shared step for both people | Shared output | Optional exploration |
|---|---|---|---|
| Days 1–2 | Learn the basic BC idea and run the existing expert agent. | A shared understanding note connecting expert demonstrations, observations, actions, training, and evaluation. | Explore unfamiliar game or machine-learning concepts. |
| Days 3–4 | Inspect examples of expert behavior and align on what a demonstration contains. | A short expert-behavior note and an agreed first experiment question. | Inspect different episodes, tools, or representations. |
| Days 5–7 | Independently build the smallest end-to-end BC baseline, compare attempts, and agree on one shared version. | A reproducible minimal baseline with run instructions and recorded result. | Try one alternative only if the shared baseline already runs. |
| Days 8–9 | Evaluate the shared baseline and identify its most important failure or uncertainty. | A failure review that separates observed evidence from guesses. | Each person may investigate a different explanation. |
| Days 10–11 | Independently attempt one agreed improvement, then teach and compare the results. | An improvement comparison and a decision to keep, reject, or revisit it. | Keep unrelated personal experiments separate from the baseline. |
| Days 12–13 | Integrate the best-understood version, reproduce the workflow, and finish incomplete shared work. | A reproducible shared workflow and final evaluation record. | Use remaining time for deeper exploration. These days also serve as buffers. |
| Day 14 | Exchange teach-backs, summarize lessons, and choose the next learning direction. | A final handoff containing lessons, open questions, and a recommendation for the next two weeks. | Record future ideas without requiring implementation. |

## Phase checkpoints

### 1. Shared orientation

Move forward when both people can explain, in their own words:

- what behavior cloning learns from an expert;
- how BC differs from the genetic search that produced the scripted champion;
- why training accuracy alone does not prove that an agent plays well.

### 2. Demonstration understanding

Move forward when both people can follow one example from expert observation to recorded decision
and explain what information would be available at inference time.

### 3. Minimal baseline

Move forward when each person has attempted the pipeline and both can reproduce the agreed shared
baseline from written instructions. Modest performance is acceptable.

### 4. Failure review

Move forward when the pair has selected one important failure or uncertainty using observed
results, rather than choosing improvements only because they sound promising.

### 5. Shared improvement

Move forward when both attempts have been compared under the same evaluation conditions and the
keep/reject decision has been recorded.

### 6. Handoff

Finish when both people can independently explain and rerun the workflow, including its known
limitations and the reason for the recommended next step.

## Shared artifacts

Keep the artifacts small and understandable:

1. Shared understanding note
2. Expert-behavior note
3. Minimal baseline and run instructions
4. Experiment log
5. Failure review
6. Improvement comparison
7. Final handoff and next-step recommendation

Every experiment-log entry should state:

- Question
- Change
- Evaluation conditions
- Result
- Interpretation
- Next decision

## Flexibility rules

- A missed day consumes buffer time; it does not create mandatory catch-up work.
- A phase may expand by one or two days if both people are learning from it.
- If availability drops, preserve the shared core and remove optional exploration first.
- If the minimal baseline takes longer than expected, use Days 8–11 to finish and analyze it rather
  than rushing into an improvement.
- If the baseline is ready early, add one exploration question instead of expanding the project
  into full reinforcement learning.
- Personal work can join the shared baseline only after the other person can reproduce or explain
  it.

## Success criteria

The two weeks are successful when:

- both people can explain the complete BC workflow;
- both have personally attempted every core phase;
- one small baseline can be reproduced from written instructions;
- at least one failure has been investigated with evidence;
- the pair has compared one shared improvement or clearly documented why improvement work was
  deferred;
- the final handoff identifies a justified next learning step.

Competition strength, PPO, a complete learned Kaggriculture agent, and large-scale tuning are
outside this two-week scope.
