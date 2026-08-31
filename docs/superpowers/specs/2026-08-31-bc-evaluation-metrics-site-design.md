# Beyond Accuracy — BC Evaluation Metrics

## Purpose

Create a new, standalone, publicly shareable Site that explains how we should evaluate the Ryo behavior-cloning model before using game score as the main measure.

The page is written for Tu. It should make the proposal understandable without requiring imitation-learning or statistics knowledge. The existing “Can It See the Farm? — Ryo BC v0” Site is a visual reference only; this will be a separate Site and project.

## Core message

A behavior-cloning model can have high offline accuracy while still failing to copy the decisions that define Ryo’s behavior. Common scheduled actions dominate the replay data, while rare state-driven deviations can be much more important. Sequence errors also matter because one wrong action can change what should happen next.

Therefore, model selection should use three replay-only metrics in this priority order:

1. Schedule-deviation macro recall
2. Step-prefix AUC@24
3. Raw top-1 accuracy

The metrics are compared in priority order, not blended into one weighted score.

## Audience and writing style

- Primary audience: Tu and teammates working on Ryo BC v0.
- Use short sentences and familiar farm actions such as WATER, HARVEST, DROP, MOVE, and PICK UP.
- Explain the question each metric answers before showing how it is calculated.
- Lead with examples; keep formulas secondary and visually optional.
- Clearly label every toy number as an illustrative example, not a measured model result.
- Avoid unexplained terms such as “distribution shift,” “conditional,” and “macro average.” When a technical term is necessary, immediately translate it into plain language.

## Narrative structure

### 1. Hero

Title: “Beyond Accuracy — BC Evaluation Metrics”

Primary statement: “A clone can be 90% accurate and still fail to imitate the policy.”

Supporting copy explains that the first evaluation goal is to copy Ryo’s choices on held-out replays. Game performance and on-policy evaluation come later.

### 2. Why accuracy is not enough: the 90% trap

Show 100 decisions made at the same time/context:

- 90 times: WATER
- 5 times: HARVEST
- 5 times: DROP

Compare two illustrative models:

- Schedule memorizer: always predicts WATER.
- State-aware clone: usually predicts WATER but changes action when the farm state calls for HARVEST or DROP.

The schedule memorizer receives 90% raw accuracy while recognizing none of the important deviations. The example introduces the central distinction: “Did the model copy the routine?” versus “Did the model notice when Ryo broke the routine?”

### 3. Priority 1: Schedule-deviation macro recall

Plain-language question: “When Ryo does something unusual for this time and actor, does the clone notice?”

The training replays establish the usual action for each day band, hour, and actor type. Held-out decisions where Ryo chooses another action are the deviations. Recall is computed separately for each deviation action, then each action gets equal weight so a frequent exception cannot hide a missed rare exception.

Illustrative example:

- Held-out deviations contain 5 HARVEST and 5 DROP decisions.
- Model predicts all 5 HARVEST decisions correctly but none of the DROP decisions.
- HARVEST recall = 5/5 = 100%.
- DROP recall = 0/5 = 0%.
- Deviation macro recall = (100% + 0%) / 2 = 50%.

Plain conclusion: the clone notices one kind of exception but completely misses another. A single combined exception accuracy could make that weakness harder to see.

### 4. Priority 2: Step-prefix AUC@24

Plain-language question: “Can the clone stay correct across a useful run of consecutive farm steps?”

One step means one complete environment turn, not one worker row. For every possible starting step, check whether the model remains correct for 1 step, 2 steps, and so on through 24 steps. The window restarts at the next step, so one error does not permanently discard everything that follows. Twenty-four is the initial horizon because Kaggriculture has 24 environment turns in an in-game day.

Illustrative example:

Predictions across eight steps:

`✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓`

- The first window stays correct for 3 steps, then breaks.
- A new window starting after the error can earn credit for the final 4 correct steps.
- A model with the same total number of correct steps but scattered errors will score worse because it cannot maintain a reliable sequence.

The Site should visualize rolling windows moving across the sequence, emphasizing that recovery still counts.

### 5. Priority 3: Raw top-1 accuracy

Plain-language question: “Across all held-out decisions, how often is the clone’s first choice exactly Ryo’s choice?”

Illustrative example:

- The model matches 92 of 100 held-out decisions.
- Raw top-1 accuracy = 92%.

Plain conclusion: this is a useful overall sanity check, but it is lowest priority because routine actions can dominate the number.

### 6. How we choose a model

Present the selection rule as a simple podium:

1. Prefer the model with the highest schedule-deviation macro recall.
2. If the models are effectively tied, prefer the higher Step-prefix AUC@24.
3. If still tied, prefer the higher raw top-1 accuracy.

Add two supporting notes:

- Do not combine the three metrics into one weighted score; weights would hide which behavior improved or regressed.
- Calculate results per held-out game and use paired whole-game bootstrap confidence intervals before calling small differences real.

The confidence-interval note should be visually secondary and written as: “We compare whole games, not shuffled individual rows, so repeated decisions from one game do not pretend to be independent evidence.”

### 7. What this proposal does not measure yet

Current Ryo data consists of replays, and there is no callable correct Ryo policy for arbitrary new states. Therefore this first evaluation does not claim to measure:

- true on-policy cloning error after the clone changes the game state;
- correct responses on synthetic novel states;
- final game performance.

These are deferred until a playable clone and a trustworthy way to label new states exist.

### 8. Shareable summary

End with a compact card Tu can screenshot or quote:

“First test whether the clone notices Ryo’s exceptions. Then test whether it stays correct across consecutive steps. Use raw accuracy only as the final sanity check.”

## Interaction design

The page has one primary interactive comparison. A two-position control switches between “Schedule memorizer” and “State-aware clone.” It updates the illustrative action predictions and the three metric cards. This interaction should demonstrate why raw accuracy alone can favor the wrong model.

The Step-prefix section includes a lightweight step-strip demonstration. Selecting or hovering over a mismatch highlights only the rolling windows that contain that mismatch. Later windows remain eligible, showing why the metric is less strict than discarding the rest of a day after one disagreement.

All controls must work with keyboard and touch input. Motion should be subtle and respect reduced-motion preferences.

## Visual direction

Use the same editorial architecture-story language as the reference Site:

- farm green and warm paper palette;
- large editorial headline typography;
- numbered narrative sections;
- bordered cards, action chips, and diagram-like step strips;
- generous spacing and strong mobile stacking;
- no decorative stock imagery;
- no model-authored SVG illustrations.

The new Site must be recognizable as a companion to the Ryo BC v0 story while remaining a distinct Site dedicated only to evaluation metrics.

## Architecture and content boundaries

- One responsive route.
- Static, hard-coded illustrative data; no database, authentication, uploads, or external APIs.
- Keep content and example values in small data objects so the explanations and visual cards remain consistent.
- Use the retained Artifact template source rather than scaffolding over it.
- Preserve the template’s structure, components, styles, assets, package configuration, and logical bindings unless the approved content requires a focused adaptation.
- Create a new Sites project; never reuse or edit the reference Site project.

## Accessibility and responsive behavior

- Use semantic headings in narrative order.
- Meet readable color contrast for text and metric states.
- Never communicate correctness through color alone; use labels and symbols.
- Provide visible focus styles and accessible labels for interactive controls.
- Keep examples readable without interaction and stack comparisons vertically on narrow screens.

## Validation

- Confirm the site builds successfully.
- Verify the hero, 90% trap, three metric sections, winner rule, limitation note, and shareable summary are present.
- Test the model comparison control and rolling-window demonstration with mouse, keyboard, and touch-sized layouts.
- Check that every illustrative number is labeled as an example and that no example is presented as an actual model measurement.
- Preview the first coherent version in Codex, then publish the completed Site publicly for sharing with Tu.

## Out of scope

- Calculating metrics from live replay files inside the Site.
- Uploading model outputs.
- Leaderboards, user accounts, or persistent results.
- Game-score evaluation.
- Claiming that replay-only metrics solve on-policy distribution shift.
