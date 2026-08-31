# Beyond Accuracy — BC Evaluation Metrics Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publicly deploy a new one-page Site that explains the three replay-only BC metrics through easy farm examples.

**Architecture:** Copy the retained Artifact architecture-story source into a new checkout, create a new Sites project, and adapt the one-page Vinext/React application without touching the reference Site. Keep explanatory copy in typed data, isolate the two interactive demonstrations in client components, and retain the template’s editorial visual system.

**Tech Stack:** Vinext, React 19, TypeScript, CSS, Vitest, Testing Library, Node test runner, OpenAI Sites.

**Spec:** `docs/superpowers/specs/2026-08-31-bc-evaluation-metrics-site-design.md`

## Global Constraints

- Title is exactly “Beyond Accuracy — BC Evaluation Metrics”.
- Use a new standalone Sites project; do not edit or reuse `appgprj_6a8e57ccc52c81919d1c1afcf3f73b22`.
- Use the retained source at `/Users/minhduy/.codex/skills/artifact-template-architecture-story-site/assets/source`; do not scaffold a replacement application.
- Preserve the retained template’s source structure, package configuration, assets, and logical bindings.
- Use short, plain-language explanations and lead with farm examples.
- Label all toy values “Illustrative example — not measured model results.”
- Metric priority is schedule-deviation macro recall, Step-prefix AUC@24, then raw top-1 accuracy.
- Compare metrics lexicographically; do not create a weighted combined score.
- One route only; no database, authentication, uploads, or external APIs.
- Do not use decorative stock imagery or model-authored SVG illustrations.
- Keep the Site usable with keyboard, touch, narrow screens, and reduced motion.

## File Structure

Working directory: `sites/beyond-accuracy-bc-evaluation-metrics/`

- `.openai/hosting.json` — new Site identity and retained logical bindings.
- `app/metric-content.ts` — typed copy, action counts, example model scores, and selection rule.
- `app/model-switcher.tsx` — retained component adapted for the schedule-memorizer versus state-aware-clone interaction.
- `app/step-prefix-demo.tsx` — rolling sequence window explanation and recovery interaction.
- `app/page.tsx` — semantic one-page narrative assembled from the content and demonstrations.
- `app/globals.css` — retained farm/editorial theme adapted for the metric story and responsive states.
- `app/layout.tsx` — title, description, favicon, and social-preview metadata.
- `public/og.png` — generated branded social card.
- `tests/content-contract.test.mjs` — checks required copy, priority order, example labels, and exclusions.
- `tests/interactions.test.tsx` — checks both interactive examples and keyboard-accessible controls.
- `tests/rendered-html.test.mjs` — checks server-rendered landmark content and metadata.

---

### Task 1: Create the independent Site checkout

**Files:**
- Create: `sites/beyond-accuracy-bc-evaluation-metrics/**` by copying the retained template source.
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/.openai/hosting.json`

**Interfaces:**
- Consumes: retained Artifact template source and its npm lockfile.
- Produces: an installable, independent Site checkout with one new opaque Sites project ID.

- [ ] **Step 1: Confirm the destination does not exist**

Run:

```bash
test ! -e sites/beyond-accuracy-bc-evaluation-metrics
```

Expected: exit code 0. If the directory exists, inspect it and stop rather than overwrite it.

- [ ] **Step 2: Copy the retained source exactly once**

Create the destination directory and copy the complete retained source, including dotfiles. Do not run `create-sites` because this task explicitly uses the Artifact template.

- [ ] **Step 3: Create a new Sites project**

Call the Sites `create_site` action exactly once with the title `Beyond Accuracy — BC Evaluation Metrics`. Write the returned opaque project ID and source configuration into this checkout’s `.openai/hosting.json`, while preserving any retained logical binding declarations.

- [ ] **Step 4: Install retained dependencies**

Run:

```bash
npm install
```

Expected: installation completes without changing the package manager or dependency versions.

- [ ] **Step 5: Establish the clean template baseline**

Run:

```bash
npm test
```

Expected: the retained template’s tests pass before content changes.

- [ ] **Step 6: Commit the independent checkout**

```bash
git add sites/beyond-accuracy-bc-evaluation-metrics
git commit -m "chore: initialize BC metrics Site"
```

---

### Task 2: Add the metric content contract and first meaningful preview

**Files:**
- Create: `sites/beyond-accuracy-bc-evaluation-metrics/app/metric-content.ts`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/tests/content-contract.test.mjs`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/page.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/model-switcher.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/globals.css`

**Interfaces:**
- Produces: `ModelId = "memorizer" | "state-aware"`, `models`, `priorityMetrics`, `exampleLabel`, and `winnerRule` from `metric-content.ts`.
- Produces: `ModelSwitcher()` as a client component that reads the shared model data.
- The page consumes those exports without duplicating metric values in JSX.

- [ ] **Step 1: Replace the content contract with failing assertions**

Write tests that read `app/metric-content.ts` and `app/page.tsx` and assert:

```js
assert.match(source, /Illustrative example — not measured model results\./);
assert.ok(source.indexOf("Schedule-deviation macro recall") < source.indexOf("Step-prefix AUC@24"));
assert.ok(source.indexOf("Step-prefix AUC@24") < source.indexOf("Raw top-1 accuracy"));
assert.match(page, /A clone can be 90% accurate and still fail to imitate the policy\./);
assert.doesNotMatch(page, /weighted score/i);
```

Use a positive explicit phrase for the selection rule elsewhere: `Compare in priority order — never blend the metrics into one score.`

- [ ] **Step 2: Run the content test and verify it fails**

Run:

```bash
node --test tests/content-contract.test.mjs
```

Expected: FAIL because the new copy and content module do not exist.

- [ ] **Step 3: Add typed illustrative content**

Create the following public interface and values in `app/metric-content.ts`:

```ts
export type ModelId = "memorizer" | "state-aware";

export type ExampleModel = {
  id: ModelId;
  label: string;
  description: string;
  predictions: { water: number; harvest: number; drop: number };
  metrics: { deviationRecall: number; prefixAuc: number; accuracy: number };
};

export const exampleLabel = "Illustrative example — not measured model results.";

export const models: Record<ModelId, ExampleModel> = {
  memorizer: {
    id: "memorizer",
    label: "Schedule memorizer",
    description: "Always chooses WATER because WATER is usually right at this time.",
    predictions: { water: 90, harvest: 0, drop: 0 },
    metrics: { deviationRecall: 0, prefixAuc: 48, accuracy: 90 },
  },
  "state-aware": {
    id: "state-aware",
    label: "State-aware clone",
    description: "Usually chooses WATER, but changes when the farm needs HARVEST or DROP.",
    predictions: { water: 86, harvest: 4, drop: 4 },
    metrics: { deviationRecall: 80, prefixAuc: 78, accuracy: 94 },
  },
};

export const priorityMetrics = [
  { rank: 1, name: "Schedule-deviation macro recall", question: "Does the clone notice when Ryo breaks the routine?" },
  { rank: 2, name: "Step-prefix AUC@24", question: "Can the clone stay correct across consecutive farm steps?" },
  { rank: 3, name: "Raw top-1 accuracy", question: "How often is the clone’s first choice exactly Ryo’s choice?" },
] as const;

export const winnerRule = "Compare in priority order — never blend the metrics into one score.";
```

- [ ] **Step 4: Write the first interaction test**

In `tests/interactions.test.tsx`, render `ModelSwitcher`, click the `State-aware clone` button, and assert that the visible values change from `0%` deviation recall to `80%` and the active button exposes `aria-pressed="true"`.

- [ ] **Step 5: Run the interaction test and verify it fails**

Run:

```bash
npx vitest run tests/interactions.test.tsx
```

Expected: FAIL because the retained switcher does not implement the new model interface.

- [ ] **Step 6: Adapt the retained model switcher**

Keep the existing component file but replace its content with a client component using `useState<ModelId>("memorizer")`. Render two native buttons with `aria-pressed`, the three predicted action counts, and the three metric values from `models[selected]`. Include the `exampleLabel` above the example.

- [ ] **Step 7: Build only the recognizable first slice**

Adapt `page.tsx` and the minimum CSS needed to show:

- the new title and hero statement;
- the 100-decision WATER/HARVEST/DROP setup;
- the working schedule-memorizer/state-aware-clone switcher;
- the three priority metric names.

Leave the retained Site intact until this slice compiles. Do not add the remaining narrative sections in this first slice.

- [ ] **Step 8: Run the focused tests**

Run:

```bash
node --test tests/content-contract.test.mjs
npx vitest run tests/interactions.test.tsx
```

Expected: both pass.

- [ ] **Step 9: Hand off the first meaningful preview**

Start the retained `npm run dev` script in a persistent session. Use the exact printed local URL for one lightweight request and require a successful response. Open that URL once in Codex and retain the returned browser-tab ID for all later handoffs. Do not perform screenshots or visual inspection unless requested.

- [ ] **Step 10: Commit the first slice**

```bash
git add sites/beyond-accuracy-bc-evaluation-metrics/app sites/beyond-accuracy-bc-evaluation-metrics/tests
git commit -m "feat: explain the BC accuracy trap"
```

---

### Task 3: Add the step-prefix recovery demonstration

**Files:**
- Create: `sites/beyond-accuracy-bc-evaluation-metrics/app/step-prefix-demo.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/tests/interactions.test.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/globals.css`

**Interfaces:**
- Produces: `StepPrefixDemo()` with `sequence = [true, true, true, false, true, true, true, true]`.
- Produces: `windowsContaining(stepIndex: number): number[]`, returning zero-based rolling-window start indices that include the selected step.
- `page.tsx` consumes `StepPrefixDemo` without passing state.

- [ ] **Step 1: Write failing unit and interaction tests**

Add assertions that:

```ts
expect(windowsContaining(3)).toEqual([0, 1, 2, 3]);
```

Render the component, activate the mismatch at step 4, verify the explanation says `Only windows touching step 4 break`, and verify a later start remains labeled `Recovery still counts`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
npx vitest run tests/interactions.test.tsx
```

Expected: FAIL because `StepPrefixDemo` and `windowsContaining` do not exist.

- [ ] **Step 3: Implement the minimal sequence model**

Implement `windowsContaining` by returning all start indices from zero through `stepIndex` for this fixed prefix demonstration. Render eight native buttons labeled `Step N: correct` or `Step N: mismatch`; use `aria-pressed` for the selected step and text labels as well as color.

- [ ] **Step 4: Add the plain-language recovery display**

Show `✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓`, highlight affected windows for the selected mismatch, and keep the post-error four-step window visibly eligible. Include these exact explanations:

- `The first run stays correct for 3 steps, then breaks.`
- `A new window starts after the error, so the final 4 correct steps still count.`
- `24 is our first horizon because one Kaggriculture day contains 24 environment turns.`

- [ ] **Step 5: Run the interaction tests**

Run:

```bash
npx vitest run tests/interactions.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the demonstration**

```bash
git add sites/beyond-accuracy-bc-evaluation-metrics/app/step-prefix-demo.tsx sites/beyond-accuracy-bc-evaluation-metrics/app/globals.css sites/beyond-accuracy-bc-evaluation-metrics/tests/interactions.test.tsx
git commit -m "feat: visualize step-prefix recovery"
```

---

### Task 4: Complete the plain-language metric story

**Files:**
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/page.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/story-rail.tsx`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/globals.css`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/tests/content-contract.test.mjs`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes: `priorityMetrics`, `winnerRule`, `ModelSwitcher`, and `StepPrefixDemo`.
- Produces: semantic sections with IDs `question`, `accuracy-trap`, `deviation-recall`, `step-prefix`, `raw-accuracy`, `decision`, `limits`, and `summary`.

- [ ] **Step 1: Write failing rendered-story assertions**

Check the rendered HTML for all eight section IDs and the exact example outcomes:

```js
assert.match(html, /HARVEST recall.*100%/s);
assert.match(html, /DROP recall.*0%/s);
assert.match(html, /Deviation macro recall.*50%/s);
assert.match(html, /matches 92 of 100 held-out decisions/s);
assert.match(html, /First test whether the clone notices Ryo’s exceptions\./);
```

Also assert that limits mention `on-policy cloning error`, `synthetic novel states`, and `final game performance` as deferred.

- [ ] **Step 2: Run rendered and content tests and verify they fail**

Run:

```bash
npm run build
node --test tests/content-contract.test.mjs tests/rendered-html.test.mjs
```

Expected: FAIL because the complete story is absent.

- [ ] **Step 3: Compose the eight-section page**

Use one `h1`, sequential `h2` headings, short paragraphs, example cards, and the retained numbered story rail. Each metric section begins with its plain-language question. Put calculations in compact cards beneath the example rather than leading with equations.

- [ ] **Step 4: Add the model-selection podium**

Render the three selection steps in rank order. Include the secondary note:

`We compare whole games, not shuffled individual rows, so repeated decisions from one game do not pretend to be independent evidence.`

- [ ] **Step 5: Add boundaries and the shareable summary**

State that the current evaluation is replay-only and defer the three unavailable measurements. End with:

`First test whether the clone notices Ryo’s exceptions. Then test whether it stays correct across consecutive steps. Use raw accuracy only as the final sanity check.`

- [ ] **Step 6: Finish responsive and accessible CSS**

Retain the farm-green and warm-paper variables, headline scale, bordered cards, chips, and numbered sections. Add a single-column layout below 760px, minimum 44px interactive targets, `:focus-visible` outlines, non-color correctness labels, and a `prefers-reduced-motion: reduce` rule.

- [ ] **Step 7: Run the complete local test suite**

Run:

```bash
npm test
npm run typecheck
npm run lint
```

Expected: all pass.

- [ ] **Step 8: Commit the complete story**

```bash
git add sites/beyond-accuracy-bc-evaluation-metrics/app sites/beyond-accuracy-bc-evaluation-metrics/tests
git commit -m "feat: complete the BC metrics story"
```

---

### Task 5: Add share metadata and the social card

**Files:**
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/app/layout.tsx`
- Create: `sites/beyond-accuracy-bc-evaluation-metrics/public/og.png`
- Modify: `sites/beyond-accuracy-bc-evaluation-metrics/tests/rendered-html.test.mjs`

**Interfaces:**
- Consumes: the stable title, subtitle, farm palette, and deployed trusted origin.
- Produces: site-wide Open Graph and X metadata using `/og.png` from a trusted absolute origin.

- [ ] **Step 1: Start the required social-preview asset task after preview**

Spawn exactly one image-only subagent with `fork_turns="none"`. Instruct it to make exactly one imagegen request for a cohesive 1200×630 landscape card, save the result outside the Site checkout, and return the path. The card must contain the exact title `Beyond Accuracy — BC Evaluation Metrics` and supporting line `A clone can be 90% accurate and still miss the policy.` in the farm-green/warm-paper visual language. It must not call Sites tools, invoke Sites skills, edit Site source, initialize a Site, or spawn another agent.

- [ ] **Step 2: Write failing metadata checks**

Assert that layout metadata contains:

```ts
title: "Beyond Accuracy — BC Evaluation Metrics"
description: "Three replay-only metrics for testing whether a behavior clone copies Ryo’s decisions, not just the clock."
```

Assert Open Graph and X entries use the same title/description and an `og.png` image.

- [ ] **Step 3: Run the metadata check and verify it fails**

Run:

```bash
node --test tests/rendered-html.test.mjs
```

Expected: FAIL because reference metadata remains.

- [ ] **Step 4: Inspect and integrate the returned card**

Inspect the image for exact, legible text and no invented words. Retry once only if unusable. Copy the accepted image into `public/og.png` and set Open Graph and X metadata in `app/layout.tsx`. Build the image URL from an explicitly trusted deployment/request origin; do not blindly trust forwarded host headers.

- [ ] **Step 5: Run metadata and build validation**

Run:

```bash
node --test tests/rendered-html.test.mjs
npm run build
```

Expected: PASS.

- [ ] **Step 6: Commit share metadata**

```bash
git add sites/beyond-accuracy-bc-evaluation-metrics/app/layout.tsx sites/beyond-accuracy-bc-evaluation-metrics/public/og.png sites/beyond-accuracy-bc-evaluation-metrics/tests/rendered-html.test.mjs
git commit -m "feat: add BC metrics share preview"
```

---

### Task 6: Verify, save, and publish the new Site

**Files:**
- Verify: `sites/beyond-accuracy-bc-evaluation-metrics/**`
- Modify only if verification finds an actual defect.

**Interfaces:**
- Consumes: the complete pushed source commit and new `.openai/hosting.json` project ID.
- Produces: one saved Sites version and one public production URL.

- [ ] **Step 1: Run final automated verification**

Run:

```bash
npm test
npm run typecheck
npm run lint
git diff --check
```

Expected: all pass and no whitespace errors.

- [ ] **Step 2: Verify the working tree scope**

Run `git status --short` and confirm that unrelated pre-existing repository changes remain untouched. Confirm all Site changes are committed.

- [ ] **Step 3: Push the exact Site source state**

Obtain the Sites source credential for the new project and push the exact committed source state required by Sites. Record the pushed commit SHA; do not save a version from an unpushed or different state.

- [ ] **Step 4: Save and deploy the version**

Use the new project ID and pushed commit SHA to save one version, then deploy that saved version publicly. If deployment is non-terminal, inspect status until it succeeds or returns a real failure.

- [ ] **Step 5: Reuse the preview tab for the live URL**

Navigate the retained Codex browser tab to the production URL. Do not open a second handoff tab. Do not perform visual QA unless the user requests it.

- [ ] **Step 6: Report the public deliverable**

Return the production URL as the primary result and summarize the three metric priorities in one sentence. Do not include internal project IDs, credentials, or build details.
