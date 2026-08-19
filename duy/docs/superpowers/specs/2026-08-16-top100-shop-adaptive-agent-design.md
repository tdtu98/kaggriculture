# Top-100 Shop-Adaptive Agent Design

**Date:** 2026-08-16  
**Target:** `duy/another_work/02_inspect_top1`  
**Runtime:** `kaggle-environments==1.32.7`

## Objective

Upgrade `02_inspect_top1` using the top 100 Kaggriculture replays from
2026-08-15. The promoted agent must beat `01_baseline3k` as strongly and
reliably as the available evidence supports.

The primary success metric is paired head-to-head performance against
`01_baseline3k`, with each benchmark seed played in both seat assignments.
Absolute final money is secondary. The design does not promise a win on every
random seed; it requires statistically supported superiority across an unseen
confirmation panel.

## Constraints

- Keep `duy/another_work/01_baseline3k/main.py` unchanged.
- Keep the final `02_inspect_top1/main.py` standalone and standard-library
  only. It must not read replay files or repository-local modules at runtime.
- Do not modify or stage the downloaded files under `duy_explore/`.
- Treat replay seeds as discovery evidence only, never promotion evidence.
- Preserve the already-staged `02_inspect_top1/main.py` controllers and tests.
- Do not replace the current agent until the candidate passes the confirmation
  gate against `01_baseline3k`.
- Do not spend benchmark time comparing the candidate with the old `02`.
- Run all development and confirmation games with
  `kaggle-environments==1.32.7`.

## Evidence

### Environment boundary

The downloaded corpus contains 100 valid JSON replays but crosses an engine
version boundary:

- 90 replays use `module_version == 1.32.7`.
- 10 replays use `module_version == 1.32.6`.

Version 1.32.7 changes the market scarcity curve for carrot, tomato, and egg
to a new `hinge` shape. The 1.32.6 replays are therefore excluded from route
selection and strategy statistics.

### Compatible replay cohort

The 90 compatible replays contain 180 player routes from 11 named agents.
`カワシギ` is the strongest recurring source policy:

- 69 games.
- 57 wins.
- Mean final money of 93,690.6.
- Mean head-to-head margin of +4,822.8.

Of those 69 routes, 68 use the newer 42-strawberry/12-melon crop plan. The
current `02` agent embeds the older 34-strawberry/20-melon plan.

### Coherent opening families

Normalized action prefixes through step 71 reveal two large coherent opening
families and one singleton. The largest family contains 36 routes, wins 31 of
its source games, and has examples of every supported livestock branch. After
excluding its lone 34-strawberry/20-melon route, it provides 35 compatible
42-strawberry/12-melon traces:

- 23 routes with 10 cows and 4 sheep.
- 4 routes with 8 cows and 6 sheep.
- 4 routes with 6 cows and 8 sheep.
- 4 routes with 6 cows and 12 sheep.

This family supplies the canonical opening and all branch evidence. Routes
from the other opening family must not be spliced into it.

### Current baseline

Under 1.32.7, the current staged `02` agent was screened against
`01_baseline3k` on seeds 0 through 19 in both seats:

- 20 wins and 20 losses.
- Mean paired margin of +1,919.55.
- Median paired margin of +534.
- Seat-zero mean margin of +2,182.05.
- Seat-one mean margin of +1,657.05.
- Deterministic 95% bootstrap interval of [-111.00, +4,019.05].

The current result is promising but does not establish superiority because
the confidence interval crosses zero and the win rate is only 50%.

## Architecture

### 1. Version-aware offline inspector

Extend `inspect_replays.py` to process a heterogeneous replay directory while
requiring compatible route evidence:

1. Validate replay shape and full 720-step completion.
2. Record `module_version` and reject route candidates not produced by
   1.32.7.
3. Locate the requested team independently in each replay.
4. Shift stored replay actions back to the observation that produced them.
5. Normalize replay-specific weed digs and market quantities for comparison.
6. Fingerprint the first 72 actions to identify coherent opening families.
7. Select the largest family that contains all four supported livestock
   branches and uses the 42-strawberry/12-melon crop plan.
8. Select a deterministic medoid route for each branch.

Medoid comparison covers field actions and market intent. Replay-specific
weed corrections use a sentinel. Sell quantities are normalized so inventory
variation does not dominate distance; purchase type and quantity remain part
of the comparison because they define production capacity.

The inspector writes deterministic, sorted artifacts containing cohort
counts, rejected-version counts, opening-family membership, branch evidence,
medoid sources, handoff checks, route actions, and weed annotations.

### 2. Handoff compatibility

The four medoids share the same opening family, but a common first 72 steps is
not sufficient to prove that an arbitrary later full-route switch is safe.
Before emitting a branch payload, the inspector must verify the handoff:

- Compare action prefixes through the proposed decision step.
- Compare route-defining purchases already attempted.
- Compare canonical farm checkpoints, actor counts, and unit positions.
- Identify the first branch-specific animal purchase, placement, and field
  maintenance operations.

If complete route prefixes are compatible, the runtime may select the full
branch continuation. If they are not compatible, the generated payload keeps
one canonical route and includes only narrow, replay-proven translations for
animal purchases, placements, and matching maintenance actions. Unsafe trace
splicing is forbidden.

### 3. Shop branch selector

The selector uses only the live `town.unlocked_shops` observation. Duplicate
shop instances are preserved, matching 1.32.7 behavior, while branch choice
depends on presence rather than uniqueness.

At step 144, record whether Yarn Store appeared among the first two unlocks.
At step 216, freeze one livestock branch for the episode:

1. Yarn Store appeared by step 144: 6 cows / 12 sheep.
2. Yarn Store first appears by step 216: 6 cows / 8 sheep.
3. Otherwise, Pizza Shop, Ice Cream Shop, or Smoothie Shop is present:
   10 cows / 4 sheep.
4. Otherwise: 8 cows / 6 sheep.

The choice never changes after step 216. Missing or malformed town data falls
back to the well-supported 10-cow/4-sheep route.

### 4. Standalone runtime agent

Embed the generated route payload in `main.py` using the existing compressed
standard-library representation. Runtime flow is:

1. Reset per-seat episode state on step zero or a backwards step.
2. Read the canonical action for the current step and selected branch.
3. Apply the existing live weed repair.
4. Apply field precondition guards.
5. Apply optional market or purchase controllers only when their feature flag
   is enabled for a benchmarked candidate.
6. Cap market orders at ten and align hand actions to the live hand count.

The public `agent(obs)` interface remains unchanged. Exceptions return one
`PASS` action for the farmer, one for every live hand, and no market orders.

## Runtime Safety

Field guards suppress invalid replay actions for:

- `FEED` without a live unfed animal and carried wheat.
- `CARE` without a live not-yet-cared animal.
- `COLLECT_FERTILIZER` without available fertilizer.
- `HARVEST` without live yield.
- `FERTILIZE` without a valid plant and carried fertilizer.
- Replay-specific `DIG` when the live tile is not a weed.

A weed blocking scheduled planting or pasture construction is dug, the
intended action is retried on the next step, and the actor replays a bounded
number of delayed route actions.

Purchase recovery, inventory-aware sale capping, and premium front-running
remain isolated behind feature flags. They are not enabled together without
ablation evidence.

## Testing

Implementation follows test-driven development. Required tests cover:

- Module-version extraction and filtering.
- Stable counts of 90 accepted and 10 rejected replays.
- Stable detection of 69 compatible `カワシギ` routes.
- Deterministic opening-family fingerprints and family selection.
- Exclusion of the lone old crop-plan route from the selected family.
- Stable branch counts of 23/4/4/4.
- Deterministic medoid selection for every branch.
- Market-intent normalization and weed-only normalization.
- Handoff acceptance and rejection.
- Exact shop-branch classification, including duplicate shops.
- Branch freeze and per-seat episode reset.
- Missing-town fallback.
- Route embedding with no repository-file dependency.
- Existing field guards, weed recovery, hand alignment, market-order cap, and
  exception fallback.
- Deterministic reruns of identical seed and seat combinations.

A smoke benchmark must finish in both seats with `DONE` statuses and rewards
equal to final money before longer screening begins.

## Development and Promotion

### Development panel

Use seeds 0 through 49 in both seats against `01_baseline3k` for 100 games per
complete candidate.

Screen candidates in causal stages:

1. New replay-derived route with no optional controllers.
2. Route plus field guards.
3. Add purchase recovery alone.
4. Add sale capping alone.
5. Add premium front-running alone.
6. Test combinations only from individually positive additions.

Reject any candidate with an agent error, non-`DONE` status, reward/money
mismatch, negative paired mean, or a regression in both seat-specific mean
margins. Select the development winner by paired mean margin, using win rate,
paired median, and worst-seed behavior as secondary criteria.

### Frozen confirmation

Before confirmation, record the candidate and `01_baseline3k` SHA-256 hashes.
Run seeds 1000 through 1099 in both seats for 200 games. This range is not used
for tuning.

Promotion requires all of the following:

- 200 valid `DONE` games and no reward/money mismatch.
- Positive paired mean margin.
- Positive paired median margin.
- More than 55% game wins.
- Positive mean margin with the candidate in seat zero.
- Positive mean margin with the candidate in seat one.
- A deterministic 10,000-resample 95% bootstrap confidence interval whose
  lower bound for paired mean margin is greater than zero.

Only a candidate passing every gate replaces
`duy/another_work/02_inspect_top1/main.py`. If no candidate passes, preserve
the current staged agent and document the failed evidence.

## Artifacts

The completed work records:

- Compatible and rejected replay inventories.
- Opening-family and branch summaries.
- Deterministic route payload and source medoids.
- Human-readable strategy findings.
- Unit-test output.
- Development ablation results.
- Frozen hashes and confirmation benchmark artifacts.
- Exact Python and `kaggle-environments` versions.

The public replay JSON files remain unmodified and untracked.
