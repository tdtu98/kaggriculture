# Top-1 Replay Inspection and Agent Improvement Design

**Date:** 2026-08-14

**Status:** Approved for implementation planning

## Objective

Create `duy/another_work/02_inspect_top1` by starting from the proven
`01_baseline3k` champion, learning the reproducible parts of the public top-1
replays in `duy_explore/top1_14_Aug`, and promoting a new agent only when a
held-out, paired-seat benchmark demonstrates that the improvement is not a
seed, seat, or replay-selection accident.

The work must produce both an inspectable strategy analysis and a standalone
Kaggle submission agent. The final agent must not read replay files or any
other repository file at runtime.

## Current Evidence

The replay set contains 13 complete 720-step episodes under the competition's
3,000-coin configuration: 12 public matches and one top-1 self-play match. The
top-1 player is named `カワシギ` in the replay metadata and appears in both
seat positions.

The public top-1 strategy differs materially from `01_baseline3k`:

- It normally builds 10 cows and 4 sheep, rather than 8 cows and 4 sheep.
- It hires 277 hands across the season, rather than 264.
- It opens with two cows, two sheep, twelve melon seeds, and seven wheat seeds.
- It buys land near steps 148 and 264, rather than steps 160 and 240.
- It uses carrots in addition to wheat, melon, and strawberry.
- It has shop-dependent livestock and premium-crop branches.
- In the inspected 10-cow/4-sheep reference replay it submits 318 `CARE` and
  321 `FEED` operations, compared with `01_baseline3k`'s 967 scheduled `CARE`
  and 290 `FEED` operations. The timing is consistent across the default-route
  replays rather than being repeated filler.
- It attempts frequent inventory-aware product sales and retains all 14
  animals in the inspected default-route episodes.

A diagnostic agent that replayed each public top-1 action trace on its original
seed and seat against `01_baseline3k` won 9 of 12 games. Its mean margin was
+2,668.25 coins and its median margin was +1,141.50 coins. This is discovery
evidence only: the traces contain seed-specific weed responses and opponent-
specific market actions, so these results cannot qualify a candidate for
promotion.

## Considered Approaches

### Copy one replay verbatim

This has the highest single-episode fidelity, but it preserves seed-specific
weed corrections, shop choices, and opponent-dependent market quantities. It
is useful as a diagnostic probe and unsuitable as the final agent.

### Patch only isolated `01_baseline3k` controllers

This has low implementation risk, but it cannot reproduce the main observed
advantage: the coordinated 14-animal production route, earlier expansion, and
different crop schedule.

### Hybrid replay reconstruction

This is the selected approach. Reconstruct a representative top-1 production
route from several consistent replays, retain the proven recovery and premium
market mechanisms from `01_baseline3k`, and add narrowly scoped live guards.
Every new controller remains independently switchable for controlled
ablation.

## Deliverables

Create the following under `duy/another_work/02_inspect_top1`:

- `main.py`: standalone submission agent with an embedded, compressed action
  schedule and all runtime controllers.
- `inspect_replays.py`: deterministic replay inspection and route-selection
  tool. It reads replay paths supplied on the command line and writes only to
  an explicitly supplied output path.
- `STRATEGY.md`: evidence-backed description of the top-1 strategy, its
  differences from `01_baseline3k`, and which lessons were retained.
- `BENCHMARK_FINDINGS.md`: final ablations, held-out results, qualification
  decision, hashes, and reproducibility details.

Add focused tests under `duy/` following the repository's existing test-file
layout. Benchmark result directories continue to live under
`duy/benchmarks/results` and use the existing timestamped naming convention.

## Replay Inspection

### Validation

For every input replay, the inspector must verify:

- Required top-level metadata and `steps` exist.
- Exactly two player states exist at every step.
- The episode contains 720 states.
- The configuration uses `startingMoney=3000`, `episodeSteps=720`,
  `turnsPerDay=24`, and `townCenterSellInterval=24`.
- Exactly one seat contains the requested top-1 team name, except self-play,
  where the requested seat must be explicitly selected.
- Actions have the expected `farmer`, `hands`, and `market` structure.

Invalid or ambiguous replay input must raise a descriptive error. The
inspector must not silently omit bad episodes.

### Action alignment

Kaggle replay state zero contains the initial placeholder action. The action
stored at replay state `t + 1` is the action produced from observation `t`.
The inspector and generated route must apply this one-step shift explicitly.
Tests must cover the first purchase/build action so an off-by-one error cannot
produce a passive 3,000-coin agent.

### Extracted evidence

For each replay and for the aggregate set, extract:

- Player seat, opponent, seed, reward, margin, and final status.
- Daily hand hires and land purchases.
- Animal, seed, product, and crop purchase totals and timing.
- Submitted field-action counts by operation.
- Market order counts and quantities by operation and item.
- Farm composition and money at daily checkpoints.
- Shop unlock sequence and inferred livestock/crop branch.
- Weed-specific `DIG` interventions, based on the actor's preceding
  observation.
- Precondition-valid `FEED`, `CARE`, collection, harvest, and sale attempts
  where the replay observation supplies enough evidence.

The report must distinguish submitted quantities from confirmed production or
sales. Invalid market requests and same-turn deposits make submitted order
totals larger than realized inventory flow.

### Branches and canonical route

Cluster replays by realized livestock targets and premium-crop purchase mix.
The default canonical branch is the largest supported 10-cow/4-sheep cluster.
Within that cluster, select a medoid replay: the replay with the smallest total
actor-action disagreement from the other cluster members after excluding
market quantities and observation-proven weed-only `DIG` actions.

Do not create a per-step majority-vote route by combining arbitrary actors
from unrelated episodes. The medoid preserves a coherent movement and
coordination schedule. Peer replays are used to select and annotate the route,
not to splice it into a new schedule.

Only implement a shop-dependent alternative branch when at least two replays
support the same branch. A singleton branch remains documented evidence but is
not added to the submission agent.

## Runtime Agent Architecture

`main.py` starts from the `01_baseline3k` runtime structure: a compressed
schedule, per-seat state, hand alignment, narrow delayed-action recovery, and
one-turn premium-sale front-running. The production schedule is replaced by
the selected top-1 route.

### Episode and seat state

Maintain independent state for seats zero and one. Reset a seat whenever the
observed step is zero or moves backwards. Mutable state includes active weed
repair transactions, purchase recovery targets, front-run debt, and the last
observed step.

### Hand alignment

Always emit exactly one action for each live hand. Truncate schedule actions
for hands that were not successfully hired and pad missing actions with
`PASS`. Never let a missed hire produce an invalid action structure.

### Field-action safety

- Observation-proven replay-specific weed `DIG` operations execute only when
  the live target is a weed; otherwise they become `PASS`.
- A scheduled `PLANT` or `BUILD_PASTURE` blocked by a live weed uses the
  existing delayed-action repair: dig, retry the intended action, then replay
  the affected actor's delayed schedule for a bounded window.
- `DIG` must not remove a live productive plant or occupied structure unless
  the canonical action was an ordinary route action rather than a replay-
  specific weed correction.
- `FEED`, `CARE`, `COLLECT_FERTILIZER`, `HARVEST`, and `FERTILIZE` are emitted
  only when the live tile and available private inventory make the action
  meaningful. An invalid guarded action becomes `PASS`; it is not replaced by
  an unrelated movement that could desynchronize the route.

### Purchase recovery

The canonical market schedule remains authoritative for ordinary timing.
Track critical cumulative targets for hires, land, cows, sheep, route-defining
seeds, and feed wheat. When a scheduled critical purchase fails because of
money or the ten-order cap, retry it at the next route-approved market slot
where it is affordable and capacity remains. Recovery must not exceed the
branch's cumulative target or displace a higher-priority animal-feed purchase.

The order priority is:

1. Wheat needed for imminent animal feed.
2. A missed animal required by an upcoming placement.
3. A missed hire required by the current day's action table.
4. Route-defining seeds required by an upcoming planting wave.
5. Land and non-urgent purchases.
6. Sales and optional market timing.

### Inventory-aware selling

For each scheduled sale, cap the quantity at the live shed stock plus
same-turn planned deposits, less inventory reserved by scheduled pickups.
Remove zero-quantity orders. The controller must preserve the ten-order cap
and must not count carried inventory unless a live actor is scheduled to place
or drop it into the shed on the same turn.

Retain the `01_baseline3k` one-turn premium front-run controller for melon,
milk, strawberry, and wool. It may move only observable unreserved stock and
must subtract exactly the moved quantity from the original next-turn sale.
Front-run debt resets between episodes.

### Failure behavior

The submission entry point catches unexpected runtime exceptions and returns
an aligned `PASS` action with no market orders. Analysis code does not use this
fallback: malformed data and violated invariants must fail loudly in tests and
the replay inspector.

## Feature Isolation and Ablation

Expose internal feature constants so tests and development-only loaders can
disable each addition without changing the schedule:

- Canonical top-1 route.
- Field-action guards.
- Purchase recovery.
- Inventory-aware sale capping.
- Premium front-running.
- Any replay-supported shop branch.

Development comparisons add one controller at a time. A controller advances
only when its candidate has no agent errors and a positive paired mean margin
against the immediately preceding variant. Non-improving complexity is
removed from the final agent.

## Testing

### Unit and replay tests

Tests must cover:

- Top-1 seat detection, including explicit self-play seat selection.
- Replay validation errors and the action one-step shift.
- Stable per-replay and aggregate strategy statistics for the supplied set.
- Branch clustering and deterministic medoid selection.
- Per-seat state reset on step zero and backwards steps.
- Hand truncation and padding.
- Weed-only `DIG` suppression and delayed planting/build recovery.
- Guards for feed, care, collection, harvest, fertilize, and destructive dig.
- Purchase recovery priority, affordability, cumulative caps, and market-order
  capacity.
- Sale quantities with shed stock, same-turn deposits, and pickup reserves.
- Front-run movement and exact debt repayment.
- Exception fallback shape.
- Standalone import with no replay or repository-file dependency.
- Deterministic reruns of identical seed and seat combinations.

### Smoke benchmark

Before ablation, run three seeds in both seats for six games. All final states
must be `DONE`, rewards must equal final money, and repeated identical runs
must match exactly.

## Benchmark and Promotion Protocol

### Discovery and development

Public replay seeds are discovery data and never count toward qualification.

Use seeds 0 through 19 in both seats for rapid ablation screening. Surviving
complete candidates then run against `01_baseline3k` on seeds 0 through 49 in
both seats. Reject any variant with an agent error, a negative paired mean
margin, or a regression in both seat-specific mean margins.

Development results may select and tune a candidate. Confirmation results may
not.

### Frozen confirmation against `01_baseline3k`

Before confirmation, record the exact candidate and baseline SHA-256 hashes.
Run seeds 1000 through 1099 in both seat assignments for 200 games.

For seed `s`, define the paired margin as the mean of the candidate's margin
when it occupies seat zero and its margin when it occupies seat one. The seed,
not the individual game, is the primary statistical unit.

Promotion requires all of the following:

- No non-`DONE` status or reward/money mismatch.
- Positive mean paired margin.
- Positive median paired margin.
- More than 55% wins across the 200 games.
- Positive mean margin with the candidate in seat zero.
- Positive mean margin with the candidate in seat one.
- A deterministic 95% bootstrap confidence interval for the mean paired
  margin whose lower bound is greater than zero. Use 10,000 bootstrap
  resamples and record the bootstrap RNG seed.

Run this holdout only after the candidate is frozen. If it fails, return to
development and assign the next untouched 100-seed range to a new frozen
candidate; do not tune against seeds 1000 through 1099.

### Robustness against `00_baseline`

Run seeds 2000 through 2049 in both seats for 100 games. Require no errors,
positive mean and median paired margins, more than 55% wins, and positive mean
margins in both seats. This panel checks that an improvement aimed at the
current champion does not depend entirely on a close-mirror market effect.

### Reproducibility artifacts

Every persisted benchmark records:

- Candidate and opponent identifiers, resolved paths, and SHA-256 hashes.
- Python and `kaggle-environments` versions.
- Full configuration and exact ordered schedule.
- Per-game seat, money, reward, status, outcome, and candidate margin.
- Per-seed paired margins and confidence-interval inputs.
- Human-readable and machine-readable summaries.

Save per-game CSV, summary JSON, and summary text using the existing benchmark
directory convention. `BENCHMARK_FINDINGS.md` links the final artifacts and
states clearly whether `02_inspect_top1` qualified.

## Scope Boundaries

- Do not modify `01_baseline3k`; it remains the immutable reference.
- Do not add a general path planner or reinforcement-learning system.
- Do not infer opponent-private inventory.
- Do not use the confirmation panel for iterative tuning.
- Do not add a shop branch supported by only one replay.
- Do not claim improvement from public matched-seed replay probes or a single
  benchmark slice.

## Acceptance Criteria

The task is complete when:

1. Replay inspection is deterministic, validated, and documented.
2. `02_inspect_top1/main.py` is a standalone, exception-safe agent derived
   from the approved hybrid architecture.
3. Unit, replay, submission, and deterministic smoke tests pass.
4. Development ablations identify the frozen candidate without consulting the
   confirmation seeds.
5. The complete 200-game confirmation and 100-game robustness panels are
   persisted with reproducibility metadata.
6. The findings document reports the promotion decision using every approved
   threshold, including the paired-seed bootstrap interval.
