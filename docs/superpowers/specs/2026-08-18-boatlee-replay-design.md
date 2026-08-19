# Replay-Driven Baseline3k Improvement Design

## Purpose

Build a reproducible replay-learning and evaluation subsystem in
`duy/another_work/02_boatlee_replay` that can improve
`duy/another_work/01_baseline3k/main.py`. The subsystem will study every
compatible top-submission replay, learn both complete core strategies and
general adaptive behavior, and promote only changes that demonstrate live
paired-game uplift against baseline3k.

This first phase deliberately optimizes against baseline3k. The existing
BoatLeeV3 leaderboard agent remains untouched and is not the binding comparison
target for this phase.

## Binding Inputs and Constraints

- Baseline source:
  `duy/another_work/01_baseline3k/main.py`
- Baseline SHA-256:
  `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`
- Replay directory:
  `duy_explore/kaggriculture-episodes-2026-08-15/top-100`
- Replay count: exactly 90 files after removal of the ten version-1.32.6
  mismatches.
- Required replay and runtime environment: Kaggriculture module version
  `1.32.7`.
- Season length: 720 steps.
- All new analysis, candidates, artifacts, and promoted output live in
  `duy/another_work/02_boatlee_replay`.
- Neither baseline3k nor BoatLeeV3 may be edited by this project.
- A submitted agent must be one standalone `main.py` with no runtime replay,
  repository-file, notebook, network, or non-standard-library dependency.
- Agent mean decision time must remain below 1 ms and p95 below 2 ms on the
  replay gate.

## Non-Goals

- Do not imitate a single replay and call it a general strategy.
- Do not use replay reward correlation as proof of causal uplift.
- Do not splice arbitrary actions from unrelated routes.
- Do not use future observations, future prices, future shop unlocks, final
  rewards, or hidden opponent state at runtime.
- Do not promote a candidate merely because it beats a built-in or random
  agent.
- Do not modify or delete the accepted replay corpus.

## System Architecture

### 1. Replay inspector

`replay_inspector.py` validates and converts all 90 replay files into compact,
deterministic evidence.

For every replay it must:

- require module version `1.32.7` and the expected 720-step configuration;
- validate two player states and action/observation availability at every
  step;
- shift stored actions by one replay state so each recorded action is aligned
  with the observation that produced it;
- inspect both seats;
- record winner, final margin, seat, opponent, shop sequence, and public farm
  trajectory;
- extract field actions, market orders, purchases, hires, land unlocks,
  planting, livestock, inventory flow, production, sales, prices, and recovery
  behavior;
- normalize only environment noise such as observation-proven random weed
  repair while preserving strategy-defining quantities and order timing; and
- produce stable, sorted JSON that regenerates byte-for-byte.

Malformed or wrong-version input is a hard failure. The inspector must not
silently skip files.

### 2. Deterministic replay split

The inspector assigns 60 replays to discovery and 30 to replay holdout using a
deterministic, documented stratified split. It must minimize distribution
differences for winner team, winner seat, opponent, shop sequence, and detected
core family, with source filename as the final tie-breaker. The assignment must
contain exactly 60 discovery and 30 holdout filenames and must regenerate
identically from the same corpus.

Discovery replays are used to form hypotheses and select route families.
Holdout replays are not used to invent or tune hypotheses; they are used only
to verify recurrence and generality.

### 3. Core-strategy lane

The core lane searches for coherent replacements for baseline3k's complete
720-step schedule. It clusters full-season strategies using:

- farmer and hand movement;
- daily hiring schedule;
- land purchases and unlock timing;
- crop layout and seed purchasing;
- animal structures, livestock mix, feeding, care, and collection;
- inventory deposits, pickups, and shed pressure;
- market purchasing and selling; and
- shop sequence and opponent context.

A core candidate must be one of:

1. a complete route from one replay trace that represents a supported cluster;
2. a medoid route that minimizes disagreement within a coherent cluster; or
3. a route assembled only at a handoff proven compatible by identical or
   equivalent live state.

A compatible handoff requires proof covering actor positions, ordered hands,
unlocked land, tile contents, purchases, seed and shed inventory, carried
inventory, money obligations, and any controller state used by the candidate.
Without that proof, the route cannot be spliced.

Core candidates are evaluated as complete replacements rather than isolated
action edits. Their adaptive overlays are disabled during the initial core
screen so core value is measurable.

### 4. Adaptive-heuristic lane

The adaptive lane mines decisions whose triggers can be computed from the
current observation. Candidate categories include:

- shop-demand-aware crop and livestock allocation;
- purchase timing and affordability recovery;
- sale ordering, quantity, and market-price timing;
- shed-capacity and carried-inventory protection;
- feeding, room, weed, and terminal-liquidation safety;
- hiring and land timing; and
- responses to public opponent production or market pressure.

#### Minimum replay support

An adaptive strategy is eligible for implementation only when all of the
following are true:

- it appears in at least 10 distinct discovery replays;
- it recurs in at least 5 distinct holdout replays;
- it spans at least 3 distinct opponents or team signatures;
- it is seen in both seats when the available supporting data contains both
  seats;
- one replay contributes at most one support unit to each hypothesis; and
- its trigger and action are expressible using current-observation fields.

This support threshold is binding. A strategy observed in only one replay,
even a very high-scoring replay, may be documented but cannot become agent
behavior.

Rare emergency behavior may be implemented only as a mechanically proven game
safety invariant, not labeled or promoted as a replay-learned adaptive
strategy. It must still pass the same live evaluation gates.

Each eligible hypothesis records:

- identifier and human-readable explanation;
- observable trigger;
- proposed action change;
- intended economic or safety effect;
- discovery and holdout replay support;
- supported seats, opponents, shops, days, and core families;
- replay reward and margin association, clearly labeled non-causal;
- known conflicts and failure risks; and
- experiment status and exact candidate hashes.

Adaptive hypotheses are first tested independently on the unchanged
baseline3k core. This isolates the value of each rule.

### 5. Candidate builder

`candidate_builder.py` generates standalone agents from the frozen baseline
source. Every generated candidate records:

- baseline source path and SHA-256;
- core-route identifier or `baseline3k`;
- enabled adaptive hypothesis identifiers;
- generated source SHA-256; and
- build-schema version.

Generation must be deterministic. Two builds of the same candidate must be
byte-identical. The builder rejects unknown identifiers, incompatible route
handoffs, duplicate feature application, changed baseline bytes, and any
runtime external-file dependency.

The agent exception path returns aligned `PASS` actions for the farmer and all
current hands with no market orders.

### 6. Evaluator

`evaluate.py` runs candidates through three gates against
`01_baseline3k/main.py`.

#### Replay gate

The candidate processes both player observations in all 90 replays. It must
have:

- zero exceptions;
- valid action dictionaries;
- farmer and hand action counts aligned with the observation;
- no more than ten market orders;
- JSON-serializable actions;
- zero determinism mismatches;
- mean decision time below 1 ms; and
- p95 decision time below 2 ms.

This gate establishes safety, coverage, determinism, and speed. It does not
claim counterfactual reward.

#### Development screen

Run seeds `0..9` in both seat assignments for 20 games. A candidate qualifies
only if:

- all games finish normally;
- rewards equal final money;
- paired mean margin is positive; and
- mean margin is positive in both candidate seats.

Core replacements and adaptive rules are screened independently. A component
with a non-positive screen cannot enter a combination.

#### Fresh confirmation

Freeze candidate bytes after screening, then run seeds `50..99` in both seat
assignments for 100 games. Promotion requires:

- exactly 100 valid games;
- paired mean margin greater than zero;
- paired median margin greater than zero;
- win rate greater than 55 percent;
- deterministic 95 percent bootstrap confidence interval with lower bound
  greater than zero;
- positive mean margin in both candidate seats;
- every game ending with `DONE`; and
- every reward matching final money.

The confirmed bytes must match the screened candidate hash. Rebuilding or
editing after screening invalidates confirmation.

### 7. Combination lane

Combinations may contain only independently confirmed components. Test the
smallest combinations first. A combination must pass the same replay,
development, and fresh-confirmation gates and must beat every included
component on the same 100-game confirmation panel before it can replace those
components.

If no candidate passes, baseline3k remains the winner and the project does not
create a promoted `main.py`.

## Artifacts

The project produces:

- `replay_catalog.json`: input hashes, validation results, discovery/holdout
  membership, compact traces, and family summaries;
- `strategy_hypotheses.json`: core families, adaptive hypotheses, support,
  risks, and experiment status;
- `EXPERIMENTS.md`: human-readable tested candidates, hashes, metrics,
  rejections, and decisions;
- benchmark result directories containing per-game and paired-seed records;
  and
- `main.py` only after full promotion.

All JSON artifacts use stable key ordering and indentation and must regenerate
byte-for-byte. Large raw observations are not duplicated into tracked
artifacts.

## Testing Strategy

Implementation follows test-driven development. Required unit and integration
coverage includes:

- replay validation and wrong-version rejection;
- stored-action shift alignment;
- both-seat extraction;
- deterministic 60/30 split and stratification invariants;
- feature extraction for purchases, hires, fields, animals, inventory,
  markets, shops, and opponents;
- core-family clustering and deterministic medoid selection;
- complete-route and handoff compatibility rejection;
- adaptive support counted once per replay;
- enforcement of 10 discovery, 5 holdout, 3 opponent, and seat-diversity
  thresholds;
- rejection of future-information triggers;
- deterministic candidate building and source-hash binding;
- standalone import and safe fallback behavior;
- replay action validation, determinism, and latency summaries;
- paired-seat scheduling, reward/money equality, bootstrap confidence
  interval, and promotion failures; and
- a real-corpus regression over the 90 accepted replay filenames.

Tests must use compact fixtures except for explicit real-corpus regressions.
No test may mutate the replay corpus, baseline3k, or BoatLeeV3.

## Execution Sequence

1. Implement and verify replay inspection and deterministic evidence artifacts.
2. Publish core families and adaptive hypotheses before altering agent behavior.
3. Review the evidence report and choose a small, evidence-ranked candidate
   set.
4. Build and screen complete core replacements with adaptive rules disabled.
5. Build and screen adaptive rules on the baseline3k core.
6. Confirm independent winners on fresh seeds.
7. Test only qualified combinations.
8. Freeze the strongest fully confirmed bytes.
9. Generate the standalone promoted `main.py` and record its SHA-256.

## Acceptance Criteria

The project is complete when either:

1. a standalone candidate passes every binding replay, development, and fresh
   confirmation gate and `main.py` is generated from the exact confirmed
   bytes; or
2. all evidence-qualified candidates are rejected, baseline3k remains the
   winner, no promoted `main.py` is created, and the artifacts fully explain
   the result.
