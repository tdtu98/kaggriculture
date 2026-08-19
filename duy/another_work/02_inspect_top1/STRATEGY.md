# Top-100 Replay Lessons and Promoted Baseline Strategy

## Binding promotion status

The full replay-route candidate was not promoted. Its best isolated tuple,
`route_only`, lost to `01_baseline3k` by a paired mean of `-2,187.77` over 100
development games. Adding replay-derived field guards, purchase recovery, sale
caps, or its original front-run controller did not produce a positive tuple.
The route evidence and safe-handoff work below remain useful analysis, but the
shop classifier and medoid routes are not the strategy in the current
`02_inspect_top1/main.py`.

The transferable lesson was narrower: top agents use stable production routes
plus discrete, shop-aware sale timing. The promoted agent therefore preserves
`01_baseline3k`'s complete 720-step schedule, field actions, purchases, hires,
land orders, planting, and animals. Its only enabled change is adaptive sale
front-running for `MELON`, `MILK`, `STRAWBERRY`, and `WOOL`.

## Compatible evidence

The inspector validates all 100 downloaded top-100 replays before selecting
route evidence. Ninety replays use Kaggriculture 1.32.7 and are accepted; ten
otherwise-valid Kaggriculture 1.32.6 replays are recorded as version
mismatches and excluded. Among the compatible files, 69 contain `カワシギ`.

Route selection uses normalized field and market timelines. Observation-proven
weed digs use a comparison sentinel, and only `SELL` quantities are removed;
purchase types and quantities remain exact. The selected 72-step opening
fingerprint is
`c860b6d9f00fed8c2fefc1300666263f0debab4074dc47ee75cb5f2b779d5f3f`.
Its coherent 35-replay family contains every supported 42-strawberry/12-melon
livestock branch:

| Branch | Family records | Medoid source |
|---|---:|---|
| 10 cows / 4 sheep | 23 | `93232089.json` |
| 8 cows / 6 sheep | 4 | `93316226.json` |
| 6 cows / 8 sheep | 4 | `93339617.json` |
| 6 cows / 12 sheep | 4 | `93399364.json` |

The other opening family is excluded: routes are never combined merely
because their season-total purchases match. Each branch medoid minimizes total
field-plus-market disagreement within its branch, with source filename as the
deterministic tie-breaker.

## Proven route handoffs

The generated schema-v2 route contains complete action traces for all four
medoids. Before a non-default trace can be emitted, the inspector proves that
all route-defining field actions before the decision are identical, cumulative
purchase attempts before the decision are identical, and canonical live farm
state at the decision observation is identical. Canonical state contains the
farmer position, ordered hand positions, unlocked quadrants, and row-major
tile kind/crop/animal descriptions. An unsafe required handoff raises
`ReplayError`; unsafe trace splicing is not permitted.

The 6-cow/12-sheep route is safe at step 144. The 6-cow/8-sheep and
8-cow/6-sheep routes are safe against the 10-cow/4-sheep default at step 216.
For those two later branches, the first normalized market-timeline difference
is step 195, but cumulative route-defining purchases and canonical farm state
are equal by step 216; all field prefixes are identical. This is why the
handoff proof compares purchase state in addition to reporting the first
market timing difference.

## Investigated shop classifier (not promoted)

The runtime selector uses only the live `town.unlocked_shops` observation.
Duplicate shop instances are preserved in observations, while classification
tests shop presence. Branch selection freezes at step 216:

1. If Yarn Store appeared among the first two unlocks by step 144, select
   6 cows / 12 sheep.
2. Otherwise, if Yarn Store first appears by step 216, select
   6 cows / 8 sheep.
3. Otherwise, if Pizza Shop, Ice Cream Shop, or Smoothie Shop is present,
   select the default 10 cows / 4 sheep.
4. Otherwise, select 8 cows / 6 sheep.

Missing or malformed town data falls back to the 10-cow/4-sheep route. Once
frozen, the branch does not change later in the episode.

## Artifact guarantees and limits

`replay_analysis.json` records the accepted filenames, rejected version
records, target summaries, selected family, branch counts, and medoid sources.
`canonical_route.json` records selector metadata, explainable handoff reports,
and the four complete routes with their weed annotations. Both files are
written with sorted keys and stable indentation and are verified by generating
them twice and comparing bytes.

The artifacts prove compatibility within the selected Kaggriculture 1.32.7
opening family. They deliberately do not claim that the ten 1.32.6 traces or
the other opening family can be replayed or spliced safely. Competitive
promotion still requires the frozen paired-seed qualification and holdout
gates described by the project plan.

## Promoted adaptive market overlay

At each decision, the overlay inspects the next four baseline actions. When a
future action sells a premium item, it may move only the currently available
quantity into the present action. It reserves stock for same-turn pickups and
existing sales, records the exact moved quantity as debt, and subtracts that
debt from the original future sale exactly once. It never looks beyond four
steps, never front-runs a non-premium product, never crosses an observed shop
demand boundary, and never exceeds the ten-market-order limit. Mutable debt is
isolated by player seat and resets when a new episode starts.

Demand deferral was tested separately and rejected. Waiting for the next
premium demand tick lost by a paired mean of `-84,609.6` in the screen because
the opponent could sell first into the shared market. The promoted flags are
therefore exactly:

```text
_ENABLE_DEMAND_DEFERRAL = False
_ENABLE_ADAPTIVE_FRONT_RUN = True
```

The screen on seeds `0..9` in both seats produced 17 wins from 20 games, a
paired mean of `+881.7`, and a paired median of `+805.0`. The exact frozen bytes
then ran once on fresh seeds `50..99` in both seats. The 100-game confirmation
finished 88-12 with paired mean `+866.54`, paired median `+785.0`, deterministic
95% paired bootstrap interval `[+769.28, +973.98]`, and positive means in both
candidate seats (`+919.22` and `+813.86`). Every game finished `DONE`, every
reward equalled final money, and every paired seed was positive.

The promoted standalone `main.py` is bound to SHA-256
`ce87aeabfe0141cfda004ee8f78ca272570766d20039b004f31d8c034ba12d06`.
Its 720-call replay profile measured `0.173089 ms` mean and `0.2085 ms` p95 per
decision. It has no runtime replay or repository-file dependency.

## Limits of the evidence

The promotion proves an uplift against `01_baseline3k` on Kaggriculture
`1.32.7` under the fixed paired-seat screen and fresh confirmation panels. It
does not prove dominance against every competitor, and the market assumptions
should be revalidated after an environment or rules change. The overlay does
not dynamically alter production, crop layout, livestock mix, or purchases;
those remain the verified baseline route.
