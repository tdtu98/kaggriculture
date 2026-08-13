# Decoded Legacy Actions: Schedule and Strategy Analysis

This document explains the decoded `_LEGACY_ACTIONS` table from `main.py`.
The complete machine-readable table is in `legacy_actions.json`.

## Artifact Fidelity

- JSON entries: **719**, representing action steps 0 through 718
- JSON size: **305,157 bytes**
- JSON SHA-256:
  `d62e8f0a0b930ddfedd560d824c320c39ff3c947f914169ea22ca013a3f0ff84`
- Round-trip check: the saved JSON is semantically equal to
  `main._LEGACY_ACTIONS`
- Implementation finding: `_REBALANCE_ACTIONS is _LEGACY_ACTIONS`; there is
  no separate rebalance action table in this artifact

The JSON root is the original array. Array index `n` is step `n`; no metadata
was inserted into the actions.

## Action Schema

Every entry has the normal Kaggriculture action structure:

```json
{
  "farmer": ["PASS"],
  "hands": [],
  "market": [["HIRE"], ["BUY_SEED", "WHEAT", 5]]
}
```

- `farmer` contains exactly one main-farmer action.
- `hands` contains one action for each hand active at that step.
- `market` is an ordered list of market operations.

Day 29 contains only steps 696 through 718 because a 720-frame episode needs
719 submitted action transitions. The wrapper also clamps any unexpectedly
larger observation step to the table's last entry.

## Headline Findings

1. **This is a scripted route, not a general farm planner.** Most movement,
   planting, watering, animal work, and selling is fixed by step number.
2. **Labor is the primary multiplier.** The table schedules 262 hires, rising
   from four hands on day 0 to frequent 10–14-hand days.
3. **Animals finance and fertilize the farm.** The base mix is nine cows and
   four sheep in thirteen pastures, supported by 212 purchased units of wheat.
4. **Only three crops are used.** The route plants wheat, strawberry, and
   melon; it completely skips carrot and tomato.
5. **Land is expanded twice.** Purchases occur at steps 160 and 240.
6. **The route changes from setup to throughput.** Early days build livestock
   and premium crops; later days use large labor teams for wheat, harvesting,
   fertilizing, and liquidation.
7. **The endgame deliberately stops long-horizon work.** Care falls on day 27
   and reaches zero on days 28–29; planting stops after day 26, while harvesting
   and selling accelerate.
8. **The raw schedule is only the base layer.** Weed repair, the Yarn Store
   livestock swap, wool release, sell ranking, and Market Relay may alter the
   action returned in a real match.

## Aggregate Operation Counts

These counts include the farmer, all hands, and market orders across the raw
719-step table.

| Operation | Scheduled count |
|---|---:|
| WATER | 1,010 |
| PASS | 994 |
| NORTH | 838 |
| WEST | 811 |
| EAST | 634 |
| SOUTH | 553 |
| HARVEST | 390 |
| COLLECT_FERTILIZER | 296 |
| FEED | 290 |
| CARE | 285 |
| HIRE | 262 |
| PLANT | 199 |
| SELL | 168 |
| BUY_PRODUCT | 140 |
| PICKUP | 135 |
| FERTILIZE | 72 |
| BUY_SEED | 63 |
| DROP | 57 |
| DIG | 40 |
| BUILD_PASTURE | 13 |
| PLACE | 13 |
| BUY_ANIMAL | 7 |
| BUY_LAND | 2 |

The maximum table width is 14 hand actions in one step and 10 market orders in
one step. Large hiring days are split across the first two steps of the day so
the route stays within the market-order cap.

### Who performs the work?

Hands perform most production actions:

- 951 of 1,010 water actions
- 369 of 390 harvest actions
- 232 of 290 feed actions
- 230 of 296 fertilizer collections
- 219 of 285 care actions
- 190 of 199 plant actions
- all 72 fertilize actions

The main farmer contributes more heavily to animal care, fertilizer collection,
pickup, and route coordination, while hired hands provide the field throughput.

## Purchases and Production Mix

| Category | Item | Scheduled quantity |
|---|---|---:|
| Hires | Farm hand | 262 |
| Animals | Cow | 9 |
| Animals | Sheep | 4 |
| Seeds | Wheat | 148 |
| Seeds | Strawberry | 37 |
| Seeds | Melon | 19 |
| Product | Wheat | 212 |
| Land | Quadrant purchase | 2 |

The 212 units of bought wheat are feed insurance. The route does not rely on
its own harvest timing to keep thirteen animals alive.

### Planting totals

| Crop | Plant actions | First plant step | Last plant step |
|---|---:|---:|---:|
| Wheat | 143 | 7 | 646 |
| Strawberry | 37 | 77 | 283 |
| Melon | 19 | 5 | 272 |

There are no carrot or tomato purchases or plantings. There are also no geese
or coops: all animal structures are pastures.

## Scheduled Sales

| Product | Sell orders | Total quantity | First sale | Last sale |
|---|---:|---:|---:|---:|
| Wheat | 39 | 455 | step 120 | step 718 |
| Strawberry | 19 | 286 | step 336 | step 672 |
| Milk | 32 | 241 | step 216 | step 716 |
| Fertilizer | 50 | 235 | step 48 | step 717 |
| Wool | 12 | 132 | step 160 | step 672 |
| Melon | 16 | 114 | step 252 | step 528 |

The raw schedule attempts to sell **1,463 product units** in total. Fertilizer
is the first recurring revenue stream, beginning on day 2. Wheat selling begins
on day 5, wool on day 6, milk on day 9, melon on day 10, and strawberry on day
14. This stagger matches the production lead time of each farm subsystem.

## Day-by-Day Schedule

Notation in the final column is:

```text
W = WATER, H = HARVEST, F = FEED, C = CARE,
CF = COLLECT_FERTILIZER, Ft = FERTILIZE
```

All values are scheduled actions, not guaranteed successful execution.

| Day | Hires | Purchases | Plantings | Sales | Land step | Field focus W/H/F/C/CF/Ft |
|---:|---:|---|---|---|---:|---|
| 0 | 4 | Cow×1; Sheep×4; Wheat seed×5; Melon seed×5; Wheat product×5 | Melon×5; Wheat×5 | — | — | W10/H0/F5/C5/CF0/Ft0 |
| 1 | 1 | — | — | — | — | W0/H0/F0/C5/CF5/Ft0 |
| 2 | 2 | Wheat product×11 | — | Fertilizer×5 | — | W10/H0/F5/C5/CF5/Ft0 |
| 3 | 3 | Wheat seed×1; Strawberry seed×3; Wheat product×5 | Strawberry×3; Wheat×1 | Fertilizer×5 | — | W9/H0/F5/C5/CF5/Ft0 |
| 4 | 3 | Wheat seed×5; Wheat product×2 | Wheat×5 | Fertilizer×5 | — | W15/H5/F5/C5/CF5/Ft0 |
| 5 | 3 | Cow×1; Wheat seed×1; Strawberry seed×4; Wheat product×7 | Strawberry×4; Wheat×1 | Wheat×17; Fertilizer×5 | — | W9/H0/F6/C6/CF5/Ft0 |
| 6 | 4 | Wheat product×5; Cow×2; Wheat seed×1; Strawberry seed×3 | Strawberry×2 | Fertilizer×8; Wool×5 | 160 | W12/H4/F7/C7/CF6/Ft0 |
| 7 | 7 | Cow×2; Wheat seed×3; Strawberry seed×9; Wheat product×14 | Strawberry×10; Wheat×4 | Wool×15; Fertilizer×3 | — | W33/H1/F10/C10/CF7/Ft0 |
| 8 | 6 | Cow×2; Wheat seed×8; Strawberry seed×2; Wheat product×7 | Wheat×8; Strawberry×2 | Fertilizer×7; Wheat×8 | — | W22/H6/F12/C12/CF10/Ft0 |
| 9 | 7 | Wheat seed×1; Wheat product×10 | Wheat×1 | Milk×6; Fertilizer×10; Wheat×7 | — | W28/H5/F12/C12/CF12/Ft0 |
| 10 | 14 | Melon seed×14; Strawberry seed×16; Wheat product×12; Wheat seed×1 | Melon×12; Strawberry×11; Wheat×1 | Wool×16; Wheat×2; Melon×30 | 240 | W43/H6/F12/C12/CF12/Ft0 |
| 11 | 10 | Wheat seed×4; Wheat product×8 | Strawberry×5; Melon×2; Wheat×4 | Milk×3; Fertilizer×12; Wheat×2 | — | W41/H4/F12/C12/CF12/Ft0 |
| 12 | 10 | Wheat seed×8; Wheat product×5 | Wheat×8 | Wheat×11; Fertilizer×9; Wool×4; Milk×3 | — | W46/H13/F12/C12/CF12/Ft3 |
| 13 | 8 | Wheat seed×1; Wheat product×11 | Wheat×1 | Wheat×24; Wool×12; Fertilizer×12 | — | W27/H5/F12/C12/CF12/Ft0 |
| 14 | 9 | Wheat product×12 | — | Milk×6; Strawberry×6; Wheat×3; Fertilizer×8 | — | W45/H2/F12/C12/CF12/Ft4 |
| 15 | 9 | Cow×1; Wheat product×5; Wheat seed×4 | Wheat×4 | Milk×24; Fertilizer×9; Wool×8 | — | W35/H18/F11/C12/CF10/Ft1 |
| 16 | 13 | Wheat seed×11; Wheat product×5 | Wheat×8 | Strawberry×14; Wheat×16; Wool×8; Milk×21 | — | W59/H14/F12/C12/CF12/Ft13 |
| 17 | 9 | Wheat product×11; Wheat seed×1 | Wheat×1 | Milk×3; Wheat×19; Strawberry×2; Fertilizer×9 | — | W28/H18/F12/C12/CF12/Ft2 |
| 18 | 11 | Wheat product×14 | — | Strawberry×28; Milk×18; Wheat×5; Fertilizer×8 | — | W54/H14/F12/C12/CF12/Ft4 |
| 19 | 13 | Wheat product×7; Wheat seed×4 | Wheat×7 | Wool×16; Strawberry×26; Milk×3; Wheat×8 | — | W53/H25/F12/C12/CF12/Ft12 |
| 20 | 14 | Wheat seed×49; Wheat product×11 | Wheat×14 | Milk×23; Strawberry×20; Wheat×15; Melon×60 | — | W49/H30/F12/C11/CF11/Ft13 |
| 21 | 12 | Wheat seed×7; Wheat product×9 | Wheat×10 | Melon×12; Strawberry×28; Milk×15; Wheat×7; Wool×16; Fertilizer×7 | — | W36/H33/F11/C13/CF11/Ft2 |
| 22 | 10 | Wheat product×13 | Wheat×3 | Strawberry×34; Melon×12; Fertilizer×11 | — | W38/H17/F13/C13/CF13/Ft2 |
| 23 | 14 | Wheat product×7; Wheat seed×7 | Wheat×18 | Strawberry×38; Milk×15; Fertilizer×8; Wheat×3 | — | W60/H24/F12/C13/CF13/Ft11 |
| 24 | 11 | Wheat seed×11 | Wheat×12 | Strawberry×32; Wheat×24; Fertilizer×11; Milk×11 | — | W41/H30/F12/C12/CF12/Ft3 |
| 25 | 12 | Wheat seed×12; Wheat product×4 | Wheat×13 | Strawberry×10; Wheat×31; Milk×14; Wool×16; Fertilizer×13 | — | W54/H22/F13/C13/CF13/Ft0 |
| 26 | 12 | Wheat product×4; Wheat seed×3 | Wheat×14 | Strawberry×16; Wheat×30; Milk×5; Fertilizer×17 | — | W50/H17/F13/C13/CF13/Ft2 |
| 27 | 11 | Wheat product×6 | — | Strawberry×26; Milk×18; Fertilizer×15; Wheat×11 | — | W38/H30/F13/C5/CF13/Ft0 |
| 28 | 10 | Wheat product×2 | — | Strawberry×6; Wheat×47; Milk×17; Fertilizer×18; Wool×16 | — | W39/H16/F5/C0/CF13/Ft0 |
| 29 | 10 | — | — | Wheat×165; Fertilizer×20; Milk×36 | — | W26/H31/F0/C0/CF6/Ft0 |

## Strategy Phases

### Days 0–6: build the animal engine

Step 0 spends aggressively on four hands, one cow, four sheep, five wheat
seeds, five melon seeds, and five feed wheat. The route establishes five
pastures and plants its first wheat and melons. Fertilizer sales start on day 2,
providing early cash before premium crops mature.

Strawberries enter on day 3. A second cow arrives on day 5, followed by two
more on day 6. The first land purchase at step 160 coincides with the first wool
sale, showing that animal revenue helps finance expansion.

### Days 7–11: scale livestock, premium crops, and land

The route expands rapidly to twelve animals, then thirteen later in the season.
It increases labor from 7 to 14 hands, buys the second land quadrant at step
240, and launches the largest melon/strawberry planting wave.

Milk selling begins on day 9. Day 10 is the capital-expansion peak: fourteen
hires, two large seed groups, the second land purchase, and the first large
melon liquidation.

### Days 12–19: operate the mixed-production machine

The farm now maintains roughly twelve animals daily while cycling wheat and
harvesting ongoing strawberries. Fertilization begins on day 12 and becomes
heavy on days 16 and 19. Labor fluctuates with the harvest workload rather than
remaining fixed.

Sales diversify across wheat, strawberry, milk, wool, and fertilizer. Melon is
temporarily absent from the sales table because the next large melon harvest is
still maturing.

### Days 20–26: convert land and labor into wheat throughput

Day 20 buys 49 wheat seeds and plants fourteen wheat tiles. From this point,
harvesting and replanting dominate. Melons are liquidated on days 20–22, while
strawberry, milk, wheat, and fertilizer provide continuous cash flow.

This phase uses 10–14 hands per day. The cost is justified only if the scripted
route stays aligned and the large number of field actions succeeds.

### Days 27–29: stop investing and liquidate

No new crops are planted after day 26. Animal care drops from thirteen actions
on day 26 to five on day 27 and zero thereafter because care bonuses no longer
have enough time to pay back. Feeding also ends on day 29.

Workers concentrate on watering the last useful crops, harvesting, collecting
remaining fertilizer, and returning inventory. Day 29 schedules sales of 165
wheat, 36 milk, and 20 fertilizer.

## Representative Steps

- **Step 0:** four hires plus all initial animals, seeds, and feed; farmer
  passes because the new hands and inventory are not available until processing.
- **Step 48:** sells five fertilizer, hires two hands, and buys six feed wheat.
- **Step 160:** sells wool and fertilizer while buying the first land quadrant.
- **Step 240:** fills all ten market slots with wool/wheat sales, land, and
  seven hires; the other seven hires are placed at step 241.
- **Step 281:** the base table sells twelve fertilizer. The Market Relay may
  move up to this quantity to step 278 and subtract the executed amount here.
- **Step 718:** nine hands pass, one hand drops inventory, and the market sells
  the last seven scheduled wheat.

## How Runtime Controllers Change the Raw Table

`legacy_actions.json` is the base schedule before these wrappers execute:

1. **Weed repair** can replace `PLANT` or `BUILD_PASTURE` with `DIG`, retry the
   intended operation, and replay delayed actions.
2. **Yarn Store conversion** can turn the two cows bought at step 192 into
   sheep and update the matching placement actions.
3. **Wool control** can append inventory- and price-aware wool sales on the
   Yarn Store route.
4. **Sell ranking** can reorder sell orders based on estimated price impact
   and town consumption.
5. **Market Relay** can move recurring fertilizer sales three turns earlier
   when both farms remain near-mirrors at all route checkpoints.
6. **Hand alignment** pads or truncates the table's hand list to match the
   actual number of hired hands.

Therefore the decoded table explains the planned economic route, while a replay
is still required to know which actions actually executed and what products
were ultimately produced or sold.

## Risks and Limitations

- Money, inventory, and tile assumptions are mostly implicit. If a purchase,
  pickup, placement, or planting fails, later scheduled actions may drift.
- The route depends heavily on costly daily hiring and successful worker paths.
- It buys feed wheat often, exposing the strategy to insufficient cash if early
  revenue is disrupted.
- It has no general response to an opponent other than public near-mirror
  detection and shared-market timing.
- Scheduled sale totals do not equal guaranteed revenue because market prices
  change, invalid orders are silent no-ops, and runtime controllers can alter
  order timing or quantity.
- The public `agent(obs)` hardcodes a configuration with
  `townCenterSellInterval = 24`; the repository guide currently describes a
  default of 12.

## Practical Interpretation

The core strategy is best summarized as:

> Use animals to create early fertilizer and premium products, spend heavily on
> labor to keep a deterministic field route moving, expand twice, transition
> into wheat-heavy throughput, and liquidate before the season ends. Add narrow
> runtime corrections only where randomness or shared-market timing threatens
> the script.

This explains why the agent crushes the simple `demo_agent`: the decoded route
uses hundreds of hires and thousands of coordinated field actions, whereas the
demo baseline operates only one farmer in a small wheat loop.
