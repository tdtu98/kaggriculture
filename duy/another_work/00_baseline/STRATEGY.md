# Boatlee V16-RC2 Market Relay Strategy

This document explains the strategy extracted from
`v16-rc2-high-score-near-mirror-market-relay.ipynb`. The executable agent is
available beside it as `main.py`.

## Strategy at a Glance

The agent follows a mostly prerecorded high-output route:

```text
early livestock and melons
    -> strawberries and land expansion
    -> large daily labor force
    -> wheat-heavy mixed production
    -> market-aware product releases
    -> final liquidation
```

Its main competitive idea is to build production capacity quickly, keep many
workers active, and make small market-timing adjustments when the opponent is
following an almost identical route.

## Phase 1: Establish Livestock and Premium Crops

At step 0, the base route schedules:

- Four farm hands
- One cow
- Four sheep
- Five wheat seeds
- Five melon seeds
- Five units of wheat for animal feed

It builds pastures immediately, places the animals, and plants five melons and
five wheat crops during the first day. Animals are central to the route because
they generate milk or wool while also supplying fertilizer.

Wheat has two roles: it is a sale crop and the daily feed reserve. The agent
buys wheat from the market throughout the season when the scripted route needs
additional feed.

## Phase 2: Add Strawberries and Expand Land

Strawberry planting begins on day 3. The farm then uses three complementary
crop types:

- **Melon:** delayed premium harvests
- **Strawberry:** recurring premium production
- **Wheat:** fast crop, sale inventory, and animal feed

The route buys land twice:

- Step 160 — day 6, hour 16
- Step 240 — day 10, hour 0

This expands the operation to three unlocked quadrants. Around the second land
purchase, the route performs a large melon-and-strawberry planting wave.

## Phase 3: Scale Labor and Maintain Animals

Labor is the engine of the strategy. The route schedules 262 `HIRE` orders
over the season. It starts with a small group, then commonly hires 8–14 hands
per day during the middle and late game.

The large workforce repeatedly performs:

- Watering
- Harvesting and replanting
- Feeding animals
- Caring for animals to bank production bonuses
- Collecting fertilizer
- Moving inventory to the shed

The embedded schedule contains 1,010 watering actions, 390 harvest actions,
290 feeding actions, 285 care actions, and 296 fertilizer-collection actions.
These are scheduled actions; random weeds or a diverged game state can make
individual orders invalid.

## Crop and Livestock Mix

The base schedule contains these purchases and plantings:

| Resource | Scheduled amount |
|---|---:|
| Wheat planted | 143 |
| Strawberry planted | 37 |
| Melon planted | 19 |
| Cows purchased | 9 |
| Sheep purchased | 4 |
| Pastures built | 13 |

It does not use carrots, tomatoes, geese, or coops. The specialization keeps
the route focused on wheat, premium crops, milk, wool, and fertilizer.

## Conditional Yarn Store Route

After step 161, the agent checks the public list of unlocked shops. It selects
the wool-oriented route when:

- `YARN_STORE` is unlocked; and
- neither `PIZZA_SHOP` nor `SMOOTHIE_SHOP` is unlocked.

When selected, the route changes the two cows purchased at step 192 into two
sheep and also changes their matching placement actions. The resulting planned
mix is approximately:

- Normal route: 9 cows and 4 sheep
- Yarn Store route: 7 cows and 6 sheep

The choice is cached for the rest of the match rather than recalculated every
turn.

## Weed Repair

Random weeds can break a prerecorded route when a worker is supposed to plant
or build a pasture. If the target tile contains a weed, the controller:

1. Replaces the intended action with `DIG`.
2. Retries the original planting or building action on the next step.
3. Replays up to eight delayed actions for that worker.

This is a narrow recovery mechanism. It protects important setup actions but
does not fully replan the farm.

## Fertilizer Strategy

Animals create a steady fertilizer stream. The base schedule contains 72
`FERTILIZE` actions, using some fertilizer to improve crop output and selling
the surplus.

The scheduled market table sells 235 fertilizer over the season. In a close
matchup, the Market Relay can move some of those sales earlier without changing
the total intended quantity.

## Wool Release Controller

On the Yarn Store route, wool is not simply sold whenever it appears. The
controller waits for acceptable prices, with thresholds that fall as the end
of the season approaches:

| Step range | Minimum wool price |
|---|---:|
| Before 480 | 170 |
| 480–599 | 120 |
| 600–671 | 80 |
| 672–718 | 1 |

Normal sales are limited to batches of 16 and spaced by at least six turns.
The agent can sell earlier when total shed inventory reaches 78, reducing the
risk of overflow. From step 713 onward, it attempts to sell all remaining
wool.

## Same-Turn Market Ordering

When an action contains multiple `SELL` orders, the agent estimates how much
each batch will lower its product's market price:

```text
estimated impact = quantity * max(0, current price - post-sale price)
```

It also estimates consumption by unlocked town shops. Products with larger
price impact or more urgent oversupply are placed earlier in the existing sell
slots. Non-sell orders keep their original positions.

## Near-Mirror Market Relay

The relay targets opponents whose farms appear to be following almost the same
production route. It compares both public farms at steps 216, 240, and 264.

The comparison includes:

- Number of farm hands
- Number of unlocked quadrants
- Counts of each crop
- Counts of cows, sheep, and geese
- Counts of pastures, coops, and weeds

The distance calculation is:

```text
absolute hand difference
+ 3 * absolute unlocked-quadrant difference
+ sum of absolute tile-category count differences
```

The relay activates only if the distance is at most 8 at all three checkpoints.
If any checkpoint fails, the base sale schedule remains unchanged.

Once active, the controller examines fertilizer sales scheduled three turns in
the future. Between steps 278 and 662, it can:

1. Find the fertilizer quantity scheduled at `t + 3`.
2. Sell up to that quantity at step `t`, limited by actual shed inventory.
3. Record the executed quantity as debt.
4. Remove exactly that quantity from the original order at `t + 3`.

For example:

```text
Base route:   step 278 -> no sale      step 281 -> SELL 12 fertilizer
With relay:  step 278 -> SELL 12      step 281 -> remove 12 from sale
```

The purpose is to reach the shared market before a similar opponent and obtain
the pre-glut price. It changes timing, not the intended two-step quantity.

## Scheduled Sales and Endgame

The embedded base schedule contains the following total sale quantities:

| Product | Scheduled quantity |
|---|---:|
| Wheat | 455 |
| Strawberry | 286 |
| Milk | 241 |
| Fertilizer | 235 |
| Wool | 132 |
| Melon | 114 |

These are schedule totals, not guaranteed match results. The wool controller
can add or delay wool orders, and invalid field actions can change realized
production.

The late game becomes increasingly wheat-heavy. At the beginning of day 20,
the route buys 46 wheat seeds for a large final production cycle. Day 29 is
dominated by harvesting and liquidation, including a scheduled sale of 165
wheat.

## Strengths

- Very high action throughput from aggressive hiring
- Multiple income streams from crops, animals, and fertilizer
- Premium crop and shop specialization
- Basic recovery from random weed interference
- Market-aware order ranking
- A targeted timing advantage against near-mirror opponents

## Limitations

- The production route is prerecorded rather than generally planned from the
  current farm state.
- Observation-based responses are narrow: weeds, the Yarn Store branch, wool
  releases, sale ranking, and the fertilizer relay.
- The embedded action table has 719 entries. The agent clamps larger step
  values to the table's final action.
- The entry point hardcodes `townCenterSellInterval = 24`, while this
  repository's current game guide describes a default of 12. Confirm the
  competition configuration before relying on the reported evaluation.
- The notebook's large evaluation tables are embedded results; the notebook
  itself reruns only a single self-play smoke game.

## Executable Fidelity

`main.py` is extracted from notebook cell 12 with only the leading
`%%writefile main.py` directive removed.

- Size: 29,901 bytes
- SHA-256:
  `3c9b6e75d1bb9cc1f23b6bf5d8821c84193d1306d5bcb74ada1628359e3fb025`
- Public entry point: `agent(obs)`

To test it locally from the repository root:

```python
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run(["another_work/main.py", "random"])
print([(state.reward, state.status) for state in env.steps[-1]])
```

For a competition archive with `main.py` at the archive root:

```bash
tar -C another_work -czf submission.tar.gz main.py
```
