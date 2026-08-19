# Top-100 shop-adaptive development findings

## Baseline3k market-overlay screen (2026-08-17)

The second experiment kept `01_baseline3k`'s complete field route and tested
only two market decisions learned from the top-100 corpus. The 20-game screen
used seeds `0..9` with the candidate in both seats. All 60 games across the
three variants finished `DONE`, and every reported reward equalled final
money.

| Variant | Games (W-L-T) | Win rate | Paired mean | Paired median | Paired mean 95% CI | Seat 0 mean | Seat 1 mean | Decision |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `control` | 20 (5-5-10) | 25% | 0.0 | 0.0 | [0.0, 0.0] | +277.4 | -277.4 | Valid paired-seat control |
| `demand_defer` | 20 (0-20-0) | 0% | -84,609.6 | -89,094.0 | [-96,794.6, -69,040.3] | -85,046.1 | -84,173.1 | Rejected |
| `adaptive_front_run` | 20 (17-3-0) | 85% | **+881.7** | **+805.0** | **[+733.5, +1,081.3]** | **+1,159.8** | **+603.6** | Frozen for confirmation |

Demand deferral failed because the shared market lets the opponent sell first
and consume the premium price. The useful replay lesson is the opposite:
retain the strong baseline's production route, but sell premium inventory a
few steps before an observed demand boundary. The adaptive controller looks
ahead at most four steps and does not cross the currently observed boundary.

The screen froze `adaptive_front_run` at SHA-256
`ce87aeabfe0141cfda004ee8f78ca272570766d20039b004f31d8c034ba12d06`.
Its source candidate was
`4c9d0d88b37c24cf0b211b12d2719bed68d0e3bcfba9a6a4dc4597236d963594`;
the untouched `01_baseline3k` opponent was
`f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`.
No screen result is a promotion decision: the frozen artifact must still pass
the predeclared 100-game panel on fresh seeds `50..99` in both seats.

## Fresh 100-game confirmation

The frozen `adaptive_front_run` bytes passed every binding promotion gate on
the untouched confirmation panel. The evaluator ran seeds `50..99` once with
the candidate in both seats. All 100 games finished `DONE`; every reward
equalled final money; all 50 seed pairs were complete; and all 50 paired-seed
margins were positive.

| Metric | Confirmation result | Gate |
| --- | ---: | --- |
| Games | 100 | Exactly 100 |
| Wins / losses / ties | 88 / 12 / 0 | More than 55 wins |
| Win rate | 88% | Above 55% |
| Paired mean margin | **+866.54** | Positive |
| Paired median margin | **+785.0** | Positive |
| Paired mean 95% bootstrap CI | **[+769.28, +973.98]** | Lower bound positive |
| Candidate-seat-0 mean | **+919.22** | Positive |
| Candidate-seat-1 mean | **+813.86** | Positive |
| Worst / best paired seed | +339.0 / +1,955.0 | Informational |
| Decision latency mean | 0.173089 ms | Below 1 ms |
| Decision latency p95 | 0.2085 ms | Below 2 ms |
| Decision latency maximum | 0.454334 ms | Informational |

The deterministic 95% interval used 10,000 paired-seed bootstrap resamples
with seed `20260814`. The latency profile made 720 calls on compatible replay
`93232089.json` for team `カワシギ`. Python was `3.12.9`, and Kaggle
Environments/Kaggriculture was `1.32.7`.

The confirmed candidate SHA-256 is exactly the frozen screen digest:
`ce87aeabfe0141cfda004ee8f78ca272570766d20039b004f31d8c034ba12d06`.
The baseline and source-candidate digests remained
`f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`
and
`4c9d0d88b37c24cf0b211b12d2719bed68d0e3bcfba9a6a4dc4597236d963594`,
respectively. The promotion failure list was empty.

The binding confirmation command was:

```text
duy/.venv/bin/python duy/another_work/02_inspect_top1/evaluate_market_overlay.py --candidate duy/another_work/02_inspect_top1/market_candidate.py --baseline duy/another_work/01_baseline3k/main.py --output-dir duy/benchmarks/results/baseline3k-market-overlay/confirm --phase confirm --screening-json duy/benchmarks/results/baseline3k-market-overlay/screen/screening.json
```

This result authorizes mechanical promotion of the exact confirmed bytes. It
does not establish superiority against other opponents or future game-version
changes; the evidence is specifically against `01_baseline3k` on Kaggriculture
1.32.7.

## Earlier replay-route experiment

## Outcome

The development panel produced **no legal winner**. Every tested candidate had
a negative paired mean margin against `01_baseline3k`, so every tuple was
rejected by the binding selection rule. The existing `candidate_main.py` was
preserved byte-for-byte; no tuple was frozen, no Task 6 latency profile was
created, and confirmation seeds `1000..1099` were not spent. Task 7
confirmation must therefore be skipped.

The primary ranking by paired mean was `route_only` (-2,187.77), `front_run`
(-2,545.21), `field_guards` (-2,550.58), `sale_cap` (-4,883.98), then
`purchase_recovery` (-81,059.71). Secondary evidence did not change the
outcome because the primary non-negative requirement was not met.

## Protocol and identities

- Development seeds: `0..49`, both candidate seats, 100 games per tuple.
- Episode length: 720 steps.
- Opponent: `another_work/01_baseline3k/main.py` only.
- Source candidate SHA-256:
  `0fb6737b37e1cbfd6dc37f568dda325e750f019caac1bb63dc5776252985e81e`.
- Baseline SHA-256:
  `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`.
- Python: `3.12.9`.
- Kaggle Environments / Kaggriculture module version: `1.32.7`.
- Bootstrap interval: paired-seed mean, 95%, 10,000 resamples, bootstrap seed
  `20260814`.
- Environment preflight: `pip check` reported no broken requirements and
  `git diff -- another_work/01_baseline3k/main.py` was empty.

Flag tuples below are ordered `(field_guards, purchase_recovery, sale_cap,
front_run)`.

## Base ablations

| Variant | Flags | Rendered SHA-256 | Games (W-L-T) | Win rate | Paired mean | Paired median | Paired mean 95% CI | Seat 0 mean | Seat 1 mean | Worst paired seed | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `route_only` | `(F,F,F,F)` | `97b346ea03e294963be84eaa926d04452977b2762d26afc0d488017982685ec2` | 100 (37-63-0) | 37% | -2,187.77 | -3,811.0 | [-4,535.25, 204.48] | -2,418.66 | -1,956.88 | -18,628.0 | Rejected: negative paired mean |
| `field_guards` | `(T,F,F,F)` | `0fb6737b37e1cbfd6dc37f568dda325e750f019caac1bb63dc5776252985e81e` | 100 (38-62-0) | 38% | -2,550.58 | -2,085.5 | [-5,282.95, 35.15] | -2,526.52 | -2,574.64 | -30,720.0 | Rejected: negative paired mean; both seats regress vs route-only |
| `purchase_recovery` | `(T,T,F,F)` | `912c867d2700fc0e015294b0f270c0af7f3052fa00bcaf91cf29d4519ce19ffd` | 100 (0-100-0) | 0% | -81,059.71 | -76,467.0 | [-86,874.89, -75,322.44] | -80,175.98 | -81,943.44 | -123,951.0 | Rejected; does not qualify for combinations |
| `sale_cap` | `(T,F,T,F)` | `5ff3ace202de1eec5ccbdf92299b5eb8be83268cb0ceffd8240b3fd45f83da47` | 100 (30-70-0) | 30% | -4,883.98 | -5,496.5 | [-7,435.60, -2,445.57] | -4,828.76 | -4,939.20 | -31,046.5 | Rejected; does not qualify for combinations |
| `front_run` | `(T,F,F,T)` | `e9c974064afe66b4b5062b2c64427497da4cdd2a30fafaed1e255e4f2c01253c` | 100 (38-62-0) | 38% | -2,545.21 | -2,085.5 | [-5,278.72, 44.55] | -2,523.76 | -2,566.66 | -30,720.0 | Rejected overall; qualifies for singleton combination test |

Raw base evidence:

- [`route_only`](../../benchmarks/results/top100-shop-adaptive/development/base/route_only/)
- [`field_guards`](../../benchmarks/results/top100-shop-adaptive/development/base/field_guards/)
- [`purchase_recovery`](../../benchmarks/results/top100-shop-adaptive/development/base/purchase_recovery/)
- [`sale_cap`](../../benchmarks/results/top100-shop-adaptive/development/base/sale_cap/)
- [`front_run`](../../benchmarks/results/top100-shop-adaptive/development/base/front_run/)
- [`base/ablations.json`](../../benchmarks/results/top100-shop-adaptive/development/base/ablations.json)

## Controller qualification and combination evidence

An optional controller qualified for combination testing only when its
isolated tuple improved paired mean over `field_guards` (-2,550.58) and did
not worsen both seat means (-2,526.52 and -2,574.64).

- `purchase_recovery` failed: paired mean delta -78,509.13 and both seats
  worsened.
- `sale_cap` failed: paired mean delta -2,333.40 and both seats worsened.
- `front_run` qualified narrowly: paired mean delta +5.37, seat 0 delta
  +2.76, and seat 1 delta +7.98.

Only the non-empty singleton `{front_run}` qualified. Its exact tuple was
already named in evaluator `VARIANTS`, so no evaluator or test change was
needed. It was run once under the separate combination root and exactly
reproduced its base evidence:

| Combination | Flags | Rendered SHA-256 | Games (W-L-T) | Win rate | Paired mean | Paired median | Paired mean 95% CI | Seat 0 mean | Seat 1 mean | Worst paired seed | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `front_run` | `(T,F,F,T)` | `e9c974064afe66b4b5062b2c64427497da4cdd2a30fafaed1e255e4f2c01253c` | 100 (38-62-0) | 38% | -2,545.21 | -2,085.5 | [-5,278.72, 44.55] | -2,523.76 | -2,566.66 | -30,720.0 | Rejected: negative paired mean |

Raw combination evidence:

- [`combinations/front_run`](../../benchmarks/results/top100-shop-adaptive/development/combinations/front_run/)
- [`combinations/ablations.json`](../../benchmarks/results/top100-shop-adaptive/development/combinations/ablations.json)

## Freeze and latency decision

There is no finalist and therefore no frozen Task 6 candidate digest. The
source candidate remains at
`0fb6737b37e1cbfd6dc37f568dda325e750f019caac1bb63dc5776252985e81e`
with its pre-existing `(T,F,F,F)` flags only because the preserve-current-on-
failure rule forbids freezing a rejected tuple.

Task 5 latency evidence for this unchanged source candidate passed on replay
`93232089` for team `カワシギ`: 720 calls, mean `0.13229870833333335 ms`,
p95 `0.151 ms` (thresholds: mean below 1.0 ms and p95 below 2.0 ms). That
historical profile is not a frozen-finalist profile and was not promoted to
Task 6 evidence. No replay or latency artifact was added to the development
result tree.
