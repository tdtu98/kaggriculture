# Baseline3k Live Market Overlay Design

**Date:** 2026-08-17

**Status:** Proposed for implementation planning

## Objective

Test whether a narrowly scoped, live market controller can improve
`01_baseline3k` without changing its proven field movement, planting,
livestock, purchases, land, or hiring schedule. The experiment learns sale
timing principles from the top-100 replays; it does not replay their field
routes.

`duy/another_work/01_baseline3k/main.py` remains byte-for-byte unchanged.
`duy/another_work/02_inspect_top1/main.py` is replaced only if a standalone
candidate passes the complete promotion gate.

## Evidence and Hypothesis

The previous top-100 route candidate lost to `01_baseline3k` by an average of
2,187.77 coins across 50 paired seeds even with all optional controllers
disabled. Diagnostic games show why: it commonly uses fewer productive crop
tiles, retains unused seeds or unplaced animals, buys excessive wheat, and
accumulates more weeds. Copying the route is therefore rejected.

The transferable replay pattern is narrower. Strong replays coordinate
premium-product sales with recurring town demand and submit large, batched
liquidations. The market consumes shop products every four turns, prices are
observable, and competing sales affect the same shared inventory. The testable
hypothesis is that bounded, live sale timing can improve `01_baseline3k` while
leaving its production system intact.

## Experimental Architecture

Build an isolated candidate from the exact `01_baseline3k` source. Its decoded
720-step action schedule and existing weed repair remain unchanged. The only
new code is a market overlay applied after the baseline action and before hand
alignment.

Two independently switchable policies may be screened:

1. **Demand deferral.** A scheduled premium sale may be delayed for at most
   four turns when its live price is depressed and an observed town-demand
   tick for that product will occur during the delay. Deferred quantities are
   capped by live, unreserved shed stock. They are retried immediately after
   demand, or at the deadline, and forcibly flushed near season end.
2. **Adaptive front-running.** Generalize the baseline's one-turn premium
   front-run to a bounded lookahead of at most four turns. Move only live,
   unreserved stock, never cross an intervening demand tick, and record exact
   debt so the corresponding future scheduled sale is reduced rather than
   duplicated.

Premium products are `MELON`, `MILK`, `STRAWBERRY`, and `WOOL`. Fertilizer and
wheat are excluded from this first experiment because they are feed/capacity
resources and follow materially different price curves.

Both policies must preserve the ten-market-order limit. Per-seat state resets
at step zero or when steps move backwards. Malformed observations fall back to
the unchanged baseline action.

## Correctness Invariants

- Never alter farmer or hand actions.
- Never alter purchases, hires, or land orders.
- Never sell more than live shed stock minus scheduled pickup reserves.
- Never duplicate a moved or deferred quantity.
- Never carry sale debt across games or seats.
- Never emit more than ten market orders.
- Flush pending deferred quantities by the final safe sale window.
- The standalone candidate must not read repository or replay files at
  runtime.

## Tests

Use test-driven development for:

- Exact action parity with `01_baseline3k` when both policies are disabled.
- Farmer/hand and non-sale market-order parity when either policy is enabled.
- Demand recognition with duplicate shops.
- Deferral, deadline retry, end-of-season flush, and state reset.
- Multi-step front-run debt creation and exact future repayment.
- Live-stock and pickup-reserve caps.
- Per-seat isolation, ten-order capacity, malformed observations, and aligned
  fallback actions.
- Standalone import and mean/p95 decision latency below 1 ms/2 ms.

## Benchmark Protocol and Promotion Gate

Use `01_baseline3k` as the only opponent.

1. Screen the control and each isolated policy on development seeds `0..9`,
   both seats (20 games per tuple). A policy advances only with no invalid
   games and a positive paired mean.
2. Run only the best advancing tuple on fresh seeds `50..99`, both seats:
   exactly 100 games of 720 turns.
3. Promote only if all games finish `DONE`, terminal reward equals money, and
   all of the following are positive from the candidate's perspective:
   paired mean, paired median, both seat means, and the lower bound of the
   deterministic 95% paired bootstrap interval. Win rate must exceed 55%.
4. If no tuple passes, preserve the existing `02_inspect_top1/main.py` and
   report the no-winner result. Do not weaken the gate after seeing results.

The fresh 100-game panel is the evidence for real uplift. Development results
select a tuple but cannot promote it.

## Deliverables

- Isolated candidate and focused tests.
- Deterministic screening and 100-game benchmark artifacts.
- Updated strategy/benchmark findings explaining the accepted or rejected
  policy.
- `02_inspect_top1/main.py` changed only by mechanically installing a verified
  winner.
- After the overall feature is integrated, remove the temporary worktree and
  return development focus to `main`, as requested.
