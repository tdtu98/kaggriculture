# Baseline3k Live Market Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and qualify a standalone 01_baseline3k-derived candidate whose only behavioral changes are bounded, live premium-sale timing policies.

**Architecture:** Mechanically seed an isolated candidate from the exact baseline source, keep its compressed action schedule and field/purchase code unchanged, and add two independently switchable market overlays. A focused evaluator screens each overlay, then runs only the best positive policy for 100 paired-seat games on fresh seeds before any promotion into 02_inspect_top1/main.py.

**Tech Stack:** Python 3.12.9, Kaggle Environments/Kaggriculture 1.32.7, standard-library unittest, the existing duy/benchmarks/benchmark.py runner, and Git.

## Global Constraints

- Keep duy/another_work/01_baseline3k/main.py byte-for-byte unchanged.
- Change duy/another_work/02_inspect_top1/main.py only after every promotion gate passes.
- The candidate must be standalone and must not read replay or repository files at runtime.
- Never alter baseline farmer actions, hand actions, purchases, hires, land orders, or the decoded 720-step schedule.
- Premium products are exactly MELON, MILK, STRAWBERRY, and WOOL.
- Preserve the ten-market-order limit and isolate mutable state by seat.
- Decision latency must remain below 1 ms mean and 2 ms p95.
- Screen on seeds 0..9 in both seats; confirm on fresh seeds 50..99 in both seats, exactly 100 games.
- Promotion requires DONE/reward validity, positive paired mean and median, positive means in both seats, win rate above 55%, and a positive deterministic 95% bootstrap lower bound.
- Preserve the existing 02 main if no candidate passes. Never weaken the gate after seeing results.
- Keep duy_explore and duy/.venv untracked and unmodified.
- After integration, remove the exact temporary worktree and return focus to main.

---

## File Structure

- Create duy/another_work/02_inspect_top1/market_candidate.py for the standalone runtime candidate.
- Create duy/another_work/02_inspect_top1/evaluate_market_overlay.py for variant rendering, screening, confirmation, and profiling.
- Create duy/test_baseline3k_market_overlay.py for parity, controller, evaluator, and standalone tests.
- Modify STRATEGY.md and BENCHMARK_FINDINGS.md with evidence and the binding decision.
- Conditionally create duy/another_work/02_inspect_top1/market_overlay_promotion.json as the small, committed identity record for a verified winner.
- Conditionally replace 02_inspect_top1/main.py with exact confirmed bytes.
- Write raw artifacts under duy/benchmarks/results/baseline3k-market-overlay and do not stage the raw result tree.

---

### Task 1: Seed the Candidate and Prove Baseline Parity

**Files:**
- Create: duy/another_work/02_inspect_top1/market_candidate.py
- Create: duy/test_baseline3k_market_overlay.py

**Interfaces:**
- Consumes: baseline _ACTIONS and agent(obs).
- Produces: _ENABLE_DEMAND_DEFERRAL, _ENABLE_ADAPTIVE_FRONT_RUN, and agent(obs) with exact baseline behavior when both flags are false.

- [ ] **Step 1: Write failing parity tests**

Add a unique-path module loader and these tests:

    class CandidateParityTests(unittest.TestCase):
        def test_candidate_embeds_exact_baseline_schedule(self):
            baseline = load_agent(BASELINE_PATH, "overlay_baseline")
            candidate = load_agent(CANDIDATE_PATH, "overlay_candidate")
            self.assertEqual(candidate._ACTIONS, baseline._ACTIONS)
            self.assertFalse(candidate._ENABLE_DEMAND_DEFERRAL)
            self.assertFalse(candidate._ENABLE_ADAPTIVE_FRONT_RUN)

    def test_disabled_candidate_matches_720_replay_observations(self):
            baseline = load_agent(BASELINE_PATH, "overlay_baseline_replay")
            candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_replay")
            replay = json.loads(PROFILE_REPLAY_PATH.read_text())
            seat = replay["info"]["TeamNames"].index("カワシギ")
            for states in replay["steps"]:
                obs = states[seat]["observation"]
                self.assertEqual(candidate.agent(obs), baseline.agent(obs))

    def test_malformed_observation_preserves_baseline_fallback_shape(self):
        baseline = load_agent(BASELINE_PATH, "overlay_baseline_bad_obs")
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_bad_obs")
        malformed = {"step": "not-an-integer", "farms": [{}, {}]}
        self.assertEqual(candidate.agent(malformed), baseline.agent(malformed))

    def test_candidate_call_has_no_runtime_file_dependency(self):
        candidate = load_agent(CANDIDATE_PATH, "overlay_candidate_no_files")
        observation = json.loads(PROFILE_REPLAY_PATH.read_text())["steps"][0][0][
            "observation"
        ]
        with mock.patch("builtins.open", side_effect=AssertionError("runtime file read")):
            action = candidate.agent(observation)
        self.assertEqual(set(action), {"farmer", "hands", "market"})

Use compatible replay 93232089.json only as development input.

- [ ] **Step 2: Run RED**

Run:

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay.CandidateParityTests

Expected: import error because market_candidate.py is absent.

- [ ] **Step 3: Seed and minimally mark the candidate**

Mechanically copy 01_baseline3k/main.py to market_candidate.py and verify equal SHA-256 before editing. Use apply_patch to add:

    _ENABLE_DEMAND_DEFERRAL = False
    _ENABLE_ADAPTIVE_FRONT_RUN = False

Place the flags after _WEED_REPLAY_STEPS and do not change agent yet.

- [ ] **Step 4: Run GREEN and baseline-integrity checks**

Run the focused tests, git diff --check, and:

    git diff --exit-code -- duy/another_work/01_baseline3k/main.py

Expected: 720-call parity passes and the baseline diff is empty.

- [ ] **Step 5: Commit only Task 1 files**

    git add duy/another_work/02_inspect_top1/market_candidate.py duy/test_baseline3k_market_overlay.py
    git commit -m "test: seed baseline market candidate"

---

### Task 2: Implement Bounded Demand Deferral

**Files:**
- Modify: market_candidate.py
- Modify: duy/test_baseline3k_market_overlay.py

**Interfaces:**
- Produces: _market_state(obs, step), _demand_units(obs, item, step), _next_demand_release(obs, item, step), and _apply_demand_deferral(action, obs, state, step).

- [ ] **Step 1: Write failing controller tests**

Use make_observation with configurable seat, step, shed, prices, shops, and hands. Test:

    def test_duplicate_yarn_store_demand(self):
        obs = make_observation(step=100, shops=["YARN_STORE", "YARN_STORE"])
        self.assertEqual(agent._demand_units(obs, "WOOL", 100), 4)

    def test_depressed_milk_sale_defers_to_post_demand_step(self):
        obs = make_observation(
            step=101, shed={"MILK": 8}, prices={"MILK": 40},
            shops=["SMOOTHIE_SHOP"],
        )
        state = agent._market_state(obs, 101)
        result = agent._apply_demand_deferral(
            pass_action(market=[["SELL", "MILK", 6]]), obs, state, 101
        )
        self.assertEqual(result["market"], [])
        self.assertEqual(
            state["deferred"]["MILK"],
            {"quantity": 6, "release_step": 105, "deadline": 105},
        )

    def test_release_is_stock_capped_and_exactly_once(self):
        state = {
            "last_step": 104,
            "deferred": {
                "MILK": {"quantity": 6, "release_step": 105, "deadline": 105}
            },
            "front_debt": {},
        }
        obs = make_observation(step=105, shed={"MILK": 5})
        first = agent._apply_demand_deferral(pass_action(), obs, state, 105)
        second = agent._apply_demand_deferral(pass_action(), obs, state, 105)
        self.assertEqual(first["market"], [["SELL", "MILK", 5]])
        self.assertEqual(second["market"], [])

Also test pickup reserves, live-stock cap, no demand, price at/above base, shed occupancy above 90, final flush from step 715, seat isolation, backwards-step reset, malformed observations, non-premium parity, non-sale order parity, unchanged farmer/hands, and the ten-order cap.

- [ ] **Step 2: Run RED**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay.DemandDeferralTests

Expected: missing helper errors.

- [ ] **Step 3: Add state and constants**

    _PREMIUM_BASE_PRICES = {
        "MELON": 250, "MILK": 160, "STRAWBERRY": 120, "WOOL": 200,
    }
    _MARKET_STATE = {
        0: {"last_step": -1, "deferred": {}, "front_debt": {}},
        1: {"last_step": -1, "deferred": {}, "front_debt": {}},
    }
    _DEFER_MAX_STEPS = 4
    _DEFER_SHED_GUARD = 90
    _FINAL_FLUSH_STEP = 715

Count duplicate shop instances. Yarn contributes two units per instance; other matching shops contribute one. Town-center demand contributes one on steps divisible by 24. Since action-step sales happen before same-step demand, release on the turn after the next observed demand tick.

Allow deferral only when price is below base, release is within four turns, shed total is at most 90, and step is below 715. Cap at live shed minus scheduled pickup reserve. Clear due state before emission for same-step retry safety.

- [ ] **Step 4: Wire behind the flag**

After the unchanged baseline _front_run call:

    if _ENABLE_DEMAND_DEFERRAL:
        market_state = _market_state(obs, step)
        action = _apply_demand_deferral(action, obs, market_state, step)

- [ ] **Step 5: Run demand and parity suites**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay.DemandDeferralTests duy.test_baseline3k_market_overlay.CandidateParityTests

Expected: all pass.

- [ ] **Step 6: Commit**

    git add duy/another_work/02_inspect_top1/market_candidate.py duy/test_baseline3k_market_overlay.py
    git commit -m "feat: add bounded premium sale deferral"

---

### Task 3: Implement Adaptive Multi-Step Front-Running

**Files:**
- Modify: market_candidate.py
- Modify: duy/test_baseline3k_market_overlay.py

**Interfaces:**
- Produces: _repay_front_debt(action, state, step) and _adaptive_front_run(action, obs, state, step).
- Debt shape: state["front_debt"][due_step][item] = quantity.

- [ ] **Step 1: Write failing tests**

Patch and restore selected future _ACTIONS entries in each test:

    def test_moves_stock_four_turns_early(self):
        replace_route_sale(step=104, item="MILK", quantity=6)
        obs = make_observation(step=100, shed={"MILK": 5}, shops=[])
        state = agent._market_state(obs, 100)
        result = agent._adaptive_front_run(pass_action(), obs, state, 100)
        self.assertEqual(result["market"], [["SELL", "MILK", 5]])
        self.assertEqual(state["front_debt"], {104: {"MILK": 5}})

    def test_does_not_cross_shop_demand(self):
        replace_route_sale(step=104, item="MILK", quantity=6)
        obs = make_observation(
            step=100, shed={"MILK": 6}, shops=["SMOOTHIE_SHOP"]
        )
        state = agent._market_state(obs, 100)
        result = agent._adaptive_front_run(pass_action(), obs, state, 100)
        self.assertEqual(result["market"], [])
        self.assertEqual(state["front_debt"], {})

    def test_repay_reduces_future_sale_once(self):
        state = {
            "last_step": 103, "deferred": {},
            "front_debt": {104: {"MILK": 5}},
        }
        action = pass_action(market=[["SELL", "MILK", 8]])
        first = agent._repay_front_debt(action, state, 104)
        second = agent._repay_front_debt(action, state, 104)
        self.assertEqual(first["market"], [["SELL", "MILK", 3]])
        self.assertEqual(second["market"], [["SELL", "MILK", 8]])

Also test the four-step boundary, non-premium exclusion, pickup/current-sale reserves, multiple sale orders, multiple due items, capacity, and state reset.

- [ ] **Step 2: Run RED**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay.AdaptiveFrontRunTests

- [ ] **Step 3: Implement repayment and lookahead**

Pop due debt before rewriting copied market orders. Subtract from matching premium sales in original order and remove zero orders.

For each premium item, scan offsets 1..4. Stop before any intermediate step whose observed town demand is positive. Select the first future route sale and move:

    min(
        future_quantity - already_recorded_debt,
        live_stock - pickup_reserve - existing_current_sales,
    )

Merge with an existing same-item sale or append only below ten orders. Record exact moved quantity under its future due step.

- [ ] **Step 4: Exclusively replace baseline front-running when enabled**

    if _ENABLE_ADAPTIVE_FRONT_RUN:
        market_state = _market_state(obs, step)
        action = _repay_front_debt(action, market_state, step)
        action = _adaptive_front_run(action, obs, market_state, step)
    else:
        state = _fr_state(obs, step)
        action = _repay(action, state, step)
        action = _front_run(action, obs, state, step)

    if _ENABLE_DEMAND_DEFERRAL:
        market_state = _market_state(obs, step)
        action = _apply_demand_deferral(action, obs, market_state, step)

- [ ] **Step 5: Run the full focused module twice**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay
    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay

Expected: identical passing counts and no leaked state.

- [ ] **Step 6: Commit**

    git add duy/another_work/02_inspect_top1/market_candidate.py duy/test_baseline3k_market_overlay.py
    git commit -m "feat: add adaptive premium front running"

---

### Task 4: Build the Deterministic Evaluator

**Files:**
- Create: duy/another_work/02_inspect_top1/evaluate_market_overlay.py
- Modify: duy/test_baseline3k_market_overlay.py

**Interfaces:**
- Consumes existing benchmark resolve_agent, run_suite, summarize, build_metadata, and write_artifacts functions.
- Produces VARIANTS, render_variant, promotion_failures, profile_candidate, and screen/confirm CLI phases.

- [ ] **Step 1: Write failing evaluator tests**

Require:

    FLAG_NAMES = (
        "_ENABLE_DEMAND_DEFERRAL",
        "_ENABLE_ADAPTIVE_FRONT_RUN",
    )
    VARIANTS = {
        "control": (False, False),
        "demand_defer": (True, False),
        "adaptive_front_run": (False, True),
    }

Test exact-once flag replacement, unknown/missing flag errors, deterministic metadata without timestamp/temp paths, and each promotion failure independently: game count, paired mean, paired median, both seat means, wins at or below 55, and bootstrap lower bound.

- [ ] **Step 2: Run RED**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay.EvaluatorTests

- [ ] **Step 3: Implement rendering and CLI**

Arguments:

    --candidate PATH
    --baseline PATH
    --output-dir PATH
    --phase {screen,confirm}
    --screening-json PATH

Screen all three variants on seeds 0..9 and write each normal benchmark artifact, each exact rendered agent source, plus screening.json. Control must tie exactly. Advance only non-control policies with positive paired mean; rank by paired mean, paired median, then name. The screening record freezes the winner name, flags, and rendered SHA-256.

Confirm reads the frozen identity from the supplied screening.json, rejects any source/hash mismatch, and runs only that winner on seeds 50..99. Write promotion.json and confirmed_candidate.py, verify the source file has the recorded digest, and return zero only when every promotion gate passes. Refuse existing output directories.

- [ ] **Step 4: Implement latency profiling**

Load the rendered candidate once and time its 720 calls on replay 93232089. Report import, mean, median, nearest-rank p95, and maximum milliseconds. Confirmation fails at mean >=1 ms or p95 >=2 ms.

- [ ] **Step 5: Run focused and full discovery**

    duy/.venv/bin/python -m unittest -v duy.test_baseline3k_market_overlay
    duy/.venv/bin/python -m unittest discover -v duy

- [ ] **Step 6: Commit**

    git add duy/another_work/02_inspect_top1/evaluate_market_overlay.py duy/test_baseline3k_market_overlay.py
    git commit -m "test: add market overlay qualification gate"

---

### Task 5: Screen and Freeze One Policy

**Files:**
- Create untracked: duy/benchmarks/results/baseline3k-market-overlay/screen
- Modify: BENCHMARK_FINDINGS.md

- [ ] **Step 1: Preflight**

Verify Kaggle Environments 1.32.7, pip check, unchanged baseline diff, and candidate/baseline SHA-256 identities.

- [ ] **Step 2: Run exactly one screen**

    duy/.venv/bin/python duy/another_work/02_inspect_top1/evaluate_market_overlay.py --candidate duy/another_work/02_inspect_top1/market_candidate.py --baseline duy/another_work/01_baseline3k/main.py --output-dir duy/benchmarks/results/baseline3k-market-overlay/screen --phase screen

Expected: 60 valid games, 20 per tuple, and exact-zero control margins.

- [ ] **Step 3: Audit and freeze**

Verify DONE/reward validity, complete seed-seat pairs, and recompute paired means independently. Freeze only the highest-ranked non-control policy with positive paired mean. Record its name, flags, rendered/source/baseline hashes, metrics, and command.

If neither policy is positive, document no winner, preserve current main, commit the report, and skip Tasks 6 and 7 promotion actions.

- [ ] **Step 4: Commit only findings**

    git add duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md
    git commit -m "docs: record market overlay screen"

---

### Task 6: Confirm the Frozen Policy on 100 Fresh Games

**Files:**
- Create untracked: duy/benchmarks/results/baseline3k-market-overlay/confirm
- Modify: BENCHMARK_FINDINGS.md

- [ ] **Step 1: Reverify frozen hash**

Render the selected flags again and require an exact SHA-256 match with Task 5. Any mismatch invalidates the screen and stops confirmation.

- [ ] **Step 2: Run once on seeds 50..99**

Use the frozen Task 5 record directly:

    duy/.venv/bin/python duy/another_work/02_inspect_top1/evaluate_market_overlay.py --candidate duy/another_work/02_inspect_top1/market_candidate.py --baseline duy/another_work/01_baseline3k/main.py --output-dir duy/benchmarks/results/baseline3k-market-overlay/confirm --phase confirm --screening-json duy/benchmarks/results/baseline3k-market-overlay/screen/screening.json

Expected: exactly 100 valid games. Exit zero means all statistical and latency gates pass; exit one is a binding rejection.

- [ ] **Step 3: Independently audit**

Require 50 unique seeds with both seats, 100 DONE rows, reward/money equality, positive paired mean/median, positive seat means, more than 55 wins, positive bootstrap lower bound, mean latency below 1 ms, and p95 below 2 ms.

- [ ] **Step 4: Record and commit the binding decision**

Record full statistics, hashes, versions, commands, and promotion failures. On rejection state explicitly that main.py remains unchanged.

    git add duy/another_work/02_inspect_top1/BENCHMARK_FINDINGS.md
    git commit -m "docs: record fresh market overlay confirmation"

---

### Task 7: Promote Only a Verified Winner and Finish

**Files:**
- Conditionally modify: 02_inspect_top1/main.py
- Conditionally create: 02_inspect_top1/market_overlay_promotion.json
- Modify: STRATEGY.md
- Modify: duy/test_baseline3k_market_overlay.py

- [ ] **Step 1: Write a failing promotion-integrity test after success only**

    def test_promoted_main_matches_committed_winner_identity(self):
        promotion = json.loads(PROMOTION_RECORD_PATH.read_text())
        actual = hashlib.sha256(PROMOTED_MAIN_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, promotion["candidate_sha256"])

Verify RED while main remains the pre-experiment agent.

- [ ] **Step 2: Mechanically install exact frozen bytes**

Copy promotion.json to market_overlay_promotion.json after removing temporary paths, and verify its candidate digest against confirmed_candidate.py. Copy the evaluator-rendered confirmed candidate through a temporary file, verify its digest again, then replace main.py mechanically. Do not hand-edit the promoted copy.

- [ ] **Step 3: Verify**

Run focused tests, full discovery, git diff --check, baseline no-diff, and a fresh latency profile. Update STRATEGY.md with the rejected replay-route lesson, the transferred market lesson, exact confirmation result, and limitations.

- [ ] **Step 4: Commit the promotion or rejection conclusion**

On success:

    git add duy/another_work/02_inspect_top1/main.py duy/another_work/02_inspect_top1/market_overlay_promotion.json duy/another_work/02_inspect_top1/STRATEGY.md duy/test_baseline3k_market_overlay.py
    git commit -m "feat: promote verified baseline market overlay"

On rejection, do not touch main.py; commit only the strategy conclusion and relevant tests/docs.

- [ ] **Step 5: Review, integrate, and clean**

Use superpowers:requesting-code-review and superpowers:finishing-a-development-branch. Verify untracked corpus, venv, raw results, and unrelated user files are not staged. Merge the reviewed branch into main, rerun focused and full tests from main, and remove only:

    /Users/minhduy/Desktop/Project/kaggle/kaggriculture/.worktrees/top100-shop-adaptive

Verify the active branch is main, the temporary worktree is absent from git worktree list, and the original duy_explore corpus remains intact.
