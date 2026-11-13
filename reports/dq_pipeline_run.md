# Decision Quality pipeline run (DQ-v0): stopped at Gate G1, blocked at Gate G2

> **Update, DQ-v4 prototype (section 9): a DQ prototype was built on Metrica, past gate G3 at the user's request. It is robust to parameter changes (G4 PASS) but does not separate intercepted from completed passes (AUC 0.54-0.55), and it rewards progression far more than risk. The expert protocol is in `dq_expert_protocol.md`.**
>
> **Update, run DQ-v2 (section 7): the target-free success model transfers across competitions but is over-confident on Metrica (slope 0.798, 95% CI 0.72-0.88; gate floor 0.8), so Stage 2 is still not passed.**
>
> **Update, run DQ-v1 (section 6): Stage 1 passes after the repairs; Stage 2 still fails.** On two new competitions, Stage 1 passes after the two repairs below. Stage 2 fails again using 360 freeze frames: a target is identified for 59% of completed vs 41% of failed passes. Sections 0-5 describe the original v0 run.

**Outcome: the run stopped at Stage 1.** The value models were built and tested against gates frozen before any download. The shot model (xG) passes. The possession-value model **V** fails out-of-competition calibration. The turnover-value model **Vopp** fails to beat a constant. Independently, Stage 2 is **structurally blocked**: no authorised data provides an outcome-independent intended target for failed passes. No DQ value was computed for any situation or player.

Scope and provenance: StatsBomb open event data, Premier League 2015/16 (all 380 matches) plus a seeded random 40 of the 64 FIFA World Cup 2018 matches, **1.248 GB downloaded** (limit 1.5 GB). Every raw file was deleted immediately after reduction; the derived tables were deleted after the analysis. Only scripts and aggregate outputs remain, in `/tmp/claude-0/-home-user-DecisionQualityEngine/c4ca3e56-cf08-52a3-b16e-487a22fb013f/scratchpad/dq/` (`PREREG_DQ_v0.txt`, `fetch.py`, `stage1.py`, `stage2.py`, `metrica_shots.py`, `*_out.txt`); `fetch.py` re-creates the data. Data: StatsBomb Open Data (free for research, attribution required). Fitting used numpy only (no new dependencies). No repository code changed; nothing committed.

## 0. Stage 0: frozen definition (`PREREG_DQ_v0.txt`, 2026-11-13T10:53:48Z, before any download)

* Unit: one on-ball decision at the release frame of an open-play ground pass or foot shot (Metrica tracking, existing decision contract). Actions: pass to each R0-admissible teammate, or shoot.
* `EV(pass) = Ps * V(receiver location) + (1 - Ps) * (-Vopp(mirrored receiver location))`; `EV(shot) = xG`. The turnover is assumed at the receiver location (stated simplification).
* `DQ_w = EV(chosen) - sum_a w(a) EV(a)` with choice-model weights (primary); `DQ_max = EV(chosen) - max_a EV(a)` (secondary). Situation-level only until Stage 6.
* Gates G1-G6 and a stop rule (stop at the first gate that fails and cannot be fixed with authorised data; no threshold changed after seeing results).

## 1. Stage 1: value models (Gate G1)

Train = 190 PL matches before 2016-01-02; test A = 190 later PL matches; test B = 40 WC 2018 matches. All three models are logistic regressions on a fixed basis with ridge 1.0. CIs come from a match bootstrap (1,000 resamples). Gate per test set: log-loss gain over the constant with CI excluding 0; calibration slope 0.8-1.2; mean prediction / mean outcome 0.85-1.15; xG AUC >= 0.75.

| Model | Test | n (positives) | Log-loss gain vs constant [95% CI] | Slope | mean p / mean y | Extra | Gate |
|---|---|---|---|---|---|---|---|
| **V** (possession ends in a goal, any on-ball event) | A (PL) | 482,461 (7,994) | +0.0028 [+0.0023, +0.0033] | 0.96 | 1.04 | | PASS |
| | B (WC) | 103,670 (1,380) | +0.0032 [+0.0024, +0.0042] | 1.06 | **1.26** | | **FAIL** |
| **Vopp** (first event after an open-play turnover) | A | 14,347 (177) | +0.0002 [-0.0001, +0.0004] | 0.88 | 1.11 | | **FAIL** |
| | B | 3,348 (38) | -0.0002 [-0.0007, +0.0004] | 0.28 | 1.23 | | **FAIL** |
| **xG** (open play, non-penalty) | A | 4,715 (448) | +0.043 [+0.035, +0.052] | 0.90 | 0.95 | AUC 0.769 (StatsBomb xG 0.789) | PASS |
| | B | 964 (81) | +0.050 [+0.032, +0.068] | 0.93 | 1.01 | AUC 0.794 (StatsBomb 0.799) | PASS |
| xG transfer to Metrica (50 open-play foot shots; weak check) | | 50 (7 goals) | predicted sum 5.53, Poisson 95% range 1-11 | | | G1 1.77 vs 2, G2 1.56 vs 3, G3 2.20 vs 2 | PASS |

Reading:
* **xG is usable.** It is well calibrated within and across competitions and close to StatsBomb's own xG (AUC 0.77-0.79 vs 0.79-0.80). Its Metrica totals are plausible. That check can only detect gross errors with 7 goals, and it depends on a linear 120x80 to 105x68 m mapping and on a goalkeeper inferred as the opponent nearest his goal.
* **V ranks well but its level does not transfer.** Reliability within the PL is excellent (top bin 0.034 predicted vs 0.032 observed). In the World Cup the possession goal rate is lower (1.33% vs 1.66%) and V over-predicts by 26% in every bin. This is the same base-rate-shift problem the Metrica pass model showed. Metrica's competition is unknown, so V's level for Metrica cannot be known.
* **Vopp has no detectable skill.** The location where the opponent regains possession does not predict, beyond a constant (~1.2-1.4%), whether that possession ends in a goal, with this basis. The failure cost of a pass is then effectively location-independent in these data.

**G1 = FAIL** under the frozen criteria. Two design changes could arguably repair it. Both are **post-hoc changes, so I have not applied them**; each is your decision:
1. Report DQ on a relative scale (DQ divided by the situation's mean EV), so a competition-wide multiplicative shift of V cancels. If V and Vopp shift by the same factor, the sign of `DQ_w` is unchanged. This is not shown for unequal shifts (here 1.26 vs 1.23 in test B).
2. Treat `Vfail` as a constant (the measured turnover base rate) instead of a location model, since no location signal was found.

## 2. Stage 2: calibrated pass-success probability (Gate G2): blocked by data

G2 needs a dataset with three things: the decision-time state, an intended target that does not depend on the outcome (for completed *and* failed passes), and at least 10 matches with transferable calibration.

| Source | Status |
|---|---|
| Metrica (3 matches, tracking) | Already failed: ranking transfers but calibration slopes are 0.39-2.76 across held-out games (`pooled_validation.md`); the target of failed passes is known only post hoc and coverage depends on the outcome |
| StatsBomb events (420 matches checked here) | **Not usable as the target.** For 63,794 incomplete passes followed by an opponent event, the pass `end_location` coincides with that opponent event's location in **96.1%** (median 0.14 units; Interception, Block, Clearance, Ball Recovery and Goal Keeper all 99.8-99.9%). It records where the ball was stopped, which is the outcome itself, not the intended receiver. Training a completion model on it would build the outcome into the feature. No freeze frames for passes outside 360 data; 360 frames share the same target problem and lack velocities |
| IDSSE, SkillCorner | Failed the release-timing gates earlier (`idsse_pilot.md`, `skillcorner_release_audit.md`) |

**G2 = blocked.** This is a data limitation, not a modelling one. Even if G1 were repaired, Ps for hypothetical passes cannot be calibrated, so `EV(pass)` cannot be computed for alternatives.

## 3. Stages 3-6: not run

* **Stage 3 (realistic options).** The motion-direction hypothesis passed its pre-registered criteria on the same three Metrica games: top-1 0.414-0.437 rising to 0.529-0.570; log loss -0.26 [-0.28, -0.24]; placebo p < 0.005; leakage check 0/2,649. A one-window-earlier velocity gave almost the same gain, so the signal is the persistent direction of travel. Confirmation needs an untouched tracking match with valid release timing, which no authorised source provides.
* **Stage 4** needs Ps (blocked). **Stage 5 (expert panel)** needs people, not data, and is the only stage that can proceed independently. **Stage 6** is not reachable.

## 4. What this establishes

* An **externally validated xG** is achievable from public data. It transfers across competitions and gives plausible totals on Metrica. This is the first value component the programme has that passed a frozen gate.
* A **location-based possession value** ranks situations but needs per-competition level adjustment. A **turnover-location value** shows no signal.
* The binding constraint on DQ is **Ps for hypothetical passes**. It needs tracking data, with release-accurate timing, across at least 10 matches, and outcome-independent targets. None of the sources examined (Metrica, StatsBomb events, IDSSE, SkillCorner) meets this.

## 5. Minimum next steps (each needs your decision)

1. **Data for G2:** a tracking dataset of at least 10 matches with release-accurate timing (25 Hz with event frames aligned to the ball leaving the foot), for example a licensed provider or a research partnership. Without it, DQ for passes cannot be computed defensibly.
2. **Decide on the two post-hoc G1 repairs** (relative-scale DQ, constant Vfail). If accepted, they must be re-registered and re-tested on a new held-out competition, not on the data that suggested them.
3. **Stage 5 can start now:** design the expert-panel protocol (situation sampling, blinding to outcome, inter-rater kappa target) and its power analysis. It tests whether "good decision" has any agreed reference, which is the largest remaining risk.
4. **Possible narrower product:** "shot vs pass" now has one validated component (xG). It still needs Ps, so it is blocked at the same point.

## 6. Run DQ-v1: repaired Stage 1 and a Stage 2 test with player positions (360)

Re-registration `PREREG_DQ_v1.txt` (2026-11-13T11:02:53Z) was written before anything was re-downloaded or refitted. **Both repairs are post-hoc**: test B suggested them. They are therefore judged only on two competitions never used before, C = UEFA Euro 2020 (51 matches) and D = FIFA World Cup 2022 (64 matches). The models are refitted identically to v0; the Metrica xG total reproduces as 5.53. This time the coefficients are saved (`stage1b_models.pkl`); no data is.
Data: gzip transfer, decompressed in memory only, nothing raw written to disk. This run transferred **435 MB** (163 MB in an attempt cut by a connection reset, then 272 MB with retries). Cumulative over both runs: 1.25 GB + 0.44 GB. The derived tables were deleted after use.

### 6.1 Stage 1 repaired: Gate G1' PASS (C and D)

* **Repair 1, relative scale:** DQ is reported only as `DQ_rel = DQ / mean_a EV(a)` within the situation, so a competition-wide level of V cancels. The gate tests the *shape* of V after one level factor `c`.
* **Repair 2, constant failure value:** `Vfail = -k * mean(V)`, with `k` = (goal rate of possessions after open-play turnovers) / (mean V) estimated on train, `k_train` = 0.814. The gate requires `k` to stay within ±25% on each test set.

| Test | Level factor c | V shape: log-loss gain [95% CI] | slope | quintile obs/pred | k (vs 0.814) | xG AUC (StatsBomb) / slope / mean p/y | Gate |
|---|---|---|---|---|---|---|---|
| A PL (seen) | 0.96 | +0.0028 [+0.0023, +0.0033] | 0.96 | 0.98-1.05 | 0.718 (-12%) | 0.769 (0.789) / 0.90 / 0.95 | pass |
| B WC18 (suggested repair) | 0.79 | +0.0032 [+0.0024, +0.0041] | 1.08 | 0.89-1.11 | 0.676 (-17%) | 0.794 (0.799) / 0.93 / 1.01 | pass |
| **C Euro 2020 (new)** | 1.18 | +0.0036 [+0.0023, +0.0048] | 1.00 | 0.93-1.05 | 0.931 (+14%) | 0.776 (0.814) / 0.89 / 0.94 | **PASS** |
| **D WC 2022 (new)** | 1.24 | +0.0033 [+0.0023, +0.0042] | 1.02 | 0.91-1.07 | 0.944 (+16%) | 0.800 (0.819) / 1.01 / 0.89 | **PASS** |

Reading:
* **V's shape transfers; its level does not.** The level moves by -21% to +24% between competitions, so only the relative scale is defensible.
* **The turnover location again carries no signal** on C and D (gains +0.0002 and +0.0005, CIs include 0). This supports the constant Vfail.
* **k varies by up to ±17%.** It is within the frozen limit, but this variation will enter `DQ_rel` as uncertainty and must be propagated in Stage 4.
* **xG holds on both new competitions,** a little below StatsBomb's own model (AUC 0.78-0.80 vs 0.81-0.82).

### 6.2 Stage 2 with 360 freeze frames: can player positions reveal the intended receiver of an intercepted pass?

StatsBomb 360 records the positions of the players **visible to the broadcast camera** at the moment of each event (teammate/opponent flag, no identities, no velocities). I applied the existing Metrica heading rule unchanged (cone 15°, gap 5°): find the teammate standing along the line from the passer to where the ball was stopped. Scope: 77,270 open-play ground passes (Euro 2020 + WC 2022); 360 frames present for 88% of both completed and incomplete passes.

| | Completed | Incomplete (failed) |
|---|---|---|
| target identified | **59.1%** (n = 65,456) | **40.7%** (n = 2,439) |
| no teammate in the cone at all | 29.1% | **46.4%** |
| visible teammates per frame (of 10) | 6.5 | 6.2 |
| identified teammate = teammate nearest the end location (completed passes) | 87.4% | n/a |

Per competition the picture is the same (Euro 2020 57.3% vs 40.0%; WC 2022 60.6% vs 41.2%). **Gate G2-target = FAIL** on the coverage gap (18 pp > 10 pp). Agreement (87% >= 80%) and frame availability (gap 0.2 pp) pass.

**Answer to the question:** yes, the positions are there (in 360 for the visible players; in Metrica for all 22 players), and they were used. The rule finds the teammate along the line of the pass for only 41% of failed passes, against 59% of completed ones. In 46% of failed passes no teammate stands along that line at all. These are passes into space or toward a runner, or the teammate is outside the camera view (only about 6 of 10 teammates are visible). The same 18-25 pp gap appeared on Metrica with full tracking (60.5% vs 35.3%), so camera coverage is not the main cause. **The problem is the definition of "intended receiver", not the data source.** Failed passes are disproportionately passes that had no clear receiver. Training Ps only on passes with an identified target would remove exactly the failures that the probability must learn from. **The run stops here (stop rule).**

### 6.3 Options that would need a new registration (not run)

1. **Target-free completion model:** predict the success of a pass in a given *direction and length* rather than to a person. A failed pass then needs no receiver; only the direction is used, never where the ball stopped. Each candidate teammate gets the Ps of a pass toward his position. The gap above disappears by construction. The remaining risk is that the length of failed passes is truncated by the interception, so the intended length must be modelled or bounded.
2. **Soft target assignment:** a probability over teammates weighted by angle (the choice model) instead of a hard rule, keeping every pass.
3. **Tracking data of at least 10 matches** with release-accurate timing remains the cleanest route, but the 18-25 pp gap on Metrica shows it does not by itself solve the target problem.

## 7. Run DQ-v2: target-free pass-success model (Stage 2 retry)

Pre-registration `PREREG_DQ_v2.txt` (2026-11-13T11:19:56Z) was written before any download or fit for this model. Data: Euro 2020 + WC 2022 events and 360, **150 MB transferred** (gzip), decompressed in memory only. The derived step tables (StatsBomb and Metrica) were deleted after the analysis; only the fitted coefficients (`tf_model.pkl`) and outputs remain.

**Model.** A discrete-time interception hazard along the pass line, in 2-unit steps.
* A failed pass is an *event* at the step where the ball was stopped. A completed pass is *censored* at its length. Neither the receiver nor the intended length is needed.
* For a hypothetical pass to a teammate at distance d: `Ps = prod_{steps < d} (1 - h)`.
* Covariates come from positions at release only (360 has no velocities): step distance, distance of the nearest opponent to the segment, its ratio to the ball's travel distance, opponents near the segment, pitch position, passer pressure, and whether the segment is inside the camera's visible area.
* Untestable assumption: the intended length is independent of the step hazard given the covariates.

| Test | Step rows (events) | LL gain/row [95% CI] | Slope | O/E | Between-match tau | Gate |
|---|---|---|---|---|---|---|
| train Euro 2020 -> test WC 2022 | 362,757 (1,635) | +0.0065 [+0.0061, +0.0069] | 0.97 | 1.13 | 0.11 | PASS |
| train WC 2022 -> test Euro 2020 | 288,104 (1,197) | +0.0061 [+0.0056, +0.0066] | 1.04 | 0.88 | 0.18 | PASS |
| **fitted on both -> Metrica (3 games, all players visible)** | 29,187 (268) | +0.0117 [+0.0074, +0.0140] | **0.798** [0.72, 0.88] (below the 0.8 floor; shown unrounded) | 1.02 | (3 games) | **FAIL** |

Metrica per game: O/E 1.12 / 0.93 / 1.02; slope 0.80 / 0.84 / 0.74 (pooled 0.798).

Reading:
* **Without a receiver, the success model works across competitions.** The level and the shape both transfer between tournaments, and between-match variation in level (tau 0.11-0.18) is within the frozen limit. This is the first success model in the programme to pass a cross-competition calibration test. The base rate did not transfer for the receiver-based Metrica model.
* **On Metrica the overall level is right (O/E 1.02), but the hazards are too extreme.** The slope is 0.80, CI 0.72-0.88, so the gap is not a rounding effect. Low hazards are under-predicted (second quintile 0.0003 predicted vs 0.0009 observed); the highest are slightly over-predicted. A plausible cause, not tested: 360 frames miss off-camera opponents, so the model learned "far from any visible opponent = very safe", while Metrica shows every opponent.
* **Excluded by design (both sources):** passes stopped within 5 units (about 4.4 m) of the passer. That is 65 of 334 Metrica interceptions (19%) vs 73 of 2,649 completions (3%), so blocks at the foot are outside the model.
* **Gate G2-cal = FAIL (Metrica transfer)**, so the stop rule applies. This is a much narrower failure than in v1: a recalibration problem rather than a structural bias.

Options requiring a new registration: (1) recalibrate the slope on Metrica with leave-one-game-out (fit on two games, test on the third; 268 events, wide intervals); (2) restrict training to segments inside the visible area, or weight by visibility, to reduce the off-camera bias; (3) more tracking matches to test either option independently.

## 8. Run DQ-v3: option 2 (visibility) and option 1 (Metrica recalibration)

Pre-registration `PREREG_DQ_v3.txt` (2026-11-13T11:39:40Z), written before re-downloading (150 MB, in memory) or fitting. Both options are post-hoc remedies for v2's Metrica slope of 0.798. Step tables were deleted after the run; coefficients are kept in `tf_model_v3.pkl`.

**Option 2: train only on what the camera saw. Prediction failed; hypothesis not supported.** Each pass is censored where it leaves the 360 visible area. This affects 10,630 of 68,809 passes, removes 6.9% of step rows, and drops 469 of 2,832 failures.

| Test | Slope | O/E | tau | Gate |
|---|---|---|---|---|
| Euro 2020 -> WC 2022 | 0.98 | **1.162** | 0.14 | FAIL (O/E) |
| WC 2022 -> Euro 2020 | 1.03 | 0.861 | **0.217** | FAIL (tau) |
| -> Metrica (pooled) | **0.705** | **1.182** | | FAIL |

The pre-stated prediction was that Metrica's slope would move toward 1 from 0.798. It moved **away**, to 0.705, and cross-competition transfer also got worse. Camera visibility is therefore **not** the explanation for the over-confidence, at least not in this simple form. The cause remains unknown. Untested candidates: the different stop-point definitions (StatsBomb event location vs Metrica ball tracking at the end frame), differences in what is recorded as a pass, and the absence of velocities.

**Option 1: recalibration on Metrica, leave-one-game-out. Passes.** For each held-out game, `logit(h_cal) = a + b*logit(h)` is fitted on the other two games' step rows.

| Base model | Fitted b per fold | Out-of-fold slope | O/E | Per-game O/E (G1 / G2 / G3) | Log loss/row before -> after | Gate |
|---|---|---|---|---|---|---|
| option-2 model (pre-registered primary) | 0.69-0.72 | 0.996 | 1.000 | 1.24 / 0.90 / 0.88 | 0.04499 -> 0.04405 | **PASS** |
| v2 model (comparison) | 0.78-0.82 | 0.999 | 1.005 | 1.20 / 0.90 / 0.93 | 0.04257 -> 0.04223 | PASS |

**Gate G2-cal (v3) = PASS, conditionally.** Conditions and caveats:
* **The calibration is Metrica-specific.** It uses labels from Metrica's own games, so any new tracking source would need its own recalibration.
* **The correction is stable but per-game level still varies.** The fitted slope is very stable across folds (b 0.69-0.72; 0.78-0.82), which suggests a consistent source-level difference rather than noise. But G1's O/E is 1.20-1.24, outside 0.85-1.15 at game level; the gate is pooled.
* **The sample is small and not fresh:** 268 events in three games that earlier work already used.
* **I recommend carrying the v2 model plus recalibration forward rather than the pre-registered option-2 base.** The option-2 base failed its own cross-competition gate; the v2 base passed it and has lower log loss after recalibration (0.04223 vs 0.04405). This choice is made after seeing results, and is labelled so.
* **Blocks at the foot are excluded:** passes stopped within 5 units, which is 19% of Metrica interceptions.

**Where the pipeline now stands:**

| Stage | Status |
|---|---|
| G1 | PASS, with the repairs |
| G2 | PASS, conditional and Metrica-recalibrated |
| G3 | **Blocked.** It requires confirming the realistic-option model, with the motion-direction feature, on an untouched tracking match with release-accurate timing; none is available among the authorised sources |
| G4 | Not run: a sensitivity-checked DQ prototype on Metrica is technically possible now, but only by running past G3 |
| G5 | Expert panel; not run |

## 9. DQ-v4: prototype on Metrica (run past gate G3, at the user's request)

Pre-registration `PREREG_DQ_v4.txt` (2026-11-13T11:57:38Z), written before any prototype computation. **Exploratory**: the realistic-option model was not confirmed on new data (G3), so everything below is conditional on it. No player-level output.

**Definition.**
* `EV_j = Ps_j * V_j - (1 - Ps_j) * k * m` for each R0 teammate j.
  * V is the PL-trained possession value at the teammate's position.
  * Ps comes from the hazard model with Metrica leave-one-game-out recalibration.
  * k = 0.814 and m = 0.0168.
* `DQ_w = EV_chosen - sum_j w_j EV_j`, with w the choice-model probabilities (including the motion feature, leave-one-game-out).
* Values are in percentage points (pp) of the possession-goal probability, up to the unknown competition factor c.
* **Shots are not included:** xG is absolute while V is only relative, so they cannot be compared on Metrica.

**Population.**
* P1: 2,568 completed passes with the receiver admissible.
* P2: 104 intercepted passes with a heading-identified target that is admissible (31% of 334 interceptions).

**Results.**
* **Magnitudes are tiny.** The mean EV of the chosen pass is 1.25 pp. The P1 `DQ_w` median is +0.016 pp (IQR -0.039 to +0.099); 59% are positive; the chosen pass is the EV-best option in 21%.
* **Stable across contexts.** By game, the P1 median is +0.018 / +0.031 / +0.006 (61% / 63% / 54% positive). It is similar across pitch thirds, and slightly higher under high passer pressure (+0.038) than low (+0.004).
* **Gate G4 (sensitivity) = PASS.** Sign agreement with the baseline in every setting: k x0.83 98.2%; k x1.17 98.0%; other folds' recalibration 98.4-99.3%; no recalibration 95.9%; R_loose 96.3%; choice weights without the motion feature 90.8% (Spearman 0.92-0.998). G4 tests the tuning parameters only, not the V or Ps models themselves.
* **Known-groups check: the pre-stated expectation was not met.** `DQ_w` barely separates passes that were later intercepted from completed ones: AUC **0.536 [0.459, 0.610]**, and 0.550 without set-piece deliveries. The chosen pass's Ps alone separates them well (0.811), and V alone at 0.634.
  * **Reason:** within a situation, EV ranges about 1.06 pp and is driven mainly by V (correlation 0.62, against 0.42 for Ps). The failure cost (k * m = 1.37 pp) is close to the value gain from progressing.
  * **Consequence:** risky forward passes come out "worth it". Intercepted passes have the same mean chosen EV as completed ones (1.24 vs 1.25 pp) despite a much lower Ps (0.81 vs 0.94).
  * **What it means:** the metric, as defined, measures *progression value relative to the expected choice* far more than *risk management*. Whether that matches expert judgement is exactly what Stage 5 must test.
* **Worked examples** (attack-oriented metres; EV in pp):
  * **Lowest `DQ_w`:** G1 event 820, 43:40. The passer (Player4, at x 41.9, under pressure 4.5 m) chose Player10, 34 m away, with Ps 0.23 and EV -0.29. The highest-EV alternative was Player6 (19 m, Ps 0.61, V 7.3, EV 3.93; choice weight 0.54). `DQ_w` -3.26. The pass was in fact completed, a reminder that a low score is ex ante, not a verdict on the outcome.
  * **Highest `DQ_w`:** G3 event 1759 (+6.63). This turned out to be a **corner kick**, see the next point.
  * **A typical case:** G1 event 989. A short safe pass under pressure, Ps 0.996, chosen = EV-best, `DQ_w` +0.016.
* **Data issue found: set-piece deliveries in the "open-play" pass population.**
  * **What:** 188 situations (184 completed, 4 intercepted; 7%) are `PASS` events recorded within 1 s of the same team's `SET PIECE` event: throw-ins 96, free kicks 65, kick-offs 16, corners 11. They are over-represented in the top 5% of `DQ_w` (10.4%).
  * **Effect here:** excluding them changes little (P1 median +0.015, 59.1% positive).
  * **Wider reach:** the same population definition (`PASS` without subtype) was used in earlier studies. Whether those `PASS` events are the deliveries themselves should be verified against Metrica's event documentation before any rerun. This was not fixed here (no repository code changes).
* **Expert panel:** the 120 pilot situations were drawn from this prototype, excluding set pieces. The protocol and sample-size calculation are in `reports/dq_expert_protocol.md`.

**Where this leaves the metric:** it is computable, internally stable, and transparent about what drives it. It does not track the most obvious ex-ante risk signal, because the value model prices turnovers cheaply. The next decisive step is the expert pilot (G5a, then G5b), not further modelling.
