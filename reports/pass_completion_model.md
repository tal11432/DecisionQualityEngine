# Pass-completion probability: methodology and held-out results

Train/select/calibrate on Metrica Sample Game 1; scored once on held-out Sample Game 2. Neither game is Ronaldo data.
Reproduce: `python -m src.pass_completion --train 1 --test 2 --out reports/pass_completion_results.txt --figdir reports`
(deterministic; identical output on repeated runs; ~45 s). Raw numbers: `reports/pass_completion_results.txt`.

## 0. Headline

| | Scenario A: **decision time** (target-free) | Scenario B: **early flight** (identified target; **not** decision-time) |
|---|---|---|
| Prediction frame | ball release t_d | release + 6 frames (0.24 s) |
| Held-out n (intercepted / completed) | 1023 (111 / 912) | 591 (37 / 554) |
| Primary logistic ROC-AUC [95% CI] | 0.583 [0.529, 0.636] | 0.767 [0.671, 0.857] |
| Best single feature set | feasibility aggregate (= primary) | feasibility margin alone: 0.863 [0.809, 0.910] |
| Exploratory boosted trees | 0.645 [0.593, 0.696] | 0.884 [0.843, 0.921] |
| Probability skill vs base rate (log-loss, paired) | primary -0.006 [-0.012, -0.0003] (marginal; Brier n.s.) | primary -0.034 [-0.062, -0.005]; Brier n.s. |
| Verdict | **no useful decision-time completion probability** | useful *ranking* for clearly directed passes; probabilities only roughly calibrated; small n |

A genuine **target-specific** decision-time model is **not possible with this data** (section 1). The decision-time model that *is* possible (A) is
real, leakage-safe and weak. The informative model (B) needs 0.24 s of ball flight and covers only ~1/3 of interceptions.

## 1. Can the intended receiver be identified at or before release?

No. Evidence (`target_identifiability`, Game 1, 753 completed passes with a known receiver, rules fixed a priori, true receiver used only for scoring):
nearest teammate 30.8% top-1, passer-velocity-aligned teammate 34.6% (586 moving passers), safest interception lane 34.3%, against 10.0% chance.
Further: Metrica gives no receiver for failed passes (`To` is empty for `BALL LOST/INTERCEPTION`), no body orientation, and no pass-target location.
Hence even a good decision-time receiver predictor could not be validated on the failures that matter.

**Correction to the earlier feasibility audit.** That evaluation took its target from the ball's heading over the first 6 post-release frames. Those
frames are *after* the decision, so it was a hybrid (inputs at release, target chosen 0.24 s later), not a decision-time prediction. It is now
re-expressed as scenario B with all features at t_p; the earlier report carries a note.

Two scenarios, never mixed (separate feature lists, tables, models, metrics and claims):
* **A** (`prediction frame t_d`): passer state, local pressure, and option-set aggregates over *all* teammates. No receiver is assumed, so **no
  target-identification exclusion exists**.
* **B** (`t_p = t_release + 6 frames`): the aimed-at teammate is identified from the ball's heading over those 6 frames (`assign_target_by_heading`,
  correct for 86-87% of assigned completed passes), then features are computed at t_p using frames <= t_p only.

## 2. Outcome labels (event type/subtype only, verified on both games)

* **Completed** = `PASS`, no subtype, receiver present, positive duration (G1 755, G2 916).
* **Intercepted** = `BALL LOST` with subtype exactly `INTERCEPTION`, passer present; that event *is* the failed pass (G1 128, G2 114).
* **Unknown, excluded and counted** (G1 173, G2 167 events): `PASS` with a subtype (aerial/special), `BALL LOST` with any other subtype
  (THEFT, HEAD, CLEARANCE, none, ...: **never** treated as interceptions), cross/head/goal-kick interceptions, passes without receiver or duration.
  Other failure modes (out of play, mis-controls) therefore do not appear: the target is *completed vs intercepted*, not completed vs any failure.
* Labels are **not** conditioned on following events. Verification only: 95.0% (G1) / 92.0% (G2) of completed passes are followed by an action of the
  receiver; 89.1% / 91.2% of interceptions are followed by an opponent event. Filtering on these would, e.g., drop completed passes whose receiver was
  immediately challenged and bias the completed class toward unpressured receptions. Examples are printed in the results file.
* Set-piece passes (`SET PIECE` events) are outside the label universe.

## 3. Features (all with a plausible football meaning; 8 in A, 9 in B; registry `FEATURES` in `src/pass_completion_features.py`)

Interception margin uses the existing feasibility model (ball 11.9 m/s prior, 7.0 m/s, 0.3 s; the latter two assumed). Velocities are causal (5-frame
backward difference). Orientation: coordinates rotated so the passer's team attacks +x.

| Feature | Unit | Definition | Frames |
|---|---|---|---|
| **A** dist_goal, angle_goal | m, rad | passer to goal centre; angle subtended by the posts | t_d |
| A nearest_opp_dist, n_opp_10m | m, count | pressure on the passer | t_d |
| A passer_speed, nearest_opp_closing_speed | m/s | causal; closing > 0 = opponent approaching | t_d-5..t_d |
| A n_safe_options, best_margin | count, s | over **all** teammates >= 3 m: count with margin < 0; safest (most negative) margin | t_d |
| **B** pass_dist | m | release point to target position at t_p | t_rel, t_p |
| B early_ball_speed | m/s | ball displacement over the first 6 frames / 0.24 s (observable only at t_p) | t_rel..t_p |
| B margin_static / margin_vel | s | interception margin ball(t_p)->target(t_p); static vs causal-velocity opponents | t_p |
| B n_opp_corridor, min_opp_corridor_dist | count, m | opponents within 1.5 m of the remaining line; nearest distance to it | t_p |
| B receiver_speed, receiver_nearest_opp_dist | m/s, m | target teammate: causal speed (history only); marking distance | t_p-5..t_p, t_p |
| B forward_gain, dist_goal_target | m | passer-minus-target distance to goal (forward = +); target distance to goal | t_rel, t_p |

Never read by any feature: event end position/time/frame, interception location, receiver identity (`to_player`), later events. Enforced structurally
(`ALLOWED_EVENT_COLUMNS` = team, period, start time, passer) and by tests.
Not used: ball speed at t_d (the ball is at the passer's feet; the 11.9 m/s population prior is used only inside the margin); receiver
extrapolation/lead (would need a validated motion model).

## 4. Models and protocol

Base rate; distance baseline (**B only**: `pass_dist`; A has no pass distance, so A uses a location-only baseline); existing-feasibility baseline
(B: `margin_static`; A: option-set aggregates); **primary ridge logistic regression** (numpy, no new dependency); **exploratory** depth-2 boosted trees (own
~60-line deterministic implementation; scikit-learn is not installed and one exploratory model did not justify the dependency). No class
weighting, SMOTE or resampling: the test class mix is untouched.
Everything fitted on Game 1 only: winsorise (1st/99th pct) + standardise, ridge strength, feature-set choice (simplest set within 1 SE of the best
5-block contiguous-time CV log-loss, fixed *before* seeing Game 2), boosting rounds, Platt calibration on out-of-fold predictions, and the F1 threshold for
"intercepted". Test-set calibration slope/intercept are diagnostics and are never applied back. Nothing was tuned on Game 2.

## 5. Samples and exclusions

| | completed kept / total | intercepted kept / total | exclusion rate (completed / intercepted) |
|---|---|---|---|
| A train G1 | 746 / 755 | 126 / 128 | 1.2% / 1.6% |
| A test G2 | 912 / 916 | 111 / 114 | 0.4% / 2.6% |
| B train G1 | 442 / 755 | 45 / 128 | 41.5% / 64.8% |
| B test G2 | 554 / 916 | 37 / 114 | 39.5% / 67.5% |

A's few exclusions: ball not with passer (7/2, 4/3), no velocity history (2/0). B's: ambiguous receiver (G2 228 / 29), no teammate in heading cone / passes
into space (110 / 37), ball displacement too small (18 / 3), no ball track after release (0 / 5), other (6 / 3).

## 6. Held-out Game 2: scenario A (decision time; n=1023; 111 intercepted; no-skill PR-AUC 0.109)

| Model | ROC-AUC | PR-AUC (intercepted) | Brier | Log-loss |
|---|---|---|---|---|
| Base rate | - | - | 0.098 [0.084, 0.112] | 0.349 [0.314, 0.384] |
| Location-only baseline | 0.461 [0.398, 0.524] | 0.116 [0.088, 0.163] | 0.098 | 0.349 |
| Feasibility-aggregate baseline = **primary** (A3: n_safe_options, best_margin) | 0.583 [0.529, 0.636] | 0.153 [0.115, 0.217] | 0.097 [0.083, 0.111] | 0.343 [0.308, 0.378] |
| Exploratory boosted trees (8 features, 25 rounds) | 0.645 [0.593, 0.696] | 0.173 [0.133, 0.229] | 0.096 [0.083, 0.110] | 0.339 [0.305, 0.374] |

* Selection: every candidate set was within 1 SE of the best on Game 1 CV (log-loss 0.404-0.414, SE 0.021): the sets are statistically indistinguishable
  in training; the rule returned the simplest (A3), which coincides with baseline 3.
* Primary vs base rate: log-loss -0.0061 [-0.0123, -0.0003] (significant, tiny); Brier -0.0012 [-0.0028, +0.0004] (not). Boosted trees vs base rate: Brier -0.0021
  [-0.0038, -0.0005], log-loss -0.0101 [-0.0159, -0.0046]; vs primary: ROC-AUC +0.063 [+0.004, +0.124], Brier/log-loss not significant.
* Calibration: primary slope 0.96, intercept -0.36 on logit P(intercepted): the model over-predicts interceptions because Game 1 has a higher
  base rate (14.4% vs 10.9%); 5 equal-count bins predicted 0.090-0.200 vs observed 0.063-0.137 (`pass_completion_reliability_A.png`). Boosted trees slope 1.41.
  Probabilities span only ~0.09-0.20: they carry almost no resolution beyond the base rate.
* Threshold (F1-optimal on Game 1 OOF, tau 0.873): TP 79, FP 530, FN 32, TN 382; precision 0.130, recall 0.712, F1 0.219 (a flag fires on 60% of passes).
  Boosted trees: TP 58, FP 274, FN 53, TN 638; P 0.175, R 0.523, F1 0.262.
* Ablation (ROC-AUC): location 0.461, pressure 0.526, feasibility 0.583, feasibility+movement 0.592, no-feasibility (6) 0.523, full 0.582.
  Feasibility aggregates beat location by +0.122 [+0.040, +0.199] AUC (log-loss -0.006, significant). Full minus no-feasibility: +0.059 [+0.003, +0.115] AUC,
  log-loss -0.005 [-0.012, +0.001] (not significant).
* Coefficients (primary, per +1 SD, 300-resample train bootstrap): n_safe_options +0.185 [+0.006, +0.350] (OR 1.20); best_margin -0.294 [-0.454, -0.101]
  (OR 0.75): completion is likelier when a clearly safe option exists. Football reading: the passer usually has an escape pass when one exists. This is
  an association; the tracked option set also reflects what the passer could see, and unobserved factors (body orientation, defender position relative to the
  passing line from the passer's view) are missing. Boosted trees rely mainly on dist_goal (49% of gain), a non-linear effect the linear model cannot use.
* By passer location (decision-time): attacking third 0.442 [0.312, 0.572], middle 0.613 [0.535, 0.687], defensive 0.557 [0.454, 0.657]. By post-hoc pass length
  (stratifier only): <10 m 0.634, 10-20 m 0.615, >=20 m 0.567 (all CIs wide, ~0.5-0.73).

## 7. Held-out Game 2: scenario B (early flight; n=591; 37 intercepted; no-skill PR-AUC 0.063)

| Model | ROC-AUC | PR-AUC (intercepted) | Brier | Log-loss |
|---|---|---|---|---|
| Base rate | - | - | 0.060 [0.044, 0.076] | 0.240 [0.197, 0.286] |
| Pass-distance baseline | 0.658 [0.558, 0.754] | 0.151 [0.086, 0.276] | 0.058 [0.043, 0.073] | 0.229 [0.188, 0.273] |
| Existing feasibility-margin baseline | 0.863 [0.809, 0.910] | 0.282 [0.178, 0.446] | 0.052 [0.039, 0.066] | 0.195 [0.157, 0.233] |
| **Primary** logistic (B5, 6 features, no feasibility) | 0.767 [0.671, 0.857] | 0.269 [0.157, 0.421] | 0.054 [0.041, 0.067] | 0.206 [0.161, 0.253] |
| Exploratory boosted trees (9 features, 50 rounds) | 0.884 [0.843, 0.921] | 0.313 [0.201, 0.462] | 0.051 [0.038, 0.064] | 0.176 [0.139, 0.216] |

* **The pre-declared selection rule chose a set that excludes feasibility.** On Game 1 CV the full set (0.225 ± 0.027) was best; the simplest set within 1 SE was B5
  (0.245). On Game 2 this primary is *worse in AUC* than the single-feature margin baseline: -0.096 [-0.188, -0.010] (significant), though Brier (+0.001) and
  log-loss (+0.012 [-0.010, +0.036]) are not significantly different. The rule was not changed after seeing Game 2. Takeaway: with 45 training positives, Game 1 CV
  cannot distinguish these sets, and the multi-feature logistic did not transfer as well as one geometric feature.
* Primary vs base rate: log-loss -0.034 [-0.062, -0.005] (significant), Brier -0.006 [-0.013, +0.000] (not). Boosted trees vs base rate: Brier -0.009
  [-0.017, -0.002], log-loss -0.064 [-0.088, -0.041]; vs primary: AUC +0.117 [+0.051, +0.191], log-loss -0.030 [-0.050, -0.011] (exploratory; interaction
  structure in margin_vel / n_opp_corridor, not selected on Game 2).
* Calibration (3 equal-count bins; fewer because of 37 positives): primary slope 0.81, intercept -0.65; predicted 0.014 / 0.041 / 0.181 vs observed 0.025 / 0.015 /
  0.147 (`pass_completion_reliability_B.png`). Boosted trees slope 1.01, intercept -0.15. Directionally calibrated at this resolution; too few events to say more.
  Test interception rate is lower than in training (6.3% vs 9.2%).
* Threshold (F1-optimal on Game 1 OOF): primary tau 0.818 TP 19 FP 54 FN 18 TN 500, P 0.260 R 0.514 F1 0.345; feasibility baseline TP 22 FP 78 FN 15 TN 476, P 0.220 R 0.595
  F1 0.321; distance baseline P 0.161 R 0.378 F1 0.226; boosted trees P 0.367 R 0.297 F1 0.328.
* **Does feasibility add predictive value?** Yes in B. Margin alone vs pass distance alone: AUC +0.204 [+0.138, +0.273], log-loss -0.035 [-0.048, -0.022]. Full vs
  no-feasibility: AUC +0.049 [+0.005, +0.104], log-loss -0.011 (n.s.). Adding distance or movement to the margin does not improve it (B4 0.855, B3 0.854 vs B2 0.863).
  Subgroup by pass length: only >= 20 m has enough intercepted passes (19): AUC 0.748 [0.606, 0.869]; shorter groups have 4 and 14 and are not reported.
* Coefficients (primary, per +1 SD): forward_gain -0.63 [-0.96, -0.40] (OR 0.53; forward progression lowers completion), receiver_nearest_opp_dist +0.43 [+0.14, +0.72]
  (tight marking lowers it), dist_goal_target +0.41 [+0.15, +0.69], pass_dist -0.32 [-0.59, -0.03], early_ball_speed -0.27 [-0.54, +0.02], receiver_speed +0.03 [-0.27, +0.30].
  Correlated features (forward_gain / dist_goal_target / pass_dist) share signal and individual coefficients are unstable; none is a causal effect.

## 8. Selection bias (scenario B target filter)

* Retention: completed 59.0% (G1) / 60.7% (G2) vs intercepted 35.7% / 33.3%. The completion rate is 0.856 / 0.891 over all A candidates but 0.907 / 0.937 in the B set
  (excluded: 0.791 / 0.829): **B's data over-represents completions**, and every B base rate and Brier/log-loss is computed on that easier subset.
* Retained passes differ at decision time (Game 2, standardised difference retained - excluded): the option set was safer for retained passes (best_margin -0.36
  completed, -0.59 intercepted), and they start farther from goal (+0.20, +0.54); pressure on the passer is similar (|SMD| <= 0.13). Retained completed passes are longer
  (median 14.6 vs 11.6 m).
* Retention is only mildly predictable from decision-time state (ROC-AUC 0.619), so the filter is driven mostly by what the ball does after release (direction clarity),
  which is unobservable at decision time.
* Scenario A on Game 2 split by the B filter: retained 0.494 [0.400, 0.586] (37/554) vs excluded 0.585 [0.518, 0.658] (74/358). A's weak skill is not concentrated in clearly
  directed passes.
* **Representativeness: B is not representative** of all passes. Results do not apply to passes into space, ambiguous targets, or all attacking decisions. No re-weighting is
  attempted: the retention mechanism (post-release ball direction) is not observed at decision time, so any weights would be arbitrary.

## 9. Conclusions and limits

1. A valid *target-specific* decision-time completion model is not possible (receiver unidentifiable before release; failures carry no receiver).
2. The decision-time model that is possible (A) discriminates weakly (AUC 0.58; 0.65 exploratory) and yields probabilities within a few points of the base rate. It is
   not good enough to underpin Decision Quality scoring.
3. The early-flight model (B) shows that interception geometry has real predictive value, but it is not decision-time, covers ~1/3 of interceptions, has 45 / 37 positives,
   and its probabilities are only roughly calibrated (base rate shifts between games). AUC alone is not evidence of reliable probabilities; the Brier/log-loss gains over the
   base rate are modest (log-loss significant, Brier not, for the primary).
4. Two games, one competition; no player-level or Ronaldo inference. The ridge strength and feature choices are weakly identified.

**What would be needed:** several hundred+ failed passes with an intended receiver or target location (provider data that records it), tracking at release, and
multiple matches for train/validation/test; ideally a receiver-choice model validated on its own.
**Recommendation:** do not proceed to action-value or Decision Quality scoring on these probabilities. Next: add Sample Game 3 (EPTS format) as a third match to check the
transfer of scenario B and the stability of A, and prototype a validated receiver-choice model.

## 10. Tests and leakage protection

`tests/test_pass_completion.py` (25 tests): label rules and unknown-outcome handling; event-column whitelist; registry timestamps; A/B prediction frames and disjoint feature
sets; feature tables invariant to scrambled outcome columns (end point, `to_player`, ...) and to every frame after the prediction frame (incl. future receiver movement);
completed and intercepted pass through identical construction; heading-based target not outcome-derived; train/test separation (marker values never reach any
fit; held-out fold unaffected by its own labels); determinism; numerics (IRLS, boosted trees, metrics, calibration bins, 1-SE rule). **Meta/mutation tests** show the
harness fails on an injected end-point leak, a future-frame leak, and a pooled train+test fit. Manual mutations of the real builder (future passer speed, true-receiver
target, future receiver speed, whole-pass ball speed) each make the tests fail. Full suite: 86 passed.
