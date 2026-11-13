# Exploratory pass-choice modelling, Metrica Games 1-3

**Status: exploratory, descriptive. This is not a Decision Quality study.** It asks only: *using information available at the pass release frame, which admissible teammate does the passer choose?* All three games already influenced earlier work (design choices, thresholds, audits), so none is a clean holdout and every number below is exploratory. Failed passes are not used. Receiver identity (the provider's `to_player`) is used only as the choice label; admissibility and features are computed for every teammate before the label is looked up; no post-release frame, ball trajectory, outcome or future event enters any rule or feature.
Method files: scratch scripts outside the repository (`/tmp/claude-0/-home-user-DecisionQualityEngine/c4ca3e56-cf08-52a3-b16e-487a22fb013f/scratchpad/choice/`: `build.py`, `coverage.py`, `model.py`, `extra.py`) and a **timestamped pre-registration** (`PREREGISTRATION.txt`, 2026-11-13T10:29:27Z) written before any model result existed. No repository code changed, so the existing tests were not re-run. This is the only new file; nothing committed.

## 1. Executive verdict

* **A simple decision-time-only conditional-logit model predicts the chosen receiver well above chance and above the nearest-teammate rule, in all three held-out games.** Primary rule R0: top-1 accuracy **0.414-0.437** vs **0.125-0.132** for uniform choice (about 3.3x) and **0.312-0.364** for the nearest admissible teammate; paired gains over nearest are +0.118, +0.067, +0.103 top-1 (intervals exclude 0).
* **It is a weak-to-moderate predictor:** about 57-59% of choices are not the top prediction. Predicted probabilities are reasonably calibrated at the top choice, and log loss is 1.45-1.58 vs 2.04-2.10 for uniform.
* **The pre-registered features omit the passer's motion direction.** On moving passers the existing velocity-aligned rule is as good as the model (G2, G3) or slightly worse (G1). A post-hoc one-feature extension (not pre-registered) raises top-1 to 0.53-0.57. This is a hypothesis, not a result to rely on.
* **The candidate rules are plausible but biased:** R0 retains 96.9% of actual receivers (2,568 of 2,649), but the 81 excluded receivers are mostly long (median 40.8 m vs 13.9 m) and 72.8% forward vs 52.8%; a tighter range cap (R_tight) cuts coverage to 88.0% with clear role and zone bias.
* **This establishes predictability of observed choices only.** It says nothing about whether choices were good.

## 2. Dataset and exclusion counts

Population: completed ground passes (`PASS` without subtype, label = completed) that are existing Scenario-A eligible (decision-time state reconstructable). Every eligible pass has all 10 other teammates tracked at release.

| Stage | G1 | G2 | G3 | Total |
|---|---|---|---|---|
| completed ground passes (labelled) | 755 | 916 | 1,067 | 2,738 |
| Scenario-A eligible (state reconstructable) | 746 | 912 | 991 | **2,649** |
| actual receiver admissible under R0 and set size >= 2 (modelled) | 730 | 892 | 946 | **2,568** |
| excluded: receiver outside R0 | 16 | 20 | 45 | 81 (47 farther than 40 m, 25 closer than 3 m, 9 offside) |

Outside scope: failed passes, headers/crosses/set pieces, and passes with unknown outcomes (not labelled completed). Game 3's second-half identity swap is kept as supplied (it removes 76 of its completed passes at the possession filter, included in the eligibility counts above).

## 3. Candidate-set definition and coverage

**Pre-registered rules (a-priori constants, not estimated from data, never changed after seeing results). R0 (primary):** teammate tracked at release; 3 m <= distance from passer <= 40 m; not offside at release (receiver's attack-oriented x greater than both the second-last opponent's and the passer's, and greater than 0). **Sensitivity:** R_tight (3-25 m), R_loose (3-60 m), R_existing (>= 3 m only, no offside; the existing option rule). No risk, lane or pressure criterion is used in any rule (to avoid circularity). The 40 m cap is a stated football-judgement constant, not derived from the data, and its adequacy is exactly what the coverage tables test.

| Rule | Mean set size (median, IQR) | Coverage, all | G1 / G2 / G3 |
|---|---|---|---|
| **R0** | 8.04 (8; 7-9) | **96.9%** (95% Wilson 96.2-97.5) | 97.9 / 97.8 / 95.5 |
| R_tight | 5.06 (5; 4-6); 39 sets with <2 candidates | 88.0% (86.7-89.1) | 87.8 / 92.1 / 84.3 |
| R_loose | 9.58 (10; 9-10) | 98.7% (98.2-99.1) | 98.9 / 99.0 / 98.3 |
| R_existing | 9.97 (10; 10-10) | 99.1% (98.6-99.4) | 99.3 / 99.1 / 98.8 |

Coverage of actual receivers by passer pitch third and role proxy (tercile of the passer's match-average attack-oriented x within the team):

| Rule | Defensive / middle / attacking third | Deep / mid / advanced role |
|---|---|---|
| R0 | 96.1 / 97.4 / 96.4% | 96.3 / 97.2 / 97.6% |
| R_tight | **82.5** / 89.5 / 90.0% | **85.0** / 89.3 / 91.1% |
| R_loose | 99.2 / 98.9 / 97.4% | 99.1 / 98.5 / 98.3% |

**Excluded receivers (R0):** 47 too far, 25 too close, 9 offside. Examples: G1 passer Player19 to Player18 at 25.1 m, 12.4 m forward, flagged offside; Player15 to Player22 at 55.9 m, 22.5 m forward. Excluded receivers differ from covered ones: 72.8% vs 52.8% forward, median distance 40.8 m vs 13.9 m. The 9 offside exclusions are completed passes (so presumably onside at the whistle), which shows the position-only offside proxy errs. Reference only: the spec's illustrative 5% exclusion limit is met by R0 (3.1% excluded) and not by R_tight (12.0%); that limit is unjustified in the earlier reports and is not used here as a criterion.

## 4. Descriptive choice patterns (R0; associations, no causal claims)

"If uniform" = what a random choice among the same candidates would give on average.

| | G1 | G2 | G3 |
|---|---|---|---|
| passes | 730 | 892 | 946 |
| forward share: actual (if uniform) | 54.7% (52.7) | 51.2% (51.3) | 53.0% (56.0) |
| receiver distance, mean m: actual (candidates) | 15.2 (21.9) | 14.1 (21.7) | 16.1 (22.0) |
| median receiver distance, actual | 13.9 | 13.1 | 15.0 |
| mean forward gain, m: actual (candidates) | +1.0 (+0.6) | +0.4 (+0.1) | +0.6 (+2.2) |
| median receiver pressure (nearest opponent, m) | 8.4 | 7.8 | 8.3 |
| nearest teammate chosen / safest lane chosen | 31.9% / 33.3% | 36.4% / 31.1% | 31.2% / 33.9% |

* Passers choose clearly **nearer** receivers than average availability (about 14-16 m vs 22 m); the forward share is **close to what chance would give** (G3 even below), so progression preference is modest.
* **Passer pressure** (nearest opponent, pooled terciles at 2.9 m and 5.7 m): forward share 41.9% under high pressure, 53.5% medium, 63.1% low (G1/G2/G3 high: 45.5/37.1/43.9%; low: 65.7/62.5/61.7%); median distance 12.2, 13.2, 16.5 m.
* **Pitch third of the passer:** forward 66.1% defensive, 51.9% middle, 38.5% attacking; **role proxy:** deep players 67.9% forward, mid 47.8%, advanced 34.0% (positions confound everything here).
* Per-game conditional-logit coefficients (standardised, ridge 1, bootstrap over passes, all with the same sign pattern; descriptive associations): distance -2.07 / -1.99 / -1.61 (G1/G2/G3); forward +1.00 / +0.94 / +0.76; |lateral| +0.70 / +0.69 / +0.50; receiver pressure +1.34 / +1.10 / +1.14 (receivers with more space are preferred); interception margin -0.02 / -0.36 / -0.10 (intervals around 0: adds little once distance and receiver space are included). Distance, forward and lateral are geometrically related, so the individual coefficients are not separately interpretable. Between games the pattern is similar in sign, with G3 shows weaker distance and forward effects.

## 5. Baseline and model results by held-out game (R0, leave-one-game-out)

Primary model: conditional logit on dist, forward, |lateral|, receiver pressure, interception margin; existing `Prep` fitted on training candidates; ridge chosen by inner leave-one-training-game-out log loss (selected 1.0, 1.0, 0.1). Baselines: uniform (analytic), nearest admissible teammate, safest admissible lane (lowest existing margin) and the existing velocity-aligned rule on moving passers. No existing choice model exists in the repository. 95% bootstrap intervals over passes (1,000 resamples).

| Held-out (n) | Method | Top-1 | MRR | Log loss |
|---|---|---|---|---|
| **G1** (730) | uniform | 0.127 | 0.341 | 2.082 |
| | nearest | 0.319 [0.286, 0.352] | 0.546 [0.521, 0.570] | n/a |
| | safest lane | 0.333 [0.297, 0.363] | 0.563 [0.538, 0.585] | n/a |
| | **conditional logit** | **0.437** [0.401, 0.474] | **0.641** [0.618, 0.665] | **1.473** [1.407, 1.540] |
| **G2** (892) | uniform | 0.125 | 0.337 | 2.098 |
| | nearest | 0.364 [0.332, 0.393] | 0.578 [0.555, 0.599] | n/a |
| | safest lane | 0.311 [0.281, 0.341] | 0.553 [0.533, 0.575] | n/a |
| | **conditional logit** | **0.430** [0.398, 0.464] | **0.642** [0.621, 0.664] | **1.451** [1.403, 1.498] |
| **G3** (946) | uniform | 0.132 | 0.350 | 2.044 |
| | nearest | 0.312 [0.282, 0.342] | 0.536 [0.514, 0.558] | n/a |
| | safest lane | 0.339 [0.311, 0.371] | 0.571 [0.551, 0.593] | n/a |
| | **conditional logit** | **0.414** [0.382, 0.446] | **0.622** [0.600, 0.645] | **1.578** [1.505, 1.648] |

Paired model minus nearest: top-1 +0.118 [0.082, 0.153] (G1), +0.067 [0.031, 0.103] (G2), +0.103 [0.071, 0.135] (G3); MRR +0.096, +0.064, +0.086. Model minus uniform log loss: -0.607, -0.648, -0.466 (intervals exclude 0).
**Existing velocity-aligned rule (passers moving >= 1 m/s; n = 570 / 691 / 713):** top-1 0.409, 0.430, 0.431 vs model on the same subset 0.444, 0.414, 0.412 (uniform about 0.13): the model is no better than this single rule on moving passers in two of three games.
**Probability quality (pooled out-of-fold):** top-choice probability vs accuracy in five equal-count bins: 0.257 to 0.265, 0.340 to 0.360, 0.413 to 0.399, 0.505 to 0.485, 0.686 to 0.624 (mildly overconfident at the top); pooled log loss 1.504 vs 2.073 for uniform.
**End-to-end (receiver excluded or set < 2 counts as a miss):** top-1 0.428 / 0.421 / 0.396 (nearest 0.312 / 0.356 / 0.298; uniform about 0.12).
**Relative to candidate-set size (R0, pooled out-of-fold):**

| Set size | passes | model top-1 | nearest top-1 | uniform top-1 | model MRR | uniform MRR |
|---|---|---|---|---|---|---|
| <= 6 | 302 | 0.556 | 0.503 | 0.182 | 0.739 | 0.430 |
| 7 | 523 | 0.468 | 0.363 | 0.143 | 0.673 | 0.370 |
| 8 | 688 | 0.413 | 0.321 | 0.125 | 0.630 | 0.340 |
| 9-10 | 1,055 | 0.377 | 0.275 | 0.107 | 0.588 | 0.307 |

Accuracy falls with set size for every method; the lift over uniform stays about 3.1-3.5x, so accuracy must never be compared across rule sets without the set size.

## 6. Sensitivity and bias analysis

**Rule sensitivity** (top-1 held-out G1 / G2 / G3; model, nearest, uniform; paired model-minus-nearest in all nine cells exceeds 0 with intervals excluding 0):

| Rule | Model | Nearest | Uniform | Model/uniform |
|---|---|---|---|---|
| R0 | 0.437 / 0.430 / 0.414 | 0.319 / 0.364 / 0.312 | 0.127 / 0.125 / 0.132 | about 3.3x |
| R_tight | 0.493 / 0.472 / 0.488 | 0.352 / 0.382 / 0.346 | 0.218 / 0.216 / 0.226 | about 2.2x |
| R_loose | 0.425 / 0.432 / 0.406 | 0.316 / 0.360 / 0.303 | 0.105 / 0.103 / 0.106 | about 4.0x |
| R_existing | 0.422 / 0.429 / 0.402 | 0.312 / 0.360 / 0.302 | 0.100 / 0.100 / 0.100 | about 4.2x |

The ordering (model > safest lane or nearest > uniform) and the +0.07..+0.14 gain over nearest are stable across definitions; the absolute accuracy is not (tighter sets look better only because there are fewer candidates and fewer covered receivers). As a consistency check, nearest-teammate top-1 under R_existing in G1 (0.312) matches the earlier repository figure (0.308); the safest-lane figure differs (0.301 vs 0.343) because this study uses the velocity-aware margin and the repository check used the static one.

**Post-hoc extension (not pre-registered; chosen after seeing the velocity-rule result; the only variant tried):** primary plus one feature, passer velocity direction toward the candidate (zero when the passer moves slower than 1 m/s, the existing threshold). Top-1 0.548 / 0.570 / 0.529 vs 0.437 / 0.430 / 0.414; paired +0.111 [0.077, 0.147], +0.138 [0.104, 0.172], +0.113 [0.083, 0.142]; log loss -0.25 to -0.26; standardised coefficient about +1.0. Because it was chosen after seeing held-out results, treat it as a hypothesis (passers tend to pass in the direction they are already moving) to confirm on new matches.

**Biases and limits:**
1. **Success-conditioned:** only completed passes; riskier long passes complete less often, so the choice patterns are those of *successful* choices, not all choices. The receiver is the provider's receiver, which could differ from the intended one after a deflection (not measurable here).
2. **Coverage-conditioned evaluation:** the main tables use passes whose receiver is admissible; end-to-end figures are given for R0. Excluded receivers are long and forward, so long forward passes are systematically under-represented in the model evaluation, and R_tight would amplify role and zone bias (deep players, defensive third).
3. **Dependence:** passes share passers, possessions and tactical situations; the bootstrap treats passes as independent, so intervals are optimistic. With three games the match-to-match variability cannot be estimated.
4. **Missing information:** no body orientation, receiver runs, roles or tactics; receiver motion at release was not used; ties and tracking noise exist; Game 3 retains its identity defect.
5. **Not a clean holdout:** the games, the heading rule and the candidate range cap were all considered in earlier work; leave-one-game-out here is a generalisation check across matches, not an untouched test.

## 7. What the results do and do not establish

| Question | Answer from this study |
|---|---|
| **Predicting the receiver selected** | Established as exploratory: decision-time geometry (distance, direction of progression, space around the receiver, lane margin) predicts which admissible teammate is chosen about 3x better than chance and 0.07-0.12 (top-1) better than the nearest-teammate rule in every held-out game; ranking quality and probabilities are reasonable; the passer's motion direction appears to matter (post-hoc). |
| **Explaining observed player preferences** | Partly: passers prefer nearer, more open (less marked) and somewhat more forward receivers; stronger forward preference under lower pressure and for deeper players; patterns have a similar sign across games with different magnitudes. These are associations among features that are geometrically collinear, in success-selected passes, not causal effects. |
| **Evaluating whether the choice was good** | **Not addressed.** A high prediction score means choices are predictable from geometry; it does not show that the predicted choice is the better one (passers can be predictably suboptimal or predictable because they follow the same heuristics). No value, counterfactual, outcome or expert information is used. |

Also not established: the realism of the admissibility rules (only their coverage was measured), generalisation to other matches or leagues, behaviour on failed passes or passes into space, or any link to Decision Quality.

## 8. Recommended next step

1. Treat this as a descriptive baseline for the choice-and-coverage question; keep the pre-registered R0 results as the reference and report the velocity-direction finding as a flagged hypothesis only.
2. Before any further modelling, test the two cheap leads on **new, untouched matches** (an approved dataset), freezing in advance: the motion-direction feature and the coverage behaviour of the 40 m cap (and whether a role-specific cap is justified without using receiver identity).
3. Do **not** build a decision score from this model. If the programme continues, the next work is the specification-level revision recommended in the design review (identification assumptions, intent ground truth, value model, expert validation), not more choice modelling on the same three games.
