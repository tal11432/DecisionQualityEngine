# Pass Decision Quality (PDQ): feasibility audit on Metrica Games 1-3

Audit only. No metric implemented, no model/feature/label/threshold/validation change, no new data. The measurements below reuse the existing Metrica code read-only
(`build_features`, `label_events`, `synchronize_events`, `interception_margin`, `assign_target_by_heading`, the existing `DecisionContract` and feasibility parameters 11.9 m/s, 7.0 m/s, 0.3 s).
They were produced by scratch scripts kept outside the repository (`/tmp/claude-0/-home-user-DecisionQualityEngine/c4ca3e56-cf08-52a3-b16e-487a22fb013f/scratchpad/pdq_audit.py`, `pdq_summary.py`, ~13 s); the reconstructed eligible population reproduces the existing scenario-A sample exactly (2,983 passes).
Existing tests: **133 passed**. Files changed: this report only; nothing committed.

## 1. Executive verdict: **NO-GO** for PDQ as defined

`PDQ = quality(chosen pass) - best quality(alternatives)` is **not operationally measurable with the data and validated methodology we have**, and the one empirical check that is possible argues against the geometry-only version:

* **There is no "quality" term.** The only per-candidate quantity we have is a geometric *feasibility* (interception margin); it is not expected value, and its probability reading does not transfer across matches (pooled validation). Value needs xG/EPV-type modelling, which is out of scope.
* **The chosen receiver cannot be known at decision time** and is only recoverable post hoc, through a rule whose coverage depends on the outcome (60.5% of completed vs 35.3% of intercepted passes).
* **Subtracting the best alternative makes the score worse, not better:** on the 1,717 identified passes the ranking AUC for interceptions is **0.818** for the chosen pass's own margin but **0.733** for `margin_chosen - margin_best_alternative` (paired difference -0.084, 95% CI -0.112..-0.058), and the best alternative alone carries no information (0.519, CI 0.472..0.566). A feasibility-only PDQ would mostly measure "did the passer pick the geometrically safest option" (which is 23% of the time, typically a backward or very short pass), not decision quality.

What *is* supported today is narrower: reconstructing the decision-time state and a **descriptive option-risk profile** (per candidate, how interceptable each lane was). That is a measurement of options, not a score of decisions.

## 2. What can be measured reliably today

* **Decision-time state at the release frame** for 96.7% of completed and 96.3% of intercepted candidate passes (table in section 4). Uses frames up to the release frame only (existing contract, covered by the existing leakage tests); positions, ball, causal 5-frame velocities.
* **The option set:** all 10 other teammates are tracked at the release frame in every eligible pass (minimum 10), so no pass lacks players. With the existing rule (candidate = teammate at least 3 m from the passer) the median is 10 eligible options (minimum 8); no pass has fewer than 2.
* **A geometric interception margin for every candidate** at the release frame, with the same code and parameters. For *identified-target* passes it ranks interceptions above completions well (AUC 0.77-0.85 per game on this population; 0.818 pooled), a **ranking** statement; probabilities are not calibrated across matches.
* **The provider receiver of completed passes**: present for 100% and within the eligible option set for 99.1% (2,624/2,649; the other 25 are < 3 m from the passer and so not candidates under the existing rule).

## 3. What cannot be measured, and why

1. **Quality/value of a pass.** Feasibility is not value: a backward pass to an unmarked defender is the safest option and has little attacking value. No value model exists (xG/EPV excluded); the calibration of any completion probability is unstable across games.
2. **The chosen receiver for intercepted passes, and for passes into space.** Metrica gives no receiver for failed passes. The only available identification is the existing heading rule over the 6 frames *after* release (84.1% agreement with the provider receiver on completed passes, 1,348/1,602): it is label-side only, never a decision-time input, and its coverage differs by outcome (below).
3. **"Realistic" alternatives.** The existing candidate rule is geometric (any tracked teammate >= 3 m away). It has no range limit, direction, offside, role or reachability filter; a 44 m pass to a far-away teammate counts as an alternative. Any realism rule would be a new, pre-registered definition.
4. **Which action was *better*.** There is no ground truth for decision quality. Outcome is not proof of a good decision (a good pass can fail; a poor one can succeed), so outcome-based checks validate only the feasibility term.
5. **Match-level generalisation.** Three matches cannot estimate between-match variance (pooled validation).

## 4. Sample counts and exclusions at each stage

S0 = labelled candidates (completed `PASS` without subtype; intercepted `BALL LOST/INTERCEPTION`). S1 = decision-time state reconstructable (existing scenario-A eligibility: synchronised, passer tracked, ball within 3 m, velocity history, opponents tracked, option exists). S2 = at least 2 options. S3a = provider receiver in the option set (**exists for completed passes only**). S3b = chosen receiver identified by the heading rule (both classes) and in the option set. S4 = S3b with at least one alternative.

| Game | Class | S0 | S1 | S2 | S3a provider receiver | S3b = S4 heading-chosen |
|---|---|---|---|---|---|---|
| 1 | completed | 755 | 746 (99%) | 746 | 741 (98%) | 440 (58%) |
| 1 | intercepted | 128 | 126 (98%) | 126 | n/a | 45 (35%) |
| 2 | completed | 916 | 912 (100%) | 912 | 904 (99%) | 554 (60%) |
| 2 | intercepted | 114 | 111 (97%) | 111 | n/a | 37 (32%) |
| 3 | completed | 1,067 | 991 (93%) | 991 | 979 (92%) | 605 (57%) |
| 3 | intercepted | 105 | 97 (92%) | 97 | n/a | 36 (34%) |
| **All** | completed | 2,738 | 2,649 (96.7%) | 2,649 | 2,624 (95.8%) | **1,599 (58.4%)** |
| **All** | intercepted | 347 | 334 (96.3%) | 334 | **0** | **118 (34.0%)** |

(S1 in Game 3 is lower because of its second-half identity swap, kept as supplied.) Why the heading rule leaves passes unidentified (share of S1 passes):

| Reason | Completed | Intercepted |
|---|---|---|
| identified | 60.5% | 35.3% |
| ambiguous receiver (runner-up within 5 deg) | 25.5% | 24.0% |
| no teammate in the heading cone (pass into space / unidentified) | **11.6%** | **32.9%** |
| ball displacement too small | 2.4% | 5.7% |
| post-release ball track unavailable | 0% | 2.1% |

**Outcome dependence and bias.** (i) The provider route to a chosen receiver exists *only because* the pass was completed (100% outcome-dependent). (ii) The heading route is outcome-agnostic by construction but loses interceptions far more often (65% vs 40%), mainly passes into space, so the surviving population is biased toward clearly directed passes and toward fewer interceptions; (iii) identification uses frames after release, so it is not decision-time; it identifies *which action was chosen*, not an input to the score. (iv) 24-25% of passes have an ambiguous chosen receiver in both classes.

## 5. Chosen versus alternatives (descriptive, 1,717 identified passes with at least one alternative, 118 interceptions)

* Median 9 alternatives per pass. The chosen pass is the **safest** option in 23.0% of passes (completed 24.5%, intercepted 2.5%); median margin chosen -0.18 s vs best alternative -0.49 s; the best alternative is itself interceptable in only 8.9%.
* **Ranking diagnostic (not a PDQ score; ranking only, no calibration claim):** AUC for intercepted vs completed on the same samples, with 1,000-resample bootstrap CIs.

| | n | `margin_chosen` | `margin_chosen - best_alt` | best alternative alone | AUC(delta) - AUC(chosen) |
|---|---|---|---|---|---|
| G1 | 485 | 0.830 [0.773, 0.882] | 0.783 [0.718, 0.840] | 0.491 [0.414, 0.579] | -0.046 [-0.089, -0.003] |
| G2 | 591 | 0.848 [0.791, 0.896] | 0.793 [0.715, 0.857] | 0.432 [0.350, 0.518] | -0.055 [-0.102, -0.014] |
| G3 | 641 | 0.770 [0.718, 0.821] | 0.608 [0.527, 0.695] | 0.639 [0.560, 0.717] | -0.162 [-0.214, -0.111] |
| All | 1,717 | 0.818 [0.787, 0.848] | 0.733 [0.690, 0.777] | 0.519 [0.472, 0.566] | -0.084 [-0.112, -0.058] |

Reading: the alternatives term adds noise for ranking in every game (and the best alternative alone is near chance, 0.43-0.64). The intervals reflect pass sampling within a match only; with 36-45 interceptions per game and three matches they are not evidence about other matches. Ranking is reported here separately from calibration, which is not assessed (no probability is produced).

## 6. Concrete examples (decision-time quantities at the release frame; chosen receiver identified post hoc)

Attack-oriented coordinates (x increases toward the goal attacked), margin in seconds (>= 0 means some opponent can reach a lane point no later than the ball), causal velocities, existing parameters.

**Intercepted pass, Game 1, event 7 (13.8 s, period 1):** Away passer Player17 at (-1.0, 31.8), nearest opponent 6.2 m. Chosen receiver by the heading rule: **Player18** (24.9 m, margin **+1.04**, interceptable); the provider gives no receiver (failed pass).

| Candidate | dist (m) | (x, y) | margin (s) | |
|---|---|---|---|---|
| Player25 | 43.9 | (-34.0, 2.9) | -1.04 | safest, but a 44 m pass back |
| Player20 | 11.8 | (-6.5, 21.4) | -0.91 | safe, short, square/back |
| Player22 | 8.3 | (7.2, 33.1) | -0.34 | safe, short |
| Player21 | 20.5 | (19.5, 31.0) | +0.88 | interceptable |
| **Player18 (chosen)** | 24.9 | (19.1, 17.1) | **+1.04** | interceptable |
| Player19 / 16 / 15 / 24 / 23 | 27.5-46.0 | | +0.55..+2.85 | interceptable |

`margin_chosen - best_alt = +2.08 s`: the chosen pass was far riskier than the safest option, and it was intercepted. But the safest option is a 44 m back pass, which no analyst would call a realistic alternative, so the number would be driven by the candidate-set definition.

**Completed pass, Game 2, event 2 (5.8 s, period 1):** Away passer Player20 at (-7.9, -0.1), nearest opponent 1.6 m. Chosen receiver: **Player18** (16.6 m, margin **-0.41**), identified both by the heading rule and by the provider (`To`). Alternatives: Player16, Player17, Player15 at -0.44/-0.44/-0.40 (similar safety), Player25 -0.47 (43 m back); more advanced candidates (Player19, 22, 23, 24, 21) are interceptable (+0.04..+0.99). `margin_chosen - best_alt = +0.06 s`: essentially a tie among safe options, while the forward candidates were riskier: the measure cannot say whether staying safe or going forward was the better decision.

## 7. Minimum additional data or methodology required (none exists today)

1. **A pre-registered Q** with a value term (e.g. possession value or progression) validated against outcomes independently of feasibility, plus a calibrated completion model per match (currently not transferable).
2. **A pre-registered option-realism rule** (range, direction, offside, receiver role), applied identically to all passes.
3. **Intended-receiver labels for failed passes at release** (not in Metrica), or an explicitly validated receiver-choice model; otherwise chosen-action identification stays post hoc and outcome-dependent.
4. **Many more matches** (the pooled analysis called for 10+) to separate between-match variance from decision effects.
5. **An external reference for "good decision"** (expert annotation or another proxy), since outcome is not proof.

## 8. Recommended next step

Do **not** build PDQ. If the question is to be pursued, the smallest defensible step is a *written, pre-registered specification* (no data touched) fixing Q, the option-realism rule and the validation target, followed by a decision on whether a value model is permitted. Until then the supportable output is the descriptive **option-risk profile** per pass (candidate margins at release), presented as geometry, never as a decision score.

Uncertainty: all statements are for three matches and 36-45 interceptions per game in the identified subset; the examples are illustrative, not representative; coverage numbers depend on the existing heading rule (cone 15 deg, gap 5 deg) which was not altered.
