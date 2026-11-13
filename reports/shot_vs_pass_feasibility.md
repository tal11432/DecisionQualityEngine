# Shot vs pass: feasibility and value audit, Metrica Games 1-3

Audit only. No model trained, no xG value invented, no repository code changed, no new data. Counts come from scratch scripts outside the repository (`/tmp/claude-0/-home-user-DecisionQualityEngine/c4ca3e56-cf08-52a3-b16e-487a22fb013f/scratchpad/shots2.py`, `shots3.py`) that reuse the existing loaders, `_Track`, synchronisation, `interception_margin` and `FeasibilityParams` read-only. Context: `pass_decision_quality_feasibility.md`, `decision_quality_research_spec.md`, `decision_quality_design_review.md`. Nothing committed.

> **Update (section 8):** the missing components were later built and validated (`dq_pipeline_run.md`). An externally validated xG model is now available, along with a target-free pass-success model recalibrated on Metrica. The comparison was then run on the 50 shots, with the caveats below. The NO-GO below was the verdict *at the time of this audit*.

## 1. Executive verdict: **NO-GO** for "shot value vs best pass value"

The research question needs three quantities (shot xG, potential xG at a receiver, completion probability of a hypothetical pass). **None of the three can be produced defensibly from what exists:**

* **No xG model, and none can be trained here.** The repository contains no shot/xG code. The data has 68 shots, 50 after the scope filters, and **7 goals** among those 50 (10 in all 68). A fitted model would be dominated by noise (a handful of goals, no goalkeeper or defender-line labels, no shot-height information). Adopting a published xG formula would import external coefficients that this study cannot validate and that were not calibrated to Metrica coordinates. That would be a silent substitute, which the task forbids.
* **No calibrated completion probability for hypothetical passes.** The pass model ranks interceptions above completions but its probabilities do not transfer between matches (pooled validation); and it was validated for *identified targets*, not hypothetical ones.
* **Potential xG at a teammate is a counterfactual** (first touch, time to control, defender reaction). It has no label and no validation in any dataset available.

What is supportable is narrower: **reconstructing the pre-shot state and listing geometrically admissible receivers** (Section 4), as an option-geometry description, without values. A CONDITIONAL GO would require an external, validated xG model and a stated receiver-value definition (Section 7); the current data supports neither.

## 2. Shot sample and eligibility counts

All counts are `SHOT` events. Tracking synchronisation succeeded for all 68; the shooter is within 3 m of the ball at the event frame in 66 of 68 (median 0.11 m; two exceptions, 3.97 m and 29.4 m, are not excluded here but flag timing noise). All 10 teammates are tracked in every case.

| Stage | G1 | G2 | G3 | Total |
|---|---|---|---|---|
| shots | 24 | 24 | 20 | **68** |
| minus headers (subtype contains HEAD) | 18 | 17 | 19 | 54 |
| minus set pieces (penalty 1, free kicks 3: previous event) | 18 | 14 | 18 | **50** (open-play foot shots) |
| goals among the 50 | 2 | 3 | 2 | 7 |
| of the 50: preceded by a same-team PASS within 1 s | 8 | 6 | 5 | 19 |
| of the 50: preceded by recovery / carry / challenge / lost ball | 10 | 8 | 13 | 31 |

Remaining ambiguity not removable with current data: Metrica gives no ball height, so volleys and shots after crosses or deep balls cannot be separated from ground shots (the exclusion of crosses and through balls applies to the *hypothetical passes*, which the shot data cannot enforce for the shot itself; 7 of the 50 follow a CROSS event, others a DEEP BALL). Shot-event timing was validated for passes, not shots.

## 3. Candidate-pass definition (geometry only, fixed here, no value, no outcome)

Same admissibility as the choice study R0, computed at the shot event frame from tracking at or before it: teammate tracked, **3-40 m** from the shooter, not offside (position-only proxy: beyond the second-last opponent and the shooter, and in the opponent half). No rule uses the receiver's shot quality, lane risk, or the shot outcome.

Coverage across the 50 open-play foot shots:

| | G1 | G2 | G3 | Total |
|---|---|---|---|---|
| shots with >= 1 admissible teammate | 18/18 | 14/14 | 18/18 | **50/50** |
| mean admissible teammates per shot | 7.3 | 7.4 | 7.3 | 7.3 |
| shots with >= 1 candidate **closer to goal** than the shooter | 15 | 10 | 12 | 37 (74%) |
| ... and with the lane not interceptable by the existing geometry (margin < 0) | 5 | 3 | 7 | 15 (30%) |

Availability by shooting distance (pooled; n small):

| Shot distance to goal | shots | >= 1 closer candidate | >= 1 closer candidate, lane open (margin < 0) |
|---|---|---|---|
| < 12 m | 9 | 4 | 2 |
| 12-20 m | 18 | 11 | 4 |
| > 20 m | 23 | 22 | 9 |

By game, closer-candidate counts (<12 / 12-20 / >20 m): G1 2/3, 4/6, 9/9; G2 1/4, 2/2, 7/8; G3 1/2, 5/10, 6/6. So availability depends strongly on the shot zone (far shots almost always have a closer teammate; close shots rarely), and the per-zone counts are 2-10, too few to compare games.

Caution: "closer to goal" is a **crude position proxy, not a value**. A teammate closer to goal may be marked, offside-adjacent, or at a worse angle, and a candidate with margin near 0 is borderline rather than "open". The interception margin is a geometric feasibility, not a calibrated probability.

## 4. Availability of each required value component

| Component | Status | What exists | What is missing |
|---|---|---|---|
| Shot xG (actual shot) | **Not available** | shooter and ball positions, 11 defenders, shot subtype/outcome as label only (10 goals in 68) | an xG model (none in repo; 7-10 goals cannot support fitting); goalkeeper role (Metrica has none, would be inferred); shot height / body part beyond HEAD flag; the outcome must not be used for alternatives |
| Potential xG at a receiver | **Not available** | receiver position and surrounding opponents at release (measured) | the same xG model, plus a counterfactual model of the first touch, reception time, and defender response; no label for it exists |
| Pass completion probability, hypothetical pass | **Not available as probability** | interception margin per candidate; ranks intercepted above completed passes for identified-target passes (AUC about 0.8) | calibrated, match-transferable probabilities (pooled validation); validation for hypothetical rather than identified targets; no information on passes that were never attempted |
| Realistic receiver identification | **Partly** | geometric admissible set above; coverage 50/50 | a realism rule validated against intent (choice study: R0 keeps 96.9% of actual pass receivers but long forward passes are under-covered); expert or behavioural evidence that a candidate was a genuine alternative |
| Expected pass value = P(complete) x potential xG | **Not computable** | none | both factors. As noted in the task, even if computed it ignores the value of failed passes and any continuation after reception, so it would not be an action-value model |

## 5. Example situations (geometry only; no values are shown because none is supported)

Attack-oriented coordinates, release at the shot frame, existing causal velocities, existing margin parameters.

**Game 1, event 135 (SHOT, ON TARGET-SAVED, preceded by a team PASS):** shooter at (48.1, 8.9), 9.9 m from goal, nearest opponent 4.9 m. Admissible teammates (distance from the shooter; distance to goal; margin in s, negative means no opponent can reach the lane point in time; nearest opponent to the receiver): Player5 6.7 m, 6.0 m from goal, margin 0.0, opponent 1.2 m; Player9 14.2 m, 9.5 m, -0.0, 7.5 m; Player10 11.2 m, 19.4 m, +0.1, 3.7 m; Player6 and Player7 farther and interceptable. Player5 is closer to goal but tightly marked with a borderline lane, Player9 is as close to goal with more space but the lane margin is zero. The data can describe these facts; it **cannot say** whether passing would have had higher expected value than the shot.

**Game 2, event 1335 (SHOT, ON TARGET-SAVED, preceded by a team PASS):** shooter at (46.3, -11.4), 13.0 m from goal, nearest opponent 2.8 m. Closest candidate to goal: Player 26, 5.0 m away, 8.6 m from goal, margin -0.1, opponent 0.6 m (tightly marked); Player23, 20.1 m, 13.4 m from goal, margin +0.5, opponent 5.9 m (more space but the lane is interceptable). Again a geometric description only.

Neither example supports "a better option existed" or "the shot was correct"; they illustrate why the option set cannot be ranked without a value model.

## 6. Main methodological limitations

1. **Value components missing** (Section 4): this is the decisive limitation. Any number produced now would depend on an unvalidated substitute.
2. **Sample size:** 50 open-play foot shots, 7 goals, 15 shots with a plausibly open closer teammate. Per game and per zone cells hold 2-18 shots; no uncertainty statement beyond "very wide" is meaningful, and between-match variance cannot be estimated from three matches.
3. **Selection bias:** shots are taken precisely when the shooter judged them best or the only action; shots where a pass was chosen are not in this sample, so passing alternatives are observed only in situations that led to a shot (a choice-selected, outcome-dependent sample). Heavy shot-distance mix (23 of 50 beyond 20 m).
4. **Counterfactual validity:** teammates' runs, the receiver's body orientation and control time, the defenders' response and the shooter's own intent are unobserved; a "closer and open" teammate may be a passive position. Offside is a position-only proxy that errs both ways.
5. **Hindsight discipline:** all candidate features use frames up to the shot event only; the shot outcome (goal, saved, off target, blocked) was used only to count goals, never for alternatives. This is verified by construction here, not by a leakage test (no code was changed).
6. **Existing reports' limits carry over:** feasibility is not value, margin is a ranking tool whose probabilities do not transfer, and pass-decision quality has no ground-truth reference (design review). Game 3 keeps its identity swap as supplied.
7. **Event-detail limits:** no ball height, no cross or volley indicator beyond the preceding event, one ambiguous goalkeeper identity.

## 7. Minimum viable next experiment

Do **not** compute shot-vs-pass values on these three games. The smallest defensible path:

1. **Decide the value source first (no data touched):** pre-register that potential xG comes from an *external, published and validated* xG model applied unchanged (inputs available in tracking: distance, angle, defender count in the cone, goalkeeper position), and specify how a hypothetical receiver shot is defined (immediate one-touch shot from the receiver's position at arrival, no continuation).
2. **Obtain the validation basis:** a dataset with enough shots to check that model's calibration on this tracking format (hundreds of shots, with goalkeeper flags). This needs explicit approval for new data; it cannot be done within the present restrictions.
3. **Pass completion for the alternative:** use ranking only (or per-match calibration) with a stated sensitivity analysis, because the existing probabilities do not transfer; keep "expected pass value" labelled as a lower bound that ignores failed-pass and continuation value.
4. **Restrict the first output to a descriptive count:** in how many shots does a candidate exist whose estimated potential xG exceeds the shot xG even under optimistic completion assumptions (an upper-bound screen), reported with its uncertainty and without any per-player score.

Until steps 1-2 are approved, the supportable deliverable is the descriptive option geometry in Sections 3 and 5.

## 8. Update: the comparison, run on the 50 shots

Definitions were frozen before computing (`PREREG_SHOT.txt`, scratch). Positions come from the tracking frame of the shot event; nothing after it is used, and the shot outcome is used only to report goals.

* **xG** = probability that the actual shot scores. The xG model was trained on Premier League 2015/16 and passed calibration on four competitions.
* **xP** = probability that a pass to teammate j arrives. It uses the target-free interception-hazard model along the straight line, with the leave-one-game-out Metrica recalibration of the shot's game.
* **xA** = potential xG: the same xG model evaluated at the teammate's position, against the same defenders. This is the user's naming. Industry "xA" usually means the xG of a shot that actually followed a pass; here it is a hypothetical immediate shot.
* **EPV_j = xP_j x xA_j.** This ignores what a failed pass costs and anything after the reception. It is a lower-bound-style pass value, not a full action value.
* **Candidates:** teammates 3-40 m away, onside by the position proxy; median 8 per shot.

**Correction found while running:** the earlier Metrica xG check counted defenders in the shot triangle *without* the goalkeeper, while training counted him. With the training convention, the 50 shots sum to **xG 4.62** (previously 5.53) against 7 goals, which is still within the Poisson 95% range.

**Results.**
* **Overall:** 35 of 50 shots (70%) had a pass option with a higher EPV than the shot's xG. The median gap (best EPV - xG) is +0.041; the sums are xG 4.62 vs best EPV 6.27.
* **By shot distance:** under 12 m, 4 of 9; 12-20 m, 10 of 18; over 20 m, 21 of 23 (median xG of shots over 20 m is 0.018).
* **By game:** 13 of 18, 10 of 14, 12 of 18.
* **The best option** has a median xP of 0.84 and a median xA of 0.116.

**Sensitivity** (all frozen in advance):

| Variant | Shots with a better pass | Sum of best EPV (vs xG 4.62) |
|---|---|---|
| xP not recalibrated | 36 (72%) | 6.02 |
| candidates 3-25 m | 35 (70%) | 6.27 |
| one extra defender in the receiver's triangle (crude "defenders react") | 34 (68%) | **4.94** |

The *share* of shots with a better option is stable. The *size* of the advantage is not: with one extra defender, the summed advantage almost disappears (median gap +0.022). The result is therefore driven by the "receiver shoots immediately, defenders frozen" assumption.

**Examples** (distances in m; EPV = xP x xA):
* **Largest pass advantage.** G2 event 1666, 74:30, Player24, shot blocked; shot from 11.8 m, xG 0.062. Teammate Player 26 was 8.8 m away and 3.1 m from goal: xP 0.80, xA 0.64, EPV 0.51. A teammate unmarked in front of goal would be clearly better *if* the pass arrives and he can shoot at once.
* **Shot clearly best.** G1 event 1213, 66:01, Player9, goal; shot from 3.5 m, xG 0.73. The best pass had EPV 0.06.
* **Typical.** G1 event 1647, 89:21, a shot from 30.5 m, off target; xG 0.011. The best pass had EPV 0.053 (xP 0.77 x xA 0.07): a long-range shot where a simple pass to a teammate 16 m from goal was worth more.

**How to read this.**
* **Directionally,** long-range shots (over 20 m) were almost always less valuable than an available pass, while close-range shots were usually the right call. This matches common analytics findings, and it is the clearest, most robust part.
* **The magnitudes are optimistic for the pass.** Three effects push in that direction: the immediate-shot assumption with frozen defenders; taking the maximum over about 8 noisy options, which inflates the best by chance; and the failure cost being ignored.
* **Remaining limits:**
  * 50 shots and 7 goals, from three matches that earlier work already used;
  * xP was trained on ground passes; the goalkeeper is inferred;
  * the offside proxy is position-only;
  * no body orientation or control time.
* **Scope:** this is a geometric "what was on" comparison, not a judgement of individual players.

## 9. At scale: 7,292 shots from StatsBomb freeze frames

Pre-registration `research/dq/PREREG_SHOT_SB.txt`, written before downloading; scripts `research/dq/shot_vs_pass_sb.py` and `shot_vs_pass_sb_eval.py`; output `shot_vs_pass_sb_out.txt`.

**Data and method.**
* **Shots:** open-play foot shots from competitions *not* used to train xG: Premier League 2015/16 second half (190 matches), WC 2018 (64), Euro 2020 (51), WC 2022 (64); 369 matches in all.
* **Positions:** each shot's freeze frame.
* **Models:** xG and xA unchanged; xP is the target-free hazard model without recalibration (it was calibrated on StatsBomb data).
* **Data handling:** 110 MB transferred and decompressed in memory; the reduced table was deleted after the run.

| | Shots | Better pass existed | Median gap (best EPV - xG) |
|---|---|---|---|
| **All** | **7,292** (687 goals) | **72.9%** [71.7, 74.1] | +0.032 |
| Premier League | 3,945 | 73.0% [71.6, 74.4] | +0.032 |
| WC 2018 | 1,245 | 74.6% [72.1, 77.3] | +0.030 |
| Euro 2020 | 972 | 71.9% [68.3, 75.5] | +0.035 |
| WC 2022 | 1,130 | 71.8% [69.1, 74.6] | +0.032 |
| shots < 12 m (median xG 0.175) | 1,535 | **43.5%** [40.9, 46.1] | |
| shots 12-20 m (median xG 0.058) | 3,015 | 72.1% [70.4, 74.0] | |
| shots > 20 m (median xG 0.019) | 2,742 | **90.3%** [89.1, 91.5] | |

95% CIs come from a match bootstrap.
* **Gap distribution:** the best pass beat the shot by more than 0.02 xG in 59.7% of shots, by more than 0.05 in 37.0%, and by more than 0.10 in 17.8%.
* **The best option** has a median xP of 0.81, a median xA of 0.112, and is a median 10.1 m away.

**Sensitivity.**
* With one extra defender in the receiver's triangle: 64.9% better, median gap +0.016, sum 657 vs xG 625. By band: 36% / 62% / 85%.
* With candidates limited to 3-25 m: unchanged at 72.9%.

**Calibration check** (the only place the outcome is used): goals vs xG are 287 vs 274 (ratio 1.05) where a better pass existed, and 400 vs 351 (ratio 1.14) where none did.
* **Overall level:** our xG under-predicts these test sets by about 10% overall (625 vs 687; the earlier gate showed 0.89-0.95). xA uses the same model, so the comparison is partly protected.
* **Uneven across groups:** the slightly larger shortfall among close-range shots means the shot side is, if anything, under-valued there.

**Reading.**
* **The pattern is the same in all four competitions and in Metrica** (70% of 50 shots): long-range shots almost always had a more valuable pass available; close-range shots were the right call more often than not.
* **The headline share is robust; the size of the advantage is not.** It shrinks by half with one reacting defender.
* **The candidate set here is incomplete.** Freeze frames show only the players in view (median 4 visible teammates), so better options may have existed off-camera. That pushes in the opposite direction to the optimistic biases: the max over options, the immediate-shot assumption, and the ignored cost of failure.
* **This is a descriptive statement about situations, not players.**

## 10. Replication on 1,322 more matches (men and women, 2017-2025)

Pre-registration `research/dq/PREREG_SHOT_SB2.txt` (written before downloading); scripts `shot_vs_pass_sb2.py` and `shot_vs_pass_sb2_eval.py`; output `shot_vs_pass_sb2_out.txt`. Models, method and thresholds are unchanged from section 9 (xP not recalibrated). 392 MB were transferred and processed in memory; the reduced tables were deleted.

**Current season.** StatsBomb's open data does not contain 2026/27 matches. The most recent men's competitions are Euro 2024 and Copa America 2024 (summer 2024); the most recent women's is the Women's Euro 2025. Israeli Premier League matches are not in any open source. A current-season run needs licensed data.

**Which matches these are.** Several "league" samples are not whole leagues. I checked the match lists: every La Liga match (2017/18-2020/21) involves Barcelona; Ligue 1 2022/23 is Paris Saint-Germain's matches only; Bundesliga 2023/24 is Bayer Leverkusen's; MLS 2023 is Inter Miami's 6 matches. The shots in those samples come from both teams in each match, but the sample is selected by one club. International tournaments are complete.

**Men (7,357 shots, 801 goals, 12 competitions):**

| Sample | Shots | Better pass existed | < 12 m / 12-20 m / > 20 m | xG-to-goals ratio (xG sum / goals) |
|---|---|---|---|---|
| **Pooled** | **7,357** | **68.6%** [67.4, 69.7] | 40 / 69 / 90% | 0.86 |
| Euro 2024 | 1,029 | 75.2% [72.9, 77.5] | 46 / 76 / 91% | 1.02 |
| Copa America 2024 | 568 | 71.7% [68.5, 75.6] | 41 / 71 / 88% | 0.97 |
| AFCON 2023 | 869 | 72.2% [69.5, 74.6] | 44 / 71 / 91% | 1.01 |
| Bundesliga 2023/24 (Leverkusen) | 738 | 73.8% [70.8, 77.2] | 44 / 76 / 93% | 0.80 |
| Ligue 1 2021/22 and 2022/23 (PSG in 22/23) | 563 / 700 | 68.0% / 65.7% | 36 / 67 / 93% and 40 / 69 / 88% | 0.80 / 0.78 |
| La Liga 2017/18-2020/21 (Barcelona) | 770 / 700 / 586 / 667 | 63.8% / 60.6% / 64.0% / 66.4% | 35-39 / 60-72 / 83-92% | 0.88 / 0.76 / 0.77 / 0.86 |
| Champions League (2 matches), MLS 2023 (6 matches) | 24 / 25 / 118 | too few to assess | | |

**Women (secondary; the models were trained on men's data): 19,773 shots, 2,025 goals, 9 competitions.** The pooled share is 72.8% [72.0, 73.4], and the band shares are 47 / 75 / 90%. Per competition the shares range from 70.6% to 75.9%. The Women's Euro 2025 (the most recent) is 75.9% [71.7, 79.3].

**Replication criteria (men):**
* **R1, xG calibration: pass.** Slope 0.96. The mean xG is 0.86 times the observed goals, only just inside the 0.85-1.15 band. xG under-predicts by 14% on this data. Per sample the deficit is concentrated in the single-club samples (0.76-0.88), while the international tournaments are close to 1.0 (0.97-1.02). So the xG model is well calibrated for tournaments and low for strong clubs with elite finishers; part of that is selection, not model error.
* **R2, ordering by shot distance: pass.** In every sample with 150 or more shots, the share with a better pass rises from under 12 m to 12-20 m to over 20 m.
* **R3, shares near the first run: pass.** Pooled 68.6% (first run 72.9%), and by band 40 / 69 / 90% (first run 43.5 / 72.1 / 90.3%). All within tolerance, but the pooled men's figure is lower than in the first run, driven by the single-club La Liga and Ligue 1 samples (60-68%).
* **R4, defender sensitivity: fail.** Adding one extra defender cut the median gap by 49-57% in the tournaments, as predicted. In La Liga it cut it by 71-99% (median gap close to 0). In those samples the advantage of passing depends almost entirely on the "receiver shoots at once, defenders frozen" assumption.

**What the extra data shows.**
* **The distance pattern replicates.** It appears in all 12 men's samples, in 9 women's competitions, and across 2017-2025. This is the most robust finding of the whole study: long-range shots were nearly always less valuable than an available pass; close-range shots were usually the right call.
* **The overall share depends on who is playing.** In tournaments it is 72-75%; in the club samples dominated by one team it is 61-68%. A team that creates better close chances has fewer shots where a pass would have been better. This differs between teams and cannot be read as a league average.
* **The women's results match the men's closely,** although the models were trained on men's data. This is a partial check that the geometry generalises. The xG under-prediction is larger there (pooled ratio 0.89 of goals) and was not tested separately against women's data.
* **Still not a judgement of players or decisions.** The caveats of section 9 apply: freeze frames show only some of the players, the receiver is assumed to shoot immediately, and the cost of failed passes is ignored.

## 11. DQ-v5: a better xG, and an *empirical* pass value (the old xA was about 5 times too high)

Pre-registration `research/dq/PREREG_DQ_v5.txt` (written before the download); scripts `fetch_v5.py`, `eval_v5.py`, `research/cases/v5_cases.py`; outputs `eval_v5_out.txt` and `v5_cases_out.txt`. Data: StatsBomb open data, all men's competitions from 2015 (2,254 matches; 989 MB transferred and processed in memory, then deleted). The test competitions never enter any training set: Euro 2024, Copa America 2024, Bundesliga 2023/24 (Leverkusen) and MLS 2023 (Inter Miami, 6 matches).

**(A) A better xG. The pre-registered gate failed, narrowly.**
* The model was retrained on 50,093 open-play shots (21 competitions) with extra features: goalkeeper offset from the shooting line, nearby defenders, first-time shot, volley, play pattern, and the type of the preceding pass. Selected by leave-one-competition-out validation (17 features).
* **On 2,983 test shots (272 goals):** log-loss gain over the old model +0.0077 [+0.0032, +0.0126]; AUC 0.775 (old 0.757; StatsBomb's own xG 0.785). The goals-to-predicted ratio is 1.06 (the old model was 0.97 pooled, 0.83 on the Bundesliga sample, where it under-predicted).
* **The gate failed on calibration slope: 0.896 against a floor of 0.9 (compared unrounded).** The model is slightly over-confident at the extremes. The retrain with only the old five features gave no improvement (+0.0003, CI includes 0), so the gain comes entirely from the new features.
* A geometry-only version (no context fields, the one usable on Israeli data) has +0.0057 [+0.0020, +0.0093].

**(B) Empirical pass value ("xA_emp"). The gate passed.**
* **Definition:** the expected xG of the shot taken by the *receiver himself* as his next action (within 6 s, same possession), given a completed ground pass to that position in the attacking half. 28,736 passes (visible target and goalkeeper); 20,559 for training and 8,177 on the test set (Euro 2024, Bundesliga, MLS).
* **Pooled test:** predicted/observed 0.974, slope 1.014, log-likelihood gain over the constant +0.0108 [+0.0081, +0.0141]. Per competition: Euro 2024 slope 0.96, Bundesliga 1.09, MLS 1.16 (170 passes).
* **The old hypothetical xA (immediate shot from the target, defenders frozen) was 5 times too high** (mean 0.022 vs 0.0044 observed, slope 1.62), and the gap is large in every distance band:

| Target distance to goal | Observed | Empirical | Old hypothetical | n (test) |
|---|---|---|---|---|
| < 12 m | 0.205 | 0.198 | 0.245 | 60 |
| 12-20 m | 0.025 | 0.024 | 0.085 | 664 |
| 20-30 m | 0.003 | 0.003 | 0.030 | 1,991 |
| > 30 m | 0.000 | 0.000 | 0.009 | 5,462 |

**(C) Shot vs pass again (2,453 test foot shots).**

| | Better pass existed | Median gap | Sum xG vs sum of best pass |
|---|---|---|---|
| Old (xG M0 vs xP x hypothetical xA) | 73.7% | +0.034 | 205.9 vs 281.9 |
| **New (xG_new vs xP x empirical xA)** | **42.0%** | -0.007 | 233.7 vs 193.1 |

By distance (new vs old): under 12 m 36% (44%); 12-20 m 45% (74%); over 20 m 42% (91%). By competition (new): Euro 2024 45%, Bundesliga 40%, Copa America 40%, MLS 33%.

**How to read this.**
* **The earlier headline ("the pass was better in ~73% of shots") was largely an artefact of the optimistic xA.** With a calibrated pass value, a better pass existed in about 42% of shots, with no systematic advantage for passing. The distance pattern is much weaker: long shots are no longer almost always worse than a pass.
* **Neither number is "the truth".** The empirical value counts only the receiver's own immediate shot: it excludes shots by later teammates and the receiver's chance-creating pass, so it is a **lower bound** on a pass's value. The old value is an optimistic **upper bound**. The real answer lies between 42% and 74%, and this analysis cannot narrow it. To narrow it, the pass value must include a value for possession after the pass (xGChain/possession value).
* **The xP-based comparison is unchanged** (hazard model, not recalibrated).

**The Israeli frames with the new models** (official shot location, camera-side orientation; positions from the frames):

| | xG_geo | Official | Best pass (#90): xP x empirical xA | Better pass in |
|---|---|---|---|---|
| First shot (71:47) | 0.170 | 0.166 | 0.78 x 0.147 = **0.115** | 36% of draws |
| Second shot (71:48) | 0.457 | 0.378 | 0.98 x 0.495 = **0.486** | 42% of draws |

* The first shot matches the official xG almost exactly (0.170 vs 0.166).
* The second overshoots (0.46 vs 0.38; Monte Carlo 0.29-0.61).
* **The #90 value for the second shot (0.495) is an extrapolation:** few training passes end that close to goal (60 test passes within 12 m), and only some defenders are visible in the frame, so the features understate crowding. Treat it as an upper estimate.
* **The conclusion is unchanged: no pass clearly better than the shot in either case.** This now holds under both the optimistic and the empirical value.

## 12. DQ-v6: pass value over the same horizon as the shot (possession chain). Gate failed, but it narrows the answer.

Pre-registration `research/dq/PREREG_DQ_v6.txt`; scripts `fetch_v6.py`, `eval_v6.py`, `research/cases/v6_cases.py`; outputs `eval_v6_out.txt`, `v6_cases_out.txt`. 447 MB transferred (383 matches of the 360-frame competitions and Copa America 2024), processed in memory, tables deleted. Test sets: Euro 2024, Bundesliga 2023/24 and MLS 2023 for passes; those plus Copa America 2024 for shots. Nothing from the test sets was used for training.

**Definition.** Horizon 10 s, StatsBomb xG of open-play shots, outcomes never used.
* **Pass chain value:** the expected xG of *all* shots by the team in the same possession within 10 s after a completed pass (receiver or teammates). It fixes the limitation of section 11, which counted only the receiver's own shot.
* **Shot value:** xG plus a continuation term (later shots in the same possession within 10 s: rebounds and second phases).
* **Failed pass:** valued 0 (possession lost; the opponent's counter is ignored).

**Calibration of the chain value on the test sets: the pre-registered gate FAILED.**
* Pooled: observed 0.0140, predicted 0.0164 (ratio **1.18**, outside the 0.85-1.15 band); slope 1.03; log-likelihood gain over the constant +0.0106 [+0.0083, +0.0132].
* Per competition: Euro 2024 ratio 1.26 (slope 1.07), Bundesliga 1.09 (0.96), MLS 1.15 (slope **1.49** on 170 passes, above the allowed 1.4).
* So the model ranks well but **over-predicts the level by about 18%**, which favours passing. By distance the fit is close near goal (under 12 m: 0.231 predicted vs 0.246 observed; n = 60) and over-predicts mid-range (12-20 m: 0.062 vs 0.053).
* The continuation term passed its gate (predicted/observed 1.09). Its size is small: mean 0.012, against a mean shot xG of 0.094.

| Target distance to goal | Receiver-only (v5) | **Chain (observed)** | Old hypothetical | n |
|---|---|---|---|---|
| < 12 m | 0.205 | **0.246** | 0.245 | 60 |
| 12-20 m | 0.025 | **0.053** | 0.085 | 664 |
| 20-30 m | 0.003 | **0.019** | 0.030 | 1,991 |
| > 30 m | 0.000 | **0.005** | 0.009 | 5,462 |

Near the goal, the old hypothetical value happens to equal the observed chain value (0.245 vs 0.246). Farther out the old value was too high by factors of 1.6-2.

**Shot vs pass, 2,453 test foot shots (the pass value is bracketed on both sides):**

| Pass value used | Better pass existed |
|---|---|
| Lower bound: receiver's own next shot (v5) | 42.4% |
| **Primary: possession chain vs xG + continuation** | **63.6%** (median gap +0.013; the pass is better by more than 0.05 in 23.7%) |
| Diagnostic: the same, rescaled to the observed level (uses test labels, so not a prediction) | **55.9%** |
| Upper bound: old hypothetical xA | 70.9% |

By distance (primary): under 12 m 39%; 12-20 m 62%; over 20 m 80%. By competition: Euro 2024 66%, Bundesliga 62%, Copa America 62%, MLS 59%. For the rescaled diagnostic: 34% / 53% / 73%.

**Reading.**
* **As predicted before the run, the answer lies between the two bounds.** The best-supported value is about **56-64%**, closer to the upper bound than to the receiver-only lower bound (42%). A pass also helps the team through the possession after reception, so counting only the receiver's own shot undervalued it.
* **The distance pattern is back:** close shots are mostly right (the pass is better in about 1 of 3), mid-range shots are a toss-up (about 55-60%), and long shots usually have a better pass (73-80%).
* **The gate failure matters:** the over-prediction of 18% pushes toward passing, which is why the rescaled 56% is more honest than 63.6%. The remaining uncertainty, 56% vs 64%, is what this data cannot resolve. Notably, the rescaling uses test labels, so it is a sensitivity check, not a result.
* **Still not a judgement of a player or a decision.** The values are expected xG of the chain, with no value for lost possession, the cost of a failed pass, or possession after 10 s.

**The Israeli frames (official shot location, camera side; shot value = xG_geo + continuation; pass = xP x chain xA):**

| | Shot value | Official xG | Best pass (#90) | Better pass in |
|---|---|---|---|---|
| First shot (71:47) | 0.188 | 0.166 | 0.78 x 0.204 = **0.160** | 41% of draws |
| Second shot (71:48) | 0.479 | 0.378 | 0.98 x 0.508 = **0.498** | 39% of draws |

The conclusion is unchanged: in both cases no pass is clearly better than the shot (the best pass exceeds the shot in 39-41% of draws, inside the uncertainty). The value of 0.50 for the second shot is again an extrapolation near goal (few training passes end that close; only some defenders visible), so it is an upper estimate. Note that our shot value there (0.48) is also above the official xG (0.378).
