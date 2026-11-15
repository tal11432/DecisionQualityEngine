# Decision Score v1.3: an absolute 0-100 value for every shot, pass and carry on the whole pitch

Model-based and not validated by experts. Pre-registration: `research/dq/PREREG_DQ_v13.txt` (amendments A-L, each written before the data it concerns was scored; K and L were written after a defect or a choice found in the first whole-pitch run, and say so). Code: `research/dq/dscore/` (`core.py`, `score.py`); data scripts `fetch_v13*.py`, `fetch_v14.py`, `fetch_v15.py`, `fetch_v16.py`; model scripts `eval_v13_models.py`, `eval_v15_models.py`, `eval_v16_models.py`; validation `eval_v14.py` (attacking third, earlier) and `eval_v16.py` (whole pitch); tracking measurement `tracking_counter.py`; tests `tests/test_dscore_*.py`, `tests/test_tracking_counter.py` (160 pass, 1 expected failure). StatsBomb open data (attribution required), 360 competitions 2015-2024; test competitions never used for fitting: Euro 2024, Bundesliga 2023/24, MLS 2023 (+ Copa America 2024 for shots).

## What it is
* **Value of an option, Q** = expected net xG of the possession over the next 10 s (xG for us minus xG for the opponent), never the outcome. Shot (from x >= 60): xG plus the later shots of the possession. Pass to a visible teammate (ground, 3-40 m): xP x chain value - (1 - xP) x L(target). Carry (the one made): P_keep x V_pos(destination) - (1 - P_keep) x L(destination). **L** is the average cost of losing the ball at that place, given who is around.
* **Score = 100 x clip((Q + C) / (0.70 + C), 0, 1)**, C = 0.231 (the mean xG of a one-on-one): 100 is a move worth 0.70 xG or more; 0 is a decision that costs 0.231 xG in expectation. Q = 0 scores 25. The score of **every option** of the moment, the best option and the **gap** to the option taken (points and xG) are computed; a gap below the noise floor (0.03 xG, 0.05 for passes over 15 m) is "no clear difference from the best option".
* Q 0.01 -> 26, 0.06 -> 31, 0.08 -> 33, 0.32 -> 59, 0.62 -> 91, 0.70 and above -> 100. Whether a shot went in never enters.
* **Whole-pitch models (amendment J):** chain value of a pass target, keep-the-ball probability and destination value of carries are refit on all open-play passes and carries (no keeper distance, because the opposing keeper is not in the frame for 80% of events, 100% in our own third). Shots are offered from x >= 60 only, with a missing keeper assumed at (118, 40). Shot features are clipped at the range they were fitted on (amendment K).

## The range of the score (what you asked about)
| zone of the ball carrier | options taken: 1st - 99th percentile | lowest and highest | share of options taken >= 50 |
|---|---|---|---|
| own third (x < 40), 25,519 | 24 - 25 | 12 - 35 | 0% |
| middle (40-80), 57,743 | 24 - 27 | 15 - 62 | 0% |
| attacking (>= 80), 26,722 | 25 - 54 | 25 - 100 | 1.3% |
| all, 109,984 | 24 - 36 | 12 - 100 | 0.3% |

* **The top is reached**: a shot at an almost open goal (0.93 xG) scores 100 and a safe 4 m pass to a teammate with the goal open (0.91 xG) scores 100; a 0.62 xG shot scores 91; 99.9% of the decisions taken in the attacking third score 90 or less.
* **The bottom is not reached by data.** A pass straight into an opponent's feet at the edge of our box scores **25**, the same as a safe pass (the model sees an average cost of losing the ball there of 0.00-0.01 xG; the data say so: observed 0.010 xG in our own 30 units). The lowest options available in a moment (the worst real pass or carry) score 13-14 at the 1st percentile, and 7 if the hypothetical unforced turnover is counted; those low values come only through the marked assumption A1 below. Ordinary decisions sit at 25, because they are worth about 0.00-0.03 xG against a 0.93 wide scale: that follows from the definition, it is not stretched.
* The 5-95 practical range is therefore **half reached**: the high end yes, the low end no. I have not rescaled anything; a stretch of the scale would be a change of definition and is your decision.

## The counter-attack end: what the data can and cannot say
* **StatsBomb 360 (positions only):** the visible numerical balance does not predict opponent chances (56,108 losses). The model cannot see a counter-attack with a numerical advantage.
* **Tracking with velocities** (Metrica 1-3 and IDSSE J03WPY, 25 Hz, 803 losses, `research/dq/tracking_counter_out.txt`): a real counter with a numerical advantage (opponent runners at >= 2.5 m/s minus our players goal-side of the ball >= 2) occurred in **2 of 803 losses (0.25%)**; with advantage >= 1 in 9 losses, about +0.12 xG each. Not enough to measure (about 150 matches would be needed; the open tracking data holds 10), so C = 0.231 stays a stipulated anchor, and the position-only proxy has precision and recall 0 for the real counters.
* **Marked assumption A1** (implemented, flagged `counter_assumed`, switchable): a loss with a visible balance >= 2 at a point with x <= 70 costs at least 0.231 xG. On the whole pitch it touches **1.07% of the decisions taken** (own third 1.15%, middle 1.54%, attacking 0%), changes 0.74% of the scores by 2.3 points on average, and 8.6% of decisions have some option flagged. Where the flagged option was taken, A1 predicts -0.013 xG while the realised value was +0.004: **the assumption disagrees with the data and is shown only because it was the instruction**; all calibration gates are evaluated without it. Extending it to the attacking third with the same trigger would flag 27% of all targets and is not done.
* Stand-alone take-ons are not scored (with the frame taken before the duel nothing visible predicts the duel: AUC 0.55).

## What validated and what did not (109,984 test decisions: 56,787 passes, 50,758 carries, 2,439 shots), per zone
| Gate (model-only value, A1 off) | own third | middle | attacking | pooled |
|---|---|---|---|---|
| Calibration: Q of the option taken vs realised net xG of the next 10 s (ratio, or difference where the realised mean is below 0.01; Spearman over deciles) | **FAIL**: Q -0.0000 vs -0.0003, Spearman 0.88 (0.83 without failed passes) | **pass**: ratio 1.02, Spearman 1.00 | ratio 1.151: **fails by 0.0009** with failed passes; **pass** without them (1.08) | pass: 1.13 (1.06 without failed passes), Spearman 1.00 |
| Gap known-groups: inside terciles of the promise of the situation, big gaps realise less | **FAIL**: right direction 3/3 but the interval excludes 0 in 1/3 | **pass** 3/3 (-0.002, -0.005, -0.007 xG) | **pass** 3/3 (-0.004, -0.012, -0.053 xG) | pass 3/3 |
| Best-option sanity: shots with Q >= 0.30 (n = 187) | | | pass: median gap 0, 93% gap 0 | |
| No regression in the attacking third against the earlier validated run | | | **pass**: Spearman 0.982, mean absolute difference 0.003 xG | |
| Stability under 0.8 m position noise and 9% / 10% / 18% value scales | pass | pass | pass | bands of 0-1 points: uninformative about resolution |
| Models (fit on train, gates on test) | | | | chain value of passes: ratio 1.10, slope 0.95; P_keep for carries: AUC 0.84, ratio 1.00; V_pos: ratio 1.05: **all pass**. xP (pass success): frozen model over-spreads (slope 0.58 middle, 0.49 attacking); in our third xP is about 0.99 for nearly every pass so its slope cannot be identified |

* **Own third:** the score there does not carry validated information. Q is about zero for every option, the realised values are about zero, and neither the ordering of the deciles nor the gap known-groups test passes; the noise-floor gap (0.03 xG) occurs in 0.0% of own-third decisions.
* **Middle:** validates, but the gap above the noise floor occurs in only 1.1% of decisions.
* **Attacking third:** validates, and it is the only zone where verdicts are meaningful: the gap exceeds the noise floor in **20.1%** of decisions (shots 36%, carries 6%, passes 3% pooled over all zones).

## Two things found during this validation (stated because they changed the numbers)
1. **Amendment K, an extrapolation defect.** Without the keeper requirement, 360 frames with no opponent visible near the ball gave the shot model absurd values (0.98 xG from 38 units). About 0.1% of decisions, but it inflated the top of the range and the gaps. Shot features are now clipped at the 99.5th percentile of the fitted range; the validation was rerun.
2. **Amendment L, the xP recalibration.** The pre-registered per-zone recalibration of xP fixes the probability of the passes that were played but raises the value of the passes that were not (a modelled 0.3 becomes 0.57): selection bias, because players attempt low-probability passes with information the model lacks. Result: the gap above the noise floor rose from 20% to 26% in the attacking third on the same decisions. **The primary score uses the frozen xP** (as in all earlier runs); the recalibrated variant is reported as a sensitivity in `research/dq/eval_v16_xpcal_out.txt` (its gates are the same except the 1.14 calibration ratio). The choice was made after seeing both, so it is a researcher degree of freedom, disclosed.

## The two stored matches (whole pitch, CSV per match in `research/cases/`, no player names)
* Leverkusen 5-1 Darmstadt: 1,300 decisions; the five Leverkusen foot goals score **59, 31, 66, 33, 91**; only 48:08 has a clear gap (shot 31 against a pass worth 56: 0.23 xG). Of 1,300 decisions the gap exceeds the noise floor in 5%, almost all in the attacking third.
* Barcelona 5-1 Alavés: 1,508 decisions; goals **35** (a pass worth 75 was available: 0.38 xG), 29, 47 (Alavés), 68, 30 and 97; the others show no clear gap.

## Limits (read before using the numbers)
* **Not validated by experts**: the score is a model of expected xG from positions; it does not know intention, body orientation, speed or what the player saw.
* **The 10 s horizon undervalues build-up** in our own half and the middle: progression creates chances later, so options there differ by thousandths of xG and the score is dominated by the average cost of a loss. This is why the own-third and middle scores say little.
* **The low end of the scale is not data-supported** (see the range and counter-attack sections).
* Shots score higher by construction; compare the gap, not the score, across decision types. Alternatives are limited to ground passes of 3-40 m to visible teammates; lofted balls, through balls, crosses, clearances and take-ons are not options; hypothetical carry alternatives are not offered; the best option is a maximum over noisy values (the winner's curse flag reads 1.12).
* Calibration of the attacking third misses its gate by 0.0009 on all decisions (failed passes: the only target known is where the ball stopped). StatsBomb 360 shows only part of the players (80% of events have no keeper in the frame). No player-level aggregation or ranking has been made or should be made before validation.
