# Expert-panel protocol for Decision Quality (Stage 5 design)

Design only; no rater has been contacted and no rating collected. The purpose is to test whether "a good passing decision" has an **agreed reference** among football experts. Only if it does can it test whether the DQ prototype (`dq_pipeline_run.md`, section 9) tracks that reference. The protocol must be frozen (registered) before the first rating.

## 1. Questions, in order

1. **Agreement (gate G5a).** Do independent experts agree on the quality of the same decisions? If they do not, there is nothing to validate against, and the programme stops at this point regardless of the metric.
2. **Convergence (gate G5b).** Does `DQ_w` correlate with the experts' mean rating? Is the expert-preferred option the EV-best option more often than the choice-model baseline would predict?
3. **Diagnosis (descriptive).** Where do the experts and the metric disagree: risk vs progression, pressure, zone?

## 2. Material and blinding

* **Situations:** the 120 pilot situations in the appendix. They are drawn from open-play Metrica passes, excluding set-piece deliveries: 100 completed passes, 20 per `DQ_w` quintile and balanced over games, plus 20 intercepted passes, deliberately over-sampled so that risky choices appear. Seed 0. Games come out 43 / 34 / 43 because the 20 interceptions were drawn at random.
* **Stimulus:** a top-down animation from the tracking data (existing pitch plotting) from 3 s before the release frame, frozen at release. Teammates are numbered and the passer is highlighted.
* **Blinding:**
  * the pass is not shown and the outcome is never shown;
  * no metric value, chosen receiver, player identity or match context is shown (Metrica identities are already anonymous);
  * presentation order is randomised per rater;
  * 12 situations (10%) are repeated, to measure within-rater consistency.
* **Disclosure:** raters are told that some passes failed, but not which.

## 3. Raters and tasks

* **Raters:** at least 3, ideally 5, independent raters (licensed coaches, UEFA B or higher, or professional performance analysts), with no contact between raters during rating. A 10-situation calibration round uses situations outside the 120. It explains the scale, not "right answers".
* **Per situation:**
  * **(a) best option:** which numbered teammate is the best pass, or "no good pass";
  * **(b) quality of a named pass:** the chosen receiver is revealed only in this step, after (a) is locked, and rated 1-5 (clearly poor to clearly best available);
  * **(c) confidence:** 1-3;
  * **(d) reason:** one optional free-text tag (risk / progression / space / timing / other).

## 4. Pre-specified analysis and gates

* **G5a, agreement:** Krippendorff's alpha (interval) on rating (b). Required: point estimate >= 0.40 and lower 95% bound >= 0.30. Also reported:
  * agreement on (a), as Fleiss' kappa over option labels;
  * within-rater test-retest on the 12 repeats.
  
  If G5a fails, stop: report the disagreement profile and do not compare against the metric.
* **G5b, convergence** (only if G5a passes):
  * Spearman correlation of `DQ_w` with the mean rating: one-sided test of a positive correlation, alpha 0.05.
  * Top-1 agreement between the expert-modal best option and the EV-argmax option, against two baselines: the choice-model argmax, and "nearest teammate".
  
  The outcome must not be used. A secondary check repeats the correlation within completed passes only.
* **Interpretation:** expert agreement with the metric supports it as a model of expert judgement, not as truth. Disagreement is informative, for example if experts weight interception risk more than the value model does.

## 5. Sample size (simulation, `power.py`, seed 0)

5-point ratings from a latent quality plus rater noise; 2,000 simulated panels per cell.

| True single-rater reliability | Raters | Situations | Krippendorff alpha: mean [95% range] |
|---|---|---|---|
| 0.5 | 3 | 60 | 0.45 [0.28, 0.60] |
| 0.5 | 3 | 120 | 0.45 [0.34, 0.55] |
| 0.5 | 5 | 200 | 0.45 [0.38, 0.52] |
| 0.3 | 3 | 120 | 0.27 [0.15, 0.38] |

Correlation detection (two-sided 5%, 80% power): rho 0.4 needs 47 situations, 0.3 needs 85, 0.2 needs 194. Averaging raters attenuates the observed correlation: with 3 raters at reliability 0.5 the observed value is 0.87 x the true one; at 0.3 it is 0.75 x.

**Recommendation.**
* **Pilot:** 120 situations x 3 raters. This estimates alpha to about ±0.1, enough to decide G5a, and detects a correlation of about 0.3.
* **Main study, if the pilot passes:** 200 new situations x 5 raters, ideally from new matches. Detecting a modest correlation (0.2-0.25) needs this size.
* **Workload:** about 120 x 1.5 min, roughly 3 hours per rater for the pilot.

## 6. Known limitations

* **Pitch-only view.** Raters see a top-down animation, not video, so body orientation, the first touch and calls from teammates are not visible. The metric does not see these either.
* **Old data.** Metrica's three matches have been analysed extensively in this programme, so the pilot is not a fresh-data test of the metric. It tests the reference (G5a) and is only a first look at convergence.
* **Selection and regeneration.** Interceptions are over-sampled, and only those with a heading-identified target are eligible (selection described in `dq_pipeline_run.md`). The researcher key (outcome, `DQ_w`, quintile) is held separately and is regenerated deterministically by the scratch pipeline.

## Appendix: pilot situations (no outcome or metric shown)

| # | Game | Period | Clock | Team | Passer | Event index |
|---|---|---|---|---|---|---|
| 1 | G1 | 1 | 00:11 | Away | Player22 | 6 |
| 2 | G1 | 1 | 00:44 | Away | Player17 | 23 |
| 3 | G1 | 1 | 04:41 | Home | Player3 | 92 |
| 4 | G1 | 1 | 05:00 | Away | Player19 | 104 |
| 5 | G1 | 1 | 05:04 | Away | Player18 | 105 |
| 6 | G1 | 1 | 06:13 | Away | Player19 | 131 |
| 7 | G1 | 1 | 09:49 | Home | Player8 | 187 |
| 8 | G1 | 1 | 09:55 | Away | Player18 | 193 |
| 9 | G1 | 1 | 12:29 | Away | Player17 | 242 |
| 10 | G1 | 1 | 15:49 | Away | Player21 | 318 |
| 11 | G1 | 1 | 15:54 | Home | Player9 | 324 |
| 12 | G1 | 1 | 18:28 | Away | Player19 | 375 |
| 13 | G1 | 1 | 25:31 | Away | Player21 | 508 |
| 14 | G1 | 1 | 34:51 | Home | Player4 | 647 |
| 15 | G1 | 1 | 36:35 | Home | Player8 | 691 |
| 16 | G1 | 1 | 39:29 | Home | Player12 | 738 |
| 17 | G1 | 1 | 43:02 | Away | Player22 | 802 |
| 18 | G1 | 1 | 43:25 | Home | Player12 | 813 |
| 19 | G1 | 1 | 44:43 | Away | Player16 | 840 |
| 20 | G1 | 2 | 48:07 | Home | Player6 | 894 |
| 21 | G1 | 2 | 48:18 | Away | Player23 | 901 |
| 22 | G1 | 2 | 48:41 | Away | Player15 | 910 |
| 23 | G1 | 2 | 49:45 | Away | Player20 | 930 |
| 24 | G1 | 2 | 50:34 | Home | Player8 | 948 |
| 25 | G1 | 2 | 55:28 | Away | Player23 | 1037 |
| 26 | G1 | 2 | 55:56 | Away | Player19 | 1047 |
| 27 | G1 | 2 | 56:20 | Home | Player9 | 1059 |
| 28 | G1 | 2 | 65:57 | Home | Player8 | 1207 |
| 29 | G1 | 2 | 67:23 | Away | Player15 | 1220 |
| 30 | G1 | 2 | 68:48 | Home | Player7 | 1244 |
| 31 | G1 | 2 | 72:44 | Away | Player20 | 1337 |
| 32 | G1 | 2 | 72:51 | Away | Player19 | 1339 |
| 33 | G1 | 2 | 78:09 | Home | Player2 | 1427 |
| 34 | G1 | 2 | 82:16 | Home | Player4 | 1506 |
| 35 | G1 | 2 | 82:55 | Away | Player16 | 1521 |
| 36 | G1 | 2 | 85:16 | Home | Player12 | 1567 |
| 37 | G1 | 2 | 86:32 | Home | Player2 | 1591 |
| 38 | G1 | 2 | 86:53 | Away | Player17 | 1600 |
| 39 | G1 | 2 | 87:00 | Home | Player14 | 1603 |
| 40 | G1 | 2 | 87:41 | Home | Player5 | 1614 |
| 41 | G1 | 2 | 88:24 | Home | Player12 | 1630 |
| 42 | G1 | 2 | 89:11 | Away | Player27 | 1644 |
| 43 | G1 | 2 | 95:15 | Away | Player27 | 1732 |
| 44 | G2 | 1 | 06:06 | Home | Player8 | 146 |
| 45 | G2 | 1 | 15:26 | Away | Player17 | 363 |
| 46 | G2 | 1 | 15:53 | Away | Player24 | 374 |
| 47 | G2 | 1 | 16:46 | Home | Player6 | 397 |
| 48 | G2 | 1 | 19:29 | Home | Player4 | 465 |
| 49 | G2 | 1 | 24:17 | Home | Player6 | 554 |
| 50 | G2 | 1 | 30:20 | Home | Player2 | 673 |
| 51 | G2 | 1 | 30:30 | Home | Player5 | 683 |
| 52 | G2 | 1 | 30:53 | Away | Player19 | 691 |
| 53 | G2 | 1 | 33:56 | Home | Player4 | 782 |
| 54 | G2 | 1 | 34:20 | Home | Player9 | 797 |
| 55 | G2 | 1 | 35:17 | Away | Player19 | 821 |
| 56 | G2 | 1 | 37:59 | Away | Player20 | 855 |
| 57 | G2 | 1 | 39:53 | Home | Player8 | 902 |
| 58 | G2 | 1 | 42:01 | Away | Player20 | 956 |
| 59 | G2 | 1 | 43:45 | Home | Player7 | 991 |
| 60 | G2 | 1 | 44:13 | Away | Player19 | 1011 |
| 61 | G2 | 2 | 46:33 | Away | Player18 | 1057 |
| 62 | G2 | 2 | 48:11 | Home | Player11 | 1083 |
| 63 | G2 | 2 | 49:16 | Home | Player10 | 1115 |
| 64 | G2 | 2 | 52:13 | Home | Player1 | 1162 |
| 65 | G2 | 2 | 54:03 | Home | Player6 | 1217 |
| 66 | G2 | 2 | 60:05 | Away | Player24 | 1334 |
| 67 | G2 | 2 | 61:16 | Home | Player3 | 1361 |
| 68 | G2 | 2 | 61:48 | Home | Player4 | 1377 |
| 69 | G2 | 2 | 62:32 | Home | Player5 | 1406 |
| 70 | G2 | 2 | 64:11 | Home | Player5 | 1441 |
| 71 | G2 | 2 | 64:18 | Home | Player9 | 1443 |
| 72 | G2 | 2 | 74:18 | Away | Player15 | 1657 |
| 73 | G2 | 2 | 82:36 | Home | Player5 | 1745 |
| 74 | G2 | 2 | 82:39 | Home | Player6 | 1748 |
| 75 | G2 | 2 | 86:07 | Home | Player3 | 1811 |
| 76 | G2 | 2 | 89:15 | Away | Player16 | 1856 |
| 77 | G2 | 2 | 91:22 | Home | Player13 | 1910 |
| 78 | G3 | 1 | 02:11 | Away | P3589 | 90 |
| 79 | G3 | 1 | 07:25 | Away | P3594 | 261 |
| 80 | G3 | 1 | 09:44 | Away | P3588 | 374 |
| 81 | G3 | 1 | 10:55 | Away | P3588 | 423 |
| 82 | G3 | 1 | 11:47 | Home | P3568 | 458 |
| 83 | G3 | 1 | 11:51 | Home | P3575 | 463 |
| 84 | G3 | 1 | 12:05 | Home | P3573 | 471 |
| 85 | G3 | 1 | 15:33 | Home | P3568 | 607 |
| 86 | G3 | 1 | 17:36 | Away | P3589 | 680 |
| 87 | G3 | 1 | 24:26 | Away | P3592 | 920 |
| 88 | G3 | 1 | 26:22 | Home | P3573 | 1008 |
| 89 | G3 | 1 | 27:28 | Home | P3571 | 1070 |
| 90 | G3 | 1 | 28:22 | Home | P3573 | 1120 |
| 91 | G3 | 1 | 37:01 | Home | P3574 | 1421 |
| 92 | G3 | 1 | 37:36 | Away | P3595 | 1442 |
| 93 | G3 | 1 | 39:39 | Away | P3588 | 1526 |
| 94 | G3 | 1 | 43:40 | Away | P3589 | 1687 |
| 95 | G3 | 1 | 44:29 | Home | P3571 | 1734 |
| 96 | G3 | 1 | 45:01 | Home | P3568 | 1754 |
| 97 | G3 | 2 | 49:44 | Away | P3586 | 1908 |
| 98 | G3 | 2 | 50:26 | Home | P3572 | 1928 |
| 99 | G3 | 2 | 53:32 | Away | P3590 | 2051 |
| 100 | G3 | 2 | 55:04 | Home | P3568 | 2104 |
| 101 | G3 | 2 | 55:16 | Away | P3592 | 2116 |
| 102 | G3 | 2 | 56:06 | Away | P3596 | 2151 |
| 103 | G3 | 2 | 56:14 | Away | P3594 | 2164 |
| 104 | G3 | 2 | 56:49 | Away | P3588 | 2197 |
| 105 | G3 | 2 | 58:56 | Away | P3587 | 2294 |
| 106 | G3 | 2 | 59:01 | Away | P3595 | 2298 |
| 107 | G3 | 2 | 64:13 | Away | P3696 | 2440 |
| 108 | G3 | 2 | 70:43 | Home | P3568 | 2663 |
| 109 | G3 | 2 | 70:47 | Home | P3579 | 2665 |
| 110 | G3 | 2 | 72:49 | Away | P3587 | 2772 |
| 111 | G3 | 2 | 74:37 | Home | P3572 | 2822 |
| 112 | G3 | 2 | 77:37 | Away | P3591 | 2947 |
| 113 | G3 | 2 | 78:16 | Away | P3698 | 2966 |
| 114 | G3 | 2 | 78:29 | Home | P3570 | 2973 |
| 115 | G3 | 2 | 79:21 | Away | P3697 | 3021 |
| 116 | G3 | 2 | 80:27 | Away | P3596 | 3061 |
| 117 | G3 | 2 | 80:53 | Home | P3572 | 3079 |
| 118 | G3 | 2 | 84:10 | Home | P3568 | 3173 |
| 119 | G3 | 2 | 90:29 | Away | P3596 | 3421 |
| 120 | G3 | 2 | 95:15 | Away | P3698 | 3604 |
