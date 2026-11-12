# Pooled leave-one-game-out validation of pass feasibility as a probability estimate

```bash
python -m src.pooled_validation --out reports/pooled_validation_results.txt --figdir reports     # ~20 s, deterministic
```
Needs Games 1-3 (`python -m src.cli download --game N --yes`). Raw output: `reports/pooled_validation_results.txt`; plots `reports/pooled_reliability_{A,B}.png`.
Features, labels and eligibility are unchanged (`pass_completion_features.py`); data as supplied (Game 3's second-half identity swap is *not* repaired; see `cross_game_validation.md`).
Scenario A (decision time, release frame, target-free) and B (early flight, release + 0.24 s, identified targets) are run and reported separately.

## 1. Methodology

Three folds per scenario: train on two games pooled, test on the third (6 experiments). Everything fitted (winsorising/scaling, ridge strength, logistic, Platt, pooled base rate) uses the two training games only; the held-out game is scored once. No held-out labels enter any deployable model.

Deployable models (the only ones whose held-out numbers are claims):
* **BASE**: completion rate of the pooled training passes (pass-weighted).
* **M2**: the existing baseline: ridge logistic + Platt on 5-block out-of-fold predictions (B: `margin_static`; A has no single score, so M2 is the existing two-feature option-set baseline `n_safe_options, best_margin`).
* **M3**: plain (unpenalised) logistic on ONE score: B `margin_static`; A `best_margin`, chosen a priori as the continuous analogue of B's margin and never selected on any test game.
* **M3x**: M3 + Platt fitted on *cross-game* out-of-fold predictions inside the training pool (model on training game a predicts b, and vice versa), the simplest training-only calibration that sees a between-game shift.

Diagnostics only (use held-out labels; not deployable, never reported as predictions): a logit shift of M3 so its mean matches the held-out rate (delta, "oracle intercept"), a slope+intercept refit on the held-out game ("oracle recalibration", an optimistic upper bound), per-game score-outcome fits, and a common-slope/game-specific-intercept fit on all three games. They separate "different base rates" from "unstable score-outcome relationship".
Metrics: ROC-AUC (ranking), PR-AUC for the interception class (no-skill = test rate), Brier and log-loss (probability accuracy) as paired differences versus BASE on identical samples, calibration slope/intercept of logit P(intercepted) (ideal 1/0; 95% bootstrap CIs over held-out decisions, 1,000 resamples; calibration CIs 300). `*` = CI excludes 0. Negative d = better than BASE.

## 2. Samples and base rates

| game | A: n (int./comp.) | A int. rate [Wilson 95%] | A excl. (comp./int.) | B: n (int./comp.) | B int. rate [Wilson 95%] | B excl. (comp./int.) |
|---|---|---|---|---|---|---|
| G1 | 872 (126/746) | 0.144 [0.123, 0.169] | 1.2% / 1.6% | 487 (45/442) | 0.092 [0.070, 0.121] | 41.5% / 64.8% |
| G2 | 1023 (111/912) | 0.109 [0.091, 0.129] | 0.4% / 2.6% | 591 (37/554) | 0.063 [0.046, 0.085] | 39.5% / 67.5% |
| G3 | 1088 (97/991) | 0.089 [0.074, 0.108] | 7.1% / 7.6% | 642 (36/606) | 0.056 [0.041, 0.077] | 43.2% / 65.7% |

Interception-rate heterogeneity: chi-square for equal rates A 15.1 (2 df, p = 0.0005), B 6.2 (p = 0.045). These treat passes as independent; passes within a match are not, so the p-values are optimistic. B keeps ~57-60% of completions but ~33-35% of interceptions in every game (clearly directed passes only; not decision time).

## 3. Results

Held-out experiments (train = the other two games). Training sizes and per-game training rates in the first columns.

### Scenario A (decision time)
| test | train n (int.) | train int. rates | test n (int.) rate | **M3 ROC-AUC** | PR-AUC (no-skill) | dBrier vs BASE | dLog-loss vs BASE | cal. slope | mean pred. P(int) vs observed |
|---|---|---|---|---|---|---|---|---|---|
| G1 | 2111 (208) | G2 .109, G3 .089 (pooled .099) | 872 (126) .144 | 0.597 [0.542, 0.649] | 0.218 (.144) | -0.0025 [-0.0046, -0.0004]* | -0.0093 [-0.0196, +0.0020] | 0.82 [0.38, 1.21] | 0.100 vs 0.145 |
| G2 | 1960 (223) | G1 .144, G3 .089 (pooled .114) | 1023 (111) .109 | 0.550 [0.496, 0.613] | 0.143 (.109) | +0.0003 [-0.0021, +0.0024] | +0.0028 [-0.0096, +0.0142] | 0.39 [0.10, 0.82] | 0.110 vs 0.109 |
| G3 | 1895 (237) | G1 .144, G2 .109 (pooled .125) | 1088 (97) .089 | 0.697 [0.642, 0.750] | 0.240 (.089) | -0.0030 [-0.0044, -0.0017]* | -0.0151 [-0.0204, -0.0098]* | 2.76 [1.99, 3.77] | 0.126 vs 0.089 |

Other deployable models (dBrier / dLog-loss vs BASE; cal. slope): M2 G1 -0.0028* / -0.0114*, 0.98; G2 -0.0004 / -0.0020, 0.57; G3 -0.0027* / -0.0122*, 2.37 (AUC 0.612 / 0.570 / 0.684). M3x G1 -0.0014* / -0.0073*, 2.34; G2 -0.0005 / -0.0027, 0.94; G3 -0.0020* / -0.0097*, 5.10.

### Scenario B (early flight; identified-target passes only)
| test | train n (int.) | train int. rates | test n (int.) rate | **M3 ROC-AUC** | PR-AUC (no-skill) | dBrier vs BASE | dLog-loss vs BASE | cal. slope | mean pred. P(int) vs observed |
|---|---|---|---|---|---|---|---|---|---|
| G1 | 1233 (73) | G2 .063, G3 .056 (pooled .059) | 487 (45) .092 | 0.829 [0.762, 0.884] | 0.292 (.092) | -0.0082 [-0.0147, -0.0026]* | -0.0509 [-0.0774, -0.0262]* | 1.12 [0.88, 1.50] | 0.066 vs 0.092 |
| G2 | 1129 (81) | G1 .092, G3 .056 (pooled .072) | 591 (37) .063 | 0.863 [0.810, 0.907] | 0.282 (.063) | -0.0065 [-0.0112, -0.0021]* | -0.0411 [-0.0582, -0.0240]* | 1.48 [1.18, 1.86] | 0.065 vs 0.063 |
| G3 | 1078 (82) | G1 .092, G2 .063 (pooled .076) | 642 (36) .056 | 0.776 [0.721, 0.825] | 0.118 (.056) | **+0.0064 [+0.0015, +0.0115]*** (worse) | -0.0035 [-0.0197, +0.0138] | 0.57 [0.39, 0.76] | 0.077 vs 0.056 |

Other models: M2 G1 -0.0081* / -0.0500*, 1.19; G2 -0.0063* / -0.0395*, 1.60; G3 +0.0057* / -0.0048, 0.59 (AUC 0.829 / 0.863 / 0.776). M3x G1 -0.0072* / -0.0453*, 1.46; G2 -0.0056* / -0.0353*, 1.91; G3 +0.0059* / -0.0044, 0.58.

**Ranking vs calibration vs skill, kept distinct.** *Ranking:* the score ranks interceptions above completions in all six folds (B 0.78-0.86 with CIs well above 0.5; A 0.55-0.70, the G2 fold's CI includes 0.5). *Calibration:* slopes are 0.39-2.76 (A) and 0.57-1.48 (B); only B-G1/G2 are near 1. *Skill vs BASE:* M3 beats the pooled base rate on both Brier and log-loss in 3/6 folds (A-G3, B-G1, B-G2), on Brier only in 1 (A-G1), not at all in A-G2, and is *worse* than BASE on Brier in B-G3 despite AUC 0.78.

## 4. Base-rate transfer analysis (central question)

Per-fold diagnostics (log-loss; "oracle" rows use held-out labels and are not achievable):

| fold | delta (logit offset to match the held-out rate) | log-loss M3 deployed | + oracle intercept | + oracle slope&intercept | share of the gap recovered by intercept alone |
|---|---|---|---|---|---|
| A, test G1 | -0.43 | 0.4143 | 0.4044 | 0.4040 | **96%** |
| A, test G2 | +0.01 | 0.3463 | 0.3463 | 0.3408 | **0%** |
| A, test G3 | +0.39 | 0.2919 | 0.2851 | 0.2764 | 44% |
| B, test G1 | -0.41 | 0.2656 | 0.2601 | 0.2595 | **91%** |
| B, test G2 | +0.05 | 0.1936 | 0.1935 | 0.1887 | **1%** |
| B, test G3 | +0.39 | 0.2156 | 0.2118 | 0.2043 | 34% |

Per-game relationship (descriptive, all three games, common scaling; change in logit P(completed) per +1 SD of the score, bootstrap 95% CI; AUC = ranking of interceptions by the score):

| | A `best_margin` slope | A AUC | B `margin_static` slope | B AUC |
|---|---|---|---|---|
| G1 | -0.445 [-0.650, -0.209] | 0.597 | -0.871 [-1.167, -0.678] | 0.829 |
| G2 | -0.254 [-0.514, -0.021] | 0.550 | -1.056 [-1.322, -0.811] | 0.863 |
| G3 | -0.986 [-1.344, -0.702] | 0.697 | -0.542 [-0.718, -0.358] | 0.776 |
| common slope; game intercepts (G1/G2/G3) | -0.511; +1.88 / +2.18 / +2.43 | | -0.813; +2.67 / +2.96 / +3.18 | |

(The common-slope / game-intercept fit is a diagnostic on all three games, not a prediction.)

Findings:
1. **Base rates differ by game and this matters.** Interception rates range 0.089-0.144 (A) and 0.056-0.092 (B). Where the held-out game is the extreme one (G1, rate highest; G3, lowest) a pooled-training base rate is off by |delta| ~ 0.4 logit units; where it is typical (G2) delta is ~0.
2. **But the base rate is not the whole story.** An intercept correction alone recovers 91-96% of the recoverable log-loss error in the two G1 folds, essentially none in the G2 folds (base rates already match, yet slopes are 0.39 (A) and 1.48 (B)), and a third to a half in the G3 folds.
3. **The score-outcome relationship is directionally stable but not stable in magnitude.** All per-game slopes are negative (riskier margin -> fewer completions) and the ranking persists after pooling, but slopes differ 2-4x (A: -0.25 vs -0.99; B: -1.06 vs -0.54) with non-overlapping CIs between G3 and G1/G2 in A and between G2 and G3 in B. A model fitted on two games therefore under-states the relationship in the held-out game where it is steeper (A-G3 slope 2.76) and over-states it where it is flatter (A-G2 0.39, B-G3 0.57).
4. **Conclusion on the mechanism:** poor probability transfer is *not primarily* one or the other. It is predominantly a base-rate shift when the held-out game has an extreme interception rate (G1 folds) and predominantly slope instability when the base rate matches (G2 folds); G3 folds show both. Neither the ranking value nor the sign of the relationship is in doubt.
5. **Does pooling help?** Modestly. M3 trained on the pooled pair is better than the average of the two single-game-trained M3 in 5/6 folds on Brier (by 0.0005-0.0009; worse by 0.0001 in B-G3) and 6/6 on log-loss (by 0.0006-0.0080), mostly by avoiding the worse single-game model (e.g. A-G1: train-G2-only log-loss 0.4109 vs train-G3-only 0.4324, pooled 0.4143). It does not fix calibration.
6. **Does a simple training-only calibration help?** No consistent gain. M2 (ridge + block-CV Platt) differs from plain M3 by <= 0.0007 Brier. M3x (cross-game Platt) is worse than M3 on Brier in 4/6 folds (better only in A-G2 and, trivially, B-G3), and its slopes are often worse (A-G3 5.10), because with only two training games the cross-game calibrator learns to shrink predictions when the held-out relationship is in fact steeper.
7. **Between-game variability cannot be estimated reliably.** The three deltas are about -0.4 / 0 / +0.4 in both scenarios (SD ~0.4 logit units, partly sampling noise in each held-out rate: SE ~0.1 (A), ~0.17 (B)). With three games the SD has 2 degrees of freedom, so its 95% interval spans roughly 0.5x to 6x the estimate. Likewise per-game slopes: three values.

## 5. Scenario B limitations (unchanged)

B is evaluated 0.24 s after release on passes whose aimed-at teammate could be identified from early ball flight. It excludes 40-43% of completions and 65-68% of interceptions in every game (see table in section 2), over-represents clearly directed, safer passes, has only 36-45 interceptions per game, and says nothing about passes into space or decision-time prediction.

## 6. Statistical caution

Only three games; the three folds share training data (each game is a training game in two folds and a test game in one), so they are not independent replications. CIs reflect sampling of passes within a held-out game only and ignore between-game variance, the dominant uncertainty here; no multiplicity correction. Passes within a match are correlated, making all CIs and the chi-square optimistic. Oracle diagnostics fit one or two parameters on the held-out labels and are optimistic by construction. Differences of 0.0005-0.001 in Brier are within noise. Treat the pattern across folds (consistent direction, inconsistent magnitude), not individual stars, as the evidence.

## 7. Conclusions

* **Ranking transfers; probabilities do not (yet).** The geometric score retains clear predictive value after pooling (B AUC 0.78-0.86; A 0.55-0.70), but its mapping to probability changes with the game: interception base rates differ (A 0.089-0.144; B 0.056-0.092) and the strength of the score-outcome relationship differs by up to 2-4x.
* **Scenario A vs B.** A is the only decision-time claim but is weak (AUC 0.55-0.70, no skill vs BASE in one fold, slopes 0.39-2.76). B ranks far better and its slopes are closer to 1, but it is not decision-time, covers a biased third of interceptions and still fails against BASE on Brier in one fold. Neither supports a transferable probability.
* **The evidence does not support building a transferable probability model from these data.** With three games, the between-game intercept and slope variability is the binding uncertainty and cannot be estimated.

## 8. Recommendation

Use the feasibility margin as a risk *score* (ranking/feature), not as a probability. The next research stage should add many more matches (ideally >= 10 with tracking + events) and fit a leave-one-match-out hierarchical (random-intercept, random-slope) logistic on the single score, to estimate between-match intercept and slope variance directly and to test whether any pre-match information explains the base rate. Without that, or without target-match labels for per-match recalibration, no transferable probability estimate is justified.

## 9. Tests

`tests/test_pooled_validation.py` (13): folds pool exactly the other two games and the base rate is the pooled, pass-weighted training rate; unequal sizes; single-score/scenario separation; held-out labels and features never reach the fitted models (marker values never reach any fit) *and the scored predictions are invariant to held-out labels*; meta-test that pooling the held-out game is detectable; cross-game Platt never uses a game's own labels; determinism; oracle diagnostics; Wilson/chi-square arithmetic; per-game relationship recovery; summary bounds. Manual mutations (held-out game in the pool, held-out base rate, held-out recalibration of M3) each make tests fail (the last was initially undetected and prompted an added test). Full suite: 117 passed.
