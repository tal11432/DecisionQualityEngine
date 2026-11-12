# Cross-game validation (Metrica Sample Games 1, 2, 3)

Reproduce (needs all three games: `python -m src.cli download --game N --yes`):
```bash
python -m src.cross_game --out reports/cross_game_results.txt --figdir reports      # data exactly as supplied (primary)
python -m src.cross_game --fix-identity --out reports/cross_game_results_fix_identity.txt   # sensitivity: detected tracking-identity swap applied
```
~100 s each, deterministic. Raw tables: `reports/cross_game_results*.txt`; reliability plots `reports/cross_game_reliability_{A,B}.png`.
Features, labels, models and selection rules are **exactly those of `reports/pass_completion_model.md`** (no new features, no re-tuning).

## 1. Game 3 integration and validation

Format (`src/ingestion/epts.py`, ~225 lines, reusing `TrackingData`, `metrica_to_metric`, `infer_attack_direction`, sync, validation, labels unchanged):

| | Games 1-2 (CSV) | Game 3 (EPTS/FIFA) |
|---|---|---|
| Tracking | one CSV per team, header rows, 28/26 columns | one `frame:x,y;...:ballx,bally` line per frame, 22 *positional* slots |
| Player identity | column header | per-segment `DataFormatSpecification` in metadata.xml (11 segments, slot contents change at each substitution); off-pitch players are NaN |
| Events | CSV, names | JSON, ids; `subtypes` null/dict/list (joined with `-` to match the CSV vocabulary) |
| Clock | time = frame/25 | time = frame/25 (verified on all 3,620 events) |
| Coordinates | normalised, y down | same normalisation/orientation (event positions equal ball positions at the event frame, unflipped) |
| Teams | Home/Away | FIFATMA (local) -> Home, FIFATMB -> Away |

Reader safeguards (tests in `tests/test_epts.py`): pitch-size mismatch, frames outside every segment or period, malformed lines (pandas would otherwise NaN-pad them), and an
attack direction that contradicts the metadata raise `ValueError`; untracked values stay NaN.

Validation (gate in `src/cross_game.py`, evaluation stops if any game fails): Game 3 passes. 143,761 frames at 25.00 Hz, no duplicate/non-monotonic/irregular timestamps, no gaps;
3,620/3,620 events synchronised with 0 s offset; no unknown or wrong-team passer ids; kick-off attack-direction inference agrees with the provider's metadata; same warnings as G1/G2
(ball untracked in 32% of frames, 358 coordinates beyond the pitch margin, 763 player / 133 ball implausible-speed jumps). Labelled outcomes use the **same event vocabulary and rules**
(G3: 1,067 plain `PASS`, 105 `BALL LOST/INTERCEPTION`); nothing was manufactured.

**Data defect found in Game 3 (second half).** Tracking slots for shirts 6 (P3573) and 13 (P3580), both Home, look exchanged relative to the events: 90% of shirt 6's and 97% of shirt 13's
second-half events have the ball at the *other* player (first half: 0.8% far from the named passer, so the slot mapping itself is right). The detector (`detect_identity_swaps`) finds nothing in
G1/G2. Only the passer lookup depends on identity (both players are teammates; team and all positional features are unaffected), so the possession filter removes those passers' events:
G3 loses 7.1% of completed and 7.6% of intercepted candidates in scenario A (G1/G2: 0.4-1.2% and 1.6-2.6%). **Default runs keep the data as supplied**; a sensitivity run applies the swap (section 6).
This is an eligibility difference: absolute metrics on G3 are not like-for-like with G1/G2.

## 2. Protocol

Every ordered pair of distinct games (6 runs) for scenario A (decision time, release frame, target-free) and B (early flight, release + 0.24 s, identified target) **separately**; A and B populations are never compared.
Per run, scaling/winsorising, ridge strength, feature-set choice (1-SE rule, 5-block time CV), boosting rounds, Platt calibration are fitted on the training game only; the test game is scored once. No
threshold is used. Metrics: ROC-AUC with 95% bootstrap CIs (1,000 resamples, resampling decisions), PR-AUC for the interception class, and paired Brier/log-loss differences versus the **training-game base-rate predictor**
on identical test samples. "Meaningful improvement" is pre-specified: *probability skill* = both paired CIs exclude 0 in the favourable direction ("yes"; one = "partial"; none = "no"); *ranking* = ROC-AUC CI above 0.5.
Models: base rate; location-only (A) / pass-distance (B) baseline; feasibility baseline (A: option-set aggregates; B: static margin); primary ridge logistic; exploratory boosted trees.
Caveats on inference: the six runs are not independent (each test game is scored twice, each training game used twice), three games is a very small number of replications, and CIs cover sampling error within a test game only, not
between-game variation. No multiplicity correction; treat isolated significances as exploratory.

## 3. Samples

| | G1 | G2 | G3 |
|---|---|---|---|
| A: n (intercepted / completed) | 872 (126 / 746) | 1023 (111 / 912) | 1088 (97 / 991) |
| A: completion rate | 0.856 | 0.891 | 0.911 |
| A: exclusion rate completed / intercepted | 1.2% / 1.6% | 0.4% / 2.6% | **7.1% / 7.6%** (identity swap) |
| B: n (intercepted / completed) | 487 (45 / 442) | 591 (37 / 554) | 642 (36 / 606) |
| B: completion rate | 0.908 | 0.937 | 0.944 |
| B: exclusion rate completed / intercepted | 41.5% / 64.8% | 39.5% / 67.5% | 43.2% / 65.7% |
| Outcome-unknown events (excluded, counted) | 173 | 167 | 237 |

B's selection (keeps ~57-60% of completed but only ~33-35% of intercepted passes) is remarkably consistent across the three games, so the *restriction* is systematic, not game-specific. Base rates differ by game, which matters for probabilities (section 5).

## 4. Results

### Scenario A: decision time (primary = A3 feasibility option-set aggregates in all three training games; it coincides with the feasibility baseline)
ROC-AUC [95% CI]; Brier / log-loss: paired difference vs training base rate (negative = better; * = CI excludes 0).

| run | test n (int.) | location baseline | **primary / feasibility** | dBrier | dLog-loss | boosted trees | dBrier | dLog-loss |
|---|---|---|---|---|---|---|---|---|
| G1 -> G2 | 1023 (111) | 0.461 | **0.583** [0.529, 0.641] | -0.0012 | -0.0061* | 0.645 [0.593, 0.701] | -0.0021* | -0.0101* |
| G1 -> G3 | 1088 (97) | 0.501 | **0.691** [0.636, 0.744] | -0.0033* | -0.0144* | 0.738 [0.687, 0.788] | -0.0047* | -0.0192* |
| G2 -> G1 | 872 (126) | 0.520 | **0.610** [0.554, 0.664] | -0.0013* | -0.0061* | 0.628 [0.572, 0.678] | -0.0026* | -0.0115* |
| G2 -> G3 | 1088 (97) | 0.488 | **0.676** [0.619, 0.733] | -0.0012* | -0.0058* | 0.656 [0.599, 0.712] | -0.0018* | -0.0104* |
| G3 -> G1 | 872 (126) | 0.453 | **0.603** [0.548, 0.655] | -0.0024 | -0.0018 | 0.657 [0.605, 0.708] | -0.0053* | -0.0210* |
| G3 -> G2 | 1023 (111) | 0.489 | **0.559** [0.506, 0.621] | +0.0006 | +0.0099 | 0.658 [0.607, 0.710] | -0.0013 | -0.0110* |

PR-AUC (no-skill = test interception rate 0.09-0.14): primary 0.14-0.24, boosted 0.15-0.25. Calibration of the primary (slope / intercept of logit P(int.), ideal 1 / 0): 0.96/-0.36, 2.36/+1.70, 3.11/+4.68, 5.33/+8.61, 0.54/-0.46, 0.33/-1.27;
boosted trees 0.77-2.66. Predicted interception probabilities occupy only ~0.09-0.21 (or 0.02-0.18 when trained on G3).

### Scenario B: early flight, identified-target passes only (primary differs by training game: G1 B5 no-feasibility, G2 B2 margin, G3 B4 distance+margin)

| run | test n (int.) | distance baseline | **feasibility baseline** | dBrier | dLog-loss | primary | boosted trees | dBrier | dLog-loss |
|---|---|---|---|---|---|---|---|---|---|
| G1 -> G2 | 591 (37) | 0.658 | **0.863** [0.810, 0.907] | -0.0075* | -0.0454* | 0.767 | 0.884 [0.842, 0.920] | -0.0090* | -0.0636* |
| G1 -> G3 | 642 (36) | 0.426 | **0.776** [0.721, 0.825] | +0.0031 | -0.0110 | 0.778 | 0.840 [0.778, 0.897] | -0.0038 | -0.0455* |
| G2 -> G1 | 487 (45) | 0.669 | **0.828** [0.762, 0.884] | -0.0084 | -0.0535* | 0.828 | 0.857 [0.803, 0.906] | -0.0158* | -0.0840* |
| G2 -> G3 | 642 (36) | 0.426 | **0.776** [0.721, 0.825] | +0.0063* | -0.0018 | 0.776 | 0.809 [0.754, 0.863] | -0.0015 | -0.0260* |
| G3 -> G1 | 487 (45) | 0.331 | **0.829** [0.762, 0.884] | -0.0062* | -0.0417* | 0.744 | 0.835 [0.781, 0.885] | -0.0119* | -0.0687* |
| G3 -> G2 | 591 (37) | 0.342 | **0.863** [0.810, 0.907] | -0.0046* | -0.0329* | 0.783 | 0.796 [0.727, 0.860] | -0.0042 | -0.0331* |

Primary logistic: AUC 0.74-0.83 (CIs of width ~0.15-0.18), probability skill yes 2 / partial 3 / no 1; feasibility baseline yes 3 / partial 1 / no 2; boosted trees yes 3 / partial 3 / no 0 (primary CIs and verdict lines in the results file). PR-AUC (no-skill 0.056-0.092): feasibility baseline 0.12-0.29, boosted 0.18-0.36.
Calibration slope / intercept (primary): 0.81/-0.65, 0.67/-1.00, 0.86/-0.03, 0.55/-1.27, 0.86/+0.13, 0.98/+0.01; on G3 as test the intercepts are strongly negative (over-predicted interceptions: G3's completion rate is highest).
Note: a monotone one-feature baseline has the *same* AUC for every training game whose fitted sign agrees (e.g. 0.776 on G3, 0.863 on G2); only Brier/log-loss then depend on the training game.

## 5. Answers to the research questions

**1. Does decision-time Model A generalise?** As a weak *ranking* signal, yes: the primary's ROC-AUC is 0.56-0.69, CI above 0.5 in 6/6 runs (lowest bound 0.506, G3 -> G2); PR-AUC beats no-skill in all runs. As a *probability* model, not reliably: it beats the training base rate on both Brier
and log-loss in 3/6 runs (all with G1 or G2 as training game), partially in 1, and not in the two G3-trained runs (indistinguishable from the base rate for G3 -> G1, worse for G3 -> G2). Predictions have almost no resolution, and calibration slopes range 0.3-5.3.
Exploratory boosted trees rank better (0.63-0.74) and beat the base rate on log-loss in 6/6 runs (Brier in 5/6), but with tiny absolute gains (0.0013-0.0053 Brier).
**2. Does early-flight Model B generalise, and how much does the restricted sample matter?** Ranking generalises well: AUC 0.78-0.86 for the single feasibility feature and 0.80-0.88 for boosted trees, in 6/6 runs, with CIs above 0.5
and no sign of a single-game artefact. But the conclusion applies only to the clearly directed passes B retains: it drops 65-68% of interceptions and 40-43% of completions in every game, retained passes are those with safer option sets and
cleaner direction (see `pass_completion_model.md` section 8), each test set has only 36-45 interceptions, and B's prediction frame is 0.24 s after release. B is not decision-time and does not cover passes into space.
**3. Does geometric feasibility beat pass distance alone?** In B: yes, in all 6 runs, with paired ROC-AUC differences +0.16 to +0.52 and log-loss differences -0.019 to -0.045, all significant. The distance relationship is itself unstable: intercepted passes are longer
in G1/G2 (AUC 0.66-0.67) but shorter in G3 (0.43, also in G3's first half, so not caused by the identity defect), so a G1/G2-trained distance model scores 0.43 on G3 and a G3-trained one 0.33-0.34 on G1/G2. In A (location baseline has no pass distance): +0.07 to +0.19 AUC,
significant in 5/6 runs (G3 -> G2 not), no skill from location alone (0.45-0.52).
**4. Are results driven by one game?** Not the ranking conclusion for feasibility, but absolute levels depend heavily on the test game: B feasibility AUC 0.78 (G3), 0.83 (G1), 0.86 (G2), so the original single-split result (G1 -> G2: 0.86-0.88)
was the most favourable of the six; in A the primary scores 0.57 (G2), 0.61 (G1), 0.68 (G3 as test), and G3-trained models are the weakest (0.56-0.60). The probability-skill failures are concentrated in runs where the training and test base rates differ most. G3 also differs in eligibility (section 1), though
the sensitivity run shows this does not change conclusions.
**5. Are probability estimates useful, or does the model merely rank?** Mostly ranking. Calibration-in-the-large does not transfer: interception rates differ by game (A 8.9-14.4%, B 5.6-9.2%), producing large intercepts in the diagnostics and failures against the training base rate. In B, probabilities
are directionally sensible (slopes 0.55-0.98) and improve log-loss over the base rate in most runs, but Brier improvements are often not significant and CIs are wide with 36-45 events. Treat them as risk scores that need per-match base-rate recalibration, not as probabilities.
**6. Is the boosted-tree model consistently better than logistic regression?** Point estimates favour it in A in 5/6 runs (paired AUC -0.02 to +0.10; significant in 4; one run -0.02, n.s.) and in B in 6/6 (+0.01 to +0.12; significant in 4/6, one borderline), with log-loss gains significant in 5/6 B runs and 3/6 A runs. The advantage varies (e.g. B: +0.117 for G1 -> G2 but +0.013 for G3 -> G2; A: -0.021 for G2 -> G3), and in B it does not
clearly beat the one-feature feasibility baseline (GBT 0.84 vs baseline 0.78 on G3, 0.88 vs 0.86 on G2, but 0.80 vs 0.86 for G3 -> G2). The comparison is also not like-for-like: the logistic uses a train-selected subset, the boosted trees all features. Exploratory; not stable enough to recommend as the primary model.

## 6. Sensitivity: tracking-identity swap applied (`--fix-identity`)

Restores G3 candidates (A: 1088 -> 1161, 97 -> 105 interceptions; B: 642 -> 684, 36 -> 39) and changes test AUCs by at most ~0.02 (the largest shifts are G3-trained B models, e.g. primary G3 -> G2 0.783 -> 0.802) (A, G1 -> G3: primary 0.691 -> 0.682; B: feasibility 0.776 -> 0.775, boosted 0.840 -> 0.846). Conclusions are unchanged.
One artefact to ignore: the G3-trained distance baseline moves from 0.33 to 0.67 on G1/G2 only because its Platt slope collapses to -0.005 (no signal), which flips the ranking sign arbitrarily; this is not evidence that distance works.

## 7. Limitations

Three games, one provider, no player-level or Ronaldo inference; only 36-126 interceptions per game and, in B, a restricted and biased sample; CIs ignore between-game variance and the runs share data; identity defect in G3 handled by exclusion
(default) or inferred swap (sensitivity) and not confirmed by the provider; probabilities require base-rate recalibration that cannot be done without target-game labels; pass-completion here still means completed vs intercepted, not all failures; scenario B is not a decision-time prediction.

## 8. Conclusion

Evidence generalises for **ranking by geometric feasibility** (strongly for identified-target passes in B, weakly for decision-time option-set aggregates in A) and for **feasibility outperforming pass distance**. Evidence does **not** support calibrated, transferable
completion probabilities or a decision-time target-specific model. Next most defensible step: a leave-one-game-out pooled evaluation of the single-feature models with an explicit estimate of between-game intercept (base-rate) variance, to establish whether any transferable probability model is possible before adding model complexity.

## 9. Tests

`tests/test_epts.py` (10): metadata, tracking slots/substitutions/NaN preservation/clock, coordinate conversion, attack-direction cross-check, explicit errors, event schema/vocabulary, synchronisation and consistency checks, labels on EPTS events, identity-swap detection and opt-in repair.
`tests/test_cross_game.py` (8): training-only fitting, identical test samples, training base-rate baseline, verdict rules, unchanged feature builder, failed gate blocks evaluation, gate fails on missing labels, identity fix is opt-in. All earlier leakage/mutation tests are unchanged. Full suite: 104 passed.
