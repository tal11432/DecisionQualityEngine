# Decision Score v1 (corrected definition): an absolute 0-100 value for every shot, pass and carry in the attacking third

Model-based and not validated by experts. Pre-registration: `research/dq/PREREG_DQ_v13.txt` (amendments A-F, each written before the data it concerns was scored). Code: `research/dq/dscore/` (`core.py`, `score.py`), data scripts `fetch_v13*.py`, `fetch_v14.py`, `fetch_v15.py`, model and validation scripts `eval_v13_models.py`, `eval_v15_models.py`, `eval_v14.py`; tests `tests/test_dscore_*.py` (the full suite: 148 pass, 2 expected failures that document a limit, see below). StatsBomb open data (attribution required), 360 competitions 2015-2024; test competitions never used for fitting: Euro 2024, Bundesliga 2023/24, MLS 2023 (+ Copa America 2024 for shots).

## A correction first
The first version of this report scored a decision by its **percentile rank among other decisions** (amendment D). That contradicted the definition: a 0.32 xG shot scored 99, a 0.06 xG shot 88 and a 0.08 xG shot 92. It was my error and it is withdrawn (amendment E). The score is now what was defined: the **expected value of the decision on an absolute scale**, independent of the outcome and of other decisions.

## What it is
* **Value of an option, Q** = expected net xG of the possession over the next 10 s (xG for us minus xG for the opponent). Shot: xG plus the later shots of the possession. Pass to a visible teammate: xP x chain value - (1 - xP) x L(target), so a low success probability lowers the value even when the receiver would be in a scoring position, but not to the level of a pass to nowhere (arrival creates the chance). Carry (the one actually made): P_keep x V_pos(destination) - (1 - P_keep) x L(destination). **L** is the average cost of losing the ball at that place, given who is around.
* **Score = 100 x clip((Q + C) / (0.70 + C), 0, 1)** with C = 0.231 (the mean xG of a one-on-one chance). 100 is a move worth 0.70 xG or more; 0 is a decision that costs 0.231 xG in expectation. Q = 0 scores 25. Only Q enters: whether the shot went in never does. The score of every option of the moment is computed, the best option and the **gap** to the option taken (in points and in xG); a gap below the noise floor of DQ-v12b (0.03 xG, 0.05 for passes over 15 m) is "no clear difference from the best option".

| Expected value Q (xG) | -0.231 | -0.012 | 0 | 0.01 | 0.06 | 0.08 | 0.32 | 0.62 | 0.70+ |
|---|---|---|---|---|---|---|---|---|---|
| Score | 0 | 24 | 25 | 26 | 31 | 33 | 59 | 91 | 100 |

## The Leverkusen 5-1 Darmstadt goals under the corrected score
| minute | Q (xG) | score now | score in the withdrawn percentile version | best option |
|---|---|---|---|---|
| 20:18 | 0.321 | **59** | 99 | the shot itself |
| 48:08 | 0.059 | **31** | 88 | a pass worth 0.252 (score 52): clear gap of 0.19 xG, 21 points |
| 60:21 | 0.383 | **66** | 99 | the shot itself |
| 66:59 | 0.080 | **33** | 92 | the shot itself |
| 82:10 | 0.620 | **91** | 100 | the shot itself |

A scored 0.06 xG shot is a poor decision with a lucky ending; the score says so. Barcelona 5-1 Alavés: goals 35 (a pass worth 72 was available: 0.34 xG, 37 points), 29, 47 (Alavés), 68, 30 and 97. Full output: `research/cases/score_match_*_out.txt` and per-match CSVs (no player names).

## What validated and what did not (17,867 test decisions: 2,424 shots, 7,920 passes, 7,523 carries)
| Gate | Result |
|---|---|
| Calibration of Q against the realised net xG of the next 10 s | Deciles of the score rise monotonically (Spearman 0.96 [0.95, 1.00]); pooled predicted/observed **1.18: fail** (limit 1.15); **without failed passes 1.11: pass**. By kind: shots 1.04, completed passes 1.07, carries 1.25, failed passes 12 (the only target known is where the ball stopped, which makes the pass look safe) |
| Gap known-groups: inside terciles of the promise of the situation, big gaps realise less | **Pass in 3/3 strata**, intervals exclude 0: realised difference -0.004, -0.010 and -0.072 xG |
| Best-option sanity: clear chances (Q >= 0.30, n = 191) | **Pass**: the shot is the best option in 92%, median gap 0 |
| Winner's-curse flag | ok: predicted/realised 1.19 for decisions where the option taken is the modelled best |
| Stability under noise (0.8 m positions, 9%/10%/18% value scales) | Pass, but uninformative on this scale: most decisions sit between 25 and 33 points, so the band is about 1 point (gap: 2 points); 86% keep their score tercile in >= 80% of draws |
| Hiding half of the visible teammates | the median change of the gap is 0 points |
| Models | turnover cost L (whole pitch) and keep-the-ball for carries pass their calibration gates; **the counter-attack feature gate failed; the leak-free take-on gate failed** (see next section) |
| Face-validity unit tests | 0.01 xG shot scores 26 (< 30), 0.99 scores >= 95, 0.32 far above 0.06; a covered lane scores below a free lane and above a pass to nowhere: **pass**. A loss that hands over a counter-attack scores <= 15 and a crowded carry is worse when a loss means a counter: **both fail** (expected failures in the test suite) |

## What the data cannot deliver (the parts of the definition that are not implemented)
The definition says the lowest score belongs to a decision that leads to an opponent counter-attack with a numerical advantage, and that a dribble into a crowd where a loss means a counter should be near zero. **I could not implement that from data.**
* In 56,108 losses of the ball in the 360 data, the visible numerical balance (opponents beyond the ball against our players behind it) does **not** predict opponent chances: the 1,322 training losses with an apparent opponent advantage realised 0.002 xG on average (an opponent shot followed in 2.4%), no more than other losses; adding the balance features to the turnover model gave no gain (log-likelihood -0.00004). A single 360 frame (only the players in view, no velocities) does not identify a dangerous counter.
* Consequence: the zero of the scale is the **stipulated** anchor (C = 0.231, the mean one-on-one xG), not a value measured from counters, and almost no decision gets near it: the worst 0.1% of unforced turnovers score about 18, and no option taken in the test set scores below 25 (`share <= 10`: 0.0%). A dribble into a crowd is lowered only through the probability of keeping the ball and the average cost of losing it at that place.
* Stand-alone take-ons are **not scored**: with the frame taken before the duel (no outcome leakage) nothing visible predicts who wins it (AUC 0.55, slope 0.34).
* What would fix it: positions with velocities (tracking data; we only have three Metrica games), or a rule-based "stress scenario" for counters. The second would be an assumption written into the score, not something measured; I have not added it.

## What the corrected score shows
* Ordinary decisions sit near the bottom of the scale: the median option taken scores 26 (10th-90th percentile 25-33; passes 25-29, carries 25-31, shots 27-53). Only 2.1% of decisions score 50 or more and 0.4% score 80 or more. This follows from the definition (typical decisions are worth about 0.02 xG against a ceiling of 0.70), and it means the score itself separates decisions only at the top; **the gap to the best option and the verdict are the useful outputs for ordinary play.**
* The gap is larger than the noise floor in 19% of decisions (shots 30%, carries 24%, passes 11%); in the rest the option taken is statistically as good as the best one found.
* The rarity column (`pct_chosen` in `decision_scores_test.csv`: the percentile of Q among 9,000 training decisions) is kept for reading how unusual a value is. It is not the Decision Score.

## Limits (read before using the numbers)
* **Not validated by experts.** The score is a model of expected xG from positions; it says nothing about intention, body orientation, speed, or the information the player had. The expert pilot (`dq_expert_protocol.md`) is the only external validation and has not been run.
* **The low end of the scale is not data-supported** (previous section).
* **Shots score higher by construction** (their values are higher than most passes); compare the gap, not the score, across decision types.
* **Alternatives are limited to visible players and ground passes**: hypothetical carry alternatives are not supported and the best option is a maximum over noisy values (flagged: realised is 1.19 times lower than predicted for these cases).
* The calibration gate failed on the pooled sample (1.18), mostly from failed passes and carries; read their scores with that in mind.
* StatsBomb 360 shows only part of the players; the Premier League 2003/04 data has no 360 frames (shots only). No player-level aggregation or ranking has been made or should be made before validation.
