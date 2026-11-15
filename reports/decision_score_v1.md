# Decision Score v1: a 0-100 value for every shot, pass and carry in the attacking third

Model-based and not validated by experts. Pre-registration: `research/dq/PREREG_DQ_v13.txt` (with amendments A-D, each written before the data it concerns was scored). Code: `research/dq/dscore/` (`core.py`, `score.py`), data scripts `fetch_v13*.py`, `fetch_v14.py`, validation `eval_v13_models.py`, `eval_v14.py`; tests `tests/test_dscore_*.py` (13 unit tests; the full suite, 146 tests, passes). StatsBomb open data (attribution required), 360 competitions 2015-2024; test competitions never used for fitting: Euro 2024, Bundesliga 2023/24, MLS 2023 (+ Copa America 2024 for shots).

## What it is
* **Value of an option, Q** = expected net xG of the possession over the next 10 s (xG for minus xG against).
  * Shot: xG plus the later shots of the possession. Pass to a visible teammate: xP x chain value - (1 - xP) x L(target). Carry (the one actually made): P_keep x V_pos(destination) - (1 - P_keep) x L(destination).
  * **L**, the cost of losing the ball *in that situation*, is a model of the expected opponent xG minus own xG after a loss, using the position and who is around (not a constant).
* **Score** = the **percentile rank of Q among the decisions taken** in a fixed random sample of 9,000 attacking-third decisions of the training competitions; the ends are pinned to the user's anchors (Q >= 0.70 -> 100; Q <= -0.231, the mean xG of a one-on-one, -> 0). 50 is about the median decision. The scores of **every option** in the moment are computed, together with the **score of the best option** and the **gap** to the option taken (in points and in xG). A gap below the noise floor of DQ-v12b (0.03 xG, 0.05 for passes over 15 m) is "no clear difference from the best option": percentile points exaggerate small differences.
* Scope: foot shots, ground passes and carries with the ball at x >= 80 and 360 frames. Not in v1: take-ons, through balls, crosses, lofted balls, clearances, hypothetical carries as alternatives.

## What validated and what did not (17,867 test decisions: 2,424 shots, 7,920 passes, 7,523 carries)
| Gate | Result |
|---|---|
| Resolution: the 10th-90th range of the score >= 20 points | **Pass**: 10 to 91 (81 points). The first, linear scale failed (8-81 points of Q-values compressed into 50-54 for ordinary passes), which led to the percentile scale |
| Calibration of Q against the realised net xG of the next 10 s | Deciles of the score rise monotonically (Spearman 0.99 [0.98, 1.00]). Pooled predicted/observed **1.18: fail** (limit 1.15); **without failed passes 1.11: pass**. By kind: shots 1.04, completed passes 1.07, carries 1.25, failed passes 12 (the only target known is where the ball stopped, which makes the pass look safe) |
| Gap known-groups: inside terciles of the promise of the situation, big gaps realise less | **Pass in 3/3 strata**, intervals exclude 0: realised difference -0.004, -0.010 and -0.072 xG |
| Best-option sanity: clear chances (Q >= 0.30, n = 191) | **Pass**: the shot is the best option in 97%; this failed while hypothetical carries were alternatives (their values reached 0.95) |
| Winner's-curse flag | ok: predicted/realised 1.17 for decisions where the option taken is the modelled best |
| Anchors | the scale pins to 100 / 0; a modelled gift (ball near our goal, an opponent next to it, only our keeper behind) scores 0 (L = 0.062 xG, five times the typical cost of a loss) |
| Stability under noise (0.8 m positions, 9%/10%/18% value scales) | **Pass**: median band 12 points for the score, 14 for the gap; 82% of decisions keep their score tercile in >= 80% of draws |
| Hiding half of the visible teammates | the median change of the gap is 0 points |
| Models | turnover danger L (whole pitch), keep-the-ball for carries and V_pos pass their calibration gates; the take-on model failed (AUC 0.955 suggests the frame already shows the result of the duel) and is not in v1; the frozen chain-value model over-predicts carry destinations (slope 0.65), the refit version passes |

## What it shows
* **Only about 19% of decisions have a gap to the best option larger than the noise floor** (shots 30%, carries 24%, passes 11%); in the other 81% the option taken is statistically as good as the best one found, even if the gap looks like 20-30 points.
* In situations of similar promise, decisions with a large gap realised clearly less in the next 10 s (table above), so the gap carries information beyond how good the situation was.
* Typical scores: ordinary passes median 40, carries 50, shots 88 (shots have higher values than most passes: a low-value long shot can still score 70+; the gap, not the absolute score, tells whether a better option existed).

## Two matches (every decision at x >= 80; CSV per match in `research/cases/`, no player names)
Leverkusen 5-1 Darmstadt (2023-09-02): 293 decisions, median score 56; the five Leverkusen foot goals scored 99, 88, 99, 92 and 100. Only the 48:08 goal had a clear gap (the shot 88 against a pass worth 98; gap 0.19 xG; the best pass is at the offside line in doubt, but the next certain one is also better). Barcelona 5-1 Alavés (2021-02-13): 293 decisions, median 47; goals 93 (gap 0.34 xG to a pass at 28:54), 76, 97, 99, 86 and 100, the others with no clear gap. Both are consistent with the earlier shot-versus-pass reviews. Full output: `score_match_*_out.txt`.

## Limits (read before using the numbers)
* **Not validated by experts.** The score is a model of expected xG from positions; it says nothing about intention, body orientation, speed, or the information the player had. The expert pilot (`dq_expert_protocol.md`) is the only external validation and has not been run.
* **The low end of the scale is thin.** In expectation a loss costs little, even in our own half (about 0.012 xG), so almost no real decision has a negative value: 0.6% of decisions score 0 and every negative value maps to 0. The one-on-one gift is a rare tail the position and context features only partly see.
* **Shots score high by construction** (their values are higher than most passes); compare the gap, not the score, across decision types.
* **Hypothetical alternatives are limited to visible players and ground passes**: carry alternatives are not supported and the best option is a maximum over noisy values (flagged: realised is 1.17 times lower than predicted for these cases).
* The calibration gate failed on the pooled sample (1.18), mostly from failed passes and carries; the scores of failed passes and of carries should be read with that in mind.
* StatsBomb 360 shows only part of the players; the Premier League 2003/04 data has no 360 frames (shots only).
