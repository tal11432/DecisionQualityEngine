# Pass-feasibility: outcome-leakage audit and correction

> **Clarification added in the pass-completion stage (`reports/pass_completion_model.md`).** The target of every evaluated pass here was identified from the ball's heading over the
> first 6 frames *after* release, i.e. 0.24 s after the decision. Inputs were taken at release, but the target selection (and therefore the sample) used post-release
> information. The evaluation is outcome-leak-free but **not a decision-time prediction**; read the wording below as "identified-target, early-flight" evidence. The pass-completion
> report separates a genuine decision-time (target-free) scenario from an early-flight scenario and reports both.

Scope: the kinematic time-to-intercept score in `src/analytics/feasibility.py` and its evaluation.
Development game = Metrica Sample Game 1, held-out game = Sample Game 2 (neither is Ronaldo data).
Raw output: `reports/pass_feasibility_results.txt`
(`python -m src.calibration --dev 1 --test 2 --out reports/pass_feasibility_results.txt`, deterministic).

## 1. Leakage found (previous commit `2313537`)

| # | Where | Problem | Verdict |
|---|---|---|---|
| L1 | `calibration.build_samples` | **Target = event `end_x/end_y`.** For completed passes this is the receiver's position at reception (~1-2 s after the decision: future receiver movement). For intercepted passes it is **where the opponent got the ball**, i.e. an opponent is at the lane end by construction. | **Genuine leakage** |
| L2 | `kinematics.snapshot_velocities` | Central difference over t±2 frames: reads 0.08 s of **future** frames. | Genuine (small) |
| L3 | `estimate_ball_speed` | Median speed of completed passes from their **end frames**. | Outcome-derived *population* statistic; one constant for all decisions. Kept as a disclosed prior, not a per-pass input. In-sample on the dev game. |
| L4 | player speed / reaction time | Selected by AUC under L1's leaky target; reported as "calibrated". | Invalid as stated; re-selected leakage-safe, **not adopted** (see §3). |
| L5 | sample construction | Samples filtered on end-point availability and built from end-point geometry; negatives and positives did not share an information contract. | Genuine |
| - | Decision frame | Undefined. Verified on Game 1: at the event start frame the ball is at the passer's feet (median 0.09 m, both classes) and its speed jumps from ~3 to ~11-12 m/s within 6 frames, so it is the **ball-release frame**. | Now defined (§2) |

Size of L1 on identical decisions (M1, actual-end-point target vs decision-time target):
dev 0.872 → 0.823 AUC, held-out 0.888 → 0.860 (paired, same samples). Because the old samples also differed, the
earlier headline (Game 2 AUC 0.87-0.88) is not directly comparable with the numbers below.

Legitimate at decision time: positions of all players and the ball at the decision frame; velocities from frames ≤ decision
frame; passer identity; the event *type/subtype* as the eventual **label**.

## 2. Decision-time contract (implemented in `src/pass_dataset.py`, config `[decision_time]`)

Decision frame t_d = ball-release frame (event start synced by timestamp, tolerance 0.02 s) + `offset_frames` (≤ 0).
The release frame is the only frame identifiable for *every* candidate; the "earliest frame with possession" is not reliably
identifiable (no possession-start event for dribbles/recoveries). A dev-only sensitivity with t_d = release - 5 frames
(0.2 s earlier) gives M1 0.804 / M3 0.802 (n=415, 37 intercepted) vs 0.823 / 0.830 at release: no material change.

| Feature | Definition | Source | Time vs t_d | Real-time observable? | Notes / limits |
|---|---|---|---|---|---|
| origin | passer position | tracking | t_d | yes | passer must be tracked and ball within 3 m |
| target | position of the **aimed-at teammate** | tracking | t_d | yes (as a *candidate*) | static position; no lead/space; receiver motion not used (if added: history only) |
| opponent positions | all tracked opponents | tracking | t_d | yes | untracked players are absent |
| opponent velocity | causal (pos[t_d] - pos[t_d-5 frames]) / dt | tracking | frames t_d-5..t_d | yes | NaN → treated as static; excluded if none valid |
| ball speed | fixed prior 11.9 m/s | median avg speed of Game-1 completed ground passes | constant | yes as a prior | average chord speed, not the speed of any particular pass; outcome-derived population statistic |
| player speed | 7.0 m/s | **assumed** | constant | n/a | not measured; dev grid diagnostic only |
| reaction time | 0.3 s | **assumed** | constant | n/a | weakly identified (below) |
| *label side only:* aimed-at teammate | teammate whose bearing from the passer is closest to the ball's heading over frames t_rel..t_rel+6 (≤15°, runner-up ≥5° worse) | tracking | **after** release | **no** | selects *which* decision-time position is the target; never a feature |
| length-only baseline | distance passer→target | tracking | t_d | yes | |

Label: completed = `PASS` without subtype; intercepted = `BALL LOST` with subtype exactly `INTERCEPTION` (that event is the failed pass;
`To` is empty). Out of scope and not counted as negatives: passes with subtypes (HEAD, CROSS, GOAL KICK, ...: 42 / 48 events in
G1 / G2), other failures (THEFT, out of play, ...), other interception subtypes.

**Why the target needs a label-side rule.** Metrica provides no intended receiver for intercepted passes and no notion of a space
target. Using the interception point is leakage, so the only defensible option is to infer the aimed-at *teammate* from the ball's
early heading, identically for completed and intercepted passes, and to **exclude** anything unidentified or ambiguous. The true
receiver (completed passes only) is used solely to score that rule: correct on **85.7% (G1) / 86.9% (G2)** of assigned completed
passes. Accuracy on intercepted passes cannot be measured.

## 3. Samples remaining (candidates → kept)

| | completed | intercepted |
|---|---|---|
| Dev G1 | 757 → **441** (58%) | 128 → **45** (35%) |
| Held-out G2 | 916 → **554** (60%) | 114 → **37** (32%) |

Main exclusions (G2, completed / intercepted): ambiguous receiver 228 / 29; no teammate in heading cone (passes into space or
unidentified) 110 / 37; ball displacement too small 18 / 3; ball untracked after release 0 / 5; possession filter 4 / 3;
target < 3 m 2 / 0. Full tables for both games are in the results file.

**Selection bias.** Intercepted passes are excluded far more often than completed ones (65-68% vs 40-42%), mostly because their
direction is ambiguous or aimed into space, so the evaluation covers clearly directed passes only. Direction of the bias on the
score is unknown. The positive class is small (45 dev, 37 held-out).

## 4. Held-out Game 2 (n = 591: 37 intercepted / 554 completed, base rate 6.3%)

Nothing was tuned on Game 2. The score is a geometric margin in seconds (>= 0 means some opponent can reach a lane point no later
than the ball). It is **a score, not a probability**; no Brier score/log loss/reliability diagram is reported because there are no
probabilities (and 37 positives could not support a reliability diagram).

| Model | AUC [95% bootstrap CI] | Dev AUC |
|---|---|---|
| M1 original static (15 m/s, 7 m/s, 0.3 s) | 0.860 [0.809, 0.906] | 0.823 |
| M2 + calibrated ball speed (11.9) | 0.852 [0.795, 0.904] | 0.825 |
| M3 + causal velocities | 0.848 [0.791, 0.901] | 0.830 |
| M4 dev-refit (6.2 m/s, 0.3 s) + causal velocities | 0.848 [0.792, 0.898] | 0.830 |
| Pass length only | 0.662 [0.563, 0.757] | 0.672 |

Paired ΔAUC (same samples): M2-M1 -0.008 [-0.019, +0.003]; M3-M2 -0.004 [-0.013, +0.004]; M4-M3 +0.000 [-0.006, +0.006]: **none
significant**, i.e. calibration and velocities do not help. Models vs length-only: M1 +0.197 [+0.129, +0.269], M3 +0.186
[+0.125, +0.251]: **significant**.

Confusion (positive = intercepted). A: margin ≥ 0 (the model's own definition, fixed a priori). B: F1-optimal threshold on dev.

| Model | thr | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| M1 | A 0 | 29 | 114 | 8 | 440 | 0.203 | 0.784 | 0.322 |
| M1 | B +0.53 | 18 | 44 | 19 | 510 | 0.290 | 0.486 | 0.364 |
| M3 | A 0 | 31 | 180 | 6 | 374 | 0.147 | 0.838 | 0.250 |
| M3 | B +0.80 | 20 | 52 | 17 | 502 | 0.278 | 0.541 | 0.367 |
| Length only | B +20.7 m | 18 | 151 | 19 | 403 | 0.107 | 0.486 | 0.175 |

Subgroups (sufficient size only: ≥ 15 per class): pass length ≥ 20 m AUC M1 0.830 [0.744, 0.901], M3 0.834 [0.756, 0.898],
length-only 0.677; Home 0.850 / Away 0.868 (M1); period 1 0.916 / period 2 0.790 (M1; overlapping CIs). Passes < 20 m have
too few intercepted examples (4 and 14) to report.

## 5. Does it show genuine predictive ability?

**Partly, and narrowly.** All model inputs are now decision-time, and the result survives removing the leak: the score ranks
intercepted above completed passes clearly better than pass length (+0.19 AUC, significant on the held-out game; moving the decision 0.2 s earlier lowers dev AUC only
from ~0.82-0.83 to ~0.80, no length comparison was run for that dev-only check). So the geometry carries real decision-time signal **conditional on a given candidate target**. What is *not* shown:

- A forecast of what a player will do: the evaluated target is identified from post-release ball heading, which is unavailable
  at decision time. In deployment the targets would be hypothetical options, which this data cannot label.
- Coverage: only ~1/3 of intercepted passes are evaluable; passes into space cannot be evaluated at all.
- Probabilities or calibration; thresholded flags are poor (precision 0.15-0.29 at a 6% base rate).
- That calibration or velocities improve anything. They did not; player speed and reaction time remain **assumptions**
  (dev grid AUC 0.822-0.830 across all 12 settings, not identified; M4's choice is not adopted into config).

## 6. What the dataset cannot support

Intended receivers/targets of failed passes; passes into space; failed passes that are not interceptions (out of play,
miskicks); possession-start timing; calibrated completion probabilities (45 / 37 positives); player-level conclusions; anything
about Ronaldo. A valid probabilistic evaluation needs provider data with an explicit intended receiver or target location for
**failed** passes (plus tracking at the decision frame), in several hundred or more failures.

## 7. Recommendation

Proceed to a pass-completion model only as a model for **identified-receiver passes**, using this score as one decision-time
feature, with the label-noise of the target rule stated, and validated on held-out games. Do not use it yet to score decisions or
passes into space.

## Reproducibility / protection

`tests/test_leakage.py` (15 tests) covers: end-point columns and `to_player` scrambled → identical samples; moving the
interception location → identical score; end point cannot rescue a pass into space; all frames after t_d mutated → identical
inputs; causal velocity ignores the future (central estimator does not); completed vs intercepted identical features;
ambiguous / unidentified / unsynchronized targets excluded with reasons; possession and decision-offset rules. Injecting either
leak (future velocities, end-point target) into the builder makes these tests fail. Full suite: 61 passed.
