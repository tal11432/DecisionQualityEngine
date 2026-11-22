# Set-piece deliveries in the "open-play ground pass" population (Metrica Sample Games 1-3)

**Status: finding + proposal. Nothing in `src/` or in the existing reports was changed.** The proposed change is `reports/set_piece_delivery_exclusion.patch` (checked with `git apply --check`, not applied). The new test `tests/test_set_piece_exclusion.py` fails on the current code on purpose (3 of 4 tests; the 4th is a negative control that passes both ways).

## 1. Verdict: SET PIECE + PASS is one action

| Evidence | Result |
|---|---|
| Metrica definitions (`documentation/events-definitions.pdf`, same file at the pinned commit and on master) | `SET PIECE`: "The player brings the ball back into play via a Set piece." `PASS`: "Possession of the ball changes between two players of the same team." The marker records the restart act, the `PASS`/`BALL LOST` records the possession transfer. The document does not say "one action" in words; that is inferred from the rest of this table. |
| Event table, all 247 SET PIECE events | Marker has zero duration (one marker -0.04 s). 241 are followed by a `PASS`/`BALL LOST`/`SHOT` by the **same team and same player** within 1 s; 100% same player; gap is exactly 0.00 s for 85% / 90% / 100% (G1/G2/G3), max 0.68 s. The other 6 are followed by `BALL OUT` (3) or by another event (3). |
| Tracking at the event frame (225 labelled deliveries vs 2,858 other candidates) | Ball untracked in all 25 frames before the event: **97% vs 0.3%**. At the event frame the tracked ball coincides with the passer (<1 cm): **97% vs 9%** (the ball track starts at the restart). Throw-ins: ball within 1 m of a touch/goal line **99%** (open play 2.8%). Kick-offs: ball within 1.5 m of the centre spot 100%. Corners: median 2.2 m from the corner flag (max 8.1 m). |
| Kinematics | Ball speed over the next 0.4 s: 9.7 m/s median (deliveries) vs 11.3 (open play): a normal delivery, not a separate preceding action. Throw-ins are hand deliveries. |

No second ball touch, no intervening event, and no possession change separates marker and pass. Side effect: the "ball within 3 m of the passer" eligibility filter is vacuous for these events (the tracked ball is the passer's position), so it cannot confirm possession.

Reproduce: `python research/set_piece_check/spot_checks.py` (output: `reports/set_piece_exclusion/spot_checks_output.txt`).

**Population size and a discrepancy with the exploratory check.** On the actual scenario-A tables (legacy labels) I get **221 deliveries of 2,983 (7.4%): 195 completed, 26 intercepted** (throw-ins 104, free kicks 54 + 23 indirect + 2 retaken, corners 21, kick-offs 16 + 1 retaken); scenario B: 106 of 1,720 (6.2%: 101 / 5). The exploratory figure (188 = 184 + 4, "7%") is cited in `reports/dq_pipeline_run.md` for the DQ prototype's situations, a different population; I did not trace it. The interception count differs most (4 vs 26). The conclusion does not depend on it. The interception rate is the same in both groups (deliveries 11.8%, open play 11.2%), so the exclusion does not move base rates (per-game scenario-A interception rates 0.144 / 0.109 / 0.089 -> 0.147 / 0.106 / 0.088).

## 2. Effect on the headline results

The scripts were rerun unchanged except for the population filter. The "without" runs reproduce the committed `reports/*_results.txt` line for line (apart from figure paths). "With" outputs: `reports/set_piece_exclusion/`; side-by-side parse: `reports/set_piece_exclusion/comparison.txt` (`python research/set_piece_check/compare_runs.py`).

The filter removes 6-7% of passes, the same ones in every table. Compare point estimates with caution: the changes below are small next to the CI widths (about ±0.05-0.06 AUC), and I did not compute paired CIs between the two populations. `*` marks a CI that excludes 0, as in the source files.

**Scenario A (decision time).** Primary ROC-AUC moves up or stays flat in every comparison; scenario B is essentially unchanged.

| | A: before -> after | B: before -> after |
|---|---|---|
| `pass_completion` G1->G2, primary AUC | 0.583 -> 0.606 | 0.767 -> 0.754 |
| same, boosted trees AUC | 0.645 -> 0.653 | 0.884 -> 0.872 |
| same, primary vs base: Brier / log-loss | -0.0012 n.s. / -0.0061* -> **-0.0018\* / -0.0087\*** | -0.0061 n.s. / -0.0337* -> -0.0040 n.s. / -0.0281* |
| same, test n (intercepted) | 1023 (111) -> 950 (101) | 591 (37) -> 558 (34) |
| `cross_game`, mean primary AUC over 6 runs | 0.620 -> 0.639 (per run +0.000 to +0.041) | 0.779 -> 0.780 (per run -0.013 to +0.016) |
| `cross_game`, primary beats base on Brier and log-loss (yes/partial/no) | 3/1/2 -> **4/0/2** | 2/3/1 -> 1/3/2 |
| `cross_game --fix-identity`, mean primary AUC | 0.617 -> 0.635 | 0.783 -> 0.783 |
| `pooled_validation` M3 AUC, test G1 / G2 / G3 | 0.597 -> 0.597 / 0.550 -> 0.580 / 0.697 -> 0.720 | 0.829 -> 0.822 / 0.863 -> 0.856 / 0.776 -> 0.788 |
| `pooled_validation`, base-rate heterogeneity chi-square (2 df) | 15.10 -> 16.11 | 6.23 -> 6.35 |

Why A improves (diagnostic, not a methodology change): `best_margin` ranks interceptions at AUC 0.628 [0.596, 0.665] on the kept open-play passes but 0.427 [0.297, 0.561] on the 221 deliveries (26 interceptions, so "no evidence of signal", not "anti-informative"). Dead-ball deliveries dilute the score.

**Statements in the reports that would be stale if the filter is adopted** (not edited here):
* `pass_completion_model.md` section 0/6: scenario A "Brier n.s." (now -0.0018 [-0.0034, -0.0002]*). The verdict "no useful decision-time probability" is unaffected: the gain is ~2% of the base-rate Brier (0.097).
* `cross_game_validation.md` section 5.1: "beats base on both in 3/6, partially 1" -> 4/6, 0; primary A AUC "0.56-0.69" -> 0.59-0.72; "lowest CI bound 0.506" -> 0.527; boosted trees "log-loss in 6/6" -> 5/6 (G3->G1 CI upper bound +0.0002). The two G3-trained runs still fail against the base rate. Section 5.4 "A: 0.57 (G2), 0.61 (G1), 0.68 (G3 as test)" -> 0.60 / 0.61 / 0.71.
* `pooled_validation.md` section 7: A "AUC 0.55-0.70" -> 0.58-0.72; B "fails against BASE on Brier in one fold" (G3 M3 +0.0064*) -> +0.0043, CI now includes 0.
* **Unchanged:** ranking transfers but probabilities do not; interception base rates differ by game (chi-square p = 0.0003 / 0.04); scenario B conclusions and selection bias; the Game 3 identity-swap caveat; all feature-set selections.

## 3. Proposed change (not applied)

`reports/set_piece_delivery_exclusion.patch` (+31/-6 lines in `src/pass_dataset.py` and `src/pass_completion_features.py`):
* new `set_piece_delivery_mask(events, max_gap_s=1.0)`: a `PASS`/`BALL LOST` whose immediately preceding event (full table, chronological, ties in file order) is a `SET PIECE` of the same team, same player, same period, ending <= 1.0 s earlier. Only type/team/player/period/time columns are read (no outcome information, so it respects the leakage contract).
* `candidate_events(events, exclude_set_piece_deliveries=True)` drops them; `label_events(...)` gives them outcome `None` with reason "unknown: set-piece delivery ..." (same mechanism as `PASS with subtype`; they show up in the `unknown` counts, as shown in the rerun accounting, e.g. G1: 72).
* `exclude_set_piece_deliveries=False` restores the legacy population, so earlier reports stay reproducible.

Test: `tests/test_set_piece_exclusion.py` (synthetic events: three deliveries must be excluded, including a failed delivery and one at 0.6 s; five near-misses must be kept: different player / team / period, gap 1.5 s, and open-play passes). On the current code 3 of 4 fail; with the patch the whole suite passes (167 passed, 1 xfailed). Note that the branch's test run is red until the patch is applied.

Design choices to confirm before adopting:
* The rule excludes **all** restart deliveries, including short free kicks and kick-offs (ground passes). A narrower rule (throw-ins, corners, free kicks only) is a one-line subtype filter.
* The 1.0 s window follows the exploratory check. Deliveries beyond it stay in: 3 events (a `PASS` at 1.44 s in G1, `BALL LOST` at 1.96 s and 3.24 s); a 2 s window would add the first two, a 5 s window all three (G2's `BALL LOST` at 3.24 s is the third), and nothing else.
* Restarts without a `SET PIECE` marker (goal kicks, and "dead-ball situations that are not a SET PIECE" per the definitions) are not covered.
* `label_events` is also used by `src/idsse_pilot.py::metrica_reference_events`, `research/choice/build.py`, `research/dq/proto_build.py` and `research/dq/tf_metrica.py`; adopting the default changes their populations. The pass-feasibility and choice studies were **not** rerun (outside the requested scope).
