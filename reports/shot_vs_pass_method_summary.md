# Shot vs pass: method summary (what is validated and what is not)

A general, league-agnostic statistical model. Everything below is built on StatsBomb open data (2,254 men's matches from 2015; 360-frame player positions for about 380 of them) and was tested on competitions held out of training. Details, numbers and caveats: `shot_vs_pass_feasibility.md` (sections 8-15), `dq_pipeline_run.md`; scripts and pre-registrations: `research/`.

## 1. Question
When a player shoots, was there a teammate to pass to who offered a higher expected attacking value than the shot?

## 2. The method in four numbers
For a given moment (positions of the players and the possession so far):
1. **Shot value** = xG of the shot (probability it scores) + a small continuation term (later shots in the same possession within 10 s, for example rebounds).
2. **xP** = probability that a pass to teammate j reaches him, from a hazard model along the pass line (defenders near the line).
3. **Chain value of the pass** = expected xG of all team shots in the same possession within 10 s after a completed pass, given where the ball arrives and who is around.
4. **Pass value** = xP x chain value (a failed pass is valued 0; the loss of possession is almost never followed by an opponent chance in the attacking third: 0.00002).
Compare the shot value with the best pass value over the realistic teammates (3-40 m, not offside by a position proxy). Because the pass value is uncertain by about +-30%, the verdict is **three-way**: pass clearly better / shot clearly better / ambiguous.

## 3. Validation status
| Component | Tested on | Result | Status |
|---|---|---|---|
| xG | 2,983 shots from Euro 2024, Copa America 2024, Bundesliga 2023/24, MLS 2023 | AUC 0.775 (StatsBomb's own 0.785); better than the first model; predicted/observed goals 1.06 | Usable; the calibration gate failed narrowly (slope 0.896 vs 0.9). Tweaks did not help |
| xP | Euro 2020 and WC 2022 cross-validation; Metrica tracking | Calibrated between competitions; over-confident on tracking data (slope 0.80) until recalibrated | Usable per data source; needs local recalibration |
| Chain value of a pass | 8,177 passes from Euro 2024, Bundesliga, MLS | Ranks well (AUC 0.81 for "a shot follows"); level over-predicted by about 12-18%; between-competition spread of the level about +-9% | Usable with a +-30% band; the calibration gate failed |
| Shot continuation | test shots | predicted/observed 1.09 | Passed |
| Cost of a failed pass | 4,692 failed passes | calibrated; negligible in the attacking third | Passed, no effect on the verdict |
| Attempts to improve the chain value (more features, two-part model, 564k-pass baseline) | same test sets | no gain; even the defenders' positions at reception add almost nothing | The data (single frames) is at its ceiling |

## 4. Main results (2,431 foot shots from the held-out competitions)
* **Share of shots with a better pass:** 64% with the central estimate, 56% if the pass value is scaled down to the observed level; an earlier version (optimistic pass value) gave 74%, an earlier lower bound (only the receiver's own shot) 42%. The truth sits in the 56-64% region.
* **Three-way verdict (pass value x0.7 to x1.3):** pass clearly better **47%**, shot clearly better **27%**, ambiguous **26%**.

| Shot distance | Pass clearly better | Shot clearly better | Ambiguous |
|---|---|---|---|
| under 12 m | 27% | **50%** | 22% |
| 12-20 m | 44% | 27% | 29% |
| over 20 m | **62%** | 12% | 26% |

Close shots are mostly the right call; long shots mostly have a better pass available. The pattern is the same in every competition and in both sexes (22 more competitions checked in an earlier replication).
* **What players do:** from the same context features, a shot/pass choice model predicts whether a player shoots with AUC 0.95 (81% of decisions within 8 m are shots, 13% beyond 20 m).

## 5. What is not established
* **Not a judgement of a player or a decision.** The values are expected xG from positions; they ignore the player's information, time pressure, body orientation, and anything after 10 s or after a lost ball.
* **No expert validation.** There is no external reference showing that "the better option by this model" is what coaches consider the better decision. An expert-panel protocol is written (`dq_expert_protocol.md`) but not run.
* **Calibration gates that failed** (xG slope, chain-value level) are reported as failures; the model is used with the uncertainty band rather than as a point estimate.
* **360 frames show only some of the players,** and the pass-value ceiling is set by static frames; tracking data with velocities could do better but was not available at scale.
* **Single-frame applications** (for example, reconstructing a broadcast frame by hand) add a position error of 2-3 m, which can move xG a lot near goal; they are illustrations, not measurements.

## 6. Reproducing
`research/README.md` lists the pipeline. Every analysis has a timestamped pre-registration written before the data was downloaded; raw and derived data are deleted after each run and the scripts regenerate them from StatsBomb open data (attribution required).
