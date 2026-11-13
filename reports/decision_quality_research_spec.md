# Research specification: measuring football Decision Quality (DQ)

Design document only. Nothing here is implemented, trained or computed. Every proposal carries one label:
**[S]** supported by current (Metrica Games 1-3) data, **[D]** requires additional data, **[H]** research hypothesis requiring validation.

## 1. Executive recommendation

1. **We cannot claim to measure DQ today, and should not.** Current assets give a decision-time state (release-frame tracking), a geometric lane-risk ranking, and nothing that values an action.
2. **Adopt a counterfactual-regret definition with an ex-ante reference value (section 2) as the *hypothesis* to be tested, not as an established metric.** It is falsifiable (section 6) and keeps DQ, execution and outcome separate.
3. **The hardest, still-open problems are not modelling problems:** (a) what "realistic alternative" means, (b) how to identify the chosen action of a failed pass, (c) what the value currency is, (d) what ground truth makes a decision "good". Each needs a pre-registered decision (section 9); (d) needs human expert annotation.
4. **Priority order:** freeze the specification and falsification criteria first (no data), then a pass-only prototype on borrowed value models (needs approved external data), then an expert-annotation validation. Kill the program if the validation criteria in section 6 fail; do not extend to player-level or Ronaldo analysis before they pass.
5. **Ronaldo 2011/12 analysis is not supportable** with tracking (none public); any event-only variant would be a different, weaker construct and must be labelled so.

## 2. Proposed formal DQ definition

**Objects.** At release time *t* the decision-maker faces a state *s* (everything observable at or before *t*), chooses an intended action *d* = (type, target). Execution *e* is how *d* was physically carried out; outcome *o* is what happened.

**Action value (ex ante).** `Q(s, d) = E[ V(next state) | s, d, execution drawn from a reference execution distribution ]`, V a possession-value currency (section 3). Q depends on *s* and *d* only; it contains the *risk* of failure but not the realised failure.

**DQ (primary hypothesis, D1: regret).** For a pre-specified option set `A(s)` (section 4): `DQ = Q(s, d_chosen) - max_{a in A(s)} Q(s, a)` (<= 0; 0 = the best available option). Equivalent to the PDQ form used so far, with the chosen action *in* the set.

| Candidate definition | Benchmark | Strengths | Weaknesses | Status |
|---|---|---|---|---|
| **D1 regret vs best option** | best option in A(s) | interpretable; absolute scale | winner's curse (max over noisy Q rewards large sets); depends on A(s); punishes risk-neutral vs risk-averse styles | primary **[H]** |
| D1' soft regret: `Q(d) - (1/beta) log sum exp(beta Q(a))` | soft-best | less set-size and noise sensitivity | needs beta; harder to explain | sensitivity **[H]** |
| D2 rank/percentile of chosen among A(s) | order only | robust to Q scale errors | ignores magnitude; size-dependent | sensitivity **[H]** |
| D3 policy-relative: `Q(d) - E_{a~pi(.|s)} Q(a)` | typical choice of comparable players | no "best option" assumption | measures unusualness, not goodness | secondary **[D]** |
| D4 expert-referenced: gap to expert-preferred action | human judgement | closest to construct | costly, subjective | validation target **[D]** |

**What exactly is evaluated:** the *choice* among options given the information available at release, not the pass that happened. **Benchmark for "good":** the best option under an ex-ante value model with reference execution (D1), cross-checked against expert judgement (D4). **Avoiding hindsight and outcome leakage:** `A(s)`, `Q` and the state use only frames <= *t*; the outcome and any post-release frame may be used only to *identify which action was chosen* (label-side) and to compute EQ/outcome terms, never in Q or A(s) (tests in section 6). **Failed passes without a known receiver:** DQ is defined on *actions*, not receivers. A pass is identified by its **target cell and lane** (direction sector plus distance band) estimated from the first ~0.24 s of ball flight (existing early-flight window, 84% receiver agreement on completed passes **[S]**). This identification is post-release and is therefore an *action-identification layer with measured error*, not a decision-time input; its selection effect is handled by bounds (section 6). **[H]**

**Information-set caveat.** The player decided before the release frame and sees what tracking does not (body orientation, communication). DQ at release is therefore a *lower-information proxy*; sensitivity to evaluating the state 0.2-0.5 s earlier is part of the plan. **[H]**

## 3. Action-value framework

Common currency: change in possession value over a horizon H (next possession end or fixed seconds), `V = P(team scores next) - P(opponent scores next)`, optionally conditioned on score and time (risk preference). Competing choices for V: **V1 zone-based (xT-like)** (simple, interpretable, estimable from many matches of events) versus **V2 tracking-based EPV** (richer, needs far more data and modelling). Recommend V1 first. **[D]**

| Action | Q(s, a) (reference execution) | Measurable now | Estimable with more data | Unavailable now |
|---|---|---|---|---|
| Pass to target cell | `p_complete * V(position after reception) + (1 - p_complete) * V(loss at target area)` | lane geometry at release; interception margin as a *ranking* feature **[S]** | `p_complete` calibrated per match; V of arrival state; loss value (transition risk) **[D]** | expected value of what happens after reception beyond V1; defender anticipation |
| Shot | `xG(s) + (1 - xG) * V(rebound/loss)` | distance/angle to goal at release **[S]** | xG model (needs thousands of shots; Metrica has about 20-25 per match) **[D]** | goalkeeper position quality, body shape, shot type |
| Carry / dribble | `p_retain(path) * V(end state) + (1 - p_retain) * V(loss)` | free space ahead, pressure geometry **[S]** | duel success model **[D]** | dribbling skill, defender intent; carry labels exist only in Game 3 |
| Retain / safe option | `V(state after safe pass or hold) - time/pressure cost` | nearest-opponent distance, available back options **[S]** | value of keeping the ball vs progression (game-state dependent) **[D]** | team instructions, tempo |

Critical points: **feasibility is not value** (the safest pass is often a back pass to a defender); Q needs the value term, and `p_complete` is *not* transferable across matches today (pooled validation), so per-match recalibration or a hierarchical structure is required **[D/H]**. Components are additive only under the stated reference-execution assumption; verifying consistency between action types (pass vs shot trade-offs) is a validation item, not an assumption **[H]**.

## 4. Realistic-alternative definition

The set `A(s)` must be fixed **before** looking at outcomes, computed from frames <= *t* only, identical for every decision, and independent of the chosen action, the recipient and the result. Hard admissibility criteria (all **[S]** to compute; thresholds are research decisions to pre-register):

| Dimension | Proposed criterion | Notes |
|---|---|---|
| Target set | teammates *and* space cells (e.g. 5 x 5 m grid) reachable by a teammate before any opponent under decision-time kinematics (time-to-arrive) | handles passes into space without an intended receiver |
| Range | min 3 m (existing); max = a population percentile of *chosen* pass lengths (not tuned on outcomes) | a 44 m back pass to a distant teammate is not an alternative |
| Direction/orientation | carrier velocity (and ball) direction before release as a proxy; exclude targets behind a blocking opponent within a short radius | **body orientation is unavailable [D]** (pose data) |
| Offside / pitch | exclude offside receivers and out-of-pitch cells | **[S]** |
| Pressure/lane | *do not* use interception risk to define availability (circular: risky options would vanish and the chosen one would always look best); risk enters through Q | key anti-circularity rule |
| Tactical context | role, formation, instructions | **[D]**; sensitivity only |
| Non-pass actions | shot if inside a pre-set range/angle; carry directions with free space; retain option | **[H]** thresholds |

Competing constructions: **A1 admissibility rules** (above); **A2 behaviour-supported options** (an option is realistic if its feature combination was chosen at least a small fraction of the time by players in comparable states in *other* matches; no outcome use, but inherits typical biases) **[D]**; **A3 expert-annotated option sets** on a sample to measure precision/recall of A1/A2 **[D]**.
**Avoiding outcome dependence:** compute `A(s)` blind to events after *t*; then add the chosen action *only for diagnostics*. A built-in falsification: if `A(s)` excludes more than about 5% of *chosen* actions, the rule is too narrow and fails the **coverage test** (section 6). **[H]**

## 5. Separating DQ, execution and outcome

Realised value `R = Q_ref(s, d) + [E(V | d, e_actual) - Q_ref(s, d)] + [R - E(V | d, e_actual)]`: the first term is **DQ-relevant** (decision), the second **execution quality (EQ)**, the third **outcome luck**. `E(V | d, e_actual)` uses the *actual* ball path, so it is post-release by design and may appear only in EQ/outcome. Reference execution (league-typical error and speed given *d*) must be fixed so that a skilled passer is not credited in DQ for attempting harder passes; the alternative, player-specific Q, merges skill into decision and is an open choice (section 9). **[H]**

| Case | DQ | EQ | Outcome | Lesson |
|---|---|---|---|---|
| 35 m shot with three unmarked teammates in the box; scores into the top corner | low (`Q(shot) ~ xG 0.02` vs pass alternatives with higher value) | high (placement far above reference) | goal | outcome would wrongly reward the decision |
| Through ball into a runner with clear lane (`Q` high); underhit, intercepted | high | low (speed/direction error beyond reference spread) | possession lost | outcome would wrongly punish the decision |
| Safe back pass under pressure, completed, when an unmarked forward option existed | depends on V: low if forward value is much higher, high otherwise | normal | success | success alone proves nothing |
| Rushed 45 m diagonal, deflected and completed by luck | low (outside A(s) or low Q) | n/a | success | completion is not evidence of a good decision |

Execution needs the intended target; for failed passes it is available only through the early-flight identification above (error and selection bias to be quantified). **[D/H]**

## 6. Validation and falsification plan

DQ would be *wrong* (a) if it only reproduces an outcome predictor, (b) if it depends on post-release information, (c) if experts disagree with it beyond what outcome explains, (d) if its ordering flips under reasonable alternative definitions.

| Test | Design | Falsifies the metric if | Status |
|---|---|---|---|
| **Leakage** | invariance: scramble all post-release frames and outcome/receiver columns, DQ must be bit-identical; compute at *t - k* frames (k = 1-5) for stability; negative control with an outcome-aware oracle should be clearly detectable | any change, or erratic jumps over k | **[S]** (extends existing leakage tests) |
| **Holdouts** | leave-one-match-out, then competition/team holdouts, later temporal (season) holdouts; freeze definitions before evaluation | results reverse in held-out matches; between-match variance dominates (needs >= 10 matches) | **[D]** (3 matches only) |
| **Selection bias** | report retention by outcome class and reason at every step; **partial-identification bounds** (assign worst/best Q to unidentified passes), inject realistic action-identification error (about 16% on completed passes), compare identified vs excluded passes on decision-time covariates | conclusions change inside the bounds | **[S]** descriptive |
| **Ranking vs calibration** | evaluate `p_complete` by Brier/slope/intercept per match and ranking by AUC separately; DQ itself is not a probability and is judged by validity tests, not calibration | a ranking-only gain is presented as probability accuracy | **[S]/[H]** |
| **Sensitivity** | set definitions (A1/A2/A3), range cut-offs, horizon H, V1 vs V2, D1 vs D1'/D2/D3, set-size effect (DQ vs `|A(s)|`) | rank correlation of decision/player ordering below a pre-set bound; DQ driven by set size | **[H]** |
| **Construct validity (human)** | independent experts, blind to outcome and metric, rate decision quality and list realistic alternatives on a stratified sample (several hundred decisions incl. failed passes, passes into space, shots, carries) with >= 3 raters | rater reliability too low to validate (e.g. Krippendorff alpha < 0.6); *partial* correlation of DQ with expert ratings, holding outcome fixed, not clearly above zero | **[D]** |
| **Not-just-outcome** | DQ must add to expert agreement beyond `p_complete(chosen)`; DQ and EQ should be weakly related; equal-DQ decisions must differ in outcome only through EQ and chance | DQ ~ `p_complete(chosen)` and adds nothing | **[H]** |
| **Curated logic cases** | expert-designed scenarios where the better option is unambiguous | ordering violations | **[D]** |
| **Reliability** | split-half (odd/even decisions) player and team scores; power analysis for decisions needed per player | low reliability: no player-level claims | **[D]** |

**Evidence required to claim DQ (not outcome prediction):** leakage tests pass; coverage of chosen actions by `A(s)` high; stable under sensitivity; significant, outcome-adjusted agreement with independent experts; incremental validity over feasibility/outcome models; replication on held-out matches and a second data provider. Pre-register thresholds before any evaluation; the numbers above are illustrative and must be set in advance.

## 7. What existing Metrica data can and cannot support

| Supported now [S] | Not supported |
|---|---|
| Release-frame state reconstruction for about 96-97% of labelled passes (existing contract and leakage tests) | value of any action (no xT/EPV/xG; about 3,000 passes, about 70 shots across three matches) |
| Geometric interception margin as a *ranking* feature (AUC about 0.8 on identified-target passes) | calibrated, transferable completion probabilities (pooled validation) |
| Candidate geometry: teammates, distances, lanes; population distribution of chosen pass lengths for range rules | realistic option sets with orientation, roles, tactics (no pose, no formation) |
| Early-flight action identification (60.5% of completed, 35.3% of intercepted passes retained; biased toward directed passes) | intended receivers of failed passes at decision time; passes into space (32.9% of interceptions unidentified) |
| Provider receiver for completed passes (outcome-dependent availability) | expert annotation; between-match variance (3 matches); carry/dribble labels (Game 3 only); score/game-state value effects |

Existing finding to respect: replacing the chosen margin by "chosen minus safest alternative" lowered the ranking AUC (0.818 to 0.733). This is consistent with the safest option being a trivial back pass, i.e. a symptom of the unrealistic option set and the missing value term, and a warning that any option-based metric can be worse than a plain outcome predictor.

## 8. Phased roadmap (gates in bold)

0. **Specification freeze (no data):** choose V, `A(s)`, D1 vs alternatives, thresholds, falsification criteria; pre-register. **[H]**
1. **Pass-only research prototype:** space-cell option set A1, V1 from externally approved event data, ex-ante `Q`, D1 with D1'/D2 sensitivities; **gate:** leakage tests pass, coverage of chosen actions >= pre-set level, sensitivity ordering stable. **[D/H]**
2. **Validation study:** expert annotation, leave-one-match-out with >= 10 matches, selection-bias bounds; **gate:** construct-validity and not-just-outcome tests pass, otherwise stop. **[D]**
3. **Expansion:** shots (xG), carries/dribbles, retain; unify in the common currency and test cross-action consistency (e.g. shot vs pass trade-offs); game state. **[D/H]**
4. **Player-level analysis:** needs reliability, power (decisions per player, role/team/opposition adjustment, shrinkage, uncertainty intervals), and the same tracking provider across many matches. **Ronaldo 2011/12:** no public tracking; options are a negotiated tracking dataset or an event-only proxy (different construct, lower validity, must not be presented as the same metric). **[D]**

## 9. Key unresolved research decisions

1. **Value currency and horizon:** xT-style vs EPV vs goals-added; role of score/time (risk appetite). Competing, no obvious winner.
2. **Option realism:** A1 vs A2 vs A3; numeric thresholds; whether to include non-pass actions in the first prototype.
3. **Benchmark:** best option (D1), soft-best, rank, policy-average or expert; how to handle noisy Q (winner's curse) and set-size dependence.
4. **Reference execution:** league-average vs player-specific; skilled players legitimately attempt harder passes, so this choice trades DQ against EQ.
5. **Chosen-action identification for failed passes:** early-flight target cell, bounds-only aggregate analysis, or restriction to identifiable passes; the 0.24 s window is a compromise, and its selection effects are not removable.
6. **Information set:** release-time proxy vs earlier decision time; the effect of missing body orientation.
7. **Ground truth:** which experts, what protocol, what agreement is enough to call the construct validated.
8. **Pooling:** whether data from different providers and frame rates can be pooled (earlier audits found release timing unusable in IDSSE and SkillCorner), and how to estimate between-match variance.
9. **Success criterion for abandoning the program:** pre-commit to what failure looks like, so that a negative result is acceptable.
