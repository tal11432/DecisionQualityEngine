# Data expansion audit: sources for >= 10 additional matches with tracking + events

Scope: research/feasibility audit only. No models, features or pipeline code were changed and **no dataset was downloaded**. Only small metadata (READMEs, license files, match lists, Git-LFS pointer files, one 4 MB event CSV and a 1.5 MB prefix of another, kept in `/tmp` outside the repository) was fetched. Several hosts were unreachable from this environment (figshare, huggingface.co, pff.com, nature.com, kloppy docs, Google Drive); claims depending on them are marked as secondary or unverified.

Evidence labels: **[V]** verified by me in this session (how is stated); **[S]** from a secondary source (search results, third-party mirrors); **[A]** assumption/estimate, not verified.

## 0. Bottom line

| Rank | Candidate | Matches | Tracking | Event labels (pass outcome) | Intercepted vs completed | Verdict |
|---|---|---|---|---|---|---|
| 1 | **IDSSE / DFL-Sportec** (Bundesliga + 2. Bundesliga 2022/23) | 7 | optical (TRACAB), 25 Hz **[S]**; kloppy loader **[V]** | successful/unsuccessful + recipient for completed **[V in kloppy code]** | interception not directly labelled; derivable? **[A]** | **GO: quality anchor** |
| 2 | **SkillCorner Open Data** (A-League 2024/25) | **20** **[V]** | broadcast, 10 fps, partly extrapolated **[V]** | successful/unsuccessful/offside **[V]** | **yes, derivable [V on 1 match]**, plus an intended-receiver link (96% / 81%) **[V]** | **GO after a 3-match pilot** |
| 3 | PFF FC World Cup 2022 | 64 **[S]** | broadcast-derived **[S]**, rate unverified | trained-annotator events **[S]**, schema unverified | unknown | **CONDITIONAL**: license + schema unverified |
| 4 | StatsBomb 360 open data | 426 **[V]** | **freeze frames only [V]** | rich events **[A]** | partly | **NO-GO for this pipeline** (no continuous tracking) |
| 5 | Impect open data | Bundesliga 23/24 **[V]** | none **[V]** | events | n/a | **NO-GO** (no tracking) |
| 6 | Signality / older SkillCorner / Last Row | 3 / 9 / 19 goals **[S]** | yes | no events **[S]** | n/a | **NO-GO** |
| 7 | Stats Perform, Second Spectrum, Hawk-Eye, Tracab, Metrica Elite | many | yes | yes | yes | **NO-GO**: proprietary/paid, no public data **[V in kloppy table]** |

**Recommendation: GO (staged and conditional).** Stage 1 IDSSE (7 matches, highest comparability, CC-BY). Stage 2 a 3-match SkillCorner pilot; if it passes the gates in section 6, extend to all 20. IDSSE alone (7) does **not** reach "10 additional matches"; IDSSE + >= 3 SkillCorner matches does; SkillCorner alone (20) does.

## 1. Current ingestion architecture (what must be reused or extended)

* **Readers:** `src/ingestion/metrica.py` (CSV, games 1-2), `src/ingestion/epts.py` (EPTS/FIFA + JSON, game 3), both producing the canonical `TrackingData` (wide per-frame table, metres, origin at centre, y up, x as recorded, `attack_dir` inferred from kick-off and cross-checked) and a snake_case events table (`team, type, subtype, period, start_*, end_*, from_player, to_player, ...`). `_load` in `src/cli.py` dispatches on `[metrica.games.N].format`.
* **Downloader:** `src/ingestion/download.py` fetches **raw GitHub files pinned to a commit**, opt-in with `--yes`. It cannot fetch Git-LFS objects, Hugging Face, figshare, or Google Drive: each new source needs an extended or separate acquisition step, with explicit approval for large files.
* **Everything downstream** (sync, validation gate, `label_events`, scenario A/B features, models) is provider-independent **provided an adapter emits the same events schema and the Metrica event vocabulary** (`PASS` without subtype = completed; `BALL LOST/INTERCEPTION` = intercepted). Labels are explicitly *not* conditioned on following events, to avoid aftermath bias.
* **Frame-based parameters** (`config [decision_time]`): causal velocity window 5 frames, early-flight window 6 frames, both defined for **25 Hz**; i.e. 0.2 s and 0.24 s. Any non-25 Hz source forces a *seconds-based* re-parameterisation (see section 4).
* Existing sample (cross-game, data as supplied): scenario A 2,983 passes / 334 interceptions in 3 matches; scenario B 1,720 passes / 118 interceptions. The binding problem (pooled validation) is **3 groups**: between-match intercept/slope variance cannot be estimated.

## 2. Candidate evidence

### 2.1 IDSSE: DFL / Sportec Solutions open dataset (rank 1)

| Property | Finding | Status |
|---|---|---|
| Matches | 7 (2 Bundesliga + 5 2. Bundesliga, 2022/23) | [S] search results for the Scientific Data paper; kloppy's loader lists 7 ids [V] (see discrepancy below) |
| Tracking | positions of all players and the ball, 25 Hz, TRACAB optical; kloppy file template `positions_raw_observed` | [S] frequency/technology; [V] file naming in kloppy source |
| Events | raw event XML with timestamps; pass `Evaluation` in {`successfullyCompleted`, `successful`} -> complete, else incomplete; `Recipient` for completed passes (may be missing per docs); `BallClaiming` event exists; kloppy infers `OUT` for passes before throw-ins/goal kicks | [V] kloppy deserializer source (raw.githubusercontent.com/PySport/kloppy) |
| Coordinates / sync / identities | not inspected (data not downloaded); kloppy transforms to canonical coordinate systems | [A] |
| License | CC-BY 4.0, credit DFL and cite Bassek, Rein, Weber, Memmert, *Sci Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y | [S]; primary page (figshare doi:10.6084/m9.figshare.28196177) unreachable here |
| Access | kloppy's `sportec.load_open_*` downloads from the Hugging Face mirror `pysport/idsse-data` | [V] URL template in kloppy source; mirror unreachable here, sizes unknown |
| Discrepancy | kloppy source lists ids J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9; a Hugging Face card seen via search lists J03WPF and J03WQF instead of J03WQQ/J03WR9. Count is 7 either way | [V] vs [S]: reconcile on download |

Links: [paper](https://www.nature.com/articles/s41597-025-04505-y), [PMC copy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/), [Hugging Face mirror](https://huggingface.co/datasets/pysport/idsse-data), [kloppy sportec loader](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/event/sportec/deserializer.py).

**Labels.** Completed passes are cleanly labelled with a recipient. Failed passes are `INCOMPLETE` (no interception flag in the kloppy mapping). Whether interceptions are separable (via `BallClaiming` type/qualifiers or the next event) is **unverified**; this is the main labelling risk. **Compatibility:** A and B should run with *unchanged feature definitions* (25 Hz, optical, full-pitch coverage, same technology class as Metrica). Highest comparability of any candidate.

### 2.2 SkillCorner Open Data, A-League 2024/25 (rank 2)

| Property | Finding | Status |
|---|---|---|
| Matches | `matches.json` lists **20** (README text says 10); all 20 have `*_dynamic_events.csv` (3.8-5.5 MB) and a Git-LFS tracking pointer | [V] HEAD/GET of all 20 |
| Tracking size | 20 LFS objects, **1,823 MB total** (85-102 MB each) | [V] pointer files |
| Tracking nature | **broadcast** tracking (computer vision on TV video), 10 fps, `is_detected` flag per player, players off-screen are **extrapolated**; "around 97% of player identities accurate"; "some data points are erroneous; apply speed/acceleration smoothing" | [V] README; 10 fps also hard-coded in kloppy's loader [V] |
| Coordinates | metres, centre origin, x long axis; pitch 105 x 68 in `match.json` (checked on 1 match). **Event x/y "are not scaled to standard pitch size and require adjustment"** | [V] README + match.json |
| Events | dynamic events: `player_possession`, `passing_option`, `off_ball_run`, `on_ball_engagement`; per pass `pass_outcome` in {successful, unsuccessful, offside} | [V] one full match (2017461) |
| Intended receiver | `targeted_passing_option_event_id` links a pass to a `passing_option` with `targeted=True`: **96% of 517 successful and 81% of 134 unsuccessful passes** in the checked match | [V] one match |
| Intercepted vs completed | of 134 unsuccessful passes the next possession's `start_type` is `pass_interception` for **80** (60%), `recovery` 21, `keep_possession` 16, `throw_in_interception` 4, other 13; 81% are followed by an opponent possession (median gap 2.2 s). Intercepted rate = 80/654 = 12.2% of passes, comparable to Metrica A (8.9-14.4%) | [V] one match; **label derived from the following possession** |
| Event timing | 24% of successful and **39% of unsuccessful** passes are zero-duration (`frame_start == frame_end`); median pass duration 12 frames (1.2 s). The release-frame semantics (and whether failed passes are timed at release or at loss) are **not documented in anything reachable** (spec PDF blocked) | [V] counts; semantics [A] |
| License | repository MIT (LICENSE header read); README asks users to credit SkillCorner | [V] |
| Loader | kloppy loads SkillCorner **tracking only**; events need a custom reader | [V] kloppy README table |

Links: [repository](https://github.com/SkillCorner/opendata), [LICENSE](https://github.com/SkillCorner/opendata/blob/master/LICENSE), [kloppy tracking loader](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/skillcorner.py).

**Compatibility with the existing pipeline.**
* *Frame rate.* The features are defined in frames at 25 Hz. At 10 fps a 5-frame window is 0.5 s, not 0.2 s. Keeping the *time* definition requires 2 frames (0.2 s, exact), but the 6-frame (0.24 s) early-flight window becomes 2 frames (0.2 s) or 3 (0.3 s): a documented deviation, not an identical feature. Linear up-sampling to 25 Hz is possible but adds no information and would make velocities smoother than the observed data. Either way this must be a pre-registered config profile, not a silent change.
* *Broadcast coverage.* Scenario A uses all opponents/teammates at release and 5 frames earlier. Extrapolated positions are model output, not observation; the share of detected players near the passer and the ball is **unknown until the LFS data is read**. A detection-based eligibility rule is needed, with its exclusion rate reported by outcome (like scenario B's existing selection analysis).
* *Scenario B.* The provider's intended-receiver link could replace heading-based identification for 81% of failed passes. That is a **new target definition**, not the audited B, and (being a post-hoc annotation) must stay label-side. It would remove most of B's outcome-asymmetric exclusion (65-68% of interceptions), but it is a separate, new scenario; recorded here as an opportunity, not planned work.
* *Labels.* "Intercepted" = unsuccessful pass whose next possession starts `pass_interception` is a *different construction* from Metrica's `BALL LOST/INTERCEPTION` (derived from the following event vs provider event type). Cross-provider label harmonisation must be validated, not assumed.

### 2.3 PFF FC / Gradient Sports, FIFA World Cup 2022 (rank 3, conditional)

| Property | Finding | Status |
|---|---|---|
| Matches | 64, tracking + event data, free dataset release | [S] PFF blog |
| Tracking | described as broadcast-derived; kloppy's loader reads per-frame `playersSmoothed` / `ballsSmoothed` with `jerseyNum`, frame rate from metadata `fps`, pitch size from stadium metadata; files `*.jsonl.bz2` + metadata + rosters | [V] kloppy tracking source (frame rate value itself **unverified**) |
| Events | annotated by trained annotators on video and tracking; kloppy event loader is marked "in progress" (⧗) | [S]; [V] kloppy table |
| Pass outcome fields / receiver / interception | unknown | unverified |
| License / access | free but via a request form; "by submitting this form you agree to our Terms of Use"; terms text not retrievable | [S]; **terms unverified** |
| Size | unknown (kloppy README links a Google Drive folder, unreachable) | unverified |

Links: [PFF release post](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data), [enhanced dataset post](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset), [kloppy PFF tracking loader](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/pff.py).
**Assessment:** the largest and most diverse source (32 national teams, one tournament), but access terms, event schema and label quality are unverified, and a custom event reader is required. Defer until the license is read and a sample is inspected.

### 2.4 Rejected or out-of-scope sources

* **StatsBomb 360** ([repo](https://github.com/statsbomb/open-data)): **426 matches** with 360 data across 12 competition-seasons **[V]** (e.g. World Cup 2022 64, Euro 2020 51, Euro 2024 51, Bundesliga 2023/24 34). But one sample file shows what it is: freeze frames at event moments, 20 players in frame (median 15 visible across 3,195 frames of that match), only `teammate/actor/keeper` flags and a location, **no player identities and no continuous frames [V]**. Causal velocities (needed by scenario A's margin) and post-release ball flight (scenario B) are impossible. It could support only a *different*, static-geometry study with new features, which is outside this task's constraints.
* **Impect open data** ([repo](https://github.com/ImpectAPI/open-data)): Bundesliga 2023/24 **event data only** [V README].
* **Signality/Hammarby (3 matches), the older SkillCorner/Friends-of-Tracking set (9 games), Last Row (19 goals):** tracking without events or only goal clips, per the Metrica README [S: github.com/metrica-sports/sample-data].
* **Commercial providers** (Stats Perform, Second Spectrum, Hawk-Eye, Tracab, Metrica Elite, Wyscout): supported by kloppy but with no public data [V kloppy table]; excluded (no paid access).

## 3. Verification matrix (the task's checklist)

| Requirement | IDSSE | SkillCorner | PFF | StatsBomb 360 |
|---|---|---|---|---|
| Usable matches | 7 [S/V] | 20 [V] | 64 [S] | 426 but unusable |
| Player + ball tracking | yes [S] | yes, broadcast, partial/extrapolated [V] | yes, broadcast [S] | **no (freeze frames)** [V] |
| Event labels | yes [V code] | yes [V] | yes [S], schema unknown | yes [A], tracking missing |
| Timestamp synchronisation | unknown until inspected | unknown; release semantics unclear [V counts] | unknown | n/a |
| Player identity consistency | unknown [A: optical, IDs in both files] | ~97% stated [V]; check with `detect_identity_swaps`-style test | unknown | no IDs [V] |
| Coordinates / frequency | converted by kloppy [A] / 25 Hz [S] | metres, centred; event coords need rescaling; 10 fps [V] | smoothed coordinates, fps in metadata [V code] | 120 x 80 pitch frame [A] |
| License / access | CC-BY 4.0 [S] | MIT + credit [V] | form + Terms [S], unread | StatsBomb user agreement [A] |

## 4. Compatibility with scenarios A and B

| | IDSSE | SkillCorner | PFF |
|---|---|---|---|
| Scenario A (release frame, causal 0.2 s velocities, all players) | Same definitions; needs only an events adapter | Needs seconds-based profile (2 frames at 10 fps) and a detection-based eligibility rule; release frame must be validated | Unknown (frame rate, coverage) |
| Scenario B (release + 0.24 s heading) | Same definitions | 0.24 s is 2.4 frames: deviation (0.2/0.3 s); heading estimates much noisier at 10 fps | Unknown |
| Outcome labels | Completed solid; interceptions need derivation (BallClaiming / next event) [unverified] | Completed solid; interception derivable [V on 1 match] but defined differently from Metrica | Unknown |
| Cross-provider comparability | Highest | Lower: different technology class, rate, label construction | Unknown |

## 5. Expected usable sample

Basis and assumptions are stated; none of the usable-after-exclusion figures can be confirmed without downloading.

| Source | Matches (listed -> expected usable after the gate) | Passes per match | Interceptions per match | Total interceptions (A, after exclusions) |
|---|---|---|---|---|
| IDSSE | 7 -> **6-7** [A] | ~800-1,000 [A: Metrica games had 885-1,030 labelled passes] | ~7-14% -> 60-140 [A] | **~350-700** |
| SkillCorner | 20 -> **12-20** [A] | **654** [V one match] | **~80** (12%) [V one match]; eligibility 60-90% of passes retained [A] | **~700-1,400** from 12-20 matches |
| PFF | 64 -> 40-64 if access is granted [A] | ? | ? | unknown |
| *Existing* | 3 | 872-1,088 (A) | 97-126 | 334 |

IDSSE + a 3-match SkillCorner pilot would give 10 additional matches and roughly 500-900 additional interceptions (about 1.5-2.7x the current 334; IDSSE ~350-700 plus the pilot ~140-220 [A]); all 27 listed matches would give 30 groups, enough to attempt between-match variance estimation, **if** the sources can be pooled (see risks). Scenario B counts would fall further (current B retains ~33-35% of interceptions); no estimate is possible for new sources.

**Implementation effort (assumption, working days):** IDSSE: XML tracking + event adapter, events mapping, validation gate, tests 3-4 (custom reader preferred over adding kloppy as a dependency; kloppy's loaders can serve as a reference); SkillCorner: LFS acquisition, JSONL tracking + dynamic-events adapter, seconds-based profile, label derivation, release-frame validation, tests 5-7; PFF: access, tracking + event readers 6-8 (+ license review).

## 6. Main methodological risks

1. **Source heterogeneity.** Optical 25 Hz vs broadcast 10 fps vs different event vocabularies: *source* becomes another random effect confounded with match. Pool only after showing comparable score-outcome relationships; keep source as a stratum in analysis.
2. **Label harmonisation.** Metrica: provider-labelled `BALL LOST/INTERCEPTION` events; SkillCorner: derived from the following possession's `start_type`; Sportec: unverified. Different constructions may shift base rates by themselves. Needs per-source label audits (like the Metrica verification) before pooling.
3. **Release-frame semantics (SkillCorner).** 39% of failed vs 24% of completed passes are zero-duration events; if the timing convention differs by outcome, scenario A/B inputs would leak or be misaligned. Must be tested against the tracking data (ball distance to passer and ball-speed jump at the event frame, as done for Metrica).
4. **Extrapolated/off-screen positions** (broadcast data). Features may be computed from modelled positions; selection by detection status could correlate with outcome. Needs reporting of exclusions by outcome class.
5. **Identity errors** (SkillCorner ~3% stated; Metrica Game 3 had a systematic swap): run the identity-consistency detector per match.
6. **Coordinate pitfalls:** SkillCorner event coordinates not scaled to pitch size; others unchecked.
7. **Narrow populations:** one league-season (A-League), one tournament (World Cup), one German season: base rates will reflect these contexts; extrapolation to other leagues is unsupported.
8. **Licenses:** IDSSE attribution/citation (CC-BY), SkillCorner credit request, PFF terms unread; never commit data (already enforced by `.gitignore`).
9. **Statistical:** more matches fix the group-count problem only if matches are exchangeable within a source; between-source differences can swamp between-match variance.

## 7. Recommended next dataset and minimal integration plan

**Next dataset: IDSSE (DFL/Sportec).** Reasons: same technology class and rate as Metrica, so scenarios A and B run with unchanged definitions; CC-BY; small; events carry recipients; kloppy offers a verified reference implementation. It is the cleanest test of whether the pipeline and findings transfer to another provider.

Minimal plan (each step gated; **no step runs without explicit approval**):
1. **Approval + inventory (0.5 d):** approve downloading the 7 matches (size unknown; expect hundreds of MB each [A]) from the Hugging Face mirror/figshare into git-ignored `data/`; record file sizes, hashes, license text; resolve the match-id discrepancy.
2. **Adapter (2-3 d):** `src/ingestion/sportec.py` reading positions XML + events XML + match-information XML into `TrackingData` and the **existing events schema**; teams mapped to Home/Away; coordinates verified against ball position at event frames and kick-off attack direction (as done for Game 3); no changes to features or models.
3. **Gate (0.5 d):** the existing validation gate plus identity-swap detection plus new checks: ball-to-passer distance at release, ball-speed jump after release, event/tracking sync >= 99%.
4. **Label audit (1 d):** inspect how interceptions can be derived (BallClaiming type/qualifiers; next event) on the 7 matches, compare the resulting interception rate and chain statistics with Metrica; **stop** if no defensible rule exists.
5. **Evaluation (0.5 d):** run scenario A and B with frozen settings as a *new held-out source* (no tuning), report exclusion rates by outcome and per-match calibration slope/intercept; do not pool with Metrica until step 6.
6. **Stage 2 pilot:** SkillCorner, 3 matches (~270 MB LFS), seconds-based profile registered before looking at results; pass criteria: release-frame check as in step 3, detected-player share near the passer and ball above a pre-set threshold, identity check, derived-interception label rate within the Metrica range. Only then scale to 20 matches (1.8 GB, approval required).

**Acceptance gates for pooling any new source:** (a) validation gate passes, (b) label audit documented, (c) per-source score-outcome slope within the range seen in Metrica or analysed with source as a stratum, (d) exclusion rates by outcome reported.

## 8. GO / NO-GO

**GO, staged and conditional.**
* **GO:** acquire and integrate IDSSE (7 matches) as the quality anchor; then pilot SkillCorner (3 matches) toward >= 10 additional matches in total, or up to 20 if the pilot passes.
* **CONDITIONAL / deferred:** PFF World Cup 2022 (64 matches): read the terms, inspect one sample, check events schema before committing.
* **NO-GO:** StatsBomb 360 (no continuous tracking; incompatible with causal-velocity and early-flight features without new feature definitions), Impect (no tracking), the tracking-only or goal-only sets, and all commercial feeds.
* **Do not proceed to pooling or hierarchical modelling** until at least IDSSE passes its gate and the SkillCorner pilot confirms that release timing and interception labels are comparable. If the SkillCorner pilot fails (release timing or detection coverage), the realistic ceiling for tracking-quality matches with consistent labels is **10 total (3 Metrica + 7 IDSSE)**, i.e. **7 additional**, which would not meet the stated goal of 10 additional matches.
