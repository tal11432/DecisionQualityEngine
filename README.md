# Football Decision Quality Engine (research MVP)

Goal: eventually measure whether a player chose the best available attacking action
(**Decision Quality**), separately from **Execution Quality** and **Outcome**.
Case study planned: Cristiano Ronaldo, Real Madrid 2011/12.

**Current scope: tracking-data foundation only.** Development uses the public Metrica
Sports sample games. These are *development fixtures, not Ronaldo data*; nothing here
describes Ronaldo, and no Decision Quality is computed. Public Ronaldo 2011/12 tracking
data is not assumed to exist.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate   # Python >= 3.11
pip install -r requirements.txt
python -m pytest                                      # no downloads needed
```

## Run
```bash
python -m src.cli download --game 1          # lists files + sizes (~66 MB), downloads nothing
python -m src.cli download --game 1 --yes    # downloads to data/metrica/Sample_Game_1/ (git-ignored); games 1, 2, 3 (~66, ~64, ~55 MB)
python -m src.cli validate --game 1          # load + validation report + event sync summary
python -m src.cli plot --frame 50000 --out outputs/frame.png
python -m src.cli plot --time 600.0 --orient Home
python -m src.cli plot --event 1 --orient Away    # event + its synchronized frame
```
`--orient TEAM` rotates the view so that team attacks towards +x. Games 2 and 3 work the same (`--game 2`, `--game 3`; Game 3 is read through the EPTS adapter automatically).

Python API: `read_tracking`, `read_events` (`src/ingestion/metrica.py`),
`synchronize_events`, `validate_tracking`, `snapshot_features`, `plot_frame`, `plot_event`.

## Architecture
| Path | Role |
|---|---|
| `config/datasets.toml` | pitch size, pinned data commit, sync tolerance, validation thresholds |
| `src/schema.py` | `TrackingData` (wide frame table, metres), `Snapshot`, `Pitch` |
| `src/ingestion/` | `metrica.py` (CSV adapter), `download.py` (opt-in, commit-pinned) |
| `src/processing/` | `coordinates.py` (normalized→metres, attack direction), `sync.py`, `validation.py` |
| `src/analytics/` | `geometry.py` (features), `feasibility.py` (time-to-intercept pass feasibility), `interfaces.py` (future modules) |
| `src/visualization/pitch.py` | pitch, players, ball, labels, event overlay |
| `tests/` | synthetic fixtures only |

Canonical coordinates: metres, origin at pitch centre, x along length, y up (Metrica's y
points down and is flipped), x orientation *as recorded*. Attacking direction per
(team, period) is **inferred from kick-off positions** (each team's median x) and errors
out if ambiguous. Geometry functions expect oriented data (goal at `(+L/2, 0)`).
Pitch size is config, not assumed: Metrica documents 105×68 m.

Sync: events are matched to the nearest frame **by timestamp within the same period**,
within `sync.tolerance_s`. Events with no frame in tolerance, missing time, missing period
or an ambiguous (duplicated) timestamp are reported as unmatched with a reason; provider
frame numbers are only cross-checked (`frame_mismatch`).

Validation checks: player ids, missing values, duplicate frames, invalid coordinates,
timestamp consistency, continuity (frame gaps, implausible speeds). Findings are
reported, never silently repaired.

Future interfaces (`src/analytics/interfaces.py`): action-value Q (xG/EPV), pass
completion, alternative-action generation, Execution Quality, and
`decision_quality = Q(selected) − max Q(alternatives)`. This is a **design interface, not a
validated model**; DQ and EQ are kept as separate quantities.

## Pass feasibility (decision-time score)
`src/analytics/feasibility.py`: ball moves along the pass line at constant speed; an opponent can
intercept if `reaction + distance/speed` ≤ ball time at some lane point. Output `margin_s` (≥ 0 ⇒
interceptable) is a geometric **score, not a probability**; no acceleration model.

**Methodology (leakage-audited; see `reports/pass_feasibility_leakage_audit.md`; note that its target is identified from 0.24 s of post-release ball flight, so it is an early-flight evaluation, not decision-time).** An earlier version used
the event's end point as the pass target (the interception location for intercepted passes) and
central-difference velocities; both leak outcome/future information and were removed. Now:
- Decision frame = ball-release frame (+ optional earlier offset). Inputs: passer, target-teammate and
  opponent positions at that frame; opponent velocities by **causal** backward difference
  (`processing/kinematics.decision_velocities`); fixed parameters. Never end points, interception
  locations, later frames or later events (`src/pass_dataset.py`, config `[decision_time]`).
- Labels from event type only (completed = `PASS`; intercepted = `BALL LOST`/`INTERCEPTION`).
- Metrica gives no intended receiver for failed passes, so the aimed-at teammate is identified from
  the ball's early post-release heading (label side only, identical for both classes; ambiguous or
  unidentified passes, including passes into space, are excluded and counted).
- Parameters: ball speed is a measured population prior (11.9 m/s, Game 1); player speed (7.0 m/s) and
  reaction time (0.3 s) are **assumptions**.

Run: `python -m src.calibration --dev 1 --test 2 --out reports/pass_feasibility_results.txt`
(needs both games downloaded). Held-out Game 2 (37 intercepted / 554 completed after filtering):
AUC 0.85-0.86 vs 0.66 for pass length alone; calibration and velocities give no significant gain.
Evidence of decision-time signal for *identified-target* passes only; no probabilities, no coverage of
passes into space, ~1/3 of intercepted passes evaluable. Details and caveats in the report.

## Pass completion probability (research, leakage-safe)
`python -m src.pass_completion --train 1 --test 2 --out reports/pass_completion_results.txt --figdir reports` trains on Sample Game 1 and
scores Game 2 once (needs both games). Full methodology, results and limits: `reports/pass_completion_model.md`.
- **Target:** P(completed) among *completed* (`PASS`) vs *intercepted* (`BALL LOST/INTERCEPTION`) passes; other possession losses are unknown and excluded, not relabelled.
- **Two separate scenarios.** **A (decision time, target-free):** features at ball release from passer state, pressure and option-set feasibility aggregates;
  ROC-AUC 0.58 [0.53, 0.64] (exploratory boosted trees 0.65): weak, probabilities close to the base rate. **B (early flight, release + 0.24 s, identified target):** includes
  pass-specific geometry; ROC-AUC 0.77-0.88, only ~1/3 of interceptions evaluable, 37 held-out positives. B is **not** a decision-time prediction.
  The intended receiver cannot be identified at/before release (30-35% top-1 vs 10% chance), and failed passes carry no receiver label.
- Primary model = ridge logistic regression (numpy); boosted trees are exploratory; all fitting and calibration on Game 1 only; reliability diagrams in `reports/`.
- Limits: two games, small positive classes, selection bias in B (retains 60% of completed but ~35% of intercepted passes), no passes into space, no Ronaldo inference.
  Not yet suitable for action-value or Decision Quality scoring.

## Cross-game validation (Games 1, 2, 3)
```bash
python -m src.cross_game --out reports/cross_game_results.txt --figdir reports                       # data as supplied
python -m src.cross_game --fix-identity --out reports/cross_game_results_fix_identity.txt            # sensitivity (opt-in repair)
```
Validation gate first (evaluation is skipped if a game fails), then train-on-one/test-on-another for all six ordered pairs, scenarios A and B
separately, with unchanged features/models and train-only fitting. Findings (`reports/cross_game_validation.md`): geometric feasibility generalises as a
*ranking* signal (B: AUC 0.78-0.86 on identified-target passes; A: 0.56-0.69, weak) and beats pass distance, but probabilities do not transfer (game base rates
differ; probability skill vs the training base rate in only about half of the runs) and B covers a biased third of interceptions. Game 3 has a second-half
tracking-identity swap (shirts 6/13) that is detected and reported; by default those passers are excluded by the possession filter.

## Data provenance & licence
Source: <https://github.com/metrica-sports/sample-data>, pinned to commit
`e706dd506b360d69d9d123d5b8026e7294b13996`. The repository carries no formal licence
file; its README asks users to use the data responsibly and **acknowledge Metrica Sports
as the source** in any public use. Data is anonymised. Check upstream terms before
redistributing; this repo never commits the data. `MANIFEST.json` (URLs, sizes, SHA-256) is
written next to the download; the hashes are local records, not upstream-published checksums.

## Limitations
- Metrica CSV (Sample Games 1–2) and EPTS/FIFA + JSON (Sample Game 3, `src/ingestion/epts.py`) are supported; other providers are not.
- Sample Game 1 contains missing data (ball untracked ~39% of frames; off-pitch/implausible-speed
  points) which are flagged, not corrected. Substitutes and unused players are NaN.
- Attack direction relies on kick-off formation; a second source of truth (GK position, metadata) is not yet used.
- Passing-lane obstruction is a static geometric heuristic; pass feasibility is an uncalibrated-probability geometric score evaluated only on identified-target passes (see above).
- Event end times are not synchronized by default (`synchronize_events(time_col="end_time_s")` supports it).
- Tracking is in wide format (one row per frame); fine for one match, not tuned for many.
