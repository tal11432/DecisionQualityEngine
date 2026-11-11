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
python -m src.cli download --game 1 --yes    # downloads to data/metrica/Sample_Game_1/ (git-ignored)
python -m src.cli validate --game 1          # load + validation report + event sync summary
python -m src.cli plot --frame 50000 --out outputs/frame.png
python -m src.cli plot --time 600.0 --orient Home
python -m src.cli plot --event 1 --orient Away    # event + its synchronized frame
```
`--orient TEAM` rotates the view so that team attacks towards +x. Game 2 works the same (`--game 2`).

Python API: `read_tracking`, `read_events` (`src/ingestion/metrica.py`),
`synchronize_events`, `validate_tracking`, `snapshot_features`, `plot_frame`, `plot_event`.

## Architecture
| Path | Role |
|---|---|
| `config/datasets.toml` | pitch size, pinned data commit, sync tolerance, validation thresholds |
| `src/schema.py` | `TrackingData` (wide frame table, metres), `Snapshot`, `Pitch` |
| `src/ingestion/` | `metrica.py` (CSV adapter), `download.py` (opt-in, commit-pinned) |
| `src/processing/` | `coordinates.py` (normalized→metres, attack direction), `sync.py`, `validation.py` |
| `src/analytics/` | `geometry.py` (provider-independent features), `interfaces.py` (future modules) |
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

## Data provenance & licence
Source: <https://github.com/metrica-sports/sample-data>, pinned to commit
`e706dd506b360d69d9d123d5b8026e7294b13996`. The repository carries no formal licence
file; its README asks users to use the data responsibly and **acknowledge Metrica Sports
as the source** in any public use. Data is anonymised. Check upstream terms before
redistributing; this repo never commits the data. `MANIFEST.json` (URLs, sizes, SHA-256) is
written next to the download; the hashes are local records, not upstream-published checksums.

## Limitations
- Only Metrica standard CSV (Sample Games 1–2). Game 3 (EPTS/FIFA + JSON events) not supported yet.
- Sample Game 1 contains missing data (ball untracked ~39% of frames; off-pitch/implausible-speed
  points) which are flagged, not corrected. Substitutes and unused players are NaN.
- Attack direction relies on kick-off formation; a second source of truth (GK position, metadata) is not yet used.
- Passing-lane obstruction is a static geometric heuristic, not pass feasibility.
- Event end times are not synchronized by default (`synchronize_events(time_col="end_time_s")` supports it).
- Tracking is in wide format (one row per frame); fine for one match, not tuned for many.
