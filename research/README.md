# Research scripts (exploratory; not part of the validated `src/` pipeline)

Scripts, pre-registrations, outputs and fitted coefficients behind the Decision Quality reports. Run everything from the repository root. Derived data tables (`*.pkl`) are not committed, except three small coefficient files; regenerate them with the scripts below.

| Folder | What | Report |
|---|---|---|
| `choice/` | Pass-choice model: candidate sets (`build.py` -> `coverage.py` -> `model.py` -> `extra.py`), motion-direction hypothesis `h1.py` | `reports/pass_choice_modeling_exploratory.md` |
| `shots/` | Metrica shot sample and candidate geometry (`shots2.py` -> `shots3.py`, writes `shots_full.pkl`) | `reports/shot_vs_pass_feasibility.md` |
| `dq/` | DQ pipeline stages 1-4 and shot vs pass | `reports/dq_pipeline_run.md`, `reports/shot_vs_pass_feasibility.md`, `reports/dq_expert_protocol.md` |

Pipeline in `dq/`:
* **Stage 1, value models:** `fetch.py` / `fetch2.py` stream StatsBomb open data, keeping nothing raw on disk; then `stage1.py` and `stage1b.py`.
* **Stage 2, target identification in 360 frames:** `stage2.py`, `stage2b.py`.
* **Target-free pass-success (xP) hazard model:** `tfcommon.py`, `tf_fetch.py`, `tf_metrica.py`, `tf_eval.py`, `tf_eval2.py`.
* **Stage 4 prototype:** `proto_build.py`, then `proto.py`.
* **Expert panel power:** `power.py`.
* **Shot vs pass:** `metrica_shots.py`, then `shot_vs_pass.py` (needs `shots/shots_full.pkl` and the `choice/` outputs).

Pre-registrations (`PREREG_*.txt`, `PREREGISTRATION*.txt`) were written, with UTC timestamps, before the analysis they govern. Outputs (`*_out.txt`, `coverage.txt`) are the printed results quoted in the reports.

Fitted coefficients:
* `stage1b_models.pkl`: V, Vopp and xG; logistic models on a standardised basis (`mu`, `sd`, `b`).
* `tf_model.pkl`: the v2 interception-hazard model, i.e. xP.
* `tf_model_v3.pkl`: the visibility-censored variant.

`models.json` holds the same coefficients in readable form, plus the Metrica leave-one-game-out recalibration parameters and the constants k and mean V.

Data: StatsBomb Open Data (free for research, attribution required) and the Metrica Sports sample data. Exploratory status and every caveat are stated in the reports. Nothing here is a validated player rating.
