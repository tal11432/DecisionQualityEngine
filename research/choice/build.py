"""Stage 1: candidate sets + coverage (NO model). Reuses existing code read-only."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from src.cli import _load
from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.pass_completion_features import _Track, build_features, label_events
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events

cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; C = DecisionContract.from_config(cfg); P = FeasibilityParams.from_config(cfg)
W = C.velocity_window_frames
rows_c, rows_p = [], []
for g in (1, 2, 3):
    td, ev = _load(cfg, g)
    out = label_events(ev).set_index("event_idx").outcome.dropna()
    fsA = build_features(td, ev, "A", tol, C, P)
    tabA = fsA.table.set_index("event_idx")
    kept = [i for i in tabA.index if out[i] == "completed"]
    T = _Track.of(td)
    sync = synchronize_events(ev.loc[kept, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    # role proxy: match-average oriented x of each player (grouping variable for reporting only)
    per = T.period
    mean_x = {}
    for k in range(len(T.pid)):
        sgn = np.where(per == 1, T.attack[(T.team[k], 1)], T.attack[(T.team[k], 2)])
        v = T.X[:, k] * sgn; ok = np.isfinite(v)
        mean_x[(T.team[k], T.pid[k])] = (float(v[ok].mean()) if ok.any() else np.nan, int(ok.sum()))
    role = {}
    for team in ("Home", "Away"):
        ps = [(m, p) for (t, p), (m, n) in mean_x.items() if t == team and n >= 25000]
        order = sorted(ps)
        for r_, (m, p) in enumerate(order):
            role[(team, p)] = ["deep", "mid", "advanced"][min(2, r_ * 3 // len(order))]
    for idx in kept:
        e = ev.loc[idx]; r0 = int(sync[idx]); team = e["team"]
        j = next(k for k in range(len(T.pid)) if T.team[k] == team and T.pid[k] == e["from_player"])
        sg = T.sign(team, r0)
        origin = np.array([T.X[r0, j], T.Y[r0, j]])
        ok = np.isfinite(T.X[r0]) & np.isfinite(T.Y[r0])
        mates = np.flatnonzero((T.team == team) & ok & (np.arange(len(T.pid)) != j))
        opp = np.flatnonzero((T.team != team) & ok)
        opp_xy = np.column_stack([T.X[r0, opp], T.Y[r0, opp]])
        dt = float(T.time[r0] - T.time[r0 - W])
        opp_v = np.column_stack([T.X[r0, opp] - T.X[r0 - W, opp], T.Y[r0, opp] - T.Y[r0 - W, opp]]) / dt
        v = origin - np.array([T.X[r0 - W, j], T.Y[r0 - W, j]]); speed = float(np.hypot(*v) / dt)
        ox_sorted = np.sort(opp_xy[:, 0] * sg)
        second_last = ox_sorted[-2] if len(ox_sorted) >= 2 else ox_sorted[-1]
        line = max(second_last, origin[0] * sg)
        rows_p.append(dict(game=g, idx=idx, period=int(T.period[r0]), team=team, passer=e["from_player"], px=origin[0] * sg, py=origin[1] * sg, speed=speed,
                           p_press=float(tabA.loc[idx, "nearest_opp_dist"]), n10=int(tabA.loc[idx, "n_opp_10m"]), role=role.get((team, e["from_player"]), "sub"),
                           receiver=e["to_player"], n_mates=len(mates)))
        for m in mates:
            xy = np.array([T.X[r0, m], T.Y[r0, m]]); vec = xy - origin
            d = float(np.hypot(*vec)); xo, yo = xy[0] * sg, xy[1] * sg
            rows_c.append(dict(game=g, idx=idx, cand=T.pid[m], dist=d, forward=xo - origin[0] * sg, lateral=abs(xy[1] - origin[1]), x_att=xo, y_att=yo,
                               rec_press=float(np.hypot(*(opp_xy - xy).T).min()), margin=float(np.clip(interception_margin(origin, xy, opp_xy, P, opp_v)["margin_s"], -10, 10)),
                               offside=bool(xo > line and xo > 0), cos_vel=float(vec @ v / (d * np.hypot(*v) + 1e-9)),
                               is_actual=bool(T.pid[m] == e["to_player"])))
Cd, Pd = pd.DataFrame(rows_c), pd.DataFrame(rows_p)
RULES = {"R0_primary": (3, 40, True), "R_tight": (3, 25, True), "R_loose": (3, 60, True), "R_existing": (3, 1e9, False)}
for name, (lo, hi, off) in RULES.items():
    Cd[name] = (Cd.dist >= lo) & (Cd.dist <= hi) & (~Cd.offside if off else True)
pd.to_pickle((Cd, Pd, RULES), sys.argv[1]); print("passes", len(Pd), "candidate rows", len(Cd))
