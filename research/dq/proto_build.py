"""DQ-v4 prototype inputs from Metrica (existing local data): situations, candidates, opponents at release."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from src.cli import _load
from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.pass_completion_features import _Track, build_features, label_events
from src.pass_dataset import DecisionContract, build_decision_samples
from src.processing.sync import synchronize_events
D = "research/dq/"
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; C = DecisionContract.from_config(cfg); P = FeasibilityParams.from_config(cfg); W = C.velocity_window_frames
cands, sits = [], []
for g in (1, 2, 3):
    td, ev = _load(cfg, g); out = label_events(ev).set_index("event_idx").outcome.dropna()
    kept = [i for i in build_features(td, ev, "A", tol, C, P).table.event_idx if out.get(i) in ("completed", "intercepted")]
    asg = build_decision_samples(td, ev, tol, C).assignments.set_index("event_idx")["assigned"]
    T = _Track.of(td)
    sync = synchronize_events(ev.loc[kept, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    for idx in kept:
        e = ev.loc[idx]; r0 = int(sync[idx]); team = e.team; sg = T.sign(team, r0)
        chosen = e.to_player if out[idx] == "completed" else asg.get(idx)
        chosen = None if (chosen is None or (isinstance(chosen, float) and np.isnan(chosen))) else chosen
        j = next(k for k in range(len(T.pid)) if T.team[k] == team and T.pid[k] == e.from_player)
        origin = np.array([T.X[r0, j], T.Y[r0, j]]); ok = np.isfinite(T.X[r0]) & np.isfinite(T.Y[r0])
        mates = np.flatnonzero((T.team == team) & ok & (np.arange(len(T.pid)) != j)); opp = np.flatnonzero((T.team != team) & ok)
        opp_xy = np.column_stack([T.X[r0, opp], T.Y[r0, opp]]); dt = float(T.time[r0] - T.time[r0 - W])
        opp_v = np.column_stack([T.X[r0, opp] - T.X[r0 - W, opp], T.Y[r0, opp] - T.Y[r0 - W, opp]]) / dt
        v = origin - np.array([T.X[r0 - W, j], T.Y[r0 - W, j]]); speed = float(np.hypot(*v) / dt)
        ox = np.sort(opp_xy[:, 0] * sg); line = max(ox[-2], origin[0] * sg)
        key = g * 100000 + idx
        sits.append(dict(key=key, game=g, idx=idx, t=float(e.start_time_s), period=int(e.period), team=team, passer=e.from_player, outcome=out[idx], chosen=chosen,
                         px=origin[0] * sg, py=origin[1] * sg, speed=speed, p_press=float(np.hypot(*(opp_xy - origin).T).min()),
                         opps=opp_xy * sg))
        for m in mates:
            xy = np.array([T.X[r0, m], T.Y[r0, m]]); vec = xy - origin; d = float(np.hypot(*vec)); xo = xy[0] * sg
            cands.append(dict(key=key, game=g, cand=T.pid[m], dist=d, forward=xo - origin[0] * sg, lateral=abs(xy[1] - origin[1]), x_att=xo, y_att=xy[1] * sg,
                              rec_press=float(np.hypot(*(opp_xy - xy).T).min()), margin=float(np.clip(interception_margin(origin, xy, opp_xy, P, opp_v)["margin_s"], -10, 10)),
                              offside=bool(xo > line and xo > 0), cos_vel=float(vec @ v / (d * np.hypot(*v) + 1e-9)), is_chosen=bool(chosen is not None and T.pid[m] == chosen)))
Cd, Sd = pd.DataFrame(cands), pd.DataFrame(sits)
Cd["F1"] = np.where(Cd.key.map(Sd.set_index("key").speed) >= 1.0, Cd.cos_vel, 0.0)
Cd["R0"] = (Cd.dist >= 3) & (Cd.dist <= 40) & ~Cd.offside; Cd["R_loose"] = (Cd.dist >= 3) & (Cd.dist <= 60) & ~Cd.offside
pd.to_pickle((Cd, Sd), D + "proto_in.pkl")
print(Sd.groupby(["game", "outcome"]).size().unstack(), "\nchosen known:", Sd.groupby("outcome").chosen.apply(lambda s: s.notna().sum()).to_dict())
