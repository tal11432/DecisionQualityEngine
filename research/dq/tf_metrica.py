"""Metrica step table: release-frame opponents (all 11 visible), stop point = tracking ball at the event end frame (label side only)."""
import sys, os; D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from tfcommon import steps, FEATS
from src.config import load_config
from src.cli import _load
from src.analytics.feasibility import FeasibilityParams
from src.pass_completion_features import _Track, build_features, label_events
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; C = DecisionContract.from_config(cfg); P = FeasibilityParams.from_config(cfg)
sb = lambda x, y: np.array([(x + 52.5) * 120 / 105, (y + 34) * 80 / 68])
rows, info, drop = [], [], {}
for g in (1, 2, 3):
    td, ev = _load(cfg, g); out = label_events(ev).set_index("event_idx").outcome.dropna()
    kept = [i for i in build_features(td, ev, "A", tol, C, P).table.event_idx if out.get(i) in ("completed", "intercepted")]
    T = _Track.of(td); frow = {int(f): r for r, f in enumerate(T.frame) if np.isfinite(f)}
    sync = synchronize_events(ev.loc[kept, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    for i in kept:
        e = ev.loc[i]; r0 = int(sync[i]); sg = T.sign(e.team, r0); re = frow.get(int(e.end_frame)) if np.isfinite(e.end_frame) else None
        if re is None or not np.isfinite(T.bx[re]): drop[(g, out[i])] = drop.get((g, out[i]), 0) + 1; continue
        j = next(k for k in range(len(T.pid)) if T.team[k] == e.team and T.pid[k] == e.from_player)
        a = sb(T.X[r0, j] * sg, T.Y[r0, j] * sg); end = sb(T.bx[re] * sg, T.by[re] * sg)
        if np.hypot(*(end - a)) < 5: drop[(g, "short-" + out[i])] = drop.get((g, "short-" + out[i]), 0) + 1; continue
        ok = np.isfinite(T.X[r0]) & (T.team != e.team); opps = np.column_stack([sb(T.X[r0, k] * sg, T.Y[r0, k] * sg) for k in np.flatnonzero(ok)]).T
        R = steps(a, end, opps, out[i] == "intercepted", None); pid = len(info)
        info.append((pid, g, int(out[i] == "intercepted"))); rows.append(np.column_stack([np.full(len(R), pid), R]))
S = pd.DataFrame(np.vstack(rows), columns=["pid"] + FEATS + ["event"]); Pi = pd.DataFrame(info, columns=["pid", "game", "failed"]); S = S.merge(Pi[["pid", "game"]], on="pid")
S.to_pickle(f"{D}/steps_metrica.pkl"); print("Metrica passes", len(Pi), "intercepted", Pi.failed.sum(), "by game", Pi.groupby("game").failed.agg(["size", "sum"]).to_dict(), "dropped", drop)
