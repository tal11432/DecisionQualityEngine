"""xG inputs for the 50 Metrica open-play foot shots (pre-shot tracking only), in StatsBomb units."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from src.cli import _load
from src.pass_completion_features import _Track
from src.processing.sync import synchronize_events
SP = "research/shots/"
S = pd.read_pickle(SP + "shots_full.pkl"); S = S[S.step2]
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]
def sb(x, y): return (x + 52.5) / 105 * 120, (y + 34) / 68 * 80     # attack-oriented metres -> StatsBomb units (linear)
GX = 120.0
def ang(x, y): return float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(px, py, sx, sy):
    def s(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = s(px, py, sx, sy, GX, 36), s(px, py, GX, 36, GX, 44), s(px, py, GX, 44, sx, sy)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
rows = []
for g in (1, 2, 3):
    td, ev = _load(cfg, g); T = _Track.of(td); sub = S[S.game == g]
    mm = synchronize_events(ev.loc[sub.idx, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    for _, r in sub.iterrows():
        e = ev.loc[r.idx]; r0 = int(mm[r.idx]); sg = T.sign(e.team, r0)
        j = next(k for k in range(len(T.pid)) if T.team[k] == e.team and T.pid[k] == e.from_player)
        sx, sy = sb(T.X[r0, j] * sg, T.Y[r0, j] * sg)
        ok = np.isfinite(T.X[r0]); opp = np.flatnonzero((T.team != e.team) & ok)
        oxy = [sb(T.X[r0, o] * sg, T.Y[r0, o] * sg) for o in opp]
        gk = min(oxy, key=lambda q: np.hypot(GX - q[0], 40 - q[1]))          # ASSUMPTION: goalkeeper = opponent nearest own goal centre
        rows.append(dict(game=g, idx=r.idx, x=sx, y=sy, dist=float(np.hypot(GX - sx, 40 - sy)), angle=ang(sx, sy), header=0,
                         n_tri=sum(intri(q[0], q[1], sx, sy) for q in oxy if q is not gk), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), goal=int(r.goal)))
M = pd.DataFrame(rows); M.to_pickle("research/dq/metrica_shots.pkl"); print(M.describe().round(2).T[["mean", "50%", "min", "max"]]); print("goals", M.goal.sum(), "n", len(M))
