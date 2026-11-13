import sys, os; D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from tfcommon import steps, FEATS as HF
from src.config import load_config
from src.cli import _load
from src.pass_completion_features import _Track
from src.processing.sync import synchronize_events
S = pd.read_pickle("research/shots/shots_full.pkl"); S = S[S.step2]
XG = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]; HM = pd.read_pickle(f"{D}/tf_model.pkl"); REC = {1: (-0.764, 0.793), 2: (-0.713, 0.778), 3: (-0.581, 0.819)}
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; GX = 120.0
sb = lambda x, y: np.array([(x + 52.5) * 120 / 105, (y + 34) * 80 / 68])
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def xg(pos, opps, gk, incl_gk=True, extra=0):
    ntri = sum(intri(o, pos) for o in opps if incl_gk or not np.allclose(o, gk)) + extra
    x = np.array([[np.hypot(GX - pos[0], 40 - pos[1]), ang(*pos), 0.0, ntri, np.hypot(GX - gk[0], 40 - gk[1])]])
    return float(1 / (1 + np.exp(-(np.column_stack([np.ones(1), (x - XG["mu"]) / XG["sd"]]) @ XG["b"])))[0])
def xp(a, end, opps, rec):
    R = steps(a, end, opps, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]
    h = 1 / (1 + np.exp(-(rec[0] + rec[1] * lg))); return float(np.exp(np.log1p(-h).sum()))
shots, cands = [], []
for g in (1, 2, 3):
    td, ev = _load(cfg, g); T = _Track.of(td); sub = S[S.game == g]
    mm = synchronize_events(ev.loc[sub.idx, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    for _, r in sub.iterrows():
        e = ev.loc[r.idx]; r0 = int(mm[r.idx]); sg = T.sign(e.team, r0); ok = np.isfinite(T.X[r0])
        j = next(k for k in range(len(T.pid)) if T.team[k] == e.team and T.pid[k] == e.from_player)
        a = sb(T.X[r0, j] * sg, T.Y[r0, j] * sg); am = np.array([T.X[r0, j] * sg, T.Y[r0, j] * sg])
        opps = [sb(T.X[r0, k] * sg, T.Y[r0, k] * sg) for k in np.flatnonzero(ok & (T.team != e.team))]; O = np.array(opps)
        gk = min(opps, key=lambda q: np.hypot(GX - q[0], 40 - q[1]))
        oxm = np.sort([T.X[r0, k] * sg for k in np.flatnonzero(ok & (T.team != e.team))]); line = max(oxm[-2], am[0])
        x_shot = xg(a, opps, gk); x_shot_nogk = xg(a, opps, gk, incl_gk=False)
        shots.append(dict(game=g, idx=r.idx, t=float(e.start_time_s), period=int(e.period), team=e.team, shooter=e.from_player, sub=r["sub"], goal=int(r.goal),
                          dist_m=float(r.shot_dist), xg=x_shot, xg_nogk=x_shot_nogk, sx=am[0], sy=am[1]))
        for k in np.flatnonzero(ok & (T.team == e.team) & (np.arange(len(T.pid)) != j)):
            pm = np.array([T.X[r0, k] * sg, T.Y[r0, k] * sg]); d = float(np.hypot(*(pm - am)))
            if not (3 <= d <= 40) or (pm[0] > line and pm[0] > 0): continue
            p = sb(*pm)
            cands.append(dict(game=g, idx=r.idx, mate=T.pid[k], d_m=d, x_m=pm[0], y_m=pm[1], goal_d=float(np.hypot(GX - p[0], 40 - p[1])) * 105 / 120,
                              xP=xp(a, p, O, REC[g]), xP_raw=xp(a, p, O, (0.0, 1.0)), xA=xg(p, opps, gk), xA_plus1=xg(p, opps, gk, extra=1)))
Sh, Cn = pd.DataFrame(shots), pd.DataFrame(cands); Cn["EPV"] = Cn.xP * Cn.xA
pd.to_pickle((Sh, Cn), f"{D}/shot_vs_pass.pkl")
print(f"shots {len(Sh)}; xG sum {Sh.xg.sum():.2f} (goalkeeper in n_tri, as trained) vs {Sh.xg_nogk.sum():.2f} (earlier convention) vs goals {Sh.goal.sum()}")
def summary(lbl, epv_col="EPV", sub=None):
    c = Cn if sub is None else Cn[sub(Cn)]; best = c.loc[c.groupby(["game", "idx"])[epv_col].idxmax()].set_index(["game", "idx"])
    m = Sh.set_index(["game", "idx"]).join(best[[epv_col, "xP", "xA", "mate", "d_m"]]); m[epv_col] = m[epv_col].fillna(0)
    better = m[epv_col] > m.xg; diff = m[epv_col] - m.xg
    print(f"[{lbl}] shots with a pass option of higher EPV: {better.sum()} of {len(m)} ({better.mean()*100:.0f}%); median (best EPV - xG) {diff.median():+.3f}; "
          f"sum xG {m.xg.sum():.2f} vs sum best EPV {m[epv_col].sum():.2f}"); return m, better
m, better = summary("primary")
m["band"] = pd.cut(m.dist_m, [0, 12, 20, 60], labels=["<12m", "12-20m", ">20m"])
print("by distance: " + "; ".join(f"{b}: {better[m.band==b].sum()}/{(m.band==b).sum()}" for b in ["<12m", "12-20m", ">20m"]) + "   by game: " + "; ".join(f"G{g}: {better[m.index.get_level_values(0)==g].sum()}/{(m.index.get_level_values(0)==g).sum()}" for g in (1, 2, 3)))
Cn["EPV_raw"] = Cn.xP_raw * Cn.xA; Cn["EPV_p1"] = Cn.xP * Cn.xA_plus1
summary("sens a: xP not recalibrated", "EPV_raw"); summary("sens b: candidates 3-25 m", sub=lambda c: c.d_m <= 25); summary("sens c: +1 defender in xA triangle", "EPV_p1")
print(f"candidates per shot median {Cn.groupby(['game','idx']).size().median():.0f}; xP of best option median {m.xP.median():.2f}; xA of best median {m.xA.median():.3f}")
m.to_pickle(f"{D}/shot_vs_pass_m.pkl")
