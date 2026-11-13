"""DQ-v4 prototype (PREREG_DQ_v4.txt)."""
import sys, os; D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from tfcommon import steps, FEATS as HF
from src.pass_completion import Prep
sys.argv = ["", D + "/../choice/cands.pkl", D + "/../choice/passes_cov.pkl", "x"]; sys.path.insert(0, D + "/../choice"); import model as mdl
Cd, Sd = pd.read_pickle(f"{D}/proto_in.pkl"); S = Sd.set_index("key")
VM = pd.read_pickle(f"{D}/stage1b_models.pkl")["V"]; HM = pd.read_pickle(f"{D}/tf_model.pkl")
K, MV = 0.814, 0.0168; REC = {1: (-0.764, 0.793), 2: (-0.713, 0.778), 3: (-0.581, 0.819)}
sb = lambda x, y: ((np.asarray(x) + 52.5) * 120 / 105, (np.asarray(y) + 34) * 80 / 68)
# ---- V at candidate positions
cx, cy = sb(Cd.x_att, Cd.y_att); dist = np.hypot(120 - cx, 40 - cy); ang = np.abs(np.arctan2(44 - cy, 120 - cx) - np.arctan2(36 - cy, 120 - cx))
Xv = np.column_stack([dist, ang, cx, dist ** 2, ang * dist]); Zv = np.column_stack([np.ones(len(Xv)), (Xv - VM["mu"]) / VM["sd"]]); Cd["V"] = 1 / (1 + np.exp(-Zv @ VM["b"]))
# ---- hazard logit per step -> per candidate sum of log(1-h) for a given recalibration
rows = []
for i, r in enumerate(Cd.itertuples()):
    s = S.loc[r.key]; a = np.array(sb(s.px, s.py)); end = np.array(sb(r.x_att, r.y_att)); op = np.column_stack(sb(s.opps[:, 0], s.opps[:, 1]))
    R = steps(a, end, op, False, None); Z = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]); rows.append(Z @ HM["b"])
Cd["hlog"] = rows
def Ps(recal):
    out = np.empty(len(Cd))
    for i, (lg, g) in enumerate(zip(Cd.hlog, Cd.game)):
        a, b = recal(g); h = 1 / (1 + np.exp(-(a + b * lg))); out[i] = np.exp(np.log1p(-h).sum())
    return out
# ---- choice weights (LOGO, ridge 1, trained on completed passes of the other games, R0)
def weights(rule, feats):
    w = np.full(len(Cd), np.nan)
    for g in (1, 2, 3):
        mdl.Cd["F1"] = np.where(mdl.Cd["pass"].map(mdl.Pd.speed) >= 1.0, mdl.Cd.cos_vel, 0.0)
        dtr, st = mdl.sets_for(rule if rule != "R0" else "R0_primary", [x for x in (1, 2, 3) if x != g])
        pr = Prep().fit(dtr[feats].to_numpy(float)); b = mdl.fit_clogit(pr.transform(dtr[feats].to_numpy(float)), mdl.chosen_idx(dtr, st), st, 1.0)
        m = (Cd.game == g) & Cd[rule]; u = pr.transform(Cd.loc[m, feats].to_numpy(float)) @ b
        e = pd.Series(np.exp(u - pd.Series(u).groupby(Cd.key[m].to_numpy()).transform("max").to_numpy()), index=Cd.index[m])
        w[m.to_numpy()] = (e / e.groupby(Cd.key[m]).transform("sum")).to_numpy()
    return w
BASEF = ["dist", "forward", "lateral", "rec_press", "margin"]
def dq(ps, w, rule, k=K):
    d = Cd[Cd[rule]].copy(); d["ps"] = ps[Cd[rule].to_numpy()]; d["w"] = w[Cd[rule].to_numpy()]; d["EV"] = 100 * (d.ps * d.V - (1 - d.ps) * k * MV)
    g = d.groupby("key"); res = pd.DataFrame({"n": g.size(), "EVw": g.apply(lambda x: (x.w * x.EV).sum()), "EVmax": g.EV.max()})
    ch = d[d.is_chosen].set_index("key"); res = res.join(ch[["EV", "ps", "V", "cand"]].rename(columns={"EV": "EVc", "ps": "ps_c", "V": "V_c", "cand": "ch"}), how="inner")
    res["DQw"] = res.EVc - res.EVw; res["DQmax"] = res.EVc - res.EVmax; res["is_best"] = res.DQmax >= -1e-12
    return res[res.n >= 2].join(S[["game", "outcome", "px", "py", "p_press", "idx", "t", "period", "team", "passer"]])
ps0 = Ps(lambda g: REC[g]); w0 = weights("R0", BASEF + ["F1"]); B = dq(ps0, w0, "R0")
P1, P2 = B[B.outcome == "completed"], B[B.outcome == "intercepted"]
print(f"situations: P1 completed {len(P1)}, P2 intercepted (heading target in R0) {len(P2)}")
for nm, d in (("P1", P1), ("P2", P2)):
    print(f"[{nm}] EV chosen mean {d.EVc.mean():.3f} pp; DQ_w mean {d.DQw.mean():+.3f} median {d.DQw.median():+.3f} (IQR {d.DQw.quantile(.25):+.3f}..{d.DQw.quantile(.75):+.3f}); share DQ_w>0 {(d.DQw>0).mean()*100:.1f}%; "
          f"DQ_max median {d.DQmax.median():+.3f}; chosen = argmax EV {d.is_best.mean()*100:.1f}%; chosen Ps mean {d.ps_c.mean():.3f}")
# known groups
y = np.r_[np.zeros(len(P1)), np.ones(len(P2))]; sc = -np.r_[P1.DQw, P2.DQw]
def auc(y, s): r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
rng = np.random.default_rng(0); bs = []
for _ in range(1000):
    i1, i2 = rng.integers(0, len(P1), len(P1)), rng.integers(0, len(P2), len(P2)); bs.append(auc(y, -np.r_[P1.DQw.to_numpy()[i1], P2.DQw.to_numpy()[i2]]))
print(f"known-groups: AUC (lower DQ_w -> intercepted) {auc(y, sc):.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}];  using chosen Ps alone {auc(y, -np.r_[P1.ps_c, P2.ps_c]):.3f}; chosen V alone {auc(y, np.r_[P1.V_c, P2.V_c]):.3f}")
print("by game (P1): " + ", ".join(f"G{g} DQ_w median {P1[P1.game==g].DQw.median():+.3f}, >0 {(P1[P1.game==g].DQw>0).mean()*100:.0f}%" for g in (1, 2, 3)))
P1 = P1.assign(third=pd.cut(P1.px, [-60, -17.5, 17.5, 60], labels=["def", "mid", "att"]), press=pd.qcut(P1.p_press, 3, labels=["high", "mid", "low"]))
print("by third (P1): " + ", ".join(f"{k} n={len(d)} median {d.DQw.median():+.3f} >0 {(d.DQw>0).mean()*100:.0f}%" for k, d in P1.groupby("third", observed=True)))
print("by passer pressure (P1): " + ", ".join(f"{k} median {d.DQw.median():+.3f} >0 {(d.DQw>0).mean()*100:.0f}%" for k, d in P1.groupby("press", observed=True)))
# ---- G4 sensitivity
def cmp(name, R):
    j = P1[["DQw"]].join(R[["DQw"]], rsuffix="_s", how="inner"); agree = (np.sign(j.DQw) == np.sign(j.DQw_s)).mean(); rho = j.DQw.rank().corr(j.DQw_s.rank())
    print(f"  {name:<36} n={len(j)} sign agreement {agree*100:.1f}%  Spearman {rho:.3f}  => {'PASS' if agree >= 0.8 else 'FAIL'}"); return agree >= 0.8
print("G4 sensitivity (P1):"); ok = []
ok += [cmp("k x0.83", dq(ps0, w0, "R0", K * 0.83)), cmp("k x1.17", dq(ps0, w0, "R0", K * 1.17))]
for f in (1, 2, 3): ok.append(cmp(f"recalibration params of fold G{f}", dq(Ps(lambda g, f=f: REC[f]), w0, "R0")))
ok.append(cmp("no recalibration", dq(Ps(lambda g: (0.0, 1.0)), w0, "R0")))
ok.append(cmp("rule R_loose", dq(ps0, weights("R_loose", BASEF + ["F1"]), "R_loose")))
ok.append(cmp("weights without motion feature", dq(ps0, weights("R0", BASEF), "R0")))
print(f"GATE G4: {'PASS' if all(ok) else 'FAIL'}")
B.to_pickle(f"{D}/proto_out.pkl"); Cd.assign(ps=ps0, w=w0).to_pickle(f"{D}/proto_cands.pkl")
