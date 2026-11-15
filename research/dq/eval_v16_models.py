"""DQ-v16 (PREREG_DQ_v13.txt amendment J): whole-pitch models for the Decision Score: chain value of a pass target (xA_chain_all), keep-the-ball probability of a carry (P_keep_all), value of having the ball at a carry destination (V_pos_all),
and the per-zone calibration (with logistic recalibration if needed) of the frozen xP hazard model. Fitted on TRAIN, gates on the test competitions; writes v16_models.pkl."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
from dscore import core, score as S
from dscore.core import Situation
Dd = pd.read_pickle(f"{D}/v16_dec.pkl")
def fit(X, y, lam=1.0, it=300, clip=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.clip(np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b), -clip, clip); b += s
        if np.abs(s).max() < 1e-8: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(1 - np.clip(p, 1e-9, 1 - 1e-9)))
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
def slope(y, p):
    Z = np.column_stack([np.ones(len(p)), lg(p)]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + 1e-9 * np.eye(2), Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
lgc = lambda p: np.clip(lg(p), -8.0, 8.0)                                  # logit clipped at +-8 for the xP calibration (xP is within 1e-9 of 1 for many own-third passes; the unclipped fit diverged)
def slope_c(y, p, ridge=1e-3):
    Z = np.column_stack([np.ones(len(p)), lgc(p)]); b = np.array([0.0, 1.0]); R = ridge * np.eye(2)
    for _ in range(200):
        q = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30))); s = np.clip(np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + R, Z.T @ (y - q) - R @ (b - np.array([0.0, 1.0]))), -2, 2); b += s
        if np.abs(s).max() < 1e-9: break
    return b
def auc(y, p): r = pd.Series(p).rank().to_numpy(); n1 = (y > 0.5).sum(); return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def mboot(v, mt, B=300):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
def zone_of(x): return np.where(x < 40, "own third", np.where(x < 80, "middle", "attacking"))
ZN = ["own third", "middle", "attacking"]
gates = {}
print(f"data: {len(Dd)} decisions {dict(Dd.kind.value_counts())}; test {int(Dd.test.sum())}; keeper in the frame in {Dd.gk.notna().mean()*100:.0f}% (own third {Dd[Dd.ax<40].gk.notna().mean()*100:.0f}%)")
def sit(r): return Situation([r.ax, r.ay], r.opps, list(r.mates), r.gk)
# ---------------------------------------------------------------- (1) xA_chain_all: completed ground passes, target anywhere
F_XA = ["dist", "angle", "x", "d2", "ad", "ldist", "n_tri", "n3", "n6", "nn", "plen", "fwd", "pdist"]
P = Dd[(Dd.kind == "pass") & (Dd.completed == 1)].copy()
ft = [core.target_features(np.array([r.ax, r.ay]), r.opps.astype(float), core.DEFAULT_GK, np.array([r.tx, r.ty])) for r in P.itertuples()]
XP = np.array([[f[k] for k in F_XA] for f in ft], float); P["y"] = np.clip(P.chain10.to_numpy(float), 0, 0.99); P["tz"] = zone_of(P.tx.to_numpy())
ptr, pte = ~P.test.to_numpy(), P.test.to_numpy()
f1 = fit(XP[ptr], P.y.to_numpy()[ptr]); p = f1(XP[pte]); y = P.y.to_numpy()[pte]; te = P[pte].assign(p=p)
ok = 0.85 <= p.mean() / y.mean() <= 1.15 and 0.8 <= slope(y, p) <= 1.2; gates["xA_chain_all"] = bool(ok)
g = ll(y, np.full(len(y), P.y[ptr].mean())) - ll(y, p); lo, hi = mboot(g, te.match.to_numpy())
print(f"\n== xA_chain_all (completed passes, target anywhere): train {int(ptr.sum())}, test {int(pte.sum())}; realised chain {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); LL gain {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}] => {'PASS' if ok else 'FAIL'}")
print("   by zone of the target: " + "; ".join(f"{z}: n={len(g_)}, realised {g_.y.mean():.4f}, predicted {g_.p.mean():.4f} (ratio {g_.p.mean()/max(g_.y.mean(),1e-9):.2f}, slope {slope(g_.y.to_numpy(), g_.p.to_numpy()):.2f})" for z, g_ in te.groupby("tz") if len(g_) > 50))
# frozen v6 model on the same test passes for reference (attacking half only is its support)
M6 = core.models()["v6"]["xA_chain"]; v6 = np.array([core.sgm(M6, [f[k] for k in M6["F"]]) for f, m_ in zip(ft, pte) if m_]); att = (te.tx >= 60).to_numpy()
print(f"   reference, frozen v6 model on test passes ending at x >= 60 (its support): realised {y[att].mean():.4f}, v6 {v6[att].mean():.4f}, new model {p[att].mean():.4f}")
# ---------------------------------------------------------------- (2) carries
C = Dd[Dd.kind == "carry"].copy(); T_ = C.test.to_numpy()
F_PK = ["dist", "angle", "x", "n_tri", "n3", "n6", "nn", "L", "fwd", "pdist", "s_n_opp3", "s_nn_opp", "s_n_opp6"]; cf = []
for r in C.itertuples():
    s_ = sit(r); Tm = np.array(s_.mates, float).reshape(-1, 2); st = S._around(s_.a, s_.O, Tm); cf.append(S.carry_features(s_, np.array([r.tx, r.ty], float), st, Tm, float(np.hypot(core.GX - r.ax, 40 - r.ay))))
XC = np.array([[f[k] for k in F_PK] for f in cf], float); keep = 1 - C.fail.to_numpy(float)
fk = fit(XC[~T_], keep[~T_]); pk = fk(XC[T_]); yk = keep[T_]; a_ = auc(1 - yk, 1 - pk); okk = 0.8 <= slope(yk, pk) <= 1.2 and 0.9 <= pk.mean() / yk.mean() <= 1.1 and a_ >= 0.60; gates["P_keep_all"] = bool(okk)
cte = C[T_].assign(pk=pk, keep=yk, z=zone_of(C.ax.to_numpy()[T_]))
print(f"\n== P_keep_all (carries anywhere): train {int((~T_).sum())}, test {int(T_.sum())}; observed keep rate {yk.mean():.4f}, predicted {pk.mean():.4f} (ratio {pk.mean()/yk.mean():.3f}, slope {slope(yk, pk):.3f}); AUC for losing the ball {a_:.3f} => {'PASS' if okk else 'FAIL'}")
print("   by zone of the start: " + "; ".join(f"{z}: n={len(g_)}, keep observed {g_.keep.mean():.4f} predicted {g_.pk.mean():.4f}" for z, g_ in cte.groupby("z")))
K = C[C.fail == 0]; kf = [f for f, fl in zip(cf, C.fail.to_numpy()) if fl == 0]; XK = np.array([[f[k] for k in F_XA] for f in kf], float); yv = np.clip(K.chain10.to_numpy(float), 0, 0.99); tk = K.test.to_numpy()
fv = fit(XK[~tk], yv[~tk]); pv = fv(XK[tk]); yt = yv[tk]; okv = 0.85 <= pv.mean() / yt.mean() <= 1.15 and 0.8 <= slope(yt, pv) <= 1.2; gates["V_pos_all"] = bool(okv)
kte = K[tk].assign(pv=pv, y=yt, z=zone_of(K.ax.to_numpy()[tk]))
print(f"\n== V_pos_all (kept carries, value of the destination): train {int((~tk).sum())}, test {int(tk.sum())}; realised chain {yt.mean():.4f}, predicted {pv.mean():.4f} (ratio {pv.mean()/yt.mean():.3f}, slope {slope(yt, pv):.3f}) => {'PASS' if okv else 'FAIL'}")
print("   by zone of the start: " + "; ".join(f"{z}: n={len(g_)}, realised {g_.y.mean():.4f} predicted {g_.pv.mean():.4f} (ratio {g_.pv.mean()/max(g_.y.mean(),1e-9):.2f})" for z, g_ in kte.groupby("z")))
# ---------------------------------------------------------------- (3) xP hazard model, per zone of the passer
PA = Dd[Dd.kind == "pass"].copy(); PA["z"] = zone_of(PA.ax.to_numpy()); trn = PA[~PA.test].groupby("z").sample(n=20000, random_state=0); tst = PA[PA.test]
xp_of = lambda df: np.array([core.xp_hazard(np.array([r.ax, r.ay], float), np.array([r.tx, r.ty], float), r.opps.astype(float)) for r in df.itertuples()])
trn = trn.assign(xp=xp_of(trn)); tst = tst.assign(xp=xp_of(tst)); cal = {}
print(f"\n== xP (frozen hazard model) vs completion, per zone of the passer (target = where the ball ended; failed passes make this approximate); TRAIN sample {len(trn)} (20,000 per zone), test {len(tst)}")
for z in ZN:
    g_ = tst[tst.z == z]; y = g_.completed.to_numpy(float); p = g_.xp.to_numpy(float); ok0 = 0.95 <= p.mean() / y.mean() <= 1.05 and 0.8 <= slope_c(y, p)[1] <= 1.2
    print(f"   {z}: n={len(g_)}, observed completion {y.mean():.4f}, xP {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope_c(y, p)[1]:.3f}, AUC {auc(y, p):.3f}) => {'PASS' if ok0 else 'FAIL'}", end="")
    if ok0: print(); gates[f"xP {z}"] = True; continue
    t_ = trn[trn.z == z]; b = slope_c(t_.completed.to_numpy(float), t_.xp.to_numpy(float)); cal[z] = (float(b[0]), float(b[1]))
    p2 = 1 / (1 + np.exp(-(b[0] + b[1] * lgc(p)))); ok1 = 0.95 <= p2.mean() / y.mean() <= 1.05 and 0.8 <= slope_c(y, p2)[1] <= 1.2; gates[f"xP {z}"] = bool(ok1)
    print(f"; recalibrated on TRAIN (a={b[0]:+.3f}, b={b[1]:.3f}, logit clipped at +-8): xP {p2.mean():.4f} (ratio {p2.mean()/y.mean():.3f}, slope {slope_c(y, p2)[1]:.3f}, AUC {auc(y, p2):.3f}) => {'PASS' if ok1 else 'NOT CALIBRATED'}")
print("\nGATES:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
pd.to_pickle(dict(xA_chain_all=dict(F=F_XA, **f1.coef), P_keep_all=dict(F=F_PK, **fk.coef), V_pos_all=dict(F=F_XA, **fv.coef), xp_cal=cal, gates=gates), f"{D}/v16_models.pkl")
