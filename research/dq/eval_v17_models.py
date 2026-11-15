"""DQ-v17 (PREREG_DQ_v13.txt amendment M): the models of the 20 s horizon variant: chain value of a pass target (xA_chain_20), value of a carry destination (V_pos_20), shot continuation (cont_20) and the cost of losing the ball (L_20).
Fitted on TRAIN with 20 s outcomes, gates on the test competitions, per zone reported; writes v17_models.pkl."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
from dscore import core, score as S
from dscore.core import Situation
Dd = pd.read_pickle(f"{D}/v17_dec.pkl"); Lo = pd.read_pickle(f"{D}/v17_loss.pkl")
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
        q = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30))); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + 1e-9 * np.eye(2), Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def mboot(v, mt, B=300):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
def zone_of(x): return np.where(x < 40, "own third", np.where(x < 80, "middle", "attacking"))
gates = {}; F_XA = ["dist", "angle", "x", "d2", "ad", "ldist", "n_tri", "n3", "n6", "nn", "plen", "fwd", "pdist"]
def sit(r): return Situation([r.ax, r.ay], r.opps.astype(float), list(r.mates.astype(float)), r.gk)
print(f"data: {len(Dd)} decisions {dict(Dd.kind.value_counts())}, test {int(Dd.test.sum())}; losses {len(Lo)} (test {int(Lo.test.sum())}); mean own20 {Dd.own20.mean():.4f} vs own10 {Dd.own10.mean():.4f}; mean chain20 of completed passes {Dd[(Dd.kind=='pass')&(Dd.completed==1)].chain20.mean():.4f} vs chain10 {Dd[(Dd.kind=='pass')&(Dd.completed==1)].chain10.mean():.4f}")
def report(name, y, p, df, ycol="y"):
    ok = 0.85 <= p.mean() / y.mean() <= 1.15 and 0.8 <= slope(y, p) <= 1.2; gates[name] = bool(ok)
    print(f"\n== {name}: test n={len(y)}; realised {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}) => {'PASS' if ok else 'FAIL'}")
    d = df.assign(y=y, p=p); print("   by zone: " + "; ".join(f"{z}: n={len(g)}, realised {g.y.mean():.4f}, predicted {g.p.mean():.4f} (ratio {g.p.mean()/max(g.y.mean(), 1e-9):.2f})" for z, g in d.groupby("zone") if len(g) > 50))
# ---------------------------------------------------------------- xA_chain_20
P = Dd[(Dd.kind == "pass") & (Dd.completed == 1)].copy(); P["zone"] = zone_of(P.tx.to_numpy())
XP = np.array([[f[k] for k in F_XA] for f in (core.target_features(np.array([r.ax, r.ay]), r.opps.astype(float), core.DEFAULT_GK, np.array([r.tx, r.ty])) for r in P.itertuples())], float)
y = np.clip(P.chain20.to_numpy(float), 0, 0.99); tr, te = ~P.test.to_numpy(), P.test.to_numpy(); f1 = fit(XP[tr], y[tr]); report("xA_chain_20 (completed passes, 20 s chain value of the target)", y[te], f1(XP[te]), P[te])
# ---------------------------------------------------------------- carries: V_pos_20 (kept carries)
C = Dd[(Dd.kind == "carry") & (Dd.fail == 0)].copy(); C["zone"] = zone_of(C.ax.to_numpy()); cf = []
for r in C.itertuples():
    s_ = sit(r); Tm = np.array(s_.mates, float).reshape(-1, 2); cf.append(S.carry_features(s_, np.array([r.tx, r.ty], float), S._around(s_.a, s_.O, Tm), Tm, float(np.hypot(core.GX - r.ax, 40 - r.ay))))
XC = np.array([[f[k] for k in F_XA] for f in cf], float); yc = np.clip(C.chain20.to_numpy(float), 0, 0.99); trc, tec = ~C.test.to_numpy(), C.test.to_numpy(); fv = fit(XC[trc], yc[trc]); report("V_pos_20 (kept carries, 20 s chain value of the destination)", yc[tec], fv(XC[tec]), C[tec])
# ---------------------------------------------------------------- cont_20 (shot continuation)
Sh = Dd[Dd.kind == "shot"].copy(); M6 = core.models()["v6"]["cont"]; F_C = M6["F"]
XS = np.array([[core.shot_features(sit(r), 0)[k] for k in F_C] for r in Sh.itertuples()], float); ys = np.clip(Sh.cont20.to_numpy(float), 0, 0.99); trs, tes = ~Sh.test.to_numpy(), Sh.test.to_numpy()
fc = fit(XS[trs], ys[trs]); Sh["zone"] = zone_of(Sh.ax.to_numpy()); report("cont_20 (later shots of the possession within 20 s of a shot)", ys[tes], fc(XS[tes]), Sh[tes])
frozen = np.array([core.sgm(M6, list(x)) for x in XS[tes]]); print(f"   reference: the frozen 10 s continuation model on the same test shots predicts {frozen.mean():.4f} against the realised 10 s continuation {np.clip(Sh.cont10.to_numpy(float), 0, .99)[tes].mean():.4f}")
# ---------------------------------------------------------------- L_20 (cost of losing the ball)
for k in ("pass", "dribble", "dispossessed"): Lo[f"k_{k}"] = (Lo.kind == k).astype(float)
Lo["d2"], Lo["ad"] = Lo.dist_own ** 2, Lo.ang_own * Lo.dist_own
F_L = ["dist_own", "ang_own", "lx", "d2", "ad", "n_opp3", "n_opp6", "nn_opp", "n_mate10", "nn_mate", "n_opp_ahead", "n_cover", "k_pass", "k_dribble", "k_dispossessed"]
ltr, lte = Lo[~Lo.test], Lo[Lo.test].copy(); Lm = {}; lte["zone"] = zone_of(lte.lx.to_numpy())
print(f"\n== L_20: danger of losing the ball over 20 s (opponent xG - own xG), losses anywhere; train {len(ltr)}, test {len(lte)}")
for nm, col in (("opp", "opp20"), ("own", "own20")):
    f = fit(ltr[F_L].to_numpy(float), np.clip(ltr[col].to_numpy(float), 0, 0.99)); yy = np.clip(lte[col].to_numpy(float), 0, 0.99); p = f(lte[F_L].to_numpy(float)); lte[f"p_{nm}"] = p; Lm[nm] = f.coef
    g = ll(yy, np.full(len(yy), np.clip(ltr[col], 0, .99).mean())) - ll(yy, p); lo, hi = mboot(g, lte.match.to_numpy()); ok = lo > 0 and 0.85 <= p.mean() / yy.mean() <= 1.15; gates[f"L20_{nm}"] = bool(ok)
    print(f"   {nm}: observed mean {yy.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/yy.mean():.3f}, slope {slope(yy, p):.3f}); LL gain {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}] => {'PASS' if ok else 'FAIL'}")
lte["L"] = lte.p_opp - lte.p_own; lte["Lobs"] = lte.opp20 - lte.own20
print("   L_20 by zone of the loss (observed / predicted): " + "; ".join(f"{z}: n={len(g)}, {g.Lobs.mean():+.4f} / {g.L.mean():+.4f}" for z, g in lte.groupby("zone")) + "; same losses over 10 s observed: " + "; ".join(f"{z}: {(g.opp10-g.own10).mean():+.4f}" for z, g in lte.groupby("zone")))
print(f"   L_20 predicted percentiles 1/50/99/99.9: " + ", ".join(f"{np.percentile(lte.L, q):+.4f}" for q in (1, 50, 99, 99.9)))
print("\nGATES:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
pd.to_pickle(dict(xA_chain_20=dict(F=F_XA, **f1.coef), V_pos_20=dict(F=F_XA, **fv.coef), cont_20=dict(F=F_C, **fc.coef), L_20=dict(F=F_L, opp=Lm["opp"], own=Lm["own"]), gates=gates), f"{D}/v17_models.pkl")
