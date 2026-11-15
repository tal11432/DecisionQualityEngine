"""DQ-v13 Phase 2 (after amendment B: whole-pitch losses, one-on-one anchor, take-on dropped): new models for the Decision Score (PREREG_DQ_v13.txt section 4): turnover danger L, keep-the-ball probabilities for carries and dribbles, value of having the ball at a position (V_pos). Gates on the test competitions."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
A = pd.read_pickle(f"{D}/v13_acts.pkl"); Lo = pd.read_pickle(f"{D}/v13_loss.pkl"); M6 = pd.read_pickle(f"{D}/v6_models.pkl")
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
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def auc(y, p): r = pd.Series(p).rank().to_numpy(); n1 = (y > 0.5).sum(); return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def mboot(v, mt):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]; return np.percentile(b, [2.5, 97.5])
def tau(d, ycol, pcol):
    rows = []
    for c, g in d.groupby("comp"):
        if len(g) < 150: continue
        rows.append((g[ycol].mean() / g[pcol].mean(), max(1, int((g[ycol] > 0).sum()))))
    r = np.array([x[0] for x in rows]); npos = np.array([x[1] for x in rows]); return float(np.sqrt(max(0.0, np.var(np.log(r), ddof=1) - np.mean(1 / npos)))), len(rows)
print(f"data: carries {int((A.kind=='carry').sum())} (fail rate {A[A.kind=='carry'].fail.mean()*100:.1f}%), dribbles {int((A.kind=='dribble').sum())} (success {A[A.kind=='dribble'].complete.mean()*100:.1f}%), losses {len(Lo)} by kind {dict(Lo.kind.value_counts())}")
gates = {}
# ---------------- L: turnover danger
for k in ("pass", "dribble", "dispossessed"): Lo[f"k_{k}"] = (Lo.kind == k).astype(float)
Lo["d2"], Lo["ad"] = Lo.dist_own ** 2, Lo.ang_own * Lo.dist_own; Lo["nn_opp"], Lo["nn_mate"] = Lo.nn_opp.clip(upper=30), Lo.nn_mate.clip(upper=30)
FL = ["dist_own", "ang_own", "lx", "d2", "ad", "n_opp3", "n_opp6", "nn_opp", "n_mate10", "nn_mate", "n_opp_ahead", "n_cover", "k_pass", "k_dribble", "k_dispossessed"]
tr, te = Lo[~Lo.test], Lo[Lo.test].copy(); Lm = {}
print("\n== L: danger of losing the ball (opponent xG - own xG in the next 10 s), losses anywhere on the pitch")
for nm, col in (("opp", "opp10"), ("own", "own10")):
    f = fit(tr[FL].to_numpy(float), np.clip(tr[col].to_numpy(float), 0, 0.99)); y = np.clip(te[col].to_numpy(float), 0, 0.99); p = f(te[FL].to_numpy(float)); te[f"p_{nm}"] = p; Lm[nm] = f.coef
    g = ll(y, np.full(len(y), np.clip(tr[col], 0, .99).mean())) - ll(y, p); lo, hi = mboot(g, te.match.to_numpy()); ok = lo > 0 and 0.85 <= p.mean() / y.mean() <= 1.15; gates[f"L_{nm}"] = ok
    print(f"   {nm}: train {len(tr)} test {len(te)}; observed mean {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); LL gain {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  => {'PASS' if ok else 'FAIL'}")
te["L"] = te.p_opp - te.p_own; te["Lobs"] = te.opp10 - te.own10
print("   by x of the loss point: " + "; ".join(f"{b}: L observed {g.Lobs.mean():+.4f} predicted {g.L.mean():+.4f} (n={len(g)})" for b, g in te.groupby(pd.cut(te.lx, [-1, 30, 50, 70, 85, 100, 121]), observed=True)))
print("   L predicted: 1st/10th/50th/90th/99th percentile " + ", ".join(f"{np.percentile(te.L, q):+.4f}" for q in (1, 10, 50, 90, 99)) + "; (old line follows) 10th/50th/90th " + ", ".join(f"{np.percentile(te.L, q):+.4f}" for q in (10, 50, 90)) + f"; share of losses with L > 0.05: {(te.L>0.05).mean()*100:.1f}%")
own = te[te.lx < 40].copy(); own["lvl"] = pd.qcut(own.n_cover, [0, .34, .67, 1.0], duplicates="drop"); print("   situation dependence inside our own 40 units (teammates behind the ball): " + "; ".join(f"{b}: L predicted {g.L.mean():+.4f}, observed {g.Lobs.mean():+.4f} (n={len(g)})" for b, g in own.groupby("lvl", observed=True)))
for c in ("opp10", "own10"): te[c + "_p"] = te["p_" + c[:3]]
t1, n1 = tau(te.assign(y=te.opp10, p=te.p_opp), "y", "p"); t2, n2 = tau(te.assign(y=te.own10, p=te.p_own), "y", "p"); print(f"   between-competition spread tau (test competitions only): opp {t1:.3f} ({n1} comps), own {t2:.3f} ({n2} comps)")
# ---------------- P_keep for carries
C = A[A.kind == "carry"].copy(); C["keep"] = 1 - C.fail; C["x"] = C.ex; C["d2"] = C.dist ** 2; C["ad"] = C.angle * C.dist
FC = ["dist", "angle", "n_tri", "gk_d", "n3", "n6", "nn", "L", "fwd", "pdist", "s_n_opp3", "s_nn_opp", "s_n_opp6"]
ctr, cte = C[~C.test], C[C.test].copy(); fk = fit(ctr[FC].to_numpy(float), ctr.keep.to_numpy(float)); cte["pk"] = fk(cte[FC].to_numpy(float)); y = cte.keep.to_numpy(float); p = cte.pk.to_numpy(float)
a_ = auc(1 - y, 1 - p); ok = 0.8 <= slope(y, p) <= 1.2 and 0.9 <= p.mean() / y.mean() <= 1.1 and a_ >= 0.60; gates["P_keep"] = ok
print(f"\n== P_keep (carries): train {len(ctr)} test {len(cte)}; observed keep rate {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); AUC for losing the ball {a_:.3f}  => {'PASS' if ok else 'FAIL'}")
print("== P_dribble: DROPPED from v1 (amendment B: failed its gate with an AUC of 0.955 that signals outcome leakage in the event-time frame)")
Dr = A[A.kind == "dribble"].copy(); FD = []; fd = None
# ---------------- V_pos: value of having the ball at the destination (kept carries and completed dribbles), frozen xA_chain vs refit
sgm = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
K = pd.concat([C[C.keep == 1].assign(kk="carry"), Dr[Dr.complete == 1].assign(kk="dribble", n_tri=Dr[Dr.complete == 1].n_tri)]).copy(); K["x"] = K.ex; K["d2"] = K.dist ** 2; K["ad"] = K.angle * K.dist; K["plen"] = K.L; K["y"] = np.clip(K.chain10, 0, 0.99)
F6 = M6["xA_chain"]["F"]; K["v0"] = [sgm(M6["xA_chain"], r[F6].to_numpy(float)) for _, r in K.iterrows()]; ktr, kte = K[~K.test], K[K.test].copy()
f1 = fit(ktr[F6].to_numpy(float), ktr.y.to_numpy(float)); kte["v1"] = f1(kte[F6].to_numpy(float)); y = kte.y.to_numpy(float)
for nm, col in (("frozen xA_chain", "v0"), ("refit on carries/dribbles", "v1")):
    p = kte[col].to_numpy(float); ok = 0.85 <= p.mean() / y.mean() <= 1.15 and 0.8 <= slope(y, p) <= 1.2; gates[f"V_pos {nm}"] = ok
    print(f"== V_pos {nm}: test n={len(kte)} (kept carries {int((kte.kk=='carry').sum())}, completed dribbles {int((kte.kk=='dribble').sum())}); realised own chain {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f})  => {'PASS' if ok else 'FAIL'}")
print("   V_pos by destination distance to goal (m): " + "; ".join(f"{b}: realised {g.y.mean():.4f} frozen {g.v0.mean():.4f} refit {g.v1.mean():.4f} (n={len(g)})" for b, g in kte.groupby(pd.cut(kte.dist * 105 / 120, [0, 12, 20, 30, 80]), observed=True)))
t3, n3 = tau(kte.assign(p=kte.v1), "y", "p"); print(f"   between-competition spread tau of V_pos (refit, test competitions): {t3:.3f} ({n3} comps)")
O1 = pd.read_pickle(f"{D}/v13_oneone.pkl"); O1tr = O1[~O1.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024")]; QLO = float(O1tr.xg.mean()); print(f"\nQ_lo (score 0) = mean xG of {len(O1tr)} one-on-one open-play shots in the training competitions = {QLO:.3f}; Q_hi = 0.70")
print("\nGATES:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
pd.to_pickle(dict(L=dict(F=FL, opp=Lm["opp"], own=Lm["own"]), P_keep=dict(F=FC, **fk.coef), V_pos_refit=dict(F=F6, **f1.coef), gates=gates, Q_lo=QLO, Q_hi=0.70), f"{D}/v13_models.pkl")
