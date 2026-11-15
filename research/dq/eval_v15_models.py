"""DQ-v15 (PREREG_DQ_v13.txt amendment E): counter-aware turnover danger L, leak-free take-on model P_dribble, and the zero anchor C of the Decision Score. Gates on the test competitions; writes v15_models.pkl."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
Lo = pd.read_pickle(f"{D}/v15_loss.pkl"); Dr = pd.read_pickle(f"{D}/v15_dribble.pkl"); M13 = pd.read_pickle(f"{D}/v13_models.pkl")
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
def mboot(v, mt, B=500):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
gates = {}
print(f"data: losses {len(Lo)} by kind {dict(Lo.kind.value_counts())}; stand-alone take-ons {len(Dr)} (success {Dr.ok.mean()*100:.1f}%)")
# ------------------------------------------------ L with numerical-balance features
for k in ("pass", "dribble", "dispossessed"): Lo[f"k_{k}"] = (Lo.kind == k).astype(float)
Lo["d2"], Lo["ad"] = Lo.dist_own ** 2, Lo.ang_own * Lo.dist_own
Lo["adv"] = Lo.n_opp_ahead - Lo.n_cover; Lo["adv_pos"] = Lo.adv.clip(0, 5); Lo["adv_x"] = Lo.adv_pos * (1 - Lo.lx / 120.0)
F_OLD = ["dist_own", "ang_own", "lx", "d2", "ad", "n_opp3", "n_opp6", "nn_opp", "n_mate10", "nn_mate", "n_opp_ahead", "n_cover", "k_pass", "k_dribble", "k_dispossessed"]
F_NEW = F_OLD + ["adv", "adv_pos", "n_free_run", "run_x", "nn_run", "adv_x"]
tr, te = Lo[~Lo.test].copy(), Lo[Lo.test].copy(); Lm = {}
print(f"\n== L: danger of losing the ball (opponent xG - own xG in the next 10 s), losses anywhere; train {len(tr)}, test {len(te)}")
for tag, FL in (("old features (v13)", F_OLD), ("with numerical balance (v15)", F_NEW)):
    P = {}
    for nm, col in (("opp", "opp10"), ("own", "own10")):
        f = fit(tr[FL].to_numpy(float), np.clip(tr[col].to_numpy(float), 0, 0.99)); y = np.clip(te[col].to_numpy(float), 0, 0.99); p = f(te[FL].to_numpy(float)); P[nm] = p
        g = ll(y, np.full(len(y), np.clip(tr[col], 0, .99).mean())) - ll(y, p); lo_, hi_ = mboot(g, te.match.to_numpy()); ok = lo_ > 0 and 0.85 <= p.mean() / y.mean() <= 1.15
        print(f"   [{tag}] {nm}: observed mean {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); LL gain over constant {g.mean():+.5f} [{lo_:+.5f},{hi_:+.5f}] => {'PASS' if ok else 'FAIL'}")
        if FL is F_NEW: gates[f"L_{nm}"] = bool(ok); Lm[nm] = f.coef
    te["L_old" if FL is F_OLD else "L"] = P["opp"] - P["own"]
te["Lobs"] = te.opp10 - te.own10
# is the new model better for the opponent part? (paired log-likelihood difference, same test set)
for nm, col in (("opp", "opp10"),):
    y = np.clip(te[col].to_numpy(float), 0, 0.99); fo = fit(tr[F_OLD].to_numpy(float), np.clip(tr[col].to_numpy(float), 0, 0.99)); fn = fit(tr[F_NEW].to_numpy(float), np.clip(tr[col].to_numpy(float), 0, 0.99))
    g = ll(y, fo(te[F_OLD].to_numpy(float))) - ll(y, fn(te[F_NEW].to_numpy(float))); lo_, hi_ = mboot(g, te.match.to_numpy()); print(f"   paired LL gain of the numerical-balance features over the old features ({nm}): {g.mean():+.5f} [{lo_:+.5f},{hi_:+.5f}]")
print("   L predicted (new): 1st/10th/50th/90th/99th/99.9th percentile " + ", ".join(f"{np.percentile(te.L, q):+.4f}" for q in (1, 10, 50, 90, 99, 99.9)) + f"; (old) max {te.L_old.max():+.3f}, 99.9th {np.percentile(te.L_old, 99.9):+.4f}")
te["adv"] = te.n_opp_ahead - te.n_cover; z = te[te.lx <= 70].copy(); z["grp"] = pd.cut(z.adv, [-99, 0, 1, 99], labels=["adv<=0", "adv=1", "adv>=2"])
rows = []
for b, g in z.groupby("grp", observed=True): rows.append((b, len(g), g.Lobs.mean(), g.L.mean(), g.L_old.mean()))
print("   by opponent numerical advantage (losses at x <= 70; test): " + "; ".join(f"{b}: n={n}, observed {o:+.4f}, predicted new {p:+.4f} / old {po:+.4f}" for b, n, o, p, po in rows))
g2 = z[z.adv >= 2]; g0 = z[z.adv <= 0]; ratio = g2.L.mean() / g2.Lobs.mean() if len(g2) and g2.Lobs.mean() > 0 else np.nan
rises = len(g2) and len(g0) and g2.L.mean() > g0.L.mean() and g2.Lobs.mean() > g0.Lobs.mean()
okc = (len(g2) < 200 or (0.75 <= ratio <= 1.25)) and bool(rises); gates["G-counter"] = bool(okc)
print(f"   G-counter: losses with adv >= 2 at x <= 70 in the test competitions n={len(g2)}, predicted/observed {ratio:.3f}, predicted L rises from adv<=0 to adv>=2: {bool(rises)} => {'PASS' if okc else 'FAIL'}" + (" (n < 200: cannot fail on the ratio)" if len(g2) < 200 else ""))
# ------------------------------------------------ the zero anchor C
trm = tr[(tr.n_opp_ahead - tr.n_cover >= 2) & (tr.lx <= 70)].copy(); trm["net"] = trm.opp10 - trm.own10; c_meas = float(trm.net.mean()); lo_, hi_ = mboot(trm.net.to_numpy(float), trm.match.to_numpy()); half = (hi_ - lo_) / 2
use = 0.15 <= c_meas <= 0.35 and half <= 0.05; C = c_meas if use else float(M13["Q_lo"])
print(f"\n== Zero anchor C: training losses with adv >= 2 and x <= 70: n={len(trm)}; mean realised net danger {c_meas:.4f} (95% CI [{lo_:.4f},{hi_:.4f}], half-width {half:.4f}); mean opponent xG {trm.opp10.mean():.4f}; share with an opponent shot {np.mean(trm.opp10 > 0)*100:.1f}%")
print(f"   rule (measured value if inside [0.15, 0.35] and half-width <= 0.05): {'MEASURED value used' if use else 'NOT met -> C = 0.231 (mean xG of one-on-one open-play shots)'}; C = {C:.4f}")
# ------------------------------------------------ take-on without outcome leakage
Dr["y"] = Dr.ok.astype(float); FD = ["dist", "angle", "n_tri", "gk_d", "n3", "n6", "nn", "n_cone", "nn_cone", "n_mate10"]
dtr, dte = Dr[~Dr.test], Dr[Dr.test].copy(); fd = fit(dtr[FD].to_numpy(float), dtr.y.to_numpy(float)); p = fd(dte[FD].to_numpy(float)); y = dte.y.to_numpy(float)
a_ = auc(y, p); ok = 0.8 <= slope(y, p) <= 1.2 and 0.9 <= p.mean() / y.mean() <= 1.1 and 0.60 <= a_ <= 0.80; gates["P_dribble"] = bool(ok)
print(f"\n== P_dribble (take-on success, features from the frame BEFORE the duel; preceding events {dict(Dr.prev_kind.value_counts())}, median {Dr.dt_prev.median():.1f} s earlier): train {len(dtr)}, test {len(dte)}; observed success {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); AUC {a_:.3f} => {'PASS' if ok else 'FAIL'}" + (" (AUC > 0.80: residual leakage suspected)" if a_ > 0.80 else ""))
print("   by number of opponents within 6 units of the preceding frame: " + "; ".join(f"{b}: observed {g.y.mean():.3f}, predicted {g.p.mean():.3f} (n={len(g)})" for b, g in dte.assign(p=p, nb=dte.n6.clip(upper=3)).groupby("nb")))
print("\nGATES:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
pd.to_pickle(dict(L=dict(F=F_NEW, opp=Lm["opp"], own=Lm["own"]), P_dribble=dict(F=FD, **fd.coef), gates=gates, C=C, C_measured=c_meas, C_rule_met=bool(use)), f"{D}/v15_models.pkl")
